"""Append-only, balanced double-entry journal for the payment demo."""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from .store import MarketplaceStore, StoreError, compact_json, utc_now


class LedgerError(StoreError):
    pass


class UnbalancedJournal(LedgerError):
    pass


class JournalConflict(LedgerError):
    pass


def _content_hash(value: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(compact_json(value).encode('utf-8')).hexdigest()}"


class Ledger:
    """Posts immutable journals and owns payable accounting transitions."""

    def __init__(self, store: MarketplaceStore, *, schema_version: int = 1) -> None:
        self.store = store
        self.schema_version = schema_version

    @staticmethod
    def _normalize_entries(
        entries: Iterable[Mapping[str, Any]], currency: str
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in entries:
            side = str(item["side"])
            amount = item["amount"]
            if side not in {"debit", "credit"}:
                raise LedgerError(f"invalid journal side {side}")
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise LedgerError("journal amount must be a non-negative integer")
            item_currency = str(item.get("currency", currency))
            if item_currency != currency:
                raise LedgerError("currencies cannot be mixed in one journal")
            normalized.append(
                {
                    "account": str(item["account"]),
                    "side": side,
                    "amount": amount,
                    "currency": item_currency,
                    "related_entry_id": item.get("related_entry_id"),
                }
            )
        if not normalized:
            raise UnbalancedJournal("journal requires at least one entry")
        debit = sum(item["amount"] for item in normalized if item["side"] == "debit")
        credit = sum(item["amount"] for item in normalized if item["side"] == "credit")
        if debit != credit:
            raise UnbalancedJournal(f"debit {debit} does not equal credit {credit}")
        return normalized

    def _post_journal(
        self,
        conn: sqlite3.Connection,
        *,
        journal_id: str,
        event_type: str,
        source_id: str,
        currency: str,
        entries: Sequence[Mapping[str, Any]],
        idempotency_key: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = self._normalize_entries(entries, currency)
        content = {
            "event_type": event_type,
            "source_id": source_id,
            "currency": currency,
            "entries": normalized,
            "metadata": dict(metadata or {}),
        }
        digest = _content_hash(content)
        existing = conn.execute(
            "SELECT * FROM journal_transactions WHERE event_type=? AND source_id=?",
            (event_type, source_id),
        ).fetchone()
        if existing:
            if existing["journal_id"] != journal_id or existing["content_hash"] != digest:
                raise JournalConflict(f"journal source {event_type}/{source_id} already posted differently")
            return dict(existing)

        now = utc_now()
        conn.execute(
            """INSERT INTO journal_transactions
               (journal_id, event_type, source_id, currency, state, content_hash,
                idempotency_key, metadata_json, schema_version, created_at)
               VALUES (?, ?, ?, ?, 'draft', ?, ?, ?, ?, ?)""",
            (
                journal_id,
                event_type,
                source_id,
                currency,
                digest,
                idempotency_key,
                compact_json(metadata or {}),
                self.schema_version,
                now,
            ),
        )
        for index, item in enumerate(normalized, 1):
            conn.execute(
                """INSERT INTO journal_entries
                   (entry_id, journal_id, account, side, amount, currency, effective_at,
                    source_event, idempotency_key, related_entry_id, schema_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"{journal_id}:entry:{index}",
                    journal_id,
                    item["account"],
                    item["side"],
                    item["amount"],
                    currency,
                    now,
                    f"{event_type}:{source_id}",
                    idempotency_key,
                    item["related_entry_id"],
                    self.schema_version,
                ),
            )
        conn.execute(
            "UPDATE journal_transactions SET state='posted', posted_at=? WHERE journal_id=?",
            (now, journal_id),
        )
        return dict(
            conn.execute("SELECT * FROM journal_transactions WHERE journal_id=?", (journal_id,)).fetchone()
        )

    def post_journal(
        self,
        *,
        journal_id: str,
        event_type: str,
        source_id: str,
        currency: str,
        entries: Sequence[Mapping[str, Any]],
        idempotency_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.store.business_transaction() as conn:
            return self._post_journal(
                conn,
                journal_id=journal_id,
                event_type=event_type,
                source_id=source_id,
                currency=currency,
                entries=entries,
                idempotency_key=idempotency_key,
                metadata=metadata,
            )

    def post_charge(
        self,
        *,
        journal_id: str,
        charge_id: str,
        order_id: str,
        payable_id: str,
        merchant_id: str,
        amount: int,
        currency: str = "USD",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Post Dr simulated_cash / Cr merchant_payable and create its principal."""
        with self.store.business_transaction() as conn:
            journal = self._post_journal(
                conn,
                journal_id=journal_id,
                event_type="charge",
                source_id=charge_id,
                currency=currency,
                entries=(
                    {"account": "simulated_cash", "side": "debit", "amount": amount},
                    {
                        "account": f"merchant_payable:{merchant_id}",
                        "side": "credit",
                        "amount": amount,
                    },
                ),
                idempotency_key=idempotency_key,
                metadata={"order_id": order_id, "payable_id": payable_id, "merchant_id": merchant_id},
            )
            now = utc_now()
            conn.execute(
                """INSERT OR IGNORE INTO payables
                   (payable_id, order_id, charge_id, merchant_id, amount, asset, state,
                    journal_id, version, schema_version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'open', ?, 1, ?, ?, ?)""",
                (
                    payable_id,
                    order_id,
                    charge_id,
                    merchant_id,
                    amount,
                    currency,
                    journal_id,
                    self.schema_version,
                    now,
                    now,
                ),
            )
            payable = conn.execute("SELECT * FROM payables WHERE payable_id=?", (payable_id,)).fetchone()
            if (
                payable is None
                or payable["charge_id"] != charge_id
                or payable["amount"] != amount
                or payable["merchant_id"] != merchant_id
            ):
                raise JournalConflict("payable principal conflicts with the charge journal")
            return {"journal": journal, "payable": dict(payable)}

    def post_refund(
        self,
        *,
        journal_id: str,
        refund_id: str,
        merchant_id: str,
        amount: int,
        payable_id: str | None = None,
        currency: str = "USD",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        with self.store.business_transaction() as conn:
            journal = self._post_journal(
                conn,
                journal_id=journal_id,
                event_type="refund",
                source_id=refund_id,
                currency=currency,
                entries=(
                    {
                        "account": f"merchant_payable:{merchant_id}",
                        "side": "debit",
                        "amount": amount,
                    },
                    {"account": "simulated_cash", "side": "credit", "amount": amount},
                ),
                idempotency_key=idempotency_key,
                metadata={"payable_id": payable_id, "merchant_id": merchant_id},
            )
            if payable_id:
                changed = conn.execute(
                    """UPDATE payables SET state='reversed', version=version+1, updated_at=?
                       WHERE payable_id=? AND state IN ('open','guaranteed','eligible','reversing')""",
                    (utc_now(), payable_id),
                ).rowcount
                if changed == 0:
                    current = conn.execute("SELECT state FROM payables WHERE payable_id=?", (payable_id,)).fetchone()
                    if current is None or current["state"] != "reversed":
                        raise JournalConflict("payable is not refundable")
            return journal

    def post_payout(
        self,
        *,
        journal_id: str,
        payout_id: str,
        merchant_id: str,
        amount: int,
        payable_ids: Sequence[str] = (),
        currency: str = "USD",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        with self.store.business_transaction() as conn:
            journal = self._post_journal(
                conn,
                journal_id=journal_id,
                event_type="payout",
                source_id=payout_id,
                currency=currency,
                entries=(
                    {
                        "account": f"merchant_payable:{merchant_id}",
                        "side": "debit",
                        "amount": amount,
                    },
                    {"account": "simulated_cash", "side": "credit", "amount": amount},
                ),
                idempotency_key=idempotency_key,
                metadata={"payable_ids": list(payable_ids), "merchant_id": merchant_id},
            )
            if payable_ids:
                placeholders = ",".join("?" for _ in payable_ids)
                rows = conn.execute(
                    f"SELECT payable_id, merchant_id, amount, state FROM payables WHERE payable_id IN ({placeholders})",
                    tuple(payable_ids),
                ).fetchall()
                if len(rows) != len(set(payable_ids)):
                    raise JournalConflict("payout contains an unknown payable")
                if any(row["merchant_id"] != merchant_id for row in rows):
                    raise JournalConflict("payout mixes merchant tenants")
                if sum(int(row["amount"]) for row in rows) != amount:
                    raise JournalConflict("payout items do not equal journal amount")
                invalid = [row["payable_id"] for row in rows if row["state"] not in {"included", "paid"}]
                if invalid:
                    raise JournalConflict(f"payables not included for payout: {invalid}")
                conn.execute(
                    f"""UPDATE payables SET state='paid', payout_id=?, version=version+1, updated_at=?
                        WHERE payable_id IN ({placeholders}) AND state='included'""",
                    (payout_id, utc_now(), *payable_ids),
                )
                conn.execute("UPDATE payout_items SET state='paid' WHERE payout_id=?", (payout_id,))
            return journal

    def get_journal(self, journal_id: str) -> dict[str, Any] | None:
        rows = self.store.fetch_business(
            "SELECT * FROM journal_transactions WHERE journal_id=?", (journal_id,)
        )
        if not rows:
            return None
        result = rows[0]
        result["entries"] = self.store.fetch_business(
            "SELECT * FROM journal_entries WHERE journal_id=? ORDER BY entry_id", (journal_id,)
        )
        return result

    def account_balance(self, account: str, currency: str = "USD") -> int:
        rows = self.store.fetch_business(
            """SELECT COALESCE(SUM(CASE side WHEN 'debit' THEN amount ELSE -amount END), 0) AS balance
               FROM journal_entries e JOIN journal_transactions j ON j.journal_id=e.journal_id
               WHERE e.account=? AND e.currency=? AND j.state='posted'""",
            (account, currency),
        )
        return int(rows[0]["balance"])

    def all_journals_balanced(self) -> bool:
        rows = self.store.fetch_business(
            """SELECT j.journal_id,
                      SUM(CASE e.side WHEN 'debit' THEN e.amount ELSE 0 END) AS debits,
                      SUM(CASE e.side WHEN 'credit' THEN e.amount ELSE 0 END) AS credits
               FROM journal_transactions j JOIN journal_entries e ON e.journal_id=j.journal_id
               WHERE j.state='posted' GROUP BY j.journal_id
               HAVING debits <> credits"""
        )
        return not rows
