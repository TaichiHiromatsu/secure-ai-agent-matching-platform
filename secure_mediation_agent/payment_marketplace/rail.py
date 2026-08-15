"""Deterministic, value-free local PaymentRail implementation."""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Mapping, Protocol

from .store import IdempotencyConflict, MarketplaceStore, StoreError, compact_json, utc_now


class PaymentRail(Protocol):
    def capabilities(self) -> Mapping[str, Any]: ...
    def get_operation(self, operation_id: str) -> dict[str, Any] | None: ...


class RailError(StoreError):
    pass


class UnknownFaultDisabled(RailError):
    pass


def _hash_request(value: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(compact_json(value).encode('utf-8')).hexdigest()}"


class LocalPaymentRail:
    """Atomic fixed-balance simulation; never communicates with an external network."""

    CUSTOMER = "demo-customer"
    PLATFORM = "mediation-platform"
    MERCHANT = "demo-merchant"

    def __init__(
        self,
        store: MarketplaceStore,
        *,
        asset: str = "USD",
        initial_customer_balance: int = 100_000,
        allow_test_faults: bool = False,
        schema_version: int = 1,
    ) -> None:
        if isinstance(initial_customer_balance, bool) or initial_customer_balance < 0:
            raise ValueError("initial balance must be a non-negative integer")
        self.store = store
        self.asset = asset
        self.allow_test_faults = allow_test_faults
        self.schema_version = schema_version
        now = utc_now()
        with self.store.business_transaction() as conn:
            for account_id, balance in (
                (self.CUSTOMER, initial_customer_balance),
                (self.PLATFORM, 0),
                (self.MERCHANT, 0),
            ):
                conn.execute(
                    """INSERT OR IGNORE INTO rail_accounts
                       (account_id, asset, balance, version, updated_at)
                       VALUES (?, ?, ?, 1, ?)""",
                    (account_id, asset, balance, now),
                )

    def capabilities(self) -> dict[str, Any]:
        return {
            "profile": "urn:secure-a2a:extensions:ap2-x402-marketplace:v1",
            "simulated": True,
            "scheme": "exact-simulated",
            "network": "demo:local",
            "asset": self.asset,
            "decimals": 2,
            "charge": True,
            "refund": True,
            "payout": True,
        }

    def get_balance(self, account_id: str) -> int:
        rows = self.store.fetch_business(
            "SELECT balance FROM rail_accounts WHERE account_id=? AND asset=?",
            (account_id, self.asset),
        )
        if not rows:
            raise RailError(f"unknown rail account {account_id}")
        return int(rows[0]["balance"])

    def settle_charge(
        self,
        *,
        operation_id: str,
        source_id: str,
        amount: int,
        idempotency_key: str,
        payer: str = CUSTOMER,
        payee: str = PLATFORM,
        fault: str | None = None,
    ) -> dict[str, Any]:
        return self._transfer(
            kind="charge",
            operation_id=operation_id,
            source_id=source_id,
            from_account=payer,
            to_account=payee,
            amount=amount,
            idempotency_key=idempotency_key,
            fault=fault,
        )

    def refund(
        self,
        *,
        operation_id: str,
        source_id: str,
        amount: int,
        idempotency_key: str,
        payer: str = PLATFORM,
        payee: str = CUSTOMER,
        fault: str | None = None,
    ) -> dict[str, Any]:
        return self._transfer(
            kind="refund",
            operation_id=operation_id,
            source_id=source_id,
            from_account=payer,
            to_account=payee,
            amount=amount,
            idempotency_key=idempotency_key,
            fault=fault,
        )

    def payout(
        self,
        *,
        operation_id: str,
        source_id: str,
        amount: int,
        idempotency_key: str,
        payer: str = PLATFORM,
        payee: str = MERCHANT,
        fault: str | None = None,
    ) -> dict[str, Any]:
        return self._transfer(
            kind="payout",
            operation_id=operation_id,
            source_id=source_id,
            from_account=payer,
            to_account=payee,
            amount=amount,
            idempotency_key=idempotency_key,
            fault=fault,
        )

    @staticmethod
    def _result(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["simulated"] = True
        result["receipt"] = (
            None if result.get("receipt_json") is None else __import__("json").loads(result["receipt_json"])
        )
        result.pop("receipt_json", None)
        return result

    def _transfer(
        self,
        *,
        kind: str,
        operation_id: str,
        source_id: str,
        from_account: str,
        to_account: str,
        amount: int,
        idempotency_key: str,
        fault: str | None,
    ) -> dict[str, Any]:
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise RailError("rail amount must be a non-negative integer")
        if from_account == to_account:
            raise RailError("rail transfer accounts must differ")
        if fault not in (None, "unknown"):
            raise RailError(f"unsupported fault {fault}")
        if fault and not self.allow_test_faults:
            raise UnknownFaultDisabled("rail fault injection is test-only")
        request = {
            "kind": kind,
            "source_id": source_id,
            "from_account": from_account,
            "to_account": to_account,
            "asset": self.asset,
            "amount": amount,
        }
        request_hash = _hash_request(request)
        now = utc_now()
        with self.store.business_transaction() as conn:
            existing = conn.execute(
                """SELECT * FROM rail_operations
                   WHERE operation_id=? OR (kind=? AND idempotency_key=?)""",
                (operation_id, kind, idempotency_key),
            ).fetchone()
            if existing:
                if existing["operation_id"] != operation_id or existing["request_hash"] != request_hash:
                    raise IdempotencyConflict(f"conflicting {kind} rail retry")
                return self._result(existing)

            source = conn.execute(
                "SELECT balance FROM rail_accounts WHERE account_id=? AND asset=?",
                (from_account, self.asset),
            ).fetchone()
            destination = conn.execute(
                "SELECT balance FROM rail_accounts WHERE account_id=? AND asset=?",
                (to_account, self.asset),
            ).fetchone()
            if source is None or destination is None:
                raise RailError("rail account or asset is not allowlisted")

            state = "unknown" if fault == "unknown" else "settled"
            applied = 0
            error_code = None
            receipt: dict[str, Any] | None = None
            if state == "settled" and int(source["balance"]) < amount:
                state = "failed"
                error_code = "INSUFFICIENT_FUNDS"
            elif state == "settled":
                self._apply_transfer(conn, from_account, to_account, amount, now)
                applied = 1
                receipt = self._receipt(operation_id, kind, source_id, amount, now)

            conn.execute(
                """INSERT INTO rail_operations
                   (operation_id, kind, source_id, from_account, to_account, asset,
                    amount, state, applied, idempotency_key, request_hash, error_code,
                    receipt_json, attempt, schema_version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (
                    operation_id,
                    kind,
                    source_id,
                    from_account,
                    to_account,
                    self.asset,
                    amount,
                    state,
                    applied,
                    idempotency_key,
                    request_hash,
                    error_code,
                    compact_json(receipt) if receipt else None,
                    self.schema_version,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM rail_operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
            return self._result(row)

    def _apply_transfer(
        self,
        conn: sqlite3.Connection,
        from_account: str,
        to_account: str,
        amount: int,
        now: str,
    ) -> None:
        debited = conn.execute(
            """UPDATE rail_accounts SET balance=balance-?, version=version+1, updated_at=?
               WHERE account_id=? AND asset=? AND balance>=?""",
            (amount, now, from_account, self.asset, amount),
        ).rowcount
        if debited != 1:
            raise RailError("INSUFFICIENT_FUNDS")
        credited = conn.execute(
            """UPDATE rail_accounts SET balance=balance+?, version=version+1, updated_at=?
               WHERE account_id=? AND asset=?""",
            (amount, now, to_account, self.asset),
        ).rowcount
        if credited != 1:
            raise RailError("destination account disappeared")

    def _receipt(
        self, operation_id: str, kind: str, source_id: str, amount: int, issued_at: str
    ) -> dict[str, Any]:
        return {
            "receiptType": "x402-settlement" if kind == "charge" else kind,
            "receiptId": f"rail-receipt:{operation_id}",
            "profile": "urn:secure-a2a:extensions:ap2-x402-marketplace:v1",
            "simulated": True,
            "status": "settled",
            "operationId": operation_id,
            "sourceId": source_id,
            "amount": str(amount),
            "asset": self.asset,
            "network": "demo:local",
            "issuedAt": issued_at,
        }

    def get_operation(self, operation_id: str) -> dict[str, Any] | None:
        rows = self.store.fetch_business(
            "SELECT * FROM rail_operations WHERE operation_id=?", (operation_id,)
        )
        return None if not rows else self._result(rows[0])

    def resolve_unknown(self, operation_id: str, *, settled: bool) -> dict[str, Any]:
        """Test/reconciliation hook that resolves, but never re-creates, an operation."""
        if not self.allow_test_faults:
            raise UnknownFaultDisabled("unknown resolution fixture is test-only")
        with self.store.business_transaction() as conn:
            row = conn.execute(
                "SELECT * FROM rail_operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if row is None:
                raise RailError(f"unknown operation {operation_id}")
            if row["state"] != "unknown":
                return self._result(row)
            now = utc_now()
            state = "failed"
            error_code = "SETTLEMENT_NOT_APPLIED"
            receipt = None
            applied = 0
            if settled:
                source = conn.execute(
                    "SELECT balance FROM rail_accounts WHERE account_id=? AND asset=?",
                    (row["from_account"], row["asset"]),
                ).fetchone()
                if source is not None and int(source["balance"]) >= int(row["amount"]):
                    self._apply_transfer(
                        conn,
                        row["from_account"],
                        row["to_account"],
                        int(row["amount"]),
                        now,
                    )
                    state = "settled"
                    applied = 1
                    error_code = None
                    receipt = self._receipt(
                        operation_id, row["kind"], row["source_id"], int(row["amount"]), now
                    )
                else:
                    error_code = "INSUFFICIENT_FUNDS"
            conn.execute(
                """UPDATE rail_operations SET state=?, applied=?, error_code=?, receipt_json=?,
                   attempt=attempt+1, updated_at=? WHERE operation_id=? AND state='unknown'""",
                (
                    state,
                    applied,
                    error_code,
                    compact_json(receipt) if receipt else None,
                    now,
                    operation_id,
                ),
            )
            result = conn.execute(
                "SELECT * FROM rail_operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
            return self._result(result)

    def reconcile_platform_cash(self, ledger: Any) -> dict[str, Any]:
        rail_balance = self.get_balance(self.PLATFORM)
        ledger_balance = int(ledger.account_balance("simulated_cash", self.asset))
        return {
            "asset": self.asset,
            "rail_balance": rail_balance,
            "ledger_balance": ledger_balance,
            "difference": rail_balance - ledger_balance,
            "balanced": rail_balance == ledger_balance,
            "simulated": True,
        }
