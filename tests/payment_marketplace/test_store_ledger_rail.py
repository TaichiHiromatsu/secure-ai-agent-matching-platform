from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from secure_mediation_agent.payment_marketplace.ledger import (
    JournalConflict,
    Ledger,
    UnbalancedJournal,
)
from secure_mediation_agent.payment_marketplace.rail import (
    LocalPaymentRail,
    UnknownFaultDisabled,
)
from secure_mediation_agent.payment_marketplace.store import (
    EvidenceAccessDenied,
    IdempotencyConflict,
    MarketplaceStore,
    ReplayDetected,
    sha256_digest,
    utc_now,
)


def new_store(tmp_path: Path) -> MarketplaceStore:
    return MarketplaceStore(tmp_path / "marketplace.db", tmp_path / "evidence.db")


def seed_order_charge(
    store: MarketplaceStore,
    *,
    suffix: str = "1",
    amount: int = 1_250,
) -> dict[str, str]:
    ids = {
        "task": f"task-{suffix}",
        "context": f"context-{suffix}",
        "order": f"order-{suffix}",
        "charge": f"charge-{suffix}",
        "payable": f"payable-{suffix}",
    }
    store.save_task(
        ids["task"],
        ids["context"],
        "input-required",
        actor_id="demo-customer",
        tenant_id="demo-customer",
        metadata={"simulated": True},
    )
    store.create_order(
        ids["order"],
        ids["task"],
        ids["context"],
        "demo-customer",
        "demo-merchant",
    )
    now = utc_now()
    with store.business_transaction() as conn:
        conn.execute(
            """INSERT INTO charges
               (charge_id, order_id, challenge_id, payer_id, pay_to, amount, asset,
                nonce, state, operation_id, idempotency_key, version, schema_version,
                created_at, updated_at)
               VALUES (?, ?, ?, 'demo-customer', 'mediation-platform', ?, 'USD', ?,
                       'settled', ?, ?, 1, 1, ?, ?)""",
            (
                ids["charge"],
                ids["order"],
                f"challenge-{suffix}",
                amount,
                f"nonce-{suffix}",
                f"rail-charge-{suffix}",
                f"charge-key-{suffix}",
                now,
                now,
            ),
        )
    return ids


def test_migration_contains_business_and_separate_evidence_schema(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    assert store.schema_versions() == {"business": 1, "evidence": 1}

    with sqlite3.connect(store.business_db) as conn:
        business_tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    assert {
        "tasks",
        "orders",
        "merchant_quotes",
        "pricing",
        "charges",
        "payables",
        "guarantees",
        "fulfillments",
        "refunds",
        "payouts",
        "payout_items",
        "journal_transactions",
        "journal_entries",
        "rail_accounts",
        "rail_operations",
        "idempotency_records",
        "used_nonces",
        "state_events",
        "evidence_intents",
    } <= business_tables
    assert "evidence" not in business_tables

    with sqlite3.connect(store.evidence_db) as conn:
        evidence_tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    assert {"evidence", "evidence_access_events"} <= evidence_tables
    assert "orders" not in evidence_tables


def test_task_order_state_idempotency_nonce_and_restart(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    store.save_task(
        "task-1",
        "context-1",
        "input-required",
        actor_id="demo-customer",
        tenant_id="demo-customer",
        metadata={"profile": "v1"},
    )
    order = store.create_order(
        "order-1", "task-1", "context-1", "demo-customer", "demo-merchant"
    )
    updated = store.update_order_state(
        "order-1",
        "awaiting_quote",
        "payment_required",
        actor_id="mediation-platform",
        reason="validated-quote",
        expected_version=order["version"],
    )
    assert updated["state"] == "payment_required"
    assert len(store.list_state_events("order", "order-1")) == 2

    assert store.begin_idempotency("charge", "demo-customer", "idem-1", "hash-a")["status"] == "new"
    store.complete_idempotency(
        "charge", "demo-customer", "idem-1", "hash-a", {"chargeId": "charge-1"}
    )
    hit = store.begin_idempotency("charge", "demo-customer", "idem-1", "hash-a")
    assert hit == {
        "status": "hit",
        "state": "completed",
        "response": {"chargeId": "charge-1"},
    }
    with pytest.raises(IdempotencyConflict):
        store.begin_idempotency("charge", "demo-customer", "idem-1", "hash-b")

    store.consume_nonce(
        "demo-customer",
        "nonce-1",
        "sha256:one",
        order_id="order-1",
        task_id="task-1",
        operation="charge",
    )
    with pytest.raises(ReplayDetected):
        store.consume_nonce(
            "demo-customer",
            "nonce-1",
            "sha256:two",
            order_id="order-2",
            task_id="task-2",
            operation="charge",
        )

    restarted = MarketplaceStore(store.business_db, store.evidence_db)
    assert restarted.get_task("task-1")["context_id"] == "context-1"
    assert restarted.get_order("order-1")["state"] == "payment_required"
    assert restarted.begin_idempotency("charge", "demo-customer", "idem-1", "hash-a")[
        "response"
    ] == {"chargeId": "charge-1"}
    with pytest.raises(ReplayDetected):
        restarted.consume_nonce(
            "demo-customer",
            "nonce-1",
            "sha256:one",
            order_id="order-1",
            task_id="task-1",
            operation="charge",
        )


def test_evidence_intent_separation_access_and_recovery(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    payload = b"signed-proof-exact-bytes"
    intent = store.put_evidence(
        intent_id="intent-1",
        evidence_id="evidence-1",
        tenant_type="customer",
        tenant_id="demo-customer",
        kind="payment-proof",
        exact_bytes=payload,
        kid="demo-customer-hmac-v1",
    )
    assert intent["state"] == "committed"
    assert intent["digest"] == sha256_digest(payload)
    metadata = store.get_evidence_metadata("evidence-1")
    assert metadata["digest"] == sha256_digest(payload)
    assert "exact_bytes" not in metadata

    assert store.read_evidence(
        "evidence-1",
        actor_id="customer-actor",
        actor_role="customer",
        tenant_type="customer",
        tenant_id="demo-customer",
    ) == payload
    with pytest.raises(EvidenceAccessDenied):
        store.read_evidence(
            "evidence-1",
            actor_id="other-customer",
            actor_role="customer",
            tenant_type="customer",
            tenant_id="other-customer",
        )
    with sqlite3.connect(store.evidence_db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM evidence_access_events WHERE allowed=0"
        ).fetchone()[0] == 1
    assert store.read_evidence(
        "evidence-1", actor_id="operator-1", actor_role="operator"
    ) == payload

    now = utc_now()
    with store.business_transaction() as conn:
        conn.execute(
            """INSERT INTO evidence_intents
               (intent_id, evidence_id, tenant_type, tenant_id, kind, digest, state,
                schema_version, created_at, updated_at)
               VALUES ('intent-missing', 'evidence-missing', 'merchant', 'demo-merchant',
                       'guarantee', 'sha256:missing', 'pending', 1, ?, ?)""",
            (now, now),
        )
    pending = store.reconcile_evidence_intent("intent-missing")
    assert pending["state"] == "pending"
    assert pending["last_error"] == "EVIDENCE_NOT_DURABLE"

    restarted = MarketplaceStore(store.business_db, store.evidence_db)
    assert restarted.get_evidence_metadata("evidence-1")["digest"] == sha256_digest(payload)
    assert restarted.reconcile_evidence_intent("intent-1")["state"] == "committed"


def test_balanced_charge_journal_payable_and_immutability(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    ids = seed_order_charge(store)
    ledger = Ledger(store)

    posted = ledger.post_charge(
        journal_id="journal-charge-1",
        charge_id=ids["charge"],
        order_id=ids["order"],
        payable_id=ids["payable"],
        merchant_id="demo-merchant",
        amount=1_250,
        idempotency_key="ledger-charge-1",
    )
    assert posted["journal"]["state"] == "posted"
    assert posted["payable"]["state"] == "open"
    assert ledger.account_balance("simulated_cash") == 1_250
    assert ledger.account_balance("merchant_payable:demo-merchant") == -1_250
    assert ledger.all_journals_balanced()

    retry = ledger.post_charge(
        journal_id="journal-charge-1",
        charge_id=ids["charge"],
        order_id=ids["order"],
        payable_id=ids["payable"],
        merchant_id="demo-merchant",
        amount=1_250,
        idempotency_key="ledger-charge-1",
    )
    assert retry["journal"]["journal_id"] == "journal-charge-1"
    assert len(store.fetch_business("SELECT * FROM journal_entries")) == 2
    assert len(store.fetch_business("SELECT * FROM payables")) == 1

    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError)):
        with store.business_transaction() as conn:
            conn.execute(
                "UPDATE journal_entries SET amount=999 WHERE journal_id='journal-charge-1'"
            )
    with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError)):
        with store.business_transaction() as conn:
            conn.execute(
                """INSERT INTO journal_entries
                   (entry_id,journal_id,account,side,amount,currency,effective_at,
                    source_event,schema_version)
                   VALUES ('late-entry','journal-charge-1','simulated_cash','debit',1,
                           'USD',?,'tamper',1)""",
                (utc_now(),),
            )

    with pytest.raises(UnbalancedJournal):
        ledger.post_journal(
            journal_id="unbalanced",
            event_type="adjustment",
            source_id="bad-1",
            currency="USD",
            entries=(
                {"account": "simulated_cash", "side": "debit", "amount": 10},
                {"account": "merchant_payable:demo-merchant", "side": "credit", "amount": 9},
            ),
        )
    with pytest.raises(JournalConflict):
        ledger.post_charge(
            journal_id="journal-charge-1",
            charge_id=ids["charge"],
            order_id=ids["order"],
            payable_id=ids["payable"],
            merchant_id="demo-merchant",
            amount=1_251,
            idempotency_key="ledger-charge-1",
        )


def test_local_rail_charge_payout_and_ledger_reconciliation(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    ids = seed_order_charge(store)
    rail = LocalPaymentRail(store)
    ledger = Ledger(store)

    charge = rail.settle_charge(
        operation_id="rail-charge-1",
        source_id=ids["charge"],
        amount=1_250,
        idempotency_key="rail-charge-key-1",
    )
    assert charge["state"] == "settled"
    assert charge["simulated"] is True
    assert rail.get_balance("demo-customer") == 98_750
    assert rail.get_balance("mediation-platform") == 1_250

    same_charge = rail.settle_charge(
        operation_id="rail-charge-1",
        source_id=ids["charge"],
        amount=1_250,
        idempotency_key="rail-charge-key-1",
    )
    assert same_charge["receipt"] == charge["receipt"]
    assert rail.get_balance("demo-customer") == 98_750
    with pytest.raises(IdempotencyConflict):
        rail.settle_charge(
            operation_id="rail-charge-other",
            source_id=ids["charge"],
            amount=1_251,
            idempotency_key="rail-charge-key-1",
        )

    ledger.post_charge(
        journal_id="journal-charge-1",
        charge_id=ids["charge"],
        order_id=ids["order"],
        payable_id=ids["payable"],
        merchant_id="demo-merchant",
        amount=1_250,
    )
    assert rail.reconcile_platform_cash(ledger)["balanced"] is True

    with store.business_transaction() as conn:
        conn.execute(
            "UPDATE payables SET state='included', payout_id='payout-1' WHERE payable_id=?",
            (ids["payable"],),
        )
        now = utc_now()
        conn.execute(
            """INSERT INTO payouts
               (payout_id, merchant_id, state, gross_amount, commission_amount, rail_cost,
                net_amount, asset, eligibility_json, operation_id, idempotency_key,
                request_hash, attempt, version, schema_version, created_at, updated_at)
               VALUES ('payout-1', 'demo-merchant', 'settling', 1250, 0, 0, 1250,
                       'USD', '{}', 'rail-payout-1', 'payout-key-1', 'hash', 1, 1, 1, ?, ?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO payout_items(payout_id, payable_id, amount, state, created_at)
               VALUES ('payout-1', ?, 1250, 'claimed', ?)""",
            (ids["payable"], now),
        )

    payout = rail.payout(
        operation_id="rail-payout-1",
        source_id="payout-1",
        amount=1_250,
        idempotency_key="rail-payout-key-1",
    )
    assert payout["state"] == "settled"
    ledger.post_payout(
        journal_id="journal-payout-1",
        payout_id="payout-1",
        merchant_id="demo-merchant",
        amount=1_250,
        payable_ids=(ids["payable"],),
    )
    assert rail.get_balance("mediation-platform") == 0
    assert rail.get_balance("demo-merchant") == 1_250
    assert ledger.account_balance("simulated_cash") == 0
    assert ledger.account_balance("merchant_payable:demo-merchant") == 0
    assert rail.reconcile_platform_cash(ledger)["difference"] == 0
    assert store.fetch_business("SELECT state FROM payables WHERE payable_id=?", (ids["payable"],))[0][
        "state"
    ] == "paid"


def test_refund_insufficient_funds_unknown_and_restart(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    ids = seed_order_charge(store, suffix="refund", amount=500)
    rail = LocalPaymentRail(store, allow_test_faults=True)
    ledger = Ledger(store)
    rail.settle_charge(
        operation_id="rail-charge-refund",
        source_id=ids["charge"],
        amount=500,
        idempotency_key="charge-refund-key",
    )
    ledger.post_charge(
        journal_id="journal-charge-refund",
        charge_id=ids["charge"],
        order_id=ids["order"],
        payable_id=ids["payable"],
        merchant_id="demo-merchant",
        amount=500,
    )

    unknown = rail.refund(
        operation_id="rail-refund-1",
        source_id="refund-1",
        amount=500,
        idempotency_key="refund-key-1",
        fault="unknown",
    )
    assert unknown["state"] == "unknown"
    assert rail.get_balance("mediation-platform") == 500

    restarted_rail = LocalPaymentRail(
        MarketplaceStore(store.business_db, store.evidence_db), allow_test_faults=True
    )
    assert restarted_rail.get_operation("rail-refund-1")["state"] == "unknown"
    resolved = restarted_rail.resolve_unknown("rail-refund-1", settled=True)
    assert resolved["state"] == "settled"
    assert restarted_rail.get_balance("demo-customer") == 100_000
    ledger = Ledger(restarted_rail.store)
    ledger.post_refund(
        journal_id="journal-refund-1",
        refund_id="refund-1",
        merchant_id="demo-merchant",
        amount=500,
        payable_id=ids["payable"],
    )
    assert restarted_rail.reconcile_platform_cash(ledger)["balanced"] is True

    failed = restarted_rail.settle_charge(
        operation_id="rail-too-large",
        source_id="charge-too-large",
        amount=100_001,
        idempotency_key="too-large-key",
    )
    assert failed["state"] == "failed"
    assert failed["error_code"] == "INSUFFICIENT_FUNDS"
    assert restarted_rail.get_balance("demo-customer") == 100_000

    production_like = LocalPaymentRail(restarted_rail.store, allow_test_faults=False)
    with pytest.raises(UnknownFaultDisabled):
        production_like.payout(
            operation_id="unknown-disabled",
            source_id="payout-unknown",
            amount=0,
            idempotency_key="unknown-disabled-key",
            fault="unknown",
        )
