from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from secure_mediation_agent.ap2.keys import ROLE_KIDS
from secure_mediation_agent.workflow.controller import Identity, WorkflowController
from secure_mediation_agent.workflow.models import MessagePart, WorkflowRequest, WorkflowState


pytestmark = [pytest.mark.restart, pytest.mark.integration]
IDENTITY = Identity("demo-tenant", "demo-customer")
ROOT = Path(__file__).resolve().parents[2]


def _create(controller: WorkflowController, suffix: str, *, paid: bool = True):
    return controller.create(
        WorkflowRequest(goal=f"recovery {suffix}", paymentRequired=paid),
        identity=IDENTITY,
        session_id=f"recovery-session-{suffix}",
        context_id=f"recovery-context-{suffix}",
        idempotency_key=f"recovery-create-{suffix}",
    )


def _approve(controller, workflow_id: str, suffix: str):
    return controller.message(
        workflow_id,
        [MessagePart(kind="text", text="承認")],
        identity=IDENTITY,
        message_id=f"recovery-message-{suffix}",
        idempotency_key=f"recovery-message-key-{suffix}",
    )


def _drain(repository, keys, operation_id: str):
    row = repository.lease_outbox("pytest-recovery", operation_id=operation_id)
    assert row is not None
    WorkflowController(repository, keys).process_leased_outbox(row, "pytest-recovery")


def _defer_merchant_start(controller: WorkflowController, suffix: str):
    view = _create(controller, suffix)
    run_outbox_operation = controller._run_outbox_operation
    controller._run_outbox_operation = lambda _: None
    interrupted = _approve(controller, view.workflow_id, f"{suffix}-plan")
    controller._run_outbox_operation = run_outbox_operation
    return interrupted, f"start:{interrupted.task_id}"


def _merchant_effect_counts(workflow_fixture) -> tuple[int, int, int]:
    repository = workflow_fixture["repository"]
    with repository._connect(workflow_fixture["paths"].merchant) as connection:
        return tuple(
            int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "merchant_tasks_v2",
                "merchant_messages_v2",
                "merchant_operations_v2",
            )
        )


def test_api_fast_path_retries_a_just_enqueued_not_due_operation(
    workflow_fixture,
) -> None:
    repository = workflow_fixture["repository"]
    controller = WorkflowController(repository, workflow_fixture["keys"])
    interrupted, operation_id = _defer_merchant_start(controller, "clock-skew")
    available_at = (datetime.now(UTC) + timedelta(milliseconds=150)).isoformat().replace(
        "+00:00", "Z"
    )
    with repository.transaction() as connection:
        connection.execute(
            "UPDATE outbox SET available_at=? WHERE operation_id=?",
            (available_at, operation_id),
        )

    assert repository.lease_outbox("not-due-probe", operation_id=operation_id) is None
    controller._run_outbox_operation(operation_id)

    assert repository.outbox_row(operation_id)["status"] == "done"
    assert repository.get_workflow(interrupted.workflow_id)["state"] == (
        WorkflowState.PAYMENT_APPROVAL_REQUIRED
    )
    assert _merchant_effect_counts(workflow_fixture) == (1, 0, 0)


def test_api_fast_path_waits_for_an_active_competing_lease_without_replay(
    workflow_fixture,
) -> None:
    repository = workflow_fixture["repository"]
    controller = WorkflowController(repository, workflow_fixture["keys"])
    interrupted, operation_id = _defer_merchant_start(controller, "competing-worker")
    leased = repository.lease_outbox(
        "competing-worker", operation_id=operation_id, lease_seconds=120
    )
    assert leased is not None

    def complete_in_competing_worker() -> None:
        time.sleep(0.15)
        WorkflowController(repository, workflow_fixture["keys"]).process_leased_outbox(
            leased, "competing-worker"
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        completion = pool.submit(complete_in_competing_worker)
        controller._run_outbox_operation(operation_id)
        completion.result(timeout=2)

    # A completed operation is an idempotent no-op for the fast path.
    controller._run_outbox_operation(operation_id)
    assert repository.outbox_row(operation_id)["status"] == "done"
    assert repository.get_workflow(interrupted.workflow_id)["state"] == (
        WorkflowState.PAYMENT_APPROVAL_REQUIRED
    )
    assert _merchant_effect_counts(workflow_fixture) == (1, 0, 0)


def _persist_keys(directory: Path, keys) -> None:
    directory.mkdir(mode=0o700)
    for role in ROLE_KIDS:
        path = directory / f"{role}.jwk"
        path.write_text(getattr(keys, role).export(private_key=True), encoding="utf-8")
        path.chmod(0o600)


def _outbox_subprocess(
    workflow_fixture, operation_id: str, key_dir: Path, failpoint: str | None
) -> subprocess.CompletedProcess[str]:
    code = """
import os
from secure_mediation_agent.ap2.keys import DemoKeySet
from secure_mediation_agent.workflow.controller import WorkflowController
from secure_mediation_agent.workflow.migrations import DatabasePaths
from secure_mediation_agent.workflow.repository import WorkflowRepository
paths = DatabasePaths.resolve(
    os.environ['PAYMENT_MARKETPLACE_DB'],
    os.environ['PAYMENT_MERCHANT_DB'],
    os.environ['PAYMENT_EVIDENCE_DB'],
)
repository = WorkflowRepository(paths)
row = repository.lease_outbox('actual-crash-worker', operation_id=os.environ['OPERATION_ID'], lease_seconds=0)
if row is None:
    raise SystemExit(70)
WorkflowController(repository, DemoKeySet.from_environment()).process_leased_outbox(
    row, 'actual-crash-worker'
)
"""
    environment = os.environ.copy()
    environment.update(
        {
            "APP_ENV": "test",
            "AP2_DEMO_KEY_DIR": str(key_dir),
            "PAYMENT_MARKETPLACE_DB": str(workflow_fixture["paths"].marketplace),
            "PAYMENT_MERCHANT_DB": str(workflow_fixture["paths"].merchant),
            "PAYMENT_EVIDENCE_DB": str(workflow_fixture["paths"].evidence),
            "PAYMENT_TEST_FAILPOINT_MARKER": str(key_dir.parent / "failpoint-fired"),
            "OPERATION_ID": operation_id,
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), environment.get("PYTHONPATH", "")]
            ),
        }
    )
    if failpoint:
        environment["PAYMENT_TEST_FAILPOINT"] = failpoint
    else:
        environment.pop("PAYMENT_TEST_FAILPOINT", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_worker_recovers_merchant_start_with_same_task_and_operation_id(
    workflow_fixture,
) -> None:
    controller = WorkflowController(
        workflow_fixture["repository"], workflow_fixture["keys"]
    )
    view = _create(controller, "merchant-start")
    controller._run_outbox_operation = lambda _: None
    interrupted = _approve(controller, view.workflow_id, "merchant-start")
    assert interrupted.state == WorkflowState.MERCHANT_TASK_STARTING
    operation_id = f"start:{interrupted.task_id}"
    _drain(workflow_fixture["repository"], workflow_fixture["keys"], operation_id)
    recovered = workflow_fixture["repository"].get_workflow(view.workflow_id)
    assert recovered["state"] == WorkflowState.PAYMENT_APPROVAL_REQUIRED
    assert recovered["merchant_task_id"] == interrupted.task_id
    assert workflow_fixture["repository"].outbox_row(operation_id)["status"] == "done"


def test_worker_recovers_authorization_from_enqueued_handoff(workflow_fixture) -> None:
    controller = WorkflowController(
        workflow_fixture["repository"], workflow_fixture["keys"]
    )
    view = _create(controller, "authorization")
    payment = _approve(controller, view.workflow_id, "authorization-plan")
    controller._run_outbox_operation = lambda _: None
    interrupted = _approve(controller, view.workflow_id, "authorization-payment")
    assert interrupted.state == WorkflowState.PAYMENT_AUTHORIZING
    approval = workflow_fixture["repository"].payment_approval(view.workflow_id)
    operation_id = f"authorize:{approval['payment_approval_id']}"
    _drain(workflow_fixture["repository"], workflow_fixture["keys"], operation_id)
    recovered = workflow_fixture["repository"].get_workflow(view.workflow_id)
    assert recovered["state"] == WorkflowState.COMPLETED
    assert workflow_fixture["repository"].counts(view.workflow_id)["settlements"] == 1
    assert workflow_fixture["repository"].outbox_row(operation_id)["status"] == "done"


@pytest.mark.parametrize(
    ("phase", "checkpoint"),
    [
        ("merchant", "external:merchant-start-returned"),
        ("merchant", "state:payment_approval_required"),
        ("authorization", "state:payment_approved"),
        ("authorization", "external:payment-submit-returned"),
        ("authorization", "state:payment_submitted"),
        ("authorization", "state:payment_verifying"),
        ("authorization", "state:fulfillment_preparing"),
        ("authorization", "state:payment_settling"),
        ("authorization", "external:settlement-returned"),
        ("authorization", "state:fulfillment_committing"),
        ("authorization", "external:fulfillment-commit-returned"),
        ("authorization", "state:completed"),
    ],
)
def test_actual_process_death_replays_same_operation_without_second_effect(
    workflow_fixture, tmp_path: Path, phase: str, checkpoint: str
) -> None:
    repository = workflow_fixture["repository"]
    controller = WorkflowController(repository, workflow_fixture["keys"])
    suffix = checkpoint.replace(":", "-")
    view = _create(controller, f"process-{suffix}")
    controller._run_outbox_operation = lambda _: None
    plan_approved = _approve(controller, view.workflow_id, f"process-plan-{suffix}")
    if phase == "merchant":
        operation_id = f"start:{plan_approved.task_id}"
    else:
        _drain(repository, workflow_fixture["keys"], f"start:{plan_approved.task_id}")
        payment = _approve(controller, view.workflow_id, f"process-payment-{suffix}")
        assert payment.state == WorkflowState.PAYMENT_AUTHORIZING
        approval = repository.payment_approval(view.workflow_id)
        operation_id = f"authorize:{approval['payment_approval_id']}"

    key_dir = tmp_path / "process-keys"
    _persist_keys(key_dir, workflow_fixture["keys"])
    crashed = _outbox_subprocess(workflow_fixture, operation_id, key_dir, checkpoint)
    assert crashed.returncode == 86, crashed.stderr
    recovered = _outbox_subprocess(workflow_fixture, operation_id, key_dir, None)
    assert recovered.returncode == 0, recovered.stderr
    final = repository.get_workflow(view.workflow_id)
    expected = (
        WorkflowState.PAYMENT_APPROVAL_REQUIRED
        if phase == "merchant"
        else WorkflowState.COMPLETED
    )
    assert final["state"] == expected
    assert repository.outbox_row(operation_id)["status"] == "done"
    if phase == "authorization":
        assert repository.counts(view.workflow_id)["settlements"] == 1
        assert repository.rail_balance("demo-customer") == 98_750
        assert repository.rail_balance("demo-merchant") == 1_250


def test_worker_reconciles_evidence_written_before_intent_ack(workflow_fixture) -> None:
    repository = workflow_fixture["repository"]
    evidence_id = "evidence:crash-before-intent-ack"
    repository.put_evidence(
        workflow_id="workflow:evidence-recovery",
        evidence_id=evidence_id,
        tenant_id="demo-tenant",
        kind="crash-fixture",
        exact_bytes=b"exact immutable evidence",
        kid=None,
        media_type="application/octet-stream",
        profile_id="x402-wire-simulation/1",
    )
    with repository.transaction() as conn:
        conn.execute(
            "UPDATE evidence_intents_v2 SET state='pending',committed_at=NULL "
            "WHERE evidence_id=?",
            (evidence_id,),
        )
    assert repository.evidence_intent_health()["pending"] == 1
    assert repository.reconcile_evidence_intents() == {
        "resolved": 1,
        "missing": 0,
        "corrupt": 0,
    }
    assert repository.evidence_intent_health() == {"pending": 0, "failed": 0}


@pytest.mark.parametrize(
    "checkpoint",
    [
        WorkflowState.PAYMENT_APPROVED,
        WorkflowState.PAYMENT_SUBMITTED,
        WorkflowState.PAYMENT_VERIFYING,
        WorkflowState.FULFILLMENT_PREPARING,
        WorkflowState.PAYMENT_SETTLING,
        WorkflowState.FULFILLMENT_COMMITTING,
    ],
)
def test_worker_resumes_each_payment_checkpoint_without_second_effect(
    workflow_fixture, checkpoint: WorkflowState
) -> None:
    repository = workflow_fixture["repository"]
    controller = WorkflowController(repository, workflow_fixture["keys"])
    suffix = checkpoint.value
    view = _create(controller, suffix)
    payment = _approve(controller, view.workflow_id, f"{suffix}-plan")
    original_transition = repository.transition
    crashed = False

    def crash_after_transition(*args, **kwargs):
        nonlocal crashed
        result = original_transition(*args, **kwargs)
        if not crashed and str(kwargs.get("to_state")) == str(checkpoint):
            crashed = True
            raise RuntimeError(f"simulated process death after {checkpoint}")
        return result

    repository.transition = crash_after_transition
    with pytest.raises(RuntimeError, match="simulated process death"):
        _approve(controller, view.workflow_id, f"{suffix}-payment")
    repository.transition = original_transition
    assert repository.get_workflow(view.workflow_id)["state"] == checkpoint
    approval = repository.payment_approval(view.workflow_id)
    operation_id = f"authorize:{approval['payment_approval_id']}"
    _drain(repository, workflow_fixture["keys"], operation_id)
    assert repository.get_workflow(view.workflow_id)["state"] == WorkflowState.COMPLETED
    assert repository.counts(view.workflow_id)["settlements"] == 1
    assert repository.rail_balance("demo-customer") == 98_750
    assert repository.rail_balance("demo-merchant") == 1_250
    assert repository.outbox_row(operation_id)["status"] == "done"


def test_orphan_scanner_recovers_plan_and_free_transient_states(workflow_fixture) -> None:
    repository = workflow_fixture["repository"]
    controller = WorkflowController(repository, workflow_fixture["keys"])
    paid = _create(controller, "orphan-plan")
    controller._start_paid_task = lambda *args, **kwargs: None
    orphaned = _approve(controller, paid.workflow_id, "orphan-plan")
    assert orphaned.state == WorkflowState.PLAN_APPROVED
    recovery = WorkflowController(repository, workflow_fixture["keys"])
    recovery.recover_workflow(repository.recoverable_workflow())
    assert repository.get_workflow(paid.workflow_id)["state"] == WorkflowState.PAYMENT_APPROVAL_REQUIRED

    free = _create(recovery, "orphan-free", paid=False)
    original_transition = repository.transition
    crashed = False

    def crash_in_free(*args, **kwargs):
        nonlocal crashed
        result = original_transition(*args, **kwargs)
        if not crashed and str(kwargs.get("to_state")) == str(WorkflowState.FREE_EXECUTING):
            crashed = True
            raise RuntimeError("simulated free execution death")
        return result

    repository.transition = crash_in_free
    with pytest.raises(RuntimeError):
        _approve(recovery, free.workflow_id, "orphan-free")
    repository.transition = original_transition
    recovery.recover_workflow(repository.recoverable_workflow())
    recovery.recover_workflow(repository.recoverable_workflow())
    assert repository.get_workflow(free.workflow_id)["state"] == WorkflowState.COMPLETED
