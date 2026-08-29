"""Supervised durable outbox worker for restart-safe workflow effects."""

from __future__ import annotations

import logging
import os
import signal
import time
from pathlib import Path

from secure_mediation_agent.ap2.keys import DemoKeySet
from secure_mediation_agent.merchant.client import HttpPaidBookingMerchant

from .controller import WorkflowController
from .migrations import DatabasePaths
from .repository import WorkflowRepository


LOGGER = logging.getLogger("workflow-outbox-worker")


def _paths() -> DatabasePaths:
    return DatabasePaths.resolve(
        os.environ.get("PAYMENT_MARKETPLACE_DB", "/app/payment-data/marketplace.db"),
        os.environ.get("PAYMENT_MERCHANT_DB", "/app/payment-data/paid-agent.db"),
        os.environ.get("PAYMENT_EVIDENCE_DB", "/app/payment-evidence/evidence.db"),
    )


def run() -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    marker_value = os.environ.get("PAYMENT_DURABLE_VOLUME_MARKER")
    evidence_marker_value = os.environ.get("PAYMENT_EVIDENCE_VOLUME_MARKER")
    ephemeral_demo = os.environ.get("EPHEMERAL_CLOUD_RUN_DEMO") == "true"
    if not ephemeral_demo:
        if not marker_value or not Path(marker_value).is_file():
            raise RuntimeError("payment data durable-volume marker is required")
        if not evidence_marker_value or not Path(evidence_marker_value).is_file():
            raise RuntimeError("payment evidence durable-volume marker is required")

    repository = WorkflowRepository(_paths())
    controller = WorkflowController(
        repository,
        DemoKeySet.from_environment(),
        merchant=HttpPaidBookingMerchant(
            os.environ.get("PAYMENT_MERCHANT_A2A_URL", "http://127.0.0.1:8005")
        ),
    )
    worker_id = os.environ.get(
        "PAYMENT_OUTBOX_WORKER_ID", f"workflow-outbox:{os.getpid()}"
    )
    stopping = False
    lease_seconds = int(os.environ.get("PAYMENT_OUTBOX_LEASE_SECONDS", "120"))

    def stop(_: int, __: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    repository.heartbeat_worker(worker_id, status="starting")
    while not stopping:
        repository.heartbeat_worker(worker_id)
        repository.reconcile_evidence_intents()
        row = repository.lease_outbox(worker_id, lease_seconds=lease_seconds)
        if row is None:
            recoverable = repository.recoverable_workflow()
            if recoverable is not None:
                controller.recover_workflow(recoverable)
                continue
            time.sleep(0.25)
            continue
        repository.heartbeat_worker(worker_id, operation_id=row["operation_id"])
        try:
            controller.process_leased_outbox(row, worker_id)
        except Exception as error:
            LOGGER.exception(
                "outbox operation %s will retry: %s",
                row["operation_id"],
                type(error).__name__,
            )
    repository.heartbeat_worker(worker_id, status="stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
