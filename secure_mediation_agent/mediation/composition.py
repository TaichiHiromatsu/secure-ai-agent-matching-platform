"""Production composition kept lazy so importing the ADK agent has no side effects."""

from __future__ import annotations

import os

from .a2a_executor import SharedA2AOperationExecutor
from .adapters import (
    DeterministicStableGate,
    HttpxA2ATransport,
    LegacyCallbackHook,
    LegacyFinalValidationAdapter,
    LegacyMatcherAdapter,
    LocalDeterministicCallbackHook,
    TypedPlannerAdapter,
)
from .controller import MediationController
from .store import InMemoryMediationStore


DEFAULT_MEDIATION_STORE_KEY_FILE = "/run/secrets/ap2-demo/mediation-store.key"


def _ephemeral_store_allowed() -> bool:
    return (
        os.environ.get("APP_ENV") == "local"
        and os.environ.get("DEV_MODE", "false").lower() == "true"
    ) or os.environ.get("EPHEMERAL_CLOUD_RUN_DEMO", "false").lower() == "true"


def _configured_store(repository):
    mode = os.environ.get("MEDIATION_STORE_MODE", "sqlite").strip().lower()
    ephemeral_cloud_run = (
        os.environ.get("EPHEMERAL_CLOUD_RUN_DEMO", "false").lower() == "true"
    )
    if ephemeral_cloud_run and mode != "memory":
        raise RuntimeError(
            "EPHEMERAL_CLOUD_RUN_DEMO=true requires MEDIATION_STORE_MODE=memory"
        )
    if mode == "sqlite":
        from .persistence import SqliteMediationStore

        return SqliteMediationStore(
            repository=repository,
            master_key=os.environ.get(
                "MEDIATION_STORE_KEY_FILE", DEFAULT_MEDIATION_STORE_KEY_FILE
            ),
        )
    if mode == "memory":
        if not _ephemeral_store_allowed():
            raise RuntimeError(
                "MEDIATION_STORE_MODE=memory is restricted to APP_ENV=local with "
                "DEV_MODE=true or EPHEMERAL_CLOUD_RUN_DEMO=true"
            )
        return InMemoryMediationStore()
    raise RuntimeError(f"unsupported MEDIATION_STORE_MODE: {mode}")


def _configured_callback_hook():
    mode = os.environ.get("MEDIATION_CALLBACK_MODE", "legacy")
    if mode == "legacy":
        return LegacyCallbackHook()
    if mode == "deterministic-local":
        if (
            os.environ.get("APP_ENV") != "local"
            or os.environ.get("DEV_MODE", "false").lower() != "true"
        ):
            raise RuntimeError(
                "MEDIATION_CALLBACK_MODE=deterministic-local is restricted to "
                "DEV_MODE=true with APP_ENV=local"
            )
        return LocalDeterministicCallbackHook()
    raise RuntimeError(f"unsupported MEDIATION_CALLBACK_MODE: {mode}")


def create_production_controller(
    *, repository=None, keys=None, store=None, callback_hook=None
) -> MediationController:
    from secure_mediation_agent.ap2.keys import DemoKeySet
    from secure_mediation_agent.payment_bridge import PaymentBridge
    from secure_mediation_agent.workflow.migrations import DatabasePaths
    from secure_mediation_agent.workflow.repository import WorkflowRepository
    from secure_mediation_agent.workflow.approval import AuthorizationService

    from .capability import PlanAuthorityOperationAuthorizer
    from .payment_bridge_adapter import DurablePaymentBridgeAdapter

    paths = DatabasePaths.resolve(
        os.getenv("PAYMENT_MARKETPLACE_DB", "/app/payment-data/marketplace.db"),
        os.getenv("PAYMENT_MERCHANT_DB", "/app/payment-data/paid-agent.db"),
        os.getenv("PAYMENT_EVIDENCE_DB", "/app/payment-evidence/evidence.db"),
    )
    resolved_repository = repository or WorkflowRepository(paths)
    resolved_keys = keys or DemoKeySet.from_environment()
    ephemeral_demo = (
        os.environ.get("EPHEMERAL_CLOUD_RUN_DEMO", "false").lower() == "true"
    )
    configured_store_mode = os.environ.get(
        "MEDIATION_STORE_MODE", "sqlite"
    ).strip().lower()
    if ephemeral_demo and configured_store_mode != "memory":
        raise RuntimeError(
            "EPHEMERAL_CLOUD_RUN_DEMO=true requires MEDIATION_STORE_MODE=memory"
        )
    if isinstance(store, InMemoryMediationStore) and not _ephemeral_store_allowed():
        raise RuntimeError(
            "InMemoryMediationStore is restricted to APP_ENV=local with DEV_MODE=true "
            "or EPHEMERAL_CLOUD_RUN_DEMO=true"
        )
    resolved_store = store if store is not None else _configured_store(resolved_repository)
    if ephemeral_demo and not isinstance(resolved_store, InMemoryMediationStore):
        raise RuntimeError(
            "EPHEMERAL_CLOUD_RUN_DEMO=true requires InMemoryMediationStore"
        )
    durability_profile = (
        "ephemeral-demo"
        if ephemeral_demo or isinstance(resolved_store, InMemoryMediationStore)
        else getattr(resolved_store, "durability_profile", "local-durable")
    )
    payment_bridge = PaymentBridge(resolved_repository, resolved_keys)
    gates = DeterministicStableGate()
    executor = SharedA2AOperationExecutor(
        callback=callback_hook or _configured_callback_hook(),
        gates=gates,
        transport=HttpxA2ATransport(),
        authorizer=PlanAuthorityOperationAuthorizer(
            AuthorizationService(payment_bridge.keys.plan_authority)
        ),
    )
    bridge = DurablePaymentBridgeAdapter(payment_bridge, executor=executor)
    return MediationController(
        store=resolved_store,
        matcher=LegacyMatcherAdapter(),
        planner=TypedPlannerAdapter(),
        executor=executor,
        gates=gates,
        payment_bridge=bridge,
        final_validator=LegacyFinalValidationAdapter(),
        durability_profile=durability_profile,
    )
