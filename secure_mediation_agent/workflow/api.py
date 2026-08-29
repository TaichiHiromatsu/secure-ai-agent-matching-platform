"""Authenticated public API for the authoritative mediation workflow."""

from __future__ import annotations

import os
import json
import hashlib
import inspect
import stat
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Awaitable, Callable, Literal

import httpx
from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from secure_mediation_agent.ap2.keys import DemoKeySet, ROLE_KIDS
from secure_mediation_agent.identity import VerifiedIdentity, verify_identity_assertion
from secure_mediation_agent.merchant.client import HttpPaidBookingMerchant
from secure_mediation_agent.mediation.controller import (
    MediationController as SessionMediationController,
)
from secure_mediation_agent.mediation.errors import MediationError, SecurityBlocked
from secure_mediation_agent.mediation.models import (
    MediationPublicView,
    PendingAction,
    SubjectScope,
    TextPart as MediationTextPart,
    TraceEvent,
)
from secure_mediation_agent.mediation.store import InMemoryMediationStore

from .controller import Identity, WorkflowController
from .errors import DomainError
from .migrations import DatabasePaths, verify
from .models import MessagePart, PublicWorkflowView, WorkflowRequest
from .repository import WorkflowRepository


class ApiModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True)


class CreateWorkflowBody(ApiModel):
    request: WorkflowRequest
    session_id: StrictStr = Field(alias="sessionId", min_length=1)
    context_id: StrictStr = Field(alias="contextId", min_length=1)


class WorkflowMessageBody(ApiModel):
    message_id: StrictStr = Field(alias="messageId", min_length=1)
    parts: list[MessagePart] = Field(min_length=1)
    expected_version: StrictInt | None = Field(default=None, alias="expectedVersion", ge=1)


class MediationTurnMessage(ApiModel):
    parts: list[MediationTextPart] = Field(min_length=1)


class MediationTurnBody(ApiModel):
    schema_version: Literal["mediation-turn-request/1"] = Field(
        default="mediation-turn-request/1", alias="schemaVersion"
    )
    request_id: StrictStr = Field(alias="requestId", min_length=8, max_length=256)
    expected_version: StrictInt | None = Field(
        default=None, alias="expectedVersion", ge=0
    )
    message: MediationTurnMessage
    selection_token: None = Field(default=None, alias="selectionToken")


class MediationTurnResponse(ApiModel):
    schema_version: Literal["mediation-turn-response/1"] = Field(
        default="mediation-turn-response/1", alias="schemaVersion"
    )
    request_id: StrictStr = Field(alias="requestId")
    mediation_session_id: StrictStr = Field(alias="mediationSessionId")
    state: StrictStr
    version: StrictInt
    pending_action: PendingAction = Field(alias="pendingAction")
    view: MediationPublicView
    trace: tuple[TraceEvent, ...]
    error: None = None


@dataclass(slots=True)
class WorkflowRuntime:
    controller: WorkflowController
    paths: DatabasePaths
    identity_verifier_key: object
    durable_marker: Path | None
    evidence_durable_marker: Path | None = None
    keys: DemoKeySet | None = None
    key_directory: Path | None = None
    merchant_probe: Callable[[], bool] | None = None
    public_route_probe: Callable[[], bool | Awaitable[bool]] | None = None
    spec_manifest: Path | None = None
    allow_ephemeral_test_dependencies: bool = False
    ephemeral_cloud_run_demo: bool = False
    mediation_controller: SessionMediationController | None = None


VERTEX_ADC_ENVIRONMENT = {
    "GOOGLE_GENAI_USE_VERTEXAI": "true",
    "GOOGLE_CLOUD_PROJECT": "gen-lang-client-0585901015",
    "GOOGLE_CLOUD_LOCATION": "global",
}
FORBIDDEN_VERTEX_API_KEY_ENVIRONMENT = ("GOOGLE_API_KEY", "GEMINI_API_KEY")


def _vertex_adc_configuration_ready() -> bool:
    return all(
        os.environ.get(name) == value
        for name, value in VERTEX_ADC_ENVIRONMENT.items()
    ) and all(name not in os.environ for name in FORBIDDEN_VERTEX_API_KEY_ENVIRONMENT)


def _default_runtime() -> WorkflowRuntime:
    paths = DatabasePaths.resolve(
        os.environ.get("PAYMENT_MARKETPLACE_DB", "/app/payment-data/marketplace.db"),
        os.environ.get("PAYMENT_MERCHANT_DB", "/app/payment-data/paid-agent.db"),
        os.environ.get("PAYMENT_EVIDENCE_DB", "/app/payment-evidence/evidence.db"),
    )
    keys = DemoKeySet.from_environment()
    marker_value = os.environ.get("PAYMENT_DURABLE_VOLUME_MARKER")
    marker = Path(marker_value).resolve() if marker_value else None
    evidence_marker_value = os.environ.get("PAYMENT_EVIDENCE_VOLUME_MARKER")
    evidence_marker = (
        Path(evidence_marker_value).resolve() if evidence_marker_value else None
    )
    merchant = HttpPaidBookingMerchant(
        os.environ.get("PAYMENT_MERCHANT_A2A_URL", "http://127.0.0.1:8005")
    )
    repository = WorkflowRepository(paths)
    from secure_mediation_agent.mediation.composition import (
        create_production_controller,
    )

    return WorkflowRuntime(
        controller=WorkflowController(
            repository,
            keys,
            merchant=merchant,
        ),
        paths=paths,
        identity_verifier_key=keys.service_auth,
        durable_marker=marker,
        evidence_durable_marker=evidence_marker,
        keys=keys,
        key_directory=Path(os.environ["AP2_DEMO_KEY_DIR"]).resolve(),
        merchant_probe=merchant.health,
        public_route_probe=lambda: _probe_public_routes(
            os.environ.get("PUBLIC_EDGE_PROBE_URL", "http://127.0.0.1:8080")
        ),
        spec_manifest=Path(__file__).resolve().parents[1] / "spec_manifest.json",
        ephemeral_cloud_run_demo=(
            os.environ.get("EPHEMERAL_CLOUD_RUN_DEMO") == "true"
        ),
        mediation_controller=create_production_controller(
            repository=repository,
            keys=keys,
        ),
    )


def _spec_pins_ready(path: Path | None) -> tuple[bool, dict[str, str]]:
    expected = {
        "ap2": "32c3be5011f481d2760e56e7b9935b0842c3da0d5f7d7b8a68402a599f1e6aa3",
        "x402": "5cdc35ed8c4d7a93bb120f1782fd06e2cc3ef19036684f772e27d0d644c66940",
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path else {}
        observed = {name: payload[name]["sha256"] for name in expected}
        enabled = payload.get("officialX402") == "DISABLED / NOT RUN"
    except Exception:
        return False, {}
    return observed == expected and enabled, observed


def _keys_ready(runtime: WorkflowRuntime) -> bool:
    if runtime.keys is None:
        return False
    for role, kid in ROLE_KIDS.items():
        key = getattr(runtime.keys, role)
        if key.get("kid") != kid or key.get("crv") != "P-256" or not key.has_private:
            return False
    if runtime.key_directory is None:
        return runtime.allow_ephemeral_test_dependencies
    try:
        for role in ROLE_KIDS:
            path = runtime.key_directory / f"{role}.jwk"
            if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
                return False
    except OSError:
        return False
    return True


async def _probe_public_routes(
    base_url: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> bool:
    """Verify the live edge method/path boundary without parsing nginx text."""

    probes = (
        ("GET", "/mediation-api/v1/view", {200, 401}),
        ("POST", "/mediation-api/v1/turns", {401, 403, 422}),
        ("GET", "/mediation-api/v1/turns", {404}),
        ("POST", "/mediation-api/v1/view", {404}),
        ("POST", "/run", {401, 403, 422}),
        ("GET", "/run", {404}),
        ("GET", "/v1/view", {404}),
        ("GET", "/payment/internal", {404}),
        ("GET", "/paid-agent/internal", {404}),
        ("GET", "/internal/authority", {404}),
    )
    try:
        async with httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(2.0),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        ) as client:
            for method, path, accepted in probes:
                response = await client.request(method, path)
                if response.status_code not in accepted:
                    return False
    except httpx.HTTPError:
        return False
    return True


async def _routes_ready(runtime: WorkflowRuntime) -> bool:
    if runtime.public_route_probe is None:
        return runtime.allow_ephemeral_test_dependencies
    try:
        result = runtime.public_route_probe()
        if inspect.isawaitable(result):
            result = await result
        return result is True
    except Exception:
        return False


def create_app(runtime: WorkflowRuntime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.runtime is None:
            try:
                application.state.runtime = _default_runtime()
            except Exception as error:  # readiness reports only the safe class
                application.state.startup_error = type(error).__name__
        yield

    app = FastAPI(
        title="Secure mediation workflow", version="2.0.0", lifespan=lifespan
    )
    app.state.runtime = runtime
    app.state.startup_error = None

    def configured_runtime() -> WorkflowRuntime:
        value = app.state.runtime
        if value is None:
            raise DomainError(
                "SERVICE_NOT_READY",
                "Workflow service is not configured.",
                "startup",
            )
        return value

    def identity(
        assertion: Annotated[str | None, Header(alias="X-Verified-Identity")] = None,
        configured: WorkflowRuntime = Depends(configured_runtime),
    ) -> VerifiedIdentity:
        if not assertion:
            raise DomainError("TENANT_BINDING_MISMATCH", "Verified identity is required.", "identity")
        try:
            return verify_identity_assertion(assertion, configured.identity_verifier_key)
        except Exception as error:
            raise DomainError(
                "TENANT_BINDING_MISMATCH",
                "Verified identity assertion is invalid.",
                "identity",
            ) from error

    def domain_identity(value: VerifiedIdentity) -> Identity:
        return Identity(tenant_id=value.tenant_id, customer_id=value.customer_id)

    def mediation_scope(value: VerifiedIdentity) -> SubjectScope:
        digest = hashlib.sha256(
            f"{value.tenant_id}\0{value.subject}".encode("utf-8")
        ).hexdigest()
        return SubjectScope(
            subject=value.subject,
            tenantId=value.tenant_id,
            adkSessionId=f"public-{digest[:32]}",
        )

    def mediation_controller(
        configured: WorkflowRuntime = Depends(configured_runtime),
    ) -> SessionMediationController:
        if configured.mediation_controller is None:
            raise DomainError(
                "SERVICE_NOT_READY",
                "Session mediation composition is not configured.",
                "mediation",
            )
        return configured.mediation_controller

    @app.exception_handler(DomainError)
    async def handle_domain_error(_: Request, error: DomainError) -> JSONResponse:
        return JSONResponse(status_code=error.http_status, content={"error": error.envelope()})

    @app.exception_handler(KeyError)
    async def handle_missing(_: Request, __: KeyError) -> JSONResponse:
        # Do not disclose whether a cross-tenant opaque identifier exists.
        error = DomainError("WORKFLOW_NOT_FOUND", "Workflow was not found.", "workflow")
        return JSONResponse(status_code=404, content={"error": error.envelope()})

    @app.exception_handler(MediationError)
    async def handle_mediation_error(_: Request, error: MediationError) -> JSONResponse:
        return JSONResponse(
            status_code=403 if isinstance(error, SecurityBlocked) else 409,
            content={
                "error": {
                    "code": error.code,
                    "message": error.safe_message,
                    "retryable": False,
                }
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(configured: WorkflowRuntime = Depends(configured_runtime)) -> JSONResponse:
        data_durable = bool(
            configured.durable_marker and configured.durable_marker.is_file()
        )
        evidence_durable = bool(
            configured.evidence_durable_marker
            and configured.evidence_durable_marker.is_file()
        )
        try:
            schemas = verify(configured.paths)
        except Exception:
            schemas = {}
        repository = configured.controller.repository
        try:
            outbox = repository.outbox_health()
            evidence_intents = repository.evidence_intent_health()
            trust = repository.trust_health()
        except Exception:
            outbox = {"liveWorkers": 0, "staleLeases": 1, "failed": 1, "overdue": 1}
            evidence_intents = {"pending": 1, "failed": 1}
            trust = {"missing": 1, "corrupt": 1}
        outbox_ready = (
            outbox["liveWorkers"] >= 1
            and outbox["staleLeases"] == 0
            and outbox["failed"] == 0
            and outbox["overdue"] == 0
        )
        evidence_ready = evidence_intents == {"pending": 0, "failed": 0}
        trust_ready = trust == {"missing": 0, "corrupt": 0}
        spec_path = configured.spec_manifest or (
            Path(__file__).resolve().parents[1] / "spec_manifest.json"
        )
        spec_ready, spec_hashes = _spec_pins_ready(spec_path)
        key_ready = _keys_ready(configured)
        routes_ready = await _routes_ready(configured)
        try:
            profile_status = configured.controller.profile.readiness()
            profile_ready = (
                profile_status.ready
                and profile_status.profile_id == "x402-wire-simulation/1"
                and profile_status.rail_mode == "simulated"
            )
        except Exception:
            profile_ready = False
        merchant_ready = bool(
            configured.merchant_probe and configured.merchant_probe()
        ) or configured.allow_ephemeral_test_dependencies
        mediation = configured.mediation_controller
        mediation_store = getattr(mediation, "store", None)
        local_ephemeral_allowed = (
            os.environ.get("APP_ENV") == "local"
            and os.environ.get("DEV_MODE", "false").lower() == "true"
        )
        if (
            mediation_store is None
            and configured.allow_ephemeral_test_dependencies
            and not configured.ephemeral_cloud_run_demo
        ):
            mediation_store_status = {
                "mode": "test-unconfigured",
                "durabilityProfile": "ephemeral-demo",
                "schemaVersion": None,
                "writable": True,
                "decryptable": True,
            }
            mediation_mode_ready = True
            mediation_profile_ready = True
            mediation_schema_ready = True
            mediation_probe_ready = True
        else:
            memory_store = isinstance(mediation_store, InMemoryMediationStore)
            mode = "memory" if memory_store else getattr(
                mediation_store, "kind", "unknown"
            )
            durability_profile = getattr(
                mediation,
                "durability_profile",
                getattr(mediation_store, "durability_profile", "unknown"),
            )
            if memory_store:
                # For the process-local demo store these booleans mean the
                # in-process store is available; no encrypted durable rows exist.
                schema_version = None
                writable = True
                decryptable = True
            else:
                try:
                    probe_method = getattr(mediation_store, "readiness_probe")
                    probe = probe_method()
                    schema_version = probe.schema_version
                    writable = probe.writable
                    decryptable = probe.decryptable
                    mode = probe.kind
                except Exception:
                    schema_version = None
                    writable = False
                    decryptable = False
            mediation_store_status = {
                "mode": mode,
                "durabilityProfile": durability_profile,
                "schemaVersion": schema_version,
                "writable": writable,
                "decryptable": decryptable,
            }
            test_memory = (
                configured.allow_ephemeral_test_dependencies and memory_store
            )
            local_memory = local_ephemeral_allowed and memory_store
            ephemeral_target = configured.ephemeral_cloud_run_demo
            expected_mode = (
                "memory"
                if ephemeral_target or test_memory or local_memory
                else "sqlite"
            )
            expected_profile = (
                "ephemeral-demo" if expected_mode == "memory" else "local-durable"
            )
            expected_schema = None if expected_mode == "memory" else 4
            mediation_mode_ready = mode == expected_mode
            mediation_profile_ready = durability_profile == expected_profile
            mediation_schema_ready = schema_version == expected_schema
            mediation_probe_ready = writable is True and decryptable is True
        common_checks = {
            "schemas": schemas == {"marketplace": 4, "merchant": 4, "evidence": 4},
            "outboxRecovery": outbox_ready,
            "evidenceIntents": evidence_ready,
            "roleKeys": key_ready,
            "trustSnapshots": trust_ready,
            "specPins": spec_ready,
            "selectedProfileOnly": profile_ready,
            "routeIsolation": routes_ready,
            "merchantA2AAndTaskStore": merchant_ready,
            "mediationComposition": (
                configured.mediation_controller is not None
                or configured.allow_ephemeral_test_dependencies
            ),
            "mediationStoreMode": mediation_mode_ready,
            "mediationStoreProfile": mediation_profile_ready,
            "mediationStoreSchema": mediation_schema_ready,
            "mediationStoreProbe": mediation_probe_ready,
        }
        if configured.ephemeral_cloud_run_demo:
            checks = {
                "ephemeralDataPathWritable": os.access(
                    configured.paths.marketplace.parent, os.W_OK
                ),
                "ephemeralEvidencePathWritable": os.access(
                    configured.paths.evidence.parent, os.W_OK
                ),
                "vertexAdcConfiguration": _vertex_adc_configuration_ready(),
                **common_checks,
            }
        else:
            checks = {
                "dataDurableVolume": data_durable,
                "evidenceDurableVolume": evidence_durable,
                **common_checks,
            }
        is_ready = all(checks.values())
        content = {
            "status": "ready" if is_ready else "not-ready",
            "target": (
                "ephemeral-cloud-run-demo"
                if configured.ephemeral_cloud_run_demo
                else "explicit-durable-single-host-single-container"
            ),
            "checks": checks,
            "schemas": schemas,
            "outbox": outbox,
            "evidenceIntents": evidence_intents,
            "trust": trust,
            "specHashes": spec_hashes,
            "profile": "x402-wire-simulation/1",
            "mediationStore": mediation_store_status,
            "railMode": "simulated",
            "officialX402": "NOT RUN",
            "wallet": "NOT RUN",
            "facilitator": "NOT RUN",
            "onChain": "NOT RUN",
        }
        if configured.ephemeral_cloud_run_demo:
            content.update(
                {
                    "durability": "NOT PROVIDED",
                    "notice": "EPHEMERAL DEMO: state and keys may reset on restart",
                }
            )
        else:
            content.update(
                {
                    "durableVolumeMarker": "PASS" if data_durable else "MISSING",
                    "evidenceDurableVolumeMarker": (
                        "PASS" if evidence_durable else "MISSING"
                    ),
                }
            )
        return JSONResponse(status_code=200 if is_ready else 503, content=content)

    @app.post(
        "/v1/turns",
        response_model=MediationTurnResponse,
        response_model_by_alias=True,
    )
    async def submit_mediation_turn(
        body: MediationTurnBody,
        idempotency_key: Annotated[
            str, Header(alias="Idempotency-Key", min_length=8)
        ],
        request_header: Annotated[
            str | None, Header(alias="X-Request-ID", min_length=8)
        ] = None,
        verified: VerifiedIdentity = Depends(identity),
        controller: SessionMediationController = Depends(mediation_controller),
    ) -> MediationTurnResponse:
        if body.request_id != idempotency_key or (
            request_header is not None and request_header != body.request_id
        ):
            raise DomainError(
                "IDEMPOTENCY_CONFLICT",
                "Turn request identifiers do not match.",
                body.request_id,
            )
        scope = mediation_scope(verified)
        view = await controller.submit(
            scope=scope,
            parts=body.message.parts,
            request_id=body.request_id,
            expected_version=body.expected_version,
        )
        request_result = controller.completed_request_result(
            scope=scope,
            parts=body.message.parts,
            request_id=body.request_id,
            expected_version=body.expected_version,
        )
        if (
            request_result is None
            or request_result.status != "completed"
            or request_result.mediation_session_id is None
            or request_result.result_version != view.version
            or request_result.view != view
        ):
            raise DomainError(
                "SERVICE_NOT_READY",
                "The mediation result was not persisted.",
                body.request_id,
            )
        return MediationTurnResponse(
            requestId=body.request_id,
            mediationSessionId=request_result.mediation_session_id,
            state=view.state.value,
            version=view.version,
            pendingAction=view.pending_action,
            view=view,
            trace=view.trace,
        )

    @app.get(
        "/v1/view",
        response_model=MediationPublicView | None,
        response_model_by_alias=True,
    )
    async def active_mediation_view(
        verified: VerifiedIdentity = Depends(identity),
        controller: SessionMediationController = Depends(mediation_controller),
    ) -> MediationPublicView | None:
        scope = mediation_scope(verified)
        session = controller.store.active_for(scope) or controller.store.latest_for(
            scope
        )
        return controller.public_view(session) if session is not None else None

    @app.post("/v1/workflows", response_model=PublicWorkflowView, response_model_by_alias=True)
    async def create_workflow(
        body: CreateWorkflowBody,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8)],
        verified: VerifiedIdentity = Depends(identity),
        configured: WorkflowRuntime = Depends(configured_runtime),
    ) -> PublicWorkflowView:
        return configured.controller.create(
            body.request,
            identity=domain_identity(verified),
            session_id=body.session_id,
            context_id=body.context_id,
            idempotency_key=idempotency_key,
        )

    @app.get("/v1/workflows/active", response_model=PublicWorkflowView | None, response_model_by_alias=True)
    async def active_workflow(
        session_id: Annotated[str, Query(alias="sessionId", min_length=1)],
        context_id: Annotated[str, Query(alias="contextId", min_length=1)],
        verified: VerifiedIdentity = Depends(identity),
        configured: WorkflowRuntime = Depends(configured_runtime),
    ) -> PublicWorkflowView | None:
        return configured.controller.active(
            identity=domain_identity(verified),
            session_id=session_id,
            context_id=context_id,
        )

    @app.get("/v1/workflows/{workflow_id}", response_model=PublicWorkflowView, response_model_by_alias=True)
    async def get_workflow(
        workflow_id: str,
        verified: VerifiedIdentity = Depends(identity),
        configured: WorkflowRuntime = Depends(configured_runtime),
    ) -> PublicWorkflowView:
        return configured.controller.get(workflow_id, identity=domain_identity(verified))

    @app.post(
        "/v1/workflows/{workflow_id}/messages",
        response_model=PublicWorkflowView,
        response_model_by_alias=True,
    )
    async def send_message(
        workflow_id: str,
        body: WorkflowMessageBody,
        idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=8)],
        verified: VerifiedIdentity = Depends(identity),
        configured: WorkflowRuntime = Depends(configured_runtime),
    ) -> PublicWorkflowView:
        return configured.controller.message(
            workflow_id,
            body.parts,
            identity=domain_identity(verified),
            message_id=body.message_id,
            idempotency_key=idempotency_key,
            expected_version=body.expected_version,
        )

    return app


app = create_app()
