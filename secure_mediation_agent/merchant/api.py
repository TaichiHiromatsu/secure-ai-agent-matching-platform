"""Private A2A HTTP boundary for the selected paid booking Merchant."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from a2a.types import Message
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from secure_mediation_agent.ap2.keys import DemoKeySet
from secure_mediation_agent.payment_profiles.registry import ProfileRegistry
from secure_mediation_agent.workflow.approval import AuthorizationService
from secure_mediation_agent.workflow.errors import DomainError
from secure_mediation_agent.workflow.migrations import DatabasePaths, verify
from secure_mediation_agent.workflow.repository import WorkflowRepository

from .service import PaidBookingMerchant


@dataclass(slots=True)
class MerchantRuntime:
    service: PaidBookingMerchant
    authorization: AuthorizationService
    paths: DatabasePaths
    extension_uri: str


def _default_runtime() -> MerchantRuntime:
    paths = DatabasePaths.resolve(
        os.environ.get("PAYMENT_MARKETPLACE_DB", "/app/payment-data/marketplace.db"),
        os.environ.get("PAYMENT_MERCHANT_DB", "/app/payment-data/paid-agent.db"),
        os.environ.get("PAYMENT_EVIDENCE_DB", "/app/payment-evidence/evidence.db"),
    )
    keys = DemoKeySet.from_environment()
    repository = WorkflowRepository(paths)
    profile = ProfileRegistry.load(
        "x402-wire-simulation/1", simulation_key=keys.simulation_signer
    )
    return MerchantRuntime(
        service=PaidBookingMerchant(repository, keys, profile),
        authorization=AuthorizationService(keys.plan_authority),
        paths=paths,
        extension_uri=profile.extension_uri,
    )


def create_app(runtime: MerchantRuntime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if application.state.runtime is None:
            try:
                application.state.runtime = _default_runtime()
            except Exception as error:
                application.state.startup_error = type(error).__name__
        yield

    app = FastAPI(title="Paid booking A2A Merchant", version="2.0.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.startup_error = None

    def configured() -> MerchantRuntime:
        value = app.state.runtime
        if value is None:
            raise DomainError(
                "SERVICE_NOT_READY", "Merchant service is not configured.", "merchant"
            )
        return value

    @app.exception_handler(DomainError)
    async def domain_error(_: Request, error: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=error.http_status, content={"error": error.envelope()}
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "paid-booking-agent"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        try:
            value = configured()
            schemas = verify(value.paths)
            profile = value.service._profile.readiness()
            ok = schemas == {"marketplace": 2, "merchant": 2, "evidence": 2} and profile.ready
        except Exception:
            schemas = {}
            ok = False
        return JSONResponse(
            status_code=200 if ok else 503,
            content={
                "status": "ready" if ok else "not-ready",
                "service": "paid-booking-agent",
                "taskStore": "sqlite-v2",
                "profile": "x402-wire-simulation/1",
                "officialX402": "NOT RUN",
                "schemas": schemas,
            },
        )

    @app.get("/.well-known/agent-card.json")
    async def agent_card() -> dict[str, Any]:
        return configured().service.agent_card().model_dump(
            mode="json", by_alias=True, exclude_none=True
        )

    @app.post("/a2a")
    async def a2a(
        request: Request,
        authorization_header: str | None = Header(default=None, alias="Authorization"),
        activation: str | None = Header(default=None, alias="X-A2A-Extensions"),
    ) -> JSONResponse:
        runtime_value = configured()
        if activation != runtime_value.extension_uri:
            raise DomainError(
                "X402_ACTIVATION_MISMATCH",
                "Selected payment extension activation is required.",
                "merchant",
            )
        if not authorization_header or not authorization_header.startswith("Bearer "):
            raise DomainError(
                "CAPABILITY_MISSING", "A signed service capability is required.", "merchant"
            )
        body = await request.json()
        if body.get("jsonrpc") != "2.0" or body.get("method") != "message/send":
            raise DomainError(
                "A2A_REQUEST_INVALID", "A2A JSON-RPC envelope is invalid.", "merchant"
            )
        params = body.get("params")
        if not isinstance(params, dict):
            raise DomainError("A2A_REQUEST_INVALID", "A2A params are required.", "merchant")
        action = params.get("action")
        supported = {
            "merchant-task:start",
            "merchant:payment-submit",
            "merchant:fulfillment-prepare",
            "merchant:fulfillment-commit",
        }
        if action not in supported:
            raise DomainError(
                "A2A_REQUEST_INVALID", "A2A operation is unsupported.", str(action)
            )
        token = authorization_header.removeprefix("Bearer ")
        try:
            claims = runtime_value.authorization.verify(
                token,
                expected_type="secure-downstream-capability+jwt",
                audience="merchant:demo-merchant",
                operation=action,
                now=int(time.time()),
            )
        except Exception as error:
            raise DomainError(
                "CAPABILITY_INVALID",
                "Signed service capability is invalid or expired.",
                str(body.get("id")),
            ) from error
        capability_record = runtime_value.service._repository.capability_record(
            str(claims.get("jti"))
        )
        if capability_record and capability_record["status"] == "invalidated":
            raise DomainError(
                "CAPABILITY_REVOKED",
                "Signed service capability has been revoked.",
                str(body.get("id")),
            )
        for claim_name, param_name in (
            ("workflowId", "workflowId"),
            ("taskId", "taskId"),
            ("orderId", "orderId"),
        ):
            if param_name in params and params.get(param_name) != claims.get(claim_name):
                raise DomainError(
                    "CAPABILITY_BINDING_MISMATCH",
                    "Service capability does not authorize this A2A Task.",
                    str(body.get("id")),
                )
        if params.get("capabilityId") and params["capabilityId"] != claims.get("jti"):
            raise DomainError(
                "CAPABILITY_BINDING_MISMATCH",
                "Capability identifier does not match the signed grant.",
                str(body.get("id")),
            )

        service = runtime_value.service
        if action == "merchant-task:start":
            if "checkoutJwt" in params:
                result: dict[str, Any] = {
                    "claims": service.verify_checkout(
                        params["checkoutJwt"],
                        workflow_id=params["workflowId"],
                        plan_digest=params["planDigest"],
                        task_id=params["taskId"],
                    )
                }
            else:
                started = service.start_task(
                    workflow_id=params["workflowId"],
                    plan_digest=params["planDigest"],
                    task_id=params["taskId"],
                    order_id=params["orderId"],
                    context_id=params["contextId"],
                    capability_id=claims["jti"],
                    activation={activation},
                    issued_at=int(params["issuedAt"]),
                    expires_at=int(params["expiresAt"]),
                    capability_token=token,
                )
                result = {
                    "task": started.task.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    ),
                    "checkoutJwt": started.checkout_jwt,
                    "checkoutHash": started.checkout_hash,
                    "requirements": started.requirements,
                    "activationEcho": started.activation_echo,
                    "checkoutChallenge": started.checkout_challenge,
                    "paymentChallenge": started.payment_challenge,
                }
        elif action == "merchant:payment-submit":
            service.submit_payment(
                message=Message.model_validate(params["message"]),
                checkout_mandate=params["checkoutMandate"],
                checkout_jwt=params["checkoutJwt"],
                checkout_nonce=params["checkoutNonce"],
                capability_id=claims["jti"],
                workflow_id=params["workflowId"],
                order_id=params["orderId"],
            )
            result = {"accepted": True}
        elif action == "merchant:fulfillment-prepare":
            result = service.prepare(params["taskId"], params["operationId"])
        else:
            task = service.complete_task(
                task_id=params["taskId"],
                context_id=params["contextId"],
                receipts=params["receipts"],
                checkout_receipt_id=params["checkoutReceiptId"],
                payment_receipt_id=params["paymentReceiptId"],
            )
            result = {
                "task": task.model_dump(mode="json", by_alias=True, exclude_none=True)
            }
        return JSONResponse(
            status_code=200,
            content={"jsonrpc": "2.0", "id": body.get("id"), "result": result},
        )

    return app


app = create_app()
