"""FastAPI/A2A entry point for marketplace payment mediation (port 8004)."""

from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .a2a_adapter import (
    PROFILE,
    jsonrpc_error,
    jsonrpc_result,
    mediation_agent_card,
    payment_metadata,
    task_result,
)
from .auth import RequestAuthenticationError, verify_request_auth
from .canonical import CanonicalizationError, digest_object, verify_payload_signature
from .config import (
    CUSTOMER_SUBJECT,
    MEDIATOR_SUBJECT,
    MERCHANT_KID,
    MERCHANT_SUBJECT,
    OPERATOR_SUBJECT,
)
from .ledger import Ledger
from .merchant_client import EndpointPolicy, HttpMerchantClient
from .rail import LocalPaymentRail
from .service import MarketplaceError, MarketplaceService
from .store import IdempotencyConflict, MarketplaceStore, ReplayDetected


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_service() -> MarketplaceService:
    business_db = os.getenv("PAYMENT_MARKETPLACE_DB", "/tmp/payment-marketplace.db")
    evidence_db = os.getenv("PAYMENT_EVIDENCE_DB", "/tmp/payment-evidence.db")
    merchant_url = os.getenv("PAYMENT_MERCHANT_URL", "http://127.0.0.1:8005")
    parsed = urlparse(merchant_url)
    allow_loopback = _bool_env("PAYMENT_DEMO_ALLOW_LOOPBACK", False)
    policy = EndpointPolicy(
        allowed_hosts=frozenset({parsed.hostname or ""}),
        allowed_ports=frozenset({parsed.port or (443 if parsed.scheme == "https" else 80)}),
        allow_loopback=allow_loopback,
    )
    store = MarketplaceStore(business_db, evidence_db)
    ledger = Ledger(store)
    rail = LocalPaymentRail(
        store,
        allow_test_faults=_bool_env("PAYMENT_DEMO_ENABLE_RAIL_FAULTS", False),
    )
    service = MarketplaceService(store, ledger, rail, HttpMerchantClient(merchant_url, policy))
    service.seed_demo_onboarding(merchant_url)
    return service


def _decode_auth_header(value: str | None) -> dict[str, Any]:
    if not value:
        raise RequestAuthenticationError("request authentication is required")
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        result = json.loads(raw)
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise RequestAuthenticationError("request authentication encoding is invalid") from exc
    if not isinstance(result, dict):
        raise RequestAuthenticationError("request authentication shape is invalid")
    return result


def _split_signed_body(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(body, dict):
        raise MarketplaceError("INVALID_SCHEMA", "JSON object body is required.")
    business = dict(body)
    auth = business.pop("requestAuth", None)
    if not isinstance(auth, dict):
        raise RequestAuthenticationError("signed requestAuth is required")
    return business, auth


def _require_extension(value: str | None) -> None:
    if value != PROFILE:
        raise MarketplaceError(
            "UNSUPPORTED_EXTENSION",
            "Activate the project-local payment extension with X-A2A-Extensions.",
        )


def _authenticate(
    service: MarketplaceService,
    auth: dict[str, Any],
    *,
    method: str,
    path: str,
    body: dict[str, Any],
    role: str,
    tenant: str,
    operation: str,
    reference: str,
) -> dict[str, str]:
    actor = verify_request_auth(
        auth,
        method=method,
        path=path,
        body=body,
        expected_role=role,
        expected_tenant=tenant,
    )
    service.store.consume_nonce(
        actor["subject"],
        actor["nonce"],
        digest_object(auth),
        order_id=reference,
        task_id=reference,
        operation=operation,
    )
    return actor


def _merchant_payout_auth(
    service: MarketplaceService, document: dict[str, Any], payout_id: str
) -> None:
    try:
        verify_payload_signature(document, expected_kid=MERCHANT_KID)
    except Exception as exc:
        raise RequestAuthenticationError("merchant payout query signature is invalid") from exc
    expected = {
        "profile": PROFILE,
        "simulated": True,
        "skill": "payout_status",
        "method": "GET",
        "path": f"/v1/payouts/{payout_id}",
        "bodyDigest": digest_object({}),
        "issuer": MERCHANT_SUBJECT,
        "audience": MEDIATOR_SUBJECT,
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise RequestAuthenticationError("merchant payout query binding is invalid")
    actor = document.get("actor")
    if actor != {"role": "merchant", "merchantId": MERCHANT_SUBJECT}:
        raise RequestAuthenticationError("merchant payout tenant is forbidden")
    if document.get("payoutId") != payout_id:
        raise RequestAuthenticationError("merchant payout ID binding is invalid")
    service.store.consume_nonce(
        MERCHANT_SUBJECT,
        str(document.get("nonce", "")),
        digest_object(document),
        order_id=payout_id,
        task_id=str(document.get("correlationId", payout_id)),
        operation="payout-status",
    )


def create_app(service: MarketplaceService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(current_app: FastAPI):
        if current_app.state.marketplace_service is None:
            try:
                current_app.state.marketplace_service = _default_service()
            except Exception as exc:  # readiness reports no secret/detail
                current_app.state.startup_error = type(exc).__name__
        yield

    application = FastAPI(
        title="Secure Mediation Marketplace Payment Agent",
        version="1.0.0-demo",
        description=(
            "Project-local AP2 v0.2 / x402 v2-shaped marketplace simulation. "
            "No real settlement, payout, or legal payment guarantee."
        ),
        lifespan=lifespan,
    )
    application.state.marketplace_service = service
    application.state.startup_error = None

    def get_service() -> MarketplaceService:
        current = application.state.marketplace_service
        if current is None:
            raise MarketplaceError(
                "INTERNAL_ERROR", "Payment marketplace is not ready.", status_code=503
            )
        return current

    @application.exception_handler(MarketplaceError)
    async def marketplace_error_handler(_request: Request, exc: MarketplaceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.envelope())

    @application.exception_handler(RequestAuthenticationError)
    async def auth_error_handler(_request: Request, exc: RequestAuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=403,
            content={
                "code": "FORBIDDEN",
                "message": str(exc),
                "retryable": False,
                "correlationId": "payment-marketplace",
            },
        )

    @application.exception_handler(IdempotencyConflict)
    @application.exception_handler(ReplayDetected)
    async def replay_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        code = "IDEMPOTENCY_CONFLICT" if isinstance(exc, IdempotencyConflict) else "REPLAY_DETECTED"
        return JSONResponse(
            status_code=409,
            content={"code": code, "message": "Request replay was rejected.", "retryable": False, "correlationId": "payment-marketplace"},
        )

    @application.exception_handler(RequestValidationError)
    @application.exception_handler(CanonicalizationError)
    @application.exception_handler(ValueError)
    async def validation_error_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"code": "INVALID_SCHEMA", "message": "Request does not match the payment schema.", "retryable": False, "correlationId": "payment-marketplace"},
        )

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "payment-marketplace", "simulated": True}

    @application.get("/ready")
    def ready() -> JSONResponse:
        try:
            ok, detail = get_service().ready()
        except Exception:
            ok, detail = False, {"profile": PROFILE, "simulated": True}
        return JSONResponse(status_code=200 if ok else 503, content={"status": "ready" if ok else "not-ready", **detail})

    @application.get("/.well-known/agent-card.json")
    def agent_card() -> dict[str, Any]:
        public_url = os.getenv("PAYMENT_PUBLIC_URL", "http://localhost:8004")
        return mediation_agent_card(public_url)

    @application.post("/v1/orders")
    def start_order(
        body: dict[str, Any],
        idempotency_key: str = Header(alias="Idempotency-Key"),
        extension: str | None = Header(default=None, alias="X-A2A-Extensions"),
    ) -> dict[str, Any]:
        _require_extension(extension)
        service = get_service()
        business, auth = _split_signed_body(body)
        _authenticate(
            service,
            auth,
            method="POST",
            path="/v1/orders",
            body=business,
            role="customer",
            tenant=CUSTOMER_SUBJECT,
            operation="order-start-request",
            reference=f"idempotency:{idempotency_key}",
        )
        return service.start_order(business, idempotency_key=idempotency_key)

    @application.post("/v1/orders/{order_id}/payment")
    def submit_payment(
        order_id: str,
        body: dict[str, Any],
        idempotency_key: str = Header(alias="Idempotency-Key"),
        extension: str | None = Header(default=None, alias="X-A2A-Extensions"),
    ) -> dict[str, Any]:
        _require_extension(extension)
        service = get_service()
        business, auth = _split_signed_body(body)
        _authenticate(
            service,
            auth,
            method="POST",
            path=f"/v1/orders/{order_id}/payment",
            body=business,
            role="customer",
            tenant=CUSTOMER_SUBJECT,
            operation="payment-submit-request",
            reference=order_id,
        )
        fault = business.pop("merchantFault", None)
        if fault and not service.rail.allow_test_faults:
            raise MarketplaceError("FORBIDDEN", "Fault controls are disabled.", status_code=403)
        return service.submit_payment(
            order_id,
            business,
            idempotency_key=idempotency_key,
            merchant_fault=str(fault) if fault else None,
        )

    @application.get("/v1/orders/{order_id}")
    def order_status(
        order_id: str,
        request_auth: str | None = Header(default=None, alias="X-Demo-Request-Auth"),
        extension: str | None = Header(default=None, alias="X-A2A-Extensions"),
    ) -> dict[str, Any]:
        _require_extension(extension)
        service = get_service()
        auth = _decode_auth_header(request_auth)
        _authenticate(
            service,
            auth,
            method="GET",
            path=f"/v1/orders/{order_id}",
            body={},
            role="customer",
            tenant=CUSTOMER_SUBJECT,
            operation="order-status-request",
            reference=order_id,
        )
        return service.order_status(order_id, customer_id=CUSTOMER_SUBJECT)

    @application.post("/internal/v1/orders/{order_id}/refunds")
    def refund_order(
        order_id: str,
        body: dict[str, Any],
        idempotency_key: str = Header(alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        service = get_service()
        business, auth = _split_signed_body(body)
        actor = _authenticate(
            service,
            auth,
            method="POST",
            path=f"/internal/v1/orders/{order_id}/refunds",
            body=business,
            role="operator",
            tenant=OPERATOR_SUBJECT,
            operation="refund-request",
            reference=order_id,
        )
        reason = str(business.get("reason") or "")
        if not reason:
            raise MarketplaceError("INVALID_SCHEMA", "Operator reason is required.")
        return service.refund_order(
            order_id,
            idempotency_key=idempotency_key,
            actor_id=actor["subject"],
            reason=reason,
        )

    @application.post("/internal/v1/orders/{order_id}/reconcile")
    def reconcile_order(order_id: str, body: dict[str, Any]) -> dict[str, Any]:
        service = get_service()
        business, auth = _split_signed_body(body)
        actor = _authenticate(
            service,
            auth,
            method="POST",
            path=f"/internal/v1/orders/{order_id}/reconcile",
            body=business,
            role="operator",
            tenant=OPERATOR_SUBJECT,
            operation="order-reconciliation",
            reference=order_id,
        )
        reason = str(business.get("reason") or "")
        if not reason:
            raise MarketplaceError("INVALID_SCHEMA", "Operator reason is required.")
        return service.reconcile_order(order_id, actor_id=actor["subject"], reason=reason)

    @application.post("/internal/v1/payouts")
    def create_payout(
        body: dict[str, Any],
        idempotency_key: str = Header(alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        service = get_service()
        business, auth = _split_signed_body(body)
        actor = _authenticate(
            service,
            auth,
            method="POST",
            path="/internal/v1/payouts",
            body=business,
            role="operator",
            tenant=OPERATOR_SUBJECT,
            operation="payout-request",
            reference=f"idempotency:{idempotency_key}",
        )
        reason = str(business.get("reason") or "")
        if not reason:
            raise MarketplaceError("INVALID_SCHEMA", "Operator reason is required.")
        merchant_id = str(business.get("merchantId") or "")
        return service.create_payout(
            merchant_id=merchant_id,
            idempotency_key=idempotency_key,
            actor_id=actor["subject"],
            reason=reason,
        )

    @application.post("/v1/payouts/{payout_id}/status")
    def payout_status(payout_id: str, body: dict[str, Any]) -> dict[str, Any]:
        service = get_service()
        _merchant_payout_auth(service, body, payout_id)
        return service.payout_status(payout_id, merchant_id=MERCHANT_SUBJECT)

    @application.post("/internal/v1/payouts/{payout_id}/reconcile")
    def reconcile_payout(payout_id: str, body: dict[str, Any]) -> dict[str, Any]:
        service = get_service()
        business, auth = _split_signed_body(body)
        actor = _authenticate(
            service,
            auth,
            method="POST",
            path=f"/internal/v1/payouts/{payout_id}/reconcile",
            body=business,
            role="operator",
            tenant=OPERATOR_SUBJECT,
            operation="payout-reconciliation",
            reference=payout_id,
        )
        reason = str(business.get("reason") or "")
        if not reason:
            raise MarketplaceError("INVALID_SCHEMA", "Operator reason is required.")
        return service.reconcile_payout(
            payout_id, actor_id=actor["subject"], reason=reason
        )

    @application.post("/a2a")
    def a2a_message(
        envelope: dict[str, Any],
        extension: str | None = Header(default=None, alias="X-A2A-Extensions"),
        idempotency_key: str = Header(alias="Idempotency-Key"),
    ) -> dict[str, Any]:
        _require_extension(extension)
        request_id = envelope.get("id")
        try:
            if envelope.get("jsonrpc") != "2.0" or envelope.get("method") != "message/send":
                raise MarketplaceError("INVALID_SCHEMA", "Unsupported A2A JSON-RPC method.")
            params = envelope.get("params") or {}
            message = params.get("message") or {}
            parts = message.get("parts") or []
            data = next((part.get("data") for part in parts if isinstance(part, dict) and isinstance(part.get("data"), dict)), None)
            if not isinstance(data, dict):
                raise MarketplaceError("INVALID_SCHEMA", "A2A payment message requires a data part.")
            action = data.get("action")
            if action == "start_order":
                business, auth = _split_signed_body(data.get("request") or {})
                service = get_service()
                _authenticate(service, auth, method="POST", path="/v1/orders", body=business, role="customer", tenant=CUSTOMER_SUBJECT, operation="a2a-order-start", reference=idempotency_key)
                result = service.start_order(business, idempotency_key=idempotency_key)
                metadata = payment_metadata(
                    status="payment-required",
                    leg="upstream",
                    order_id=result["orderId"],
                    merchant_id=result["merchantId"],
                    quote_id=result["quoteId"],
                    correlation_id=result["correlationId"],
                    requirement=result["requirement"],
                    ap2=result["ap2"],
                    receipts=[],
                    pricing=result["pricing"],
                    trustedSurfaceInput=result["trustedSurfaceInput"],
                )
                task = task_result(task_id=result["taskId"], context_id=result["contextId"], state="input-required", metadata=metadata, message="Human Present payment approval is required.")
                return jsonrpc_result(request_id, task)
            if action == "submit_payment":
                order_id = str(data.get("orderId"))
                business, auth = _split_signed_body(data.get("request") or {})
                service = get_service()
                _authenticate(service, auth, method="POST", path=f"/v1/orders/{order_id}/payment", body=business, role="customer", tenant=CUSTOMER_SUBJECT, operation="a2a-payment-submit", reference=order_id)
                result = service.submit_payment(order_id, business, idempotency_key=idempotency_key)
                ap2_receipt = next(
                    (
                        receipt
                        for receipt in result.get("receipts", [])
                        if receipt.get("receiptType") == "ap2-payment"
                    ),
                    None,
                )
                metadata = payment_metadata(
                    status="payment-completed",
                    leg="upstream",
                    order_id=result["orderId"],
                    merchant_id=MERCHANT_SUBJECT,
                    quote_id=service._owned_order(order_id, CUSTOMER_SUBJECT)["quote_id"],
                    correlation_id=result["correlationId"],
                    settlement={"simulated": True, "state": result["state"]},
                    ap2={"paymentReceipt": ap2_receipt} if ap2_receipt else None,
                    receipts=result.get("receipts", []),
                )
                task = task_result(task_id=result["taskId"], context_id=result["contextId"], state="completed", metadata=metadata, message="Simulated marketplace payment and fulfillment completed.")
                return jsonrpc_result(request_id, task)
            if action == "payout_status":
                signed = data.get("request")
                payout_id = str(data.get("payoutId"))
                if not isinstance(signed, dict):
                    raise MarketplaceError("INVALID_SCHEMA", "Signed merchant payout query is required.")
                service = get_service()
                _merchant_payout_auth(service, signed, payout_id)
                result = service.payout_status(payout_id, merchant_id=MERCHANT_SUBJECT)
                return jsonrpc_result(request_id, result)
            raise MarketplaceError("INVALID_SCHEMA", "Unknown A2A payment action.")
        except MarketplaceError as exc:
            return jsonrpc_error(request_id, -32001, exc.message, exc.envelope())
        except RequestAuthenticationError as exc:
            return jsonrpc_error(request_id, -32003, "Forbidden", {"code": "FORBIDDEN", "message": str(exc), "retryable": False})

    return application


app = create_app()
