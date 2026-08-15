"""FastAPI entrypoint for the project-local paid booking merchant (port 8005)."""

from __future__ import annotations

import os

import uvicorn
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

try:  # Supports namespace-package loading used by the test suite.
    from .models import (
        MEDIATOR_ID,
        MERCHANT_ID,
        PROFILE,
        SDK_PACKAGE,
        SDK_VERSION,
        WIRE_PROTOCOL_VERSION,
        ErrorEnvelope,
        FulfillmentRequest,
        FulfillmentResponse,
        FulfillmentStatus,
        PayoutStatusRequestInput,
        QuoteRequest,
        QuoteResponse,
        SignedPayoutStatusRequest,
    )
    from .service import MerchantError, MerchantService
except ImportError:  # pragma: no cover - used by `python app.py`.
    from models import (  # type: ignore[no-redef]
        MEDIATOR_ID,
        MERCHANT_ID,
        PROFILE,
        SDK_PACKAGE,
        SDK_VERSION,
        WIRE_PROTOCOL_VERSION,
        ErrorEnvelope,
        FulfillmentRequest,
        FulfillmentResponse,
        FulfillmentStatus,
        PayoutStatusRequestInput,
        QuoteRequest,
        QuoteResponse,
        SignedPayoutStatusRequest,
    )
    from service import MerchantError, MerchantService  # type: ignore[no-redef]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _service_from_environment() -> MerchantService | None:
    merchant_key = os.environ.get("PAYMENT_DEMO_MERCHANT_HMAC_KEY")
    mediator_key = os.environ.get("PAYMENT_DEMO_MEDIATOR_HMAC_KEY")
    if not merchant_key or not mediator_key:
        return None
    return MerchantService(
        merchant_key=merchant_key.encode("utf-8"),
        mediator_keys={"demo-mediator-hmac-v1": mediator_key.encode("utf-8")},
        database_path=os.environ.get(
            "PAYMENT_DEMO_MERCHANT_DB", "/tmp/paid-booking-agent.sqlite3"
        ),
        allow_test_faults=_bool_env("PAYMENT_DEMO_ENABLE_MERCHANT_FAULTS"),
    )


def _agent_card(public_url: str) -> dict:
    """Return the exact payment-aware capabilities without key material."""

    return {
        "name": "Paid Demo Booking Agent",
        "description": (
            "Project-local v2-shaped simulation merchant. It accepts mediator-signed "
            "platform credit, not x402 exact settlement or a legal payment guarantee."
        ),
        "url": public_url.rstrip("/") + "/a2a",
        "version": "1.0.0-demo",
        "protocolVersion": WIRE_PROTOCOL_VERSION,
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
            "extensions": [
                {
                    "uri": PROFILE,
                    "required": True,
                    "params": {
                        "profile": PROFILE,
                        "simulated": True,
                        "sdkPackage": SDK_PACKAGE,
                        "sdkVersion": SDK_VERSION,
                        "wireProtocolVersion": WIRE_PROTOCOL_VERSION,
                        "roles": ["merchant"],
                        "merchantCredit": {
                            "schemes": ["platform-credit"],
                            "networks": ["demo:mediation-ledger"],
                            "assets": [{"asset": "USD", "decimals": 2}],
                            "payTo": [MERCHANT_ID],
                        },
                    },
                }
            ],
        },
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "paid_booking",
                "name": "Paid demo booking",
                "description": (
                    "Issues a signed merchant quote requirement and fulfils only after "
                    "a valid mediator-signed platform-credit guarantee."
                ),
                "tags": ["payment", "booking", "platform-credit", "simulation"],
                "examples": ["Quote and fulfil one deterministic demo booking"],
            },
            {
                "id": "fulfillment_status",
                "name": "Fulfillment status",
                "description": "Returns the authoritative status for an order/guarantee pair.",
                "tags": ["payment", "status", "simulation"],
                "examples": ["Check a booking after a response timeout"],
            },
            {
                "id": "payout_status",
                "name": "Payout status query",
                "description": (
                    "Creates a merchant-signed request for the marketplace authoritative "
                    "payout status endpoint."
                ),
                "tags": ["payment", "payout", "status", "simulation"],
                "examples": ["Create a signed status query for payout-123"],
            },
        ],
        "provider": {"organization": MERCHANT_ID, "url": public_url},
        "documentationUrl": public_url.rstrip("/") + "/docs",
    }


def create_app(merchant_service: MerchantService | None = None) -> FastAPI:
    application = FastAPI(
        title="Paid Demo Booking Agent",
        version="1.0.0-demo",
        description=(
            "A project-local AP2/x402-shaped simulation merchant. This is not real "
            "settlement, payout, AP2 conformance, or a legal guarantee."
        ),
    )
    application.state.merchant_service = merchant_service or _service_from_environment()

    @application.exception_handler(MerchantError)
    async def merchant_error_handler(_request: Request, exc: MerchantError) -> JSONResponse:
        envelope = ErrorEnvelope(
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            correlation_id=exc.correlation_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=envelope.model_dump(by_alias=True, mode="json", exclude_none=True),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        # Do not echo Pydantic's input field: it might be a customer proof that this
        # service must neither receive nor reflect.
        envelope = ErrorEnvelope(
            code="INVALID_SCHEMA",
            message="Request does not match the merchant payment schema.",
            retryable=False,
            correlation_id="merchant",
        )
        return JSONResponse(
            status_code=422,
            content=envelope.model_dump(by_alias=True, mode="json", exclude_none=True),
        )

    def get_service(request: Request) -> MerchantService:
        service = request.app.state.merchant_service
        if service is None:
            raise MerchantError(
                "INTERNAL_ERROR",
                "Merchant signing configuration is unavailable.",
                status_code=503,
                retryable=False,
            )
        return service

    @application.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "paid-booking-agent", "simulated": True}

    @application.get("/ready")
    def ready(request: Request) -> JSONResponse:
        service = request.app.state.merchant_service
        is_ready = service is not None and service.ready()
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={
                "status": "ready" if is_ready else "not-ready",
                "profile": PROFILE,
                "simulated": True,
            },
        )

    @application.get("/.well-known/agent-card.json")
    def agent_card() -> dict:
        public_url = os.environ.get(
            "PAID_BOOKING_AGENT_PUBLIC_URL", "http://localhost:8005"
        ).rstrip("/")
        return _agent_card(public_url)

    @application.post("/v1/quotes", response_model=QuoteResponse, response_model_by_alias=True)
    def create_quote(
        body: QuoteRequest,
        service: MerchantService = Depends(get_service),
    ) -> QuoteResponse:
        return service.create_quote(body)

    @application.post(
        "/v1/fulfillments",
        response_model=FulfillmentResponse,
        response_model_by_alias=True,
    )
    def fulfill(
        body: FulfillmentRequest,
        service: MerchantService = Depends(get_service),
        test_fault: str | None = Header(default=None, alias="X-Demo-Test-Fault"),
    ) -> FulfillmentResponse:
        return service.fulfill(body, fault=test_fault or "success")

    @application.get(
        "/v1/fulfillments/{order_id}/{guarantee_id}",
        response_model=FulfillmentStatus,
        response_model_by_alias=True,
    )
    def fulfillment_status(
        order_id: str,
        guarantee_id: str,
        service: MerchantService = Depends(get_service),
    ) -> FulfillmentStatus:
        return service.get_fulfillment(order_id, guarantee_id)

    @application.post(
        "/v1/payout-status-requests",
        response_model=SignedPayoutStatusRequest,
        response_model_by_alias=True,
    )
    def payout_status_request(
        body: PayoutStatusRequestInput,
        service: MerchantService = Depends(get_service),
    ) -> SignedPayoutStatusRequest:
        return service.create_payout_status_request(body)

    @application.post("/a2a")
    def a2a_message(
        envelope: dict,
        request: Request,
        test_fault: str | None = Header(default=None, alias="X-Demo-Test-Fault"),
    ) -> dict:
        """Minimal wire-0.3 JSON-RPC adapter for mediator-to-merchant calls."""

        request_id = envelope.get("id")
        if envelope.get("jsonrpc") != "2.0" or envelope.get("method") != "message/send":
            raise MerchantError("INVALID_SCHEMA", "Unsupported A2A JSON-RPC method.")
        message = (envelope.get("params") or {}).get("message") or {}
        parts = message.get("parts") or []
        data = next(
            (
                part.get("data")
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("data"), dict)
            ),
            None,
        )
        if not isinstance(data, dict):
            raise MerchantError("INVALID_SCHEMA", "A2A merchant request requires a data part.")
        service = get_service(request)
        action = data.get("action")
        if action == "quote":
            result = service.create_quote(QuoteRequest.model_validate(data.get("request") or {}))
        elif action == "fulfill":
            result = service.fulfill(
                FulfillmentRequest.model_validate(data.get("request") or {}),
                fault=test_fault or "success",
            )
        elif action == "fulfillment_status":
            result = service.get_fulfillment(
                str(data.get("orderId") or ""), str(data.get("guaranteeId") or "")
            )
        else:
            raise MerchantError("INVALID_SCHEMA", "Unsupported merchant A2A action.")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result.model_dump(by_alias=True, mode="json"),
        }

    return application


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="0.0.0.0", port=8005)
