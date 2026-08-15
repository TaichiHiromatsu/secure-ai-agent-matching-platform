"""Deterministic, payment-aware merchant domain service.

Only merchant quote material and mediator guarantees are accepted here.  Customer
payment proofs and credentials intentionally have no representation in this module.
The documented HMAC fixture values are injected by tests or the runtime environment;
they are never embedded in production source or returned by this service.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

try:  # Supports both namespace-package and direct path loading.
    from .models import (
        FIXED_MERCHANDISE_AMOUNT,
        FIXED_PRODUCT_ID,
        FULFILLMENT_TERMS,
        FULFILLMENT_TERMS_DIGEST,
        MEDIATOR_ID,
        MEDIATOR_KID,
        MERCHANT_ID,
        MERCHANT_KID,
        PAYOUT_TERMS_VERSION,
        PRICING_POLICY_VERSION,
        PROFILE,
        AcceptedTerms,
        FulfillmentRequest,
        FulfillmentResponse,
        FulfillmentStatus,
        GuaranteeClaims,
        MerchantQuoteRequirement,
        PayoutStatusRequestInput,
        QuoteRequest,
        QuoteResponse,
        ReceiptEnvelope,
        Signature,
        SignedPayoutStatusRequest,
    )
except ImportError:  # pragma: no cover - exercised by `python app.py`.
    from models import (  # type: ignore[no-redef]
        FIXED_MERCHANDISE_AMOUNT,
        FIXED_PRODUCT_ID,
        FULFILLMENT_TERMS,
        FULFILLMENT_TERMS_DIGEST,
        MEDIATOR_ID,
        MEDIATOR_KID,
        MERCHANT_ID,
        MERCHANT_KID,
        PAYOUT_TERMS_VERSION,
        PRICING_POLICY_VERSION,
        PROFILE,
        AcceptedTerms,
        FulfillmentRequest,
        FulfillmentResponse,
        FulfillmentStatus,
        GuaranteeClaims,
        MerchantQuoteRequirement,
        PayoutStatusRequestInput,
        QuoteRequest,
        QuoteResponse,
        ReceiptEnvelope,
        Signature,
        SignedPayoutStatusRequest,
    )


FaultMode = Literal["success", "failure", "timeout"]


class MerchantError(Exception):
    """Safe project-local error that can cross the HTTP boundary."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        retryable: bool = False,
        correlation_id: str = "merchant",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.correlation_id = correlation_id


class MerchantTimeout(MerchantError):
    """Test-only timeout after the authoritative merchant commit."""

    def __init__(self, correlation_id: str) -> None:
        super().__init__(
            "SETTLEMENT_UNKNOWN",
            "Merchant response timed out; query fulfillment status before retrying.",
            status_code=504,
            retryable=True,
            correlation_id=correlation_id,
        )


def _validate_json(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        raise ValueError(f"float is not permitted at {path}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string object key at {path}")
            _validate_json(item, f"{path}.{key}")
        return
    raise ValueError(f"unsupported JSON value at {path}")


def canonical_bytes(value: Any) -> bytes:
    """Return Appendix A sorted compact UTF-8 JSON bytes."""

    _validate_json(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def digest_document(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def _unsigned(document: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(document))
    unsigned.pop("signature", None)
    return unsigned


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def sign_document(document: Mapping[str, Any], *, kid: str, key: bytes) -> dict[str, Any]:
    """Return a new top-level signed document using the project-local profile."""

    unsigned = _unsigned(document)
    value = _b64url(hmac.new(key, canonical_bytes(unsigned), hashlib.sha256).digest())
    signed = copy.deepcopy(unsigned)
    signed["signature"] = {"alg": "HS256", "kid": kid, "value": value}
    return signed


def verify_document(
    document: Mapping[str, Any],
    *,
    keys: Mapping[str, bytes],
    expected_kid: str | None = None,
    correlation_id: str = "merchant",
) -> None:
    signature = document.get("signature")
    if not isinstance(signature, dict) or signature.get("alg") != "HS256":
        raise MerchantError(
            "INVALID_SIGNATURE",
            "Signature is missing or uses an unsupported algorithm.",
            correlation_id=correlation_id,
        )
    kid = signature.get("kid")
    if not isinstance(kid, str) or kid not in keys or (expected_kid and kid != expected_kid):
        raise MerchantError(
            "UNKNOWN_KID",
            "The signing key is unknown or inactive.",
            correlation_id=correlation_id,
        )
    expected = sign_document(document, kid=kid, key=keys[kid])["signature"]["value"]
    actual = signature.get("value")
    if not isinstance(actual, str) or not hmac.compare_digest(actual, expected):
        raise MerchantError(
            "INVALID_SIGNATURE",
            "Signature verification failed.",
            correlation_id=correlation_id,
        )


def _compact_checkout_jwt(claims: Mapping[str, Any], *, key: bytes) -> str:
    header = {"alg": "HS256", "kid": MERCHANT_KID, "typ": "JWT"}
    header_segment = _b64url(canonical_bytes(header))
    claims_segment = _b64url(canonical_bytes(dict(claims)))
    signing_input = f"{header_segment}.{claims_segment}".encode("ascii")
    signature = _b64url(hmac.new(key, signing_input, hashlib.sha256).digest())
    return f"{header_segment}.{claims_segment}.{signature}"


class MerchantService:
    """SQLite-backed quote and exactly-once fulfillment service."""

    def __init__(
        self,
        *,
        merchant_key: bytes,
        mediator_keys: Mapping[str, bytes],
        database_path: str = ":memory:",
        clock: Callable[[], int] | None = None,
        id_factory: Callable[[], str] | None = None,
        nonce_factory: Callable[[], str] | None = None,
        allow_test_faults: bool = False,
        quote_ttl_seconds: int = 300,
    ) -> None:
        if not merchant_key or MEDIATOR_KID not in mediator_keys or not mediator_keys[MEDIATOR_KID]:
            raise ValueError("merchant and mediator HMAC keys must be injected")
        if database_path != ":memory:":
            Path(database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._merchant_key = bytes(merchant_key)
        self._mediator_keys = {kid: bytes(key) for kid, key in mediator_keys.items()}
        self._clock = clock or (lambda: int(time.time()))
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._nonce_factory = nonce_factory or (lambda: uuid.uuid4().hex)
        self.allow_test_faults = allow_test_faults
        self._quote_ttl_seconds = quote_ttl_seconds
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        if database_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def _migrate(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS merchant_quotes (
                    order_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    quote_id TEXT NOT NULL UNIQUE,
                    request_digest TEXT NOT NULL,
                    requirement_json TEXT NOT NULL,
                    requirement_digest TEXT NOT NULL,
                    accepted_json TEXT NOT NULL,
                    checkout_jwt TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS merchant_fulfillments (
                    order_id TEXT NOT NULL,
                    guarantee_id TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    quote_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('fulfilled', 'failed')),
                    fulfillment_id TEXT NOT NULL UNIQUE,
                    guarantee_digest TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (order_id, guarantee_id),
                    FOREIGN KEY (order_id) REFERENCES merchant_quotes(order_id)
                );
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def ready(self) -> bool:
        try:
            row = self._connection.execute("SELECT 1 AS ok").fetchone()
            return bool(row and row["ok"] == 1 and self._merchant_key and self._mediator_keys)
        except sqlite3.Error:
            return False

    def create_quote(self, request: QuoteRequest) -> QuoteResponse:
        request_document = request.model_dump(by_alias=True, mode="json")
        request_digest = digest_document(request_document)
        with self._lock:
            existing = self._connection.execute(
                "SELECT * FROM merchant_quotes WHERE order_id = ?", (request.order_id,)
            ).fetchone()
            if existing:
                if not hmac.compare_digest(existing["request_digest"], request_digest):
                    raise MerchantError(
                        "IDEMPOTENCY_CONFLICT",
                        "The order already has a quote for different input.",
                        status_code=409,
                        correlation_id=request.correlation_id,
                    )
                return self._quote_from_row(existing)

            now = int(self._clock())
            quote_id = f"quote-{self._id_factory()}"
            checkout_claims = {
                "audience": MEDIATOR_ID,
                "exp": now + self._quote_ttl_seconds,
                "fulfillmentTermsDigest": FULFILLMENT_TERMS_DIGEST,
                "iat": now,
                "issuer": MERCHANT_ID,
                "merchantId": MERCHANT_ID,
                "orderId": request.order_id,
                "pricingPolicyVersion": PRICING_POLICY_VERSION,
                "product": {
                    "description": "Deterministic paid booking demo",
                    "merchandiseAmount": FIXED_MERCHANDISE_AMOUNT,
                    "productId": FIXED_PRODUCT_ID,
                    "quantity": 1,
                },
                "quoteId": quote_id,
                "taskId": request.task_id,
            }
            checkout_jwt = _compact_checkout_jwt(checkout_claims, key=self._merchant_key)
            accepted = {
                "scheme": "platform-credit",
                "network": "demo:mediation-ledger",
                "amount": FIXED_MERCHANDISE_AMOUNT,
                "asset": "USD",
                "decimals": 2,
                "payTo": MERCHANT_ID,
                "maxTimeoutSeconds": 300,
                "extra": {
                    "profile": PROFILE,
                    "simulated": True,
                    "quoteId": quote_id,
                    "orderId": request.order_id,
                    "merchantId": MERCHANT_ID,
                    "pricingPolicyVersion": PRICING_POLICY_VERSION,
                    "fulfillmentTermsDigest": FULFILLMENT_TERMS_DIGEST,
                },
            }
            unsigned_requirement = {
                "x402Version": 2,
                "profile": PROFILE,
                "simulated": True,
                "resource": {
                    "url": f"a2a://{MERCHANT_ID}/orders/{request.order_id}",
                    "description": "Project-local simulated paid booking quote",
                    "mimeType": "application/json",
                },
                "accepts": [accepted],
                "quote": {
                    "issuer": MERCHANT_ID,
                    "audience": MEDIATOR_ID,
                    "orderId": request.order_id,
                    "taskId": request.task_id,
                    "quoteId": quote_id,
                    "merchantId": MERCHANT_ID,
                    "product": {
                        "productId": FIXED_PRODUCT_ID,
                        "description": "Deterministic paid booking demo",
                        "quantity": 1,
                        "merchandiseAmount": FIXED_MERCHANDISE_AMOUNT,
                    },
                    "pricingPolicyVersion": PRICING_POLICY_VERSION,
                    "fulfillmentTerms": FULFILLMENT_TERMS,
                    "fulfillmentTermsDigest": FULFILLMENT_TERMS_DIGEST,
                    "checkoutJwt": checkout_jwt,
                    "iat": now,
                    "exp": now + self._quote_ttl_seconds,
                },
            }
            signed_requirement = sign_document(
                unsigned_requirement, kid=MERCHANT_KID, key=self._merchant_key
            )
            requirement = MerchantQuoteRequirement.model_validate(signed_requirement)
            serialized = requirement.model_dump(by_alias=True, mode="json")
            requirement_digest = digest_document(serialized)
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO merchant_quotes (
                        order_id, task_id, quote_id, request_digest, requirement_json,
                        requirement_digest, accepted_json, checkout_jwt, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.order_id,
                        request.task_id,
                        quote_id,
                        request_digest,
                        canonical_bytes(serialized).decode("utf-8"),
                        requirement_digest,
                        canonical_bytes(accepted).decode("utf-8"),
                        checkout_jwt,
                        now,
                    ),
                )
            return QuoteResponse(
                requirement=requirement,
                checkout_jwt=checkout_jwt,
                quote_digest=requirement_digest,
            )

    def _quote_from_row(self, row: sqlite3.Row) -> QuoteResponse:
        requirement = MerchantQuoteRequirement.model_validate_json(row["requirement_json"])
        return QuoteResponse(
            requirement=requirement,
            checkout_jwt=row["checkout_jwt"],
            quote_digest=row["requirement_digest"],
        )

    def fulfill(
        self,
        request: FulfillmentRequest,
        *,
        fault: FaultMode = "success",
    ) -> FulfillmentResponse:
        if fault not in ("success", "failure", "timeout"):
            raise MerchantError(
                "INVALID_SCHEMA",
                "Unknown fault fixture.",
                correlation_id=request.correlation_id,
            )
        if fault != "success" and not self.allow_test_faults:
            raise MerchantError(
                "FORBIDDEN",
                "Merchant fault fixtures are disabled.",
                status_code=403,
                correlation_id=request.correlation_id,
            )

        payment_payload = request.payment_payload
        claims = payment_payload.payload
        request_digest = digest_document(
            payment_payload.model_dump(by_alias=True, mode="json")
        )
        with self._lock:
            existing = self._connection.execute(
                """
                SELECT * FROM merchant_fulfillments
                WHERE order_id = ? AND guarantee_id = ?
                """,
                (claims.order_id, claims.guarantee_id),
            ).fetchone()
            if existing:
                if not hmac.compare_digest(existing["request_digest"], request_digest):
                    raise MerchantError(
                        "IDEMPOTENCY_CONFLICT",
                        "The guarantee was already processed with different input.",
                        status_code=409,
                        correlation_id=request.correlation_id,
                    )
                return self._response_from_row(existing, idempotent=True)

            quote = self._connection.execute(
                "SELECT * FROM merchant_quotes WHERE order_id = ? AND quote_id = ?",
                (claims.order_id, claims.quote_id),
            ).fetchone()
            if quote is None:
                raise MerchantError(
                    "QUOTE_MISMATCH",
                    "No matching merchant quote exists.",
                    correlation_id=request.correlation_id,
                )
            self._verify_guarantee(request, quote)

            now = int(self._clock())
            state: Literal["fulfilled", "failed"] = (
                "failed" if fault == "failure" else "fulfilled"
            )
            fulfillment_id = f"fulfillment-{self._id_factory()}"
            # The order receipt binds the complete x402-shaped PaymentPayload,
            # including the exact accepted terms, rather than only its claims body.
            guarantee_digest = request_digest
            receipt_id = f"merchant-receipt-{self._id_factory()}"
            receipt_document = {
                "receiptType": "merchant-order",
                "receiptId": receipt_id,
                "profile": PROFILE,
                "simulated": True,
                "status": state,
                "issuedAt": now,
                "issuer": MERCHANT_ID,
                "subject": MEDIATOR_ID,
                "orderId": claims.order_id,
                "quoteId": claims.quote_id,
                "guaranteeId": claims.guarantee_id,
                "fulfillmentId": fulfillment_id,
                "relatedDigests": {
                    "guarantee": guarantee_digest,
                    "merchantQuoteRequirement": quote["requirement_digest"],
                },
            }
            receipt = ReceiptEnvelope.model_validate(
                sign_document(receipt_document, kid=MERCHANT_KID, key=self._merchant_key)
            )
            receipt_json = canonical_bytes(
                receipt.model_dump(by_alias=True, mode="json")
            ).decode("utf-8")
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO merchant_fulfillments (
                        order_id, guarantee_id, request_digest, quote_id, state,
                        fulfillment_id, guarantee_digest, receipt_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claims.order_id,
                        claims.guarantee_id,
                        request_digest,
                        claims.quote_id,
                        state,
                        fulfillment_id,
                        guarantee_digest,
                        receipt_json,
                        now,
                    ),
                )
            response = FulfillmentResponse(
                state=state,
                fulfillment_id=fulfillment_id,
                receipt=receipt,
                idempotent=False,
            )
            if fault == "timeout":
                raise MerchantTimeout(request.correlation_id)
            return response

    def _verify_guarantee(self, request: FulfillmentRequest, quote: sqlite3.Row) -> None:
        payment_payload = request.payment_payload
        claims = payment_payload.payload
        correlation_id = request.correlation_id
        accepted = payment_payload.accepted.model_dump(by_alias=True, mode="json")
        expected_accepted = json.loads(quote["accepted_json"])
        if not hmac.compare_digest(canonical_bytes(accepted), canonical_bytes(expected_accepted)):
            raise MerchantError(
                "QUOTE_MISMATCH",
                "Guarantee accepted terms do not match the signed quote.",
                correlation_id=correlation_id,
            )
        if claims.merchant_quote_requirement_digest != quote["requirement_digest"]:
            raise MerchantError(
                "QUOTE_MISMATCH",
                "Guarantee does not bind the signed merchant quote.",
                correlation_id=correlation_id,
            )
        if claims.task_id != quote["task_id"] or claims.merchant_id != MERCHANT_ID:
            raise MerchantError(
                "GUARANTEE_INVALID",
                "Guarantee order, task, or merchant binding is invalid.",
                correlation_id=correlation_id,
            )
        if claims.payable_amount != expected_accepted["amount"]:
            raise MerchantError(
                "AMOUNT_MISMATCH",
                "Guarantee payable amount does not match the quote.",
                correlation_id=correlation_id,
            )
        now = int(self._clock())
        if claims.iat > now:
            raise MerchantError(
                "NOT_YET_VALID",
                "Guarantee is not yet valid.",
                retryable=True,
                correlation_id=correlation_id,
            )
        if claims.exp <= now:
            raise MerchantError(
                "EXPIRED",
                "Guarantee has expired.",
                correlation_id=correlation_id,
            )
        claims_document = claims.model_dump(by_alias=True, mode="json")
        verify_document(
            claims_document,
            keys=self._mediator_keys,
            expected_kid=MEDIATOR_KID,
            correlation_id=correlation_id,
        )

    def _response_from_row(
        self, row: sqlite3.Row, *, idempotent: bool
    ) -> FulfillmentResponse:
        return FulfillmentResponse(
            state=row["state"],
            fulfillment_id=row["fulfillment_id"],
            receipt=ReceiptEnvelope.model_validate_json(row["receipt_json"]),
            idempotent=idempotent,
        )

    def get_fulfillment(self, order_id: str, guarantee_id: str) -> FulfillmentStatus:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM merchant_fulfillments
                WHERE order_id = ? AND guarantee_id = ?
                """,
                (order_id, guarantee_id),
            ).fetchone()
        if row is None:
            raise MerchantError(
                "INVALID_SCHEMA",
                "Fulfillment was not found.",
                status_code=404,
            )
        return FulfillmentStatus(
            order_id=order_id,
            guarantee_id=guarantee_id,
            state=row["state"],
            fulfillment_id=row["fulfillment_id"],
            guarantee_digest=row["guarantee_digest"],
            receipt=ReceiptEnvelope.model_validate_json(row["receipt_json"]),
        )

    def create_payout_status_request(
        self, request: PayoutStatusRequestInput
    ) -> SignedPayoutStatusRequest:
        timestamp = int(self._clock())
        document = {
            "profile": PROFILE,
            "simulated": True,
            "skill": "payout_status",
            "method": "GET",
            "path": f"/v1/payouts/{request.payout_id}",
            "bodyDigest": digest_document({}),
            "issuer": MERCHANT_ID,
            "audience": MEDIATOR_ID,
            "actor": {"role": "merchant", "merchantId": MERCHANT_ID},
            "payoutId": request.payout_id,
            "correlationId": request.correlation_id,
            "nonce": self._nonce_factory(),
            "timestamp": timestamp,
        }
        return SignedPayoutStatusRequest.model_validate(
            sign_document(document, kid=MERCHANT_KID, key=self._merchant_key)
        )

    def count_fulfillments(self) -> int:
        """Return the authoritative side-effect count for tests/readiness diagnostics."""

        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM merchant_fulfillments"
        ).fetchone()
        return int(row["count"])
