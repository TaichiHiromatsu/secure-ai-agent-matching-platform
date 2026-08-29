"""Strict wire models for the payment-aware demo merchant.

The module deliberately contains no customer mandate or proof model.  The merchant
accepts only a marketplace-issued guarantee plus the minimum order identifiers it
needs to fulfil a booking.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


PROFILE = "urn:secure-a2a:extensions:ap2-x402-marketplace:v1"
WIRE_PROTOCOL_VERSION = "0.3.0"
SDK_PACKAGE = "a2a-sdk"
SDK_VERSION = "0.3.19"
MERCHANT_ID = "demo-merchant"
MERCHANT_KID = "demo-merchant-hmac-v1"
MEDIATOR_ID = "mediation-platform"
MEDIATOR_KID = "demo-mediator-hmac-v1"
PRICING_POLICY_VERSION = "zero-fee-v1"
PAYOUT_TERMS_VERSION = "manual-payout-v1"
FULFILLMENT_TERMS = "Confirm one deterministic demo booking after guarantee verification."
FULFILLMENT_TERMS_DIGEST = "sha256:" + hashlib.sha256(
    FULFILLMENT_TERMS.encode("utf-8")
).hexdigest()
FIXED_PRODUCT_ID = "demo-paid-booking"
FIXED_MERCHANDISE_AMOUNT = "1250"


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class StrictModel(BaseModel):
    """Base model that rejects unrecognised (including customer-proof) fields."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
        strict=True,
    )


class Signature(StrictModel):
    alg: Literal["HS256"] = "HS256"
    kid: str
    value: str = Field(min_length=1)


class QuoteRequest(StrictModel):
    order_id: str = Field(min_length=1, max_length=128)
    task_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=128)
    product_id: Literal[FIXED_PRODUCT_ID] = FIXED_PRODUCT_ID
    quantity: Literal[1] = 1
    audience: Literal[MEDIATOR_ID] = MEDIATOR_ID


class Resource(StrictModel):
    url: str
    description: str
    mime_type: Literal["application/json"] = "application/json"


class AcceptedExtra(StrictModel):
    profile: Literal[PROFILE] = PROFILE
    simulated: Literal[True] = True
    quote_id: str
    order_id: str
    merchant_id: Literal[MERCHANT_ID] = MERCHANT_ID
    pricing_policy_version: Literal[PRICING_POLICY_VERSION] = PRICING_POLICY_VERSION
    fulfillment_terms_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class AcceptedTerms(StrictModel):
    scheme: Literal["platform-credit"] = "platform-credit"
    network: Literal["demo:mediation-ledger"] = "demo:mediation-ledger"
    amount: str = Field(pattern=r"^(0|[1-9][0-9]*)$")
    asset: Literal["USD"] = "USD"
    decimals: Literal[2] = 2
    pay_to: Literal[MERCHANT_ID] = MERCHANT_ID
    max_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    extra: AcceptedExtra


class Product(StrictModel):
    product_id: Literal[FIXED_PRODUCT_ID] = FIXED_PRODUCT_ID
    description: str
    quantity: Literal[1] = 1
    merchandise_amount: str = Field(pattern=r"^(0|[1-9][0-9]*)$")


class QuoteClaims(StrictModel):
    issuer: Literal[MERCHANT_ID] = MERCHANT_ID
    audience: Literal[MEDIATOR_ID] = MEDIATOR_ID
    order_id: str
    task_id: str
    quote_id: str
    merchant_id: Literal[MERCHANT_ID] = MERCHANT_ID
    product: Product
    pricing_policy_version: Literal[PRICING_POLICY_VERSION] = PRICING_POLICY_VERSION
    fulfillment_terms: Literal[FULFILLMENT_TERMS] = FULFILLMENT_TERMS
    fulfillment_terms_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    checkout_jwt: str = Field(min_length=3)
    iat: int = Field(ge=0)
    exp: int = Field(ge=0)

    @field_validator("exp")
    @classmethod
    def _exp_is_integer(cls, value: int) -> int:
        # bool is an int subclass; payment times must never accept it.
        if isinstance(value, bool):
            raise ValueError("exp must be an integer timestamp")
        return value


class MerchantQuoteRequirement(StrictModel):
    x402_version: Literal[2] = 2
    profile: Literal[PROFILE] = PROFILE
    simulated: Literal[True] = True
    resource: Resource
    accepts: list[AcceptedTerms] = Field(min_length=1, max_length=1)
    quote: QuoteClaims
    signature: Signature


class QuoteResponse(StrictModel):
    requirement: MerchantQuoteRequirement
    checkout_jwt: str
    quote_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class GuaranteeClaims(StrictModel):
    kind: Literal["payment-guarantee"] = "payment-guarantee"
    profile: Literal[PROFILE] = PROFILE
    simulated: Literal[True] = True
    guarantee_id: str = Field(min_length=1, max_length=128)
    merchant_quote_requirement_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    order_id: str = Field(min_length=1, max_length=128)
    task_id: str = Field(min_length=1, max_length=128)
    quote_id: str = Field(min_length=1, max_length=128)
    merchant_id: Literal[MERCHANT_ID] = MERCHANT_ID
    upstream_x402_receipt_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    upstream_ap2_receipt_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    payable_journal_transaction_id: str = Field(min_length=1, max_length=128)
    payable_entry_id: str = Field(min_length=1, max_length=128)
    payable_amount: str = Field(pattern=r"^(0|[1-9][0-9]*)$")
    commission: Literal["0"] = "0"
    currency: Literal["USD"] = "USD"
    payout_terms_version: Literal[PAYOUT_TERMS_VERSION] = PAYOUT_TERMS_VERSION
    iat: int = Field(ge=0)
    exp: int = Field(ge=0)
    signature: Signature


class GuaranteePaymentPayload(StrictModel):
    x402_version: Literal[2] = 2
    accepted: AcceptedTerms
    payload: GuaranteeClaims


class FulfillmentRequest(StrictModel):
    payment_payload: GuaranteePaymentPayload
    correlation_id: str = Field(min_length=1, max_length=128)


class ReceiptEnvelope(StrictModel):
    receipt_type: Literal["merchant-order"] = "merchant-order"
    receipt_id: str
    profile: Literal[PROFILE] = PROFILE
    simulated: Literal[True] = True
    status: Literal["fulfilled", "failed"]
    issued_at: int
    issuer: Literal[MERCHANT_ID] = MERCHANT_ID
    subject: Literal[MEDIATOR_ID] = MEDIATOR_ID
    order_id: str
    quote_id: str
    guarantee_id: str
    fulfillment_id: str
    related_digests: dict[str, str]
    signature: Signature


class FulfillmentResponse(StrictModel):
    state: Literal["fulfilled", "failed"]
    fulfillment_id: str
    receipt: ReceiptEnvelope
    idempotent: bool


class FulfillmentStatus(StrictModel):
    order_id: str
    guarantee_id: str
    state: Literal["fulfilled", "failed"]
    fulfillment_id: str
    guarantee_digest: str
    receipt: ReceiptEnvelope


class PayoutStatusRequestInput(StrictModel):
    payout_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=128)


class SignedPayoutStatusRequest(StrictModel):
    profile: Literal[PROFILE] = PROFILE
    simulated: Literal[True] = True
    skill: Literal["payout_status"] = "payout_status"
    method: Literal["GET"] = "GET"
    path: str
    body_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    issuer: Literal[MERCHANT_ID] = MERCHANT_ID
    audience: Literal[MEDIATOR_ID] = MEDIATOR_ID
    actor: dict[str, str]
    payout_id: str
    correlation_id: str
    nonce: str
    timestamp: int
    signature: Signature


class ErrorEnvelope(StrictModel):
    code: str
    message: str
    retryable: bool
    correlation_id: str
    details: dict[str, Any] | None = None
