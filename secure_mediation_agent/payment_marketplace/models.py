"""Strict domain and wire models for the marketplace payment simulation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .config import (
    ASSET,
    CURRENCY,
    DECIMALS,
    PRICING_POLICY_VERSION,
    PROFILE_URI,
    ROUNDING_RULE,
    SIMULATED,
    UPSTREAM_NETWORK,
    UPSTREAM_PAYEE_ID,
    UPSTREAM_PAYEE_NAME,
)


NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
Base64UrlSha256 = Annotated[
    StrictStr,
    Field(pattern=r"^[A-Za-z0-9_-]{43}$"),
]
Sha256Digest = Annotated[
    StrictStr,
    Field(pattern=r"^sha256:[0-9a-f]{64}$"),
]
Base64UrlValue = Annotated[
    StrictStr,
    Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$"),
]
MinorUnitString = Annotated[
    StrictStr,
    Field(pattern=r"^(0|[1-9][0-9]*)$"),
]


class StrictModel(BaseModel):
    """Base model that rejects coercion and unknown fields."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    def wire_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """Return the project wire representation using declared aliases."""

        return self.model_dump(by_alias=True, exclude_none=exclude_none)


class ActorRole(StrEnum):
    CUSTOMER = "customer"
    MERCHANT = "merchant"
    OPERATOR = "operator"


class OrderState(StrEnum):
    AWAITING_QUOTE = "awaiting_quote"
    PAYMENT_REQUIRED = "payment_required"
    CHARGE_PROCESSING = "charge_processing"
    PAYABLE_POSTED = "payable_posted"
    GUARANTEE_ISSUED = "guarantee_issued"
    FULFILLING = "fulfilling"
    REFUND_REQUIRED = "refund_required"
    REFUNDING = "refunding"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    COMPLETED = "completed"
    REFUNDED = "refunded"
    FAILED = "failed"


class ChargeState(StrEnum):
    REQUIRED = "required"
    VERIFIED = "verified"
    SETTLING = "settling"
    SETTLED = "settled"
    UNKNOWN = "unknown"
    FAILED = "failed"


class PayableState(StrEnum):
    OPEN = "open"
    GUARANTEED = "guaranteed"
    ELIGIBLE = "eligible"
    INCLUDED = "included"
    REVERSING = "reversing"
    PAID = "paid"
    REVERSED = "reversed"


class GuaranteeState(StrEnum):
    ISSUED = "issued"
    DELIVERED = "delivered"
    DELIVERY_UNKNOWN = "delivery_unknown"
    ACCEPTED = "accepted"
    EXPIRED = "expired"


class FulfillmentState(StrEnum):
    AUTHORIZED = "authorized"
    PROCESSING = "processing"
    FULFILLED = "fulfilled"
    UNKNOWN = "unknown"
    FAILED = "failed"


class RefundState(StrEnum):
    REQUIRED = "required"
    SETTLING = "settling"
    SETTLED = "settled"
    UNKNOWN = "unknown"
    FAILED = "failed"


class PayoutState(StrEnum):
    CREATED = "created"
    SETTLING = "settling"
    PAID = "paid"
    UNKNOWN = "unknown"
    FAILED = "failed"


class PaymentLeg(StrEnum):
    UPSTREAM = "upstream"
    MERCHANT_CREDIT = "merchant-credit"
    REFUND = "refund"
    PAYOUT = "payout"


class PaymentStatus(StrEnum):
    PAYMENT_REQUIRED = "payment-required"
    PAYMENT_SUBMITTED = "payment-submitted"
    PAYMENT_VERIFIED = "payment-verified"
    PAYMENT_COMPLETED = "payment-completed"
    PAYMENT_FAILED = "payment-failed"
    GUARANTEE_ISSUED = "guarantee-issued"
    FULFILLMENT_COMPLETED = "fulfillment-completed"
    REFUND_REQUIRED = "refund-required"
    REFUND_COMPLETED = "refund-completed"
    PAYOUT_PENDING = "payout-pending"
    PAYOUT_COMPLETED = "payout-completed"
    PAYOUT_FAILED = "payout-failed"


class ReceiptType(StrEnum):
    X402_SETTLEMENT = "x402-settlement"
    AP2_PAYMENT = "ap2-payment"
    MARKETPLACE_GUARANTEE = "marketplace-guarantee"
    MERCHANT_ORDER = "merchant-order"
    REFUND = "refund"
    PAYOUT = "payout"


class ErrorCode(StrEnum):
    UNSUPPORTED_EXTENSION = "UNSUPPORTED_EXTENSION"
    UNSUPPORTED_PROFILE = "UNSUPPORTED_PROFILE"
    UNSUPPORTED_A2A_VERSION = "UNSUPPORTED_A2A_VERSION"
    UNSUPPORTED_MANDATE_TYPE = "UNSUPPORTED_MANDATE_TYPE"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    INVALID_SIGNATURE = "INVALID_SIGNATURE"
    UNKNOWN_KID = "UNKNOWN_KID"
    EXPIRED = "EXPIRED"
    NOT_YET_VALID = "NOT_YET_VALID"
    AUDIENCE_MISMATCH = "AUDIENCE_MISMATCH"
    REPLAY_DETECTED = "REPLAY_DETECTED"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    QUOTE_MISMATCH = "QUOTE_MISMATCH"
    PAYEE_MISMATCH = "PAYEE_MISMATCH"
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    ASSET_MISMATCH = "ASSET_MISMATCH"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    MERCHANT_NOT_ONBOARDED = "MERCHANT_NOT_ONBOARDED"
    MERCHANT_SUSPENDED = "MERCHANT_SUSPENDED"
    GUARANTEE_INVALID = "GUARANTEE_INVALID"
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
    SETTLEMENT_UNKNOWN = "SETTLEMENT_UNKNOWN"
    LEDGER_POST_FAILED = "LEDGER_POST_FAILED"
    PAYOUT_NOT_ELIGIBLE = "PAYOUT_NOT_ELIGIBLE"
    PAYOUT_UNKNOWN = "PAYOUT_UNKNOWN"
    REFUND_UNKNOWN = "REFUND_UNKNOWN"
    FORBIDDEN = "FORBIDDEN"
    SSRF_BLOCKED = "SSRF_BLOCKED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


RETRYABLE_ERROR_CODES = frozenset(
    {
        ErrorCode.NOT_YET_VALID,
        ErrorCode.SETTLEMENT_UNKNOWN,
        ErrorCode.LEDGER_POST_FAILED,
        ErrorCode.PAYOUT_UNKNOWN,
        ErrorCode.REFUND_UNKNOWN,
    }
)


class Signature(StrictModel):
    alg: Literal["HS256"] = "HS256"
    kid: StrictStr = Field(min_length=1)
    value: Base64UrlValue


class StableError(StrictModel):
    code: ErrorCode
    message: StrictStr = Field(min_length=1)
    retryable: StrictBool
    correlation_id: StrictStr = Field(alias="correlationId", min_length=1)
    details: dict[str, Any] | None = None

    @model_validator(mode="after")
    def retryability_matches_contract(self) -> "StableError":
        if self.retryable != (self.code in RETRYABLE_ERROR_CODES):
            raise ValueError("retryable does not match the stable error contract")
        return self


class PricingBreakdown(StrictModel):
    policy_version: Literal["zero-fee-v1"] = Field(
        default=PRICING_POLICY_VERSION,
        alias="policyVersion",
    )
    merchandise_amount: NonNegativeInt = Field(alias="merchandiseAmount")
    customer_surcharge: Literal[0] = Field(default=0, alias="customerSurcharge")
    collection_rail_cost: Literal[0] = Field(default=0, alias="collectionRailCost")
    customer_total: NonNegativeInt = Field(alias="customerTotal")
    provider_commission: Literal[0] = Field(default=0, alias="providerCommission")
    payout_rail_cost: Literal[0] = Field(default=0, alias="payoutRailCost")
    merchant_payable_amount: NonNegativeInt = Field(alias="merchantPayableAmount")
    asset: Literal["USD"] = ASSET
    currency: Literal["USD"] = CURRENCY
    network: Literal["demo:local"] = UPSTREAM_NETWORK
    decimals: Literal[2] = DECIMALS
    rounding_rule: Literal["minor-unit-exact"] = Field(
        default=ROUNDING_RULE,
        alias="roundingRule",
    )
    calculated_at: datetime = Field(alias="calculatedAt")

    @field_validator("calculated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("calculatedAt must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_zero_fee_equations(self) -> "PricingBreakdown":
        if self.customer_total != self.merchandise_amount:
            raise ValueError("zero-fee customer total must equal merchandise amount")
        if self.merchant_payable_amount != self.merchandise_amount:
            raise ValueError("zero-fee merchant payable must equal merchandise amount")
        return self


def calculate_zero_fee_pricing(
    merchandise_amount: int,
    *,
    calculated_at: datetime,
) -> PricingBreakdown:
    """Calculate the immutable MVP pricing breakdown using integer minor units."""

    return PricingBreakdown(
        merchandiseAmount=merchandise_amount,
        customerTotal=merchandise_amount,
        merchantPayableAmount=merchandise_amount,
        calculatedAt=calculated_at,
    )


class CheckoutMandate(StrictModel):
    """AP2 v0.2 closed Checkout Mandate claims used by the simulation."""

    vct: Literal["mandate.checkout.1"] = "mandate.checkout.1"
    checkout_jwt: StrictStr = Field(min_length=1)
    checkout_hash: Base64UrlSha256
    iat: NonNegativeInt
    exp: PositiveInt

    @model_validator(mode="after")
    def expiry_after_issue(self) -> "CheckoutMandate":
        if self.exp <= self.iat:
            raise ValueError("exp must be later than iat")
        return self


class PaymentPayee(StrictModel):
    id: Literal["mediation-platform"] = UPSTREAM_PAYEE_ID
    name: Literal["Secure Mediation Marketplace"] = UPSTREAM_PAYEE_NAME


class PaymentAmount(StrictModel):
    amount: NonNegativeInt
    currency: Literal["USD"] = CURRENCY


class PaymentInstrument(StrictModel):
    id: StrictStr = Field(min_length=1)
    type: StrictStr = Field(min_length=1)
    description: StrictStr | None = None


class PaymentMandate(StrictModel):
    """AP2 v0.2 closed Payment Mandate claims used by the simulation."""

    vct: Literal["mandate.payment.1"] = "mandate.payment.1"
    transaction_id: Base64UrlSha256
    payee: PaymentPayee
    payment_amount: PaymentAmount
    payment_instrument: PaymentInstrument
    iat: NonNegativeInt
    exp: PositiveInt

    @model_validator(mode="after")
    def expiry_after_issue(self) -> "PaymentMandate":
        if self.exp <= self.iat:
            raise ValueError("exp must be later than iat")
        return self


class ProjectAuthorization(StrictModel):
    """Project-local authorization kept outside the AP2 mandate claims."""

    profile: Literal[PROFILE_URI] = PROFILE_URI
    simulated: Literal[True] = SIMULATED
    subject: StrictStr = Field(min_length=1)
    kid: StrictStr = Field(min_length=1)
    audience: StrictStr = Field(min_length=1)
    nonce: StrictStr = Field(min_length=1)
    order_id: StrictStr = Field(alias="orderId", min_length=1)
    task_id: StrictStr = Field(alias="taskId", min_length=1)
    quote_id: StrictStr = Field(alias="quoteId", min_length=1)
    challenge_id: StrictStr = Field(alias="challengeId", min_length=1)
    checkout_mandate_digest: Sha256Digest = Field(alias="checkoutMandateDigest")
    payment_mandate_digest: Sha256Digest = Field(alias="paymentMandateDigest")
    asset: Literal["USD"] = ASSET
    network: Literal["demo:local"] = UPSTREAM_NETWORK
    decimals: Literal[2] = DECIMALS
    iat: NonNegativeInt
    exp: PositiveInt
    signature: Signature | None = None

    @model_validator(mode="after")
    def validate_authorization(self) -> "ProjectAuthorization":
        if self.exp <= self.iat:
            raise ValueError("exp must be later than iat")
        if self.signature is not None and self.signature.kid != self.kid:
            raise ValueError("signature kid must match authorization kid")
        return self


class TrustedSurfaceDisplay(StrictModel):
    checkout_jwt: StrictStr = Field(min_length=1)
    pricing: PricingBreakdown
    payee: PaymentPayee
    payment_instrument: PaymentInstrument


class TrustedSurfaceApproval(StrictModel):
    display: TrustedSurfaceDisplay
    checkout_mandate: CheckoutMandate = Field(alias="checkoutMandate")
    payment_mandate: PaymentMandate = Field(alias="paymentMandate")
    authorization: ProjectAuthorization

    @model_validator(mode="after")
    def mandates_share_checkout_hash(self) -> "TrustedSurfaceApproval":
        if self.checkout_mandate.checkout_hash != self.payment_mandate.transaction_id:
            raise ValueError("Payment transaction_id must equal Checkout checkout_hash")
        return self


class PaymentResource(StrictModel):
    url: StrictStr = Field(min_length=1)
    description: StrictStr = Field(min_length=1)
    mime_type: Literal["application/json"] = Field(
        default="application/json",
        alias="mimeType",
    )


class PaymentAcceptanceExtra(StrictModel):
    profile: Literal[PROFILE_URI] = PROFILE_URI
    simulated: Literal[True] = SIMULATED
    quote_digest: Sha256Digest = Field(alias="quoteDigest")


class PaymentAcceptance(StrictModel):
    scheme: StrictStr = Field(min_length=1)
    network: StrictStr = Field(min_length=1)
    amount: MinorUnitString
    asset: Literal["USD"] = ASSET
    decimals: Literal[2] = DECIMALS
    pay_to: StrictStr = Field(alias="payTo", min_length=1)
    max_timeout_seconds: PositiveInt | None = Field(
        default=None,
        alias="maxTimeoutSeconds",
    )
    extra: PaymentAcceptanceExtra | dict[str, Any]


class PaymentRequired(StrictModel):
    x402_version: Literal[2] = Field(default=2, alias="x402Version")
    resource: PaymentResource
    accepts: list[PaymentAcceptance] = Field(min_length=1, max_length=1)


class PaymentPayload(StrictModel):
    x402_version: Literal[2] = Field(default=2, alias="x402Version")
    resource: PaymentResource | None = None
    accepted: PaymentAcceptance
    payload: dict[str, Any]


class ReceiptEnvelope(StrictModel):
    receipt_type: ReceiptType = Field(alias="receiptType")
    receipt_id: StrictStr = Field(alias="receiptId", min_length=1)
    profile: Literal[PROFILE_URI] = PROFILE_URI
    simulated: Literal[True] = SIMULATED
    status: StrictStr = Field(min_length=1)
    issued_at: datetime = Field(alias="issuedAt")
    issuer: StrictStr = Field(min_length=1)
    subject: StrictStr = Field(min_length=1)
    order_id: StrictStr = Field(alias="orderId", min_length=1)
    related_digests: dict[str, Sha256Digest] = Field(alias="relatedDigests")
    references: dict[str, StrictStr] = Field(default_factory=dict)
    signature: Signature | None = None

    @field_validator("issued_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("issuedAt must be timezone-aware")
        return value


class AP2PaymentMetadata(StrictModel):
    checkout_mandate: CheckoutMandate | None = Field(
        default=None,
        alias="checkoutMandate",
    )
    payment_mandate: PaymentMandate | None = Field(
        default=None,
        alias="paymentMandate",
    )
    authorization: ProjectAuthorization | None = None
    payment_receipt: ReceiptEnvelope | None = Field(
        default=None,
        alias="paymentReceipt",
    )


class X402PaymentMetadata(StrictModel):
    extension: Literal[PROFILE_URI] = PROFILE_URI
    profile: Literal[PROFILE_URI] = PROFILE_URI
    simulated: Literal[True] = SIMULATED
    status: PaymentStatus
    leg: PaymentLeg
    receipts: list[ReceiptEnvelope] = Field(default_factory=list)
    requirement: PaymentRequired | None = None
    payload: PaymentPayload | None = None
    settlement: dict[str, Any] | None = None
    error: StableError | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> "X402PaymentMetadata":
        if self.status == PaymentStatus.PAYMENT_REQUIRED and self.requirement is None:
            raise ValueError("payment-required status requires requirement")
        if self.status == PaymentStatus.PAYMENT_SUBMITTED and self.payload is None:
            raise ValueError("payment-submitted status requires payload")
        if self.status == PaymentStatus.PAYMENT_COMPLETED:
            if self.settlement is None or not self.receipts:
                raise ValueError("payment-completed requires settlement and receipts")
        if self.status == PaymentStatus.PAYMENT_FAILED and self.error is None:
            raise ValueError("payment-failed status requires error")
        return self


class MarketplacePaymentMetadata(StrictModel):
    order_id: StrictStr = Field(alias="orderId", min_length=1)
    merchant_id: StrictStr = Field(alias="merchantId", min_length=1)
    quote_id: StrictStr = Field(alias="quoteId", min_length=1)
    correlation_id: StrictStr = Field(alias="correlationId", min_length=1)
    payable_entry_id: StrictStr | None = Field(default=None, alias="payableEntryId")
    guarantee_id: StrictStr | None = Field(default=None, alias="guaranteeId")
    fulfillment_id: StrictStr | None = Field(default=None, alias="fulfillmentId")
    refund_id: StrictStr | None = Field(default=None, alias="refundId")
    payout_id: StrictStr | None = Field(default=None, alias="payoutId")


class PaymentMetadataEnvelope(StrictModel):
    x402_payment: X402PaymentMetadata = Field(alias="x402.payment")
    marketplace_payment: MarketplacePaymentMetadata = Field(alias="marketplace.payment")
    ap2_payment: AP2PaymentMetadata | None = Field(default=None, alias="ap2.payment")
