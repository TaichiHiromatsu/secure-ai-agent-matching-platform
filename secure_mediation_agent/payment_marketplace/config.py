"""Fixed configuration for the project-local AP2/x402 simulation profile.

The HMAC material in this module is a public, test-only vector.  Callers must
still treat it like a credential at runtime: it must never be returned from an
API, written to a log, or copied into an LLM prompt/artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping


PROFILE_URI: Final = "urn:secure-a2a:extensions:ap2-x402-marketplace:v1"
PROFILE_VERSION: Final = 1
SIMULATED: Final = True

A2A_EXTENSION_HEADER: Final = "X-A2A-Extensions"
A2A_SDK_PACKAGE: Final = "a2a-sdk"
A2A_SDK_VERSION: Final = "0.3.19"
A2A_WIRE_PROTOCOL_VERSION: Final = "0.3.0"

CURRENCY: Final = "USD"
ASSET: Final = "USD"
DECIMALS: Final = 2
ROUNDING_RULE: Final = "minor-unit-exact"
PRICING_POLICY_VERSION: Final = "zero-fee-v1"

UPSTREAM_SCHEME: Final = "exact-simulated"
UPSTREAM_NETWORK: Final = "demo:local"
UPSTREAM_PAYEE_ID: Final = "mediation-platform"
UPSTREAM_PAYEE_NAME: Final = "Secure Mediation Marketplace"
UPSTREAM_MAX_TIMEOUT_SECONDS: Final = 300

MERCHANT_CREDIT_SCHEME: Final = "platform-credit"
MERCHANT_CREDIT_NETWORK: Final = "demo:mediation-ledger"

CUSTOMER_SUBJECT: Final = "demo-customer"
CUSTOMER_KID: Final = "demo-customer-hmac-v1"
MEDIATOR_SUBJECT: Final = "mediation-platform"
MEDIATOR_KID: Final = "demo-mediator-hmac-v1"
MERCHANT_SUBJECT: Final = "demo-merchant"
MERCHANT_KID: Final = "demo-merchant-hmac-v1"
OPERATOR_SUBJECT: Final = "demo-operator"
OPERATOR_KID: Final = "demo-operator-hmac-v1"

CUSTOMER_INITIAL_BALANCE: Final = 100_000
PLATFORM_INITIAL_BALANCE: Final = 0
MERCHANT_INITIAL_BALANCE: Final = 0


class UnknownKeyIdError(ValueError):
    """Raised when a signature references an unregistered key identifier."""


@dataclass(frozen=True, slots=True, repr=False)
class TestKeyRecord:
    """A test signing identity whose repr deliberately redacts key bytes."""

    subject: str
    kid: str
    _key: bytes

    def __repr__(self) -> str:
        return (
            f"TestKeyRecord(subject={self.subject!r}, kid={self.kid!r}, "
            "key=<redacted>)"
        )

    def signing_key(self) -> bytes:
        """Return the test key to deterministic crypto code only."""

        return self._key


_TEST_KEY_REGISTRY: Final[Mapping[str, TestKeyRecord]] = MappingProxyType(
    {
        CUSTOMER_KID: TestKeyRecord(
            CUSTOMER_SUBJECT,
            CUSTOMER_KID,
            b"test-only-demo-customer-key-v1",
        ),
        MEDIATOR_KID: TestKeyRecord(
            MEDIATOR_SUBJECT,
            MEDIATOR_KID,
            b"test-only-demo-mediator-key-v1",
        ),
        MERCHANT_KID: TestKeyRecord(
            MERCHANT_SUBJECT,
            MERCHANT_KID,
            b"test-only-demo-merchant-key-v1",
        ),
        OPERATOR_KID: TestKeyRecord(
            OPERATOR_SUBJECT,
            OPERATOR_KID,
            b"test-only-demo-operator-key-v1",
        ),
    }
)

TEST_KEY_IDS: Final[tuple[str, ...]] = tuple(_TEST_KEY_REGISTRY)

INITIAL_RAIL_BALANCES: Final[Mapping[str, int]] = MappingProxyType(
    {
        CUSTOMER_SUBJECT: CUSTOMER_INITIAL_BALANCE,
        MEDIATOR_SUBJECT: PLATFORM_INITIAL_BALANCE,
        MERCHANT_SUBJECT: MERCHANT_INITIAL_BALANCE,
    }
)


def resolve_test_key(kid: str) -> bytes:
    """Resolve a fixed test key without exposing the registry through APIs."""

    try:
        return _TEST_KEY_REGISTRY[kid].signing_key()
    except KeyError as exc:
        raise UnknownKeyIdError(f"unknown key id: {kid}") from exc


def subject_for_kid(kid: str) -> str:
    """Return the fixed subject bound to ``kid``."""

    try:
        return _TEST_KEY_REGISTRY[kid].subject
    except KeyError as exc:
        raise UnknownKeyIdError(f"unknown key id: {kid}") from exc


def public_test_key_registry() -> tuple[dict[str, str], ...]:
    """Return non-secret key metadata suitable for readiness checks."""

    return tuple(
        {"subject": record.subject, "kid": record.kid}
        for record in _TEST_KEY_REGISTRY.values()
    )
