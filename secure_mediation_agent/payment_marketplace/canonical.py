"""Deterministic JSON, digest, and test-only HMAC helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .config import UnknownKeyIdError, resolve_test_key
from .models import Signature


class CanonicalizationError(ValueError):
    """Raised when input cannot be represented by the project canonical JSON."""


class DuplicateKeyError(CanonicalizationError):
    """Raised when parsed JSON contains the same object key more than once."""


class InvalidSignatureError(ValueError):
    """Raised when an HMAC signature does not verify."""


KeyResolver = Callable[[str], bytes]


def _reject_float(_: str) -> None:
    raise CanonicalizationError("floating-point JSON numbers are not supported")


def _reject_constant(value: str) -> None:
    raise CanonicalizationError(f"non-finite JSON number is not supported: {value}")


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def loads_strict(data: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting duplicates, floats, and non-finite numbers."""

    if isinstance(data, bytearray):
        data = bytes(data)
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalizationError("JSON must be valid UTF-8") from exc
    if not isinstance(data, str):
        raise TypeError("JSON input must be str, bytes, or bytearray")
    try:
        return json.loads(
            data,
            object_pairs_hook=_pairs_without_duplicates,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except CanonicalizationError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise CanonicalizationError("invalid JSON") from exc


def _model_to_wire(value: BaseModel) -> Any:
    return value.model_dump(mode="json", by_alias=True, exclude_none=True)


def _validate_json_value(value: Any, path: str = "$") -> Any:
    if isinstance(value, BaseModel):
        return _validate_json_value(_model_to_wire(value), path)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise CanonicalizationError(f"float is not supported at {path}")
    if isinstance(value, list):
        return [
            _validate_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(f"object key is not a string at {path}")
            normalized[key] = _validate_json_value(item, f"{path}.{key}")
        return normalized
    raise CanonicalizationError(
        f"unsupported canonical JSON type at {path}: {type(value).__name__}"
    )


def _signature_payload(value: Any) -> Any:
    normalized = _validate_json_value(value)
    if not isinstance(normalized, dict):
        raise CanonicalizationError("a signed payload must be a JSON object")
    unsigned = dict(normalized)
    unsigned.pop("signature", None)
    return unsigned


def canonical_bytes(value: Any) -> bytes:
    """Serialize JSON using recursively sorted keys and compact UTF-8."""

    normalized = _validate_json_value(value)
    try:
        text = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CanonicalizationError("value is not canonicalizable JSON") from exc


def canonical_json(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def base64url_encode(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("base64url input must be bytes")
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def base64url_decode(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise CanonicalizationError("base64url value must be a non-empty string")
    if "=" in value:
        raise CanonicalizationError("base64url padding is not allowed")
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, UnicodeEncodeError) as exc:
        raise CanonicalizationError("invalid base64url value") from exc
    if base64url_encode(decoded) != value:
        raise CanonicalizationError("non-canonical base64url value")
    return decoded


def sha256_bytes(data: bytes) -> bytes:
    if not isinstance(data, bytes):
        raise TypeError("SHA-256 input must be bytes")
    return hashlib.sha256(data).digest()


def base64url_sha256(data: bytes) -> str:
    """Return an unpadded base64url SHA-256 digest."""

    return base64url_encode(sha256_bytes(data))


def checkout_hash(checkout_jwt: str) -> str:
    """Hash the exact UTF-8 bytes of the Checkout Mandate field value."""

    if not isinstance(checkout_jwt, str) or not checkout_jwt:
        raise CanonicalizationError("checkout_jwt must be a non-empty string")
    try:
        exact_bytes = checkout_jwt.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CanonicalizationError("checkout_jwt must be valid UTF-8") from exc
    return base64url_sha256(exact_bytes)


def digest_object(value: Any) -> str:
    """Return the PROFILE-018 canonical object digest."""

    return f"sha256:{hashlib.sha256(canonical_bytes(value)).hexdigest()}"


def sign_payload(
    value: Any,
    *,
    kid: str,
    key_resolver: KeyResolver = resolve_test_key,
) -> Signature:
    """Sign a JSON object, excluding its top-level ``signature`` field."""

    key = key_resolver(kid)
    message = canonical_bytes(_signature_payload(value))
    signature = base64url_encode(hmac.new(key, message, hashlib.sha256).digest())
    return Signature(kid=kid, value=signature)


def verify_payload_signature(
    value: Any,
    signature: Signature | dict[str, Any] | None = None,
    *,
    key_resolver: KeyResolver = resolve_test_key,
    expected_kid: str | None = None,
) -> None:
    """Verify an HS256 object signature or raise a fail-closed exception."""

    normalized = _validate_json_value(value)
    if not isinstance(normalized, dict):
        raise InvalidSignatureError("a signed payload must be a JSON object")

    embedded = normalized.get("signature")
    if signature is None:
        if embedded is None:
            raise InvalidSignatureError("signature is missing")
        signature = Signature.model_validate(embedded)
    elif not isinstance(signature, Signature):
        signature = Signature.model_validate(signature)

    if signature.alg != "HS256":
        raise InvalidSignatureError("unsupported signature algorithm")
    if expected_kid is not None and signature.kid != expected_kid:
        raise InvalidSignatureError("signature key is not the expected issuer key")
    try:
        key = key_resolver(signature.kid)
    except UnknownKeyIdError:
        raise

    message = canonical_bytes(_signature_payload(normalized))
    expected = base64url_encode(hmac.new(key, message, hashlib.sha256).digest())
    if not hmac.compare_digest(expected, signature.value):
        raise InvalidSignatureError("signature verification failed")


def with_signature(
    value: Any,
    *,
    kid: str,
    key_resolver: KeyResolver = resolve_test_key,
) -> dict[str, Any]:
    """Return a new wire dictionary with a deterministic signature attached."""

    normalized = _signature_payload(value)
    signature = sign_payload(normalized, kid=kid, key_resolver=key_resolver)
    return {**normalized, "signature": signature.wire_dict()}
