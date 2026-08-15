"""Canonical bytes and digests used by authorization and evidence records."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import rfc8785


def _reject_float(value: Any) -> None:
    if isinstance(value, float):
        raise TypeError("floating point values are forbidden in signed domain objects")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical object keys must be strings")
            _reject_float(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_float(child)


def canonical_bytes(value: Any) -> bytes:
    """Return RFC 8785 bytes after enforcing the no-float amount policy."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True, exclude_none=True)
    _reject_float(value)
    return rfc8785.dumps(value)


def canonical_json(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def sha256_digest(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical_digest(value: Any) -> str:
    return sha256_digest(canonical_bytes(value))


def parse_json_strict(raw: str | bytes) -> Any:
    """Parse JSON while rejecting duplicate object names and floats."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    parsed = json.loads(raw, object_pairs_hook=object_pairs, parse_float=lambda _: (_ for _ in ()).throw(ValueError("floats are forbidden")))
    _reject_float(parsed)
    return parsed
