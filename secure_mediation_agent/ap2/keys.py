"""File-backed, role-separated P-256 signing keys."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, fields
from pathlib import Path

from jwcrypto.jwk import JWK


ROLE_KIDS = {
    "plan_authority": "demo-plan-authority-1",
    "user_root": "demo-user-credential-issuer-1",
    "trusted_surface": "demo-trusted-surface-1",
    "merchant": "demo-merchant-1",
    "credential_provider": "demo-credential-provider-1",
    "simulation_signer": "demo-simulation-signer-1",
    "mpp": "demo-mpp-1",
    "service_auth": "demo-service-auth-1",
}


def generate_key(kid: str) -> JWK:
    key = JWK.generate(kty="EC", crv="P-256")
    value = json.loads(key.export())
    # Keep the private JWK minimal.  The pinned AP2/sd-jwt stack derives
    # ES256 from P-256 and rejects some otherwise-valid key metadata while
    # walking the delegated holder ``cnf`` chain.
    value["kid"] = kid
    return JWK.from_json(json.dumps(value))


def public_key(key: JWK) -> JWK:
    return JWK.from_json(key.export_public())


def load_role_key(directory: str | Path, role: str) -> JWK:
    """Load one configured private role key without exposing sibling authorities."""

    if role not in ROLE_KIDS:
        raise ValueError(f"unknown signing role: {role}")
    root = Path(directory).resolve(strict=True)
    path = (root / f"{role}.jwk").resolve(strict=True)
    if path.parent != root:
        raise ValueError("role key escaped configured secret directory")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(f"role key {path} must not be group/world accessible")
    key = JWK.from_json(path.read_text(encoding="utf-8"))
    if (
        key.get("kid") != ROLE_KIDS[role]
        or key.get("crv") != "P-256"
        or not key.has_private
    ):
        raise ValueError(f"invalid role key for {role}")
    return key


@dataclass(frozen=True, slots=True)
class DemoKeySet:
    plan_authority: JWK
    user_root: JWK
    trusted_surface: JWK
    merchant: JWK
    credential_provider: JWK
    simulation_signer: JWK
    mpp: JWK
    service_auth: JWK

    @classmethod
    def generate_for_test(cls) -> "DemoKeySet":
        return cls(**{role: generate_key(kid) for role, kid in ROLE_KIDS.items()})

    @classmethod
    def load(cls, directory: str | Path) -> "DemoKeySet":
        values = {role: load_role_key(directory, role) for role in ROLE_KIDS}
        return cls(**values)

    def public_manifest(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for item in fields(self):
            key = getattr(self, item.name)
            result[item.name] = {
                "issuer": ROLE_KIDS[item.name].rsplit("-1", 1)[0],
                "kid": key.get("kid"),
                "jwk": json.loads(key.export_public()),
                "status": "active",
                "version": "demo-es256-v1",
            }
        return result

    @classmethod
    def from_environment(cls) -> "DemoKeySet":
        directory = os.environ.get("AP2_DEMO_KEY_DIR")
        if not directory:
            if os.environ.get("APP_ENV") == "test":
                return cls.generate_for_test()
            raise RuntimeError("AP2_DEMO_KEY_DIR is required; startup never generates role keys")
        return cls.load(directory)
