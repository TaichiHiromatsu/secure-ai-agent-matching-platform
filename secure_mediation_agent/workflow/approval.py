"""Exact approval dispatcher and project-local ES256 authorizations."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any, Iterable

from ap2.sdk.jwt_helper import create_jwt, verify_jwt
from jwcrypto.jwk import JWK

from secure_mediation_agent.ap2.keys import public_key

from .errors import DomainError
from .models import MessagePart, WorkflowState


class ApprovalAction(StrEnum):
    APPROVE_PLAN = "approve-plan"
    APPROVE_PAYMENT = "approve-payment"
    REJECT_CURRENT = "reject-current"
    HANDLE_MESSAGE = "handle-message"


def dispatch(parts: Iterable[MessagePart], state: WorkflowState | str) -> ApprovalAction:
    values = tuple(parts)
    exact_approval = (
        len(values) == 1
        and values[0].kind == "text"
        and values[0].text == "承認"
    )
    exact_rejection = (
        len(values) == 1
        and values[0].kind == "text"
        and values[0].text == "拒否"
    )
    current = WorkflowState(state)
    if exact_approval and current == WorkflowState.PLAN_APPROVAL_REQUIRED:
        return ApprovalAction.APPROVE_PLAN
    if exact_approval and current == WorkflowState.PAYMENT_APPROVAL_REQUIRED:
        return ApprovalAction.APPROVE_PAYMENT
    if exact_approval:
        raise DomainError(
            "APPROVAL_NOT_PENDING",
            "No approval is pending for this workflow state.",
            "approval",
            current_state=current,
        )
    if current in {
        WorkflowState.PLAN_APPROVAL_REQUIRED,
        WorkflowState.PAYMENT_APPROVAL_REQUIRED,
    }:
        if exact_rejection:
            return ApprovalAction.REJECT_CURRENT
        raise DomainError(
            "APPROVAL_EXACT_TOKEN_REQUIRED",
            "Approval requires one text part exactly equal to 承認.",
            "approval",
            current_state=current,
            expected_action="承認 or 拒否",
        )
    return ApprovalAction.HANDLE_MESSAGE


class AuthorizationService:
    issuer = "secure-mediation-plan-authority"

    def __init__(self, key: JWK) -> None:
        self._key = key

    def issue_plan_authorization(self, claims: dict[str, Any]) -> str:
        payload = {
            "typ": "secure-plan-authorization+jwt",
            "iss": self.issuer,
            **claims,
        }
        return self._sign(payload)

    def issue_capability(self, claims: dict[str, Any]) -> str:
        payload = {
            "typ": "secure-downstream-capability+jwt",
            "iss": self.issuer,
            **claims,
        }
        return self._sign(payload)

    def verify(
        self,
        token: str,
        *,
        expected_type: str,
        audience: str,
        operation: str | None = None,
        now: int | None = None,
    ) -> dict[str, Any]:
        payload = verify_jwt(token, public_key(self._key))
        current = int(time.time()) if now is None else now
        if payload.get("iss") != self.issuer or payload.get("typ") != expected_type:
            raise ValueError("authorization issuer/type mismatch")
        if payload.get("aud") != audience:
            raise ValueError("authorization audience mismatch")
        if operation is not None and payload.get("operation") != operation:
            raise ValueError("authorization operation mismatch")
        if not isinstance(payload.get("iat"), int) or not isinstance(payload.get("exp"), int):
            raise ValueError("authorization time claims missing")
        if current < payload["iat"] - 300 or current > payload["exp"]:
            raise ValueError("authorization expired or not yet valid")
        return payload

    def _sign(self, payload: dict[str, Any]) -> str:
        return create_jwt(
            {"alg": "ES256", "kid": self._key.get("kid"), "typ": "JWT"},
            payload,
            self._key,
        )
