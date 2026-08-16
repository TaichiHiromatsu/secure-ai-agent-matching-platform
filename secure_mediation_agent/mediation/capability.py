"""Operation-scoped Merchant capability creation for the exact A2A request."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from secure_mediation_agent.workflow.approval import AuthorizationService

from .a2a_executor import A2AOperation
from .errors import SecurityBlocked


class PlanAuthorityOperationAuthorizer:
    def __init__(self, authorization: AuthorizationService) -> None:
        self.authorization = authorization

    async def authorize(self, operation: A2AOperation) -> A2AOperation:
        if operation.agent.canonical_agent_id != "agent-005":
            return operation
        params = operation.request.get("params")
        if not isinstance(params, dict):
            raise SecurityBlocked("CAPABILITY_INPUT_INVALID", "A2A params are invalid.")
        required = {
            "action",
            "workflowId",
            "taskId",
            "orderId",
            "capabilityId",
        }
        if not required.issubset(params) or not all(
            isinstance(params[name], str) and params[name] for name in required
        ):
            raise SecurityBlocked(
                "CAPABILITY_INPUT_INVALID", "The paid A2A capability scope is incomplete."
            )
        now = int(datetime.now(timezone.utc).timestamp())
        token = self.authorization.issue_capability(
            {
                "ver": 1,
                "aud": "merchant:demo-merchant",
                "sub": "secure-mediator",
                "jti": params["capabilityId"],
                "iat": now,
                "nbf": now,
                "exp": now + 300,
                "operation": params["action"],
                "workflowId": params["workflowId"],
                "taskId": params["taskId"],
                "orderId": params["orderId"],
                "requestDigest": operation.request_digest,
            }
        )
        return operation.model_copy(update={"authorization": token})
