"""Only public ADK boundary for the internal secure mediation controller."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import hashlib
import os
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import PrivateAttr

from secure_mediation_agent.ap2.keys import load_role_key
from secure_mediation_agent.identity import (
    ADK_IDENTITY_STATE_KEY,
    verify_identity_assertion,
)
from .authority import HttpMediationAuthority
from .canonical import canonical_bytes
from .errors import MediationError
from .models import MediationPublicView, SubjectScope, TextPart


class SecureMediationAdapter(BaseAgent):
    """ADK-compatible UI adapter for the workflow-owned mediation authority."""

    _authority: HttpMediationAuthority | None = PrivateAttr(default=None)
    _authority_factory: Callable[[], HttpMediationAuthority] = PrivateAttr()

    def __init__(
        self,
        *,
        authority: HttpMediationAuthority | None = None,
        authority_factory: Callable[[], HttpMediationAuthority] = HttpMediationAuthority,
        **data: Any,
    ) -> None:
        super().__init__(**data)
        self._authority = authority
        self._authority_factory = authority_factory

    def _resolved_authority(self) -> HttpMediationAuthority:
        if self._authority is None:
            self._authority = self._authority_factory()
        return self._authority

    @staticmethod
    def _scope(context: InvocationContext) -> SubjectScope:
        assertion = context.session.state.get(ADK_IDENTITY_STATE_KEY)
        key_dir = os.environ.get("AP2_DEMO_KEY_DIR")
        if not isinstance(assertion, str) or not assertion or not key_dir:
            raise MediationError(
                "VERIFIED_IDENTITY_REQUIRED", "検証済みの利用者情報がありません。"
            )
        try:
            identity = verify_identity_assertion(
                assertion, load_role_key(key_dir, "service_auth")
            )
        except Exception as error:
            raise MediationError(
                "VERIFIED_IDENTITY_MISMATCH", "検証済みの利用者情報が一致しません。"
            ) from error
        subject = identity.subject
        tenant_id = identity.tenant_id
        actual_session = str(context.session.id)
        if (
            not actual_session
            or str(context.session.user_id) != subject
        ):
            raise MediationError(
                "VERIFIED_IDENTITY_MISMATCH", "検証済みの利用者情報が一致しません。"
            )
        return SubjectScope(
            subject=subject,
            tenantId=tenant_id,
            adkSessionId=actual_session,
        )

    @staticmethod
    def _parts(context: InvocationContext) -> tuple[TextPart, ...]:
        if context.user_content is None:
            return ()
        values: list[TextPart] = []
        for part in context.user_content.parts or ():
            if part.text is None:
                raise MediationError(
                    "TEXT_ONLY_RELEASE", "このリリースではテキストだけを利用できます。"
                )
            values.append(TextPart(text=part.text))
        return tuple(values)

    @staticmethod
    def _approval_display(view: Any) -> str:
        if view.approval_target is None:
            return ""
        target_json = canonical_bytes(view.approval_target).decode("utf-8")
        target_digest = view.approval_target_digest or "-"
        return (
            f"\n承認対象 (canonical JSON): {target_json}"
            f"\n承認対象digest: {target_digest}"
            "\nこの対象を承認する場合のみ、メッセージ全体を完全一致「承認」として"
            "送信してください。"
        )

    @staticmethod
    def _request_id(context: InvocationContext) -> str:
        value = (
            f"{context.session.id}\0{context.invocation_id}".encode("utf-8")
        )
        return f"adk-{hashlib.sha256(value).hexdigest()}"

    @classmethod
    def _reply(cls, view: MediationPublicView) -> str:
        durability_notice = (
            "デモ環境: 再起動すると進行中の状態は失われます（耐久性保証なし）。\n"
            if view.durability_profile == "ephemeral-demo"
            else ""
        )
        refs = " / ".join(
            value
            for value in (view.plan_ref, view.step_ref, view.task_ref)
            if value is not None
        )
        return (
            f"{durability_notice}{view.message}\n状態: {view.state.value}"
            f"\n参照: {refs or '-'}"
            f"{cls._approval_display(view)}"
            "\nシミュレーション: x402 wire-shape fixture (NOT CONFORMANT)"
        )

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        try:
            # Validate the assertion against the actual ADK user/session before
            # forwarding it.  The workflow process verifies it again and owns
            # the only controller/store; this process never creates a fallback.
            self._scope(context)
            assertion = context.session.state[ADK_IDENTITY_STATE_KEY]
            view = await self._resolved_authority().turn(
                assertion=assertion,
                parts=self._parts(context),
                request_id=self._request_id(context),
            )
            reply = self._reply(view)
        except MediationError as error:
            reply = (
                f"処理できませんでした: {error.code}\n{error.safe_message}"
                "\nシミュレーション: x402 wire-shape fixture (NOT CONFORMANT)"
            )
        except Exception:
            reply = (
                "処理できませんでした: MEDIATION_INTERNAL_ERROR"
                "\nシミュレーション: x402 wire-shape fixture (NOT CONFORMANT)"
            )
        yield Event(
            invocationId=context.invocation_id,
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=reply)]),
        )
