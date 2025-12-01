from typing import Literal, Optional, Any, List
from pydantic import BaseModel, Field


# Core event envelope
class JuryStreamEvent(BaseModel):
    type: Literal[
        "evaluation_started",
        "phase_started",
        "phase_change",
        "juror_evaluation",
        "consensus_check",
        "round_started",
        "discussion_start",
        "juror_statement",
        "round_completed",
        "round_complete",
        "final_judgment",
        "evaluation_completed",
        "stage_update",
        "score_update",
        "submission_state_change",
        "model_switch",
        "safety_block",
        "message_chunk",
        "tool_call",
        "tool_response",
        "error",
    ]
    # Optional payload fields; individual handlers may use subsets.
    content: Optional[str] = None
    role: Optional[str] = None
    model: Optional[str] = None
    sequence: Optional[int] = None
    timestamp: Optional[float] = None
    # Generic payload for backward compatibility
    data: Optional[Any] = None
    # Common judge fields
    juror: Optional[str] = None
    verdict: Optional[str] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    phase: Optional[str] = None
    phaseNumber: Optional[int] = None
    round: Optional[int] = None
    positionChanged: Optional[bool] = None
    newVerdict: Optional[str] = None
    newScore: Optional[float] = None
    consensusStatus: Optional[str] = None
    consensusReached: Optional[bool] = None
    consensusVerdict: Optional[str] = None
    safety_categories: Optional[List[str]] = Field(default=None, description="Categories that triggered safety block")
    block_reason: Optional[str] = None
    # AISI 4軸スコア（Phase 1用）
    taskCompletion: Optional[int] = None  # 0-40
    toolUsage: Optional[int] = None       # 0-30
    autonomy: Optional[int] = None        # 0-20
    safety: Optional[int] = None          # 0-10


def validate_event_dict(payload: dict) -> dict:
    """
    Validate and normalize outgoing SSE/WS payloads for Jury Judge stream.
    Returns the cleaned dict (original on validation failure to avoid crashing).
    """
    try:
        return JuryStreamEvent(**payload).model_dump(exclude_none=True)
    except Exception:
        # Do not raise; keep original for best-effort delivery
        return payload
