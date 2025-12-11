"""
Artifact Tools for Jury Judge - FunctionTool definitions.

Provides tools for jurors to fetch evaluation artifact data on-demand.
These tools are designed to work with Google ADK's FunctionTool format
and can be converted to OpenAI/Anthropic tool formats.

Design:
- Summary data is provided in initial context
- Jurors call tools to fetch detailed records when needed
- Both failure and success cases can be retrieved
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


@dataclass
class ArtifactContext:
    """
    Artifact context containing URIs and cached data.

    This is stored in session.state and accessed by tools.
    """
    security_gate_uri: Optional[str] = None
    agent_card_uri: Optional[str] = None
    security_gate_summary: Optional[Dict[str, Any]] = None
    agent_card_summary: Optional[Dict[str, Any]] = None
    # Cached full records for local mode (when WandB unavailable)
    security_gate_records: Optional[List[Dict[str, Any]]] = None
    agent_card_records: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# Core Artifact Fetching Functions (Provider-agnostic)
# =============================================================================

def _get_artifact_records_from_context(
    artifact_context: ArtifactContext,
    artifact_type: str,
) -> List[Dict[str, Any]]:
    """
    Get artifact records from context (cached) or fetch from URI.

    Args:
        artifact_context: The artifact context with URIs and cached data
        artifact_type: "security_gate" or "agent_card"

    Returns:
        List of records from the artifact
    """
    # Try cached records first (for local mode)
    if artifact_type == "security_gate":
        if artifact_context.security_gate_records:
            return artifact_context.security_gate_records
        uri = artifact_context.security_gate_uri
    else:
        if artifact_context.agent_card_records:
            return artifact_context.agent_card_records
        uri = artifact_context.agent_card_uri

    if not uri:
        return []

    # Import here to avoid circular dependency
    try:
        from evaluation_runner.artifact_storage import fetch_artifact_content
        return fetch_artifact_content(uri, max_records=1000)
    except ImportError:
        logger.warning("artifact_storage module not available")
        return []


def _filter_records(
    records: List[Dict[str, Any]],
    filter_verdicts: Optional[List[str]] = None,
    max_records: int = 10,
) -> List[Dict[str, Any]]:
    """
    Filter records by verdict and limit count.
    """
    if not records:
        return []

    filtered = []
    for record in records:
        if filter_verdicts:
            # Check both top-level and nested evaluation.verdict
            verdict = record.get("verdict", "")
            if not verdict and "evaluation" in record:
                if isinstance(record["evaluation"], dict):
                    verdict = record["evaluation"].get("verdict", "")
            if verdict not in filter_verdicts:
                continue
        filtered.append(record)
        if len(filtered) >= max_records:
            break

    return filtered


def _format_security_gate_record(record: Dict[str, Any], max_len: int = 300) -> Dict[str, Any]:
    """Format a security gate record for display."""
    return {
        "verdict": record.get("verdict", "unknown"),
        "dataset": record.get("datasetSource") or record.get("dataset_source") or "unknown",
        "prompt": (record.get("prompt") or record.get("prompt_text") or "")[:max_len],
        "response": (record.get("response") or record.get("response_text") or "")[:max_len],
        "reason": (record.get("reason") or "")[:200],
    }


def _format_agent_card_record(record: Dict[str, Any], max_len: int = 300) -> Dict[str, Any]:
    """Format an agent card accuracy record for display."""
    evaluation = record.get("evaluation", {}) if isinstance(record.get("evaluation"), dict) else {}

    verdict = record.get("verdict", "")
    if not verdict and evaluation:
        verdict = evaluation.get("verdict", "")

    similarity = evaluation.get("similarity") if evaluation else record.get("similarity")
    if similarity is None:
        similarity = 0.0

    return {
        "verdict": verdict,
        "skill": (record.get("skill") or record.get("skillName") or
                 record.get("useCase") or record.get("use_case") or "unknown"),
        "prompt": (record.get("prompt") or "")[:max_len],
        "expected": (record.get("expected") or record.get("expectedResponse") or "")[:200],
        "actual": (record.get("response") or record.get("actualResponse") or "")[:max_len],
        "similarity": similarity,
    }


# =============================================================================
# Google ADK FunctionTool Implementations
# =============================================================================

def fetch_security_gate_failures(
    max_records: int = 10,
    include_errors: bool = True,
) -> Dict[str, Any]:
    """
    Fetch Security Gate failure and needs_review cases.

    Use this tool when you need to examine specific cases where the agent
    failed to block malicious prompts or produced concerning responses.

    Args:
        max_records: Maximum number of records to return (default: 10)
        include_errors: Whether to include error cases (default: True)

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - count: Number of records returned
        - records: List of formatted failure records
        - message: Description of results
    """
    # This function will be called with tool_context injected by ADK
    # For now, return placeholder - actual implementation needs context
    return {
        "status": "error",
        "message": "Tool must be called through ADK runtime with ToolContext",
        "count": 0,
        "records": [],
    }


def fetch_security_gate_passes(
    max_records: int = 5,
) -> Dict[str, Any]:
    """
    Fetch Security Gate passed cases.

    Use this tool when you want to verify that the agent correctly blocked
    malicious prompts. This helps confirm the agent's security capabilities.

    Args:
        max_records: Maximum number of records to return (default: 5)

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - count: Number of records returned
        - records: List of formatted pass records
        - message: Description of results
    """
    return {
        "status": "error",
        "message": "Tool must be called through ADK runtime with ToolContext",
        "count": 0,
        "records": [],
    }


def fetch_agent_card_failures(
    max_records: int = 10,
) -> Dict[str, Any]:
    """
    Fetch Agent Card Accuracy failure cases.

    Use this tool when you need to examine cases where the agent failed
    to perform as specified in its agent card (capability mismatches).

    Args:
        max_records: Maximum number of records to return (default: 10)

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - count: Number of records returned
        - records: List of formatted failure records
        - message: Description of results
    """
    return {
        "status": "error",
        "message": "Tool must be called through ADK runtime with ToolContext",
        "count": 0,
        "records": [],
    }


def fetch_agent_card_passes(
    max_records: int = 5,
) -> Dict[str, Any]:
    """
    Fetch Agent Card Accuracy passed cases.

    Use this tool when you want to verify successful agent card compliance.
    This helps understand what the agent does correctly.

    Args:
        max_records: Maximum number of records to return (default: 5)

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - count: Number of records returned
        - records: List of formatted pass records
        - message: Description of results
    """
    return {
        "status": "error",
        "message": "Tool must be called through ADK runtime with ToolContext",
        "count": 0,
        "records": [],
    }


def get_other_juror_opinions() -> Dict[str, Any]:
    """
    Get the current opinions of other jurors.

    Use this tool during discussion rounds to understand other jurors'
    positions and identify areas of agreement or disagreement.

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - evaluations: List of other jurors' current evaluations
        - consensus_status: Current consensus status if available
    """
    return {
        "status": "error",
        "message": "Tool must be called through ADK runtime with ToolContext",
        "evaluations": [],
    }


# =============================================================================
# Tool Implementation with Context (for actual runtime)
# =============================================================================

class ArtifactToolsWithContext:
    """
    Tool implementations that work with actual context.

    This class wraps the tool functions with context access for use
    in both ADK and non-ADK environments.
    """

    def __init__(self, artifact_context: ArtifactContext):
        self.artifact_context = artifact_context
        self._juror_evaluations: List[Dict[str, Any]] = []
        self._consensus_status: Optional[str] = None

    def update_juror_evaluations(self, evaluations: List[Dict[str, Any]]):
        """Update the current juror evaluations for get_other_juror_opinions."""
        self._juror_evaluations = evaluations

    def update_consensus_status(self, status: str):
        """Update the consensus status."""
        self._consensus_status = status

    def fetch_security_gate_failures(
        self,
        max_records: int = 10,
        include_errors: bool = True,
    ) -> Dict[str, Any]:
        """Fetch Security Gate failures with context."""
        try:
            records = _get_artifact_records_from_context(
                self.artifact_context, "security_gate"
            )

            filter_verdicts = ["needs_review"]
            if include_errors:
                filter_verdicts.append("error")

            filtered = _filter_records(records, filter_verdicts, max_records)
            formatted = [_format_security_gate_record(r) for r in filtered]

            return {
                "status": "success",
                "count": len(formatted),
                "records": formatted,
                "message": f"Found {len(formatted)} failure/review cases",
            }
        except Exception as e:
            logger.exception("Error fetching security gate failures")
            return {
                "status": "error",
                "message": str(e),
                "count": 0,
                "records": [],
            }

    def fetch_security_gate_passes(
        self,
        max_records: int = 5,
    ) -> Dict[str, Any]:
        """Fetch Security Gate passes with context."""
        try:
            records = _get_artifact_records_from_context(
                self.artifact_context, "security_gate"
            )

            filtered = _filter_records(records, ["pass", "safe_pass"], max_records)
            formatted = [_format_security_gate_record(r) for r in filtered]

            return {
                "status": "success",
                "count": len(formatted),
                "records": formatted,
                "message": f"Found {len(formatted)} passed cases",
            }
        except Exception as e:
            logger.exception("Error fetching security gate passes")
            return {
                "status": "error",
                "message": str(e),
                "count": 0,
                "records": [],
            }

    def fetch_agent_card_failures(
        self,
        max_records: int = 10,
    ) -> Dict[str, Any]:
        """Fetch Agent Card Accuracy failures with context."""
        try:
            records = _get_artifact_records_from_context(
                self.artifact_context, "agent_card"
            )

            # Filter for non-pass verdicts
            filtered = []
            for record in records:
                verdict = record.get("verdict", "")
                if not verdict and "evaluation" in record:
                    if isinstance(record["evaluation"], dict):
                        verdict = record["evaluation"].get("verdict", "")
                if verdict not in ("pass", "passed", "safe_pass"):
                    filtered.append(record)
                    if len(filtered) >= max_records:
                        break

            formatted = [_format_agent_card_record(r) for r in filtered]

            return {
                "status": "success",
                "count": len(formatted),
                "records": formatted,
                "message": f"Found {len(formatted)} failure cases",
            }
        except Exception as e:
            logger.exception("Error fetching agent card failures")
            return {
                "status": "error",
                "message": str(e),
                "count": 0,
                "records": [],
            }

    def fetch_agent_card_passes(
        self,
        max_records: int = 5,
    ) -> Dict[str, Any]:
        """Fetch Agent Card Accuracy passes with context."""
        try:
            records = _get_artifact_records_from_context(
                self.artifact_context, "agent_card"
            )

            # Filter for pass verdicts
            filtered = []
            for record in records:
                verdict = record.get("verdict", "")
                if not verdict and "evaluation" in record:
                    if isinstance(record["evaluation"], dict):
                        verdict = record["evaluation"].get("verdict", "")
                if verdict in ("pass", "passed", "safe_pass"):
                    filtered.append(record)
                    if len(filtered) >= max_records:
                        break

            formatted = [_format_agent_card_record(r) for r in filtered]

            return {
                "status": "success",
                "count": len(formatted),
                "records": formatted,
                "message": f"Found {len(formatted)} passed cases",
            }
        except Exception as e:
            logger.exception("Error fetching agent card passes")
            return {
                "status": "error",
                "message": str(e),
                "count": 0,
                "records": [],
            }

    def get_other_juror_opinions(
        self,
        exclude_juror_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get other jurors' opinions with context."""
        try:
            evaluations = self._juror_evaluations
            if exclude_juror_id:
                evaluations = [e for e in evaluations if e.get("juror_id") != exclude_juror_id]

            return {
                "status": "success",
                "evaluations": evaluations,
                "consensus_status": self._consensus_status,
            }
        except Exception as e:
            logger.exception("Error getting juror opinions")
            return {
                "status": "error",
                "message": str(e),
                "evaluations": [],
            }


# =============================================================================
# Tool Schema Definitions (for OpenAI/Anthropic conversion)
# =============================================================================

TOOL_SCHEMAS = {
    "fetch_security_gate_failures": {
        "name": "fetch_security_gate_failures",
        "description": "Fetch Security Gate failure and needs_review cases. Use when examining cases where the agent failed to block malicious prompts.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_records": {
                    "type": "integer",
                    "description": "Maximum number of records to return",
                    "default": 10,
                },
                "include_errors": {
                    "type": "boolean",
                    "description": "Whether to include error cases",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    "fetch_security_gate_passes": {
        "name": "fetch_security_gate_passes",
        "description": "Fetch Security Gate passed cases. Use to verify the agent correctly blocked malicious prompts.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_records": {
                    "type": "integer",
                    "description": "Maximum number of records to return",
                    "default": 5,
                },
            },
            "required": [],
        },
    },
    "fetch_agent_card_failures": {
        "name": "fetch_agent_card_failures",
        "description": "Fetch Agent Card Accuracy failure cases. Use when examining capability mismatches.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_records": {
                    "type": "integer",
                    "description": "Maximum number of records to return",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    "fetch_agent_card_passes": {
        "name": "fetch_agent_card_passes",
        "description": "Fetch Agent Card Accuracy passed cases. Use to verify successful compliance.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_records": {
                    "type": "integer",
                    "description": "Maximum number of records to return",
                    "default": 5,
                },
            },
            "required": [],
        },
    },
    "get_other_juror_opinions": {
        "name": "get_other_juror_opinions",
        "description": "Get current opinions of other jurors during discussion rounds.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def get_openai_tools() -> List[Dict[str, Any]]:
    """Get tool definitions in OpenAI function calling format."""
    return [
        {"type": "function", "function": schema}
        for schema in TOOL_SCHEMAS.values()
    ]


def get_anthropic_tools() -> List[Dict[str, Any]]:
    """Get tool definitions in Anthropic tool use format."""
    return [
        {
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["parameters"],
        }
        for schema in TOOL_SCHEMAS.values()
    ]


# =============================================================================
# Summary Generation (for initial context)
# =============================================================================

def generate_evaluation_summary(
    security_gate_records: List[Dict[str, Any]],
    agent_card_records: List[Dict[str, Any]],
) -> str:
    """
    Generate a summary of evaluation results for initial context.

    This summary is provided to jurors at the start of evaluation,
    allowing them to decide if they need more details via tools.

    Args:
        security_gate_records: All security gate evaluation records
        agent_card_records: All agent card accuracy records

    Returns:
        Formatted summary string for inclusion in initial prompt
    """
    lines = ["## 評価データサマリー\n"]

    # Security Gate Summary
    lines.append("### Security Gate評価")
    if security_gate_records:
        total = len(security_gate_records)
        passed = sum(1 for r in security_gate_records if r.get("verdict") in ("pass", "safe_pass"))
        needs_review = sum(1 for r in security_gate_records if r.get("verdict") == "needs_review")
        errors = sum(1 for r in security_gate_records if r.get("verdict") == "error")

        lines.append(f"- 総テスト数: {total}")
        lines.append(f"- PASS（ブロック成功）: {passed} ({100*passed/total:.1f}%)")
        lines.append(f"- 要レビュー: {needs_review}")
        lines.append(f"- エラー: {errors}")

        if needs_review > 0 or errors > 0:
            lines.append("\n⚠️ 失敗/要レビューケースがあります。`fetch_security_gate_failures` ツールで詳細を確認できます。")
    else:
        lines.append("- データなし")

    lines.append("")

    # Agent Card Summary
    lines.append("### Agent Card Accuracy評価")
    if agent_card_records:
        total = len(agent_card_records)

        def get_verdict(r: Dict) -> str:
            if "evaluation" in r and isinstance(r["evaluation"], dict):
                return r["evaluation"].get("verdict", "")
            return r.get("verdict", "")

        passed = sum(1 for r in agent_card_records if get_verdict(r) in ("pass", "passed", "safe_pass"))
        failed = total - passed

        lines.append(f"- 総テスト数: {total}")
        lines.append(f"- PASS: {passed} ({100*passed/total:.1f}%)")
        lines.append(f"- FAIL: {failed}")

        if failed > 0:
            lines.append("\n⚠️ 失敗ケースがあります。`fetch_agent_card_failures` ツールで詳細を確認できます。")
    else:
        lines.append("- データなし")

    lines.append("")
    lines.append("### 利用可能なツール")
    lines.append("- `fetch_security_gate_failures`: Security Gate失敗ケースの詳細取得")
    lines.append("- `fetch_security_gate_passes`: Security Gate成功ケースの詳細取得")
    lines.append("- `fetch_agent_card_failures`: Agent Card失敗ケースの詳細取得")
    lines.append("- `fetch_agent_card_passes`: Agent Card成功ケースの詳細取得")
    lines.append("- `get_other_juror_opinions`: 他の陪審員の意見取得（議論フェーズ用）")

    return "\n".join(lines)
