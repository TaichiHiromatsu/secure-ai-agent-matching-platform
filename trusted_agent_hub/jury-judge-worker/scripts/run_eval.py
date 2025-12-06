#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from csv import DictReader
from pathlib import Path
from typing import Any, Dict, List, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from jury_judge_worker import MCTSJudgeOrchestrator, dispatch_questions, generate_questions
from jury_judge_worker.llm_judge import LLMJudge, LLMJudgeConfig
from jury_judge_worker.wandb_logger import WandbConfig, init_wandb, log_artifact, log_metrics, update_config

ROOT = Path(__file__).resolve().parents[3]
PROJECT_SCENARIO = ROOT / "prototype/jury-judge-worker/scenarios/generic_eval.yaml"
OUTPUT_DIR = ROOT / "prototype/jury-judge-worker/out"
AISEV_ROOT = Path(os.environ.get("AISEV_HOME", ROOT / "third_party/aisev"))
AISEV_DATASET_DIR = AISEV_ROOT / "backend/dataset/output"
AISEV_DATASET_CACHE: Dict[str, List[Dict[str, str]]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inspect evaluation using aisev scenario")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--scenario", default=os.environ.get("INSPECT_SCENARIO", str(PROJECT_SCENARIO)))
    parser.add_argument("--artifacts", default=os.environ.get("ARTIFACTS_DIR"), help="Path to sandbox artifacts directory")
    parser.add_argument("--manifest", default=os.environ.get("MANIFEST_PATH", str(ROOT / "prompts/aisi/manifest.tier3.json")))
    parser.add_argument("--enable-judge-panel", action="store_true", help="Generate jury_judge_report.jsonl via Judge Panel PoC")
    parser.add_argument("--agent-card", help="AgentCard JSON path used for Judge Panel question generation")
    parser.add_argument("--relay-endpoint", help="A2A Relay endpoint for executing judge questions")
    parser.add_argument("--relay-token", help="Bearer token for the A2A Relay endpoint")
    parser.add_argument("--submission-id", help="Submission identifier used for Human Review deeplink")
    parser.add_argument("--judge-max-questions", type=int, default=int(os.environ.get("JUDGE_MAX_QUESTIONS", 5)))
    parser.add_argument("--judge-timeout", type=float, default=float(os.environ.get("JUDGE_TIMEOUT", 15.0)))
    parser.add_argument(
        "--judge-dry-run",
        action="store_true",
        default=os.environ.get("JUDGE_DRY_RUN", "false").lower() == "true",
        help="Do not call the real agent during Judge Panel execution",
    )
    parser.add_argument(
        "--judge-llm-enabled",
        action="store_true",
        default=os.environ.get("JUDGE_LLM_ENABLED", "false").lower() == "true",
        help="Enable LLM-as-a-Judge scoring"
    )
    parser.add_argument("--judge-llm-provider", default=os.environ.get("JUDGE_LLM_PROVIDER", "openai"))
    parser.add_argument("--judge-llm-model", default=os.environ.get("JUDGE_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--judge-llm-temperature", type=float, default=float(os.environ.get("JUDGE_LLM_TEMPERATURE", 0.1)))
    parser.add_argument("--judge-llm-max-output", type=int, default=int(os.environ.get("JUDGE_LLM_MAX_OUTPUT", 512)))
    parser.add_argument("--judge-llm-base-url", default=os.environ.get("JUDGE_LLM_BASE_URL"))
    parser.add_argument(
        "--judge-llm-dry-run",
        action="store_true",
        default=os.environ.get("JUDGE_LLM_DRY_RUN", "false").lower() == "true",
        help="Skip actual LLM API calls even if enabled"
    )
    parser.add_argument("--wandb-run-id", help="Reuse an existing W&B Run ID for Inspect outputs")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "agent-store-sandbox"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "local"))
    parser.add_argument("--wandb-base-url", default=os.environ.get("WANDB_BASE_URL", "https://wandb.ai"))
    parser.add_argument("--wandb-enabled", default=os.environ.get("WANDB_DISABLED", "false").lower() != "true", type=lambda v: str(v).lower() == 'true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = OUTPUT_DIR / args.agent_id / args.revision
    output_path.mkdir(parents=True, exist_ok=True)

    wandb_config = WandbConfig(
        enabled=bool(args.wandb_enabled),
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_id=args.wandb_run_id or f"inspect-{args.agent_id}-{args.revision}",
        base_url=args.wandb_base_url,
    )
    wandb_run = init_wandb(wandb_config) if wandb_config.enabled else None

    artifacts_dir = Path(args.artifacts)
    response_samples_file = artifacts_dir / "response_samples.jsonl"
    policy_score_file = artifacts_dir / "policy_score.json"

    print(f"[DEBUG] Checking for response_samples.jsonl at: {response_samples_file.absolute()}", file=sys.stderr)
    print(f"[DEBUG] artifacts_dir exists: {artifacts_dir.exists()}", file=sys.stderr)
    print(f"[DEBUG] artifacts_dir contents: {list(artifacts_dir.iterdir()) if artifacts_dir.exists() else 'N/A'}", file=sys.stderr)

    if not response_samples_file.exists():
        raise FileNotFoundError(f"response_samples.jsonl が見つかりません。パス: {response_samples_file.absolute()}")

    if args.enable_judge_panel and not args.agent_card:
        raise ValueError("Judge Panelを有効にする場合は --agent-card を指定してください")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest が見つかりません: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    question_map = _load_questions(manifest_path.parent, manifest.get("questionFiles", []))

    records = _load_response_records(response_samples_file, question_map)
    dataset_path = output_path / "inspect_dataset.jsonl"
    _write_inspect_dataset(dataset_path, records)

    use_placeholder = os.environ.get("INSPECT_USE_PLACEHOLDER", "false").lower() == "true"
    inspect_info: Dict[str, Any] | None = None
    if use_placeholder:
        latencies, eval_results, compliance_ratio, inspect_info = _placeholder_eval(records)
    else:
        latencies, eval_results, compliance_ratio, inspect_info = _run_inspect_eval(records, output_path)

    policy_score = 0.0
    if policy_score_file.exists():
        policy_score_data = json.loads(policy_score_file.read_text(encoding="utf-8"))
        policy_score = policy_score_data.get("score", 0.0)

    passed = sum(1 for entry in eval_results if entry.get("compliant"))
    total = len(eval_results)
    summary = {
        "agentId": args.agent_id,
        "revision": args.revision,
        "outputDir": str(output_path),
        "score": compliance_ratio,
        "policyScore": policy_score,
        "avgLatencyMs": sum(latencies) / len(latencies) if latencies else None,
        "passed": passed,
        "total": total,
        "notes": "Inspect評価 (inspect_aiが有効な場合はCLI実行、それ以外はプレースホルダー)"
    }

    if inspect_info:
        summary["inspect"] = inspect_info

    details_path = output_path / "details.json"
    details_path.write_text(json.dumps(eval_results, indent=2, ensure_ascii=False), encoding="utf-8")

    log_metrics(wandb_config, {
        "functional/score": float(compliance_ratio),
        "functional/passed": float(passed),
        "functional/total": float(total),
        "functional/avg_latency_ms": float(summary.get("avgLatencyMs") or 0.0),
    })

    llm_config = None
    if args.judge_llm_enabled:
        llm_config = LLMJudgeConfig(
            enabled=True,
            provider=args.judge_llm_provider,
            model=args.judge_llm_model,
            temperature=args.judge_llm_temperature,
            max_output_tokens=args.judge_llm_max_output,
            base_url=args.judge_llm_base_url,
            dry_run=bool(args.judge_llm_dry_run or args.judge_dry_run),
        )

    if args.enable_judge_panel and args.agent_card:
        judge_summary = _run_judge_panel(
            output_path,
            agent_card_path=Path(args.agent_card),
            relay_endpoint=args.relay_endpoint,
            relay_token=args.relay_token,
            max_questions=args.judge_max_questions,
            timeout=args.judge_timeout,
            dry_run=args.judge_dry_run,
            wandb_config=wandb_config if wandb_config.enabled else None,
            llm_config=llm_config,
        )
        summary["judgePanel"] = judge_summary

    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if wandb_run:
        wandb_run.finish()


def _load_questions(base_dir: Path, files: List[str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for rel_path in files:
        question_path = base_dir / rel_path
        if not question_path.exists():
            continue
        data = json.loads(question_path.read_text(encoding="utf-8"))
        question_id = data.get("id")
        if question_id:
            enriched = _apply_aisev_metadata(data)
            result[question_id] = enriched
    return result


def _apply_aisev_metadata(question: Dict[str, Any]) -> Dict[str, Any]:
    aisev_cfg = question.get("aisev")
    if not aisev_cfg:
        return question

    dataset_name = aisev_cfg.get("dataset")
    if not dataset_name:
        raise ValueError(f"aisev設定に dataset がありません: {question.get('id')}")

    rows = _load_aisev_dataset(dataset_name)
    candidates = rows

    gsn_code = aisev_cfg.get("gsnPerspective")
    if gsn_code:
        candidates = [row for row in candidates if row.get("gsn_perspective") == gsn_code]

    ten_perspective = aisev_cfg.get("tenPerspective")
    if ten_perspective:
        candidates = [row for row in candidates if row.get("ten_perspective") == ten_perspective]

    text_contains = aisev_cfg.get("textContains")
    if text_contains:
        candidates = [row for row in candidates if text_contains in row.get("text", "")]

    row_index = int(aisev_cfg.get("rowIndex", 0))
    if not candidates:
        raise ValueError(f"aisev dataset '{dataset_name}' に一致する行が見つかりません (questionId={question.get('id')}).")
    if row_index >= len(candidates) or row_index < 0:
        raise IndexError(f"aisev dataset '{dataset_name}' の rowIndex={row_index} が範囲外です (候補数={len(candidates)}).")

    row = candidates[row_index]

    question["prompt"] = row.get("text", question.get("prompt", ""))
    question["expectedBehaviour"] = row.get("requirement", question.get("expectedBehaviour", ""))
    question["notes"] = question.get("notes") or f"aisev dataset {dataset_name} / {row.get('gsn_perspective')}"
    question["scorer"] = row.get("scorer", "requirement")
    question["aisevMeta"] = {
        "dataset": dataset_name,
        "tenPerspective": row.get("ten_perspective"),
        "gsnPerspective": row.get("gsn_perspective")
    }
    return question


def _load_aisev_dataset(dataset_name: str) -> List[Dict[str, str]]:
    if dataset_name in AISEV_DATASET_CACHE:
        return AISEV_DATASET_CACHE[dataset_name]

    dataset_path = AISEV_DATASET_DIR / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"aisev dataset が見つかりません: {dataset_path}")

    rows: List[Dict[str, str]] = []
    with dataset_path.open(encoding="utf-8-sig") as f:
        reader = DictReader(f)
        for row in reader:
            # Skip empty rows
            if not any(value for value in row.values()):
                continue
            rows.append(row)

    AISEV_DATASET_CACHE[dataset_name] = rows
    return rows


def _load_response_records(response_samples_file: Path, question_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not response_samples_file.exists():
        return records

    with response_samples_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            question_id = sample.get("questionId")
            question = question_map.get(question_id, {})
            tolerance = question.get("tolerance") or {}
            record = {
                "questionId": question_id,
                "prompt": question.get("prompt") or sample.get("inputText") or "",
                "inputText": sample.get("inputText") or "",
                "expectedBehaviour": question.get("expectedBehaviour") or "",
                "outputText": sample.get("outputText") or "",
                "latencyMs": sample.get("latencyMs", 0),
                "tokensOut": sample.get("tokensOut"),
                "tolerance": tolerance,
                "notes": question.get("notes"),
                "scorer": question.get("scorer", "requirement"),
                "aisevMeta": question.get("aisevMeta"),
            }
            records.append(record)
    return records


def _placeholder_eval(records: List[Dict[str, Any]]) -> Tuple[List[float], List[Dict[str, Any]], float, Dict[str, Any] | None]:
    latencies: List[float] = []
    eval_results: List[Dict[str, Any]] = []
    passed = 0
    total = 0

    for record in records:
        question_id = record.get("questionId")
        output_text = record.get("outputText", "")
        tolerance = record.get("tolerance") or {}
        allowed = tolerance.get("allowedPhrases") or []
        blocked = tolerance.get("blockedPhrases") or []

        lower_output = output_text.lower()
        is_allowed = all(word.lower() not in lower_output for word in blocked)
        contains_required = True
        if allowed:
            contains_required = any(word.lower() in lower_output for word in allowed)

        compliant = bool(is_allowed and contains_required)
        if compliant:
            passed += 1
        total += 1

        latencies.append(record.get("latencyMs", 0))
        eval_results.append({
            "questionId": question_id,
            "output": output_text,
            "allowedPhrases": allowed,
            "blockedPhrases": blocked,
            "compliant": compliant,
            "notes": record.get("expectedBehaviour"),
            "scorer": record.get("scorer"),
            "aisevMeta": record.get("aisevMeta")
        })

    compliance_ratio = passed / total if total else 0.0
    return latencies, eval_results, compliance_ratio, None


def _run_inspect_eval(
    records: List[Dict[str, Any]],
    output_path: Path
) -> Tuple[List[float], List[Dict[str, Any]], float, Dict[str, Any] | None]:
    try:
        from inspect_ai.dataset import Sample
        from inspect_ai.model import ModelOutput
        from inspect_ai.scorer import model_graded_qa
        from inspect_ai.solver import Generate, TaskState, solver
        from inspect_ai._eval.eval import eval as inspect_eval
        from inspect_ai._eval.task import Task
    except ImportError as error:
        print(f"[jury-judge-worker] inspect_ai import failed: {error}. Falling back to placeholder evaluation.")
        return _placeholder_eval(records)

    if not records:
        return [], [], 0.0, None

    log_dir = output_path / "inspect_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    judge_model = os.environ.get("INSPECT_GRADER_MODEL", "mockllm/model")

    @solver
    def replay_solver():
        async def solve(state: TaskState, generate: Generate):
            output_text = state.metadata.get("outputText", "")
            state.output = ModelOutput.from_content(
                model=os.environ.get("INSPECT_REPLAY_MODEL", "replay"),
                content=output_text
            )
            state.completed = True
            return state
        return solve

    samples: List[Sample] = []
    for record in records:
        metadata = {
            "questionId": record.get("questionId"),
            "expectedBehaviour": record.get("expectedBehaviour"),
            "tolerance": record.get("tolerance"),
            "latencyMs": record.get("latencyMs"),
            "tokensOut": record.get("tokensOut"),
            "notes": record.get("notes"),
            "outputText": record.get("outputText"),
            "scorer": record.get("scorer"),
            "aisevMeta": record.get("aisevMeta"),
        }
        sample = Sample(
            input=record.get("prompt") or record.get("inputText") or "",
            target=record.get("expectedBehaviour") or "",
            id=record.get("questionId"),
            metadata=metadata
        )
        samples.append(sample)

    task = Task(
        dataset=samples,
        solver=replay_solver(),
        scorer=model_graded_qa(model=judge_model)
    )

    try:
        logs = inspect_eval(
            tasks=[task],
            log_dir=str(log_dir),
            log_format="json",
            model=None,
            display="none",
            score_display=False,
        )
    except Exception as error:
        print(f"[jury-judge-worker] inspect eval execution failed: {error}. Falling back to placeholder evaluation.")
        return _placeholder_eval(records)

    if not logs:
        print("[jury-judge-worker] inspect eval returned no logs. Falling back to placeholder evaluation.")
        return _placeholder_eval(records)

    log = logs[0]
    scorer_name = log.results.scores[0].name if log.results.scores else "model_graded_qa"

    latencies: List[float] = []
    eval_results: List[Dict[str, Any]] = []
    passed = 0

    for sample in log.samples:
        metadata = sample.metadata or {}
        question_id = metadata.get("questionId") or sample.id
        score = (sample.scores or {}).get(scorer_name)
        grade = getattr(score, "value", None)
        compliant = grade == "C"
        if compliant:
            passed += 1
        latencies.append(metadata.get("latencyMs", 0))

        eval_results.append({
            "questionId": question_id,
            "output": sample.output.completion if sample.output else "",
            "expectedBehaviour": metadata.get("expectedBehaviour"),
            "allowedPhrases": (metadata.get("tolerance") or {}).get("allowedPhrases"),
            "blockedPhrases": (metadata.get("tolerance") or {}).get("blockedPhrases"),
            "grade": grade,
            "judgeModel": judge_model,
            "explanation": getattr(score, "explanation", None),
            "compliant": compliant,
            "latencyMs": metadata.get("latencyMs"),
            "notes": metadata.get("notes"),
            "scorer": metadata.get("scorer"),
            "aisevMeta": metadata.get("aisevMeta"),
        })

    compliance_ratio = passed / len(log.samples) if log.samples else 0.0

    aggregated_metrics = {}
    for eval_score in log.results.scores or []:
        aggregated_metrics[eval_score.name] = {
            metric_name: metric.value for metric_name, metric in (eval_score.metrics or {}).items()
        }

    inspect_info: Dict[str, Any] | None = None
    if log.location:
        inspect_info = {
            "logFile": str(log.location),
            "judgeModel": judge_model,
            "metrics": aggregated_metrics
        }
        log_info = {
            "logFile": str(log.location),
            "judgeModel": judge_model,
            "metrics": aggregated_metrics
        }
        (output_path / "inspect_log_index.json").write_text(
            json.dumps(log_info, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    return latencies, eval_results, compliance_ratio, inspect_info


def _run_judge_panel(
    output_path: Path,
    *,
    agent_card_path: Path,
    relay_endpoint: str | None,
    relay_token: str | None,
    max_questions: int,
    timeout: float,
    dry_run: bool,
    wandb_config: WandbConfig | None,
    llm_config: LLMJudgeConfig | None,
) -> Dict[str, Any]:
    judge_dir = output_path / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)

    questions = generate_questions(agent_card_path, max_questions=max_questions)
    executions = dispatch_questions(
        questions,
        relay_endpoint=relay_endpoint,
        relay_token=relay_token,
        timeout=timeout,
        dry_run=dry_run,
    )

    llm_summary = {
        "enabled": bool(llm_config and llm_config.enabled),
        "model": llm_config.model if llm_config and llm_config.enabled else None,
        "dryRun": bool(llm_config.dry_run) if llm_config else False,
        "error": None,
        "calls": 0,
        "provider": llm_config.provider if llm_config else None,
        "temperature": llm_config.temperature if llm_config else None,
        "maxOutputTokens": llm_config.max_output_tokens if llm_config else None,
        "baseUrl": llm_config.base_url if llm_config else None
    }

    # Multi-Model Judge Panel (GPT-4o, Claude 3.5, Gemini 1.5 Pro)の初期化
    panel_judge_instance = None
    use_panel = False
    try:
        from jury_judge_worker.panel_judge import MultiModelJudgePanel
        # 環境変数から各モデルの有効化を確認
        enable_openai = bool(os.getenv("OPENAI_API_KEY"))
        enable_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        enable_google = bool(os.getenv("GOOGLE_API_KEY"))

        if enable_openai or enable_anthropic or enable_google:
            panel_judge_instance = MultiModelJudgePanel(
                dry_run=llm_config.dry_run if llm_config else False,
                enable_openai=enable_openai,
                enable_anthropic=enable_anthropic,
                enable_google=enable_google
            )
            use_panel = True
            llm_summary["panelEnabled"] = True
            llm_summary["panelModels"] = panel_judge_instance.models
    except Exception as error:
        llm_summary["panelError"] = str(error)
        llm_summary["panelEnabled"] = False

    # Single LLM Judge (フォールバック)
    llm_judge_instance = None
    if llm_config and llm_config.enabled and not use_panel:
        try:
            llm_judge_instance = LLMJudge(llm_config)
        except Exception as error:  # pragma: no cover - env specific
            llm_summary["error"] = str(error)
            llm_summary["enabled"] = False

    orchestrator = MCTSJudgeOrchestrator(
        llm_judge=llm_judge_instance,
        panel_judge=panel_judge_instance,
        use_panel=use_panel
    )
    verdicts = orchestrator.run_panel(questions, executions)
    llm_summary["calls"] = orchestrator.llm_calls if llm_judge_instance else 0

    report_path = judge_dir / "jury_judge_report.jsonl"
    with report_path.open("w", encoding="utf-8") as f:
        for verdict in verdicts:
            execution = next((item for item in executions if item.question_id == verdict.question_id), None)
            record = {
                "questionId": verdict.question_id,
                "prompt": execution.prompt if execution else "",
                "response": execution.response if execution else "",
                "latencyMs": execution.latency_ms if execution else None,
                "responseSnippet": execution.response_snippet if execution else None,
                "score": verdict.score,
                "verdict": verdict.verdict,
                "rationale": verdict.rationale,
                "judgeNotes": verdict.judge_notes,
                "relayEndpoint": execution.relay_endpoint if execution else None,
                "responseStatus": execution.status if execution else None,
                "responseError": execution.error if execution else None,
                "httpStatus": execution.http_status if execution else None,
                "relayAttempts": execution.attempts if execution else None,
                "relayRetryErrors": execution.retry_errors if execution else None,
                "flags": verdict.flags,
                "llmScore": verdict.llm_score,
                "llmVerdict": verdict.llm_verdict,
                "llmRationale": verdict.llm_rationale,
                "llmProvider": llm_config.provider if llm_config else None,
                "llmModel": llm_config.model if llm_config else None,
                "llmTemperature": llm_config.temperature if llm_config else None,
                "llmMaxOutputTokens": llm_config.max_output_tokens if llm_config else None,
                "llmDryRun": llm_config.dry_run if llm_config else None,
                "llmBaseUrl": llm_config.base_url if llm_config else None,
                "traceId": execution.trace_id if execution else None,
                # Multi-Model Judge Panel results
                "panelVerdicts": verdict.panel_verdicts if hasattr(verdict, 'panel_verdicts') else None,
                "panelAggregatedVerdict": verdict.panel_aggregated_verdict if hasattr(verdict, 'panel_aggregated_verdict') else None,
                "panelMinorityVeto": verdict.panel_minority_veto if hasattr(verdict, 'panel_minority_veto') else None,
                # AISI Inspect criteria scores
                "aisiTaskCompletion": verdict.aisi_task_completion if hasattr(verdict, 'aisi_task_completion') else None,
                "aisiToolUsage": verdict.aisi_tool_usage if hasattr(verdict, 'aisi_tool_usage') else None,
                "aisiAutonomy": verdict.aisi_autonomy if hasattr(verdict, 'aisi_autonomy') else None,
                "aisiSafety": verdict.aisi_safety if hasattr(verdict, 'aisi_safety') else None,
                "aisiTotalScore": verdict.aisi_total_score if hasattr(verdict, 'aisi_total_score') else None,
                # Stage-based Multi-Model Panel results
                "stagePanelVerdicts": verdict.stage_panel_verdicts if hasattr(verdict, 'stage_panel_verdicts') else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    relay_log_path = judge_dir / "relay_logs.jsonl"
    with relay_log_path.open("w", encoding="utf-8") as relay_log:
        for execution in executions:
            relay_log.write(json.dumps({
                "questionId": execution.question_id,
                "prompt": execution.prompt,
                "response": execution.response,
                "responseSnippet": execution.response_snippet,
                "status": execution.status,
                "error": execution.error,
                "latencyMs": execution.latency_ms,
                "httpStatus": execution.http_status,
                "relayEndpoint": execution.relay_endpoint,
                "flags": execution.flags,
                "attempts": execution.attempts,
                "retryErrors": execution.retry_errors,
                "traceId": execution.trace_id,
            }, ensure_ascii=False) + "\n")

    summary = {
        "questions": len(verdicts),
        "approved": sum(1 for v in verdicts if v.verdict == "approve"),
        "manual": sum(1 for v in verdicts if v.verdict == "manual"),
        "rejected": sum(1 for v in verdicts if v.verdict == "reject"),
        "notes": "Judge Panel PoC",
        "flagged": sum(1 for exec in executions if exec.flags),
        "relayErrors": sum(1 for exec in executions if exec.status == "error"),
        "relayRetries": sum(max(exec.attempts - 1, 0) for exec in executions),
        "llmJudge": llm_summary,
    }
    summary_path = judge_dir / "judge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    review_ui_base = os.environ.get("HUMAN_REVIEW_BASE_URL") or os.environ.get("REVIEW_UI_BASE_URL")
    review_ui_url = None
    if review_ui_base and args.submission_id:
        review_ui_url = f"{review_ui_base.rstrip('/')}/review/ui/{args.submission_id}"

    if wandb_config:
        questions_total = float(summary["questions"] or 0)
        reject_count = float(summary.get("rejected") or 0)
        manual_count = float(summary.get("manual") or 0)
        relay_error_count = float(summary.get("relayErrors") or 0)
        relay_retry_count = float(summary.get("relayRetries") or 0)

        update_config(wandb_config, {
            "judge_llm_enabled": 1 if llm_summary.get("enabled") else 0,
            "judge_llm_provider": llm_summary.get("provider"),
            "judge_llm_model": llm_summary.get("model"),
            "judge_llm_temperature": llm_summary.get("temperature"),
            "judge_llm_max_tokens": llm_summary.get("maxOutputTokens"),
            "judge_llm_base_url": llm_summary.get("baseUrl"),
            "judge_llm_dry_run": bool(llm_summary.get("dryRun")),
            "judge_relay_endpoint": relay_endpoint,
            "judge_manual_ratio": (manual_count / questions_total) if questions_total else 0,
            "judge_reject_ratio": (reject_count / questions_total) if questions_total else 0,
            "judge_relay_error_ratio": (relay_error_count / questions_total) if questions_total else 0,
            "judge_relay_retry_ratio": (relay_retry_count / questions_total) if questions_total else 0,
            "judge_review_ui_url": review_ui_url,
            "jury_judge_report_path": str(report_path),
            "judge_relay_path": str(relay_log_path)
        })
        log_artifact(wandb_config, report_path, name=f"judge-report-{wandb_config.run_id}")
        log_artifact(wandb_config, summary_path, name=f"judge-summary-{wandb_config.run_id}")
        log_artifact(wandb_config, relay_log_path, name=f"judge-relay-{wandb_config.run_id}")
        log_metrics(wandb_config, {
            "judge/questions": float(summary["questions"]),
            "judge/approved": float(summary["approved"]),
            "judge/manual": float(summary["manual"]),
            "judge/rejected": float(summary["rejected"]),
            "judge/flagged": float(summary.get("flagged") or 0),
            "judge/llm_calls": float(llm_summary.get("calls") or 0),
            "judge/relay_errors": relay_error_count,
            "judge/relay_retries": relay_retry_count,
            "judge/llm_enabled": float(1 if llm_summary.get("enabled") else 0),
            "judge/llm_temperature": float(llm_summary.get("temperature") or 0.0),
            "judge/llm_max_tokens": float(llm_summary.get("maxOutputTokens") or 0.0),
            "judge/manual_ratio": (manual_count / questions_total) if questions_total else 0.0,
            "judge/reject_ratio": (reject_count / questions_total) if questions_total else 0.0,
            "judge/relay_error_ratio": (relay_error_count / questions_total) if questions_total else 0.0,
            "judge/relay_retry_ratio": (relay_retry_count / questions_total) if questions_total else 0.0,
        })
    return summary


def _write_inspect_dataset(dataset_path: Path, records: List[Dict[str, Any]]) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as out_file:
        for record in records:
            serialized = {
                "id": record.get("questionId"),
                "input": record.get("prompt"),
                "expected": record.get("expectedBehaviour"),
                "output": record.get("outputText"),
                "latencyMs": record.get("latencyMs"),
                "tokensOut": record.get("tokensOut"),
                "tolerance": record.get("tolerance"),
                "notes": record.get("notes"),
                "scorer": record.get("scorer"),
                "aisevMeta": record.get("aisevMeta"),
            }
            out_file.write(json.dumps(serialized, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
