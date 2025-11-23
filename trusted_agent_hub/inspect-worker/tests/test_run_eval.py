from __future__ import annotations

import importlib.util
import json
from pathlib import Path

RUN_EVAL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
_spec = importlib.util.spec_from_file_location("inspect_worker_run_eval", RUN_EVAL_PATH)
assert _spec and _spec.loader
_run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_eval)  # type: ignore[arg-type]


def test_load_response_records_merges_question_fields(tmp_path: Path) -> None:
    samples_path = tmp_path / "response_samples.jsonl"
    samples_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "questionId": "privacy.data_retention",
                        "inputText": "input",
                        "outputText": "回答",
                        "latencyMs": 1200,
                        "tokensOut": 140,
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        encoding="utf-8",
    )

    question_map = {
        "privacy.data_retention": {
            "prompt": "prompt from manifest",
            "expectedBehaviour": "expected behaviour",
            "tolerance": {"allowedPhrases": ["24時間"]},
            "notes": "note",
            "scorer": "requirement",
            "aisevMeta": {"dataset": "demo"},
        }
    }

    records = _run_eval._load_response_records(samples_path, question_map)
    assert len(records) == 1
    record = records[0]
    assert record["prompt"] == "prompt from manifest"
    assert record["expectedBehaviour"] == "expected behaviour"
    assert record["tolerance"]["allowedPhrases"] == ["24時間"]
    assert record["aisevMeta"]["dataset"] == "demo"


def test_placeholder_eval_respects_tolerance_rules() -> None:
    records = [
        {
            "questionId": "safety.general",
            "outputText": "This output leaks secret",
            "tolerance": {"blockedPhrases": ["secret"]},
            "expectedBehaviour": "avoid secret",
            "latencyMs": 250,
        },
        {
            "questionId": "privacy.data_retention",
            "outputText": "We delete in 24h",
            "tolerance": {"allowedPhrases": ["24h"]},
            "expectedBehaviour": "mention retention",
            "latencyMs": 500,
        },
    ]

    latencies, eval_results, compliance_ratio, info = _run_eval._placeholder_eval(records)
    assert latencies == [250, 500]
    assert compliance_ratio == 0.5
    assert not eval_results[0]["compliant"]
    assert eval_results[1]["compliant"]
    assert info is None


def test_write_inspect_dataset_persists_expected_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    records = [
        {
            "questionId": "privacy.data_retention",
            "prompt": "prompt",
            "expectedBehaviour": "expected",
            "outputText": "output",
            "latencyMs": 42,
            "tokensOut": 17,
            "tolerance": {"allowedPhrases": ["ok"]},
            "notes": "note",
            "scorer": "requirement",
            "aisevMeta": {"dataset": "demo"},
        }
    ]

    _run_eval._write_inspect_dataset(dataset_path, records)
    lines = dataset_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["id"] == "privacy.data_retention"
    assert data["tolerance"]["allowedPhrases"] == ["ok"]
    assert data["aisevMeta"]["dataset"] == "demo"
