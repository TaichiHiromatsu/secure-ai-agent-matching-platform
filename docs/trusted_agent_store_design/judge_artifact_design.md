# Judge Panel Input Slimming with Weave Artifacts

## Goal
- Avoid Judge Panel token overflow and Gemini safety blocks by passing only summaries plus Artifact URIs for Security Gate (SG) and Agent Card Accuracy (ACA) details.

## Scope
- Trusted Agent Store pipeline (PreCheck → SG → ACA → Jury Judge → Human Review).
- Applies to data handed from `submissions.py` into Jury Judge (MAGI) including CASPER.

## High-level Design
1) **Persist full SG/ACA detail to Weave Artifacts**
   - Store per-submission JSONL with all prompts/responses/reasons.
   - Capture metadata: dataset_source, priority, verdict, timing, endpoint.
2) **Compress in-memory payload for Judge**
   - Provide: small summary (counts/ratios), category breakdown, representative samples (bounded), and `artifact_uri`.
   - No full prompts/responses inline.
3) **Judge prompt contract**
   - Include summary + `artifact_uri` and instruct “fetch artifact if more evidence needed; otherwise reason from summary”.
   - CASPER uses sanitized snippets only; Artifact holds full text to consult.

## Data Flow
```
SG/ACA runner → full_results.jsonl (artifact) → compress_*() → judge_input (summary + artifact_uri) → LLM
```

## Structures
### Security Gate compressed payload
```json
{
  "summary": {"total": N, "blocked": B, "needs_review": R, "errors": E, "blocked_rate": "83%"},
  "by_dataset": {"aisi_security": {...}, ...},
  "samples": {
    "blocked": [ {"prompt_snippet": "...", "reason": "...", "dataset": "aisi_security"}, ... ],
    "needs_review": [ {"prompt": "...", "response": "...", "dataset": "..."}, ... ],
    "errors": [ {"prompt": "...", "error": "...", "dataset": "..."} ]
  },
  "artifacts": {"full_report": "weave://runs/<id>/artifacts/sg_full.jsonl"}
}
```

### ACA compressed payload
```json
{
  "summary": {"total_scenarios": M, "passed": P, "failed": F, "avg_similarity": 0.85},
  "skills": {"SkillName": {"passed": true, "similarity": 0.92, "reason": ""}},
  "failures": [ {"skill": "...", "prompt": "...", "expected": "...", "actual": "...", "similarity": 0.45} ],
  "artifacts": {"full_report": "weave://runs/<id>/artifacts/aca_full.jsonl"}
}
```

## API / Code Touch Points
- `trusted_agent_store/app/routers/submissions.py`
  - Add `compress_security_results` / `compress_functional_results` helpers.
  - After SG/ACA finish: write full JSONL to Weave, capture `artifact_uri`, call compressors, pass compressed payloads to `run_judge_panel`.
- `trusted_agent_store/jury-judge-worker/jury_judge_worker/jury_judge_collaborative.py`
  - Update `_build_evaluation_context` to use summaries; add note in prompt template: "Refer to artifact if more evidence is required".
- `trusted_agent_store/jury-judge-worker/jury_judge_worker/llm_judge.py`
  - CASPER already sanitized; ensure prompts mention artifact availability instead of inlining long logs.

## Prompt Guidance for Judges
- Add to role prompts: "You received only a summary. If evidence is insufficient, fetch the artifact URI and cite findings concisely." (Japanese wording to match existing prompts.)

## Non-Goals
- No change to UI/DB schemas; full detail still stored in artifacts and existing storage.
- No auto-RAG retrieval yet (can be added later by fetching artifact → short summary → feed back to LLM).

## Risks / Mitigations
- **Artifact unavailable**: keep minimal inline samples; fall back gracefully.
- **Evidence loss**: needs_review/errors remain full-text inline; blocked samples capped but artifact keeps all.
- **Latency**: artifact write/read adds I/O; mitigate with async + caching.

## Rollout Plan
1) Implement compressors + artifact write in `submissions.py` (feature flag optional).
2) Update judge prompts/context to summary+URI; verify token count <5k typical.
3) E2E test with SG=30, ACA=5, normal/abusive agents; confirm CASPER no safety block.
4) Monitor Weave for artifact size/time; adjust sampling caps if needed.

## Acceptance Criteria
- Judge input total tokens < 6k for SG=30, ACA=5.
- CASPER Phase1 safety blocks reduced vs baseline.
- Needs_review/errors cases remain fully inspectable via artifact.
