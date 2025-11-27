#!/usr/bin/env bash
set -euo pipefail

AGENT_ID=${AGENT_ID:-demo-agent}
REVISION=${REVISION:-rev1}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-/tmp/artifacts}
MANIFEST_PATH=${MANIFEST_PATH:-/app/prompts/aisi/manifest.tier3.json}

python prototype/inspect-worker/scripts/run_eval.py \
  --agent-id "$AGENT_ID" \
  --revision "$REVISION" \
  --artifacts "$ARTIFACTS_DIR" \
  --manifest "$MANIFEST_PATH"
