#!/usr/bin/env python3
"""Offline verifier for a completed simulation workflow evidence graph."""

from __future__ import annotations

import argparse
import json

from secure_mediation_agent.ap2.evidence_verifier import verify_evidence_graph
from secure_mediation_agent.ap2.keys import DemoKeySet
from secure_mediation_agent.workflow.repository import WorkflowRepository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow_id")
    parser.add_argument("--marketplace", default="/app/payment-data/marketplace.db")
    parser.add_argument("--merchant", default="/app/payment-data/paid-agent.db")
    parser.add_argument("--evidence", default="/app/payment-evidence/evidence.db")
    parser.add_argument("--key-dir", default="/run/secrets/ap2-demo")
    args = parser.parse_args()
    repository = WorkflowRepository.open(args.marketplace, args.merchant, args.evidence)
    report = verify_evidence_graph(repository, DemoKeySet.load(args.key_dir), args.workflow_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
