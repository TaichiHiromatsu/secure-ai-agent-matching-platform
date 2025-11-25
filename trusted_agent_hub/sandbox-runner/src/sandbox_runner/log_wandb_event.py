from __future__ import annotations

import argparse
import json
import os
import sys


def _import_wandb():
    try:
        import wandb  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        print('[wandb_event] wandb package not available, skipping logging', file=sys.stderr)
        return None
    return wandb


def main() -> int:
    parser = argparse.ArgumentParser(description='Log a single event to an existing W&B run')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--entity', required=True)
    parser.add_argument('--event', required=True, help='JSON payload to log via wandb.log')
    parser.add_argument('--base-url', help='Optional W&B base URL (defaults to https://wandb.ai)')
    parser.add_argument('--summary', action='store_true', help='Also update wandb.summary with the payload')
    args = parser.parse_args()

    wandb = _import_wandb()
    if wandb is None:
        return 0

    if args.base_url:
        os.environ.setdefault('WANDB_BASE_URL', args.base_url.rstrip('/'))

    try:
        payload = json.loads(args.event)
    except json.JSONDecodeError:
        print('[wandb_event] invalid event JSON payload', file=sys.stderr)
        return 1

    run = wandb.init(  # type: ignore[union-attr]
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        resume='allow',
        name=args.run_id,
        settings=wandb.Settings(start_method='thread')  # type: ignore[attr-defined]
    )
    try:
        wandb.log(payload)  # type: ignore[attr-defined]
        if args.summary:
            wandb.summary.update(payload)  # type: ignore[attr-defined]
    finally:
        if run is not None:
            run.finish()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
