#!/usr/bin/env python3
"""Plan, apply, verify, or explicitly restore the three AP2/x402 v2 databases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from secure_mediation_agent.workflow.migrations import (
    DatabasePaths,
    backup_once,
    migrate,
    verify,
)


EXPECTED_NAMES = ("marketplace.db", "paid-agent.db", "evidence.db")


def _raw_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    values = tuple(
        Path(value).expanduser().resolve()
        for value in (args.marketplace, args.merchant, args.evidence)
    )
    if tuple(path.name for path in values) != EXPECTED_NAMES:
        raise SystemExit(f"Refusing paths: filenames must be exactly {EXPECTED_NAMES}")
    return values


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"


def _inventory(values: tuple[Path, Path, Path]) -> dict[str, object]:
    return {
        "profile": "x402-wire-simulation/1",
        "target": "explicit-durable-single-host-single-container",
        "databases": [
            {"path": str(path), "exists": path.is_file(), "sha256": _digest(path)}
            for path in values
        ],
    }


def _restore(manifest_path: Path, values: tuple[Path, Path, Path], confirm: str) -> None:
    if confirm != "RESTORE-PRE-CUTOVER":
        raise SystemExit("Restore requires --confirm RESTORE-PRE-CUTOVER")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_source = {item["source"]: item for item in manifest.get("backups", [])}
    for target in values:
        item = by_source.get(str(target))
        if not item or not item.get("backup"):
            raise SystemExit(f"Manifest has no recoverable backup for {target}")
        backup = Path(item["backup"]).resolve(strict=True)
        if _digest(backup) != item["sha256"]:
            raise SystemExit(f"Backup checksum mismatch for {backup}")
    for target in values:
        backup = Path(by_source[str(target)]["backup"]).resolve(strict=True)
        temporary = target.with_suffix(target.suffix + ".restore-tmp")
        shutil.copy2(backup, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(target)
    print(json.dumps({"status": "restored-pre-cutover", **_inventory(values)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "apply", "verify", "restore-pre-cutover"))
    parser.add_argument("--marketplace", default="/app/payment-data/marketplace.db")
    parser.add_argument("--merchant", default="/app/payment-data/paid-agent.db")
    parser.add_argument("--evidence", default="/app/payment-evidence/evidence.db")
    parser.add_argument("--backup-dir", default="/app/payment-data/migration-backups")
    parser.add_argument("--manifest")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    values = _raw_paths(args)
    if args.command == "plan":
        print(json.dumps({"status": "planned", **_inventory(values)}, indent=2))
        return
    paths = DatabasePaths.resolve(*values)
    if args.command == "apply":
        manifest = backup_once(paths, args.backup_dir)
        print(
            json.dumps(
                {"status": "applied", "backupManifest": str(manifest), "schemas": migrate(paths)},
                indent=2,
            )
        )
    elif args.command == "verify":
        print(json.dumps({"status": "verified", "schemas": verify(paths)}, indent=2))
    else:
        if not args.manifest:
            raise SystemExit("Restore requires --manifest")
        _restore(Path(args.manifest).expanduser().resolve(strict=True), values, args.confirm or "")


if __name__ == "__main__":
    main()
