#!/usr/bin/env python3
"""Provision persistent demo P-256 role keys with strict file permissions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from secure_mediation_agent.ap2.keys import ROLE_KIDS, generate_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Dedicated persistent key directory")
    args = parser.parse_args()
    target = Path(args.directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target, 0o700)
    for role, kid in ROLE_KIDS.items():
        path = target / f"{role}.jwk"
        if path.exists():
            continue
        temporary = target / f".{role}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(generate_key(kid).export(private_key=True))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        os.chmod(path, 0o600)
    print(f"Provisioned {len(ROLE_KIDS)} persistent demo role keys in {target}")


if __name__ == "__main__":
    main()
