"""Deterministic, test-only process-death failpoints for recovery proof."""

from __future__ import annotations

import os
from pathlib import Path


FAILPOINT_EXIT_CODE = 86


def crash_after(checkpoint: str) -> None:
    """Exit after one named durable/external checkpoint, at most once.

    Failpoints are impossible outside ``APP_ENV=test`` and require an explicit
    marker path. The exclusive marker makes a restart with the same operation
    deterministic instead of entering a crash loop.
    """

    if os.environ.get("APP_ENV") != "test":
        return
    if os.environ.get("PAYMENT_TEST_FAILPOINT") != checkpoint:
        return
    marker_value = os.environ.get("PAYMENT_TEST_FAILPOINT_MARKER")
    if not marker_value:
        return
    marker = Path(marker_value)
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(checkpoint + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os._exit(FAILPOINT_EXIT_CODE)
