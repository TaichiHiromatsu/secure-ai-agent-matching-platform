"""Persistent A2A 0.3.19 TaskStore backed by the Merchant authority DB."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from a2a.server.context import ServerCallContext
from a2a.server.tasks import TaskStore
from a2a.types import Task


class SQLiteTaskStore(TaskStore):
    def __init__(self, database: str | Path) -> None:
        self.database = str(database)

    async def save(self, task: Task, context: ServerCallContext | None = None) -> None:
        wire = json.dumps(
            task.model_dump(mode="json", by_alias=True, exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with sqlite3.connect(self.database) as conn:
            row = conn.execute(
                "SELECT workflow_id,order_id,version FROM merchant_tasks_v2 WHERE task_id=?",
                (task.id,),
            ).fetchone()
            if row is None:
                raise ValueError("reserved Merchant Task must exist before SDK save")
            conn.execute(
                "UPDATE merchant_tasks_v2 SET state=?,task_json=?,version=version+1,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE task_id=? AND version=?",
                (task.status.state.value, wire, task.id, row[2]),
            )

    async def get(self, task_id: str, context: ServerCallContext | None = None) -> Task | None:
        with sqlite3.connect(self.database) as conn:
            row = conn.execute(
                "SELECT task_json FROM merchant_tasks_v2 WHERE task_id=?", (task_id,)
            ).fetchone()
        return None if row is None else Task.model_validate(json.loads(row[0]))

    async def delete(self, task_id: str, context: ServerCallContext | None = None) -> None:
        raise PermissionError("Merchant Tasks are append-only for this release")
