"""
SSE router for real-time jury judge updates.
"""

import asyncio
import json
import logging
from typing import Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter()


class SSEManager:
    """Simple per-submission SSE fan-out using asyncio.Queue."""

    def __init__(self) -> None:
        # submission_id -> list of queues
        self._connections: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, submission_id: str) -> asyncio.Queue:
        """Register a new subscriber and return its queue."""
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._connections.setdefault(submission_id, []).append(queue)
        logger.info(f"SSE connected for submission {submission_id} (total={len(self._connections.get(submission_id, []))})")
        return queue

    async def disconnect(self, submission_id: str, queue: asyncio.Queue) -> None:
        """Remove subscriber queue."""
        async with self._lock:
            if submission_id in self._connections:
                try:
                    self._connections[submission_id].remove(queue)
                    if not self._connections[submission_id]:
                        del self._connections[submission_id]
                except ValueError:
                    pass
        logger.info(f"SSE disconnected for submission {submission_id}")

    async def send(self, submission_id: str, data: dict) -> None:
        """Send data to all subscribers of a submission."""
        async with self._lock:
            queues = list(self._connections.get(submission_id, []))
        if not queues:
            return
        payload = json.dumps(data, ensure_ascii=False)
        for q in queues:
            # Non-blocking put: drop if queue is full (should not block streaming loop)
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning("SSE queue full; dropping message")


manager = SSEManager()


def get_sse_manager() -> SSEManager:
    return manager


@router.get("/sse/submissions/{submission_id}/judge")
async def sse_judge_stream(request: Request, submission_id: str):
    """
    SSE endpoint for Jury Judge real-time updates.
    """
    queue = await manager.connect(submission_id)

    async def event_generator():
        try:
            while True:
                # Abort if client disconnects
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=25.0)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat
                    yield "event: ping\ndata: {}\n\n"
        finally:
            await manager.disconnect(submission_id, queue)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
        "Connection": "keep-alive",
    }

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
