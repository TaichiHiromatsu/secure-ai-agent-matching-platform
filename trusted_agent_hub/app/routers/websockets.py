"""
WebSocket router for real-time jury judge evaluation updates
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Dict, Optional
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """WebSocket接続管理クラス"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, submission_id: str, websocket: WebSocket):
        """WebSocket接続を確立"""
        await websocket.accept()
        self.active_connections[submission_id] = websocket
        logger.info(f"WebSocket connected for submission {submission_id}")

    def disconnect(self, submission_id: str):
        """WebSocket接続を切断"""
        if submission_id in self.active_connections:
            del self.active_connections[submission_id]
            logger.info(f"WebSocket disconnected for submission {submission_id}")

    async def send_progress(self, submission_id: str, data: dict):
        """進捗情報を送信"""
        if submission_id in self.active_connections:
            try:
                await self.active_connections[submission_id].send_json(data)
            except Exception as e:
                logger.error(f"Failed to send WebSocket message to {submission_id}: {e}")
                self.disconnect(submission_id)

    async def broadcast(self, data: dict):
        """全接続にブロードキャスト"""
        disconnected = []
        for submission_id, connection in self.active_connections.items():
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Failed to broadcast to {submission_id}: {e}")
                disconnected.append(submission_id)

        # 切断された接続をクリーンアップ
        for sub_id in disconnected:
            self.disconnect(sub_id)


# グローバルConnectionManagerインスタンス
manager = ConnectionManager()


@router.websocket("/ws/submissions/{submission_id}/judge")
async def websocket_judge_progress(
    websocket: WebSocket,
    submission_id: str,
    token: Optional[str] = Query(None)
):
    """
    Jury Judge評価のリアルタイム進捗WebSocket

    Args:
        websocket: WebSocket connection
        submission_id: 提出ID
        token: 認証トークン（オプション）
    """
    # TODO: トークン検証を追加（将来の実装）
    # if token and not verify_token(token, submission_id):
    #     await websocket.close(code=1008)  # Policy Violation
    #     return

    await manager.connect(submission_id, websocket)

    try:
        # 接続維持ループ
        while True:
            # クライアントからのメッセージを受信（ハートビート）
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                # ハートビート応答
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # タイムアウト時は接続維持のため何もしない
                pass

    except WebSocketDisconnect:
        manager.disconnect(submission_id)
        logger.info(f"Client disconnected: {submission_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {submission_id}: {e}")
        manager.disconnect(submission_id)


def get_connection_manager() -> ConnectionManager:
    """ConnectionManagerインスタンスを取得（他のモジュールから使用）"""
    return manager
