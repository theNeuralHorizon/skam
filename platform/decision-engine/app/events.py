"""WebSocket event manager for real-time event broadcasting."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import structlog
from fastapi import WebSocket

from app.metrics import websocket_connections

logger = structlog.get_logger(__name__)


class EventManager:
    """Manages WebSocket connections and broadcasts events to all connected clients."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection and register it."""
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
        websocket_connections.set(len(self._connections))
        logger.info("websocket_connected", total_connections=len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from the pool."""
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)
        websocket_connections.set(len(self._connections))
        logger.info("websocket_disconnected", total_connections=len(self._connections))

    async def broadcast(self, event: dict[str, Any]) -> None:
        """Broadcast an event to all connected WebSocket clients.

        Stale connections are automatically removed on send failure.
        """
        if not self._connections:
            return

        event.setdefault("timestamp", datetime.utcnow().isoformat())

        payload = json.dumps(event, default=str)
        stale: list[WebSocket] = []

        async with self._lock:
            targets = list(self._connections)

        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
                logger.warning("websocket_send_failed", stale=True)

        if stale:
            async with self._lock:
                for ws in stale:
                    if ws in self._connections:
                        self._connections.remove(ws)
            websocket_connections.set(len(self._connections))

    async def emit_anomaly_detected(self, alert_data: dict[str, Any]) -> None:
        await self.broadcast({"type": "anomaly_detected", "data": alert_data})

    async def emit_recovery_started(self, action_data: dict[str, Any]) -> None:
        await self.broadcast({"type": "recovery_started", "data": action_data})

    async def emit_recovery_completed(self, action_data: dict[str, Any]) -> None:
        await self.broadcast({"type": "recovery_completed", "data": action_data})

    async def emit_recovery_failed(self, action_data: dict[str, Any]) -> None:
        await self.broadcast({"type": "recovery_failed", "data": action_data})
