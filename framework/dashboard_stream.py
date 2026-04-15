"""
framework/dashboard_stream.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Live agentlog streaming helpers for the dashboard websocket.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import WebSocket


async def tail_agentlog(path: Path) -> AsyncIterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                try:
                    yield json.loads(stripped)
                except Exception:
                    pass

        while True:
            line = handle.readline()
            if line:
                stripped = line.strip()
                if stripped:
                    try:
                        yield json.loads(stripped)
                    except Exception:
                        pass
            else:
                await asyncio.sleep(0.08)


async def stream_agentlog(ws: WebSocket, path: Path) -> None:
    await ws.send_text(json.dumps({"_type": "file", "filename": path.name}))
    async for event in tail_agentlog(path):
        await ws.send_text(json.dumps(event))
        if event.get("type") == "session_end":
            break
