from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from framework.dashboard_stream import stream_agentlog, tail_agentlog


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_tail_agentlog_yields_initial_and_appended_events(tmp_path: Path) -> None:
    path = tmp_path / "run.agentlog"
    path.write_text(
        "not-json\n" + json.dumps({"type": "session_start", "run_id": "abc12345"}) + "\n",
        encoding="utf-8",
    )
    events = tail_agentlog(path)

    first = await events.__anext__()
    assert first["type"] == "session_start"

    async def append_event() -> None:
        await asyncio.sleep(0.12)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "session_end", "run_id": "abc12345"}) + "\n")

    task = asyncio.create_task(append_event())
    second = await events.__anext__()
    await task
    await events.aclose()

    assert second["type"] == "session_end"


@pytest.mark.asyncio
async def test_stream_agentlog_sends_header_and_stops_on_session_end(tmp_path: Path) -> None:
    path = tmp_path / "run.agentlog"
    path.write_text(
        json.dumps({"type": "session_start", "run_id": "abc12345"}) + "\n"
        + json.dumps({"type": "session_end", "run_id": "abc12345"}) + "\n",
        encoding="utf-8",
    )
    ws = _FakeWebSocket()

    await stream_agentlog(ws, path)

    assert json.loads(ws.sent[0]) == {"_type": "file", "filename": "run.agentlog"}
    assert json.loads(ws.sent[1])["type"] == "session_start"
    assert json.loads(ws.sent[2])["type"] == "session_end"
