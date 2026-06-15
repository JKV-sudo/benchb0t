from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient


class _FakeAgentAPI:
    def __init__(self, *args, **kwargs) -> None:
        self.model = kwargs.get("model", "fake")

    def chat_with_stream(self, messages, tools=None, *, on_text_delta=None, **kwargs):
        # Simulate assistant that first thinks, then calls a tool, then answers.
        if on_text_delta:
            on_text_delta("Let me check that.")
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check that.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_benchbot_status",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }


@pytest.mark.asyncio
async def test_chat_stream_emits_tool_lifecycle_events(dashboard_test_runtime, monkeypatch) -> None:
    from framework import api as api_mod

    monkeypatch.setattr(api_mod, "AgentAPI", _FakeAgentAPI)

    app = dashboard_test_runtime["app"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        resp = await client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "status"}],
                "page": "dashboard",
                "allow_control": True,
                "base_url": "http://localhost:11434/v1",
                "model": "hermes3",
            },
        )

    assert resp.status_code == 200
    lines = [line.strip() for line in resp.text.splitlines() if line.strip()]
    events = []
    for line in lines:
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            events.append({"_type": "done"})
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            pass

    assert events[-1]["_type"] == "done"
    types = [ev.get("_type") for ev in events if "_type" in ev]
    assert "tool_calls" in types
    assert "tool_start" in types
    assert "tool_result" in types

    tool_calls_event = next(ev for ev in events if ev.get("_type") == "tool_calls")
    assert tool_calls_event["tool_calls"][0]["name"] == "get_benchbot_status"

    tool_start_event = next(ev for ev in events if ev.get("_type") == "tool_start")
    assert tool_start_event["tool"] == "get_benchbot_status"

    tool_result_event = next(ev for ev in events if ev.get("_type") == "tool_result")
    assert tool_result_event["tool"] == "get_benchbot_status"
    assert "message" in tool_result_event
    assert "data" in tool_result_event
