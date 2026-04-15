from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from framework.container import ContainerError
from framework.recorder import Recorder, load_agentlog
import framework.runner as runner


class _ToolContainer:
    def __init__(self) -> None:
        self.exec_commands: list[str] = []
        self.writes: list[tuple[str, str]] = []

    def exec(self, command: str) -> tuple[int, str]:
        self.exec_commands.append(command)
        if command.startswith("bash -c"):
            return 0, "1234\n"
        if command.startswith("kill -0"):
            return 0, "RUNNING\n"
        if command.startswith("tail -30"):
            return 0, "server ready\n"
        if command.startswith("python3 /tmp/_benchbot_patch.py"):
            return 0, "patched\n"
        return 0, f"exec:{command}"

    def read_file(self, path: str) -> str:
        if path == "/missing":
            raise ContainerError("missing file")
        return f"content:{path}"

    def write_file(self, path: str, content: str) -> None:
        if path == "/forbidden":
            raise ContainerError("no write")
        self.writes.append((path, content))


class _SequenceAPI:
    def __init__(self, responses: list[Any], *, fail_stream: bool = False, fallback_response: dict | None = None) -> None:
        self._responses = list(responses)
        self._fail_stream = fail_stream
        self._fallback_response = fallback_response
        self.chat_calls = 0
        self.chat_with_stream_calls = 0

    def chat_with_stream(self, messages, tools=None, on_text_delta=None):
        self.chat_with_stream_calls += 1
        if self._fail_stream:
            raise RuntimeError("stream down")
        response = self._responses.pop(0)
        if on_text_delta is not None:
            for delta in response.get("deltas", []):
                on_text_delta(delta)
        return response["payload"]

    def chat(self, messages, tools=None):
        self.chat_calls += 1
        return self._fallback_response


def _recorder(tmp_path: Path) -> Recorder:
    return Recorder(tmp_path, "l99-test", "hermes")


def test_dispatch_tool_handles_common_tools() -> None:
    container = _ToolContainer()

    assert runner.dispatch_tool("bash", {"command": "echo hi"}, container) == (0, "exec:echo hi")
    assert runner.dispatch_tool("read_file", {"path": "/tmp/x"}, container) == (0, "content:/tmp/x")
    assert runner.dispatch_tool("write_file", {"path": "/tmp/x", "content": "abc"}, container) == (0, "Written to /tmp/x")
    assert runner.dispatch_tool("list_dir", {"path": "/workspace"}, container) == (0, "exec:ls -lah /workspace 2>&1")
    http_code, http_out = runner.dispatch_tool(
        "http_request",
        {"method": "POST", "url": "http://localhost:3000/api", "headers": {"X-Test": "1"}, "body": '{"ok":true}'},
        container,
    )
    assert http_code == 0
    assert "curl -s -X POST" in http_out
    bg_code, bg_out = runner.dispatch_tool("run_background", {"command": "npm run dev", "wait_seconds": 2}, container)
    assert bg_code == 0
    assert "PID=1234" in bg_out
    patch_code, patch_out = runner.dispatch_tool(
        "patch_file",
        {"path": "/workspace/app.py", "old": "foo", "new": "bar"},
        container,
    )
    assert patch_code == 0
    assert patch_out.strip() == "patched"
    assert runner.dispatch_tool("unknown", {}, container) == (1, "Unknown tool: unknown")


def test_dispatch_tool_surfaces_container_errors() -> None:
    container = _ToolContainer()

    assert runner.dispatch_tool("read_file", {"path": "/missing"}, container) == (1, "missing file")
    assert runner.dispatch_tool("write_file", {"path": "/forbidden", "content": "x"}, container) == (1, "no write")


def test_run_agent_loop_finishes_on_stop(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.start({"level": {"name": "Smoke Test"}}, {"harness": {"type": "openai_compat"}})
    api = _SequenceAPI(
        [
            {
                "deltas": ["done"],
                "payload": {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            }
        ]
    )

    ok = runner.run_agent_loop(
        api=api,
        container=_ToolContainer(),
        recorder=recorder,
        task_cfg={"instruction": "solve it", "max_turns": 3, "timeout_s": 30},
        tools_list=["bash"],
    )
    recorder.end(score={"total": 100})

    assert ok is True
    events = load_agentlog(recorder.path)
    assert [event["type"] for event in events if event["type"] in ("message", "message_delta")] == [
        "message",
        "message_delta",
        "message",
    ]


def test_run_agent_loop_executes_tool_calls_then_finishes(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.start({"level": {"name": "Smoke Test"}}, {"harness": {"type": "openai_compat"}})
    container = _ToolContainer()
    api = _SequenceAPI(
        [
            {
                "payload": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            },
            {
                "payload": {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "done"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            },
        ]
    )

    ok = runner.run_agent_loop(
        api=api,
        container=container,
        recorder=recorder,
        task_cfg={"instruction": "solve it", "max_turns": 4, "timeout_s": 30},
        tools_list=["bash"],
    )

    assert ok is True
    assert container.exec_commands == ["ls"]
    assert recorder.tool_calls[0]["tool"] == "bash"
    assert recorder.tool_calls[0]["output"] == "exec:ls"


def test_run_agent_loop_falls_back_to_non_streaming(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.start({"level": {"name": "Smoke Test"}}, {"harness": {"type": "openai_compat"}})
    api = _SequenceAPI(
        [],
        fail_stream=True,
        fallback_response={
            "choices": [
                {
                    "message": {"role": "assistant", "content": "fallback ok"},
                    "finish_reason": "stop",
                }
            ]
        },
    )

    ok = runner.run_agent_loop(
        api=api,
        container=_ToolContainer(),
        recorder=recorder,
        task_cfg={"instruction": "solve it", "max_turns": 2, "timeout_s": 30},
        tools_list=["bash"],
    )

    assert ok is True
    assert api.chat_with_stream_calls == 1
    assert api.chat_calls == 1
