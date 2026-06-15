from __future__ import annotations

import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from framework import dashboard as dashboard_mod


@pytest.mark.asyncio
async def test_history_page_renders(dashboard_test_runtime) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        resp = await client.get("/history")

    assert resp.status_code == 200
    assert "RUN ARCHIVE" in resp.text
    assert "HISTORY" in resp.text


@pytest.mark.asyncio
async def test_history_api_lists_runs_with_artifact_counts(dashboard_test_runtime) -> None:
    write_run = dashboard_test_runtime["write_run"]
    write_run(run_id="abc12345", model="hermes", screenshot=True, bundle=True)
    write_run(run_id="def67890", model="gpt-4.1", snapshot=True)

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        resp = await client.get("/api/history?limit=10")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total"] == 2
    runs_by_id = {run["id"]: run for run in payload["runs"]}
    assert runs_by_id["abc12345"]["artifact_counts"]["screenshots"] == 1
    assert runs_by_id["abc12345"]["artifact_counts"]["bundles"] == 1
    assert runs_by_id["def67890"]["artifact_counts"]["snapshots"] == 1


@pytest.mark.asyncio
async def test_replay_api_includes_artifacts(dashboard_test_runtime) -> None:
    write_run = dashboard_test_runtime["write_run"]
    write_run(run_id="abc12345", model="hermes", screenshot=True, bundle=True)

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        resp = await client.get("/api/replays/abc12345")

    assert resp.status_code == 200
    payload = resp.json()
    artifact_kinds = [artifact["kind"] for artifact in payload["artifacts"]]
    assert payload["run"]["id"] == "abc12345"
    assert "preview_screenshot" in artifact_kinds
    assert "result_bundle" in artifact_kinds


@pytest.mark.asyncio
async def test_compare_api_returns_metric_diff_and_timeline_pairs(dashboard_test_runtime) -> None:
    write_run = dashboard_test_runtime["write_run"]
    write_run(run_id="abc12345", model="hermes", score_total=84.0, duration_s=19.0, turns=6, tool_calls_n=10, screenshot=True)
    write_run(run_id="def67890", model="gpt-4.1", score_total=93.0, duration_s=14.0, turns=4, tool_calls_n=7, bundle=True)

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        resp = await client.get("/api/compare?left_run_id=abc12345&right_run_id=def67890")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["summary"]["winner"] == "right"
    metric_ids = {metric["id"]: metric for metric in payload["summary"]["metrics"]}
    assert metric_ids["score_total"]["better"] == "right"
    assert metric_ids["duration_s"]["better"] == "right"
    assert len(payload["timeline_pairs"]) >= 3


@pytest.mark.asyncio
async def test_log_and_artifact_routes_serve_saved_files(dashboard_test_runtime) -> None:
    write_run = dashboard_test_runtime["write_run"]
    write_run(run_id="abc12345", model="hermes", screenshot=True)

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        log_resp = await client.get("/api/logs/abc12345")
        artifact_resp = await client.get("/api/artifacts/abc12345/preview.png")

    assert log_resp.status_code == 200
    assert '"run_id": "abc12345"' in log_resp.text
    assert artifact_resp.status_code == 200
    assert artifact_resp.content == b"png-bytes"


@pytest.mark.asyncio
async def test_credentials_status_stop_and_preflight_routes(
    dashboard_test_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Proc:
        def __init__(self, pid: int, alive: bool = True) -> None:
            self.pid = pid
            self._alive = alive
            self.terminated = False

        def poll(self):
            return None if self._alive else 0

        def terminate(self) -> None:
            self.terminated = True
            self._alive = False

    dashboard_mod.state.active_procs = [Proc(1234)]
    monkeypatch.setattr(dashboard_mod, "check_docker", lambda: {"ok": True, "msg": "ready"})
    monkeypatch.setattr(dashboard_mod, "check_api", lambda base_url: {"ok": True, "msg": base_url})
    monkeypatch.setattr(dashboard_mod, "probe_preview_status", lambda port, path="/": {"up": True, "status": 200, "url": f"http://localhost:{port}{path}"})

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        save_resp = await client.post(
            "/api/credentials",
            json={"base_url": "http://localhost:11434/v1", "model": "hermes3", "api_key": "secret"},
        )
        creds_resp = await client.get("/api/credentials")
        status_resp = await client.get("/api/status")
        preflight_resp = await client.get("/api/preflight?base_url=http://localhost:11434/v1")
        preview_resp = await client.get("/api/preview-status?port=49312&path=/")
        stop_resp = await client.post("/api/stop")

    assert save_resp.status_code == 200
    assert creds_resp.json()["model"] == "hermes3"
    assert status_resp.json()["status"] == "running"
    assert preflight_resp.json()["docker"]["ok"] is True
    assert preview_resp.json()["up"] is True
    assert stop_resp.json() == {"status": "stopped"}


@pytest.mark.asyncio
async def test_stats_levels_runs_and_parsed_level_routes(dashboard_test_runtime) -> None:
    write_run = dashboard_test_runtime["write_run"]
    project_dir = dashboard_test_runtime["project_dir"]
    level_path = project_dir / "levels" / "l99-test.yaml"
    level_path.write_text(
        "level:\n"
        "  id: l99-test\n"
        "  name: Smoke Test\n"
        "  difficulty: 1\n"
        "  category: web\n"
        "  tags: [smoke]\n"
        "container:\n"
        "  image: python:3.11-slim\n"
        "  working_dir: /workspace\n"
        "  packages:\n"
        "    apt: [curl]\n"
        "task:\n"
        "  instruction: Run smoke test\n"
        "  max_turns: 5\n"
        "  timeout_s: 20\n"
        "tools: [bash]\n"
        "preview:\n"
        "  port: 3000\n"
        "  path: /\n"
        "evaluation:\n"
        "  efficiency_target: 5\n"
        "  criteria:\n"
        "    - id: smoke\n"
        "      description: should pass\n"
        "      check: echo ok\n"
        "      weight: 1.0\n",
        encoding="utf-8",
    )
    write_run(run_id="abc12345", model="hermes", screenshot=True)

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        summary_resp = await client.get("/api/stats/summary")
        models_resp = await client.get("/api/stats/models")
        levels_resp = await client.get("/api/stats/levels")
        runs_resp = await client.get("/api/runs?limit=10")
        meta_resp = await client.get("/api/runs/meta")
        parsed_resp = await client.get("/api/levels/l99-test/parsed")
        save_level_resp = await client.post(
            "/api/levels/save",
            json={
                "filename": "saved-level",
                "content": (
                    "level:\n"
                    "  id: saved-level\n"
                    "  name: Saved Level\n"
                    "  difficulty: 1\n"
                    "  category: web\n"
                    "container:\n"
                    "  image: python:3.11-slim\n"
                    "  working_dir: /workspace\n"
                    "task:\n"
                    "  instruction: Run saved level\n"
                    "tools: [bash]\n"
                    "evaluation:\n"
                    "  criteria: []\n"
                ),
            },
        )

    assert summary_resp.json()["total_runs"] == 1
    assert models_resp.json()[0]["model"] == "hermes"
    assert levels_resp.json()[0]["level_id"] == "l99-test"
    assert runs_resp.json()["total"] == 1
    assert meta_resp.json()["models"] == ["hermes"]
    assert parsed_resp.json()["id"] == "l99-test"
    assert save_level_resp.json()["status"] == "saved"
    assert (project_dir / "levels" / "saved-level.yaml").exists()


@pytest.mark.asyncio
async def test_save_level_route_rejects_invalid_yaml(dashboard_test_runtime) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        resp = await client.post(
            "/api/levels/save",
            json={
                "filename": "broken-level",
                "content": "level:\n  id: broken-level\n",
            },
        )

    assert resp.status_code == 400
    assert "Invalid level config" in resp.json()["error"]


@pytest.mark.asyncio
async def test_validate_level_route_returns_warnings_for_draft_format(dashboard_test_runtime) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        resp = await client.post(
            "/api/levels/validate",
            json={
                "filename": "l-my-level",
                "content": (
                    "level:\n"
                    "  id: l-my-level\n"
                    "  name: Draft Level\n"
                    "  difficulty: 2\n"
                    "  category: webapp\n"
                    "container:\n"
                    "  image: node:20-slim\n"
                    "  working_dir: /workspace\n"
                    "task:\n"
                    "  instruction: Build a site\n"
                    "tools:\n"
                    "  - bash\n"
                    "  - run_background\n"
                    "preview:\n"
                    "  port: 3000\n"
                    "evaluation:\n"
                    "  type: script\n"
                    "  efficiency_target: 3\n"
                    "  criteria: []\n"
                ),
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["valid"] is True
    assert any("Recommended filename format" in warning for warning in payload["warnings"])
    assert any("Recommended level.id format" in warning for warning in payload["warnings"])


@pytest.mark.asyncio
async def test_validate_level_route_returns_structured_errors(dashboard_test_runtime) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        resp = await client.post(
            "/api/levels/validate",
            json={
                "filename": "broken-level",
                "content": (
                    "level:\n"
                    "  id: broken-level\n"
                    "  name: Broken Level\n"
                    "  difficulty: 2\n"
                    "  category: webapp\n"
                    "container:\n"
                    "  image: https://example.com/not-a-docker-image.png\n"
                    "  working_dir: /workspace\n"
                    "task:\n"
                    "  instruction: Build a site\n"
                    "tools:\n"
                    "  - bash\n"
                    "  - run_background\n"
                    "preview:\n"
                    "  port: 3000\n"
                    "evaluation:\n"
                    "  type: script\n"
                    "  efficiency_target: 3\n"
                    "  criteria: []\n"
                ),
            },
        )

    assert resp.status_code == 400
    payload = resp.json()
    assert payload["valid"] is False
    assert any("Docker image reference" in error for error in payload["errors"])


@pytest.mark.asyncio
async def test_chat_route_handles_missing_and_successful_provider(
    dashboard_test_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        missing_resp = await client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert "no endpoint configured" in missing_resp.text

    class FakeAgentAPI:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def stream_chat(self, messages):
            yield "hello "
            yield "world"

    monkeypatch.setattr("framework.api.AgentAPI", FakeAgentAPI)

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        ok_resp = await client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "base_url": "http://localhost:11434/v1",
                "model": "hermes3",
                "page": "analytics",
            },
        )

    assert ok_resp.status_code == 200
    assert 'data: {"delta": "hello "}' in ok_resp.text
    assert 'data: {"delta": "world"}' in ok_resp.text


@pytest.mark.asyncio
async def test_chat_route_can_configure_dashboard_via_tools(
    dashboard_test_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "configure_benchbot_run",
                                    "arguments": json.dumps(
                                        {"all_levels": True, "save_result_bundle": True}
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Configured the dashboard run options.",
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    ]

    class FakeAgentAPI:
        def __init__(self, **kwargs) -> None:
            return None

        def chat(self, messages, tools=None):
            return responses.pop(0)

        def chat_with_stream(self, messages, tools=None, *, on_text_delta=None):
            response = self.chat(messages, tools)
            message = response["choices"][0].get("message", {})
            content = message.get("content") or ""
            if content and on_text_delta:
                on_text_delta(content)
            return response

    monkeypatch.setattr("framework.api.AgentAPI", FakeAgentAPI)

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        resp = await client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "run all levels and save the bundle"}],
                "base_url": "http://localhost:11434/v1",
                "model": "hermes3",
                "page": "dashboard",
            },
        )

    assert resp.status_code == 200
    assert '"_type": "tool_result"' in resp.text
    assert '"all_levels": true' in resp.text
    assert '"save_result_bundle": true' in resp.text
    assert 'Configured the dashboard run options.' in resp.text


@pytest.mark.asyncio
async def test_chat_route_can_start_run_via_tool(
    dashboard_test_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness_path = dashboard_test_runtime["project_dir"] / "harnesses" / "hermes.yaml"
    harness_path.write_text("harness:\n  name: hermes\n", encoding="utf-8")
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "start_benchbot_run",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Started the run."},
                    "finish_reason": "stop",
                }
            ]
        },
    ]

    class Proc:
        def __init__(self, pid: int = 8888) -> None:
            self.pid = pid
            self.stdout = type("Stdout", (), {"readline": lambda self: ""})()

        def poll(self):
            return None

    class FakeAgentAPI:
        def __init__(self, **kwargs) -> None:
            return None

        def chat(self, messages, tools=None):
            return responses.pop(0)

        def chat_with_stream(self, messages, tools=None, *, on_text_delta=None):
            response = self.chat(messages, tools)
            message = response["choices"][0].get("message", {})
            content = message.get("content") or ""
            if content and on_text_delta:
                on_text_delta(content)
            return response

    monkeypatch.setattr("framework.api.AgentAPI", FakeAgentAPI)
    monkeypatch.setattr(dashboard_mod.subprocess, "Popen", lambda *args, **kwargs: Proc())
    monkeypatch.setattr(dashboard_mod.asyncio, "create_task", lambda coro: None)
    monkeypatch.setattr(dashboard_mod, "_drain_runner", lambda proc, prefix="": None)

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        resp = await client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "start the benchmark"}],
                "base_url": "http://localhost:11434/v1",
                "model": "hermes3",
                "page": "dashboard",
                "providers": [{"base_url": "http://localhost:11434/v1", "model": "hermes3", "api_key": "", "label": "hermes3"}],
                "all_levels": True,
            },
        )

    assert resp.status_code == 200
    assert '"_type": "run_started"' in resp.text
    assert '"status": "started"' in resp.text
    assert 'Started the run.' in resp.text


@pytest.mark.asyncio
async def test_chat_route_can_create_level_via_tool(
    dashboard_test_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "create_benchbot_level",
                                    "arguments": json.dumps(
                                        {
                                            "id": "l15-italian-restaurant",
                                            "name": "Italian Restaurant",
                                            "difficulty": 3,
                                            "category": "webapp",
                                            "image": "node:20-slim",
                                            "instruction": "Build a full Italian restaurant page with 15 menu items.",
                                            "tools": ["BASH", "write", "curl"],
                                            "criteria": [
                                                {
                                                    "id": "menu_items_count",
                                                    "description": "renders 15 menu items",
                                                    "check": "grep -q 15 menu.txt",
                                                    "weight": 2,
                                                }
                                            ],
                                            "preview_port": 3000,
                                            "save": True,
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "I created the level file."},
                    "finish_reason": "stop",
                }
            ]
        },
    ]

    class FakeAgentAPI:
        def __init__(self, **kwargs) -> None:
            return None

        def chat(self, messages, tools=None):
            return responses.pop(0)

        def chat_with_stream(self, messages, tools=None, *, on_text_delta=None):
            response = self.chat(messages, tools)
            message = response["choices"][0].get("message", {})
            content = message.get("content") or ""
            if content and on_text_delta:
                on_text_delta(content)
            return response

    monkeypatch.setattr("framework.api.AgentAPI", FakeAgentAPI)

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        resp = await client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Erstell mir ein neues italienisches Restaurant Level"}],
                "base_url": "http://localhost:11434/v1",
                "model": "hermes3",
                "page": "dashboard",
            },
        )

    assert resp.status_code == 200
    assert '"_type": "level_saved"' in resp.text
    assert 'l15-italian-restaurant' in resp.text
    saved_path = dashboard_test_runtime["project_dir"] / "levels" / "l15-italian-restaurant.yaml"
    assert saved_path.exists()
    saved_text = saved_path.read_text(encoding="utf-8")
    assert 'setup_script: ""' in saved_text
    assert "  - http_request" in saved_text
    assert "  - run_background" in saved_text


@pytest.mark.asyncio
async def test_chat_route_validates_builder_level_drafts(
    dashboard_test_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "create_benchbot_level",
                                    "arguments": json.dumps(
                                        {
                                            "id": "l17-invalid-draft",
                                            "name": "Invalid Draft",
                                            "difficulty": 3,
                                            "category": "webapp",
                                            "image": "https://images.example.com/not-a-docker-image.png",
                                            "instruction": "Build a preview app.",
                                            "tools": ["bash", "run_background"],
                                            "criteria": [
                                                {
                                                    "id": "server_responds",
                                                    "description": "server responds",
                                                    "check": "curl -sf http://localhost:3000/",
                                                    "weight": 1,
                                                }
                                            ],
                                            "preview_port": 3000,
                                            "save": False,
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    ]

    class FakeAgentAPI:
        def __init__(self, **kwargs) -> None:
            return None

        def chat(self, messages, tools=None):
            return responses.pop(0)

        def chat_with_stream(self, messages, tools=None, *, on_text_delta=None):
            response = self.chat(messages, tools)
            message = response["choices"][0].get("message", {})
            content = message.get("content") or ""
            if content and on_text_delta:
                on_text_delta(content)
            return response

    monkeypatch.setattr("framework.api.AgentAPI", FakeAgentAPI)

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        resp = await client.post(
            "/api/chat",
            json={
                "messages": [{"role": "user", "content": "Erstell mir einen Level-Draft"}],
                "base_url": "http://localhost:11434/v1",
                "model": "hermes3",
                "page": "builder",
            },
        )

    assert resp.status_code == 200
    assert '"error"' in resp.text
    assert "Docker image reference" in resp.text
    assert not (dashboard_test_runtime["project_dir"] / "levels" / "l17-invalid-draft.yaml").exists()


@pytest.mark.asyncio
async def test_run_route_rejects_missing_provider_and_conflict(
    dashboard_test_runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Proc:
        def __init__(self, pid: int = 7777) -> None:
            self.pid = pid
            self.stdout = type("Stdout", (), {"readline": lambda self: ""})()

        def poll(self):
            return None

    harness_path = dashboard_test_runtime["project_dir"] / "harnesses" / "hermes.yaml"
    harness_path.write_text("harness:\n  name: hermes\n", encoding="utf-8")

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        missing_provider = await client.post("/api/run", json={"messages": []})

    assert missing_provider.status_code == 400

    dashboard_mod.state.active_procs = [Proc(1111)]
    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        conflict = await client.post(
            "/api/run",
            json={"base_url": "http://localhost:11434/v1", "model": "hermes3"},
        )

    assert conflict.status_code == 409

    dashboard_mod.state.active_procs = []
    monkeypatch.setattr(dashboard_mod.subprocess, "Popen", lambda *args, **kwargs: Proc())
    monkeypatch.setattr(dashboard_mod.asyncio, "create_task", lambda coro: None)
    monkeypatch.setattr(dashboard_mod, "_drain_runner", lambda proc, prefix="": None)

    async with AsyncClient(
        transport=ASGITransport(app=dashboard_test_runtime["app"]),
        base_url="http://testserver",
    ) as client:
        ok_resp = await client.post(
            "/api/run",
            json={
                "base_url": "http://localhost:11434/v1",
                "model": "hermes3",
                "capture_preview_screenshot": True,
            },
        )

    assert ok_resp.status_code == 200
    assert ok_resp.json()["status"] == "started"
