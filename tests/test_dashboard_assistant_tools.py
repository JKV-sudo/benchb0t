from __future__ import annotations

import pytest

from framework import dashboard as dashboard_mod
from framework.dashboard import _execute_dashboard_assistant_tool
from framework.dashboard_assistant import build_initial_assistant_state


@pytest.mark.asyncio
async def test_create_benchbot_level_derives_filename(dashboard_test_runtime) -> None:
    project_dir = dashboard_test_runtime["project_dir"]
    assistant_state = build_initial_assistant_state(
        __chat_request(),
        {},
        stored_providers=[],
    )

    result = await _execute_dashboard_assistant_tool(
        "create_benchbot_level",
        {
            "id": "l42-test-tool",
            "name": "Test Tool Level",
            "difficulty": 1,
            "category": "file-operations",
            "image": "python:3.11-slim",
            "instruction": "Write hello to output.txt",
            "tools": ["bash"],
            "criteria": [
                {
                    "id": "ok",
                    "description": "file exists",
                    "check": "test -f /workspace/output.txt",
                }
            ],
            "filename": "this-should-be-ignored.yaml",
            "save": True,
        },
        assistant_state=assistant_state,
        page="dashboard",
    )

    assert result["ok"] is True
    assert result["event_type"] == "level_saved"
    assert result["data"]["filename"] == "l42-test-tool.yaml"
    assert result["data"]["saved_path"].endswith("levels/l42-test-tool.yaml")
    assert (project_dir / "levels" / "l42-test-tool.yaml").exists()


@pytest.mark.asyncio
async def test_edit_benchbot_level_patches_existing(dashboard_test_runtime) -> None:
    project_dir = dashboard_test_runtime["project_dir"]
    level_path = project_dir / "levels" / "l50-edit-me.yaml"
    level_path.write_text(
        "level:\n"
        "  id: l50-edit-me\n"
        "  name: Old Name\n"
        "  difficulty: 1\n"
        "  category: general\n"
        "container:\n"
        "  image: python:3.11-slim\n"
        "  working_dir: /workspace\n"
        "task:\n"
        "  instruction: Old instruction\n"
        "tools: [bash]\n"
        "evaluation:\n"
        "  criteria: []\n",
        encoding="utf-8",
    )
    assistant_state = build_initial_assistant_state(
        __chat_request(),
        {},
        stored_providers=[],
    )

    result = await _execute_dashboard_assistant_tool(
        "edit_benchbot_level",
        {
            "level_id": "l50-edit-me",
            "name": "New Name",
            "instruction": "New instruction",
            "save": True,
        },
        assistant_state=assistant_state,
        page="dashboard",
    )

    assert result["ok"] is True
    assert result["event_type"] == "level_saved"
    content = level_path.read_text(encoding="utf-8")
    assert "New Name" in content
    assert "New instruction" in content
    assert "id:         l50-edit-me" in content


@pytest.mark.asyncio
async def test_get_benchbot_stats_and_history(dashboard_test_runtime) -> None:
    write_run = dashboard_test_runtime["write_run"]
    write_run(run_id="abc12345", model="hermes")
    write_run(run_id="def67890", model="gpt-4.1")

    assistant_state = build_initial_assistant_state(
        __chat_request(),
        {},
        stored_providers=[],
    )

    stats = await _execute_dashboard_assistant_tool(
        "get_benchbot_stats", {}, assistant_state=assistant_state, page="dashboard"
    )
    assert stats["ok"] is True
    assert stats["data"]["total_runs"] == 2
    assert stats["data"]["total_models"] == 2

    history = await _execute_dashboard_assistant_tool(
        "get_benchbot_run_history",
        {"limit": 10, "model": "hermes"},
        assistant_state=assistant_state,
        page="dashboard",
    )
    assert history["ok"] is True
    assert len(history["data"]["runs"]) == 1
    assert history["data"]["runs"][0]["model"] == "hermes"


@pytest.mark.asyncio
async def test_get_benchbot_run_detail_and_compare(dashboard_test_runtime) -> None:
    write_run = dashboard_test_runtime["write_run"]
    write_run(run_id="abc12345", model="hermes", score_total=84.0)
    write_run(run_id="def67890", model="gpt-4.1", score_total=93.0)

    assistant_state = build_initial_assistant_state(
        __chat_request(),
        {},
        stored_providers=[],
    )

    detail = await _execute_dashboard_assistant_tool(
        "get_benchbot_run_detail",
        {"run_id": "abc12345"},
        assistant_state=assistant_state,
        page="dashboard",
    )
    assert detail["ok"] is True
    assert detail["data"]["run"]["id"] == "abc12345"

    comparison = await _execute_dashboard_assistant_tool(
        "compare_benchbot_runs",
        {"left_run_id": "abc12345", "right_run_id": "def67890"},
        assistant_state=assistant_state,
        page="dashboard",
    )
    assert comparison["ok"] is True
    assert comparison["data"]["summary"]["winner"] == "right"


@pytest.mark.asyncio
async def test_list_and_detail_models(dashboard_test_runtime) -> None:
    write_run = dashboard_test_runtime["write_run"]
    write_run(run_id="abc12345", model="hermes")

    assistant_state = build_initial_assistant_state(
        __chat_request(),
        {},
        stored_providers=[],
    )

    models = await _execute_dashboard_assistant_tool(
        "list_benchbot_models", {}, assistant_state=assistant_state, page="dashboard"
    )
    assert models["ok"] is True
    assert "hermes" in models["data"]["models"]

    detail = await _execute_dashboard_assistant_tool(
        "get_benchbot_model_detail",
        {"model": "hermes"},
        assistant_state=assistant_state,
        page="dashboard",
    )
    assert detail["ok"] is True
    assert detail["data"]["model"] == "hermes"


def __chat_request() -> "ChatRequest":
    from framework.dashboard_models import ChatRequest

    return ChatRequest(
        messages=[],
        base_url="http://localhost:11434/v1",
        model="hermes3",
    )
