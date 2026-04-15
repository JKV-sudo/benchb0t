from __future__ import annotations

from pathlib import Path

from framework.dashboard_assistant import (
    assistant_state_ui_patch,
    build_initial_assistant_state,
    build_level_patch_from_args,
    build_run_request_from_assistant_state,
    chat_request_to_provider_dicts,
    lint_level_content,
    list_levels_for_assistant,
    render_level_yaml_from_patch,
    resolve_level_reference,
    save_level_content,
)
from framework.dashboard_models import ChatRequest, ProviderRequest


def test_chat_request_to_provider_dicts_prefers_request_then_creds() -> None:
    req = ChatRequest(
        messages=[],
        providers=[ProviderRequest(base_url=" http://localhost:11434/v1 ", model=" hermes3 ", api_key="x", label="Hermes")],
    )

    providers = chat_request_to_provider_dicts(req, {"base_url": "http://fallback", "model": "fallback"})
    assert providers == [
        {
            "base_url": "http://localhost:11434/v1",
            "model": "hermes3",
            "api_key": "x",
            "label": "Hermes",
        }
    ]

    fallback = chat_request_to_provider_dicts(
        ChatRequest(messages=[]),
        {"base_url": "http://fallback", "model": "fallback", "api_key": "k"},
    )
    assert fallback[0]["model"] == "fallback"


def test_build_initial_assistant_state_and_run_request() -> None:
    state = build_initial_assistant_state(
        ChatRequest(
            messages=[],
            base_url="http://localhost:11434/v1",
            model="hermes3",
            level="levels/l99-test.yaml",
            all_levels=False,
            save_result_bundle=True,
        ),
        {},
    )

    assert state["providers"][0]["model"] == "hermes3"
    assert state["save_result_bundle"] is True

    patch = assistant_state_ui_patch(state)
    assert patch["level"] == "levels/l99-test.yaml"
    assert patch["providers"][0]["model"] == "hermes3"

    run_req = build_run_request_from_assistant_state(state)
    assert run_req.model == "hermes3"
    assert run_req.save_result_bundle is True


def test_list_levels_and_resolve_level_reference(tmp_path: Path) -> None:
    levels_dir = tmp_path / "levels"
    levels_dir.mkdir()
    levels_dir.joinpath("l99-test.yaml").write_text(
        "level:\n"
        "  id: l99-test\n"
        "  name: Smoke Test\n"
        "  difficulty: 1\n"
        "  category: web\n"
        "container:\n"
        "  image: python:3.11-slim\n"
        "  working_dir: /workspace\n"
        "task:\n"
        "  instruction: Run smoke test\n"
        "tools: [bash]\n"
        "evaluation:\n"
        "  criteria: []\n",
        encoding="utf-8",
    )
    levels_dir.joinpath("l01-old.yaml").write_text(
        "level:\n"
        "  id: l01-old\n"
        "  name: Old\n"
        "  difficulty: 1\n"
        "  category: web\n"
        "  deprecated: true\n"
        "container:\n"
        "  image: python:3.11-slim\n"
        "  working_dir: /workspace\n"
        "task:\n"
        "  instruction: Old\n"
        "tools: [bash]\n"
        "evaluation:\n"
        "  criteria: []\n",
        encoding="utf-8",
    )

    levels = list_levels_for_assistant(tmp_path)
    assert [level["id"] for level in levels] == ["l99-test"]
    assert resolve_level_reference(tmp_path, "l99-test").endswith("l99-test.yaml")
    assert resolve_level_reference(tmp_path, "l99-test.yaml").endswith("l99-test.yaml")


def test_build_render_and_save_level_patch(tmp_path: Path) -> None:
    patch = build_level_patch_from_args(
        {
            "id": "l15-italian-restaurant",
            "name": "Italian Restaurant",
            "difficulty": 3,
            "category": "webapp",
            "image": "node:20-slim",
            "instruction": "Build a restaurant website with 15 menu items.",
            "tools": ["bash", "write_file", "run_background"],
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
    )

    content = render_level_yaml_from_patch(patch)
    saved = save_level_content(tmp_path, "l15-italian-restaurant.yaml", content)

    assert saved.name == "l15-italian-restaurant.yaml"
    assert "setup_script: \"\"" in content
    assert "preview:" in content
    assert "menu_items_count" in content
    assert saved.read_text(encoding="utf-8") == content


def test_build_level_patch_normalizes_tool_aliases_and_preview() -> None:
    patch = build_level_patch_from_args(
        {
            "id": "l16-bella-italia-v2",
            "name": "Bella Italia V2",
            "difficulty": 4,
            "category": "webapp",
            "image": "node:20-slim",
            "instruction": "Build the app.",
            "tools": ["BASH", "curl", "serve", "write"],
            "criteria": [
                {
                    "id": "server_responds",
                    "description": "server responds",
                    "check": "curl -sf http://localhost:3000/",
                }
            ],
            "preview_port": 3000,
        }
    )

    assert patch["tools"] == ["bash", "http_request", "run_background", "write_file"]


def test_lint_level_content_reports_builder_warnings(tmp_path: Path) -> None:
    content = (
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
    )

    report = lint_level_content(tmp_path, "l-my-level.yaml", content)

    assert report["valid"] is True
    assert any("Recommended filename format" in warning for warning in report["warnings"])
    assert any("Recommended level.id format" in warning for warning in report["warnings"])
    assert any("Add at least one evaluation criterion" in warning for warning in report["warnings"])
