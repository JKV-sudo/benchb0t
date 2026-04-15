from __future__ import annotations

import json
from pathlib import Path

from framework.dashboard_context import (
    build_analytics_context,
    build_builder_context,
    build_chat_context,
    load_live_session_context,
)
from framework.store import Store


class _FakeStore:
    def get_summary(self):
        return {
            "total_runs": 3,
            "total_models": 2,
            "total_levels": 2,
            "avg_score": 74.5,
            "best_score": 93.0,
            "total_stars": 11,
        }

    def get_model_stats(self):
        return [
            {
                "model": "gpt-4.1",
                "avg_score": 93.0,
                "best_score": 93.0,
                "run_count": 1,
                "total_stars": 5,
                "avg_turns": 4,
                "timeouts": 0,
            }
        ]

    def get_level_stats(self):
        return [
            {
                "level_id": "l99-test",
                "difficulty": 1,
                "avg_score": 88.0,
                "best_score": 93.0,
                "run_count": 2,
                "pass_rate": 0.5,
            }
        ]

    def get_mode_comparison(self):
        return [
            {
                "model": "gpt-4.1",
                "level_id": "l99-test",
                "mode": "guided",
                "avg_score": 93.0,
                "avg_turns": 4,
                "timeouts": 0,
            }
        ]

    def get_runs(self, limit: int = 20):
        return [
            {
                "ts": 1_700_000_000.0,
                "model": "gpt-4.1",
                "level_id": "l99-test",
                "score_total": 93.0,
                "stars": 5,
                "turns": 4,
                "tool_calls_n": 7,
                "timed_out": 0,
            }
        ]


def test_load_live_session_context_formats_agentlog(tmp_path: Path) -> None:
    log_path = tmp_path / "20260414_l99-test_abc12345.agentlog"
    events = [
        {"type": "session_start", "run_id": "abc12345", "model": "hermes", "level_id": "l99-test", "mode": "guided"},
        {"type": "tool_call", "tool": "bash", "args": {"command": "ls"}},
        {"type": "tool_result", "exit_code": 0, "output": "file.txt"},
        {"type": "message", "role": "assistant", "content": "done"},
        {"type": "session_end", "score": {"total": 88.0}, "timed_out": False, "duration_s": 12.5},
    ]
    log_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    context = load_live_session_context("abc12345", tmp_path)

    assert "LIVE SESSION LOG" in context
    assert "[START] model=hermes level=l99-test mode=guided" in context
    assert "[TOOL] bash" in context
    assert "[END] score=88.0 timed_out=False duration=12.5s" in context


def test_build_analytics_context_includes_model_and_level_stats() -> None:
    context = build_analytics_context(_FakeStore())

    assert "ANALYTICS page" in context
    assert "MODEL LEADERBOARD:" in context
    assert "gpt-4.1: avg=93.0" in context
    assert "LEVEL DIFFICULTY vs PASS RATE:" in context
    assert "l99-test diff=1 avg=88.0 pass=50% runs=2" in context


def test_build_builder_context_includes_current_yaml_and_examples(tmp_path: Path) -> None:
    levels_dir = tmp_path / "levels"
    levels_dir.mkdir()
    (levels_dir / "l99-test.yaml").write_text(
        "level:\n  id: l99-test\n  name: Smoke Test\ncontainer:\n  image: python:3.11-slim\n"
        "task:\n  instruction: Run smoke test\n"
        "tools: [bash]\n"
        "evaluation:\n  criteria: []\n",
        encoding="utf-8",
    )

    context = build_builder_context(tmp_path, current_level_yaml="level:\n  id: draft-level\n")

    assert "LEVEL BUILDER page" in context
    assert "CURRENT LEVEL BEING EDITED" in context
    assert "draft-level" in context
    assert "EXISTING LEVELS AS REFERENCE" in context
    assert "l99-test" in context
    assert "<level_patch>" in context


def test_build_chat_context_includes_summary_levels_and_recent_logs(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    levels_dir = project_dir / "levels"
    levels_dir.mkdir()
    levels_dir.joinpath("l99-test.yaml").write_text(
        "level:\n  id: l99-test\n  name: Smoke Test\n  difficulty: 1\n  category: web\n"
        "task:\n  instruction: Fix app\n  max_turns: 5\n"
        "evaluation:\n  criteria:\n    - id: smoke\n",
        encoding="utf-8",
    )
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    runs_dir.joinpath("20260414_l99-test_abc12345.agentlog").write_text(
        json.dumps({"type": "tool_call", "tool": "bash", "args": {"command": "ls"}}) + "\n"
        + json.dumps({"type": "tool_result", "exit_code": 0, "output": "ok"}) + "\n",
        encoding="utf-8",
    )

    context = build_chat_context(
        store=_FakeStore(),
        project_dir=project_dir,
        runs_dir=runs_dir,
        active_run_id="",
        page="dashboard",
    )

    assert "BENCHMARK SUMMARY: 3 total runs" in context
    assert 'l99-test "Smoke Test" diff=1 cat=web' in context
    assert "RECENT AGENT LOGS" in context
    assert "CALL bash" in context

