from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from framework.store import Store


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def dashboard_test_runtime(tmp_path: Path):
    from framework import dashboard as dashboard_mod

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "levels").mkdir()
    (project_dir / "harnesses").mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    creds_file = tmp_path / ".benchb0t_creds.json"
    store = Store(tmp_path / "benchb0t.db").init()

    original = {
        "project_dir": dashboard_mod.state.project_dir,
        "runs_dir": dashboard_mod.state.runs_dir,
        "creds_file": dashboard_mod.state.creds_file,
        "store": dashboard_mod.state.store,
        "loaded_config": dashboard_mod.state.loaded_config,
        "active_procs": dashboard_mod.state.active_procs,
        "run_batch_started_at": dashboard_mod.state.run_batch_started_at,
        "runner_log": dashboard_mod.state.runner_log,
    }

    dashboard_mod.state.project_dir = project_dir
    dashboard_mod.state.runs_dir = runs_dir
    dashboard_mod.state.creds_file = creds_file
    dashboard_mod.state.store = store
    dashboard_mod.state.loaded_config = None
    dashboard_mod.state.active_procs = []
    dashboard_mod.state.run_batch_started_at = 0.0
    dashboard_mod.state.runner_log.clear()

    def write_run(
        *,
        run_id: str,
        model: str,
        level_id: str = "l99-test",
        level_name: str = "Smoke Test",
        score_total: float = 88.0,
        duration_s: float = 18.0,
        turns: int = 5,
        tool_calls_n: int = 9,
        timed_out: bool = False,
        preview_ready_offset: float | None = 4.5,
        screenshot: bool = False,
        bundle: bool = False,
        snapshot: bool = False,
    ) -> dict[str, Any]:
        result = {
            "run_id": run_id,
            "ts": 1_700_000_000.0,
            "level_id": level_id,
            "level_name": level_name,
            "difficulty": 1,
            "harness": "hermes",
            "mode": "unguided",
            "model": model,
            "base_url": "http://localhost:11434/v1",
            "log_path": str(runs_dir / f"20260414-120000_{level_id}_hermes_{run_id}.agentlog"),
            "timed_out": timed_out,
            "score": {
                "total": score_total,
                "dimensions": {
                    "completion": score_total * 0.5,
                    "efficiency": score_total * 0.2,
                    "self_correction": score_total * 0.2,
                    "path_quality": score_total * 0.1,
                },
                "penalties": {
                    "extra_calls": 0,
                    "backtracks": 0,
                    "timeout": 5 if timed_out else 0,
                    "retry": 0,
                },
                "criteria": [],
            },
            "turns": turns,
            "tool_calls_n": tool_calls_n,
            "duration_s": duration_s,
            "host_preview_port": 49312 if preview_ready_offset is not None else None,
            "preview_linger_seconds": 60 if preview_ready_offset is not None else 0,
            "preview_expires_at": 1_700_000_000.0 + duration_s + 60 if preview_ready_offset is not None else None,
            "artifacts": [],
        }
        store.record_run(result)

        events: list[dict[str, Any]] = [
            {
                "type": "session_start",
                "ts": 1_700_000_000.0,
                "run_id": run_id,
                "level_id": level_id,
                "level_name": level_name,
                "harness": "hermes",
                "model": model,
                "provider_label": model,
            }
        ]
        if preview_ready_offset is not None:
            events.append(
                {
                    "type": "preview_ready",
                    "ts": 1_700_000_000.0 + preview_ready_offset,
                    "run_id": run_id,
                    "level_id": level_id,
                    "harness": "hermes",
                    "host_preview_port": 49312,
                    "preview_path": "/",
                }
            )
        events.extend(
            [
                {
                    "type": "message",
                    "ts": 1_700_000_003.0,
                    "run_id": run_id,
                    "level_id": level_id,
                    "harness": "hermes",
                    "role": "assistant",
                    "content": "Done.",
                },
                {
                    "type": "session_end",
                    "ts": 1_700_000_000.0 + duration_s,
                    "run_id": run_id,
                    "level_id": level_id,
                    "duration_s": duration_s,
                    "timed_out": timed_out,
                    "score": {"total": score_total},
                },
            ]
        )

        artifact_dir = runs_dir / "artifacts" / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if screenshot:
            screenshot_path = artifact_dir / "preview.png"
            screenshot_path.write_bytes(b"png-bytes")
            events.insert(
                -1,
                {
                    "type": "artifact",
                    "ts": 1_700_000_004.0,
                    "run_id": run_id,
                    "kind": "preview_screenshot",
                    "label": "Preview screenshot",
                    "path": str(screenshot_path),
                    "size_bytes": screenshot_path.stat().st_size,
                },
            )
        if bundle:
            bundle_path = artifact_dir / f"{run_id}-result-bundle.zip"
            bundle_path.write_bytes(b"zip-bytes")
            events.insert(
                -1,
                {
                    "type": "artifact",
                    "ts": 1_700_000_004.2,
                    "run_id": run_id,
                    "kind": "result_bundle",
                    "label": "Result bundle",
                    "path": str(bundle_path),
                    "size_bytes": bundle_path.stat().st_size,
                },
            )
        if snapshot:
            snapshot_path = artifact_dir / "container-snapshot.json"
            snapshot_path.write_text('{"image_ref":"benchb0t:test"}', encoding="utf-8")
            events.insert(
                -1,
                {
                    "type": "artifact",
                    "ts": 1_700_000_004.4,
                    "run_id": run_id,
                    "kind": "container_snapshot",
                    "label": "Container snapshot",
                    "path": str(snapshot_path),
                    "image_ref": "benchb0t:test",
                    "size_bytes": snapshot_path.stat().st_size,
                },
            )

        log_path = Path(result["log_path"])
        log_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )
        return result

    try:
        yield {
            "app": dashboard_mod.app,
            "runs_dir": runs_dir,
            "project_dir": project_dir,
            "store": store,
            "write_run": write_run,
        }
    finally:
        dashboard_mod.state.project_dir = original["project_dir"]
        dashboard_mod.state.runs_dir = original["runs_dir"]
        dashboard_mod.state.creds_file = original["creds_file"]
        dashboard_mod.state.store = original["store"]
        dashboard_mod.state.loaded_config = original["loaded_config"]
        dashboard_mod.state.active_procs = original["active_procs"]
        dashboard_mod.state.run_batch_started_at = original["run_batch_started_at"]
        dashboard_mod.state.runner_log = original["runner_log"]
