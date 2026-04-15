from __future__ import annotations

from pathlib import Path

from framework.dashboard_history import (
    build_history_inventory,
    find_run_log_path,
    summarize_artifacts,
)


def test_find_run_log_path_prefers_plain_agentlog(tmp_path: Path) -> None:
    plain = tmp_path / "20260413_l99-test_abc12345.agentlog"
    gz = tmp_path / "20260413_l99-test_abc12345.agentlog.gz"
    plain.write_text("{}", encoding="utf-8")
    gz.write_text("{}", encoding="utf-8")

    found = find_run_log_path(tmp_path, "abc12345")

    assert found == plain


def test_summarize_artifacts_tracks_preview_bundle_and_snapshot() -> None:
    summary = summarize_artifacts([
        {"kind": "preview_screenshot", "name": "preview.png"},
        {"kind": "result_bundle", "name": "run.zip"},
        {"kind": "container_snapshot", "name": "container-snapshot.json"},
        {"kind": "json", "name": "meta.json"},
    ])

    assert summary["counts"] == {
        "total": 4,
        "screenshots": 1,
        "bundles": 1,
        "snapshots": 1,
        "other": 1,
    }
    assert summary["preview_artifact"]["name"] == "preview.png"


def test_build_history_inventory_attaches_artifacts_and_log(tmp_path: Path) -> None:
    run = {
        "id": "abc12345",
        "level_id": "l99-test",
        "level_name": "Smoke Test",
        "model": "hermes",
        "score_total": 88.0,
        "timed_out": 0,
    }
    log_path = tmp_path / "20260413_l99-test_abc12345.agentlog"
    log_path.write_text("{}", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts" / "abc12345"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "preview.png").write_bytes(b"png")

    items = build_history_inventory([run], tmp_path)

    assert len(items) == 1
    assert items[0]["log_url"] == "/api/logs/abc12345"
    assert items[0]["artifact_counts"]["screenshots"] == 1
    assert items[0]["preview_artifact"]["name"] == "preview.png"
