"""
framework/dashboard_history.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Helpers for shaping saved run history and artifact inventory for the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from framework.dashboard_replay import build_artifact_records


def find_run_log_path(runs_dir: Path, run_id: str) -> Path | None:
    """Locate the stored agentlog for a run, preferring plain text over gzip."""
    patterns = [f"*_{run_id}.agentlog", f"*_{run_id}.agentlog.gz"]
    for pattern in patterns:
        for candidate in sorted(Path(runs_dir).glob(pattern)):
            return candidate
    return None


def summarize_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "total": len(artifacts),
        "screenshots": 0,
        "bundles": 0,
        "snapshots": 0,
        "other": 0,
    }
    preview_artifact: dict[str, Any] | None = None

    for artifact in artifacts:
        kind = artifact.get("kind", "")
        if kind == "preview_screenshot":
            counts["screenshots"] += 1
            preview_artifact = preview_artifact or artifact
        elif kind == "result_bundle":
            counts["bundles"] += 1
        elif kind == "container_snapshot":
            counts["snapshots"] += 1
        else:
            counts["other"] += 1

    return {
        "counts": counts,
        "preview_artifact": preview_artifact,
    }


def build_history_run_entry(run: dict[str, Any], runs_dir: Path) -> dict[str, Any]:
    """Attach artifact and log metadata to one stored run row."""
    run_id = str(run.get("id") or run.get("run_id") or "")
    artifacts = build_artifact_records(runs_dir, run_id)
    artifact_summary = summarize_artifacts(artifacts)
    log_path = find_run_log_path(runs_dir, run_id)

    return {
        **dict(run),
        "artifacts": artifacts,
        "artifact_counts": artifact_summary["counts"],
        "preview_artifact": artifact_summary["preview_artifact"],
        "log_url": f"/api/logs/{run_id}" if log_path else "",
        "log_name": log_path.name if log_path else "",
        "replay_url": f"/api/replays/{run_id}",
    }


def build_history_inventory(runs: list[dict[str, Any]], runs_dir: Path) -> list[dict[str, Any]]:
    """Build UI-ready run history entries newest-first."""
    return [build_history_run_entry(run, runs_dir) for run in runs]
