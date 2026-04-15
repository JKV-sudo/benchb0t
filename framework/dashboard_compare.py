"""
framework/dashboard_compare.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Helpers for comparing two stored runs and their replay trajectories.
"""

from __future__ import annotations

from typing import Any


def _started_at(payload: dict[str, Any]) -> float:
    session = payload.get("replay", {}).get("summary", {}).get("session", {})
    return float(session.get("started_at") or 0.0)


def _artifacts_by_kind(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in payload.get("artifacts", []):
        kind = str(artifact.get("kind", "file"))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _preview_ready_offset(payload: dict[str, Any]) -> float | None:
    started_at = _started_at(payload)
    for step in payload.get("replay", {}).get("timeline", []):
        if step.get("type") == "preview_ready" and step.get("ts"):
            ts = float(step["ts"])
            return round(ts - started_at, 2) if started_at else round(ts, 2)
    return None


def _step_signature(step: dict[str, Any] | None) -> str:
    if not step:
        return ""
    return "|".join(
        [
            str(step.get("type", "")),
            str(step.get("tool", "")),
            str(step.get("label", "")),
            str(step.get("text_preview", "")),
            str(step.get("output_preview", "")),
        ]
    )


def _metric(
    metric_id: str,
    label: str,
    left: Any,
    right: Any,
    *,
    direction: str,
) -> dict[str, Any]:
    better = "equal"
    delta: float | None = None

    if direction == "higher":
        left_num = float(left or 0)
        right_num = float(right or 0)
        delta = round(left_num - right_num, 2)
        if left_num > right_num:
            better = "left"
        elif right_num > left_num:
            better = "right"
    elif direction == "lower":
        left_num = float(left or 0)
        right_num = float(right or 0)
        delta = round(left_num - right_num, 2)
        if left_num < right_num:
            better = "left"
        elif right_num < left_num:
            better = "right"
    elif direction == "false_is_better":
        left_bool = bool(left)
        right_bool = bool(right)
        if left_bool != right_bool:
            better = "right" if left_bool else "left"
    elif direction == "optional_lower":
        if left is None and right is None:
            better = "equal"
        elif left is None:
            better = "right"
        elif right is None:
            better = "left"
        else:
            left_num = float(left)
            right_num = float(right)
            delta = round(left_num - right_num, 2)
            if left_num < right_num:
                better = "left"
            elif right_num < left_num:
                better = "right"

    return {
        "id": metric_id,
        "label": label,
        "left": left,
        "right": right,
        "delta": delta,
        "direction": direction,
        "better": better,
    }


def build_compare_summary(left_payload: dict[str, Any], right_payload: dict[str, Any]) -> dict[str, Any]:
    left_run = left_payload.get("run", {})
    right_run = right_payload.get("run", {})
    left_artifacts = _artifacts_by_kind(left_payload)
    right_artifacts = _artifacts_by_kind(right_payload)

    metrics = [
        _metric(
            "score_total",
            "Score",
            float(left_run.get("score_total", 0) or 0),
            float(right_run.get("score_total", 0) or 0),
            direction="higher",
        ),
        _metric(
            "duration_s",
            "Duration",
            float(left_run.get("duration_s", 0) or 0),
            float(right_run.get("duration_s", 0) or 0),
            direction="lower",
        ),
        _metric(
            "turns",
            "Turns",
            int(left_run.get("turns", 0) or 0),
            int(right_run.get("turns", 0) or 0),
            direction="lower",
        ),
        _metric(
            "tool_calls_n",
            "Tool Calls",
            int(left_run.get("tool_calls_n", 0) or 0),
            int(right_run.get("tool_calls_n", 0) or 0),
            direction="lower",
        ),
        _metric(
            "timed_out",
            "Timeout",
            bool(left_run.get("timed_out", 0)),
            bool(right_run.get("timed_out", 0)),
            direction="false_is_better",
        ),
        _metric(
            "preview_ready_s",
            "Preview Ready",
            _preview_ready_offset(left_payload),
            _preview_ready_offset(right_payload),
            direction="optional_lower",
        ),
        _metric(
            "screenshots",
            "Screenshots",
            int(left_artifacts.get("preview_screenshot", 0)),
            int(right_artifacts.get("preview_screenshot", 0)),
            direction="higher",
        ),
        _metric(
            "bundles",
            "Bundles",
            int(left_artifacts.get("result_bundle", 0)),
            int(right_artifacts.get("result_bundle", 0)),
            direction="higher",
        ),
        _metric(
            "snapshots",
            "Snapshots",
            int(left_artifacts.get("container_snapshot", 0)),
            int(right_artifacts.get("container_snapshot", 0)),
            direction="higher",
        ),
    ]

    score_metric = metrics[0]
    winner = score_metric["better"]
    if winner == "equal":
        timeout_metric = next((metric for metric in metrics if metric["id"] == "timed_out"), None)
        if timeout_metric and timeout_metric["better"] in {"left", "right"}:
            winner = timeout_metric["better"]

    return {
        "same_level": left_run.get("level_id") == right_run.get("level_id"),
        "same_model": left_run.get("model") == right_run.get("model"),
        "level_id": left_run.get("level_id"),
        "left_run_id": left_run.get("id"),
        "right_run_id": right_run.get("id"),
        "winner": winner,
        "metrics": metrics,
    }


def build_timeline_pairs(
    left_payload: dict[str, Any],
    right_payload: dict[str, Any],
    *,
    limit: int = 40,
) -> list[dict[str, Any]]:
    left_timeline = list(left_payload.get("replay", {}).get("timeline", []))
    right_timeline = list(right_payload.get("replay", {}).get("timeline", []))
    left_started_at = _started_at(left_payload)
    right_started_at = _started_at(right_payload)

    pairs: list[dict[str, Any]] = []
    max_len = min(max(len(left_timeline), len(right_timeline)), limit)
    for index in range(max_len):
        left_step = left_timeline[index] if index < len(left_timeline) else None
        right_step = right_timeline[index] if index < len(right_timeline) else None
        left_offset = None
        right_offset = None
        if left_step and left_step.get("ts"):
            left_offset = round(float(left_step["ts"]) - left_started_at, 2) if left_started_at else None
        if right_step and right_step.get("ts"):
            right_offset = round(float(right_step["ts"]) - right_started_at, 2) if right_started_at else None

        same_type = bool(left_step and right_step and left_step.get("type") == right_step.get("type"))
        diverged = _step_signature(left_step) != _step_signature(right_step)
        pairs.append(
            {
                "index": index,
                "left": left_step,
                "right": right_step,
                "left_offset_s": left_offset,
                "right_offset_s": right_offset,
                "same_type": same_type,
                "diverged": diverged,
            }
        )
    return pairs


def build_compare_payload(left_payload: dict[str, Any], right_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "left": left_payload,
        "right": right_payload,
        "summary": build_compare_summary(left_payload, right_payload),
        "timeline_pairs": build_timeline_pairs(left_payload, right_payload),
    }
