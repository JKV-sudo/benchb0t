from __future__ import annotations

from framework.dashboard_compare import (
    build_compare_payload,
    build_compare_summary,
    build_timeline_pairs,
)


def _payload(
    *,
    run_id: str,
    model: str,
    score_total: float,
    duration_s: float,
    turns: int,
    tool_calls_n: int,
    timed_out: bool,
    preview_ready_offset: float | None,
    artifacts: list[dict],
) -> dict:
    timeline = [
        {
            "type": "session_start",
            "ts": 100.0,
            "label": "Session started",
            "meta": {
                "run_id": run_id,
                "level_id": "l99-test",
                "level_name": "Smoke Test",
                "model": model,
                "started_at": 100.0,
            },
        },
    ]
    if preview_ready_offset is not None:
        timeline.append(
            {
                "type": "preview_ready",
                "ts": 100.0 + preview_ready_offset,
                "label": "Preview ready",
                "meta": {"host_preview_port": 49312, "path": "/"},
            }
        )
    timeline.extend(
        [
            {
                "type": "assistant_message",
                "ts": 103.0,
                "label": "Assistant turn 1",
                "text_preview": "Done.",
            },
            {
                "type": "session_end",
                "ts": 100.0 + duration_s,
                "label": "Session ended",
                "score_total": score_total,
                "timed_out": timed_out,
            },
        ]
    )
    return {
        "run": {
            "id": run_id,
            "level_id": "l99-test",
            "level_name": "Smoke Test",
            "model": model,
            "score_total": score_total,
            "duration_s": duration_s,
            "turns": turns,
            "tool_calls_n": tool_calls_n,
            "timed_out": int(timed_out),
            "stars": 4,
        },
        "artifacts": artifacts,
        "replay": {
            "summary": {
                "session": {
                    "started_at": 100.0,
                    "level_id": "l99-test",
                    "level_name": "Smoke Test",
                    "model": model,
                }
            },
            "timeline": timeline,
        },
    }


def test_build_compare_summary_picks_winner_and_metrics() -> None:
    left = _payload(
        run_id="left1234",
        model="hermes",
        score_total=88.0,
        duration_s=18.0,
        turns=5,
        tool_calls_n=9,
        timed_out=False,
        preview_ready_offset=4.5,
        artifacts=[{"kind": "preview_screenshot"}],
    )
    right = _payload(
        run_id="right567",
        model="gpt-4.1",
        score_total=93.0,
        duration_s=24.0,
        turns=7,
        tool_calls_n=11,
        timed_out=False,
        preview_ready_offset=2.0,
        artifacts=[
            {"kind": "preview_screenshot"},
            {"kind": "result_bundle"},
        ],
    )

    summary = build_compare_summary(left, right)

    assert summary["same_level"] is True
    assert summary["winner"] == "right"
    metric_ids = {metric["id"]: metric for metric in summary["metrics"]}
    assert metric_ids["score_total"]["better"] == "right"
    assert metric_ids["duration_s"]["better"] == "left"
    assert metric_ids["preview_ready_s"]["better"] == "right"
    assert metric_ids["bundles"]["better"] == "right"


def test_build_timeline_pairs_aligns_steps_by_index() -> None:
    left = _payload(
        run_id="left1234",
        model="hermes",
        score_total=88.0,
        duration_s=18.0,
        turns=5,
        tool_calls_n=9,
        timed_out=False,
        preview_ready_offset=4.5,
        artifacts=[],
    )
    right = _payload(
        run_id="right567",
        model="gpt-4.1",
        score_total=93.0,
        duration_s=24.0,
        turns=7,
        tool_calls_n=11,
        timed_out=False,
        preview_ready_offset=None,
        artifacts=[],
    )

    pairs = build_timeline_pairs(left, right)

    assert pairs[0]["same_type"] is True
    assert pairs[1]["left"]["type"] == "preview_ready"
    assert pairs[1]["right"]["type"] == "assistant_message"
    assert pairs[1]["diverged"] is True


def test_build_compare_payload_returns_summary_and_pairs() -> None:
    left = _payload(
        run_id="left1234",
        model="hermes",
        score_total=88.0,
        duration_s=18.0,
        turns=5,
        tool_calls_n=9,
        timed_out=False,
        preview_ready_offset=4.5,
        artifacts=[],
    )
    right = _payload(
        run_id="right567",
        model="gpt-4.1",
        score_total=93.0,
        duration_s=24.0,
        turns=7,
        tool_calls_n=11,
        timed_out=False,
        preview_ready_offset=2.0,
        artifacts=[],
    )

    payload = build_compare_payload(left, right)

    assert payload["summary"]["winner"] == "right"
    assert payload["left"]["run"]["id"] == "left1234"
    assert payload["right"]["run"]["id"] == "right567"
    assert len(payload["timeline_pairs"]) >= 3
