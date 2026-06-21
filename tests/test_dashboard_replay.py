from __future__ import annotations

from pathlib import Path

from framework.dashboard_replay import (
    build_artifact_records,
    build_replay_payload,
    build_replay_run_record,
    build_run_replay,
)


def test_build_run_replay_summarizes_timeline() -> None:
    events = [
        {
            "type": "session_start",
            "ts": 100.0,
            "run_id": "abc12345",
            "level_id": "l99-test",
            "level_name": "Smoke Test",
            "model": "hermes",
        },
        {
            "type": "preview_ready",
            "ts": 101.0,
            "host_preview_port": 49312,
            "preview_path": "/",
        },
        {
            "type": "tool_call",
            "ts": 102.0,
            "call_id": "call-1",
            "tool": "bash",
            "args": {"command": "echo hi"},
        },
        {
            "type": "tool_result",
            "ts": 103.0,
            "call_id": "call-1",
            "exit_code": 0,
            "output": "hi\n",
        },
        {
            "type": "message",
            "ts": 103.5,
            "role": "user",
            "content": "Please verify the output.",
        },
        {
            "type": "message",
            "ts": 104.0,
            "role": "assistant",
            "content": "Done.",
        },
        {
            "type": "artifact",
            "ts": 104.5,
            "kind": "preview_screenshot",
            "path": "/tmp/runs/artifacts/abc12345/preview.png",
            "label": "Preview screenshot",
        },
        {
            "type": "session_end",
            "ts": 105.0,
            "duration_s": 5.0,
            "timed_out": False,
            "score": {"total": 100.0},
            "preview_linger_seconds": 60,
            "preview_expires_at": 165.0,
        },
    ]

    replay = build_run_replay(events)

    assert replay["summary"]["event_count"] == 8
    assert replay["summary"]["turn_count"] == 1
    assert replay["summary"]["tool_call_count"] == 1
    assert replay["summary"]["has_preview"] is True
    assert replay["summary"]["preview"]["host_preview_port"] == 49312
    assert replay["summary"]["session"]["score"]["total"] == 100.0

    timeline_types = [item["type"] for item in replay["timeline"]]
    assert timeline_types == [
        "session_start",
        "preview_ready",
        "tool_call",
        "tool_result",
        "user_message",
        "assistant_message",
        "artifact",
        "session_end",
    ]


def test_build_replay_run_record_falls_back_to_log_metadata(tmp_path: Path) -> None:
    events = [
        {
            "type": "session_start",
            "ts": 100.0,
            "run_id": "abc12345",
            "level_id": "l99-test",
            "level_name": "Smoke Test",
            "model": "hermes",
        },
        {
            "type": "message",
            "ts": 101.0,
            "role": "assistant",
            "content": "Done.",
        },
        {
            "type": "session_end",
            "ts": 102.0,
            "duration_s": 2.0,
            "timed_out": False,
            "score": {"total": 87.0},
        },
    ]

    record = build_replay_run_record("abc12345", events, tmp_path / "run.agentlog")

    assert record["id"] == "abc12345"
    assert record["level_id"] == "l99-test"
    assert record["model"] == "hermes"
    assert record["score_total"] == 87.0
    assert record["stars"] == 4
    assert record["turns"] == 1
    assert record["tool_calls_n"] == 0


def test_build_replay_payload_uses_db_run_when_present(tmp_path: Path) -> None:
    events = [
        {
            "type": "session_start",
            "ts": 100.0,
            "run_id": "abc12345",
            "level_id": "l99-test",
            "level_name": "Smoke Test",
            "model": "hermes",
        },
        {
            "type": "session_end",
            "ts": 102.0,
            "duration_s": 2.0,
            "timed_out": False,
            "score": {"total": 87.0},
        },
    ]
    db_run = {
        "id": "abc12345",
        "level_id": "l99-test",
        "level_name": "Smoke Test",
        "model": "db-model",
        "score_total": 91.0,
        "stars": 4,
        "timed_out": 0,
    }

    payload = build_replay_payload(
        "abc12345",
        events,
        tmp_path / "run.agentlog",
        runs_dir=tmp_path,
        db_run=db_run,
    )

    assert payload["run"]["model"] == "db-model"
    assert payload["run"]["score_total"] == 91.0
    assert payload["replay"]["summary"]["event_count"] == 2
    assert payload["artifacts"] == []
    assert payload["log_found"] is True


def test_build_artifact_records_indexes_saved_files(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "abc12345"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "preview.png").write_bytes(b"png-bytes")
    (artifact_dir / "abc12345-result-bundle.zip").write_bytes(b"zip-bytes")
    (artifact_dir / "container-snapshot.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "anomalies.json").write_text(
        '{"summary": {"count": 2, "severity": "medium"}, "items": [{"kind": "timeout", "severity": "high"}], "llm_summary": "rough run"}',
        encoding="utf-8",
    )

    records = build_artifact_records(tmp_path, "abc12345")

    kinds = [record["kind"] for record in records]
    assert kinds == [
        "result_bundle",
        "anomalies",
        "container_snapshot",
        "preview_screenshot",
    ]
    assert records[-1]["is_image"] is True
    assert records[-1]["url"] == "/api/artifacts/abc12345/preview.png"

    # The anomalies record should carry the parsed report inline.
    anomaly = records[kinds.index("anomalies")]
    assert anomaly["anomaly_summary"] == {"count": 2, "severity": "medium"}
    assert len(anomaly["anomaly_items"]) == 1
    assert anomaly["anomaly_llm_summary"] == "rough run"


def test_anomaly_record_handles_corrupt_json(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "deadbeef"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "anomalies.json").write_text("{not valid json", encoding="utf-8")

    records = build_artifact_records(tmp_path, "deadbeef")

    assert len(records) == 1
    assert records[0]["kind"] == "anomalies"
    # No inline summary keys when the file fails to parse.
    assert "anomaly_summary" not in records[0]
