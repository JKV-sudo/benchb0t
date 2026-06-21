from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from framework.anomalies import (
    detect_anomalies,
    llm_anomaly_summary,
    write_anomalies_report,
)


def _call(tool: str, args: dict[str, Any] | None = None, *, exit_code: int = 0, output: str = "", call_id: str = "c1") -> dict[str, Any]:
    return {
        "call_id": call_id,
        "tool": tool,
        "args": args or {},
        "exit_code": exit_code,
        "output": output,
    }


# ── detect_anomalies ──────────────────────────────────────────────────────────


def test_clean_run_has_no_anomalies() -> None:
    calls = [_call("bash", {"cmd": "ls"}, exit_code=0, output="ok", call_id="c1")]
    report = detect_anomalies(tool_calls=calls, timed_out=False)
    assert report["summary"]["count"] == 0
    assert report["summary"]["severity"] == "none"
    assert report["items"] == []
    assert report["llm_summary"] is None


def test_failed_tool_is_flagged_high() -> None:
    calls = [_call("bash", {"cmd": "false"}, exit_code=1, output="boom", call_id="c1")]
    report = detect_anomalies(tool_calls=calls, timed_out=False)
    kinds = [it["kind"] for it in report["items"]]
    assert "failed_tool" in kinds
    assert report["summary"]["severity"] in {"medium", "high"}


def test_timeout_is_flagged() -> None:
    report = detect_anomalies(tool_calls=[], timed_out=True, duration_s=120.0)
    assert any(it["kind"] == "timeout" for it in report["items"])
    assert report["summary"]["severity"] == "high"


def test_forced_retry_from_score_penalties() -> None:
    score = {"penalties": {"retry": 20.0}}
    report = detect_anomalies(tool_calls=[], timed_out=False, score=score)
    retry_items = [it for it in report["items"] if it["kind"] == "forced_retry"]
    assert len(retry_items) == 1
    assert "20.0" in retry_items[0]["detail"]


def test_retry_loop_detected() -> None:
    calls = [
        _call("bash", {"cmd": "npm test"}, call_id="c1"),
        _call("bash", {"cmd": "npm test"}, call_id="c2"),
        _call("bash", {"cmd": "npm test"}, call_id="c3"),
    ]
    report = detect_anomalies(tool_calls=calls, timed_out=False)
    loops = [it for it in report["items"] if it["kind"] == "retry_loop"]
    assert len(loops) == 1
    assert "3 identical" in loops[0]["detail"]


def test_two_identical_calls_do_not_trigger_loop() -> None:
    calls = [
        _call("bash", {"cmd": "ls"}, call_id="c1"),
        _call("bash", {"cmd": "ls"}, call_id="c2"),
    ]
    report = detect_anomalies(tool_calls=calls, timed_out=False)
    assert not any(it["kind"] == "retry_loop" for it in report["items"])


def test_self_correction_write_then_patch() -> None:
    calls = [
        _call("write_file", {"path": "app.js"}, call_id="c1"),
        _call("patch_file", {"path": "app.js"}, call_id="c2"),
    ]
    report = detect_anomalies(tool_calls=calls, timed_out=False)
    corrections = [it for it in report["items"] if it["kind"] == "self_correction"]
    assert len(corrections) == 1
    assert "app.js" in corrections[0]["detail"]


def test_long_output_flagged() -> None:
    big = "x" * 5000
    calls = [_call("bash", {"cmd": "cat big"}, output=big, call_id="c1")]
    report = detect_anomalies(tool_calls=calls, timed_out=False)
    longs = [it for it in report["items"] if it["kind"] == "long_output"]
    assert len(longs) == 1
    assert "5000" in longs[0]["detail"]


def test_duration_spike_from_events() -> None:
    # Three fast calls (1s each) + one slow call (60s). Median = 1s, spike = 60s.
    base = 1000.0
    events = [
        {"type": "tool_call", "call_id": "c1", "ts": base},
        {"type": "tool_result", "call_id": "c1", "ts": base + 1},
        {"type": "tool_call", "call_id": "c2", "ts": base + 2},
        {"type": "tool_result", "call_id": "c2", "ts": base + 3},
        {"type": "tool_call", "call_id": "c3", "ts": base + 4},
        {"type": "tool_result", "call_id": "c3", "ts": base + 5},
        {"type": "tool_call", "call_id": "c4", "ts": base + 6},
        {"type": "tool_result", "call_id": "c4", "ts": base + 66},
    ]
    calls = [
        _call("bash", call_id="c1"),
        _call("bash", call_id="c2"),
        _call("bash", call_id="c3"),
        _call("bash", call_id="c4"),
    ]
    report = detect_anomalies(tool_calls=calls, events=events, timed_out=False)
    spikes = [it for it in report["items"] if it["kind"] == "duration_spike"]
    assert len(spikes) == 1
    assert spikes[0]["call_id"] == "c4"


def test_no_duration_spike_below_floor() -> None:
    # All calls fast — even a relative spike stays under the 15s floor.
    base = 1000.0
    events = [
        {"type": "tool_call", "call_id": "c1", "ts": base},
        {"type": "tool_result", "call_id": "c1", "ts": base + 0.1},
        {"type": "tool_call", "call_id": "c2", "ts": base + 1},
        {"type": "tool_result", "call_id": "c2", "ts": base + 5},  # 4s, 40x median but <15s floor
    ]
    calls = [_call("bash", call_id="c1"), _call("bash", call_id="c2")]
    report = detect_anomalies(tool_calls=calls, events=events, timed_out=False)
    assert not any(it["kind"] == "duration_spike" for it in report["items"])


def test_severity_none_low_medium_high_progression() -> None:
    none = detect_anomalies(tool_calls=[_call("bash", call_id="c1", exit_code=0)], timed_out=False)
    assert none["summary"]["severity"] == "none"

    low = detect_anomalies(
        tool_calls=[_call("write_file", {"path": "f"}, call_id="c1"), _call("patch_file", {"path": "f"}, call_id="c2")],
        timed_out=False,
    )
    assert low["summary"]["severity"] == "low"

    medium = detect_anomalies(
        tool_calls=[
            _call("bash", {"cmd": "x"}, call_id="c1"),
            _call("bash", {"cmd": "x"}, call_id="c2"),
            _call("bash", {"cmd": "x"}, call_id="c3"),
        ],
        timed_out=False,
    )
    assert medium["summary"]["severity"] == "medium"

    high = detect_anomalies(tool_calls=[], timed_out=True, duration_s=120.0)
    assert high["summary"]["severity"] == "high"


def test_items_capped_at_max() -> None:
    # Generate many failed calls to exceed the report cap.
    calls = [
        _call("bash", {"cmd": str(i)}, exit_code=1, output="err", call_id=f"c{i}")
        for i in range(200)
    ]
    report = detect_anomalies(tool_calls=calls, timed_out=False)
    assert len(report["items"]) <= 50


# ── write_anomalies_report ────────────────────────────────────────────────────


def test_write_anomalies_report_creates_json(tmp_path: Path) -> None:
    report = {"summary": {"count": 1, "severity": "low"}, "items": [{"kind": "timeout"}], "llm_summary": None}
    dest = tmp_path / "sub" / "anomalies.json"
    meta = write_anomalies_report(report=report, dest_path=dest)

    assert dest.exists()
    assert meta["kind"] == "anomalies"
    assert meta["size_bytes"] > 0
    loaded = json.loads(dest.read_text(encoding="utf-8"))
    assert loaded["summary"]["count"] == 1


def test_write_anomalies_report_handles_disk_error(tmp_path: Path) -> None:
    dest = tmp_path / "anomalies.json"
    dest.write_text("blocker", encoding="utf-8")
    dest.chmod(0o444)  # read-only
    report = {"summary": {"count": 0, "severity": "none"}, "items": [], "llm_summary": None}
    meta = write_anomalies_report(report=report, dest_path=dest)
    # Should not raise; returns a metadata dict regardless.
    assert meta["kind"] == "anomalies"


# ── llm_anomaly_summary ───────────────────────────────────────────────────────


class _FakeAPI:
    """Minimal AgentAPI stand-in for judge testing."""

    def __init__(self, response: dict[str, Any] | None = None, *, raises: bool = False) -> None:
        self._response = response
        self._raises = raises
        self.received: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.received.append({"messages": messages, **kwargs})
        if self._raises:
            raise RuntimeError("boom")
        return self._response or {"choices": [{"message": {"content": "All good."}}]}


def test_llm_summary_returns_text() -> None:
    api = _FakeAPI(response={"choices": [{"message": {"content": "Clean run."}}]})
    report = detect_anomalies(tool_calls=[], timed_out=False)
    result = {"level_id": "l1", "score": {"total": 90.0}, "duration_s": 5.0}
    text = llm_anomaly_summary(api, report=report, result=result)  # type: ignore[arg-type]
    assert text == "Clean run."


def test_llm_summary_returns_none_on_failure() -> None:
    api = _FakeAPI(raises=True)
    report = detect_anomalies(tool_calls=[], timed_out=False)
    result = {"level_id": "l1", "score": {"total": 90.0}}
    assert llm_anomaly_summary(api, report=report, result=result) is None  # type: ignore[arg-type]


def test_llm_summary_returns_none_on_empty_response() -> None:
    api = _FakeAPI(response={"choices": [{"message": {"content": ""}}]})
    report = detect_anomalies(tool_calls=[], timed_out=False)
    result = {"level_id": "l1"}
    assert llm_anomaly_summary(api, report=report, result=result) is None  # type: ignore[arg-type]


def test_llm_summary_prompt_includes_key_metrics() -> None:
    api = _FakeAPI()
    report = detect_anomalies(
        tool_calls=[_call("bash", exit_code=1, call_id="c1")],
        timed_out=False,
    )
    result = {
        "level_id": "l4-express-api",
        "model": "test-model",
        "duration_s": 42.0,
        "turns": 7,
        "tool_calls_n": 1,
        "timed_out": False,
        "score": {"total": 33.0},
    }
    llm_anomaly_summary(api, report=report, result=result)  # type: ignore[arg-type]
    user_msg = api.received[0]["messages"][1]["content"]
    assert "l4-express-api" in user_msg
    assert "test-model" in user_msg
    assert "failed_tool" in user_msg
