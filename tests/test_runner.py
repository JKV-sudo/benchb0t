from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from framework.config import FrameworkConfig, LoadedFrameworkConfig
from framework.container import ContainerError
from framework.recorder import load_agentlog
import framework.runner as runner
from framework.scorer import ScoreBreakdown
from framework.store import Store


class _StubValidated:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self._payload


class _FakeAPIClient:
    def __init__(self, model: str) -> None:
        self.model = model


def _loaded_framework_config(
    tmp_path: Path,
    *,
    preview_linger_seconds: int = 60,
    default_max_turns: int = 20,
    default_timeout_s: int = 120,
) -> LoadedFrameworkConfig:
    config = FrameworkConfig.model_validate(
        {
            "framework": {
                "preview_linger_seconds": preview_linger_seconds,
                "runs_dir": "runs",
            },
            "agent": {
                "default_max_turns": default_max_turns,
                "default_timeout_s": default_timeout_s,
            },
        }
    )
    return LoadedFrameworkConfig(
        path=tmp_path / "config.yaml",
        project_dir=tmp_path,
        config=config,
    )


def _level_payload(
    *,
    preview: bool = False,
    preview_path: str = "/",
    forced_retry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "level": {
            "id": "l99-test",
            "name": "Smoke Test",
            "difficulty": 1,
            "category": "web",
        },
        "container": {
            "image": "python:3.11-slim",
            "working_dir": "/workspace",
        },
        "task": {
            "instruction": "Fix the sample app",
        },
        "tools": ["bash"],
        "modes": {
            "guided": {
                "system_prompt": "Use the guided playbook.",
            }
        },
        "evaluation": {
            "type": "exact_match",
            "efficiency_target": 1,
            "criteria": [],
        },
    }
    if preview:
        payload["preview"] = {
            "port": 3000,
            "path": preview_path,
        }
    if forced_retry is not None:
        payload["forced_retry"] = forced_retry
    return payload


def _harness_payload(*, mode: str | None = None) -> dict[str, Any]:
    harness = {
        "name": "hermes",
        "type": "openai_compat",
        "endpoint": {
            "base_url": "http://localhost:11434/v1",
            "api_key_env": "BENCHBOT_API_KEY",
        },
        "model_defaults": {
            "model": "hermes3",
            "temperature": 0.2,
            "max_tokens": 4096,
        },
        "container": {
            "cpu_limit": "2",
            "memory_limit": "4g",
            "max_parallel": 1,
        },
    }
    if mode is not None:
        harness["mode"] = mode
    return {"harness": harness}


def test_resolve_preview_linger_seconds() -> None:
    framework_cfg = FrameworkConfig.model_validate(
        {"framework": {"preview_linger_seconds": 42}}
    )

    assert runner._resolve_preview_linger_seconds(framework_cfg, {}, None) == 0
    assert (
        runner._resolve_preview_linger_seconds(
            framework_cfg,
            {"preview": {"linger_seconds": 12}},
            43123,
        )
        == 12
    )
    assert (
        runner._resolve_preview_linger_seconds(
            framework_cfg,
            {"preview": {"port": 3000}},
            43123,
        )
        == 42
    )


def test_run_level_records_result_store_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    framework_cfg = _loaded_framework_config(
        tmp_path,
        preview_linger_seconds=37,
        default_max_turns=9,
        default_timeout_s=77,
    )
    store = Store(tmp_path / "benchb0t.db").init()
    seen: dict[str, Any] = {}
    sleep_calls: list[int] = []

    class FakeAgentAPI:
        @classmethod
        def from_harness(cls, harness: dict[str, Any], defaults: dict[str, Any]) -> _FakeAPIClient:
            seen["harness"] = harness
            seen["defaults"] = defaults
            return _FakeAPIClient("api-model")

    class FakeLevelContainer:
        def __init__(self, level_cfg: dict[str, Any], framework_cfg: dict[str, Any], level_id: str) -> None:
            seen["container_init"] = {
                "level_cfg": level_cfg,
                "framework_cfg": framework_cfg,
                "level_id": level_id,
            }
            self.host_preview_port = 45678
            self.stopped = False

        def start(self) -> None:
            seen["container_started"] = True

        def stop(self) -> None:
            self.stopped = True
            seen["container_stopped"] = True

        def exec(self, command: str) -> tuple[int, str]:
            seen.setdefault("exec_calls", []).append(command)
            return 0, f"ok:{command}"

    def fake_run_agent_loop(
        *,
        api: Any,
        container: Any,
        recorder: Any,
        task_cfg: dict[str, Any],
        tools_list: list[str],
        system_prompt: str | None = None,
    ) -> bool:
        seen["task_cfg"] = task_cfg
        seen["system_prompt"] = system_prompt
        seen["tools_list"] = tools_list
        recorder.record_message("user", task_cfg["instruction"])
        cid = recorder.record_tool_call("bash", {"command": "echo ok"}, call_id="tool-1")
        recorder.record_tool_result(
            cid,
            "ok",
            exit_code=0,
            tool="bash",
            args={"command": "echo ok"},
        )
        recorder.record_message("assistant", "All set.")
        return True

    fake_breakdown = ScoreBreakdown(
        completion=0.9,
        efficiency=1.0,
        self_correction=1.0,
        path_quality=1.0,
    )

    class FakeScorer:
        def __init__(self, eval_cfg: dict[str, Any], scoring_cfg: dict[str, Any]) -> None:
            seen["scorer_init"] = {
                "eval_cfg": eval_cfg,
                "scoring_cfg": scoring_cfg,
            }

        def score(self, **_: Any) -> ScoreBreakdown:
            return fake_breakdown

    def fake_capture_preview_screenshot(*, host_port: int, preview_path: str, dest_path: Path) -> dict[str, Any]:
        dest_path.write_bytes(b"png-bytes")
        return {
            "kind": "preview_screenshot",
            "label": "Preview screenshot",
            "path": str(dest_path),
            "url": f"http://localhost:{host_port}{preview_path}",
            "size_bytes": dest_path.stat().st_size,
        }

    def fake_save_container_snapshot(
        *,
        container: Any,
        artifact_dir: Path,
        level_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        metadata_path = artifact_dir / "container-snapshot.json"
        metadata_path.write_text(
            json.dumps({"image_ref": f"benchb0t/{level_id}:{run_id}"}),
            encoding="utf-8",
        )
        return {
            "kind": "container_snapshot",
            "label": "Container snapshot",
            "path": str(metadata_path),
            "image_ref": f"benchb0t/{level_id}:{run_id}",
            "size_bytes": metadata_path.stat().st_size,
        }

    monkeypatch.setenv("BENCHBOT_MODEL", "fixture-model")
    monkeypatch.setenv("BENCHBOT_BASE_URL", "http://fixture/v1")
    monkeypatch.setattr(runner, "load_level_config", lambda _path: _StubValidated(_level_payload(preview=True, preview_path="/preview")))
    monkeypatch.setattr(runner, "load_harness_config", lambda _path: _StubValidated(_harness_payload(mode="guided")))
    monkeypatch.setattr(runner, "AgentAPI", FakeAgentAPI)
    monkeypatch.setattr(runner, "LevelContainer", FakeLevelContainer)
    monkeypatch.setattr(runner, "run_agent_loop", fake_run_agent_loop)
    monkeypatch.setattr(runner, "Scorer", FakeScorer)
    monkeypatch.setattr(runner, "capture_preview_screenshot_artifact", fake_capture_preview_screenshot)
    monkeypatch.setattr(runner, "save_container_snapshot_artifact", fake_save_container_snapshot)
    monkeypatch.setattr(runner, "llm_anomaly_summary", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_print_result", lambda result: None)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    result = runner.run_level(
        level_path=tmp_path / "level.yaml",
        harness_path=tmp_path / "harness.yaml",
        framework_cfg=framework_cfg,
        store=store,
        capture_preview_screenshot=True,
        save_result_bundle=True,
        save_container_snapshot=True,
    )

    assert result["mode"] == "guided"
    assert result["model"] == "fixture-model"
    assert result["base_url"] == "http://fixture/v1"
    assert result["host_preview_port"] == 45678
    assert result["preview_linger_seconds"] == 37
    assert sleep_calls == [37]
    assert result["turns"] == 1
    assert result["tool_calls_n"] == 1
    assert seen["task_cfg"]["max_turns"] == 9
    assert seen["task_cfg"]["timeout_s"] == 77
    assert seen["defaults"]["timeout_s"] == 77
    assert seen["defaults"]["max_tokens"] == 4096
    assert seen["system_prompt"] == "Use the guided playbook."

    artifact_kinds = [artifact["kind"] for artifact in result["artifacts"]]
    assert artifact_kinds == [
        "preview_screenshot",
        "container_snapshot",
        "anomalies",
        "result_bundle",
    ]

    saved = store.get_run_by_id(result["run_id"])
    assert saved is not None
    assert saved["model"] == "fixture-model"
    assert saved["score_total"] == pytest.approx(96.0)
    assert saved["tool_calls_n"] == 1

    log_events = load_agentlog(result["log_path"])
    event_types = [event["type"] for event in log_events]
    assert "preview_ready" in event_types
    assert event_types.count("artifact") == 3
    session_end = next(event for event in log_events if event["type"] == "session_end")
    assert session_end["preview_linger_seconds"] == 37

    bundle_path = Path(result["artifacts"][-1]["path"])
    assert bundle_path.exists()
    with zipfile.ZipFile(bundle_path) as bundle:
        assert set(bundle.namelist()) == {
            "anomalies.json",
            "container-snapshot.json",
            "preview.png",
            "result.json",
            Path(result["log_path"]).name,
        }
        bundle_result = json.loads(bundle.read("result.json").decode("utf-8"))
        assert bundle_result["run_id"] == result["run_id"]
        assert [artifact["kind"] for artifact in bundle_result["artifacts"]] == [
            "preview_screenshot",
            "container_snapshot",
            "anomalies",
        ]
        anomalies = json.loads(bundle.read("anomalies.json").decode("utf-8"))
        assert "summary" in anomalies
        assert "items" in anomalies


def test_run_level_applies_retry_penalty_after_forced_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    framework_cfg = _loaded_framework_config(tmp_path)
    loop_calls: list[int] = []
    scorer_calls: list[int] = []

    class FakeAgentAPI:
        @classmethod
        def from_harness(cls, harness: dict[str, Any], defaults: dict[str, Any]) -> _FakeAPIClient:
            return _FakeAPIClient("api-model")

    class FakeLevelContainer:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.host_preview_port = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def exec(self, command: str) -> tuple[int, str]:
            return 0, command

    def fake_run_agent_loop(
        *,
        api: Any,
        container: Any,
        recorder: Any,
        task_cfg: dict[str, Any],
        tools_list: list[str],
        system_prompt: str | None = None,
    ) -> bool:
        loop_calls.append(len(loop_calls) + 1)
        recorder.record_message("user", task_cfg["instruction"])
        recorder.record_message("assistant", f"Attempt {len(loop_calls)}")
        return True

    score_sequence = [
        ScoreBreakdown(completion=0.2, efficiency=1.0, self_correction=1.0, path_quality=1.0),
        ScoreBreakdown(completion=0.9, efficiency=1.0, self_correction=1.0, path_quality=1.0),
    ]

    class FakeScorer:
        def __init__(self, eval_cfg: dict[str, Any], scoring_cfg: dict[str, Any]) -> None:
            return None

        def score(self, **_: Any) -> ScoreBreakdown:
            scorer_calls.append(len(scorer_calls) + 1)
            return score_sequence[len(scorer_calls) - 1]

    monkeypatch.setattr(
        runner,
        "load_level_config",
        lambda _path: _StubValidated(
            _level_payload(
                forced_retry={
                    "enabled": True,
                    "max_retries": 2,
                    "penalty_per_retry": 7.5,
                    "completion_threshold": 0.8,
                }
            )
        ),
    )
    monkeypatch.setattr(runner, "load_harness_config", lambda _path: _StubValidated(_harness_payload()))
    monkeypatch.setattr(runner, "AgentAPI", FakeAgentAPI)
    monkeypatch.setattr(runner, "LevelContainer", FakeLevelContainer)
    monkeypatch.setattr(runner, "run_agent_loop", fake_run_agent_loop)
    monkeypatch.setattr(runner, "Scorer", FakeScorer)
    monkeypatch.setattr(runner, "_print_result", lambda result: None)

    result = runner.run_level(
        level_path=tmp_path / "level.yaml",
        harness_path=tmp_path / "harness.yaml",
        framework_cfg=framework_cfg,
    )

    assert loop_calls == [1, 2]
    assert scorer_calls == [1, 2]
    assert result["timed_out"] is False
    assert result["score"]["penalties"]["retry"] == pytest.approx(7.5)
    assert result["score"]["total"] == pytest.approx(88.5)


def test_run_level_returns_zero_score_on_container_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    framework_cfg = _loaded_framework_config(tmp_path)
    holder: dict[str, Any] = {}

    class FakeAgentAPI:
        @classmethod
        def from_harness(cls, harness: dict[str, Any], defaults: dict[str, Any]) -> _FakeAPIClient:
            return _FakeAPIClient("api-model")

    class FakeLevelContainer:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.host_preview_port = None
            self.stopped = False
            holder["container"] = self

        def start(self) -> None:
            raise ContainerError("docker unavailable")

        def stop(self) -> None:
            self.stopped = True

        def exec(self, command: str) -> tuple[int, str]:
            return 0, command

    monkeypatch.setattr(runner, "load_level_config", lambda _path: _StubValidated(_level_payload()))
    monkeypatch.setattr(runner, "load_harness_config", lambda _path: _StubValidated(_harness_payload()))
    monkeypatch.setattr(runner, "AgentAPI", FakeAgentAPI)
    monkeypatch.setattr(runner, "LevelContainer", FakeLevelContainer)
    monkeypatch.setattr(runner, "_print_result", lambda result: None)

    result = runner.run_level(
        level_path=tmp_path / "level.yaml",
        harness_path=tmp_path / "harness.yaml",
        framework_cfg=framework_cfg,
    )

    assert result["score"]["total"] == 0
    assert result["score"]["error"] == "docker unavailable"
    assert result["host_preview_port"] is None
    assert holder["container"].stopped is True

    log_events = load_agentlog(result["log_path"])
    session_end = next(event for event in log_events if event["type"] == "session_end")
    assert session_end["score"]["error"] == "docker unavailable"
    assert session_end["score"]["total"] == 0
