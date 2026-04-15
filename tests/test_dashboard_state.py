from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from framework.config import FrameworkConfig, LoadedFrameworkConfig
from framework.dashboard_models import ProviderRequest, RunRequest
from framework.dashboard_state import DashboardState


def test_load_and_save_creds_roundtrip(tmp_path: Path) -> None:
    state = DashboardState(creds_file=tmp_path / ".benchb0t_creds.json")

    state.save_creds(
        {
            "base_url": "http://localhost:11434/v1",
            "model": "hermes3",
            "api_key": "secret",
            "providers": [{"base_url": "http://localhost:11434/v1"}],
            "ignored": "nope",
        }
    )

    assert state.load_creds() == {
        "base_url": "http://localhost:11434/v1",
        "model": "hermes3",
        "api_key": "secret",
        "providers": [{"base_url": "http://localhost:11434/v1"}],
    }


def test_providers_from_request_supports_multi_and_single() -> None:
    state = DashboardState()
    multi = state.providers_from_request(
        RunRequest(
            providers=[
                ProviderRequest(base_url=" http://localhost:11434/v1 ", model=" hermes3 ", api_key="x", label="Hermes"),
                ProviderRequest(base_url="", model="", api_key="", label=""),
            ]
        )
    )
    single = state.providers_from_request(
        RunRequest(base_url=" http://localhost:1234/v1 ", model=" lmstudio-model ", api_key="k")
    )

    assert multi == [
        {
            "base_url": "http://localhost:11434/v1",
            "model": "hermes3",
            "api_key": "x",
            "label": "Hermes",
        }
    ]
    assert single == [
        {
            "base_url": "http://localhost:1234/v1",
            "model": "lmstudio-model",
            "api_key": "k",
            "label": "lmstudio-model",
        }
    ]


def test_alive_procs_and_reset_run_batch() -> None:
    state = DashboardState()
    running = SimpleNamespace(pid=1001, poll=lambda: None)
    finished = SimpleNamespace(pid=1002, poll=lambda: 0)
    state.active_procs = [running, finished]
    state.runner_log.append("old line")

    alive = state.alive_procs()
    before = time.time()
    state.reset_run_batch()

    assert [proc.pid for proc in alive] == [1001]
    assert list(state.runner_log) == []
    assert state.active_procs == []
    assert state.run_batch_started_at >= before


def test_apply_runtime_config_sets_paths_and_store(tmp_path: Path) -> None:
    config = FrameworkConfig.model_validate({"framework": {"runs_dir": "custom-runs"}})
    loaded = LoadedFrameworkConfig(
        path=tmp_path / "config.yaml",
        project_dir=tmp_path,
        config=config,
    )
    state = DashboardState()

    state.apply_runtime_config(loaded)

    assert state.project_dir == tmp_path
    assert state.runs_dir == tmp_path / "custom-runs"
    assert state.creds_file == tmp_path / ".benchb0t_creds.json"
    assert state.resolve_runtime_path(Path("levels/test.yaml")) == tmp_path / "levels/test.yaml"
    assert state.store is not None

