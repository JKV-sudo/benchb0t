from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import framework.container as container_mod
from framework.container import ContainerError, LevelContainer


class _FakeContainer:
    def __init__(
        self,
        *,
        name: str = "benchb0t-fake",
        statuses: list[str] | None = None,
        ports: dict[str, list[dict[str, str]]] | None = None,
        exec_responses: dict[str, tuple[int, bytes]] | None = None,
        put_archive_ok: bool = True,
        logs_text: bytes = b"",
    ) -> None:
        self.name = name
        self._statuses = list(statuses or ["running"])
        self.status = self._statuses[0]
        self.ports = ports or {}
        self._exec_responses = exec_responses or {}
        self._put_archive_ok = put_archive_ok
        self._logs_text = logs_text
        self.stopped = False
        self.removed = False
        self.commit_calls: list[tuple[str, str]] = []

    def reload(self) -> None:
        if len(self._statuses) > 1:
            self._statuses.pop(0)
        self.status = self._statuses[0]

    def exec_run(self, cmd: list[str], workdir: str, demux: bool = False) -> tuple[int, bytes]:
        command = cmd[-1]
        return self._exec_responses.get(command, (0, b"ok"))

    def put_archive(self, dest_dir: str, tarstream) -> bool:
        return self._put_archive_ok

    def commit(self, repository: str, tag: str) -> None:
        self.commit_calls.append((repository, tag))

    def stop(self, timeout: int = 10) -> None:
        self.stopped = True

    def remove(self, force: bool = True) -> None:
        self.removed = True

    def logs(self) -> bytes:
        return self._logs_text


class _FakeImages:
    def __init__(self, *, present: set[str] | None = None) -> None:
        self.present = set(present or set())
        self.pulled: list[str] = []

    def get(self, image: str) -> str:
        if image not in self.present:
            raise container_mod.ImageNotFound("missing")
        return image

    def pull(self, image: str) -> None:
        self.pulled.append(image)
        self.present.add(image)


class _FakeContainers:
    def __init__(self, container: _FakeContainer) -> None:
        self._container = container
        self.run_kwargs: dict[str, Any] | None = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return self._container


class _FakeClient:
    def __init__(self, container: _FakeContainer, *, present_images: set[str] | None = None) -> None:
        self.images = _FakeImages(present=present_images)
        self.containers = _FakeContainers(container)

    def ping(self) -> None:
        return None


def test_resolve_env_parse_volumes_and_cpu_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCHBOT_SECRET", "abc123")
    container = LevelContainer(
        level_cfg={"image": "python:3.11-slim", "cpu_limit": "1.5"},
        framework_cfg={"default_cpu_limit": "2"},
        level_id="l99-test",
    )

    assert container._cpu_quota() == 150000
    assert container._resolve_env({"API_KEY": "${BENCHBOT_SECRET}", "MODE": "dev"}) == {
        "API_KEY": "abc123",
        "MODE": "dev",
    }
    parsed = container._parse_volumes(["./data:/workspace/data:ro", "/tmp/cache"])
    assert parsed["/Users/jacobs/Documents/Claude/Projects/benchb0t/data"]["bind"] == "/workspace/data"
    assert parsed["/tmp/cache"]["bind"] == "/tmp/cache"


def test_start_publishes_preview_port_and_runs_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_container = _FakeContainer(
        statuses=["created", "running"],
        ports={"3000/tcp": [{"HostPort": "49312"}]},
    )
    fake_client = _FakeClient(fake_container, present_images={"python:3.11-slim"})
    installed: list[dict[str, list[str]]] = []

    monkeypatch.setattr(LevelContainer, "_make_client", staticmethod(lambda: fake_client))
    monkeypatch.setattr(
        LevelContainer,
        "_install_packages",
        lambda self, packages: installed.append(packages),
    )

    container = LevelContainer(
        level_cfg={
            "image": "python:3.11-slim",
            "working_dir": "/workspace",
            "preview_port": 3000,
            "packages": {"pip": ["pytest"]},
            "setup_script": "echo ready",
        },
        framework_cfg={"pull_policy": "if_not_present", "default_memory_limit": "4g"},
        level_id="l99-test",
    )

    container.start()

    assert container.host_preview_port == 49312
    assert installed == [{"pip": ["pytest"]}]
    assert fake_client.containers.run_kwargs is not None
    assert fake_client.containers.run_kwargs["ports"] == {"3000/tcp": None}
    assert "network_mode" not in fake_client.containers.run_kwargs


def test_start_raises_when_setup_script_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_container = _FakeContainer(
        statuses=["running"],
        exec_responses={"echo broken": (1, b"setup failed")},
    )
    fake_client = _FakeClient(fake_container, present_images={"python:3.11-slim"})

    monkeypatch.setattr(LevelContainer, "_make_client", staticmethod(lambda: fake_client))
    monkeypatch.setattr(LevelContainer, "_install_packages", lambda self, packages: None)

    container = LevelContainer(
        level_cfg={
            "image": "python:3.11-slim",
            "working_dir": "/workspace",
            "setup_script": "echo broken",
        },
        framework_cfg={"pull_policy": "if_not_present"},
        level_id="l99-test",
    )

    with pytest.raises(ContainerError, match="Setup script failed"):
        container.start()


def test_wait_for_running_raises_with_logs() -> None:
    container = LevelContainer(
        level_cfg={"image": "python:3.11-slim"},
        framework_cfg={"pull_policy": "never"},
        level_id="l99-test",
    )
    container._container = _FakeContainer(statuses=["exited"], logs_text=b"boom")

    with pytest.raises(ContainerError, match="Container exited immediately"):
        container._wait_for_running()


def test_write_file_raises_when_put_archive_fails() -> None:
    container = LevelContainer(
        level_cfg={"image": "python:3.11-slim"},
        framework_cfg={},
        level_id="l99-test",
    )
    container._container = _FakeContainer(put_archive_ok=False)

    with pytest.raises(ContainerError, match="put_archive returned False"):
        container.write_file("/workspace/app.py", "print('hi')")


def test_snapshot_and_stop_cleanup() -> None:
    container = LevelContainer(
        level_cfg={"image": "python:3.11-slim"},
        framework_cfg={},
        level_id="l99-test",
    )
    fake_container = _FakeContainer()
    container._container = fake_container

    image_ref = container.snapshot()
    container.stop()

    assert image_ref.startswith("benchb0t:benchb0t-snapshot-l99-test-")
    assert fake_container.commit_calls[0][0] == "benchb0t"
    assert fake_container.stopped is True
    assert fake_container.removed is True


def test_ensure_image_pull_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeImageNotFound(Exception):
        pass

    monkeypatch.setattr(container_mod, "ImageNotFound", FakeImageNotFound)
    container = LevelContainer(
        level_cfg={"image": "python:3.11-slim"},
        framework_cfg={},
        level_id="l99-test",
    )
    container._client = _FakeClient(_FakeContainer(), present_images={"present:latest"})

    container._ensure_image("present:latest", "if_not_present")
    assert container._client.images.pulled == []

    container._ensure_image("missing:latest", "if_not_present")
    container._ensure_image("always:latest", "always")
    container._ensure_image("never:latest", "never")

    assert container._client.images.pulled == ["missing:latest", "always:latest"]

