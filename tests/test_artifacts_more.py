from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

import framework.artifacts as artifacts


class _FakeHTTPResponse:
    def __init__(self, status: int, payload: bytes = b"") -> None:
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_wait_for_preview_accepts_http_error_under_500(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_http_error(req, timeout=2):
        raise urllib.error.HTTPError(req.full_url, 404, "missing", hdrs=None, fp=None)

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", raise_http_error)

    assert artifacts.wait_for_preview("http://localhost:3000") is True


def test_capture_preview_screenshot_skips_without_npx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(artifacts.shutil, "which", lambda name: None)

    result = artifacts.capture_preview_screenshot(
        host_port=3000,
        preview_path="/",
        dest_path=tmp_path / "preview.png",
    )

    assert result is None


def test_capture_preview_screenshot_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dest_path = tmp_path / "preview.png"

    monkeypatch.setattr(artifacts.shutil, "which", lambda name: "/usr/bin/npx")
    monkeypatch.setattr(artifacts, "wait_for_preview", lambda url, timeout_s=12.0: True)

    def fake_run(cmd, check, capture_output, text, timeout):
        dest_path.write_bytes(b"png-bytes")
        return type("Proc", (), {"stdout": "saved", "stderr": ""})()

    monkeypatch.setattr(artifacts.subprocess, "run", fake_run)

    result = artifacts.capture_preview_screenshot(
        host_port=49312,
        preview_path="/app",
        dest_path=dest_path,
    )

    assert result is not None
    assert result["kind"] == "preview_screenshot"
    assert result["url"] == "http://localhost:49312/app"
    assert Path(result["path"]).exists()


def test_save_container_snapshot_success_and_failure(tmp_path: Path) -> None:
    class GoodContainer:
        def snapshot(self) -> str:
            return "benchb0t:snapshot"

    result = artifacts.save_container_snapshot(
        container=GoodContainer(),
        artifact_dir=tmp_path,
        level_id="l99-test",
        run_id="abc12345",
    )

    assert result is not None
    metadata = json.loads((tmp_path / "container-snapshot.json").read_text(encoding="utf-8"))
    assert metadata["image_ref"] == "benchb0t:snapshot"

    class BadContainer:
        def snapshot(self) -> str:
            raise RuntimeError("docker offline")

    assert (
        artifacts.save_container_snapshot(
            container=BadContainer(),
            artifact_dir=tmp_path,
            level_id="l99-test",
            run_id="abc12345",
        )
        is None
    )

