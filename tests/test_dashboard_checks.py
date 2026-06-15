from __future__ import annotations

import json
import urllib.error

import pytest

import framework.dashboard_checks as checks


class _FakeHTTPResponse:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status = status
        self._payload = payload or {}

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_normalize_url_adds_scheme_and_v1() -> None:
    assert checks.normalize_url("localhost:11434") == "http://localhost:11434/v1"
    assert checks.normalize_url("http://localhost:11434/") == "http://localhost:11434/v1"
    assert checks.normalize_url("https://api.openai.com/v1") == "https://api.openai.com/v1"


def test_check_api_success_and_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        checks.urllib.request,
        "urlopen",
        lambda req, timeout=3: _FakeHTTPResponse(200),
    )
    assert checks.check_api("localhost:11434") == {"ok": True, "msg": "reachable · HTTP 200"}

    def raise_http_error(req, timeout=3):
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(checks.urllib.request, "urlopen", raise_http_error)
    assert checks.check_api("localhost:11434") == {"ok": True, "msg": "reachable · HTTP 401"}


def test_check_api_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        checks.urllib.request,
        "urlopen",
        lambda req, timeout=3: (_ for _ in ()).throw(RuntimeError("connection refused\ntrace")),
    )

    assert checks.check_api("localhost:11434") == {"ok": False, "msg": "connection refused"}
    assert checks.check_api("") == {"ok": None, "msg": "not configured"}


def test_detect_providers_sync_uses_env_and_local_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "claude-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENCODE_BASE_URL", "http://127.0.0.1:4321/v1")
    monkeypatch.setenv("OPENCODE_API_KEY", "opencode-key")

    def fake_probe(base_url: str, timeout: float = 1.2) -> list[str]:
        if "4321" in base_url:
            return ["opencode-local"]
        if "11434" in base_url:
            return ["llama3", "qwen"]
        raise RuntimeError("offline")

    monkeypatch.setattr(checks, "probe_local_oai", fake_probe)

    providers = checks.detect_providers_sync()
    provider_ids = [provider["id"] for provider in providers]

    assert "claude" in provider_ids
    assert "openai" in provider_ids
    assert "ollama" in provider_ids
    ollama = next(provider for provider in providers if provider["id"] == "ollama")
    assert ollama["models"] == ["llama3", "qwen"]


def test_probe_preview_status_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        checks.urllib.request,
        "urlopen",
        lambda req, timeout=2: _FakeHTTPResponse(204),
    )
    assert checks.probe_preview_status(49312, "/") == {
        "up": True,
        "status": 204,
        "url": "http://localhost:49312/",
    }

    monkeypatch.setattr(
        checks.urllib.request,
        "urlopen",
        lambda req, timeout=2: (_ for _ in ()).throw(RuntimeError("refused")),
    )
    failure = checks.probe_preview_status(49312, "/app")
    assert failure["up"] is False
    assert failure["url"] == "http://localhost:49312/app"

