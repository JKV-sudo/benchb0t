"""
framework/dashboard_checks.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Connectivity, readiness, and provider-detection helpers for the dashboard.
Consolidated normalize_url to framework.utils for single source of truth.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any

from framework.utils import normalize_url


def check_docker() -> dict[str, Any]:
    try:
        import docker as _docker

        client = _docker.from_env()
        client.ping()
        info = client.info()
        version = info.get("ServerVersion", "?")
        containers = len(client.containers.list(all=True))
        label = "container" if containers == 1 else "containers"
        return {"ok": True, "msg": f"v{version} · {containers} {label}"}
    except Exception as exc:
        short = str(exc).split("(")[0].strip()
        return {"ok": False, "msg": short or "daemon not reachable"}


def check_api(base_url: str) -> dict[str, Any]:
    if not base_url:
        return {"ok": None, "msg": "not configured"}

    normalized = base_url if base_url.startswith(("http://", "https://")) else "http://" + base_url
    normalized = normalized.rstrip("/")
    try:
        req = urllib.request.Request(normalized + "/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return {"ok": True, "msg": f"reachable · HTTP {resp.status}"}
    except urllib.error.HTTPError as exc:
        return {"ok": True, "msg": f"reachable · HTTP {exc.code}"}
    except Exception as exc:
        short = str(exc).split("\n")[0][:72]
        return {"ok": False, "msg": short}


def probe_local_oai(base_url: str, timeout: float = 1.5) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
        return [model["id"] for model in data.get("data", [])]


def detect_providers_sync() -> list[dict[str, Any]]:
    detected: list[dict[str, Any]] = []

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        detected.append(
            {
                "id": "claude",
                "label": "Claude (Anthropic)",
                "base_url": "https://api.anthropic.com/v1",
                "model": "claude-sonnet-4-5",
                "api_key": anthropic_key,
                "source": "env:ANTHROPIC_API_KEY",
                "models": [
                    "claude-opus-4-5",
                    "claude-sonnet-4-5",
                    "claude-haiku-4-5",
                ],
            }
        )

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        detected.append(
            {
                "id": "openai",
                "label": "OpenAI / Codex",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4.1",
                "api_key": openai_key,
                "source": "env:OPENAI_API_KEY",
                "models": ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "o3", "o4-mini"],
            }
        )

    local_candidates = [
        {
            "id": "ollama",
            "label": "Ollama",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "source": "localhost:11434",
        },
        {
            "id": "lmstudio",
            "label": "LM Studio",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "source": "localhost:1234",
        },
        {
            "id": "opencode",
            "label": "OpenCode",
            "base_url": "http://localhost:3000/v1",
            "api_key": "opencode",
            "source": "localhost:3000",
        },
        {
            "id": "vllm",
            "label": "vLLM",
            "base_url": "http://localhost:8000/v1",
            "api_key": "vllm",
            "source": "localhost:8000",
        },
    ]

    open_code_url = os.getenv("OPENCODE_BASE_URL", "")
    if open_code_url:
        local_candidates.insert(
            0,
            {
                "id": "opencode-env",
                "label": "OpenCode (env)",
                "base_url": open_code_url.rstrip("/"),
                "api_key": os.getenv("OPENCODE_API_KEY", "opencode"),
                "source": "env:OPENCODE_BASE_URL",
            },
        )

    for candidate in local_candidates:
        try:
            models = probe_local_oai(candidate["base_url"], timeout=1.2)
            entry = dict(candidate)
            entry["models"] = models
            if models:
                entry["model"] = models[0]
            detected.append(entry)
        except (urllib.error.URLError, OSError, json.JSONDecodeError, socket.timeout):
            # Local provider unreachable — skip it
            pass

    return detected


def probe_preview_status(port: int, path: str = "/") -> dict[str, Any]:
    url = f"http://localhost:{port}{path}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return {"up": True, "status": resp.status, "url": url}
    except urllib.error.HTTPError as exc:
        return {"up": True, "status": exc.code, "url": url}
    except Exception as exc:
        return {"up": False, "error": str(exc)[:80], "url": url}
