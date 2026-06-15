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


def _classify_connection_error(exc: Exception, base_url: str) -> dict[str, Any]:
    """Return a machine-readable diagnosis plus human-friendly tip."""
    text = str(exc).lower()
    host = base_url.split("://", 1)[-1].split("/")[0]

    if isinstance(exc, urllib.error.HTTPError):
        code = exc.code
        if code == 401:
            return {
                "kind": "auth",
                "message": f"HTTP 401 — wrong API key for {host}",
                "tip": "Check the API key for this provider.",
            }
        if code == 403:
            return {
                "kind": "auth",
                "message": f"HTTP 403 — access denied on {host}",
                "tip": "Verify the API key has permission to list models.",
            }
        if code == 404:
            return {
                "kind": "endpoint",
                "message": f"HTTP 404 — /models not found on {host}",
                "tip": "This host may not be an OpenAI-compatible endpoint. Try the full /v1 URL.",
            }
        return {
            "kind": "http",
            "message": f"HTTP {code} on {host}",
            "tip": "Endpoint reachable but returned an error.",
        }

    if "certificate" in text or "ssl" in text:
        return {
            "kind": "tls",
            "message": f"TLS certificate problem with {host}",
            "tip": "Use http:// instead of https:// for local servers, or update system certificates.",
        }

    if "timeout" in text or isinstance(exc, socket.timeout):
        return {
            "kind": "timeout",
            "message": f"Connection to {host} timed out",
            "tip": "The host is slow or the firewall is blocking the port.",
        }

    if "refused" in text or "nodename" in text or "name or service" in text:
        return {
            "kind": "network",
            "message": f"Cannot reach {host}",
            "tip": "Make sure the server is running and the URL/port are correct.",
        }

    return {
        "kind": "unknown",
        "message": str(exc).split("\n")[0][:72],
        "tip": "Could not connect. Check the URL and network.",
    }


def check_api(base_url: str) -> dict[str, Any]:
    if not base_url:
        return {"ok": None, "msg": "not configured"}

    normalized = normalize_url(base_url)
    try:
        req = urllib.request.Request(normalized + "/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return {"ok": True, "msg": f"reachable · HTTP {resp.status}"}
    except urllib.error.HTTPError as exc:
        return {"ok": True, "msg": f"reachable · HTTP {exc.code}"}
    except Exception as exc:
        short = str(exc).split("\n")[0].split("(")[0].strip()
        return {"ok": False, "msg": short or "unreachable"}


def probe_local_oai(base_url: str, timeout: float = 1.5) -> list[str]:
    url = normalize_url(base_url).rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
        return [model["id"] for model in data.get("data", [])]


def diagnose_provider(base_url: str, api_key: str = "", timeout: float = 3.0) -> dict[str, Any]:
    """
    Probe a single OpenAI-compatible endpoint and return a full diagnosis.

    Includes reachability, model list, error classification and a tip.
    """
    normalized = normalize_url(base_url)
    result: dict[str, Any] = {
        "base_url": normalized,
        "api_key": api_key,
        "reachable": False,
        "models": [],
        "model": "",
        "kind": "unknown",
        "message": "",
        "tip": "",
    }

    try:
        models = probe_local_oai(normalized, timeout=timeout)
        result.update(
            {
                "reachable": True,
                "models": models,
                "model": models[0] if models else "",
                "kind": "ok",
                "message": f"reachable · {len(models)} model(s)",
                "tip": "Ready to use.",
            }
        )
        return result
    except Exception as exc:
        diag = _classify_connection_error(exc, normalized)
        result.update(
            {
                "reachable": False,
                "kind": diag["kind"],
                "message": diag["message"],
                "tip": diag["tip"],
            }
        )
        return result


def detect_providers_sync(
    extra_hosts: list[str] | None = None,
    timeout: float = 1.5,
) -> list[dict[str, Any]]:
    """
    Probe all known provider locations (env vars + common local ports).
    Returns a JSON array of available providers with pre-filled config.
    Used by the dashboard to show one-click provider presets.
    """
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
                "reachable": True,
                "kind": "ok",
                "message": "configured from environment",
                "tip": "Click to add.",
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
                "reachable": True,
                "kind": "ok",
                "message": "configured from environment",
                "tip": "Click to add.",
            }
        )

    env_base_url = os.getenv("BENCHBOT_BASE_URL", "")
    env_api_key = os.getenv("BENCHBOT_API_KEY", "")
    if env_base_url:
        entry = diagnose_provider(env_base_url, env_api_key, timeout=timeout)
        entry.update(
            {
                "id": "env-benchbot",
                "label": f"env: BENCHBOT_BASE_URL",
                "source": "env:BENCHBOT_BASE_URL",
            }
        )
        detected.append(entry)

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

    for host in (extra_hosts or []):
        local_candidates.append(
            {
                "id": f"custom-{host}",
                "label": host,
                "base_url": host,
                "api_key": "",
                "source": f"custom:{host}",
            }
        )

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
        entry = diagnose_provider(candidate["base_url"], candidate.get("api_key", ""), timeout=timeout)
        entry.update(
            {
                "id": candidate["id"],
                "label": candidate["label"],
                "source": candidate["source"],
                "api_key": candidate.get("api_key", ""),
            }
        )
        if entry["reachable"] or candidate.get("source", "").startswith("custom:"):
            detected.append(entry)

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
