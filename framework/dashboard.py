"""
framework/dashboard.py
~~~~~~~~~~~~~~~~~~~~~~
benchb0t live dashboard.

Usage
─────
  python -m framework.dashboard
  → open http://localhost:7860
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from framework.config import (
    FrameworkConfigError,
    LevelValidationError,
    LoadedFrameworkConfig,
    load_framework_config,
    load_level_config,
)
from framework.store import Store

logger = logging.getLogger(__name__)

app = FastAPI(title="benchb0t", docs_url=None, redoc_url=None)

_runs_dir: Path = Path("runs")
_project_dir: Path = Path(".").resolve()
_active_procs: list[subprocess.Popen] = []
_active_proc_lock = asyncio.Lock()
_run_batch_started_at = 0.0

_CREDS_FILE: Path = Path(".benchb0t_creds.json")
_CREDS_KEYS = ("base_url", "model", "api_key", "providers")

_store: Store | None = None   # initialised in main()
_loaded_config: LoadedFrameworkConfig | None = None

# Rolling buffer for subprocess (runner) stdout — shown in UI when run fails
_runner_log: collections.deque = collections.deque(maxlen=600)


# ── Models ─────────────────────────────────────────────────────────────────────

class ProviderRequest(BaseModel):
    base_url: str
    model: str
    api_key: str = ""
    label: str = ""


class RunRequest(BaseModel):
    base_url:   str = ""
    model:      str = ""
    api_key:    str = ""
    level:      str = ""        # empty = use all_levels
    all_levels: bool = False
    providers:  list[ProviderRequest] = []


# ── Credentials ────────────────────────────────────────────────────────────────

def _load_creds() -> dict:
    try:
        if _CREDS_FILE.exists():
            return json.loads(_CREDS_FILE.read_text())
    except Exception:
        pass
    return {}

def _save_creds(data: dict) -> None:
    try:
        safe = {k: data[k] for k in _CREDS_KEYS if k in data}
        _CREDS_FILE.write_text(json.dumps(safe, indent=2))
    except Exception as e:
        logger.warning("Could not save credentials: %s", e)


def _providers_from_request(req: RunRequest) -> list[dict[str, str]]:
    providers: list[dict[str, str]] = []
    if req.providers:
        for p in req.providers:
            base_url = p.base_url.strip()
            model = p.model.strip()
            if not base_url or not model:
                continue
            providers.append({
                "base_url": base_url,
                "model": model,
                "api_key": p.api_key.strip(),
                "label": (p.label or model).strip(),
            })
    elif req.base_url.strip() and req.model.strip():
        providers.append({
            "base_url": req.base_url.strip(),
            "model": req.model.strip(),
            "api_key": req.api_key.strip(),
            "label": req.model.strip(),
        })
    return providers


def _save_provider_creds(providers: list[dict[str, str]]) -> None:
    if not providers:
        return
    first = providers[0]
    _save_creds({
        "base_url": first["base_url"],
        "model": first["model"],
        "api_key": first["api_key"],
        "providers": providers,
    })


def _alive_procs() -> list[subprocess.Popen]:
    global _active_procs
    _active_procs = [proc for proc in _active_procs if proc.poll() is None]
    return list(_active_procs)


# ── REST API ───────────────────────────────────────────────────────────────────

@app.get("/api/levels")
def list_levels() -> JSONResponse:
    levels = []
    for p in sorted((_project_dir / "levels").glob("*.yaml")):
        try:
            level = load_level_config(p)
            if level.is_deprecated:
                continue
            levels.append({
                "path":         str(p),
                "id":           level.level.id,
                "name":         level.level.name,
                "difficulty":   level.level.difficulty,
                "category":     level.level.category,
                "instruction":  level.task.instruction.strip(),
                "tools":        list(level.tools),
                "max_turns":    level.task.max_turns if level.task.max_turns is not None else "?",
                "timeout_s":    level.task.timeout_s if level.task.timeout_s is not None else "?",
                "preview_port": level.preview.port if level.preview else "",
                "preview_path": level.preview.path if level.preview else "/",
            })
        except LevelValidationError:
            levels.append({"path": str(p), "id": p.stem, "name": p.stem,
                           "difficulty": 1, "instruction": "", "tools": [],
                           "max_turns": "?", "timeout_s": "?",
                           "preview_port": "", "preview_path": "/"})
        except Exception:
            levels.append({"path": str(p), "id": p.stem, "name": p.stem,
                           "difficulty": 1, "instruction": "", "tools": [],
                           "max_turns": "?", "timeout_s": "?",
                           "preview_port": "", "preview_path": "/"})
    return JSONResponse(levels)


@app.get("/api/credentials")
def get_credentials() -> JSONResponse:
    return JSONResponse(_load_creds())


@app.post("/api/credentials")
async def save_credentials(req: RunRequest) -> JSONResponse:
    _save_provider_creds(_providers_from_request(req))
    return JSONResponse({"status": "saved"})


@app.get("/api/status")
def get_status() -> JSONResponse:
    alive = _alive_procs()
    if alive:
        return JSONResponse({
            "status": "running",
            "count": len(alive),
            "pids": [proc.pid for proc in alive],
        })
    return JSONResponse({"status": "idle"})


@app.post("/api/run")
async def start_run(req: RunRequest) -> JSONResponse:
    global _active_procs, _run_batch_started_at
    async with _active_proc_lock:
        if _alive_procs():
            return JSONResponse({"error": "run already in progress"}, status_code=409)

        providers = _providers_from_request(req)
        if not providers:
            return JSONResponse({"error": "no provider configured"}, status_code=400)

        _save_provider_creds(providers)

        harnesses = sorted((_project_dir / "harnesses").glob("*.yaml"))
        if not harnesses:
            return JSONResponse({"error": "no harness files found"}, status_code=500)

        _runner_log.clear()
        _active_procs = []
        _run_batch_started_at = time.time()

        for idx, provider in enumerate(providers, start=1):
            env = {
                **os.environ,
                "BENCHBOT_BASE_URL": _normalize_url(provider["base_url"]),
                "BENCHBOT_MODEL":    provider["model"],
                "BENCHBOT_API_KEY":  provider["api_key"] or "benchbot",
                "BENCHBOT_PROVIDER_SLOT": str(idx),
                "BENCHBOT_PROVIDER_LABEL": provider["label"],
                "PYTHONUNBUFFERED":  "1",
            }

            cmd = [sys.executable, "-m", "framework.runner",
                   "--no-prompt", "--harness", str(harnesses[0])]

            if req.all_levels or not req.level:
                cmd.append("--all-levels")
            else:
                cmd += ["--level", req.level]

            logger.info("Spawning provider %d: %s", idx, " ".join(cmd))
            proc = subprocess.Popen(
                cmd, cwd=str(_project_dir), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            _active_procs.append(proc)
            asyncio.create_task(_drain_runner(proc, prefix=f"[P{idx} {provider['model']}] "))

    return JSONResponse({
        "status": "started",
        "pids": [proc.pid for proc in _active_procs],
        "providers": len(_active_procs),
    })


async def _drain_runner(proc: subprocess.Popen, prefix: str = "") -> None:
    """Read runner subprocess stdout line-by-line into the rolling log buffer."""
    loop = asyncio.get_event_loop()
    while True:
        line: str = await loop.run_in_executor(None, proc.stdout.readline)
        if not line:
            break
        stripped = line.rstrip()
        tagged = f"{prefix}{stripped}" if prefix else stripped
        _runner_log.append(tagged)
        logger.debug("[runner] %s", tagged)


@app.get("/api/runner-log")
def get_runner_log() -> JSONResponse:
    """Return the buffered stdout of the most recent runner subprocess."""
    return JSONResponse({"lines": list(_runner_log)})


# ── Preflight checks ───────────────────────────────────────────────────────────

def _check_docker() -> dict:
    try:
        import docker as _docker
        client = _docker.from_env()
        client.ping()
        info = client.info()
        ver  = info.get("ServerVersion", "?")
        nc   = len(client.containers.list(all=True))
        return {"ok": True, "msg": f"v{ver} · {nc} container{'s' if nc != 1 else ''}"}
    except Exception as exc:
        short = str(exc).split("(")[0].strip()
        return {"ok": False, "msg": short or "daemon not reachable"}


def _check_api(base_url: str) -> dict:
    if not base_url:
        return {"ok": None, "msg": "not configured"}
    norm = base_url if base_url.startswith(("http://", "https://")) else "http://" + base_url
    norm = norm.rstrip("/")
    try:
        req = urllib.request.Request(norm + "/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return {"ok": True, "msg": f"reachable · HTTP {resp.status}"}
    except urllib.error.HTTPError as exc:
        # 401/403/404 still means the server is responding
        return {"ok": True, "msg": f"reachable · HTTP {exc.code}"}
    except Exception as exc:
        short = str(exc).split("\n")[0][:72]
        return {"ok": False, "msg": short}


# ── Provider auto-detection ────────────────────────────────────────────────────

def _probe_local_oai(base_url: str, timeout: float = 1.5) -> list[str]:
    """
    Probe an OpenAI-compatible /v1/models endpoint.
    Returns a list of model IDs (possibly empty) if the server is up.
    Raises on connection error.
    """
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
        return [m["id"] for m in data.get("data", [])]


def _detect_providers_sync() -> list[dict]:
    """
    Synchronously probe all known provider locations.
    Returns a list of detected-provider dicts ready to send to the frontend.
    """
    detected: list[dict] = []

    # ── Cloud providers (check env vars) ──────────────────────────────────────

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        detected.append({
            "id":       "claude",
            "label":    "Claude (Anthropic)",
            "base_url": "https://api.anthropic.com/v1",
            "model":    "claude-sonnet-4-5",
            "api_key":  anthropic_key,
            "source":   "env:ANTHROPIC_API_KEY",
            "models":   [
                "claude-opus-4-5",
                "claude-sonnet-4-5",
                "claude-haiku-4-5",
            ],
        })

    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        detected.append({
            "id":       "openai",
            "label":    "OpenAI / Codex",
            "base_url": "https://api.openai.com/v1",
            "model":    "gpt-4.1",
            "api_key":  openai_key,
            "source":   "env:OPENAI_API_KEY",
            "models":   ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "o3", "o4-mini"],
        })

    # ── Local servers ─────────────────────────────────────────────────────────

    local_candidates = [
        {
            "id":       "ollama",
            "label":    "Ollama",
            "base_url": "http://localhost:11434/v1",
            "api_key":  "ollama",
            "source":   "localhost:11434",
        },
        {
            "id":       "lmstudio",
            "label":    "LM Studio",
            "base_url": "http://localhost:1234/v1",
            "api_key":  "lm-studio",
            "source":   "localhost:1234",
        },
        {
            "id":       "opencode",
            "label":    "OpenCode",
            "base_url": "http://localhost:3000/v1",
            "api_key":  "opencode",
            "source":   "localhost:3000",
        },
        {
            "id":       "vllm",
            "label":    "vLLM",
            "base_url": "http://localhost:8000/v1",
            "api_key":  "vllm",
            "source":   "localhost:8000",
        },
    ]

    # Also check env-configured OpenCode
    oc_url = os.getenv("OPENCODE_BASE_URL", "")
    if oc_url:
        local_candidates.insert(0, {
            "id":       "opencode-env",
            "label":    "OpenCode (env)",
            "base_url": oc_url.rstrip("/"),
            "api_key":  os.getenv("OPENCODE_API_KEY", "opencode"),
            "source":   "env:OPENCODE_BASE_URL",
        })

    for candidate in local_candidates:
        try:
            models = _probe_local_oai(candidate["base_url"], timeout=1.2)
            entry = dict(candidate)
            entry["models"] = models
            if models:
                entry["model"] = models[0]
            detected.append(entry)
        except Exception:
            pass  # server not running — silently skip

    return detected


@app.get("/api/detect-providers")
async def detect_providers() -> JSONResponse:
    """
    Probe all known provider locations (env vars + common local ports).
    Returns a JSON array of available providers with pre-filled config.
    Used by the dashboard to show one-click provider presets.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _detect_providers_sync)
    return JSONResponse(result)


@app.get("/api/preview-status")
async def preview_status(port: int, path: str = "/") -> JSONResponse:
    """
    Check whether an HTTP server is listening on localhost:port.
    Called by the dashboard to show a 'waiting…' state until the agent's
    dev server comes up.
    """
    url = f"http://localhost:{port}{path}"
    loop = asyncio.get_event_loop()
    def _probe():
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return {"up": True, "status": resp.status, "url": url}
        except urllib.error.HTTPError as exc:
            # Any HTTP response (even 4xx) means the server is up
            return {"up": True, "status": exc.code, "url": url}
        except Exception as exc:
            return {"up": False, "error": str(exc)[:80], "url": url}
    result = await loop.run_in_executor(None, _probe)
    return JSONResponse(result)


@app.get("/api/preflight")
async def preflight(base_url: str = "") -> JSONResponse:
    """Run all readiness checks and return pass/fail per check."""
    loop = asyncio.get_event_loop()

    # Docker ping runs blocking I/O — offload to thread pool
    docker_res = await loop.run_in_executor(None, _check_docker)
    api_res    = await loop.run_in_executor(None, _check_api, _normalize_url(base_url) if base_url else "")

    levels    = list((_project_dir / "levels").glob("*.yaml"))
    harnesses = list((_project_dir / "harnesses").glob("*.yaml"))

    return JSONResponse({
        "docker":  docker_res,
        "api":     api_res,
        "levels":  {
            "ok":  len(levels) > 0,
            "msg": f"{len(levels)} level{'s' if len(levels) != 1 else ''} found" if levels else "no levels in ./levels/",
        },
        "harness": {
            "ok":  len(harnesses) > 0,
            "msg": harnesses[0].name if harnesses else "no harness in ./harnesses/",
        },
    })


@app.get("/api/stats/summary")
def stats_summary() -> JSONResponse:
    return JSONResponse(_store.get_summary() if _store else {})

@app.get("/api/stats/models")
def stats_models() -> JSONResponse:
    return JSONResponse(_store.get_model_stats() if _store else [])

@app.get("/api/stats/levels")
def stats_levels() -> JSONResponse:
    return JSONResponse(_store.get_level_stats() if _store else [])

@app.get("/api/stats/model-detail/{model:path}")
def stats_model_detail(model: str) -> JSONResponse:
    """
    Full per-level breakdown for a single model.
    Used by the analytics Dex entry to render the level-conquest grid
    and rich stat display without re-querying the full model list.
    """
    if not _store:
        return JSONResponse({"error": "store not available"}, status_code=503)
    data = _store.get_model_detail(model)
    return JSONResponse(data)

@app.get("/api/runs")
def list_runs(
    limit:     int  = 50,
    offset:    int  = 0,
    model:     str  = "",
    level_id:  str  = "",
    min_stars: int  = -1,
) -> JSONResponse:
    """
    Return runs newest-first with optional pagination and filters.

    Query params
    ------------
    limit     : max rows (default 50)
    offset    : skip N rows (for pagination)
    model     : filter by exact model name
    level_id  : filter by exact level id
    min_stars : only runs with ≥ N stars (-1 = no filter)
    """
    if not _store:
        return JSONResponse([])
    runs = _store.get_runs(
        limit=limit,
        offset=offset,
        model=model,
        level_id=level_id,
        min_stars=min_stars if min_stars >= 0 else None,
    )
    total = _store.get_run_count(model=model, level_id=level_id)
    return JSONResponse({"runs": runs, "total": total, "limit": limit, "offset": offset})


@app.get("/api/runs/meta")
def runs_meta() -> JSONResponse:
    """
    Returns distinct models + levels stored in the DB.
    Used by the history UI to populate filter dropdowns.
    """
    if not _store:
        return JSONResponse({"models": [], "levels": []})
    return JSONResponse({
        "models": _store.get_distinct_models(),
        "levels": _store.get_distinct_levels(),
    })


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> JSONResponse:
    """
    Return full data for a single run identified by its 8-char hex id.
    Also returns the parsed agentlog events so the UI can replay the session.
    """
    if not _store:
        return JSONResponse({"error": "store not available"}, status_code=503)

    run = _store.get_run_by_id(run_id)
    if not run:
        return JSONResponse({"error": f"run {run_id!r} not found"}, status_code=404)

    # Try to locate and parse the agentlog file by run_id suffix
    events: list[dict] = []
    log_found = False
    try:
        for log_path in sorted(_runs_dir.glob(f"*_{run_id}.agentlog")):
            from framework.recorder import load_agentlog
            events = load_agentlog(log_path)
            log_found = True
            break
    except Exception as exc:
        logger.warning("Could not load agentlog for run %s: %s", run_id, exc)

    return JSONResponse({
        "run":       run,
        "events":    events,
        "log_found": log_found,
    })


# ── BenchBot-AI chat ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages:      list[dict]   # OpenAI-format message history from the UI
    base_url:      str = ""
    model:         str = ""
    api_key:       str = ""
    active_run_id: str = ""     # dashboard: inject live agentlog for this run
    page:          str = "dashboard"  # "dashboard" | "analytics" | "builder"
    page_context:  str = ""     # builder: current level YAML from the form


def _load_live_session_context(run_id: str) -> str:
    """
    Load the agentlog for a running/just-finished session by run_id.
    Returns a formatted string block for injection into the system prompt.
    """
    if not run_id:
        return ""
    try:
        from framework.recorder import load_agentlog
        for log_path in sorted(_runs_dir.glob(f"*_{run_id}.agentlog")):
            events = load_agentlog(log_path)
            lines: list[str] = [f"LIVE SESSION LOG (run_id={run_id}):"]
            for ev in events:
                t = ev.get("type", "?")
                if t == "session_start":
                    lines.append(
                        f"  [START] model={ev.get('model','?')} "
                        f"level={ev.get('level_id','?')} mode={ev.get('mode','?')}"
                    )
                elif t == "tool_call":
                    args_str = str(ev.get("args", {}))[:120]
                    lines.append(f"  [TOOL] {ev.get('tool','?')}({args_str})")
                elif t == "tool_result":
                    out = str(ev.get("output", ""))[:120].replace("\n", " ")
                    lines.append(f"  [RES] exit={ev.get('exit_code',0)} {out}")
                elif t == "message" and ev.get("role") == "assistant":
                    content = str(ev.get("content", ""))[:200].replace("\n", " ")
                    lines.append(f"  [AI] {content}")
                elif t == "session_end":
                    sc = ev.get("score", {})
                    lines.append(
                        f"  [END] score={sc.get('total',0):.1f} "
                        f"timed_out={ev.get('timed_out',False)} "
                        f"duration={ev.get('duration_s',0):.1f}s"
                    )
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("live session context load failed for %s: %s", run_id, exc)
    return ""


def _build_analytics_context() -> str:
    """
    Page-specific context for the Analytics page.
    Focuses on comparative model analysis, trends, and recommendations.
    """
    parts: list[str] = [
        "You are BenchBot-AI on the ANALYTICS page. "
        "Your role: interpret benchmark data, compare model performance, "
        "spot trends, explain why models succeed or fail on specific levels. "
        "Be specific — cite exact scores, pass rates, turn counts. "
        "Suggest which model to use for which task type. "
        "Format: plain text, no markdown.",
        "",
    ]
    if not _store:
        return "\n".join(parts)
    try:
        models = _store.get_model_stats()
        if models:
            parts.append("MODEL LEADERBOARD:")
            for i, m in enumerate(models[:15], 1):
                parts.append(
                    f"  #{i} {m['model']}: avg={m['avg_score']} best={m['best_score']} "
                    f"runs={m['run_count']} stars={m.get('total_stars',0)} "
                    f"turns={m.get('avg_turns','?')} timeouts={m.get('timeouts',0)}"
                )
            parts.append("")
        levels = _store.get_level_stats()
        if levels:
            parts.append("LEVEL DIFFICULTY vs PASS RATE:")
            for l in levels:
                pr = round((l.get("pass_rate") or 0) * 100)
                parts.append(
                    f"  {l['level_id']} diff={l.get('difficulty',1)} "
                    f"avg={l['avg_score']} pass={pr}% runs={l['run_count']}"
                )
            parts.append("")
        cmp = _store.get_mode_comparison() if hasattr(_store, "get_mode_comparison") else []
        if cmp:
            parts.append("GUIDED vs UNGUIDED COMPARISON (sample):")
            for row in cmp[:20]:
                parts.append(
                    f"  {row['model']} · {row['level_id']} · {row['mode']}: "
                    f"avg={row['avg_score']} turns={row['avg_turns']} timeouts={row['timeouts']}"
                )
            parts.append("")
    except Exception as exc:
        logger.warning("analytics context: %s", exc)
    return "\n".join(parts)


def _build_builder_context(current_level_yaml: str = "") -> str:
    """
    Page-specific context for the Builder page.
    Makes the LLM an expert level designer that understands the YAML schema
    and can output <level_patch> JSON blocks to edit the form directly.
    """
    # Load existing levels as examples
    example_levels: list[str] = []
    levels_dir = _project_dir / "levels"
    if levels_dir.exists():
        for p in sorted(levels_dir.glob("*.yaml"))[:5]:
            try:
                example_levels.append(p.read_text(encoding="utf-8")[:1200])
            except Exception:
                pass

    parts: list[str] = [
        "You are BenchBot-AI on the LEVEL BUILDER page. "
        "You are an expert benchb0t level designer. "
        "You help the user create, refine, and debug benchmark levels for LLM agents. "
        "You know the full YAML schema and what makes a good level. "
        "IMPORTANT: When you suggest concrete changes to the level, output a <level_patch> block "
        "with a JSON object containing ONLY the fields to change. "
        "The user can then click APPLY PATCH to fill those fields into the form automatically. "
        "Format answers in plain text. Keep <level_patch> blocks short and valid JSON.",
        "",
        "LEVEL YAML SCHEMA REFERENCE:",
        """  level:
    id: string          # e.g. l5-data-pipeline (used as filename)
    name: string        # human-readable display name
    difficulty: 1-5     # 1=trivial, 5=expert
    category: string    # file-operations | backend | webapp | game | data | networking
    tags: [list]

  container:
    image: string       # e.g. python:3.11-slim, node:20-slim, ubuntu:22.04
    working_dir: /workspace
    packages:
      apt: [curl, git, ...]
      pip: [requests, pandas, ...]
      npm: []           # installed globally with npm install -g
    setup_script: |     # shell script, runs before agent gets the task

  preview:              # optional — only for levels that start a web server
    port: 3000
    path: /

  task:
    instruction: |      # exact task description given to the agent
    max_turns: 15
    timeout_s: 120

  tools:                # subset of: [bash, read_file, write_file, list_dir, http_request, run_background, patch_file]
    - bash
    - write_file

  evaluation:
    efficiency_target: 5   # ideal number of tool calls
    criteria:
      - id: unique_snake_case_id
        description: "human-readable check description"
        type: script
        check: "bash command that exits 0 on success"
        weight: 1.0     # higher = more important

  forced_retry:          # optional
    enabled: true
    max_retries: 2
    penalty_per_retry: 10
    completion_threshold: 0.5""",
        "",
        "AVAILABLE DOCKER IMAGES (tested and recommended):",
        "  python:3.11-slim  — Python tasks (pip packages available)",
        "  node:20-slim      — Node.js / Express / React (npm available)",
        "  ubuntu:22.04      — General Linux (apt available)",
        "  python:3.11-slim  — also works for shell + curl tasks with apt:curl",
        "",
        "LEVEL DESIGN PRINCIPLES:",
        "  1. Instructions must be UNAMBIGUOUS — the agent has no context beyond the task text",
        "  2. Setup script must create all needed files/dirs BEFORE the agent starts",
        "  3. Evaluation criteria must be SHELL-TESTABLE (exit 0 = pass)",
        "  4. efficiency_target = ideal tool calls for a skilled agent (not minimum possible)",
        "  5. For web servers: always add run_background to tools, set preview: port",
        "  6. Criteria weights sum roughly to ~10 for a balanced score",
        "",
    ]

    if current_level_yaml.strip():
        parts.append("CURRENT LEVEL BEING EDITED (from the form):")
        parts.append(current_level_yaml[:3000])
        parts.append("")

    if example_levels:
        parts.append("EXISTING LEVELS AS REFERENCE (first 5):")
        for i, ex in enumerate(example_levels, 1):
            parts.append(f"--- example {i} ---")
            parts.append(ex[:800])
            parts.append("")

    parts.append(
        "PATCH FORMAT EXAMPLE — when suggesting level changes, use exactly this format:\n"
        "<level_patch>\n"
        '{"name": "My Level", "difficulty": 3, "instruction": "Do X and Y...", '
        '"setup_script": "mkdir -p /workspace\\necho hello > /workspace/input.txt", '
        '"efficiency_target": 5}\n'
        "</level_patch>\n"
        "Patchable fields: name, difficulty, category, tags, image, working_dir, "
        "apt, pip, npm, setup_script, instruction, max_turns, timeout_s, efficiency_target, "
        "tools, criteria (array of {id,description,check,weight})."
    )
    return "\n".join(parts)


def _build_chat_context(active_run_id: str = "", page: str = "dashboard", page_context: str = "") -> str:
    """
    Dispatch to the right context builder based on which page is calling.
    Each page gets a tailored system prompt + relevant data.
    """
    if page == "analytics":
        return _build_analytics_context()
    if page == "builder":
        return _build_builder_context(current_level_yaml=page_context)

    # ── Dashboard (default) ─────────────────────────────────────────────────
    # Live DB stats + recent agentlogs + optional live session context.
    parts: list[str] = [
        "You are BenchBot-AI, an embedded analyst inside the benchb0t LLM-agent "
        "benchmarking framework. You have real-time access to benchmark data and "
        "can answer questions about model performance, level difficulty, run logs, "
        "and scoring. Be concise, technical, and specific — cite actual numbers "
        "from the data. Format answers in plain text (no markdown).",
        "",
    ]

    # ── Live session context (injected first when a run is active) ────────────
    if active_run_id:
        live_ctx = _load_live_session_context(active_run_id)
        if live_ctx:
            parts.append("⚡ ACTIVE SESSION — the user is watching this run live:")
            parts.append(live_ctx)
            parts.append("")

    # ── DB stats ─────────────────────────────────────────────────────────────
    if _store:
        try:
            s = _store.get_summary()
            parts.append(
                f"BENCHMARK SUMMARY: {s.get('total_runs',0)} total runs | "
                f"{s.get('total_models',0)} models | "
                f"{s.get('total_levels',0)} levels | "
                f"avg score {s.get('avg_score',0)} | "
                f"best score {s.get('best_score',0)} | "
                f"{s.get('total_stars',0)} total stars"
            )
            parts.append("")

            models = _store.get_model_stats()
            if models:
                parts.append("MODEL LEADERBOARD:")
                for i, m in enumerate(models[:10], 1):
                    parts.append(
                        f"  #{i} {m['model']}: avg={m['avg_score']} "
                        f"best={m['best_score']} runs={m['run_count']} "
                        f"stars={m.get('total_stars',0)} "
                        f"turns={m.get('avg_turns','?')} "
                        f"timeouts={m.get('timeouts',0)}"
                    )
                parts.append("")

            levels = _store.get_level_stats()
            if levels:
                parts.append("LEVEL STATS:")
                for l in levels:
                    pr = round((l.get('pass_rate') or 0) * 100)
                    parts.append(
                        f"  {l['level_id']} (diff={l.get('difficulty',1)}) "
                        f"avg={l['avg_score']} best={l['best_score']} "
                        f"pass_rate={pr}% runs={l['run_count']}"
                    )
                parts.append("")

            recent = _store.get_runs(limit=20)
            if recent:
                parts.append("RECENT RUNS (last 20, newest first):")
                for r in recent:
                    from datetime import datetime
                    ts = datetime.fromtimestamp(r['ts']).strftime('%m-%d %H:%M')
                    parts.append(
                        f"  [{ts}] {r['model']} on {r['level_id']}: "
                        f"score={r['score_total']:.1f} stars={r['stars']} "
                        f"turns={r['turns']} tools={r['tool_calls_n']} "
                        f"{'TIMEOUT' if r.get('timed_out') else 'ok'}"
                    )
                parts.append("")
        except Exception as exc:
            logger.warning("chat context: DB query failed: %s", exc)

    # ── Level definitions ─────────────────────────────────────────────────────
    levels_dir = _project_dir / "levels"
    if levels_dir.exists():
        try:
            yamls = sorted(levels_dir.glob("*.yaml"))[:12]
            if yamls:
                parts.append("LEVEL DEFINITIONS:")
            for p in yamls:
                cfg = yaml.safe_load(p.read_text())
                lvl  = cfg.get("level", {})
                task = cfg.get("task",  {})
                ev   = cfg.get("evaluation", {})
                crits = [c.get("id","?") for c in ev.get("criteria", [])]
                parts.append(
                    f"  {lvl.get('id','?')} \"{lvl.get('name','?')}\" "
                    f"diff={lvl.get('difficulty',1)} cat={lvl.get('category','?')} "
                    f"max_turns={task.get('max_turns','?')} "
                    f"criteria=[{','.join(crits)}]"
                )
                instr = (task.get("instruction") or "").strip()
                if instr:
                    parts.append(f"    instruction: {instr[:200]}")
            parts.append("")
        except Exception as exc:
            logger.warning("chat context: level YAML read failed: %s", exc)

    # ── Recent agentlogs ──────────────────────────────────────────────────────
    if _runs_dir.exists():
        try:
            log_files = sorted(
                _runs_dir.glob("*.agentlog"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:3]
            if log_files:
                parts.append("RECENT AGENT LOGS (last 3 runs, tool calls only):")
            for lf in log_files:
                try:
                    lines = lf.read_text().strip().split("\n")
                    tool_events = []
                    for line in lines:
                        try:
                            ev = json.loads(line)
                            if ev.get("type") in ("tool_call", "tool_result"):
                                tool = ev.get("tool", ev.get("name", "?"))
                                args = str(ev.get("args", ""))[:80]
                                out  = str(ev.get("output", ""))[:120]
                                ec   = ev.get("exit_code", 0)
                                if ev["type"] == "tool_call":
                                    tool_events.append(f"    CALL {tool}({args})")
                                else:
                                    tool_events.append(
                                        f"    -> exit={ec} {out[:80]}"
                                    )
                        except Exception:
                            continue
                    parts.append(f"  [{lf.stem}]")
                    parts.extend(tool_events[:30])  # cap at 30 events per log
                except Exception:
                    continue
            parts.append("")
        except Exception as exc:
            logger.warning("chat context: agentlog read failed: %s", exc)

    return "\n".join(parts)


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest) -> StreamingResponse:
    """
    Stream a chat completion from the configured LLM with live benchmark
    context injected as the system message.

    The frontend sends the full conversation history; this endpoint prepends
    the context system message and streams tokens back as SSE.
    """
    from framework.api import AgentAPI

    # Resolve endpoint — request > saved creds > env
    creds    = _load_creds()
    base_url = req.base_url.strip() or creds.get("base_url", "") or os.getenv("BENCHBOT_BASE_URL", "")
    model    = req.model.strip()    or creds.get("model",    "") or os.getenv("BENCHBOT_MODEL",    "")
    api_key  = req.api_key.strip()  or creds.get("api_key",  "") or os.getenv("BENCHBOT_API_KEY",  "benchbot")

    if not base_url or not model:
        async def _err():
            yield "data: {\"error\": \"no endpoint configured\"}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    # Normalise URL
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    context_system = _build_chat_context(
        active_run_id=req.active_run_id or "",
        page=req.page or "dashboard",
        page_context=req.page_context or "",
    )
    messages = [{"role": "system", "content": context_system}] + list(req.messages)

    def _stream():
        try:
            client = AgentAPI(
                base_url=base_url,
                api_key=api_key or "benchbot",
                model=model,
                temperature=0.3,
                max_tokens=1024,
                timeout=60.0,
            )
            for delta in client.stream_chat(messages):
                if delta:
                    payload = json.dumps({"delta": delta})
                    yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.warning("chat stream error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/api/stop")
async def stop_run() -> JSONResponse:
    global _active_procs
    async with _active_proc_lock:
        for proc in _alive_procs():
            proc.terminate()
        _active_procs = []
    return JSONResponse({"status": "stopped"})


@app.get("/api/levels/{stem}/parsed")
def load_level_parsed(stem: str) -> JSONResponse:
    """Return a level YAML's fields as structured JSON for the builder to populate."""
    p = _project_dir / "levels" / (stem if stem.endswith(".yaml") else stem + ".yaml")
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        cfg  = yaml.safe_load(p.read_text())
        lvl  = cfg.get("level",     {})
        cont = cfg.get("container", {})
        task = cfg.get("task",      {})
        prev = cfg.get("preview",   {})
        pkgs = cont.get("packages", {})
        return JSONResponse({
            "id":          lvl.get("id", ""),
            "name":        lvl.get("name", ""),
            "difficulty":  lvl.get("difficulty", 1),
            "category":    lvl.get("category", ""),
            "tags":        ", ".join(lvl.get("tags", [])),
            "image":       cont.get("image", "python:3.11-slim"),
            "working_dir": cont.get("working_dir", "/workspace"),
            "apt":         " ".join(pkgs.get("apt", [])),
            "pip":         " ".join(pkgs.get("pip", [])),
            "npm":         " ".join(pkgs.get("npm", [])),
            "setup_script": (cont.get("setup_script") or "").strip(),
            "instruction": (task.get("instruction") or "").strip(),
            "max_turns":   task.get("max_turns", 15),
            "timeout_s":   task.get("timeout_s", 90),
            "tools":       cfg.get("tools", []),
            "efficiency_target": cfg.get("evaluation", {}).get("efficiency_target", 5),
            "criteria":    [
                {
                    "id":     c.get("id", ""),
                    "desc":   c.get("description", ""),
                    "check":  c.get("check", ""),
                    "weight": c.get("weight", 1.0),
                }
                for c in cfg.get("evaluation", {}).get("criteria", [])
            ],
            "preview_port": prev.get("port", ""),
            "preview_path": prev.get("path", "/"),
            "forced_retry": cfg.get("forced_retry", None),
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


class SaveLevelRequest(BaseModel):
    filename: str   # e.g. "l4-my-level.yaml"
    content:  str   # raw YAML text

@app.post("/api/levels/save")
async def save_level(req: SaveLevelRequest) -> JSONResponse:
    name = req.filename.strip()
    if not name.endswith(".yaml"):
        name += ".yaml"
    # Reject path traversal attempts
    if "/" in name or "\\" in name or name.startswith("."):
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    dest = _project_dir / "levels" / name
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(req.content, encoding="utf-8")
        return JSONResponse({"status": "saved", "path": str(dest)})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ── WebSocket ──────────────────────────────────────────────────────────────────

async def _tail(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                try: yield json.loads(s)
                except: pass
        while True:
            line = fh.readline()
            if line:
                s = line.strip()
                if s:
                    try: yield json.loads(s)
                    except: pass
            else:
                await asyncio.sleep(0.08)

async def _stream_log(ws: WebSocket, path: Path) -> None:
    await ws.send_text(json.dumps({"_type": "file", "filename": path.name}))
    async for ev in _tail(path):
        await ws.send_text(json.dumps(ev))
        if ev.get("type") == "session_end":
            break


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    seen: set[Path] = set()
    tasks: dict[Path, asyncio.Task] = {}
    connected_at = time.time()
    try:
        while True:
            cutoff = max(connected_at, _run_batch_started_at - 0.5)
            for path in sorted(_runs_dir.glob("*.agentlog"), key=lambda p: p.stat().st_mtime):
                if path in seen:
                    continue
                try:
                    if path.stat().st_mtime < cutoff:
                        continue
                except FileNotFoundError:
                    continue
                seen.add(path)
                tasks[path] = asyncio.create_task(_stream_log(ws, path))

            done = [path for path, task in tasks.items() if task.done()]
            for path in done:
                try:
                    await tasks[path]
                except Exception:
                    pass
                tasks.pop(path, None)

            await asyncio.sleep(0.25)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        for task in tasks.values():
            task.cancel()


# ── Template loader ───────────────────────────────────────────────────────────
# HTML lives in framework/templates/ — keeping Python and markup separate.

_TEMPLATES = Path(__file__).parent / "templates"


def _template(name: str) -> str:
    """Load a template file; raises FileNotFoundError if missing (fast-fail)."""
    return (_TEMPLATES / name).read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_template("dashboard.html"))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    if not url.rstrip("/").endswith("/v1"):
        url = url.rstrip("/") + "/v1"
    return url


def _resolve_runtime_path(path: Path) -> Path:
    if _loaded_config is None or path.is_absolute():
        return path
    return _loaded_config.resolve_path(path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--runs", type=Path, default=None)
    args = p.parse_args()

    try:
        loaded_config = load_framework_config(args.config)
    except (FileNotFoundError, FrameworkConfigError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    root_level = getattr(
        logging,
        loaded_config.config.framework.log_level.upper(),
        logging.INFO,
    )
    logging.getLogger().setLevel(root_level)

    global _runs_dir, _project_dir, _store, _CREDS_FILE, _loaded_config
    _loaded_config = loaded_config
    _project_dir = loaded_config.project_dir
    _runs_dir = loaded_config.runs_dir if args.runs is None else _resolve_runtime_path(args.runs)
    _CREDS_FILE = _project_dir / ".benchb0t_creds.json"
    _runs_dir.mkdir(parents=True, exist_ok=True)

    _store = Store(loaded_config.db_path).init()

    print(f"\n  🎮  benchb0t → http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


# ── Frontend ───────────────────────────────────────────────────────────────────



@app.get("/builder", response_class=HTMLResponse)
async def builder() -> HTMLResponse:
    return HTMLResponse(_template("builder.html"))




@app.get("/analytics", response_class=HTMLResponse)
async def analytics() -> HTMLResponse:
    return HTMLResponse(_template("analytics.html"))




if __name__ == "__main__":
    main()
