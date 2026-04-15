"""
framework/dashboard.py
~~~~~~~~~~~~~~~~~~~~~~
benchb0t live dashboard (FastAPI app listening on port 7860).

Usage: python -m framework.dashboard and open http://localhost:7860
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from framework.utils import ok_response, error_response, normalize_url

from framework.config import (
    FrameworkConfigError,
    LevelValidationError,
    load_framework_config,
    load_level_config,
)
from framework.dashboard_assistant import (
    assistant_control_prompt,
    assistant_tool_schemas,
    build_level_patch_from_args,
    assistant_state_ui_patch,
    build_initial_assistant_state,
    build_run_request_from_assistant_state,
    list_levels_for_assistant,
    lint_level_content,
    render_level_yaml_from_patch,
    resolve_level_reference,
    save_level_content,
    validate_level_content,
)
from framework.dashboard_checks import (
    check_api,
    check_docker,
    detect_providers_sync,
    probe_preview_status,
)
from framework.dashboard_compare import build_compare_payload
from framework.dashboard_context import build_chat_context
from framework.dashboard_history import build_history_inventory, find_run_log_path
from framework.dashboard_models import ChatRequest, RunRequest, SaveLevelRequest
from framework.dashboard_replay import (
    build_artifact_records,
    build_replay_payload,
    build_replay_run_record,
    build_run_replay,
)
from framework.dashboard_state import DashboardState
from framework.dashboard_stream import stream_agentlog

logger = logging.getLogger(__name__)

app = FastAPI(title="benchb0t", docs_url=None, redoc_url=None)
state = DashboardState()


def _load_replay_payload_for_run(run_id: str) -> tuple[dict[str, Any] | None, int]:
    log_path = find_run_log_path(state.runs_dir, run_id)
    if log_path is None:
        return {"error": f"replay {run_id!r} not found"}, 404

    try:
        from framework.recorder import load_agentlog

        events = load_agentlog(log_path)
        db_run = state.store.get_run_by_id(run_id) if state.store else None
        payload = build_replay_payload(
            run_id,
            events,
            log_path,
            runs_dir=state.runs_dir,
            db_run=db_run,
        )
        return payload, 200
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        # Log file missing or corrupted
        logger.warning("Could not load replay %s: %s", run_id, exc)
        return {"error": str(exc)}, 500


def _json_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, *, size: int = 220) -> list[str]:
    text = text or ""
    if len(text) <= size:
        return [text] if text else []
    return [text[idx: idx + size] for idx in range(0, len(text), size)]


async def _start_run_impl(req: RunRequest) -> tuple[dict[str, Any], int]:
    async with state.active_proc_lock:
        if state.alive_procs():
            return {"error": "run already in progress"}, 409

        providers = state.providers_from_request(req)
        if not providers:
            return {"error": "no provider configured"}, 400

        state.save_provider_creds(providers)

        harnesses = sorted((state.project_dir / "harnesses").glob("*.yaml"))
        if not harnesses:
            return {"error": "no harness files found"}, 500

        state.reset_run_batch()

        for idx, provider in enumerate(providers, start=1):
            env = {
                **os.environ,
                "BENCHBOT_BASE_URL": normalize_url(provider["base_url"]),
                "BENCHBOT_MODEL": provider["model"],
                "BENCHBOT_API_KEY": provider["api_key"] or "benchbot",
                "BENCHBOT_PROVIDER_SLOT": str(idx),
                "BENCHBOT_PROVIDER_LABEL": provider["label"],
                "PYTHONUNBUFFERED": "1",
            }

            cmd = [sys.executable, "-m", "framework.runner", "--no-prompt", "--harness", str(harnesses[0])]
            if req.all_levels or not req.level:
                cmd.append("--all-levels")
            else:
                cmd += ["--level", req.level]
            if req.capture_preview_screenshot:
                cmd.append("--capture-preview-screenshot")
            if req.save_result_bundle:
                cmd.append("--save-result-bundle")
            if req.save_container_snapshot:
                cmd.append("--save-container-snapshot")

            logger.info("Spawning provider %d: %s", idx, " ".join(cmd))
            proc = subprocess.Popen(
                cmd,
                cwd=str(state.project_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            state.active_procs.append(proc)
            asyncio.create_task(_drain_runner(proc, prefix=f"[P{idx} {provider['model']}] "))

    return {
        "status": "started",
        "pids": [proc.pid for proc in state.active_procs],
        "providers": len(state.active_procs),
    }, 200


async def _stop_run_impl() -> dict[str, Any]:
    async with state.active_proc_lock:
        for proc in state.alive_procs():
            proc.terminate()
        state.active_procs = []
    return {"status": "stopped"}


async def _execute_dashboard_assistant_tool(
    tool_name: str,
    args: dict[str, Any],
    *,
    assistant_state: dict[str, Any],
    page: str,
) -> dict[str, Any]:
    if tool_name == "get_benchbot_status":
        providers = assistant_state.get("providers", [])
        status = "running" if state.alive_procs() else "idle"
        first = providers[0] if providers else {}
        message = (
            f"Dashboard is {status}. "
            f"Provider 1 is {first.get('model', 'not configured')} @ {first.get('base_url', 'not configured')}. "
            f"Selection is {'all levels' if assistant_state.get('all_levels') else (assistant_state.get('level') or 'no level selected')}."
        )
        return {
            "ok": True,
            "message": message,
            "data": {
                "status": status,
                "active_pids": [proc.pid for proc in state.alive_procs()],
                "ui_patch": assistant_state_ui_patch(assistant_state),
            },
            "ui_patch": assistant_state_ui_patch(assistant_state),
        }

    if tool_name == "list_benchbot_levels":
        levels = list_levels_for_assistant(state.project_dir, limit=int(args.get("limit", 12) or 12))
        return {
            "ok": True,
            "message": f"Found {len(levels)} selectable levels.",
            "data": {"levels": levels},
        }

    if tool_name == "detect_benchbot_providers":
        providers = detect_providers_sync()
        return {
            "ok": True,
            "message": f"Detected {len(providers)} provider preset(s).",
            "data": {"providers": providers},
        }

    if tool_name == "run_benchbot_preflight":
        providers = assistant_state.get("providers", [])
        base_url = providers[0]["base_url"] if providers else ""
        docker_res = check_docker()
        api_res = check_api(normalize_url(base_url) if base_url else "")
        preflight = {
            "docker": docker_res,
            "api": api_res,
            "levels": {"ok": bool(list((state.project_dir / "levels").glob("*.yaml")))},
            "harness": {"ok": bool(list((state.project_dir / "harnesses").glob("*.yaml")))},
        }
        return {
            "ok": True,
            "message": (
                f"Preflight: docker={docker_res.get('ok')} api={api_res.get('ok')} "
                f"levels={preflight['levels']['ok']} harness={preflight['harness']['ok']}."
            ),
            "data": preflight,
        }

    if tool_name == "configure_benchbot_provider":
        slot = max(1, min(int(args.get("slot", 1) or 1), 2))
        providers = list(assistant_state.get("providers", []))
        while len(providers) < slot:
            providers.append({"base_url": "", "model": "", "api_key": "", "label": ""})
        providers[slot - 1] = {
            "base_url": normalize_url(str(args.get("base_url", "")).strip()),
            "model": str(args.get("model", "")).strip(),
            "api_key": str(args.get("api_key", "")).strip(),
            "label": str(args.get("label", "")).strip() or str(args.get("model", "")).strip(),
        }
        assistant_state["providers"] = [provider for provider in providers if provider.get("base_url") and provider.get("model")]
        assistant_state["parallel_compare"] = bool(assistant_state["parallel_compare"] or len(assistant_state["providers"]) > 1 or slot == 2)
        state.save_provider_creds(assistant_state["providers"])
        ui_patch = assistant_state_ui_patch(assistant_state)
        return {
            "ok": True,
            "message": (
                f"Configured provider {slot} to use {providers[slot - 1]['model']} "
                f"at {providers[slot - 1]['base_url']}."
            ),
            "ui_patch": ui_patch,
            "data": {"providers": assistant_state["providers"]},
        }

    if tool_name == "configure_benchbot_run":
        if "level" in args and str(args.get("level", "")).strip():
            assistant_state["level"] = resolve_level_reference(state.project_dir, str(args["level"]))
            assistant_state["all_levels"] = False
        if "all_levels" in args:
            assistant_state["all_levels"] = bool(args.get("all_levels"))
            if assistant_state["all_levels"]:
                assistant_state["level"] = ""
        for key in (
            "capture_preview_screenshot",
            "save_result_bundle",
            "save_container_snapshot",
            "parallel_compare",
        ):
            if key in args:
                assistant_state[key] = bool(args.get(key))
        ui_patch = assistant_state_ui_patch(assistant_state)
        target = "all levels" if assistant_state.get("all_levels") else (assistant_state.get("level") or "current selection")
        return {
            "ok": True,
            "message": f"Updated run options for {target}.",
            "ui_patch": ui_patch,
            "data": {"run": ui_patch},
        }

    if tool_name == "start_benchbot_run":
        if "level" in args and str(args.get("level", "")).strip():
            assistant_state["level"] = resolve_level_reference(state.project_dir, str(args["level"]))
            assistant_state["all_levels"] = False
        if "all_levels" in args:
            assistant_state["all_levels"] = bool(args.get("all_levels"))
            if assistant_state["all_levels"]:
                assistant_state["level"] = ""
        for key in (
            "capture_preview_screenshot",
            "save_result_bundle",
            "save_container_snapshot",
        ):
            if key in args:
                assistant_state[key] = bool(args.get(key))
        run_req = build_run_request_from_assistant_state(assistant_state)
        payload, status = await _start_run_impl(run_req)
        ui_patch = assistant_state_ui_patch(assistant_state)
        event_type = "run_started" if status == 200 else "tool"
        message = "Benchmark run started." if status == 200 else payload.get("error", "Could not start run.")
        return {
            "ok": status == 200,
            "message": message,
            "event_type": event_type,
            "ui_patch": ui_patch,
            "data": payload,
        }

    if tool_name == "stop_benchbot_run":
        payload = await _stop_run_impl()
        return {
            "ok": True,
            "message": "Stopped the active benchmark batch.",
            "event_type": "run_stopped",
            "data": payload,
            "ui_patch": assistant_state_ui_patch(assistant_state),
        }

    if tool_name == "create_benchbot_level":
        patch = build_level_patch_from_args(args)
        content = render_level_yaml_from_patch(patch)
        save_requested = bool(args.get("save", page != "builder"))
        filename = str(args.get("filename", "")).strip() or f"{patch['id']}.yaml"
        saved_path = ""
        if save_requested:
            saved_path = str(save_level_content(state.project_dir, filename, content))
        else:
            validate_level_content(state.project_dir, filename, content)
        message = (
            f"Created level {patch['id']} and saved it to levels/{Path(saved_path).name}."
            if saved_path
            else f"Created a draft for level {patch['id']}."
        )
        payload = {
            "level_id": patch["id"],
            "filename": filename,
            "content": content,
            "saved_path": saved_path,
        }
        result = {
            "ok": True,
            "message": message,
            "event_type": "level_saved" if saved_path else "level_patch",
            "data": payload,
            "level_patch": patch,
        }
        if page == "builder":
            result["ui_patch"] = {"level_patch": patch}
        return result

    return {
        "ok": False,
        "message": f"Unknown dashboard assistant tool: {tool_name}",
        "data": {},
    }


# ── REST API ───────────────────────────────────────────────────────────────────

@app.get("/api/levels")
def list_levels() -> JSONResponse:
    levels = []
    for p in sorted((state.project_dir / "levels").glob("*.yaml")):
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
        except (LevelValidationError, FileNotFoundError) as exc:
            # Invalid or missing level file — return placeholder
            logger.debug("Skipping invalid level %s: %s", p.name, exc)
            levels.append({"path": str(p), "id": p.stem, "name": p.stem,
                           "difficulty": 1, "instruction": "", "tools": [],
                           "max_turns": "?", "timeout_s": "?",
                           "preview_port": "", "preview_path": "/"})
    return JSONResponse(levels)


@app.get("/api/credentials")
def get_credentials() -> JSONResponse:
    return JSONResponse(state.load_creds())


@app.post("/api/credentials")
async def save_credentials(req: RunRequest) -> JSONResponse:
    state.save_provider_creds(state.providers_from_request(req))
    return JSONResponse({"status": "saved"})


@app.get("/api/status")
def get_status() -> JSONResponse:
    alive = state.alive_procs()
    if alive:
        return JSONResponse({
            "status": "running",
            "count": len(alive),
            "pids": [proc.pid for proc in alive],
        })
    return JSONResponse({"status": "idle"})


@app.post("/api/run")
async def start_run(req: RunRequest) -> JSONResponse:
    payload, status = await _start_run_impl(req)
    return JSONResponse(payload, status_code=status)


async def _drain_runner(proc: subprocess.Popen, prefix: str = "") -> None:
    """Read runner subprocess stdout line-by-line into the rolling log buffer."""
    loop = asyncio.get_event_loop()
    while True:
        line: str = await loop.run_in_executor(None, proc.stdout.readline)
        if not line:
            break
        stripped = line.rstrip()
        state.record_runner_output(stripped, prefix=prefix)


@app.get("/api/runner-log")
def get_runner_log() -> JSONResponse:
    """Return the buffered stdout of the most recent runner subprocess."""
    return JSONResponse({"lines": list(state.runner_log)})


@app.get("/api/detect-providers")
async def detect_providers() -> JSONResponse:
    """
    Probe all known provider locations (env vars + common local ports).
    Returns a JSON array of available providers with pre-filled config.
    Used by the dashboard to show one-click provider presets.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, detect_providers_sync)
    return JSONResponse(result)


@app.get("/api/preview-status")
async def preview_status(port: int, path: str = "/") -> JSONResponse:
    """
    Check whether an HTTP server is listening on localhost:port.
    Called by the dashboard to show a 'waiting…' state until the agent's
    dev server comes up.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, probe_preview_status, port, path)
    return JSONResponse(result)


@app.get("/api/preflight")
async def preflight(base_url: str = "") -> JSONResponse:
    """Run all readiness checks and return pass/fail per check."""
    loop = asyncio.get_event_loop()

    # Docker ping runs blocking I/O — offload to thread pool
    docker_res = await loop.run_in_executor(None, check_docker)
    api_res = await loop.run_in_executor(
        None,
        check_api,
        normalize_url(base_url) if base_url else "",
    )

    levels = list((state.project_dir / "levels").glob("*.yaml"))
    harnesses = list((state.project_dir / "harnesses").glob("*.yaml"))

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
    return JSONResponse(state.store.get_summary() if state.store else {})

@app.get("/api/stats/models")
def stats_models() -> JSONResponse:
    return JSONResponse(state.store.get_model_stats() if state.store else [])

@app.get("/api/stats/levels")
def stats_levels() -> JSONResponse:
    return JSONResponse(state.store.get_level_stats() if state.store else [])

@app.get("/api/stats/model-detail/{model:path}")
def stats_model_detail(model: str) -> JSONResponse:
    """
    Full per-level breakdown for a single model.
    Used by the analytics Dex entry to render the level-conquest grid
    and rich stat display without re-querying the full model list.
    """
    if not state.store:
        return JSONResponse({"error": "store not available"}, status_code=503)
    data = state.store.get_model_detail(model)
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
    if not state.store:
        return JSONResponse([])
    runs = state.store.get_runs(
        limit=limit,
        offset=offset,
        model=model,
        level_id=level_id,
        min_stars=min_stars if min_stars >= 0 else None,
    )
    total = state.store.get_run_count(model=model, level_id=level_id)
    return JSONResponse({"runs": runs, "total": total, "limit": limit, "offset": offset})


@app.get("/api/runs/meta")
def runs_meta() -> JSONResponse:
    """
    Returns distinct models + levels stored in the DB.
    Used by the history UI to populate filter dropdowns.
    """
    if not state.store:
        return JSONResponse({"models": [], "levels": []})
    return JSONResponse({
        "models": state.store.get_distinct_models(),
        "levels": state.store.get_distinct_levels(),
    })


@app.get("/api/history")
def history_inventory(limit: int = 120) -> JSONResponse:
    """Return recent runs enriched with artifact and log metadata."""
    if not state.store:
        return JSONResponse({"runs": [], "total": 0})

    runs = state.store.get_runs(limit=max(1, limit), offset=0)
    items = build_history_inventory(runs, state.runs_dir)
    return JSONResponse({"runs": items, "total": len(items)})


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> JSONResponse:
    """
    Return full data for a single run identified by its 8-char hex id.
    Also returns the parsed agentlog events so the UI can replay the session.
    """
    if not state.store:
        return JSONResponse({"error": "store not available"}, status_code=503)

    run = state.store.get_run_by_id(run_id)
    if not run:
        return JSONResponse({"error": f"run {run_id!r} not found"}, status_code=404)

    # Try to locate and parse the agentlog file by run_id suffix
    events: list[dict] = []
    log_found = False
    try:
        log_path = find_run_log_path(state.runs_dir, run_id)
        if log_path is not None:
            from framework.recorder import load_agentlog
            events = load_agentlog(log_path)
            log_found = True
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        # Log file missing or corrupted — graceful degradation
        logger.warning("Could not load agentlog for run %s: %s", run_id, exc)

    return JSONResponse({
        "run":       run,
        "events":    events,
        "replay":    build_run_replay(events),
        "artifacts": build_artifact_records(state.runs_dir, run_id),
        "log_found": log_found,
    })


@app.get("/api/replays/recent")
def list_recent_replays(limit: int = 12) -> JSONResponse:
    log_paths = sorted(
        state.runs_dir.glob("*.agentlog"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[: max(1, limit)]

    items: list[dict[str, Any]] = []
    for log_path in log_paths:
        try:
            from framework.recorder import load_agentlog

            events = load_agentlog(log_path)
            if not events:
                continue
            run_id = str(events[0].get("run_id") or log_path.stem.split("_")[-1])
            db_run = state.store.get_run_by_id(run_id) if state.store else None
            items.append(build_replay_run_record(run_id, events, log_path, db_run=db_run))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            # Log file missing or corrupted — skip it
            logger.warning("Could not index replay log %s: %s", log_path, exc)

    return JSONResponse({"runs": items})


@app.get("/api/replays/{run_id}")
def get_replay(run_id: str) -> JSONResponse:
    payload, status = _load_replay_payload_for_run(run_id)
    return JSONResponse(payload, status_code=status)


@app.get("/api/compare")
def compare_runs(left_run_id: str, right_run_id: str) -> JSONResponse:
    left_payload, left_status = _load_replay_payload_for_run(left_run_id)
    if left_status != 200:
        return JSONResponse(left_payload, status_code=left_status)

    right_payload, right_status = _load_replay_payload_for_run(right_run_id)
    if right_status != 200:
        return JSONResponse(right_payload, status_code=right_status)

    return JSONResponse(build_compare_payload(left_payload, right_payload))


@app.get("/api/artifacts/{run_id}/{name:path}")
def get_artifact(run_id: str, name: str):
    artifact_root = (state.runs_dir / "artifacts" / run_id).resolve()
    candidate = (artifact_root / name).resolve()

    if artifact_root not in candidate.parents or not candidate.is_file():
        return JSONResponse({"error": "artifact not found"}, status_code=404)

    return FileResponse(candidate)


@app.get("/api/logs/{run_id}")
def get_run_log(run_id: str):
    log_path = find_run_log_path(state.runs_dir, run_id)
    if log_path is None:
        return JSONResponse({"error": "log not found"}, status_code=404)
    return FileResponse(log_path)


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
    creds = state.load_creds()
    base_url = req.base_url.strip() or creds.get("base_url", "") or os.getenv("BENCHBOT_BASE_URL", "")
    model    = req.model.strip()    or creds.get("model",    "") or os.getenv("BENCHBOT_MODEL",    "")
    api_key  = req.api_key.strip()  or creds.get("api_key",  "") or os.getenv("BENCHBOT_API_KEY",  "benchbot")

    if not base_url or not model:
        async def _err():
            yield "data: {\"error\": \"no endpoint configured\"}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    base_url = normalize_url(base_url)

    context_system = build_chat_context(
        store=state.store,
        project_dir=state.project_dir,
        runs_dir=state.runs_dir,
        active_run_id=req.active_run_id or "",
        page=req.page or "dashboard",
        page_context=req.page_context or "",
    )
    if req.page in ("dashboard", "builder") and req.allow_control:
        context_system += "\n\n" + assistant_control_prompt(req.page)
    messages = [{"role": "system", "content": context_system}] + list(req.messages)

    async def _stream():
        try:
            client = AgentAPI(
                base_url=base_url,
                api_key=api_key or "benchbot",
                model=model,
                temperature=0.3,
                max_tokens=1024,
                timeout=60.0,
            )
            if req.page not in ("dashboard", "builder") or not req.allow_control:
                for delta in client.stream_chat(messages):
                    if delta:
                        yield _json_sse({"delta": delta})
                return

            assistant_state = build_initial_assistant_state(req, creds)
            tool_summaries: list[str] = []
            for _ in range(6):
                response = client.chat(messages, tools=assistant_tool_schemas(req.page))
                choice = response["choices"][0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls") or []
                content = (message.get("content") or "").strip()

                if tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": tool_calls,
                        }
                    )
                    for idx, tool_call in enumerate(tool_calls):
                        fn = tool_call.get("function", {})
                        call_id = tool_call.get("id") or f"tool_{idx}"
                        tool_name = fn.get("name", "")
                        try:
                            args = json.loads(fn.get("arguments", "{}") or "{}")
                        except json.JSONDecodeError:
                            args = {}

                        result = await _execute_dashboard_assistant_tool(
                            tool_name,
                            args,
                            assistant_state=assistant_state,
                            page=req.page,
                        )
                        tool_summaries.append(result.get("message", ""))
                        yield _json_sse(
                            {
                                "_type": result.get("event_type", "tool"),
                                "tool": tool_name,
                                "message": result.get("message", ""),
                                "ui_patch": result.get("ui_patch"),
                                "data": result.get("data", {}),
                                "level_patch": result.get("level_patch"),
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": tool_name,
                                "content": json.dumps(result, ensure_ascii=False),
                            }
                        )
                    continue

                final_text = content or "\n".join(summary for summary in tool_summaries if summary).strip()
                if not final_text:
                    final_text = "No additional changes were needed."
                for chunk in _chunk_text(final_text):
                    yield _json_sse({"delta": chunk})
                break
            else:
                yield _json_sse({"error": "assistant control loop exceeded max tool steps"})
        except Exception as exc:
            logger.warning("chat stream error: %s", exc)
            yield _json_sse({"error": str(exc)})
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/api/stop")
async def stop_run() -> JSONResponse:
    return JSONResponse(await _stop_run_impl())


@app.get("/api/levels/{stem}/parsed")
def load_level_parsed(stem: str) -> JSONResponse:
    """Return a level YAML's fields as structured JSON for the builder to populate."""
    p = state.project_dir / "levels" / (stem if stem.endswith(".yaml") else stem + ".yaml")
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

@app.post("/api/levels/save")
async def save_level(req: SaveLevelRequest) -> JSONResponse:
    try:
        dest = save_level_content(
            state.project_dir,
            req.filename,
            req.content,
        )
        validation = lint_level_content(state.project_dir, req.filename, req.content)
        return JSONResponse({
            "status": "saved",
            "path": str(dest),
            "validation": validation,
            "warnings": validation.get("warnings", []),
        })
    except ValueError as exc:
        validation = lint_level_content(state.project_dir, req.filename, req.content)
        error_summary = (
            validation.get("errors", [str(exc)])[0]
            if validation.get("errors") == ["invalid filename"]
            else "Invalid level config"
        )
        return JSONResponse(
            {
                "error": error_summary,
                "errors": validation.get("errors", []),
                "warnings": validation.get("warnings", []),
                "validation": validation,
            },
            status_code=400,
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/levels/validate")
async def validate_level(req: SaveLevelRequest) -> JSONResponse:
    validation = lint_level_content(state.project_dir, req.filename, req.content)
    status_code = 200 if validation.get("valid") else 400
    return JSONResponse(validation, status_code=status_code)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    seen: set[Path] = set()
    tasks: dict[Path, asyncio.Task] = {}
    connected_at = time.time()
    try:
        while True:
            cutoff = max(connected_at, state.run_batch_started_at - 0.5)
            for path in sorted(state.runs_dir.glob("*.agentlog"), key=lambda p: p.stat().st_mtime):
                if path in seen:
                    continue
                try:
                    if path.stat().st_mtime < cutoff:
                        continue
                except FileNotFoundError:
                    continue
                seen.add(path)
                tasks[path] = asyncio.create_task(stream_agentlog(ws, path))

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

    state.apply_runtime_config(
        loaded_config,
        runs_override=args.runs,
    )

    print(f"\n  🎮  benchb0t → http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


# ── Frontend ───────────────────────────────────────────────────────────────────



@app.get("/builder", response_class=HTMLResponse)
async def builder() -> HTMLResponse:
    return HTMLResponse(_template("builder.html"))




@app.get("/analytics", response_class=HTMLResponse)
async def analytics() -> HTMLResponse:
    return HTMLResponse(_template("analytics.html"))


@app.get("/history", response_class=HTMLResponse)
async def history() -> HTMLResponse:
    return HTMLResponse(_template("history.html"))




if __name__ == "__main__":
    main()
