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
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from framework.store import Store

logger = logging.getLogger(__name__)

app = FastAPI(title="benchb0t", docs_url=None, redoc_url=None)

_runs_dir:    Path = Path("runs")
_project_dir: Path = Path(".")
_active_procs: list[subprocess.Popen] = []
_active_proc_lock = asyncio.Lock()
_run_batch_started_at = 0.0

_CREDS_FILE = Path(".benchb0t_creds.json")
_CREDS_KEYS = ("base_url", "model", "api_key", "providers")

_store: Store | None = None   # initialised in main()

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
            cfg  = yaml.safe_load(p.read_text())
            lvl  = cfg.get("level", {})
            task = cfg.get("task", {})
            prev = cfg.get("preview", {})
            levels.append({
                "path":         str(p),
                "id":           lvl.get("id", p.stem),
                "name":         lvl.get("name", p.stem),
                "difficulty":   lvl.get("difficulty", 1),
                "category":     lvl.get("category", ""),
                "instruction":  (task.get("instruction") or "").strip(),
                "tools":        cfg.get("tools", []),
                "max_turns":    task.get("max_turns", "?"),
                "timeout_s":    task.get("timeout_s", "?"),
                "preview_port": prev.get("port", ""),
                "preview_path": prev.get("path", "/"),
            })
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

@app.get("/api/runs")
def list_runs(limit: int = 200) -> JSONResponse:
    return JSONResponse(_store.get_runs(limit=limit) if _store else [])


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


# ── HTML ───────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(HTML)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    if not url.rstrip("/").endswith("/v1"):
        url = url.rstrip("/") + "/v1"
    return url


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--runs", default="runs")
    args = p.parse_args()

    global _runs_dir, _project_dir, _store
    _runs_dir    = Path(args.runs)
    _project_dir = Path(".").resolve()
    _runs_dir.mkdir(parents=True, exist_ok=True)

    _store = Store(_project_dir / "benchb0t.db").init()

    print(f"\n  🎮  benchb0t → http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


# ── Frontend ───────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>benchb0t</title>
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
<style>
:root {
  --bg:      #0a0800;
  --panel:   #100d00;
  --panel2:  #151000;
  --b1:      #241e00;
  --b2:      #3a3000;
  --b3:      #4a3f00;
  --y1:      #ffd700;   /* main yellow */
  --y2:      #ffb300;   /* amber */
  --y3:      #ff8c00;   /* deep amber / warning */
  --ydk:     #7a6000;   /* dark yellow for bars */
  --text:    #ffe87a;   /* warm text */
  --dim:     #5a4a10;
  --dim2:    #3a3000;
  --green:   #39ff14;   /* neon green — success only */
  --red:     #ff3a3a;   /* error only */
  --cyan:    #00e5ff;
  --cyan-dk: #007f8f;
  --violet:  #b36cff;
  --violet-dk:#5e2f91;
  --font:    'Press Start 2P', monospace;
  --term-bg: #050400;
  --term-y:  #ffc200;   /* amber terminal text */
  --term-dk: #1a1400;   /* dark amber for dim terminal text */
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--font); font-size: 10px; overflow: hidden; }

/* CRT scanlines */
body::after {
  content: ''; position: fixed; inset: 0; z-index: 9999; pointer-events: none;
  background: repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,.08) 3px, rgba(0,0,0,.08) 4px);
}

/* ── GRID ── */
.root { display: grid; grid-template-columns: 270px 1fr; grid-template-rows: 52px 1fr 38px; height: 100vh; }

/* ── HEADER ── */
header {
  grid-column: 1/-1;
  background: var(--panel);
  border-bottom: 3px solid var(--y1);
  display: flex; align-items: center; padding: 0 18px; gap: 14px;
  box-shadow: 0 2px 20px rgba(255,215,0,.15);
}
.logo {
  color: var(--y1); font-size: 14px; letter-spacing: 4px;
  text-shadow: 0 0 20px var(--y1), 0 0 50px rgba(255,215,0,.3);
}
.hsep { color: var(--b3); }
.hmeta { font-size: 6px; color: var(--dim); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dim); display: inline-block; margin-right: 6px; vertical-align: middle; }
.dot.live { background: var(--y1); box-shadow: 0 0 10px var(--y1); animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.2} }
.h-stars { margin-left: auto; color: var(--y1); font-size: 11px; text-shadow: 0 0 12px rgba(255,215,0,.6); }

/* ── SIDEBAR ── */
aside {
  background: var(--panel); border-right: 2px solid var(--b1);
  overflow-y: auto; padding: 12px 10px; display: flex; flex-direction: column; gap: 8px;
}
aside::-webkit-scrollbar { width: 3px; }
aside::-webkit-scrollbar-thumb { background: var(--b2); }

.card { border: 2px solid var(--b2); padding: 11px; background: var(--panel2); }
.ct { color: var(--dim); font-size: 6px; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 9px; }

/* status chip */
.rstatus {
  display: inline-block; padding: 3px 8px; font-size: 6px; letter-spacing: 2px;
  border: 1px solid var(--dim); color: var(--dim); margin-bottom: 9px; text-transform: uppercase;
}
.rstatus.running {
  border-color: var(--y1); color: var(--y1);
  box-shadow: 0 0 8px rgba(255,215,0,.3); animation: pulse 1.2s infinite;
}

/* form fields */
.field { margin-bottom: 9px; }
.field label { display: block; font-size: 6px; color: var(--dim); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
.field input, .field select {
  width: 100%; background: var(--bg); border: 2px solid var(--b2);
  color: var(--text); font-family: var(--font); font-size: 7px; padding: 7px 8px; outline: none;
  -webkit-appearance: none;
}
.field input:focus, .field select:focus { border-color: var(--y1); box-shadow: 0 0 0 1px rgba(255,215,0,.2); }
.field select {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%235a4a10'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: right 7px center; padding-right: 20px; cursor: pointer;
}
.field select option { background: var(--panel); }
.field-chk { display: flex; align-items: center; gap: 8px; margin-bottom: 9px; }
.field-chk input[type=checkbox] { width: 13px; height: 13px; accent-color: var(--y1); cursor: pointer; }
.field-chk label { font-size: 6px; color: var(--dim); cursor: pointer; text-transform: uppercase; letter-spacing: 1px; }
.provider-stack { display: flex; flex-direction: column; gap: 9px; margin-bottom: 8px; }
.provider-block {
  border: 1px solid var(--b2); background: rgba(0,0,0,.12);
  padding: 8px 8px 2px;
}
.provider-optional { display: none; }
.provider-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 7px;
}
.provider-label {
  font-size: 6px; color: var(--y2); letter-spacing: 2px; text-transform: uppercase;
}
.provider-note {
  font-size: 5px; color: var(--dim); letter-spacing: 1px; text-transform: uppercase;
}

/* START button — big, yellow, pixel press */
.btn-run {
  width: 100%; padding: 13px; font-family: var(--font); font-size: 9px; letter-spacing: 3px;
  background: var(--y1); color: #000; border: none; cursor: pointer;
  box-shadow: 0 5px 0 var(--ydk), 0 0 24px rgba(255,215,0,.2);
  transition: transform .08s, box-shadow .08s;
  position: relative;
}
.btn-run::before {
  content: ''; position: absolute; inset: 0;
  background: repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(0,0,0,.04) 4px, rgba(0,0,0,.04) 8px);
}
.btn-run:hover { filter: brightness(1.12); }
.btn-run:active { transform: translateY(3px); box-shadow: 0 2px 0 var(--ydk); }
.btn-run:disabled { background: var(--b2); color: var(--dim2); box-shadow: none; cursor: not-allowed; }
.btn-run:disabled::before { display: none; }

.btn-stop {
  width: 100%; padding: 8px; font-family: var(--font); font-size: 7px; letter-spacing: 2px;
  background: transparent; color: var(--y3); border: 2px solid var(--y3); cursor: pointer; margin-top: 7px;
  transition: background .15s;
}
.btn-stop:hover { background: rgba(255,140,0,.08); }
.btn-stop:disabled { opacity: .2; cursor: not-allowed; }

/* sidebar stats */
.kv { display: flex; justify-content: space-between; margin-bottom: 5px; }
.kv .k { font-size: 6px; color: var(--dim); }
.kv .v { font-size: 6px; color: var(--text); }

/* ── MAIN ── */
main { overflow: hidden; display: flex; flex-direction: column; background: var(--bg); }

.main-hdr {
  background: var(--panel); border-bottom: 2px solid var(--b1);
  padding: 7px 14px; display: flex; align-items: center; gap: 12px;
  font-size: 6px; color: var(--dim); flex-shrink: 0;
}
.run-total { margin-left: auto; color: var(--y1); font-size: 7px; }

/* benchmark rows */
.term-grid {
  flex: 1; display: flex; flex-direction: column;
  overflow: auto; gap: 12px; padding: 12px;
  align-items: flex-start;
}
.term-grid::-webkit-scrollbar { width: 4px; height: 4px; }
.term-grid::-webkit-scrollbar-thumb { background: var(--b2); }

.track-legend {
  display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  padding: 6px 8px; border: 1px solid var(--b1); background: rgba(0,0,0,.18);
}
.track-chip {
  font-size: 6px; letter-spacing: 1px; text-transform: uppercase;
  border: 1px solid currentColor; color: var(--dim);
  padding: 3px 8px; font-family: monospace;
}
.track-chip.provider-1 { color: var(--y1); }
.track-chip.provider-2 { color: var(--cyan); }
.track-chip.summary { color: var(--violet); }

.bench-row {
  width: 100%;
  border: 1px solid var(--b1);
  background: rgba(0,0,0,.14);
}
.row-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 8px; padding: 7px 10px; background: var(--panel);
  border-bottom: 1px solid var(--b1);
}
.row-title {
  font-size: 7px; color: var(--text); letter-spacing: 1px;
}
.row-meta {
  font-size: 6px; color: var(--dim); letter-spacing: 1px; text-transform: uppercase;
}
.row-panels {
  display: grid; gap: 12px; padding: 12px;
  overflow-x: auto;
  width: 100%;
}
.row-panels::-webkit-scrollbar { height: 4px; }
.row-panels::-webkit-scrollbar-thumb { background: var(--b2); }

/* ── TERMINAL PANEL ── */
.term {
  --accent: var(--y1);
  --accent-dk: var(--ydk);
  --accent-glow: rgba(255,215,0,.18);
  display: flex; flex-direction: column;
  min-width: 360px; width: 360px; flex-shrink: 0;
  border: 2px solid var(--b1);
  background: var(--term-bg);
}
.term.provider-2 {
  --accent: var(--cyan);
  --accent-dk: var(--cyan-dk);
  --accent-glow: rgba(0,229,255,.18);
}

/* title bar — thick yellow stripe when active */
.term-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 8px 0 10px; height: 34px;
  background: var(--panel); border-bottom: 2px solid var(--b1);
  flex-shrink: 0; gap: 6px;
}
.term.is-active .term-bar { border-bottom-color: var(--accent); box-shadow: 0 2px 12px var(--accent-glow); }
.term.is-done   .term-bar { border-bottom-color: var(--accent); }

.term-title { font-size: 7px; color: var(--dim); letter-spacing: 1px; }
.term.is-active .term-title { color: var(--text); }
.term.is-done   .term-title { color: var(--accent); }
.term-provider {
  font-size: 6px; color: var(--dim); font-family: monospace;
  letter-spacing: 1px;
}
.term.is-active .term-provider { color: var(--accent); }

.pill {
  font-size: 5px; padding: 2px 6px; letter-spacing: 1px;
  border: 1px solid var(--dim); color: var(--dim); text-transform: uppercase;
}
.pill.active { border-color: var(--accent); color: var(--accent); animation: pulse 1s infinite; }
.pill.done   { border-color: var(--y2); color: var(--y2); }
.pill.error  { border-color: var(--red); color: var(--red); }
.term.is-done .pill.done { border-color: var(--accent); color: var(--accent); }

/* output body */
.term-out {
  flex: 1; overflow-y: auto; overflow-x: hidden;
  padding: 8px 10px; font-family: monospace; font-size: 11px; line-height: 1.75;
  color: var(--term-y); background: var(--term-bg);
}
.term-out::-webkit-scrollbar { width: 3px; }
.term-out::-webkit-scrollbar-thumb { background: var(--b2); }

/* terminal lines */
.tl { display: flex; gap: 8px; }
.tl .ts  { color: var(--term-dk); font-size: 9px; flex-shrink: 0; user-select: none; }
.tl .lc  { flex: 1; word-break: break-all; }
.tl.info { color: var(--dim); font-size: 9px; }
.tl.call { color: var(--term-y); }
.tl.ok   { color: var(--green); }
.tl.fail { color: var(--red); }
.tl.done { color: var(--y1); font-size: 11px; }
.tl.msg  { color: #4a3f00; font-size: 9px; }
.tn  { color: var(--y1); }   /* tool name */
.arg { color: var(--dim); }  /* args */
.snip{ color: #3a3000; }     /* output snippet */

/* blinking cursor when active */
@keyframes cur { 0%,100%{opacity:1}50%{opacity:0} }
.cursor { display: inline-block; width: 7px; height: 11px; background: var(--term-y); animation: cur .8s infinite; vertical-align: middle; margin-left: 3px; }

/* .term-score and .score-* rules moved to SCORE BREAKDOWN block above */

/* ── FOOTER ── */
footer {
  grid-column: 1/-1; background: var(--panel); border-top: 2px solid var(--b1);
  display: flex; align-items: center; padding: 0 16px; gap: 16px; font-size: 6px; color: var(--dim);
}
.f-total { margin-left: auto; color: var(--y1); font-size: 7px; }

/* ── PREFLIGHT INDICATORS ── */
.pf-row {
  display: flex; align-items: center; gap: 7px;
  padding: 5px 0; border-bottom: 1px solid var(--b1);
}
.pf-row:last-child { border-bottom: none; }
.pf-dot {
  width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
  box-shadow: 0 0 5px currentColor;
}
.pf-dot.ok      { background: var(--green); color: var(--green); }
.pf-dot.fail    { background: var(--red);   color: var(--red); }
.pf-dot.warn    { background: var(--y3);    color: var(--y3); }
.pf-dot.pending { background: var(--dim);   color: var(--dim); animation: pf-pulse .9s infinite alternate; }
@keyframes pf-pulse { to { opacity: .2; } }
.pf-label { font-size: 6px; color: var(--dim); width: 46px; flex-shrink: 0; letter-spacing: 1px; }
.pf-msg   { font-size: 6px; color: var(--text); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pf-msg.fail { color: var(--red); }
.pf-refresh {
  background: none; border: 1px solid var(--b3); color: var(--dim);
  font-family: var(--font); font-size: 7px; cursor: pointer; padding: 1px 5px;
}
.pf-refresh:hover { border-color: var(--y1); color: var(--y1); }
.pf-blocked { font-size: 6px; color: var(--red); margin-top: 5px; letter-spacing: 1px; display: none; }

/* hazard stripe accent (decorative) */
.hazard {
  height: 4px; width: 100%;
  background: repeating-linear-gradient(90deg, var(--y1) 0, var(--y1) 12px, #000 12px, #000 24px);
  opacity: .35; flex-shrink: 0;
}

/* ── TITLE BAR HUD ── */
.term-bar-left  { display: flex; align-items: center; gap: 6px; }
.term-bar-right { display: flex; align-items: center; gap: 6px; }
.hud-turns {
  font-size: 6px; color: var(--dim); letter-spacing: 1px; display: none;
  font-family: monospace;
}
.hud-active {
  font-size: 6px; color: var(--accent); letter-spacing: 1px; display: none;
  max-width: 72px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  animation: pulse 0.9s infinite;
}

/* ── MISSION BRIEFING ── */
.briefing {
  flex-shrink: 0; border-bottom: 2px solid var(--b1);
  background: var(--panel2);
}
.brief-toggle {
  display: flex; align-items: center; gap: 7px;
  width: 100%; padding: 5px 10px;
  background: transparent; border: none; border-bottom: 1px solid transparent;
  font-family: var(--font); font-size: 6px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--dim); cursor: pointer; text-align: left;
  transition: color .15s;
}
.brief-toggle:hover { color: var(--accent); }
.brief-arr { color: var(--ydk); transition: transform .2s; }
.briefing.open .brief-arr { transform: rotate(90deg); }
.briefing.open .brief-toggle { border-bottom-color: var(--b1); }
.brief-body {
  display: none; padding: 6px 10px 8px;
}
.briefing.open .brief-body { display: block; }
.brief-instr {
  font-family: monospace; font-size: 10px; line-height: 1.65;
  color: var(--text); margin-bottom: 7px;
  max-height: 64px; overflow-y: auto;
  white-space: pre-wrap; word-break: break-word;
  border-left: 2px solid var(--ydk); padding-left: 8px;
}
.brief-instr::-webkit-scrollbar { width: 2px; }
.brief-instr::-webkit-scrollbar-thumb { background: var(--b2); }
.brief-footer { display: flex; justify-content: space-between; align-items: center; gap: 6px; }
.tool-badges  { display: flex; gap: 4px; flex-wrap: wrap; }
.tool-badge {
  font-size: 6px; padding: 2px 6px; letter-spacing: 1px;
  border: 1px solid var(--ydk); color: var(--ydk);
  font-family: monospace; cursor: default;
  transition: border-color .15s, color .15s;
}
.tool-badge:hover { border-color: var(--y2); color: var(--y2); }
.tool-badge.active-tool { border-color: var(--accent); color: var(--accent); animation: pulse 0.8s infinite; }
.brief-limits { font-size: 6px; color: var(--dim2); white-space: nowrap; }

/* ── AGENT SPEECH BUBBLES ── */
.bubble {
  background: var(--panel2); border: 2px solid var(--b2);
  padding: 7px 10px 7px 28px; margin: 6px 0 4px;
  font-family: monospace; font-size: 10px; color: var(--text); line-height: 1.6;
  position: relative; word-break: break-word;
}
.bubble::before {
  content: '🤖'; position: absolute;
  left: 6px; top: 6px; font-size: 11px;
}
/* pixel-art pointer triangle */
.bubble::after {
  content: ''; position: absolute;
  left: -6px; top: 10px;
  border-top: 4px solid transparent; border-bottom: 4px solid transparent;
  border-right: 6px solid var(--b2);
}
.term.is-active .bubble { border-color: var(--b3); }
.term.is-active .bubble::after { border-right-color: var(--b3); }
.bubble.streaming { border-color: var(--accent); }
.bubble.streaming::after { border-right-color: var(--accent); }
.bubble-trunc { color: var(--dim); font-size: 9px; cursor: pointer; }
.bubble-trunc:hover { color: var(--y2); }

/* ── TOOL CALL CARDS ── */
.call-card {
  border-left: 3px solid var(--b3);
  background: rgba(5,4,0,.7);
  padding: 5px 8px; margin: 3px 0;
  font-family: monospace; font-size: 9px;
  transition: border-color .3s;
}
.call-card.running { border-left-color: var(--accent); }
.call-card.ok      { border-left-color: var(--green); }
.call-card.fail    { border-left-color: var(--red); }
.cc-head  { display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px; }
.cc-tool  { color: var(--accent); font-size: 10px; letter-spacing: 1px; }
.cc-spin  { font-size: 8px; color: var(--dim); }
.cc-args  { color: var(--dim); word-break: break-all; margin-bottom: 3px; font-size: 8px; }
.cc-args .cc-key   { color: var(--b3); }
.cc-args .cc-val   { color: var(--dim); }
.cc-result { color: var(--term-dk); font-size: 8px; margin-top: 2px; }
.cc-snip   { color: #2a2400; }
.cc-ok     { color: var(--green); }
.cc-fail   { color: var(--red); }

/* ── ARTIFACT CARDS (write_file output) ── */
.artifact-card {
  border: 2px solid var(--b2); margin: 6px 0;
  background: var(--panel2); font-family: monospace;
}
.artifact-hdr {
  display: flex; align-items: center; justify-content: space-between;
  padding: 4px 8px; background: var(--panel);
  border-bottom: 1px solid var(--b1);
  font-size: 8px; color: var(--y2); letter-spacing: 1px;
}
.artifact-hdr .artifact-path { color: var(--y1); }
.artifact-hdr .artifact-bytes { color: var(--dim); font-size: 7px; }
.artifact-body {
  padding: 6px 8px; font-size: 9px; line-height: 1.6;
  color: var(--term-y); white-space: pre; overflow-x: auto;
  max-height: 120px; overflow-y: auto;
}
.artifact-body::-webkit-scrollbar { width: 2px; height: 2px; }
.artifact-body::-webkit-scrollbar-thumb { background: var(--b2); }
.artifact-toggle {
  width: 100%; padding: 3px 8px; background: transparent; border: none;
  border-top: 1px solid var(--b1); font-family: var(--font); font-size: 6px;
  color: var(--dim); cursor: pointer; text-align: center; letter-spacing: 1px;
}
.artifact-toggle:hover { color: var(--y2); }

/* ── HTTP PREVIEW CARD ── */
.http-card {
  border: 2px solid var(--b3); margin: 6px 0;
  background: var(--panel2); font-family: monospace;
}
.http-hdr {
  display: flex; align-items: center; gap: 8px;
  padding: 4px 8px; background: var(--panel);
  border-bottom: 1px solid var(--b1);
  font-size: 8px; color: var(--y2);
}
.http-method { color: var(--y1); font-size: 9px; }
.http-status { padding: 1px 5px; border: 1px solid; font-size: 7px; }
.http-status.ok   { border-color: var(--green); color: var(--green); }
.http-status.fail { border-color: var(--red);   color: var(--red); }
.http-preview {
  padding: 6px 8px; font-size: 9px; line-height: 1.6; color: var(--term-y);
  white-space: pre; overflow-x: auto; max-height: 100px; overflow-y: auto;
}
.http-preview-frame {
  width: 100%; height: 220px; border: none;
  background: #fff;
}

/* ── LIVE PREVIEW STRIP ── */
.preview-strip {
  flex-shrink: 0;
  border-top: 2px solid var(--b2);
  display: flex;
  flex-direction: column;
}
.preview-strip-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 10px;
  background: var(--panel);
  font-size: 8px;
  color: var(--y2);
  letter-spacing: 0.5px;
}
.preview-reload-btn {
  background: none;
  border: 1px solid var(--b3);
  color: var(--dim);
  font-family: inherit;
  font-size: 8px;
  cursor: pointer;
  padding: 2px 6px;
}
.preview-reload-btn:hover { border-color: var(--y1); color: var(--y1); }
.term.provider-2 .preview-reload-btn:hover { border-color: var(--cyan); color: var(--cyan); }
.preview-frame { transition: opacity 1s; }
.preview-frame {
  width: 100%;
  height: 260px;
  border: none;
  background: #fff;
  display: block;
}

/* ── SCORE BREAKDOWN ── */
.term-score { flex-shrink: 0; background: var(--panel); border-top: 2px solid var(--b1); padding: 6px 10px; }
.term.is-done .term-score { border-top-color: var(--accent); }
.score-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px; }
.score-stars { font-size: 13px; letter-spacing: 3px; color: var(--b3); }
.score-stars .s-filled { color: var(--accent); text-shadow: 0 0 8px var(--accent-glow); }
.score-num { font-size: 12px; color: var(--dim); }
.score-num.has-score { color: var(--accent); text-shadow: 0 0 10px var(--accent-glow); }
.score-bar-o { height: 6px; background: var(--bg); border: 1px solid var(--b2); margin-bottom: 8px; }
.score-bar-i { height: 100%; width: 0; background: linear-gradient(90deg, var(--accent-dk), var(--accent)); transition: width .8s ease; }
@keyframes starpop { 0%{transform:scale(0) rotate(-30deg);opacity:0} 65%{transform:scale(1.3)} 100%{transform:scale(1);opacity:1} }
.star-anim { display: inline-block; animation: starpop .3s cubic-bezier(.34,1.56,.64,1) both; }

.score-breakdown { border-top: 1px solid var(--b1); padding-top: 7px; margin-top: 4px; }

/* ── SUMMARY CARDS ── */
.summary-card,
.batch-summary {
  min-width: 290px;
  background: linear-gradient(180deg, rgba(179,108,255,.12), rgba(10,8,0,.85));
  border: 2px solid rgba(179,108,255,.35);
  color: #f0deff;
  padding: 12px;
}
.summary-card:not(.pending),
.batch-summary:not(.pending) {
  border-color: var(--violet);
  box-shadow: 0 0 20px rgba(179,108,255,.16);
}
.summary-card.pending,
.batch-summary.pending {
  background: linear-gradient(180deg, rgba(94,47,145,.12), rgba(10,8,0,.82));
  border-color: rgba(179,108,255,.22);
}
.summary-head,
.batch-summary-head {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  margin-bottom: 10px;
}
.summary-kicker,
.batch-summary-kicker {
  font-size: 6px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--violet);
}
.summary-chip,
.batch-summary-chip {
  font-size: 5px; letter-spacing: 1px; text-transform: uppercase;
  color: #e6cfff; border: 1px solid rgba(179,108,255,.45);
  padding: 2px 6px;
}
.summary-title,
.batch-summary-title {
  font-size: 8px; color: #fff; margin-bottom: 8px; line-height: 1.5;
}
.summary-empty,
.batch-summary-empty {
  font-size: 7px; color: #caa8ef; line-height: 1.7;
}
.summary-provider-list,
.batch-provider-list {
  display: flex; flex-direction: column; gap: 8px;
}
.summary-provider-entry,
.batch-provider-entry {
  border-left: 3px solid var(--violet);
  padding: 6px 8px; background: rgba(0,0,0,.22);
}
.summary-provider-entry.provider-1,
.batch-provider-entry.provider-1 { border-left-color: var(--y1); }
.summary-provider-entry.provider-2,
.batch-provider-entry.provider-2 { border-left-color: var(--cyan); }
.summary-provider-head,
.batch-provider-head {
  display: flex; justify-content: space-between; gap: 8px;
  font-size: 7px; color: #fff;
}
.summary-provider-meta,
.batch-provider-meta {
  margin-top: 4px; font-size: 6px; color: #cfb4f3; line-height: 1.6;
}
.summary-verdict,
.batch-verdict {
  margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(179,108,255,.28);
  font-size: 7px; color: #fff;
}
.batch-summary {
  width: 100%;
  min-width: 0;
  box-shadow: 0 0 24px rgba(179,108,255,.18);
}

/* dimension bars */
.dim-row { display: flex; align-items: center; gap: 5px; margin-bottom: 4px; }
.dim-label { font-size: 5px; color: var(--dim); letter-spacing: 1px; width: 52px; flex-shrink: 0; }
.dim-wt    { font-size: 5px; color: var(--dim2); width: 18px; flex-shrink: 0; text-align: right; }
.dim-bar-o { flex: 1; height: 5px; background: var(--bg); border: 1px solid var(--b2); }
.dim-bar-i { height: 100%; background: linear-gradient(90deg, var(--ydk), var(--y2)); transition: width .6s ease; }
.dim-bar-i.full   { background: linear-gradient(90deg, #39aa00, var(--green)); }
.dim-bar-i.zero   { background: var(--red); }
.dim-pct   { font-size: 5px; color: var(--text); width: 26px; flex-shrink: 0; text-align: right; }
.pen-row   { display: flex; align-items: center; gap: 5px; margin-top: 3px; }
.pen-label { font-size: 5px; letter-spacing: 1px; color: var(--red); width: 52px; flex-shrink: 0; }
.pen-val   { font-size: 5px; color: var(--red); margin-left: auto; }

/* per-criterion chips */
.criteria-row { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; padding-top: 5px; border-top: 1px solid var(--b1); }
.crit-chip {
  font-size: 5px; padding: 2px 5px; letter-spacing: 1px;
  border: 1px solid; font-family: monospace;
  max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  cursor: default;
}
.crit-chip.pass { border-color: var(--green); color: var(--green); }
.crit-chip.fail { border-color: var(--red);   color: var(--red); }
</style>
</head>
<body>
<div class="root">

<!-- HEADER -->
<header>
  <span class="logo">benchb0t</span>
  <span class="hsep">│</span>
  <span><span class="dot" id="dot"></span><span class="hmeta" id="conn-lbl">CONNECTING</span></span>
  <span class="hsep">│</span>
  <span class="hmeta" id="log-file">—</span>
  <span class="h-stars" id="h-stars"></span>
  <div style="margin-left:auto;display:flex;gap:8px">
    <a href="/builder"   style="font-size:6px;letter-spacing:2px;color:var(--dim);text-decoration:none;border:1px solid var(--dim);padding:3px 8px" onmouseover="this.style.color='var(--y1)';this.style.borderColor='var(--y1)'" onmouseout="this.style.color='var(--dim)';this.style.borderColor='var(--dim)'">BUILDER ↗</a>
    <a href="/analytics" style="font-size:6px;letter-spacing:2px;color:var(--dim);text-decoration:none;border:1px solid var(--dim);padding:3px 8px" onmouseover="this.style.color='var(--y1)';this.style.borderColor='var(--y1)'" onmouseout="this.style.color='var(--dim)';this.style.borderColor='var(--dim)'">ANALYTICS ↗</a>
  </div>
</header>

<!-- SIDEBAR -->
<aside>
  <!-- PREFLIGHT STATUS CARD -->
  <div class="card" id="pf-card" style="margin-bottom:10px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
      <span class="ct" style="margin-bottom:0">Readiness</span>
      <button class="pf-refresh" onclick="runPreflight()" title="Re-check">↺</button>
    </div>
    <div class="pf-row">
      <span class="pf-dot pending" id="pf-docker-dot"></span>
      <span class="pf-label">Docker</span>
      <span class="pf-msg" id="pf-docker-msg">checking…</span>
    </div>
    <div class="pf-row">
      <span class="pf-dot pending" id="pf-api-dot"></span>
      <span class="pf-label">API</span>
      <span class="pf-msg" id="pf-api-msg">checking…</span>
    </div>
    <div class="pf-row">
      <span class="pf-dot pending" id="pf-levels-dot"></span>
      <span class="pf-label">Levels</span>
      <span class="pf-msg" id="pf-levels-msg">checking…</span>
    </div>
    <div class="pf-row">
      <span class="pf-dot pending" id="pf-harness-dot"></span>
      <span class="pf-label">Harness</span>
      <span class="pf-msg" id="pf-harness-msg">checking…</span>
    </div>
    <div class="pf-blocked" id="pf-blocked">▲ FIX ABOVE BEFORE RUNNING</div>
  </div>

  <div class="card" style="border-color:var(--b3)">
    <div class="ct">▶ Launch</div>
    <div id="run-status" class="rstatus">IDLE</div>

    <div class="provider-stack">
      <div class="provider-block">
        <div class="provider-head">
          <span class="provider-label">Provider 1</span>
          <span class="provider-note">required</span>
        </div>
        <div class="field"><label>Base URL</label>
          <input id="f-url" type="text" placeholder="svslai02:8080"></div>
        <div class="field"><label>Model</label>
          <input id="f-model" type="text" placeholder="llama3"></div>
        <div class="field"><label>API Key</label>
          <input id="f-key" type="password" placeholder="(empty for local)"></div>
      </div>

      <div class="field-chk" style="margin-bottom:0">
        <input type="checkbox" id="f-parallel">
        <label for="f-parallel">Parallel compare with second provider</label>
      </div>

      <div class="provider-block provider-optional" id="provider-2-wrap">
        <div class="provider-head">
          <span class="provider-label">Provider 2</span>
          <span class="provider-note">optional</span>
        </div>
        <div class="field"><label>Base URL</label>
          <input id="f-url-2" type="text" placeholder="api.openai.com/v1"></div>
        <div class="field"><label>Model</label>
          <input id="f-model-2" type="text" placeholder="gpt-4.1"></div>
        <div class="field"><label>API Key</label>
          <input id="f-key-2" type="password" placeholder="sk-..."></div>
      </div>
    </div>

    <div class="field"><label>Level</label>
      <select id="f-level"></select></div>
    <div class="field-chk">
      <input type="checkbox" id="f-all">
      <label for="f-all">All levels</label>
    </div>
    <div class="hazard" style="margin-bottom:10px"></div>
    <button class="btn-run" id="btn-run" onclick="startRun()">▶ &nbsp;START</button>
    <button class="btn-stop" id="btn-stop" onclick="stopRun()" disabled>■ &nbsp;STOP</button>
  </div>

  <div class="card" id="side-live" style="display:none">
    <div class="ct">Live</div>
    <div class="kv"><span class="k">Provider</span> <span class="v" id="sl-provider">—</span></div>
    <div class="kv"><span class="k">Level</span> <span class="v" id="sl-level">—</span></div>
    <div class="kv"><span class="k">Turns</span> <span class="v" id="sl-turns">0</span></div>
    <div class="kv"><span class="k">Tools</span> <span class="v" id="sl-tools">0</span></div>
    <div class="kv"><span class="k">Time</span>  <span class="v" id="sl-time">—</span></div>
  </div>

  <!-- Runner error log — shown only when runner exits with errors -->
  <div id="runner-err-card" style="display:none;margin-top:10px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
      <span class="ct" style="color:var(--red);margin-bottom:0">⚠ RUNNER ERROR</span>
      <button id="runner-err-copy" onclick="copyRunnerErr()" style="
        background:none;border:1px solid #ff3a3a55;color:var(--red);
        font-family:var(--font);font-size:6px;cursor:pointer;padding:2px 6px;
      ">📋 COPY</button>
    </div>
    <pre id="runner-err-pre" style="
      font-family:var(--font);font-size:6px;line-height:1.7;
      color:#ff8080;background:rgba(255,0,0,.06);border:1px solid #ff3a3a55;
      padding:8px;margin:0;overflow-x:auto;white-space:pre-wrap;
      max-height:260px;overflow-y:auto;
    "></pre>
  </div>
</aside>

<!-- MAIN -->
<main>
  <div class="main-hdr">
    <span id="run-label">no run yet</span>
    <span class="run-total" id="run-total"></span>
    <button onclick="checkRunnerLog();document.getElementById('runner-err-card').style.display='block'"
      style="margin-left:auto;background:none;border:1px solid var(--b3);color:var(--dim);
             font-family:var(--font);font-size:6px;cursor:pointer;padding:2px 7px"
      title="Show runner output log">📋 LOG</button>
  </div>
  <div class="term-grid" id="term-grid"></div>
</main>

<!-- FOOTER -->
<footer>
  <span>benchb0t v0.1.0</span>
  <span class="hsep">│</span>
  <span id="f-runid">—</span>
  <span class="f-total" id="f-total"></span>
</footer>

</div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let runStatus = 'idle';
let allLevels = [];
let terminals = {};       // id → terminal state
let totalStars = 0;
let timerInt   = null;
let curLevelId = null;
let batchState = null;

function providerSuffix(slot) {
  return slot === 1 ? '' : '-2';
}

function setProviderFields(slot, provider) {
  const suffix = providerSuffix(slot);
  document.getElementById('f-url' + suffix).value   = provider?.base_url || '';
  document.getElementById('f-model' + suffix).value = provider?.model || '';
  document.getElementById('f-key' + suffix).value   = provider?.api_key || '';
}

function getProviderConfig(slot) {
  const suffix = providerSuffix(slot);
  return {
    base_url: document.getElementById('f-url' + suffix).value.trim(),
    model: document.getElementById('f-model' + suffix).value.trim(),
    api_key: document.getElementById('f-key' + suffix).value.trim(),
    label: document.getElementById('f-model' + suffix).value.trim(),
  };
}

function toggleSecondaryProvider(force) {
  const enabled = (force !== undefined) ? force : document.getElementById('f-parallel').checked;
  document.getElementById('provider-2-wrap').style.display = enabled ? 'block' : 'none';
}

function buildPanels(levels, providers) {
  return providers.flatMap((provider, idx) => levels.map(level => ({
    ...level,
    id: `p${idx + 1}--${level.id}`,
    levelId: level.id,
    providerSlot: idx + 1,
    providerLabel: provider.label || provider.model,
    providerTitle: providers.length > 1 ? `P${idx + 1} · ${provider.label || provider.model}` : (provider.label || provider.model),
  })));
}

function providerClass(slot) {
  return 'provider-' + slot;
}

function initBatchState(panels) {
  const providerMap = new Map();
  const rows = {};

  panels.forEach(panel => {
    if (!providerMap.has(panel.providerSlot)) {
      providerMap.set(panel.providerSlot, {
        slot: panel.providerSlot,
        label: panel.providerLabel || panel.providerTitle || `Provider ${panel.providerSlot}`,
      });
    }
    if (!rows[panel.levelId]) {
      rows[panel.levelId] = {
        levelId: panel.levelId,
        levelName: panel.name,
        difficulty: panel.difficulty,
        results: {},
      };
    }
  });

  batchState = {
    rows,
    providerMeta: Array.from(providerMap.values()).sort((a, b) => a.slot - b.slot),
    expectedPanels: panels.length,
    completedPanels: 0,
  };
}

function renderTrackLegend() {
  if (!batchState?.providerMeta?.length) return '';
  const tracks = batchState.providerMeta.map(meta =>
    `<span class="track-chip ${providerClass(meta.slot)}">Lane ${meta.slot} · ${esc(meta.label)}</span>`
  ).join('');
  return `<div class="track-legend">${tracks}<span class="track-chip summary">Summary</span></div>`;
}

function renderSummaryShell(levelId, levelName) {
  return `
    <aside class="summary-card pending" id="summary-${levelId}">
      <div class="summary-head">
        <span class="summary-kicker">Level Summary</span>
        <span class="summary-chip" id="summary-chip-${levelId}">waiting</span>
      </div>
      <div class="summary-title">${esc(levelName)}</div>
      <div id="summary-body-${levelId}">
        <div class="summary-empty">Waiting for provider results…</div>
      </div>
    </aside>`;
}

function renderBatchSummaryShell() {
  return `
    <section class="batch-summary pending" id="batch-summary">
      <div class="batch-summary-head">
        <span class="batch-summary-kicker">Benchmark Summary</span>
        <span class="batch-summary-chip" id="batch-summary-chip">running</span>
      </div>
      <div class="batch-summary-title">Overall comparison appears here after the full batch completes.</div>
      <div id="batch-summary-body">
        <div class="batch-summary-empty">Waiting for completed rows…</div>
      </div>
    </section>`;
}

function renderRowSummaryBody(row) {
  const providerEntries = batchState.providerMeta.map(meta => {
    const result = row.results[meta.slot];
    if (!result) {
      return `
        <div class="summary-provider-entry ${providerClass(meta.slot)}">
          <div class="summary-provider-head">
            <span>${esc(meta.label)}</span>
            <span>pending</span>
          </div>
          <div class="summary-provider-meta">Waiting for this provider to finish this level.</div>
        </div>`;
    }
    return `
      <div class="summary-provider-entry ${providerClass(meta.slot)}">
        <div class="summary-provider-head">
          <span>${esc(meta.label)}</span>
          <span>${result.total.toFixed(1)} / 100</span>
        </div>
        <div class="summary-provider-meta">
          ${renderStarsPlain(result.stars)} · ${result.turns} turns · ${result.tools} tools${result.timedOut ? ' · timeout' : ''}
        </div>
      </div>`;
  }).join('');

  const doneCount = Object.keys(row.results).length;
  const expected = batchState.providerMeta.length;
  if (expected === 1 || doneCount < expected) {
    return `<div class="summary-provider-list">${providerEntries}</div>`;
  }

  const ranked = Object.values(row.results).sort((a, b) => b.total - a.total);
  const spread = ranked.length > 1 ? (ranked[0].total - ranked[1].total).toFixed(1) : '0.0';
  return `
    <div class="summary-provider-list">${providerEntries}</div>
    <div class="summary-verdict">
      Winner: ${esc(ranked[0].label)} by ${spread} pts
    </div>`;
}

function updateLevelSummary(levelId) {
  if (!batchState?.rows?.[levelId]) return;
  const row = batchState.rows[levelId];
  const body = document.getElementById('summary-body-' + levelId);
  const chip = document.getElementById('summary-chip-' + levelId);
  const card = document.getElementById('summary-' + levelId);
  if (!body || !chip || !card) return;

  const doneCount = Object.keys(row.results).length;
  const expected = batchState.providerMeta.length;
  chip.textContent = doneCount >= expected ? 'complete' : `${doneCount}/${expected} done`;
  card.classList.toggle('pending', doneCount < expected);
  body.innerHTML = renderRowSummaryBody(row);
}

function renderBatchSummaryBody() {
  const rows = Object.values(batchState?.rows || {});
  const providerStats = batchState.providerMeta.map(meta => {
    const results = rows.map(row => row.results[meta.slot]).filter(Boolean);
    const totalStarsLocal = results.reduce((sum, r) => sum + r.stars, 0);
    const avg = results.length ? results.reduce((sum, r) => sum + r.total, 0) / results.length : 0;
    const best = results.length ? Math.max(...results.map(r => r.total)) : 0;
    return {
      ...meta,
      avg,
      best,
      totalStars: totalStarsLocal,
      count: results.length,
    };
  });

  const providerList = providerStats.map(stat => `
    <div class="batch-provider-entry ${providerClass(stat.slot)}">
      <div class="batch-provider-head">
        <span>${esc(stat.label)}</span>
        <span>${stat.avg.toFixed(1)} avg</span>
      </div>
      <div class="batch-provider-meta">
        ${stat.totalStars}★ total · ${stat.best.toFixed(1)} best · ${stat.count} finished panels
      </div>
    </div>
  `).join('');

  const ranked = [...providerStats].sort((a, b) => b.avg - a.avg);
  const verdict = ranked.length > 1
    ? `Best overall: ${esc(ranked[0].label)} leads by ${(ranked[0].avg - ranked[1].avg).toFixed(1)} avg points`
    : `Completed ${rows.length} level${rows.length === 1 ? '' : 's'} on ${esc(ranked[0]?.label || 'provider')}`;

  return `
    <div class="batch-provider-list">${providerList}</div>
    <div class="batch-verdict">${verdict}</div>
  `;
}

function updateBatchSummary() {
  const wrap = document.getElementById('batch-summary');
  const body = document.getElementById('batch-summary-body');
  const chip = document.getElementById('batch-summary-chip');
  if (!wrap || !body || !chip || !batchState) return;

  const complete = batchState.completedPanels >= batchState.expectedPanels && batchState.expectedPanels > 0;
  chip.textContent = complete ? 'complete' : `${batchState.completedPanels}/${batchState.expectedPanels} done`;
  wrap.classList.toggle('pending', !complete);
  if (!complete) {
    body.innerHTML = `<div class="batch-summary-empty">Waiting for all provider lanes to finish before computing the final comparison.</div>`;
    return;
  }
  body.innerHTML = renderBatchSummaryBody();
  document.getElementById('run-label').textContent = 'bench complete';
  document.getElementById('run-label').style.color = 'var(--violet)';
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
  await loadCreds();
  await loadLevels();
  await runPreflight();
}

async function loadCreds() {
  try {
    const c = await fetch('/api/credentials').then(r => r.json());
    const providers = Array.isArray(c.providers) && c.providers.length
      ? c.providers
      : [{ base_url: c.base_url || '', model: c.model || '', api_key: c.api_key || '' }];
    setProviderFields(1, providers[0] || {});
    const hasSecond = !!providers[1];
    document.getElementById('f-parallel').checked = hasSecond;
    toggleSecondaryProvider(hasSecond);
    setProviderFields(2, providers[1] || {});
  } catch(e) {}
}

// ── Preflight ─────────────────────────────────────────────────────────────────
let _pfDebounce = null;

function setPfRow(key, result) {
  const dot = document.getElementById('pf-' + key + '-dot');
  const msg = document.getElementById('pf-' + key + '-msg');
  if (!dot || !msg) return;
  const ok = result.ok;
  dot.className = 'pf-dot ' + (ok === true ? 'ok' : ok === false ? 'fail' : 'warn');
  msg.textContent = result.msg || '';
  msg.className   = 'pf-msg' + (ok === false ? ' fail' : '');
}

async function runPreflight() {
  // Set all to pending while checking
  ['docker','api','levels','harness'].forEach(k => {
    const d = document.getElementById('pf-' + k + '-dot');
    const m = document.getElementById('pf-' + k + '-msg');
    if (d) d.className = 'pf-dot pending';
    if (m) { m.textContent = 'checking…'; m.className = 'pf-msg'; }
  });

  const url = getProviderConfig(1).base_url || '';
  try {
    const qs  = url ? '?base_url=' + encodeURIComponent(url) : '';
    const res = await fetch('/api/preflight' + qs).then(r => r.json());
    setPfRow('docker',  res.docker);
    setPfRow('api',     res.api);
    setPfRow('levels',  res.levels);
    setPfRow('harness', res.harness);

    // Block run button if critical checks fail
    const critical = res.docker.ok && res.levels.ok && res.harness.ok;
    const btn = document.getElementById('btn-run');
    const blk = document.getElementById('pf-blocked');
    if (!critical && runStatus !== 'running') {
      if (btn) btn.disabled = true;
      if (blk) blk.style.display = 'block';
    } else {
      if (btn && runStatus !== 'running') btn.disabled = false;
      if (blk) blk.style.display = 'none';
    }

    // Color the preflight card border
    const card = document.getElementById('pf-card');
    if (card) card.style.borderColor = critical ? 'var(--b2)' : 'var(--red)';
  } catch(e) {
    ['docker','api','levels','harness'].forEach(k => {
      setPfRow(k, { ok: null, msg: 'check failed' });
    });
  }
}

// Re-check when URL field changes (debounced 1.2s)
document.addEventListener('DOMContentLoaded', () => {
  const urlInput = document.getElementById('f-url');
  const urlInput2 = document.getElementById('f-url-2');
  const parallel = document.getElementById('f-parallel');
  if (parallel) {
    parallel.addEventListener('change', () => toggleSecondaryProvider());
  }
  if (urlInput) {
    urlInput.addEventListener('input', () => {
      clearTimeout(_pfDebounce);
      _pfDebounce = setTimeout(runPreflight, 1200);
    });
  }
  if (urlInput2) {
    urlInput2.addEventListener('input', () => {
      clearTimeout(_pfDebounce);
      _pfDebounce = setTimeout(runPreflight, 1200);
    });
  }
});

// Light background re-check every 30s
setInterval(runPreflight, 30_000);

async function loadLevels() {
  allLevels = await fetch('/api/levels').then(r => r.json());
  const sel = document.getElementById('f-level');
  const stars = ['','★','★★','★★★','★★★★','★★★★★'];
  sel.innerHTML = allLevels.map(l =>
    `<option value="${l.path}">${stars[l.difficulty]||'?'} ${l.name}</option>`
  ).join('');
}

// ── Run control ───────────────────────────────────────────────────────────────
async function startRun() {
  const providers = [getProviderConfig(1)];
  const parallel = document.getElementById('f-parallel').checked;
  if (!providers[0].base_url || !providers[0].model) {
    alert('Fill in Base URL and Model for Provider 1');
    return;
  }
  if (parallel) {
    const provider2 = getProviderConfig(2);
    if (!provider2.base_url || !provider2.model) {
      alert('Fill in Base URL and Model for Provider 2');
      return;
    }
    providers.push(provider2);
  }
  const level  = document.getElementById('f-level').value;
  const allLvl = document.getElementById('f-all').checked;

  // Build terminals
  const toRun = allLvl ? allLevels : allLevels.filter(l => l.path === level);
  buildTerminals(buildPanels(toRun, providers));

  document.getElementById('run-label').textContent = 'running…';
  document.getElementById('run-label').style.color = 'var(--y2)';

  const res  = await fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      base_url: providers[0].base_url,
      model: providers[0].model,
      api_key: providers[0].api_key,
      level,
      all_levels: allLvl,
      providers,
    }),
  });
  const data = await res.json();
  if (res.ok) setStatus('running');
  else        alert('Error: ' + (data.error || res.statusText));
}

async function stopRun() {
  await fetch('/api/stop', { method: 'POST' });
  setStatus('idle');
}

function setStatus(s) {
  const wasRunning = runStatus === 'running';
  runStatus = s;
  const chip = document.getElementById('run-status');
  chip.textContent = s.toUpperCase();
  chip.className = 'rstatus ' + (s === 'running' ? 'running' : '');
  document.getElementById('btn-run').disabled  = (s === 'running');
  document.getElementById('btn-stop').disabled = (s !== 'running');
  if (s === 'idle') {
    clearInterval(timerInt);
    // When a run just finished, fetch the runner output and surface any errors
    if (wasRunning) checkRunnerLog();
  }
  if (s === 'running') {
    // hide previous error card
    document.getElementById('runner-err-card').style.display = 'none';
    document.getElementById('runner-err-pre').textContent = '';
  }
}

function copyRunnerErr() {
  const text = document.getElementById('runner-err-pre')?.textContent || '';
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('runner-err-copy');
    if (btn) { btn.textContent = '✔ COPIED'; setTimeout(() => { btn.textContent = '📋 COPY'; }, 1800); }
  }).catch(() => {});
}

async function checkRunnerLog() {
  try {
    const data = await fetch('/api/runner-log').then(r => r.json());
    const lines = data.lines || [];
    if (!lines.length) return;

    // Look for error indicators
    const errLines = lines.filter(l =>
      /error|traceback|exception|failed|exit code [^0]|no such file|not found|cannot|unable/i.test(l)
    );

    const card = document.getElementById('runner-err-card');
    const pre  = document.getElementById('runner-err-pre');

    if (errLines.length > 0) {
      // Show last 80 lines total (the tail is most relevant)
      pre.textContent = lines.slice(-80).join('\n');
      card.style.display = 'block';
    } else {
      card.style.display = 'none';
    }
  } catch(e) {}
}

setInterval(async () => {
  const d = await fetch('/api/status').then(r => r.json()).catch(() => ({}));
  if (d.status && d.status !== runStatus) setStatus(d.status);
}, 2500);

// ── Terminal panels ───────────────────────────────────────────────────────────
function buildTerminals(levels) {
  const grid = document.getElementById('term-grid');
  grid.innerHTML = '';
  terminals = {};
  totalStars = 0;
  curLevelId = null;
  updateGlobalStars();
  document.getElementById('side-live').style.display = 'none';

  initBatchState(levels);
  grid.insertAdjacentHTML('beforeend', renderTrackLegend());

  const grouped = {};
  levels.forEach(level => {
    if (!grouped[level.levelId]) grouped[level.levelId] = [];
    grouped[level.levelId].push(level);
  });

  Object.values(grouped).forEach(group => {
    group.sort((a, b) => a.providerSlot - b.providerSlot);
    const first = group[0];
    const row = document.createElement('section');
    row.className = 'bench-row';
    row.id = 'row-' + first.levelId;
    row.innerHTML = `
      <div class="row-head">
        <span class="row-title">${esc(first.name)}</span>
        <span class="row-meta">${first.levelId} · difficulty ${first.difficulty}</span>
      </div>
      <div class="row-panels" id="row-panels-${first.levelId}"></div>
    `;
    grid.appendChild(row);

    const panelsWrap = document.getElementById('row-panels-' + first.levelId);
    panelsWrap.style.gridTemplateColumns = `${group.map(() => '360px').join(' ')} 290px`;

    group.forEach(l => {
      const el = document.createElement('div');
      el.className = `term ${providerClass(l.providerSlot)}`;
      el.id = 'term-' + l.id;

      const badgesHtml = (l.tools || []).map(t =>
        `<span class="tool-badge" id="tbadge-${l.id}-${t}">${esc(t)}</span>`
      ).join('');
      const instr = (l.instruction || '').trim();

      el.innerHTML = `
        <div class="term-bar">
          <div class="term-bar-left">
            <span class="term-title">${esc(l.name)}</span>
            <span class="term-provider">${esc(l.providerTitle || '')}</span>
          </div>
          <div class="term-bar-right">
            <span class="hud-turns" id="hud-t-${l.id}">T:0</span>
            <span class="hud-active" id="hud-a-${l.id}"></span>
            <span class="pill" id="pill-${l.id}">waiting</span>
          </div>
        </div>
        <div class="briefing open" id="brief-${l.id}">
          <button class="brief-toggle" onclick="toggleBrief('${l.id}')">
            <span class="brief-arr">▶</span> MISSION BRIEF
          </button>
          <div class="brief-body">
            <div class="brief-instr">${esc(instr)}</div>
            <div class="brief-footer">
              <div class="tool-badges">${badgesHtml}</div>
              <div class="brief-limits">max ${l.max_turns} turns · ${l.timeout_s}s</div>
            </div>
          </div>
        </div>
        <div class="term-out" id="out-${l.id}"></div>
        ${l.preview_port ? `
        <div class="preview-strip" id="prev-strip-${l.id}">
          <div class="preview-strip-bar">
            <span id="prev-lbl-${l.id}">🌐 PREVIEW — waiting for server…</span>
            <div style="display:flex;gap:6px;align-items:center">
              <span id="prev-status-dot-${l.id}" class="pf-dot pending" title="server status"></span>
              <a id="prev-open-${l.id}"
                 href="http://localhost:${l.preview_port}${l.preview_path||'/'}"
                 target="_blank" rel="noopener"
                 style="font-family:var(--font);font-size:8px;color:var(--dim);text-decoration:none;
                        border:1px solid var(--b3);padding:2px 6px;pointer-events:none">
                ↗ OPEN
              </a>
              <button class="preview-reload-btn" onclick="reloadPreview('${l.id}')">↺</button>
            </div>
          </div>
          <div id="prev-wait-${l.id}" style="
            height:260px;display:flex;align-items:center;justify-content:center;
            background:var(--panel);color:var(--dim);font-size:7px;letter-spacing:2px;flex-direction:column;gap:10px">
            <span style="font-size:18px;animation:pf-pulse .9s infinite alternate">⏳</span>
            WAITING FOR SERVER ON :${l.preview_port}
          </div>
          <iframe id="prev-${l.id}"
            src=""
            class="preview-frame"
            style="display:none"
            sandbox="allow-scripts allow-same-origin allow-forms"
            title="live preview">
          </iframe>
        </div>` : ''}
        <div class="term-score" id="sc-${l.id}">
          <div class="score-top">
            <span class="score-stars" id="stars-${l.id}">☆☆☆☆☆</span>
            <span class="score-num"   id="num-${l.id}">— / 100</span>
          </div>
          <div class="score-bar-o" style="margin-bottom:0"><div class="score-bar-i" id="bar-${l.id}"></div></div>
        </div>
      `;
      panelsWrap.appendChild(el);

      terminals[l.id] = {
        turns: 0,
        tools: 0,
        startTs: null,
        callCards: {},
        previewPort: l.preview_port || null,
        previewPath: l.preview_path || '/',
        liveTurnActive: false,
        liveBubbleId: null,
        liveMessageText: '',
        levelId: l.levelId || l.id,
        providerLabel: l.providerLabel || '',
        providerSlot: l.providerSlot || 1,
        completed: false,
      };
    });

    panelsWrap.insertAdjacentHTML('beforeend', renderSummaryShell(first.levelId, first.name));
  });

  grid.insertAdjacentHTML('beforeend', renderBatchSummaryShell());
  updateBatchSummary();
}

// ── Output helpers ────────────────────────────────────────────────────────────
function setPill(lid, state) {
  const p = document.getElementById('pill-' + lid);
  if (p) { p.className = 'pill ' + state; p.textContent = state; }
}

function toggleBrief(lid) {
  const wrap = document.getElementById('brief-' + lid);
  if (!wrap) return;
  wrap.classList.toggle('open');
}

function reloadPreview(lid) {
  const frame = document.getElementById('prev-' + lid);
  if (!frame) return;
  // Only reload if the iframe is already visible
  if (frame.style.display === 'none') return;
  const src = frame.src;
  frame.src = '';
  setTimeout(() => { frame.src = src; }, 80);
}

// ── Preview server poller ─────────────────────────────────────────────────────
// Called when a session starts for a level that has a preview port.
// Polls /api/preview-status every 2s until the server inside the container
// responds, then shows the iframe.
const _previewPollers = {}; // lid → intervalId

function startPreviewPoller(lid, port, path) {
  const strip = document.getElementById('prev-strip-' + lid);
  if (!strip) return;

  // Clear any previous poller for this level
  if (_previewPollers[lid]) clearInterval(_previewPollers[lid]);

  const dot   = document.getElementById('prev-status-dot-' + lid);
  const lbl   = document.getElementById('prev-lbl-' + lid);
  const wait  = document.getElementById('prev-wait-' + lid);
  const frame = document.getElementById('prev-' + lid);
  const open  = document.getElementById('prev-open-' + lid);
  const url   = `http://localhost:${port}${path}`;
  let attempts = 0;

  async function probe() {
    attempts++;
    try {
      const res = await fetch(`/api/preview-status?port=${port}&path=${encodeURIComponent(path)}`)
                        .then(r => r.json());
      if (res.up) {
        // Server is up — show iframe, enable open link
        clearInterval(_previewPollers[lid]);
        delete _previewPollers[lid];

        if (dot)   { dot.className = 'pf-dot ok'; }
        if (lbl)   { lbl.textContent = `🌐 LIVE — localhost:${port}${path}`; }
        if (wait)  { wait.style.display = 'none'; }
        if (frame) { frame.src = url; frame.style.display = 'block'; }
        if (open)  { open.href = url; open.style.color = 'var(--y2)'; open.style.pointerEvents = 'auto'; }
      } else {
        // Still waiting
        if (dot) dot.className = 'pf-dot pending';
        if (lbl) lbl.textContent = `⏳ waiting for server on :${port}… (${attempts * 2}s)`;
        // Give up after 3 minutes
        if (attempts >= 90) {
          clearInterval(_previewPollers[lid]);
          delete _previewPollers[lid];
          if (dot) dot.className = 'pf-dot fail';
          if (lbl) lbl.textContent = `✘ server never started on :${port}`;
          if (wait) wait.innerHTML =
            `<span style="color:var(--red);font-size:7px">SERVER DID NOT START ON :${port}</span>`;
        }
      }
    } catch(e) {}
  }

  // First probe immediately, then every 2s
  probe();
  _previewPollers[lid] = setInterval(probe, 2000);
}

function stopPreviewPoller(lid) {
  if (_previewPollers[lid]) {
    clearInterval(_previewPollers[lid]);
    delete _previewPollers[lid];
  }
}

function bumpTurnCounter(lid) {
  const t = terminals[lid];
  if (!t) return;
  t.turns++;
  const hudT = document.getElementById('hud-t-' + lid);
  if (hudT) hudT.textContent = 'T:' + t.turns;
  document.getElementById('sl-turns').textContent = t.turns;
}

function ensureAssistantTurnStarted(lid) {
  const t = terminals[lid];
  if (!t || t.liveTurnActive) return t;
  t.liveTurnActive = true;
  bumpTurnCounter(lid);
  return t;
}

function renderBubbleBody(text) {
  const clean = String(text || '').trim();
  if (!clean) return '';
  const MAX = 220;
  const shown = clean.length > MAX ? clean.slice(0, MAX) : clean;
  const more  = clean.length > MAX
    ? `<span class="bubble-trunc" onclick="this.parentNode.querySelector('.bubble-full').style.display='block';this.remove()">… show more</span>
       <span class="bubble-full" style="display:none">${esc(clean.slice(MAX))}</span>`
    : '';
  return `${esc(shown)}${more}`;
}

function ensureLiveBubble(lid) {
  const t = terminals[lid];
  if (!t) return null;
  if (!t.liveBubbleId) {
    const bubbleId = 'bubble-' + lid + '-' + Date.now();
    t.liveBubbleId = bubbleId;
    appendToTerm(lid, `<div class="bubble streaming" id="${bubbleId}"><span class="cursor"></span></div>`);
  }
  return document.getElementById(t.liveBubbleId);
}

function setLiveBubbleText(lid, text, done) {
  const bubble = ensureLiveBubble(lid);
  if (!bubble) return;
  const body = renderBubbleBody(text) || '<span style="color:var(--dim)">…</span>';
  bubble.innerHTML = body + (done ? '' : '<span class="cursor"></span>');
  bubble.classList.toggle('streaming', !done);
  const out = document.getElementById('out-' + lid);
  if (out) out.scrollTop = out.scrollHeight;
}

// ── Event handler ─────────────────────────────────────────────────────────────
function onEvent(ev) {
  const lid = ev.panel_id || ev.level_id || curLevelId;

  // ── session_start ──────────────────────────────────────────────────────────
  if (ev.type === 'session_start' && lid) {
    curLevelId = lid;
    const el = document.getElementById('term-' + lid);
    if (el) {
      el.classList.remove('is-done');
      el.classList.add('is-active');
    }
    setPill(lid, 'active');
    const st = ev.ts || (Date.now() / 1000);
    terminals[lid] = {
      ...(terminals[lid] || {}),
      turns: 0,
      tools: 0,
      startTs: st,
      callCards: {},
      liveTurnActive: false,
      liveBubbleId: null,
      liveMessageText: '',
    };
    const hudT = document.getElementById('hud-t-' + lid);
    if (hudT) { hudT.textContent = 'T:0'; hudT.style.display = 'inline'; }

    // Sidebar live card
    document.getElementById('side-live').style.display = 'block';
    document.getElementById('sl-provider').textContent = ev.provider_label || ev.model || terminals[lid]?.providerLabel || '—';
    document.getElementById('sl-level').textContent = ev.level_name || ev.level_id || lid;
    document.getElementById('sl-turns').textContent = '0';
    document.getElementById('sl-tools').textContent = '0';
    document.getElementById('sl-time').textContent  = '0s';
    document.getElementById('f-runid').textContent  = 'run ' + (ev.run_id || '').slice(0, 8);

    clearInterval(timerInt);
    timerInt = setInterval(() => {
      const t2 = terminals[lid];
      if (!t2?.startTs) return;
      const secs = ((Date.now() / 1000) - t2.startTs).toFixed(0);
      document.getElementById('sl-time').textContent = secs + 's';
    }, 400);

    appendToTerm(lid, '<div class="tl info"><span class="lc">── SESSION START ──</span></div>');

    // If this level has a preview port, start polling for the server
    const t0 = terminals[lid];
    if (t0?.previewPort) {
      startPreviewPoller(lid, t0.previewPort, t0.previewPath || '/');
    }
  }

  // ── message_delta ──────────────────────────────────────────────────────────
  else if (ev.type === 'message_delta' && ev.role === 'assistant' && lid) {
    const delta = ev.delta || '';
    if (!delta) return;
    const t = ensureAssistantTurnStarted(lid);
    if (!t) return;
    t.liveMessageText = (t.liveMessageText || '') + delta;
    setLiveBubbleText(lid, t.liveMessageText, false);
  }

  // ── message ────────────────────────────────────────────────────────────────
  else if (ev.type === 'message' && ev.role === 'assistant' && lid) {
    const t = ensureAssistantTurnStarted(lid);

    // Extract text from content (string or array)
    let text = '';
    if (typeof ev.content === 'string') {
      text = ev.content;
    } else if (Array.isArray(ev.content)) {
      const tc = ev.content.find(c => c && c.type === 'text');
      if (tc) text = tc.text || '';
    }
    text = text.trim();

    if (t && t.liveBubbleId) {
      const finalText = text || t.liveMessageText || '';
      if (finalText.trim()) {
        setLiveBubbleText(lid, finalText, true);
      } else {
        const bubble = document.getElementById(t.liveBubbleId);
        if (bubble) bubble.remove();
      }
      t.liveBubbleId = null;
      t.liveMessageText = '';
      t.liveTurnActive = false;
      return;
    }

    if (text) {
      appendToTerm(lid, `<div class="bubble">${renderBubbleBody(text)}</div>`);
    }

    if (t) {
      t.liveBubbleId = null;
      t.liveMessageText = '';
      t.liveTurnActive = false;
    }
  }

  // ── tool_call ──────────────────────────────────────────────────────────────
  else if (ev.type === 'tool_call' && lid) {
    const t = terminals[lid];
    if (t) {
      t.tools++;
      document.getElementById('sl-tools').textContent = t.tools;
    }

    const callId = ev.call_id || ('c' + Date.now());
    const args   = ev.args || {};

    // Format args nicely — show each key:value on one row (compact)
    const argsHtml = Object.entries(args).map(([k, v]) => {
      const vs = String(v).slice(0, 120).replace(/\n/g, '↵');
      return `<span class="cc-key">${esc(k)}</span>=<span class="cc-val">${esc(vs)}</span>`;
    }).join('  ');

    const cardHtml = `
      <div class="call-card running" id="cc-${callId}" data-callid="${callId}">
        <div class="cc-head">
          <span class="cc-tool">▶ ${esc(ev.tool)}</span>
          <span class="cc-spin" id="ccsp-${callId}"><span class="cursor"></span></span>
        </div>
        <div class="cc-args">${argsHtml || '<span class="cc-val">(no args)</span>'}</div>
        <div class="cc-result" id="ccr-${callId}"></div>
      </div>`;
    appendToTerm(lid, cardHtml);

    // Store card metadata so tool_result can render the right widget
    if (t) t.callCards[callId] = { tool: ev.tool, args: ev.args || {} };

    // HUD active tool badge
    const hudA = document.getElementById('hud-a-' + lid);
    if (hudA) { hudA.textContent = '▶ ' + ev.tool; hudA.style.display = 'inline'; }

    // Highlight tool badge in briefing
    (document.querySelectorAll('.tool-badge')).forEach(b => b.classList.remove('active-tool'));
    const badge = document.getElementById('tbadge-' + lid + '-' + ev.tool);
    if (badge) badge.classList.add('active-tool');
  }

  // ── tool_result ────────────────────────────────────────────────────────────
  else if (ev.type === 'tool_result' && lid) {
    const ok     = ev.exit_code === 0;
    const callId = ev.call_id || '';
    const output = ev.output || '';
    const snip   = (output.split('\n').find(l => l.trim()) || '').slice(0, 100);

    // Update the call card
    const card = document.getElementById('cc-' + callId);
    if (card) {
      card.classList.remove('running');
      card.classList.add(ok ? 'ok' : 'fail');
      const spin = document.getElementById('ccsp-' + callId);
      if (spin) spin.innerHTML = ok
        ? '<span class="cc-ok">✔</span>'
        : '<span class="cc-fail">✘ exit ' + ev.exit_code + '</span>';
      const res = document.getElementById('ccr-' + callId);
      if (res && snip) res.innerHTML = `<span class="cc-snip">${esc(snip)}</span>`;
    }

    // Render rich result widget based on what tool was called
    const meta = terminals[lid]?.callCards?.[callId];
    if (meta && ok) {
      if (meta.tool === 'write_file') {
        appendToTerm(lid, renderArtifact(meta.args.path || '?', meta.args.content || output));
      } else if (meta.tool === 'http_request') {
        appendToTerm(lid, renderHttpPreview(meta.args, output));
      }
    }

    const out = document.getElementById('out-' + lid);
    if (out) out.scrollTop = out.scrollHeight;

    // Clear HUD active + tool badge highlight
    const hudA = document.getElementById('hud-a-' + lid);
    if (hudA) hudA.style.display = 'none';
    document.querySelectorAll('.tool-badge.active-tool').forEach(b => b.classList.remove('active-tool'));
  }

  // ── session_end ────────────────────────────────────────────────────────────
  else if (ev.type === 'session_end') {
    clearInterval(timerInt);
    const t = terminals[lid];
    if (t) {
      t.liveTurnActive = false;
      t.liveBubbleId = null;
      t.liveMessageText = '';
    }
    const sc    = ev.score || {};
    const total = sc.total || 0;
    const stars = scoreToStars(total);

    const barEl   = document.getElementById('bar-'   + lid);
    const numEl   = document.getElementById('num-'   + lid);
    const starsEl = document.getElementById('stars-' + lid);
    if (barEl)   barEl.style.width = total + '%';
    if (numEl)   { numEl.textContent = total.toFixed(1) + ' / 100'; numEl.className = 'score-num has-score'; }
    if (starsEl) starsEl.innerHTML = renderStars(stars, true);

    // Inject full score breakdown below the bar
    const scoreEl = document.getElementById('sc-' + lid);
    if (scoreEl) scoreEl.insertAdjacentHTML('beforeend', renderScoreBreakdown(sc));

    const el = document.getElementById('term-' + lid);
    if (el) {
      el.classList.remove('is-active');
      el.classList.add('is-done');
    }
    setPill(lid, ev.timed_out ? 'error' : 'done');

    // Clear HUD active tool
    const hudA = document.getElementById('hud-a-' + lid);
    if (hudA) hudA.style.display = 'none';

    appendToTerm(lid,
      `<div class="tl done"><span class="lc">★ ${total.toFixed(1)} / 100 &nbsp; ${renderStarsPlain(stars)}${ev.timed_out ? ' ⚠ TIMEOUT' : ''}</span></div>`
    );

    if (t && !t.completed) {
      t.completed = true;
      batchState.completedPanels++;
    }
    if (t && batchState?.rows?.[t.levelId]) {
      batchState.rows[t.levelId].results[t.providerSlot] = {
        label: t.providerLabel || `Provider ${t.providerSlot}`,
        total,
        stars,
        turns: t.turns,
        tools: t.tools,
        timedOut: !!ev.timed_out,
      };
      updateLevelSummary(t.levelId);
      updateBatchSummary();
    }

    totalStars += stars;
    updateGlobalStars();
    document.getElementById('run-label').textContent = 'last run: ' + new Date().toISOString().slice(11, 19);
    document.getElementById('run-label').style.color = 'var(--dim)';

    // Stop any running preview poller for this level
    stopPreviewPoller(lid);

    // Freeze preview strip — container stops shortly after session_end.
    // Keep iframe + open link visible so user can inspect before it goes dark.
    const prevStrip = document.getElementById('prev-strip-' + lid);
    if (prevStrip) {
      const lbl = document.getElementById('prev-lbl-' + lid);
      if (lbl) lbl.textContent = '⚠ SERVER STOPPING — grab it now ↗';
      prevStrip.style.borderTop = '2px solid var(--y3)';
      // After 8s the container is gone — dim the iframe
      setTimeout(() => {
        const fr = document.getElementById('prev-' + lid);
        if (fr) { fr.style.opacity = '0.35'; fr.style.pointerEvents = 'none'; }
        if (lbl) lbl.textContent = '🌐 preview offline (container stopped)';
      }, 8000);
    }

    curLevelId = null;
  }
}

// ── Terminal output helper ────────────────────────────────────────────────────
function appendToTerm(lid, html) {
  const out = document.getElementById('out-' + lid);
  if (!out) return;
  const d = document.createElement('div');
  d.innerHTML = html;
  while (d.firstChild) out.appendChild(d.firstChild);
  out.scrollTop = out.scrollHeight;
}

// ── Stars ─────────────────────────────────────────────────────────────────────
function scoreToStars(s) { return s>=95?5 : s>=80?4 : s>=60?3 : s>=35?2 : s>0?1 : 0; }

function renderStars(n, animated) {
  let r = '';
  for (let i = 0; i < 5; i++) {
    if (i < n) {
      const delay = animated ? `animation-delay:${i*160}ms;` : '';
      r += `<span class="s-filled${animated?' star-anim':''}" style="${delay}">★</span>`;
    } else {
      r += `<span>☆</span>`;
    }
  }
  return r;
}

function renderStarsPlain(n) {
  return '★'.repeat(n) + '☆'.repeat(5-n);
}

function updateGlobalStars() {
  const t = totalStars;
  const s = t > 0 ? t + ' ★' : '';
  document.getElementById('h-stars').textContent   = s;
  document.getElementById('run-total').textContent = t > 0 ? 'Total: ' + s : '';
  document.getElementById('f-total').textContent   = t > 0 ? s + ' earned' : '';
}

// ── Artifact renderer ─────────────────────────────────────────────────────────
function renderArtifact(path, content) {
  const bytes = new TextEncoder().encode(content).length;
  const uid   = 'art-' + Math.random().toString(36).slice(2, 8);
  const MAX   = 600;
  const shown = content.slice(0, MAX);
  const hasMore = content.length > MAX;
  const ext   = path.split('.').pop() || '';
  return `
    <div class="artifact-card">
      <div class="artifact-hdr">
        <span>📄 <span class="artifact-path">${esc(path)}</span></span>
        <span class="artifact-bytes">${bytes}B · ${esc(ext)}</span>
      </div>
      <pre class="artifact-body" id="${uid}">${esc(shown)}${hasMore ? '\n…' : ''}</pre>
      ${hasMore ? `<button class="artifact-toggle" onclick="
        const el=document.getElementById('${uid}');
        const full=${JSON.stringify(content)};
        el.textContent=full; this.remove();
      ">▼ show all ${bytes}B</button>` : ''}
    </div>`;
}

// ── HTTP preview renderer ─────────────────────────────────────────────────────
function renderHttpPreview(args, output) {
  const method = (args.method || 'GET').toUpperCase();
  const url    = args.url || args.uri || '';
  const status = (() => {
    const m = output.match(/^HTTP\/\S+\s+(\d+)/m) || output.match(/"status":\s*(\d+)/);
    return m ? parseInt(m[1]) : null;
  })();
  const statusOk   = status && status < 400;
  const statusHtml = status
    ? `<span class="http-status ${statusOk ? 'ok' : 'fail'}">${status}</span>`
    : '';
  const isHtml = output.trimStart().startsWith('<') || /<html/i.test(output);
  const body   = output.slice(0, 800);
  const previewBody = isHtml
    ? `<div style="font-size:8px;color:var(--dim);padding:4px 8px">[HTML — use webserver preview for rendering]</div>
       <pre class="http-preview">${esc(body.slice(0, 300))}</pre>`
    : `<pre class="http-preview">${esc(body)}</pre>`;
  return `
    <div class="http-card">
      <div class="http-hdr">
        <span class="http-method">${esc(method)}</span>
        <span style="color:var(--dim);font-size:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1">${esc(url)}</span>
        ${statusHtml}
      </div>
      ${previewBody}
    </div>`;
}

// ── Score breakdown renderer ──────────────────────────────────────────────────
function renderScoreBreakdown(sc) {
  const dims  = sc.dimensions || {};
  const pens  = sc.penalties  || {};
  const crits = sc.criteria   || [];

  const DIMS = [
    { key: 'completion',      label: 'COMPLETE',  wt: 40 },
    { key: 'efficiency',      label: 'EFFICIENCY', wt: 25 },
    { key: 'self_correction', label: 'RECOVERY',  wt: 20 },
    { key: 'path_quality',    label: 'PATH',      wt: 15 },
  ];

  const dimRows = DIMS.map(d => {
    const pct  = Math.round(dims[d.key] ?? 0);
    const cls  = pct >= 99 ? 'full' : pct === 0 ? 'zero' : '';
    return `<div class="dim-row">
      <span class="dim-label">${d.label}</span>
      <span class="dim-wt">${d.wt}%</span>
      <div class="dim-bar-o"><div class="dim-bar-i ${cls}" style="width:${pct}%"></div></div>
      <span class="dim-pct">${pct}%</span>
    </div>`;
  }).join('');

  const penTotal = (pens.extra_calls || 0) + (pens.backtracks || 0) + (pens.timeout || 0);
  const penRow   = penTotal > 0 ? `<div class="pen-row">
    <span class="pen-label">PENALTIES</span>
    <span class="pen-val">−${penTotal.toFixed(1)} pts</span>
  </div>` : '';

  const critChips = crits.map(c =>
    `<span class="crit-chip ${c.passed ? 'pass' : 'fail'}" title="${esc(c.id)}">
      ${c.passed ? '✔' : '✘'} ${esc(c.id)}
    </span>`
  ).join('');

  return `<div class="score-breakdown">
    <div class="dim-bars">${dimRows}${penRow}</div>
    ${crits.length ? `<div class="criteria-row">${critChips}</div>` : ''}
  </div>`;
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connect() {
  const ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onopen = () => {
    document.getElementById('dot').className = 'dot live';
    document.getElementById('conn-lbl').textContent = 'LIVE';
  };
  ws.onmessage = (m) => {
    try {
      const d = JSON.parse(m.data);
      if (d._type === 'file') { document.getElementById('log-file').textContent = d.filename; return; }
      onEvent(d);
    } catch(e) {}
  };
  ws.onclose = () => {
    document.getElementById('dot').className = 'dot';
    document.getElementById('conn-lbl').textContent = 'RECONNECTING';
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

init();
connect();
</script>
</body>
</html>"""


@app.get("/builder", response_class=HTMLResponse)
async def builder() -> HTMLResponse:
    return HTMLResponse(BUILDER_HTML)


BUILDER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>benchb0t · level builder</title>
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#0a0800; --panel:#100d00; --panel2:#151000;
  --b1:#241e00; --b2:#3a3000; --b3:#4a3f00;
  --y1:#ffd700; --y2:#ffb300; --y3:#ff8c00;
  --ydk:#7a6000; --text:#ffe87a; --dim:#5a4a10; --dim2:#3a3000;
  --green:#39ff14; --red:#ff3a3a;
  --font:'Press Start 2P',monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font);font-size:10px;overflow:hidden}
body::after{content:'';position:fixed;inset:0;z-index:9999;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,.07) 3px,rgba(0,0,0,.07) 4px)}

/* HEADER */
header{background:var(--panel);border-bottom:3px solid var(--y1);padding:0 20px;height:52px;
  display:flex;align-items:center;gap:14px;box-shadow:0 2px 20px rgba(255,215,0,.15);z-index:50;flex-shrink:0}
.logo{color:var(--y1);font-size:13px;letter-spacing:3px;text-shadow:0 0 20px var(--y1)}
.sep{color:var(--b3)}
.nav-a{font-size:6px;color:var(--dim);letter-spacing:2px;text-decoration:none;
  border:1px solid var(--dim);padding:3px 8px}
.nav-a:hover{color:var(--y1);border-color:var(--y1)}
.hdr-badge{font-size:8px;letter-spacing:2px;color:var(--y2)}

/* ROOT SPLIT */
.root{display:flex;height:calc(100vh - 52px)}
.left{width:420px;flex-shrink:0;overflow-y:auto;border-right:2px solid var(--b1);
  display:flex;flex-direction:column;gap:0}
.left::-webkit-scrollbar{width:3px}
.left::-webkit-scrollbar-thumb{background:var(--b2)}
.right{flex:1;display:flex;flex-direction:column;overflow:hidden}

/* SECTIONS */
.section{border-bottom:2px solid var(--b1)}
.sec-hdr{display:flex;align-items:center;gap:8px;padding:10px 14px;cursor:pointer;
  background:var(--panel);user-select:none}
.sec-hdr:hover{background:var(--panel2)}
.sec-arr{font-size:8px;color:var(--ydk);transition:transform .15s}
.sec-hdr.open .sec-arr{transform:rotate(90deg)}
.sec-title{font-size:6px;letter-spacing:3px;color:var(--dim);text-transform:uppercase}
.sec-body{padding:12px 14px;display:none;flex-direction:column;gap:8px}
.sec-body.open{display:flex}

/* FORM FIELDS */
.field{display:flex;flex-direction:column;gap:4px}
.field label{font-size:6px;color:var(--dim);letter-spacing:1px;text-transform:uppercase}
.field input,.field textarea,.field select{
  background:var(--bg);border:2px solid var(--b2);color:var(--text);
  font-family:var(--font);font-size:7px;padding:7px 9px;outline:none;resize:vertical}
.field input:focus,.field textarea:focus,.field select:focus{border-color:var(--y1)}
.field select{cursor:pointer}
.field select option{background:var(--panel)}
.field textarea{min-height:72px;line-height:1.7;font-family:monospace;font-size:9px}
.field-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.field-row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.hint{font-size:5px;color:var(--dim2);letter-spacing:1px;margin-top:2px}

/* pkg strip — show all three side by side */
.pkg-strip{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:0}

/* preset row — select + optional custom input */
.preset-row{display:flex;flex-direction:column;gap:4px}
.preset-row select,.preset-row input{width:100%;box-sizing:border-box}

/* combobox hint */
.field input[list]{padding-right:22px}

/* DIFFICULTY STARS */
.diff-row{display:flex;gap:6px;align-items:center}
.diff-star{font-size:16px;color:var(--b3);cursor:pointer;transition:color .1s;user-select:none}
.diff-star.on{color:var(--y1);text-shadow:0 0 10px rgba(255,215,0,.5)}
.diff-star:hover{color:var(--y2)}

/* TOOL CHECKBOXES */
.tool-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.tool-chk{display:flex;align-items:center;gap:7px;cursor:pointer;
  border:2px solid var(--b2);padding:6px 8px}
.tool-chk:hover{border-color:var(--b3)}
.tool-chk.checked{border-color:var(--ydk)}
.tool-chk input{accent-color:var(--y1);width:12px;height:12px;cursor:pointer}
.tool-chk span{font-size:6px;color:var(--dim);letter-spacing:1px;font-family:monospace}
.tool-chk.checked span{color:var(--y2)}

/* CRITERIA */
.criteria-list{display:flex;flex-direction:column;gap:8px}
.crit-card{border:2px solid var(--b2);background:var(--panel2)}
.crit-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:6px 9px;border-bottom:1px solid var(--b1);cursor:pointer}
.crit-hdr:hover{background:var(--panel)}
.crit-label{font-size:6px;color:var(--dim);letter-spacing:1px}
.crit-del{font-size:9px;color:var(--red);cursor:pointer;padding:0 4px}
.crit-del:hover{color:#ff7a7a}
.crit-body{padding:9px;display:flex;flex-direction:column;gap:7px}

.btn-add{
  background:transparent;border:2px solid var(--b2);color:var(--dim);
  font-family:var(--font);font-size:6px;letter-spacing:2px;padding:8px;
  cursor:pointer;width:100%;text-align:center;transition:border-color .15s,color .15s}
.btn-add:hover{border-color:var(--y2);color:var(--y2)}

/* LOAD BAR */
.load-bar{display:flex;gap:8px;align-items:center;padding:10px 14px;
  background:var(--panel2);border-bottom:2px solid var(--b1);flex-shrink:0}
.load-bar select{flex:1;background:var(--bg);border:2px solid var(--b2);color:var(--text);
  font-family:var(--font);font-size:7px;padding:6px 8px;outline:none;cursor:pointer}
.load-bar select:focus{border-color:var(--y1)}
.load-bar select option{background:var(--panel)}
.btn-load{padding:6px 12px;font-family:var(--font);font-size:6px;letter-spacing:2px;
  background:transparent;color:var(--y2);border:2px solid var(--y2);cursor:pointer;white-space:nowrap}
.btn-load:hover{background:rgba(255,179,0,.08)}
.tpl-lbl{font-size:5px;color:var(--dim);letter-spacing:2px;white-space:nowrap}

/* ACTION BUTTONS */
.actions{padding:14px;display:flex;gap:8px;border-top:2px solid var(--b1);background:var(--panel);flex-shrink:0}
.btn-save{
  flex:1;padding:11px;font-family:var(--font);font-size:8px;letter-spacing:3px;
  background:var(--y1);color:#000;border:none;cursor:pointer;
  box-shadow:0 4px 0 var(--ydk);transition:transform .08s,box-shadow .08s}
.btn-save:active{transform:translateY(3px);box-shadow:0 1px 0 var(--ydk)}
.btn-copy{
  padding:11px 14px;font-family:var(--font);font-size:7px;letter-spacing:2px;
  background:transparent;color:var(--y2);border:2px solid var(--y2);cursor:pointer}
.btn-copy:hover{background:rgba(255,179,0,.08)}
.save-msg{font-size:6px;color:var(--green);letter-spacing:1px;align-self:center;display:none}

/* YAML PREVIEW */
.preview-hdr{background:var(--panel);border-bottom:2px solid var(--b1);
  padding:8px 14px;font-size:6px;color:var(--dim);letter-spacing:3px;flex-shrink:0}
.yaml-out{flex:1;overflow:auto;padding:14px;font-family:monospace;font-size:10px;
  line-height:1.8;color:var(--text);background:var(--bg)}
.yaml-out::-webkit-scrollbar{width:3px}
.yaml-out::-webkit-scrollbar-thumb{background:var(--b2)}
/* syntax highlight classes */
.yc{color:var(--dim)}    /* comment */
.yk{color:var(--y1)}     /* key */
.yv{color:var(--text)}   /* value */
.ys{color:var(--y2)}     /* string value */
.yn{color:var(--green)}  /* number */
.yd{color:var(--dim)}    /* dashes/punct */
</style>
</head>
<body>
<header>
  <span class="logo">benchb0t</span>
  <span class="sep">│</span>
  <span class="hdr-badge">LEVEL BUILDER</span>
  <div style="margin-left:auto;display:flex;gap:10px">
    <a class="nav-a" href="/">← LIVE</a>
    <a class="nav-a" href="/analytics">ANALYTICS</a>
  </div>
</header>

<div class="root">

<!-- ── LEFT: FORM ── -->
<div class="left">

  <!-- LOAD / TEMPLATE BAR -->
  <div class="load-bar">
    <span class="tpl-lbl">LOAD</span>
    <select id="load-sel">
      <option value="">— existing level —</option>
    </select>
    <button class="btn-load" onclick="loadLevel()">EDIT ↓</button>
    <span class="tpl-lbl" style="margin-left:4px">TPL</span>
    <select id="tpl-sel" onchange="applyTemplate(this.value)">
      <option value="">— template —</option>
      <option value="webapp">🌐 Webapp</option>
      <option value="file">📄 File task</option>
      <option value="api">🔌 API fetch</option>
      <option value="data">📊 Data pipeline</option>
    </select>
  </div>

  <!-- datalists for comboboxes -->
  <datalist id="dl-images">
    <option value="python:3.11-slim">
    <option value="python:3.12-slim">
    <option value="python:3.10-slim">
    <option value="node:20-slim">
    <option value="node:18-slim">
    <option value="node:22-slim">
    <option value="ubuntu:22.04">
    <option value="ubuntu:24.04">
    <option value="debian:bookworm-slim">
    <option value="alpine:3.19">
    <option value="golang:1.22-alpine">
    <option value="rust:1.77-slim">
    <option value="php:8.3-cli">
    <option value="ruby:3.3-slim">
    <option value="openjdk:21-slim">
    <option value="postgres:16-alpine">
    <option value="redis:7-alpine">
  </datalist>
  <datalist id="dl-workdir">
    <option value="/workspace">
    <option value="/app">
    <option value="/home/user">
    <option value="/tmp/work">
    <option value="/root">
  </datalist>

  <!-- LEVEL INFO -->
  <div class="section">
    <div class="sec-hdr open" onclick="toggleSec(this)">
      <span class="sec-arr">▶</span>
      <span class="sec-title">Level info</span>
    </div>
    <div class="sec-body open">
      <div class="field">
        <label>Name</label>
        <input id="f-name" type="text" placeholder="My Awesome Level" oninput="onNameInput()">
      </div>
      <div class="field">
        <label>ID <span class="hint">(auto-generated, editable)</span></label>
        <input id="f-id" type="text" placeholder="l-my-awesome-level" oninput="sync()">
      </div>
      <div class="field-row">
        <div class="field">
          <label>Category</label>
          <select id="f-cat" onchange="sync()">
            <option value="general">general</option>
            <option value="file-operations">file-operations</option>
            <option value="webapp">webapp</option>
            <option value="api">api</option>
            <option value="data">data</option>
            <option value="networking">networking</option>
            <option value="code-generation">code-generation</option>
            <option value="database">database</option>
            <option value="devops">devops</option>
            <option value="security">security</option>
          </select>
        </div>
        <div class="field">
          <label>Tags <span class="hint">(comma-sep)</span></label>
          <input id="f-tags" type="text" placeholder="beginner, read, write" oninput="sync()">
        </div>
      </div>
      <div class="field">
        <label>Difficulty</label>
        <div class="diff-row" id="diff-stars">
          <span class="diff-star on"  onclick="setDiff(1)">★</span>
          <span class="diff-star off" onclick="setDiff(2)">★</span>
          <span class="diff-star off" onclick="setDiff(3)">★</span>
          <span class="diff-star off" onclick="setDiff(4)">★</span>
          <span class="diff-star off" onclick="setDiff(5)">★</span>
        </div>
      </div>
    </div>
  </div>

  <!-- CONTAINER -->
  <div class="section">
    <div class="sec-hdr open" onclick="toggleSec(this)">
      <span class="sec-arr">▶</span>
      <span class="sec-title">Container</span>
    </div>
    <div class="sec-body open">
      <div class="field-row">
        <div class="field">
          <label>Image</label>
          <input id="f-image" type="text" list="dl-images" value="python:3.11-slim"
                 placeholder="python:3.11-slim" oninput="onImageChange(this.value)" autocomplete="off">
          <span class="hint">type or pick ↓</span>
        </div>
        <div class="field">
          <label>Working dir</label>
          <input id="f-workdir" type="text" list="dl-workdir" value="/workspace"
                 placeholder="/workspace" oninput="sync()" autocomplete="off">
        </div>
      </div>

      <!-- Package strip — only show sections relevant to the image -->
      <div class="pkg-strip">
        <div class="field pkg-apt" id="pkg-apt-wrap">
          <label>APT <span class="hint">space-sep</span></label>
          <input id="f-apt" type="text" placeholder="curl jq git" oninput="sync()">
        </div>
        <div class="field pkg-pip" id="pkg-pip-wrap">
          <label>PIP <span class="hint">space-sep</span></label>
          <input id="f-pip" type="text" placeholder="requests pandas" oninput="sync()">
        </div>
        <div class="field pkg-npm" id="pkg-npm-wrap">
          <label>NPM <span class="hint">space-sep</span></label>
          <input id="f-npm" type="text" placeholder="axios express" oninput="sync()">
        </div>
      </div>

      <div class="field">
        <label>Setup script <span class="hint">runs before agent starts</span></label>
        <textarea id="f-setup" placeholder="mkdir -p /workspace&#10;printf 'hello' > /workspace/input.txt" oninput="sync()"></textarea>
      </div>

      <!-- Preview port (collapsed by default, shown when image is node / user sets it) -->
      <div class="field-row" id="preview-fields">
        <div class="field">
          <label>🌐 Preview port</label>
          <select id="f-port-sel" onchange="onPortPreset(this.value)">
            <option value="">none</option>
            <option value="3000">3000 (Node/Vite)</option>
            <option value="5173">5173 (Vite default)</option>
            <option value="8000">8000 (Python)</option>
            <option value="8080">8080 (generic)</option>
            <option value="5000">5000 (Flask)</option>
            <option value="custom">custom…</option>
          </select>
        </div>
        <div class="field" id="f-port-custom-wrap" style="display:none">
          <label>Port number</label>
          <input id="f-port" type="number" placeholder="3000" min="1" max="65535" oninput="sync()">
        </div>
        <div class="field" id="f-ppath-wrap" style="display:none">
          <label>Preview path</label>
          <input id="f-ppath" type="text" placeholder="/" value="/" oninput="sync()">
        </div>
      </div>
    </div>
  </div>

  <!-- TASK -->
  <div class="section">
    <div class="sec-hdr open" onclick="toggleSec(this)">
      <span class="sec-arr">▶</span>
      <span class="sec-title">Task</span>
    </div>
    <div class="sec-body open">
      <div class="field">
        <label>Instruction</label>
        <textarea id="f-instr" style="min-height:110px"
          placeholder="Describe exactly what the agent must accomplish…" oninput="sync()"></textarea>
      </div>
      <div class="field-row">
        <div class="field">
          <label>Max turns</label>
          <div class="preset-row">
            <select id="f-turns-sel" onchange="setPreset('f-turns',this.value)">
              <option value="10">10 — quick</option>
              <option value="15" selected>15 — normal</option>
              <option value="20">20 — standard</option>
              <option value="30">30 — extended</option>
              <option value="50">50 — deep</option>
              <option value="custom">custom…</option>
            </select>
            <input id="f-turns" type="number" value="15" min="1" max="200"
                   style="display:none" oninput="sync()">
          </div>
        </div>
        <div class="field">
          <label>Timeout</label>
          <div class="preset-row">
            <select id="f-timeout-sel" onchange="setPreset('f-timeout',this.value)">
              <option value="60">60s — quick</option>
              <option value="90" selected>90s — normal</option>
              <option value="120">120s — standard</option>
              <option value="180">180s — long</option>
              <option value="300">300s — webapp</option>
              <option value="600">600s — max</option>
              <option value="custom">custom…</option>
            </select>
            <input id="f-timeout" type="number" value="90" min="10" max="600"
                   style="display:none" oninput="sync()">
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TOOLS -->
  <div class="section">
    <div class="sec-hdr open" onclick="toggleSec(this)">
      <span class="sec-arr">▶</span>
      <span class="sec-title">Tools</span>
    </div>
    <div class="sec-body open">
      <div class="tool-grid">
        <label class="tool-chk checked" id="tc-bash">
          <input type="checkbox" checked onchange="syncTool(this,'bash')"> <span>bash</span>
        </label>
        <label class="tool-chk checked" id="tc-read_file">
          <input type="checkbox" checked onchange="syncTool(this,'read_file')"> <span>read_file</span>
        </label>
        <label class="tool-chk checked" id="tc-write_file">
          <input type="checkbox" checked onchange="syncTool(this,'write_file')"> <span>write_file</span>
        </label>
        <label class="tool-chk" id="tc-http_request">
          <input type="checkbox" onchange="syncTool(this,'http_request')"> <span>http_request</span>
        </label>
      </div>
    </div>
  </div>

  <!-- EVALUATION -->
  <div class="section">
    <div class="sec-hdr open" onclick="toggleSec(this)">
      <span class="sec-arr">▶</span>
      <span class="sec-title">Evaluation</span>
    </div>
    <div class="sec-body open">
      <div class="field">
        <label>Efficiency target <span class="hint">(ideal # of tool calls)</span></label>
        <select id="f-eff-sel" onchange="setPreset('f-eff',this.value)">
          <option value="3">3 — trivial</option>
          <option value="5" selected>5 — simple</option>
          <option value="8">8 — moderate</option>
          <option value="12">12 — complex</option>
          <option value="20">20 — webapp</option>
          <option value="custom">custom…</option>
        </select>
        <input id="f-eff" type="number" value="5" min="1"
               style="display:none;margin-top:4px" oninput="sync()">
      </div>
      <div class="criteria-list" id="crit-list"></div>
      <button class="btn-add" onclick="addCriterion()">+ ADD CRITERION</button>
    </div>
  </div>

  <!-- ACTIONS -->
  <div class="actions">
    <button class="btn-copy" onclick="copyYaml()">📋 COPY</button>
    <button class="btn-save" onclick="saveLevel()">💾 SAVE LEVEL</button>
    <span class="save-msg" id="save-msg">✔ SAVED</span>
  </div>

</div><!-- /left -->

<!-- ── RIGHT: YAML PREVIEW ── -->
<div class="right">
  <div class="preview-hdr">YAML PREVIEW — live</div>
  <div class="yaml-out" id="yaml-out"></div>
</div>

</div><!-- /root -->

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let difficulty = 1;
let tools      = ['bash','read_file','write_file'];
let critCount  = 0;

// ── Init ──────────────────────────────────────────────────────────────────────
window.onload = async () => {
  await populateLoadDropdown();
  addCriterion({ id:'output_exists', desc:'output file must exist', check:"test -f /workspace/output.txt", weight:1.0 });
  // initialise smart field states
  onImageChange(document.getElementById('f-image')?.value || '');
  onPortPreset(document.getElementById('f-port-sel')?.value || 'none');
  sync();
};

async function populateLoadDropdown() {
  try {
    const levels = await fetch('/api/levels').then(r => r.json());
    const sel = document.getElementById('load-sel');
    sel.innerHTML = '<option value="">— existing level —</option>' +
      levels.map(l => `<option value="${esc(l.id)}">${esc(l.name)} (${esc(l.id)})</option>`).join('');
  } catch(e) {}
}

async function loadLevel() {
  const stem = document.getElementById('load-sel').value;
  if (!stem) return;
  try {
    const d = await fetch('/api/levels/' + encodeURIComponent(stem) + '/parsed').then(r => r.json());
    if (d.error) { alert('Error: ' + d.error); return; }
    setField('f-name',    d.name);
    setField('f-id',      d.id);
    setField('f-cat',     d.category);
    setField('f-tags',    d.tags);
    setField('f-image',   d.image);
    onImageChange(d.image || '');
    setField('f-workdir', d.working_dir);
    setField('f-apt',     d.apt);
    setField('f-pip',     d.pip);
    setField('f-npm',     d.npm);
    setField('f-setup',   d.setup_script);
    setField('f-instr',   d.instruction);
    setField('f-turns',   d.max_turns);
    setField('f-timeout', d.timeout_s);
    setField('f-eff',     d.efficiency_target);
    setField('f-port',    d.preview_port || '');
    setField('f-ppath',   d.preview_path || '/');
    setDiff(d.difficulty || 1);

    // tools
    tools = [];
    ['bash','read_file','write_file','http_request'].forEach(t => {
      const cb  = document.querySelector(`#tc-${t} input`);
      const lbl = document.getElementById('tc-' + t);
      const on  = (d.tools || []).includes(t);
      if (cb)  cb.checked = on;
      if (lbl) lbl.classList.toggle('checked', on);
      if (on)  tools.push(t);
    });

    // criteria — clear and re-add
    document.getElementById('crit-list').innerHTML = '';
    critCount = 0;
    (d.criteria || []).forEach(c => addCriterion(c));

    sync();
  } catch(e) { alert('Failed to load level: ' + e); }
}

// ── Section toggle ────────────────────────────────────────────────────────────
function toggleSec(hdr) {
  hdr.classList.toggle('open');
  const body = hdr.nextElementSibling;
  body.classList.toggle('open');
}

// ── Difficulty ────────────────────────────────────────────────────────────────
function setDiff(n) {
  difficulty = n;
  document.querySelectorAll('#diff-stars .diff-star').forEach((s, i) => {
    s.className = 'diff-star ' + (i < n ? 'on' : 'off');
  });
  sync();
}

// ── Tools ─────────────────────────────────────────────────────────────────────
function syncTool(cb, name) {
  const lbl = document.getElementById('tc-' + name);
  if (cb.checked) { if (!tools.includes(name)) tools.push(name); lbl.classList.add('checked'); }
  else            { tools = tools.filter(t => t !== name); lbl.classList.remove('checked'); }
  sync();
}

// ── Name → ID slug ────────────────────────────────────────────────────────────
function onNameInput() {
  const name = val('f-name');
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');
  document.getElementById('f-id').value = slug ? 'l-' + slug : '';
  sync();
}

// ── Criteria ──────────────────────────────────────────────────────────────────
function addCriterion(defaults) {
  const idx = ++critCount;
  const d = defaults || {};
  const card = document.createElement('div');
  card.className = 'crit-card';
  card.id = 'crit-' + idx;
  card.innerHTML = `
    <div class="crit-hdr" onclick="toggleCrit(${idx})">
      <span class="crit-label">CRITERION #${idx}</span>
      <span class="crit-del" onclick="event.stopPropagation();deleteCrit(${idx})">✕</span>
    </div>
    <div class="crit-body" id="crit-body-${idx}">
      <div class="field-row">
        <div class="field">
          <label>ID</label>
          <input class="ci" id="ci-id-${idx}" type="text" value="${esc(d.id||'')}" placeholder="output_exists" oninput="sync()">
        </div>
        <div class="field">
          <label>Weight</label>
          <input class="ci" id="ci-w-${idx}" type="number" value="${d.weight||1.0}" min="0.1" step="0.5" oninput="sync()">
        </div>
      </div>
      <div class="field">
        <label>Description</label>
        <input class="ci" id="ci-desc-${idx}" type="text" value="${esc(d.desc||'')}" placeholder="what this checks" oninput="sync()">
      </div>
      <div class="field">
        <label>Shell check</label>
        <textarea class="ci" id="ci-check-${idx}" oninput="sync()">${esc(d.check||'')}</textarea>
      </div>
    </div>`;
  document.getElementById('crit-list').appendChild(card);
  sync();
}

function toggleCrit(idx) {
  document.getElementById('crit-body-' + idx).classList.toggle('open');
}

function deleteCrit(idx) {
  const el = document.getElementById('crit-' + idx);
  if (el) el.remove();
  sync();
}

function getCriteria() {
  const items = [];
  document.querySelectorAll('.crit-card').forEach(card => {
    const idx = card.id.split('-')[1];
    const id    = val('ci-id-'   + idx);
    const desc  = val('ci-desc-' + idx);
    const check = val('ci-check-' + idx);
    const w     = parseFloat(val('ci-w-' + idx)) || 1.0;
    if (id || check) items.push({ id, desc, check, weight: w });
  });
  return items;
}

// ── YAML builder ──────────────────────────────────────────────────────────────
function buildYaml() {
  const id      = val('f-id')      || 'l-my-level';
  const name    = val('f-name')    || 'My Level';
  const cat     = val('f-cat')     || 'general';
  const tagsRaw = val('f-tags');
  const tags    = tagsRaw ? tagsRaw.split(',').map(t=>t.trim()).filter(Boolean) : [];
  const image   = val('f-image')   || 'python:3.11-slim';
  const workdir = val('f-workdir') || '/workspace';
  const apt     = val('f-apt').split(/[\s,]+/).filter(Boolean);
  const pip     = val('f-pip').split(/[\s,]+/).filter(Boolean);
  const npm     = val('f-npm').split(/[\s,]+/).filter(Boolean);
  const setup   = val('f-setup');
  const instr   = val('f-instr');
  const turns   = val('f-turns')   || '15';
  const timeout = val('f-timeout') || '90';
  const eff     = val('f-eff')     || '5';
  const port    = val('f-port');
  const ppath   = val('f-ppath')   || '/';
  const crits   = getCriteria();

  const yTags  = tags.length  ? '[' + tags.join(', ')  + ']' : '[]';
  const yApt   = apt.length   ? '[' + apt.join(', ')   + ']' : '[]';
  const yPip   = pip.length   ? '[' + pip.join(', ')   + ']' : '[]';
  const yNpm   = npm.length   ? '[' + npm.join(', ')   + ']' : '[]';
  const yTools = tools.map(t => '  - ' + t).join('\n');

  // Indent multiline strings for YAML block scalar
  const indentBlock = (s, indent) =>
    s.split('\n').map((l, i) => (i === 0 ? '' : ' '.repeat(indent)) + l).join('\n');
  const setupBlock = setup
    ? '|\n    ' + setup.split('\n').join('\n    ')
    : 'null';
  const instrBlock = instr
    ? '|\n    ' + instr.split('\n').join('\n    ')
    : '""';

  const critYaml = crits.map(c => {
    const checkBlock = c.check.includes('\n')
      ? '|\n        ' + c.check.split('\n').join('\n        ')
      : JSON.stringify(c.check);
    return [
      '    - id:          ' + c.id,
      '      description: "' + c.desc.replace(/"/g, '\\"') + '"',
      '      type:        script',
      '      check:       ' + checkBlock,
      '      weight:      ' + c.weight,
    ].join('\n');
  }).join('\n\n');

  const parts = [
    '# ──────────────────────────────────────────────────────────────────────',
    '# ' + name,
    '# Difficulty: ' + '★'.repeat(difficulty) + '☆'.repeat(5-difficulty),
    '# Generated by benchb0t level builder',
    '# ──────────────────────────────────────────────────────────────────────',
    '',
    'level:',
    '  id:         ' + id,
    '  name:       "' + name + '"',
    '  difficulty: ' + difficulty,
    '  category:   ' + cat,
    '  tags:       ' + yTags,
    '',
    'container:',
    '  image:       ' + image,
    '  working_dir: ' + workdir,
    '  env:',
    '    TASK_ENV: sandbox',
    '  volumes: []',
    '  packages:',
    '    apt: ' + yApt,
    '    pip: ' + yPip,
    '    npm: ' + yNpm,
    '  setup_script: ' + setupBlock,
    '',
  ];

  // preview block — only emit when a port is specified
  if (port) {
    parts.push(
      'preview:',
      '  port: ' + port,
      '  path: "' + ppath + '"',
      '',
    );
  }

  parts.push(
    'task:',
    '  instruction: ' + instrBlock,
    '  context: null',
    '  max_turns:  ' + turns,
    '  timeout_s:  ' + timeout,
    '',
    'tools:',
    yTools || '  - bash',
    '',
    'evaluation:',
    '  type:              script',
    '  efficiency_target: ' + eff,
    '  criteria:',
    critYaml || '    []',
  );

  return parts.join('\n');
}

// ── Live preview with syntax highlighting ─────────────────────────────────────
function sync() {
  const yaml  = buildYaml();
  const lines = yaml.split('\n');
  const html  = lines.map(line => {
    const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    if (/^\s*#/.test(line))   return `<span class="yc">${esc(line)}</span>`;
    const m = line.match(/^(\s*)([\w_]+)(:)(.*)/);
    if (m) {
      const val = m[4];
      let vspan;
      if (/^\s*\|/.test(val))       vspan = `<span class="yd">${esc(val)}</span>`;
      else if (/^\s*[\d.]+$/.test(val.trim())) vspan = `<span class="yn">${esc(val)}</span>`;
      else if (/^\s*"/.test(val))   vspan = `<span class="ys">${esc(val)}</span>`;
      else if (/^\s*\[/.test(val))  vspan = `<span class="ys">${esc(val)}</span>`;
      else                          vspan = `<span class="yv">${esc(val)}</span>`;
      return `${esc(m[1])}<span class="yk">${esc(m[2])}</span><span class="yd">${esc(m[3])}</span>${vspan}`;
    }
    if (/^\s*-\s/.test(line))  return `<span class="ys">${esc(line)}</span>`;
    return `<span class="yv">${esc(line)}</span>`;
  }).join('\n');
  document.getElementById('yaml-out').innerHTML = html;
}

// ── Save ──────────────────────────────────────────────────────────────────────
async function saveLevel() {
  const id       = val('f-id') || 'l-my-level';
  const filename = id + '.yaml';
  const content  = buildYaml();
  const res = await fetch('/api/levels/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename, content }),
  });
  const data = await res.json();
  const msg = document.getElementById('save-msg');
  if (res.ok) {
    msg.textContent = '✔ SAVED → levels/' + filename;
    msg.style.color = 'var(--green)';
  } else {
    msg.textContent = '✘ ' + (data.error || 'error');
    msg.style.color = 'var(--red)';
  }
  msg.style.display = 'inline';
  setTimeout(() => { msg.style.display = 'none'; }, 3500);
}

function copyYaml() {
  navigator.clipboard.writeText(buildYaml()).catch(() => {});
  const btn = document.querySelector('.btn-copy');
  btn.textContent = '✔ COPIED';
  setTimeout(() => { btn.textContent = '📋 COPY'; }, 1500);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function val(id) { const el = document.getElementById(id); return el ? el.value.trim() : ''; }
function esc(s)  { return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function setField(id, v) {
  const el = document.getElementById(id);
  if (!el) return;
  const s = (v == null) ? '' : String(v);
  el.value = s;
  // sync associated selects (preset dropdowns)
  const sel = document.getElementById(id + '-sel');
  if (sel) {
    const opt = [...sel.options].find(o => o.value === s);
    if (opt) { sel.value = s; showCustomInput(id, false); }
    else if (s) { sel.value = 'custom'; showCustomInput(id, true); }
    else { sel.selectedIndex = 0; showCustomInput(id, false); }
  }
  // sync port select
  if (id === 'f-port') {
    const ps = document.getElementById('f-port-sel');
    if (ps) {
      if (!s) { ps.value = 'none'; showPortCustom(false); }
      else {
        const has = [...ps.options].find(o => o.value === s);
        ps.value = has ? s : 'custom';
        showPortCustom(ps.value === 'custom');
      }
    }
  }
  // sync category select
  if (id === 'f-cat') {
    const cs = document.getElementById('f-cat');
    if (cs.tagName === 'SELECT') {
      const has = [...cs.options].find(o => o.value === s);
      if (!has) {
        const opt = document.createElement('option');
        opt.value = s; opt.textContent = s;
        cs.appendChild(opt);
      }
      cs.value = s;
    }
  }
}

// ── Preset selects ────────────────────────────────────────────────────────────
function showCustomInput(fieldId, show) {
  const inp = document.getElementById(fieldId);
  if (inp) inp.style.display = show ? 'block' : 'none';
}

function setPreset(fieldId, val) {
  if (val === 'custom') {
    showCustomInput(fieldId, true);
    document.getElementById(fieldId).focus();
  } else {
    showCustomInput(fieldId, false);
    setField(fieldId, val);
  }
  sync();
}

// ── Image combobox — auto-show/hide pkg sections ───────────────────────────
const NODE_IMGS   = /^node:/i;
const PYTHON_IMGS = /^python:|^django:|^flask:/i;

function onImageChange(v) {
  // highlight relevant package inputs
  const isNode   = NODE_IMGS.test(v);
  const isPython = PYTHON_IMGS.test(v);
  const npmWrap  = document.getElementById('pkg-npm-wrap');
  const pipWrap  = document.getElementById('pkg-pip-wrap');
  const prevF    = document.getElementById('preview-fields');
  if (npmWrap) npmWrap.style.opacity  = isNode   ? '1' : '0.4';
  if (pipWrap) pipWrap.style.opacity  = isPython ? '1' : '0.4';
  // auto-suggest port for node images
  if (isNode) {
    const ps = document.getElementById('f-port-sel');
    if (ps && ps.value === 'none') { ps.value = '3000'; onPortPreset('3000'); }
  }
  sync();
}

// ── Port preset ───────────────────────────────────────────────────────────────
function showPortCustom(show) {
  const wrap = document.getElementById('f-port-custom-wrap');
  const ppath = document.getElementById('f-ppath-wrap');
  if (wrap) wrap.style.display = show ? 'block' : 'none';
  if (ppath) ppath.style.display = show || document.getElementById('f-port-sel')?.value !== 'none' ? 'flex' : 'none';
}

function onPortPreset(v) {
  if (v === 'none') {
    setField('f-port', '');
    showPortCustom(false);
    document.getElementById('f-ppath-wrap').style.display = 'none';
  } else if (v === 'custom') {
    showPortCustom(true);
    document.getElementById('f-port').focus();
  } else {
    showPortCustom(false);
    setField('f-port', v);
    document.getElementById('f-ppath-wrap').style.display = 'flex';
  }
  sync();
}

// ── Templates ─────────────────────────────────────────────────────────────────
function applyTemplate(type) {
  if (!type) return;
  // reset template dropdown so it can be reselected
  document.getElementById('tpl-sel').value = '';

  const T = {
    webapp: {
      name:    'Webapp Level',
      cat:     'webapp',
      tags:    'webapp,node,server',
      image:   'node:20-slim',
      workdir: '/workspace',
      apt:     'curl',
      pip:     '',
      npm:     '',
      setup:   'mkdir -p /workspace',
      instr: [
        'Create a simple Express.js web server that:',
        '1. Listens on port 3000 bound to 0.0.0.0 (NOT localhost)',
        '2. Has a GET / route that responds with {"status":"ok"}',
        '3. Starts in the background (use & at end of command)',
        '4. Verify it is running: curl -sf http://localhost:3000/',
        '',
        'Example server (server.js):',
        '  const express = require("express");',
        '  const app = express();',
        '  app.get("/", (req, res) => res.json({status:"ok"}));',
        '  app.listen(3000, "0.0.0.0", () => console.log("ready"));',
        '',
        'Start it: node server.js &',
        'Then verify: curl -sf http://localhost:3000/',
      ].join('\n'),
      turns:   '20',
      timeout: '120',
      eff:     '8',
      port:    '3000',
      ppath:   '/',
      diff:    2,
      tools:   ['bash', 'write_file'],
      criteria: [
        { id:'server_responds', desc:'server responds on port 3000', check:'curl -sf http://localhost:3000/', weight: 1.5 },
        { id:'server_file_exists', desc:'server.js exists in /workspace', check:'test -f /workspace/server.js', weight: 0.5 },
      ],
    },
    file: {
      name:    'File Transform Level',
      cat:     'file-operations',
      tags:    'beginner,file,transform',
      image:   'python:3.11-slim',
      workdir: '/workspace',
      apt:     '',
      pip:     '',
      npm:     '',
      setup:   'mkdir -p /workspace\nprintf "hello world\\nline two\\nline three" > /workspace/input.txt',
      instr:   'Read /workspace/input.txt, convert all text to UPPERCASE, and write the result to /workspace/output.txt.',
      turns:   '10',
      timeout: '60',
      eff:     '3',
      port:    '',
      ppath:   '/',
      diff:    1,
      tools:   ['bash', 'read_file', 'write_file'],
      criteria: [
        { id:'output_exists', desc:'output.txt exists', check:'test -f /workspace/output.txt', weight: 1.0 },
        { id:'output_uppercase', desc:'output is uppercase', check:'grep -q "HELLO WORLD" /workspace/output.txt', weight: 1.5 },
      ],
    },
    api: {
      name:    'API Fetch Level',
      cat:     'networking',
      tags:    'api,http,json',
      image:   'python:3.11-slim',
      workdir: '/workspace',
      apt:     'curl',
      pip:     'requests',
      npm:     '',
      setup:   'mkdir -p /workspace',
      instr:   'Fetch the JSON from https://httpbin.org/json and save the "slideshow.title" field value as plain text to /workspace/title.txt.',
      turns:   '10',
      timeout: '60',
      eff:     '4',
      port:    '',
      ppath:   '/',
      diff:    2,
      tools:   ['bash', 'write_file'],
      criteria: [
        { id:'title_file_exists', desc:'title.txt exists', check:'test -f /workspace/title.txt', weight: 1.0 },
        { id:'title_correct', desc:'title contains expected text', check:'grep -qi "sample" /workspace/title.txt', weight: 2.0 },
      ],
    },
    data: {
      name:    'Data Pipeline Level',
      cat:     'data',
      tags:    'data,csv,pandas,analysis',
      image:   'python:3.11-slim',
      workdir: '/workspace',
      apt:     '',
      pip:     'pandas',
      npm:     '',
      setup: [
        'mkdir -p /workspace',
        'python3 -c "',
        'import csv, random, datetime',
        'rows = [[\"date\",\"sales\"]]',
        'for i in range(30):',
        '    d = datetime.date(2024,1,1) + datetime.timedelta(days=i)',
        '    rows.append([str(d), random.randint(100,1000)])',
        'with open(\"/workspace/sales.csv\",\"w\",newline=\"\") as f:',
        '    csv.writer(f).writerows(rows)',
        '"',
      ].join('\n'),
      instr:   'Analyze /workspace/sales.csv and write a summary to /workspace/summary.txt that includes:\n- Total sales\n- Average daily sales (rounded to 2 decimal places)\n- The date with the highest sales',
      turns:   '15',
      timeout: '90',
      eff:     '5',
      port:    '',
      ppath:   '/',
      diff:    2,
      tools:   ['bash', 'read_file', 'write_file'],
      criteria: [
        { id:'summary_exists', desc:'summary.txt exists', check:'test -f /workspace/summary.txt', weight: 1.0 },
        { id:'has_total', desc:'summary mentions total', check:'grep -qi "total" /workspace/summary.txt', weight: 1.0 },
        { id:'has_average', desc:'summary mentions average', check:'grep -qi "average\\|avg" /workspace/summary.txt', weight: 1.0 },
      ],
    },
  };

  const tpl = T[type];
  if (!tpl) return;

  setField('f-name',    tpl.name);
  setField('f-id',      'l-' + type + '-level');
  setField('f-cat',     tpl.cat);
  setField('f-tags',    tpl.tags);
  setField('f-image',   tpl.image);
  onImageChange(tpl.image || '');
  setField('f-workdir', tpl.workdir);
  setField('f-apt',     tpl.apt);
  setField('f-pip',     tpl.pip);
  setField('f-npm',     tpl.npm);
  setField('f-setup',   tpl.setup);
  setField('f-instr',   tpl.instr);
  setField('f-turns',   tpl.turns);
  setField('f-timeout', tpl.timeout);
  setField('f-eff',     tpl.eff);
  setField('f-port',    tpl.port);
  setField('f-ppath',   tpl.ppath);
  setDiff(tpl.diff);

  // tools
  tools = [];
  ['bash','read_file','write_file','http_request'].forEach(t => {
    const cb  = document.querySelector(`#tc-${t} input`);
    const lbl = document.getElementById('tc-' + t);
    const on  = tpl.tools.includes(t);
    if (cb)  cb.checked = on;
    if (lbl) lbl.classList.toggle('checked', on);
    if (on)  tools.push(t);
  });

  // criteria
  document.getElementById('crit-list').innerHTML = '';
  critCount = 0;
  (tpl.criteria || []).forEach(c => addCriterion(c));

  sync();
}
</script>
</body>
</html>"""


@app.get("/analytics", response_class=HTMLResponse)
async def analytics() -> HTMLResponse:
    return HTMLResponse(ANALYTICS_HTML)


ANALYTICS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>benchb0t · analytics</title>
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
<style>
:root {
  --bg:    #0a0800; --panel: #100d00; --panel2: #151000;
  --b1:    #241e00; --b2:    #3a3000; --b3:    #4a3f00;
  --y1:    #ffd700; --y2:    #ffb300; --y3:    #ff8c00;
  --ydk:   #7a6000; --text:  #ffe87a; --dim:   #5a4a10; --dim2: #3a3000;
  --green: #39ff14; --red:   #ff3a3a;
  --font:  'Press Start 2P', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { min-height: 100%; background: var(--bg); color: var(--text); font-family: var(--font); font-size: 10px; }
body::after {
  content:''; position:fixed; inset:0; z-index:9999; pointer-events:none;
  background: repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,.07) 3px, rgba(0,0,0,.07) 4px);
}
a { color: var(--y2); text-decoration: none; }
a:hover { color: var(--y1); }

/* ── HEADER ── */
header {
  background: var(--panel); border-bottom: 3px solid var(--y1);
  padding: 0 20px; height: 52px;
  display: flex; align-items: center; gap: 14px;
  box-shadow: 0 2px 20px rgba(255,215,0,.15);
  position: sticky; top: 0; z-index: 100;
}
.logo { color: var(--y1); font-size: 13px; letter-spacing: 3px; text-shadow: 0 0 20px var(--y1); }
.sep  { color: var(--b3); }
.nav-lnk { font-size: 7px; color: var(--dim); letter-spacing: 2px; }
.nav-lnk:hover { color: var(--y1); }
.hdr-r { margin-left: auto; display: flex; gap: 14px; align-items: center; }
.badge { font-size: 6px; padding: 3px 8px; border: 1px solid var(--dim); color: var(--dim); letter-spacing: 1px; }

/* ── PAGE ── */
.page { max-width: 1300px; margin: 0 auto; padding: 20px 20px 40px; }

/* ── SECTION TITLES ── */
.sec-title {
  font-size: 7px; letter-spacing: 3px; color: var(--dim);
  text-transform: uppercase; margin-bottom: 12px;
  padding-bottom: 5px; border-bottom: 1px solid var(--b2);
}

/* ── SUMMARY CARDS ── */
.cards { display: flex; gap: 10px; margin-bottom: 28px; flex-wrap: wrap; }
.card {
  background: var(--panel); border: 2px solid var(--b2);
  padding: 14px 16px; min-width: 130px; flex: 1;
}
.card-val { font-size: 22px; color: var(--y1); text-shadow: 0 0 16px rgba(255,215,0,.4); margin-bottom: 7px; }
.card-lbl { font-size: 6px; color: var(--dim); letter-spacing: 2px; }

/* ── TWO-COL LAYOUT ── */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 28px; }
@media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }

/* ── TABLES ── */
.tbl-wrap { background: var(--panel); border: 2px solid var(--b2); overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th {
  text-align: left; font-size: 5px; letter-spacing: 2px; color: var(--dim);
  padding: 7px 10px; border-bottom: 2px solid var(--b2); text-transform: uppercase;
  white-space: nowrap;
}
td { font-size: 7px; padding: 6px 10px; border-bottom: 1px solid var(--b1); white-space: nowrap; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--panel2); }
.td-score { color: var(--y1); }
.td-dim   { color: var(--dim); font-size: 6px; }
.td-stars { letter-spacing: 2px; }
.rank { color: var(--ydk); font-size: 6px; }
.bar-cell { min-width: 80px; }
.mini-bar-o { height: 4px; background: var(--bg); border: 1px solid var(--b2); }
.mini-bar-i { height: 100%; background: linear-gradient(90deg, var(--ydk), var(--y1)); }
.diff-dots  { letter-spacing: 2px; color: var(--y1); font-size: 9px; }
.pass-rate  { font-size: 6px; }
.pass-rate.good { color: var(--green); }
.pass-rate.mid  { color: var(--y2); }
.pass-rate.bad  { color: var(--red); }

/* ── MODEL FINGERPRINT ── */
.fingerprint-wrap {
  background: var(--panel); border: 2px solid var(--b2);
  margin-bottom: 28px;
}
.fp-header {
  padding: 10px 14px; border-bottom: 2px solid var(--b1);
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
}
.fp-body { display: flex; align-items: flex-start; gap: 20px; padding: 14px; flex-wrap: wrap; }
.fp-select {
  background: var(--bg); border: 2px solid var(--b2);
  color: var(--text); font-family: var(--font); font-size: 7px;
  padding: 5px 8px; outline: none; cursor: pointer;
}
.fp-select:focus { border-color: var(--y1); }
.fp-select option { background: var(--panel); }
.radar-svg { flex-shrink: 0; }
.fp-dims { flex: 1; min-width: 160px; display: flex; flex-direction: column; gap: 8px; justify-content: center; }
.fp-dim-row { display: flex; align-items: center; gap: 6px; }
.fp-dim-label { font-size: 5px; color: var(--dim); letter-spacing: 1px; width: 64px; flex-shrink: 0; }
.fp-dim-bar-o { flex: 1; height: 6px; background: var(--bg); border: 1px solid var(--b2); }
.fp-dim-bar-i { height: 100%; background: linear-gradient(90deg, var(--ydk), var(--y1)); transition: width .5s ease; }
.fp-dim-pct { font-size: 6px; color: var(--text); width: 28px; text-align: right; flex-shrink: 0; }

/* ── RECENT RUNS TABLE ── */
.run-model  { color: var(--y2); }
.run-level  { color: var(--text); }
.run-time   { color: var(--dim); font-size: 6px; }
.run-dur    { color: var(--dim); font-size: 6px; }
.s-on  { color: var(--y1); }
.s-off { color: var(--b3); }

/* ── EMPTY STATE ── */
.empty { text-align: center; padding: 40px; color: var(--dim); font-size: 8px; line-height: 2.5; }
</style>
</head>
<body>
<header>
  <span class="logo">benchb0t</span>
  <span class="sep">│</span>
  <span style="font-size:8px;letter-spacing:2px;color:var(--y2)">ANALYTICS</span>
  <div class="hdr-r">
    <a class="nav-lnk" href="/builder">BUILDER</a>
    <a class="nav-lnk" href="/">← LIVE DASHBOARD</a>
  </div>
</header>

<div class="page">

  <!-- SUMMARY CARDS -->
  <div class="sec-title">overview</div>
  <div class="cards" id="cards">
    <div class="card"><div class="card-val" id="s-runs">—</div><div class="card-lbl">total runs</div></div>
    <div class="card"><div class="card-val" id="s-models">—</div><div class="card-lbl">models tested</div></div>
    <div class="card"><div class="card-val" id="s-levels">—</div><div class="card-lbl">levels run</div></div>
    <div class="card"><div class="card-val" id="s-avg">—</div><div class="card-lbl">avg score</div></div>
    <div class="card"><div class="card-val" id="s-best">—</div><div class="card-lbl">best score</div></div>
    <div class="card"><div class="card-val" id="s-stars">—</div><div class="card-lbl">total stars</div></div>
  </div>

  <!-- MODEL FINGERPRINT -->
  <div class="sec-title">model fingerprint</div>
  <div class="fingerprint-wrap">
    <div class="fp-header">
      <span style="font-size:6px;color:var(--dim);letter-spacing:2px">SELECT MODEL</span>
      <select class="fp-select" id="fp-sel" onchange="renderFingerprint()">
        <option value="">— pick a model —</option>
      </select>
      <span style="font-size:6px;color:var(--dim)" id="fp-meta"></span>
    </div>
    <div class="fp-body">
      <svg class="radar-svg" id="radar" width="200" height="200" viewBox="0 0 200 200">
        <!-- reference diamond -->
        <polygon points="100,20 180,100 100,180 20,100"
          fill="none" stroke="#241e00" stroke-width="1"/>
        <!-- axis lines -->
        <line x1="100" y1="20" x2="100" y2="180" stroke="#1a1600" stroke-width="1"/>
        <line x1="20"  y1="100" x2="180" y2="100" stroke="#1a1600" stroke-width="1"/>
        <!-- 50% ring -->
        <polygon points="100,60 140,100 100,140 60,100"
          fill="none" stroke="#1a1600" stroke-width="1" stroke-dasharray="3,3"/>
        <!-- model polygon -->
        <polygon id="radar-poly" points="100,100 100,100 100,100 100,100"
          fill="rgba(255,215,0,.12)" stroke="#ffd700" stroke-width="2"/>
        <!-- axis labels -->
        <text x="100" y="14" text-anchor="middle" font-size="7" fill="#5a4a10" font-family="monospace">COMPLETE</text>
        <text x="188" y="103" text-anchor="start" font-size="7" fill="#5a4a10" font-family="monospace">EFF</text>
        <text x="100" y="194" text-anchor="middle" font-size="7" fill="#5a4a10" font-family="monospace">PATH</text>
        <text x="12"  y="103" text-anchor="end" font-size="7" fill="#5a4a10" font-family="monospace">REC</text>
      </svg>
      <div class="fp-dims">
        <div class="fp-dim-row">
          <span class="fp-dim-label">COMPLETION</span>
          <div class="fp-dim-bar-o"><div class="fp-dim-bar-i" id="fp-c" style="width:0%"></div></div>
          <span class="fp-dim-pct" id="fp-cv">—</span>
        </div>
        <div class="fp-dim-row">
          <span class="fp-dim-label">EFFICIENCY</span>
          <div class="fp-dim-bar-o"><div class="fp-dim-bar-i" id="fp-e" style="width:0%"></div></div>
          <span class="fp-dim-pct" id="fp-ev">—</span>
        </div>
        <div class="fp-dim-row">
          <span class="fp-dim-label">RECOVERY</span>
          <div class="fp-dim-bar-o"><div class="fp-dim-bar-i" id="fp-r" style="width:0%"></div></div>
          <span class="fp-dim-pct" id="fp-rv">—</span>
        </div>
        <div class="fp-dim-row">
          <span class="fp-dim-label">PATH QUAL</span>
          <div class="fp-dim-bar-o"><div class="fp-dim-bar-i" id="fp-p" style="width:0%"></div></div>
          <span class="fp-dim-pct" id="fp-pv">—</span>
        </div>
      </div>
    </div>
  </div>

  <!-- MODEL LEADERBOARD + LEVEL BREAKDOWN -->
  <div class="two-col">
    <div>
      <div class="sec-title">model leaderboard</div>
      <div class="tbl-wrap">
        <table id="model-tbl">
          <thead><tr>
            <th>#</th><th>model</th><th>runs</th>
            <th>avg score</th><th>best</th>
            <th>turns</th><th>timeouts</th>
          </tr></thead>
          <tbody id="model-body"></tbody>
        </table>
      </div>
    </div>
    <div>
      <div class="sec-title">level breakdown</div>
      <div class="tbl-wrap">
        <table id="level-tbl">
          <thead><tr>
            <th>diff</th><th>level</th><th>runs</th>
            <th>avg</th><th>best</th>
            <th>pass rate</th><th>avg turns</th>
          </tr></thead>
          <tbody id="level-body"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- RECENT RUNS -->
  <div class="sec-title">recent runs</div>
  <div class="tbl-wrap">
    <table>
      <thead><tr>
        <th>time</th><th>model</th><th>level</th>
        <th>score</th><th>stars</th>
        <th>turns</th><th>tools</th><th>duration</th>
      </tr></thead>
      <tbody id="runs-body"></tbody>
    </table>
  </div>

</div><!-- /page -->

<script>
let allModels = [];

async function load() {
  const [sum, models, levels, runs] = await Promise.all([
    fetch('/api/stats/summary').then(r => r.json()).catch(() => ({})),
    fetch('/api/stats/models').then(r => r.json()).catch(() => []),
    fetch('/api/stats/levels').then(r => r.json()).catch(() => []),
    fetch('/api/runs?limit=50').then(r => r.json()).catch(() => []),
  ]);

  allModels = models;
  renderSummary(sum);
  renderModelTable(models);
  renderLevelTable(levels);
  renderRunsTable(runs);
  populateModelSelect(models);
  if (models.length > 0) renderFingerprint(models[0]);
}

// ── Summary cards ──────────────────────────────────────────────────────────
function renderSummary(s) {
  setText('s-runs',   s.total_runs    ?? '0');
  setText('s-models', s.total_models  ?? '0');
  setText('s-levels', s.total_levels  ?? '0');
  setText('s-avg',    (s.avg_score    ?? 0) + '');
  setText('s-best',   (s.best_score   ?? 0) + '');
  setText('s-stars',  (s.total_stars  ?? 0) + ' ★');
}

// ── Model leaderboard ──────────────────────────────────────────────────────
function renderModelTable(models) {
  const tbody = document.getElementById('model-body');
  if (!models.length) { tbody.innerHTML = '<tr><td colspan="7" class="empty">no runs yet — start a benchmark!</td></tr>'; return; }
  tbody.innerHTML = models.map((m, i) => `
    <tr onclick="renderFingerprint(allModels[${i}])" style="cursor:pointer">
      <td class="rank">${i+1}</td>
      <td class="run-model">${esc(m.model)}</td>
      <td>${m.run_count}</td>
      <td>
        <div class="td-score">${m.avg_score}</div>
        <div class="bar-cell"><div class="mini-bar-o"><div class="mini-bar-i" style="width:${m.avg_score}%"></div></div></div>
      </td>
      <td class="td-score">${m.best_score}</td>
      <td class="td-dim">${m.avg_turns ?? '—'}</td>
      <td style="color:${m.timeouts > 0 ? 'var(--red)' : 'var(--dim)'}">${m.timeouts ?? 0}</td>
    </tr>`).join('');
}

// ── Level breakdown ────────────────────────────────────────────────────────
function renderLevelTable(levels) {
  const tbody = document.getElementById('level-body');
  if (!levels.length) { tbody.innerHTML = '<tr><td colspan="7" class="empty">no runs yet</td></tr>'; return; }
  tbody.innerHTML = levels.map(l => {
    const pr   = l.pass_rate != null ? Math.round(l.pass_rate * 100) : null;
    const pCls = pr == null ? '' : pr >= 80 ? 'good' : pr >= 50 ? 'mid' : 'bad';
    const dots = '★'.repeat(l.difficulty) + '☆'.repeat(5 - l.difficulty);
    return `<tr>
      <td><span class="diff-dots" style="font-size:7px">${dots}</span></td>
      <td class="run-level">${esc(l.level_name || l.level_id)}</td>
      <td>${l.run_count}</td>
      <td class="td-score">${l.avg_score}</td>
      <td class="td-score">${l.best_score}</td>
      <td class="pass-rate ${pCls}">${pr != null ? pr + '%' : '—'}</td>
      <td class="td-dim">${l.avg_turns ?? '—'}</td>
    </tr>`;
  }).join('');
}

// ── Recent runs ────────────────────────────────────────────────────────────
function renderRunsTable(runs) {
  const tbody = document.getElementById('runs-body');
  if (!runs.length) { tbody.innerHTML = '<tr><td colspan="8" class="empty">no runs yet — start a benchmark to see data here</td></tr>'; return; }
  tbody.innerHTML = runs.map(r => {
    const stars = r.stars || 0;
    const starHtml = '<span class="s-on">' + '★'.repeat(stars) + '</span><span class="s-off">' + '☆'.repeat(5-stars) + '</span>';
    const dt = new Date(r.ts * 1000);
    const timeStr = dt.toISOString().replace('T', ' ').slice(0, 16);
    return `<tr>
      <td class="run-time">${timeStr}</td>
      <td class="run-model">${esc(r.model)}</td>
      <td class="run-level">${esc(r.level_id)}</td>
      <td class="td-score">${(r.score_total || 0).toFixed(1)}</td>
      <td class="td-stars">${starHtml}</td>
      <td class="td-dim">${r.turns ?? '—'}</td>
      <td class="td-dim">${r.tool_calls_n ?? '—'}</td>
      <td class="run-dur">${(r.duration_s || 0).toFixed(1)}s</td>
    </tr>`;
  }).join('');
}

// ── Model fingerprint ──────────────────────────────────────────────────────
function populateModelSelect(models) {
  const sel = document.getElementById('fp-sel');
  sel.innerHTML = '<option value="">— pick a model —</option>' +
    models.map((m, i) => `<option value="${i}">${esc(m.model)}</option>`).join('');
  if (models.length) sel.value = '0';
}

function renderFingerprint(m) {
  if (!m) {
    const idx = document.getElementById('fp-sel').value;
    m = allModels[parseInt(idx)];
    if (!m) return;
  }
  // Also sync selector
  const sel = document.getElementById('fp-sel');
  const idx = allModels.findIndex(x => x.model === m.model);
  if (idx >= 0) sel.value = String(idx);

  const c = m.avg_completion      ?? 0;
  const e = m.avg_efficiency      ?? 0;
  const r = m.avg_self_correction ?? 0;
  const p = m.avg_path_quality    ?? 0;

  // Update bars
  setBar('fp-c', c); setText('fp-cv', c + '%');
  setBar('fp-e', e); setText('fp-ev', e + '%');
  setBar('fp-r', r); setText('fp-rv', r + '%');
  setBar('fp-p', p); setText('fp-pv', p + '%');

  // Update meta
  setText('fp-meta', `runs: ${m.run_count}  avg: ${m.avg_score}  best: ${m.best_score}`);

  // Update SVG radar polygon
  // axes: top=completion, right=efficiency, bottom=path_quality, left=self_correction
  const R = 80;   // max radius from center (100,100)
  const cx = 100, cy = 100;
  const topX    = cx,         topY    = cy - R * (c / 100);
  const rightX  = cx + R * (e / 100), rightY  = cy;
  const bottomX = cx,         bottomY = cy + R * (p / 100);
  const leftX   = cx - R * (r / 100), leftY   = cy;
  const poly = document.getElementById('radar-poly');
  if (poly) poly.setAttribute('points',
    `${topX},${topY} ${rightX},${rightY} ${bottomX},${bottomY} ${leftX},${leftY}`
  );
}

// ── Helpers ────────────────────────────────────────────────────────────────
function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }
function setBar(id, pct) { const el = document.getElementById(id); if (el) el.style.width = pct + '%'; }
function esc(s) { return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

load();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
