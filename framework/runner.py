"""
framework/runner.py
~~~~~~~~~~~~~~~~~~~
benchb0t main runner — the orchestration heart of the framework.

Usage
─────
  python -m framework.runner --level levels/l1-single-file.yaml \\
                              --harness harnesses/slavko.yaml

  # Run all levels sequentially:
  python -m framework.runner --all-levels --harness harnesses/slavko.yaml

Flow per level
──────────────
  1. Load level YAML + harness YAML + config.yaml
  2. Start Docker container (LevelContainer)
  3. Open Recorder
  4. Feed task instruction to agent via AgentAPI
  5. Agentic loop: receive response → dispatch tool calls → feed results back
  6. Stop loop on finish_reason="stop" | max_turns | timeout
  7. Score the session (Scorer)
  8. Close Recorder + stop container
  9. Print result summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from framework.api import AgentAPI
from framework.artifacts import (
    capture_preview_screenshot as capture_preview_screenshot_artifact,
    run_artifacts_dir,
    save_container_snapshot as save_container_snapshot_artifact,
    save_result_bundle as save_result_bundle_artifact,
)
from framework.config import (
    FrameworkConfig,
    FrameworkConfigError,
    HarnessValidationError,
    LevelValidationError,
    LoadedFrameworkConfig,
    load_framework_config,
    load_harness_config,
    load_level_config,
)
from framework.container import LevelContainer, ContainerError
from framework.recorder import Recorder
from framework.scorer import Scorer
from framework.store import Store
from framework.types import RunResult

def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

logger = logging.getLogger(__name__)


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "bash": {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command inside the sandbox container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"],
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file inside the container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path inside the container"}
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file inside the container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Absolute path inside the container"},
                    "content": {"type": "string", "description": "File content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "http_request": {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Make an HTTP request from inside the container using curl.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method":  {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                    "url":     {"type": "string"},
                    "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                    "body":    {"type": "string"},
                },
                "required": ["method", "url"],
            },
        },
    },
    "list_dir": {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory inside the container with file sizes and types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute directory path to list"},
                },
                "required": ["path"],
            },
        },
    },
    "run_background": {
        "type": "function",
        "function": {
            "name": "run_background",
            "description": (
                "Launch a long-running command in the background (e.g. a dev server or API). "
                "Returns the PID, waits `wait_seconds`, then reports if the process is still alive "
                "plus the last lines of its log. Much safer than using bash with & because you get "
                "real feedback on whether the process started successfully."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run in the background",
                    },
                    "wait_seconds": {
                        "type": "integer",
                        "description": "Seconds to wait before checking if the process is alive (1-10, default 4)",
                    },
                    "log_file": {
                        "type": "string",
                        "description": "Path for capturing stdout+stderr (default: /tmp/bg_proc.log)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    "patch_file": {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Replace the first occurrence of an exact string in a file. "
                "Ideal for targeted edits (e.g. change a config value, fix a single line) "
                "without rewriting the entire file. Fails if the pattern is not found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "old":  {"type": "string", "description": "Exact string to find (must be present)"},
                    "new":  {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
}


# ── Tool dispatch ─────────────────────────────────────────────────────────────

def dispatch_tool(
    tool_name: str,
    args: dict[str, Any],
    container: LevelContainer,
) -> tuple[int, str]:
    """
    Execute a tool call inside the container and return (exit_code, output).
    """
    if tool_name == "bash":
        return container.exec(args["command"])

    if tool_name == "read_file":
        try:
            content = container.read_file(args["path"])
            return 0, content
        except ContainerError as exc:
            return 1, str(exc)

    if tool_name == "write_file":
        try:
            container.write_file(args["path"], args["content"])
            return 0, f"Written to {args['path']}"
        except ContainerError as exc:
            return 1, str(exc)

    if tool_name == "http_request":
        method  = args.get("method", "GET")
        url     = args["url"]
        headers = args.get("headers", {})
        body    = args.get("body", "")
        header_flags = " ".join(f'-H "{k}: {v}"' for k, v in headers.items())
        data_flag    = f"-d '{body}'" if body else ""
        cmd = f"curl -s -X {method} {header_flags} {data_flag} '{url}'"
        return container.exec(cmd)

    if tool_name == "list_dir":
        return container.exec(f"ls -lah {args['path']} 2>&1")

    if tool_name == "run_background":
        cmd        = args["command"]
        wait       = max(1, min(int(args.get("wait_seconds", 4)), 10))
        log_file   = args.get("log_file", "/tmp/bg_proc.log")
        # Launch detached: subshell + disown so it survives when bash exits
        launch = f"bash -c '({cmd}) > {log_file} 2>&1 & echo $!'"
        ec, out = container.exec(launch)
        pid = out.strip().splitlines()[-1] if out.strip() else ""
        if not pid.isdigit():
            return 1, f"Failed to launch background process.\n{out}"
        # Wait, then check liveness
        container.exec(f"sleep {wait}")
        _, alive_out = container.exec(
            f"kill -0 {pid} 2>/dev/null && echo RUNNING || echo EXITED"
        )
        _, tail = container.exec(f"tail -30 {log_file} 2>/dev/null || echo '(no log)'")
        status = alive_out.strip()
        return 0, f"PID={pid}  status={status}\n\n--- last log lines ---\n{tail}"

    if tool_name == "patch_file":
        path = args["path"]
        old  = args["old"]
        new  = args["new"]
        # Write a temp Python patch script into the container via put_archive
        # (avoids ALL shell quoting issues — repr() produces safe Python literals)
        script = (
            "import sys\n"
            f"path = {repr(path)}\n"
            f"old  = {repr(old)}\n"
            f"new  = {repr(new)}\n"
            "try:\n"
            "    content = open(path).read()\n"
            "except FileNotFoundError:\n"
            "    print(f'ERROR: file not found: {path}'); sys.exit(1)\n"
            "if old not in content:\n"
            "    print(f'ERROR: pattern not found in {path}'); sys.exit(1)\n"
            "open(path, 'w').write(content.replace(old, new, 1))\n"
            "print(f'Patched {path} — 1 occurrence replaced')\n"
        )
        try:
            container.write_file("/tmp/_benchbot_patch.py", script)
        except ContainerError as exc:
            return 1, f"patch_file: could not write helper script: {exc}"
        return container.exec("python3 /tmp/_benchbot_patch.py")

    return 1, f"Unknown tool: {tool_name}"


# ── Agent loop ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPTS: dict[str, str] = {
    "unguided": (
        "You are a capable software engineer agent running inside a Docker sandbox. "
        "Complete the assigned task. "
        "When done, respond with a plain text summary — no more tool calls."
    ),
    "guided": (
        "You are a precise, step-by-step software engineering agent running inside a "
        "Docker container sandbox.\n\n"
        "GENERAL APPROACH:\n"
        "1. Start by using list_dir and read_file to understand what is already scaffolded.\n"
        "2. Write new files with write_file; use patch_file for targeted edits to existing files.\n"
        "3. Use run_background (not bash with &) to launch servers or long-running processes — "
        "it returns the PID, waits, then reports if the process is alive and shows the log tail.\n"
        "4. After starting a server, always verify it responds (e.g. curl http://localhost:PORT/) "
        "before declaring done.\n"
        "5. If a process fails to start, read its log file for the error.\n\n"
        "Complete the task efficiently and correctly. "
        "When done, respond with a plain text summary — no more tool calls."
    ),
}


def _resolve_system_prompt(level_cfg: dict[str, Any], mode: str) -> str:
    """
    Return the system prompt for the given mode.

    Priority:
      1. level YAML  → modes.<mode>.system_prompt
      2. Built-in    → _SYSTEM_PROMPTS[mode]
      3. Fallback    → _SYSTEM_PROMPTS["unguided"]
    """
    mode_cfg = level_cfg.get("modes", {}).get(mode, {})
    if mode_cfg.get("system_prompt"):
        return mode_cfg["system_prompt"].strip()
    return _SYSTEM_PROMPTS.get(mode, _SYSTEM_PROMPTS["unguided"])


def run_agent_loop(
    api: AgentAPI,
    container: LevelContainer,
    recorder: Recorder,
    task_cfg: dict[str, Any],
    tools_list: list[str],
    system_prompt: str | None = None,
) -> bool:
    """
    Agentic turn loop.

    Returns True if the agent completed without timeout, False if timed out.
    """
    max_turns  = task_cfg.get("max_turns", 20)
    timeout_s  = task_cfg.get("timeout_s", 120)
    deadline   = time.time() + timeout_s

    # Build tool schemas for the tools declared in the level YAML
    active_tools = [TOOL_SCHEMAS[t] for t in tools_list if t in TOOL_SCHEMAS]

    # Seed the conversation with the task instruction
    system_msg = system_prompt or _SYSTEM_PROMPTS["unguided"]
    messages: list[dict[str, Any]] = [
        {"role": "system",  "content": system_msg},
        {"role": "user",    "content": task_cfg.get("instruction", "")},
    ]
    if task_cfg.get("context"):
        messages.append({"role": "user", "content": f"Context:\n{task_cfg['context']}"})

    recorder.record_message("user", task_cfg.get("instruction", ""))

    for turn in range(max_turns):
        if time.time() > deadline:
            logger.warning("⏰ Timeout after turn %d", turn)
            return False

        logger.info("── Turn %d/%d ──────────────────────────────", turn + 1, max_turns)

        try:
            response = api.chat_with_stream(
                messages,
                tools=active_tools if active_tools else None,
                on_text_delta=lambda delta: recorder.record_message_delta("assistant", delta),
            )
        except RuntimeError as exc:
            logger.warning(
                "Streaming failed on turn %d, falling back to non-streaming: %s",
                turn + 1,
                exc,
            )
            try:
                response = api.chat(messages, tools=active_tools if active_tools else None)
            except RuntimeError as exc:
                logger.error("API error on turn %d: %s", turn + 1, exc)
                break

        choice  = response["choices"][0]
        message = choice["message"]
        reason  = choice.get("finish_reason", "")

        # Record assistant message
        recorder.record_message("assistant", message.get("content") or "")

        # No tool calls → agent is done
        if reason == "stop" or not message.get("tool_calls"):
            logger.info("✅ Agent finished (finish_reason=%s)", reason)
            return True

        # Dispatch tool calls
        tool_calls = message.get("tool_calls", [])
        tool_results_for_messages: list[dict[str, Any]] = []

        for tc in tool_calls:
            fn        = tc["function"]
            tool_name = fn["name"]
            call_id   = tc.get("id", "")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError as exc:
                # Agent generated malformed JSON for tool arguments — this is a bug
                # Record the error and continue so the agent sees the failure
                args = {}
                logger.error(
                    "Tool %s has malformed JSON arguments (call_id=%s): %s",
                    tool_name, call_id, exc
                )

            cid = recorder.record_tool_call(tool_name, args, call_id=call_id)

            if time.time() > deadline:
                recorder.record_tool_result(cid, "Timeout", exit_code=1, tool=tool_name, args=args)
                return False

            exit_code, output = dispatch_tool(tool_name, args, container)
            recorder.record_tool_result(cid, output, exit_code=exit_code, tool=tool_name, args=args)

            tool_results_for_messages.append({
                "tool_call_id": call_id,
                "role":         "tool",
                "name":         tool_name,
                "content":      output,
            })

        # Append assistant turn + tool results to message history
        messages.append({
            "role":       "assistant",
            "content":    message.get("content") or "",
            "tool_calls": tool_calls,
        })
        messages.extend(tool_results_for_messages)

    logger.warning("Agent reached max_turns (%d) without stopping", max_turns)
    return False


# ── Level runner ──────────────────────────────────────────────────────────────

def _resolve_preview_linger_seconds(
    framework_cfg: FrameworkConfig,
    level_cfg: dict[str, Any],
    host_preview_port: int | None,
) -> int:
    """Return how long a preview container should stay alive after scoring."""
    if not host_preview_port:
        return 0

    preview_cfg = level_cfg.get("preview", {}) or {}
    override = preview_cfg.get("linger_seconds")
    if override is not None:
        return int(override)

    return int(framework_cfg.framework.preview_linger_seconds)


def run_level(
    level_path: Path,
    harness_path: Path,
    framework_cfg: LoadedFrameworkConfig,
    store: "Store | None" = None,
    mode: str = "unguided",
    capture_preview_screenshot: bool = False,
    save_result_bundle: bool = False,
    save_container_snapshot: bool = False,
) -> RunResult:
    """
    Run a single level and return a result dict with score + metadata.

    Parameters
    ----------
    mode : str
        "guided" or "unguided". Controls the system prompt sent to the agent.
        "guided" provides tool-usage hints and a step-by-step approach.
        "unguided" gives only a minimal system prompt — the agent must figure
        out the approach itself.
    """
    cfg = framework_cfg.config
    level_cfg = load_level_config(level_path).model_dump(mode="python", exclude_none=True)
    harness_cfg = load_harness_config(harness_path).model_dump(mode="python", exclude_none=True)
    # Harness YAML can override mode: guided | unguided
    mode = harness_cfg.get("harness", {}).get("mode", mode)
    task_cfg = cfg.agent.apply_task_defaults(level_cfg.get("task", {}))

    level_id     = level_cfg["level"]["id"]
    harness_name = harness_cfg["harness"]["name"]

    logger.info("═══════════════════════════════════════════════")
    logger.info("🎮 Level: %s  │  Harness: %s  │  Mode: %s", level_id, harness_name, mode)
    logger.info("═══════════════════════════════════════════════")

    runs_dir = framework_cfg.runs_dir
    compress = cfg.recorder.compress
    scoring_cfg = cfg.scoring.model_dump(mode="python")

    recorder = Recorder(runs_dir, level_id, harness_name, compress=compress)

    api = AgentAPI.from_harness(
        harness_cfg.get("harness", {}),
        defaults=cfg.agent.api_defaults(task_cfg),
    )

    timed_out = False
    score_summary: dict[str, Any] = {}
    host_preview_port: int | None = None   # set inside try-block if container starts OK
    preview_linger_seconds = 0
    preview_expires_at: float | None = None
    result_duration_s = 0.0
    session_ended = False
    provider_slot = max(1, int(os.getenv("BENCHBOT_PROVIDER_SLOT", "1") or "1"))
    provider_label = os.getenv("BENCHBOT_PROVIDER_LABEL") or os.getenv("BENCHBOT_MODEL", "") or harness_name
    panel_id = f"p{provider_slot}--{level_id}"
    container: LevelContainer | None = None
    preview_path = level_cfg.get("preview", {}).get("path", "/")
    artifact_dir: Path | None = None
    artifacts: list[dict[str, Any]] = []

    try:
        recorder.start(
            level_cfg,
            harness_cfg,
            model=api.model,
            base_url=os.getenv("BENCHBOT_BASE_URL", ""),
            provider_slot=provider_slot,
            provider_label=provider_label,
            panel_id=panel_id,
        )

        # LevelContainer created inside try so Docker errors are caught cleanly
        # Merge preview port into container cfg so the container can publish it.
        container_cfg = dict(level_cfg.get("container", {}))
        preview_cfg   = level_cfg.get("preview", {})
        if preview_cfg.get("port"):
            container_cfg["preview_port"] = preview_cfg["port"]
        container = LevelContainer(
            level_cfg=container_cfg,
            framework_cfg=cfg.container.model_dump(mode="python"),
            level_id=level_id,
        )

        system_prompt = _resolve_system_prompt(level_cfg, mode)
        logger.info("System prompt mode=%s (%d chars)", mode, len(system_prompt))

        # Forced-retry config (optional block in level YAML)
        retry_cfg             = level_cfg.get("forced_retry", {})
        retry_enabled         = bool(retry_cfg.get("enabled", False))
        max_retries           = int(retry_cfg.get("max_retries", 2))
        penalty_per_retry     = float(retry_cfg.get("penalty_per_retry", 10.0))
        completion_threshold  = float(retry_cfg.get("completion_threshold", 0.5))

        container.start()

        timed_out = not run_agent_loop(
            api=api,
            container=container,
            recorder=recorder,
            task_cfg=task_cfg,
            tools_list=level_cfg.get("tools", []),
            system_prompt=system_prompt,
        )

        # Capture the dynamic host port Docker assigned (may be None for non-preview levels)
        host_preview_port = container.host_preview_port

        # Notify the dashboard of the real host port so the iframe uses it.
        # This event is written after container.start() — the only point where
        # Docker has assigned the ephemeral port.
        if host_preview_port:
            recorder.record_preview_ready(host_preview_port, path=preview_path)

        # Score while the container is still alive (script checks need it)
        scorer = Scorer(level_cfg.get("evaluation", {}), scoring_cfg)
        breakdown = scorer.score(
            tool_calls=recorder.tool_calls,
            timed_out=timed_out,
            container_exec=container.exec,
        )

        # ── Forced-retry loop ─────────────────────────────────────────────
        # If enabled and the agent did not meet the completion threshold,
        # give it another attempt — each retry costs penalty_per_retry points.
        retries_used = 0
        if retry_enabled and max_retries > 0:
            while (
                breakdown.completion < completion_threshold
                and retries_used < max_retries
            ):
                retries_used += 1
                logger.warning(
                    "🔁 Forced retry %d/%d — completion %.0f%% < threshold %.0f%%",
                    retries_used, max_retries,
                    breakdown.completion * 100,
                    completion_threshold * 100,
                )
                # Re-run the agent loop (container state is preserved)
                timed_out = not run_agent_loop(
                    api=api,
                    container=container,
                    recorder=recorder,
                    task_cfg=task_cfg,
                    tools_list=level_cfg.get("tools", []),
                    system_prompt=system_prompt,
                )
                breakdown = scorer.score(
                    tool_calls=recorder.tool_calls,
                    timed_out=timed_out,
                    container_exec=container.exec,
                )

        # Apply accumulated retry penalty to the final breakdown
        breakdown.penalty_retry = retries_used * penalty_per_retry
        if retries_used:
            logger.info(
                "Retry penalty applied: %d × %.1f = −%.1f pts",
                retries_used, penalty_per_retry, breakdown.penalty_retry,
            )

        score_summary = breakdown.to_score_summary()
        result_duration_s = round(time.time() - recorder._started, 2)
        preview_linger_seconds = _resolve_preview_linger_seconds(cfg, level_cfg, host_preview_port)
        if preview_linger_seconds > 0:
            preview_expires_at = time.time() + preview_linger_seconds

        if capture_preview_screenshot or save_result_bundle or save_container_snapshot:
            artifact_dir = run_artifacts_dir(runs_dir, recorder.run_id)

        if artifact_dir and host_preview_port and capture_preview_screenshot:
            screenshot = capture_preview_screenshot_artifact(
                host_port=host_preview_port,
                preview_path=preview_path,
                dest_path=artifact_dir / "preview.png",
            )
            if screenshot:
                artifacts.append(screenshot)
                recorder.record_artifact(
                    screenshot["kind"],
                    screenshot["path"],
                    label=screenshot["label"],
                    url=screenshot.get("url", ""),
                    size_bytes=screenshot.get("size_bytes", 0),
                )

        if artifact_dir and save_container_snapshot:
            snapshot = save_container_snapshot_artifact(
                container=container,
                artifact_dir=artifact_dir,
                level_id=level_id,
                run_id=recorder.run_id,
            )
            if snapshot:
                artifacts.append(snapshot)
                recorder.record_artifact(
                    snapshot["kind"],
                    snapshot["path"],
                    label=snapshot["label"],
                    image_ref=snapshot.get("image_ref", ""),
                    size_bytes=snapshot.get("size_bytes", 0),
                )

        recorder.end(
            score=score_summary,
            timed_out=timed_out,
            preview_linger_seconds=preview_linger_seconds,
            preview_expires_at=preview_expires_at,
            artifacts=artifacts,
        )
        session_ended = True

        if host_preview_port and preview_linger_seconds > 0:
            logger.info(
                "Keeping preview alive on http://localhost:%d for %ds",
                host_preview_port,
                preview_linger_seconds,
            )
            time.sleep(preview_linger_seconds)

    except ContainerError as exc:
        logger.error("Container error in level %s: %s", level_id, exc)
        score_summary = {"error": str(exc), "total": 0}
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error in level %s: %s", level_id, exc, exc_info=True)
        score_summary = {"error": str(exc), "total": 0}
    finally:
        if not session_ended:
            result_duration_s = round(time.time() - recorder._started, 2)
            recorder.end(
                score=score_summary,
                timed_out=timed_out,
                preview_linger_seconds=preview_linger_seconds,
                preview_expires_at=preview_expires_at,
                artifacts=artifacts,
            )
        if container is not None:
            container.stop()

    level_meta = level_cfg.get("level", {})
    result = RunResult(
        run_id=recorder.run_id,
        ts=recorder._started,
        level_id=level_id,
        level_name=level_meta.get("name", level_id),
        difficulty=level_meta.get("difficulty", 1),
        harness=harness_name,
        mode=mode,
        model=os.getenv("BENCHBOT_MODEL", ""),
        base_url=os.getenv("BENCHBOT_BASE_URL", ""),
        log_path=str(recorder.path),
        timed_out=timed_out,
        score=score_summary,
        turns=recorder.turn_count,
        tool_calls_n=len(recorder.tool_calls),
        duration_s=result_duration_s,
        # Dynamic host port Docker assigned — only set for preview levels.
        # Consumers (dashboard iframe) use this; None means no preview.
        host_preview_port=host_preview_port,
        preview_linger_seconds=preview_linger_seconds,
        preview_expires_at=preview_expires_at,
        artifacts=artifacts,
    )

    if artifact_dir and save_result_bundle:
        bundle = save_result_bundle_artifact(
            artifact_dir=artifact_dir,
            run_id=recorder.run_id,
            result=result.to_dict(),
            log_path=recorder.path,
        )
        if bundle:
            artifacts.append(bundle)

    if store is not None:
        store.record_run(result.to_dict())

    _print_result(result.to_dict())
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_result(result: dict[str, Any]) -> None:
    score = result.get("score", {})
    total = score.get("total", 0)
    bar   = "█" * int(total / 5) + "░" * (20 - int(total / 5))
    mode  = result.get("mode", "unguided")
    print(f"\n{'─'*50}")
    print(f"  Level   : {result['level_id']}")
    print(f"  Harness : {result['harness']}  [{mode}]")
    print(f"  Score   : [{bar}] {total:.1f}/100")
    if result.get("timed_out"):
        print("  ⚠️  Timed out")
    print(f"  Log     : {result['log_path']}")
    print(f"{'─'*50}\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="benchb0t — LLM agent benchmark runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--level",      type=Path, help="Path to a single level YAML file")
    p.add_argument("--all-levels", action="store_true", help="Run all levels/*.yaml files")
    p.add_argument("--harness",    type=Path, required=True, help="Path to harness YAML file")
    p.add_argument("--config",     type=Path, default=Path("config.yaml"), help="Framework config (default: config.yaml)")
    p.add_argument("--env",        type=Path, default=Path(".env"), help=".env file to load (default: .env)")
    p.add_argument("--no-prompt",  action="store_true", help="Skip the interactive boot screen (use ENV vars directly)")
    p.add_argument(
        "--capture-preview-screenshot",
        action="store_true",
        help="Capture a preview screenshot for levels that start a web server",
    )
    p.add_argument(
        "--save-result-bundle",
        action="store_true",
        help="Save a ZIP bundle with result.json, agentlog, and saved artifacts",
    )
    p.add_argument(
        "--save-container-snapshot",
        action="store_true",
        help="Commit the final container state to a Docker image and save metadata",
    )
    p.add_argument(
        "--mode",
        choices=["guided", "unguided"],
        default="unguided",
        help=(
            "Agent system prompt mode (default: unguided). "
            "'guided' provides tool-usage hints and a step-by-step approach. "
            "'unguided' gives only a minimal system prompt. "
            "Can also be set per-harness via harness.mode in the harness YAML."
        ),
    )
    return p


def _normalize_url(url: str) -> str:
    """
    Ensure the endpoint URL has a scheme and ends with /v1.

    Examples
    --------
    "svslai02:8080"             → "http://svslai02:8080/v1"
    "http://svslai02:8080"      → "http://svslai02:8080/v1"
    "http://svslai02:8080/v1"   → "http://svslai02:8080/v1"   (unchanged)
    "https://api.openai.com/v1" → "https://api.openai.com/v1" (unchanged)
    """
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    if not url.rstrip("/").endswith("/v1"):
        url = url.rstrip("/") + "/v1"
    return url


def _boot_screen(framework_cfg: FrameworkConfig) -> None:
    """
    Interactive boot prompt shown once at startup.

    Asks the user for endpoint URL, model name, and optional API key.
    Values are written directly into os.environ so the rest of the framework
    picks them up via AgentAPI.from_harness() without needing a .env file.

    The user can press Enter to accept the suggested default for any field.
    """
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   🎮  benchb0t  —  LLM Agent Benchmark Framework    ║")
    print(f"║   v{framework_cfg.framework.version:<51}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Gather current defaults (from env or config)
    current_url   = os.getenv("BENCHBOT_BASE_URL", "http://localhost:11434/v1")
    current_model = os.getenv("BENCHBOT_MODEL",    "llama3")
    current_key   = os.getenv("BENCHBOT_API_KEY",  "")

    print("  Configure your endpoint (press Enter to keep the default):")
    print()

    # ── Endpoint URL ─────────────────────────────────────────────────────────
    prompt_url = f"  Base URL [{current_url}]: "
    try:
        val = input(prompt_url).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Aborted.")
        sys.exit(0)
    base_url = _normalize_url(val or current_url)
    os.environ["BENCHBOT_BASE_URL"] = base_url

    # ── Model name ────────────────────────────────────────────────────────────
    prompt_model = f"  Model     [{current_model}]: "
    try:
        val = input(prompt_model).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Aborted.")
        sys.exit(0)
    model = val or current_model
    os.environ["BENCHBOT_MODEL"] = model

    # ── API key (optional) ────────────────────────────────────────────────────
    key_display = "(none — press Enter to skip)" if not current_key else f"{'*' * min(8, len(current_key))}…"
    prompt_key = f"  API key   [{key_display}]: "
    try:
        val = input(prompt_key).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Aborted.")
        sys.exit(0)
    api_key = val or current_key or "benchbot"
    os.environ["BENCHBOT_API_KEY"] = api_key

    print()
    print(f"  ✔  Endpoint : {base_url}")
    print(f"  ✔  Model    : {model}")
    print(f"  ✔  API key  : {'(none)' if api_key == 'benchbot' and not val and not current_key else '***'}")
    print()
    print("  💡  Live dashboard: python -m framework.dashboard")
    print("      then open  →  http://localhost:7860")
    print()


def main() -> None:
    args = _build_parser().parse_args()

    try:
        framework_cfg = load_framework_config(args.config)
    except (FileNotFoundError, FrameworkConfigError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    env_path = framework_cfg.resolve_path(args.env)
    if env_path.exists():
        load_dotenv(env_path)

    _configure_logging(framework_cfg.config.framework.log_level)

    # Interactive boot screen — always shown unless --no-prompt is passed
    if not getattr(args, "no_prompt", False):
        _boot_screen(framework_cfg.config)

    harness_path = framework_cfg.resolve_path(args.harness)
    if not harness_path.exists():
        logger.error("Harness file not found: %s", harness_path)
        sys.exit(1)

    # Collect levels to run
    if args.all_levels:
        levels = []
        for candidate in sorted(framework_cfg.levels_dir.glob("*.yaml")):
            try:
                if load_level_config(candidate).is_deprecated:
                    continue
            except LevelValidationError as exc:
                logger.error(str(exc))
                sys.exit(1)
            levels.append(candidate)
        if not levels:
            logger.error("No level files found in %s", framework_cfg.levels_dir)
            sys.exit(1)
    elif args.level:
        level_path = framework_cfg.resolve_path(args.level)
        if not level_path.exists():
            logger.error("Level file not found: %s", level_path)
            sys.exit(1)
        try:
            load_level_config(level_path)
        except LevelValidationError as exc:
            logger.error(str(exc))
            sys.exit(1)
        levels = [level_path]
    else:
        logger.error("Specify --level <file> or --all-levels")
        sys.exit(1)

    try:
        load_harness_config(harness_path)
    except HarnessValidationError as exc:
        logger.error(str(exc))
        sys.exit(1)

    store = Store(framework_cfg.db_path).init()

    results = []
    for level_path in levels:
        result = run_level(
            level_path,
            harness_path,
            framework_cfg,
            store=store,
            mode=args.mode,
            capture_preview_screenshot=args.capture_preview_screenshot,
            save_result_bundle=args.save_result_bundle,
            save_container_snapshot=args.save_container_snapshot,
        )
        results.append(result)

    # Final summary
    if len(results) > 1:
        avg = sum(r["score"].get("total", 0) for r in results) / len(results)
        mode_used = results[0].get("mode", "unguided")
        print(f"\n{'═'*50}")
        print(f"  Run complete — {len(results)} levels │ mode: {mode_used} │ avg score: {avg:.1f}/100")
        print(f"{'═'*50}\n")


if __name__ == "__main__":
    main()
