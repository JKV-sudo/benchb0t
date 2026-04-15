"""
framework/dashboard_context.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Context builders for the dashboard's embedded chat assistant.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import yaml

from framework.recorder import load_agentlog
from framework.store import Store

logger = logging.getLogger(__name__)


def load_live_session_context(run_id: str, runs_dir: Path) -> str:
    if not run_id:
        return ""

    try:
        for log_path in sorted(runs_dir.glob(f"*_{run_id}.agentlog")):
            events = load_agentlog(log_path)
            lines: list[str] = [f"LIVE SESSION LOG (run_id={run_id}):"]
            for event in events:
                event_type = event.get("type", "?")
                if event_type == "session_start":
                    lines.append(
                        f"  [START] model={event.get('model','?')} "
                        f"level={event.get('level_id','?')} mode={event.get('mode','?')}"
                    )
                elif event_type == "tool_call":
                    args_str = str(event.get("args", {}))[:120]
                    lines.append(f"  [TOOL] {event.get('tool','?')}({args_str})")
                elif event_type == "tool_result":
                    output = str(event.get("output", ""))[:120].replace("\n", " ")
                    lines.append(f"  [RES] exit={event.get('exit_code',0)} {output}")
                elif event_type == "message" and event.get("role") == "assistant":
                    content = str(event.get("content", ""))[:200].replace("\n", " ")
                    lines.append(f"  [AI] {content}")
                elif event_type == "session_end":
                    score = event.get("score", {})
                    lines.append(
                        f"  [END] score={score.get('total',0):.1f} "
                        f"timed_out={event.get('timed_out',False)} "
                        f"duration={event.get('duration_s',0):.1f}s"
                    )
            return "\n".join(lines)
    except (FileNotFoundError, json.JSONDecodeError, IOError) as exc:
        # Log file missing, corrupted, or unreadable
        logger.warning("live session context load failed for %s: %s", run_id, exc)

    return ""


def build_analytics_context(store: Store | None) -> str:
    parts: list[str] = [
        "You are BenchBot-AI on the ANALYTICS page. "
        "Your role: interpret benchmark data, compare model performance, "
        "spot trends, explain why models succeed or fail on specific levels. "
        "Be specific — cite exact scores, pass rates, turn counts. "
        "Suggest which model to use for which task type. "
        "Format: plain text, no markdown.",
        "",
    ]
    if not store:
        return "\n".join(parts)

    try:
        models = store.get_model_stats()
        if models:
            parts.append("MODEL LEADERBOARD:")
            for index, model in enumerate(models[:15], 1):
                parts.append(
                    f"  #{index} {model['model']}: avg={model['avg_score']} "
                    f"best={model['best_score']} runs={model['run_count']} "
                    f"stars={model.get('total_stars',0)} "
                    f"turns={model.get('avg_turns','?')} "
                    f"timeouts={model.get('timeouts',0)}"
                )
            parts.append("")

        levels = store.get_level_stats()
        if levels:
            parts.append("LEVEL DIFFICULTY vs PASS RATE:")
            for level in levels:
                pass_rate = round((level.get("pass_rate") or 0) * 100)
                parts.append(
                    f"  {level['level_id']} diff={level.get('difficulty',1)} "
                    f"avg={level['avg_score']} pass={pass_rate}% runs={level['run_count']}"
                )
            parts.append("")

        comparison = store.get_mode_comparison() if hasattr(store, "get_mode_comparison") else []
        if comparison:
            parts.append("GUIDED vs UNGUIDED COMPARISON (sample):")
            for row in comparison[:20]:
                parts.append(
                    f"  {row['model']} · {row['level_id']} · {row['mode']}: "
                    f"avg={row['avg_score']} turns={row['avg_turns']} timeouts={row['timeouts']}"
                )
            parts.append("")
    except (KeyError, TypeError, AttributeError, sqlite3.OperationalError) as exc:
        # Database access failures or malformed response — graceful degradation
        logger.warning("analytics context: %s", exc)

    return "\n".join(parts)


def build_builder_context(project_dir: Path, current_level_yaml: str = "") -> str:
    example_levels: list[str] = []
    levels_dir = project_dir / "levels"
    if levels_dir.exists():
        for path in sorted(levels_dir.glob("*.yaml"))[:5]:
            try:
                example_levels.append(path.read_text(encoding="utf-8")[:1200])
            except (FileNotFoundError, UnicodeDecodeError, OSError):
                # Example level unreadable — skip it
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
        for index, example in enumerate(example_levels, 1):
            parts.append(f"--- example {index} ---")
            parts.append(example[:800])
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


def build_chat_context(
    *,
    store: Store | None,
    project_dir: Path,
    runs_dir: Path,
    active_run_id: str = "",
    page: str = "dashboard",
    page_context: str = "",
) -> str:
    if page == "analytics":
        return build_analytics_context(store)
    if page == "builder":
        return build_builder_context(project_dir, current_level_yaml=page_context)

    parts: list[str] = [
        "You are BenchBot-AI, an embedded analyst inside the benchb0t LLM-agent "
        "benchmarking framework. You have real-time access to benchmark data and "
        "can answer questions about model performance, level difficulty, run logs, "
        "and scoring. Be concise, technical, and specific — cite actual numbers "
        "from the data. Format answers in plain text (no markdown).",
        "",
    ]

    if active_run_id:
        live_context = load_live_session_context(active_run_id, runs_dir)
        if live_context:
            parts.append("ACTIVE SESSION:")
            parts.append(live_context)
            parts.append("")

    if store:
        try:
            summary = store.get_summary()
            parts.append(
                f"BENCHMARK SUMMARY: {summary.get('total_runs',0)} total runs | "
                f"{summary.get('total_models',0)} models | "
                f"{summary.get('total_levels',0)} levels | "
                f"avg score {summary.get('avg_score',0)} | "
                f"best score {summary.get('best_score',0)} | "
                f"{summary.get('total_stars',0)} total stars"
            )
            parts.append("")

            models = store.get_model_stats()
            if models:
                parts.append("MODEL LEADERBOARD:")
                for index, model in enumerate(models[:10], 1):
                    parts.append(
                        f"  #{index} {model['model']}: avg={model['avg_score']} "
                        f"best={model['best_score']} runs={model['run_count']} "
                        f"stars={model.get('total_stars',0)} "
                        f"turns={model.get('avg_turns','?')} "
                        f"timeouts={model.get('timeouts',0)}"
                    )
                parts.append("")

            levels = store.get_level_stats()
            if levels:
                parts.append("LEVEL STATS:")
                for level in levels:
                    pass_rate = round((level.get("pass_rate") or 0) * 100)
                    parts.append(
                        f"  {level['level_id']} (diff={level.get('difficulty',1)}) "
                        f"avg={level['avg_score']} best={level['best_score']} "
                        f"pass_rate={pass_rate}% runs={level['run_count']}"
                    )
                parts.append("")

            recent = store.get_runs(limit=20)
            if recent:
                parts.append("RECENT RUNS (last 20, newest first):")
                for run in recent:
                    from datetime import datetime

                    ts = datetime.fromtimestamp(run["ts"]).strftime("%m-%d %H:%M")
                    parts.append(
                        f"  [{ts}] {run['model']} on {run['level_id']}: "
                        f"score={run['score_total']:.1f} stars={run['stars']} "
                        f"turns={run['turns']} tools={run['tool_calls_n']} "
                        f"{'TIMEOUT' if run.get('timed_out') else 'ok'}"
                    )
                parts.append("")
        except (KeyError, TypeError, AttributeError, sqlite3.OperationalError) as exc:
            # Database access failures or malformed response — graceful degradation
            logger.warning("chat context: DB query failed: %s", exc)

    levels_dir = project_dir / "levels"
    if levels_dir.exists():
        try:
            yaml_files = sorted(levels_dir.glob("*.yaml"))[:12]
            if yaml_files:
                parts.append("LEVEL DEFINITIONS:")
            for path in yaml_files:
                cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
                level = cfg.get("level", {})
                task = cfg.get("task", {})
                evaluation = cfg.get("evaluation", {})
                criteria = [criterion.get("id", "?") for criterion in evaluation.get("criteria", [])]
                parts.append(
                    f"  {level.get('id','?')} \"{level.get('name','?')}\" "
                    f"diff={level.get('difficulty',1)} cat={level.get('category','?')} "
                    f"max_turns={task.get('max_turns','?')} "
                    f"criteria=[{','.join(criteria)}]"
                )
                instruction = (task.get("instruction") or "").strip()
                if instruction:
                    parts.append(f"    instruction: {instruction[:200]}")
            parts.append("")
        except (FileNotFoundError, UnicodeDecodeError, yaml.YAMLError, OSError) as exc:
            # Level YAML file unreadable or malformed — graceful degradation
            logger.warning("chat context: level YAML read failed: %s", exc)

    if runs_dir.exists():
        try:
            log_files = sorted(
                runs_dir.glob("*.agentlog"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:3]
            if log_files:
                parts.append("RECENT AGENT LOGS (last 3 runs, tool calls only):")
            for log_file in log_files:
                try:
                    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
                    tool_events: list[str] = []
                    for line in lines:
                        try:
                            event = json.loads(line)
                            if event.get("type") in ("tool_call", "tool_result"):
                                tool = event.get("tool", event.get("name", "?"))
                                args = str(event.get("args", ""))[:80]
                                output = str(event.get("output", ""))[:120]
                                exit_code = event.get("exit_code", 0)
                                if event["type"] == "tool_call":
                                    tool_events.append(f"    CALL {tool}({args})")
                                else:
                                    tool_events.append(f"    -> exit={exit_code} {output[:80]}")
                        except json.JSONDecodeError:
                            # Malformed JSON line — skip it
                            continue
                    parts.append(f"  [{log_file.stem}]")
                    parts.extend(tool_events[:30])
                except (FileNotFoundError, UnicodeDecodeError, OSError):
                    # Agentlog file unreadable — skip it
                    continue
            parts.append("")
        except (FileNotFoundError, OSError) as exc:
            # Agentlog directory unreadable — graceful degradation
            logger.warning("chat context: agentlog read failed: %s", exc)

    return "\n".join(parts)
