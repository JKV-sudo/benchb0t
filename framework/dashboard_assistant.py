"""
framework/dashboard_assistant.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tool schemas and state helpers for BenchBot-AI dashboard control.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
import uuid

from framework.config import KNOWN_TOOLS, LEVEL_FILENAME_RE, LevelValidationError, load_level_config
from framework.dashboard_models import ChatRequest, ProviderRequest, RunRequest

_TOOL_ALIASES = {
    "curl": "http_request",
    "http": "http_request",
    "fetch": "http_request",
    "read": "read_file",
    "write": "write_file",
    "edit": "patch_file",
    "edit_file": "patch_file",
    "background": "run_background",
    "serve": "run_background",
    "ls": "list_dir",
}
_RECOMMENDED_LEVEL_ID_RE = re.compile(r"^l\d{1,3}-[a-z0-9-]+$")


def assistant_control_prompt(page: str) -> str:
    base = (
        "You can directly operate benchb0t. "
        "When the user asks to configure providers, pick levels, toggle saved artifacts, "
        "run readiness checks, start runs, stop runs, or create benchmark levels, "
        "use the available tools instead of merely describing the steps. "
        "After using tools, briefly confirm what changed."
    )
    if page == "builder":
        return (
            base
            + " On the builder page, prefer creating a structured level patch and save the YAML "
              "when the user explicitly asks to create or save the level. "
              f"Valid agent tools are: {', '.join(sorted(KNOWN_TOOLS))}."
        )
    return base


def assistant_tool_schemas(page: str) -> list[dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_benchbot_status",
                "description": "Read the current benchb0t dashboard state, run selection, and active run status.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_benchbot_levels",
                "description": "List available benchmark levels that can be selected in the dashboard.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "detect_benchbot_providers",
                "description": "Probe common local and configured API providers and return ready-to-use presets.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_benchbot_preflight",
                "description": "Run readiness checks for Docker, API reachability, levels, and harness availability.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "configure_benchbot_provider",
                "description": "Set provider credentials and model selection for slot 1 or 2 in the dashboard UI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slot": {"type": "integer", "enum": [1, 2]},
                        "base_url": {"type": "string"},
                        "model": {"type": "string"},
                        "api_key": {"type": "string"},
                        "label": {"type": "string"},
                    },
                    "required": ["slot", "base_url", "model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "configure_benchbot_run",
                "description": "Change the selected level, run scope, and artifact capture options in the dashboard UI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string"},
                        "all_levels": {"type": "boolean"},
                        "capture_preview_screenshot": {"type": "boolean"},
                        "save_result_bundle": {"type": "boolean"},
                        "save_container_snapshot": {"type": "boolean"},
                        "parallel_compare": {"type": "boolean"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "start_benchbot_run",
                "description": "Start a benchmark run using the current dashboard configuration, optionally overriding run options.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string"},
                        "all_levels": {"type": "boolean"},
                        "capture_preview_screenshot": {"type": "boolean"},
                        "save_result_bundle": {"type": "boolean"},
                        "save_container_snapshot": {"type": "boolean"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stop_benchbot_run",
                "description": "Stop the currently running benchmark batch.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_benchbot_level",
                "description": (
                    "Create a new benchb0t level draft from structured fields, validate it, "
                    "and optionally save it to levels/<id>.yaml. Also returns a builder-form patch."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                        "category": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "image": {"type": "string"},
                        "working_dir": {"type": "string"},
                        "apt": {"type": "array", "items": {"type": "string"}},
                        "pip": {"type": "array", "items": {"type": "string"}},
                        "npm": {"type": "array", "items": {"type": "string"}},
                        "setup_script": {"type": "string"},
                        "instruction": {"type": "string"},
                        "max_turns": {"type": "integer", "minimum": 1},
                        "timeout_s": {"type": "integer", "minimum": 1},
                        "tools": {
                            "type": "array",
                            "items": {"type": "string", "enum": sorted(KNOWN_TOOLS)},
                        },
                        "efficiency_target": {"type": "integer", "minimum": 0},
                        "criteria": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "description": {"type": "string"},
                                    "check": {"type": "string"},
                                    "weight": {"type": "number"},
                                },
                                "required": ["id", "description", "check"],
                            },
                        },
                        "preview_port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "preview_path": {"type": "string"},
                        "retry_enabled": {"type": "boolean"},
                        "retry_max": {"type": "integer", "minimum": 0},
                        "retry_penalty": {"type": "number", "minimum": 0},
                        "retry_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                        "filename": {"type": "string"},
                        "save": {"type": "boolean"},
                    },
                    "required": [
                        "id",
                        "name",
                        "difficulty",
                        "category",
                        "image",
                        "instruction",
                        "tools",
                        "criteria",
                    ],
                },
            },
        },
    ]
    return tools


def chat_request_to_provider_dicts(req: ChatRequest, creds: dict[str, Any]) -> list[dict[str, str]]:
    providers: list[dict[str, str]] = []
    if req.providers:
        for provider in req.providers:
            base_url = provider.base_url.strip()
            model = provider.model.strip()
            if not base_url or not model:
                continue
            providers.append(
                {
                    "base_url": base_url,
                    "model": model,
                    "api_key": provider.api_key.strip(),
                    "label": (provider.label or model).strip(),
                }
            )
    elif req.base_url.strip() and req.model.strip():
        providers.append(
            {
                "base_url": req.base_url.strip(),
                "model": req.model.strip(),
                "api_key": req.api_key.strip(),
                "label": req.model.strip(),
            }
        )
    elif creds.get("providers"):
        for provider in creds.get("providers", []):
            base_url = str(provider.get("base_url", "")).strip()
            model = str(provider.get("model", "")).strip()
            if not base_url or not model:
                continue
            providers.append(
                {
                    "base_url": base_url,
                    "model": model,
                    "api_key": str(provider.get("api_key", "")).strip(),
                    "label": str(provider.get("label", model)).strip(),
                }
            )
    elif creds.get("base_url") and creds.get("model"):
        providers.append(
            {
                "base_url": str(creds.get("base_url", "")).strip(),
                "model": str(creds.get("model", "")).strip(),
                "api_key": str(creds.get("api_key", "")).strip(),
                "label": str(creds.get("model", "")).strip(),
            }
        )
    return providers


def build_initial_assistant_state(req: ChatRequest, creds: dict[str, Any]) -> dict[str, Any]:
    providers = chat_request_to_provider_dicts(req, creds)
    return {
        "providers": providers,
        "parallel_compare": bool(req.parallel_compare or len(providers) > 1),
        "level": req.level.strip(),
        "all_levels": bool(req.all_levels),
        "capture_preview_screenshot": bool(req.capture_preview_screenshot),
        "save_result_bundle": bool(req.save_result_bundle),
        "save_container_snapshot": bool(req.save_container_snapshot),
    }


def assistant_state_ui_patch(assistant_state: dict[str, Any]) -> dict[str, Any]:
    providers = [
        {
            "base_url": provider.get("base_url", ""),
            "model": provider.get("model", ""),
            "api_key": provider.get("api_key", ""),
            "label": provider.get("label", provider.get("model", "")),
        }
        for provider in assistant_state.get("providers", [])
    ]
    return {
        "providers": providers,
        "parallel_compare": bool(assistant_state.get("parallel_compare", False) or len(providers) > 1),
        "level": assistant_state.get("level", ""),
        "all_levels": bool(assistant_state.get("all_levels", False)),
        "capture_preview_screenshot": bool(assistant_state.get("capture_preview_screenshot", True)),
        "save_result_bundle": bool(assistant_state.get("save_result_bundle", False)),
        "save_container_snapshot": bool(assistant_state.get("save_container_snapshot", False)),
    }


def build_run_request_from_assistant_state(assistant_state: dict[str, Any]) -> RunRequest:
    providers = [
        ProviderRequest(
            base_url=provider.get("base_url", ""),
            model=provider.get("model", ""),
            api_key=provider.get("api_key", ""),
            label=provider.get("label", provider.get("model", "")),
        )
        for provider in assistant_state.get("providers", [])
    ]
    first = providers[0] if providers else ProviderRequest(base_url="", model="", api_key="", label="")
    return RunRequest(
        base_url=first.base_url,
        model=first.model,
        api_key=first.api_key,
        level=str(assistant_state.get("level", "")),
        all_levels=bool(assistant_state.get("all_levels", False)),
        capture_preview_screenshot=bool(assistant_state.get("capture_preview_screenshot", True)),
        save_result_bundle=bool(assistant_state.get("save_result_bundle", False)),
        save_container_snapshot=bool(assistant_state.get("save_container_snapshot", False)),
        providers=providers,
    )


def list_levels_for_assistant(project_dir: Path, *, limit: int = 12) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted((project_dir / "levels").glob("*.yaml")):
        try:
            level = load_level_config(path)
            if level.is_deprecated:
                continue
            items.append(
                {
                    "id": level.level.id,
                    "name": level.level.name,
                    "path": str(path),
                    "difficulty": level.level.difficulty,
                    "category": level.level.category,
                }
            )
        except LevelValidationError:
            continue
        if len(items) >= max(1, limit):
            break
    return items


def resolve_level_reference(project_dir: Path, value: str) -> str:
    requested = value.strip()
    if not requested:
        return ""

    candidate = Path(requested)
    if candidate.exists():
        return str(candidate)

    levels_dir = project_dir / "levels"
    direct = levels_dir / requested
    if direct.exists():
        return str(direct)
    if not requested.endswith(".yaml") and (levels_dir / f"{requested}.yaml").exists():
        return str(levels_dir / f"{requested}.yaml")

    for path in sorted(levels_dir.glob("*.yaml")):
        try:
            level = load_level_config(path)
        except LevelValidationError:
            continue
        if level.is_deprecated:
            continue
        if requested in {level.level.id, path.stem, path.name}:
            return str(path)

    raise ValueError(f"Level {requested!r} not found")


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
        return [item.strip() for item in raw if item.strip()]
    return [str(value).strip()]


def _inline_yaml_list(values: list[str]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(json.dumps(value, ensure_ascii=False) for value in values) + "]"


def build_level_patch_from_args(args: dict[str, Any]) -> dict[str, Any]:
    tools: list[str] = []
    unknown_tools: list[str] = []
    for tool in _as_string_list(args.get("tools")) or ["bash"]:
        token = tool.strip().lower()
        normalized = _TOOL_ALIASES.get(token, token)
        if normalized not in KNOWN_TOOLS:
            unknown_tools.append(tool)
            continue
        if normalized not in tools:
            tools.append(normalized)
    if unknown_tools:
        allowed = ", ".join(sorted(KNOWN_TOOLS))
        raise ValueError(f"unknown tools: {', '.join(unknown_tools)}. Allowed tools: {allowed}")

    criteria = []
    for criterion in args.get("criteria", []) or []:
        criteria.append(
            {
                "id": str(criterion.get("id", "")).strip(),
                "desc": str(criterion.get("description", "")).strip(),
                "check": str(criterion.get("check", "")).strip(),
                "weight": float(criterion.get("weight", 1.0) or 1.0),
            }
        )

    patch = {
        "id": str(args.get("id", "")).strip(),
        "name": str(args.get("name", "")).strip(),
        "difficulty": int(args.get("difficulty", 1) or 1),
        "category": str(args.get("category", "webapp")).strip(),
        "tags": _as_string_list(args.get("tags")),
        "image": str(args.get("image", "node:20-slim")).strip() or "node:20-slim",
        "working_dir": str(args.get("working_dir", "/workspace")).strip() or "/workspace",
        "apt": _as_string_list(args.get("apt")),
        "pip": _as_string_list(args.get("pip")),
        "npm": _as_string_list(args.get("npm")),
        "setup_script": str(args.get("setup_script", "") or ""),
        "instruction": str(args.get("instruction", "")).strip(),
        "max_turns": int(args.get("max_turns", 20) or 20),
        "timeout_s": int(args.get("timeout_s", 120) or 120),
        "tools": tools or ["bash"],
        "efficiency_target": int(args.get("efficiency_target", 8) or 8),
        "criteria": criteria,
        "preview_port": args.get("preview_port"),
        "preview_path": str(args.get("preview_path", "/") or "/"),
        "retry_enabled": bool(args.get("retry_enabled", False)),
        "retry_max": int(args.get("retry_max", 2) or 2),
        "retry_penalty": float(args.get("retry_penalty", 10.0) or 10.0),
        "retry_threshold": float(args.get("retry_threshold", 0.5) or 0.5),
    }
    if patch["preview_port"] and "run_background" not in patch["tools"]:
        patch["tools"].append("run_background")
    return patch


def render_level_yaml_from_patch(patch: dict[str, Any]) -> str:
    def block(value: str, indent: int = 4) -> str:
        if not value:
            return '""'
        return "|\n" + "\n".join(" " * indent + line for line in value.splitlines())

    crit_lines: list[str] = []
    for criterion in patch.get("criteria", []):
        check = str(criterion.get("check", ""))
        check_value = (
            "|\n" + "\n".join(" " * 8 + line for line in check.splitlines())
            if "\n" in check
            else json.dumps(check, ensure_ascii=False)
        )
        crit_lines.append(
            "\n".join(
                [
                    f"    - id:          {criterion.get('id', '')}",
                    f"      description: {json.dumps(str(criterion.get('desc', '')), ensure_ascii=False)}",
                    "      type:        script",
                    f"      check:       {check_value}",
                    f"      weight:      {float(criterion.get('weight', 1.0) or 1.0)}",
                ]
            )
        )

    lines = [
        "# ──────────────────────────────────────────────────────────────────────",
        f"# {patch['name']}",
        "# Generated by BenchBot-AI",
        "# ──────────────────────────────────────────────────────────────────────",
        "",
        "level:",
        f"  id:         {patch['id']}",
        f"  name:       {json.dumps(patch['name'], ensure_ascii=False)}",
        f"  difficulty: {int(patch['difficulty'])}",
        f"  category:   {patch['category']}",
        f"  tags:       {_inline_yaml_list(patch.get('tags', []))}",
        "",
        "container:",
        f"  image:       {patch['image']}",
        f"  working_dir: {patch['working_dir']}",
        "  env:",
        "    TASK_ENV: sandbox",
        "  volumes: []",
        "  packages:",
        f"    apt: {_inline_yaml_list(patch.get('apt', []))}",
        f"    pip: {_inline_yaml_list(patch.get('pip', []))}",
        f"    npm: {_inline_yaml_list(patch.get('npm', []))}",
        f"  setup_script: {block(str(patch.get('setup_script', '')))}",
        "",
    ]

    preview_port = patch.get("preview_port")
    if preview_port:
        lines.extend(
            [
                "preview:",
                f"  port: {int(preview_port)}",
                f"  path: {json.dumps(str(patch.get('preview_path', '/') or '/'), ensure_ascii=False)}",
                "",
            ]
        )

    lines.extend(
        [
            "task:",
            f"  instruction: {block(patch['instruction'])}",
            "  context: null",
            f"  max_turns:  {int(patch['max_turns'])}",
            f"  timeout_s:  {int(patch['timeout_s'])}",
            "",
            "tools:",
        ]
    )
    for tool in patch.get("tools", []) or ["bash"]:
        lines.append(f"  - {tool}")
    lines.extend(
        [
            "",
            "evaluation:",
            "  type:              script",
            f"  efficiency_target: {int(patch['efficiency_target'])}",
            "  criteria:",
        ]
    )
    if crit_lines:
        lines.extend(crit_lines)
    else:
        lines.append("    []")

    if patch.get("retry_enabled"):
        lines.extend(
            [
                "",
                "forced_retry:",
                "  enabled:              true",
                f"  max_retries:          {int(patch.get('retry_max', 2))}",
                f"  penalty_per_retry:    {float(patch.get('retry_penalty', 10.0))}",
                f"  completion_threshold: {float(patch.get('retry_threshold', 0.5))}",
            ]
        )

    return "\n".join(lines) + "\n"


def validate_level_content(project_dir: Path, filename: str, content: str) -> None:
    _load_validated_level_content(project_dir, filename, content)


def lint_level_content(project_dir: Path, filename: str, content: str) -> dict[str, Any]:
    try:
        normalized_name, model = _load_validated_level_content(project_dir, filename, content)
    except ValueError as exc:
        issues = _extract_validation_issues(str(exc))
        return {
            "valid": False,
            "errors": issues,
            "warnings": [],
            "filename": filename.strip() or "unnamed.yaml",
        }

    return {
        "valid": True,
        "errors": [],
        "warnings": _lint_level_warnings(normalized_name, model),
        "filename": normalized_name,
    }


def _normalize_level_filename(filename: str) -> str:
    name = filename.strip()
    if not name.endswith(".yaml"):
        name += ".yaml"
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("invalid filename")
    return name


def _load_validated_level_content(
    project_dir: Path,
    filename: str,
    content: str,
) -> tuple[str, Any]:
    levels_dir = project_dir / "levels"
    levels_dir.mkdir(parents=True, exist_ok=True)
    name = _normalize_level_filename(filename)
    temp_path = levels_dir / f".benchb0t-validate-{uuid.uuid4().hex[:8]}-{name}"
    try:
        temp_path.write_text(content, encoding="utf-8")
        model = load_level_config(temp_path)
        return name, model
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _extract_validation_issues(message: str) -> list[str]:
    issues: list[str] = []
    for raw_line in message.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Invalid level config in "):
            continue
        if line.startswith("- "):
            issues.append(line[2:].strip())
        else:
            issues.append(line)
    return issues or [message.strip()]


def _lint_level_warnings(filename: str, model) -> list[str]:
    warnings: list[str] = []
    if not LEVEL_FILENAME_RE.match(filename):
        warnings.append("Recommended filename format is l<number>-slug.yaml.")
    if not _RECOMMENDED_LEVEL_ID_RE.match(model.level.id):
        warnings.append("Recommended level.id format is l<number>-slug.")
    if not model.evaluation.criteria:
        warnings.append("Add at least one evaluation criterion so the level can be scored meaningfully.")
    if model.level.category in {"webapp", "fullstack"} and model.preview is None:
        warnings.append("Webapp and fullstack levels should usually define a preview port.")
    return warnings


def save_level_content(project_dir: Path, filename: str, content: str) -> Path:
    name = _normalize_level_filename(filename)
    validate_level_content(project_dir, name, content)
    dest = project_dir / "levels" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return dest
