"""
framework/dashboard_replay.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Helpers for turning raw agentlog events into replay-friendly timeline data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _clip_text(value: Any, limit: int = 200) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[:limit]


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        text = content.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    chunks = [
                        str(item.get("text", "")).strip()
                        for item in parsed
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    if chunks:
                        return " ".join(chunk for chunk in chunks if chunk).strip()
            except Exception:
                pass
        return text

    if isinstance(content, list):
        chunks = [
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(chunk for chunk in chunks if chunk).strip()

    return str(content or "").strip()


def score_to_stars(score_total: float) -> int:
    if score_total >= 95:
        return 5
    if score_total >= 80:
        return 4
    if score_total >= 60:
        return 3
    if score_total >= 35:
        return 2
    if score_total > 0:
        return 1
    return 0


def artifact_kind_for_path(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg"):
        return "preview_screenshot" if "preview" in name else "image"
    if name.endswith(".zip"):
        return "result_bundle"
    if name == "container-snapshot.json":
        return "container_snapshot"
    if name.endswith(".json"):
        return "json"
    return "file"


def build_artifact_records(runs_dir: Path, run_id: str) -> list[dict[str, Any]]:
    artifact_dir = Path(runs_dir) / "artifacts" / run_id
    if not artifact_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for path in sorted(artifact_dir.iterdir()):
        if not path.is_file():
            continue
        kind = artifact_kind_for_path(path)
        records.append(
            {
                "name": path.name,
                "kind": kind,
                "label": path.stem.replace("-", " "),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "is_image": kind in {"preview_screenshot", "image"},
                "url": f"/api/artifacts/{run_id}/{path.name}",
            }
        )
    return records


def build_run_replay(events: list[dict[str, Any]]) -> dict[str, Any]:
    timeline: list[dict[str, Any]] = []
    tool_calls: dict[str, dict[str, Any]] = {}
    turn_count = 0
    tool_call_count = 0
    preview: dict[str, Any] = {}
    session: dict[str, Any] = {}

    for index, event in enumerate(events):
        event_type = event.get("type", "unknown")
        ts = event.get("ts")

        if event_type == "session_start":
            session = {
                "run_id": event.get("run_id", ""),
                "level_id": event.get("level_id", ""),
                "level_name": event.get("level_name", ""),
                "model": event.get("model", ""),
                "provider_label": event.get("provider_label", ""),
                "started_at": ts,
            }
            timeline.append(
                {
                    "index": index,
                    "type": "session_start",
                    "ts": ts,
                    "label": "Session started",
                    "meta": session,
                }
            )
        elif event_type == "preview_ready":
            preview = {
                "host_preview_port": event.get("host_preview_port"),
                "path": event.get("preview_path", "/"),
            }
            timeline.append(
                {
                    "index": index,
                    "type": "preview_ready",
                    "ts": ts,
                    "label": "Preview ready",
                    "meta": preview,
                }
            )
        elif event_type == "message" and event.get("role") in {"user", "assistant"}:
            role = str(event.get("role"))
            text = _message_text(event.get("content", ""))
            turn = None
            if role == "assistant":
                turn_count += 1
                turn = turn_count

            timeline.append(
                {
                    "index": index,
                    "type": f"{role}_message",
                    "ts": ts,
                    "turn": turn,
                    "label": (
                        f"Assistant turn {turn_count}"
                        if role == "assistant"
                        else "User message"
                    ),
                    "text_preview": _clip_text(text, limit=320),
                }
            )
        elif event_type == "tool_call":
            tool_call_count += 1
            call_id = event.get("call_id", f"call-{index}")
            tool_calls[call_id] = {
                "tool": event.get("tool", ""),
                "args": event.get("args", {}),
                "call_index": tool_call_count,
            }
            timeline.append(
                {
                    "index": index,
                    "type": "tool_call",
                    "ts": ts,
                    "call_id": call_id,
                    "tool": event.get("tool", ""),
                    "call_index": tool_call_count,
                    "label": f"Tool call {tool_call_count}: {event.get('tool', '?')}",
                    "args": event.get("args", {}),
                }
            )
        elif event_type == "tool_result":
            call_id = event.get("call_id", "")
            meta = tool_calls.get(call_id, {})
            exit_code = int(event.get("exit_code", 0) or 0)
            timeline.append(
                {
                    "index": index,
                    "type": "tool_result",
                    "ts": ts,
                    "call_id": call_id,
                    "tool": meta.get("tool", ""),
                    "call_index": meta.get("call_index"),
                    "exit_code": exit_code,
                    "ok": exit_code == 0,
                    "label": f"Tool result: {meta.get('tool', '?')}",
                    "output_preview": _clip_text(event.get("output", ""), limit=320),
                }
            )
        elif event_type == "artifact":
            timeline.append(
                {
                    "index": index,
                    "type": "artifact",
                    "ts": ts,
                    "kind": event.get("kind", "file"),
                    "path": event.get("path", ""),
                    "size_bytes": event.get("size_bytes", 0),
                    "label": event.get("label", "Artifact saved"),
                    "detail": event.get("image_ref", "") or event.get("url", "") or event.get("path", ""),
                }
            )
        elif event_type == "session_end":
            session.update(
                {
                    "ended_at": ts,
                    "duration_s": event.get("duration_s", 0),
                    "timed_out": bool(event.get("timed_out", False)),
                    "score": event.get("score", {}),
                    "preview_linger_seconds": event.get("preview_linger_seconds", 0),
                    "preview_expires_at": event.get("preview_expires_at"),
                }
            )
            timeline.append(
                {
                    "index": index,
                    "type": "session_end",
                    "ts": ts,
                    "label": "Session ended",
                    "score_total": event.get("score", {}).get("total", 0),
                    "timed_out": bool(event.get("timed_out", False)),
                    "preview_linger_seconds": event.get("preview_linger_seconds", 0),
                }
            )

    return {
        "summary": {
            "event_count": len(events),
            "turn_count": turn_count,
            "tool_call_count": tool_call_count,
            "has_preview": bool(preview),
            "session": session,
            "preview": preview,
        },
        "timeline": timeline,
    }


def build_replay_run_record(
    run_id: str,
    events: list[dict[str, Any]],
    log_path: Path,
    db_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    replay = build_run_replay(events)
    summary = replay["summary"]
    session = summary.get("session", {})

    if db_run is not None:
        record = dict(db_run)
        record.setdefault("id", run_id)
        record.setdefault("log_path", str(log_path))
        return record

    score_total = float((session.get("score") or {}).get("total", 0) or 0)
    return {
        "id": run_id,
        "ts": session.get("started_at") or 0,
        "model": session.get("model", ""),
        "base_url": "",
        "harness": "",
        "mode": "unguided",
        "level_id": session.get("level_id", ""),
        "level_name": session.get("level_name", session.get("level_id", "")),
        "difficulty": 0,
        "score_total": score_total,
        "duration_s": session.get("duration_s", 0),
        "turns": summary.get("turn_count", 0),
        "tool_calls_n": summary.get("tool_call_count", 0),
        "timed_out": int(bool(session.get("timed_out", False))),
        "stars": score_to_stars(score_total),
        "log_path": str(log_path),
    }


def build_replay_payload(
    run_id: str,
    events: list[dict[str, Any]],
    log_path: Path,
    runs_dir: Path,
    db_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    replay = build_run_replay(events)
    run = build_replay_run_record(run_id, events, log_path, db_run=db_run)
    return {
        "run": run,
        "events": events,
        "replay": replay,
        "artifacts": build_artifact_records(runs_dir, run_id),
        "log_found": True,
        "log_path": str(log_path),
    }
