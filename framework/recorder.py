"""
framework/recorder.py
~~~~~~~~~~~~~~~~~~~~~
Session recorder — captures every tool call, message, and event as an
.agentlog file (newline-delimited JSON).

.agentlog format
────────────────
One JSON object per line, each with a mandatory ``type`` field:

  {"type": "session_start",  "ts": ..., "level_id": ..., "harness": ...}
  {"type": "message_delta",  "ts": ..., "role": "assistant", "delta": ...}
  {"type": "message",        "ts": ..., "role": "user"|"assistant", "content": ...}
  {"type": "tool_call",      "ts": ..., "tool": ..., "args": ..., "call_id": ...}
  {"type": "tool_result",    "ts": ..., "call_id": ..., "exit_code": ..., "output": ...}
  {"type": "session_end",    "ts": ..., "score": ..., "timed_out": bool}

All timestamps are Unix epoch floats.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Recorder:
    """
    Writes an .agentlog file for one benchmark session.

    Parameters
    ----------
    runs_dir : str | Path
        Directory where run logs are stored (from config.yaml → framework.runs_dir).
    level_id : str
        Level identifier (e.g. "l1-single-file").
    harness_name : str
        Harness name (e.g. "slavko").
    compress : bool
        When True, gzip the .agentlog file on close.
    """

    def __init__(
        self,
        runs_dir: str | Path,
        level_id: str,
        harness_name: str,
        compress: bool = False,
    ) -> None:
        self.level_id     = level_id
        self.harness_name = harness_name
        self.compress     = compress

        self._run_id     = uuid.uuid4().hex[:8]
        self._started    = time.time()
        self._tool_calls: list[dict[str, Any]] = []
        self._turn_count = 0   # incremented on each assistant message
        self._event_context: dict[str, Any] = {
            "run_id": self._run_id,
            "level_id": self.level_id,
            "harness": self.harness_name,
        }

        # Build output path
        runs_path = Path(runs_dir)
        runs_path.mkdir(parents=True, exist_ok=True)
        ts_label   = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._started))
        filename   = f"{ts_label}_{level_id}_{harness_name}_{self._run_id}.agentlog"
        self._path = runs_path / filename
        # Line-buffered writes keep the live dashboard in sync with the runner.
        self._file = self._path.open("w", encoding="utf-8", buffering=1)

        logger.info("Recording session → %s", self._path)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def path(self) -> Path:
        return self._path

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Immutable view of recorded tool call records."""
        return list(self._tool_calls)

    def start(
        self,
        level_cfg: dict[str, Any],
        harness_cfg: dict[str, Any],
        **metadata: Any,
    ) -> None:
        """Write the session_start event."""
        self._event_context.update({
            "level_name": level_cfg.get("level", {}).get("name", self.level_id),
            "harness_type": harness_cfg.get("harness", {}).get("type", "unknown"),
            **metadata,
        })
        self._write(
            type="session_start",
        )

    @property
    def turn_count(self) -> int:
        """Number of assistant turns recorded so far."""
        return self._turn_count

    def record_message(self, role: str, content: str | list[Any]) -> None:
        """Record a chat message (user or assistant turn)."""
        if role == "assistant":
            self._turn_count += 1
        self._write(
            type="message",
            role=role,
            content=content if isinstance(content, str) else json.dumps(content),
        )

    def record_message_delta(self, role: str, delta: str) -> None:
        """Record an incremental assistant text chunk for live streaming."""
        if not delta:
            return
        self._write(
            type="message_delta",
            role=role,
            delta=delta,
        )

    def record_tool_call(
        self,
        tool: str,
        args: dict[str, Any],
        call_id: str | None = None,
    ) -> str:
        """
        Record a tool invocation. Returns the call_id so the caller can pair
        it with the corresponding result.
        """
        cid = call_id or uuid.uuid4().hex[:12]
        self._write(type="tool_call", tool=tool, args=args, call_id=cid)
        logger.debug("tool_call [%s] %s(%s)", cid[:6], tool, _truncate(args))
        return cid

    def record_tool_result(
        self,
        call_id: str,
        output: str,
        exit_code: int = 0,
        tool: str = "",
        args: dict[str, Any] | None = None,
    ) -> None:
        """Record the output of a tool call and store it for the scorer."""
        self._write(
            type="tool_result",
            call_id=call_id,
            exit_code=exit_code,
            output=output[:4096],  # cap stored output to keep logs manageable
        )
        record = {
            "call_id":   call_id,
            "tool":      tool,
            "args":      args or {},
            "exit_code": exit_code,
            "output":    output,
        }
        self._tool_calls.append(record)
        logger.debug(
            "tool_result [%s] exit=%d output=%s",
            call_id[:6], exit_code, _truncate(output),
        )

    def record_preview_ready(self, host_port: int, path: str = "/") -> None:
        """
        Write a preview_ready event once Docker has assigned a host port.
        Called after container.start() so the real ephemeral port is known.
        The dashboard uses this to point the preview iframe at the right port.
        """
        self._write(type="preview_ready", host_preview_port=host_port, preview_path=path)
        logger.info("Preview ready on host port %d (path=%s)", host_port, path)

    def end(
        self,
        score: dict[str, Any] | None = None,
        timed_out: bool = False,
        **metadata: Any,
    ) -> None:
        """Write the session_end event, flush, and (optionally) compress."""
        duration = round(time.time() - self._started, 2)
        self._write(
            type="session_end",
            run_id=self._run_id,
            level_id=self.level_id,
            duration_s=duration,
            timed_out=timed_out,
            score=score or {},
            **metadata,
        )
        self._file.flush()
        self._file.close()
        logger.info(
            "Session %s ended — duration=%.1fs timed_out=%s score=%s",
            self._run_id, duration, timed_out, score,
        )

        if self.compress:
            self._gzip_log()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _write(self, **kwargs: Any) -> None:
        payload = {**self._event_context, **kwargs}
        payload.setdefault("ts", time.time())
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()

    def _gzip_log(self) -> None:
        gz_path = self._path.with_suffix(".agentlog.gz")
        with self._path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
            dst.write(src.read())
        self._path.unlink()
        logger.debug("Compressed log → %s", gz_path)


# ── Replay helper ─────────────────────────────────────────────────────────────

def load_agentlog(path: str | Path) -> list[dict[str, Any]]:
    """
    Load and parse an .agentlog file (plain or gzipped).
    Returns a list of event dicts in the order they were recorded.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    events = []
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[call-overload]
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# ── Utilities ─────────────────────────────────────────────────────────────────

def _truncate(value: Any, limit: int = 80) -> str:
    """Return a short string representation for logging."""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"
