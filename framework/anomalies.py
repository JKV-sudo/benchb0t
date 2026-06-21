"""
framework/anomalies.py
~~~~~~~~~~~~~~~~~~~~~~
Post-run anomaly detection.

Analyzes recorded tool calls, agentlog events, and run metadata to surface
notable events: failed tools, timeouts, forced-retry usage, retry loops,
oversized outputs, self-corrections, and duration spikes.

The result is written as ``anomalies.json`` into the run artifact directory
and is picked up automatically by ``save_result_bundle()`` (it zips every
file in the artifact dir). Optionally enriched with a narrative LLM summary
(best-effort, never fatal to the run).

Report shape
------------
    {
      "summary": {"count": int, "severity": "none|low|medium|high"},
      "items": [ {"kind": str, "severity": str, "detail": str, ...}, ... ],
      "llm_summary": str | null
    }
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any, TYPE_CHECKING

from framework.utils import truncate

if TYPE_CHECKING:
    from framework.api import AgentAPI

logger = logging.getLogger(__name__)

# ── Tunable heuristics ────────────────────────────────────────────────────────
LONG_OUTPUT_THRESHOLD = 4096        # chars — outputs above this are "long"
LOOP_REPEAT_THRESHOLD = 3           # identical consecutive tool calls → loop
DURATION_SPIKE_FACTOR = 4.0         # multiple of median per-tool duration
DURATION_SPIKE_MIN_S = 15.0         # absolute floor below which a spike is ignored
MAX_LLM_INPUT_CHARS = 6000          # cap on the tool-call digest sent to the judge
MAX_ITEMS_REPORTED = 50             # hard cap so the report stays readable


def detect_anomalies(
    *,
    tool_calls: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
    timed_out: bool = False,
    score: dict[str, Any] | None = None,
    duration_s: float = 0.0,
) -> dict[str, Any]:
    """
    Build an anomaly report from one run's recorded activity.

    Parameters mirror what ``framework/runner.py`` already has in scope after
    a run finishes. Pure function — no I/O, no network.
    """
    score = score or {}
    items: list[dict[str, Any]] = []

    durations = _tool_durations(events or [])
    median_duration = statistics.median(durations.values()) if durations else 0.0

    # ── Failed tool calls (exit_code != 0) ────────────────────────────────────
    failed_tools = set()
    for call in tool_calls:
        if call.get("exit_code", 0) != 0:
            items.append({
                "kind": "failed_tool",
                "severity": "high",
                "tool": call.get("tool", ""),
                "call_id": call.get("call_id", ""),
                "detail": f"exit {call.get('exit_code')} — {truncate(str(call.get('output', '')), 160)}",
            })
            failed_tools.add(call.get("tool", ""))

    # ── Timeout ───────────────────────────────────────────────────────────────
    if timed_out:
        items.append({
            "kind": "timeout",
            "severity": "high",
            "detail": f"agent loop timed out after {duration_s:.1f}s",
        })

    # ── Forced-retry usage ────────────────────────────────────────────────────
    retry_penalty = float(score.get("penalties", {}).get("retry", 0) or 0)
    if retry_penalty > 0:
        items.append({
            "kind": "forced_retry",
            "severity": "medium",
            "detail": f"forced retry applied (penalty −{retry_penalty:.1f} pts)",
        })

    # ── Retry loops: identical tool+args repeated N times in a row ────────────
    items.extend(_detect_loops(tool_calls))

    # ── Self-corrections: write_file then patch_file on the same path ────────
    items.extend(_detect_self_corrections(tool_calls))

    # ── Long outputs (pre-truncation length, only available on tool_calls) ───
    for call in tool_calls:
        out_len = len(str(call.get("output", "")))
        if out_len > LONG_OUTPUT_THRESHOLD:
            items.append({
                "kind": "long_output",
                "severity": "low",
                "tool": call.get("tool", ""),
                "call_id": call.get("call_id", ""),
                "detail": f"output {out_len} chars (>{LONG_OUTPUT_THRESHOLD})",
            })

    # ── Duration spikes: one tool call far slower than the median ────────────
    for cid, dur in durations.items():
        if (
            dur >= DURATION_SPIKE_MIN_S
            and median_duration > 0
            and dur >= median_duration * DURATION_SPIKE_FACTOR
        ):
            tool_name = _tool_for_call_id(tool_calls, cid)
            items.append({
                "kind": "duration_spike",
                "severity": "low",
                "tool": tool_name,
                "call_id": cid,
                "detail": f"{dur:.1f}s vs {median_duration:.1f}s median",
            })

    # Cap the number of reported items to keep the report readable.
    if len(items) > MAX_ITEMS_REPORTED:
        items = items[:MAX_ITEMS_REPORTED]

    severity = _overall_severity(items)
    return {
        "summary": {"count": len(items), "severity": severity},
        "items": items,
        "llm_summary": None,
    }


def write_anomalies_report(
    *,
    report: dict[str, Any],
    dest_path: Path,
) -> dict[str, Any]:
    """
    Write an anomaly report as JSON to ``dest_path``.

    Returns a metadata dict suitable for ``recorder.record_artifact()`` and the
    result.artifacts list. Never raises — disk errors are logged and a minimal
    fallback is returned.
    """
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Anomaly report write failed for %s: %s", dest_path, exc)
        return {
            "kind": "anomalies",
            "label": "Anomaly report",
            "path": str(dest_path),
            "size_bytes": 0,
            "error": str(exc),
        }

    logger.info(
        "Anomaly report saved: %s (%d items, severity=%s)",
        dest_path,
        report.get("summary", {}).get("count", 0),
        report.get("summary", {}).get("severity", "none"),
    )
    return {
        "kind": "anomalies",
        "label": "Anomaly report",
        "path": str(dest_path),
        "size_bytes": dest_path.stat().st_size,
    }


def llm_anomaly_summary(
    api: "AgentAPI",
    *,
    report: dict[str, Any],
    result: dict[str, Any],
) -> str | None:
    """
    Ask the agent model for a short narrative summary of the run.

    Best-effort: any API failure is logged and ``None`` is returned so the run
    is never affected. The returned text is folded into the report under
    ``llm_summary``.
    """
    prompt = _build_judge_prompt(report, result)
    try:
        response = api.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise benchmark analyst. Summarize what "
                        "happened in this agent run and call out notable "
                        "behaviour. Max 4 sentences, plain prose, no lists."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        text = _extract_assistant_text(response)
        if text:
            logger.info("LLM anomaly summary generated (%d chars)", len(text))
        return text or None
    except Exception as exc:  # noqa: BLE001 — best-effort judge
        logger.warning("LLM anomaly summary failed: %s", exc)
        return None


# ── Internal helpers ──────────────────────────────────────────────────────────


def _tool_durations(events: list[dict[str, Any]]) -> dict[str, float]:
    """Pair tool_call/tool_result events by call_id to compute durations (s)."""
    starts: dict[str, float] = {}
    durations: dict[str, float] = {}
    for ev in events:
        cid = ev.get("call_id")
        if not cid:
            continue
        ev_type = ev.get("type")
        if ev_type == "tool_call":
            starts[cid] = float(ev.get("ts", 0.0))
        elif ev_type == "tool_result":
            start = starts.get(cid)
            if start is not None:
                durations[cid] = max(0.0, float(ev.get("ts", start)) - start)
    return durations


def _detect_loops(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag runs of identical tool + args repeated back-to-back."""
    items: list[dict[str, Any]] = []
    run_start = 0
    for idx in range(1, len(tool_calls) + 1):
        prev = tool_calls[idx - 1] if idx > 0 else None
        cur = tool_calls[idx] if idx < len(tool_calls) else None
        sig_prev = _call_signature(prev) if prev else None
        sig_cur = _call_signature(cur) if cur else None
        if cur is not None and sig_prev == sig_cur:
            continue
        # End of a run (or end of list): check its length
        run_len = idx - run_start
        if run_len >= LOOP_REPEAT_THRESHOLD and prev is not None:
            items.append({
                "kind": "retry_loop",
                "severity": "medium",
                "tool": prev.get("tool", ""),
                "detail": f"{run_len} identical '{prev.get('tool', '')}' calls in a row",
            })
        run_start = idx
    return items


def _detect_self_corrections(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag write_file → patch_file (or vice versa) on the same path."""
    items: list[dict[str, Any]] = []
    seen_writes: dict[str, str] = {}  # path → tool that wrote it
    for call in tool_calls:
        tool = call.get("tool", "")
        args = call.get("args", {}) or {}
        path = args.get("path") or args.get("file") or ""
        if not path:
            continue
        if tool in {"write_file", "patch_file"}:
            prior = seen_writes.get(path)
            if prior and prior != tool:
                items.append({
                    "kind": "self_correction",
                    "severity": "low",
                    "tool": tool,
                    "detail": f"{prior} → {tool} on {path}",
                })
            seen_writes[path] = tool
    return items


def _call_signature(call: dict[str, Any] | None) -> tuple[str, str] | None:
    if call is None:
        return None
    return (call.get("tool", ""), str(call.get("args", "")))


def _tool_for_call_id(tool_calls: list[dict[str, Any]], call_id: str) -> str:
    for call in tool_calls:
        if call.get("call_id") == call_id:
            return call.get("tool", "")
    return ""


def _overall_severity(items: list[dict[str, Any]]) -> str:
    """Roll up per-item severities into one overall bucket."""
    if not items:
        return "none"
    ranks = {"high": 3, "medium": 2, "low": 1}
    worst = max(ranks.get(it.get("severity", "low"), 1) for it in items)
    high_n = sum(1 for it in items if it.get("severity") == "high")
    # Two or more high-severity items → high; otherwise worst wins but
    # a single low → low, single medium → medium.
    if high_n >= 2 or worst >= 3:
        return "high"
    if worst >= 2:
        return "medium"
    return "low"


def _build_judge_prompt(report: dict[str, Any], result: dict[str, Any]) -> str:
    """Compact, token-friendly digest of the run for the LLM judge."""
    score = result.get("score", {})
    digest_lines = [
        f"Level: {result.get('level_id', '?')}",
        f"Model: {result.get('model', '?')}",
        f"Duration: {result.get('duration_s', 0):.1f}s",
        f"Turns: {result.get('turns', 0)}",
        f"Tool calls: {result.get('tool_calls_n', 0)}",
        f"Timed out: {result.get('timed_out', False)}",
        f"Score: {score.get('total', 0):.1f}/100",
        f"Anomaly count: {report.get('summary', {}).get('count', 0)} "
        f"({report.get('summary', {}).get('severity', 'none')})",
        "Anomalies:",
    ]
    for item in report.get("items", [])[:12]:
        digest_lines.append(
            f"  - [{item.get('severity', '?')}] {item.get('kind', '?')}: "
            f"{item.get('detail', '')}"
        )
    digest = "\n".join(digest_lines)
    if len(digest) > MAX_LLM_INPUT_CHARS:
        digest = digest[:MAX_LLM_INPUT_CHARS] + "\n…(truncated)"
    return digest


def _extract_assistant_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {}) or {}
    return (message.get("content") or "").strip()
