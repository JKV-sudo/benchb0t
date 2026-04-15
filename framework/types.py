"""
framework/types.py
~~~~~~~~~~~~~~~~~~
Consolidated type definitions and shared data structures for benchb0t.

This module centralizes all cross-module domain types: tool calls, results,
scores, artifacts, and messages. Using shared types eliminates dict literals,
ensures consistent shapes across the codebase, and makes dependencies explicit.

Type categories
  ChatMessage         — LLM message (user/assistant/system)
  ToolCall & Result   — Agent tool invocations
  ScoreDimension      — Scoring breakdown
  RunResult           — Complete level run result (runner → store → dashboard)
  Artifact            — Optional run artifacts (screenshots, snapshots, bundles)
  AgentLogEvent       — Recorder event types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal



@dataclass
class ToolCallFunction:
    """Identifies a single tool function call within a response."""

    name: str
    arguments: str  # JSON string


@dataclass
class ToolCall:
    """
    One tool invocation from the agent's response.

    Corresponds to OpenAI's tool_calls format.
    """

    id: str
    type: str = "function"
    function: ToolCallFunction | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible dict for message history."""
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name if self.function else "",
                "arguments": self.function.arguments if self.function else "",
            },
        }


@dataclass
class ChatMessage:
    """
    One turn in the agent conversation.

    Used in message history, recorded events, and dashboard streaming.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible dict."""
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return result




@dataclass
class ToolCallRecord:
    """
    Complete record of one tool invocation from recorder.

    Passed to scorer and stored in run results.
    """

    call_id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0
    output: str = ""




@dataclass
class CriterionResult:
    """Result of evaluating a single criterion (from evaluation config)."""

    criterion_id: str
    passed: bool
    weight: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to summary dict."""
        return {
            "id": self.criterion_id,
            "passed": self.passed,
            "weight": self.weight,
            "notes": self.notes,
        }


@dataclass
class ScorePenalties:
    """Penalty deductions from the final score."""

    extra_calls: float = 0.0
    backtracks: float = 0.0
    timeout: float = 0.0
    retry: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to summary dict."""
        return {
            "extra_calls": round(self.extra_calls, 2),
            "backtracks": round(self.backtracks, 2),
            "timeout": round(self.timeout, 2),
            "retry": round(self.retry, 2),
        }


@dataclass
class ScoreDimensions:
    """Normalized scoring dimensions (each 0–1)."""

    completion: float = 0.0
    efficiency: float = 0.0
    self_correction: float = 0.0
    path_quality: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to summary dict."""
        return {
            "completion": round(self.completion * 100, 2),
            "efficiency": round(self.efficiency * 100, 2),
            "self_correction": round(self.self_correction * 100, 2),
            "path_quality": round(self.path_quality * 100, 2),
        }


@dataclass
class ScoreSummary:
    """Final score summary — used by store and dashboard."""

    total: float = 0.0
    dimensions: ScoreDimensions = field(default_factory=ScoreDimensions)
    penalties: ScorePenalties = field(default_factory=ScorePenalties)
    criteria: list[CriterionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested summary dict."""
        return {
            "total": round(self.total, 2),
            "dimensions": self.dimensions.to_dict(),
            "penalties": self.penalties.to_dict(),
            "criteria": [c.to_dict() for c in self.criteria],
        }




@dataclass
class PreviewScreenshot:
    """Captured screenshot of the preview URL."""

    kind: str = "preview_screenshot"
    label: str = "Preview screenshot"
    path: str = ""
    url: str = ""
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to event dict."""
        return {
            "kind": self.kind,
            "label": self.label,
            "path": self.path,
            "url": self.url,
            "size_bytes": self.size_bytes,
        }


@dataclass
class ContainerSnapshot:
    """Committed Docker image of the final container state."""

    kind: str = "container_snapshot"
    label: str = "Container snapshot"
    path: str = ""
    image_ref: str = ""
    level_id: str = ""
    run_id: str = ""
    created_at: float = 0.0
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to event dict."""
        return {
            "kind": self.kind,
            "label": self.label,
            "path": self.path,
            "image_ref": self.image_ref,
            "level_id": self.level_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
        }


@dataclass
class ResultBundle:
    """Portable ZIP bundle of result.json, log, and artifacts."""

    kind: str = "result_bundle"
    label: str = "Result bundle"
    path: str = ""
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to event dict."""
        return {
            "kind": self.kind,
            "label": self.label,
            "path": self.path,
            "size_bytes": self.size_bytes,
        }


# Union for any artifact type
Artifact = PreviewScreenshot | ContainerSnapshot | ResultBundle




@dataclass
class RunResult:
    """
    Complete result of running a single benchmark level.

    Produced by runner.run_level(), stored by Store, and displayed by dashboard.
    This is the primary data structure flowing through the system.
    """

    # Identity
    run_id: str
    level_id: str
    level_name: str = ""
    harness: str = ""
    mode: str = "unguided"  # "guided" | "unguided"

    # Metadata
    ts: float = 0.0  # Unix epoch when run started
    duration_s: float = 0.0
    difficulty: int = 1

    # Execution
    model: str = ""
    base_url: str = ""
    turns: int = 0
    tool_calls_n: int = 0
    timed_out: bool = False

    # Scoring & results
    score: ScoreSummary = field(default_factory=ScoreSummary)
    log_path: str = ""

    # Preview (optional)
    host_preview_port: int | None = None
    preview_linger_seconds: int = 0
    preview_expires_at: float | None = None

    # Artifacts (optional)
    artifacts: list[Artifact] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dict for storage/transmission."""
        return {
            "run_id": self.run_id,
            "ts": self.ts,
            "level_id": self.level_id,
            "level_name": self.level_name,
            "difficulty": self.difficulty,
            "harness": self.harness,
            "mode": self.mode,
            "model": self.model,
            "base_url": self.base_url,
            "log_path": self.log_path,
            "timed_out": self.timed_out,
            "score": self.score.to_dict(),
            "turns": self.turns,
            "tool_calls_n": self.tool_calls_n,
            "duration_s": self.duration_s,
            "host_preview_port": self.host_preview_port,
            "preview_linger_seconds": self.preview_linger_seconds,
            "preview_expires_at": self.preview_expires_at,
            "artifacts": [a.to_dict() if hasattr(a, "to_dict") else a for a in self.artifacts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunResult":
        """Reconstruct from a dict (e.g., from store or JSON)."""
        score_dict = data.get("score", {})
        dimensions = score_dict.get("dimensions", {})
        penalties = score_dict.get("penalties", {})

        return cls(
            run_id=data.get("run_id", ""),
            ts=data.get("ts", 0.0),
            level_id=data.get("level_id", ""),
            level_name=data.get("level_name", ""),
            difficulty=data.get("difficulty", 1),
            harness=data.get("harness", ""),
            mode=data.get("mode", "unguided"),
            model=data.get("model", ""),
            base_url=data.get("base_url", ""),
            log_path=data.get("log_path", ""),
            timed_out=data.get("timed_out", False),
            score=ScoreSummary(
                total=score_dict.get("total", 0.0),
                dimensions=ScoreDimensions(
                    completion=dimensions.get("completion", 0) / 100,
                    efficiency=dimensions.get("efficiency", 0) / 100,
                    self_correction=dimensions.get("self_correction", 0) / 100,
                    path_quality=dimensions.get("path_quality", 0) / 100,
                ),
                penalties=ScorePenalties(
                    extra_calls=penalties.get("extra_calls", 0),
                    backtracks=penalties.get("backtracks", 0),
                    timeout=penalties.get("timeout", 0),
                    retry=penalties.get("retry", 0),
                ),
                criteria=[
                    CriterionResult(
                        criterion_id=c.get("id", ""),
                        passed=c.get("passed", False),
                        weight=c.get("weight", 1.0),
                        notes=c.get("notes", ""),
                    )
                    for c in score_dict.get("criteria", [])
                ],
            ),
            turns=data.get("turns", 0),
            tool_calls_n=data.get("tool_calls_n", 0),
            duration_s=data.get("duration_s", 0.0),
            host_preview_port=data.get("host_preview_port"),
            preview_linger_seconds=data.get("preview_linger_seconds", 0),
            preview_expires_at=data.get("preview_expires_at"),
            artifacts=data.get("artifacts", []),
        )




@dataclass
class SessionStartEvent:
    """Event written when a benchmark session begins."""

    run_id: str
    level_id: str
    harness: str
    type: str = "session_start"
    ts: float = 0.0
    level_name: str = ""
    harness_type: str = ""
    model: str = ""
    base_url: str = ""
    provider_slot: int = 1
    provider_label: str = ""
    panel_id: str = ""


@dataclass
class MessageEvent:
    """Event for a complete chat message."""

    role: Literal["user", "assistant", "system"]
    content: str
    type: str = "message"
    ts: float = 0.0


@dataclass
class MessageDeltaEvent:
    """Event for streaming text deltas."""

    role: str
    delta: str
    type: str = "message_delta"
    ts: float = 0.0


@dataclass
class ToolCallEvent:
    """Event for a tool invocation."""

    call_id: str
    tool: str
    args: dict[str, Any]
    type: str = "tool_call"
    ts: float = 0.0


@dataclass
class ToolResultEvent:
    """Event for tool result."""

    call_id: str
    type: str = "tool_result"
    ts: float = 0.0
    exit_code: int = 0
    output: str = ""


@dataclass
class PreviewReadyEvent:
    """Event written when preview port is ready."""

    host_preview_port: int
    type: str = "preview_ready"
    ts: float = 0.0
    preview_path: str = "/"


@dataclass
class ArtifactEvent:
    """Event for saved artifacts."""

    kind: str
    path: str
    type: str = "artifact"
    ts: float = 0.0
    label: str = ""
    url: str = ""
    image_ref: str = ""
    size_bytes: int = 0


@dataclass
class SessionEndEvent:
    """Event written when the session ends."""

    type: str = "session_end"
    ts: float = 0.0
    score: dict[str, Any] = field(default_factory=dict)
    timed_out: bool = False
    preview_linger_seconds: int = 0
    preview_expires_at: float | None = None


# Union of all event types
AgentLogEvent = (
    SessionStartEvent
    | MessageEvent
    | MessageDeltaEvent
    | ToolCallEvent
    | ToolResultEvent
    | PreviewReadyEvent
    | ArtifactEvent
    | SessionEndEvent
)
