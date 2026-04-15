"""
framework/scorer.py
~~~~~~~~~~~~~~~~~~~
Scoring engine for benchb0t.

Scoring model
─────────────
  Final score  =  Σ(dimension_score × weight)  −  Σ(penalties)
  Normalised to [0, 100].

Dimensions
  completion      40% — did the agent complete the task?
  efficiency      25% — how close to the efficiency_target tool call count?
  self_correction 20% — recovery quality after mistakes
  path_quality    15% — directness / absence of backtracks

Penalties (subtracted before normalisation)
  extra_tool_call  −0.5 per call beyond efficiency_target
  backtrack        −1.0 per detected backtrack step
  timeout          −5.0 flat if the agent timed out
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from framework.types import (
    CriterionResult,
    ScoreDimensions,
    ScorePenalties,
    ScoreSummary,
)

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """
    Mutable working state during scoring. Converted to immutable ScoreSummary.

    Maintains internal weights and intermediate calculations that are not
    exposed in the final result; the final summary is a ScoreSummary.
    """

    completion: float = 0.0  # 0–1
    efficiency: float = 0.0  # 0–1
    self_correction: float = 0.0  # 0–1
    path_quality: float = 0.0  # 0–1

    completion_weight: float = 0.40
    efficiency_weight: float = 0.25
    self_correction_weight: float = 0.20
    path_quality_weight: float = 0.15

    penalty_extra_calls: float = 0.0
    penalty_backtracks: float = 0.0
    penalty_timeout: float = 0.0
    # Accumulated penalty for forced-retry attempts (−penalty_per_retry × retries used)
    penalty_retry: float = 0.0

    criteria_results: list[CriterionResult] = field(default_factory=list)

    @property
    def total(self) -> float:
        """Weighted sum minus penalties, clamped to [0, 100]."""
        raw = (
            self.completion * self.completion_weight * 100.0
            + self.efficiency * self.efficiency_weight * 100.0
            + self.self_correction * self.self_correction_weight * 100.0
            + self.path_quality * self.path_quality_weight * 100.0
            - self.penalty_extra_calls
            - self.penalty_backtracks
            - self.penalty_timeout
            - self.penalty_retry
        )
        return max(0.0, min(100.0, raw))

    def summary(self) -> dict[str, Any]:
        """Convert to dict for compatibility."""
        return {
            "total": round(self.total, 2),
            "dimensions": {
                "completion": round(self.completion * 100, 2),
                "efficiency": round(self.efficiency * 100, 2),
                "self_correction": round(self.self_correction * 100, 2),
                "path_quality": round(self.path_quality * 100, 2),
            },
            "penalties": {
                "extra_calls": round(self.penalty_extra_calls, 2),
                "backtracks": round(self.penalty_backtracks, 2),
                "timeout": round(self.penalty_timeout, 2),
                "retry": round(self.penalty_retry, 2),
            },
            "criteria": [
                {
                    "id": r.criterion_id,
                    "passed": r.passed,
                    "weight": r.weight,
                    "notes": r.notes,
                }
                for r in self.criteria_results
            ],
        }

    def to_score_summary(self) -> ScoreSummary:
        """Convert to immutable ScoreSummary for use in RunResult."""
        return ScoreSummary(
            total=self.total,
            dimensions=ScoreDimensions(
                completion=self.completion,
                efficiency=self.efficiency,
                self_correction=self.self_correction,
                path_quality=self.path_quality,
            ),
            penalties=ScorePenalties(
                extra_calls=self.penalty_extra_calls,
                backtracks=self.penalty_backtracks,
                timeout=self.penalty_timeout,
                retry=self.penalty_retry,
            ),
            criteria=self.criteria_results,
        )


class Scorer:
    """
    Evaluates a completed agent session against a level's evaluation config.

    Parameters
    ----------
    eval_cfg : dict
        The ``evaluation`` block from a level YAML.
    scoring_cfg : dict
        Framework-wide ``scoring`` block from config.yaml (weights + penalties).
    """

    def __init__(self, eval_cfg: dict[str, Any], scoring_cfg: dict[str, Any]) -> None:
        self._eval = eval_cfg
        raw_weights = scoring_cfg.get("weights", {})
        self._weights = {
            "completion": float(raw_weights.get("completion", 0.40)),
            "efficiency": float(raw_weights.get("efficiency", 0.25)),
            "self_correction": float(raw_weights.get("self_correction", 0.20)),
            "path_quality": float(raw_weights.get("path_quality", 0.15)),
        }
        raw_penalties = scoring_cfg.get("penalties", {})
        self._penalties_cfg = {
            "extra_tool_call": abs(float(raw_penalties.get("extra_tool_call", 0.5))),
            "backtrack": abs(float(raw_penalties.get("backtrack", 1.0))),
            "timeout": abs(float(raw_penalties.get("timeout", 5.0))),
        }

    def score(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        timed_out: bool = False,
        container_exec: Callable[[str], tuple[int, str]] | None = None,  # callable(cmd) → (exit_code, output)
        judge_fn: Callable[[str], str] | None = None,        # callable(prompt) → str
    ) -> ScoreBreakdown:
        """
        Compute the full score for a session.

        Parameters
        ----------
        tool_calls : list[dict]
            Ordered list of tool call records from the recorder.
        timed_out : bool
            Whether the session hit the timeout.
        container_exec : callable, optional
            Function to run a shell command in the container for script-based checks.
        judge_fn : callable, optional
            Function to call an LLM judge for llm_judge evaluations.
        """
        bd = ScoreBreakdown(
            completion_weight=self._weights["completion"],
            efficiency_weight=self._weights["efficiency"],
            self_correction_weight=self._weights["self_correction"],
            path_quality_weight=self._weights["path_quality"],
        )

        criteria = self._eval.get("criteria", [])
        eval_type = self._eval.get("type", "script")

        for criterion in criteria:
            result = self._evaluate_criterion(
                criterion,
                eval_type=eval_type,
                container_exec=container_exec,
                judge_fn=judge_fn,
                tool_calls=tool_calls,
            )
            bd.criteria_results.append(result)

        if criteria:
            weighted_pass = sum(r.weight for r in bd.criteria_results if r.passed)
            total_weight  = sum(r.weight for r in bd.criteria_results)
            bd.completion = weighted_pass / total_weight if total_weight > 0 else 0.0
        else:
            bd.completion = 0.0

        efficiency_target = self._eval.get("efficiency_target", len(tool_calls))
        actual_calls = len(tool_calls)
        extra = max(0, actual_calls - efficiency_target)
        penalty_per = self._penalties_cfg.get("extra_tool_call", 0.5)

        if efficiency_target > 0:
            bd.efficiency = max(0.0, 1.0 - (extra / efficiency_target))
        else:
            bd.efficiency = 1.0
        bd.penalty_extra_calls = extra * penalty_per

        bd.self_correction = self._score_self_correction(tool_calls)

        backtrack_count, bd.path_quality = self._score_path_quality(tool_calls)
        bd.penalty_backtracks = backtrack_count * self._penalties_cfg.get("backtrack", 1.0)

        if timed_out:
            bd.penalty_timeout = self._penalties_cfg.get("timeout", 5.0)

        logger.info("Score: %.1f/100 %s", bd.total, bd.summary())
        return bd

    def _evaluate_criterion(
        self,
        criterion: dict[str, Any],
        *,
        eval_type: str,
        container_exec: Callable[[str], tuple[int, str]] | None,
        judge_fn: Callable[[str], str] | None,
        tool_calls: list[dict[str, Any]],
    ) -> CriterionResult:
        cid    = criterion["id"]
        weight = float(criterion.get("weight", 1.0))
        check  = criterion.get("check", "")
        criterion_eval_type = criterion.get("type", eval_type)

        try:
            if criterion_eval_type == "script":
                passed, notes = self._check_script(check, container_exec)
            elif criterion_eval_type == "exact_match":
                passed, notes = self._check_exact_match(check, tool_calls)
            elif criterion_eval_type == "llm_judge":
                passed, notes = self._check_llm_judge(
                    criterion.get("description", check), judge_fn
                )
            else:
                logger.warning(
                    "Unknown eval type '%s' for criterion %s, treating as failed",
                    criterion_eval_type,
                    cid,
                )
                passed, notes = False, f"Unknown eval type: {criterion_eval_type}"
        except Exception as exc:  # noqa: BLE001
            logger.error("Criterion %s evaluation error: %s", cid, exc)
            passed, notes = False, str(exc)

        logger.debug("Criterion %s → %s | %s", cid, "PASS" if passed else "FAIL", notes)
        return CriterionResult(criterion_id=cid, passed=passed, weight=weight, notes=notes)

    @staticmethod
    def _check_script(
        command: str, container_exec: Callable[[str], tuple[int, str]] | None
    ) -> tuple[bool, str]:
        """Run a shell check command; exit code 0 = pass."""
        if container_exec is None:
            return False, "No container_exec provided for script evaluation"
        exit_code, output = container_exec(command)
        return exit_code == 0, output.strip()[:500]

    @staticmethod
    def _check_exact_match(
        expected: str, tool_calls: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        """
        Check whether the expected string appears in any tool call output.
        """
        for call in tool_calls:
            output = str(call.get("output", ""))
            if expected in output:
                return True, f"Found '{expected}' in tool output"
        return False, f"'{expected}' not found in any tool call output"

    @staticmethod
    def _check_llm_judge(description: str, judge_fn: Callable[[str], str] | None) -> tuple[bool, str]:
        """Ask an LLM judge to evaluate the criterion."""
        if judge_fn is None:
            return False, "No judge_fn provided for llm_judge evaluation"
        prompt = (
            f"Evaluate whether the following criterion was met. "
            f"Reply with PASS or FAIL followed by a one-sentence reason.\n\n"
            f"Criterion: {description}"
        )
        verdict = judge_fn(prompt).strip()
        passed = verdict.upper().startswith("PASS")
        return passed, verdict[:300]

    @staticmethod
    def _score_self_correction(tool_calls: list[dict[str, Any]]) -> float:
        """
        Heuristic: if the agent retried a failed tool call and succeeded on a
        subsequent attempt, award full self-correction credit.
        Score = 1.0 if no errors OR all errors were eventually recovered.
        Score = 0.0 if there were unrecovered errors.
        """
        errors = [c for c in tool_calls if c.get("exit_code", 0) != 0]
        if not errors:
            return 1.0

        # Check whether each error tool was called successfully later
        recovered = 0
        for err_call in errors:
            tool_name = err_call.get("tool")
            err_idx   = tool_calls.index(err_call)
            later = tool_calls[err_idx + 1:]
            if any(
                c.get("tool") == tool_name and c.get("exit_code", 0) == 0
                for c in later
            ):
                recovered += 1

        return recovered / len(errors)

    @staticmethod
    def _score_path_quality(
        tool_calls: list[dict[str, Any]]
    ) -> tuple[int, float]:
        """
        Detect backtracks: a tool call that undoes a previous tool call.
        Currently uses a simple heuristic — same tool + same args in reverse direction.

        Returns (backtrack_count, quality_score_0_to_1).
        """
        if len(tool_calls) <= 1:
            return 0, 1.0

        backtracks = 0
        seen: list[tuple[str, str]] = []
        for call in tool_calls:
            key = (call.get("tool", ""), str(call.get("args", "")))
            if key in seen:
                backtracks += 1
            else:
                seen.append(key)

        quality = max(0.0, 1.0 - backtracks / len(tool_calls))
        return backtracks, quality
