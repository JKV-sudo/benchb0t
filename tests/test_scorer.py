from __future__ import annotations

import pytest

from framework.scorer import Scorer


TOOL_CALLS = [
    {
        "tool": "bash",
        "args": {"command": "echo TARGET"},
        "output": "TARGET",
        "exit_code": 0,
    },
    {
        "tool": "bash",
        "args": {"command": "echo TARGET"},
        "output": "TARGET again",
        "exit_code": 0,
    },
]


def _scoring_cfg(
    *,
    completion: float = 0.0,
    efficiency: float = 0.0,
    self_correction: float = 0.0,
    path_quality: float = 0.0,
    extra_tool_call: float = 0.0,
    backtrack: float = 0.0,
    timeout: float = 0.0,
) -> dict:
    return {
        "weights": {
            "completion": completion,
            "efficiency": efficiency,
            "self_correction": self_correction,
            "path_quality": path_quality,
        },
        "penalties": {
            "extra_tool_call": extra_tool_call,
            "backtrack": backtrack,
            "timeout": timeout,
        },
    }


def test_scoring_weights_change_total() -> None:
    eval_cfg = {
        "type": "exact_match",
        "efficiency_target": 1,
        "criteria": [
            {
                "id": "contains-target",
                "description": "tool output contains TARGET",
                "check": "TARGET",
                "weight": 1.0,
            }
        ],
    }

    completion_only = Scorer(
        eval_cfg,
        _scoring_cfg(completion=1.0),
    ).score(TOOL_CALLS)
    efficiency_only = Scorer(
        eval_cfg,
        _scoring_cfg(efficiency=1.0),
    ).score(TOOL_CALLS)
    path_quality_only = Scorer(
        eval_cfg,
        _scoring_cfg(path_quality=1.0),
    ).score(TOOL_CALLS)

    assert completion_only.total == pytest.approx(100.0)
    assert efficiency_only.total == pytest.approx(0.0)
    assert path_quality_only.total == pytest.approx(50.0)


def test_penalties_are_subtracted_from_total() -> None:
    eval_cfg = {
        "type": "exact_match",
        "efficiency_target": 1,
        "criteria": [
            {
                "id": "contains-target",
                "description": "tool output contains TARGET",
                "check": "TARGET",
                "weight": 1.0,
            }
        ],
    }

    score = Scorer(
        eval_cfg,
        _scoring_cfg(
            completion=1.0,
            extra_tool_call=2.0,
            backtrack=5.0,
            timeout=7.0,
        ),
    ).score(TOOL_CALLS, timed_out=True)

    assert score.penalty_extra_calls == pytest.approx(2.0)
    assert score.penalty_backtracks == pytest.approx(5.0)
    assert score.penalty_timeout == pytest.approx(7.0)
    assert score.total == pytest.approx(86.0)


def test_criterion_type_overrides_top_level_eval_type() -> None:
    eval_cfg = {
        "type": "script",
        "efficiency_target": 0,
        "criteria": [
            {
                "id": "contains-target",
                "description": "tool output contains TARGET",
                "type": "exact_match",
                "check": "TARGET",
                "weight": 1.0,
            }
        ],
    }

    score = Scorer(
        eval_cfg,
        _scoring_cfg(completion=1.0),
    ).score(TOOL_CALLS[:1])

    assert score.criteria_results[0].passed is True
    assert score.total == pytest.approx(100.0)
