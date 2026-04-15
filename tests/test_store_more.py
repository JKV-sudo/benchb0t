from __future__ import annotations

from pathlib import Path

from framework.store import Store


def _record(
    store: Store,
    *,
    run_id: str,
    model: str,
    level_id: str,
    mode: str,
    total: float,
    stars: int | None = None,
    timed_out: bool = False,
) -> None:
    store.record_run(
        {
            "run_id": run_id,
            "ts": 1_700_000_000.0 + len(run_id),
            "model": model,
            "base_url": "http://localhost:11434/v1",
            "harness": "hermes",
            "mode": mode,
            "level_id": level_id,
            "level_name": level_id,
            "difficulty": 1,
            "score": {
                "total": total,
                "dimensions": {
                    "completion": total * 0.5,
                    "efficiency": total * 0.2,
                    "self_correction": total * 0.2,
                    "path_quality": total * 0.1,
                },
                "penalties": {"extra_calls": 0, "backtracks": 0, "timeout": 5 if timed_out else 0, "retry": 0},
                "criteria": [{"id": "smoke", "passed": total >= 80}],
            },
            "duration_s": 12.0,
            "turns": 4,
            "tool_calls_n": 7,
            "timed_out": timed_out,
        }
    )


def test_store_filters_and_detail_queries(tmp_path: Path) -> None:
    store = Store(tmp_path / "benchb0t.db").init()
    _record(store, run_id="aaa11111", model="hermes", level_id="l1", mode="guided", total=88.0)
    _record(store, run_id="bbb22222", model="hermes", level_id="l1", mode="unguided", total=61.0, timed_out=True)
    _record(store, run_id="ccc33333", model="gpt-4.1", level_id="l2", mode="guided", total=95.0)

    summary = store.get_summary()
    assert summary["total_runs"] == 3
    assert summary["total_models"] == 2

    runs = store.get_runs(model="hermes", timed_out=False)
    assert len(runs) == 1
    assert runs[0]["model"] == "hermes"

    assert store.get_run_count(model="hermes") == 2
    assert store.get_distinct_models() == ["gpt-4.1", "hermes"]
    assert [level["level_id"] for level in store.get_distinct_levels()] == ["l1", "l2"]

    detail = store.get_model_detail("hermes")
    assert detail["overall"]["run_count"] == 2
    assert detail["per_level"][0]["level_id"] == "l1"

    comparison = store.get_mode_comparison()
    assert {(row["model"], row["mode"]) for row in comparison} >= {
        ("hermes", "guided"),
        ("hermes", "unguided"),
    }

