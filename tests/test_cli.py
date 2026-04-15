from __future__ import annotations

import argparse
import json
from pathlib import Path

from framework import cli


def test_validate_command_passes_for_repo(monkeypatch, capsys, repo_root: Path) -> None:
    monkeypatch.chdir(repo_root)

    rc = cli._cmd_validate(argparse.Namespace(paths=[]))
    captured = capsys.readouterr()

    assert rc == 0
    assert "Validation passed:" in captured.out
    assert "deprecated level" in captured.out


def test_list_hides_deprecated_levels(monkeypatch, capsys, repo_root: Path) -> None:
    monkeypatch.chdir(repo_root)

    rc = cli._cmd_list(argparse.Namespace(include_deprecated=False))
    captured = capsys.readouterr()

    assert rc == 0
    assert "l9-tattoo-studio" in captured.out
    assert "l-tatto-studio" not in captured.out
    assert "deprecated level hidden" in captured.out


def test_list_can_include_deprecated_levels(monkeypatch, capsys, repo_root: Path) -> None:
    monkeypatch.chdir(repo_root)

    rc = cli._cmd_list(argparse.Namespace(include_deprecated=True))
    captured = capsys.readouterr()

    assert rc == 0
    assert "STATE" in captured.out
    assert "l-tatto-studio" in captured.out
    assert "deprecated" in captured.out


def test_iter_validate_files_detects_levels_and_harnesses(tmp_path: Path) -> None:
    levels_dir = tmp_path / "levels"
    harnesses_dir = tmp_path / "harnesses"
    levels_dir.mkdir()
    harnesses_dir.mkdir()
    level_file = levels_dir / "l99-test.yaml"
    harness_file = harnesses_dir / "hermes.yaml"
    level_file.write_text("level: {}\n", encoding="utf-8")
    harness_file.write_text("harness: {}\n", encoding="utf-8")

    assert cli._iter_validate_files(levels_dir) == [("level", level_file)]
    assert cli._iter_validate_files(harnesses_dir) == [("harness", harness_file)]


def test_export_writes_json_and_csv(tmp_path: Path, capsys) -> None:
    from framework.store import Store

    db_path = tmp_path / "benchb0t.db"
    store = Store(db_path).init()
    store.record_run(
        {
            "run_id": "abc12345",
            "ts": 1_700_000_000.0,
            "model": "hermes",
            "base_url": "http://localhost:11434/v1",
            "harness": "hermes",
            "mode": "guided",
            "level_id": "l99-test",
            "level_name": "Smoke Test",
            "difficulty": 1,
            "score": {"total": 88.0, "dimensions": {}, "penalties": {}, "criteria": []},
            "duration_s": 9.0,
            "turns": 3,
            "tool_calls_n": 4,
            "timed_out": False,
        }
    )

    json_out = tmp_path / "runs"
    csv_out = tmp_path / "runs_csv"

    assert cli._cmd_export(argparse.Namespace(db=str(db_path), output=str(json_out), format="json")) == 0
    json_payload = json.loads(json_out.with_suffix(".json").read_text(encoding="utf-8"))
    assert json_payload[0]["id"] == "abc12345"

    assert cli._cmd_export(argparse.Namespace(db=str(db_path), output=str(csv_out), format="csv")) == 0
    csv_text = csv_out.with_suffix(".csv").read_text(encoding="utf-8")
    assert "abc12345" in csv_text


def test_build_parser_accepts_new_run_flags() -> None:
    parser = cli._build_parser()

    args = parser.parse_args(
        [
            "run",
            "--level",
            "levels/l99-test.yaml",
            "--harness",
            "harnesses/hermes.yaml",
            "--capture-preview-screenshot",
            "--save-result-bundle",
            "--save-container-snapshot",
        ]
    )

    assert args.capture_preview_screenshot is True
    assert args.save_result_bundle is True
    assert args.save_container_snapshot is True
