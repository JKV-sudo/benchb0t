from __future__ import annotations

import argparse
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
