from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from framework.config import (
    LevelValidationError,
    load_framework_config,
    load_harness_config,
    load_level_config,
)


def test_load_framework_config_resolves_paths(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            framework:
              version: "0.1.0"
              name: benchb0t
              log_level: INFO
              runs_dir: artifacts/runs
              max_parallel_levels: 2
              preview_linger_seconds: 75
            scoring: {}
            container: {}
            agent: {}
            recorder: {}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    loaded = load_framework_config(cfg_path)

    assert loaded.project_dir == tmp_path
    assert loaded.runs_dir == tmp_path / "artifacts" / "runs"
    assert loaded.db_path == tmp_path / "benchb0t.db"
    assert loaded.config.framework.preview_linger_seconds == 75


def test_load_level_config_rejects_negative_preview_linger(tmp_path: Path) -> None:
    level_path = tmp_path / "l10-preview.yaml"
    level_path.write_text(
        textwrap.dedent(
            """
            level:
              id: l10-preview
              name: Preview Test
              difficulty: 1
              category: web
            container:
              image: python:3.11-slim
              working_dir: /workspace
            task:
              instruction: Start a preview
            tools:
              - bash
            preview:
              port: 3000
              linger_seconds: -5
            evaluation:
              type: script
              efficiency_target: 0
              criteria: []
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(LevelValidationError, match="preview.linger_seconds"):
        load_level_config(level_path)


def test_load_level_config_rejects_url_image_and_prose_checks(tmp_path: Path) -> None:
    level_path = tmp_path / "l10-broken-webapp.yaml"
    level_path.write_text(
        textwrap.dedent(
            """
            level:
              id: l10-broken-webapp
              name: Broken Webapp
              difficulty: 3
              category: webapp
            container:
              image: https://images.example.com/preview.png
              working_dir: /workspace
            task:
              instruction: Build a webapp
            tools:
              - bash
              - run_background
            preview:
              port: 3000
            evaluation:
              type: script
              efficiency_target: 0
              criteria:
                - id: server_responds
                  description: Validate the server
                  type: script
                  check: Check if the site responds on port 3000
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(LevelValidationError, match="container.image must be a Docker image reference"):
        load_level_config(level_path)


def test_load_level_config_rejects_preview_without_run_background(tmp_path: Path) -> None:
    level_path = tmp_path / "l10-preview-missing-tool.yaml"
    level_path.write_text(
        textwrap.dedent(
            """
            level:
              id: l10-preview-missing-tool
              name: Preview Missing Tool
              difficulty: 2
              category: webapp
            container:
              image: node:20-slim
              working_dir: /workspace
            task:
              instruction: Build and serve an app
            tools:
              - bash
              - write_file
            preview:
              port: 3000
            evaluation:
              type: script
              efficiency_target: 0
              criteria: []
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(LevelValidationError, match="preview levels must include run_background"):
        load_level_config(level_path)


def test_repo_sample_configs_validate(repo_root: Path) -> None:
    framework_cfg = load_framework_config(repo_root / "config.yaml")
    level_cfg = load_level_config(repo_root / "levels" / "l99-test.yaml")
    harness_cfg = load_harness_config(repo_root / "harnesses" / "hermes.yaml")

    assert framework_cfg.config.framework.preview_linger_seconds == 60
    assert level_cfg.level.id == "l99-test"
    assert harness_cfg.harness.name == "hermes"
