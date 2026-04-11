# Repository Guidelines

## Project Structure & Module Organization
`framework/` contains the runtime: `runner.py` loads YAML, starts Docker containers, drives the agent loop, and scores runs; `container.py` and `api.py` isolate Docker and OpenAI-compatible API access; `store.py` persists results to `benchb0t.db`; `dashboard.py` serves the FastAPI UI and embedded frontend. Add new benchmark content in `levels/*.yaml` and model profiles in `harnesses/*.yaml`. Treat `runs/`, `benchb0t.db`, and `benchb0t.db-*` as generated artifacts, not source files.

## Build, Test, and Development Commands
This repo uses direct Python module entrypoints rather than a package script or `Makefile`.

- `python -m framework.runner --level levels/l1-single-file.yaml --harness harnesses/slavko.yaml` runs one level.
- `python -m framework.runner --level levels/l99-test.yaml --harness harnesses/hermes.yaml --no-prompt` is the fastest end-to-end smoke test.
- `python -m framework.runner --all-levels --harness harnesses/hermes.yaml --no-prompt` executes the full benchmark set.
- `python -m framework.dashboard --host 0.0.0.0 --port 7860` launches the live dashboard backed by `benchb0t.db`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, module docstrings, `from __future__ import annotations`, and type hints on public APIs. Use `snake_case` for functions and helpers, `PascalCase` for classes such as `AgentAPI`, `LevelContainer`, and `Store`. YAML files stay lowercase and descriptive: `levels/l<number>-slug.yaml`, `harnesses/<name>.yaml`. If you add a tool, update both `TOOL_SCHEMAS` and `dispatch_tool()` in `framework/runner.py`.

## Testing Guidelines
No `tests/` package or pytest configuration is present in this snapshot, so validation is runtime-based. Run `l99-test` before broader changes, then rerun a representative level for the area you touched; for example, use `l4-express-api.yaml` for preview/server work. When changing persistence or UI code, start the dashboard and confirm the new run appears in the SQLite-backed views.

## Security & Configuration Tips
Keep secrets in `.env` or environment variables only; `.env` and `.benchb0t_creds.json` are already ignored. Do not commit `runs/*.agentlog`, `__pycache__/`, or database artifacts. Keep level setup scripts deterministic so Docker-based runs remain reproducible.

## Commit & Pull Request Guidelines
This checkout does not include `.git`, so repository-specific commit patterns and PR templates could not be verified from history. Until that metadata is available, keep commit subjects imperative and scoped to the touched area, and include the exact runner or dashboard command you used to validate the change.
