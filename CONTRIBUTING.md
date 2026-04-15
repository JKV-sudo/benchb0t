# Contributing to benchb0t

Thanks for your interest. This document covers the three most common contribution types: levels, tools, and framework code.

## Quick start

```bash
git clone https://github.com/benchb0t/benchb0t
cd benchb0t
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your endpoint + key
benchbot dash          # open http://localhost:7860
```

## Adding a level

Levels live in `levels/*.yaml`. The schema is documented in [README.md](README.md).

1. Copy an existing level as a starting point, e.g. `cp levels/l1-single-file.yaml levels/l-my-level.yaml`
2. Edit the YAML — at minimum: `level.id`, `level.name`, `container.image`, `task.instruction`, `evaluation.criteria`
3. Validate it: `benchbot validate levels/l-my-level.yaml`
4. Test it: `benchbot run --level levels/l-my-level.yaml`
5. Open a PR. Include the output of a passing run in the PR description.

Good levels have:
- A clear, unambiguous instruction
- At least two shell-checkable criteria (`type: script`)
- An `efficiency_target` that reflects a competent solution
- A unique `level.id` that starts with `l-`

## Adding a tool

Tools are the functions the agent can call inside the container. They live in `framework/runner.py`.

1. Add your tool's JSON schema to `TOOL_SCHEMAS` in `runner.py`
2. Add a handler branch in `dispatch_tool()` that calls `container.exec()` or another container method
3. Return a `(exit_code: int, output: str)` tuple — exit 0 for success, non-zero for failure
4. Add at least one test in `tests/test_runner.py`

## Framework code

- Run `ruff check framework/ tests/` before committing
- Run `mypy framework/` — no new `Any` without a comment explaining why
- Run `pytest` — all tests must pass
- Keep modules focused: if a file grows past ~300 lines, consider splitting it
- The module layering is Foundation → Services → Orchestration → UI → Entrypoint — don't import upward

## Reporting bugs

Use the GitHub issue tracker. Include:
- Your Python version and OS
- The level YAML if the bug is level-specific
- The relevant lines from the agentlog (in `runs/`)
- Steps to reproduce

## License

By contributing you agree that your contribution is released under the [MIT License](LICENSE).
