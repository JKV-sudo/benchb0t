"""
framework/cli.py
~~~~~~~~~~~~~~~~
Unified `benchbot` command-line interface.

Usage
─────
  benchbot run   --level levels/l1.yaml --harness harnesses/hermes.yaml
  benchbot run   --all-levels --harness harnesses/hermes.yaml --mode guided
  benchbot dash  [--host 0.0.0.0] [--port 7860]
  benchbot list              # list all available levels
  benchbot validate levels/  # validate level YAML files
  benchbot export            # export run history to CSV / JSON

Each sub-command delegates to the appropriate framework module so the
individual modules (runner.py, dashboard.py) remain independently runnable
via `python -m framework.runner` for backwards compatibility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from framework.config import (
    LEVEL_FILENAME_RE,
    HarnessValidationError,
    LevelValidationError,
    load_harness_config,
    load_level_config,
)


# ── Sub-command handlers ──────────────────────────────────────────────────────

def _cmd_run(args: argparse.Namespace) -> int:
    """Delegate to framework.runner.main() with the parsed args injected."""
    # Build a sys.argv that runner's own argparse understands, then call main().
    import sys as _sys
    from framework.runner import main as runner_main

    argv = ["benchbot-run"]
    if args.level:
        argv += ["--level", str(args.level)]
    if args.all_levels:
        argv.append("--all-levels")
    argv += ["--harness", str(args.harness)]
    if args.config:
        argv += ["--config", str(args.config)]
    if args.env:
        argv += ["--env", str(args.env)]
    if args.mode:
        argv += ["--mode", args.mode]
    if args.no_prompt:
        argv.append("--no-prompt")

    _sys.argv = argv
    runner_main()
    return 0


def _cmd_dash(args: argparse.Namespace) -> int:
    """Start the live dashboard."""
    from framework.dashboard import main as dash_main
    import sys as _sys

    argv = ["benchbot-dash", "--host", args.host, "--port", str(args.port)]
    if args.config:
        argv += ["--config", str(args.config)]
    _sys.argv = argv
    dash_main()
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """Print all available levels."""
    levels_dir = Path("levels")
    if not levels_dir.exists():
        print("No levels/ directory found. Run from the benchb0t project root.", file=sys.stderr)
        return 1

    rows = []
    hidden_deprecated = 0
    for f in sorted(levels_dir.glob("*.yaml")):
        try:
            level = load_level_config(f)
            if level.is_deprecated and not args.include_deprecated:
                hidden_deprecated += 1
                continue
            row = [
                level.level.id,
                "★" * level.level.difficulty,
                level.level.category,
                level.level.name,
                "✓" if level.modes else "—",
                str(level.preview.port if level.preview else "—"),
            ]
            if args.include_deprecated:
                row.append("deprecated" if level.is_deprecated else "active")
            rows.append(tuple(row))
        except Exception:
            row = [f.stem, "?", "?", "?", "—", "—"]
            if args.include_deprecated:
                row.append("invalid")
            rows.append(tuple(row))

    hdr = ("ID", "DIFF", "CATEGORY", "NAME", "MODES", "PORT")
    if args.include_deprecated:
        hdr += ("STATE",)
    widths = [max(len(r[i]) for r in [hdr] + rows) for i in range(len(hdr))]
    sep = "  ".join("─" * w for w in widths)

    print()
    print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(hdr)))
    print("  " + sep)
    for row in rows:
        print("  " + "  ".join(v.ljust(widths[i]) for i, v in enumerate(row)))
    print()
    print(f"  {len(rows)} level{'s' if len(rows) != 1 else ''} shown from levels/")
    if hidden_deprecated:
        print(f"  {hidden_deprecated} deprecated level hidden; use --include-deprecated to show it")
    print()
    return 0


def _iter_validate_files(target: Path) -> list[tuple[str, Path]]:
    if target.is_dir():
        if target.name == "levels":
            return [("level", p) for p in sorted(target.glob("*.yaml"))]
        if target.name == "harnesses":
            return [("harness", p) for p in sorted(target.glob("*.yaml"))]
        raise ValueError(f"Unsupported directory for validation: {target}")

    if target.is_file():
        if "levels" in target.parts:
            return [("level", target)]
        if "harnesses" in target.parts:
            return [("harness", target)]

        try:
            load_level_config(target)
            return [("level", target)]
        except (LevelValidationError, FileNotFoundError):
            pass
        try:
            load_harness_config(target)
            return [("harness", target)]
        except (HarnessValidationError, FileNotFoundError):
            pass
        raise ValueError(f"Could not determine validation type for {target}")

    raise FileNotFoundError(f"Path not found: {target}")


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate level and harness YAML files."""
    targets = [Path(p) for p in args.paths] if args.paths else [Path("levels"), Path("harnesses")]
    errors: list[str] = []
    warnings: list[str] = []
    ok_count = 0
    active_level_ids: dict[str, Path] = {}

    for target in targets:
        try:
            items = _iter_validate_files(target)
        except (ValueError, FileNotFoundError) as exc:
            errors.append(str(exc))
            continue

        for kind, path in items:
            try:
                if kind == "level":
                    level = load_level_config(path)
                    ok_count += 1

                    if not level.is_deprecated and not LEVEL_FILENAME_RE.match(path.name):
                        errors.append(
                            f"{path}: non-deprecated levels must match l{{nn}}-slug.yaml"
                        )
                    if level.is_deprecated:
                        replacement = (
                            f" -> use {level.level.replaced_by}"
                            if level.level.replaced_by
                            else ""
                        )
                        warnings.append(f"{path}: deprecated level{replacement}")
                    else:
                        seen = active_level_ids.get(level.level.id)
                        if seen and seen != path:
                            errors.append(
                                f"{path}: duplicate active level id {level.level.id!r} "
                                f"(already defined in {seen})"
                            )
                        active_level_ids[level.level.id] = path
                else:
                    load_harness_config(path)
                    ok_count += 1
            except (LevelValidationError, HarnessValidationError, FileNotFoundError) as exc:
                errors.append(str(exc))

    for warning in warnings:
        print(f"WARN: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"\nValidation failed: {len(errors)} error(s), {len(warnings)} warning(s).", file=sys.stderr)
        return 1

    print(f"Validation passed: {ok_count} file(s), {len(warnings)} warning(s).")
    print()
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Export run history from the SQLite store."""
    import json
    import csv
    from framework.store import Store

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    store = Store(db_path).init()
    runs = store.get_runs(limit=10_000)

    out_path = Path(args.output)
    fmt = args.format

    if fmt == "json":
        out_path = out_path.with_suffix(".json") if out_path.suffix != ".json" else out_path
        out_path.write_text(json.dumps(runs, indent=2))
        print(f"Exported {len(runs)} runs → {out_path}")

    elif fmt == "csv":
        out_path = out_path.with_suffix(".csv") if out_path.suffix != ".csv" else out_path
        if not runs:
            print("No runs to export.")
            return 0
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=runs[0].keys())
            writer.writeheader()
            writer.writerows(runs)
        print(f"Exported {len(runs)} runs → {out_path}")

    return 0


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="benchbot",
        description="benchb0t — LLM agent benchmark framework",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    root.add_argument("--version", action="version", version="benchb0t 0.1.0")
    sub = root.add_subparsers(dest="command", metavar="<command>")

    # ── run ──────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run one or all benchmark levels")
    g = p_run.add_mutually_exclusive_group(required=True)
    g.add_argument("--level",      type=Path, metavar="YAML", help="Path to a single level YAML")
    g.add_argument("--all-levels", action="store_true",       help="Run every level in levels/")
    p_run.add_argument("--harness",   type=Path, required=True, metavar="YAML")
    p_run.add_argument("--config",    type=Path, default=Path("config.yaml"))
    p_run.add_argument("--env",       type=Path, default=Path(".env"))
    p_run.add_argument(
        "--mode", choices=["guided", "unguided"], default="unguided",
        help="Agent system prompt mode (default: unguided)",
    )
    p_run.add_argument("--no-prompt", action="store_true", help="Skip interactive boot screen")

    # ── dash ─────────────────────────────────────────────────────────────────
    p_dash = sub.add_parser("dash", help="Start the live dashboard (http://localhost:7860)")
    p_dash.add_argument("--host",   default="0.0.0.0")
    p_dash.add_argument("--port",   type=int, default=7860)
    p_dash.add_argument("--config", type=Path, default=Path("config.yaml"))

    # ── list ─────────────────────────────────────────────────────────────────
    p_list = sub.add_parser("list", help="List all available levels")
    p_list.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include deprecated levels in the listing",
    )

    # ── validate ─────────────────────────────────────────────────────────────
    p_val = sub.add_parser("validate", help="Validate level and harness YAML files")
    p_val.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="Files or directories to validate (default: levels harnesses)",
    )

    # ── export ───────────────────────────────────────────────────────────────
    p_exp = sub.add_parser("export", help="Export run history to CSV or JSON")
    p_exp.add_argument("--format",  choices=["csv", "json"], default="csv")
    p_exp.add_argument("--output",  default="benchbot_runs", metavar="FILE")
    p_exp.add_argument("--db",      default="benchb0t.db",   metavar="PATH")

    return root


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "run":    _cmd_run,
        "dash":   _cmd_dash,
        "list":   _cmd_list,
        "validate": _cmd_validate,
        "export": _cmd_export,
    }
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
