#!/usr/bin/env python3
"""
scripts/precommit-clean.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-commit guard for benchb0t.

Scans staged changes for personal / environment-specific data that should
never be committed (internal hostnames, home-directory paths, API keys, …)
and blocks the commit if any are found.

The script only inspects staged content via `git show :<path>` so unstaged
files or local working-tree noise cannot accidentally leak in.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


# Regexes matching things that must not be committed.
# Each tuple is (pattern, description, file_whitelist).
PATTERNS: list[tuple[re.Pattern[str], str, list[str]]] = [
    # Internal / personal hostnames (exact svslai/svsl- names)
    (re.compile(r"\bsvslai\d+\b", re.IGNORECASE), "internal hostname", []),
    (re.compile(r"\bsvsl-\w+", re.IGNORECASE), "internal hostname", []),
    # Private IPv4 addresses (skip example/placeholder files)
    (
        re.compile(
            r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
            r"172\.(?:1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3})\b"
        ),
        "private IP address",
        [],
    ),
    # macOS / Linux home directories with a username
    (
        re.compile(r"/Users/[A-Za-z0-9_\-]+(/[A-Za-z0-9_\-\. ]*)*"),
        "macOS home-directory path",
        [],
    ),
    (
        re.compile(r"/home/[A-Za-z][A-Za-z0-9_\-]*(/[A-Za-z0-9_\-\. ]*)+"),
        "Linux home-directory path",
        [],
    ),
    # Plain API keys / secrets in common formats (skip example/placeholder files)
    (
        re.compile(r"\b(sk-[A-Za-z0-9]{20,}|sk-[a-z]-[A-Za-z0-9]{20,})\b"),
        "API secret key",
        [".env.example"],
    ),
    # Real credential assignments: a literal value after =/: that looks like a secret.
    # Skips empty strings, env-var fallbacks, obvious placeholders, variable names,
    # object member access, and || / && fallbacks.
    (
        re.compile(
            r"(?:api[_-]?key|token|secret)\s*[:=]\s*['\"]?"
            r"(?![\s'\"]*(?:None|null|nil|empty|benchbot|your-key|YOUR_|example|placeholder|\$\{|os\.getenv|[A-Za-z_][A-Za-z0-9_\.\[\]?]*(?:\|\||&&)?))"
            r"[^'\"\s]{12,}['\"]?",
            re.IGNORECASE,
        ),
        "credential assignment",
        [".env.example", "scripts/precommit-clean.py"],
    ),
    # Email addresses (not in example docs)
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "email address",
        [".env.example"],
    ),
]


# File paths that are always allowed to contain the patterns above
# because they are examples / documentation / tool config.
PATH_WHITELIST: set[str] = {
    ".env.example",
    ".gitignore",
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
    "ROADMAP.md",
    "pyproject.toml",
    "scripts/precommit-clean.py",
}


def get_staged_paths() -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def read_staged_blob(path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f":{path}"],
        capture_output=True,
        check=True,
    )
    return result.stdout


def is_text(data: bytes) -> bool:
    return b"\x00" not in data


def scan(lines: Iterable[str], path: str) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(lines, start=1):
        for pattern, description, file_whitelist in PATTERNS:
            if path in file_whitelist or Path(path).name in file_whitelist:
                continue
            if pattern.search(line):
                findings.append((lineno, description, line.rstrip()))
                break  # one finding per line is enough
    return findings


def main() -> int:
    staged = get_staged_paths()
    if not staged:
        return 0

    dirty = False
    for path in staged:
        if path in PATH_WHITELIST or Path(path).name in PATH_WHITELIST:
            continue

        raw = read_staged_blob(path)
        if not is_text(raw):
            continue

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")

        findings = scan(text.splitlines(), path)
        if findings:
            dirty = True
            print(f"\n🚫 {path}", file=sys.stderr)
            for lineno, description, line in findings:
                snippet = line if len(line) <= 120 else line[:117] + "..."
                print(f"   line {lineno}: {description}: {snippet}", file=sys.stderr)

    if dirty:
        print(
            "\n❌ Commit blocked: staged changes contain personal / environment-specific data.\n"
            "   Remove the flagged content before committing.\n"
            "   You can bypass this check with git commit --no-verify (not recommended).",
            file=sys.stderr,
        )
        return 1

    print("✅ No personal / environment-specific data detected in staged changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
