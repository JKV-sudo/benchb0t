#!/bin/sh
# Git pre-commit hook for benchb0t.
# Installed automatically by scripts/install-precommit.py; do not edit by hand.

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)/scripts"
python3 "${SCRIPT_DIR}/precommit-clean.py"
