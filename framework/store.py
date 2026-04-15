"""
framework/store.py
~~~~~~~~~~~~~~~~~~
SQLite-backed persistence for benchmark run results.
One row per level-run; queried by the analytics dashboard.

Thread-safety model
───────────────────
A single sqlite3.Connection is shared across all threads in a process
(check_same_thread=False).  Every public method acquires self._lock so
concurrent FastAPI handlers never touch the connection simultaneously.

Cross-process safety
────────────────────
The runner is a subprocess that writes while the dashboard reads.
WAL journal mode handles this: readers never block writers and vice-versa.
If WAL is unavailable (e.g. NFS/FUSE mounts in CI), we fall back to
MEMORY journal mode which is safe for a single writer.

Schema is intentionally flat — no joins — so queries stay readable for
first-time contributors.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Strip SQL comments from schema — older SQLite/Python combos can
# mishandle inline '--' comments inside executescript().
# Base schema — only columns that existed from the beginning.
# New columns are added via _MIGRATIONS below so existing databases upgrade cleanly.
_SCHEMA_BASE = """
CREATE TABLE IF NOT EXISTS runs (
    id                    TEXT    PRIMARY KEY,
    ts                    REAL    NOT NULL,
    model                 TEXT    NOT NULL DEFAULT '',
    base_url              TEXT    NOT NULL DEFAULT '',
    harness               TEXT    NOT NULL DEFAULT '',
    level_id              TEXT    NOT NULL DEFAULT '',
    level_name            TEXT    NOT NULL DEFAULT '',
    difficulty            INTEGER NOT NULL DEFAULT 1,
    score_total           REAL    NOT NULL DEFAULT 0,
    score_completion      REAL    NOT NULL DEFAULT 0,
    score_efficiency      REAL    NOT NULL DEFAULT 0,
    score_self_correction REAL    NOT NULL DEFAULT 0,
    score_path_quality    REAL    NOT NULL DEFAULT 0,
    penalty_extra_calls   REAL    NOT NULL DEFAULT 0,
    penalty_backtracks    REAL    NOT NULL DEFAULT 0,
    penalty_timeout       REAL    NOT NULL DEFAULT 0,
    duration_s            REAL    NOT NULL DEFAULT 0,
    turns                 INTEGER NOT NULL DEFAULT 0,
    tool_calls_n          INTEGER NOT NULL DEFAULT 0,
    timed_out             INTEGER NOT NULL DEFAULT 0,
    criteria_passed       INTEGER NOT NULL DEFAULT 0,
    criteria_total        INTEGER NOT NULL DEFAULT 0,
    stars                 INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_model    ON runs(model);
CREATE INDEX IF NOT EXISTS idx_runs_level_id ON runs(level_id);
CREATE INDEX IF NOT EXISTS idx_runs_ts       ON runs(ts);
"""

# Forward-only column migrations.  Add new entries here when the schema grows.
# Each entry is a single ALTER TABLE statement; the column name is parsed from
# position [4] ("ALTER TABLE runs ADD COLUMN <name> ...") to skip if present.
_MIGRATIONS: list[str] = [
    "ALTER TABLE runs ADD COLUMN mode TEXT NOT NULL DEFAULT 'unguided'",
    "ALTER TABLE runs ADD COLUMN penalty_retry REAL NOT NULL DEFAULT 0",
]

# Indexes that depend on migrated columns — created after _MIGRATIONS run.
_SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_runs_mode ON runs(mode);
"""


def _stars(score: float) -> int:
    return (
        5 if score >= 95 else
        4 if score >= 80 else
        3 if score >= 60 else
        2 if score >= 35 else
        1 if score >  0  else
        0
    )


class Store:
    """
    Thread-safe SQLite store.

    Usage
    ─────
      store = Store(Path("benchb0t.db")).init()
      store.record_run(result_dict)   # called from runner subprocess
      store.get_summary()             # called from dashboard handlers
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()   # serialises all DB access within this process

    def init(self) -> "Store":
        # timeout=15 → callers block up to 15 s waiting for a cross-process lock
        # rather than raising immediately.
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            timeout=15,
        )
        self._conn.row_factory = sqlite3.Row

        # WAL = safe concurrent reads+writes across processes.
        # Fall back to MEMORY if the filesystem rejects WAL (NFS/FUSE/CI mounts).
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.OperationalError:
                logger.warning("WAL mode unavailable — falling back to MEMORY journal")
                self._conn.execute("PRAGMA journal_mode=MEMORY")
                self._conn.execute("PRAGMA synchronous=OFF")

            # 1. Create base table + pre-existing indexes (idempotent).
            self._conn.executescript(_SCHEMA_BASE)

            # 2. Run forward-only column migrations so indexes below can reference them.
            existing_cols = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(runs)").fetchall()
            }
            for stmt in _MIGRATIONS:
                try:
                    col = stmt.split()[4]   # "ALTER TABLE runs ADD COLUMN <col> …"
                except IndexError:
                    col = None
                if col and col in existing_cols:
                    continue
                try:
                    self._conn.execute(stmt)
                    logger.info("Migration applied: %s", stmt[:60])
                except sqlite3.OperationalError as exc:
                    logger.debug("Migration skipped (%s): %s", exc, stmt[:60])

            # 3. Create indexes that depend on migrated columns (idempotent).
            self._conn.executescript(_SCHEMA_INDEXES)
            self._conn.commit()

        logger.info("Store ready at %s", self._path)
        return self

    def record_run(self, result: dict[str, Any]) -> None:
        """Persist a completed level result. Idempotent (INSERT OR REPLACE)."""
        if not self._conn:
            return
        sc    = result.get("score", {})
        dims  = sc.get("dimensions", {})
        pens  = sc.get("penalties", {})
        crits = sc.get("criteria", [])
        total = sc.get("total", 0)

        row = {
            "id":                    result.get("run_id", f"r{int(time.time())}"),
            "ts":                    result.get("ts", time.time()),
            "model":                 result.get("model", ""),
            "base_url":              result.get("base_url", ""),
            "harness":               result.get("harness", ""),
            "mode":                  result.get("mode", "unguided"),
            "level_id":              result.get("level_id", ""),
            "level_name":            result.get("level_name", result.get("level_id", "")),
            "difficulty":            result.get("difficulty", 1),
            "score_total":           total,
            "score_completion":      dims.get("completion", 0),
            "score_efficiency":      dims.get("efficiency", 0),
            "score_self_correction": dims.get("self_correction", 0),
            "score_path_quality":    dims.get("path_quality", 0),
            "penalty_extra_calls":   pens.get("extra_calls", 0),
            "penalty_backtracks":    pens.get("backtracks", 0),
            "penalty_timeout":       pens.get("timeout", 0),
            "penalty_retry":         pens.get("retry", 0),
            "duration_s":            result.get("duration_s", 0),
            "turns":                 result.get("turns", 0),
            "tool_calls_n":          result.get("tool_calls_n", 0),
            "timed_out":             int(bool(result.get("timed_out", False))),
            "criteria_passed":       sum(1 for c in crits if c.get("passed")),
            "criteria_total":        len(crits),
            "stars":                 _stars(total),
        }

        try:
            with self._lock:
                self._conn.execute(
                    """INSERT OR REPLACE INTO runs (
                        id, ts, model, base_url, harness, mode,
                        level_id, level_name, difficulty,
                        score_total,
                        score_completion, score_efficiency,
                        score_self_correction, score_path_quality,
                        penalty_extra_calls, penalty_backtracks, penalty_timeout,
                        duration_s, turns, tool_calls_n, timed_out,
                        criteria_passed, criteria_total, stars,
                        penalty_retry
                    ) VALUES (
                        :id, :ts, :model, :base_url, :harness, :mode,
                        :level_id, :level_name, :difficulty,
                        :score_total,
                        :score_completion, :score_efficiency,
                        :score_self_correction, :score_path_quality,
                        :penalty_extra_calls, :penalty_backtracks, :penalty_timeout,
                        :duration_s, :turns, :tool_calls_n, :timed_out,
                        :criteria_passed, :criteria_total, :stars,
                        :penalty_retry
                    )""",
                    row,
                )
                self._conn.commit()
            logger.info(
                "Stored run %s — level=%s model=%s score=%.1f",
                row["id"], row["level_id"], row["model"], total,
            )
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as exc:
            # Database integrity or operational failure — must be visible
            logger.error("Failed to store run %s: %s", row.get("id", "?"), exc)
            raise

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        if not self._conn:
            return []
        with self._lock:
            cur = self._conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def get_summary(self) -> dict:
        rows = self._query("""
            SELECT
                COUNT(*)                           AS total_runs,
                COUNT(DISTINCT model)              AS total_models,
                COUNT(DISTINCT level_id)           AS total_levels,
                ROUND(AVG(score_total),   1)       AS avg_score,
                ROUND(MAX(score_total),   1)       AS best_score,
                COALESCE(SUM(stars),      0)       AS total_stars,
                COALESCE(SUM(timed_out),  0)       AS total_timeouts
            FROM runs
        """)
        return rows[0] if rows else {}

    def get_model_stats(self) -> list[dict]:
        return self._query("""
            SELECT
                model,
                COUNT(*)                              AS run_count,
                ROUND(AVG(score_total),         1)    AS avg_score,
                ROUND(MAX(score_total),         1)    AS best_score,
                ROUND(AVG(score_completion),    1)    AS avg_completion,
                ROUND(AVG(score_efficiency),    1)    AS avg_efficiency,
                ROUND(AVG(score_self_correction),1)   AS avg_self_correction,
                ROUND(AVG(score_path_quality),  1)    AS avg_path_quality,
                COALESCE(SUM(stars),            0)    AS total_stars,
                ROUND(AVG(duration_s),          1)    AS avg_duration,
                ROUND(AVG(turns),               1)    AS avg_turns,
                ROUND(AVG(tool_calls_n),        1)    AS avg_tools,
                COALESCE(SUM(timed_out),        0)    AS timeouts
            FROM runs
            GROUP BY model
            ORDER BY avg_score DESC
        """)

    def get_level_stats(self) -> list[dict]:
        return self._query("""
            SELECT
                level_id, level_name, difficulty,
                COUNT(*)                              AS run_count,
                ROUND(AVG(score_total),   1)          AS avg_score,
                ROUND(MAX(score_total),   1)          AS best_score,
                ROUND(AVG(turns),         1)          AS avg_turns,
                ROUND(AVG(duration_s),    1)          AS avg_duration,
                ROUND(
                    SUM(criteria_passed) * 1.0 /
                    NULLIF(SUM(criteria_total), 0),
                    2
                )                                     AS pass_rate
            FROM runs
            GROUP BY level_id
            ORDER BY difficulty, level_id
        """)

    def get_runs(
        self,
        limit: int = 200,
        offset: int = 0,
        model: str = "",
        level_id: str = "",
        min_stars: int | None = None,
        timed_out: bool | None = None,
    ) -> list[dict]:
        """
        Return runs newest-first with optional filters.

        Parameters
        ----------
        limit / offset : pagination
        model         : exact match (empty = all models)
        level_id      : exact match (empty = all levels)
        min_stars     : only runs with stars >= this value
        timed_out     : True = only timeouts, False = no timeouts, None = both
        """
        clauses: list[str] = []
        params:  list[Any] = []

        if model:
            clauses.append("model = ?")
            params.append(model)
        if level_id:
            clauses.append("level_id = ?")
            params.append(level_id)
        if min_stars is not None:
            clauses.append("stars >= ?")
            params.append(min_stars)
        if timed_out is not None:
            clauses.append("timed_out = ?")
            params.append(1 if timed_out else 0)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params += [limit, offset]
        return self._query(
            f"SELECT * FROM runs {where} ORDER BY ts DESC LIMIT ? OFFSET ?",
            tuple(params),
        )

    def get_run_count(
        self,
        model: str = "",
        level_id: str = "",
    ) -> int:
        """Total number of runs matching the given filters (for pagination)."""
        clauses: list[str] = []
        params:  list[Any] = []
        if model:
            clauses.append("model = ?")
            params.append(model)
        if level_id:
            clauses.append("level_id = ?")
            params.append(level_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._query(f"SELECT COUNT(*) AS n FROM runs {where}", tuple(params))
        return rows[0]["n"] if rows else 0

    def get_run_by_id(self, run_id: str) -> dict | None:
        """
        Fetch a single run by its primary key (8-char hex id).
        Returns None if not found.
        """
        rows = self._query("SELECT * FROM runs WHERE id = ?", (run_id,))
        return rows[0] if rows else None

    def get_distinct_models(self) -> list[str]:
        """Sorted list of all distinct model names in the DB."""
        rows = self._query("SELECT DISTINCT model FROM runs ORDER BY model")
        return [r["model"] for r in rows if r.get("model")]

    def get_distinct_levels(self) -> list[dict]:
        """Distinct levels (id + name) ordered by difficulty."""
        rows = self._query(
            "SELECT DISTINCT level_id, level_name, difficulty FROM runs ORDER BY difficulty, level_id"
        )
        return [
            {"level_id": r["level_id"], "level_name": r["level_name"], "difficulty": r["difficulty"]}
            for r in rows
        ]

    def get_model_detail(self, model: str) -> dict:
        """
        Per-level breakdown for one model.

        Returns:
          - overall stats (same shape as get_model_stats row)
          - per_level list: each level with best score, best stars, run count, avg turns
        Used by the analytics Dex entry to render the level-conquest grid.
        """
        overall = self._query(
            """
            SELECT
                model,
                COUNT(*)                               AS run_count,
                ROUND(AVG(score_total),          1)    AS avg_score,
                ROUND(MAX(score_total),          1)    AS best_score,
                ROUND(AVG(score_completion),     1)    AS avg_completion,
                ROUND(AVG(score_efficiency),     1)    AS avg_efficiency,
                ROUND(AVG(score_self_correction),1)    AS avg_self_correction,
                ROUND(AVG(score_path_quality),   1)    AS avg_path_quality,
                COALESCE(SUM(stars),             0)    AS total_stars,
                ROUND(AVG(turns),                1)    AS avg_turns,
                ROUND(AVG(duration_s),           1)    AS avg_duration,
                COALESCE(SUM(timed_out),         0)    AS timeouts
            FROM runs WHERE model = ?
            """,
            (model,),
        )
        per_level = self._query(
            """
            SELECT
                level_id, level_name, difficulty,
                COUNT(*)                             AS run_count,
                ROUND(MAX(score_total), 1)           AS best_score,
                MAX(stars)                           AS best_stars,
                ROUND(AVG(turns),       1)           AS avg_turns,
                COALESCE(SUM(timed_out), 0)          AS timeouts
            FROM runs WHERE model = ?
            GROUP BY level_id
            ORDER BY difficulty, level_id
            """,
            (model,),
        )
        return {
            "model":     model,
            "overall":   overall[0] if overall else {},
            "per_level": per_level,
        }

    def get_mode_comparison(self) -> list[dict]:
        """
        Per-level, per-model comparison of guided vs unguided scores.
        Returns rows with both scores side-by-side so the dashboard can
        render a diff table: which models need hand-holding and which don't.
        """
        return self._query("""
            SELECT
                level_id,
                model,
                mode,
                COUNT(*)                        AS run_count,
                ROUND(AVG(score_total),    1)   AS avg_score,
                ROUND(AVG(turns),          1)   AS avg_turns,
                ROUND(AVG(tool_calls_n),   1)   AS avg_tools,
                COALESCE(SUM(timed_out),   0)   AS timeouts
            FROM runs
            GROUP BY level_id, model, mode
            ORDER BY level_id, model, mode
        """)
