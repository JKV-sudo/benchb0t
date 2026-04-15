"""
framework/dashboard_state.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shared mutable runtime state for the live dashboard server.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framework.config import LoadedFrameworkConfig
from framework.dashboard_models import RunRequest
from framework.store import Store

logger = logging.getLogger(__name__)

_CREDS_KEYS = ("base_url", "model", "api_key", "providers")


@dataclass
class DashboardState:
    runs_dir: Path = Path("runs")
    project_dir: Path = field(default_factory=lambda: Path(".").resolve())
    creds_file: Path = Path(".benchb0t_creds.json")
    store: Store | None = None
    loaded_config: LoadedFrameworkConfig | None = None
    active_procs: list[subprocess.Popen] = field(default_factory=list)
    active_proc_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    run_batch_started_at: float = 0.0
    runner_log: collections.deque[str] = field(
        default_factory=lambda: collections.deque(maxlen=600)
    )

    def load_creds(self) -> dict[str, Any]:
        try:
            if self.creds_file.exists():
                return json.loads(self.creds_file.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def save_creds(self, data: dict[str, Any]) -> None:
        try:
            safe = {key: data[key] for key in _CREDS_KEYS if key in data}
            self.creds_file.write_text(json.dumps(safe, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not save credentials: %s", exc)

    def providers_from_request(self, req: RunRequest) -> list[dict[str, str]]:
        providers: list[dict[str, str]] = []
        if req.providers:
            for provider in req.providers:
                base_url = provider.base_url.strip()
                model = provider.model.strip()
                if not base_url or not model:
                    continue
                providers.append(
                    {
                        "base_url": base_url,
                        "model": model,
                        "api_key": provider.api_key.strip(),
                        "label": (provider.label or model).strip(),
                    }
                )
        elif req.base_url.strip() and req.model.strip():
            providers.append(
                {
                    "base_url": req.base_url.strip(),
                    "model": req.model.strip(),
                    "api_key": req.api_key.strip(),
                    "label": req.model.strip(),
                }
            )
        return providers

    def save_provider_creds(self, providers: list[dict[str, str]]) -> None:
        if not providers:
            return
        first = providers[0]
        self.save_creds(
            {
                "base_url": first["base_url"],
                "model": first["model"],
                "api_key": first["api_key"],
                "providers": providers,
            }
        )

    def alive_procs(self) -> list[subprocess.Popen]:
        self.active_procs = [proc for proc in self.active_procs if proc.poll() is None]
        return list(self.active_procs)

    def record_runner_output(self, line: str, prefix: str = "") -> None:
        tagged = f"{prefix}{line}" if prefix else line
        self.runner_log.append(tagged)
        logger.debug("[runner] %s", tagged)

    def reset_run_batch(self) -> None:
        self.runner_log.clear()
        self.active_procs = []
        self.run_batch_started_at = time.time()

    def resolve_runtime_path(self, path: Path) -> Path:
        if self.loaded_config is None or path.is_absolute():
            return path
        return self.loaded_config.resolve_path(path)

    def apply_runtime_config(
        self,
        loaded_config: LoadedFrameworkConfig,
        *,
        runs_override: Path | None = None,
    ) -> None:
        self.loaded_config = loaded_config
        self.project_dir = loaded_config.project_dir
        self.runs_dir = (
            loaded_config.runs_dir
            if runs_override is None
            else self.resolve_runtime_path(runs_override)
        )
        self.creds_file = self.project_dir / ".benchb0t_creds.json"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(loaded_config.db_path).init()
