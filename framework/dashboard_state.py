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
    active_batch: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    DEFAULT_SETTINGS: dict[str, Any] = field(
        default_factory=lambda: {
            "default_harness": "",
            "capture_preview_screenshot": True,
            "save_result_bundle": False,
            "save_container_snapshot": False,
            "auto_detect_providers": False,
            "crt_scanlines": True,
            "assistant_language": "en",
            "confirm_stop_run": True,
            "auto_refresh_interval_s": 5,
            "show_tool_previews": True,
        }
    )

    def _active_batch_path(self) -> Path:
        return self.runs_dir / ".active_batch.json"

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
        self.active_batch = {}
        self._save_active_batch()

    def set_active_batch(self, batch: dict[str, Any]) -> None:
        self.active_batch = dict(batch, started_at=time.time())
        self._save_active_batch()

    def clear_active_batch(self) -> None:
        self.active_batch = {}
        self._save_active_batch()

    def _save_active_batch(self) -> None:
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            self._active_batch_path().write_text(
                json.dumps(self.active_batch, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save active batch: %s", exc)

    def load_active_batch(self) -> dict[str, Any]:
        try:
            path = self._active_batch_path()
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not load active batch: %s", exc)
        return {}

    def load_settings(self) -> dict[str, Any]:
        """Load settings from the store, falling back to defaults."""
        stored = self.store.get_settings() if self.store else {}
        merged = dict(self.DEFAULT_SETTINGS)
        merged.update(stored)
        self.settings = merged
        return merged

    def save_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Persist setting updates and return the merged settings."""
        merged = dict(self.settings)
        merged.update(updates)
        self.settings = merged
        if self.store:
            self.store.save_settings(merged)
        return merged

    def get_setting(self, key: str) -> Any:
        """Return a single setting value (default if missing)."""
        return self.settings.get(key, self.DEFAULT_SETTINGS.get(key))

    def resolve_runtime_path(self, path: Path) -> Path:
        if self.loaded_config is None or path.is_absolute():
            return path
        return self.loaded_config.resolve_path(path)

    def load_providers(self) -> list[dict[str, Any]]:
        if self.store:
            return self.store.get_providers()
        return []

    def save_providers(self, providers: list[dict[str, Any]]) -> None:
        if self.store:
            self.store.save_providers(providers)
        self.save_provider_creds(providers)

    def has_providers(self) -> bool:
        if self.store:
            return self.store.has_providers()
        creds = self.load_creds()
        return bool(creds.get("providers") or (creds.get("base_url") and creds.get("model")))

    def migrate_legacy_creds(self) -> list[dict[str, Any]]:
        """Import providers from legacy .benchb0t_creds.json into the store."""
        providers: list[dict[str, Any]] = []
        creds = self.load_creds()
        for provider in creds.get("providers", []):
            if provider.get("base_url") and provider.get("model"):
                providers.append(dict(provider, id=f"legacy-{provider['model']}", enabled=True))
        if not providers and creds.get("base_url") and creds.get("model"):
            providers.append(
                {
                    "id": f"legacy-{creds['model']}",
                    "label": creds.get("model", ""),
                    "base_url": creds["base_url"],
                    "model": creds["model"],
                    "api_key": creds.get("api_key", ""),
                    "source": "legacy:creds",
                    "enabled": True,
                }
            )
        if providers and self.store:
            self.store.save_providers(providers)
        return providers

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
        self.active_batch = self.load_active_batch()
        self.load_settings()
        if not self.has_providers():
            self.migrate_legacy_creds()
