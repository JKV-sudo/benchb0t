"""
framework/artifacts.py
~~~~~~~~~~~~~~~~~~~~~~
Helpers for saving optional run artifacts such as preview screenshots,
portable result bundles, and Docker image snapshots.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from framework.container import LevelContainer

logger = logging.getLogger(__name__)


def run_artifacts_dir(runs_dir: Path, run_id: str) -> Path:
    """Return the artifact directory for one run, creating it if needed."""
    path = Path(runs_dir) / "artifacts" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def wait_for_preview(url: str, timeout_s: float = 12.0, interval_s: float = 0.4) -> bool:
    """Poll the preview URL until it responds or the timeout expires."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status < 500
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                return True
            # Server error (5xx) — retry
        except (urllib.error.URLError, OSError):
            # Network transience (timeout, connection refused, etc.) — retry
            pass
        time.sleep(interval_s)
    return False


def capture_preview_screenshot(
    *,
    host_port: int,
    preview_path: str,
    dest_path: Path,
    wait_timeout_s: float = 12.0,
) -> dict[str, Any] | None:
    """
    Capture a PNG screenshot of the preview URL using Playwright CLI.

    This is best-effort by design. When the local machine does not have `npx`
    or Playwright browsers available, the run should still succeed.
    """
    preview_url = f"http://localhost:{host_port}{preview_path}"
    if shutil.which("npx") is None:
        logger.info("Skipping preview screenshot: npx is not available")
        return None
    if not wait_for_preview(preview_url, timeout_s=wait_timeout_s):
        logger.info("Skipping preview screenshot: preview never became reachable at %s", preview_url)
        return None

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "npx",
        "playwright",
        "screenshot",
        "--browser",
        "chromium",
        "--device",
        "Desktop Chrome",
        "--full-page",
        "--timeout",
        "10000",
        "--wait-for-timeout",
        "1200",
        preview_url,
        str(dest_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        logger.info("Preview screenshot saved: %s", dest_path)
        if proc.stdout.strip():
            logger.debug("playwright screenshot stdout: %s", proc.stdout.strip())
    except subprocess.CalledProcessError as exc:
        logger.warning("Preview screenshot: playwright exited %d: %s", exc.returncode, exc.stderr)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Preview screenshot: playwright timed out after 45s")
        return None
    except FileNotFoundError:
        logger.warning("Preview screenshot: npx executable not found")
        return None

    if not dest_path.exists():
        return None

    return {
        "kind": "preview_screenshot",
        "label": "Preview screenshot",
        "path": str(dest_path),
        "url": preview_url,
        "size_bytes": dest_path.stat().st_size,
    }


def save_container_snapshot(
    *,
    container: LevelContainer,
    artifact_dir: Path,
    level_id: str,
    run_id: str,
) -> dict[str, Any] | None:
    """
    Commit the container to a Docker image and write metadata for later reuse.
    """
    try:
        image_ref = container.snapshot()
    except Exception as exc:
        # Snapshot failures are best-effort; log and continue
        # Can fail if: Docker not responsive, disk full, invalid container state, etc.
        logger.warning("Container snapshot failed for %s/%s: %s", level_id, run_id, exc)
        return None

    metadata = {
        "kind": "container_snapshot",
        "label": "Container snapshot",
        "image_ref": image_ref,
        "level_id": level_id,
        "run_id": run_id,
        "created_at": time.time(),
    }
    metadata_path = artifact_dir / "container-snapshot.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["path"] = str(metadata_path)
    metadata["size_bytes"] = metadata_path.stat().st_size
    return metadata


def save_result_bundle(
    *,
    artifact_dir: Path,
    run_id: str,
    result: dict[str, Any],
    log_path: Path,
) -> dict[str, Any] | None:
    """
    Save a portable ZIP bundle with the result JSON, agentlog, and any
    previously written artifact files for the same run.
    """
    bundle_path = artifact_dir / f"{run_id}-result-bundle.zip"
    result_json = json.dumps(result, indent=2, ensure_ascii=False)

    try:
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("result.json", result_json)
            if log_path.exists():
                bundle.write(log_path, arcname=log_path.name)
            for artifact_path in sorted(artifact_dir.iterdir()):
                if not artifact_path.is_file() or artifact_path == bundle_path:
                    continue
                bundle.write(artifact_path, arcname=artifact_path.name)
    except (OSError, zipfile.BadZipFile) as exc:
        # ZIP creation failures (disk full, permission denied, etc.)
        logger.warning("Result bundle creation failed for %s: %s", run_id, exc)
        return None

    return {
        "kind": "result_bundle",
        "label": "Result bundle",
        "path": str(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
    }
