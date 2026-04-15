from __future__ import annotations

import json
import zipfile
from pathlib import Path

from framework.artifacts import run_artifacts_dir, save_result_bundle


def test_save_result_bundle_includes_result_and_agentlog(tmp_path: Path) -> None:
    artifact_dir = run_artifacts_dir(tmp_path, "abc12345")
    log_path = tmp_path / "20260413_l99-test_abc12345.agentlog"
    log_path.write_text('{"type":"session_end","run_id":"abc12345"}\n', encoding="utf-8")
    screenshot_path = artifact_dir / "preview.png"
    screenshot_path.write_bytes(b"png-bytes")

    result = {
        "run_id": "abc12345",
        "level_id": "l99-test",
        "score": {"total": 88.0},
        "artifacts": [],
    }

    bundle = save_result_bundle(
        artifact_dir=artifact_dir,
        run_id="abc12345",
        result=result,
        log_path=log_path,
    )

    assert bundle is not None
    bundle_path = Path(bundle["path"])
    assert bundle_path.exists()

    with zipfile.ZipFile(bundle_path) as archive:
        names = sorted(archive.namelist())
        assert names == [
            "20260413_l99-test_abc12345.agentlog",
            "preview.png",
            "result.json",
        ]
        result_json = json.loads(archive.read("result.json").decode("utf-8"))
        assert result_json["run_id"] == "abc12345"
