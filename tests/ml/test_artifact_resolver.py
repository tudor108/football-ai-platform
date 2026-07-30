from __future__ import annotations

import json
from pathlib import Path

from ml.utils.artifact_resolver import LEGACY_ARTIFACTS, ResolutionStatus, resolve_latest_artifacts
from ml.utils.run_context import RunContext


def _write_legacy(root: Path) -> None:
    for relative in LEGACY_ARTIFACTS.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "data", encoding="utf-8")


def _successful_run(tmp_path: Path) -> tuple[RunContext, dict[str, Path]]:
    config = tmp_path / "config.yaml"
    config.write_text("features: [a]\n", encoding="utf-8")
    context = RunContext(tmp_path / "output" / "ml", config, repo_root=tmp_path)
    artifacts: dict[str, Path] = {}
    for name, relative in LEGACY_ARTIFACTS.items():
        path = context.artifact_path(relative.as_posix())
        path.write_text("{}" if path.suffix == ".json" else "data", encoding="utf-8")
        artifacts[name] = path
    context.succeed(artifacts)
    return context, artifacts


def test_valid_latest_pointer(tmp_path: Path) -> None:
    context, _ = _successful_run(tmp_path)
    result = resolve_latest_artifacts(context.output_root)
    assert result.status == ResolutionStatus.VERSIONED_RUN
    assert result.run_id == context.run_id


def test_absent_pointer_uses_flat_fallback(tmp_path: Path) -> None:
    root = tmp_path / "output" / "ml"
    _write_legacy(root)
    result = resolve_latest_artifacts(root)
    assert result.status == ResolutionStatus.LEGACY_FALLBACK


def test_corrupt_pointer_uses_fallback_when_available(tmp_path: Path) -> None:
    root = tmp_path / "output" / "ml"
    _write_legacy(root)
    (root / "latest_run.json").write_text("{broken", encoding="utf-8")
    result = resolve_latest_artifacts(root)
    assert result.status == ResolutionStatus.LEGACY_FALLBACK
    assert result.message and "Invalid" in result.message


def test_pointer_to_failed_or_missing_run_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "output" / "ml"
    root.mkdir(parents=True)
    (root / "latest_run.json").write_text(json.dumps({"run_id": "missing"}), encoding="utf-8")
    assert resolve_latest_artifacts(root).status == ResolutionStatus.INVALID_POINTER


def test_no_pointer_and_no_legacy_is_no_artifacts(tmp_path: Path) -> None:
    assert resolve_latest_artifacts(tmp_path / "nothing").status == ResolutionStatus.NO_ARTIFACTS
