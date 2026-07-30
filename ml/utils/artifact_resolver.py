"""Resolve coherent versioned ML artifacts with temporary legacy fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from ml.utils.run_context import SUCCEEDED


class ResolutionStatus(str, Enum):
    VERSIONED_RUN = "VERSIONED_RUN"
    LEGACY_FALLBACK = "LEGACY_FALLBACK"
    INVALID_POINTER = "INVALID_POINTER"
    NO_ARTIFACTS = "NO_ARTIFACTS"


LEGACY_ARTIFACTS = {
    "clusters": Path("clusters/clusters.csv"),
    "metrics": Path("metrics/metrics.json"),
    "best_model_summary": Path("metrics/best_model_summary.json"),
    "cluster_interpretation": Path("metrics/cluster_interpretation.json"),
    "cluster_report": Path("report/cluster_report.md"),
}


@dataclass(frozen=True)
class ArtifactResolution:
    status: ResolutionStatus
    root: Path | None
    run_id: str | None
    artifacts: Mapping[str, Path]
    message: str | None = None


def _legacy_resolution(output_root: Path, invalid_message: str | None = None) -> ArtifactResolution:
    paths = {name: output_root / relative for name, relative in LEGACY_ARTIFACTS.items()}
    if all(path.is_file() for path in paths.values()):
        return ArtifactResolution(
            status=ResolutionStatus.LEGACY_FALLBACK,
            root=output_root,
            run_id=None,
            artifacts=paths,
            message=invalid_message,
        )
    status = ResolutionStatus.INVALID_POINTER if invalid_message else ResolutionStatus.NO_ARTIFACTS
    return ArtifactResolution(status=status, root=None, run_id=None, artifacts={}, message=invalid_message)


def resolve_latest_artifacts(output_root: str | Path = "output/ml") -> ArtifactResolution:
    """Resolve a valid completed run, falling back to coherent flat artifacts."""
    root = Path(output_root)
    pointer_path = root / "latest_run.json"
    if not pointer_path.is_file():
        return _legacy_resolution(root)
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_id = str(pointer["run_id"])
        run_dir = root / "runs" / run_id
        manifest_path = run_dir / "manifest.json"
        if not run_dir.is_dir() or not manifest_path.is_file():
            raise ValueError("Latest pointer references a missing run or manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != run_id or manifest.get("status") != SUCCEEDED:
            raise ValueError("Latest pointer does not reference a SUCCEEDED matching manifest")
        artifacts: dict[str, Path] = {}
        for record in manifest.get("required_artifacts", []):
            name = str(record["name"])
            path = run_dir / Path(str(record["path"]))
            if not path.is_file():
                raise ValueError(f"Required artifact is missing: {name}")
            artifacts[name] = path
        missing_names = set(LEGACY_ARTIFACTS) - set(artifacts)
        if missing_names:
            raise ValueError(f"Manifest lacks required artifact names: {sorted(missing_names)}")
        for record in manifest.get("optional_artifacts", []):
            path = run_dir / Path(str(record["path"]))
            if path.is_file():
                artifacts[str(record["name"])] = path
        return ArtifactResolution(
            status=ResolutionStatus.VERSIONED_RUN,
            root=run_dir,
            run_id=run_id,
            artifacts=artifacts,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return _legacy_resolution(root, f"Invalid latest pointer: {error}")


def find_previous_successful_run(output_root: str | Path, exclude_run_id: str) -> ArtifactResolution:
    """Find the newest valid successful historical run other than ``exclude_run_id``."""
    root = Path(output_root)
    runs_root = root / "runs"
    if not runs_root.is_dir():
        return ArtifactResolution(ResolutionStatus.NO_ARTIFACTS, None, None, {})
    for run_dir in sorted((path for path in runs_root.iterdir() if path.is_dir()), reverse=True):
        if run_dir.name == exclude_run_id:
            continue
        manifest_path = run_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != SUCCEEDED:
                continue
            artifacts = {
                str(item["name"]): run_dir / Path(str(item["path"]))
                for item in manifest.get("required_artifacts", [])
            }
            if all(path.is_file() for path in artifacts.values()):
                return ArtifactResolution(ResolutionStatus.VERSIONED_RUN, run_dir, run_dir.name, artifacts)
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return ArtifactResolution(ResolutionStatus.NO_ARTIFACTS, None, None, {})
