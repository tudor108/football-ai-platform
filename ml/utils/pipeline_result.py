"""Typed result returned by the clustering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PipelineResult:
    run_id: str
    status: str
    run_dir: Path
    manifest_path: Path
    algorithm: str
    candidate_id: str
    selected_parameters: Mapping[str, Any]
    artifact_paths: Mapping[str, Path]
    warnings: tuple[str, ...] = ()
