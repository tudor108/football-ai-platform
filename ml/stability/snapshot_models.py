"""Typed models shared by temporal stability modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_id: str
    snapshot_type: str
    cutoff_date: datetime
    directory: Path
    clusters: pd.DataFrame
    centroids: pd.DataFrame
    processed_matrix: Any
    metadata: Mapping[str, Any]
    warnings: tuple[str, ...] = field(default_factory=tuple)
