"""Adjusted Rand Index comparisons with explicit compatibility metadata."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
from sklearn.metrics import adjusted_rand_score


def compare_partitions(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    previous_id: str,
    current_id: str,
    minimum_common_teams: int = 2,
    previous_metadata: Mapping[str, Any] | None = None,
    current_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate ARI on common teams; raw label permutations are harmless."""
    previous_label = "raw_cluster_id" if "raw_cluster_id" in previous else "cluster"
    current_label = "raw_cluster_id" if "raw_cluster_id" in current else "cluster"
    common = previous[["team_id", previous_label]].merge(
        current[["team_id", current_label]], on="team_id", suffixes=("_previous", "_current")
    )
    warnings: list[str] = []
    method_keys = ("feature_schema_hash", "preprocessing_hash", "algorithm", "selected_parameters")
    method_changed = any(
        (previous_metadata or {}).get(key) != (current_metadata or {}).get(key)
        for key in method_keys
        if (previous_metadata or {}).get(key) is not None and (current_metadata or {}).get(key) is not None
    )
    if method_changed:
        warnings.append("Feature schema, preprocessing, algorithm, or selected parameters changed.")
    if len(common) < minimum_common_teams:
        status = "INSUFFICIENT_COMMON_TEAMS"
        ari = None
    else:
        ari = float(adjusted_rand_score(common.iloc[:, 1], common.iloc[:, 2]))
        status = "METHOD_CHANGED" if method_changed else "COMPARABLE"
    return {
        "current_run_id": current_id,
        "previous_run_id": previous_id,
        "common_team_count": int(len(common)),
        "current_team_count": int(current["team_id"].nunique()),
        "previous_team_count": int(previous["team_id"].nunique()),
        "adjusted_rand_index": ari,
        "comparison_status": status,
        "compatibility_warnings": warnings,
    }
