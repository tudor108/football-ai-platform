"""Align arbitrary raw cluster labels to stable cluster identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AlignmentResult:
    mapping: dict[int, str]
    events: tuple[dict[str, Any], ...]
    overlap_matrix: dict[str, dict[str, float]]


def _next_stable_ids(existing: Iterable[str]) -> Iterable[str]:
    maximum = 0
    for value in existing:
        if str(value).startswith("SC") and str(value)[2:].isdigit():
            maximum = max(maximum, int(str(value)[2:]))
    while True:
        maximum += 1
        yield f"SC{maximum:03d}"


def _membership_sets(df: pd.DataFrame, cluster_column: str) -> dict[Any, set[str]]:
    return {
        cluster: set(group["team_id"].astype(str))
        for cluster, group in df.groupby(cluster_column, dropna=False)
    }


def align_cluster_labels(
    previous: pd.DataFrame | None,
    current: pd.DataFrame,
    feature_columns: list[str],
    membership_weight: float = 0.75,
    feature_profile_weight: float = 0.25,
) -> AlignmentResult:
    """Align current raw IDs to previous stable IDs using overlap and profiles."""
    if "raw_cluster_id" not in current:
        raise ValueError("current requires raw_cluster_id")
    current_ids = sorted(int(value) for value in current["raw_cluster_id"].dropna().unique())
    if previous is None or previous.empty:
        generator = _next_stable_ids([])
        mapping = {raw: next(generator) for raw in current_ids}
        events = tuple(
            {"event_type": "NEW_CLUSTER", "current_raw_cluster_id": raw, "stable_cluster_id": stable}
            for raw, stable in mapping.items()
        )
        return AlignmentResult(mapping, events, {})

    required = {"raw_cluster_id", "stable_cluster_id", "team_id"}
    if not required.issubset(previous.columns):
        raise ValueError(f"previous requires columns: {sorted(required)}")
    previous_stable = sorted(previous["stable_cluster_id"].dropna().astype(str).unique())
    previous_sets = _membership_sets(previous, "stable_cluster_id")
    current_sets = _membership_sets(current, "raw_cluster_id")

    overlap = np.zeros((len(previous_stable), len(current_ids)), dtype=float)
    distances = np.zeros_like(overlap)
    usable_features = [name for name in feature_columns if name in previous.columns and name in current.columns]
    previous_profiles = previous.groupby("stable_cluster_id")[usable_features].mean() if usable_features else None
    current_profiles = current.groupby("raw_cluster_id")[usable_features].mean() if usable_features else None
    all_profiles = pd.concat([previous_profiles, current_profiles]) if usable_features else None
    scales = all_profiles.std(ddof=0).replace(0, 1.0) if all_profiles is not None else None

    for row, stable_id in enumerate(previous_stable):
        for column, raw_id in enumerate(current_ids):
            intersection = len(previous_sets[stable_id] & current_sets[raw_id])
            union = len(previous_sets[stable_id] | current_sets[raw_id])
            overlap[row, column] = intersection / union if union else 0.0
            if usable_features and scales is not None:
                delta = (previous_profiles.loc[stable_id] - current_profiles.loc[raw_id]) / scales
                distances[row, column] = float(np.linalg.norm(delta.to_numpy(dtype=float)))

    normalized_distance = distances / max(float(distances.max()), 1.0)
    score = membership_weight * overlap + feature_profile_weight * (1.0 - normalized_distance)
    previous_indices, current_indices = linear_sum_assignment(-score)
    mapping: dict[int, str] = {}
    events: list[dict[str, Any]] = []
    for row, column in zip(previous_indices, current_indices):
        stable_id = previous_stable[row]
        raw_id = current_ids[column]
        if overlap[row, column] > 0 or score[row, column] > feature_profile_weight:
            mapping[raw_id] = stable_id

    generator = _next_stable_ids(previous_stable)
    for raw_id in current_ids:
        if raw_id not in mapping:
            stable_id = next(generator)
            mapping[raw_id] = stable_id
            events.append({"event_type": "NEW_CLUSTER", "current_raw_cluster_id": raw_id, "stable_cluster_id": stable_id})

    overlap_records = {
        stable_id: {str(raw_id): float(overlap[row, column]) for column, raw_id in enumerate(current_ids)}
        for row, stable_id in enumerate(previous_stable)
    }
    for row, stable_id in enumerate(previous_stable):
        related = [current_ids[column] for column in range(len(current_ids)) if overlap[row, column] >= 0.20]
        if len(related) > 1:
            events.append({"event_type": "POSSIBLE_SPLIT", "previous_stable_cluster_id": stable_id, "current_raw_clusters": related})
        if not any(mapped == stable_id for mapped in mapping.values()):
            events.append({"event_type": "DISAPPEARED_CLUSTER", "previous_stable_cluster_id": stable_id})
    for column, raw_id in enumerate(current_ids):
        related = [previous_stable[row] for row in range(len(previous_stable)) if overlap[row, column] >= 0.20]
        if len(related) > 1:
            events.append({"event_type": "POSSIBLE_MERGE", "current_raw_cluster_id": raw_id, "previous_stable_clusters": related})
    return AlignmentResult(mapping, tuple(events), overlap_records)


def apply_stable_ids(clusters: pd.DataFrame, mapping: dict[int, str]) -> pd.DataFrame:
    """Return a copy containing stable IDs while preserving raw IDs."""
    output = clusters.copy()
    output["stable_cluster_id"] = output["raw_cluster_id"].map(lambda value: mapping[int(value)])
    return output
