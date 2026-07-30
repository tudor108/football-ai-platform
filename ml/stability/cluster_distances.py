"""Distance-based cluster membership strength in preprocessed feature space."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def calculate_assignment_strength(
    teams: pd.DataFrame,
    processed_matrix: np.ndarray,
    cluster_column: str = "raw_cluster_id",
    minimum_cluster_size: int = 3,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Calculate bounded assignment strength; it is not a probability."""
    if len(teams) != len(processed_matrix):
        raise ValueError("teams and processed_matrix must have equal row counts")
    labels = teams[cluster_column].to_numpy()
    cluster_ids = sorted(np.unique(labels))
    centroids = {cluster: processed_matrix[labels == cluster].mean(axis=0) for cluster in cluster_ids}
    sizes = {cluster: int((labels == cluster).sum()) for cluster in cluster_ids}
    records: list[dict[str, Any]] = []
    for position, (_, team) in enumerate(teams.reset_index(drop=True).iterrows()):
        own = labels[position]
        own_distance = float(np.linalg.norm(processed_matrix[position] - centroids[own]))
        alternatives = [
            (cluster, float(np.linalg.norm(processed_matrix[position] - centroid)))
            for cluster, centroid in centroids.items()
            if cluster != own
        ]
        nearest_cluster, nearest_distance = min(alternatives, key=lambda item: item[1]) if alternatives else (None, own_distance)
        margin = nearest_distance - own_distance
        strength = float(np.clip(margin / max(nearest_distance, epsilon), 0.0, 1.0)) if alternatives else 0.0
        status = "RELIABLE"
        if sizes[own] < minimum_cluster_size:
            status = "UNRELIABLE_SMALL_CLUSTER"
        records.append(
            {
                "team_id": team.get("team_id"),
                "team_name": team.get("team_name"),
                "raw_cluster_id": int(own),
                "stable_cluster_id": team.get("stable_cluster_id"),
                "cluster_size": sizes[own],
                "distance_to_own_centroid": own_distance,
                "distance_to_nearest_other_centroid": nearest_distance,
                "nearest_other_cluster": int(nearest_cluster) if nearest_cluster is not None else None,
                "separation_margin": margin,
                "raw_assignment_strength": strength,
                "assignment_strength_status": status,
            }
        )
    return pd.DataFrame(records)
