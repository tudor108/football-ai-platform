"""Compare final season-level clusterings and raw feature distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ml.clustering.algorithms import _fit_predict, _preprocess_matrix
from ml.features.build_team_features import build_features, build_team_level_from_match_features
from ml.stability.ari_analysis import compare_partitions
from ml.stability.historical_snapshots import filter_completed_matches
from ml.utils.run_context import atomic_write_json


def compare_seasons(
    match_df: pd.DataFrame,
    config: dict[str, Any],
    selected_algorithm: str,
    selected_parameters: dict[str, Any],
    stability_dir: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Compare the latest two seasons or return a safe NOT_AVAILABLE result."""
    seasons = sorted(match_df["season"].dropna().unique())
    output_path = stability_dir / "season_comparison.json"
    if len(seasons) < 2:
        result = {"status": "NOT_AVAILABLE", "reason": "At least two seasons are required."}
        atomic_write_json(output_path, result)
        return result, {"season_comparison": output_path}

    previous_season, current_season = seasons[-2], seasons[-1]
    partitions: dict[Any, pd.DataFrame] = {}
    for season in (previous_season, current_season):
        season_matches = match_df.loc[match_df["season"] == season].copy()
        maximum_date = pd.to_datetime(season_matches["date"], errors="coerce", utc=True).max()
        complete = filter_completed_matches(season_matches, maximum_date.to_pydatetime())
        features = build_features(build_team_level_from_match_features(complete), config)
        matrix = _preprocess_matrix(features, config)
        _, labels = _fit_predict(selected_algorithm, selected_parameters, matrix)
        features["raw_cluster_id"] = np.asarray(labels, dtype=int)
        partitions[season] = features

    previous = partitions[previous_season]
    current = partitions[current_season]
    comparison = compare_partitions(previous, current, str(previous_season), str(current_season), minimum_common_teams=2)
    previous_teams = set(previous["team_name"].astype(str))
    current_teams = set(current["team_name"].astype(str))
    feature_drift: list[dict[str, Any]] = []
    for feature in config["features"]:
        before = pd.to_numeric(previous[feature], errors="coerce").dropna().to_numpy()
        after = pd.to_numeric(current[feature], errors="coerce").dropna().to_numpy()
        pooled = float(np.sqrt((np.var(before) + np.var(after)) / 2.0))
        mean_difference = float(np.mean(after) - np.mean(before))
        feature_drift.append(
            {
                "feature": feature,
                "mean_difference": mean_difference,
                "standardized_mean_difference": mean_difference / pooled if pooled > 0 else 0.0,
                "wasserstein_distance": float(wasserstein_distance(before, after)),
            }
        )
    prev_dist = previous["raw_cluster_id"].value_counts(normalize=True)
    curr_dist = current["raw_cluster_id"].value_counts(normalize=True)
    all_clusters = prev_dist.index.union(curr_dist.index)
    total_variation = float(0.5 * sum(abs(prev_dist.get(cluster, 0) - curr_dist.get(cluster, 0)) for cluster in all_clusters))
    result = {
        "status": "AVAILABLE",
        "previous_season": str(previous_season),
        "current_season": str(current_season),
        "teams_only_previous": sorted(previous_teams - current_teams),
        "teams_only_current": sorted(current_teams - previous_teams),
        "ari_comparison": comparison,
        "cluster_distribution_total_variation_distance": total_variation,
        "feature_drift": feature_drift,
        "interpretation_note": "Distribution changes are descriptive and do not establish causality.",
    }
    atomic_write_json(output_path, result)
    plot_path = stability_dir / "plots" / "season_feature_drift.png"
    plot_data = pd.DataFrame(feature_drift).sort_values("standardized_mean_difference", key=lambda values: values.abs())
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(plot_data["feature"], plot_data["standardized_mean_difference"])
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_title("Season feature drift (standardized mean difference)")
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return result, {"season_comparison": output_path, "season_feature_drift_plot": plot_path}
