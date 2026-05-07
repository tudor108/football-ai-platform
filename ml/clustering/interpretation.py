"""Cluster interpretation for football team archetypes."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _auto_label(means: pd.Series, global_means: pd.Series) -> str:
    if means.get("avg_goals_scored", 0) > global_means.get("avg_goals_scored", 0) and means.get("win_rate", 0) > global_means.get("win_rate", 0):
        return "elite_attack"
    if means.get("avg_goals_conceded", 1) < global_means.get("avg_goals_conceded", 1) and means.get("defense_strength", 0) > global_means.get("defense_strength", 0):
        return "defensive_wall"
    if means.get("std_goal_diff", 0) > global_means.get("std_goal_diff", 0):
        return "high_variance"
    if means.get("draw_rate", 0) > global_means.get("draw_rate", 0):
        return "balanced_control"
    return "mid_table_profile"


def interpret_clusters(best_labels: pd.DataFrame, features_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    features = config["features"]
    df = features_df.copy().merge(best_labels, on="team_id", how="left")
    global_means = df[features].mean(numeric_only=True)

    cluster_info: dict[str, Any] = {}
    for cluster_id in sorted(df["cluster"].dropna().unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        means = cluster_df[features].mean(numeric_only=True)
        diff = means - global_means
        top_features = diff.abs().sort_values(ascending=False).index.tolist()[:5]

        cluster_info[str(int(cluster_id))] = {
            "label": _auto_label(means, global_means),
            "n_teams": int(cluster_df["team_id"].nunique()),
            "teams": sorted(cluster_df["team_name"].dropna().astype(str).unique().tolist()),
            "top_features": top_features,
            "means": {k: float(v) for k, v in means.to_dict().items()},
            "diff_vs_global": {k: float(v) for k, v in diff.to_dict().items()},
        }
    return cluster_info
