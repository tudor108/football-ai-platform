"""Cluster interpretation for football team archetypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusterNarrative:
    label: str
    description: str
    football_interpretation: str
    strengths: list[str]
    weaknesses: list[str]


def _score_profile(means: pd.Series, global_means: pd.Series) -> dict[str, float]:
    return {
        "attack": float(means.get("avg_goals_scored", 0) - global_means.get("avg_goals_scored", 0)),
        "defense": float(global_means.get("avg_goals_conceded", 0) - means.get("avg_goals_conceded", 0)),
        "results": float(means.get("win_rate", 0) - global_means.get("win_rate", 0)),
        "control": float(means.get("draw_rate", 0) - global_means.get("draw_rate", 0)),
        "volatility": float(means.get("std_goal_diff", 0) - global_means.get("std_goal_diff", 0)),
    }


def _narrative_from_profile(
    profile: dict[str, float],
    cluster_size: int,
    labels_in_use: set[str],
) -> ClusterNarrative:
    attack, defense, results, control, volatility = (
        profile["attack"],
        profile["defense"],
        profile["results"],
        profile["control"],
        profile["volatility"],
    )

    if cluster_size == 1:
        base = ClusterNarrative(
            label="Outlier underperforming team",
            description="A singleton profile separated from all other teams.",
            football_interpretation="This team behaves differently from the league pattern and needs individual diagnosis.",
            strengths=["Unique tactical footprint"],
            weaknesses=["Low peer similarity", "Potential structural instability"],
        )
        if base.label in labels_in_use:
            return ClusterNarrative(
                label=f"{base.label} (isolated case {len(labels_in_use)+1})",
                description=base.description,
                football_interpretation=base.football_interpretation,
                strengths=base.strengths,
                weaknesses=base.weaknesses,
            )
        return base

    if cluster_size == 2:
        if attack > 0.35 and results > 0.12:
            base = ClusterNarrative(
                label="Elite attacking giants",
                description="A compact pair of top-end attacking teams.",
                football_interpretation="Two clubs with high scoring output and strong winning profile.",
                strengths=["Explosive attack", "High points efficiency"],
                weaknesses=["Small sample for generalization"],
            )
        elif defense > 0.25:
            base = ClusterNarrative(
                label="Defensive elite contenders",
                description="A compact pair defined by defensive control.",
                football_interpretation="Two teams that suppress opponent output and manage margins well.",
                strengths=["Defensive structure", "Game-state control"],
                weaknesses=["Can rely on narrow scorelines"],
            )
        else:
            base = ClusterNarrative(
                label="Niche tactical pair",
                description="A very small cluster with uncommon style indicators.",
                football_interpretation="Two teams share a narrow tactical profile not widely replicated in the league.",
                strengths=["Clear stylistic identity"],
                weaknesses=["Low representativeness", "Sensitive to form swings"],
            )
        if base.label in labels_in_use:
            return ClusterNarrative(
                label=f"{base.label} (pair variant {len(labels_in_use)+1})",
                description=base.description,
                football_interpretation=base.football_interpretation,
                strengths=base.strengths,
                weaknesses=base.weaknesses,
            )
        return base

    if attack > 0.2 and results > 0.08:
        base = ClusterNarrative(
            label="Elite attacking giants",
            description="Top-end scoring profile with strong winning efficiency.",
            football_interpretation="High chance creation and conversion; these teams push match tempo and dominate scorelines.",
            strengths=["Scoring power", "Winning consistency", "High attacking upside"],
            weaknesses=["Can be vulnerable in transition if over-committing forward"],
        )
    elif defense > 0.2 and results > 0.03:
        base = ClusterNarrative(
            label="Defensive elite contenders",
            description="Concede less than league average while maintaining positive results.",
            football_interpretation="Structured defensive blocks, disciplined spacing, and efficient game management.",
            strengths=["Defensive solidity", "Match control", "Reliable point accumulation"],
            weaknesses=["Lower scoring explosiveness in open games"],
        )
    elif control > 0.08 and abs(attack) < 0.12 and abs(defense) < 0.12:
        base = ClusterNarrative(
            label="Balanced mid-table teams",
            description="Stable but moderate profile close to league baseline.",
            football_interpretation="Competitive and organized teams that trade risk for consistency.",
            strengths=["Balance", "Low collapse risk", "Tactical flexibility"],
            weaknesses=["Limited elite ceiling", "Can struggle to break top blocks"],
        )
    elif volatility > 0.15:
        base = ClusterNarrative(
            label="High-variance transition teams",
            description="Large match-to-match performance swings.",
            football_interpretation="Aggressive styles or unstable phases produce both big wins and heavy drops.",
            strengths=["Upset potential", "Explosive periods"],
            weaknesses=["Inconsistent outcomes", "Harder to forecast"],
        )
    else:
        base = ClusterNarrative(
            label="Competitive mixed-profile teams",
            description="Intermediate profile with mixed strengths and trade-offs.",
            football_interpretation="Capable teams with no single dominant tactical axis.",
            strengths=["Adaptability", "Moderate stability"],
            weaknesses=["No standout competitive edge"],
        )

    # Keep labels unique in case multiple clusters fit the same narrative.
    if base.label in labels_in_use:
        return ClusterNarrative(
            label=f"{base.label} (variant {len(labels_in_use) + 1})",
            description=base.description,
            football_interpretation=base.football_interpretation,
            strengths=base.strengths,
            weaknesses=base.weaknesses,
        )
    return base


def _confidence_score(diff: pd.Series, cluster_size: int) -> float:
    separation = float(diff.abs().head(5).mean()) if not diff.empty else 0.0
    size_factor = min(1.0, cluster_size / 6.0)
    raw = 0.45 + 0.35 * np.tanh(2.0 * separation) + 0.20 * size_factor
    return float(round(max(0.0, min(0.99, raw)), 3))


def interpret_clusters(
    best_labels: pd.DataFrame,
    features_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    features = config["features"]
    df = features_df.copy().merge(best_labels, on="team_id", how="left")
    global_means = df[features].mean(numeric_only=True)

    cluster_info: dict[str, Any] = {}
    used_labels: set[str] = set()
    for cluster_id in sorted(df["cluster"].dropna().unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        cluster_size = int(cluster_df["team_id"].nunique())
        means = cluster_df[features].mean(numeric_only=True)
        diff = (means - global_means).sort_values(key=lambda s: s.abs(), ascending=False)
        top_features = diff.index.tolist()[:6]

        profile = _score_profile(means, global_means)
        narrative = _narrative_from_profile(profile, cluster_size, used_labels)
        used_labels.add(narrative.label)

        warning = None
        is_outlier_like = False
        if cluster_size <= 2:
            warning = "Small cluster size (<=2): treat interpretation as fragile."
            is_outlier_like = True
        if cluster_size == 1:
            warning = "outlier-like singleton cluster"
            is_outlier_like = True

        cluster_info[str(int(cluster_id))] = {
            "label": narrative.label,
            "description": narrative.description,
            "football_interpretation": narrative.football_interpretation,
            "strengths": narrative.strengths,
            "weaknesses": narrative.weaknesses,
            "n_teams": cluster_size,
            "teams": sorted(cluster_df["team_name"].dropna().astype(str).unique().tolist()),
            "top_differentiating_features": top_features,
            "confidence_score": _confidence_score(diff, cluster_size),
            "warning": warning,
            "is_outlier_like": is_outlier_like,
            "means": {k: float(v) for k, v in means.to_dict().items()},
            "diff_vs_global": {k: float(v) for k, v in diff.to_dict().items()},
        }
    return cluster_info
