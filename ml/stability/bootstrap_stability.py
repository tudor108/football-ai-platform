"""Fixture-level bootstrap stability analysis for clustering."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ml.clustering.algorithms import _fit_predict, _preprocess_matrix
from ml.features.build_team_features import build_features, build_team_level_from_match_features
from ml.stability.label_alignment import align_cluster_labels
from ml.utils.run_context import atomic_write_json


def run_bootstrap_stability(
    match_df: pd.DataFrame,
    reference_clusters: pd.DataFrame,
    config: dict[str, Any],
    selected_algorithm: str,
    selected_parameters: dict[str, Any],
    stability_dir: Path,
    smoke: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """Resample fixture IDs and measure partition and membership consistency."""
    bootstrap_cfg = config.get("stability", {}).get("bootstrap", {})
    requested = int(bootstrap_cfg.get("smoke_iterations", 10) if smoke else bootstrap_cfg.get("iterations", 100))
    seed = int(bootstrap_cfg.get("random_state", 42))
    coverage_min = float(bootstrap_cfg.get("minimum_team_coverage_ratio", 0.8))
    rng = np.random.default_rng(seed)
    fixture_ids = match_df["fixture_id"].dropna().unique()
    reference = reference_clusters.copy()
    reference_ids = set(reference["team_id"].astype(str))
    same_counts: Counter[str] = Counter()
    present_counts: Counter[str] = Counter()
    alternatives: dict[str, Counter[str]] = defaultdict(Counter)
    aris: list[float] = []
    invalid_reasons: Counter[str] = Counter()
    coverages: list[float] = []

    for iteration in range(requested):
        try:
            sampled = rng.choice(fixture_ids, size=len(fixture_ids), replace=True)
            pieces: list[pd.DataFrame] = []
            for draw_index, fixture_id in enumerate(sampled):
                rows = match_df.loc[match_df["fixture_id"] == fixture_id].copy()
                rows["fixture_id"] = rows["fixture_id"].astype(str) + f"__bootstrap_{iteration}_{draw_index}"
                pieces.append(rows)
            bootstrap_matches = pd.concat(pieces, ignore_index=True)
            team_features = build_team_level_from_match_features(bootstrap_matches)
            features = build_features(team_features, config)
            coverage = len(set(features["team_id"].astype(str)) & reference_ids) / max(len(reference_ids), 1)
            coverages.append(coverage)
            if coverage < coverage_min:
                raise ValueError("TEAM_COVERAGE_BELOW_THRESHOLD")
            matrix = _preprocess_matrix(features, config)
            _, labels = _fit_predict(selected_algorithm, selected_parameters, matrix)
            current = features.copy()
            current["raw_cluster_id"] = np.asarray(labels, dtype=int)
            alignment = align_cluster_labels(
                reference,
                current,
                config["features"],
                membership_weight=float(config.get("stability", {}).get("label_alignment", {}).get("membership_weight", 0.75)),
                feature_profile_weight=float(
                    config.get("stability", {}).get("label_alignment", {}).get("feature_profile_weight", 0.25)
                ),
            )
            current["stable_cluster_id"] = current["raw_cluster_id"].map(alignment.mapping)
            common = reference[["team_id", "raw_cluster_id", "stable_cluster_id"]].merge(
                current[["team_id", "raw_cluster_id", "stable_cluster_id"]], on="team_id", suffixes=("_reference", "_bootstrap")
            )
            if len(common) < 2:
                raise ValueError("INSUFFICIENT_COMMON_TEAMS")
            aris.append(float(adjusted_rand_score(common["raw_cluster_id_reference"], common["raw_cluster_id_bootstrap"])))
            for _, row in common.iterrows():
                key = str(row["team_id"])
                present_counts[key] += 1
                assigned = str(row["stable_cluster_id_bootstrap"])
                reference_stable = str(row["stable_cluster_id_reference"])
                if assigned == reference_stable:
                    same_counts[key] += 1
                else:
                    alternatives[key][assigned] += 1
        except Exception as error:  # noqa: BLE001 - invalid iterations are data
            invalid_reasons[str(error) or error.__class__.__name__] += 1

    team_records: list[dict[str, Any]] = []
    for _, row in reference.iterrows():
        key = str(row["team_id"])
        alternative, alternative_count = alternatives[key].most_common(1)[0] if alternatives[key] else (None, 0)
        present = present_counts[key]
        team_records.append(
            {
                "team_id": row["team_id"],
                "team_name": row.get("team_name"),
                "reference_stable_cluster_id": row.get("stable_cluster_id"),
                "valid_iterations_present": present,
                "same_reference_cluster_count": same_counts[key],
                "bootstrap_stability": same_counts[key] / present if present else None,
                "most_common_alternative_cluster": alternative,
                "alternative_cluster_frequency": alternative_count / present if present else 0.0,
            }
        )
    team_df = pd.DataFrame(team_records)
    values = np.asarray(aris, dtype=float)
    summary = {
        "requested_iterations": requested,
        "valid_iterations": int(len(aris)),
        "invalid_iterations": int(requested - len(aris)),
        "mean_ari": float(values.mean()) if len(values) else None,
        "std_ari": float(values.std()) if len(values) else None,
        "median_ari": float(np.median(values)) if len(values) else None,
        "p05_ari": float(np.percentile(values, 5)) if len(values) else None,
        "p95_ari": float(np.percentile(values, 95)) if len(values) else None,
        "minimum_team_coverage": float(min(coverages)) if coverages else None,
        "invalid_iteration_reasons": dict(invalid_reasons),
        "interpretation_note": "Bootstrap stability is membership consistency under fixture resampling, not a probability of correctness.",
    }
    stability_dir.mkdir(parents=True, exist_ok=True)
    team_path = stability_dir / "bootstrap_team_stability.csv"
    summary_path = stability_dir / "bootstrap_summary.json"
    plot_path = stability_dir / "plots" / "bootstrap_stability.png"
    team_df.to_csv(team_path, index=False, encoding="utf-8-sig")
    atomic_write_json(summary_path, summary)
    figure, axis = plt.subplots(figsize=(9, 5))
    plot_df = team_df.dropna(subset=["bootstrap_stability"]).sort_values("bootstrap_stability")
    axis.bar(plot_df["team_name"].astype(str), plot_df["bootstrap_stability"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Bootstrap membership stability")
    axis.set_title("Team bootstrap stability (not a probability)")
    axis.tick_params(axis="x", rotation=75)
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return team_df, summary, {"bootstrap_team_stability": team_path, "bootstrap_summary": summary_path, "bootstrap_stability_plot": plot_path}
