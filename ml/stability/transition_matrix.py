"""Team cluster transitions and stable-ID transition matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_team_transitions(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    previous_id: str,
    current_id: str,
    bootstrap_stability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one transition record per team present in either partition."""
    columns = ["team_id", "team_name", "raw_cluster_id", "stable_cluster_id", "cluster_label", "raw_assignment_strength"]
    prev = previous[[column for column in columns if column in previous]].copy()
    curr = current[[column for column in columns if column in current]].copy()
    merged = prev.merge(curr, on="team_id", how="outer", suffixes=("_previous", "_current"))
    if bootstrap_stability is not None and not bootstrap_stability.empty:
        merged = merged.merge(bootstrap_stability[["team_id", "bootstrap_stability"]], on="team_id", how="left")
    else:
        merged["bootstrap_stability"] = np.nan
    merged["team_name"] = merged.get("team_name_current").fillna(merged.get("team_name_previous"))
    merged["transition_detected"] = (
        merged.get("stable_cluster_id_previous").notna()
        & merged.get("stable_cluster_id_current").notna()
        & (merged.get("stable_cluster_id_previous") != merged.get("stable_cluster_id_current"))
    )

    def classify(row: pd.Series) -> str:
        if pd.isna(row.get("stable_cluster_id_previous")):
            return "NEW_TEAM"
        if pd.isna(row.get("stable_cluster_id_current")):
            return "TEAM_NOT_PRESENT"
        return "CLUSTER_CHANGED" if bool(row["transition_detected"]) else "STABLE_PROFILE"

    merged["transition_classification"] = merged.apply(classify, axis=1)
    merged["supporting_evidence"] = merged.apply(
        lambda row: json.dumps(
            {
                "current_assignment_strength": row.get("raw_assignment_strength_current"),
                "bootstrap_stability": row.get("bootstrap_stability"),
            },
            default=str,
        ),
        axis=1,
    )
    merged["previous_snapshot_or_run"] = previous_id
    merged["current_snapshot_or_run"] = current_id
    rename = {
        "raw_cluster_id_previous": "previous_raw_cluster_id",
        "raw_cluster_id_current": "current_raw_cluster_id",
        "stable_cluster_id_previous": "previous_stable_cluster_id",
        "stable_cluster_id_current": "current_stable_cluster_id",
        "cluster_label_previous": "previous_cluster_label",
        "cluster_label_current": "current_cluster_label",
        "raw_assignment_strength_previous": "previous_assignment_strength",
        "raw_assignment_strength_current": "current_assignment_strength",
    }
    return merged.rename(columns=rename)[
        [
            "team_id",
            "team_name",
            "previous_snapshot_or_run",
            "current_snapshot_or_run",
            "previous_raw_cluster_id",
            "current_raw_cluster_id",
            "previous_stable_cluster_id",
            "current_stable_cluster_id",
            "previous_cluster_label",
            "current_cluster_label",
            "previous_assignment_strength",
            "current_assignment_strength",
            "bootstrap_stability",
            "transition_detected",
            "transition_classification",
            "supporting_evidence",
        ]
    ]


def build_transition_matrix(transitions: pd.DataFrame) -> pd.DataFrame:
    """Return raw counts and row-normalized proportions for stable IDs."""
    valid = transitions.dropna(subset=["previous_stable_cluster_id", "current_stable_cluster_id"])
    if valid.empty:
        return pd.DataFrame(
            columns=["previous_stable_cluster_id", "current_stable_cluster_id", "count", "row_proportion"]
        )
    counts = (
        valid.groupby(["previous_stable_cluster_id", "current_stable_cluster_id"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = counts.groupby("previous_stable_cluster_id")["count"].transform("sum")
    counts["row_proportion"] = counts["count"] / totals
    labels = valid.groupby("previous_stable_cluster_id")["previous_cluster_label"].first().to_dict()
    current_labels = valid.groupby("current_stable_cluster_id")["current_cluster_label"].first().to_dict()
    counts["previous_cluster_label"] = counts["previous_stable_cluster_id"].map(labels)
    counts["current_cluster_label"] = counts["current_stable_cluster_id"].map(current_labels)
    return counts


def plot_transition_matrix(matrix: pd.DataFrame, output_path: Path) -> bool:
    """Create a readable count heatmap; return False when no data exists."""
    if matrix.empty:
        return False
    pivot = matrix.pivot(index="previous_stable_cluster_id", columns="current_stable_cluster_id", values="count").fillna(0)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(pivot.values, cmap="Blues")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_xlabel("Current stable cluster")
    axis.set_ylabel("Previous stable cluster")
    axis.set_title("Team transition matrix")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            axis.text(column, row, str(int(pivot.iloc[row, column])), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Team count")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def write_transition_artifacts(
    transitions: pd.DataFrame,
    matrix: pd.DataFrame,
    comparison: dict[str, Any],
    stability_dir: Path,
) -> dict[str, Path]:
    """Write transition CSVs and comparison JSON."""
    transitions_path = stability_dir / "team_transitions.csv"
    matrix_path = stability_dir / "transition_matrix.csv"
    comparison_path = stability_dir / "run_comparison.json"
    transitions.to_csv(transitions_path, index=False, encoding="utf-8-sig")
    matrix.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    comparison_path.write_text(json.dumps(comparison, indent=2, default=str) + "\n", encoding="utf-8")
    return {"team_transitions": transitions_path, "transition_matrix": matrix_path, "run_comparison": comparison_path}
