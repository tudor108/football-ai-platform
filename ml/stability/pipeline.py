"""Orchestrate all stability analyses for one versioned clustering run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from business.cluster_stability_rules import generate_stability_alerts
from ml.stability.ari_analysis import compare_partitions
from ml.stability.bootstrap_stability import run_bootstrap_stability
from ml.stability.cluster_distances import calculate_assignment_strength
from ml.stability.historical_snapshots import build_historical_snapshots
from ml.stability.label_alignment import AlignmentResult, align_cluster_labels, apply_stable_ids
from ml.stability.season_comparison import compare_seasons
from ml.stability.stability_report import write_stability_report
from ml.stability.transition_matrix import (
    build_team_transitions,
    build_transition_matrix,
    plot_transition_matrix,
    write_transition_artifacts,
)
from ml.utils.artifact_resolver import ResolutionStatus, find_previous_successful_run
from ml.utils.run_context import atomic_write_json


@dataclass(frozen=True)
class StabilityResult:
    current_clusters: pd.DataFrame
    artifacts: Mapping[str, Path]
    warnings: tuple[str, ...]
    summary: Mapping[str, Any]


def _load_previous(output_root: Path, current_run_id: str) -> tuple[pd.DataFrame | None, dict[str, Any], str | None]:
    resolution = find_previous_successful_run(output_root, current_run_id)
    if resolution.status != ResolutionStatus.VERSIONED_RUN or resolution.root is None:
        return None, {}, None
    clusters_path = resolution.artifacts.get("clusters")
    if clusters_path is None:
        return None, {}, None
    previous = pd.read_csv(clusters_path, encoding="utf-8-sig")
    if "raw_cluster_id" not in previous and "cluster" in previous:
        previous["raw_cluster_id"] = previous["cluster"]
    if "stable_cluster_id" not in previous:
        previous["stable_cluster_id"] = previous["raw_cluster_id"].map(lambda value: f"SC{int(value) + 1:03d}")
    manifest = json.loads((resolution.root / "manifest.json").read_text(encoding="utf-8"))
    return previous, manifest, resolution.run_id


def _plot_assignment_strength(assignments: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    data = assignments.sort_values("raw_assignment_strength")
    axis.bar(data["team_name"].astype(str), data["raw_assignment_strength"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Assignment strength (not probability)")
    axis.tick_params(axis="x", rotation=75)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_ari_timeline(comparisons: list[dict[str, Any]], path: Path) -> bool:
    usable = [item for item in comparisons if item.get("adjusted_rand_index") is not None]
    if not usable:
        return False
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(range(len(usable)), [item["adjusted_rand_index"] for item in usable], marker="o")
    axis.set_xticks(range(len(usable)), [item["current_run_id"] for item in usable], rotation=45, ha="right")
    axis.set_ylabel("Adjusted Rand Index")
    axis.set_title("Partition similarity timeline")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def run_stability_analysis(
    run_id: str,
    run_dir: Path,
    output_root: Path,
    match_df: pd.DataFrame,
    features_df: pd.DataFrame,
    current_labels: pd.DataFrame,
    processed_matrix: Any,
    config: dict[str, Any],
    best: dict[str, Any],
    coverage: Mapping[str, Any],
    smoke: bool = False,
) -> StabilityResult:
    """Run temporal, bootstrap, transition, season, alert, and report analyses."""
    stability_dir = run_dir / "stability"
    plots_dir = stability_dir / "plots"
    artifacts: dict[str, Path] = {}
    warnings: list[str] = []
    current = features_df.merge(current_labels[["team_id", "cluster"]], on="team_id", how="inner")
    current = current.rename(columns={"cluster": "raw_cluster_id"})

    snapshots = build_historical_snapshots(
        match_df,
        config,
        best["algorithm"],
        best["params"],
        stability_dir / "snapshots",
    )
    history_rows: list[pd.DataFrame] = []
    mappings: list[dict[str, Any]] = []
    ari_timeline: list[dict[str, Any]] = []
    previous_snapshot: pd.DataFrame | None = None
    previous_snapshot_id: str | None = None
    last_alignment: AlignmentResult | None = None
    rules = config.get("stability", {}).get("business_rules", {})
    min_cluster_size = int(rules.get("minimum_cluster_size", 3))
    alignment_cfg = config.get("stability", {}).get("label_alignment", {})
    for snapshot in snapshots:
        alignment = align_cluster_labels(
            previous_snapshot,
            snapshot.clusters,
            config["features"],
            float(alignment_cfg.get("membership_weight", 0.75)),
            float(alignment_cfg.get("feature_profile_weight", 0.25)),
        )
        aligned = apply_stable_ids(snapshot.clusters, alignment.mapping)
        assignments = calculate_assignment_strength(aligned, snapshot.processed_matrix, minimum_cluster_size=min_cluster_size)
        aligned = aligned.merge(
            assignments.drop(columns=["team_name", "raw_cluster_id", "stable_cluster_id"], errors="ignore"),
            on="team_id",
            how="left",
        )
        aligned.to_csv(snapshot.directory / "clusters.csv", index=False, encoding="utf-8-sig")
        history = aligned[["team_id", "team_name", "raw_cluster_id", "stable_cluster_id", "raw_assignment_strength"]].copy()
        history["snapshot_id"] = snapshot.snapshot_id
        history["cutoff_date"] = snapshot.cutoff_date.isoformat()
        history_rows.append(history)
        mappings.append({"snapshot_id": snapshot.snapshot_id, "mapping": alignment.mapping, "events": list(alignment.events)})
        if previous_snapshot is not None and previous_snapshot_id:
            ari_timeline.append(
                compare_partitions(
                    previous_snapshot,
                    aligned,
                    previous_snapshot_id,
                    snapshot.snapshot_id,
                    int(rules.get("minimum_common_teams_for_ari", 10)),
                )
            )
        previous_snapshot = aligned
        previous_snapshot_id = snapshot.snapshot_id
        last_alignment = alignment

    previous_run, previous_manifest, previous_run_id = _load_previous(output_root, run_id)
    alignment_reference = previous_run if previous_run is not None else previous_snapshot
    current_alignment = align_cluster_labels(
        alignment_reference,
        current,
        config["features"],
        float(alignment_cfg.get("membership_weight", 0.75)),
        float(alignment_cfg.get("feature_profile_weight", 0.25)),
    )
    current = apply_stable_ids(current, current_alignment.mapping)
    current_assignments = calculate_assignment_strength(current, processed_matrix, minimum_cluster_size=min_cluster_size)
    current = current.merge(
        current_assignments.drop(columns=["team_name", "raw_cluster_id", "stable_cluster_id"], errors="ignore"),
        on="team_id",
        how="left",
    )
    mappings.append({"run_id": run_id, "mapping": current_alignment.mapping, "events": list(current_alignment.events)})

    comparison_reference = previous_run if previous_run is not None else previous_snapshot
    comparison_reference_id = previous_run_id or previous_snapshot_id
    current_metadata = {
        "feature_schema_hash": coverage.get("feature_schema_hash"),
        "preprocessing_hash": coverage.get("preprocessing_hash"),
        "algorithm": best["algorithm"],
        "selected_parameters": best["params"],
    }
    if comparison_reference is not None and comparison_reference_id:
        run_comparison = compare_partitions(
            comparison_reference,
            current,
            comparison_reference_id,
            run_id,
            int(rules.get("minimum_common_teams_for_ari", 10)),
            previous_manifest,
            current_metadata,
        )
        ari_timeline.append(run_comparison)
    else:
        run_comparison = {
            "current_run_id": run_id,
            "previous_run_id": None,
            "adjusted_rand_index": None,
            "comparison_status": "NO_PREVIOUS_RUN",
            "common_team_count": 0,
            "compatibility_warnings": [],
        }

    bootstrap_df, bootstrap_summary, bootstrap_artifacts = run_bootstrap_stability(
        match_df,
        current,
        config,
        best["algorithm"],
        best["params"],
        stability_dir,
        smoke=smoke,
    )
    artifacts.update(bootstrap_artifacts)
    current = current.merge(bootstrap_df[["team_id", "bootstrap_stability"]], on="team_id", how="left")

    if comparison_reference is not None and comparison_reference_id:
        transitions = build_team_transitions(comparison_reference, current, comparison_reference_id, run_id, bootstrap_df)
    else:
        empty_reference = current.iloc[0:0].copy()
        transitions = build_team_transitions(empty_reference, current, "NONE", run_id, bootstrap_df)
    transition_matrix = build_transition_matrix(transitions)
    artifacts.update(write_transition_artifacts(transitions, transition_matrix, run_comparison, stability_dir))
    transition_plot = plots_dir / "transition_matrix.png"
    if plot_transition_matrix(transition_matrix, transition_plot):
        artifacts["transition_matrix_plot"] = transition_plot

    season_result, season_artifacts = compare_seasons(match_df, config, best["algorithm"], best["params"], stability_dir)
    artifacts.update(season_artifacts)
    alerts = generate_stability_alerts(
        run_id,
        current,
        run_comparison,
        bootstrap_summary,
        transitions,
        current_alignment.events,
        season_result,
        rules,
    )
    alerts_path = stability_dir / "alerts.json"
    atomic_write_json(alerts_path, {"run_id": run_id, "alerts": alerts})
    artifacts["alerts"] = alerts_path

    mappings_path = stability_dir / "label_mappings.json"
    atomic_write_json(mappings_path, {"mappings": mappings})
    artifacts["label_mappings"] = mappings_path
    history_path = stability_dir / "team_cluster_history.csv"
    history_df = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame(
        columns=["team_id", "team_name", "raw_cluster_id", "stable_cluster_id", "raw_assignment_strength", "snapshot_id", "cutoff_date"]
    )
    current_history = current[["team_id", "team_name", "raw_cluster_id", "stable_cluster_id", "raw_assignment_strength"]].copy()
    current_history["snapshot_id"] = run_id
    current_history["cutoff_date"] = coverage.get("data_as_of_date")
    history_df = pd.concat([history_df, current_history], ignore_index=True)
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    artifacts["team_cluster_history"] = history_path

    assignment_plot = plots_dir / "assignment_strength.png"
    _plot_assignment_strength(current, assignment_plot)
    artifacts["assignment_strength_plot"] = assignment_plot
    ari_plot = plots_dir / "ari_timeline.png"
    if _plot_ari_timeline(ari_timeline, ari_plot):
        artifacts["ari_timeline_plot"] = ari_plot

    report_path = stability_dir / "stability_report.md"
    write_stability_report(
        report_path,
        run_id,
        coverage,
        {"algorithm": best["algorithm"], "candidate_id": best["candidate_id"], "params": best["params"], "metric": best["metric"]},
        [snapshot.metadata for snapshot in snapshots],
        ari_timeline,
        bootstrap_summary,
        current,
        transitions,
        season_result,
        alerts,
    )
    artifacts["stability_report"] = report_path
    summary = {
        "snapshot_count": len(snapshots),
        "run_comparison": run_comparison,
        "bootstrap_summary": bootstrap_summary,
        "season_comparison": season_result,
        "alert_count": len(alerts),
    }
    return StabilityResult(current, artifacts, tuple(warnings), summary)
