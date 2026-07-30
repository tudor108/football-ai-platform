"""Responsible, configurable alert rules for cluster stability evidence."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

import pandas as pd


def _alert(
    alert_type: str,
    severity: str,
    entity_type: str,
    entity_id: str,
    run_id: str,
    message: str,
    evidence: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    action: str,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    identity = f"{run_id}|{snapshot_id}|{alert_type}|{entity_type}|{entity_id}"
    return {
        "alert_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
        "alert_type": alert_type,
        "severity": severity,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "message": message,
        "evidence": dict(evidence),
        "thresholds_used": dict(thresholds),
        "recommended_review_action": action,
    }


def generate_stability_alerts(
    run_id: str,
    current_teams: pd.DataFrame,
    ari_comparison: Mapping[str, Any] | None,
    bootstrap_summary: Mapping[str, Any] | None,
    transitions: pd.DataFrame | None,
    alignment_events: Iterable[Mapping[str, Any]],
    season_comparison: Mapping[str, Any],
    rules: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply operational thresholds without claiming causality or probability."""
    alerts: list[dict[str, Any]] = []
    minimum_cluster_size = int(rules.get("minimum_cluster_size", 3))
    low_strength = float(rules.get("assignment_strength_low", 0.4))
    high_strength = float(rules.get("assignment_strength_high", 0.7))
    low_bootstrap = float(rules.get("bootstrap_stability_low", 0.6))
    high_bootstrap = float(rules.get("bootstrap_stability_high", 0.8))
    ari_stable = float(rules.get("ari_stable_min", 0.8))
    ari_review = float(rules.get("ari_review_min", 0.6))

    for cluster_id, group in current_teams.groupby("stable_cluster_id"):
        size = len(group)
        if size == 1:
            alerts.append(
                _alert("SINGLETON_CLUSTER", "HIGH", "CLUSTER", str(cluster_id), run_id, "Cluster singleton; assignment strength is not trustworthy.", {"cluster_size": size}, {"minimum_cluster_size": minimum_cluster_size}, "Review the team and feature coverage manually.")
            )
        elif size < minimum_cluster_size:
            alerts.append(
                _alert("SMALL_CLUSTER", "MEDIUM", "CLUSTER", str(cluster_id), run_id, "Cluster size is below the operational threshold.", {"cluster_size": size}, {"minimum_cluster_size": minimum_cluster_size}, "Treat cluster narratives as fragile.")
            )

    for _, team in current_teams.iterrows():
        strength = team.get("raw_assignment_strength")
        if pd.notna(strength) and float(strength) < low_strength:
            alerts.append(
                _alert("LOW_ASSIGNMENT_STRENGTH", "MEDIUM", "TEAM", str(team["team_id"]), run_id, "Team is close to another cluster boundary; this score is not a probability.", {"assignment_strength": float(strength)}, {"assignment_strength_low": low_strength}, "Review nearest-cluster features before drawing conclusions.")
            )
        bootstrap = team.get("bootstrap_stability")
        if pd.notna(bootstrap) and float(bootstrap) < low_bootstrap:
            alerts.append(
                _alert("LOW_BOOTSTRAP_STABILITY", "MEDIUM", "TEAM", str(team["team_id"]), run_id, "Team membership changes often under fixture resampling.", {"bootstrap_stability": float(bootstrap)}, {"bootstrap_stability_low": low_bootstrap}, "Review data coverage and boundary proximity.")
            )

    if ari_comparison:
        status = ari_comparison.get("comparison_status")
        ari = ari_comparison.get("adjusted_rand_index")
        if status == "METHOD_CHANGED":
            alerts.append(_alert("METHOD_CHANGED", "MEDIUM", "RUN", run_id, run_id, "Compared partitions used different methods; ARI is diagnostic only.", {"comparison_status": status}, {}, "Separate method change from data evolution."))
        if ari is not None and float(ari) < ari_review:
            alerts.append(_alert("LOW_GLOBAL_ARI", "HIGH", "RUN", run_id, run_id, "Global partition similarity is below the review threshold.", {"adjusted_rand_index": float(ari)}, {"ari_review_min": ari_review}, "Review method compatibility and team-level transitions."))
        elif ari is not None and float(ari) < ari_stable:
            alerts.append(_alert("ARI_REVIEW_REQUIRED", "MEDIUM", "RUN", run_id, run_id, "Partition similarity is below the operational stable threshold.", {"adjusted_rand_index": float(ari)}, {"ari_stable_min": ari_stable}, "Review transitions; do not infer causality."))

    if bootstrap_summary and bootstrap_summary.get("valid_iterations", 0) < bootstrap_summary.get("requested_iterations", 0):
        alerts.append(_alert("PARTIAL_BOOTSTRAP_RESULT", "MEDIUM", "RUN", run_id, run_id, "Some bootstrap iterations were invalid and are reported explicitly.", bootstrap_summary, {}, "Inspect invalid_iteration_reasons."))

    if transitions is not None:
        for _, transition in transitions.loc[transitions["transition_detected"] == True].iterrows():  # noqa: E712
            strength = transition.get("current_assignment_strength")
            bootstrap = transition.get("bootstrap_stability")
            if pd.notna(strength) and float(strength) < low_strength:
                kind, severity, message = "UNCERTAIN_TRANSITION", "MEDIUM", "A cluster transition exists, but current assignment strength is low."
            elif pd.notna(strength) and pd.notna(bootstrap) and float(strength) >= high_strength and float(bootstrap) >= high_bootstrap:
                kind, severity, message = "MEANINGFUL_PROFILE_SHIFT", "MEDIUM", "The available evidence is consistent with a profile shift; it does not establish a tactical cause."
            else:
                continue
            alerts.append(_alert(kind, severity, "TEAM", str(transition["team_id"]), run_id, message, {"assignment_strength": strength, "bootstrap_stability": bootstrap}, {"assignment_strength_low": low_strength, "assignment_strength_high": high_strength, "bootstrap_stability_high": high_bootstrap}, "Review feature changes and match coverage."))

    for event in alignment_events:
        if event.get("event_type") in {"POSSIBLE_SPLIT", "POSSIBLE_MERGE"}:
            kind = "POSSIBLE_CLUSTER_SPLIT" if event["event_type"] == "POSSIBLE_SPLIT" else "POSSIBLE_CLUSTER_MERGE"
            alerts.append(_alert(kind, "MEDIUM", "CLUSTER", str(event), run_id, "Membership overlap suggests a possible structural cluster change.", event, {}, "Inspect memberships and feature profiles; do not infer causality."))

    if season_comparison.get("status") == "NOT_AVAILABLE":
        alerts.append(_alert("SEASON_COMPARISON_UNAVAILABLE", "INFO", "RUN", run_id, run_id, str(season_comparison.get("reason")), {}, {}, "Add a second season before interpreting seasonal evolution."))
    return alerts
