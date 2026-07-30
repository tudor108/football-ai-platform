"""Human-readable responsible interpretation of clustering stability."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def write_stability_report(
    output_path: Path,
    run_id: str,
    coverage: Mapping[str, Any],
    best_summary: Mapping[str, Any],
    snapshots: Sequence[Mapping[str, Any]],
    ari_timeline: Sequence[Mapping[str, Any]],
    bootstrap_summary: Mapping[str, Any],
    assignments: pd.DataFrame,
    transitions: pd.DataFrame,
    season_comparison: Mapping[str, Any],
    alerts: Sequence[Mapping[str, Any]],
) -> None:
    """Write a Markdown report with metric definitions and explicit limitations."""
    metric = best_summary.get("metric", {})
    low_assignments = int((assignments.get("raw_assignment_strength", pd.Series(dtype=float)) < 0.4).sum())
    changed = int(transitions.get("transition_detected", pd.Series(dtype=bool)).fillna(False).sum())
    lines = [
        "# Cluster Stability Report",
        "",
        "## 1. Executive summary",
        f"Run `{run_id}` contains a versioned clustering and stability analysis.",
        f"The selected candidate is `{best_summary.get('candidate_id')}` using `{best_summary.get('algorithm')}`.",
        f"Generated temporal snapshots: {len(snapshots)}. Detected team transitions: {changed}.",
        "",
        "## 2. Run identity and data coverage",
        f"- Run ID: `{run_id}`",
        f"- Data as of: `{coverage.get('data_as_of_date')}`",
        f"- Matches: {coverage.get('match_count')}",
        f"- Teams: {coverage.get('team_count')}",
        f"- Leagues: {coverage.get('league_ids')}",
        f"- Seasons: {coverage.get('seasons')}",
        "",
        "## 3. Current clustering quality",
        f"- Silhouette: {metric.get('silhouette')}",
        f"- Davies-Bouldin: {metric.get('davies_bouldin')}",
        f"- Calinski-Harabasz: {metric.get('calinski_harabasz')}",
        f"- Composite: {metric.get('composite')}",
        "Quality describes separation/compactness; it does not measure temporal stability.",
        "",
        "## 4. Cluster-size warnings",
    ]
    size_alerts = [alert for alert in alerts if alert.get("alert_type") in {"SINGLETON_CLUSTER", "SMALL_CLUSTER"}]
    lines.extend([f"- {alert['message']} Evidence: `{alert['evidence']}`" for alert in size_alerts] or ["- No configured small-cluster alert."])
    lines.extend(["", "## 5. ARI between snapshots"])
    for comparison in ari_timeline:
        lines.append(
            f"- `{comparison.get('previous_run_id')}` → `{comparison.get('current_run_id')}`: "
            f"ARI={comparison.get('adjusted_rand_index')}, status={comparison.get('comparison_status')}"
        )
    if not ari_timeline:
        lines.append("- Not enough snapshots or historical runs for comparison.")
    lines.extend(
        [
            "",
            "## 6. Bootstrap stability summary",
            f"- Requested/valid/invalid: {bootstrap_summary.get('requested_iterations')} / {bootstrap_summary.get('valid_iterations')} / {bootstrap_summary.get('invalid_iterations')}",
            f"- Mean ARI: {bootstrap_summary.get('mean_ari')}",
            f"- Median ARI: {bootstrap_summary.get('median_ari')}",
            "Bootstrap stability is consistency under fixture resampling, not a probability of correctness.",
            "",
            "## 7. Team assignment-strength summary",
            f"Teams below the initial 0.40 review threshold: {low_assignments}.",
            "Assignment strength is a bounded distance margin, not a calibrated probability.",
            "",
            "## 8. Important team transitions",
        ]
    )
    important = transitions.loc[transitions.get("transition_detected", False) == True] if not transitions.empty else transitions  # noqa: E712
    for _, row in important.head(15).iterrows():
        lines.append(
            f"- {row.get('team_name')}: {row.get('previous_stable_cluster_id')} → {row.get('current_stable_cluster_id')}; "
            f"assignment strength={row.get('current_assignment_strength')}, bootstrap={row.get('bootstrap_stability')}."
        )
    if important.empty:
        lines.append("- No supported stable-ID transition in the available comparison.")
    lines.extend(
        [
            "",
            "## 9. Season comparison",
            f"Status: `{season_comparison.get('status')}`. {season_comparison.get('reason', '')}",
            "",
            "## 10. Business alerts",
        ]
    )
    lines.extend([f"- **{alert['severity']} {alert['alert_type']}** — {alert['message']}" for alert in alerts] or ["- No alerts."])
    lines.extend(
        [
            "",
            "## 11. Limitations and responsible interpretation",
            "- Cluster IDs require alignment because raw labels are arbitrary.",
            "- ARI measures partition similarity and does not establish which partition is better.",
            "- A movement is not proof of a tactical or causal change.",
            "- Singleton and small-cluster assignment strength is explicitly unreliable.",
            "- Bootstrap results depend on fixture resampling and the available match sample.",
            "",
            "## 12. Metric definitions",
            "- **Silhouette**: cohesion versus separation in the preprocessed feature space.",
            "- **ARI**: label-permutation-invariant agreement between two partitions; 1 is identical, around 0 is chance-like, and values may be negative.",
            "- **Assignment strength**: `(nearest_other - own) / nearest_other`, clipped to `[0, 1]`; not a probability.",
            "- **Bootstrap stability**: fraction of valid resamples in which a team aligns to its reference stable cluster.",
            "- **Transition matrix**: counts and row proportions between aligned stable clusters.",
            "",
            "## 13. Five presentation-ready conclusions",
            f"1. Run `{run_id}` is reproducible through its manifest, hashes, and immutable artifact directory.",
            f"2. The selected model is `{best_summary.get('algorithm')}` with `{best_summary.get('params')}`.",
            f"3. The analysis generated {len(snapshots)} leakage-safe temporal snapshots.",
            f"4. Bootstrap used {bootstrap_summary.get('valid_iterations')} valid fixture resamples; uncertainty is reported explicitly.",
            f"5. There are {len(alerts)} structured review alerts; none should be interpreted causally.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
