"""Reporting layer for human-readable clustering outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _score_explainer() -> str:
    return (
        "- `Silhouette`: higher is better, measures separation vs cohesion.\n"
        "- `Davies-Bouldin`: lower is better, measures average cluster overlap.\n"
        "- `Calinski-Harabasz`: higher is better, ratio of between/within cluster dispersion.\n"
        "- `Composite`: weighted score used in this project for robust model selection."
    )


def _recommend_3_vs_4(metrics: dict[str, Any]) -> str:
    score3 = -1e9
    score4 = -1e9
    for m in metrics.values():
        if not isinstance(m, dict):
            continue
        if "n_clusters" not in m or "composite" not in m:
            continue
        n_clusters = m["n_clusters"]
        comp = m.get("composite", -1e9)
        if n_clusters == 3:
            score3 = max(score3, comp)
        if n_clusters == 4:
            score4 = max(score4, comp)
    if score4 > score3:
        return "Recommendation: 4 clusters currently capture structure better than 3."
    if score3 > score4:
        return "Recommendation: 3 clusters are currently more stable/compact than 4."
    return "Recommendation: 3 and 4 clusters are similar; keep 4 for richer segmentation unless governance requires simplicity."


def _cluster_balance_line(best_metric: dict[str, Any]) -> str:
    sizes = best_metric.get("cluster_sizes", {})
    if not sizes:
        return "Cluster balance: n/a"
    values = np.array(list(sizes.values()), dtype=float)
    ratio = float(values.min() / values.max()) if values.max() > 0 else 0.0
    return f"Cluster balance ratio (min/max): {ratio:.3f}. Sizes: {sizes}"


def write_cluster_report(
    output_report_path: str | Path,
    best_summary: dict[str, Any],
    metrics: dict[str, Any],
    interpretations: dict[str, Any],
) -> None:
    out_path = Path(output_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_metric = best_summary["metric"]
    lines: list[str] = []
    lines.append("# Team Clustering Report")
    lines.append("")
    lines.append("## Model Selection")
    lines.append(
        f"Selected candidate: `{best_summary['candidate_id']}` using `{best_summary['algorithm']}` "
        f"with params `{best_summary['params']}`."
    )
    lines.append(
        "This model was selected because it achieved the strongest composite quality among valid candidates "
        "under configured cluster-count constraints."
    )
    lines.append("")
    lines.append("### Core Scores")
    lines.append(f"- Silhouette: {_fmt(best_metric.get('silhouette'))}")
    lines.append(f"- Davies-Bouldin: {_fmt(best_metric.get('davies_bouldin'))}")
    lines.append(f"- Calinski-Harabasz: {_fmt(best_metric.get('calinski_harabasz'))}")
    lines.append(f"- Composite: {_fmt(best_metric.get('composite'))}")
    lines.append("")
    lines.append("### Score Interpretation")
    lines.append(_score_explainer())
    lines.append("")
    lines.append("## Cluster-by-Cluster Interpretation")
    for cid in sorted(interpretations.keys(), key=lambda x: int(x)):
        cluster = interpretations[cid]
        lines.append(f"### Cluster {cid}: {cluster['label']}")
        lines.append(f"- Description: {cluster['description']}")
        lines.append(f"- Football interpretation: {cluster['football_interpretation']}")
        lines.append(f"- Teams ({cluster['n_teams']}): {', '.join(cluster['teams'])}")
        lines.append(f"- Strengths: {', '.join(cluster['strengths'])}")
        lines.append(f"- Weaknesses: {', '.join(cluster['weaknesses'])}")
        lines.append(f"- Top differentiating features: {', '.join(cluster['top_differentiating_features'])}")
        lines.append(f"- Confidence score: {_fmt(cluster['confidence_score'])}")
        if cluster.get("warning"):
            lines.append(f"- Warning: {cluster['warning']}")
        lines.append("")
    lines.append("## Outlier Analysis")
    outliers = [f"Cluster {cid}" for cid, c in interpretations.items() if c.get("is_outlier_like")]
    if outliers:
        lines.append(f"Outlier-like clusters detected: {', '.join(outliers)}.")
    else:
        lines.append("No outlier-like clusters detected.")
    lines.append("")
    lines.append("## Cluster Balance Analysis")
    lines.append(_cluster_balance_line(best_metric))
    lines.append("")
    lines.append("## Recommendation: 3 vs 4 Clusters")
    lines.append(_recommend_3_vs_4(metrics))
    lines.append("")
    lines.append("## Football & Business Insights")
    lines.append(
        "- Use elite clusters for benchmark strategy and scouting templates.\n"
        "- Use balanced/mid clusters for incremental optimization programs.\n"
        "- Track singleton/outlier clusters separately with case-by-case tactical review."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
