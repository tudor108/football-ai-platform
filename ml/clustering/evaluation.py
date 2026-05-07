"""Evaluation and best-model selection for clustering candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def _cluster_sizes(labels: np.ndarray) -> dict[str, int]:
    sizes = pd.Series(labels).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in sizes.items()}


def _evaluate_labels(X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    has_noise = np.any(labels == -1)
    valid_mask = labels != -1 if has_noise else np.ones(len(labels), dtype=bool)
    valid_labels = labels[valid_mask]
    valid_X = X[valid_mask]

    unique_valid = np.unique(valid_labels)
    n_clusters = int(len(unique_valid))
    n_noise = int(np.sum(labels == -1))
    noise_ratio = float(n_noise / len(labels))

    out: dict[str, Any] = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "cluster_sizes": _cluster_sizes(labels),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
        "balance_ratio": None,
    }

    if n_clusters < 2:
        return out

    valid_sizes = pd.Series(valid_labels).value_counts()
    out["balance_ratio"] = float(valid_sizes.min() / valid_sizes.max())
    out["silhouette"] = float(silhouette_score(valid_X, valid_labels))
    out["davies_bouldin"] = float(davies_bouldin_score(valid_X, valid_labels))
    out["calinski_harabasz"] = float(calinski_harabasz_score(valid_X, valid_labels))
    return out


def _composite_score(metric: dict[str, Any]) -> float:
    if metric["silhouette"] is None:
        return -1e9
    db_component = 1.0 / (1.0 + float(metric["davies_bouldin"]))
    ch_component = np.log1p(float(metric["calinski_harabasz"])) / 10.0
    noise_penalty = 1.0 - float(metric["noise_ratio"])
    balance = float(metric["balance_ratio"] or 0.0)
    return float(
        0.45 * float(metric["silhouette"])
        + 0.25 * db_component
        + 0.15 * ch_component
        + 0.10 * noise_penalty
        + 0.05 * balance
    )


def _selection_value(metric: dict[str, Any], select_by: str) -> float:
    if select_by == "davies_bouldin":
        return -float(metric["davies_bouldin"]) if metric["davies_bouldin"] is not None else -1e9
    if select_by == "calinski_harabasz":
        return float(metric["calinski_harabasz"]) if metric["calinski_harabasz"] is not None else -1e9
    if select_by == "composite":
        return _composite_score(metric)
    return float(metric["silhouette"]) if metric["silhouette"] is not None else -1e9


def evaluate_clustering_results(results: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate each candidate and choose the best according to config rule."""
    X = results["X_proc"]
    candidates = results.get("candidates", [])
    select_by = config.get("comparison", {}).get("select_by", "composite")
    min_clusters = int(config.get("comparison", {}).get("min_clusters", 2))
    max_clusters = int(config.get("comparison", {}).get("max_clusters", 8))

    metrics: dict[str, Any] = {}
    best_candidate: dict[str, Any] | None = None
    best_value = -1e9

    for candidate in candidates:
        cid = candidate["candidate_id"]
        if "error" in candidate:
            metrics[cid] = {"algorithm": candidate["algorithm"], "params": candidate["params"], "error": candidate["error"]}
            continue

        m = _evaluate_labels(X, candidate["labels"])
        m["algorithm"] = candidate["algorithm"]
        m["params"] = candidate["params"]
        m["composite"] = _composite_score(m)
        metrics[cid] = m

        if m["n_clusters"] < min_clusters or m["n_clusters"] > max_clusters:
            continue
        value = _selection_value(m, select_by)
        if value > best_value:
            best_value = value
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("No valid clustering candidate found. Check data size and parameter grids.")

    best = {
        "candidate_id": best_candidate["candidate_id"],
        "algorithm": best_candidate["algorithm"],
        "params": best_candidate["params"],
        "labels": best_candidate["labels"],
        "metric": metrics[best_candidate["candidate_id"]],
    }
    return metrics, best
