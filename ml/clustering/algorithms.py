"""Clustering algorithm runner with parameter-grid search across multiple models."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


def _expand_grid(params: dict[str, Any]) -> list[dict[str, Any]]:
    if "param_grid" in params:
        grid = params["param_grid"]
        if isinstance(grid, list):
            return [item for item in grid if isinstance(item, dict)]
    scalar_params = {k: v for k, v in params.items() if k != "enabled"}
    list_keys = [k for k, v in scalar_params.items() if isinstance(v, list)]
    if not list_keys:
        return [scalar_params]
    fixed = {k: v for k, v in scalar_params.items() if k not in list_keys}
    combos: list[dict[str, Any]] = []
    for values in product(*[scalar_params[k] for k in list_keys]):
        candidate = fixed.copy()
        for key, value in zip(list_keys, values):
            candidate[key] = value
        combos.append(candidate)
    return combos


def _preprocess_matrix(features_df: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    feats = config["features"]
    X = features_df[feats].copy()

    scaler_type = config.get("preprocessing", {}).get("scaler", "standard")
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "none":
        scaler = None
    else:
        scaler = StandardScaler()

    X_values = X.to_numpy(dtype=float)
    if scaler is not None:
        X_values = scaler.fit_transform(X_values)

    if config.get("preprocessing", {}).get("pca", False):
        raw_components = int(config.get("preprocessing", {}).get("pca_components", 2))
        max_components = max(1, min(raw_components, X_values.shape[1], X_values.shape[0]))
        pca = PCA(n_components=max_components, random_state=config.get("random_state", 42))
        X_values = pca.fit_transform(X_values)
    return X_values


def _fit_predict(algo_name: str, algo_params: dict[str, Any], X_proc: np.ndarray) -> tuple[Any, np.ndarray]:
    if algo_name == "KMeans":
        model = KMeans(**algo_params)
        labels = model.fit_predict(X_proc)
    elif algo_name == "MiniBatchKMeans":
        model = MiniBatchKMeans(**algo_params)
        labels = model.fit_predict(X_proc)
    elif algo_name == "GaussianMixture":
        model = GaussianMixture(**algo_params)
        labels = model.fit_predict(X_proc)
    elif algo_name == "AgglomerativeClustering":
        model = AgglomerativeClustering(**algo_params)
        labels = model.fit_predict(X_proc)
    elif algo_name == "DBSCAN":
        model = DBSCAN(**algo_params)
        labels = model.fit_predict(X_proc)
    elif algo_name == "SpectralClustering":
        params = algo_params.copy()
        if "n_neighbors" in params:
            params["n_neighbors"] = max(2, min(int(params["n_neighbors"]), X_proc.shape[0] - 1))
        model = SpectralClustering(**params)
        labels = model.fit_predict(X_proc)
    elif algo_name == "Birch":
        model = Birch(**algo_params)
        labels = model.fit_predict(X_proc)
    elif algo_name == "MeanShift":
        model = MeanShift(**algo_params)
        labels = model.fit_predict(X_proc)
    elif algo_name == "HDBSCAN" and HDBSCAN_AVAILABLE:
        model = hdbscan.HDBSCAN(**algo_params)
        labels = model.fit_predict(X_proc)
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
    return model, labels


def run_clustering_algorithms(features_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Run all enabled clustering algorithms across parameter grids."""
    X_proc = _preprocess_matrix(features_df, config)
    candidates: list[dict[str, Any]] = []

    algorithms = config.get("algorithms", {})
    for algo_name, algo_cfg in algorithms.items():
        if not isinstance(algo_cfg, dict):
            continue
        if algo_cfg.get("enabled", True) is False:
            continue
        param_sets = _expand_grid(algo_cfg)
        for idx, params in enumerate(param_sets, start=1):
            try:
                model, labels = _fit_predict(algo_name, params, X_proc)
                candidates.append(
                    {
                        "candidate_id": f"{algo_name}#{idx}",
                        "algorithm": algo_name,
                        "params": params,
                        "model": model,
                        "labels": np.asarray(labels),
                    }
                )
            except Exception as error:  # noqa: BLE001 - keep the sweep running
                candidates.append(
                    {
                        "candidate_id": f"{algo_name}#{idx}",
                        "algorithm": algo_name,
                        "params": params,
                        "error": str(error),
                    }
                )

    return {"X_proc": X_proc, "features_df": features_df, "candidates": candidates}
