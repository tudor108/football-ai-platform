"""MLflow tracking utilities for the clustering pipeline."""

from __future__ import annotations

import os
from typing import Any

import mlflow


def mlflow_run(config: dict[str, Any], best: dict[str, Any], metrics: dict[str, Any]) -> None:
    experiment = config.get("mlflow", {}).get("experiment", "team_clustering")
    mlflow.set_experiment(experiment)

    run_name = f"{best['algorithm']}_{best['candidate_id']}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("algorithm", best["algorithm"])
        mlflow.log_param("candidate_id", best["candidate_id"])
        for k, v in best["params"].items():
            mlflow.log_param(f"param_{k}", v)
        mlflow.log_param("features", ",".join(config["features"]))

        best_metric = best["metric"]
        for key in ("silhouette", "davies_bouldin", "calinski_harabasz", "composite", "noise_ratio", "balance_ratio"):
            value = best_metric.get(key)
            if value is not None:
                mlflow.log_metric(key, float(value))

        clusters_path = os.path.join(config["output"]["clusters_dir"], "clusters.csv")
        metrics_path = os.path.join(config["output"]["metrics_dir"], "metrics.json")
        interpretation_path = os.path.join(config["output"]["metrics_dir"], "cluster_interpretation.json")
        for artifact in (clusters_path, metrics_path, interpretation_path):
            if os.path.exists(artifact):
                mlflow.log_artifact(artifact)

        plots_dir = config["output"]["plots_dir"]
        for plot_name in ("pca_scatter.png", "cluster_sizes.png"):
            plot_path = os.path.join(plots_dir, plot_name)
            if os.path.exists(plot_path):
                mlflow.log_artifact(plot_path)

        config_path = config.get("config_path")
        if config_path and os.path.exists(config_path):
            mlflow.log_artifact(config_path)
