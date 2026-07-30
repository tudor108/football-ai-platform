"""MLflow tracking utilities for the clustering pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import mlflow
except ImportError:  # MLflow is optional when tracking is disabled.
    mlflow = None  # type: ignore[assignment]


def mlflow_run(
    config: dict[str, Any],
    best: dict[str, Any],
    metrics: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> None:
    if mlflow is None:
        raise RuntimeError("mlflow is required when mlflow.enabled=true")
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

        for artifact in artifact_paths.values():
            path = Path(artifact)
            if path.is_file():
                mlflow.log_artifact(str(path))

        config_path = config.get("config_path")
        if config_path and Path(config_path).is_file():
            mlflow.log_artifact(str(config_path))
