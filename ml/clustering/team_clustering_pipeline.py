"""Team clustering pipeline with local-first feature source and model sweep."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.clustering.algorithms import run_clustering_algorithms
from ml.clustering.evaluation import evaluate_clustering_results
from ml.clustering.interpretation import interpret_clusters
from ml.clustering.visualization import visualize_clusters
from ml.reporting.cluster_reporting import write_cluster_report
from ml.features.build_team_features import (
    build_features,
    build_team_level_from_match_features,
    load_local_match_features,
)
from ml.utils.bigquery_client import get_team_features_from_bq, load_clusters_to_bq
from ml.utils.logging_utils import setup_logging
from ml.utils.mlflow_utils import mlflow_run
from ml.utils.validation import validate_config


def _ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in ("clusters_dir", "metrics_dir", "plots_dir", "report_dir"):
        out_dir = Path(config["output"][key])
        out_dir.mkdir(parents=True, exist_ok=True)


def _load_source_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    source = config["input"].get("source", "local_derived")
    if source == "bigquery":
        raw = get_team_features_from_bq(config)
        return raw
    if source == "csv":
        path = config["input"].get("path")
        if not path:
            raise ValueError("input.path is required when input.source=csv")
        return pd.read_csv(path, encoding="utf-8-sig")

    local_path = config["input"].get("path", "output/derived/fact_match_features.csv")
    match_df = load_local_match_features(local_path)
    return build_team_level_from_match_features(match_df)


def main(config_path: str) -> None:
    setup_logging()
    logger = logging.getLogger("team_clustering_pipeline")

    logger.info("Loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["config_path"] = config_path
    validate_config(config)
    _ensure_output_dirs(config)

    raw_df = _load_source_dataframe(config)
    features_df = build_features(raw_df, config)
    if "team_id" not in features_df.columns:
        raise ValueError("Feature source must contain team_id")
    if "team_name" not in features_df.columns:
        features_df["team_name"] = features_df["team_id"].astype(str)

    run_results = run_clustering_algorithms(features_df, config)
    metrics, best = evaluate_clustering_results(run_results, config)

    x_proc = run_results["X_proc"]
    pca_x = x_proc[:, 0]
    pca_y = x_proc[:, 1] if x_proc.shape[1] > 1 else [0.0] * len(pca_x)

    best_labels = pd.DataFrame(
        {
            "team_id": features_df["team_id"].values,
            "team_name": features_df["team_name"].values,
            "cluster": best["labels"],
            "best_algorithm": best["algorithm"],
            "candidate_id": best["candidate_id"],
            "pca_x": pca_x,
            "pca_y": pca_y,
        }
    )

    interpretations = interpret_clusters(best_labels[["team_id", "cluster"]], features_df, config)

    # Enrich cluster-level semantics into team-level output.
    enrich_rows: list[dict[str, Any]] = []
    for cid, info in interpretations.items():
        diff_items = list(info["diff_vs_global"].items())[:5]
        row = {
            "cluster": int(cid),
            "cluster_label": info["label"],
            "cluster_description": info["description"],
            "strengths": "; ".join(info["strengths"]),
            "weaknesses": "; ".join(info["weaknesses"]),
            "is_outlier_like": info["is_outlier_like"],
            "cluster_warning": info.get("warning"),
            "cluster_confidence_score": info["confidence_score"],
        }
        for idx in range(5):
            if idx < len(diff_items):
                feat_name, feat_value = diff_items[idx]
                row[f"feat_{idx+1}_name"] = feat_name
                row[f"feat_{idx+1}_value"] = abs(float(feat_value))
            else:
                row[f"feat_{idx+1}_name"] = None
                row[f"feat_{idx+1}_value"] = None
        enrich_rows.append(row)
    enrich_df = pd.DataFrame(enrich_rows)
    best_labels = best_labels.merge(enrich_df, on="cluster", how="left")

    try:
        visualize_clusters(
            best_labels=best_labels,
            features_df=features_df,
            features=config["features"],
            out_dir=config["output"]["plots_dir"],
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("Visualization failed: %s", error)

    clusters_path = os.path.join(config["output"]["clusters_dir"], "clusters.csv")
    metrics_path = os.path.join(config["output"]["metrics_dir"], "metrics.json")
    interpretation_path = os.path.join(config["output"]["metrics_dir"], "cluster_interpretation.json")
    summary_path = os.path.join(config["output"]["metrics_dir"], "best_model_summary.json")
    report_path = os.path.join(config["output"]["report_dir"], "cluster_report.md")

    best_labels.to_csv(clusters_path, index=False, encoding="utf-8-sig")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(interpretation_path, "w", encoding="utf-8") as f:
        json.dump(interpretations, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "algorithm": best["algorithm"],
                "candidate_id": best["candidate_id"],
                "params": best["params"],
                "metric": best["metric"],
            },
            f,
            indent=2,
        )
    write_cluster_report(
        output_report_path=report_path,
        best_summary={
            "algorithm": best["algorithm"],
            "candidate_id": best["candidate_id"],
            "params": best["params"],
            "metric": best["metric"],
        },
        metrics=metrics,
        interpretations=interpretations,
    )

    if config.get("mlflow", {}).get("enabled", False):
        mlflow_run(config, best, metrics)

    if config.get("output", {}).get("load_to_bigquery", False):
        load_clusters_to_bq(clusters_path, config)

    logger.info("Pipeline completed successfully. Best candidate: %s (%s)", best["candidate_id"], best["algorithm"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Team Clustering Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    main(args.config)
