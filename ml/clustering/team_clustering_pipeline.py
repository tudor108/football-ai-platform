"""Team clustering pipeline with local-first feature source and model sweep."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.clustering.algorithms import run_clustering_algorithms
from ml.clustering.evaluation import evaluate_clustering_results
from ml.clustering.interpretation import interpret_clusters
from ml.reporting.cluster_reporting import write_cluster_report
from ml.features.build_team_features import (
    build_features,
    build_team_level_from_match_features,
    load_local_match_features,
)
from ml.utils.bigquery_client import get_team_features_from_bq, load_clusters_to_bq
from ml.utils.logging_utils import setup_logging
from ml.utils.mlflow_utils import mlflow_run
from ml.utils.pipeline_result import PipelineResult
from ml.utils.run_context import (
    SUCCEEDED,
    RunContext,
    atomic_write_json,
    feature_schema_hash,
    preprocessing_hash,
    sha256_file,
)
from ml.utils.validation import validate_config


def _ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in ("clusters_dir", "metrics_dir", "plots_dir", "report_dir"):
        out_dir = Path(config["output"][key])
        out_dir.mkdir(parents=True, exist_ok=True)


def _match_metadata(match_df: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(match_df.get("date"), errors="coerce", utc=True)
    latest = dates.max()
    return {
        "data_as_of_date": latest.isoformat() if pd.notna(latest) else None,
        "league_ids": sorted(str(value) for value in match_df.get("league_id", pd.Series(dtype=object)).dropna().unique()),
        "seasons": sorted(str(value) for value in match_df.get("season", pd.Series(dtype=object)).dropna().unique()),
        "match_count": int(match_df["fixture_id"].nunique()) if "fixture_id" in match_df else None,
    }


def _load_source_dataframe(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    source = config["input"].get("source", "local_derived")
    if source == "bigquery":
        raw = get_team_features_from_bq(config)
        reference = {"table": config["input"].get("table"), "query": config["input"].get("query")}
        metadata = {
            "input_source": "bigquery",
            "input_path_or_reference": reference,
            "input_hash": None,
            "input_hash_status": "NOT_AVAILABLE_FOR_NON_FILE_INPUT",
        }
        if {"fixture_id", "home_team_id", "away_team_id"}.issubset(raw.columns):
            metadata.update(_match_metadata(raw))
            return build_team_level_from_match_features(raw), raw, metadata
        return raw, None, metadata
    if source == "csv":
        path = config["input"].get("path")
        if not path:
            raise ValueError("input.path is required when input.source=csv")
        input_path = Path(path)
        raw = pd.read_csv(input_path, encoding="utf-8-sig")
        metadata = {
            "input_source": "csv",
            "input_path_or_reference": input_path.as_posix(),
            "input_hash": sha256_file(input_path),
            "input_hash_status": "AVAILABLE",
        }
        if {"fixture_id", "home_team_id", "away_team_id"}.issubset(raw.columns):
            metadata.update(_match_metadata(raw))
            return build_team_level_from_match_features(raw), raw, metadata
        return raw, None, metadata

    local_path = Path(config["input"].get("path", "output/derived/fact_match_features.csv"))
    match_df = load_local_match_features(local_path)
    metadata = {
        "input_source": "local_derived",
        "input_path_or_reference": local_path.as_posix(),
        "input_hash": sha256_file(local_path),
        "input_hash_status": "AVAILABLE",
        **_match_metadata(match_df),
    }
    return build_team_level_from_match_features(match_df), match_df, metadata


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_legacy_projection(output_root: Path, artifacts: dict[str, Path]) -> None:
    destinations = {
        "clusters": output_root / "clusters" / "clusters.csv",
        "metrics": output_root / "metrics" / "metrics.json",
        "best_model_summary": output_root / "metrics" / "best_model_summary.json",
        "cluster_interpretation": output_root / "metrics" / "cluster_interpretation.json",
        "cluster_report": output_root / "report" / "cluster_report.md",
    }
    for name, destination in destinations.items():
        _atomic_copy(artifacts[name], destination)
    for name, source in artifacts.items():
        if name.startswith("plot_") and source.is_file():
            _atomic_copy(source, output_root / "plots" / source.name)


def main(config_path: str, smoke: bool = False) -> PipelineResult:
    setup_logging()
    logger = logging.getLogger("team_clustering_pipeline")
    logger.info("Loading config from %s", config_path)
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["config_path"] = config_path
    validate_config(config)

    output_root = Path(config.get("output", {}).get("root_dir", "output/ml"))
    context = RunContext(
        output_root=output_root,
        config_path=Path(config_path),
        snapshot_mode=config.get("stability", {}).get("snapshot_mode"),
        repo_root=Path(__file__).resolve().parents[2],
    )
    artifacts: dict[str, Path] = {}
    warnings: list[str] = []
    try:
        config["output"].update(
            {
                "clusters_dir": str(context.run_dir / "clusters"),
                "metrics_dir": str(context.run_dir / "metrics"),
                "plots_dir": str(context.run_dir / "plots"),
                "report_dir": str(context.run_dir / "report"),
            }
        )
        source_df, match_df, input_metadata = _load_source_dataframe(config)
        context.update_metadata(**input_metadata)
        features_df = build_features(source_df, config)
        if "team_id" not in features_df.columns:
            raise ValueError("Feature source must contain team_id")
        if "team_name" not in features_df.columns:
            features_df["team_name"] = features_df["team_id"].astype(str)
        schema_hash = feature_schema_hash(config["features"], features_df.dtypes.to_dict())
        prep_hash = preprocessing_hash(config.get("preprocessing", {}))
        context.update_metadata(
            team_count=int(features_df["team_id"].nunique()),
            feature_schema_hash=schema_hash,
            preprocessing_hash=prep_hash,
        )

        run_results = run_clustering_algorithms(features_df, config)
        metrics, best = evaluate_clustering_results(run_results, config)
        context.update_metadata(
            algorithm=best["algorithm"],
            candidate_id=best["candidate_id"],
            selected_parameters=best["params"],
        )
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
        # Keep the official feature vector next to every assignment so
        # downstream read-only consumers can calculate personalized distances
        # without reconstructing or changing the canonical clustering run.
        profile_columns = [
            feature for feature in config["features"] if feature in features_df.columns
        ]
        best_labels = best_labels.merge(
            features_df[["team_id", *profile_columns]],
            on="team_id",
            how="left",
            validate="one_to_one",
        )
        interpretations = interpret_clusters(best_labels[["team_id", "cluster"]], features_df, config)

        enrich_rows: list[dict[str, Any]] = []
        for cid, info in interpretations.items():
            diff_items = list(info["diff_vs_global"].items())[:5]
            row: dict[str, Any] = {
                "cluster": int(cid),
                "cluster_label": info["label"],
                "cluster_description": info["description"],
                "strengths": "; ".join(info["strengths"]),
                "weaknesses": "; ".join(info["weaknesses"]),
                "is_outlier_like": info["is_outlier_like"],
                "cluster_warning": info.get("warning"),
                "cluster_confidence_score": info["confidence_score"],
            }
            for index in range(5):
                if index < len(diff_items):
                    feature_name, feature_value = diff_items[index]
                    row[f"feat_{index + 1}_name"] = feature_name
                    row[f"feat_{index + 1}_value"] = abs(float(feature_value))
                else:
                    row[f"feat_{index + 1}_name"] = None
                    row[f"feat_{index + 1}_value"] = None
            enrich_rows.append(row)
        best_labels = best_labels.merge(pd.DataFrame(enrich_rows), on="cluster", how="left")

        if config.get("stability", {}).get("enabled", True):
            if match_df is None:
                warnings.append("Stability analysis skipped because match-level input is unavailable.")
            else:
                coverage = {
                    **input_metadata,
                    "team_count": int(features_df["team_id"].nunique()),
                    "feature_schema_hash": schema_hash,
                    "preprocessing_hash": prep_hash,
                }
                from ml.stability.pipeline import run_stability_analysis

                stability = run_stability_analysis(
                    context.run_id,
                    context.run_dir,
                    output_root,
                    match_df,
                    features_df,
                    best_labels[["team_id", "cluster"]],
                    x_proc,
                    config,
                    best,
                    coverage,
                    smoke=smoke,
                )
                stability_columns = [
                    "team_id",
                    "raw_cluster_id",
                    "stable_cluster_id",
                    "cluster_size",
                    "distance_to_own_centroid",
                    "distance_to_nearest_other_centroid",
                    "nearest_other_cluster",
                    "separation_margin",
                    "raw_assignment_strength",
                    "assignment_strength_status",
                    "bootstrap_stability",
                ]
                best_labels = best_labels.merge(stability.current_clusters[stability_columns], on="team_id", how="left")
                artifacts.update(stability.artifacts)
                warnings.extend(stability.warnings)

        try:
            from ml.clustering.visualization import visualize_clusters

            visualize_clusters(
                best_labels=best_labels,
                features_df=features_df,
                features=config["features"],
                out_dir=config["output"]["plots_dir"],
            )
        except Exception as error:  # noqa: BLE001
            warning = f"Visualization failed: {error}"
            logger.warning(warning)
            warnings.append(warning)

        clusters_path = Path(config["output"]["clusters_dir"]) / "clusters.csv"
        metrics_path = Path(config["output"]["metrics_dir"]) / "metrics.json"
        interpretation_path = Path(config["output"]["metrics_dir"]) / "cluster_interpretation.json"
        summary_path = Path(config["output"]["metrics_dir"]) / "best_model_summary.json"
        report_path = Path(config["output"]["report_dir"]) / "cluster_report.md"
        best_labels.to_csv(clusters_path, index=False, encoding="utf-8-sig")
        atomic_write_json(metrics_path, metrics)
        atomic_write_json(interpretation_path, interpretations)
        best_summary = {
            "algorithm": best["algorithm"],
            "candidate_id": best["candidate_id"],
            "params": best["params"],
            "metric": best["metric"],
        }
        atomic_write_json(summary_path, best_summary)
        write_cluster_report(report_path, best_summary, metrics, interpretations)
        artifacts.update(
            {
                "clusters": clusters_path,
                "metrics": metrics_path,
                "cluster_interpretation": interpretation_path,
                "best_model_summary": summary_path,
                "cluster_report": report_path,
            }
        )
        for plot_path in Path(config["output"]["plots_dir"]).glob("*.png"):
            artifacts[f"plot_{plot_path.stem}"] = plot_path

        required_names = ("clusters", "metrics", "best_model_summary", "cluster_interpretation", "cluster_report")
        required = {name: artifacts[name] for name in required_names}
        optional = {name: path for name, path in artifacts.items() if name not in required}
        for warning in warnings:
            context.add_warning(warning)
        context.succeed(required, optional)
    except Exception as error:
        context.fail(error)
        raise

    if config.get("output", {}).get("legacy_projection", {}).get("enabled", True):
        try:
            _publish_legacy_projection(output_root, artifacts)
        except Exception as error:  # noqa: BLE001
            warnings.append(f"Legacy projection failed: {error}")
    if config.get("mlflow", {}).get("enabled", False):
        try:
            mlflow_run(config, best, metrics, artifacts)
        except Exception as error:  # noqa: BLE001
            warnings.append(f"MLflow publication failed: {error}")
    if config.get("output", {}).get("load_to_bigquery", False):
        try:
            load_clusters_to_bq(str(clusters_path), config)
        except Exception as error:  # noqa: BLE001
            warnings.append(f"BigQuery cluster publication failed: {error}")

    logger.info("Pipeline completed successfully. Best candidate: %s (%s)", best["candidate_id"], best["algorithm"])
    return PipelineResult(
        run_id=context.run_id,
        status=SUCCEEDED,
        run_dir=context.run_dir,
        manifest_path=context.manifest_path,
        algorithm=best["algorithm"],
        candidate_id=best["candidate_id"],
        selected_parameters=best["params"],
        artifact_paths=artifacts,
        warnings=tuple(warnings),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Team Clustering Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--smoke", action="store_true", help="Use smoke-sized stability settings")
    args = parser.parse_args()
    main(args.config, smoke=args.smoke)
