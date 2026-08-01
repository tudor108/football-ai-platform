"""Build Agent Search-ready structured documents from one immutable ML run."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _plain(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            if value:
                return value
            continue
        if pd.isna(value) or not str(value).strip():
            continue
        return value
    return None


def _document_id(*parts: Any) -> str:
    value = "-".join(str(part).lower() for part in parts if part is not None)
    value = re.sub(r"[^a-z0-9-]+", "-", value).strip("-")
    value = re.sub(r"-+", "-", value)
    if not value:
        raise ValueError("Agent Search document id cannot be empty.")
    return value[:63].rstrip("-")


def build_agent_search_documents(
    run_id: str,
    clusters: pd.DataFrame,
    interpretations: dict[Any, dict[str, Any]],
    metrics: dict[str, Any],
    best_model: dict[str, Any],
    run_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create one run overview and one semantic document per team."""
    required = {"team_id", "team_name", "cluster"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"Clusters are missing columns: {sorted(missing)}")
    metadata = dict(run_metadata or {})
    documents: list[dict[str, Any]] = []

    overview_content = (
        f"Football clustering run {run_id}. The selected model was "
        f"{best_model.get('algorithm')} candidate {best_model.get('candidate_id')}. "
        f"It analyzed {clusters['team_id'].nunique()} teams. "
        f"Data coverage ends at {metadata.get('data_as_of_date')}. "
        "Metrics and cluster movements are descriptive; they do not establish causality."
    )
    documents.append(
        {
            "id": _document_id("run", run_id, "overview"),
            "document_type": "run_overview",
            "title": f"Football clustering run {run_id}",
            "content": overview_content,
            "run_id": run_id,
            "run_date": metadata.get("data_as_of_date"),
            "algorithm": best_model.get("algorithm"),
            "candidate_id": best_model.get("candidate_id"),
            "team_count": int(clusters["team_id"].nunique()),
            "metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True),
            "source_artifact": f"runs/{run_id}/metrics/metrics.json",
        }
    )

    excluded_from_features = {
        "team_id",
        "team_name",
        "cluster",
        "best_algorithm",
        "candidate_id",
        "pca_x",
        "pca_y",
        "cluster_label",
        "cluster_description",
        "strengths",
        "weaknesses",
        "is_outlier_like",
        "cluster_warning",
        "cluster_confidence_score",
    }
    feature_columns = [
        column
        for column in clusters.columns
        if column not in excluded_from_features
        and not column.startswith("feat_")
        and pd.api.types.is_numeric_dtype(clusters[column])
    ]
    for _, row in clusters.iterrows():
        cluster_id = int(row["cluster"])
        info = interpretations.get(cluster_id, interpretations.get(str(cluster_id), {}))
        cluster_label = _coalesce(row.get("cluster_label"), info.get("label"))
        cluster_description = _coalesce(
            row.get("cluster_description"), info.get("description")
        )
        strengths = _coalesce(row.get("strengths"), info.get("strengths"))
        weaknesses = _coalesce(row.get("weaknesses"), info.get("weaknesses"))
        if isinstance(strengths, list):
            strengths = "; ".join(str(value) for value in strengths)
        if isinstance(weaknesses, list):
            weaknesses = "; ".join(str(value) for value in weaknesses)
        feature_values = {
            column: _plain(row[column])
            for column in feature_columns
            if pd.notna(row[column])
        }
        differentiators = info.get("top_differentiating_features", [])
        if not differentiators:
            differentiators = [
                row.get(f"feat_{index}_name")
                for index in range(1, 6)
                if pd.notna(row.get(f"feat_{index}_name"))
            ]
        content = " ".join(
            part
            for part in (
                f"In run {run_id}, {row['team_name']} was assigned to official cluster {cluster_id}.",
                f"Cluster label: {cluster_label}." if cluster_label is not None else "",
                (
                    f"Description: {cluster_description}."
                    if cluster_description is not None
                    else ""
                ),
                f"Strengths: {strengths}." if strengths is not None else "",
                f"Weaknesses: {weaknesses}." if weaknesses is not None else "",
                f"Key differentiating features: {', '.join(str(value) for value in differentiators)}.",
                "This assignment is descriptive and the official cluster was not changed by user preferences.",
            )
            if part
        )
        documents.append(
            {
                "id": _document_id("run", run_id, "team", row["team_id"]),
                "document_type": "team_cluster_assignment",
                "title": f"{row['team_name']} in clustering run {run_id}",
                "content": content,
                "run_id": run_id,
                "run_date": metadata.get("data_as_of_date"),
                "team_id": str(_plain(row["team_id"])),
                "team_name": str(row["team_name"]),
                "cluster_id": cluster_id,
                "cluster_label": _plain(cluster_label),
                "feature_values_json": json.dumps(feature_values, ensure_ascii=False, sort_keys=True),
                "source_artifact": f"runs/{run_id}/clusters/clusters.csv",
            }
        )
    return documents


def write_agent_search_documents(path: Path, documents: list[dict[str, Any]]) -> Path:
    """Atomically write newline-delimited structured documents."""
    if not documents:
        raise ValueError("At least one Agent Search document is required.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for document in documents:
                handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
