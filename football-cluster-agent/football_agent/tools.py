from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery, storage

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "")
CLUSTER_BUCKET = os.getenv("CLUSTER_BUCKET", "")
BQ_DATASET = os.getenv("BQ_DATASET", "")
CLUSTER_OUTPUT_PREFIX = os.getenv("CLUSTER_OUTPUT_PREFIX", "output/ml").strip("/")

if not PROJECT_ID:
    raise ValueError("Missing GOOGLE_CLOUD_PROJECT environment variable.")
if not CLUSTER_BUCKET:
    raise ValueError("Missing CLUSTER_BUCKET environment variable.")
if not BQ_DATASET:
    raise ValueError("Missing BQ_DATASET environment variable.")

_STORAGE = storage.Client(project=PROJECT_ID)
_BQ = bigquery.Client(project=PROJECT_ID)

_ALLOWED_SQL = re.compile(r"^\s*select\b", re.IGNORECASE)
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|create|truncate|grant|revoke)\b",
    re.IGNORECASE,
)
_DISALLOWED_PUBLIC = re.compile(r"\bbigquery-public-data\b", re.IGNORECASE)


def _list_blobs(prefix: str) -> list[tuple[str, datetime]]:
    blobs = list(_STORAGE.list_blobs(CLUSTER_BUCKET, prefix=prefix))
    out: list[tuple[str, datetime]] = []
    for blob in blobs:
        if blob.updated:
            out.append((blob.name, blob.updated.astimezone(timezone.utc)))
    return out


def _normalize_path(path: str) -> str:
    clean = path.strip().lstrip("/")
    if not clean:
        raise ValueError("Path cannot be empty.")
    if clean.startswith(CLUSTER_OUTPUT_PREFIX + "/"):
        return clean
    return f"{CLUSTER_OUTPUT_PREFIX}/{clean}"


def list_latest_clustering_run() -> dict[str, Any]:
    prefix = f"{CLUSTER_OUTPUT_PREFIX}/"
    blobs = _list_blobs(prefix)
    if not blobs:
        return {
            "status": "not_found",
            "message": f"No clustering artifacts found under gs://{CLUSTER_BUCKET}/{prefix}",
        }
    latest_blob, latest_updated = max(blobs, key=lambda x: x[1])
    return {
        "status": "ok",
        "bucket": CLUSTER_BUCKET,
        "prefix": prefix,
        "latest_blob": latest_blob,
        "latest_updated_utc": latest_updated.isoformat(),
        "blob_count": len(blobs),
    }


def list_output_artifact_timestamps(folder: str = "", limit: int = 200) -> dict[str, Any]:
    """List output/ml artifact files with UTC last-updated timestamps.

    Args:
        folder: Optional folder under output prefix (e.g., "metrics", "clusters", "report").
        limit: Max number of files to return.
    """
    safe_limit = max(1, min(int(limit), 1000))
    clean_folder = folder.strip().strip("/")
    prefix = f"{CLUSTER_OUTPUT_PREFIX}/"
    if clean_folder:
        prefix = f"{prefix}{clean_folder}/"

    bucket = _STORAGE.bucket(CLUSTER_BUCKET)
    blobs = list(bucket.list_blobs(prefix=prefix))
    file_rows: list[dict[str, Any]] = []
    for blob in blobs:
        # Skip virtual folder placeholders.
        if blob.name.endswith("/"):
            continue
        updated = blob.updated.astimezone(timezone.utc).isoformat() if blob.updated else None
        file_rows.append(
            {
                "path": blob.name,
                "updated_utc": updated,
                "size_bytes": int(blob.size or 0),
            }
        )

    # Most recent first.
    file_rows.sort(key=lambda x: x["updated_utc"] or "", reverse=True)
    return {
        "status": "ok",
        "bucket": CLUSTER_BUCKET,
        "prefix": prefix,
        "total_files": len(file_rows),
        "files": file_rows[:safe_limit],
    }


def read_gcs_file(path: str) -> str:
    blob_name = _normalize_path(path)
    blob = _STORAGE.bucket(CLUSTER_BUCKET).blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"Blob not found: gs://{CLUSTER_BUCKET}/{blob_name}")
    return blob.download_as_text()


def _read_json(relative_path: str) -> dict[str, Any]:
    return json.loads(read_gcs_file(relative_path))


def read_latest_cluster_interpretation() -> dict[str, Any]:
    return _read_json("metrics/cluster_interpretation.json")


def read_latest_cluster_report() -> str:
    return read_gcs_file("report/cluster_report.md")


def read_latest_metrics() -> dict[str, Any]:
    return _read_json("metrics/metrics.json")


def read_latest_best_model_summary() -> dict[str, Any]:
    return _read_json("metrics/best_model_summary.json")


def _read_clusters_df() -> pd.DataFrame:
    csv_text = read_gcs_file("clusters/clusters.csv")
    return pd.read_csv(io.StringIO(csv_text))


def query_bigquery(sql: str, max_rows: int = 500) -> list[dict[str, Any]]:
    if not _ALLOWED_SQL.search(sql or ""):
        raise ValueError("Only SELECT statements are allowed.")
    if _FORBIDDEN_SQL.search(sql):
        raise ValueError("Write/DDL keyword detected. Query blocked.")
    if _DISALLOWED_PUBLIC.search(sql):
        raise ValueError(
            f"Query blocked: use only `{PROJECT_ID}.{BQ_DATASET}` tables. "
            "Public datasets are disabled for this agent deployment."
        )
    if f"`{PROJECT_ID}.{BQ_DATASET}." not in sql and f"{PROJECT_ID}.{BQ_DATASET}." not in sql:
        raise ValueError(
            f"Query blocked: target dataset must be `{PROJECT_ID}.{BQ_DATASET}`."
        )

    query_job = _BQ.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            maximum_bytes_billed=5 * 10**9,
            use_query_cache=True,
        ),
    )
    rows = query_job.result(page_size=max_rows)
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if idx >= max_rows:
            break
        output.append(dict(row.items()))
    return output


def explain_cluster(cluster_id: int) -> dict[str, Any]:
    interpretation = read_latest_cluster_interpretation()
    clusters_df = _read_clusters_df()
    cid = str(cluster_id)

    cluster_info: dict[str, Any] | None = None
    if cid in interpretation:
        cluster_info = interpretation[cid]
    elif isinstance(interpretation.get("clusters"), list):
        for item in interpretation["clusters"]:
            if str(item.get("cluster_id")) == cid:
                cluster_info = item
                break

    if cluster_info is None:
        raise ValueError(f"Cluster {cluster_id} not found.")

    teams: list[str] = []
    if {"team_name", "cluster"}.issubset(clusters_df.columns):
        teams = (
            clusters_df.loc[clusters_df["cluster"] == cluster_id, "team_name"]
            .dropna()
            .astype(str)
            .tolist()
        )

    return {
        "cluster_id": cluster_id,
        "label": cluster_info.get("label"),
        "description": cluster_info.get("description"),
        "football_interpretation": cluster_info.get("football_interpretation"),
        "strengths": cluster_info.get("strengths", []),
        "weaknesses": cluster_info.get("weaknesses", []),
        "top_differentiating_features": cluster_info.get("top_differentiating_features", []),
        "confidence_score": cluster_info.get("confidence_score"),
        "warning": cluster_info.get("warning"),
        "is_outlier_like": cluster_info.get("is_outlier_like"),
        "teams_in_cluster": teams,
        "team_count": len(teams),
    }


def compare_teams(team_a: str, team_b: str) -> dict[str, Any]:
    clusters_df = _read_clusters_df()
    required_columns = {"team_name", "cluster"}
    if not required_columns.issubset(clusters_df.columns):
        raise ValueError("clusters.csv must include team_name and cluster columns.")

    def find_team(name: str) -> dict[str, Any] | None:
        match = clusters_df[clusters_df["team_name"].str.lower() == name.lower()]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    row_a = find_team(team_a)
    row_b = find_team(team_b)
    if row_a is None or row_b is None:
        missing = [n for n, r in ((team_a, row_a), (team_b, row_b)) if r is None]
        raise ValueError(f"Team(s) not found in clusters.csv: {', '.join(missing)}")

    sql = f"""
    SELECT team_name, rank, points, goals_diff
    FROM `{PROJECT_ID}.{BQ_DATASET}.fact_standings_snapshot`
    WHERE LOWER(team_name) IN (LOWER(@team_a), LOWER(@team_b))
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("team_a", "STRING", team_a),
            bigquery.ScalarQueryParameter("team_b", "STRING", team_b),
        ]
    )
    standings = [dict(row.items()) for row in _BQ.query(sql, job_config=job_config).result()]

    return {
        "team_a_cluster_row": row_a,
        "team_b_cluster_row": row_b,
        "same_cluster": int(row_a["cluster"]) == int(row_b["cluster"]),
        "bigquery_standings_snapshot": standings,
    }
