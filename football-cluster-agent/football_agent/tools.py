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

from .analytics import (
    CLUSTERING_FEATURES,
    build_team_profiles_from_matches,
    personalized_similarity,
    predict_match_from_history,
    summarize_latest_matches,
)

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


def _read_match_features_df() -> pd.DataFrame:
    """Read the completed match feature history used by personalized tools."""
    sql = f"""
    SELECT
      fixture_id,
      date,
      league_id,
      season,
      home_team_id,
      home_team_name,
      away_team_id,
      away_team_name,
      goals_home,
      goals_away,
      status_short,
      home_form_pts_lastN,
      home_form_wins_lastN,
      home_form_draws_lastN,
      home_form_losses_lastN,
      home_form_gf_avg_lastN,
      home_form_ga_avg_lastN,
      home_form_gd_avg_lastN,
      away_form_pts_lastN,
      away_form_wins_lastN,
      away_form_draws_lastN,
      away_form_losses_lastN,
      away_form_gf_avg_lastN,
      away_form_ga_avg_lastN,
      away_form_gd_avg_lastN
    FROM `{PROJECT_ID}.{BQ_DATASET}.fact_match_features`
    WHERE goals_home IS NOT NULL
      AND goals_away IS NOT NULL
    """
    rows = [
        dict(row.items())
        for row in _BQ.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                maximum_bytes_billed=5 * 10**9,
                use_query_cache=True,
            ),
        ).result()
    ]
    if not rows:
        raise ValueError("No completed matches found in fact_match_features.")
    return pd.DataFrame(rows)


def _read_personalization_profiles(clusters_df: pd.DataFrame) -> pd.DataFrame:
    """Prefer self-contained cluster artifacts, with a BigQuery fallback."""
    required_profile_columns = {"team_id", "team_name", *CLUSTERING_FEATURES}
    if required_profile_columns.issubset(clusters_df.columns):
        return clusters_df[
            ["team_id", "team_name", *CLUSTERING_FEATURES]
        ].copy()
    return build_team_profiles_from_matches(_read_match_features_df())


def query_bigquery(sql: str, max_rows: int = 500) -> list[dict[str, Any]]:
    """Run a read-only query against a known project table.

    Known match tables are `fact_match_features` and `fact_matches`; there is
    no table named `matches`. Prefer get_latest_available_matches for questions
    about the newest available match data.
    """
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


def get_latest_available_matches(limit: int = 10) -> dict[str, Any]:
    """Show data coverage, available statistics, and newest completed matches.

    Use this tool for questions such as "what is your newest data?", "when were
    the latest matches?", or "what recent match statistics do you have?".
    The result describes the latest data stored in BigQuery, not live scores.

    Args:
        limit: Number of newest completed matches to return, from 1 to 50.
    """
    return summarize_latest_matches(_read_match_features_df(), limit=limit)


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


def personalized_team_similarity(
    team_name: str,
    attack_preference: float = 0.0,
    defense_preference: float = 0.0,
    results_preference: float = 0.0,
    recent_form_preference: float = 0.0,
    consistency_preference: float = 0.0,
    top_n: int = 5,
) -> dict[str, Any]:
    """Find similar and recommended teams through a user's personal lens.

    Preference values must be between -1 and 1:
    - positive means the user prefers more/stronger of that dimension;
    - negative means the user prefers the opposite (for example, -1
      consistency means a preference for volatile teams);
    - zero means that dimension is neutral.

    The magnitude also controls how much the dimension influences distance.
    Similarity values are relative indexes and must never be called
    probabilities.
    """
    clusters_df = _read_clusters_df()
    profiles_df = _read_personalization_profiles(clusters_df)
    return personalized_similarity(
        profiles=profiles_df,
        clusters=clusters_df,
        team_name=team_name,
        preferences={
            "attack": attack_preference,
            "defense": defense_preference,
            "results": results_preference,
            "recent_form": recent_form_preference,
            "consistency": consistency_preference,
        },
        top_n=top_n,
    )


def predict_match_from_stats(
    home_team: str,
    away_team: str,
) -> dict[str, Any]:
    """Produce an explainable, uncalibrated statistical match forecast.

    The tool uses historical home/away scoring rates, Bayesian smoothing,
    recent form, and an independent Poisson goals baseline. Results are not
    guarantees and do not include lineups, injuries, suspensions, or odds.
    """
    return predict_match_from_history(
        matches=_read_match_features_df(),
        home_team=home_team,
        away_team=away_team,
    )
