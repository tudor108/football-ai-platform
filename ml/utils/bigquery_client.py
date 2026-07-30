"""BigQuery client utilities for the clustering pipeline."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

try:
    from google.cloud import bigquery
except ImportError:  # BigQuery is optional for local-only ML runs.
    bigquery = None  # type: ignore[assignment]


def get_team_features_from_bq(config: dict[str, Any]) -> pd.DataFrame:
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery is required for input.source=bigquery")
    project_id = os.environ["GCP_PROJECT_ID"]
    dataset = os.environ["BQ_DATASET"]
    table = config["input"].get("table", "fact_match_features")
    query = config["input"].get("query")

    client = bigquery.Client(project=project_id)
    if query:
        return client.query(query).to_dataframe()
    sql = f"SELECT * FROM `{project_id}.{dataset}.{table}`"
    return client.query(sql).to_dataframe()


def load_clusters_to_bq(clusters_path: str, config: dict[str, Any]) -> None:
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery is required when output.load_to_bigquery=true")
    project_id = os.environ["GCP_PROJECT_ID"]
    dataset = os.environ["BQ_DATASET"]
    table = config["output"].get("bq_table", "ml_team_clusters")

    client = bigquery.Client(project=project_id)
    df = pd.read_csv(clusters_path, encoding="utf-8-sig")
    job = client.load_table_from_dataframe(
        df,
        f"{project_id}.{dataset}.{table}",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    )
    job.result()
