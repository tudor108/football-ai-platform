"""Load all processed CSVs into BigQuery as managed tables.

Reads:
- output/tables/*.csv  -> dataset.<filename without .csv>
- output/derived/*.csv -> dataset.<filename without .csv>

Each table is fully replaced (WRITE_TRUNCATE) on every run; schema is auto-detected.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import bigquery


DEFAULT_DATASET = "football_analytics"
DEFAULT_LOCATION = "europe-central2"

INPUT_FOLDERS = [
    Path("output") / "tables",
    Path("output") / "derived",
]


def get_project_id() -> str:
    load_dotenv()
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("Missing GCP_PROJECT_ID in .env")
    return project_id


def get_dataset_id() -> str:
    load_dotenv()
    return os.getenv("BQ_DATASET", DEFAULT_DATASET)


def get_location() -> str:
    load_dotenv()
    return os.getenv("BQ_LOCATION", DEFAULT_LOCATION)


def get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=get_project_id())


def ensure_dataset(client: bigquery.Client, dataset_id: str, location: str) -> None:
    """Create the dataset if it doesn't exist."""
    full_id = f"{client.project}.{dataset_id}"
    try:
        client.get_dataset(full_id)
        print(f"[INFO] Dataset exists: {full_id}")
    except Exception:
        ds = bigquery.Dataset(full_id)
        ds.location = location
        client.create_dataset(ds, exists_ok=True)
        print(f"[INFO] Created dataset: {full_id} ({location})")


def load_csv_to_table(
    client: bigquery.Client,
    csv_path: Path,
    dataset_id: str,
) -> None:
    """Load one CSV into BigQuery, fully replacing the destination table."""
    table_name = csv_path.stem  # filename without .csv
    table_ref = f"{client.project}.{dataset_id}.{table_name}"

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        allow_quoted_newlines=True,
        encoding="UTF-8",  # CSVs are UTF-8 with BOM; BigQuery handles it.
        max_bad_records=10,
    )

    print(f"[INFO] Loading {csv_path.name} -> {table_ref}")
    with csv_path.open("rb") as fh:
        job = client.load_table_from_file(fh, table_ref, job_config=job_config)
    job.result()  # wait for completion

    table = client.get_table(table_ref)
    print(f"[INFO]   -> {table.num_rows} rows, {len(table.schema)} cols")


def main() -> None:
    print("[INFO] === load_to_bigquery: start ===")
    client = get_bq_client()
    dataset_id = get_dataset_id()
    location = get_location()
    ensure_dataset(client, dataset_id, location)

    total = 0
    for folder in INPUT_FOLDERS:
        if not folder.exists():
            print(f"[WARN] Skipping missing folder: {folder}")
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            try:
                load_csv_to_table(client, csv_path, dataset_id)
                total += 1
            except Exception as error:  # noqa: BLE001
                print(f"[ERROR] Failed loading {csv_path.name}: {error}")

    print(f"[INFO] === load_to_bigquery: done ({total} tables loaded) ===")


if __name__ == "__main__":
    main()
