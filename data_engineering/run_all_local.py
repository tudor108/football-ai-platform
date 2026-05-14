"""Run the full local pipeline end-to-end:
  1. Extract raw JSON from API-Football -> data/raw/
  2. Upload data/raw/ to GCS
  3. Transform raw GCS JSON -> output/tables/*.csv
  4. Derive analytical tables -> output/derived/*.csv
  5. Run ML clustering -> output/ml/*
  6. Upload output/* to GCS under processed/
  7. Load CSVs into BigQuery

Idempotent: skip-existing logic prevents duplicate work.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path when executed as:
#   python data_engineering/run_all_local.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from extractdatafromapi.gcs_uploader import upload_local_folder
from extractdatafromapi.main import run_extraction
from data_engineering import derive_tables_local, load_to_bigquery, transform_data_local
from ml.clustering.team_clustering_pipeline import main as run_team_clustering


def step(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"=== {title}")
    print("=" * 70)


def main() -> None:
    step("Step 1/5: Extract from API-Football -> data/raw/")
    run_extraction()

    step("Step 2/5: Upload raw data to GCS (gs://.../raw/)")
    raw_root = Path("data") / "raw"
    if raw_root.exists():
        upload_local_folder(raw_root)
    else:
        print("[WARN] No data/raw folder to upload (extraction may have skipped everything).")

    step("Step 3/5: Transform raw JSON from GCS -> output/tables/")
    transform_data_local.main()

    step("Step 4/5: Build derived analytical tables -> output/derived/")
    derive_tables_local.main()

    step("Step 5/7: Run ML clustering locally (output/derived -> output/ml)")
    try:
        run_team_clustering("ml/configs/team_clustering_config.yaml")
    except Exception as error:  # noqa: BLE001
        print(f"[ERROR] ML clustering step failed: {error}")

    step("Step 6/7: Upload processed CSVs to GCS (gs://.../processed/)")
    output_root = Path("output")
    if output_root.exists():
        # Uploaded as processed/tables/*, processed/derived/*, processed/ml/*
        upload_local_folder(output_root, skip_existing=False)
    else:
        print("[WARN] No output folder to upload.")

    step("Step 7/7: Load CSVs into BigQuery dataset")
    try:
        load_to_bigquery.main()
    except Exception as error:  # noqa: BLE001
        print(f"[ERROR] BigQuery load failed: {error}")

    print("\n[INFO] === Pipeline complete ===")


if __name__ == "__main__":
    main()
