"""Run the full local pipeline end-to-end:
  1. Extract raw JSON from API-Football -> data/raw/
  2. Upload data/raw/ to GCS
  3. Transform raw GCS JSON -> output/tables/*.csv
  4. Derive analytical tables -> output/derived/*.csv
  5. Upload output/tables/ + output/derived/ to GCS under processed/

Idempotent: skip-existing logic prevents duplicate work.
"""

from __future__ import annotations

from pathlib import Path

from extractdatafromapi.gcs_uploader import upload_local_folder
from extractdatafromapi.main import run_extraction
import transform_data_local
import derive_tables_local
import load_to_bigquery


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

    step("Step 5/6: Upload processed CSVs to GCS (gs://.../processed/)")
    output_root = Path("output")
    if output_root.exists():
        # Uploaded as processed/tables/* and processed/derived/*
        upload_local_folder(output_root)
    else:
        print("[WARN] No output folder to upload.")

    step("Step 6/6: Load CSVs into BigQuery dataset")
    try:
        load_to_bigquery.main()
    except Exception as error:  # noqa: BLE001
        print(f"[ERROR] BigQuery load failed: {error}")

    print("\n[INFO] === Pipeline complete ===")


if __name__ == "__main__":
    main()
