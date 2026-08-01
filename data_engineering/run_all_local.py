"""Run the full local pipeline end-to-end:
  1. Extract raw JSON from API-Football -> data/raw/
  2. Upload data/raw/ to GCS
  3. Transform raw GCS JSON -> output/tables/*.csv
  4. Derive analytical tables -> output/derived/*.csv
  5. Run ML clustering -> output/ml/*
  6. Upload tables/derived plus only the completed ML run to GCS
  7. Load CSVs into BigQuery

Idempotent: skip-existing logic prevents duplicate work.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path when executed as:
#   python data_engineering/run_all_local.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from extractdatafromapi.gcs_uploader import (
    RemotePointerState,
    publish_completed_run_and_latest,
    read_remote_latest_pointer,
    upload_folder_to_prefix,
    upload_local_folder,
)
from extractdatafromapi.main import run_extraction
from data_engineering import derive_tables_local, load_to_bigquery, transform_data_local
from ml.clustering.team_clustering_pipeline import main as run_team_clustering
from ml.utils.pipeline_result import PipelineResult
from ml.utils.run_context import SUCCEEDED
from business.agent_search_cloud import import_documents_from_gcs


def step(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"=== {title}")
    print("=" * 70)


def main() -> None:
    step("Step 1/7: Extract from API-Football -> data/raw/")
    run_extraction()

    step("Step 2/7: Upload raw data to GCS (gs://.../raw/)")
    raw_root = Path("data") / "raw"
    if raw_root.exists():
        upload_local_folder(raw_root)
    else:
        print("[WARN] No data/raw folder to upload (extraction may have skipped everything).")

    step("Step 3/7: Transform raw JSON from GCS -> output/tables/")
    transform_data_local.main()

    step("Step 4/7: Build derived analytical tables -> output/derived/")
    derive_tables_local.main()

    # "Local" means this process' filesystem: developer workstation or the
    # ephemeral GitHub Actions runner. No ML artifact is published yet.
    pointer_state: RemotePointerState | None = None
    try:
        pointer_state = read_remote_latest_pointer()
        print(
            "[INFO] Remote latest before ML: "
            f"run_id={pointer_state.run_id}, generation={pointer_state.generation}"
        )
    except Exception as error:  # noqa: BLE001
        print(f"[ERROR] Cannot safely read remote latest pointer: {error}")

    step("Step 5/7: Run ML locally -> output/ml/runs/{run_id}/")
    ml_result: PipelineResult | None = None
    try:
        ml_result = run_team_clustering("ml/configs/team_clustering_config.yaml")
        print(
            f"[INFO] Local ML run complete: {ml_result.run_id} "
            f"({ml_result.status}) at {ml_result.run_dir}"
        )
    except Exception as error:  # noqa: BLE001
        print(f"[ERROR] ML clustering step failed: {error}")

    step("Step 6/7: Publish processed data and completed ML run to GCS")
    for local_folder, remote_prefix in (
        (Path("output") / "tables", "output/tables"),
        (Path("output") / "derived", "output/derived"),
    ):
        if local_folder.is_dir():
            summary = upload_folder_to_prefix(
                local_folder,
                remote_prefix,
                skip_existing=False,
            )
            print(f"[INFO] Published {remote_prefix}: {summary}")

    if ml_result is None or ml_result.status != SUCCEEDED:
        print("[WARN] No SUCCEEDED ML run; versioned ML publication skipped.")
    elif pointer_state is None:
        print("[WARN] Remote pointer state unavailable; ML publication skipped to avoid an unsafe latest update.")
    else:
        try:
            latest_path = ml_result.run_dir.parents[1] / "latest_run.json"
            upload_result = publish_completed_run_and_latest(
                run_dir=ml_result.run_dir,
                local_pointer_path=latest_path,
                expected_remote_state=pointer_state,
                publish_legacy=True,
            )
            print(
                f"[INFO] Uploaded immutable run {upload_result.run_id}: "
                f"uploaded={upload_result.uploaded}, "
                f"skipped={upload_result.skipped_existing}, "
                f"verified={upload_result.verified}"
            )
            print(f"[INFO] Remote latest_run.json now points to {ml_result.run_id}")
            data_store_id = os.getenv("AGENT_SEARCH_DATA_STORE_ID", "").strip()
            if data_store_id:
                project_id = (
                    os.getenv("GCP_PROJECT_ID", "").strip()
                    or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
                )
                bucket = os.getenv("GCS_BUCKET", "").strip() or os.getenv(
                    "CLUSTER_BUCKET", ""
                ).strip()
                location = os.getenv("AGENT_SEARCH_LOCATION", "global").strip()
                try:
                    search_import = import_documents_from_gcs(
                        project_id=project_id,
                        data_store_id=data_store_id,
                        gcs_uri=(
                            f"gs://{bucket}/output/ml/runs/{ml_result.run_id}/"
                            "search/agent_search_documents.jsonl"
                        ),
                        location=location,
                        wait=False,
                    )
                    print(
                        "[INFO] Agent Search incremental import started: "
                        f"{search_import.operation_name}"
                    )
                except Exception as error:  # noqa: BLE001
                    print(
                        "[WARN] Completed run remains published, but optional "
                        f"Agent Search indexing was skipped/failed: {error}"
                    )
            else:
                print(
                    "[INFO] AGENT_SEARCH_DATA_STORE_ID is unset; semantic history "
                    "indexing skipped."
                )
        except Exception as error:  # noqa: BLE001
            print(
                "[ERROR] Versioned ML publication failed; remote latest pointer "
                f"was not intentionally advanced: {error}"
            )

    step("Step 7/7: Load CSVs into BigQuery dataset")
    try:
        load_to_bigquery.main()
    except Exception as error:  # noqa: BLE001
        print(f"[ERROR] BigQuery load failed: {error}")

    print("\n[INFO] === Pipeline complete ===")


if __name__ == "__main__":
    main()
