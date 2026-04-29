"""Upload local raw data files to a Google Cloud Storage bucket."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from google.cloud import storage


def get_bucket_name() -> str:
    """Read the target bucket name from environment variables."""
    load_dotenv()
    bucket_name = os.getenv("GCS_BUCKET")
    if not bucket_name:
        raise ValueError("Missing GCS_BUCKET in .env")
    return bucket_name


def get_project_id() -> str:
    """Read the GCP project id from environment variables."""
    load_dotenv()
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("Missing GCP_PROJECT_ID in .env")
    return project_id


def get_storage_client() -> storage.Client:
    """Build an authenticated GCS client using Application Default Credentials."""
    return storage.Client(project=get_project_id())


def list_existing_blob_names(bucket_name: str, prefix: str = "") -> set[str]:
    """Return the set of object names that already exist in the bucket under a prefix."""
    client = get_storage_client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    return {blob.name for blob in blobs}


def iter_local_files(local_root: Path) -> Iterable[Path]:
    """Yield every file under a local root directory."""
    for path in local_root.rglob("*"):
        if path.is_file():
            yield path


def upload_file(bucket_name: str, local_path: Path, blob_name: str) -> None:
    """Upload a single local file to the bucket under the given blob name."""
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))


def upload_local_folder(
    local_root: Path,
    bucket_name: str | None = None,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Upload every file under local_root to the bucket, preserving the relative structure."""
    bucket = bucket_name or get_bucket_name()

    if not local_root.exists():
        raise FileNotFoundError(f"Local folder not found: {local_root}")

    existing: set[str] = set()
    if skip_existing:
        print(f"[INFO] Listing existing blobs in gs://{bucket}/")
        existing = list_existing_blob_names(bucket)
        print(f"[INFO] Found {len(existing)} existing blobs")

    summary = {"uploaded": 0, "skipped": 0, "failed": 0}
    for local_path in iter_local_files(local_root):
        # Use forward slashes for blob names regardless of OS.
        relative = local_path.relative_to(local_root.parent).as_posix()
        if skip_existing and relative in existing:
            summary["skipped"] += 1
            continue
        try:
            print(f"[INFO] Uploading {relative}")
            upload_file(bucket, local_path, relative)
            summary["uploaded"] += 1
        except Exception as error:  # noqa: BLE001
            print(f"[ERROR] Upload failed for {relative}: {error}")
            summary["failed"] += 1

    print(f"[INFO] Upload summary: {summary}")
    return summary


def main() -> None:
    """CLI entry point: upload data/raw to the configured bucket."""
    local_root = Path("data") / "raw"
    upload_local_folder(local_root)


if __name__ == "__main__":
    main()
