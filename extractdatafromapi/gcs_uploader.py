"""Upload local raw data files to a Google Cloud Storage bucket."""

from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

try:
    from google.cloud import storage
except ImportError:  # Cloud support is optional for local tests.
    storage = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RemotePointerState:
    """Remote latest pointer observed before a potentially long-running job."""

    generation: int | None
    run_id: str | None


@dataclass(frozen=True)
class RunUploadResult:
    """Result of publishing one immutable versioned run prefix."""

    run_id: str
    remote_prefix: str
    uploaded: int
    skipped_existing: int
    verified: int
    required_artifacts_verified: bool


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
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage is required for live GCS publication; "
            "install project requirements or inject a mock client"
        )
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


def download_blob_to_file(bucket_name: str, blob_name: str, local_path: Path) -> None:
    """Download a single blob to a local file path."""
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))


def upload_file(bucket_name: str, local_path: Path, blob_name: str) -> None:
    """Upload a single local file to the bucket under the given blob name."""
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))


def _normalize_blob_prefix(prefix: str) -> str:
    clean = prefix.strip().strip("/")
    if not clean:
        raise ValueError("GCS prefix cannot be empty")
    return clean


def _blob_size(blob: Any) -> int | None:
    blob.reload()
    return int(blob.size) if blob.size is not None else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _upload_immutable_file(bucket: Any, local_path: Path, blob_name: str) -> bool:
    """Upload once; an existing object must match size and is never overwritten."""
    blob = bucket.blob(blob_name)
    expected_hash = _sha256_file(local_path)
    if blob.exists():
        remote_size = _blob_size(blob)
        if remote_size != local_path.stat().st_size:
            raise RuntimeError(
                f"Immutable blob already exists with different size: {blob_name} "
                f"(remote={remote_size}, local={local_path.stat().st_size})"
            )
        remote_hash = (blob.metadata or {}).get("sha256")
        if remote_hash != expected_hash:
            raise RuntimeError(
                f"Immutable blob already exists without the expected SHA-256: {blob_name}"
            )
        return False
    blob.metadata = {"sha256": expected_hash}
    blob.upload_from_filename(str(local_path), if_generation_match=0)
    return True


def _verify_blob(bucket: Any, local_path: Path, blob_name: str) -> None:
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise RuntimeError(f"Expected GCS object is missing: {blob_name}")
    remote_size = _blob_size(blob)
    if remote_size != local_path.stat().st_size:
        raise RuntimeError(
            f"GCS size verification failed for {blob_name}: "
            f"remote={remote_size}, local={local_path.stat().st_size}"
        )
    remote_hash = (blob.metadata or {}).get("sha256")
    expected_hash = _sha256_file(local_path)
    if remote_hash != expected_hash:
        raise RuntimeError(f"GCS SHA-256 metadata verification failed for {blob_name}")


def read_remote_latest_pointer(
    output_prefix: str = "output/ml",
    bucket_name: str | None = None,
    client: storage.Client | None = None,
) -> RemotePointerState:
    """Read latest pointer and its GCS generation for concurrency protection."""
    bucket_id = bucket_name or get_bucket_name()
    storage_client = client or get_storage_client()
    blob = storage_client.bucket(bucket_id).blob(
        f"{_normalize_blob_prefix(output_prefix)}/latest_run.json"
    )
    if not blob.exists():
        return RemotePointerState(generation=None, run_id=None)
    blob.reload()
    try:
        payload = json.loads(blob.download_as_text())
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("Remote latest_run.json is not valid JSON") from error
    run_id = payload.get("run_id")
    return RemotePointerState(
        generation=int(blob.generation) if blob.generation is not None else None,
        run_id=str(run_id) if run_id else None,
    )


def upload_completed_run(
    run_dir: Path,
    output_prefix: str = "output/ml",
    bucket_name: str | None = None,
    client: storage.Client | None = None,
) -> RunUploadResult:
    """Publish one SUCCEEDED run, with its manifest uploaded and verified last."""
    local_run_dir = Path(run_dir)
    manifest_path = local_run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_id") or "")
    if manifest.get("status") != "SUCCEEDED":
        raise ValueError(f"Only SUCCEEDED runs may be uploaded, got {manifest.get('status')}")
    if run_id != local_run_dir.name:
        raise ValueError(f"Manifest run_id {run_id!r} does not match directory {local_run_dir.name!r}")

    required_paths: list[Path] = []
    for record in manifest.get("required_artifacts", []):
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid required artifact path: {relative}")
        local_path = local_run_dir / relative
        if not local_path.is_file():
            raise FileNotFoundError(f"Missing required artifact before upload: {local_path}")
        required_paths.append(local_path)

    bucket_id = bucket_name or get_bucket_name()
    storage_client = client or get_storage_client()
    bucket = storage_client.bucket(bucket_id)
    remote_prefix = f"{_normalize_blob_prefix(output_prefix)}/runs/{run_id}"
    local_files = sorted(
        (path for path in iter_local_files(local_run_dir) if path != manifest_path),
        key=lambda path: path.relative_to(local_run_dir).as_posix(),
    )
    uploaded = 0
    skipped = 0
    for local_path in local_files:
        relative = local_path.relative_to(local_run_dir).as_posix()
        blob_name = f"{remote_prefix}/{relative}"
        if _upload_immutable_file(bucket, local_path, blob_name):
            uploaded += 1
        else:
            skipped += 1

    for local_path in required_paths:
        relative = local_path.relative_to(local_run_dir).as_posix()
        _verify_blob(bucket, local_path, f"{remote_prefix}/{relative}")

    # A remote run is discoverable as complete only after its SUCCEEDED manifest exists.
    manifest_blob_name = f"{remote_prefix}/manifest.json"
    if _upload_immutable_file(bucket, manifest_path, manifest_blob_name):
        uploaded += 1
    else:
        skipped += 1
    _verify_blob(bucket, manifest_path, manifest_blob_name)
    return RunUploadResult(
        run_id=run_id,
        remote_prefix=remote_prefix,
        uploaded=uploaded,
        skipped_existing=skipped,
        verified=len(required_paths) + 1,
        required_artifacts_verified=True,
    )


def publish_legacy_projection(
    run_dir: Path,
    output_prefix: str = "output/ml",
    bucket_name: str | None = None,
    client: storage.Client | None = None,
) -> int:
    """Overwrite only mandatory flat artifacts after the versioned run is verified."""
    local_run_dir = Path(run_dir)
    manifest = json.loads((local_run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "SUCCEEDED":
        raise ValueError("Legacy projection requires a SUCCEEDED run")
    bucket_id = bucket_name or get_bucket_name()
    storage_client = client or get_storage_client()
    bucket = storage_client.bucket(bucket_id)
    prefix = _normalize_blob_prefix(output_prefix)
    published = 0
    for record in manifest.get("required_artifacts", []):
        relative = Path(str(record["path"]))
        local_path = local_run_dir / relative
        if not local_path.is_file():
            raise FileNotFoundError(f"Missing legacy source artifact: {local_path}")
        blob = bucket.blob(f"{prefix}/{relative.as_posix()}")
        blob.upload_from_filename(str(local_path))
        published += 1
    return published


def publish_latest_pointer(
    local_pointer_path: Path,
    expected_remote_state: RemotePointerState,
    output_prefix: str = "output/ml",
    bucket_name: str | None = None,
    client: storage.Client | None = None,
) -> None:
    """Publish latest last, only if the remote pointer has not changed meanwhile."""
    pointer_path = Path(local_pointer_path)
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    if payload.get("status") != "SUCCEEDED":
        raise ValueError("latest_run.json may point only to a SUCCEEDED run")
    bucket_id = bucket_name or get_bucket_name()
    storage_client = client or get_storage_client()
    blob = storage_client.bucket(bucket_id).blob(
        f"{_normalize_blob_prefix(output_prefix)}/latest_run.json"
    )
    if_generation_match = (
        expected_remote_state.generation
        if expected_remote_state.generation is not None
        else 0
    )
    blob.upload_from_filename(
        str(pointer_path),
        if_generation_match=if_generation_match,
    )


def publish_completed_run_and_latest(
    run_dir: Path,
    local_pointer_path: Path,
    expected_remote_state: RemotePointerState,
    output_prefix: str = "output/ml",
    bucket_name: str | None = None,
    client: storage.Client | None = None,
    publish_legacy: bool = True,
) -> RunUploadResult:
    """Publish an immutable run, compatibility files, and latest pointer last."""
    result = upload_completed_run(
        run_dir,
        output_prefix=output_prefix,
        bucket_name=bucket_name,
        client=client,
    )
    if not result.required_artifacts_verified:
        raise RuntimeError("Required remote run artifacts were not verified")
    if publish_legacy:
        publish_legacy_projection(
            run_dir,
            output_prefix=output_prefix,
            bucket_name=bucket_name,
            client=client,
        )
    publish_latest_pointer(
        local_pointer_path,
        expected_remote_state,
        output_prefix=output_prefix,
        bucket_name=bucket_name,
        client=client,
    )
    return result


def upload_folder_to_prefix(
    local_root: Path,
    remote_prefix: str,
    bucket_name: str | None = None,
    skip_existing: bool = False,
    client: storage.Client | None = None,
) -> dict[str, int]:
    """Upload one local folder to an explicit GCS prefix without adjacent history."""
    root = Path(local_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Local folder not found: {root}")
    bucket_id = bucket_name or get_bucket_name()
    storage_client = client or get_storage_client()
    bucket = storage_client.bucket(bucket_id)
    prefix = _normalize_blob_prefix(remote_prefix)
    summary = {"uploaded": 0, "skipped": 0, "failed": 0}
    for local_path in iter_local_files(root):
        relative = local_path.relative_to(root).as_posix()
        blob = bucket.blob(f"{prefix}/{relative}")
        try:
            if skip_existing and blob.exists():
                summary["skipped"] += 1
                continue
            blob.upload_from_filename(str(local_path))
            summary["uploaded"] += 1
        except Exception as error:  # noqa: BLE001 - preserve per-file summary
            print(f"[ERROR] Upload failed for {prefix}/{relative}: {error}")
            summary["failed"] += 1
    return summary


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
