from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from extractdatafromapi.gcs_uploader import (
    RemotePointerState,
    publish_latest_pointer,
    publish_completed_run_and_latest,
    publish_legacy_projection,
    read_remote_latest_pointer,
    upload_completed_run,
)
from ml.utils.run_context import RunContext


class FakeBlob:
    def __init__(self, bucket: "FakeBucket", name: str) -> None:
        self.bucket = bucket
        self.name = name
        self._pending_metadata: dict[str, str] | None = None

    @property
    def size(self) -> int | None:
        record = self.bucket.objects.get(self.name)
        return len(record["data"]) if record else None

    @property
    def generation(self) -> int | None:
        record = self.bucket.objects.get(self.name)
        return int(record["generation"]) if record else None

    @property
    def metadata(self) -> dict[str, str] | None:
        record = self.bucket.objects.get(self.name)
        return dict(record.get("metadata") or {}) if record else self._pending_metadata

    @metadata.setter
    def metadata(self, value: dict[str, str] | None) -> None:
        self._pending_metadata = dict(value or {})

    def exists(self) -> bool:
        return self.name in self.bucket.objects

    def reload(self) -> None:
        return None

    def download_as_text(self) -> str:
        return self.bucket.objects[self.name]["data"].decode("utf-8")

    def upload_from_filename(
        self,
        filename: str,
        if_generation_match: int | None = None,
    ) -> None:
        current = self.bucket.objects.get(self.name)
        current_generation = int(current["generation"]) if current else 0
        if if_generation_match is not None and if_generation_match != current_generation:
            raise RuntimeError("generation precondition failed")
        data = Path(filename).read_bytes()
        self.bucket.objects[self.name] = {
            "data": data,
            "generation": current_generation + 1,
            "metadata": self._pending_metadata or (current or {}).get("metadata", {}),
        }
        self.bucket.upload_order.append(self.name)


class FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.upload_order: list[str] = []

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)


class FakeClient:
    def __init__(self) -> None:
        self.fake_bucket = FakeBucket()

    def bucket(self, _: str) -> FakeBucket:
        return self.fake_bucket


def _successful_run(tmp_path: Path) -> RunContext:
    config = tmp_path / "config.yaml"
    config.write_text("features: [a]\n", encoding="utf-8")
    context = RunContext(tmp_path / "output" / "ml", config, repo_root=tmp_path)
    artifacts: dict[str, Path] = {}
    for name, relative in {
        "clusters": "clusters/clusters.csv",
        "metrics": "metrics/metrics.json",
        "best_model_summary": "metrics/best_model_summary.json",
        "cluster_interpretation": "metrics/cluster_interpretation.json",
        "cluster_report": "report/cluster_report.md",
    }.items():
        path = context.artifact_path(relative)
        path.write_text("{}" if path.suffix == ".json" else "content", encoding="utf-8")
        artifacts[name] = path
    context.succeed(artifacts)
    return context


def test_completed_run_uploads_manifest_after_artifacts_and_pointer_last(tmp_path: Path) -> None:
    context = _successful_run(tmp_path)
    client = FakeClient()
    initial = read_remote_latest_pointer(bucket_name="bucket", client=client)

    result = publish_completed_run_and_latest(
        context.run_dir,
        context.latest_pointer_path,
        initial,
        bucket_name="bucket",
        client=client,
    )

    order = client.fake_bucket.upload_order
    remote_manifest = f"output/ml/runs/{context.run_id}/manifest.json"
    assert result.required_artifacts_verified is True
    assert order.index(remote_manifest) > max(
        index
        for index, name in enumerate(order)
        if name.startswith(f"output/ml/runs/{context.run_id}/") and name != remote_manifest
    )
    assert order[-1] == "output/ml/latest_run.json"


def test_failed_run_cannot_be_published(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("features: [a]\n", encoding="utf-8")
    context = RunContext(tmp_path / "output" / "ml", config, repo_root=tmp_path)
    context.fail(ValueError("model failed"))
    client = FakeClient()
    with pytest.raises(ValueError, match="SUCCEEDED"):
        upload_completed_run(context.run_dir, bucket_name="bucket", client=client)
    assert client.fake_bucket.objects == {}


def test_upload_is_scoped_to_current_run_and_does_not_reupload_history(tmp_path: Path) -> None:
    context = _successful_run(tmp_path)
    client = FakeClient()
    client.fake_bucket.objects["output/ml/runs/old-run/manifest.json"] = {
        "data": b"old",
        "generation": 1,
    }
    upload_completed_run(context.run_dir, bucket_name="bucket", client=client)
    assert "output/ml/runs/old-run/manifest.json" not in client.fake_bucket.upload_order
    assert all(
        name.startswith(f"output/ml/runs/{context.run_id}/")
        for name in client.fake_bucket.upload_order
    )


def test_concurrent_latest_change_blocks_pointer_overwrite(tmp_path: Path) -> None:
    context = _successful_run(tmp_path)
    client = FakeClient()
    latest = client.fake_bucket.blob("output/ml/latest_run.json")
    previous_file = tmp_path / "previous.json"
    previous_file.write_text(json.dumps({"run_id": "previous"}), encoding="utf-8")
    latest.upload_from_filename(str(previous_file), if_generation_match=0)
    observed = read_remote_latest_pointer(bucket_name="bucket", client=client)

    concurrent_file = tmp_path / "concurrent.json"
    concurrent_file.write_text(json.dumps({"run_id": "concurrent"}), encoding="utf-8")
    latest.upload_from_filename(str(concurrent_file), if_generation_match=observed.generation)

    with pytest.raises(RuntimeError, match="generation precondition"):
        publish_latest_pointer(
            context.latest_pointer_path,
            observed,
            bucket_name="bucket",
            client=client,
        )
    assert read_remote_latest_pointer(bucket_name="bucket", client=client).run_id == "concurrent"


def test_initial_pointer_requires_nonexistent_remote_pointer(tmp_path: Path) -> None:
    context = _successful_run(tmp_path)
    client = FakeClient()
    publish_latest_pointer(
        context.latest_pointer_path,
        RemotePointerState(generation=None, run_id=None),
        bucket_name="bucket",
        client=client,
    )
    assert read_remote_latest_pointer(bucket_name="bucket", client=client).run_id == context.run_id
