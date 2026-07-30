from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ml.utils.run_context import FAILED, RUNNING, SUCCEEDED, RunContext, sha256_file, sha256_json


def make_context(tmp_path: Path, when: datetime | None = None) -> RunContext:
    config = tmp_path / "config.yaml"
    config.write_text("features: [a]\n", encoding="utf-8")
    return RunContext(tmp_path / "output" / "ml", config, now=when, repo_root=tmp_path)


def test_run_id_is_windows_safe_utc_and_collision_is_deterministic(tmp_path: Path) -> None:
    when = datetime(2026, 7, 22, 10, 35, tzinfo=timezone.utc)
    first = make_context(tmp_path, when)
    second = make_context(tmp_path, when)
    assert re.fullmatch(r"\d{8}T\d{6}Z", first.run_id)
    assert second.run_id == f"{first.run_id}-01"
    assert first.run_dir != second.run_dir


def test_hashes_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("same bytes", encoding="utf-8")
    assert sha256_file(source) == sha256_file(source)
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_running_to_succeeded_updates_latest_and_becomes_immutable(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    artifact = context.artifact_path("clusters/clusters.csv")
    artifact.write_text("team_id,cluster\n1,0\n", encoding="utf-8")
    assert context.manifest["status"] == RUNNING
    context.succeed({"clusters": artifact})
    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    latest = json.loads(context.latest_pointer_path.read_text(encoding="utf-8"))
    assert manifest["status"] == SUCCEEDED
    assert latest["run_id"] == context.run_id
    with pytest.raises(RuntimeError):
        context.update_metadata(team_count=10)


def test_failed_run_does_not_replace_previous_latest(tmp_path: Path) -> None:
    previous = make_context(tmp_path)
    artifact = previous.artifact_path("clusters/clusters.csv")
    artifact.write_text("ok", encoding="utf-8")
    previous.succeed({"clusters": artifact})
    latest_before = previous.latest_pointer_path.read_bytes()

    failed = make_context(tmp_path)
    failed.fail(ValueError("safe failure"))
    manifest = json.loads(failed.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == FAILED
    assert failed.latest_pointer_path.read_bytes() == latest_before


def test_missing_required_artifact_prevents_success(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    with pytest.raises(FileNotFoundError):
        context.succeed({"missing": context.artifact_path("metrics/missing.json")})
    assert json.loads(context.manifest_path.read_text(encoding="utf-8"))["status"] == RUNNING
