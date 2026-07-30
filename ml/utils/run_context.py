"""Versioned, failure-safe lifecycle management for ML runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_schema_hash(feature_names: Iterable[str], dtypes: Mapping[str, Any]) -> str:
    """Hash ordered feature names and their effective pandas dtypes."""
    schema = [{"name": name, "dtype": str(dtypes.get(name, "unknown"))} for name in feature_names]
    return sha256_json(schema)


def preprocessing_hash(preprocessing: Mapping[str, Any]) -> str:
    """Hash preprocessing configuration deterministically."""
    return sha256_json(dict(preprocessing))


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically using a temporary sibling and ``os.replace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _safe_error_message(error: BaseException, limit: int = 1000) -> str:
    message = str(error).replace("\r", " ").replace("\n", " ").strip()
    return message[:limit] or error.__class__.__name__


@dataclass
class RunContext:
    """Own the local lifecycle and metadata of one versioned ML run."""

    output_root: Path
    config_path: Path
    analysis_type: str = "team_clustering_stability"
    snapshot_mode: str | None = None
    now: datetime | None = None
    repo_root: Path | None = None
    run_id: str = field(init=False)
    run_dir: Path = field(init=False)
    manifest_path: Path = field(init=False)
    latest_pointer_path: Path = field(init=False)
    started_at: datetime = field(init=False)
    manifest: dict[str, Any] = field(init=False)
    _finalized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.config_path = Path(self.config_path)
        self.started_at = (self.now or utc_now()).astimezone(timezone.utc)
        self.run_id = self._allocate_run_id(self.started_at)
        self.run_dir = self.output_root / "runs" / self.run_id
        self.manifest_path = self.run_dir / "manifest.json"
        self.latest_pointer_path = self.output_root / "latest_run.json"
        self._create_directories()
        config_hash = sha256_file(self.config_path) if self.config_path.is_file() else None
        root = self.repo_root or self.config_path.resolve().parents[2]
        self.manifest = {
            "run_id": self.run_id,
            "status": RUNNING,
            "run_date_utc": self.started_at.date().isoformat(),
            "started_at_utc": self.started_at.isoformat(),
            "completed_at_utc": None,
            "failed_at_utc": None,
            "duration_seconds": None,
            "data_as_of_date": None,
            "analysis_type": self.analysis_type,
            "snapshot_mode": self.snapshot_mode,
            "league_ids": [],
            "seasons": [],
            "match_count": None,
            "team_count": None,
            "input_source": None,
            "input_path_or_reference": None,
            "input_hash": None,
            "input_hash_status": "NOT_EVALUATED",
            "config_path": self._relative_or_posix(self.config_path, root),
            "config_hash": config_hash,
            "feature_schema_hash": None,
            "preprocessing_hash": None,
            "algorithm": None,
            "candidate_id": None,
            "selected_parameters": {},
            "git_commit": _git_commit(root),
            "python_version": platform.python_version(),
            "required_artifacts": [],
            "optional_artifacts": [],
            "warnings": [],
            "error": None,
        }
        atomic_write_json(self.manifest_path, self.manifest)

    def _allocate_run_id(self, timestamp: datetime) -> str:
        base = timestamp.strftime("%Y%m%dT%H%M%SZ")
        runs_root = self.output_root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        candidate = base
        suffix = 0
        while (runs_root / candidate).exists():
            suffix += 1
            candidate = f"{base}-{suffix:02d}"
        return candidate

    def _create_directories(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        for relative in (
            "clusters",
            "metrics",
            "plots",
            "report",
            "stability/snapshots",
            "stability/plots",
        ):
            (self.run_dir / relative).mkdir(parents=True, exist_ok=False)

    @staticmethod
    def _relative_or_posix(path: Path, base: Path) -> str:
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def artifact_path(self, relative_path: str) -> Path:
        """Resolve a run-relative artifact path without escaping the run."""
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Artifact path must be run-relative: {relative_path}")
        return self.run_dir / relative

    def update_metadata(self, **values: Any) -> None:
        """Update the RUNNING manifest with verified metadata."""
        self._require_running()
        self.manifest.update(values)
        atomic_write_json(self.manifest_path, self.manifest)

    def add_warning(self, warning: str) -> None:
        self._require_running()
        if warning not in self.manifest["warnings"]:
            self.manifest["warnings"].append(warning)
            atomic_write_json(self.manifest_path, self.manifest)

    def succeed(
        self,
        required_artifacts: Mapping[str, Path],
        optional_artifacts: Mapping[str, Path] | None = None,
    ) -> None:
        """Validate artifacts, finalize the manifest, then advance latest atomically."""
        self._require_running()
        missing = [name for name, path in required_artifacts.items() if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing required run artifacts: {', '.join(sorted(missing))}")

        completed = utc_now()
        self.manifest.update(
            {
                "status": SUCCEEDED,
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": round((completed - self.started_at).total_seconds(), 6),
                "required_artifacts": [
                    {"name": name, "path": self._run_relative(path)}
                    for name, path in sorted(required_artifacts.items())
                ],
                "optional_artifacts": [
                    {"name": name, "path": self._run_relative(path)}
                    for name, path in sorted((optional_artifacts or {}).items())
                    if Path(path).is_file()
                ],
                "error": None,
            }
        )
        atomic_write_json(self.manifest_path, self.manifest)
        pointer = {
            "run_id": self.run_id,
            "status": SUCCEEDED,
            "run_path": (Path("runs") / self.run_id).as_posix(),
            "manifest_path": (Path("runs") / self.run_id / "manifest.json").as_posix(),
            "updated_at_utc": completed.isoformat(),
        }
        atomic_write_json(self.latest_pointer_path, pointer)
        self._finalized = True

    def fail(self, error: BaseException) -> None:
        """Finalize the run as FAILED without changing the latest pointer."""
        if self._finalized or self.manifest.get("status") != RUNNING:
            return
        failed = utc_now()
        self.manifest.update(
            {
                "status": FAILED,
                "failed_at_utc": failed.isoformat(),
                "duration_seconds": round((failed - self.started_at).total_seconds(), 6),
                "error": {
                    "type": error.__class__.__name__,
                    "message": _safe_error_message(error),
                },
            }
        )
        atomic_write_json(self.manifest_path, self.manifest)
        self._finalized = True

    def _run_relative(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.run_dir.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(f"Artifact is outside run directory: {path}") from error

    def _require_running(self) -> None:
        if self._finalized or self.manifest.get("status") != RUNNING:
            raise RuntimeError(f"Run {self.run_id} is immutable after finalization")
