"""Validation utilities for clustering configuration."""

from __future__ import annotations

from typing import Any


def validate_config(config: dict[str, Any]) -> None:
    forbidden = ["credentials", "destructive_ops"]
    for key in forbidden:
        if key in config:
            raise ValueError(f"Forbidden config key: {key}")

    required = ["features", "algorithms", "preprocessing", "input", "output"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if not isinstance(config["features"], list) or not config["features"]:
        raise ValueError("`features` must be a non-empty list")

    input_source = config["input"].get("source", "local_derived")
    allowed_sources = {"local_derived", "bigquery", "csv"}
    if input_source not in allowed_sources:
        raise ValueError(f"Unsupported input.source `{input_source}`. Allowed: {sorted(allowed_sources)}")

    output_root = config["output"].get("root_dir", "output/ml")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ValueError("`output.root_dir` must be a non-empty path string")

    legacy = config["output"].get("legacy_projection", {"enabled": True})
    if not isinstance(legacy, dict) or not isinstance(legacy.get("enabled", True), bool):
        raise ValueError("`output.legacy_projection.enabled` must be boolean")

    stability = config.get("stability", {"enabled": False})
    if not isinstance(stability, dict) or not isinstance(stability.get("enabled", False), bool):
        raise ValueError("`stability.enabled` must be boolean")
