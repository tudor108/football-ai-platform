"""Regenerate markdown report from existing ML metric artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from ml.reporting.cluster_reporting import write_cluster_report


def regenerate_report(
    metrics_path: str | Path = "output/ml/metrics/metrics.json",
    summary_path: str | Path = "output/ml/metrics/best_model_summary.json",
    interpretation_path: str | Path = "output/ml/metrics/cluster_interpretation.json",
    report_path: str | Path = "output/ml/report/cluster_report.md",
) -> None:
    metrics_file = Path(metrics_path)
    summary_file = Path(summary_path)
    interpretation_file = Path(interpretation_path)

    if not metrics_file.exists() or not summary_file.exists() or not interpretation_file.exists():
        raise FileNotFoundError("Missing one or more required ML metric files to regenerate report.")

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    best_summary = json.loads(summary_file.read_text(encoding="utf-8"))
    interpretations = json.loads(interpretation_file.read_text(encoding="utf-8"))

    write_cluster_report(report_path, best_summary, metrics, interpretations)
