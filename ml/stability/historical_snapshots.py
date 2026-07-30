"""Leakage-safe temporal clustering snapshots by round or month."""

from __future__ import annotations

import json
import re
from calendar import monthrange
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.clustering.algorithms import _fit_predict, _preprocess_matrix
from ml.clustering.evaluation import _evaluate_labels
from ml.features.build_team_features import build_features, build_team_level_from_match_features
from ml.stability.snapshot_models import SnapshotResult
from ml.utils.run_context import atomic_write_json, preprocessing_hash


COMPLETED_STATUSES = {"FT", "AET", "PEN"}


def parse_round_number(value: Any) -> int | None:
    """Parse the final integer from a round label without guessing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    match = re.search(r"(?:^|\D)(\d+)\s*$", str(value).strip())
    return int(match.group(1)) if match else None


def filter_completed_matches(match_df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """Return completed matches no later than an aware UTC cutoff."""
    df = match_df.copy()
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    cutoff_utc = pd.Timestamp(cutoff).tz_convert("UTC") if pd.Timestamp(cutoff).tzinfo else pd.Timestamp(cutoff, tz="UTC")
    status = df.get("status_short", pd.Series("FT", index=df.index)).astype(str).str.upper()
    complete = status.isin(COMPLETED_STATUSES)
    scored = pd.to_numeric(df.get("goals_home"), errors="coerce").notna() & pd.to_numeric(
        df.get("goals_away"), errors="coerce"
    ).notna()
    return df.loc[dates.notna() & (dates <= cutoff_utc) & complete & scored].copy()


def _month_end_cutoffs(dates: pd.Series) -> list[datetime]:
    valid = pd.to_datetime(dates, errors="coerce", utc=True).dropna()
    if valid.empty:
        return []
    periods = sorted({(value.year, value.month) for value in valid})
    return [
        datetime(year, month, monthrange(year, month)[1], 23, 59, 59, tzinfo=timezone.utc)
        for year, month in periods
    ]


def _round_cutoffs(match_df: pd.DataFrame, step: int) -> tuple[list[tuple[str, datetime]], list[str]]:
    parsed = match_df["round"].map(parse_round_number)
    warnings: list[str] = []
    failed = match_df.loc[parsed.isna(), "round"].dropna().astype(str).unique().tolist()
    if failed:
        warnings.append(f"Unparsed round labels were excluded: {sorted(failed)}")
    work = match_df.assign(_round_number=parsed, _date=pd.to_datetime(match_df["date"], errors="coerce", utc=True))
    rounds = sorted(int(value) for value in work["_round_number"].dropna().unique() if int(value) % step == 0)
    if work["_round_number"].notna().any():
        final_round = int(work["_round_number"].max())
        if final_round not in rounds:
            rounds.append(final_round)
    cutoffs: list[tuple[str, datetime]] = []
    for round_number in rounds:
        cutoff = work.loc[work["_round_number"] <= round_number, "_date"].max()
        if pd.notna(cutoff):
            cutoffs.append((f"round_{round_number:02d}", cutoff.to_pydatetime()))
    return cutoffs, warnings


def _snapshot_centroids(clusters: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    labels = clusters["raw_cluster_id"].to_numpy()
    for cluster_id in sorted(np.unique(labels)):
        vector = matrix[labels == cluster_id].mean(axis=0)
        row: dict[str, Any] = {"raw_cluster_id": int(cluster_id), "cluster_size": int((labels == cluster_id).sum())}
        row.update({f"component_{index + 1}": float(value) for index, value in enumerate(vector)})
        records.append(row)
    return pd.DataFrame(records)


def build_historical_snapshots(
    match_df: pd.DataFrame,
    config: dict[str, Any],
    selected_algorithm: str,
    selected_parameters: dict[str, Any],
    snapshots_root: Path,
) -> list[SnapshotResult]:
    """Build and persist configured temporal snapshots without future leakage."""
    stability = config.get("stability", {})
    snapshot_cfg = stability.get("snapshots", {})
    mode = stability.get("snapshot_mode", "round")
    minimum_matches = int(snapshot_cfg.get("minimum_matches_per_team", 8))
    warnings: list[str] = []
    if mode == "month":
        cutoffs = [(f"month_{cutoff:%Y_%m}", cutoff) for cutoff in _month_end_cutoffs(match_df["date"])]
    elif mode == "round":
        cutoffs, warnings = _round_cutoffs(match_df, int(snapshot_cfg.get("round_step", 5)))
    else:
        raise ValueError(f"Unsupported stability.snapshot_mode: {mode}")

    results: list[SnapshotResult] = []
    for snapshot_id, cutoff in cutoffs:
        filtered = filter_completed_matches(match_df, cutoff)
        if filtered.empty:
            continue
        team_features = build_team_level_from_match_features(filtered)
        excluded = team_features.loc[team_features["games"] < minimum_matches, ["team_id", "team_name", "games"]].copy()
        included = team_features.loc[team_features["games"] >= minimum_matches].copy()
        if len(included) < 2:
            continue
        features = build_features(included, config)
        matrix = _preprocess_matrix(features, config)
        _, labels = _fit_predict(selected_algorithm, selected_parameters, matrix)
        clusters = features.copy()
        clusters["raw_cluster_id"] = np.asarray(labels, dtype=int)
        clusters["cluster_label"] = clusters["raw_cluster_id"].map(lambda value: f"Raw cluster {value}")
        centroids = _snapshot_centroids(clusters, matrix)
        metric = _evaluate_labels(matrix, np.asarray(labels))

        snapshot_dir = snapshots_root / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        clusters_path = snapshot_dir / "clusters.csv"
        centroids_path = snapshot_dir / "centroids.csv"
        metadata_path = snapshot_dir / "snapshot_metadata.json"
        clusters.to_csv(clusters_path, index=False, encoding="utf-8-sig")
        centroids.to_csv(centroids_path, index=False, encoding="utf-8-sig")
        metadata = {
            "snapshot_id": snapshot_id,
            "snapshot_type": mode,
            "cutoff_date": cutoff.astimezone(timezone.utc).isoformat(),
            "season": sorted(str(value) for value in filtered["season"].dropna().unique()),
            "league_id": sorted(str(value) for value in filtered["league_id"].dropna().unique()),
            "match_count": int(filtered["fixture_id"].nunique()),
            "team_count": int(clusters["team_id"].nunique()),
            "excluded_teams": excluded.to_dict(orient="records"),
            "algorithm": selected_algorithm,
            "selected_parameters": selected_parameters,
            "quality_metrics": metric,
            "preprocessing": config.get("preprocessing", {}),
            "preprocessing_hash": preprocessing_hash(config.get("preprocessing", {})),
            "warnings": warnings,
        }
        atomic_write_json(metadata_path, metadata)
        results.append(
            SnapshotResult(
                snapshot_id=snapshot_id,
                snapshot_type=mode,
                cutoff_date=cutoff,
                directory=snapshot_dir,
                clusters=clusters,
                centroids=centroids,
                processed_matrix=matrix,
                metadata=metadata,
                warnings=tuple(warnings),
            )
        )
    return results
