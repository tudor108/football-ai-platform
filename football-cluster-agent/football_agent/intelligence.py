"""Pure business-facing football intelligence helpers.

Cloud adapters load pandas DataFrames and delegate here.  Every score returned
by this module is explainable and explicitly labelled as either a probability,
an index, or a heuristic scenario estimate.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .analytics import (
    CLUSTERING_FEATURES,
    personalized_similarity,
    poisson_outcome_summary,
    predict_match_from_history,
)


def _normalize(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return " ".join(ascii_value.casefold().split())


def _plain(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value.item() if hasattr(value, "item") else value


def _require(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _resolve_name(values: Iterable[Any], requested: str, label: str = "Team") -> str:
    normalized_requested = _normalize(requested)
    candidates: dict[str, set[str]] = {}
    for value in values:
        if pd.isna(value):
            continue
        original = str(value)
        candidates.setdefault(_normalize(original), set()).add(original)
    exact = candidates.get(normalized_requested)
    if exact:
        if len(exact) > 1:
            raise ValueError(f"Ambiguous {label.lower()} name: {requested}")
        return next(iter(exact))
    partial = sorted(
        {
            original
            for normalized, originals in candidates.items()
            if normalized_requested in normalized
            for original in originals
        }
    )
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ValueError(f"Ambiguous {label.lower()} name: {requested}; matches {partial}")
    raise ValueError(f"{label} not found: {requested}")


def _match_frame(matches: pd.DataFrame, completed_only: bool = True) -> pd.DataFrame:
    _require(
        matches,
        {
            "fixture_id",
            "date",
            "home_team_name",
            "away_team_name",
            "goals_home",
            "goals_away",
        },
        "Match data",
    )
    frame = matches.copy()
    # API snapshots may mix second and microsecond precision in the same file.
    frame["date_dt"] = pd.to_datetime(
        frame["date"], errors="coerce", utc=True, format="mixed"
    )
    frame["goals_home"] = pd.to_numeric(frame["goals_home"], errors="coerce")
    frame["goals_away"] = pd.to_numeric(frame["goals_away"], errors="coerce")
    frame = frame.dropna(subset=["date_dt"])
    if completed_only:
        frame = frame.dropna(subset=["goals_home", "goals_away"])
    return frame.sort_values("date_dt")


def _team_long_history(frame: pd.DataFrame, team_name: str) -> pd.DataFrame:
    home = frame.loc[frame["home_team_name"] == team_name].copy()
    home["opponent"] = home["away_team_name"]
    home["is_home"] = True
    home["goals_for"] = home["goals_home"]
    home["goals_against"] = home["goals_away"]
    away = frame.loc[frame["away_team_name"] == team_name].copy()
    away["opponent"] = away["home_team_name"]
    away["is_home"] = False
    away["goals_for"] = away["goals_away"]
    away["goals_against"] = away["goals_home"]
    history = pd.concat([home, away], ignore_index=True).sort_values("date_dt")
    history["points"] = np.select(
        [
            history["goals_for"] > history["goals_against"],
            history["goals_for"] == history["goals_against"],
        ],
        [3.0, 1.0],
        default=0.0,
    )
    history["result"] = np.select(
        [
            history["goals_for"] > history["goals_against"],
            history["goals_for"] == history["goals_against"],
        ],
        ["W", "D"],
        default="L",
    )
    return history


def _team_form_profile(frame: pd.DataFrame, team_name: str, window: int = 5) -> dict[str, Any]:
    history = _team_long_history(frame, team_name)
    if history.empty:
        raise ValueError(f"No completed matches found for {team_name}.")
    recent = history.tail(max(1, int(window)))
    return {
        "matches_used": int(len(recent)),
        "points_per_game": round(float(recent["points"].mean()), 3),
        "goals_for_per_game": round(float(recent["goals_for"].mean()), 3),
        "goals_against_per_game": round(float(recent["goals_against"].mean()), 3),
        "win_rate": round(float((recent["result"] == "W").mean()), 3),
        "form": "".join(recent["result"].astype(str).tolist()),
        "latest_results": [
            {
                "date_utc": row["date_dt"].isoformat(),
                "opponent": str(row["opponent"]),
                "venue": "home" if bool(row["is_home"]) else "away",
                "score": f"{int(row['goals_for'])}-{int(row['goals_against'])}",
                "result": str(row["result"]),
            }
            for _, row in recent.sort_values("date_dt", ascending=False).iterrows()
        ],
    }


def _cluster_summary(clusters: pd.DataFrame, team_name: str) -> dict[str, Any] | None:
    if clusters.empty or not {"team_name", "cluster"}.issubset(clusters.columns):
        return None
    normalized = clusters["team_name"].astype(str).map(_normalize)
    rows = clusters.loc[normalized == _normalize(team_name)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "cluster_id": int(row["cluster"]),
        "cluster_label": _plain(row.get("cluster_label")),
        "assignment_strength": _plain(row.get("assignment_strength")),
    }


def _numeric_stat_value(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip().replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _team_stat_averages(
    team_stats: pd.DataFrame,
    team_name: str,
    recent_fixture_ids: set[Any],
) -> dict[str, float]:
    required = {"fixture_id", "team_name", "stat_type", "stat_value"}
    if team_stats.empty or not required.issubset(team_stats.columns):
        return {}
    rows = team_stats.loc[
        (team_stats["team_name"].astype(str).map(_normalize) == _normalize(team_name))
        & team_stats["fixture_id"].isin(recent_fixture_ids)
    ].copy()
    if rows.empty:
        return {}
    rows["numeric_value"] = rows["stat_value"].map(_numeric_stat_value)
    preferred = {
        "Shots on Goal",
        "Shots off Goal",
        "Total Shots",
        "Ball Possession",
        "Total passes",
        "Passes accurate",
        "Corner Kicks",
        "Fouls",
    }
    grouped = (
        rows.loc[rows["stat_type"].isin(preferred)]
        .dropna(subset=["numeric_value"])
        .groupby("stat_type")["numeric_value"]
        .mean()
    )
    return {str(key): round(float(value), 3) for key, value in grouped.items()}


def _key_players(top_scorers: pd.DataFrame, team_name: str, limit: int = 3) -> list[dict[str, Any]]:
    required = {"team_name", "player_name", "goals", "assists", "minutes"}
    if top_scorers.empty or not required.issubset(top_scorers.columns):
        return []
    rows = top_scorers.loc[
        top_scorers["team_name"].astype(str).map(_normalize) == _normalize(team_name)
    ].copy()
    for column in ("goals", "assists", "minutes", "goals_per_90", "rating_avg"):
        if column in rows.columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["contributions"] = rows["goals"].fillna(0) + rows["assists"].fillna(0)
    rows = rows.sort_values(["contributions", "minutes"], ascending=False).head(limit)
    return [
        {
            "player_name": str(row["player_name"]),
            "goals": round(float(row["goals"]), 2) if pd.notna(row["goals"]) else None,
            "assists": round(float(row["assists"]), 2) if pd.notna(row["assists"]) else None,
            "minutes": round(float(row["minutes"]), 1) if pd.notna(row["minutes"]) else None,
            "goals_per_90": (
                round(float(row["goals_per_90"]), 3)
                if "goals_per_90" in rows.columns and pd.notna(row.get("goals_per_90"))
                else None
            ),
        }
        for _, row in rows.iterrows()
    ]


def build_opponent_dossier(
    matches: pd.DataFrame,
    clusters: pd.DataFrame,
    home_team: str,
    away_team: str,
    team_stats: pd.DataFrame | None = None,
    top_scorers: pd.DataFrame | None = None,
    recent_window: int = 5,
) -> dict[str, Any]:
    """Build a compact pre-match dossier from available evidence."""
    frame = _match_frame(matches)
    all_names = pd.concat([frame["home_team_name"], frame["away_team_name"]])
    resolved_home = _resolve_name(all_names, home_team)
    resolved_away = _resolve_name(all_names, away_team)
    if resolved_home == resolved_away:
        raise ValueError("Home and away team must be different.")

    home_profile = _team_form_profile(frame, resolved_home, recent_window)
    away_profile = _team_form_profile(frame, resolved_away, recent_window)
    home_ids = set(_team_long_history(frame, resolved_home).tail(recent_window)["fixture_id"])
    away_ids = set(_team_long_history(frame, resolved_away).tail(recent_window)["fixture_id"])
    h2h = frame.loc[
        ((frame["home_team_name"] == resolved_home) & (frame["away_team_name"] == resolved_away))
        | ((frame["home_team_name"] == resolved_away) & (frame["away_team_name"] == resolved_home))
    ].tail(5)

    forecast = predict_match_from_history(frame, resolved_home, resolved_away)
    newest = frame["date_dt"].max()
    return {
        "fixture": {"home_team": resolved_home, "away_team": resolved_away},
        "generated_from_data_through_utc": newest.isoformat(),
        "team_profiles": {
            "home": {
                **home_profile,
                "official_cluster": _cluster_summary(clusters, resolved_home),
                "recent_match_stat_averages": _team_stat_averages(
                    team_stats if team_stats is not None else pd.DataFrame(),
                    resolved_home,
                    home_ids,
                ),
                "key_players": _key_players(
                    top_scorers if top_scorers is not None else pd.DataFrame(),
                    resolved_home,
                ),
            },
            "away": {
                **away_profile,
                "official_cluster": _cluster_summary(clusters, resolved_away),
                "recent_match_stat_averages": _team_stat_averages(
                    team_stats if team_stats is not None else pd.DataFrame(),
                    resolved_away,
                    away_ids,
                ),
                "key_players": _key_players(
                    top_scorers if top_scorers is not None else pd.DataFrame(),
                    resolved_away,
                ),
            },
        },
        "head_to_head": [
            {
                "date_utc": row["date_dt"].isoformat(),
                "home_team": str(row["home_team_name"]),
                "away_team": str(row["away_team_name"]),
                "score": f"{int(row['goals_home'])}-{int(row['goals_away'])}",
            }
            for _, row in h2h.sort_values("date_dt", ascending=False).iterrows()
        ],
        "forecast": forecast,
        "limitations": [
            "This dossier uses stored match data, not live observations.",
            "The forecast remains uncalibrated and is not betting advice.",
            "Video, confirmed lineup, injury, and training-load evidence is included only when supplied by a source.",
        ],
    }


def evaluate_prediction_records(records: pd.DataFrame) -> dict[str, Any]:
    """Evaluate stored 1/X/2 predictions with proper probabilistic metrics."""
    required = {"home_win", "draw", "away_win", "actual_result"}
    _require(records, required, "Prediction records")
    frame = records.copy()
    probabilities = frame[["home_win", "draw", "away_win"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_results = frame["actual_result"].isin(["H", "D", "A"])
    valid_probs = probabilities.notna().all(axis=1) & (probabilities >= 0).all(axis=1)
    frame = frame.loc[valid_results & valid_probs].copy()
    probabilities = probabilities.loc[frame.index].to_numpy(dtype=float)
    if frame.empty:
        raise ValueError("No valid prediction records are available for evaluation.")
    row_sums = probabilities.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("Prediction probabilities must have positive mass.")
    probabilities = probabilities / row_sums[:, None]
    label_to_index = {"H": 0, "D": 1, "A": 2}
    actual_indices = np.array([label_to_index[value] for value in frame["actual_result"]])
    targets = np.eye(3)[actual_indices]
    brier = float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))
    chosen = np.clip(probabilities[np.arange(len(frame)), actual_indices], 1e-12, 1.0)
    log_loss = float(-np.mean(np.log(chosen)))
    predicted_indices = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predicted_indices == actual_indices))
    confidence = probabilities.max(axis=1)
    correct = (predicted_indices == actual_indices).astype(float)

    calibration: list[dict[str, Any]] = []
    for lower, upper in ((0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.75), (0.75, 1.01)):
        mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        calibration.append(
            {
                "confidence_range": f"{lower:.2f}-{min(upper, 1.0):.2f}",
                "predictions": int(mask.sum()),
                "mean_confidence": round(float(confidence[mask].mean()), 4),
                "observed_accuracy": round(float(correct[mask].mean()), 4),
            }
        )
    return {
        "evaluated_predictions": int(len(frame)),
        "multiclass_brier_score": round(brier, 5),
        "log_loss": round(log_loss, 5),
        "top_choice_accuracy": round(accuracy, 4),
        "calibration_buckets": calibration,
        "interpretation": {
            "brier": "Lower is better; zero is perfect.",
            "log_loss": "Lower is better and confident wrong forecasts are penalized strongly.",
            "accuracy": "Useful but insufficient without calibration metrics.",
        },
    }


def backtest_match_forecasts(
    matches: pd.DataFrame,
    min_prior_matches: int = 5,
    max_evaluated: int = 200,
) -> dict[str, Any]:
    """Walk forward through history without letting a forecast see its outcome."""
    frame = _match_frame(matches)
    records: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        prior = frame.loc[frame["date_dt"] < row["date_dt"]].copy()
        if prior.empty:
            continue
        home_count = int(
            ((prior["home_team_name"] == row["home_team_name"]) | (prior["away_team_name"] == row["home_team_name"])).sum()
        )
        away_count = int(
            ((prior["home_team_name"] == row["away_team_name"]) | (prior["away_team_name"] == row["away_team_name"])).sum()
        )
        if min(home_count, away_count) < min_prior_matches:
            continue
        try:
            forecast = predict_match_from_history(
                prior,
                str(row["home_team_name"]),
                str(row["away_team_name"]),
            )
        except ValueError:
            continue
        probabilities = forecast["outcome_probabilities"]
        actual_result = (
            "H"
            if row["goals_home"] > row["goals_away"]
            else "D"
            if row["goals_home"] == row["goals_away"]
            else "A"
        )
        records.append(
            {
                "fixture_id": _plain(row["fixture_id"]),
                "date_utc": row["date_dt"].isoformat(),
                "home_team": str(row["home_team_name"]),
                "away_team": str(row["away_team_name"]),
                **probabilities,
                "actual_result": actual_result,
            }
        )
    if max_evaluated > 0:
        records = records[-int(max_evaluated) :]
    if not records:
        raise ValueError("Not enough chronological history for a walk-forward backtest.")
    evaluation = evaluate_prediction_records(pd.DataFrame(records))
    return {
        **evaluation,
        "method": "Walk-forward backtest; every forecast uses only earlier matches.",
        "sample_recent_predictions": records[-10:],
        "warnings": [
            "The evaluated Poisson model is still uncalibrated.",
            "Repeated matches are not statistically independent and this is an exploratory audit.",
        ],
    }


def generate_business_alerts(
    matches: pd.DataFrame,
    clusters: pd.DataFrame,
    previous_clusters: pd.DataFrame | None = None,
    as_of: datetime | None = None,
    stale_after_days: int = 2,
    form_delta_threshold: float = 0.75,
) -> dict[str, Any]:
    """Detect actionable data, form, volatility, and cluster-movement signals."""
    frame = _match_frame(matches)
    reference = as_of or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    latest = frame["date_dt"].max().to_pydatetime()
    age_days = max(0, int((reference.astimezone(timezone.utc) - latest).total_seconds() // 86400))
    alerts: list[dict[str, Any]] = []
    if age_days > stale_after_days:
        alerts.append(
            {
                "type": "data_stale",
                "severity": "high",
                "entity": "dataset",
                "message": f"Newest completed match data is {age_days} days old.",
                "evidence": {"latest_match_utc": latest.isoformat()},
            }
        )

    names = sorted(set(frame["home_team_name"]) | set(frame["away_team_name"]))
    volatility_rows: list[tuple[str, float]] = []
    for team in names:
        history = _team_long_history(frame, team)
        if len(history) < 8:
            continue
        season_ppg = float(history["points"].mean())
        recent_ppg = float(history.tail(5)["points"].mean())
        delta = recent_ppg - season_ppg
        if abs(delta) >= form_delta_threshold:
            alerts.append(
                {
                    "type": "form_change",
                    "severity": "medium",
                    "entity": team,
                    "direction": "up" if delta > 0 else "down",
                    "message": f"Recent points per game differs from the season rate by {delta:+.2f}.",
                    "evidence": {
                        "recent_ppg": round(recent_ppg, 3),
                        "season_ppg": round(season_ppg, 3),
                        "recent_matches": 5,
                    },
                }
            )
        volatility_rows.append((team, float((history["goals_for"] - history["goals_against"]).std(ddof=0))))
    if volatility_rows:
        cutoff = float(np.quantile([value for _, value in volatility_rows], 0.75))
        for team, volatility in volatility_rows:
            if volatility >= cutoff:
                alerts.append(
                    {
                        "type": "high_result_volatility",
                        "severity": "low",
                        "entity": team,
                        "message": "Goal-difference volatility is in the highest league quartile.",
                        "evidence": {
                            "goal_difference_std": round(volatility, 3),
                            "league_upper_quartile": round(cutoff, 3),
                        },
                    }
                )

    if (
        previous_clusters is not None
        and not previous_clusters.empty
        and {"team_name", "cluster"}.issubset(clusters.columns)
        and {"team_name", "cluster"}.issubset(previous_clusters.columns)
    ):
        current = clusters[["team_name", "cluster"]].copy()
        previous = previous_clusters[["team_name", "cluster"]].copy()
        current["key"] = current["team_name"].map(_normalize)
        previous["key"] = previous["team_name"].map(_normalize)
        comparison = current.merge(previous, on="key", suffixes=("_current", "_previous"))
        for _, row in comparison.loc[comparison["cluster_current"] != comparison["cluster_previous"]].iterrows():
            alerts.append(
                {
                    "type": "cluster_movement",
                    "severity": "medium",
                    "entity": str(row["team_name_current"]),
                    "message": "Official cluster assignment changed between compatible runs.",
                    "evidence": {
                        "previous_cluster": int(row["cluster_previous"]),
                        "current_cluster": int(row["cluster_current"]),
                    },
                    "warning": "Movement is descriptive and must not be interpreted as causality.",
                }
            )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda item: (severity_order[item["severity"]], str(item["entity"])))
    return {
        "generated_at_utc": reference.astimezone(timezone.utc).isoformat(),
        "latest_data_utc": latest.isoformat(),
        "alert_count": len(alerts),
        "alerts": alerts,
        "rules": {
            "stale_after_days": stale_after_days,
            "form_delta_threshold": form_delta_threshold,
            "volatility_rule": "highest league quartile",
        },
    }


def _aggregate_player_profiles(player_stats: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fixture_id",
        "team_name",
        "player_id",
        "player_name",
        "position",
        "minutes",
        "rating",
    }
    _require(player_stats, required, "Player match statistics")
    frame = player_stats.copy()
    numeric_columns = [
        "minutes",
        "rating",
        "goals_total",
        "goals_assists",
        "shots_on",
        "passes_key",
        "passes_accuracy",
        "tackles_total",
        "tackles_blocks",
        "tackles_interceptions",
        "duels_total",
        "duels_won",
        "dribbles_attempts",
        "dribbles_success",
    ]
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby(
        ["team_name", "player_id", "player_name", "position"], dropna=False
    )
    profiles = grouped.agg(
        appearances=("fixture_id", "nunique"),
        minutes=("minutes", "sum"),
        rating=("rating", "mean"),
        rating_std=("rating", "std"),
        goals=("goals_total", "sum"),
        assists=("goals_assists", "sum"),
        shots_on=("shots_on", "sum"),
        key_passes=("passes_key", "sum"),
        pass_accuracy=("passes_accuracy", "mean"),
        tackles=("tackles_total", "sum"),
        blocks=("tackles_blocks", "sum"),
        interceptions=("tackles_interceptions", "sum"),
        duels=("duels_total", "sum"),
        duels_won=("duels_won", "sum"),
        dribbles=("dribbles_attempts", "sum"),
        dribbles_success=("dribbles_success", "sum"),
    ).reset_index()
    denominator = profiles["minutes"].replace(0, np.nan)
    for column in ("goals", "assists", "shots_on", "key_passes", "tackles", "blocks", "interceptions"):
        profiles[f"{column}_per90"] = profiles[column] * 90.0 / denominator
    profiles["duel_win_rate"] = profiles["duels_won"] / profiles["duels"].replace(0, np.nan)
    profiles["dribble_success_rate"] = profiles["dribbles_success"] / profiles["dribbles"].replace(0, np.nan)
    profiles["rating_std"] = profiles["rating_std"].fillna(0.0)
    return profiles


def _percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    fill = numeric.median() if numeric.notna().any() else 0.0
    ranks = numeric.fillna(fill).rank(pct=True, method="average")
    return ranks if higher_is_better else 1.0 - ranks


def rank_player_team_fit(
    player_stats: pd.DataFrame,
    clusters: pd.DataFrame,
    team_name: str,
    position: str = "",
    attack_preference: float = 0.0,
    defense_preference: float = 0.0,
    creation_preference: float = 0.0,
    reliability_preference: float = 0.0,
    min_minutes: float = 180.0,
    top_n: int = 10,
) -> dict[str, Any]:
    """Rank external players by transparent role and team-style fit indexes."""
    _require(clusters, {"team_name"}, "Cluster data")
    profiles = _aggregate_player_profiles(player_stats)
    resolved_team = _resolve_name(profiles["team_name"], team_name)
    preferences = {
        "attack": float(attack_preference),
        "defense": float(defense_preference),
        "creation": float(creation_preference),
        "reliability": float(reliability_preference),
    }
    if any(value < -1 or value > 1 for value in preferences.values()):
        raise ValueError("Player-fit preferences must be between -1 and 1.")

    candidates = profiles.loc[
        (profiles["team_name"].map(_normalize) != _normalize(resolved_team))
        & (profiles["minutes"] >= float(min_minutes))
    ].copy()
    if position.strip():
        aliases = {
            "gk": "g",
            "goalkeeper": "g",
            "defender": "d",
            "defence": "d",
            "midfielder": "m",
            "midfield": "m",
            "forward": "f",
            "attacker": "f",
            "striker": "f",
        }
        requested_position = aliases.get(_normalize(position), _normalize(position)[:1])
        candidates = candidates.loc[
            candidates["position"].astype(str).map(_normalize).str[:1] == requested_position
        ]
    if candidates.empty:
        raise ValueError("No players satisfy the requested position and minimum minutes.")

    candidates["attack_index"] = (
        _percentile(candidates["goals_per90"]) * 0.45
        + _percentile(candidates["shots_on_per90"]) * 0.25
        + _percentile(candidates["dribble_success_rate"]) * 0.15
        + _percentile(candidates["rating"]) * 0.15
    )
    candidates["defense_index"] = (
        _percentile(candidates["tackles_per90"]) * 0.35
        + _percentile(candidates["interceptions_per90"]) * 0.30
        + _percentile(candidates["blocks_per90"]) * 0.15
        + _percentile(candidates["duel_win_rate"]) * 0.20
    )
    candidates["creation_index"] = (
        _percentile(candidates["assists_per90"]) * 0.35
        + _percentile(candidates["key_passes_per90"]) * 0.35
        + _percentile(candidates["pass_accuracy"]) * 0.15
        + _percentile(candidates["rating"]) * 0.15
    )
    candidates["reliability_index"] = (
        _percentile(candidates["minutes"]) * 0.35
        + _percentile(candidates["rating"]) * 0.40
        + _percentile(candidates["rating_std"], higher_is_better=False) * 0.25
    )

    base_weights = {"attack": 0.25, "defense": 0.25, "creation": 0.25, "reliability": 0.25}
    team_rows = clusters.loc[
        clusters["team_name"].astype(str).map(_normalize) == _normalize(resolved_team)
    ]
    if not team_rows.empty:
        team_row = team_rows.iloc[0]
        attack_value = pd.to_numeric(pd.Series([team_row.get("attack_strength")]), errors="coerce").iloc[0]
        defense_value = pd.to_numeric(pd.Series([team_row.get("defense_strength")]), errors="coerce").iloc[0]
        if pd.notna(attack_value) and pd.notna(defense_value):
            total = abs(float(attack_value)) + abs(float(defense_value))
            if total > 0:
                base_weights["attack"] += 0.15 * abs(float(attack_value)) / total
                base_weights["defense"] += 0.15 * abs(float(defense_value)) / total

    adjusted = {
        key: max(0.05, weight * (1.0 + preferences[key]))
        for key, weight in base_weights.items()
    }
    weight_total = sum(adjusted.values())
    weights = {key: value / weight_total for key, value in adjusted.items()}
    candidates["fit_index"] = 100.0 * sum(
        candidates[f"{key}_index"] * weight for key, weight in weights.items()
    )
    candidates = candidates.sort_values("fit_index", ascending=False).head(max(1, min(int(top_n), 50)))
    return {
        "target_team": resolved_team,
        "position_filter": position or None,
        "minimum_minutes": float(min_minutes),
        "weights": {key: round(value, 4) for key, value in weights.items()},
        "official_cluster_unchanged": _cluster_summary(clusters, resolved_team),
        "recommended_players": [
            {
                "player_id": _plain(row["player_id"]),
                "player_name": str(row["player_name"]),
                "current_team": str(row["team_name"]),
                "position": _plain(row["position"]),
                "fit_index": round(float(row["fit_index"]), 1),
                "minutes": round(float(row["minutes"]), 1),
                "rating": round(float(row["rating"]), 2) if pd.notna(row["rating"]) else None,
                "dimension_indexes": {
                    key: round(float(row[f"{key}_index"] * 100), 1)
                    for key in weights
                },
            }
            for _, row in candidates.iterrows()
        ],
        "warnings": [
            "Fit indexes are relative ranking scores, not transfer-success probabilities.",
            "No transfer fee, contract, personality, medical, or tracking data is included.",
            "Small or incomplete player samples can materially change the ranking.",
        ],
    }


def _player_impacts(player_stats: pd.DataFrame, team_name: str) -> pd.DataFrame:
    profiles = _aggregate_player_profiles(player_stats)
    team = profiles.loc[profiles["team_name"].map(_normalize) == _normalize(team_name)].copy()
    if team.empty:
        return team
    attack_raw = team["goals"].fillna(0) + 0.7 * team["assists"].fillna(0) + 0.1 * team["key_passes"].fillna(0)
    defense_raw = team["tackles"].fillna(0) + team["interceptions"].fillna(0) + 0.5 * team["blocks"].fillna(0)
    team["attack_share"] = attack_raw / max(float(attack_raw.sum()), 1.0)
    team["defense_share"] = defense_raw / max(float(defense_raw.sum()), 1.0)
    team["availability_weight"] = team["minutes"] / max(float(team["minutes"].max()), 1.0)
    return team


def simulate_lineup_absences(
    matches: pd.DataFrame,
    player_stats: pd.DataFrame,
    home_team: str,
    away_team: str,
    missing_home_players: Sequence[str] = (),
    missing_away_players: Sequence[str] = (),
) -> dict[str, Any]:
    """Apply bounded, transparent player-impact scenarios to a baseline forecast."""
    baseline = predict_match_from_history(matches, home_team, away_team)
    resolved_home = baseline["fixture"]["home_team"]
    resolved_away = baseline["fixture"]["away_team"]
    home_impacts = _player_impacts(player_stats, resolved_home)
    away_impacts = _player_impacts(player_stats, resolved_away)

    def selected_impacts(frame: pd.DataFrame, names: Sequence[str]) -> tuple[list[dict[str, Any]], float, float]:
        selected: list[dict[str, Any]] = []
        attack = 0.0
        defense = 0.0
        for requested in names:
            resolved = _resolve_name(frame["player_name"], requested, label="Player")
            row = frame.loc[frame["player_name"].astype(str).map(_normalize) == _normalize(resolved)].iloc[0]
            attack_effect = min(0.20, float(row["attack_share"]) * float(row["availability_weight"]) * 0.6)
            defense_effect = min(0.15, float(row["defense_share"]) * float(row["availability_weight"]) * 0.5)
            attack += attack_effect
            defense += defense_effect
            selected.append(
                {
                    "player_name": resolved,
                    "estimated_attack_xg_reduction_share": round(attack_effect, 4),
                    "estimated_defensive_xg_increase_share": round(defense_effect, 4),
                }
            )
        return selected, min(0.35, attack), min(0.25, defense)

    home_selected, home_attack_loss, home_defense_loss = selected_impacts(home_impacts, missing_home_players)
    away_selected, away_attack_loss, away_defense_loss = selected_impacts(away_impacts, missing_away_players)
    base_home_xg = float(baseline["expected_goals"]["home"])
    base_away_xg = float(baseline["expected_goals"]["away"])
    scenario_home_xg = max(0.1, base_home_xg * (1.0 - home_attack_loss) * (1.0 + away_defense_loss))
    scenario_away_xg = max(0.1, base_away_xg * (1.0 - away_attack_loss) * (1.0 + home_defense_loss))
    scenario_markets = poisson_outcome_summary(scenario_home_xg, scenario_away_xg)
    deltas = {
        key: round(
            scenario_markets["outcome_probabilities"][key]
            - baseline["outcome_probabilities"][key],
            4,
        )
        for key in ("home_win", "draw", "away_win")
    }
    return {
        "fixture": baseline["fixture"],
        "baseline": baseline,
        "scenario": {
            "missing_home_players": home_selected,
            "missing_away_players": away_selected,
            "expected_goals": {
                "home": round(scenario_home_xg, 2),
                "away": round(scenario_away_xg, 2),
            },
            **scenario_markets,
            "probability_delta_vs_baseline": deltas,
        },
        "warnings": [
            "This is a heuristic what-if scenario, not an injury or lineup prediction.",
            "Player effects are bounded estimates from stored match contributions and are not causal estimates.",
            "Interactions between replacement players and tactics are not modeled.",
        ],
    }


def build_personalized_fan_briefing(
    matches: pd.DataFrame,
    clusters: pd.DataFrame,
    top_scorers: pd.DataFrame,
    followed_teams: Sequence[str],
    preferences: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Create a concise briefing and optional preference-based recommendations."""
    if not followed_teams:
        raise ValueError("At least one followed team is required.")
    if len(followed_teams) > 10:
        raise ValueError("A briefing supports at most 10 followed teams.")
    frame = _match_frame(matches, completed_only=False)
    completed = frame.dropna(subset=["goals_home", "goals_away"])
    all_names = pd.concat([frame["home_team_name"], frame["away_team_name"]])
    team_sections: list[dict[str, Any]] = []
    resolved_followed: list[str] = []
    now = pd.Timestamp.now(tz="UTC")
    for requested in followed_teams:
        team = _resolve_name(all_names, requested)
        resolved_followed.append(team)
        profile = _team_form_profile(completed, team, 5)
        upcoming = frame.loc[
            ((frame["home_team_name"] == team) | (frame["away_team_name"] == team))
            & ((frame["date_dt"] > now) | frame["goals_home"].isna() | frame["goals_away"].isna())
        ].head(1)
        next_match = None
        if not upcoming.empty:
            row = upcoming.iloc[0]
            next_match = {
                "date_utc": row["date_dt"].isoformat(),
                "home_team": str(row["home_team_name"]),
                "away_team": str(row["away_team_name"]),
                "status": _plain(row.get("status_short")),
            }
        team_sections.append(
            {
                "team": team,
                "official_cluster": _cluster_summary(clusters, team),
                "recent_form": profile,
                "next_stored_fixture": next_match,
                "key_players": _key_players(top_scorers, team),
            }
        )

    recommendation = None
    preference_values = dict(preferences or {})
    if preference_values and set(CLUSTERING_FEATURES).issubset(clusters.columns):
        recommendation = personalized_similarity(
            profiles=clusters[["team_id", "team_name", *CLUSTERING_FEATURES]],
            clusters=clusters,
            team_name=resolved_followed[0],
            preferences=preference_values,
            top_n=5,
        )
    return {
        "followed_teams": resolved_followed,
        "team_briefings": team_sections,
        "personalized_discovery": recommendation,
        "warnings": [
            "Briefing dates refer to stored data and may not be live.",
            "Personalized similarity indexes are relative indexes, not probabilities.",
        ],
    }


def build_live_match_companion(
    fixtures: pd.DataFrame,
    events: pd.DataFrame,
    team_stats: pd.DataFrame,
    player_stats: pd.DataFrame,
    fixture_id: int,
) -> dict[str, Any]:
    """Summarize the newest stored snapshot for one fixture; live-ready, not live-guaranteed."""
    frame = _match_frame(fixtures, completed_only=False)
    rows = frame.loc[pd.to_numeric(frame["fixture_id"], errors="coerce") == int(fixture_id)]
    if rows.empty:
        raise ValueError(f"Fixture not found: {fixture_id}")
    fixture = rows.iloc[-1]
    if not events.empty:
        _require(events, {"fixture_id"}, "Event data")
    fixture_events = events.loc[
        pd.to_numeric(events.get("fixture_id"), errors="coerce") == int(fixture_id)
    ].copy() if not events.empty else pd.DataFrame()
    if not fixture_events.empty:
        fixture_events["time_elapsed"] = pd.to_numeric(fixture_events.get("time_elapsed"), errors="coerce")
        fixture_events = fixture_events.sort_values("time_elapsed")
    if not team_stats.empty:
        _require(team_stats, {"fixture_id", "team_name", "stat_type", "stat_value"}, "Team statistics")
    fixture_team_stats = team_stats.loc[
        pd.to_numeric(team_stats.get("fixture_id"), errors="coerce") == int(fixture_id)
    ].copy() if not team_stats.empty else pd.DataFrame()
    stat_summary: dict[str, dict[str, float | None]] = {}
    if not fixture_team_stats.empty:
        for team, group in fixture_team_stats.groupby("team_name"):
            stat_summary[str(team)] = {
                str(row["stat_type"]): _numeric_stat_value(row["stat_value"])
                for _, row in group.iterrows()
            }
    if not player_stats.empty:
        _require(player_stats, {"fixture_id", "team_name", "player_name"}, "Player statistics")
    fixture_players = player_stats.loc[
        pd.to_numeric(player_stats.get("fixture_id"), errors="coerce") == int(fixture_id)
    ].copy() if not player_stats.empty else pd.DataFrame()
    top_players: list[dict[str, Any]] = []
    if not fixture_players.empty and "rating" in fixture_players.columns:
        fixture_players["rating_numeric"] = pd.to_numeric(fixture_players["rating"], errors="coerce")
        top_players = [
            {
                "player_name": str(row["player_name"]),
                "team_name": str(row["team_name"]),
                "rating": round(float(row["rating_numeric"]), 2),
            }
            for _, row in fixture_players.dropna(subset=["rating_numeric"])
            .sort_values("rating_numeric", ascending=False)
            .head(5)
            .iterrows()
        ]
    status = str(fixture.get("status_short") or "unknown")
    live_statuses = {"1H", "HT", "2H", "ET", "BT", "P", "INT", "LIVE"}
    return {
        "fixture": {
            "fixture_id": int(fixture_id),
            "date_utc": fixture["date_dt"].isoformat(),
            "home_team": str(fixture["home_team_name"]),
            "away_team": str(fixture["away_team_name"]),
            "score": {
                "home": _plain(fixture["goals_home"]),
                "away": _plain(fixture["goals_away"]),
            },
            "status": status,
            "is_live_according_to_snapshot": status.upper() in live_statuses,
        },
        "events": [
            {
                "minute": _plain(row.get("time_elapsed")),
                "extra_minute": _plain(row.get("time_extra")),
                "team": _plain(row.get("team_name")),
                "player": _plain(row.get("player_name")),
                "type": _plain(row.get("type")),
                "detail": _plain(row.get("detail")),
            }
            for _, row in fixture_events.tail(10).iterrows()
        ],
        "team_statistics": stat_summary,
        "top_player_ratings": top_players,
        "warning": "This is the newest stored snapshot. It is live only if the ingestion schedule is live.",
    }


def attach_video_evidence(
    events: pd.DataFrame,
    fixture_id: int,
    video_index: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Attach licensed clip references when supplied, otherwise expose the contract."""
    required_events = {"fixture_id", "time_elapsed", "team_name", "type", "detail"}
    _require(events, required_events, "Event data")
    fixture_events = events.loc[
        pd.to_numeric(events["fixture_id"], errors="coerce") == int(fixture_id)
    ].copy()
    if fixture_events.empty:
        raise ValueError(f"No events found for fixture {fixture_id}.")
    evidence = video_index if video_index is not None else pd.DataFrame()
    available = not evidence.empty and {
        "fixture_id",
        "event_minute",
        "video_url",
    }.issubset(evidence.columns)
    timeline: list[dict[str, Any]] = []
    for _, event in fixture_events.sort_values("time_elapsed").iterrows():
        item = {
            "minute": _plain(event.get("time_elapsed")),
            "team": _plain(event.get("team_name")),
            "player": _plain(event.get("player_name")),
            "event_type": _plain(event.get("type")),
            "detail": _plain(event.get("detail")),
            "video": None,
        }
        if available:
            matches = evidence.loc[
                (pd.to_numeric(evidence["fixture_id"], errors="coerce") == int(fixture_id))
                & (pd.to_numeric(evidence["event_minute"], errors="coerce") == pd.to_numeric(pd.Series([event.get("time_elapsed")]), errors="coerce").iloc[0])
            ]
            if not matches.empty:
                video = matches.iloc[0]
                item["video"] = {
                    "url": str(video["video_url"]),
                    "clip_start_seconds": _plain(video.get("clip_start_seconds")),
                    "clip_end_seconds": _plain(video.get("clip_end_seconds")),
                    "license": _plain(video.get("license")),
                }
        timeline.append(item)
    return {
        "fixture_id": int(fixture_id),
        "video_evidence_available": any(item["video"] for item in timeline),
        "timeline": timeline,
        "required_video_index_schema": [
            "fixture_id",
            "event_minute",
            "video_url",
            "clip_start_seconds",
            "clip_end_seconds",
            "license",
        ],
        "warning": (
            "No licensed video index is connected; event evidence is statistical only."
            if not any(item["video"] for item in timeline)
            else "Only clips explicitly present in the licensed video index are returned."
        ),
    }
