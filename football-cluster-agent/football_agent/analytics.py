"""Pure statistical helpers for personalized similarity and match forecasts.

This module intentionally has no Google Cloud dependencies. Cloud-facing tools
load the data, then delegate deterministic calculations to these functions.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import pandas as pd


CLUSTERING_FEATURES = (
    "avg_goals_scored",
    "avg_goals_conceded",
    "avg_goal_diff",
    "std_goal_diff",
    "win_rate",
    "draw_rate",
    "loss_rate",
    "avg_points",
    "form_pts_avg",
    "form_gf_avg",
    "form_ga_avg",
    "form_gd_avg",
    "form_wins_avg",
    "form_draws_avg",
    "form_losses_avg",
    "home_ratio",
    "attack_strength",
    "defense_strength",
    "form_score",
)

PREFERENCE_DIMENSIONS: dict[str, dict[str, float]] = {
    "attack": {
        "avg_goals_scored": 1.0,
        "attack_strength": 1.0,
        "form_gf_avg": 0.6,
    },
    "defense": {
        "avg_goals_conceded": -1.0,
        "defense_strength": 1.0,
        "form_ga_avg": -0.6,
    },
    "results": {
        "win_rate": 1.0,
        "loss_rate": -1.0,
        "avg_points": 1.0,
        "avg_goal_diff": 0.8,
    },
    "recent_form": {
        "form_pts_avg": 1.0,
        "form_gd_avg": 0.8,
        "form_score": 1.0,
        "form_wins_avg": 0.5,
        "form_losses_avg": -0.5,
    },
    "consistency": {
        "std_goal_diff": -1.0,
    },
}


def _normalize_team_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(ascii_value.casefold().split())


def _plain_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _numeric_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def summarize_latest_matches(
    matches: pd.DataFrame,
    limit: int = 10,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Return JSON-safe coverage metadata and the newest completed matches."""
    required = {
        "fixture_id",
        "date",
        "home_team_name",
        "away_team_name",
        "goals_home",
        "goals_away",
    }
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Match data is missing required columns: {sorted(missing)}")

    safe_limit = max(1, min(int(limit), 50))
    frame = matches.copy()
    frame["date_dt"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["goals_home"] = pd.to_numeric(frame["goals_home"], errors="coerce")
    frame["goals_away"] = pd.to_numeric(frame["goals_away"], errors="coerce")
    frame = frame.dropna(
        subset=["date_dt", "goals_home", "goals_away"]
    ).sort_values("date_dt", ascending=False)
    if frame.empty:
        raise ValueError("No completed matches with valid dates and scores are available.")

    latest_date = frame["date_dt"].max()
    earliest_date = frame["date_dt"].min()
    reference_time = as_of or datetime.now(timezone.utc)
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    age_days = max(
        0,
        int(
            (
                reference_time.astimezone(timezone.utc)
                - latest_date.to_pydatetime()
            ).total_seconds()
            // 86400
        ),
    )

    recent_form_fields = [
        column
        for column in (
            "home_form_pts_lastN",
            "home_form_wins_lastN",
            "home_form_draws_lastN",
            "home_form_losses_lastN",
            "home_form_gf_avg_lastN",
            "home_form_ga_avg_lastN",
            "home_form_gd_avg_lastN",
            "away_form_pts_lastN",
            "away_form_wins_lastN",
            "away_form_draws_lastN",
            "away_form_losses_lastN",
            "away_form_gf_avg_lastN",
            "away_form_ga_avg_lastN",
            "away_form_gd_avg_lastN",
        )
        if column in frame.columns
    ]

    latest_matches: list[dict[str, Any]] = []
    for _, row in frame.head(safe_limit).iterrows():
        latest_matches.append(
            {
                "fixture_id": _plain_scalar(row["fixture_id"]),
                "match_date_utc": row["date_dt"].isoformat(),
                "home_team": str(row["home_team_name"]),
                "away_team": str(row["away_team_name"]),
                "home_goals": _plain_scalar(row["goals_home"]),
                "away_goals": _plain_scalar(row["goals_away"]),
                "league_id": _plain_scalar(row.get("league_id")),
                "season": _plain_scalar(row.get("season")),
                "status": _plain_scalar(row.get("status_short")),
            }
        )

    return {
        "source_table": "fact_match_features",
        "coverage": {
            "completed_match_count": int(len(frame)),
            "earliest_match_utc": earliest_date.isoformat(),
            "latest_match_utc": latest_date.isoformat(),
            "data_age_days": age_days,
            "seasons": sorted(
                {
                    _plain_scalar(value)
                    for value in frame.get("season", pd.Series(dtype=object)).dropna()
                },
                key=str,
            ),
            "league_ids": sorted(
                {
                    _plain_scalar(value)
                    for value in frame.get("league_id", pd.Series(dtype=object)).dropna()
                },
                key=str,
            ),
        },
        "available_match_statistics": [
            "home and away teams",
            "final score",
            "league",
            "season",
            *recent_form_fields,
        ],
        "latest_matches": latest_matches,
        "warning": (
            f"The newest completed match is {age_days} days old; describe the data "
            "as the latest available in this dataset, not as live data."
        ),
    }


def build_team_profiles_from_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate match-level rows into the feature schema used by clustering."""
    required = {
        "fixture_id",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "goals_home",
        "goals_away",
    }
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Match data is missing required columns: {sorted(missing)}")

    frame = matches.copy()
    frame["goals_home"] = pd.to_numeric(frame["goals_home"], errors="coerce")
    frame["goals_away"] = pd.to_numeric(frame["goals_away"], errors="coerce")
    frame = frame.dropna(subset=["goals_home", "goals_away"])
    if frame.empty:
        raise ValueError("No completed matches with numeric scores are available.")

    home = pd.DataFrame(
        {
            "fixture_id": frame["fixture_id"],
            "team_id": frame["home_team_id"],
            "team_name": frame["home_team_name"],
            "is_home": 1,
            "goals_for": frame["goals_home"],
            "goals_against": frame["goals_away"],
            "form_pts_lastN": _numeric_or_nan(frame, "home_form_pts_lastN"),
            "form_gf_avg_lastN": _numeric_or_nan(frame, "home_form_gf_avg_lastN"),
            "form_ga_avg_lastN": _numeric_or_nan(frame, "home_form_ga_avg_lastN"),
            "form_gd_avg_lastN": _numeric_or_nan(frame, "home_form_gd_avg_lastN"),
            "form_wins_lastN": _numeric_or_nan(frame, "home_form_wins_lastN"),
            "form_draws_lastN": _numeric_or_nan(frame, "home_form_draws_lastN"),
            "form_losses_lastN": _numeric_or_nan(frame, "home_form_losses_lastN"),
        }
    )
    away = pd.DataFrame(
        {
            "fixture_id": frame["fixture_id"],
            "team_id": frame["away_team_id"],
            "team_name": frame["away_team_name"],
            "is_home": 0,
            "goals_for": frame["goals_away"],
            "goals_against": frame["goals_home"],
            "form_pts_lastN": _numeric_or_nan(frame, "away_form_pts_lastN"),
            "form_gf_avg_lastN": _numeric_or_nan(frame, "away_form_gf_avg_lastN"),
            "form_ga_avg_lastN": _numeric_or_nan(frame, "away_form_ga_avg_lastN"),
            "form_gd_avg_lastN": _numeric_or_nan(frame, "away_form_gd_avg_lastN"),
            "form_wins_lastN": _numeric_or_nan(frame, "away_form_wins_lastN"),
            "form_draws_lastN": _numeric_or_nan(frame, "away_form_draws_lastN"),
            "form_losses_lastN": _numeric_or_nan(frame, "away_form_losses_lastN"),
        }
    )
    long_frame = pd.concat([home, away], ignore_index=True)
    long_frame["goal_diff"] = long_frame["goals_for"] - long_frame["goals_against"]
    long_frame["is_win"] = (
        long_frame["goals_for"] > long_frame["goals_against"]
    ).astype(float)
    long_frame["is_draw"] = (
        long_frame["goals_for"] == long_frame["goals_against"]
    ).astype(float)
    long_frame["is_loss"] = (
        long_frame["goals_for"] < long_frame["goals_against"]
    ).astype(float)
    long_frame["points"] = long_frame["is_win"] * 3 + long_frame["is_draw"]

    profiles = (
        long_frame.groupby(["team_id", "team_name"], dropna=False)
        .agg(
            games=("fixture_id", "nunique"),
            avg_goals_scored=("goals_for", "mean"),
            avg_goals_conceded=("goals_against", "mean"),
            avg_goal_diff=("goal_diff", "mean"),
            std_goal_diff=("goal_diff", "std"),
            win_rate=("is_win", "mean"),
            draw_rate=("is_draw", "mean"),
            loss_rate=("is_loss", "mean"),
            avg_points=("points", "mean"),
            form_pts_avg=("form_pts_lastN", "mean"),
            form_gf_avg=("form_gf_avg_lastN", "mean"),
            form_ga_avg=("form_ga_avg_lastN", "mean"),
            form_gd_avg=("form_gd_avg_lastN", "mean"),
            form_wins_avg=("form_wins_lastN", "mean"),
            form_draws_avg=("form_draws_lastN", "mean"),
            form_losses_avg=("form_losses_lastN", "mean"),
            home_ratio=("is_home", "mean"),
        )
        .reset_index()
    )
    profiles["attack_strength"] = profiles["avg_goals_scored"] * (
        1 + profiles["win_rate"]
    )
    profiles["defense_strength"] = (
        1 / (1 + profiles["avg_goals_conceded"].clip(lower=0))
    ) * (1 + profiles["draw_rate"])
    profiles["form_score"] = (
        profiles["form_pts_avg"].fillna(0)
        + profiles["form_gd_avg"].fillna(0)
        + profiles["avg_points"].fillna(0)
    )
    return profiles


def _validated_preferences(preferences: Mapping[str, float]) -> dict[str, float]:
    unknown = set(preferences) - set(PREFERENCE_DIMENSIONS)
    if unknown:
        raise ValueError(f"Unknown preference dimensions: {sorted(unknown)}")
    validated: dict[str, float] = {}
    for dimension in PREFERENCE_DIMENSIONS:
        value = float(preferences.get(dimension, 0.0))
        if not -1.0 <= value <= 1.0:
            raise ValueError(
                f"Preference {dimension!r} must be between -1 and 1, got {value}."
            )
        validated[dimension] = value
    return validated


def _robust_standardize(
    profiles: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    numeric = profiles[features].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    filled = numeric.fillna(medians).fillna(0.0)
    iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
    standard_deviation = numeric.std()
    scale = iqr.where(iqr.abs() > 1e-12, standard_deviation)
    scale = scale.where(scale.abs() > 1e-12, 1.0).fillna(1.0)
    return (filled - medians.fillna(0.0)) / scale


def _resolve_team_row(frame: pd.DataFrame, team_name: str) -> int:
    target = _normalize_team_name(team_name)
    normalized = frame["team_name"].astype(str).map(_normalize_team_name)
    matches = frame.index[normalized == target].tolist()
    if not matches:
        raise ValueError(f"Team not found: {team_name}")
    if len(matches) > 1:
        raise ValueError(f"Team name is ambiguous: {team_name}")
    return int(matches[0])


def _relative_similarity(distance: float, reference_distance: float) -> float:
    safe_reference = max(float(reference_distance), 1e-9)
    return round(100.0 * math.exp(-math.log(2.0) * distance / safe_reference), 1)


def personalized_similarity(
    profiles: pd.DataFrame,
    clusters: pd.DataFrame,
    team_name: str,
    preferences: Mapping[str, float],
    top_n: int = 5,
) -> dict[str, Any]:
    """Calculate weighted team similarity and preference-fit recommendations.

    Returned similarity and fit values are relative indexes, not probabilities.
    """
    if not {"team_id", "team_name"}.issubset(profiles.columns):
        raise ValueError("Profiles must include team_id and team_name.")
    if not {"team_id", "cluster"}.issubset(clusters.columns):
        raise ValueError("Clusters must include team_id and cluster.")

    safe_top_n = max(1, min(int(top_n), 10))
    validated = _validated_preferences(preferences)
    available_features = [
        feature for feature in CLUSTERING_FEATURES if feature in profiles.columns
    ]
    if len(available_features) < 3:
        raise ValueError("At least three clustering features are required.")

    profile_frame = profiles.drop_duplicates(subset=["team_id"]).reset_index(drop=True)
    cluster_columns = ["team_id", "cluster"]
    if "cluster_label" in clusters.columns:
        cluster_columns.append("cluster_label")
    joined = profile_frame.merge(
        clusters[cluster_columns].drop_duplicates(subset=["team_id"]),
        on="team_id",
        how="left",
        validate="one_to_one",
    )
    seed_index = _resolve_team_row(joined, team_name)
    standardized = _robust_standardize(joined, available_features)

    feature_weights = {feature: 1.0 for feature in available_features}
    for dimension, preference in validated.items():
        for feature, orientation in PREFERENCE_DIMENSIONS[dimension].items():
            if feature in feature_weights:
                feature_weights[feature] += 2.0 * abs(preference) * abs(orientation)

    weight_series = pd.Series(feature_weights)
    seed_vector = standardized.loc[seed_index]
    distances = (
        standardized.sub(seed_vector)
        .pow(2)
        .mul(weight_series, axis="columns")
        .sum(axis=1)
        .div(float(weight_series.sum()))
        .pow(0.5)
    )
    peer_distances = distances.drop(index=seed_index)
    distance_reference = float(peer_distances.median())
    if not math.isfinite(distance_reference) or distance_reference <= 0:
        distance_reference = float(peer_distances.max()) or 1.0

    non_zero_preferences = {
        key: value for key, value in validated.items() if abs(value) > 1e-12
    }
    raw_fit = pd.Series(0.0, index=joined.index)
    if non_zero_preferences:
        preference_mass = sum(abs(value) for value in non_zero_preferences.values())
        for dimension, preference in non_zero_preferences.items():
            dimension_parts: list[pd.Series] = []
            for feature, orientation in PREFERENCE_DIMENSIONS[dimension].items():
                if feature in standardized.columns:
                    dimension_parts.append(standardized[feature] * orientation)
            if dimension_parts:
                dimension_score = pd.concat(dimension_parts, axis=1).mean(axis=1)
                raw_fit = raw_fit + preference * dimension_score
        raw_fit = raw_fit / preference_mass
        fit_index = raw_fit.rank(method="average", pct=True) * 100.0
    else:
        fit_index = pd.Series(float("nan"), index=joined.index)

    peer_order = peer_distances.sort_values().index[:safe_top_n]
    similar_teams: list[dict[str, Any]] = []
    for index in peer_order:
        row = joined.loc[index]
        similar_teams.append(
            {
                "team_name": str(row["team_name"]),
                "official_cluster": (
                    int(row["cluster"]) if pd.notna(row.get("cluster")) else None
                ),
                "weighted_distance": round(float(distances.loc[index]), 4),
                "relative_similarity_index": _relative_similarity(
                    float(distances.loc[index]), distance_reference
                ),
                "preference_fit_index": (
                    round(float(fit_index.loc[index]), 1)
                    if pd.notna(fit_index.loc[index])
                    else None
                ),
            }
        )

    cluster_distances: list[dict[str, Any]] = []
    valid_cluster_rows = joined[joined["cluster"].notna()]
    if not valid_cluster_rows.empty:
        centroid_frame = standardized.loc[valid_cluster_rows.index].copy()
        centroid_frame["cluster"] = valid_cluster_rows["cluster"].values
        centroids = centroid_frame.groupby("cluster")[available_features].mean()
        centroid_distance_values: list[tuple[Any, float]] = []
        for cluster_id, centroid in centroids.iterrows():
            delta = centroid - seed_vector
            distance = math.sqrt(
                float(delta.pow(2).mul(weight_series).sum())
                / float(weight_series.sum())
            )
            centroid_distance_values.append((cluster_id, distance))
        centroid_reference = pd.Series(
            [distance for _, distance in centroid_distance_values]
        ).median()
        if not math.isfinite(float(centroid_reference)) or centroid_reference <= 0:
            centroid_reference = 1.0
        for cluster_id, distance in sorted(
            centroid_distance_values, key=lambda item: item[1]
        ):
            cluster_distances.append(
                {
                    "cluster": int(cluster_id),
                    "weighted_distance_to_centroid": round(distance, 4),
                    "relative_cluster_similarity_index": _relative_similarity(
                        distance, float(centroid_reference)
                    ),
                }
            )

    recommended_teams: list[dict[str, Any]] = []
    if non_zero_preferences:
        recommendation_order = (
            fit_index.drop(index=seed_index).sort_values(ascending=False).index[
                :safe_top_n
            ]
        )
        for index in recommendation_order:
            row = joined.loc[index]
            recommended_teams.append(
                {
                    "team_name": str(row["team_name"]),
                    "official_cluster": (
                        int(row["cluster"]) if pd.notna(row.get("cluster")) else None
                    ),
                    "preference_fit_index": round(float(fit_index.loc[index]), 1),
                    "relative_similarity_to_reference_team": _relative_similarity(
                        float(distances.loc[index]), distance_reference
                    ),
                }
            )

    seed_row = joined.loc[seed_index]
    warnings = [
        "Similarity and preference-fit values are relative indexes, not probabilities.",
        "Personal preferences change feature importance, not official cluster assignments or raw statistics.",
    ]
    if not non_zero_preferences:
        warnings.append(
            "All preferences are neutral, so preference-based recommendations are unavailable."
        )

    return {
        "reference_team": str(seed_row["team_name"]),
        "official_cluster": (
            int(seed_row["cluster"]) if pd.notna(seed_row.get("cluster")) else None
        ),
        "official_cluster_label": seed_row.get("cluster_label"),
        "preferences": validated,
        "feature_weights": {
            key: round(value, 3) for key, value in feature_weights.items()
        },
        "features_used": available_features,
        "similar_teams": similar_teams,
        "distance_to_each_cluster": cluster_distances,
        "recommended_teams_for_preferences": recommended_teams,
        "reference_team_preference_fit_index": (
            round(float(fit_index.loc[seed_index]), 1)
            if pd.notna(fit_index.loc[seed_index])
            else None
        ),
        "warnings": warnings,
    }


def _resolve_team_name_from_matches(matches: pd.DataFrame, requested: str) -> str:
    names = pd.concat(
        [
            matches["home_team_name"].astype(str),
            matches["away_team_name"].astype(str),
        ],
        ignore_index=True,
    ).drop_duplicates()
    target = _normalize_team_name(requested)
    found = names[names.map(_normalize_team_name) == target].tolist()
    if not found:
        raise ValueError(f"Team not found in match history: {requested}")
    if len(found) > 1:
        raise ValueError(f"Team name is ambiguous in match history: {requested}")
    return str(found[0])


def _smoothed_average(
    values: pd.Series, prior_mean: float, prior_matches: float = 5.0
) -> tuple[float, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    count = int(len(numeric))
    estimate = (float(numeric.sum()) + prior_mean * prior_matches) / (
        count + prior_matches
    )
    return estimate, count


def _recent_points_per_game(
    matches: pd.DataFrame, team_name: str, limit: int = 5
) -> tuple[float, int]:
    home_rows = matches[matches["home_team_name"] == team_name].copy()
    home_rows["goals_for"] = home_rows["goals_home"]
    home_rows["goals_against"] = home_rows["goals_away"]
    away_rows = matches[matches["away_team_name"] == team_name].copy()
    away_rows["goals_for"] = away_rows["goals_away"]
    away_rows["goals_against"] = away_rows["goals_home"]
    team_rows = pd.concat([home_rows, away_rows], ignore_index=True)
    team_rows = team_rows.sort_values("date_dt").tail(limit)
    if team_rows.empty:
        return 1.5, 0
    wins = (team_rows["goals_for"] > team_rows["goals_against"]).astype(float)
    draws = (team_rows["goals_for"] == team_rows["goals_against"]).astype(float)
    return float((wins * 3 + draws).mean()), int(len(team_rows))


def _poisson_probabilities(expected_goals: float, max_goals: int) -> list[float]:
    return [
        math.exp(-expected_goals)
        * (expected_goals**goals)
        / math.factorial(goals)
        for goals in range(max_goals + 1)
    ]


def poisson_outcome_summary(
    expected_home: float,
    expected_away: float,
    max_goals: int = 8,
) -> dict[str, Any]:
    """Convert expected goals into normalized, uncalibrated match markets."""
    if expected_home < 0 or expected_away < 0:
        raise ValueError("Expected goals cannot be negative.")
    if max_goals < 3:
        raise ValueError("max_goals must be at least 3.")

    home_goal_probabilities = _poisson_probabilities(expected_home, max_goals)
    away_goal_probabilities = _poisson_probabilities(expected_away, max_goals)
    score_probabilities = [
        (home_goals, away_goals, home_probability * away_probability)
        for home_goals, home_probability in enumerate(home_goal_probabilities)
        for away_goals, away_probability in enumerate(away_goal_probabilities)
    ]
    probability_mass = sum(probability for _, _, probability in score_probabilities)
    if probability_mass <= 0:
        raise ValueError("Poisson score grid has no usable probability mass.")
    normalized_scores = [
        (home_goals, away_goals, probability / probability_mass)
        for home_goals, away_goals, probability in score_probabilities
    ]
    home_win = sum(
        probability
        for home_goals, away_goals, probability in normalized_scores
        if home_goals > away_goals
    )
    draw = sum(
        probability
        for home_goals, away_goals, probability in normalized_scores
        if home_goals == away_goals
    )
    away_win = sum(
        probability
        for home_goals, away_goals, probability in normalized_scores
        if home_goals < away_goals
    )
    over_2_5 = sum(
        probability
        for home_goals, away_goals, probability in normalized_scores
        if home_goals + away_goals >= 3
    )
    both_score = sum(
        probability
        for home_goals, away_goals, probability in normalized_scores
        if home_goals >= 1 and away_goals >= 1
    )
    likely_scores = sorted(
        normalized_scores, key=lambda item: item[2], reverse=True
    )[:3]
    return {
        "outcome_probabilities": {
            "home_win": round(home_win, 4),
            "draw": round(draw, 4),
            "away_win": round(away_win, 4),
        },
        "goal_market_probabilities": {
            "over_2_5": round(over_2_5, 4),
            "both_teams_to_score": round(both_score, 4),
        },
        "most_likely_scores": [
            {
                "score": f"{home_goals}-{away_goals}",
                "probability": round(probability, 4),
            }
            for home_goals, away_goals, probability in likely_scores
        ],
    }


def predict_match_from_history(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    max_goals: int = 8,
) -> dict[str, Any]:
    """Return an uncalibrated Poisson forecast grounded in match history."""
    required = {
        "fixture_id",
        "date",
        "league_id",
        "season",
        "home_team_name",
        "away_team_name",
        "goals_home",
        "goals_away",
    }
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Match history is missing columns: {sorted(missing)}")
    if _normalize_team_name(home_team) == _normalize_team_name(away_team):
        raise ValueError("Home and away team must be different.")

    frame = matches.copy()
    frame["goals_home"] = pd.to_numeric(frame["goals_home"], errors="coerce")
    frame["goals_away"] = pd.to_numeric(frame["goals_away"], errors="coerce")
    frame["date_dt"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["goals_home", "goals_away", "date_dt"])
    if frame.empty:
        raise ValueError("No completed matches are available for prediction.")

    resolved_home = _resolve_team_name_from_matches(frame, home_team)
    resolved_away = _resolve_team_name_from_matches(frame, away_team)

    def contexts_for(team: str) -> set[tuple[Any, Any]]:
        team_rows = frame[
            (frame["home_team_name"] == team) | (frame["away_team_name"] == team)
        ]
        return set(
            team_rows[["league_id", "season"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

    common_contexts = contexts_for(resolved_home) & contexts_for(resolved_away)
    if not common_contexts:
        raise ValueError(
            f"No common league/season data for {resolved_home} and {resolved_away}."
        )
    context = max(
        common_contexts,
        key=lambda item: frame[
            (frame["league_id"] == item[0]) & (frame["season"] == item[1])
        ]["date_dt"].max(),
    )
    league_id, season = context
    context_matches = frame[
        (frame["league_id"] == league_id) & (frame["season"] == season)
    ].copy()

    league_home_average = float(context_matches["goals_home"].mean())
    league_away_average = float(context_matches["goals_away"].mean())
    if league_home_average <= 0 or league_away_average <= 0:
        raise ValueError("League goal averages are not usable for a Poisson model.")

    home_history = context_matches[
        context_matches["home_team_name"] == resolved_home
    ]
    away_history = context_matches[
        context_matches["away_team_name"] == resolved_away
    ]
    home_scored, home_games = _smoothed_average(
        home_history["goals_home"], league_home_average
    )
    home_conceded, _ = _smoothed_average(
        home_history["goals_away"], league_away_average
    )
    away_scored, away_games = _smoothed_average(
        away_history["goals_away"], league_away_average
    )
    away_conceded, _ = _smoothed_average(
        away_history["goals_home"], league_home_average
    )

    home_form, home_form_games = _recent_points_per_game(
        context_matches, resolved_home
    )
    away_form, away_form_games = _recent_points_per_game(
        context_matches, resolved_away
    )
    home_form_factor = max(0.9, min(1.1, 1 + 0.10 * ((home_form - 1.5) / 1.5)))
    away_form_factor = max(0.9, min(1.1, 1 + 0.10 * ((away_form - 1.5) / 1.5)))

    expected_home = (
        league_home_average
        * (home_scored / league_home_average)
        * (away_conceded / league_home_average)
        * home_form_factor
    )
    expected_away = (
        league_away_average
        * (away_scored / league_away_average)
        * (home_conceded / league_away_average)
        * away_form_factor
    )
    expected_home = max(0.15, min(4.0, expected_home))
    expected_away = max(0.15, min(4.0, expected_away))

    markets = poisson_outcome_summary(expected_home, expected_away, max_goals)

    latest_match = context_matches["date_dt"].max()
    age_days = int((pd.Timestamp.now(tz="UTC") - latest_match).days)
    minimum_sample = min(home_games, away_games)
    if minimum_sample >= 15 and age_days <= 45:
        confidence = "high"
    elif minimum_sample >= 8 and age_days <= 180:
        confidence = "medium"
    else:
        confidence = "low"

    warnings = [
        "These are model-implied, uncalibrated probabilities, not guarantees.",
        "The baseline does not include confirmed lineups, injuries, suspensions, odds, or tactical matchup data.",
    ]
    if age_days > 180:
        warnings.append(
            f"The newest match in this data slice is {age_days} days old, so confidence is reduced."
        )

    return {
        "fixture": {
            "home_team": resolved_home,
            "away_team": resolved_away,
            "league_id": _plain_scalar(league_id),
            "season": _plain_scalar(season),
        },
        "expected_goals": {
            "home": round(expected_home, 2),
            "away": round(expected_away, 2),
        },
        **markets,
        "evidence": {
            "league_home_goals_average": round(league_home_average, 3),
            "league_away_goals_average": round(league_away_average, 3),
            "home_team_home_matches": home_games,
            "away_team_away_matches": away_games,
            "home_recent_points_per_game": round(home_form, 3),
            "away_recent_points_per_game": round(away_form, 3),
            "home_recent_matches": home_form_games,
            "away_recent_matches": away_form_games,
            "latest_data_utc": latest_match.isoformat(),
        },
        "confidence": confidence,
        "warnings": warnings,
        "method": "Bayesian-smoothed home/away scoring rates + recent form + independent Poisson goals",
    }
