"""Feature engineering utilities for team-level clustering (local-first)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_local_match_features(path: str | Path) -> pd.DataFrame:
    """Load local match-level features produced by derive_tables_local.py."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing local features file: {csv_path}")
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def build_team_level_from_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per team by aggregating home/away match features."""
    required = {
        "fixture_id",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "goals_home",
        "goals_away",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fact_match_features is missing required columns: {sorted(missing)}")

    home = pd.DataFrame(
        {
            "fixture_id": df["fixture_id"],
            "team_id": df["home_team_id"],
            "team_name": df["home_team_name"],
            "is_home": 1,
            "goals_for": pd.to_numeric(df.get("goals_home"), errors="coerce"),
            "goals_against": pd.to_numeric(df.get("goals_away"), errors="coerce"),
            "form_pts_lastN": pd.to_numeric(df.get("home_form_pts_lastN"), errors="coerce"),
            "form_gf_avg_lastN": pd.to_numeric(df.get("home_form_gf_avg_lastN"), errors="coerce"),
            "form_ga_avg_lastN": pd.to_numeric(df.get("home_form_ga_avg_lastN"), errors="coerce"),
            "form_gd_avg_lastN": pd.to_numeric(df.get("home_form_gd_avg_lastN"), errors="coerce"),
            "form_wins_lastN": pd.to_numeric(df.get("home_form_wins_lastN"), errors="coerce"),
            "form_draws_lastN": pd.to_numeric(df.get("home_form_draws_lastN"), errors="coerce"),
            "form_losses_lastN": pd.to_numeric(df.get("home_form_losses_lastN"), errors="coerce"),
        }
    )
    away = pd.DataFrame(
        {
            "fixture_id": df["fixture_id"],
            "team_id": df["away_team_id"],
            "team_name": df["away_team_name"],
            "is_home": 0,
            "goals_for": pd.to_numeric(df.get("goals_away"), errors="coerce"),
            "goals_against": pd.to_numeric(df.get("goals_home"), errors="coerce"),
            "form_pts_lastN": pd.to_numeric(df.get("away_form_pts_lastN"), errors="coerce"),
            "form_gf_avg_lastN": pd.to_numeric(df.get("away_form_gf_avg_lastN"), errors="coerce"),
            "form_ga_avg_lastN": pd.to_numeric(df.get("away_form_ga_avg_lastN"), errors="coerce"),
            "form_gd_avg_lastN": pd.to_numeric(df.get("away_form_gd_avg_lastN"), errors="coerce"),
            "form_wins_lastN": pd.to_numeric(df.get("away_form_wins_lastN"), errors="coerce"),
            "form_draws_lastN": pd.to_numeric(df.get("away_form_draws_lastN"), errors="coerce"),
            "form_losses_lastN": pd.to_numeric(df.get("away_form_losses_lastN"), errors="coerce"),
        }
    )

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]
    long_df["is_win"] = (long_df["goals_for"] > long_df["goals_against"]).astype(float)
    long_df["is_draw"] = (long_df["goals_for"] == long_df["goals_against"]).astype(float)
    long_df["is_loss"] = (long_df["goals_for"] < long_df["goals_against"]).astype(float)
    long_df["points"] = long_df["is_win"] * 3 + long_df["is_draw"]

    team_df = (
        long_df.groupby(["team_id", "team_name"], dropna=False)
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

    team_df["attack_strength"] = team_df["avg_goals_scored"] * (1 + team_df["win_rate"])
    team_df["defense_strength"] = (1 / (1 + team_df["avg_goals_conceded"].clip(lower=0))) * (
        1 + team_df["draw_rate"]
    )
    team_df["form_score"] = (
        team_df["form_pts_avg"].fillna(0) + team_df["form_gd_avg"].fillna(0) + team_df["avg_points"].fillna(0)
    )
    return team_df


def build_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Ensure clustering features exist and are numeric/clean."""
    features = config["features"]
    out = df.copy()

    # A few robust derivations in case source data only has base columns.
    if "win_rate" in features and "win_rate" not in out and {"wins", "games"} <= set(out.columns):
        out["win_rate"] = out["wins"] / out["games"].replace(0, np.nan)
    if "draw_rate" in features and "draw_rate" not in out and {"draws", "games"} <= set(out.columns):
        out["draw_rate"] = out["draws"] / out["games"].replace(0, np.nan)
    if "loss_rate" in features and "loss_rate" not in out and {"losses", "games"} <= set(out.columns):
        out["loss_rate"] = out["losses"] / out["games"].replace(0, np.nan)

    for feat in features:
        if feat not in out.columns:
            out[feat] = 0.0
        out[feat] = pd.to_numeric(out[feat], errors="coerce")

    fillna_strategy = config.get("preprocessing", {}).get("fillna", "median")
    if fillna_strategy == "mean":
        out[features] = out[features].fillna(out[features].mean())
    elif fillna_strategy == "zero":
        out[features] = out[features].fillna(0.0)
    else:
        out[features] = out[features].fillna(out[features].median())

    return out
