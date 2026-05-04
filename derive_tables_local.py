"""Derive analytical tables from the base CSVs produced by transform_data_local.py.

Reads CSVs from ./output/tables/ and writes derived/aggregated tables to
./output/derived/. No GCS access, no API calls. Pure pandas.

Goal: produce ML-ready feature tables and analytical aggregates with rich semantics
so downstream models have plenty of signal (form, h2h, rolling stats, top scorers, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("output") / "tables"
OUT_DIR = Path("output") / "derived"

ROLLING_WINDOW = 5  # last N matches for "form" features


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def _read(name: str) -> pd.DataFrame:
    """Read a base CSV; return empty DataFrame if missing."""
    path = BASE_DIR / name
    if not path.exists():
        print(f"[WARN] Missing base table: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"[INFO] Loaded {name}: {len(df)} rows, {len(df.columns)} cols")
    return df


def _write(df: pd.DataFrame, name: str) -> None:
    """Write a derived CSV with UTF-8 BOM (Excel-friendly)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Wrote {len(df):>7} rows -> {path}")


# --------------------------------------------------------------------------- #
# Derived: dimensions
# --------------------------------------------------------------------------- #

def build_dim_players(season_stats: pd.DataFrame, match_stats: pd.DataFrame) -> pd.DataFrame:
    """One row per player with stable attributes (latest seen wins)."""
    frames: list[pd.DataFrame] = []
    if not season_stats.empty:
        cols = [
            "player_id", "player_name", "first_name", "last_name", "age",
            "nationality", "height", "weight", "birth_date", "birth_place",
            "birth_country", "photo",
        ]
        present = [c for c in cols if c in season_stats.columns]
        frames.append(season_stats[present])
    if not match_stats.empty:
        cols = ["player_id", "player_name"]
        present = [c for c in cols if c in match_stats.columns]
        frames.append(match_stats[present])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True).dropna(subset=["player_id"])
    combined = combined.drop_duplicates(subset=["player_id"], keep="last").reset_index(drop=True)
    return combined


def build_dim_venues(teams: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per venue. Combines venues from teams + fixtures."""
    frames: list[pd.DataFrame] = []
    if not teams.empty:
        cols = ["venue_id", "venue_name", "venue_city", "venue_capacity", "venue_surface"]
        present = [c for c in cols if c in teams.columns]
        frames.append(teams[present])
    if not matches.empty:
        cols = ["venue_id", "venue_name", "venue_city"]
        present = [c for c in cols if c in matches.columns]
        frames.append(matches[present])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True).dropna(subset=["venue_id"])
    combined = combined.drop_duplicates(subset=["venue_id"], keep="last").reset_index(drop=True)
    return combined


def build_dim_coaches(lineups: pd.DataFrame) -> pd.DataFrame:
    """One row per coach with their team appearances count."""
    if lineups.empty or "coach_id" not in lineups.columns:
        return pd.DataFrame()
    df = lineups.dropna(subset=["coach_id"]).copy()
    grp = (
        df.groupby(["coach_id", "coach_name"], dropna=False)
          .agg(matches_coached=("fixture_id", "nunique"),
               teams_coached=("team_id", "nunique"))
          .reset_index()
    )
    return grp


# --------------------------------------------------------------------------- #
# Derived: match-level
# --------------------------------------------------------------------------- #

def build_team_match_stats_wide(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Pivot long fact_match_team_stats into wide form: 1 row per (fixture, team)."""
    if team_stats.empty:
        return pd.DataFrame()

    df = team_stats.copy()
    # Convert percent strings ("64%") to floats; keep numeric stats numeric.
    def _to_number(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if s.endswith("%"):
            try:
                return float(s[:-1])
            except ValueError:
                return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    df["stat_value_num"] = df["stat_value"].map(_to_number)

    wide = df.pivot_table(
        index=["fixture_id", "team_id", "team_name"],
        columns="stat_type",
        values="stat_value_num",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    # Sanitize column names a bit.
    wide.columns = [
        c.lower().replace(" ", "_").replace("%", "pct").replace("-", "_") if isinstance(c, str) else c
        for c in wide.columns
    ]
    return wide


def build_match_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert wide fact_matches into long form: 2 rows per match (one per side)."""
    if matches.empty:
        return pd.DataFrame()

    needed = {
        "fixture_id", "date", "league_id", "season", "round",
        "home_team_id", "home_team_name", "away_team_id", "away_team_name",
        "goals_home", "goals_away", "status_short",
    }
    missing = needed - set(matches.columns)
    if missing:
        print(f"[WARN] match_long: missing cols {missing}")

    home = pd.DataFrame({
        "fixture_id": matches.get("fixture_id"),
        "date": matches.get("date"),
        "league_id": matches.get("league_id"),
        "season": matches.get("season"),
        "round": matches.get("round"),
        "team_id": matches.get("home_team_id"),
        "team_name": matches.get("home_team_name"),
        "opponent_id": matches.get("away_team_id"),
        "opponent_name": matches.get("away_team_name"),
        "is_home": True,
        "goals_for": matches.get("goals_home"),
        "goals_against": matches.get("goals_away"),
        "status_short": matches.get("status_short"),
    })
    away = pd.DataFrame({
        "fixture_id": matches.get("fixture_id"),
        "date": matches.get("date"),
        "league_id": matches.get("league_id"),
        "season": matches.get("season"),
        "round": matches.get("round"),
        "team_id": matches.get("away_team_id"),
        "team_name": matches.get("away_team_name"),
        "opponent_id": matches.get("home_team_id"),
        "opponent_name": matches.get("home_team_name"),
        "is_home": False,
        "goals_for": matches.get("goals_away"),
        "goals_against": matches.get("goals_home"),
        "status_short": matches.get("status_short"),
    })
    long = pd.concat([home, away], ignore_index=True)

    # Result label
    def _result(row):
        gf, ga = row["goals_for"], row["goals_against"]
        if pd.isna(gf) or pd.isna(ga):
            return None
        if gf > ga:
            return "W"
        if gf < ga:
            return "L"
        return "D"

    long["result"] = long.apply(_result, axis=1)
    long["points"] = long["result"].map({"W": 3, "D": 1, "L": 0})
    long["goal_diff"] = long["goals_for"] - long["goals_against"]
    long["date_dt"] = pd.to_datetime(long["date"], errors="coerce", utc=True)
    long = long.sort_values(["team_id", "date_dt"]).reset_index(drop=True)
    return long


def build_team_form(match_long: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Rolling form features per team: last N matches W/D/L and goal averages.

    Uses .shift(1) so each row's features describe the team's state ENTERING the match
    (no leakage of the current match's result).
    """
    if match_long.empty:
        return pd.DataFrame()

    df = match_long.copy()
    df = df.dropna(subset=["team_id", "date_dt"]).sort_values(["team_id", "date_dt"])

    # Numeric indicators so .rolling() works (cannot operate on string "result").
    df["_is_win"] = (df["result"] == "W").astype(float)
    df["_is_draw"] = (df["result"] == "D").astype(float)
    df["_is_loss"] = (df["result"] == "L").astype(float)

    grp = df.groupby("team_id", sort=False)
    df["form_pts_lastN"] = grp["points"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    ).reset_index(level=0, drop=True)
    df["form_gf_avg_lastN"] = grp["goals_for"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    df["form_ga_avg_lastN"] = grp["goals_against"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    df["form_gd_avg_lastN"] = grp["goal_diff"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    df["form_wins_lastN"] = grp["_is_win"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    ).reset_index(level=0, drop=True)
    df["form_draws_lastN"] = grp["_is_draw"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    ).reset_index(level=0, drop=True)
    df["form_losses_lastN"] = grp["_is_loss"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).sum()
    ).reset_index(level=0, drop=True)

    keep = [
        "fixture_id", "date", "league_id", "season", "team_id", "team_name",
        "opponent_id", "opponent_name", "is_home", "result", "points",
        "goals_for", "goals_against", "goal_diff",
        "form_pts_lastN", "form_wins_lastN", "form_draws_lastN", "form_losses_lastN",
        "form_gf_avg_lastN", "form_ga_avg_lastN", "form_gd_avg_lastN",
    ]
    return df[keep].reset_index(drop=True)


def build_match_features(matches: pd.DataFrame, team_form: pd.DataFrame) -> pd.DataFrame:
    """Per-match feature row joining home and away pre-match form. ML-ready."""
    if matches.empty or team_form.empty:
        return pd.DataFrame()

    home_form = team_form[team_form["is_home"]].add_prefix("home_")
    home_form = home_form.rename(columns={"home_fixture_id": "fixture_id"})
    away_form = team_form[~team_form["is_home"]].add_prefix("away_")
    away_form = away_form.rename(columns={"away_fixture_id": "fixture_id"})

    base_cols = [
        "fixture_id", "date", "league_id", "season", "round",
        "home_team_id", "home_team_name", "away_team_id", "away_team_name",
        "goals_home", "goals_away", "status_short",
    ]
    base_present = [c for c in base_cols if c in matches.columns]
    base = matches[base_present].copy()

    feature_cols_form = [
        "form_pts_lastN", "form_wins_lastN", "form_draws_lastN", "form_losses_lastN",
        "form_gf_avg_lastN", "form_ga_avg_lastN", "form_gd_avg_lastN",
    ]
    home_keep = ["fixture_id"] + [f"home_{c}" for c in feature_cols_form]
    away_keep = ["fixture_id"] + [f"away_{c}" for c in feature_cols_form]

    out = base.merge(home_form[home_keep], on="fixture_id", how="left")
    out = out.merge(away_form[away_keep], on="fixture_id", how="left")

    # Final outcome label (useful as ML target).
    def _label(row):
        gh, ga = row.get("goals_home"), row.get("goals_away")
        if pd.isna(gh) or pd.isna(ga):
            return None
        if gh > ga:
            return "H"
        if gh < ga:
            return "A"
        return "D"

    out["target_result"] = out.apply(_label, axis=1)
    out["target_total_goals"] = out["goals_home"].fillna(0) + out["goals_away"].fillna(0)
    return out


def build_head_to_head(matches: pd.DataFrame) -> pd.DataFrame:
    """All-time head-to-head aggregates between every team pair (order-independent)."""
    if matches.empty:
        return pd.DataFrame()

    df = matches.dropna(subset=["home_team_id", "away_team_id", "goals_home", "goals_away"]).copy()
    if df.empty:
        return pd.DataFrame()

    # Make order-independent pair: smaller id first.
    df["team_a_id"] = np.minimum(df["home_team_id"], df["away_team_id"]).astype("Int64")
    df["team_b_id"] = np.maximum(df["home_team_id"], df["away_team_id"]).astype("Int64")
    df["a_is_home"] = df["team_a_id"] == df["home_team_id"]
    df["a_goals"] = np.where(df["a_is_home"], df["goals_home"], df["goals_away"])
    df["b_goals"] = np.where(df["a_is_home"], df["goals_away"], df["goals_home"])
    df["a_wins"] = (df["a_goals"] > df["b_goals"]).astype(int)
    df["b_wins"] = (df["a_goals"] < df["b_goals"]).astype(int)
    df["draws"] = (df["a_goals"] == df["b_goals"]).astype(int)

    grp = (
        df.groupby(["team_a_id", "team_b_id"], dropna=False)
          .agg(matches_played=("fixture_id", "nunique"),
               team_a_wins=("a_wins", "sum"),
               team_b_wins=("b_wins", "sum"),
               draws=("draws", "sum"),
               team_a_goals=("a_goals", "sum"),
               team_b_goals=("b_goals", "sum"))
          .reset_index()
    )
    return grp


# --------------------------------------------------------------------------- #
# Derived: player-level
# --------------------------------------------------------------------------- #

def build_top_scorers(match_player_stats: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Goals/assists per player per (league, season), aggregated from match-level data."""
    if match_player_stats.empty:
        return pd.DataFrame()

    if not matches.empty and "fixture_id" in matches.columns:
        m = matches[["fixture_id", "league_id", "season"]].drop_duplicates("fixture_id")
        df = match_player_stats.merge(m, on="fixture_id", how="left")
    else:
        df = match_player_stats.copy()
        df["league_id"] = np.nan
        df["season"] = np.nan

    for col in ["goals_total", "goals_assists", "minutes", "shots_total", "shots_on"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    grp = (
        df.groupby(["league_id", "season", "team_id", "team_name", "player_id", "player_name"], dropna=False)
          .agg(matches_played=("fixture_id", "nunique"),
               minutes=("minutes", "sum"),
               goals=("goals_total", "sum"),
               assists=("goals_assists", "sum"),
               shots_total=("shots_total", "sum"),
               shots_on=("shots_on", "sum"))
          .reset_index()
    )
    grp["goal_contributions"] = grp["goals"] + grp["assists"]
    grp["goals_per_90"] = np.where(
        grp["minutes"] > 0, grp["goals"] * 90.0 / grp["minutes"], 0.0
    )
    grp = grp.sort_values(["league_id", "season", "goals", "assists"], ascending=[True, True, False, False])
    return grp.reset_index(drop=True)


def build_player_minutes_summary(match_player_stats: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season) playing time + average rating summary."""
    if match_player_stats.empty:
        return pd.DataFrame()
    if not matches.empty and "fixture_id" in matches.columns:
        m = matches[["fixture_id", "season"]].drop_duplicates("fixture_id")
        df = match_player_stats.merge(m, on="fixture_id", how="left")
    else:
        df = match_player_stats.copy()
        df["season"] = np.nan

    df["minutes"] = pd.to_numeric(df.get("minutes"), errors="coerce")
    df["rating"] = pd.to_numeric(df.get("rating"), errors="coerce")
    df["substitute"] = df.get("substitute")

    grp = (
        df.groupby(["season", "player_id", "player_name"], dropna=False)
          .agg(matches=("fixture_id", "nunique"),
               minutes_total=("minutes", "sum"),
               minutes_avg=("minutes", "mean"),
               rating_avg=("rating", "mean"),
               starts=("substitute", lambda s: (s == False).sum()),  # noqa: E712
               sub_apps=("substitute", lambda s: (s == True).sum()))  # noqa: E712
          .reset_index()
    )
    return grp


# --------------------------------------------------------------------------- #
# Derived: referee
# --------------------------------------------------------------------------- #

def build_referee_stats(matches: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per referee: matches officiated and average yellow/red card counts."""
    if matches.empty or "referee" not in matches.columns:
        return pd.DataFrame()

    base = matches.dropna(subset=["referee"]).copy()
    if events.empty or "type" not in events.columns:
        cards_per_match = pd.DataFrame(columns=["fixture_id", "yellow_cards", "red_cards"])
    else:
        ev = events.copy()
        ev["is_yellow"] = (ev["type"].astype(str).str.lower() == "card") & \
                          (ev["detail"].astype(str).str.lower().str.contains("yellow", na=False))
        ev["is_red"] = (ev["type"].astype(str).str.lower() == "card") & \
                       (ev["detail"].astype(str).str.lower().str.contains("red", na=False))
        cards_per_match = (
            ev.groupby("fixture_id")
              .agg(yellow_cards=("is_yellow", "sum"),
                   red_cards=("is_red", "sum"))
              .reset_index()
        )

    merged = base.merge(cards_per_match, on="fixture_id", how="left")
    merged["yellow_cards"] = merged["yellow_cards"].fillna(0)
    merged["red_cards"] = merged["red_cards"].fillna(0)

    grp = (
        merged.groupby("referee", dropna=False)
              .agg(matches_officiated=("fixture_id", "nunique"),
                   yellow_total=("yellow_cards", "sum"),
                   red_total=("red_cards", "sum"),
                   yellow_avg=("yellow_cards", "mean"),
                   red_avg=("red_cards", "mean"))
              .reset_index()
              .sort_values("matches_officiated", ascending=False)
    )
    return grp.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    """Build all derived tables from base CSVs."""
    print("[INFO] === derive_tables_local: start ===")

    teams = _read("dim_teams.csv")
    matches = _read("fact_matches.csv")
    standings = _read("fact_standings.csv")
    season_stats = _read("fact_player_season_stats.csv")
    events = _read("fact_match_events.csv")
    team_stats = _read("fact_match_team_stats.csv")
    lineups = _read("fact_match_lineups.csv")
    match_player_stats = _read("fact_match_player_stats.csv")

    # Dimensions
    _write(build_dim_players(season_stats, match_player_stats), "dim_players.csv")
    _write(build_dim_venues(teams, matches), "dim_venues.csv")
    _write(build_dim_coaches(lineups), "dim_coaches.csv")

    # Match-level
    wide = build_team_match_stats_wide(team_stats)
    _write(wide, "fact_match_team_stats_wide.csv")

    match_long = build_match_long(matches)
    _write(match_long, "fact_match_long.csv")

    team_form = build_team_form(match_long, window=ROLLING_WINDOW)
    _write(team_form, "fact_team_form.csv")

    features = build_match_features(matches, team_form)
    _write(features, "fact_match_features.csv")

    h2h = build_head_to_head(matches)
    _write(h2h, "fact_head_to_head.csv")

    # Player-level
    _write(build_top_scorers(match_player_stats, matches), "fact_top_scorers.csv")
    _write(build_player_minutes_summary(match_player_stats, matches), "fact_player_minutes_summary.csv")

    # Referee
    _write(build_referee_stats(matches, events), "fact_referee_stats.csv")

    # Pass-through (helpful for one-stop folder)
    if not standings.empty:
        _write(standings, "fact_standings_snapshot.csv")

    print("[INFO] === derive_tables_local: done ===")


if __name__ == "__main__":
    main()
