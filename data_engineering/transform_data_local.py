"""Transform raw API-Football JSON in GCS into clean tabular CSVs saved locally.

Reads JSON blobs from gs://$GCS_BUCKET/raw/api_football/{entity}/{date}/*.json,
flattens them into pandas DataFrames, deduplicates, and writes CSVs under
./output/tables/. No BigQuery loading is performed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google.cloud import storage


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RAW_PREFIX = "raw/api_football"
OUTPUT_DIR = Path("output") / "tables"

ENTITY_LEAGUES = "leagues"
ENTITY_TEAMS = "teams"
ENTITY_FIXTURES = "fixtures"
ENTITY_STANDINGS = "standings"
ENTITY_PLAYERS = "players"
ENTITY_FIXTURE_EVENTS = "fixture_events"
ENTITY_FIXTURE_STATISTICS = "fixture_statistics"
ENTITY_FIXTURE_LINEUPS = "fixture_lineups"
ENTITY_FIXTURE_PLAYERS = "fixture_players"


def get_bucket_name() -> str:
    """Read the target GCS bucket name from the environment."""
    load_dotenv()
    bucket_name = os.getenv("GCS_BUCKET")
    if not bucket_name:
        raise ValueError("Missing GCS_BUCKET environment variable")
    return bucket_name


def get_storage_client() -> storage.Client:
    """Return an authenticated GCS client (uses Application Default Credentials)."""
    load_dotenv()
    project_id = os.getenv("GCP_PROJECT_ID")
    if project_id:
        return storage.Client(project=project_id)
    return storage.Client()


# --------------------------------------------------------------------------- #
# GCS I/O
# --------------------------------------------------------------------------- #

def list_gcs_files(prefix: str) -> list[str]:
    """List every blob name in the configured bucket under a given prefix."""
    client = get_storage_client()
    bucket_name = get_bucket_name()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    names = [blob.name for blob in blobs if blob.name.endswith(".json")]
    print(f"[INFO] Found {len(names)} JSON files under gs://{bucket_name}/{prefix}")
    return names


def read_json_from_gcs(blob_name: str) -> dict:
    """Read a single JSON blob and return it as a Python dict."""
    client = get_storage_client()
    bucket = client.bucket(get_bucket_name())
    blob = bucket.blob(blob_name)
    raw_bytes = blob.download_as_bytes()
    return json.loads(raw_bytes)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _now_date_iso() -> str:
    """Return today's date (UTC) as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _add_metadata(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Attach ingestion_date + source_file metadata columns."""
    if df.empty:
        return df
    df = df.copy()
    df["ingestion_date"] = _now_date_iso()
    df["source_file"] = source_file
    return df


def _safe_response(payload: dict) -> list[dict]:
    """Return the API-Football `response` array as a list, defensively."""
    response = payload.get("response", []) if isinstance(payload, dict) else []
    return response if isinstance(response, list) else []


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def transform_teams(data: dict, source_file: str) -> pd.DataFrame:
    """Flatten the /teams payload into one row per team."""
    rows: list[dict] = []
    for item in _safe_response(data):
        if not isinstance(item, dict):
            continue
        team = item.get("team", {}) or {}
        venue = item.get("venue", {}) or {}
        rows.append(
            {
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "team_code": team.get("code"),
                "country": team.get("country"),
                "founded": team.get("founded"),
                "national": team.get("national"),
                "logo": team.get("logo"),
                "venue_id": venue.get("id"),
                "venue_name": venue.get("name"),
                "venue_city": venue.get("city"),
                "venue_capacity": venue.get("capacity"),
                "venue_surface": venue.get("surface"),
            }
        )
    df = pd.DataFrame(rows)
    return _add_metadata(df, source_file)


def transform_fixtures(data: dict, source_file: str) -> pd.DataFrame:
    """Flatten the /fixtures payload into one row per fixture."""
    rows: list[dict] = []
    for item in _safe_response(data):
        if not isinstance(item, dict):
            continue
        fixture = item.get("fixture", {}) or {}
        league = item.get("league", {}) or {}
        teams = item.get("teams", {}) or {}
        goals = item.get("goals", {}) or {}
        score = item.get("score", {}) or {}
        venue = fixture.get("venue", {}) or {}
        status = fixture.get("status", {}) or {}
        home = teams.get("home", {}) or {}
        away = teams.get("away", {}) or {}
        score_ht = score.get("halftime", {}) or {}
        score_ft = score.get("fulltime", {}) or {}
        score_et = score.get("extratime", {}) or {}
        score_pen = score.get("penalty", {}) or {}

        rows.append(
            {
                "fixture_id": fixture.get("id"),
                "referee": fixture.get("referee"),
                "timezone": fixture.get("timezone"),
                "date": fixture.get("date"),
                "timestamp": fixture.get("timestamp"),
                "venue_id": venue.get("id"),
                "venue_name": venue.get("name"),
                "venue_city": venue.get("city"),
                "status_long": status.get("long"),
                "status_short": status.get("short"),
                "status_elapsed": status.get("elapsed"),
                "league_id": league.get("id"),
                "league_name": league.get("name"),
                "league_country": league.get("country"),
                "season": league.get("season"),
                "round": league.get("round"),
                "home_team_id": home.get("id"),
                "home_team_name": home.get("name"),
                "home_team_winner": home.get("winner"),
                "away_team_id": away.get("id"),
                "away_team_name": away.get("name"),
                "away_team_winner": away.get("winner"),
                "goals_home": goals.get("home"),
                "goals_away": goals.get("away"),
                "score_ht_home": score_ht.get("home"),
                "score_ht_away": score_ht.get("away"),
                "score_ft_home": score_ft.get("home"),
                "score_ft_away": score_ft.get("away"),
                "score_et_home": score_et.get("home"),
                "score_et_away": score_et.get("away"),
                "score_pen_home": score_pen.get("home"),
                "score_pen_away": score_pen.get("away"),
            }
        )
    df = pd.DataFrame(rows)
    return _add_metadata(df, source_file)


def transform_standings(data: dict, source_file: str) -> pd.DataFrame:
    """Flatten the /standings payload into one row per (league, season, team)."""
    rows: list[dict] = []
    for item in _safe_response(data):
        if not isinstance(item, dict):
            continue
        league = item.get("league", {}) or {}
        league_id = league.get("id")
        league_name = league.get("name")
        league_country = league.get("country")
        season = league.get("season")

        # standings is a list of groups (e.g., regular season, relegation), each a list of team rows.
        groups = league.get("standings") or []
        for group in groups:
            if not isinstance(group, list):
                continue
            for entry in group:
                if not isinstance(entry, dict):
                    continue
                team = entry.get("team", {}) or {}
                all_stats = entry.get("all", {}) or {}
                home_stats = entry.get("home", {}) or {}
                away_stats = entry.get("away", {}) or {}
                all_goals = all_stats.get("goals", {}) or {}

                rows.append(
                    {
                        "league_id": league_id,
                        "league_name": league_name,
                        "league_country": league_country,
                        "season": season,
                        "group": entry.get("group"),
                        "team_id": team.get("id"),
                        "team_name": team.get("name"),
                        "rank": entry.get("rank"),
                        "points": entry.get("points"),
                        "goals_diff": entry.get("goalsDiff"),
                        "form": entry.get("form"),
                        "status": entry.get("status"),
                        "description": entry.get("description"),
                        "played": all_stats.get("played"),
                        "win": all_stats.get("win"),
                        "draw": all_stats.get("draw"),
                        "lose": all_stats.get("lose"),
                        "goals_for": all_goals.get("for"),
                        "goals_against": all_goals.get("against"),
                        "home_played": home_stats.get("played"),
                        "home_win": home_stats.get("win"),
                        "home_draw": home_stats.get("draw"),
                        "home_lose": home_stats.get("lose"),
                        "away_played": away_stats.get("played"),
                        "away_win": away_stats.get("win"),
                        "away_draw": away_stats.get("draw"),
                        "away_lose": away_stats.get("lose"),
                        "update": entry.get("update"),
                    }
                )
    df = pd.DataFrame(rows)
    return _add_metadata(df, source_file)


def transform_leagues(data: dict, source_file: str) -> pd.DataFrame:
    """Flatten /leagues into one row per (league, season)."""
    rows: list[dict] = []
    for item in _safe_response(data):
        if not isinstance(item, dict):
            continue
        league = item.get("league", {}) or {}
        country = item.get("country", {}) or {}
        seasons = item.get("seasons", []) or []
        league_id = league.get("id")
        if not seasons:
            seasons = [{}]
        for season_entry in seasons:
            if not isinstance(season_entry, dict):
                continue
            coverage = season_entry.get("coverage", {}) or {}
            fixtures_cov = coverage.get("fixtures", {}) or {}
            rows.append(
                {
                    "league_id": league_id,
                    "league_name": league.get("name"),
                    "league_type": league.get("type"),
                    "league_logo": league.get("logo"),
                    "country_name": country.get("name"),
                    "country_code": country.get("code"),
                    "country_flag": country.get("flag"),
                    "season": season_entry.get("year"),
                    "season_start": season_entry.get("start"),
                    "season_end": season_entry.get("end"),
                    "season_current": season_entry.get("current"),
                    "coverage_standings": coverage.get("standings"),
                    "coverage_players": coverage.get("players"),
                    "coverage_top_scorers": coverage.get("top_scorers"),
                    "coverage_top_assists": coverage.get("top_assists"),
                    "coverage_top_cards": coverage.get("top_cards"),
                    "coverage_injuries": coverage.get("injuries"),
                    "coverage_predictions": coverage.get("predictions"),
                    "coverage_odds": coverage.get("odds"),
                    "coverage_fixtures_events": fixtures_cov.get("events"),
                    "coverage_fixtures_lineups": fixtures_cov.get("lineups"),
                    "coverage_fixtures_statistics_fixtures": fixtures_cov.get("statistics_fixtures"),
                    "coverage_fixtures_statistics_players": fixtures_cov.get("statistics_players"),
                }
            )
    df = pd.DataFrame(rows)
    return _add_metadata(df, source_file)


def transform_players(data: dict, source_file: str) -> pd.DataFrame:
    """Flatten /players into one row per (player, team, league, season).

    Each player can have multiple statistics blocks (one per team/league).
    """
    rows: list[dict] = []
    for item in _safe_response(data):
        if not isinstance(item, dict):
            continue
        player = item.get("player", {}) or {}
        birth = player.get("birth", {}) or {}
        statistics = item.get("statistics", []) or []
        for stat in statistics:
            if not isinstance(stat, dict):
                continue
            team = stat.get("team", {}) or {}
            league = stat.get("league", {}) or {}
            games = stat.get("games", {}) or {}
            substitutes = stat.get("substitutes", {}) or {}
            shots = stat.get("shots", {}) or {}
            goals = stat.get("goals", {}) or {}
            passes = stat.get("passes", {}) or {}
            tackles = stat.get("tackles", {}) or {}
            duels = stat.get("duels", {}) or {}
            dribbles = stat.get("dribbles", {}) or {}
            fouls = stat.get("fouls", {}) or {}
            cards = stat.get("cards", {}) or {}
            penalty = stat.get("penalty", {}) or {}

            rows.append(
                {
                    "player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "first_name": player.get("firstname"),
                    "last_name": player.get("lastname"),
                    "age": player.get("age"),
                    "nationality": player.get("nationality"),
                    "height": player.get("height"),
                    "weight": player.get("weight"),
                    "injured": player.get("injured"),
                    "photo": player.get("photo"),
                    "birth_date": birth.get("date"),
                    "birth_place": birth.get("place"),
                    "birth_country": birth.get("country"),
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "league_id": league.get("id"),
                    "league_name": league.get("name"),
                    "league_country": league.get("country"),
                    "season": league.get("season"),
                    "position": games.get("position"),
                    "rating": games.get("rating"),
                    "captain": games.get("captain"),
                    "appearances": games.get("appearences"),
                    "lineups": games.get("lineups"),
                    "minutes": games.get("minutes"),
                    "subs_in": substitutes.get("in"),
                    "subs_out": substitutes.get("out"),
                    "subs_bench": substitutes.get("bench"),
                    "shots_total": shots.get("total"),
                    "shots_on": shots.get("on"),
                    "goals_total": goals.get("total"),
                    "goals_conceded": goals.get("conceded"),
                    "goals_assists": goals.get("assists"),
                    "goals_saves": goals.get("saves"),
                    "passes_total": passes.get("total"),
                    "passes_key": passes.get("key"),
                    "passes_accuracy": passes.get("accuracy"),
                    "tackles_total": tackles.get("total"),
                    "tackles_blocks": tackles.get("blocks"),
                    "tackles_interceptions": tackles.get("interceptions"),
                    "duels_total": duels.get("total"),
                    "duels_won": duels.get("won"),
                    "dribbles_attempts": dribbles.get("attempts"),
                    "dribbles_success": dribbles.get("success"),
                    "dribbles_past": dribbles.get("past"),
                    "fouls_drawn": fouls.get("drawn"),
                    "fouls_committed": fouls.get("committed"),
                    "cards_yellow": cards.get("yellow"),
                    "cards_yellowred": cards.get("yellowred"),
                    "cards_red": cards.get("red"),
                    "penalty_won": penalty.get("won"),
                    "penalty_committed": penalty.get("commited"),
                    "penalty_scored": penalty.get("scored"),
                    "penalty_missed": penalty.get("missed"),
                    "penalty_saved": penalty.get("saved"),
                }
            )
    df = pd.DataFrame(rows)
    return _add_metadata(df, source_file)


def _fixture_id_from_filename(source_file: str) -> int | None:
    """Pull the fixture id out of a filename like 'fixture_12345_events.json'."""
    name = source_file.rsplit("/", 1)[-1]
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] == "fixture":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def transform_fixture_events(data: dict, source_file: str) -> pd.DataFrame:
    """Flatten /fixtures/events into one row per event (goal/card/sub)."""
    fixture_id = _fixture_id_from_filename(source_file)
    rows: list[dict] = []
    for event in _safe_response(data):
        if not isinstance(event, dict):
            continue
        time = event.get("time", {}) or {}
        team = event.get("team", {}) or {}
        player = event.get("player", {}) or {}
        assist = event.get("assist", {}) or {}
        rows.append(
            {
                "fixture_id": fixture_id,
                "time_elapsed": time.get("elapsed"),
                "time_extra": time.get("extra"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "assist_id": assist.get("id"),
                "assist_name": assist.get("name"),
                "type": event.get("type"),
                "detail": event.get("detail"),
                "comments": event.get("comments"),
            }
        )
    df = pd.DataFrame(rows)
    return _add_metadata(df, source_file)


def transform_fixture_statistics(data: dict, source_file: str) -> pd.DataFrame:
    """Flatten /fixtures/statistics into one row per (fixture, team, stat_type)."""
    fixture_id = _fixture_id_from_filename(source_file)
    rows: list[dict] = []
    for item in _safe_response(data):
        if not isinstance(item, dict):
            continue
        team = item.get("team", {}) or {}
        statistics = item.get("statistics", []) or []
        for stat in statistics:
            if not isinstance(stat, dict):
                continue
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "stat_type": stat.get("type"),
                    "stat_value": stat.get("value"),
                }
            )
    df = pd.DataFrame(rows)
    return _add_metadata(df, source_file)


def transform_fixture_lineups(data: dict, source_file: str) -> pd.DataFrame:
    """Flatten /fixtures/lineups into one row per (fixture, team, player)."""
    fixture_id = _fixture_id_from_filename(source_file)
    rows: list[dict] = []
    for item in _safe_response(data):
        if not isinstance(item, dict):
            continue
        team = item.get("team", {}) or {}
        coach = item.get("coach", {}) or {}
        formation = item.get("formation")

        # Starting XI
        for entry in item.get("startXI", []) or []:
            if not isinstance(entry, dict):
                continue
            player = entry.get("player", {}) or {}
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "formation": formation,
                    "coach_id": coach.get("id"),
                    "coach_name": coach.get("name"),
                    "player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "player_number": player.get("number"),
                    "player_pos": player.get("pos"),
                    "player_grid": player.get("grid"),
                    "is_starter": True,
                }
            )

        # Substitutes
        for entry in item.get("substitutes", []) or []:
            if not isinstance(entry, dict):
                continue
            player = entry.get("player", {}) or {}
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "formation": formation,
                    "coach_id": coach.get("id"),
                    "coach_name": coach.get("name"),
                    "player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "player_number": player.get("number"),
                    "player_pos": player.get("pos"),
                    "player_grid": player.get("grid"),
                    "is_starter": False,
                }
            )
    df = pd.DataFrame(rows)
    return _add_metadata(df, source_file)


def transform_fixture_players(data: dict, source_file: str) -> pd.DataFrame:
    """Flatten /fixtures/players into one row per (fixture, team, player)."""
    fixture_id = _fixture_id_from_filename(source_file)
    rows: list[dict] = []
    for team_block in _safe_response(data):
        if not isinstance(team_block, dict):
            continue
        team = team_block.get("team", {}) or {}
        for player_entry in team_block.get("players", []) or []:
            if not isinstance(player_entry, dict):
                continue
            player = player_entry.get("player", {}) or {}
            stats_list = player_entry.get("statistics", []) or []
            stat = stats_list[0] if stats_list and isinstance(stats_list[0], dict) else {}
            games = stat.get("games", {}) or {}
            offsides = stat.get("offsides")
            shots = stat.get("shots", {}) or {}
            goals = stat.get("goals", {}) or {}
            passes = stat.get("passes", {}) or {}
            tackles = stat.get("tackles", {}) or {}
            duels = stat.get("duels", {}) or {}
            dribbles = stat.get("dribbles", {}) or {}
            fouls = stat.get("fouls", {}) or {}
            cards = stat.get("cards", {}) or {}
            penalty = stat.get("penalty", {}) or {}

            rows.append(
                {
                    "fixture_id": fixture_id,
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "minutes": games.get("minutes"),
                    "number": games.get("number"),
                    "position": games.get("position"),
                    "rating": games.get("rating"),
                    "captain": games.get("captain"),
                    "substitute": games.get("substitute"),
                    "offsides": offsides,
                    "shots_total": shots.get("total"),
                    "shots_on": shots.get("on"),
                    "goals_total": goals.get("total"),
                    "goals_conceded": goals.get("conceded"),
                    "goals_assists": goals.get("assists"),
                    "goals_saves": goals.get("saves"),
                    "passes_total": passes.get("total"),
                    "passes_key": passes.get("key"),
                    "passes_accuracy": passes.get("accuracy"),
                    "tackles_total": tackles.get("total"),
                    "tackles_blocks": tackles.get("blocks"),
                    "tackles_interceptions": tackles.get("interceptions"),
                    "duels_total": duels.get("total"),
                    "duels_won": duels.get("won"),
                    "dribbles_attempts": dribbles.get("attempts"),
                    "dribbles_success": dribbles.get("success"),
                    "dribbles_past": dribbles.get("past"),
                    "fouls_drawn": fouls.get("drawn"),
                    "fouls_committed": fouls.get("committed"),
                    "cards_yellow": cards.get("yellow"),
                    "cards_red": cards.get("red"),
                    "penalty_won": penalty.get("won"),
                    "penalty_committed": penalty.get("commited"),
                    "penalty_scored": penalty.get("scored"),
                    "penalty_missed": penalty.get("missed"),
                    "penalty_saved": penalty.get("saved"),
                }
            )
    df = pd.DataFrame(rows)
    return _add_metadata(df, source_file)


# --------------------------------------------------------------------------- #
# Pipeline per entity
# --------------------------------------------------------------------------- #

def _process_entity(
    entity: str,
    transform_fn,
    dedup_subset: list[str],
    output_filename: str,
    extra_prefixes: list[str] | None = None,
) -> pd.DataFrame:
    """Generic pipeline: list -> read -> transform -> concat -> dedup -> save."""
    prefixes = [f"{RAW_PREFIX}/{entity}/"]
    if extra_prefixes:
        prefixes.extend(extra_prefixes)

    blob_names: list[str] = []
    for prefix in prefixes:
        blob_names.extend(list_gcs_files(prefix))

    frames: list[pd.DataFrame] = []
    for blob_name in blob_names:
        print(f"[INFO] Processing {blob_name}")
        try:
            payload = read_json_from_gcs(blob_name)
            df = transform_fn(payload, blob_name)
        except Exception as error:  # noqa: BLE001 - per-file isolation
            print(f"[ERROR] Skipped {blob_name}: {error}")
            continue
        if not df.empty:
            frames.append(df)

    if not frames:
        print(f"[WARN] No rows produced for entity '{entity}'")
        combined = pd.DataFrame()
    else:
        combined = pd.concat(frames, ignore_index=True)
        before = len(combined)
        # Build a temporary dedup key that tolerates NaN (legitimate for events/lineups).
        dedup_helper = combined[dedup_subset].astype(object).fillna("__NA__")
        mask = ~dedup_helper.duplicated(keep="last")
        combined = combined.loc[mask].reset_index(drop=True)
        print(f"[INFO] {entity}: {before} rows -> {len(combined)} after dedup on {dedup_subset}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_filename
    # utf-8-sig writes a BOM so Excel opens the CSV as UTF-8 instead of Windows-1252.
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Wrote {len(combined)} rows -> {output_path}")
    return combined


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _repair_existing_csv_encoding() -> None:
    """Re-write any pre-existing CSVs in OUTPUT_DIR with a UTF-8 BOM so Excel reads them correctly."""
    if not OUTPUT_DIR.exists():
        return
    for csv_path in OUTPUT_DIR.glob("*.csv"):
        try:
            with csv_path.open("rb") as fh:
                head = fh.read(3)
            if head.startswith(b"\xef\xbb\xbf"):
                continue  # already has BOM
            df = pd.read_csv(csv_path, encoding="utf-8")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"[INFO] Repaired encoding (added BOM) -> {csv_path}")
        except Exception as error:  # noqa: BLE001
            print(f"[WARN] Could not repair {csv_path}: {error}")


def main() -> None:
    """Build all dim/fact tables from raw GCS data."""
    print("[INFO] === transform_data_local: start ===")
    _repair_existing_csv_encoding()

    _process_entity(
        entity=ENTITY_LEAGUES,
        transform_fn=transform_leagues,
        dedup_subset=["league_id", "season"],
        output_filename="dim_leagues.csv",
        extra_prefixes=["leagues/"],  # legacy path before the raw/api_football/ restructure
    )
    _process_entity(
        entity=ENTITY_TEAMS,
        transform_fn=transform_teams,
        dedup_subset=["team_id"],
        output_filename="dim_teams.csv",
    )
    _process_entity(
        entity=ENTITY_FIXTURES,
        transform_fn=transform_fixtures,
        dedup_subset=["fixture_id"],
        output_filename="fact_matches.csv",
    )
    _process_entity(
        entity=ENTITY_STANDINGS,
        transform_fn=transform_standings,
        dedup_subset=["team_id", "league_id", "season"],
        output_filename="fact_standings.csv",
    )
    _process_entity(
        entity=ENTITY_PLAYERS,
        transform_fn=transform_players,
        dedup_subset=["player_id", "team_id", "league_id", "season"],
        output_filename="fact_player_season_stats.csv",
    )
    _process_entity(
        entity=ENTITY_FIXTURE_EVENTS,
        transform_fn=transform_fixture_events,
        dedup_subset=["fixture_id", "team_id", "player_id", "time_elapsed", "time_extra", "type", "detail"],
        output_filename="fact_match_events.csv",
    )
    _process_entity(
        entity=ENTITY_FIXTURE_STATISTICS,
        transform_fn=transform_fixture_statistics,
        dedup_subset=["fixture_id", "team_id", "stat_type"],
        output_filename="fact_match_team_stats.csv",
    )
    _process_entity(
        entity=ENTITY_FIXTURE_LINEUPS,
        transform_fn=transform_fixture_lineups,
        dedup_subset=["fixture_id", "team_id", "player_id"],
        output_filename="fact_match_lineups.csv",
    )
    _process_entity(
        entity=ENTITY_FIXTURE_PLAYERS,
        transform_fn=transform_fixture_players,
        dedup_subset=["fixture_id", "team_id", "player_id"],
        output_filename="fact_match_player_stats.csv",
    )

    print("[INFO] === transform_data_local: done ===")


if __name__ == "__main__":
    main()
