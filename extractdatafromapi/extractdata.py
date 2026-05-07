from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from requests import Response


BASE_URL = "https://v3.football.api-sports.io"
SOURCE_NAME = "api_football"
DEFAULT_TIMEOUT = 60
DEFAULT_RETRIES = 3
# Free tier API-Football allows 10 requests/minute -> stay safely below that.
REQUEST_INTERVAL_SECONDS = 7.0
# API-Football free plan currently limits players endpoint pagination to page <= 3.
MAX_PLAYERS_PAGE = 3

_API_KEY_CACHE: str | None = None
_LAST_REQUEST_TS: float = 0.0


def load_api_key() -> str:
    """Load the API-Football key from the environment, caching after the first read."""
    global _API_KEY_CACHE
    if _API_KEY_CACHE:
        return _API_KEY_CACHE
    load_dotenv()
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key:
        raise ValueError("Missing API_FOOTBALL_KEY in .env")
    _API_KEY_CACHE = api_key
    return api_key


def build_headers(api_key: str) -> dict[str, str]:
    """Create the authenticated headers required by API-Football."""
    return {"x-apisports-key": api_key}


def throttle() -> None:
    """Pause between API calls so we never exceed the free-tier rate limit."""
    global _LAST_REQUEST_TS
    now = time.monotonic()
    elapsed = now - _LAST_REQUEST_TS
    wait = REQUEST_INTERVAL_SECONDS - elapsed
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST_TS = time.monotonic()


def check_payload_errors(payload: dict[str, Any], endpoint: str) -> None:
    """Raise if API-Football reports errors inside a 200 OK payload."""
    errors = payload.get("errors")
    if isinstance(errors, dict) and errors:
        raise RuntimeError(f"API errors on {endpoint}: {errors}")
    if isinstance(errors, list) and errors:
        raise RuntimeError(f"API errors on {endpoint}: {errors}")


def call_api(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Call an API-Football endpoint with throttling, retries, timeout, and logging."""
    api_key = load_api_key()
    headers = build_headers(api_key)
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"

    for attempt in range(1, DEFAULT_RETRIES + 1):
        throttle()
        try:
            print(f"[INFO] Calling {url} params={params} (attempt {attempt}/{DEFAULT_RETRIES})")
            response = requests.get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as error:
            print(f"[WARN] Network error on {endpoint}: {error}")
            if attempt == DEFAULT_RETRIES:
                raise RuntimeError(f"Network failure after {DEFAULT_RETRIES} attempts: {endpoint}") from error
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 429:
            print(f"[WARN] Rate limited on {endpoint}; sleeping 60s before retry.")
            if attempt == DEFAULT_RETRIES:
                raise RuntimeError(f"Rate-limited after {DEFAULT_RETRIES} attempts: {endpoint}")
            time.sleep(60)
            continue

        if 400 <= response.status_code < 500:
            raise requests.HTTPError(
                f"Client error {response.status_code} on {endpoint}: {response.text}",
                response=response,
            )

        if 500 <= response.status_code < 600:
            print(f"[WARN] Server error {response.status_code} on {endpoint}")
            if attempt == DEFAULT_RETRIES:
                raise requests.HTTPError(
                    f"Server error {response.status_code} after {DEFAULT_RETRIES} attempts: {endpoint}",
                    response=response,
                )
            time.sleep(2 ** attempt)
            continue

        if response.status_code != 200:
            raise requests.HTTPError(
                f"Unexpected status {response.status_code} on {endpoint}: {response.text}",
                response=response,
            )

        payload = response.json()
        check_payload_errors(payload, endpoint)
        return payload

    raise RuntimeError(f"Unreachable retry state for endpoint: {endpoint}")


def validate_response(response: Response) -> None:
    """Validate the HTTP response (kept for backward compatibility)."""
    if response.status_code != 200:
        raise requests.HTTPError(
            f"API returned status {response.status_code}: {response.text}",
            response=response,
        )


def build_output_path(source: str, entity: str, filename: str) -> Path:
    """Compute the dated output path for an entity file and ensure folders exist."""
    extraction_date = date.today().isoformat()
    output_dir = Path("data") / "raw" / source / entity / extraction_date
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


_REMOTE_BLOB_NAMES_CACHE: set[str] | None = None


def get_remote_blob_filenames() -> set[str]:
    """Return the set of just-the-filenames already present in the GCS bucket (cached)."""
    global _REMOTE_BLOB_NAMES_CACHE
    if _REMOTE_BLOB_NAMES_CACHE is not None:
        return _REMOTE_BLOB_NAMES_CACHE

    try:
        from extractdatafromapi.gcs_uploader import list_existing_blob_names, get_bucket_name
        bucket = get_bucket_name()
        full_names = list_existing_blob_names(bucket)
        # Keep just the filename portion so skip checks are filename-based.
        _REMOTE_BLOB_NAMES_CACHE = {name.rsplit("/", 1)[-1] for name in full_names}
        print(f"[INFO] Loaded {len(_REMOTE_BLOB_NAMES_CACHE)} existing remote filenames")
    except Exception as error:  # noqa: BLE001 - GCS is optional locally
        print(f"[WARN] Could not list remote blobs (continuing with local-only skip): {error}")
        _REMOTE_BLOB_NAMES_CACHE = set()

    return _REMOTE_BLOB_NAMES_CACHE


def file_already_exists(source: str, entity: str, filename: str) -> bool:
    """Return True if the filename already exists locally OR in the remote bucket."""
    entity_root = Path("data") / "raw" / source / entity
    if entity_root.exists():
        if list(entity_root.glob(f"*/{filename}")):
            return True
    return filename in get_remote_blob_filenames()


def ensure_local_copy(source: str, entity: str, filename: str) -> Path | None:
    """Make sure a previously-saved file is available locally; download it from GCS if not."""
    entity_root = Path("data") / "raw" / source / entity
    if entity_root.exists():
        local_matches = sorted(entity_root.glob(f"*/{filename}"), reverse=True)
        if local_matches:
            return local_matches[0]

    # Look up the full blob path in GCS by matching the trailing filename.
    try:
        from extractdatafromapi.gcs_uploader import (
            download_blob_to_file,
            get_bucket_name,
            list_existing_blob_names,
        )
        bucket = get_bucket_name()
        prefix = f"raw/{source}/{entity}/"
        all_blobs = list_existing_blob_names(bucket, prefix=prefix)
        candidates = sorted(
            (name for name in all_blobs if name.endswith(f"/{filename}")),
            reverse=True,
        )
        if not candidates:
            return None
        blob_name = candidates[0]
        # Mirror the GCS path locally: strip the leading "raw/" prefix.
        relative = blob_name[len("raw/"):] if blob_name.startswith("raw/") else blob_name
        local_path = Path("data") / "raw" / relative
        print(f"[INFO] Downloading {blob_name} -> {local_path}")
        download_blob_to_file(bucket, blob_name, local_path)
        return local_path
    except Exception as error:  # noqa: BLE001 - GCS optional locally
        print(f"[WARN] Could not fetch {filename} from GCS: {error}")
        return None


def save_json(data: dict[str, Any], source: str, entity: str, filename: str) -> Path:
    """Save a JSON payload to the raw data folder using a dated directory structure."""
    output_path = build_output_path(source, entity, filename)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, ensure_ascii=False, indent=4)

    print(f"[INFO] Saved {entity} data to {output_path}")
    return output_path


def extract_leagues(country: str, season: int) -> dict[str, Any]:
    """Extract league metadata for a country and season; skip if today's file exists."""
    country_slug = slugify(country)
    filename = f"leagues_{country_slug}_{season}.json"
    if file_already_exists(SOURCE_NAME, "leagues", filename):
        print(f"[SKIP] leagues already extracted today: {filename}")
        return {}
    data = call_api("leagues", {"country": country, "season": season})
    save_json(data, SOURCE_NAME, "leagues", filename)
    return data


def extract_teams(league_id: int, season: int) -> dict[str, Any]:
    """Extract teams for a league season; skip if today's file exists."""
    filename = f"teams_league_{league_id}_{season}.json"
    if file_already_exists(SOURCE_NAME, "teams", filename):
        print(f"[SKIP] teams already extracted today: {filename}")
        return {}
    data = call_api("teams", {"league": league_id, "season": season})
    save_json(data, SOURCE_NAME, "teams", filename)
    return data


def extract_standings(league_id: int, season: int) -> dict[str, Any]:
    """Extract standings for a league season; skip if today's file exists."""
    filename = f"standings_league_{league_id}_{season}.json"
    if file_already_exists(SOURCE_NAME, "standings", filename):
        print(f"[SKIP] standings already extracted today: {filename}")
        return {}
    data = call_api("standings", {"league": league_id, "season": season})
    save_json(data, SOURCE_NAME, "standings", filename)
    return data


def extract_fixtures(league_id: int, season: int) -> dict[str, Any]:
    """Extract fixtures for a league season; skip if today's file exists."""
    filename = f"fixtures_league_{league_id}_{season}.json"
    if file_already_exists(SOURCE_NAME, "fixtures", filename):
        print(f"[SKIP] fixtures already extracted today: {filename}")
        return {}
    data = call_api("fixtures", {"league": league_id, "season": season})
    save_json(data, SOURCE_NAME, "fixtures", filename)
    return data


def extract_players_all_pages(league_id: int, season: int) -> list[dict[str, Any]]:
    """Extract every paginated players page; skip pages already saved today."""
    all_pages: list[dict[str, Any]] = []

    page_one_name = f"players_league_{league_id}_{season}_page_1.json"
    if file_already_exists(SOURCE_NAME, "players", page_one_name):
        print(f"[SKIP] players page 1 already extracted.")
        existing_path = ensure_local_copy(SOURCE_NAME, "players", page_one_name)
        if existing_path is None:
            raise FileNotFoundError(
                f"players page 1 marked as existing but could not be located: {page_one_name}"
            )
        with existing_path.open("r", encoding="utf-8") as fh:
            first_page = json.load(fh)
    else:
        first_page = call_api("players", {"league": league_id, "season": season, "page": 1})
        save_json(first_page, SOURCE_NAME, "players", page_one_name)

    all_pages.append(first_page)
    total_pages = get_total_pages(first_page)

    capped_total_pages = min(total_pages, MAX_PLAYERS_PAGE)
    if capped_total_pages < total_pages:
        print(
            f"[WARN] players endpoint reports {total_pages} pages, "
            f"but current plan supports up to page {MAX_PLAYERS_PAGE}. "
            f"Extracting first {capped_total_pages} pages only."
        )

    for page_number in range(2, capped_total_pages + 1):
        filename = f"players_league_{league_id}_{season}_page_{page_number}.json"
        if file_already_exists(SOURCE_NAME, "players", filename):
            print(f"[SKIP] players page {page_number} already extracted today.")
            continue
        page_data = call_api("players", {"league": league_id, "season": season, "page": page_number})
        save_json(page_data, SOURCE_NAME, "players", filename)
        all_pages.append(page_data)

    return all_pages


def get_total_pages(data: dict[str, Any]) -> int:
    """Read the total page count from an API-Football paginated response."""
    paging = data.get("paging")
    if not isinstance(paging, dict):
        raise ValueError("Missing paging information in players response")

    total_pages = paging.get("total")
    if not isinstance(total_pages, int) or total_pages < 1:
        raise ValueError("Invalid paging total in players response")

    return total_pages


def extract_all_for_league(league_id: int, season: int) -> dict[str, Any]:
    """Extract teams, standings, fixtures, and all player pages with per-entity isolation."""
    results: dict[str, Any] = {}
    steps = (
        ("teams", lambda: extract_teams(league_id, season)),
        ("standings", lambda: extract_standings(league_id, season)),
        ("fixtures", lambda: extract_fixtures(league_id, season)),
        ("players", lambda: extract_players_all_pages(league_id, season)),
    )
    for name, runner in steps:
        try:
            results[name] = runner()
        except Exception as error:  # noqa: BLE001 - keep going for other entities
            print(f"[ERROR] Failed to extract {name}: {error}")
            results[name] = None
    return results


def slugify(value: str) -> str:
    """Create a simple filename-safe slug from an input string."""
    return value.strip().lower().replace(" ", "_")


def load_fixture_ids_from_saved_fixtures(league_id: int, season: int) -> list[int]:
    """Read fixture ids from the most recent saved fixtures snapshot for a league/season."""
    target_filename = f"fixtures_league_{league_id}_{season}.json"

    fixtures_path = ensure_local_copy(SOURCE_NAME, "fixtures", target_filename)
    if fixtures_path is None:
        raise FileNotFoundError(
            f"No saved fixtures file for league {league_id} season {season} (local or remote)"
        )
    with fixtures_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    response = payload.get("response", [])
    fixture_ids: list[int] = []
    for item in response:
        fixture = item.get("fixture") if isinstance(item, dict) else None
        if isinstance(fixture, dict) and isinstance(fixture.get("id"), int):
            fixture_ids.append(fixture["id"])

    print(f"[INFO] Loaded {len(fixture_ids)} fixture ids from {fixtures_path}")
    return fixture_ids


def extract_fixture_endpoint(
    fixture_id: int,
    endpoint: str,
    entity: str,
    filename_suffix: str,
) -> dict[str, Any] | None:
    """Generic helper to extract a fixture-level endpoint with skip + save logic."""
    filename = f"fixture_{fixture_id}_{filename_suffix}.json"
    if file_already_exists(SOURCE_NAME, entity, filename):
        print(f"[SKIP] {entity} already extracted today for fixture {fixture_id}")
        return None
    data = call_api(endpoint, {"fixture": fixture_id})
    save_json(data, SOURCE_NAME, entity, filename)
    return data


def extract_fixture_events(fixture_id: int) -> dict[str, Any] | None:
    """Extract events for a single fixture."""
    return extract_fixture_endpoint(fixture_id, "fixtures/events", "fixture_events", "events")


def extract_fixture_statistics(fixture_id: int) -> dict[str, Any] | None:
    """Extract statistics for a single fixture."""
    return extract_fixture_endpoint(fixture_id, "fixtures/statistics", "fixture_statistics", "statistics")


def extract_fixture_lineups(fixture_id: int) -> dict[str, Any] | None:
    """Extract lineups for a single fixture."""
    return extract_fixture_endpoint(fixture_id, "fixtures/lineups", "fixture_lineups", "lineups")


def extract_fixture_players(fixture_id: int) -> dict[str, Any] | None:
    """Extract player stats for a single fixture."""
    return extract_fixture_endpoint(fixture_id, "fixtures/players", "fixture_players", "players")


def extract_all_fixture_details(
    league_id: int,
    season: int,
    max_fixtures: int | None = None,
) -> dict[str, dict[str, int]]:
    """Extract events, statistics, lineups, and players for many fixtures with isolation."""
    fixture_ids = load_fixture_ids_from_saved_fixtures(league_id, season)
    if max_fixtures is not None:
        fixture_ids = fixture_ids[:max_fixtures]

    print(f"[INFO] Extracting details for {len(fixture_ids)} fixtures")
    summary: dict[str, dict[str, int]] = {
        "events": {"ok": 0, "failed": 0},
        "statistics": {"ok": 0, "failed": 0},
        "lineups": {"ok": 0, "failed": 0},
        "players": {"ok": 0, "failed": 0},
    }

    extractors = (
        ("events", extract_fixture_events),
        ("statistics", extract_fixture_statistics),
        ("lineups", extract_fixture_lineups),
        ("players", extract_fixture_players),
    )

    for index, fixture_id in enumerate(fixture_ids, start=1):
        print(f"[INFO] ({index}/{len(fixture_ids)}) Fixture {fixture_id}")
        for label, fn in extractors:
            try:
                fn(fixture_id)
                summary[label]["ok"] += 1
            except Exception as error:  # noqa: BLE001 - per-fixture isolation
                summary[label]["failed"] += 1
                print(f"[ERROR] fixture {fixture_id} {label}: {error}")

    print(f"[INFO] Fixture details summary: {summary}")
    return summary


def main() -> None:
    """Run the default extraction flow for Spain and La Liga."""
    season = 2024
    extract_leagues(country="Spain", season=season)
    extract_all_for_league(league_id=140, season=season)
    extract_all_fixture_details(league_id=140, season=season, max_fixtures=50)


if __name__ == "__main__":
    main()
