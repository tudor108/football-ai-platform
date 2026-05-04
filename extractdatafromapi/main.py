"""End-to-end pipeline entrypoint: extract API-Football data and upload to GCS."""

from __future__ import annotations

from pathlib import Path

from extractdatafromapi.extractdata import (
    extract_all_fixture_details,
    extract_all_for_league,
    extract_leagues,
)
from extractdatafromapi.gcs_uploader import upload_local_folder


SEASON = 2024
LA_LIGA_ID = 140
MAX_FIXTURES_PER_RUN = 25


def run_extraction() -> None:
    """Run the daily extraction flow with per-step isolation."""
    steps = (
        ("leagues", lambda: extract_leagues(country="Spain", season=SEASON)),
        ("league bundle", lambda: extract_all_for_league(league_id=LA_LIGA_ID, season=SEASON)),
        (
            "fixture details",
            lambda: extract_all_fixture_details(
                league_id=LA_LIGA_ID,
                season=SEASON,
                max_fixtures=MAX_FIXTURES_PER_RUN,
            ),
        ),
    )
    for name, runner in steps:
        try:
            runner()
        except Exception as error:  # noqa: BLE001 - keep going so other steps + upload run
            print(f"[ERROR] Extraction step '{name}' failed: {error}")


def run_upload() -> None:
    """Upload all newly-extracted files to the GCS bucket."""
    upload_local_folder(Path("data") / "raw")


def main() -> None:
    """Run extraction followed by upload to GCS."""
    print("[INFO] === Step 1: extract from API ===")
    run_extraction()
    print("[INFO] === Step 2: upload to GCS ===")
    try:
        run_upload()
    except Exception as error:  # noqa: BLE001
        print(f"[ERROR] Upload step failed: {error}")
        raise
    print("[INFO] Pipeline complete.")


if __name__ == "__main__":
    main()
