"""FastAPI facade over the same deterministic tools used by the Gemini agent."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from football_agent.tools import (
    audit_match_forecast_quality,
    create_fan_briefing,
    generate_opponent_dossier,
    get_latest_available_matches,
    get_match_companion,
    get_video_evidence,
    list_business_alerts,
    recommend_players_for_team,
    search_historical_analytics,
    simulate_match_absences,
)

app = FastAPI(
    title="Football Analytics Business API",
    version="1.0.0",
    description="White-label JSON access to stored-data analytics and scenarios.",
)


class AbsenceScenario(BaseModel):
    home_team: str
    away_team: str
    missing_home_players: list[str] = Field(default_factory=list)
    missing_away_players: list[str] = Field(default_factory=list)


class FanBriefingRequest(BaseModel):
    followed_teams: list[str] = Field(min_length=1, max_length=10)
    attack_preference: float = Field(0.0, ge=-1, le=1)
    defense_preference: float = Field(0.0, ge=-1, le=1)
    results_preference: float = Field(0.0, ge=-1, le=1)
    recent_form_preference: float = Field(0.0, ge=-1, le=1)
    consistency_preference: float = Field(0.0, ge=-1, le=1)


def _call(function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:  # cloud service failures are not client validation errors
        raise HTTPException(status_code=503, detail=f"Analytics source unavailable: {error}") from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/data/latest")
def latest_matches(limit: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
    return _call(get_latest_available_matches, limit)


@app.get("/v1/dossiers/opponent")
def opponent_dossier(home_team: str, away_team: str) -> dict[str, Any]:
    return _call(generate_opponent_dossier, home_team, away_team)


@app.get("/v1/alerts")
def alerts() -> dict[str, Any]:
    return _call(list_business_alerts)


@app.get("/v1/forecasts/audit")
def forecast_audit(max_evaluated: int = Query(200, ge=20, le=500)) -> dict[str, Any]:
    return _call(audit_match_forecast_quality, max_evaluated)


@app.get("/v1/scouting/player-fit")
def player_fit(
    team_name: str,
    position: str = "",
    attack_preference: float = Query(0.0, ge=-1, le=1),
    defense_preference: float = Query(0.0, ge=-1, le=1),
    creation_preference: float = Query(0.0, ge=-1, le=1),
    reliability_preference: float = Query(0.0, ge=-1, le=1),
    min_minutes: float = Query(180.0, ge=0),
    top_n: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    return _call(
        recommend_players_for_team,
        team_name,
        position,
        attack_preference,
        defense_preference,
        creation_preference,
        reliability_preference,
        min_minutes,
        top_n,
    )


@app.post("/v1/scenarios/absences")
def absence_scenario(request: AbsenceScenario) -> dict[str, Any]:
    return _call(
        simulate_match_absences,
        request.home_team,
        request.away_team,
        request.missing_home_players,
        request.missing_away_players,
    )


@app.post("/v1/fans/briefing")
def fan_briefing(request: FanBriefingRequest) -> dict[str, Any]:
    return _call(create_fan_briefing, **request.model_dump())


@app.get("/v1/matches/{fixture_id}/companion")
def match_companion(fixture_id: int) -> dict[str, Any]:
    return _call(get_match_companion, fixture_id)


@app.get("/v1/matches/{fixture_id}/video-evidence")
def video_evidence(fixture_id: int) -> dict[str, Any]:
    return _call(get_video_evidence, fixture_id)


@app.get("/v1/history/search")
def history_search(
    query: str,
    team_name: str = "",
    run_id: str = "",
    max_results: int = Query(5, ge=1, le=10),
) -> dict[str, Any]:
    return _call(search_historical_analytics, query, team_name, run_id, max_results)
