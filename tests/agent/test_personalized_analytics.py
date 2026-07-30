from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "football-cluster-agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from football_agent.analytics import (  # noqa: E402
    personalized_similarity,
    predict_match_from_history,
    summarize_latest_matches,
)


def test_personalized_similarity_preserves_cluster_and_ranks_attack_peer() -> None:
    profiles = pd.DataFrame(
        [
            {
                "team_id": 1,
                "team_name": "Alpha",
                "avg_goals_scored": 3.0,
                "avg_goals_conceded": 1.0,
                "attack_strength": 5.4,
                "defense_strength": 0.6,
                "win_rate": 0.8,
                "avg_points": 2.5,
                "std_goal_diff": 1.1,
            },
            {
                "team_id": 2,
                "team_name": "Beta",
                "avg_goals_scored": 2.8,
                "avg_goals_conceded": 1.1,
                "attack_strength": 5.0,
                "defense_strength": 0.58,
                "win_rate": 0.75,
                "avg_points": 2.35,
                "std_goal_diff": 1.0,
            },
            {
                "team_id": 3,
                "team_name": "Gamma",
                "avg_goals_scored": 0.8,
                "avg_goals_conceded": 0.7,
                "attack_strength": 1.1,
                "defense_strength": 0.9,
                "win_rate": 0.35,
                "avg_points": 1.2,
                "std_goal_diff": 0.4,
            },
            {
                "team_id": 4,
                "team_name": "Delta",
                "avg_goals_scored": 1.2,
                "avg_goals_conceded": 2.2,
                "attack_strength": 1.5,
                "defense_strength": 0.35,
                "win_rate": 0.2,
                "avg_points": 0.8,
                "std_goal_diff": 1.8,
            },
        ]
    )
    clusters = pd.DataFrame(
        [
            {"team_id": 1, "cluster": 7, "cluster_label": "Attack"},
            {"team_id": 2, "cluster": 7, "cluster_label": "Attack"},
            {"team_id": 3, "cluster": 2, "cluster_label": "Control"},
            {"team_id": 4, "cluster": 4, "cluster_label": "Volatile"},
        ]
    )

    result = personalized_similarity(
        profiles,
        clusters,
        "Alpha",
        {
            "attack": 1.0,
            "defense": 0.0,
            "results": 0.7,
            "recent_form": 0.0,
            "consistency": 0.0,
        },
        top_n=2,
    )

    assert result["official_cluster"] == 7
    assert result["similar_teams"][0]["team_name"] == "Beta"
    assert result["similar_teams"][0]["relative_similarity_index"] <= 100
    assert len(result["distance_to_each_cluster"]) == 3
    assert "probability" not in result["similar_teams"][0]
    assert any(
        "not probabilities" in warning.lower() for warning in result["warnings"]
    )


def _forecast_history() -> pd.DataFrame:
    newest = datetime.now(timezone.utc) - timedelta(days=1)
    rows: list[dict[str, object]] = []
    for index in range(12):
        match_date = newest - timedelta(days=index)
        rows.append(
            {
                "fixture_id": 1000 + index,
                "date": match_date.isoformat(),
                "league_id": 140,
                "season": 2026,
                "home_team_name": "Alpha",
                "away_team_name": "Gamma",
                "goals_home": 3,
                "goals_away": 1 if index % 3 == 0 else 0,
            }
        )
        rows.append(
            {
                "fixture_id": 2000 + index,
                "date": match_date.isoformat(),
                "league_id": 140,
                "season": 2026,
                "home_team_name": "Gamma",
                "away_team_name": "Beta",
                "goals_home": 3,
                "goals_away": 1 if index % 4 == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


def test_poisson_forecast_is_normalized_and_marks_probabilities_uncalibrated() -> None:
    result = predict_match_from_history(
        _forecast_history(),
        home_team="Alpha",
        away_team="Beta",
    )

    probabilities = result["outcome_probabilities"]
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=0.001)
    assert probabilities["home_win"] > probabilities["away_win"]
    assert result["expected_goals"]["home"] > result["expected_goals"]["away"]
    assert result["confidence"] in {"medium", "high"}
    assert any(
        "uncalibrated" in warning.lower() for warning in result["warnings"]
    )


def test_poisson_forecast_rejects_unknown_team() -> None:
    with pytest.raises(ValueError, match="Team not found"):
        predict_match_from_history(
            _forecast_history(),
            home_team="Unknown FC",
            away_team="Beta",
        )


def test_latest_match_summary_reports_coverage_and_staleness() -> None:
    matches = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "date": "2025-05-01T19:00:00+00:00",
                "league_id": 140,
                "season": 2024,
                "home_team_name": "Alpha",
                "away_team_name": "Beta",
                "goals_home": 2,
                "goals_away": 1,
                "status_short": "FT",
                "home_form_pts_lastN": 10,
            },
            {
                "fixture_id": 2,
                "date": "2025-05-10T17:00:00+00:00",
                "league_id": 140,
                "season": 2024,
                "home_team_name": "Gamma",
                "away_team_name": "Alpha",
                "goals_home": 0,
                "goals_away": 3,
                "status_short": "FT",
                "home_form_pts_lastN": 7,
            },
        ]
    )

    result = summarize_latest_matches(
        matches,
        limit=1,
        as_of=datetime(2025, 5, 20, 17, tzinfo=timezone.utc),
    )

    assert result["source_table"] == "fact_match_features"
    assert result["coverage"]["completed_match_count"] == 2
    assert result["coverage"]["latest_match_utc"].startswith("2025-05-10")
    assert result["coverage"]["data_age_days"] == 10
    assert result["latest_matches"][0]["fixture_id"] == 2
    assert "home_form_pts_lastN" in result["available_match_statistics"]
    assert "live data" in result["warning"]
