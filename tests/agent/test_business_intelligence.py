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

from football_agent.intelligence import (  # noqa: E402
    attach_video_evidence,
    backtest_match_forecasts,
    build_live_match_companion,
    build_opponent_dossier,
    build_personalized_fan_briefing,
    generate_business_alerts,
    rank_player_team_fit,
    simulate_lineup_absences,
)


def _matches(include_upcoming: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    teams = ["Alpha", "Beta", "Gamma"]
    for index in range(30):
        home = teams[index % 3]
        away = teams[(index + 1) % 3]
        home_goals = 3 if home == "Alpha" else 1
        away_goals = 0 if away == "Beta" else 1
        rows.append(
            {
                "fixture_id": 1000 + index,
                "date": (start + timedelta(days=index)).isoformat(),
                "league_id": 140,
                "season": 2024,
                "home_team_name": home,
                "away_team_name": away,
                "goals_home": home_goals,
                "goals_away": away_goals,
                "status_short": "FT",
            }
        )
    if include_upcoming:
        rows.append(
            {
                "fixture_id": 9999,
                "date": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
                "league_id": 140,
                "season": 2026,
                "home_team_name": "Alpha",
                "away_team_name": "Beta",
                "goals_home": None,
                "goals_away": None,
                "status_short": "NS",
            }
        )
    return pd.DataFrame(rows)


def _clusters() -> pd.DataFrame:
    rows = []
    for team_id, team, cluster in ((1, "Alpha", 1), (2, "Beta", 2), (3, "Gamma", 1)):
        rows.append(
            {
                "team_id": team_id,
                "team_name": team,
                "cluster": cluster,
                "cluster_label": "Attacking" if cluster == 1 else "Defensive",
                "assignment_strength": 0.8,
                "avg_goals_scored": 2.5 if team == "Alpha" else 1.0,
                "avg_goals_conceded": 0.8 if team == "Alpha" else 1.5,
                "attack_strength": 4.0 if team == "Alpha" else 1.5,
                "defense_strength": 0.8 if team == "Alpha" else 0.5,
                "win_rate": 0.7 if team == "Alpha" else 0.3,
                "avg_points": 2.2 if team == "Alpha" else 1.1,
                "std_goal_diff": 0.7 if team == "Alpha" else 1.3,
            }
        )
    return pd.DataFrame(rows)


def _player_stats() -> pd.DataFrame:
    rows = []
    players = [
        ("Alpha", 1, "Alpha Star", "F", 5, 2, 2, 1),
        ("Alpha", 2, "Alpha Wall", "D", 0, 0, 1, 8),
        ("Beta", 3, "Beta Creator", "M", 2, 6, 7, 3),
        ("Gamma", 4, "Gamma Forward", "F", 6, 1, 4, 1),
    ]
    for team, player_id, name, position, goals, assists, key_passes, tackles in players:
        for appearance in range(4):
            rows.append(
                {
                    "fixture_id": 2000 + appearance,
                    "team_name": team,
                    "player_id": player_id,
                    "player_name": name,
                    "position": position,
                    "minutes": 90,
                    "rating": 7.0 + goals / 20,
                    "goals_total": goals / 4,
                    "goals_assists": assists / 4,
                    "shots_on": goals / 2,
                    "passes_key": key_passes / 4,
                    "passes_accuracy": 82,
                    "tackles_total": tackles / 4,
                    "tackles_blocks": 1,
                    "tackles_interceptions": tackles / 5,
                    "duels_total": 8,
                    "duels_won": 5,
                    "dribbles_attempts": 3,
                    "dribbles_success": 2,
                }
            )
    return pd.DataFrame(rows)


def _top_scorers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team_name": "Alpha", "player_name": "Alpha Star", "goals": 5, "assists": 2, "minutes": 360},
            {"team_name": "Beta", "player_name": "Beta Creator", "goals": 2, "assists": 6, "minutes": 360},
            {"team_name": "Gamma", "player_name": "Gamma Forward", "goals": 6, "assists": 1, "minutes": 360},
        ]
    )


def test_dossier_is_grounded_and_keeps_forecast_warning() -> None:
    result = build_opponent_dossier(
        _matches(), _clusters(), "Alpha", "Beta", top_scorers=_top_scorers()
    )
    assert result["fixture"] == {"home_team": "Alpha", "away_team": "Beta"}
    assert result["team_profiles"]["home"]["official_cluster"]["cluster_id"] == 1
    assert result["team_profiles"]["home"]["key_players"][0]["player_name"] == "Alpha Star"
    assert any("uncalibrated" in warning.lower() for warning in result["forecast"]["warnings"])


def test_backtest_is_walk_forward_and_reports_calibration_metrics() -> None:
    result = backtest_match_forecasts(_matches(), min_prior_matches=3, max_evaluated=50)
    assert result["evaluated_predictions"] > 0
    assert result["method"].startswith("Walk-forward")
    assert 0 <= result["top_choice_accuracy"] <= 1
    assert result["multiclass_brier_score"] >= 0


def test_alerts_mark_stale_data_and_never_claim_causality() -> None:
    previous = _clusters().copy()
    previous.loc[previous["team_name"] == "Alpha", "cluster"] = 9
    result = generate_business_alerts(
        _matches(),
        _clusters(),
        previous_clusters=previous,
        as_of=datetime(2025, 3, 1, tzinfo=timezone.utc),
        stale_after_days=2,
    )
    assert any(alert["type"] == "data_stale" for alert in result["alerts"])
    movement = next(alert for alert in result["alerts"] if alert["type"] == "cluster_movement")
    assert "not be interpreted as causality" in movement["warning"]


def test_player_fit_is_relative_index_not_probability() -> None:
    result = rank_player_team_fit(
        _player_stats(), _clusters(), "Alpha", position="M", min_minutes=1
    )
    assert result["recommended_players"][0]["player_name"] == "Beta Creator"
    assert "probabilities" in result["warnings"][0]
    assert "probability" not in result["recommended_players"][0]


def test_absence_simulation_is_bounded_and_explicitly_non_causal() -> None:
    result = simulate_lineup_absences(
        _matches(),
        _player_stats(),
        "Alpha",
        "Beta",
        missing_home_players=["Alpha Star"],
    )
    assert result["scenario"]["expected_goals"]["home"] <= result["baseline"]["expected_goals"]["home"]
    assert any("not causal" in warning for warning in result["warnings"])


def test_fan_briefing_keeps_official_cluster_and_labels_personalization() -> None:
    result = build_personalized_fan_briefing(
        _matches(include_upcoming=True),
        _clusters(),
        _top_scorers(),
        ["Alpha"],
        preferences={"attack": 1, "defense": 0, "results": 0, "recent_form": 0, "consistency": 0},
    )
    assert result["team_briefings"][0]["official_cluster"]["cluster_id"] == 1
    assert result["team_briefings"][0]["next_stored_fixture"] is not None
    assert any("not probabilities" in warning for warning in result["warnings"])


def test_match_companion_never_implies_snapshot_is_live() -> None:
    fixtures = _matches().head(1)
    fixture_id = int(fixtures.iloc[0]["fixture_id"])
    events = pd.DataFrame([{"fixture_id": fixture_id, "time_elapsed": 12, "team_name": "Alpha", "player_name": "A", "type": "Goal", "detail": "Normal Goal"}])
    stats = pd.DataFrame([{"fixture_id": fixture_id, "team_name": "Alpha", "stat_type": "Total Shots", "stat_value": "10"}])
    players = pd.DataFrame([{"fixture_id": fixture_id, "team_name": "Alpha", "player_name": "A", "rating": 8.0}])
    result = build_live_match_companion(fixtures, events, stats, players, fixture_id)
    assert result["fixture"]["is_live_according_to_snapshot"] is False
    assert "newest stored snapshot" in result["warning"].lower()


def test_video_contract_returns_only_explicitly_licensed_links() -> None:
    events = pd.DataFrame([{"fixture_id": 1, "time_elapsed": 12, "team_name": "Alpha", "player_name": "A", "type": "Goal", "detail": "Normal Goal"}])
    no_video = attach_video_evidence(events, 1)
    assert no_video["video_evidence_available"] is False
    video_index = pd.DataFrame([{"fixture_id": 1, "event_minute": 12, "video_url": "https://licensed.example/clip", "license": "provider-contract"}])
    with_video = attach_video_evidence(events, 1, video_index)
    assert with_video["timeline"][0]["video"]["url"] == "https://licensed.example/clip"
    assert with_video["timeline"][0]["video"]["license"] == "provider-contract"
