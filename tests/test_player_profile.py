"""Player profile (services/player_profile.py, GET /api/players/profile): totals, per-90 rates,
the points breakdown by FPL's rules, and the gameweek/season timelines."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api_main  # noqa: E402
from config.settings import CURRENT_SEASON, SEASONS  # noqa: E402
from services import player_profile as pp  # noqa: E402

INFO = {"name": "Bukayo Saka", "position": "MID", "team": "Arsenal", "last_team": "Arsenal",
        "status": "d", "news": "Knock", "chance_of_playing_next_round": 75, "fpl_code": 223340}


def match(gw=1, season=CURRENT_SEASON, **kw):
    row = {"season": season, "gameweek": gw, "team": "Arsenal", "was_home": True, "home_team": "Arsenal",
           "away_team": "Chelsea", "home_score": 2, "away_score": 0, "position": "MID", "minutes": 90,
           "total_points": 2, "goals": 0, "assists": 0, "clean_sheets": 0, "goals_conceded": 1, "saves": 0,
           "bonus": 0, "bps": 10, "yellow_cards": 0, "red_cards": 0, "influence": 10.0, "creativity": 20.0,
           "threat": 30.0, "ict_index": 6.0, "value": 100, "selected": 1000, "starts": 1, "xg": 0.3,
           "xa": 0.2, "xgi": 0.5, "xgc": 1.0, "tackles": 2, "recoveries": 5, "cbi": 1,
           "defensive_contribution": 8, "own_goals": None, "penalties_missed": None, "penalties_saved": None}
    row.update(kw)
    return row


def test_match_points_follow_fpl_rules():
    # MID, 90 minutes, goal + assist + clean sheet + 3 bonus + yellow = 2 + 5 + 3 + 1 + 3 - 1 = 13
    parts = pp.match_points(match(goals=1, assists=1, clean_sheets=1, goals_conceded=0, bonus=3,
                                  yellow_cards=1, total_points=13), "MID")
    assert parts["goals"] == 5 and parts["clean_sheets"] == 1 and parts["cards"] == -1
    assert parts["other"] == 0
    # DEF conceding 3 loses 1; 10 defensive contributions earn 2
    parts = pp.match_points(match(goals_conceded=3, defensive_contribution=10, total_points=3), "DEF")
    assert parts["conceded"] == -1 and parts["defensive"] == 2 and parts["other"] == 0
    # GK: 7 saves = 2 points; a missed penalty not stored falls under other
    parts = pp.match_points(match(saves=7, goals_conceded=0, clean_sheets=1, total_points=6), "GK")
    assert parts["saves"] == 2 and parts["clean_sheets"] == 4 and parts["other"] == -2


def test_did_not_play_scores_nothing_but_other():
    assert pp.match_points(match(minutes=0, total_points=0), "MID")["appearance"] == 0


def test_season_summary_totals_rates_and_timeline():
    rows = [match(1, total_points=10, goals=1, minutes=90), match(2, minutes=0, total_points=0, starts=0),
            match(3, total_points=3, minutes=45, was_home=False, home_team="Chelsea", away_team="Arsenal",
                  home_score=1, away_score=1)]
    out = pp.summarize(rows, INFO, CURRENT_SEASON)
    assert out["matches"] == 3 and out["appearances"] == 2
    assert out["totals"]["total_points"] == 13 and out["totals"]["minutes"] == 135
    assert out["per90"]["goals"] == round(1 / 135 * 90, 2)
    assert out["points_per_game"] == 6.5 and out["price"] == 10.0
    assert [g["gameweek"] for g in out["gameweeks"]] == [1, 2, 3]
    assert out["gameweeks"][2]["opponents"] == [{"team": "Chelsea", "home": False, "score": "1-1"}]
    assert out["gameweeks"][0]["opponents"][0]["score"] == "2-0"
    assert out["status"] == "d" and out["seasons"] == []
    assert sum(b["points"] for b in out["breakdown"]) == 13          # the breakdown adds up
    assert out["totals"]["own_goals"] is None                        # not stored yet: null, not 0


def test_double_gameweek_is_one_entry():
    out = pp.summarize([match(5, total_points=6), match(5, total_points=2)], INFO, CURRENT_SEASON)
    assert len(out["gameweeks"]) == 1 and out["gameweeks"][0]["points"] == 8
    assert len(out["gameweeks"][0]["opponents"]) == 2


def test_rates_need_minutes():
    out = pp.summarize([match(1, minutes=10, goals=1, total_points=6)], INFO, CURRENT_SEASON)
    assert out["per90"]["goals"] is None


def test_career_summary_has_seasons_and_no_old_availability():
    past = SEASONS[-2]
    rows = [match(1, season=past, total_points=5), match(1, total_points=7, xg=None)]
    out = pp.summarize(rows, INFO, None)
    assert out["season"] == "All seasons" and out["gameweeks"] == []
    assert [s["season"] for s in out["seasons"]] == [past, CURRENT_SEASON]
    assert out["seasons"][0]["points"] == 5 and out["totals"]["xg"] == 0.3
    past_only = pp.summarize([match(1, season=past)], INFO, past)
    assert past_only["status"] is None and past_only["news"] is None


def test_get_profile_queries_and_errors():
    conn = MagicMock()
    conn.execute_query.side_effect = [[INFO], [match(1, total_points=4)]]
    out = pp.get_player_profile(conn, "Bukayo Saka", CURRENT_SEASON, photo_for=lambda n, t, c: f"p/{c}",
                                predicted_for=lambda n: 5.5)
    assert out["photo"] == "p/223340" and out["predicted_points"] == 5.5 and "_code" not in out
    query, params = conn.execute_query.call_args.args
    assert params == {"player_name": "Bukayo Saka", "season": CURRENT_SEASON}
    assert "$season IS NULL OR r.season = $season" in query
    with pytest.raises(pp.NotFoundError):
        pp.get_player_profile(MagicMock(), "x", "1999-00")
    empty = MagicMock()
    empty.execute_query.return_value = []
    with pytest.raises(pp.NotFoundError):
        pp.get_player_profile(empty, "Nobody")


def test_route():
    conn = MagicMock()
    conn.execute_query.side_effect = [[INFO], [match(1, total_points=4)]]
    state = dict(api_main.app_state, neo4j_conn=conn, avatar_service=None, player_index=None)
    with patch.dict(api_main.app_state, state), patch.object(api_main, "get_predictions_df", return_value=None):
        client = TestClient(api_main.app)
        resp = client.get(f"/api/players/profile?name=Bukayo%20Saka&season={CURRENT_SEASON}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Bukayo Saka" and body["totals"]["total_points"] == 4
        assert body["predicted_points"] is None
        conn.execute_query.side_effect = [[]]
        assert client.get("/api/players/profile?name=Nobody").status_code == 404
        assert client.get("/api/players/profile?name=x").status_code == 422
