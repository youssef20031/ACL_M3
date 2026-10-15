"""Leakage and shape guarantees of ml/points_model.py (synthetic data, no files)."""
import numpy as np
import pandas as pd
import pytest

from ml import points_model as pm


def _match_rows(season, gw, fixture, kickoff, home, away, home_goals, away_goals, players):
    """players: {(name, team, position): points}"""
    rows = []
    for (name, team, pos), pts in players.items():
        if team not in (home, away):
            continue
        rows.append({
            "season": season, "name": name, "position": pos, "team": team, "fixture": fixture,
            "kickoff_time": kickoff, "GW": gw, "was_home": team == home,
            "team_h_score": home_goals, "team_a_score": away_goals, "total_points": pts,
            "minutes": 90, "starts": 1, "goals_scored": 0, "assists": 0, "bonus": 0, "bps": 10,
            "ict_index": 5.0, "influence": 10.0, "creativity": 10.0, "threat": 10.0,
            "clean_sheets": 0, "goals_conceded": 0, "saves": 0, "value": 60, "selected": 1000,
        })
    return rows


@pytest.fixture
def history():
    a = ("Alice", "Arsenal", "MID")
    b = ("Bob", "Chelsea", "FWD")
    c = ("Cara", "Arsenal", "DEF")
    rows = []
    rows += _match_rows("2026-27", 1, 1, "2026-08-22T14:00:00Z", "Arsenal", "Chelsea", 2, 0, {a: 2, b: 1, c: 6})
    rows += _match_rows("2026-27", 2, 2, "2026-08-29T14:00:00Z", "Chelsea", "Arsenal", 1, 1, {a: 4, b: 5, c: 2})
    # GW3: Arsenal double gameweek (fixtures 3 and 4)
    rows += _match_rows("2026-27", 3, 3, "2026-09-05T14:00:00Z", "Arsenal", "Chelsea", 3, 1, {a: 12, b: 2, c: 1})
    rows += _match_rows("2026-27", 3, 4, "2026-09-08T19:00:00Z", "Arsenal", "Leeds", 1, 0, {a: 8, c: 6})
    df = pd.DataFrame(rows)
    for col in pm.HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    return df[pm.HISTORY_COLUMNS]


def test_one_feature_row_per_input_row(history):
    feats = pm.build_features(history)
    assert len(feats) == len(history)  # v2's opponent merges multiplied rows ~20x
    assert set(pm.feature_columns()) <= set(feats.columns)


def test_form_uses_only_earlier_gameweeks(history):
    f = pm.build_features(history).set_index(["name", "fixture"])
    assert np.isnan(f.loc[("Alice", 1), "total_points_r3"])       # nothing before GW1
    assert f.loc[("Alice", 2), "total_points_r3"] == pytest.approx(2)
    assert f.loc[("Alice", 3), "total_points_r3"] == pytest.approx(3)  # mean(2, 4)
    # Second fixture of the double gameweek doesn't see the first (12 points)
    assert f.loc[("Alice", 4), "total_points_r3"] == pytest.approx(3)
    assert f.loc[("Alice", 4), "season_matches_before"] == f.loc[("Alice", 3), "season_matches_before"] == 2


def test_team_strength_excludes_the_match_itself(history):
    f = pm.build_features(history).set_index(["name", "fixture"])
    # Arsenal scored 2 then 1 before GW3: both GW3 fixtures see gf = 1.5, not the GW3 goals
    assert f.loc[("Alice", 3), "team_gf"] == pytest.approx(1.5)
    assert f.loc[("Alice", 4), "team_gf"] == pytest.approx(1.5)
    # GW1 has no history: prior for gf/ga, NaN for xG
    assert f.loc[("Alice", 1), "team_gf"] == pm.NEW_TEAM_PRIOR["team_gf"]
    assert f.loc[("Alice", 3), "opp_ga"] == pytest.approx(1.5)  # Chelsea conceded 2, 1


def test_upcoming_rows_see_full_history_but_not_themselves(history):
    upcoming = pd.DataFrame([{
        "season": "2026-27", "name": "Alice", "position": "MID", "team": "Arsenal", "opponent": "Chelsea",
        "fixture": 99, "kickoff_time": "2026-09-15T14:00:00Z", "GW": 4, "was_home": True, "value": 61,
        "selected_rank": 0.9,
    }])
    f = pm.build_features(history, upcoming)
    up = f[f["is_upcoming"]].iloc[0]
    assert up["total_points_r3"] == pytest.approx((4 + 12 + 8) / 3)
    assert up["fixtures_in_gw"] == 1 and up["was_home"] == 1.0 and up["price"] == pytest.approx(6.1)
    assert up["selected_rank"] == pytest.approx(0.9)
    assert np.isnan(up["total_points"])
    assert up["team_gf"] == pytest.approx((2 + 1 + 3 + 1) / 4)  # all of Arsenal's matches so far


def test_last_match_features_see_only_the_previous_gameweek(history):
    f = pm.build_features(history).set_index(["name", "fixture"])
    assert np.isnan(f.loc[("Alice", 1), "total_points_r1"])
    assert f.loc[("Alice", 2), "total_points_r1"] == 2
    assert f.loc[("Alice", 2), "minutes_r1"] == 90 and f.loc[("Alice", 2), "started_r1"] == 1
    assert f.loc[("Alice", 3), "total_points_r1"] == 4
    # Double gameweek: the second fixture still sees GW2, not the first fixture's 12 points
    assert f.loc[("Alice", 4), "total_points_r1"] == 4


def test_upcoming_rows_get_last_match_and_20_match_team_form(history):
    upcoming = pd.DataFrame([{
        "season": "2026-27", "name": "Alice", "position": "MID", "team": "Arsenal", "opponent": "Chelsea",
        "fixture": 99, "kickoff_time": "2026-09-15T14:00:00Z", "GW": 4, "was_home": True, "value": 61,
    }])
    up = pm.build_features(history, upcoming).query("is_upcoming").iloc[0]
    assert up["total_points_r1"] == 8 and up["minutes_r1"] == 90   # GW3's second fixture
    assert up["team_gf20"] == pytest.approx((2 + 1 + 3 + 1) / 4)  # fewer than 20 matches: all of them
    assert up["opp_ga20"] == pytest.approx((2 + 1 + 3) / 3)       # Chelsea conceded 2, 1, 3


def test_team_strength_windows_of_6_and_20_matches():
    a, b = ("Ann", "Arsenal", "FWD"), ("Ben", "Brentford", "DEF")
    rows = []
    for gw in range(1, 26):
        goals = 0 if gw <= 14 else 3  # Arsenal: 14 blanks, then 3 a game
        home, away = ("Arsenal", "Brentford") if gw % 2 else ("Brentford", "Arsenal")
        hg, ag = (goals, 1) if home == "Arsenal" else (1, goals)
        kickoff = (pd.Timestamp("2026-08-01T14:00:00Z") + pd.Timedelta(days=7 * gw)).isoformat()
        rows += _match_rows("2026-27", gw, gw, kickoff, home, away, hg, ag, {a: 2, b: 2})
    df = pd.DataFrame(rows)
    for col in pm.HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    f = pm.build_features(df[pm.HISTORY_COLUMNS]).set_index(["name", "fixture"])
    # Before GW25: the last 6 matches (19-24) all had 3 goals; the last 20 (5-24) had 10 x 0 and 10 x 3
    assert f.loc[("Ann", 25), "team_gf"] == pytest.approx(3.0)
    assert f.loc[("Ann", 25), "team_gf20"] == pytest.approx(1.5)
    assert f.loc[("Ben", 25), "opp_gf20"] == pytest.approx(1.5)  # the same numbers from Brentford's side


def test_evaluate_sums_double_gameweeks_per_player():
    df = pd.DataFrame({"season": "s", "GW": 1, "name": [f"p{i}" for i in range(40)] + ["p0"],
                       "pred": list(range(40)) + [100], "total_points": list(range(40)) + [5]})
    m = pm.evaluate(df, "pred")
    assert m.captain_points == 5  # p0's two fixtures summed: pred 100 -> top pick, actual 0 + 5
