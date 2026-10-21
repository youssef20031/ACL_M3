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


# --------------------------------------------------------------------------- other competitions

from ml import other_competitions as oc  # noqa: E402


@pytest.fixture
def other():
    """Arsenal's non-PL matches around the `history` fixture: a UCL tie between GW1 and GW2, an
    EFL Cup tie inside the GW3 double gameweek (after its deadline), a scheduled UCL tie and an
    FA Cup round without a date yet."""
    matches = pd.DataFrame([
        {"match_id": "m1", "season": "2026-27", "competition": "UCL", "kickoff_time": "2026-08-26T19:00:00Z",
         "team": "Arsenal", "opponent": "Inter", "status": "finished"},
        {"match_id": "m2", "season": "2026-27", "competition": "EFLC", "kickoff_time": "2026-09-06T19:00:00Z",
         "team": "Arsenal", "opponent": "Hull City", "status": "finished"},
        {"match_id": "m3", "season": "2026-27", "competition": "UCL", "kickoff_time": "2026-09-16T19:00:00Z",
         "team": "Arsenal", "opponent": "Ajax", "status": "scheduled"},
        {"match_id": "m4", "season": "2026-27", "competition": "FAC", "kickoff_time": None,
         "team": "Arsenal", "opponent": None, "status": "scheduled"},
    ])
    matches["kickoff_time"] = pd.to_datetime(matches["kickoff_time"], utc=True)
    apps = pd.DataFrame([
        {"match_id": "m1", "team": "Arsenal", "name": "Alice", "started": 1, "minutes": 90, "goals": 1, "assists": 0},
        {"match_id": "m1", "team": "Arsenal", "name": "Cara", "started": 0, "minutes": 0, "goals": 0, "assists": 0},
        {"match_id": "m2", "team": "Arsenal", "name": "Alice", "started": 0, "minutes": 0, "goals": 0, "assists": 0},
        {"match_id": "m2", "team": "Arsenal", "name": "Cara", "started": 1, "minutes": 90, "goals": 0, "assists": 2},
    ])
    return matches, apps


def _oc_features(history, other, upcoming=None):
    f = pm.build_features(history, upcoming)
    return f.join(oc.other_competition_features(f, *other))


def test_other_competition_outcomes_before_the_deadline_only(history, other):
    f = _oc_features(history, other).set_index(["name", "fixture"])
    assert np.isnan(f.loc[("Alice", 1), "oc_minutes_last"])         # no cup match yet
    assert f.loc[("Alice", 2), "oc_minutes_last"] == 90 and f.loc[("Alice", 2), "oc_started_last"] == 1
    assert f.loc[("Cara", 2), "oc_minutes_last"] == 0 and f.loc[("Cara", 2), "oc_squad_last"] == 1
    # GW2: the UCL tie (Wed) was the club's latest match; GW1 + UCL inside the 7 days before the deadline
    assert f.loc[("Alice", 2), "ac_minutes_last"] == 90
    assert f.loc[("Alice", 2), "ac_minutes_7d"] == 180
    # Double gameweek: the EFL Cup tie between fixtures 3 and 4 is after the deadline, so both
    # fixtures see the UCL tie (Cara on the bench), not Cara's 90 minutes in the cup
    for col in oc.PLAYER_FEATURES:
        a, b = f.loc[("Cara", 3), col], f.loc[("Cara", 4), col]
        assert (np.isnan(a) and np.isnan(b)) or a == b, col
    assert f.loc[("Cara", 4), "oc_minutes_last"] == 0
    assert f.loc[("Alice", 4), "ac_minutes_last"] == 90          # GW2, the last match before the deadline
    # Chelsea play no other competitions
    assert np.isnan(f.loc[("Bob", 3), "oc_minutes_last"]) and f.loc[("Bob", 3), "oc_prev_days"] == oc.CAP_DAYS


def test_other_competition_schedule_uses_dates_around_the_fixture(history, other):
    f = _oc_features(history, other).set_index(["name", "fixture"])
    assert f.loc[("Alice", 2), "oc_prev_days"] == pytest.approx(2 + 19 / 24)   # Wed 19:00 -> Sat 14:00
    assert f.loc[("Alice", 1), "oc_next_days"] == pytest.approx(4 + 5 / 24)
    assert f.loc[("Alice", 1), "oc_next_tier"] == oc.TIE_TIER["UCL"]
    # Fixture 4 (Tue) comes two days after the EFL Cup tie: its date is known before the deadline
    assert f.loc[("Alice", 4), "oc_prev_days"] == pytest.approx(2.0)
    assert f.loc[("Alice", 4), "ac_days_since"] == pytest.approx(2.0)
    assert f.loc[("Alice", 3), "ac_matches_14d"] == 3                    # GW1, UCL, GW2
    assert f.loc[("Alice", 3), "oc_euro_tier"] == 3 and f.loc[("Alice", 3), "oc_cups_alive"] == 2
    assert f.loc[("Bob", 3), "oc_euro_tier"] == 0 and f.loc[("Bob", 3), "sub_euro_club"] == 0


def test_other_competition_features_ignore_later_outcomes(history, other):
    """Rewriting every outcome from GW3's deadline on (cup and league) changes nothing up to GW3."""
    matches, apps = other
    before = _oc_features(history, other)
    cut = history.loc[history["GW"] == 3, "kickoff_time"].min() - pd.Timedelta(hours=oc.CUTOFF_HOURS)
    late_ids = set(matches.loc[matches["kickoff_time"] >= cut, "match_id"])
    apps2 = apps.copy()
    apps2.loc[apps2["match_id"].isin(late_ids), ["started", "minutes", "goals", "assists"]] = [1, 90, 3, 3]
    hist2 = history.copy()
    late = hist2["kickoff_time"] >= cut
    hist2.loc[late, ["minutes", "starts", "goals_scored", "assists"]] = [0, 0, 0, 0]
    after = _oc_features(hist2, (matches, apps2))
    upto = before["GW"] <= 3
    cols = oc.PLAYER_FEATURES + oc.STATUS_FEATURES + oc.SCHEDULE_FEATURES
    pd.testing.assert_frame_equal(before.loc[upto, cols], after.loc[upto, cols])


def test_other_competition_features_for_upcoming_rows(history, other):
    upcoming = pd.DataFrame([{
        "season": "2026-27", "name": n, "position": "MID", "team": "Arsenal", "opponent": "Chelsea",
        "fixture": 99, "kickoff_time": "2026-09-12T14:00:00Z", "GW": 4, "was_home": True, "value": 61,
    } for n in ("Alice", "Cara")])
    f = _oc_features(history, other, upcoming).query("is_upcoming").set_index("name")
    assert f.loc["Cara", "oc_minutes_last"] == 90 and f.loc["Alice", "oc_minutes_last"] == 0   # the EFL Cup tie
    assert f.loc["Cara", "oc_minutes_r3"] == pytest.approx(45)                               # mean(90, 0)
    assert f.loc["Alice", "oc_next_days"] == pytest.approx(4 + 5 / 24)                       # the scheduled UCL tie
    assert f.loc["Alice", "oc_euro_tier"] == 3
    assert f.loc["Alice", "oc_cups_alive"] == 1           # out of the EFL Cup, FA Cup round not scheduled yet
    assert f.loc["Cara", "ac_minutes_14d"] == 4 * 90   # GW2, GW3 x2, EFL Cup (the UCL tie was 18 days back)
    # All-competition form: 2 assists in 450 minutes (5 appearances; the UCL bench doesn't count)
    assert f.loc["Cara", "ac_ga_p90_r8"] == pytest.approx(2 / 450 * 90)


# --------------------------------------------------------------------------- absences (backtest candidate)

from scripts.analysis import ml_backtest as bt  # noqa: E402


def _player_run(minutes):
    """One Arsenal player, one fixture a week; points 2 when he plays, 0 when he doesn't."""
    rows = []
    for i, m in enumerate(minutes, start=1):
        kickoff = (pd.Timestamp("2026-08-01T14:00:00Z") + pd.Timedelta(days=7 * i)).isoformat()
        rows += _match_rows("2026-27", i, i, kickoff, "Arsenal", "Chelsea", 1, 0, {("Dan", "Arsenal", "MID"): 2})
        rows[-1].update(minutes=m, starts=float(m >= 60), total_points=2 if m else 0)
    df = pd.DataFrame(rows)
    for col in pm.HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    return pm.build_features(df[pm.HISTORY_COLUMNS])


def test_absence_form_without_absences_equals_the_model_form(history):
    f = pm.build_features(history)
    a = bt.absence_form(f, min_run=np.inf)
    for col in bt.FORM_A:
        pd.testing.assert_series_equal(a[f"{col}_a"], f[col], check_names=False)
    assert (a["absent_run"] == 0).all() and (a["since_return"] == bt.ABSENCE_CAP).all()


def test_absence_form_leaves_out_long_runs_once_they_are_long():
    # 3 starts, 4 matches out (injury), back for 2
    f = _player_run([90, 90, 90, 0, 0, 0, 0, 90, 90]).set_index("GW")
    a = bt.absence_form(f.reset_index()).set_index(f.index)
    assert a.loc[5, "minutes_r1_a"] == 0 and a.loc[5, "absent_run"] == 0   # one zero: could be rotation
    assert a.loc[6, "minutes_r3_a"] == pytest.approx(30)                   # 90, 0, 0
    assert a.loc[7, "absent_run"] == 3 and a.loc[7, "minutes_r3_a"] == 90  # the run is now an absence
    assert a.loc[8, "absent_run"] == 4 and a.loc[8, "total_points_r6_a"] == 2
    assert a.loc[9, "absent_run"] == 0 and a.loc[9, "since_return"] == 1
    assert a.loc[9, "minutes_r6_a"] == 90                                  # the absence doesn't drag his form
    assert f.loc[9, "minutes_r6"] == pytest.approx(30)                     # ... as it does in the model's


def test_absence_form_keeps_zeros_of_players_who_were_not_starting():
    a = bt.absence_form(_player_run([20, 0, 0, 0, 0, 0, 30]))
    assert (a["absent_run"] == 0).all()
    assert a["minutes_r3_a"].iloc[-1] == 0


def test_absence_form_ignores_later_matches():
    base = bt.absence_form(_player_run([90, 90, 90, 0, 0, 90, 90, 90]))
    later = bt.absence_form(_player_run([90, 90, 90, 0, 0, 0, 0, 0]))   # the same up to GW5's outcome
    pd.testing.assert_frame_equal(base.iloc[:6], later.iloc[:6])


# --------------------------------------------------------------------------- bookmaker odds (backtest candidate)

from ml import odds as od  # noqa: E402


def _fd_rows(rows):
    """football-data.co.uk columns from (date, home, away, avg 1X2, avg over/under, closing 1X2)."""
    out = []
    for date, home, away, x12, ou, x12c in rows:
        r = {"Div": "E0", "Date": date, "HomeTeam": home, "AwayTeam": away,
             "AvgH": x12[0], "AvgD": x12[1], "AvgA": x12[2], "Avg>2.5": ou[0], "Avg<2.5": ou[1]}
        if x12c:
            r.update({"AvgCH": x12c[0], "AvgCD": x12c[1], "AvgCA": x12c[2], "AvgC>2.5": ou[0], "AvgC<2.5": ou[1]})
        out.append(r)
    return pd.DataFrame(out)


@pytest.fixture
def odds_history():
    """Arsenal v Chelsea in GW1 (Sat), Chelsea v Arsenal in GW2, and a GW3 double gameweek for
    Arsenal: at Leeds on Saturday and Leeds at home on the Tuesday after the deadline."""
    a, b, c = ("Alice", "Arsenal", "MID"), ("Bob", "Chelsea", "FWD"), ("Lee", "Leeds", "DEF")
    rows = []
    rows += _match_rows("2026-27", 1, 1, "2026-08-22T14:00:00Z", "Arsenal", "Chelsea", 2, 0, {a: 2, b: 1})
    rows += _match_rows("2026-27", 2, 2, "2026-08-29T14:00:00Z", "Chelsea", "Arsenal", 1, 1, {a: 4, b: 5})
    rows += _match_rows("2026-27", 3, 3, "2026-09-05T14:00:00Z", "Leeds", "Arsenal", 1, 3, {a: 12, c: 2})
    rows += _match_rows("2026-27", 3, 4, "2026-09-08T19:00:00Z", "Arsenal", "Leeds", 1, 0, {a: 8, c: 1})
    df = pd.DataFrame(rows)
    for col in pm.HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    raw = _fd_rows([
        ("22/08/2026", "Arsenal", "Chelsea", (2.0, 3.5, 4.0), (1.9, 1.9), (1.8, 3.6, 4.5)),
        ("29/08/2026", "Chelsea", "Arsenal", (2.5, 3.4, 2.8), (2.0, 1.8), (2.5, 3.4, 2.8)),
        ("05/09/2026", "Leeds", "Arsenal", (4.2, 3.6, 1.9), (1.8, 2.0), (4.2, 3.6, 1.9)),
        ("08/09/2026", "Arsenal", "Leeds", (1.4, 4.8, 8.0), (1.6, 2.4), (1.3, 5.2, 10.0)),
    ])
    return df[pm.HISTORY_COLUMNS], od.parse_odds(raw, "2026-27")


def test_odds_poisson_fit_recovers_the_goal_rates():
    lam_h, lam_a = np.array([1.8, 0.6, 1.3, 3.0]), np.array([0.9, 2.4, 1.3, 0.4])
    ph, pa = od._pmf(lam_h), od._pmf(lam_a)
    grid = ph[:, :, None] * pa[:, None, :]                       # P(home = i, away = j)
    i, j = np.meshgrid(od._K, od._K, indexing="ij")
    p_home, p_away = (grid * (i > j)).sum((1, 2)), (grid * (i < j)).sum((1, 2))
    p_over = (grid * (i + j > 2.5)).sum((1, 2))
    got_h, got_a = od.fit_goals(p_home, p_away, p_over)
    assert got_h == pytest.approx(lam_h, rel=1e-4) and got_a == pytest.approx(lam_a, rel=1e-4)
    nan_h, _ = od.fit_goals([0.5], [0.3], [np.nan])
    assert np.isnan(nan_h[0])


def test_odds_parse_removes_the_margin_and_maps_club_names():
    raw = _fd_rows([("21/08/2026", "Man United", "Tottenham", (2.0, 3.5, 4.0), (1.9, 1.9), None),
                    ("22/08/2026", "Hull", "Coventry", (np.nan, np.nan, np.nan), (np.nan, np.nan), None)])
    raw.loc[1, ["B365H", "B365D", "B365A", "B365>2.5", "B365<2.5"]] = [2.5, 3.2, 3.0, 2.1, 1.75]
    o = od.parse_odds(raw, "2026-27").set_index(["home", "away"])
    first = o.loc[("Man Utd", "Spurs")]
    margin = 1 / 2.0 + 1 / 3.5 + 1 / 4.0
    assert first["p_home"] == pytest.approx(0.5 / margin) and first["p_over"] == pytest.approx(0.5)
    assert first[["p_home", "p_draw", "p_away"]].sum() == pytest.approx(1.0)
    assert np.isnan(first["p_home_c"])                       # no closing prices in this file
    # Bet365 fills in where the market average is missing
    assert o.loc[("Hull City", "Coventry City"), "p_home"] == pytest.approx(0.4 / (0.4 + 1 / 3.2 + 1 / 3.0))


def test_odds_collection_times():
    kickoffs = pd.Series(pd.to_datetime([
        "2026-08-21T19:00:00Z",   # Fri 20:00 BST -> Fri 17:00 BST
        "2026-08-24T19:00:00Z",   # Mon -> the Friday before
        "2027-01-13T20:00:00Z",   # Wed (GMT) -> Tue 13:00
        "2026-12-26T12:30:00Z",   # Sat -> Fri 25 Dec 17:00
    ], utc=True))
    got = od.collected_at(kickoffs)
    assert list(got) == list(pd.to_datetime(["2026-08-21T16:00:00Z", "2026-08-21T16:00:00Z",
                                             "2027-01-12T13:00:00Z", "2026-12-25T17:00:00Z"], utc=True))


def test_odds_features_from_each_teams_side_before_the_deadline(odds_history):
    history, table = odds_history
    f = pm.build_features(history)
    f = f.join(od.odds_features(f, table)).set_index(["name", "fixture"])
    o = table.set_index(["home", "away"])
    alice, bob = f.loc[("Alice", 1)], f.loc[("Bob", 1)]
    assert alice["odds_win"] == pytest.approx(o.loc[("Arsenal", "Chelsea"), "p_home"])
    assert bob["odds_win"] == pytest.approx(o.loc[("Arsenal", "Chelsea"), "p_away"])
    assert alice["odds_draw"] == bob["odds_draw"]
    assert alice["odds_gf"] == bob["odds_ga"] and alice["odds_ga"] == bob["odds_gf"]
    assert alice["odds_cs"] == pytest.approx(np.exp(-alice["odds_ga"]))
    assert alice["odds_gf"] > alice["odds_ga"]                               # the favourite
    assert f.loc[("Alice", 2), "odds_win"] == pytest.approx(o.loc[("Chelsea", "Arsenal"), "p_away"])
    # GW3 double gameweek: Saturday's prices (collected Friday) are known at the deadline, the
    # Tuesday match's prices (collected that Tuesday) aren't; closing prices are an upper bound
    assert np.isfinite(f.loc[("Alice", 3), od.FEATURES]).all()
    assert f.loc[("Alice", 4), od.FEATURES].isna().all() and f.loc[("Lee", 4), od.FEATURES].isna().all()
    assert np.isfinite(f.loc[("Alice", 4), od.CLOSING_FEATURES]).all()
    assert f.loc[("Alice", 1), "odds_win_c"] > alice["odds_win"]             # the price shortened


def test_odds_features_ignore_results_and_later_prices(odds_history):
    history, table = odds_history
    before = pm.build_features(history)
    before = before.join(od.odds_features(before, table))
    hist2 = history.copy()
    hist2.loc[hist2["GW"] >= 2, ["team_h_score", "team_a_score", "total_points"]] = [0, 5, 0]
    table2 = table.copy()
    table2.loc[table2["date"] > pd.Timestamp("2026-08-23").date(), ["p_home", "p_away"]] = [0.1, 0.8]
    after = pm.build_features(hist2)
    after = after.join(od.odds_features(after, table2))
    gw1 = before["GW"] == 1
    pd.testing.assert_frame_equal(before.loc[gw1, od.FEATURES + od.CLOSING_FEATURES],
                                  after.loc[gw1, od.FEATURES + od.CLOSING_FEATURES])


def test_odds_features_for_upcoming_rows(odds_history):
    history, table = odds_history
    upcoming = pd.DataFrame([{
        "season": "2026-27", "name": n, "position": p, "team": t, "opponent": o, "fixture": 99,
        "kickoff_time": "2026-09-12T14:00:00Z", "GW": 4, "was_home": h, "value": 61,
    } for n, p, t, o, h in (("Lee", "DEF", "Leeds", "Chelsea", False), ("Bob", "FWD", "Chelsea", "Leeds", True))])
    fixtures = od.parse_odds(_fd_rows([("12/09/2026", "Chelsea", "Leeds", (1.7, 3.9, 4.8), (1.95, 1.85), None)]),
                             "2026-27")
    f = pm.build_features(history, upcoming)
    f = f.join(od.odds_features(f, pd.concat([table, fixtures], ignore_index=True))).query("is_upcoming")
    f = f.set_index("name")
    assert f.loc["Lee", "odds_win"] == pytest.approx(fixtures["p_away"].iloc[0])
    assert f.loc["Bob", "odds_win"] == pytest.approx(fixtures["p_home"].iloc[0])
    assert np.isnan(f.loc["Lee", "odds_win_c"])                              # no closing price yet
