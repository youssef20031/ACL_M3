"""Leakage guarantees of the backtest's opponent-by-position (oppos) and head-to-head (h2h)
candidate features in scripts/analysis/ml_backtest.py (synthetic data, no files)."""
import numpy as np
import pandas as pd
import pytest

from ml import points_model as pm
from scripts.analysis import ml_backtest as bt

K = bt.H2H_PRIOR
ALICE, CARA = ("Alice", "Arsenal", "MID"), ("Cara", "Arsenal", "DEF")
BOB, LEE = ("Bob", "Chelsea", "FWD"), ("Lee", "Leeds", "FWD")
# (GW, fixture, kickoff, home, away, {player: (points, minutes, xGI)})
MATCHES = [
    (1, 1, "2026-08-22T14:00:00Z", "Arsenal", "Chelsea", {ALICE: (2, 90, 0.2), CARA: (6, 90, 0.1), BOB: (1, 90, 0.3)}),
    (2, 2, "2026-08-29T14:00:00Z", "Chelsea", "Arsenal", {ALICE: (4, 90, 0.4), CARA: (2, 90, 0.0), BOB: (5, 90, 0.6)}),
    # GW3: Arsenal double gameweek; Leeds (promoted) play their first match in its second fixture
    (3, 3, "2026-09-05T14:00:00Z", "Arsenal", "Chelsea", {ALICE: (12, 90, 1.5), CARA: (1, 90, 0.1), BOB: (2, 90, 0.5)}),
    (3, 4, "2026-09-08T19:00:00Z", "Arsenal", "Leeds", {ALICE: (8, 90, 0.9), CARA: (6, 90, 0.0), LEE: (2, 90, 0.2)}),
    (4, 5, "2026-09-12T14:00:00Z", "Chelsea", "Arsenal", {ALICE: (3, 90, 0.3), CARA: (2, 90, 0.0), BOB: (0, 0, 0.0)}),
    (5, 6, "2026-09-19T14:00:00Z", "Leeds", "Arsenal", {ALICE: (5, 90, 0.5), CARA: (2, 90, 0.0), LEE: (3, 90, 0.4)}),
]


def _history(matches=MATCHES) -> pd.DataFrame:
    rows = []
    for gw, fixture, kickoff, home, away, players in matches:
        for (name, team, pos), (pts, mins, xgi) in players.items():
            rows.append({
                "season": "2026-27", "name": name, "position": pos, "team": team, "fixture": fixture,
                "kickoff_time": kickoff, "GW": gw, "was_home": team == home, "team_h_score": 1,
                "team_a_score": 0, "total_points": pts, "minutes": mins, "starts": float(mins >= 60),
                "expected_goal_involvements": xgi, "value": 60, "selected": 1000,
            })
    df = pd.DataFrame(rows)
    for col in pm.HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    return df[pm.HISTORY_COLUMNS]


def _features(history: pd.DataFrame) -> pd.DataFrame:
    df = pm.build_features(history)
    oppos, h2h = bt._opp_position_concession(df), bt._head_to_head(df)
    assert oppos.index.equals(df.index) and h2h.index.equals(df.index)  # one row per input row
    return pd.concat([df, oppos, h2h], axis=1).set_index(["name", "fixture"])


def test_opponent_concession_uses_only_earlier_gameweeks():
    f = _features(_history())
    assert np.isnan(f.loc[("Alice", 1), "opp_pos_pts"])                # Chelsea have no match yet
    assert f.loc[("Alice", 2), "opp_pos_pts"] == pytest.approx(2)
    # Midfielders scored 2 and 4 against Chelsea before GW3: not Alice's 12 in this fixture
    assert f.loc[("Alice", 3), "opp_pos_pts"] == pytest.approx(3)
    assert f.loc[("Alice", 3), "opp_pos_xgi"] == pytest.approx(0.3)
    assert f.loc[("Alice", 5), "opp_pos_pts"] == pytest.approx((2 + 4 + 12) / 3)
    assert f.loc[("Cara", 5), "opp_pos_pts"] == pytest.approx((6 + 2 + 1) / 3)  # defenders, separately
    # Arsenal's double gameweek from the opponents' side: both GW3 fixtures see what Arsenal
    # conceded to forwards before GW3 (Bob's 1 and 5), not Bob's 2 in fixture 3
    assert f.loc[("Bob", 3), "opp_pos_pts"] == pytest.approx(3)
    assert f.loc[("Lee", 4), "opp_pos_pts"] == pytest.approx(3)
    # Promoted club: NaN before its first match, then only its own matches
    assert np.isnan(f.loc[("Alice", 4), "opp_pos_pts"]) and np.isnan(f.loc[("Alice", 4), "opp_pos_xgi20"])
    assert f.loc[("Alice", 6), "opp_pos_pts"] == pytest.approx(8)
    # Bob's 0 minutes in GW4 isn't an appearance: forwards scored 1, 5, 2, 2 in Arsenal's other matches
    assert f.loc[("Lee", 6), "opp_pos_pts"] == pytest.approx(10 / 4)


def test_opponent_concession_windows_of_6_and_20_matches():
    ann, ben = ("Ann", "Arsenal", "FWD"), ("Ben", "Brentford", "DEF")
    matches = []
    for gw in range(1, 26):
        home, away = ("Arsenal", "Brentford") if gw % 2 else ("Brentford", "Arsenal")
        kickoff = (pd.Timestamp("2026-08-01T14:00:00Z") + pd.Timedelta(days=7 * gw)).isoformat()
        matches.append((gw, gw, kickoff, home, away, {ann: (0 if gw <= 14 else 6, 90, 0.0), ben: (2, 90, 0.0)}))
    f = _features(_history(matches))
    # Before GW25 Ann scored 6 against Brentford in each of the last 6 matches, 0 in 10 of the last 20
    assert f.loc[("Ann", 25), "opp_pos_pts"] == pytest.approx(6)
    assert f.loc[("Ann", 25), "opp_pos_pts20"] == pytest.approx(3)


def test_head_to_head_uses_only_earlier_meetings():
    f = _features(_history())
    # Alice's level (points per appearance before the gameweek): GW2 2, GW3 3 (both fixtures), GW4 6.5.
    # Her residuals against Chelsea: GW1 none (no level yet), GW2 4 - 2 = 2, GW3 12 - 3 = 9
    assert f.loc[("Alice", 1), "h2h_meetings"] == 0 and f.loc[("Alice", 1), "h2h_resid"] == 0
    assert f.loc[("Alice", 2), "h2h_meetings"] == 0
    assert f.loc[("Alice", 3), "h2h_meetings"] == 1
    assert f.loc[("Alice", 3), "h2h_resid"] == pytest.approx(2 / (1 + K))  # not this fixture's +9
    assert f.loc[("Alice", 5), "h2h_meetings"] == 2
    assert f.loc[("Alice", 5), "h2h_resid"] == pytest.approx((2 + 9) / (2 + K))
    # Against Leeds: GW3's second fixture is measured against the level before GW3 (3, not 6)
    assert f.loc[("Alice", 4), "h2h_meetings"] == 0
    assert f.loc[("Alice", 6), "h2h_resid"] == pytest.approx((8 - 3) / (1 + K))
    # A player who doesn't play gets the same pre-match value (Bob: 0 minutes in fixture 5)
    assert f.loc[("Bob", 5), "h2h_meetings"] == 2
    assert f.loc[("Bob", 5), "h2h_resid"] == pytest.approx(((5 - 1) + (2 - 3)) / (2 + K))


def test_neither_feature_uses_the_gameweek_it_predicts():
    """Changing every outcome of GW3 (Arsenal's double gameweek) leaves both features of its two
    fixtures unchanged, while GW4 sees the change."""
    cols = bt.G_OPPOS + bt.G_H2H
    base = _features(_history())
    changed = _history()
    gw3 = changed["GW"] == 3
    changed.loc[gw3, "total_points"] += 10
    changed.loc[gw3, "expected_goal_involvements"] *= 4
    changed.loc[gw3 & (changed["name"] == "Bob"), ["minutes", "total_points"]] = 0
    after = _features(changed)
    pd.testing.assert_frame_equal(base.loc[base["GW"] == 3, cols], after.loc[after["GW"] == 3, cols])
    assert after.loc[("Alice", 5), "opp_pos_pts"] == pytest.approx((2 + 4 + 22) / 3)
    assert after.loc[("Alice", 5), "h2h_resid"] == pytest.approx((2 + 19) / (2 + K))
