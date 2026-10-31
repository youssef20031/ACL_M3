"""Expected points for scoring backtest picks (scripts/analysis/xpts.py): FPL's rules, the luck
it removes, and the per-season, per-position calibration."""
import math

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import xpts as xp


def match(position="MID", minutes=90, goals=0, assists=0, xg=0.0, xa=0.0, xgc=1.0, conceded=1, saves=0,
          bonus=0, dc=np.nan, cards=0, season="2024-25"):
    """One player-match with total_points by FPL's rules (cards as the only extra)."""
    cs = int(minutes >= 60 and conceded == 0)
    points = (2 if minutes >= 60 else 1 if minutes > 0 else 0) + goals * xp.GOAL_POINTS[position] + 3 * assists
    points += cs * xp.CS_POINTS[position] + bonus - cards
    if position in ("GK", "DEF"):
        points -= conceded // 2
    if position == "GK":
        points += saves // 3
    if position != "GK" and not np.isnan(dc) and dc >= xp.DC_THRESHOLD[position]:
        points += xp.DC_POINTS
    return {"season": season, "position": position, "minutes": minutes, "goals_scored": goals, "assists": assists,
            "clean_sheets": cs, "goals_conceded": conceded, "saves": saves, "bonus": bonus,
            "defensive_contribution": dc, "expected_goals": xg, "expected_assists": xa,
            "expected_goals_conceded": xgc, "total_points": points}


def test_conceded_penalty_is_the_poisson_expectation():
    for lam in (0.0, 0.4, 1.5, 3.2):
        direct = sum((k // 2) * math.exp(-lam) * lam ** k / math.factorial(k) for k in range(40))
        assert xp.conceded_penalty([lam])[0] == pytest.approx(direct, abs=1e-9)


def test_actual_terms_reproduce_fpl_points():
    df = pd.DataFrame([match("DEF", goals=1, conceded=3, bonus=2, dc=11), match("GK", saves=7, conceded=0),
                       match("FWD", minutes=30, goals=2, assists=1, cards=1)])
    _, actual = xp.terms(df)
    assert actual["rest"].tolist() == [0, 0, -1]        # only the yellow card is left over
    assert actual.loc[0, "defensive"] == 2 and actual.loc[1, "saves"] == 2


def test_dc_points_need_the_position_threshold():
    pos = pd.Series(["DEF", "DEF", "MID", "MID", "GK", "FWD"])
    dc = pd.Series([10, 9, 12, 11, 30, np.nan])
    assert xp.dc_points(pos, dc).tolist() == [2, 0, 2, 0, 0, 0]


def test_finishing_luck_is_removed():
    df = pd.DataFrame([match("FWD", goals=1, xg=0.5, bonus=3), match("FWD", goals=0, xg=0.5),
                       match("FWD", goals=0, xg=0.1)])
    x = xp.expected_points(df)
    assert x[0] == pytest.approx(x[1])                   # same chances, one scored
    assert x[0] > x[2]


def test_who_did_not_play_scores_zero_and_missing_stats_are_nan():
    df = pd.DataFrame([match(minutes=0, conceded=0), match(xg=np.nan, xa=np.nan, xgc=np.nan, season="2022-23"),
                       match(xg=0.3)])
    x = xp.expected_points(df)
    assert x[0] == 0 and np.isnan(x[1]) and np.isfinite(x[2])


def test_each_season_and_position_keeps_its_actual_level():
    rng = np.random.default_rng(0)
    rows = []
    for season in ("2023-24", "2024-25"):
        for pos in ("GK", "DEF", "MID", "FWD"):
            for _ in range(80):                          # under BONUS_MIN_ROWS: the group's mean bonus
                xg, xa, xgc = rng.exponential(0.2), rng.exponential(0.15), rng.exponential(1.3)
                rows.append(match(pos, minutes=int(rng.choice([20, 75, 90])), goals=int(rng.poisson(xg)),
                                  assists=int(rng.poisson(2 * xa)), xg=xg, xa=xa, xgc=xgc,
                                  conceded=int(rng.poisson(xgc)), saves=int(rng.integers(0, 6)),
                                  bonus=int(rng.choice([0, 0, 0, 1, 3])), cards=int(rng.random() < 0.1),
                                  season=season))
    df = pd.DataFrame(rows)
    df["xpts"] = xp.expected_points(df)
    level = df.groupby(["season", "position"])[["total_points", "xpts"]].mean()
    assert level["xpts"].to_numpy() == pytest.approx(level["total_points"].to_numpy())
    # assists are scaled up to FPL's count (twice xA here), not left at 3 x xA
    expected, actual = xp.terms(df)
    assert actual["assists"].sum() > 1.5 * expected["assists"].sum()


def test_bonus_is_fitted_and_clipped():
    rng = np.random.default_rng(1)
    xg = rng.uniform(0, 1, xp.BONUS_MIN_ROWS + 50)
    goals = (xg > 0.5).astype(int)
    # 30-minute forwards: the appearance point and goals are the only other terms
    df = pd.DataFrame([match("FWD", minutes=30, xg=g, goals=k, bonus=3 * k) for g, k in zip(xg, goals)])
    x = xp.expected_points(df)
    bonus = x - 1 - 4 * xg * goals.sum() / xg.sum()
    assert bonus.min() == pytest.approx(0, abs=1e-9) and bonus.max() == pytest.approx(3, abs=1e-9)   # clipped
    assert bonus[xg > 0.9].mean() > bonus[xg < 0.1].mean() + 2
