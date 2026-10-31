"""Team ratings fitted to bookmaker prices (ml/odds.market_ratings, rating_features) and the
projections' odds-ahead fill (pipeline/predict_next_gw.fill_odds_ahead)."""
import numpy as np
import pandas as pd
import pytest

from ml import odds as od
from ml import points_model as pm
from pipeline import predict_next_gw as pnp

ATTACK = {"ARS": 0.30, "LIV": 0.25, "CHE": 0.05, "EVE": -0.20, "BUR": -0.35}
DEFENCE = {"ARS": 0.25, "LIV": 0.10, "CHE": 0.00, "EVE": -0.10, "BUR": -0.30}
BASE, HOME = 0.25, 0.15


def priced(home, away, base=BASE, attack=ATTACK, defence=DEFENCE):
    """Margin-free prices of a match whose goals are Poisson with the given ratings."""
    lam_h = np.exp(base + HOME + attack[home] - defence[away])
    lam_a = np.exp(base + attack[away] - defence[home])
    p_home, p_draw, p_away = (float(x[0]) for x in od._one_x_two(np.array([lam_h]), np.array([lam_a])))
    total = lam_h + lam_a
    p_over = 1 - np.exp(-total) * (1 + total + total ** 2 / 2)
    return {"home": home, "away": away, "p_home": p_home, "p_draw": p_draw, "p_away": p_away, "p_over": p_over}


def league(dates=("2026-09-01", "2026-10-01")):
    rows = [{**priced(h, a), "season": "2026-27", "date": pd.Timestamp(d).date()}
            for d in dates for h in ATTACK for a in ATTACK if h != a]
    return pd.DataFrame(rows)


def test_ratings_reproduce_the_market_expected_goals():
    r = od.market_ratings(league(), ridge=0.0)
    fixture = pd.DataFrame({"team": ["ARS", "BUR"], "opponent": ["BUR", "ARS"], "was_home": [1, 0]})
    f = od.rating_features(fixture, r)
    want_h = np.exp(BASE + HOME + ATTACK["ARS"] - DEFENCE["BUR"])
    want_a = np.exp(BASE + ATTACK["BUR"] - DEFENCE["ARS"])
    assert f.loc[0, "odds_gf"] == pytest.approx(want_h, rel=0.03)
    assert f.loc[0, "odds_ga"] == pytest.approx(want_a, rel=0.03)
    assert f.loc[1, "odds_gf"] == pytest.approx(f.loc[0, "odds_ga"])
    assert f.loc[1, "odds_cs"] == pytest.approx(np.exp(-want_h), rel=0.03)
    assert f.loc[0, "odds_win"] > 0.5 > f.loc[1, "odds_win"]
    assert f.loc[0, "odds_win"] + f.loc[0, "odds_draw"] + f.loc[1, "odds_win"] == pytest.approx(1.0)
    assert r["draw_scale"] == pytest.approx(1.0, abs=0.02)       # Poisson prices: the draw needs no scaling
    shrunk = od.rating_features(fixture, od.market_ratings(league()))   # the default ridge pulls toward average
    assert want_a < shrunk.loc[0, "odds_ga"] < shrunk.loc[0, "odds_gf"] < want_h


def test_recent_prices_count_more():
    old = league(("2026-08-01",))
    new = pd.DataFrame([{**priced(h, a, attack={**ATTACK, "EVE": 0.30}), "season": "2026-27",
                         "date": pd.Timestamp("2026-10-20").date()} for h in ATTACK for a in ATTACK if h != a])
    r = od.market_ratings(pd.concat([old, new], ignore_index=True), ridge=0.0)
    gf = od.rating_features(pd.DataFrame({"team": ["EVE"], "opponent": ["CHE"], "was_home": [1]}), r)["odds_gf"][0]
    before, after = (np.exp(BASE + HOME + a - DEFENCE["CHE"]) for a in (ATTACK["EVE"], 0.30))
    assert (before + after) / 2 < gf < after                   # closer to the recent strength


def test_no_prices_no_ratings():
    assert od.market_ratings(od.empty_odds()) is None


def test_fill_odds_ahead_fills_only_rows_without_prices():
    rows = pd.DataFrame({"team": ["ARS", "CHE"], "opponent": ["BUR", "LIV"], "was_home": [1, 0],
                         **{c: [np.nan, 0.4] for c in od.FEATURES}})
    extras = pm.ExtraInputs(codes=None, odds=league(), snapshots=None)
    assert pnp.fill_odds_ahead(rows, extras) == 1
    assert rows.loc[0, od.FEATURES].notna().all() and (rows.loc[1, od.FEATURES] == 0.4).all()
    assert pnp.fill_odds_ahead(rows.assign(**{c: np.nan for c in od.FEATURES}), None) == 0
    assert pnp.ODDS_AHEAD is False                               # off until the replay says otherwise
