"""The fitted chance-of-playing multiplier of the backtest (scripts/analysis/ml_backtest.py, post hook
cop_fit, variants cop_fit and v4_fit). The models are replaced by a stub, so nothing is trained."""
import json

import numpy as np
import pandas as pd
import pytest

from ml import snapshots as sn
from scripts.analysis import ml_backtest as bt

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
# What each flag's players really score, as a share of the model's prediction
TRUE = {"d75": 0.6, "d50": 0.25, "d25": 0.1, "out": 0.0}
LITERAL = {"none": 1.0, "a": 1.0, **bt.COP_FLAGS}


def _feats(rows=2000, seed=0, scale=1.0) -> pd.DataFrame:
    """Rows whose `x` is the stub's prediction; unflagged players score `scale` x, flagged ones TRUE."""
    rng = np.random.default_rng(seed)
    flags = np.array(["none", "a", "a", "a", "a", "a", "d75", "d50", "d25", "out"])[np.arange(rows) % 10]
    parts = []
    for s in SEASONS:
        x = rng.gamma(2.0, 1.5, rows)
        share = np.array([TRUE.get(f, scale) for f in flags])
        parts.append(pd.DataFrame({
            "season": s, "GW": np.arange(rows) % 10 + 1, "position": "MID", "x": x, "total_points": x * share,
            sn.COP: [LITERAL[f] for f in flags], "snap_age_days": np.where(flags == "none", np.nan, 1.0)}))
    return pd.concat(parts, ignore_index=True)


def test_factors_recover_each_flags_share_relative_to_unflagged_players():
    for scale in (1.0, 1.3):        # a model that under-predicts everyone needs no flag correction
        f = _feats(scale=scale)
        fit = bt.cop_fit_factors(f, f["x"])
        for flag, share in TRUE.items():
            assert fit[flag]["raw"] == pytest.approx(share / scale)
            assert fit[flag]["factor"] == pytest.approx(share / scale, abs=0.01)   # 800 rows: the prior barely counts
            assert fit[flag]["rows"] == 800


def test_a_flag_with_few_rows_is_shrunk_towards_the_literal_factor():
    f = _feats()
    few = f[(f[sn.COP] != 0.5) | (np.arange(len(f)) < 30)]
    fit = bt.cop_fit_factors(few, few["x"])
    assert fit["d50"]["rows"] == 3
    assert 0.25 < fit["d50"]["factor"] < 0.5
    none = bt.cop_fit_factors(f[f[sn.COP] != 0.25], f.loc[f[sn.COP] != 0.25, "x"])
    assert none["d25"] == {"factor": 0.25, "raw": None, "rows": 0}


def test_negative_predictions_are_floored_like_served_ones():
    f = _feats()
    pred = f["x"].where(f[sn.COP] != 0.75, -1.0)
    assert bt.cop_fit_factors(f, pred)["d75"]["factor"] == bt.COP_FIT_MAX    # no predicted points: clipped


def test_multiplier_uses_the_fitted_factor_per_flag_and_1_without_a_snapshot():
    f = _feats(rows=10)
    fit = {flag: {"factor": k, "raw": k, "rows": 1} for flag, k in zip(bt.COP_FLAGS, (0.7, 0.3, 0.1, 0.02))}
    mult = bt.cop_fit_multiplier(f, fit)
    expected = {"none": 1.0, "a": 1.0, "d75": 0.7, "d50": 0.3, "d25": 0.1, "out": 0.02}
    flags = np.array(["none", "a", "a", "a", "a", "a", "d75", "d50", "d25", "out"])
    assert np.allclose(mult[:10], [expected[x] for x in flags])
    with pytest.raises(ValueError):
        bt._post(f, f["x"], bt.VARIANTS["v4_fit"])


# ----------------------------------------------------------------------------- run_variant / run_inseason

class StubRegressor:
    """Stands in for xgboost.XGBRegressor and predicts the column `x`."""

    def __init__(self, n_estimators=None, **kwargs):
        self.best_iteration = 9

    def fit(self, X, y, **kwargs):
        return self

    def evals_result(self):
        return {"validation_0": {"rmse": list(np.r_[np.linspace(2.0, 1.0, 10), np.linspace(1.01, 1.5, 20)])}}

    def predict(self, X, **kwargs):
        return X["x"].to_numpy(dtype=float)

    def get_booster(self):
        return type("Booster", (), {"get_score": lambda self, importance_type=None: {}})()


@pytest.fixture
def stubbed(monkeypatch):
    import xgboost
    monkeypatch.setattr(xgboost, "XGBRegressor", StubRegressor)
    scored = {}

    def score_unit(rows, seed_preds, key, sink=None, reuse=None):
        scored[key] = {s: np.asarray(p, dtype=float).copy() for s, p in seed_preds.items()}
        return {"summary": {}, "seed_list": sorted(seed_preds)}, {s: {"top20": 0.0, "rmse": 0.0} for s in seed_preds}

    monkeypatch.setattr(bt, "score_unit", score_unit)
    monkeypatch.setattr(bt, "SEEDS", (0,))
    monkeypatch.setattr(bt, "TEST_SEASONS", ["2024-25", "2025-26"])
    monkeypatch.setattr(bt, "INSEASON", ("2025-26",))
    return scored


def _run(variant, feats):
    return bt.run_variant(feats, variant, {**bt.VARIANTS[variant], "features": ["x"]})


def test_cop_fit_scales_every_protocol_by_factors_from_the_validation_season(stubbed):
    f = _feats()
    res = _run("v4_fit", f)
    assert list(res["folds"]) == ["2024-25", "2025-26"]      # no burn-in fold
    test = f[f["season"] == "2025-26"]
    fit = res["folds"]["2025-26"]["seeds"][0]["cop_fit"]
    assert fit["d50"]["raw"] == pytest.approx(0.25) and 0.25 < fit["d50"]["factor"] < 0.3   # 200 rows: a little prior
    expected = test["x"].to_numpy() * bt.cop_fit_multiplier(test, fit)
    for proto in ("es", "refit", "inseason"):
        assert np.allclose(stubbed[("v4_fit", proto, "2025-26")][0], expected)
    assert res["cop_fit_inseason"]["2025-26"][0] == fit


def test_cop_fit_never_uses_the_test_seasons_actuals(stubbed):
    f = _feats()
    _run("v4_fit", f)
    before = dict(stubbed)
    g = f.copy()
    m = g["season"] == "2025-26"
    g.loc[m, "total_points"] = 0.0
    stubbed.clear()
    _run("v4_fit", g)
    for key in (k for k in before if k[2] == "2025-26"):
        assert np.array_equal(before[key][0], stubbed[key][0])


def test_v4_raw_is_unscaled_and_v4_is_literal(stubbed):
    f = _feats()
    test = f[f["season"] == "2025-26"]
    _run("v4_raw", f)
    _run("v4", f)
    assert np.array_equal(stubbed[("v4_raw", "refit", "2025-26")][0], test["x"].to_numpy())
    assert np.array_equal(stubbed[("v4", "refit", "2025-26")][0], test["x"].to_numpy() * test[sn.COP].to_numpy())


def test_factors_survive_json_and_reach_the_report(stubbed):
    res = json.loads(json.dumps(_run("v4_fit", _feats())))
    rows = bt._cop_fit_lines({"v4_fit": res})
    assert len(rows) == 3          # es/refit for both folds, and the in-season replay
    assert all("(0.25) n=200" in r and "(0.60) n=200" in r for r in rows)
