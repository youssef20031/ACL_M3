"""Winner's-curse backtest variants (scripts/analysis/ml_backtest.py): refit_noscale and the tail
shrink of shrink_tail / shrink_tail_pos. The models are replaced by a stub, so nothing is trained."""
import numpy as np
import pandas as pd
import pytest

from scripts.analysis import ml_backtest as bt

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
POSITIONS = ["GK", "DEF", "MID", "FWD"]
BEST = 10            # the stub's early-stopped tree count


def _feats(seed=0, rows=400) -> pd.DataFrame:
    """Synthetic backtest rows whose `x` is the stub model's prediction and whose actuals are
    over-predicted in the upper tail (slope 0.4 above the 90th percentile)."""
    rng = np.random.default_rng(seed)
    parts = []
    for s in SEASONS:
        x = rng.gamma(2.0, 1.5, rows)
        h = np.maximum(0.0, x - np.quantile(x, 0.9))
        parts.append(pd.DataFrame({"season": s, "GW": np.arange(rows) % 10 + 1,
                                   "position": np.array(POSITIONS)[np.arange(rows) % 4],
                                   "x": x, "total_points": x - 0.4 * h + rng.normal(0, 0.3, rows)}))
    return pd.concat(parts, ignore_index=True)


class StubRegressor:
    """Stands in for xgboost.XGBRegressor: records its tree count and predicts the column `x`."""
    made = []

    def __init__(self, n_estimators=None, **kwargs):
        self.n_estimators, self.best_iteration = n_estimators, BEST - 1
        StubRegressor.made.append(n_estimators)

    def fit(self, X, y, **kwargs):
        return self

    def evals_result(self):
        curve = np.r_[np.linspace(2.0, 1.0, BEST), np.linspace(1.01, 1.5, 20)]
        return {"validation_0": {"rmse": list(curve)}}

    def predict(self, X, **kwargs):
        return X["x"].to_numpy(dtype=float)

    def get_booster(self):
        return type("Booster", (), {"get_score": lambda self, importance_type=None: {}})()


@pytest.fixture
def stubbed(monkeypatch):
    """run_variant with the stub model, one seed, and score_unit replaced by a recorder."""
    import xgboost
    monkeypatch.setattr(xgboost, "XGBRegressor", StubRegressor)
    StubRegressor.made = []
    scored = {}

    def score_unit(rows, seed_preds, key, sink=None, reuse=None):
        scored[key] = {s: np.asarray(p, dtype=float).copy() for s, p in seed_preds.items()}
        return {"summary": {}, "seed_list": sorted(seed_preds)}, {s: {"top20": 0.0, "rmse": 0.0} for s in seed_preds}

    monkeypatch.setattr(bt, "score_unit", score_unit)
    monkeypatch.setattr(bt, "SEEDS", (0, 1))
    monkeypatch.setattr(bt, "TEST_SEASONS", ["2025-26", "2024-25"])   # out of order on purpose
    monkeypatch.setattr(bt, "INSEASON", ("2025-26", "2024-25"))
    return scored


def _run(variant, feats=None):
    spec = {**bt.VARIANTS[variant], "features": ["x"]}
    return bt.run_variant(_feats() if feats is None else feats, variant, spec)


# ----------------------------------------------------------------------------- the shrink formula

def test_tail_shrink_leaves_rows_below_kappa_unchanged():
    params = {"all": {"kappa": 5.0, "lambda": 0.5}}
    out = bt.tail_shrink([1.0, 5.0, 7.0, 9.0], ["all"] * 4, params)
    assert out.tolist() == [1.0, 5.0, 6.0, 7.0]


def test_tail_params_is_the_clipped_least_squares_slope():
    rng = np.random.default_rng(1)
    pred = rng.gamma(2.0, 1.5, 2000)
    kappa = np.quantile(pred, 0.9)
    h = np.maximum(0.0, pred - kappa)
    for slope, want in ((0.3, 0.3), (0.9, bt.TAIL_MAX), (-0.5, 0.0)):
        q = bt.tail_params(pred, pred - slope * h, ["all"] * len(pred))["all"]
        assert q["kappa"] == pytest.approx(kappa)
        assert q["lambda"] == pytest.approx(want)
    # with noise: the slope through zero of (pred - y) on h
    y = pred - 0.2 * h + rng.normal(0, 0.5, len(pred))
    slope = np.linalg.lstsq(h[:, None], pred - y, rcond=None)[0][0]
    assert bt.tail_params(pred, y, ["all"] * len(pred))["all"]["lambda"] == pytest.approx(np.clip(slope, 0, 0.6))


def test_tail_params_per_position_are_independent():
    rng = np.random.default_rng(2)
    pred = rng.gamma(2.0, 1.5, 4000)
    pos = np.array(POSITIONS)[np.arange(len(pred)) % 4]
    slopes = {"GK": 0.1, "DEF": 0.2, "MID": 0.3, "FWD": 0.5}
    y = pred.copy()
    for p, s in slopes.items():
        m = pos == p
        y[m] -= s * np.maximum(0.0, pred[m] - np.quantile(pred[m], 0.9))
    params = bt.tail_params(pred, y, pos)
    for p, s in slopes.items():
        assert params[p]["kappa"] == pytest.approx(np.quantile(pred[pos == p], 0.9))
        assert params[p]["lambda"] == pytest.approx(s)
    y2 = y.copy()
    y2[pos == "GK"] = 0.0                                      # change only the goalkeepers
    params2 = bt.tail_params(pred, y2, pos)
    assert all(params2[p] == params[p] for p in ("DEF", "MID", "FWD"))
    shrunk = bt.tail_shrink(pred, pos, {"FWD": params["FWD"]})   # other positions untouched
    assert np.array_equal(shrunk[pos != "FWD"], pred[pos != "FWD"])


# ----------------------------------------------------------------------------- the fold history

def test_first_fold_is_uncorrected_and_later_folds_use_only_earlier_ones():
    f = _feats()
    hist = bt.TailHistory("all")
    first = f[f["season"] == "2024-25"]
    out, info = hist.correct(("refit", 0), first, first["x"], {})
    assert np.array_equal(out, first["x"].to_numpy()) and info == {"fit_rows": 0, "groups": {}}
    # another protocol or seed has its own history
    assert hist.correct(("es", 0), first, first["x"], {})[1]["fit_rows"] == 0
    assert hist.correct(("refit", 1), first, first["x"], {})[1]["fit_rows"] == 0
    second = f[f["season"] == "2025-26"]
    out2, info2 = hist.correct(("refit", 0), second, second["x"], {})
    assert info2["fit_rows"] == len(first)
    assert info2["groups"]["all"] == bt.tail_params(first["x"], first["total_points"], ["all"] * len(first))["all"]
    assert 0.2 < info2["groups"]["all"]["lambda"] < 0.6 and not np.array_equal(out2, second["x"].to_numpy())


def test_correction_never_uses_the_current_folds_actuals():
    f = _feats()
    first, second = f[f["season"] == "2024-25"], f[f["season"] == "2025-26"]
    scrambled = second.assign(total_points=np.random.default_rng(3).permutation(second["total_points"].to_numpy()) * 5)
    outs = []
    for cur in (second, scrambled):
        hist = bt.TailHistory("position")
        hist.correct(("inseason", 0), first, first["x"], {})
        outs.append(hist.correct(("inseason", 0), cur, cur["x"], {}))
    assert np.array_equal(outs[0][0], outs[1][0]) and outs[0][1] == outs[1][1]
    assert set(outs[0][1]["groups"]) == set(POSITIONS)


# ----------------------------------------------------------------------------- run_variant / run_inseason

def test_shrink_variants_run_folds_in_order_and_log_their_parameters(stubbed):
    f = _feats()
    for variant in ("shrink_tail", "shrink_tail_pos"):
        res = _run(variant)
        # 2024-25 ran first and only started the history (unscored); 2025-26 is fitted on it
        assert list(res["folds"]) == ["2025-26"] and list(res["inseason"]) == ["2025-26"]
        assert list(res["tail_inseason"]) == ["2025-26"]
        for proto in ("es", "refit"):
            later = res["folds"]["2025-26"]["seeds"][0]["tail"][proto]
            assert later["fit_rows"] == 400
            assert set(later["groups"]) == ({"all"} if variant == "shrink_tail" else set(POSITIONS))
        assert res["tail_inseason"]["2025-26"][1]["fit_rows"] == 400
        for proto in ("es", "refit", "inseason"):
            assert (variant, proto, "2024-25") not in stubbed
            later = stubbed[(variant, proto, "2025-26")][0]
            assert later.max() < f.loc[f["season"] == "2025-26", "x"].max()


def test_es_test_curve_uses_the_es_shrink(stubbed, monkeypatch):
    curves = []
    monkeypatch.setattr(bt, "CURVE_POINTS", (5,))
    monkeypatch.setattr(bt, "summarize", lambda t: {})
    monkeypatch.setattr(bt, "gw_table", lambda test, pred, squads=True: curves.append(np.array(pred)) or pd.DataFrame())
    _run("shrink_tail")
    f = _feats()
    assert np.array_equal(curves[0], f.loc[f["season"] == "2024-25", "x"].to_numpy())   # first fold: no shrink
    assert np.array_equal(curves[1], stubbed[("shrink_tail", "es", "2025-26")][0])     # the es protocol's shrink


def test_shrink_variant_ignores_the_current_folds_actuals(stubbed):
    f = _feats()
    _run("shrink_tail_pos", f)
    before = {k: v for k, v in stubbed.items() if k[2] == "2025-26"}
    g = f.copy()
    m = g["season"] == "2025-26"
    g.loc[m, "total_points"] = np.random.default_rng(4).permutation(g.loc[m, "total_points"].to_numpy()) * 3
    stubbed.clear()
    _run("shrink_tail_pos", g)
    for k, v in before.items():
        assert all(np.array_equal(v[s], stubbed[k][s]) for s in v)


def test_current_is_not_shrunk(stubbed):
    res = _run("current")
    assert "tail_inseason" not in res and "tail" not in res["folds"]["2025-26"]["seeds"][0]
    f = _feats()
    raw = f.loc[f["season"] == "2025-26", "x"].to_numpy()
    assert np.array_equal(stubbed[("current", "refit", "2025-26")][0], raw)


def test_refit_noscale_keeps_the_early_stopped_tree_count(stubbed):
    assert bt._refit_trees(100, 3000, 1000, {}) == 300
    assert bt._refit_trees(3000, 3000, 1000, {}) == 4000
    assert bt._refit_trees(100, 3000, 1000, bt.VARIANTS["refit_noscale"]) == 100
    res = _run("refit_noscale")
    for s in ("2024-25", "2025-26"):
        info = res["folds"][s]["seeds"][0]
        assert info["best_iter"] == BEST and info["n_refit"] == BEST
    refits = [n for n in StubRegressor.made if n != bt.NMAX]          # es fits use NMAX trees
    assert refits and all(n == BEST for n in refits)                  # refit and every in-season block
    StubRegressor.made = []
    res = _run("current")
    assert res["folds"]["2024-25"]["seeds"][0]["n_refit"] == 2 * BEST  # train 1 season, refit on 2
    assert max(n for n in StubRegressor.made if n != bt.NMAX) > 2 * BEST


def test_tail_parameters_survive_json_and_reach_the_report(stubbed):
    import json
    res = json.loads(json.dumps(_run("shrink_tail_pos")))
    rows = bt._tail_lines({"shrink_tail_pos": res})
    assert len(rows) == 3 and all("| 2025-26 |" in r for r in rows)   # es, refit, inseason; 2024-25 unscored
    assert all("| 2 | 400 |" in r and all(p in r for p in POSITIONS) for r in rows)
