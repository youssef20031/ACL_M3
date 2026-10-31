"""Squad scoring in the ML backtest (scripts/analysis/ml_backtest.py): FPL auto-subs and captaincy,
the optimiser's pool, the per-gameweek squad and position metrics, the two-level seed bootstrap and
scoring saved predictions from several runs. Small synthetic data only; no training."""
import argparse
import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import ml_backtest as bt
from services import team_optimizer as svc_opt

opt = bt.opt
C = opt.Candidate


def squad(captain="M1", vice="F1"):
    """A 4-4-2 XI with a bench of GK, DEF, MID, FWD (in that order)."""
    starters = [C("G1", "GK", "A", 4.0, 45)] + [C(f"D{i}", "DEF", "A", 4.0, 45) for i in range(1, 5)]
    starters += [C(f"M{i}", "MID", "B", 6.0 - i * 0.1, 60) for i in range(1, 5)]
    starters += [C(f"F{i}", "FWD", "C", 5.0, 70) for i in range(1, 3)]
    bench = [C("G2", "GK", "D", 1.0, 40), C("D5", "DEF", "D", 2.0, 40), C("M5", "MID", "E", 2.0, 45),
             C("F3", "FWD", "E", 2.0, 45)]
    by_id = {c.id: c for c in starters}
    return opt.Selection(starters=starters, bench=bench, captain=by_id[captain], vice_captain=by_id[vice])


def actual_for(sel, points=None, dnp=()):
    """(points, minutes) for every squad player: 2 points by default, 0 for the ones who didn't play."""
    points = points or {}
    return {c.id: ((0, 0) if c.id in dnp else (points.get(c.id, 2), 90)) for c in sel.starters + sel.bench}


# ----------------------------------------------------------------------------- auto-subs and captaincy

def test_auto_subs_live_in_team_optimizer():
    from services import model_review as mr
    assert mr.apply_auto_subs is svc_opt.apply_auto_subs
    xi = [{"name": n, "position": p, "played": n != "F1"} for n, p in
          [("G", "GK"), ("D1", "DEF"), ("D2", "DEF"), ("D3", "DEF"), ("M1", "MID"), ("M2", "MID"), ("M3", "MID"),
           ("M4", "MID"), ("M5", "MID"), ("F1", "FWD")]]
    bench = [{"name": "G2", "position": "GK", "played": True}, {"name": "D4", "position": "DEF", "played": True},
             {"name": "F2", "position": "FWD", "played": True}]
    final, _, subs = svc_opt.apply_auto_subs(xi, bench)
    assert subs == [("F1", "F2")]          # D4 would leave no forward
    assert "F2" in {p["name"] for p in final}


def test_everyone_plays():
    sel = squad()
    act = actual_for(sel, {"M1": 10})
    s = bt.score_squad(sel, act)
    assert s["sq_xi"] == 10 + 2 * 10
    assert s["sq_fpl"] == s["sq_xi"] + 10      # captain doubled
    assert s["sq_dnp"] == 0
    assert s["sq_pred"] == pytest.approx(sum(c.points for c in sel.starters) + 5.9)


def test_vice_takes_the_armband_when_the_captain_does_not_play():
    sel = squad()
    s = bt.score_squad(sel, actual_for(sel, {"F1": 7, "D5": 3}, dnp={"M1"}))
    # D5 (first outfield sub) replaces M1: 5-3-2 is valid
    assert s["sq_xi"] == 2 * 9 + 7 + 3
    assert s["sq_fpl"] == s["sq_xi"] + 7
    assert s["sq_dnp"] == 1


def test_no_armband_when_captain_and_vice_both_miss():
    sel = squad()
    s = bt.score_squad(sel, actual_for(sel, {"D5": 3, "M5": 4}, dnp={"M1", "F1"}))
    # D5 for M1, then M5 for F1 (one forward left is enough)
    assert s["sq_xi"] == 2 * 9 + 3 + 4
    assert s["sq_fpl"] == s["sq_xi"]
    assert s["sq_dnp"] == 2


def test_goalkeeper_only_for_goalkeeper_and_formation_respected():
    sel = squad()
    s = bt.score_squad(sel, actual_for(sel, {"G2": 6, "F3": 5, "D5": 1, "M5": 1}, dnp={"G1", "F1", "F2"}))
    # G2 for G1; D5 for F1 (F2 still counts as the forward); for F2, M5 would leave no forward: F3
    assert s["sq_dnp"] == 3
    assert s["sq_xi"] == 6 + 2 * 8 + 1 + 5


# ----------------------------------------------------------------------------- the pool

def test_squad_pool_keeps_the_best_and_the_cheapest(monkeypatch):
    monkeypatch.setattr(bt, "SQ_TOP_PER_POS", 3)
    monkeypatch.setattr(bt, "SQ_CHEAPEST", 2)
    g = pd.DataFrame({"name": [f"M{i}" for i in range(10)] + ["Neg", "NoPrice"],
                      "position": ["MID"] * 12, "team": ["T"] * 12,
                      "price": np.array([10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 4.6, 4.5, 6.0, 6.5, 4.0, np.nan], dtype="float32"),
                      "pred_sq": [9, 8, 7, 6, 5, 4, 1, 0.5, 2, 3, -1, 9]})
    pool = {c.id: c for c in bt.squad_pool(g)}
    assert set(pool) == {"M0", "M1", "M2", "M6", "M7"}   # top 3 + the 2 cheapest with points > 0 and a price
    assert pool["M6"].cost == 46 and pool["M0"].cost == 100


# ----------------------------------------------------------------------------- per-gameweek table

def league(gws=(1, 2), teams=10, seed=0):
    """Ten clubs, 2 GK / 6 DEF / 6 MID / 4 FWD each; 1 in 10 players doesn't play."""
    rng = np.random.default_rng(seed)
    rows = []
    for gw in gws:
        for t in range(teams):
            for pos, n in (("GK", 2), ("DEF", 6), ("MID", 6), ("FWD", 4)):
                for k in range(n):
                    plays = rng.random() > 0.1
                    rows.append({"season": "2025-26", "GW": gw, "fixture": gw * 100 + t // 2, "team": f"T{t}",
                                 "name": f"T{t} {pos}{k}", "position": pos, "price": 4.0 + 0.5 * k,
                                 "total_points": float(rng.integers(1, 12)) if plays else 0.0,
                                 "minutes": 90.0 if plays else 0.0, "clean_sheets": 0.0})
    df = pd.DataFrame(rows)
    df["price"] = df["price"].astype("float32")
    return df


def test_perfect_predictions_score_what_they_predict():
    rows = league()
    t = bt.gw_table(rows, rows["total_points"].to_numpy())
    assert len(t) == 2
    for _, r in t.iterrows():
        assert r["sq_dnp"] == 0                 # players who don't play are predicted 0: not in the pool
        assert r["sq_fpl"] == pytest.approx(r["sq_pred"])
        assert r["capt"] == r["pred_capt"] and r["top20"] == pytest.approx(r["pred20"])
        assert r["sq_xi"] < r["sq_fpl"]
        for p in bt.POS_TOPK:
            assert r[f"top_{p}"] == pytest.approx(r[f"pred_{p}"])
    s = bt.summarize(t)
    assert s["bias_DEF"] == 0 and s["sq_failed"] == 0 and s["sq_solves"] == 2


def test_position_top_k_and_bias():
    rows = league(gws=(1,))
    pred = rows["total_points"].to_numpy() + np.where(rows["position"] == "DEF", 0.5, 0.0)
    t = bt.gw_table(rows, pred, squads=False)
    defs = rows[rows["position"] == "DEF"].assign(pred=pred[rows["position"] == "DEF"])
    top = defs.sort_values("pred", ascending=False, kind="mergesort").head(10)
    assert t.loc[0, "top_DEF"] == pytest.approx(top["total_points"].mean())
    assert bt.summarize(t)["bias_DEF"] == pytest.approx(0.5) and bt.summarize(t)["bias_MID"] == 0
    assert "sq_xi" not in t


def test_captain_with_vice_fallback():
    rows = league(gws=(1,))
    pred = rows["total_points"].to_numpy().copy()
    dnp = rows.index[rows["minutes"] == 0][0]
    pred[dnp] = 50.0                            # top prediction, didn't play
    t = bt.gw_table(rows, pred, squads=False)
    second = rows.loc[np.argsort(-pred, kind="mergesort")[1], "total_points"]
    assert t.loc[0, "capt"] == 0 and t.loc[0, "capt_fpl"] == second


def test_double_gameweek_floors_each_fixture_for_the_squad(monkeypatch):
    rows = league(gws=(1,))
    extra = rows.iloc[[0]].copy()               # a second fixture for the first player
    extra["fixture"] = 999
    rows = pd.concat([rows, extra], ignore_index=True)
    pred = np.full(len(rows), 1.0)
    pred[0], pred[-1] = -1.0, 3.0
    seen = {}
    monkeypatch.setattr(bt, "squad_metrics", lambda g: seen.setdefault("g", g) is None or {})
    bt.gw_table(rows, pred)
    g = seen["g"].set_index("name")
    first = rows.loc[0, "name"]
    assert g.loc[first, "pred"] == pytest.approx(2.0) and g.loc[first, "pred_sq"] == pytest.approx(3.0)


# ----------------------------------------------------------------------------- two-level bootstrap

def seed_tables(offsets, shift=0.0, gws=60, seed=0):
    rng = np.random.default_rng(seed)
    effect = rng.normal(0, 2, gws)              # the gameweek part, shared by every variant
    rows = []
    for s, off in enumerate(offsets):
        for g in range(gws):
            rows.append({"season": "2025-26", "GW": g + 1, "seed": s, "top20": 4 + effect[g] + off + shift,
                         "se": 4.0 * (1 + shift), "rows": 1.0})
    return pd.DataFrame(rows)


def test_boot2_without_seed_noise_recovers_a_shift_exactly():
    b = bt._boot2(seed_tables([0, 0, 0]), seed_tables([0, 0], shift=0.25), n=2000)
    assert b["top20"] == pytest.approx((0.25, 0.25, 0.25))
    assert b["rmse"][0] == pytest.approx(np.sqrt(5.0) - 2.0)


def test_boot2_puts_the_seed_lottery_in_the_ci():
    b = bt._boot2(seed_tables([-1, 0, 1]), seed_tables([-1, 0, 1], shift=0.25), n=4000)
    lo, hi = b["top20"][1], b["top20"][2]
    assert b["top20"][0] == pytest.approx(0.25)
    assert lo < 0.25 < hi and hi - lo > 0.5     # gameweek effects cancel; seed offsets don't


def test_seed_spread():
    mean, sd, se = bt._seed_spread(seed_tables([-1, 0, 1]), "top20")
    assert sd == pytest.approx(1.0) and se > 0


# ----------------------------------------------------------------------------- saved predictions

def write_chunk(tmp_path, name, variants, rows):
    """A run's three files, as `run` writes them, from fake predictions."""
    rng = np.random.default_rng(sum(map(ord, name)))
    sink, out = bt.Sink(), []
    for variant, seeds in variants.items():
        preds = {s: rows["total_points"].to_numpy() + rng.normal(0, 2, len(rows)) for s in seeds}
        unit, per_seed = bt.score_unit(rows, preds, (variant, "refit", "2025-26"), sink)
        out.append({"variant": variant, "features": 3, "folds": {"2025-26": {
            "train": ["2023-24"], "valid": "2024-25", "refit": unit,
            "seeds": {s: {"best_iter": 10 + s, "n_refit": 12, "argmin_valid": 9, "refit": sm}
                      for s, sm in per_seed.items()}}}})
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(out))
    sink.save(str(path))
    return path


def test_score_pools_seeds_and_reuses_tables(tmp_path, monkeypatch):
    rows = league()
    write_chunk(tmp_path, "chunk_a", {"current": [0, 1], "other": [0, 1, 2]}, rows)
    write_chunk(tmp_path, "chunk_b", {"current": [2]}, rows)
    preds = pd.read_parquet(tmp_path / "chunk_a.preds.parquet")
    assert {"pred|current|refit|0", "pred|other|refit|2"} <= set(preds.columns) and len(preds) == len(rows)

    args = argparse.Namespace(preds=[str(tmp_path / "chunk_*.preds.parquet")], out=str(tmp_path / "res.json"),
                              pool=False, reuse=True, no_squads=False)
    with pytest.raises(SystemExit):
        bt.cmd_score(args)

    solves = []
    real = opt.best_squad
    monkeypatch.setattr(opt, "best_squad", lambda *a, **k: solves.append(1) or real(*a, **k))
    args.pool = True
    bt.cmd_score(args)
    assert len(solves) == 2                     # only current's new 3-seed average, one squad per gameweek
    res = {r["variant"]: r for r in json.loads((tmp_path / "res.json").read_text())}
    fold = res["current"]["folds"]["2025-26"]
    assert fold["refit"]["seed_list"] == [0, 1, 2]
    assert fold["seeds"]["0"]["best_iter"] == 10 and fold["seeds"]["2"]["best_iter"] == 12
    assert "sq_fpl" in fold["seeds"]["2"]["refit"]

    # the pooled average is what scoring all three seeds at once gives
    p = {s: preds[f"pred|current|refit|{s}"].to_numpy(dtype=float) for s in (0, 1)}
    p[2] = pd.read_parquet(tmp_path / "chunk_b.preds.parquet")["pred|current|refit|2"].to_numpy(dtype=float)
    direct, _ = bt.score_unit(rows, p, ("current", "refit", "2025-26"))
    assert fold["refit"]["summary"]["sq_fpl"] == pytest.approx(direct["summary"]["sq_fpl"])
    assert fold["refit"]["summary"]["rmse"] == pytest.approx(direct["summary"]["rmse"], rel=1e-5)

    tables = pd.read_parquet(tmp_path / "res.gw.parquet")
    assert sorted(tables.loc[tables["variant"] == "current", "seed"].unique()) == [bt.ENS, 0, 1, 2]

    report = tmp_path / "report.md"
    bt.cmd_report(argparse.Namespace(paths=[str(tmp_path / "res.json")], out=str(report), ref="current"))
    text = report.read_text(encoding="utf-8")
    assert "the optimiser's squad" in text and "Single models" in text and "Seed variance" in text
    assert "| other | 3/3 |" in text


def test_seed_parsing_and_matrix(capsys):
    assert bt.parse_seeds("0-2, 5") == [0, 1, 2, 5]
    with pytest.raises(SystemExit):
        bt.parse_seeds("0-2,1")
    bt.cmd_matrix(argparse.Namespace(variants="current,dc", seeds="", ref="current", ref_seeds="0-11", chunk=5))
    jobs = json.loads(capsys.readouterr().out.split("=", 1)[1])
    assert [j["tag"] for j in jobs] == ["current-s0", "current-s5", "current-s10", "dc"]
    assert jobs[2]["seeds"] == "10,11" and jobs[3]["seeds"] == "0,1,2"


# ----------------------------------------------------------------------------- expected points

def test_picks_are_scored_on_expected_points_too():
    rows = league()
    rows["xpts"] = np.where(rows["minutes"] > 0, rows["total_points"] / 2, 0.0)
    pred = rows["total_points"].to_numpy()
    t = bt.gw_table(rows, pred)
    for gw, g in rows.groupby("GW"):
        r = t[t["GW"] == gw].iloc[0]
        top = g.assign(pred=pred[g.index]).sort_values("pred", ascending=False, kind="mergesort")
        assert r["top20_xp"] == pytest.approx(top["xpts"].head(20).mean())
        assert r["top20_xp"] == pytest.approx(r["top20"] / 2)
        assert r["capt_fpl_xp"] == pytest.approx(r["capt_fpl"] / 2)
        assert r["sq_xi_xp"] == pytest.approx(r["sq_xi"] / 2) and r["sq_fpl_xp"] == pytest.approx(r["sq_fpl"] / 2)
    assert set(bt.XP_METRICS) <= set(bt.summarize(t))


def test_a_pick_without_expected_stats_has_no_expected_score():
    rows = league(gws=(1,))
    rows["xpts"] = rows["total_points"]
    pred = rows["total_points"].to_numpy()
    rows.loc[int(np.argmax(pred)), "xpts"] = np.nan          # the captain
    t = bt.gw_table(rows, pred, squads=False)
    assert np.isnan(t.loc[0, "top20_xp"]) and np.isnan(t.loc[0, "capt_fpl_xp"]) and not np.isnan(t.loc[0, "top20"])


def test_score_adds_expected_points_to_older_predictions(tmp_path, monkeypatch):
    rows = league()
    write_chunk(tmp_path, "chunk_a", {"current": [0, 1]}, rows)      # saved without xpts
    feats = rows.assign(xpts=rows["total_points"] / 2)
    feats.to_pickle(tmp_path / "feats.pkl.gz")
    solves = []
    real = opt.best_squad
    monkeypatch.setattr(opt, "best_squad", lambda *a, **k: solves.append(1) or real(*a, **k))
    args = argparse.Namespace(preds=[str(tmp_path / "chunk_*.preds.parquet")], out=str(tmp_path / "res.json"),
                              pool=True, reuse=True, no_squads=False, feats=str(tmp_path / "feats.pkl.gz"))
    bt.cmd_score(args)
    assert len(solves) == 6                     # the saved tables lack expected points: every table again
    unit = json.loads((tmp_path / "res.json").read_text())[0]["folds"]["2025-26"]["refit"]
    assert unit["summary"]["top20_xp"] == pytest.approx(unit["summary"]["top20"] / 2)


# ----------------------------------------------------------------------------- tree counts and seed ensembles

class Staged:
    """A fitted model stand-in: predict(X, iteration_range=(0, k)) is perfect at the tree counts in
    `good` and noise at the others."""

    def __init__(self, rows, good):
        self.rows, self.good = rows, set(good)

    def predict(self, X, iteration_range):
        k = iteration_range[1]
        if k in self.good:
            return self.rows.loc[X.index, "total_points"].to_numpy(dtype=float)
        return np.random.default_rng(k).normal(0, 3, len(X))


def test_stop_trees_keeps_the_rmse_stop_by_default_and_halves_it_on_request():
    rows = league(gws=(1,))
    assert bt._stop_trees(None, rows, rows, [], 120, {}) == (120, {})
    assert bt._stop_trees(None, rows, rows, [], 121, {"tree_frac": 0.5})[0] == 60


def test_stop_by_picks_takes_the_count_whose_validation_picks_scored_most():
    va = league(gws=(1, 2, 3))
    used, info = bt._stop_trees(Staged(va, good={20, 30, 40}), va, va, [], 95, {"stop": "picks"})
    assert used == 30                                      # the middle of the plateau after smoothing
    assert list(info["stop_grid"]) == ["10", "20", "30", "40", "50", "60", "70", "80", "90", "95"]


def test_ensemble_gain_is_the_averaged_predictions_minus_the_mean_single_model():
    gws = pd.DataFrame({"season": "2025-26", "GW": range(1, 31)})
    ens = gws.assign(top20=5.1, sq_fpl=60.5)
    seeds = pd.concat([gws.assign(seed=s, top20=5.0 + off, sq_fpl=60.0 + off) for s, off in enumerate((-0.2, 0, 0.2))])
    g = bt._ens_gain(ens, seeds, n=500)
    assert g["top20"] == pytest.approx((0.1, 0.1, 0.1)) and g["sq_fpl"] == pytest.approx((0.5, 0.5, 0.5))
    assert "capt_fpl" not in g


def test_tree_count_variants_exist():
    assert bt.VARIANTS["v4_trees05"]["tree_frac"] == 0.5 and bt.VARIANTS["v4_stop_picks"]["stop"] == "picks"
    assert bt.VARIANTS["v4_trees05"]["post"] == bt.VARIANTS["v4_prod"]["post"]
