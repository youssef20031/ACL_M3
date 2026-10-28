"""
Evaluation of the points model (ml/points_model.py) for the project report: the production v4
by default, or the v3 shadow with --version v3 (same protocol, same baselines).

    python -m ml.evaluate_points_v3                 # full run, a few minutes
    python -m ml.evaluate_points_v3 --n-boot 500    # quicker bootstrap
    python -m ml.evaluate_points_v3 --version v3    # the shadow model

Regenerates everything in docs/ml_evaluation/: metrics.json, the PNG figures and README.md
(the written report, filled from metrics.json so its numbers can't drift).

Protocol (identical to ml/train_points_v3.py, which train_points_v4 calls):
  train 2020-21 .. 2023-24, early stopping on 2024-25, test on 2025-26 and the current
  season so far. Ablations and seed repeats are retrained under the same protocol. The
  2026-10 features (last match, 20-match team strength) were chosen with a rolling-origin
  backtest (scripts/analysis/ml_backtest.py) whose folds include the test seasons.

Conventions:
  - Regression metrics are per player per fixture (one row each).
  - Pick metrics are per gameweek with double gameweeks summed per player first, as in
    ml.points_model.evaluate. "top-k" = mean actual points of the k highest predictions;
    "captain" = top-1; "hit@20" = share of our top 20 who scored at least the gameweek's
    20th-best score; the oracle ranks by the actual points (perfect hindsight).
  - Every method is scored on the same rows. Baselines without history fall back to 0
    (season pts/game falls back to the last-6 average first).
  - Uncertainty: bootstrap over gameweeks (stratified by season), fixed seed.
  - metrics.json keys the evaluated model as "model" (protocol.version says which); players are
    keyed by player_key (v4: the FPL code), as in training.
"""
import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
import zlib
from typing import Dict, List

import numpy as np
import pandas as pd

from ml import points_model as pm
from ml import train_points_v3 as tp
from ml.train_points_v3 import TRAIN_SEASONS, VALID_SEASONS

logger = logging.getLogger(__name__)

OUT_DIR = os.path.join("docs", "ml_evaluation")
SEED = 42
KS = (1, 10, 20, 50)
TARGET = "total_points"

METHOD_COLS = {
    "model": "pred_model", "season_ppg": "b_ppg", "last6": "b_last6", "last3": "b_last3",
    "pos_mean": "b_pos_mean", "zero": "b_zero",
}
HONEST_BASELINES = ["season_ppg", "last6", "last3", "pos_mean"]
LABELS = {
    "model": "v4 (XGBoost)", "season_ppg": "Season pts/game", "last6": "Last-6 avg", "last3": "Last-3 avg",
    "pos_mean": "Position mean", "zero": "Zero / random pick", "fpl_xP": "FPL xP (leaky)",
    "oracle": "Oracle (hindsight)", "live": "Production model (in-sample)",
}
# Reference palette (dataviz skill), fixed per method across every figure
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {
    "model": "#2a78d6", "season_ppg": "#eb6834", "last3": "#1baf7a", "last6": "#eda100",
    "pos_mean": "#9b9a96", "zero": "#c3c2b7", "fpl_xP": "#e34948", "oracle": "#0b0b0b", "live": "#4a3aa7",
}
SCOPE_COLORS = {"2025-26": "#2a78d6", "2026-27": "#eb6834", "all": "#0b0b0b"}
POS_COLORS = {"GK": "#e87ba4", "DEF": "#008300", "MID": "#4a3aa7", "FWD": "#e34948"}

GROUP_LABELS = {
    "form": "Form (last match, rolling 3/6)", "season_to_date": "Season to date",
    "team_opponent_strength": "Team / opponent strength", "context": "Fixture context",
    "price_ownership": "Price & ownership", "position": "Position",
    "ratings": "Team ratings (v4)", "market": "Transfers & price changes (v4)", "odds": "Bookmaker prices (v4)",
    "availability": "FPL availability (v4)",
}
VERSION = pm.MODEL_VERSION   # the evaluated model; set by _use_version
OUTCOME_COLS = [
    "total_points", "minutes", "starts", "goals_scored", "assists", "bonus", "bps", "ict_index",
    "influence", "creativity", "threat", "clean_sheets", "goals_conceded", "saves", "expected_goals",
    "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
    "defensive_contribution", "team_h_score", "team_a_score", "xP",
]


# --------------------------------------------------------------------------- helpers

def _rng(key: str) -> np.random.Generator:
    """Independent, reproducible stream per analysis (results don't depend on call order)."""
    return np.random.default_rng([SEED, zlib.crc32(key.encode())])


def _use_version(version: str) -> None:
    """Name the evaluated model in the figures and the README."""
    global VERSION
    VERSION = version
    LABELS["model"] = f"{version} (XGBoost)"


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else round(v, 4)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def reg_metrics(y, p) -> Dict[str, float]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = p - y
    sst = ((y - y.mean()) ** 2).sum()
    r = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and y.std() > 0 and p.std() > 0 else float("nan")
    return {"rows": int(len(y)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r2": float(1 - (e ** 2).sum() / sst) if sst > 0 else float("nan"), "bias": float(e.mean()),
            "pearson_r": r, "mean_pred": float(p.mean()), "mean_actual": float(y.mean())}


class Ranker:
    """Per-gameweek pick metrics over a fixed set of rows (double gameweeks summed per player).

    Predictions are passed as arrays aligned with `rows` (after reset_index)."""

    def __init__(self, rows: pd.DataFrame, min_players: int = 30):
        rows = rows.reset_index(drop=True)
        g = rows.groupby(["season", "GW", "name"], sort=True)
        self.pg_id = g.ngroup().to_numpy()
        self.pg_keys = g.size().reset_index()[["season", "GW", "name"]]
        self.n_pg = len(self.pg_keys)
        gw = self.pg_keys.groupby(["season", "GW"], sort=True)
        self.gw_of_pg = gw.ngroup().to_numpy()
        self.gw_keys = gw.size().reset_index(name="players")
        self.counts = self.gw_keys["players"].to_numpy()
        self.G = len(self.counts)
        self.gw_start = np.r_[0, np.cumsum(self.counts)[:-1]]
        self.y_rows = rows[TARGET].to_numpy(float)
        self.actual = self.agg(self.y_rows)
        order, rank = self._order(self.actual)
        thr = np.full(self.G, -np.inf)
        sel = rank == 19
        thr[self.gw_of_pg[order][sel]] = self.actual[order][sel]
        self.thr20 = thr
        self.keep = self.counts >= min_players
        self.row_gw = self.gw_of_pg[self.pg_id]
        self.row_counts = np.bincount(self.row_gw, minlength=self.G).astype(float)
        self._ra = None

    def agg(self, pred_rows) -> np.ndarray:
        return np.bincount(self.pg_id, weights=np.asarray(pred_rows, float), minlength=self.n_pg)

    def _order(self, p):
        order = np.lexsort((-p, self.gw_of_pg))  # by gameweek, then prediction descending
        rank = np.arange(self.n_pg) - self.gw_start[self.gw_of_pg[order]]
        return order, rank

    def _spearman(self, p) -> np.ndarray:
        g = pd.Series(self.gw_of_pg)
        if self._ra is None:
            ra = pd.Series(self.actual).groupby(g).rank()
            self._ra = ra - ra.groupby(g).transform("mean")
        rp = pd.Series(p).groupby(g).rank()
        dp = rp - rp.groupby(g).transform("mean")
        da = self._ra
        num = (dp * da).groupby(g).sum()
        den = np.sqrt((dp ** 2).groupby(g).sum() * (da ** 2).groupby(g).sum())
        return (num / den.replace(0, np.nan)).reindex(range(self.G)).to_numpy()

    def per_gw(self, pred_rows, spearman: bool = True) -> pd.DataFrame:
        pred_rows = np.asarray(pred_rows, float)
        p = self.agg(pred_rows)
        order, rank = self._order(p)
        gs, a = self.gw_of_pg[order], self.actual[order]
        out = self.gw_keys.copy()
        for k in KS:
            m = rank < k
            out[f"top{k}"] = np.bincount(gs[m], weights=a[m], minlength=self.G) / np.minimum(k, self.counts)
        m = rank < 20
        hits = (a[m] >= self.thr20[gs[m]]).astype(float)
        out["hit20"] = np.bincount(gs[m], weights=hits, minlength=self.G) / np.minimum(20, self.counts)
        if spearman:
            out["spearman"] = self._spearman(p)
        out["mae"] = np.bincount(self.row_gw, weights=np.abs(pred_rows - self.y_rows), minlength=self.G) / self.row_counts
        out["rows"] = self.row_counts
        return out[self.keep].reset_index(drop=True)

    def pg_frame(self, pred_rows) -> pd.DataFrame:
        """Player-gameweek table with the prediction's rank inside its gameweek (1 = top)."""
        p = self.agg(pred_rows)
        order, rank = self._order(p)
        r = np.empty(self.n_pg, int)
        r[order] = rank + 1
        out = self.pg_keys.copy()
        out["pred"], out["actual"], out["rank"] = p, self.actual, r
        out["players"] = self.counts[self.gw_of_pg]
        return out


RANK_METRICS = ["spearman"] + [f"top{k}" for k in KS] + ["hit20"]


def summarize_gw(tbl: pd.DataFrame) -> Dict[str, float]:
    out = {m: float(tbl[m].mean()) for m in RANK_METRICS if m in tbl}
    out["gameweeks"] = int(len(tbl))
    return out


def boot_dist(tbl: pd.DataFrame, col: str, key: str, n_boot: int, weight: str = None) -> np.ndarray:
    """Bootstrap distribution of the (weighted) mean of a per-gameweek column, resampling
    gameweeks with replacement within each season."""
    rng = _rng(key)
    num, den = np.zeros(n_boot), np.zeros(n_boot)
    for _, part in tbl.groupby("season"):
        v = part[col].to_numpy(float)
        w = part[weight].to_numpy(float) if weight else np.ones(len(v))
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        num += (v * w)[idx].sum(axis=1)
        den += w[idx].sum(axis=1)
    return num / den


def ci_summary(tbl: pd.DataFrame, col: str, key: str, n_boot: int, weight: str = None, diff: bool = False) -> dict:
    w = tbl[weight].to_numpy(float) if weight else np.ones(len(tbl))
    est = float((tbl[col].to_numpy(float) * w).sum() / w.sum())
    dist = boot_dist(tbl, col, key, n_boot, weight)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    out = {"estimate": est, "ci95": [float(lo), float(hi)]}
    if diff:
        out["p_two_sided"] = float(min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean())))
        out["significant_5pct"] = bool(lo > 0 or hi < 0)
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Per-gameweek a - b (both from the same Ranker, so rows align)."""
    out = a[["season", "GW", "rows"]].copy()
    for c in cols:
        out[c] = a[c].to_numpy() - b[c].to_numpy()
    return out


def feature_groups(cols: List[str]) -> Dict[str, List[str]]:
    groups = {
        "form": [c for c in cols if c.endswith(("_r1", "_r3", "_r6"))],
        "season_to_date": [c for c in cols if c in ("season_matches_before", "season_ppg",
                                                     "season_minutes_share", "career_matches_before")],
        "team_opponent_strength": [c for c in cols if c.startswith(("team_", "opp_"))],
        "context": [c for c in cols if c in ("fixtures_in_gw", "was_home", "days_rest")],
        "price_ownership": [c for c in cols if c in ("price", "selected_rank")],
        "position": [c for c in cols if c.startswith("pos_")],
        "ratings": [c for c in cols if c.startswith("rt_")],
        "market": [c for c in cols if c.startswith(("tr_", "price_change_"))],
        "odds": [c for c in cols if c.startswith("odds_")],
        "availability": [c for c in cols if c.startswith("snap_")],
    }
    assert sorted(sum(groups.values(), [])) == sorted(cols), "every feature belongs to exactly one group"
    return {g: gc for g, gc in groups.items() if gc}


def fit_cols(train: pd.DataFrame, valid: pd.DataFrame, cols: List[str], seed: int = 0):
    """ml.points_model.fit with a chosen feature list and seed (seed 0 == pm.fit's default)."""
    import xgboost as xgb
    model = xgb.XGBRegressor(n_estimators=3000, early_stopping_rounds=100, random_state=seed, **pm.XGB_PARAMS)
    model.fit(train[cols], train[TARGET], eval_set=[(valid[cols], valid[TARGET])], verbose=False)
    return model


def calibration_table(y, p, bins: int = 10) -> List[dict]:
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    g = d.groupby("bin")
    t = pd.DataFrame({"mean_pred": g["p"].mean(), "mean_actual": g["y"].mean(), "rows": g.size(),
                      "se": g["y"].std() / np.sqrt(g.size())})
    return t.reset_index().to_dict("records")


def calib_slope(y, p) -> Dict[str, float]:
    slope, intercept = np.polyfit(np.asarray(p, float), np.asarray(y, float), 1)
    return {"slope": float(slope), "intercept": float(intercept)}


# --------------------------------------------------------------------------- analyses

def baseline_predictions(test: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    rng = _rng("tie-jitter")
    jitter = rng.uniform(0, 1e-6, len(test))  # break ties of the constant baselines randomly
    pos_mean = train.groupby("position")[TARGET].mean()
    test["b_zero"] = jitter
    test["b_pos_mean"] = test["position"].map(pos_mean).to_numpy() + jitter
    test["b_last3"] = test["total_points_r3"].fillna(0)
    test["b_last6"] = test["total_points_r6"].fillna(0)
    test["b_ppg"] = test["season_ppg"].fillna(test["total_points_r6"]).fillna(0)
    test["b_xp"] = test["xP"]
    return test


def evaluate_scopes(test: pd.DataFrame, ranker: Ranker, method_cols: Dict[str, str], seasons: List[str]):
    """Regression + pick metrics per season and combined. Returns (results, per-gameweek tables)."""
    tables = {m: ranker.per_gw(test[c].to_numpy()) for m, c in method_cols.items()}
    tables["oracle"] = ranker.per_gw(test[TARGET].to_numpy())
    results = {}
    for scope in seasons + ["all"]:
        mask = np.ones(len(test), bool) if scope == "all" else (test["season"] == scope).to_numpy()
        played = mask & (test["minutes"] > 0).to_numpy()
        res = {}
        for m, c in method_cols.items():
            t = tables[m] if scope == "all" else tables[m][tables[m]["season"] == scope]
            res[m] = {"regression": reg_metrics(test.loc[mask, TARGET], test.loc[mask, c]),
                      "regression_played": reg_metrics(test.loc[played, TARGET], test.loc[played, c]),
                      "ranking": summarize_gw(t)}
        t = tables["oracle"] if scope == "all" else tables["oracle"][tables["oracle"]["season"] == scope]
        res["oracle"] = {"ranking": summarize_gw(t)}
        results[scope] = res
    return results, tables


def significance(tables, best: str, seasons: List[str], n_boot: int) -> dict:
    out = {"best_baseline": best, "n_boot": n_boot, "method": "bootstrap over gameweeks, stratified by season",
           "model": {}, "baselines": {}, "model_minus_best": {}, "model_minus_each_top20": {}, "gameweeks_won": {}}
    from scipy.stats import binomtest
    for scope in seasons + ["all"]:
        sel = (lambda t: t) if scope == "all" else (lambda t, s=scope: t[t["season"] == s].reset_index(drop=True))
        mod = sel(tables["model"])
        out["model"][scope] = {m: ci_summary(mod, m, f"model-{scope}-{m}", n_boot) for m in ["top1", "top10", "top20", "top50", "spearman", "hit20"]}
        out["model"][scope]["mae"] = ci_summary(mod, "mae", f"model-{scope}-mae", n_boot, weight="rows")
        out["baselines"][scope] = {b: {"top20": ci_summary(sel(tables[b]), "top20", f"{b}-{scope}", n_boot)}
                                   for b in HONEST_BASELINES}
        d = paired(mod, sel(tables[best]), ["top1", "top10", "top20", "top50", "spearman", "hit20", "mae"])
        out["model_minus_best"][scope] = {m: ci_summary(d, m, f"diff-{scope}-{m}", n_boot, diff=True,
                                                        weight="rows" if m == "mae" else None)
                                          for m in ["top1", "top10", "top20", "top50", "spearman", "hit20", "mae"]}
        out["model_minus_each_top20"][scope] = {
            b: ci_summary(paired(mod, sel(tables[b]), ["top20"]), "top20", f"diff-{scope}-{b}", n_boot, diff=True)
            for b in HONEST_BASELINES}
        wins = int((d["top20"] > 0).sum())
        losses = int((d["top20"] < 0).sum())
        p = binomtest(wins, wins + losses).pvalue if wins + losses else float("nan")
        out["gameweeks_won"][scope] = {"wins": wins, "losses": losses, "ties": int((d["top20"] == 0).sum()),
                                       "sign_test_p": float(p)}
    return out


def breakdown(test: pd.DataFrame, group: pd.Series, levels: List[str], methods: List[str], k: int = 10) -> dict:
    out = {}
    for lvl in levels:
        mask = (group == lvl).to_numpy()
        sub = test[mask].reset_index(drop=True)
        if not len(sub):
            continue
        r = Ranker(sub, min_players=10)
        res = {"rows": int(mask.sum()), "played_share": float((sub["minutes"] > 0).mean()), "methods": {}}
        for m in methods:
            col = METHOD_COLS[m]
            tbl = r.per_gw(sub[col].to_numpy())
            res["methods"][m] = {**reg_metrics(sub[TARGET], sub[col]),
                                 f"within_top{k}": float(tbl[f"top{k}"].mean()),
                                 "within_spearman": float(tbl["spearman"].mean())}
        tbl = r.per_gw(sub[TARGET].to_numpy(), spearman=False)
        res["oracle_within_top10"] = float(tbl[f"top{k}"].mean())
        out[lvl] = res
    return out


def permutation_importance(model, X: pd.DataFrame, ranker: Ranker, groups: Dict[str, List[str]], repeats: int):
    cols = list(X.columns)
    y = ranker.y_rows
    base = model.predict(X)
    base_mae = float(np.abs(base - y).mean())
    base_top20 = float(ranker.per_gw(base, spearman=False)["top20"].mean())
    rng = _rng("permutation")
    feat = {}
    Xp = X.copy()
    for c in cols:
        d_mae, d_top = [], []
        orig = Xp[c].to_numpy().copy()
        for _ in range(repeats):
            Xp[c] = orig[rng.permutation(len(orig))]
            p = model.predict(Xp)
            d_mae.append(np.abs(p - y).mean() - base_mae)
            d_top.append(base_top20 - ranker.per_gw(p, spearman=False)["top20"].mean())
        Xp[c] = orig
        feat[c] = {"mae_increase": float(np.mean(d_mae)), "mae_increase_std": float(np.std(d_mae)),
                   "top20_drop": float(np.mean(d_top))}
    grp = {}
    for gname, gcols in groups.items():
        d_mae, d_top = [], []
        origs = {c: Xp[c].to_numpy().copy() for c in gcols}
        for _ in range(repeats):
            perm = rng.permutation(len(Xp))
            for c in gcols:
                Xp[c] = origs[c][perm]  # permute the group's columns jointly
            p = model.predict(Xp)
            d_mae.append(np.abs(p - y).mean() - base_mae)
            d_top.append(base_top20 - ranker.per_gw(p, spearman=False)["top20"].mean())
        for c in gcols:
            Xp[c] = origs[c]
        grp[gname] = {"n_features": len(gcols), "mae_increase": float(np.mean(d_mae)),
                      "mae_increase_std": float(np.std(d_mae)), "top20_drop": float(np.mean(d_top)),
                      "top20_drop_std": float(np.std(d_top))}
    return {"base_mae": base_mae, "base_top20": base_top20, "repeats": repeats, "features": feat, "groups": grp}


def lookahead_audit(history: pd.DataFrame, feats_all: pd.DataFrame, cols: List[str], season: str, gw: int,
                    extras=None) -> dict:
    """Scramble every outcome of `season` GW >= gw, rebuild features, and check that no feature
    of any row up to that gameweek changed (rows of the next gameweek must change: power check)."""
    h2 = history.copy()
    mask = ((h2["season"] == season) & (h2["GW"] >= gw)).to_numpy()
    rng = _rng("lookahead")
    for c in OUTCOME_COLS:
        vals = h2.loc[mask, c].to_numpy()
        h2.loc[mask, c] = vals[rng.permutation(len(vals))]
    f2 = pm.build_features(h2, extras=extras)
    a = feats_all.set_index("_rid")
    b = f2.set_index("_rid").reindex(a.index)
    va, vb = a[cols].to_numpy(float), b[cols].to_numpy(float)
    same = np.isclose(va, vb, rtol=0, atol=1e-9, equal_nan=True).all(axis=1)
    s, g = a["season"].to_numpy(), a["GW"].to_numpy()
    upto = (s < season) | ((s == season) & (g <= gw))
    at = (s == season) & (g == gw)
    nxt = (s == season) & (g == gw + 1)
    return {"scrambled": f"{season} GW>={gw}: {len(OUTCOME_COLS)} outcome columns shuffled ({int(mask.sum()):,} rows)",
            "rows_checked_up_to_gw": int(upto.sum()), "rows_changed_up_to_gw": int((~same[upto]).sum()),
            "rows_at_gw": int(at.sum()), "rows_changed_at_gw": int((~same[at]).sum()),
            "rows_next_gw": int(nxt.sum()), "rows_changed_next_gw": int((~same[nxt]).sum())}


def single_feature_ranking(feats: pd.DataFrame, cols: List[str], train_mask, seasons: List[str],
                           model_preds: Dict[str, pd.Series]) -> dict:
    """Rank players by one column alone (sign chosen on the training seasons) and score the top 20."""
    tr = feats[train_mask]
    signs = {c: (1.0 if np.nan_to_num(tr[c].corr(tr[TARGET])) >= 0 else -1.0) for c in cols}
    signs["xP"] = 1.0
    out = {}
    for s in seasons:
        rows = feats[feats["season"] == s].reset_index(drop=True)
        r = Ranker(rows)
        res = {}
        for c in cols + ["xP"]:
            v = rows[c].to_numpy(float)
            if np.isnan(v).mean() > 0.5:
                continue
            v = signs[c] * v
            v = np.where(np.isnan(v), np.nanmin(v) - 1, v)  # missing values are never picked
            res[c] = float(r.per_gw(v, spearman=False)["top20"].mean())
        honest = {c: v for c, v in res.items() if c != "xP"}
        best = sorted(honest.items(), key=lambda kv: -kv[1])
        out[s] = {"xP_top20": res.get("xP"), "best_honest_feature": best[0][0], "best_honest_top20": best[0][1],
                  "top5_honest": best[:5],
                  "oracle_top20": float(r.per_gw(rows[TARGET].to_numpy(), spearman=False)["top20"].mean()),
                  "xP_corr": float(rows["xP"].corr(rows[TARGET])) if rows["xP"].notna().mean() > 0.5 else None}
        if s in model_preds:
            out[s]["model_top20"] = float(r.per_gw(model_preds[s].to_numpy(), spearman=False)["top20"].mean())
    return out


def v2_duplication_check(history_csv: str, season: str = "2022-23") -> dict:
    """Run the legacy FeatureEngineer on one season and measure row inflation."""
    from ml.feature_engineering import FeatureEngineer
    raw = pd.read_csv(history_csv, low_memory=False)
    s = raw[raw["season_x"] == season].copy()
    lvl = logging.getLogger("ml.feature_engineering").level
    logging.getLogger("ml.feature_engineering").setLevel(logging.WARNING)
    out = FeatureEngineer().engineer_features(s, is_training=True, lag_features=["total_points"])
    logging.getLogger("ml.feature_engineering").setLevel(lvl)
    hi = out[out["total_points"] >= 5]
    # Honest reference: how often a player's NEXT match repeats a >=5-point score
    h = s.sort_values(["name", "kickoff_time"]).copy()
    h["next"] = h.groupby("name")["total_points"].shift(-1)
    h = h[(h["total_points"] >= 5) & h["next"].notna()]
    return {"season": season, "input_rows": int(len(s)), "output_rows": int(len(out)),
            "inflation_factor": float(len(out) / len(s)),
            "share_target_equals_same_match_points_ge5": float((hi["upcoming"] == hi["total_points"]).mean()),
            "honest_share_next_match_equals_points_ge5": float((h["next"] == h["total_points"]).mean())}


def noise_ceiling(test: pd.DataFrame, train_rows: pd.DataFrame, valid_rows: pd.DataFrame) -> dict:
    """Two context estimates of how predictable per-fixture points are at all."""
    # (1) Share of variance between player-seasons (one-way ANOVA ICC): the R^2 of a predictor
    #     that knew every player's true average for the season but nothing match-specific.
    d = test[[TARGET, "season", "name"]].copy()
    d["k"] = d["season"] + "|" + d["name"]
    n = d.groupby("k").size()
    d = d[d["k"].isin(n[n >= 5].index)]
    g = d.groupby("k")[TARGET]
    ni, mi = g.size().to_numpy(float), g.mean().to_numpy()
    N, K, m = ni.sum(), len(ni), d[TARGET].mean()
    ssb = (ni * (mi - m) ** 2).sum()
    ssw = ((d[TARGET] - g.transform("mean")) ** 2).sum()
    msb, msw = ssb / (K - 1), ssw / (N - K)
    n0 = (N - (ni ** 2).sum() / N) / (K - 1)
    vb = max((msb - msw) / n0, 0)
    icc = vb / (vb + msw)
    # (2) "Process" model fed the SAME match's minutes, start and xG/xA/xGC (information no
    #     pre-deadline model can have): how much variance remains even then (finishing, bonus, cards).
    pcols = ["minutes", "started", "expected_goals", "expected_assists", "expected_goals_conceded",
             "was_home"] + [f"pos_{p}" for p in pm.POSITIONS]
    tr = train_rows[train_rows["expected_goals"].notna()]
    model = fit_cols(tr, valid_rows, pcols)
    p = model.predict(test[pcols])
    return {"icc_rows": int(N), "icc_player_seasons": int(K), "between_player_season_variance_share": float(icc),
            "process_model": {"features": pcols, "train_rows": int(len(tr)), "train_seasons": sorted(tr["season"].unique().tolist()),
                              **reg_metrics(test[TARGET], p)},
            "target_sd": float(test[TARGET].std()), "zero_minute_share": float((test["minutes"] == 0).mean()),
            "zero_points_share": float((test[TARGET] == 0).mean())}


def error_analysis(test: pd.DataFrame, ranker: Ranker, history: pd.DataFrame, best: str) -> dict:
    t = test.copy()
    t["resid"] = t[TARGET] - t["pred_model"]
    show = ["season", "GW", "name", "position", "team", "opponent", "price", "pred_model", TARGET, "minutes",
            "goals_scored", "assists", "clean_sheets", "bonus", "minutes_r3"]
    under = t.sort_values("resid", ascending=False).head(15)[show]
    over = t.sort_values("resid").head(15)[show]
    out = {"largest_under_predictions": under.to_dict("records"), "largest_over_predictions": over.to_dict("records")}

    # Hauls (player-gameweek >= 10 points): were they on our radar?
    pg = ranker.pg_frame(t["pred_model"].to_numpy())
    pgb = ranker.pg_frame(t[METHOD_COLS[best]].to_numpy())
    pos = t.groupby(["season", "GW", "name"])["position"].first()
    pg["position"] = pos.reindex(pd.MultiIndex.from_frame(pg[["season", "GW", "name"]])).to_numpy()
    hauls = pg["actual"] >= 10
    out["hauls"] = {
        "count": int(hauls.sum()), "share_of_player_gameweeks": float(hauls.mean()),
        "model_in_top20": float((pg.loc[hauls, "rank"] <= 20).mean()),
        "model_in_top50": float((pg.loc[hauls, "rank"] <= 50).mean()),
        f"{best}_in_top20": float((pgb.loc[hauls, "rank"] <= 20).mean()),
        f"{best}_in_top50": float((pgb.loc[hauls, "rank"] <= 50).mean()),
        "model_median_rank_percentile": float((pg.loc[hauls, "rank"] / pg.loc[hauls, "players"]).median()),
        "mean_pred": float(pg.loc[hauls, "pred"].mean()),
        "by_position": {p: {"hauls": int((hauls & (pg["position"] == p)).sum()),
                            "share_of_hauls": float((pg.loc[hauls, "position"] == p).mean()),
                            "share_of_model_top20_picks": float((pg.loc[pg["rank"] <= 20, "position"] == p).mean()),
                            "model_in_top50": float((pg.loc[hauls & (pg["position"] == p), "rank"] <= 50).mean())}
                        for p in pm.POSITIONS},
    }

    abs_err = (t["pred_model"] - t[TARGET]).abs()
    total = abs_err.sum()
    zero = t["minutes"] == 0
    regular_out = zero & (t["minutes_r3"] >= 60)
    out["availability"] = {
        "zero_minute_rows_share": float(zero.mean()), "zero_minute_mean_pred": float(t.loc[zero, "pred_model"].mean()),
        "zero_minute_share_of_abs_error": float(abs_err[zero].sum() / total),
        "regulars_who_missed_rows": int(regular_out.sum()),
        "regulars_who_missed_share_of_rows": float(regular_out.mean()),
        "regulars_who_missed_mean_pred": float(t.loc[regular_out, "pred_model"].mean()),
        "regulars_who_missed_share_of_abs_error": float(abs_err[regular_out].sum() / total),
    }

    def bucket_stats(groups: pd.Series, order: List[str]) -> dict:
        res = {}
        for lvl in order:
            mk = (groups == lvl).to_numpy()
            if mk.sum() == 0:
                continue
            res[lvl] = {**reg_metrics(t.loc[mk, TARGET], t.loc[mk, "pred_model"]),
                        "played_share": float((t.loc[mk, "minutes"] > 0).mean())}
        return res

    cm = t["career_matches_before"]
    hist_b = pd.Series(np.select([cm == 0, cm < 10, cm < 38], ["0 (debut)", "1-9", "10-37"], "38+"), index=t.index)
    out["by_career_history"] = bucket_stats(hist_b, ["0 (debut)", "1-9", "10-37", "38+"])
    mr = t["minutes_r3"]
    min_b = pd.Series(np.select([mr.isna(), mr == 0, mr <= 30, mr <= 60], ["no history", "0", "1-30", "31-60"], "61-90"),
                      index=t.index)
    out["by_recent_minutes"] = bucket_stats(min_b, ["no history", "0", "1-30", "31-60", "61-90"])

    teams = history.groupby("season")["team"].unique().to_dict()
    ss = sorted(teams)
    promoted = {b: set(teams[b]) - set(teams[a]) for a, b in zip(ss, ss[1:])}
    is_prom = pd.Series([tm in promoted.get(s, set()) for s, tm in zip(t["season"], t["team"])], index=t.index)
    out["promoted_teams"] = {s: sorted(promoted.get(s, [])) for s in sorted(t["season"].unique())}
    out["by_promoted"] = bucket_stats(is_prom.map({True: "promoted", False: "established"}), ["promoted", "established"])

    multi = t.groupby(["season", "GW", "name"])["team"].nunique()
    dup = t.groupby(["season", "fixture", "name"]).size()
    out["name_collisions"] = {"player_gameweeks_with_two_teams": int((multi > 1).sum()),
                              "duplicate_rows_same_fixture_and_name": int((dup > 1).sum()),
                              "duplicate_names": sorted({k[2] for k in dup[dup > 1].index})}
    return out


def dgw_analysis(test: pd.DataFrame, ranker: Ranker, tables, best: str) -> dict:
    tf = test.drop_duplicates(["season", "GW", "team", "fixture"]).groupby(["season", "GW", "team"]).size()
    doubles = tf[tf > 1].reset_index().groupby(["season", "GW"]).size()
    dgws = doubles[doubles >= 2]  # a real double gameweek has at least two teams playing twice
    if dgws.empty:
        return {"double_gameweeks": []}
    rows_dgw = test["fixtures_in_gw"] > 1
    res = {"double_gameweeks": [f"{s} GW{g} ({n} teams)" for (s, g), n in dgws.items()],
           "rows_double": reg_metrics(test.loc[rows_dgw, TARGET], test.loc[rows_dgw, "pred_model"]),
           "rows_single": reg_metrics(test.loc[~rows_dgw, TARGET], test.loc[~rows_dgw, "pred_model"])}
    pg = ranker.pg_frame(test["pred_model"].to_numpy())
    pg_best = ranker.pg_frame(test[METHOD_COLS[best]].to_numpy())
    nfix = test.groupby(["season", "GW", "name"])["fixtures_in_gw"].max()
    pg["dgw_player"] = nfix.reindex(pd.MultiIndex.from_frame(pg[["season", "GW", "name"]])).to_numpy() > 1
    in_dgw = pd.MultiIndex.from_frame(pg[["season", "GW"]]).isin(dgws.index)
    sub, subb = pg[in_dgw], pg_best[in_dgw]
    act_rank = sub.groupby(["season", "GW"])["actual"].rank(ascending=False, method="first")
    res["share_dgw_players_all"] = float(sub["dgw_player"].mean())
    res["share_dgw_players_model_top20"] = float(sub.loc[sub["rank"] <= 20, "dgw_player"].mean())
    res["share_dgw_players_best_baseline_top20"] = float(sub.loc[(subb["rank"] <= 20).to_numpy(), "dgw_player"].mean())
    res["share_dgw_players_actual_top20"] = float(sub.loc[act_rank <= 20, "dgw_player"].mean())
    key = pd.MultiIndex.from_frame(tables["model"][["season", "GW"]]).isin(dgws.index)
    res["top20_in_dgws"] = {m: float(tables[m].loc[key, "top20"].mean()) for m in ["model", best, "last3", "oracle"]}
    res["top20_in_other_gws"] = {m: float(tables[m].loc[~key, "top20"].mean()) for m in ["model", best, "last3", "oracle"]}
    return res


def phase_analysis(tables, seasons: List[str], methods: List[str]) -> dict:
    out = {}
    for scope in seasons + ["all"]:
        out[scope] = {}
        for phase, cond in [("GW1-5", lambda g: g <= 5), ("GW6+", lambda g: g > 5)]:
            row = {}
            for m in methods:
                t = tables[m] if scope == "all" else tables[m][tables[m]["season"] == scope]
                t = t[cond(t["GW"])]
                if len(t):
                    row[m] = {"gameweeks": int(len(t)), "top20": float(t["top20"].mean()),
                              "spearman": float(t["spearman"].mean()) if "spearman" in t else None}
            if row:
                out[scope][phase] = row
    return out


# --------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": "#8a8984",
        "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2,
    })
    return plt


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return name


def fig_topk(m, out_dir):
    plt = _plt()
    res = m["results"]["all"]
    methods = ["model", "season_ppg", "last6", "last3", "pos_mean", "zero"]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4), sharey=True)
    y = np.arange(len(methods))
    for ax, k in zip(axes, KS):
        vals = [res[mm]["ranking"][f"top{k}"] for mm in methods]
        ax.barh(y, vals, color=[COLORS[mm] for mm in methods], height=0.62, edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + 0.08, yi, f"{v:.2f}", va="center", fontsize=8, color=TEXT)
        orc = res["oracle"]["ranking"][f"top{k}"]
        ax.axvline(orc, color=TEXT, ls="--", lw=1)
        ax.text(orc - 0.1, -0.95, f"oracle {orc:.1f}", ha="right", va="center", fontsize=7.5, color=TEXT2)
        ax.set_xlim(0, orc * 1.05)
        ax.set_ylim(len(methods) - 0.5, -1.4)
        ax.set_title("Captain (top-1)" if k == 1 else f"Top-{k} picks")
        ax.set_xlabel("Mean actual points per pick")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[mm] for mm in methods])
    fig.suptitle("Pick quality per gameweek, test seasons 2025-26 + 2026-27 (double gameweeks summed per player)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_topk_vs_baselines.png")


def fig_timeseries(tables, best, seasons, out_dir):
    plt = _plt()
    sizes = [max(len(tables["model"][tables["model"]["season"] == s]), 1) for s in seasons]
    fig, axes = plt.subplots(1, len(seasons), figsize=(12.5, 3.6), sharey=True,
                             gridspec_kw={"width_ratios": [max(s, 4) for s in sizes]})
    axes = np.atleast_1d(axes)
    methods = ["model", best] + [b for b in ["last3"] if b != best]
    for ax, s in zip(axes, seasons):
        for mm in methods:
            t = tables[mm][tables[mm]["season"] == s]
            ax.plot(t["GW"], t["top20"], color=COLORS[mm], marker="o", ms=3.5, label=LABELS[mm],
                    lw=2 if mm == "model" else 1.4, alpha=1 if mm == "model" else 0.9)
        ax.set_title(s)
        ax.set_xlabel("Gameweek")
    axes[0].set_ylabel("Mean actual points of top-20 picks")
    axes[0].legend(loc="upper left", ncol=3)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_top20_by_gameweek.png")


def fig_significance(sig, seasons, out_dir, tables, best):
    plt = _plt()
    fig, (axc, ax, ax2) = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [0.8, 1, 1]})
    names = {"top1": "Captain", "top10": "Top-10", "top20": "Top-20", "top50": "Top-50"}
    scopes = seasons + ["all"]
    # Captain gets its own panel: with 5 gameweeks its 2026-27 interval would flatten the others
    for a, metrics, title in [(axc, ["top1"], "Captain pick"), (ax, ["top10", "top20", "top50"], "Top-k picks")]:
        for j, scope in enumerate(scopes):
            for i, mt in enumerate(metrics):
                r = sig["model_minus_best"][scope][mt]
                yy = i + (j - (len(scopes) - 1) / 2) * 0.22
                a.plot(r["ci95"], [yy, yy], color=SCOPE_COLORS[scope], lw=2)
                a.plot(r["estimate"], yy, "o", color=SCOPE_COLORS[scope], ms=6, mec=SURFACE, mew=1.5,
                       label=("combined" if scope == "all" else scope) if i == 0 else None)
        a.axvline(0, color=TEXT2, lw=1)
        a.set_yticks(range(len(metrics)))
        a.set_yticklabels([names[mt] for mt in metrics])
        a.set_ylim(len(metrics) - 0.5, -0.5)
        a.set_xlabel(f"{VERSION} minus {LABELS[best]} (points per pick)")
        a.set_title(f"{title}: difference, 95% CI")
        a.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    d = paired(tables["model"], tables[best], ["top20"])
    dist = boot_dist(d, "top20", "diff-all-top20", sig["n_boot"])
    ax2.hist(dist, bins=50, color=COLORS["model"], edgecolor=SURFACE, linewidth=0.5)
    lo, hi = sig["model_minus_best"]["all"]["top20"]["ci95"]
    for v in (lo, hi):
        ax2.axvline(v, color=TEXT, ls="--", lw=1)
    ax2.axvline(0, color=TEXT2, lw=1)
    ax2.set_xlabel(f"Bootstrap mean of {VERSION} minus best baseline, top-20 (combined)")
    ax2.set_ylabel("Bootstrap replicates")
    ax2.set_title(f"Top-20 difference: 95% CI [{lo:.2f}, {hi:.2f}]")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_significance.png")


def fig_breakdowns(bd, best, out_dir):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), gridspec_kw={"width_ratios": [4, 4, 2]})
    titles = {"position": "By position", "price_band": "By price band (£m)", "home_away": "Home vs away"}
    methods = ["model", best, "last3"] if best != "last3" else ["model", best]
    for ax, key in zip(axes, ["position", "price_band", "home_away"]):
        levels = list(bd[key].keys())
        x = np.arange(len(levels))
        w = 0.8 / (len(methods) + 1)
        for i, mm in enumerate(methods + ["oracle"]):
            if mm == "oracle":
                vals = [bd[key][l]["oracle_within_top10"] for l in levels]
                ax.bar(x + i * w, vals, w, color=SURFACE, edgecolor=TEXT, linewidth=1, hatch="///", label=LABELS[mm])
            else:
                vals = [bd[key][l]["methods"][mm]["within_top10"] for l in levels]
                ax.bar(x + i * w, vals, w, color=COLORS[mm], edgecolor=SURFACE, linewidth=1.5, label=LABELS[mm])
        ax.set_xticks(x + w * len(methods) / 2)
        ax.set_xticklabels(levels)
        ax.set_title(titles[key])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean actual points, top-10 picks\nwithin the group per gameweek")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_breakdowns.png")


def fig_calibration(cal, out_dir):
    plt = _plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    lim = 0
    for mm in ["model", "season_ppg"]:
        t = pd.DataFrame(cal["overall"][mm])
        ax.errorbar(t["mean_pred"], t["mean_actual"], yerr=1.96 * t["se"], color=COLORS[mm], marker="o", ms=5,
                    lw=1.6, capsize=2, label=LABELS[mm])
        lim = max(lim, t["mean_pred"].max(), t["mean_actual"].max())
    ax.plot([0, lim * 1.05], [0, lim * 1.05], color=TEXT2, ls="--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted points (decile of predictions)")
    ax.set_ylabel("Mean actual points (95% CI)")
    ax.set_title("Reliability, all test rows")
    ax.legend(loc="upper left")
    lim2 = 0
    for p in pm.POSITIONS:
        t = pd.DataFrame(cal["by_position"][p])
        ax2.plot(t["mean_pred"], t["mean_actual"], color=POS_COLORS[p], marker="o", ms=4, lw=1.6, label=p)
        lim2 = max(lim2, t["mean_pred"].max(), t["mean_actual"].max())
    ax2.plot([0, lim2 * 1.05], [0, lim2 * 1.05], color=TEXT2, ls="--", lw=1)
    ax2.set_xlabel("Mean predicted points (decile within position)")
    ax2.set_ylabel("Mean actual points")
    ax2.set_title(f"Reliability by position ({VERSION})")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_calibration.png")


def fig_importance(imp, out_dir, n=20):
    plt = _plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    gain = sorted(imp["gain_share"].items(), key=lambda kv: -kv[1])[:n]
    ax.barh(range(len(gain)), [v for _, v in gain], color=COLORS["model"], height=0.65)
    ax.set_yticks(range(len(gain)))
    ax.set_yticklabels([k for k, _ in gain])
    ax.invert_yaxis()
    ax.set_xlabel("Share of total split gain")
    ax.set_title(f"XGBoost gain (top {n})")
    ax.grid(axis="y", visible=False)
    perm = sorted(imp["permutation"]["features"].items(), key=lambda kv: -kv[1]["mae_increase"])[:n]
    ax2.barh(range(len(perm)), [v["mae_increase"] for _, v in perm], xerr=[v["mae_increase_std"] for _, v in perm],
             color=COLORS["model"], height=0.65, error_kw={"ecolor": TEXT2, "lw": 1})
    ax2.set_yticks(range(len(perm)))
    ax2.set_yticklabels([k for k, _ in perm])
    ax2.invert_yaxis()
    ax2.set_xlabel("Increase in test MAE when the feature is shuffled")
    ax2.set_title(f"Permutation importance on the test seasons (top {n})")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_feature_importance.png")


def fig_groups(imp, out_dir):
    plt = _plt()
    g = imp["permutation"]["groups"]
    keys = sorted(g, key=lambda k: -g[k]["mae_increase"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    y = np.arange(len(keys))
    ax.barh(y, [g[k]["mae_increase"] for k in keys], xerr=[g[k]["mae_increase_std"] for k in keys],
            color=COLORS["model"], height=0.6, error_kw={"ecolor": TEXT2, "lw": 1})
    ax.set_yticks(y)
    ax.set_yticklabels([f"{GROUP_LABELS[k]} ({g[k]['n_features']})" for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("Increase in test MAE")
    ax.set_title("Group permutation: accuracy")
    ax2.barh(y, [g[k]["top20_drop"] for k in keys], xerr=[g[k]["top20_drop_std"] for k in keys],
             color=COLORS["model"], height=0.6, error_kw={"ecolor": TEXT2, "lw": 1})
    ax2.axvline(0, color=TEXT2, lw=1)
    ax2.set_xlabel("Drop in top-20 points per pick")
    ax2.set_title("Group permutation: pick quality")
    for a in (ax, ax2):
        a.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_feature_groups.png")


def fig_ablations(abl, out_dir):
    plt = _plt()
    keys = [k for k in abl["variants"] if k != "full"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
    y = np.arange(len(keys))
    for i, k in enumerate(keys):
        r = abl["variants"][k]["top20_diff_vs_full"]
        ax.plot(r["ci95"], [i, i], color=COLORS["model"], lw=2)
        ax.plot(r["estimate"], i, "o", color=COLORS["model"], ms=6, mec=SURFACE, mew=1.5)
    ax.axvline(0, color=TEXT2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([abl["variants"][k]["label"] for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("Top-20 points per pick vs full model (95% CI)")
    ax.set_title("Ablations: pick quality")
    full_mae = abl["variants"]["full"]["test"]["mae"]
    vals = [abl["variants"][k]["test"]["mae"] - full_mae for k in keys]
    ax2.barh(y, vals, color=COLORS["model"], height=0.6)
    span = max(abs(v) for v in vals) or 1
    for yi, v in zip(y, vals):
        ax2.text(v + (0.03 if v >= 0 else -0.03) * span, yi, f"{v:+.4f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=7.5, color=TEXT)
    ax2.set_xlim(min(min(vals), 0) - 0.35 * span, max(max(vals), 0) + 0.35 * span)
    ax2.axvline(0, color=TEXT2, lw=1)
    ax2.set_xlabel(f"Test MAE minus full model ({full_mae:.3f})")
    ax2.set_title("Ablations: accuracy (higher = worse)")
    for a in (ax, ax2):
        a.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_ablations.png")


def fig_single_feature(sf, out_dir):
    plt = _plt()
    seasons = list(sf.keys())
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(seasons))
    series = [("fpl_xP", "xP_top20", "FPL xP alone"), ("pos_mean", "best_honest_top20", "Best single honest feature"),
              ("model", "model_top20", f"{VERSION} (out of sample)")]
    w = 0.26
    for i, (ck, key, lab) in enumerate(series):
        vals = [sf[s].get(key) for s in seasons]
        xs = [xx + (i - 1) * w for xx, v in zip(x, vals) if v is not None]
        vs = [v for v in vals if v is not None]
        ax.bar(xs, vs, w, color=COLORS[ck], edgecolor=SURFACE, linewidth=1.5, label=lab)
        for xx, v in zip(xs, vs):
            ax.text(xx, v + 0.08, f"{v:.1f}", ha="center", fontsize=7.5, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(best: {sf[s]['best_honest_feature']})" for s in seasons], fontsize=8)
    ax.set_ylabel("Mean actual points of top-20 picks")
    ax.set_title("Ranking by a single column: FPL's historical xP vs pre-deadline features")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_leakage_single_feature.png")


def fig_error_patterns(err, out_dir):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    for ax, key, title in [(axes[0], "by_career_history", "By career appearances before the match"),
                           (axes[1], "by_recent_minutes", "By average minutes over the last 3 matches")]:
        d = err[key]
        lv = list(d.keys())
        x = np.arange(len(lv))
        ax.bar(x - 0.2, [d[l]["mean_pred"] for l in lv], 0.4, color=COLORS["model"], edgecolor=SURFACE, lw=1.5,
               label="Mean predicted")
        ax.bar(x + 0.2, [d[l]["mean_actual"] for l in lv], 0.4, color=COLORS["pos_mean"], edgecolor=SURFACE, lw=1.5,
               label="Mean actual")
        for xx, l in zip(x, lv):
            ax.text(xx, max(d[l]["mean_pred"], d[l]["mean_actual"]) + 0.08, f"n={d[l]['rows']:,}",
                    ha="center", fontsize=7.5, color=TEXT2)
        ax.set_xticks(x)
        ax.set_xticklabels(lv)
        ax.set_ylim(0, max(max(d[l]["mean_pred"], d[l]["mean_actual"]) for l in lv) * 1.18)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Points per fixture")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_error_patterns.png")


def fig_distribution(test, out_dir):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    bins = np.arange(-4, 25) - 0.5
    ax.hist(test[TARGET].clip(-3, 23), bins=bins, color=COLORS["pos_mean"], label="Actual points", edgecolor=SURFACE)
    ax.hist(test["pred_model"].clip(-3, 23), bins=np.arange(-3, 23.01, 0.25), histtype="step", color=COLORS["model"], lw=1.8,
            label=f"{VERSION} prediction (0.25-pt bins)")
    ax.set_yscale("log")
    ax.set_xlabel("Points in one fixture (clipped to 23)")
    ax.set_ylabel("Rows (log scale)")
    ax.set_title("Target distribution: most rows are 0-minute non-appearances or 1-2 point cameos")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "fig_points_distribution.png")


# --------------------------------------------------------------------------- README

def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _ci(r, nd=2):
    return f"{_f(r['estimate'], nd)} [{_f(r['ci95'][0], nd)}, {_f(r['ci95'][1], nd)}]"


def _sd(r, nd=2):
    est = r["estimate"]
    return f"{'+' if est >= 0 else ''}{_f(est, nd)} [{_f(r['ci95'][0], nd)}, {_f(r['ci95'][1], nd)}]"


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_readme(m: dict) -> str:
    seasons = m["protocol"]["test_seasons"]
    R = m["results"]
    best = m["significance"]["best_baseline"]
    sig = m["significance"]
    A = R["all"]
    ma, bba, orc, rnd = A["model"], A[best], A["oracle"], A["zero"]
    d20 = sig["model_minus_best"]["all"]["top20"]
    dcap = sig["model_minus_best"]["all"]["top1"]
    headroom = (ma["ranking"]["top20"] - rnd["ranking"]["top20"]) / (orc["ranking"]["top20"] - rnd["ranking"]["top20"])
    headroom_b = (bba["ranking"]["top20"] - rnd["ranking"]["top20"]) / (orc["ranking"]["top20"] - rnd["ranking"]["top20"])
    sigword = lambda r: "significant at 5%" if r["significant_5pct"] else "not significant at 5%"
    V = m["protocol"].get("version", "v3")
    L = []
    w = L.append
    w(f"# Evaluation of the {V} points model")
    w("")
    w(f"Generated by `python -m ml.evaluate_points_v3 --version {V}` on {m['meta']['generated_at']} (git `{m['meta']['git_commit']}`, "
      f"runtime {m['meta']['runtime_seconds']:.0f}s). Every number below is read from [`metrics.json`](metrics.json); "
      "rerun the script to refresh both.")
    w("")
    w("## Summary")
    w("")
    w(f"- **Top-20 picks:** on the held-out test seasons ({' + '.join(seasons)}, {ma['ranking']['gameweeks']} gameweeks) "
      f"the 20 players {V} rates highest each gameweek score **{_ci(sig['model']['all']['top20'])}** points each (95% CI), "
      f"versus {bba['ranking']['top20']:.2f} for the best simple baseline ({LABELS[best].lower()}) and "
      f"{rnd['ranking']['top20']:.2f} for a random pick. Difference **{_sd(d20)}**, {sigword(d20)}; {V} wins "
      f"{sig['gameweeks_won']['all']['wins']} of {sig['gameweeks_won']['all']['wins'] + sig['gameweeks_won']['all']['losses']} "
      f"decided gameweeks (sign test p = {sig['gameweeks_won']['all']['sign_test_p']:.3f}).")
    w(f"- **Headroom:** perfect hindsight would average {orc['ranking']['top20']:.2f} for top-20; {V} closes "
      f"{100 * headroom:.0f}% of the gap between random picks and that oracle ({LABELS[best].lower()}: {100 * headroom_b:.0f}%).")
    w(f"- **Captain (top-1):** {ma['ranking']['top1']:.2f} vs {bba['ranking']['top1']:.2f} points "
      f"(difference {_sd(dcap)}, {sigword(dcap)}). Single picks are noisy: the CI is wide.")
    vp = ma["regression_played"]
    w(f"- **Accuracy:** test MAE {ma['regression']['mae']:.3f} / RMSE {ma['regression']['rmse']:.3f} / "
      f"R² {ma['regression']['r2']:.3f}, versus MAE {bba['regression']['mae']:.3f} / R² {bba['regression']['r2']:.3f} for "
      f"{LABELS[best].lower()}. Overall bias is {ma['regression']['bias']:+.3f}. On rows where the player did play "
      f"({100 * vp['rows'] / ma['regression']['rows']:.0f}%), predictions are {-vp['bias']:.2f} points low on average "
      f"(they are expectations that include the chance of not playing) and correlate r = {vp['pearson_r']:.2f} with the outcome.")
    w(f"- **Rank correlation:** mean per-gameweek Spearman {ma['ranking']['spearman']:.3f} vs "
      f"{A['last3']['ranking']['spearman']:.3f} for the last-3 average. Spearman is dominated by separating "
      f"non-playing from playing players, which simple form already does; {V}'s advantage is at the top of the table.")
    nc = m["noise_ceiling"]
    w(f"- **Points are noisy:** {100 * nc['zero_minute_share']:.0f}% of rows are 0-minute non-appearances; even a model given "
      f"the match's own minutes and xG/xA reaches only R² {nc['process_model']['r2']:.2f}, and the between-player-season "
      f"share of variance is {nc['between_player_season_variance_share']:.2f}. {V}'s R² of {ma['regression']['r2']:.2f} "
      + ("is about what knowing every player's true season average would give, without knowing that average."
         if abs(ma["regression"]["r2"] - nc["between_player_season_variance_share"]) < 0.03 else
         "should be read against those ceilings."))
    la = m["leakage_audit"]
    w(f"- **No leakage found:** {la['row_count']['feature_rows']:,} feature rows for {la['row_count']['input_rows']:,} "
      f"input rows; scrambling every outcome from {la['lookahead']['scrambled'].split(':')[0]} onwards changed "
      f"{la['lookahead']['rows_changed_up_to_gw']} of {la['lookahead']['rows_checked_up_to_gw']:,} earlier feature rows; "
      f"the strongest single feature correlates {la['max_abs_corr']['test'][0][1]:.2f} with the target. "
      f"The legacy v2 pipeline inflated rows {la['v2_duplication']['inflation_factor']:.0f}x.")
    ab = m["ablations"]["variants"]
    worst = min((k for k in ab if k != "full"), key=lambda k: ab[k]["top20_diff_vs_full"]["estimate"])
    sigabl = [ab[k]["label"].lower() for k in ab if k != "full" and ab[k]["top20_diff_vs_full"]["significant_5pct"]]
    sst = m["seed_stability"]
    w(f"- **Ablations** (3 seeds each): the biggest loss comes from *{ab[worst]['label'].lower()}* "
      f"({_sd(ab[worst]['top20_diff_vs_full'])} top-20 points). Significant losses: "
      f"{', '.join(sigabl) if sigabl else 'none'}.")
    w(f"- **Training noise:** across {len(sst['seeds'])} XGBoost seeds, top-20 is {sst['top20']['mean']:.2f} ± "
      f"{sst['top20']['std']:.2f} and captain {sst['top1']['mean']:.2f} ± {sst['top1']['std']:.2f}. The headline uses "
      f"seed 0, which is what `ml/train_points_{V}.py` trains.")
    w("")
    w("## Task and data")
    w("")
    w("The model predicts a player's FPL points for one fixture, before the gameweek deadline. The app uses the "
      "predictions to rank players (best picks, captain, best value), so pick quality matters more than squared error.")
    w("")
    rows = [[s, f"{m['data']['rows_by_season'][s]:,}", m["data"]["gameweeks_by_season"][s], role]
            for s, role in m["data"]["season_roles"].items()]
    w(_table(["Season", "Rows (player x fixture)", "Gameweeks", "Role"], rows))
    w("")
    w(f"Sources: `data/cleaned_merged_seasons_cleaned.csv` (2020-21 to 2025-26) and the current-season file "
      f"`{m['data']['live_csv']}` ({seasons[-1]} GW1-{m['data']['gameweeks_by_season'][seasons[-1]]}), loaded with "
      "`ml.points_model.load_history` exactly as in training. Features: "
      f"{m['protocol']['n_features']} columns from `ml.points_model.feature_columns('{V}')`"
      + ("; the extra inputs come from `ml.points_model.load_extra_inputs` (FPL player codes, football-data.co.uk "
         "prices, FPL availability snapshots), with these shares of rows covered: "
         + ", ".join(f"{k} {v:.0%}" for k, v in m["data"]["input_coverage"]["all"].items())
         if m["data"].get("input_coverage") else "") + ".")
    w("")
    w("![Target distribution](fig_points_distribution.png)")
    w("")
    w("## Protocol")
    w("")
    w(f"- **Temporal split, as in `ml/train_points_{V}.py`:** train {', '.join(m['protocol']['train_seasons'])}; early "
      f"stopping on {', '.join(m['protocol']['valid_seasons'])} (best iteration {m['protocol']['best_trees']} trees); test "
      f"{' and '.join(seasons)}. Hyperparameters are `ml.points_model.XGB_PARAMS`, chosen on the validation season. "
      "Ablations and seed repeats use the same split. The last-match and 20-match team-strength features were added "
      "in October 2026 after a rolling-origin backtest (`scripts/analysis/ml_backtest.py`) whose folds include the test "
      "seasons; they also win on the 2023-24 and 2024-25 folds alone, but the test numbers are no longer fully untouched.")
    w("- **No-leakage design:** one row per player per fixture; player features use only earlier gameweeks (a double "
      "gameweek's two fixtures share the pre-deadline values); team strength comes from a separate team-per-match table, "
      "also rolled over earlier gameweeks; FPL's historical xP is not a feature (it was recorded inside the gameweek).")
    w("- **Metrics:** regression metrics per row (player x fixture). Pick metrics per gameweek after summing a player's "
      "double-gameweek fixtures: top-k = mean actual points of the k highest predictions; captain = top-1; hit@20 = share "
      "of our top 20 who scored at least the gameweek's 20th-best score; oracle = ranking by actual points. Gameweek "
      "values are averaged with equal weight.")
    w("- **Baselines on identical rows:** zero (equivalent to a random pick for ranking), training-set position mean, "
      "last-3 and last-6 average points, and season-to-date points per game (falling back to last-6, then 0, before a "
      "player's first match of the season). Missing histories fall back to 0, so every method is scored on the same "
      f"{ma['regression']['rows']:,} rows. (The training report drops rows a baseline cannot score, so its baseline "
      "numbers differ slightly.)")
    w(f"- **Uncertainty:** {sig['n_boot']} bootstrap resamples of gameweeks, stratified by season, fixed seed "
      f"{m['meta']['seed']}. Paired differences resample the same gameweeks for both methods.")
    w("")
    w("## Results vs baselines")
    w("")
    w(f"Combined test seasons ({ma['ranking']['gameweeks']} gameweeks, {ma['regression']['rows']:,} rows). "
      "Bias = mean(predicted - actual).")
    w("")
    methods = ["model", "season_ppg", "last6", "last3", "pos_mean", "zero"]
    rows = []
    for mm in methods:
        r, k = A[mm]["regression"], A[mm]["ranking"]
        name = f"**{LABELS[mm]}**" if mm == "model" else LABELS[mm]
        rp = A[mm]["regression_played"]
        rows.append([name, _f(r["mae"], 3), _f(r["rmse"], 3), _f(r["r2"], 3), _f(r["bias"], 3),
                     _f(rp["mae"], 3), _f(rp["bias"], 3), _f(rp["pearson_r"], 3), _f(k["spearman"], 3), _f(k["top1"]),
                     _f(k["top10"]), _f(k["top20"]), _f(k["top50"]), _f(k["hit20"])])
    k = orc["ranking"]
    rows.append([LABELS["oracle"], "", "", "", "", "", "", "", "1", _f(k["top1"]), _f(k["top10"]), _f(k["top20"]),
                 _f(k["top50"]), "1"])
    w(_table(["Method", "MAE", "RMSE", "R²", "Bias", "MAE (played)", "Bias (played)", "r (played)", "Spearman",
              "Captain", "Top-10", "Top-20", "Top-50", "Hit@20"], rows))
    w("")
    w(f"\"Played\" = the {vp['rows']:,} rows with minutes > 0. Every method is biased low there because it is scored "
      "against outcomes conditional on playing; r shows how well it orders those players.")
    w("")
    w("Per test season:")
    w("")
    rows = []
    for s in seasons:
        for mm in ["model", "season_ppg", "last6", "last3"] + (["fpl_xP"] if "fpl_xP" in R[s] else []):
            r, k = R[s][mm]["regression"], R[s][mm]["ranking"]
            name = f"**{LABELS[mm]}**" if mm == "model" else LABELS[mm]
            rows.append([s, name, _f(r["mae"], 3), _f(r["r2"], 3), _f(k["spearman"], 3), _f(k["top1"]), _f(k["top20"]),
                         _f(k["hit20"])])
        rows.append([s, LABELS["oracle"], "", "", "", _f(R[s]["oracle"]["ranking"]["top1"]),
                     _f(R[s]["oracle"]["ranking"]["top20"]), ""])
    w(_table(["Season", "Method", "MAE", "R²", "Spearman", "Captain", "Top-20", "Hit@20"], rows))
    w("")
    if "fpl_xP" in R.get("2025-26", {}):
        w("FPL's historical xP is shown for 2025-26 for completeness only. It is not a fair baseline: in these files it "
          "was recorded inside the gameweek (see the leakage audit), and in 2025-26 it does not behave like the "
          "pre-deadline `ep_next` the app sees at prediction time.")
        w("")
    w("![Pick quality vs baselines](fig_topk_vs_baselines.png)")
    w("")
    w("![Top-20 per gameweek](fig_top20_by_gameweek.png)")
    w("")
    w("## Significance")
    w("")
    w(f"Best honest baseline by combined top-20: **{LABELS[best]}**. 95% bootstrap CIs:")
    w("")
    rows = []
    for s in seasons + ["all"]:
        dd = sig["model_minus_best"][s]
        gwk = sig["gameweeks_won"][s]
        rows.append([("combined" if s == "all" else s), _ci(sig["model"][s]["top20"]), _ci(sig["baselines"][s][best]["top20"]),
                     _sd(dd["top20"]), (f"< {1 / sig['n_boot']:.4f}" if dd["top20"]["p_two_sided"] == 0
                                        else _f(dd["top20"]["p_two_sided"], 3)),
                     _sd(dd["top1"]), _sd(dd["spearman"], 3),
                     _sd(dd["mae"], 3), f"{gwk['wins']}-{gwk['losses']}"])
    w(_table(["Scope", f"{V} top-20", f"{LABELS[best]} top-20", "Δ top-20", "p (Δ top-20)", "Δ captain", "Δ Spearman",
              "Δ MAE", "GWs won-lost"], rows))
    w("")
    w("Top-20 difference against every baseline (combined): " + "; ".join(
        f"{LABELS[b]} {_sd(sig['model_minus_each_top20']['all'][b])}" for b in HONEST_BASELINES) + ".")
    w("")
    s_last = seasons[-1]
    w(f"The {s_last} interval rests on only {R[s_last]['model']['ranking']['gameweeks']} gameweeks, so treat it as indicative.")
    w("")
    w("![Significance](fig_significance.png)")
    w("")
    w("## Breakdowns")
    w("")
    bd = m["breakdowns"]
    w("Row-level accuracy and within-group pick quality (top-10 within the group, per gameweek):")
    w("")
    rows = []
    for key, title in [("position", "Position"), ("price_band", "Price"), ("home_away", "Venue")]:
        for lvl, r in bd[key].items():
            v, b = r["methods"]["model"], r["methods"][best]
            rows.append([title, lvl, f"{r['rows']:,}", f"{100 * r['played_share']:.0f}%", _f(v["mae"], 3), _f(v["bias"], 3),
                         _f(b["mae"], 3), _f(v["within_spearman"], 3), _f(v["within_top10"]), _f(b["within_top10"]),
                         _f(r["oracle_within_top10"])])
    w(_table(["Split", "Group", "Rows", "Played", f"{V} MAE", f"{V} bias", f"{LABELS[best]} MAE", f"{V} Spearman",
              f"{V} top-10", f"{LABELS[best]} top-10", "Oracle top-10"], rows))
    w("")
    w("![Breakdowns](fig_breakdowns.png)")
    w("")
    dg = m["double_gameweeks"]
    if dg.get("double_gameweeks"):
        w(f"**Double gameweeks** in the test data: {', '.join(dg['double_gameweeks'])}. Per fixture, {V}'s MAE is "
          f"{dg['rows_double']['mae']:.3f} on double-gameweek rows vs {dg['rows_single']['mae']:.3f} on single ones "
          f"(bias {dg['rows_double']['bias']:+.3f} vs {dg['rows_single']['bias']:+.3f}). Because a player's fixtures are "
          f"summed, every method heavily favours doubling players: they are {100 * dg['share_dgw_players_all']:.0f}% of players in those "
          f"gameweeks, {100 * dg['share_dgw_players_actual_top20']:.0f}% of the real top 20, and "
          f"{100 * dg['share_dgw_players_model_top20']:.0f}% of {V}'s top 20 ({LABELS[best].lower()}: "
          f"{100 * dg['share_dgw_players_best_baseline_top20']:.0f}%). Top-20 in double gameweeks: {V} "
          f"{dg['top20_in_dgws']['model']:.2f}, {LABELS[best].lower()} {dg['top20_in_dgws'][best]:.2f}, oracle "
          f"{dg['top20_in_dgws']['oracle']:.2f} (other gameweeks: {V} {dg['top20_in_other_gws']['model']:.2f}, "
          f"{LABELS[best].lower()} {dg['top20_in_other_gws'][best]:.2f}).")
        w("")
    ph = m["phase"]
    rows = []
    for s in seasons + ["all"]:
        for phase, r in ph[s].items():
            rows.append([("combined" if s == "all" else s), phase, r["model"]["gameweeks"], _f(r["model"]["top20"]),
                         _f(r[best]["top20"]), _f(r["last3"]["top20"]), _f(r["oracle"]["top20"]), _f(r["model"]["spearman"], 3),
                         _f(r[best]["spearman"], 3)])
    w("**Early season vs the rest** (in GW1-5 season-to-date stats are thin and rolling form spans the summer break):")
    w("")
    w(_table(["Scope", "Phase", "GWs", f"{V} top-20", f"{LABELS[best]} top-20", "Last-3 top-20", "Oracle top-20",
              f"{V} Spearman", f"{LABELS[best]} Spearman"], rows))
    w("")
    w("## Calibration")
    w("")
    cal = m["calibration"]
    w(f"Predicted-points deciles against mean actual points. Linear fit of actual on predicted: {V} slope "
      f"{cal['slope']['model']['slope']:.2f} (intercept {cal['slope']['model']['intercept']:+.2f}); "
      f"{LABELS['season_ppg'].lower()} {cal['slope']['season_ppg']['slope']:.2f}; last-3 {cal['slope']['last3']['slope']:.2f}. "
      "A slope near 1 means the predictions can be read as expected points; raw form averages are over-dispersed "
      "(slope well below 1) because they don't shrink towards the mean.")
    w("")
    top = cal["overall"]["model"][-1]
    w(f"In the top decile {V} predicts {top['mean_pred']:.2f} and players score {top['mean_actual']:.2f} on average. "
      "By position: " + "; ".join(f"{p} slope {cal['slope_by_position'][p]['slope']:.2f}" for p in pm.POSITIONS) + ".")
    w("")
    w("![Calibration](fig_calibration.png)")
    w("")
    w("## Feature importance")
    w("")
    imp = m["feature_importance"]
    gain = sorted(imp["gain_share"].items(), key=lambda kv: -kv[1])[:10]
    perm = sorted(imp["permutation"]["features"].items(), key=lambda kv: -kv[1]["mae_increase"])[:10]
    w(f"Gain is measured on the early-stopped model ({m['protocol']['best_trees']} trees); permutation importance "
      f"shuffles one column of the test set ({imp['permutation']['repeats']} repeats) and records the MAE increase "
      f"(base MAE {imp['permutation']['base_mae']:.3f}) and the drop in top-20 points (base {imp['permutation']['base_top20']:.2f}).")
    w("")
    w(_table(["Rank", "Gain (share)", "Permutation (ΔMAE, Δtop-20)"],
             [[i + 1, f"{g[0]} ({g[1]:.3f})", f"{p[0]} (+{p[1]['mae_increase']:.4f}, {p[1]['top20_drop']:+.2f})"]
              for i, (g, p) in enumerate(zip(gain, perm))]))
    w("")
    w("![Feature importance](fig_feature_importance.png)")
    w("")
    grp = imp["permutation"]["groups"]
    rows = [[GROUP_LABELS[g], r["n_features"], f"{imp['group_gain_share'][g]:.3f}", f"+{r['mae_increase']:.4f}",
             f"{r['top20_drop']:+.2f}"] for g, r in sorted(grp.items(), key=lambda kv: -kv[1]["mae_increase"])]
    w("Group-level view (columns of a group shuffled together):")
    w("")
    w(_table(["Group", "Features", "Gain share", "ΔMAE when shuffled", "Top-20 drop"], rows))
    w("")
    w("![Feature groups](fig_feature_groups.png)")
    w("")
    w("## Ablations")
    w("")
    w(f"Each variant is retrained from scratch under the same protocol (train 2020-21..2023-24, early stopping on "
      f"2024-25) with XGBoost seeds {m['ablations']['seeds']}. Metrics are averaged over the seeds (per gameweek for "
      "pick metrics), and the CI is a paired gameweek bootstrap against the full model averaged the same way, so the "
      "full-model row here is a seed average and differs slightly from the headline (seed 0).")
    w("")
    rows = []
    for key, r in ab.items():
        rows.append([r["label"], r["n_features"], "/".join(str(t) for t in r["trees_by_seed"]), _f(r["valid"]["top20"]),
                     _f(r["test"]["mae"], 3),
                     _f(r["test"]["r2"], 3), _f(r["test"]["spearman"], 3), _f(r["test"]["top1"]), _f(r["test"]["top20"]),
                     "" if key == "full" else _sd(r["top20_diff_vs_full"])])
    w(_table(["Variant", "Features", "Trees", "Valid top-20", "Test MAE", "Test R²", "Spearman", "Captain", "Top-20",
              "Δ top-20 vs full"], rows))
    w("")
    sst = m["seed_stability"]
    w(f"Seed stability (full model, seeds {sst['seeds']}): top-20 {sst['top20']['mean']:.3f} ± {sst['top20']['std']:.3f} "
      f"(range {sst['top20']['min']:.3f}-{sst['top20']['max']:.3f}), MAE {sst['mae']['mean']:.4f} ± {sst['mae']['std']:.4f}, "
      f"captain {sst['top1']['mean']:.2f} ± {sst['top1']['std']:.2f}. Seed 0 (the training script's default) gives "
      f"{sst['runs'][0]['top20']:.3f}, so the headline top-20 is "
      f"{'near the top of' if sst['runs'][0]['top20'] > sst['top20']['mean'] + 0.5 * sst['top20']['std'] else 'within'} "
      "the seed range. Seed-averaged ablation differences smaller than about one seed standard deviation should be "
      "read as noise even when their CI excludes 0.")
    w("")
    w("![Ablations](fig_ablations.png)")
    w("")
    w("## Leakage audit")
    w("")
    rc = la["row_count"]
    w(f"1. **One feature row per input row.** `build_features` returned {rc['feature_rows']:,} rows for {rc['input_rows']:,} "
      f"input rows ({'identical' if rc['equal'] else 'MISMATCH'}); every input row id appears exactly once "
      f"({rc['duplicated_ids']} duplicated ids).")
    lk = la["lookahead"]
    w(f"2. **No look-ahead (perturbation test).** {lk['scrambled']}, then rebuilt the features. Rows up to and including "
      f"that gameweek changed: **{lk['rows_changed_up_to_gw']} of {lk['rows_checked_up_to_gw']:,}** "
      f"({lk['rows_changed_at_gw']} of {lk['rows_at_gw']} at the gameweek itself). Power check: "
      f"{lk['rows_changed_next_gw']} of {lk['rows_next_gw']} rows of the following gameweek changed, as they should.")
    mc = la["max_abs_corr"]
    w(f"3. **No suspiciously strong single feature.** Highest |Pearson r| with the target on the test seasons: "
      + ", ".join(f"`{c}` {v:.3f}" for c, v in mc["test"][:5]) + f" (training seasons: `{mc['train'][0][0]}` "
      f"{mc['train'][0][1]:.3f}). For contrast, FPL's xP correlates "
      + ", ".join(f"{v:.2f} ({s})" for s, v in mc["xP_by_season"].items() if v is not None) + ".")
    sf = la["single_feature_ranking"]
    w("4. **Rank by a single column.** This is the test that exposed the xP leak: a pre-deadline signal should not "
      "pick much better than the best honest feature.")
    w("")
    rows = [[s, _f(r["xP_top20"]), f"{r['best_honest_feature']} ({_f(r['best_honest_top20'])})", _f(r.get("model_top20")),
             _f(r["oracle_top20"]), _f(r["xP_corr"], 3)] for s, r in sf.items()]
    w(_table(["Season", "xP alone top-20", "Best honest single feature", f"{V} top-20", "Oracle top-20", "corr(xP, points)"], rows))
    w("")
    w(f"{V} is shown only where it is out of sample (2024-25 is the early-stopping season, so mildly optimistic). In the "
      f"seasons where xP ranks far above every honest feature and above {V}, it carries same-gameweek information "
      "(team news/lineups), so it cannot be a feature or a fair baseline.")
    w("")
    w("![Single-feature ranking](fig_leakage_single_feature.png)")
    w("")
    vt = la["validation_vs_test"]
    w(f"5. **Validation vs test consistency.** Train (in-sample) MAE {vt['train']['mae']:.3f} / top-20 {vt['train']['top20']:.2f}; "
      f"validation {vt['valid']['mae']:.3f} / {vt['valid']['top20']:.2f}; test {vt['test']['mae']:.3f} / "
      f"{vt['test']['top20']:.2f}. Row-level error is nearly the same in and out of sample, so the model is not "
      "memorising rows. Pick quality is higher in-sample, as expected for trees that have seen those hauls. The "
      "validation top-20 is above the test seasons' even though 2024-25's oracle is lower "
      f"({la['single_feature_ranking'].get(VALID_SEASONS[0], {}).get('oracle_top20', float('nan')):.2f} vs "
      f"{orc['ranking']['top20']:.2f}): part of that gap is the mild optimism of early-stopping on that season, part "
      "is season-to-season variation. There is no sign of the near-perfect in-sample fit that leakage produces.")
    v2 = la["v2_duplication"]
    if "inflation_factor" in v2:
        w(f"6. **Why the legacy v2 numbers (R² 0.72-0.87) were inflated.** Running the old `FeatureEngineer` on "
          f"{v2['season']} turns {v2['input_rows']:,} rows into {v2['output_rows']:,} (**{v2['inflation_factor']:.1f}x**): "
          f"its opponent-strength merges join per-player values on (season, team, GW). The `shift(-1)` \"next match\" "
          f"target then mostly lands on a copy of the same match: for rows scoring 5+ points the target equals the row's "
          f"own points {100 * v2['share_target_equals_same_match_points_ge5']:.0f}% of the time, against "
          f"{100 * v2['honest_share_next_match_equals_points_ge5']:.0f}% for a player's genuine next match. Together with "
          "context features describing the previous match and the leaky xP, that explains the R² far above the "
          f"{nc['process_model']['r2']:.2f} that even same-match minutes and xG allow.")
    w("")
    w("## Error analysis")
    w("")
    ea = m["error_analysis"]
    av = ea["availability"]
    w(f"- **Availability is the largest error source.** 0-minute rows are {100 * av['zero_minute_rows_share']:.0f}% of the "
      f"test set and {100 * av['zero_minute_share_of_abs_error']:.0f}% of total absolute error. Players who averaged 60+ "
      f"minutes over their last three matches but then didn't play ({av['regulars_who_missed_rows']:,} rows, "
      f"{100 * av['regulars_who_missed_share_of_rows']:.1f}%) were predicted {av['regulars_who_missed_mean_pred']:.2f} "
      f"points on average and account for {100 * av['regulars_who_missed_share_of_abs_error']:.0f}% of the absolute error. "
      "These are injuries, suspensions and rotation that the historical data can't see; the live pipeline applies FPL's "
      "availability flags after the model, which this offline evaluation does not.")
    h = ea["hauls"]
    w(f"- **Hauls (10+ points in a gameweek)** are {h['count']} player-gameweeks ({100 * h['share_of_player_gameweeks']:.1f}%). "
      f"{100 * h['model_in_top20']:.0f}% were in {V}'s top 20 and {100 * h['model_in_top50']:.0f}% in its top 50 "
      f"({LABELS[best].lower()}: {100 * h[f'{best}_in_top20']:.0f}% / {100 * h[f'{best}_in_top50']:.0f}%); the median "
      f"haul came from a player ranked in the top {100 * h['model_median_rank_percentile']:.0f}% of that gameweek. By position: "
      + "; ".join(f"{p} {r['hauls']} hauls ({100 * r['share_of_hauls']:.0f}% of hauls, {100 * r['share_of_model_top20_picks']:.0f}% "
                  f"of {V} top-20 picks, {100 * r['model_in_top50']:.0f}% caught in top 50)" for p, r in h["by_position"].items()) + ".")
    ch = ea["by_career_history"]
    w("- **Little history:** " + "; ".join(f"{k} prior matches: n={r['rows']:,}, predicted {r['mean_pred']:.2f} vs actual "
                                         f"{r['mean_actual']:.2f}, MAE {r['mae']:.2f}" for k, r in ch.items()) + ".")
    bp = ea["by_promoted"]
    if "promoted" in bp:
        w(f"- **Promoted teams** ({'; '.join(f'{s}: ' + ', '.join(t) for s, t in ea['promoted_teams'].items())}): "
          f"bias {bp['promoted']['bias']:+.3f} vs {bp['established']['bias']:+.3f} for established teams, MAE "
          f"{bp['promoted']['mae']:.3f} vs {bp['established']['mae']:.3f}.")
    rm = ea["by_recent_minutes"]
    w("- **Rotation risk** (average minutes over the last three matches): " + "; ".join(
        f"{k}: played {100 * r['played_share']:.0f}%, predicted {r['mean_pred']:.2f} vs actual {r['mean_actual']:.2f}"
        for k, r in rm.items()) + ".")
    w("")
    w("![Error patterns](fig_error_patterns.png)")
    w("")
    show = lambda r: [r["season"], r["GW"], r["name"], r["position"], f"{r['team']} v {r['opponent']}",
                      _f(r["pred_model"]), int(r["total_points"]), int(r["minutes"]), int(r["goals_scored"]),
                      int(r["assists"])]
    hdr = ["Season", "GW", "Player", "Pos", "Fixture", "Predicted", "Actual", "Min", "G", "A"]
    w("Largest under-predictions (hauls nobody saw coming):")
    w("")
    w(_table(hdr, [show(r) for r in ea["largest_under_predictions"][:10]]))
    w("")
    w("Largest over-predictions (confident picks that blanked):")
    w("")
    w(_table(hdr, [show(r) for r in ea["largest_over_predictions"][:10]]))
    w("")
    w("## Limitations and future work")
    w("")
    nm = ea["name_collisions"]
    if V == "v3":
        w(f"- **No historical availability data.** Injuries, suspensions and rotation news aren't in the training data, so "
          "the model predicts an expected value over \"plays\" and \"doesn't play\". Adding the FPL status/chance-of-playing "
          "history (or a separate minutes model) is the most promising improvement.")
    else:
        w("- **Thin availability data.** FPL's status before each deadline comes from snapshots of vaastav's "
          "players_raw.csv history (about 35 of 38 gameweeks a season to 2024-25, a dozen since) and our own archive "
          "from GW9 2026-27; without a snapshot the availability features are missing, and the model predicts an expected "
          "value over \"plays\" and \"doesn't play\".")
    w(f"- **Player identity is {'a name string' if V == 'v3' else 'the FPL code'}.** In the test seasons "
      f"{nm['duplicate_rows_same_fixture_and_name']} (fixture, player) pairs have two rows "
      f"({', '.join(nm['duplicate_names']) or 'none'}), and "
      f"{nm['player_gameweeks_with_two_teams']} player-gameweeks mix two teams; such players' form features blend two "
      "records." + (" A player whose name is spelt differently across seasons also loses his history." if V == "v3" else
                    " Players are keyed by their FPL code, so a renamed player keeps his history."))
    w(f"- **Small current-season sample.** {seasons[-1]} has {R[seasons[-1]]['model']['ranking']['gameweeks']} gameweeks, so its "
      "intervals are wide and early-season effects dominate it.")
    w(f"- **xG features exist only from 2023-24.** Earlier training seasons have them missing, so the model learns "
      "xG-based form from one training season.")
    w(f"- **Irreducible noise.** The oracle top-20 is {orc['ranking']['top20']:.2f} against {V}'s "
      f"{ma['ranking']['top20']:.2f}, but the oracle uses outcomes; a model told the match's own minutes, starts and "
      f"xG/xA/xGC still only reaches R² {nc['process_model']['r2']:.2f} (MAE {nc['process_model']['mae']:.3f}), and "
      f"knowing each player-season's true average explains about {nc['between_player_season_variance_share']:.2f} of "
      "the per-fixture variance. Goals, assists and bonus are rare and lumpy, so large single-match errors are expected.")
    w("- **Single model, squared-error objective.** A Tweedie/zero-inflated objective or a two-stage minutes x points "
      "model could fit the zero-inflated target better; ranking objectives (pairwise) could target pick quality directly.")
    w("")
    lv = m["live_model"]
    w("## Production model check")
    w("")
    w(f"The served `points_{V}.json` ({lv['n_trees']} trees, refit on {', '.join(lv['seasons'])}) scored on the "
      f"same test rows: MAE {lv['all']['regression']['mae']:.3f}, R² {lv['all']['regression']['r2']:.3f}, Spearman "
      f"{lv['all']['ranking']['spearman']:.3f}, captain {lv['all']['ranking']['top1']:.2f}, top-20 "
      f"{lv['all']['ranking']['top20']:.2f}. **This is optimistic**: the production model was trained on these seasons. "
      "It only confirms that the deployed file loads and behaves like the evaluated one; the honest numbers are the ones above.")
    w("")
    cc = m["consistency_with_training_report"]
    w(f"Cross-check: `ml.points_model.evaluate` on this run's {V} predictions gives top-20 {cc['this_run']['top20_points']:.3f} / "
      f"MAE {cc['this_run']['mae']:.3f} (combined), "
      + (f"versus {cc['report']['top20_points']:.3f} / {cc['report']['mae']:.3f} in `ml/models/points_{V}_report.json`."
         if cc.get("report") else "(no training report found to compare against)."))
    w("")
    w("## How to reproduce")
    w("")
    w("```bash")
    w(f".venv/Scripts/python.exe -m ml.evaluate_points_v3 --version {V}   # writes docs/ml_evaluation/")
    w(".venv/Scripts/python.exe -m ml.evaluate_points_v3 --n-boot 500 --perm-repeats 1   # quicker")
    w("```")
    w("")
    w(f"Environment of this run: Python {m['meta']['python']}, XGBoost {m['meta']['xgboost']}, pandas {m['meta']['pandas']}. "
      "Seeds are fixed (XGBoost seed 0 as in training, NumPy streams derived from seed "
      f"{m['meta']['seed']}); reruns on the same data reproduce the numbers. When the current-season file grows, the "
      f"{seasons[-1]} numbers change with it.")
    w("")
    return "\n".join(L)


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the points model")
    parser.add_argument("--version", default=pm.MODEL_VERSION, choices=["v3", "v4"],
                        help="v4 (production, default) or v3 (the shadow)")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--perm-repeats", type=int, default=3)
    parser.add_argument("--skip-v2-check", action="store_true", help="Skip running the legacy FeatureEngineer")
    parser.add_argument("--readme-only", action="store_true", help="Rebuild README.md from an existing metrics.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.readme_only:
        with open(os.path.join(args.out_dir, "metrics.json"), encoding="utf-8") as f:
            m = json.load(f)
        _use_version(m["protocol"].get("version", "v3"))
        with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(build_readme(m))
        return 0
    _use_version(args.version)
    np.random.seed(SEED)
    t0 = time.time()
    lap = lambda msg: logger.info("%s (%.0fs)", msg, time.time() - t0)

    # ---- data and features, exactly as ml/train_points_v3.py (load_training_features)
    live_csv, status_csv, live_season = tp._live_files()
    history = pm.load_history(live_csv=live_csv, live_season=live_season)
    history["_rid"] = np.arange(len(history))
    extras = None
    if args.version != "v3":
        extras = pm.load_extra_inputs(live_dir=tp.LIVE_DATA_DIR, season=live_season, status_csv=status_csv,
                                      ratings_prior=False)
    feats_all = pm.build_features(history, extras=extras)
    # One key per player (v4: his FPL code), as training evaluates; double gameweeks are summed by it
    feats_all["name"] = feats_all[pm.PLAYER_KEY]
    lap("features built")
    cols = pm.feature_columns(args.version)
    feats = feats_all[feats_all[TARGET].notna()]
    test_seasons = sorted(set(feats["season"]) - set(TRAIN_SEASONS) - set(VALID_SEASONS))
    train = feats[feats["season"].isin(TRAIN_SEASONS)]
    valid = feats[feats["season"].isin(VALID_SEASONS)]
    test = feats[feats["season"].isin(test_seasons)].reset_index(drop=True).copy()

    m: dict = {"meta": {}, "protocol": {
        "version": args.version, "train_seasons": TRAIN_SEASONS, "valid_seasons": VALID_SEASONS,
        "test_seasons": test_seasons, "n_features": len(cols), "features": cols, "xgb_params": pm.XGB_PARAMS,
        "rows": {"train": len(train), "valid": len(valid), "test": len(test)}}}
    roles = {s: ("train" if s in TRAIN_SEASONS else "validation (early stopping)" if s in VALID_SEASONS else "test")
             for s in sorted(feats["season"].unique())}
    m["data"] = {"rows_by_season": feats.groupby("season").size().to_dict(),
                 "gameweeks_by_season": feats.groupby("season")["GW"].nunique().to_dict(),
                 "season_roles": roles, "live_csv": live_csv}
    if extras is not None:
        m["data"]["input_coverage"] = tp.input_coverage(feats)
        gaps = tp.odds_gaps(m["data"]["input_coverage"])   # the same guard as training
        if gaps:
            raise RuntimeError(f"too few rows with odds ({', '.join(f'{s}: {c:.0%}' for s, c in gaps.items())}): "
                               f"fetch them into ODDS_DIR first (python -m scripts.data.fetch_odds)")

    # ---- main model: pm.fit, exactly as training
    model = pm.fit(train, valid, cols=cols)
    best_trees = int(model.best_iteration) + 1
    m["protocol"]["best_trees"] = best_trees
    test["pred_model"] = model.predict(test[cols])
    test = baseline_predictions(test, train)
    lap(f"model trained ({best_trees} trees)")

    ranker = Ranker(test)
    results, tables = evaluate_scopes(test, ranker, METHOD_COLS, test_seasons)
    # FPL xP only where recorded (2025-26), on that season's rows
    for s in test_seasons:
        part = test[test["season"] == s].reset_index(drop=True)
        if part["xP"].notna().mean() > 0.5:
            part = part[part["xP"].notna()].reset_index(drop=True)
            r = Ranker(part)
            tbl = r.per_gw(part["b_xp"].to_numpy())
            results[s]["fpl_xP"] = {"regression": reg_metrics(part[TARGET], part["b_xp"]),
                                    "regression_played": reg_metrics(part.loc[part["minutes"] > 0, TARGET],
                                                                     part.loc[part["minutes"] > 0, "b_xp"]),
                                    "ranking": summarize_gw(tbl), "rows_with_xP": int(len(part))}
    m["results"] = results
    best = max(HONEST_BASELINES, key=lambda b: results["all"][b]["ranking"]["top20"])
    m["per_gameweek"] = {mm: tables[mm][["season", "GW", "top1", "top20", "spearman"]].to_dict("records")
                         for mm in ["model", best, "last3", "oracle"]}
    m["significance"] = significance(tables, best, test_seasons, args.n_boot)
    lap(f"results and bootstrap done (best baseline: {best})")

    # ---- breakdowns
    # Bands keep >= ~20 players per gameweek so a within-band top-10 is still a choice
    bands = ["<5.0", "5.0-5.9", "6.0-7.4", "7.5+"]
    price_band = pd.Series(np.select([test["price"] < 5.0, test["price"] < 6.0, test["price"] < 7.5], bands[:3], bands[3]))
    bmethods = ["model", best] + [b for b in ["last3"] if b != best]
    m["breakdowns"] = {
        "position": breakdown(test, test["position"], pm.POSITIONS, bmethods),
        "price_band": breakdown(test, price_band, bands, bmethods),
        "home_away": breakdown(test, test["was_home"].map({1.0: "home", 0.0: "away"}), ["home", "away"], bmethods),
    }
    m["double_gameweeks"] = dgw_analysis(test, ranker, tables, best)
    m["phase"] = phase_analysis(tables, test_seasons, list(dict.fromkeys(["model", best, "last3", "oracle"])))

    # ---- calibration
    m["calibration"] = {
        "overall": {mm: calibration_table(test[TARGET], test[METHOD_COLS[mm]]) for mm in ["model", "season_ppg", "last3"]},
        "by_position": {p: calibration_table(test.loc[test["position"] == p, TARGET], test.loc[test["position"] == p, "pred_model"])
                        for p in pm.POSITIONS},
        "slope": {mm: calib_slope(test[TARGET], test[METHOD_COLS[mm]]) for mm in ["model", "season_ppg", "last6", "last3"]},
        "slope_by_position": {p: calib_slope(test.loc[test["position"] == p, TARGET], test.loc[test["position"] == p, "pred_model"])
                              for p in pm.POSITIONS},
    }
    lap("breakdowns and calibration done")

    # ---- feature importance (gain on the early-stopped trees only)
    booster = model.get_booster()[:best_trees]
    total_gain = booster.get_score(importance_type="total_gain")
    tg = sum(total_gain.values())
    groups = feature_groups(cols)
    gain_share = {c: total_gain.get(c, 0.0) / tg for c in cols}
    m["feature_importance"] = {
        "gain_share": gain_share,
        "avg_gain": booster.get_score(importance_type="gain"),
        "group_gain_share": {g: sum(gain_share[c] for c in gc) for g, gc in groups.items()},
        "groups": groups,
        "permutation": permutation_importance(model, test[cols], ranker, groups, args.perm_repeats),
    }
    lap("feature importance done")

    # ---- ablations and seed stability (same protocol, retrained)
    variants = {
        "full": ("Full model", cols),
        "no_strength": ("Without team/opponent strength", [c for c in cols if c not in groups["team_opponent_strength"]]),
        "no_price_ownership": ("Without price & ownership rank", [c for c in cols if c not in groups["price_ownership"]]),
        "no_context": ("Without fixture context", [c for c in cols if c not in groups["context"]]),
        "no_season": ("Without season-to-date", [c for c in cols if c not in groups["season_to_date"]]),
        "form_only": ("Form only (+ position)", groups["form"] + groups["position"]),
        "no_last_match": ("Without last-match features", [c for c in cols if not c.endswith("_r1")]),
        "no_team20": ("Without 20-match team strength", [c for c in cols if not c.endswith(("gf20", "ga20"))]),
        "r3_only": ("Rolling window 3 only", [c for c in cols if not c.endswith(("_r1", "_r6"))]),
        "r6_only": ("Rolling window 6 only", [c for c in cols if not c.endswith(("_r1", "_r3"))]),
    }
    for g, label in (("ratings", "Without team ratings"), ("market", "Without transfers & price changes"),
                     ("odds", "Without bookmaker prices"), ("availability", "Without FPL availability")):
        if g in groups:   # v4's additions
            variants[f"no_{g}"] = (label, [c for c in cols if c not in groups[g]])
    vranker = Ranker(valid)

    def run_variant(vcols, seed):
        mdl = fit_cols(train, valid, vcols, seed=seed)
        p = mdl.predict(test[vcols])
        vt = vranker.per_gw(mdl.predict(valid[vcols]), spearman=False)
        return {"trees": int(mdl.best_iteration) + 1, "pred": p, "tbl": ranker.per_gw(p),
                "valid_top20": float(vt["top20"].mean())}

    # Seed stability of the full model; seed 0 reproduces pm.fit exactly
    seeds = [0, 1, 2, 3, 4]
    full_runs, srows = {}, []
    for sd in seeds:
        run = run_variant(cols, sd)
        full_runs[sd] = run
        if sd == 0:
            m["protocol"]["fit_wrapper_matches_pm_fit_max_abs_diff"] = float(np.abs(run["pred"] - test["pred_model"].to_numpy()).max())
        srows.append({"seed": sd, "trees": run["trees"], "mae": float(np.abs(run["pred"] - test[TARGET]).mean()),
                      **{k: float(run["tbl"][k].mean()) for k in ["top1", "top20", "spearman"]}})
    sdf = pd.DataFrame(srows)
    m["seed_stability"] = {"seeds": seeds, "runs": srows,
                           **{k: {"mean": float(sdf[k].mean()), "std": float(sdf[k].std(ddof=1)),
                                  "min": float(sdf[k].min()), "max": float(sdf[k].max())}
                              for k in ["mae", "top1", "top20", "spearman"]}}
    lap("seed stability done")

    # Ablations: every variant trained with ABL_SEEDS; per-gameweek metrics averaged over seeds
    # before the paired bootstrap, so seed luck doesn't masquerade as a feature effect
    abl_seeds = [0, 1, 2]

    def seed_avg(tbls):
        out = tbls[0][["season", "GW", "players", "rows"]].copy()
        for c in RANK_METRICS + ["mae"]:
            out[c] = np.mean([t[c].to_numpy() for t in tbls], axis=0)
        return out

    full_avg = seed_avg([full_runs[s]["tbl"] for s in abl_seeds])
    abl = {"seeds": abl_seeds, "variants": {}}
    for key, (label, vcols) in variants.items():
        runs = [full_runs[s] for s in abl_seeds] if key == "full" else [run_variant(vcols, s) for s in abl_seeds]
        avg = full_avg if key == "full" else seed_avg([r["tbl"] for r in runs])
        regs = [reg_metrics(test[TARGET], r["pred"]) for r in runs]
        res = {"label": label, "n_features": len(vcols), "trees_by_seed": [r["trees"] for r in runs],
               "valid": {"top20": float(np.mean([r["valid_top20"] for r in runs]))},
               "test": {**{k: float(np.mean([g[k] for g in regs])) for k in ("mae", "rmse", "r2", "bias")},
                        **summarize_gw(avg)},
               "test_top20_by_season": {s: float(avg.loc[avg["season"] == s, "top20"].mean()) for s in test_seasons}}
        if key != "full":
            d = paired(avg, full_avg, ["top20", "mae"])
            res["top20_diff_vs_full"] = ci_summary(d, "top20", f"abl-{key}", args.n_boot, diff=True)
            res["mae_diff_vs_full"] = ci_summary(d, "mae", f"abl-mae-{key}", args.n_boot, weight="rows", diff=True)
        abl["variants"][key] = res
        lap(f"ablation {key} done")
    m["ablations"] = abl

    # ---- leakage audit
    la = {"row_count": {"input_rows": int(len(history)), "feature_rows": int(len(feats_all)),
                        "equal": bool(len(history) == len(feats_all)),
                        "duplicated_ids": int(feats_all["_rid"].duplicated().sum())}}
    corr_test = test[cols].corrwith(test[TARGET]).abs().sort_values(ascending=False)
    corr_train = train[cols].corrwith(train[TARGET]).abs().sort_values(ascending=False)
    xp_corr = {s: (float(feats.loc[feats["season"] == s, "xP"].corr(feats.loc[feats["season"] == s, TARGET]))
                   if feats.loc[feats["season"] == s, "xP"].notna().mean() > 0.5 else None)
               for s in sorted(feats["season"].unique())}
    la["max_abs_corr"] = {"test": list(corr_test.head(10).items()), "train": list(corr_train.head(10).items()),
                          "xP_by_season": xp_corr}
    mid_gw = 20 if (test.loc[test["season"] == test_seasons[0], "GW"].max() or 0) > 21 else 2
    la["lookahead"] = lookahead_audit(history, feats_all, cols, test_seasons[0], mid_gw, extras)
    lap("look-ahead audit done")
    valid_pred = pd.Series(model.predict(valid[cols]), index=valid.index)
    model_by_season = {VALID_SEASONS[0]: valid_pred.reset_index(drop=True)}
    for s in test_seasons:
        model_by_season[s] = test.loc[test["season"] == s, "pred_model"].reset_index(drop=True)
    sf_seasons = [s for s in ["2023-24"] + VALID_SEASONS + test_seasons if s in set(feats["season"])]
    la["single_feature_ranking"] = single_feature_ranking(
        feats, cols, feats["season"].isin(TRAIN_SEASONS), sf_seasons, model_by_season)
    tr_pred = model.predict(train[cols])
    tr_t = Ranker(train).per_gw(tr_pred, spearman=False)
    va_t = vranker.per_gw(valid_pred.to_numpy(), spearman=False)
    la["validation_vs_test"] = {
        "train": {"mae": float(np.abs(tr_pred - train[TARGET]).mean()), "top20": float(tr_t["top20"].mean())},
        "valid": {"mae": float(np.abs(valid_pred - valid[TARGET]).mean()), "top20": float(va_t["top20"].mean()),
                  **{k: v for k, v in reg_metrics(valid[TARGET], valid_pred).items() if k in ("rmse", "r2")}},
        "test": {"mae": results["all"]["model"]["regression"]["mae"], "top20": results["all"]["model"]["ranking"]["top20"],
                 "rmse": results["all"]["model"]["regression"]["rmse"], "r2": results["all"]["model"]["regression"]["r2"]},
    }
    if args.skip_v2_check:
        la["v2_duplication"] = {"skipped": True}
    else:
        try:
            la["v2_duplication"] = v2_duplication_check(pm.HISTORY_CSV)
        except Exception as exc:  # legacy code: never let it break the evaluation
            logger.warning("v2 duplication check failed: %s", exc)
            la["v2_duplication"] = {"error": str(exc)}
    m["leakage_audit"] = la
    lap("leakage audit done")

    # ---- noise ceiling, error analysis
    m["noise_ceiling"] = noise_ceiling(test, train, valid)
    m["error_analysis"] = error_analysis(test, ranker, history, best)
    lap("error analysis done")

    # ---- production model (optimistic: trained on the test seasons too)
    booster_live, meta_live = pm.load_model(version=args.version)
    test["pred_live"] = pm.predict(booster_live, meta_live, test)
    live_tbl = ranker.per_gw(test["pred_live"].to_numpy())
    m["live_model"] = {"note": "Refit on all seasons including the test seasons: in-sample, optimistic.",
                       "n_trees": meta_live.get("n_trees"), "seasons": meta_live.get("seasons"),
                       "trained_at": meta_live.get("trained_at")}
    for scope in test_seasons + ["all"]:
        mk = np.ones(len(test), bool) if scope == "all" else (test["season"] == scope).to_numpy()
        t = live_tbl if scope == "all" else live_tbl[live_tbl["season"] == scope]
        m["live_model"][scope] = {"regression": reg_metrics(test.loc[mk, TARGET], test.loc[mk, "pred_live"]),
                                  "ranking": summarize_gw(t)}

    # ---- cross-check with ml.points_model.evaluate and the training report
    cc = {"this_run": pm.evaluate(test, "pred_model").as_dict()}
    rp = tp.report_path_for(args.version)
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            rep = json.load(f)
        cc["report"] = rep.get("test", {}).get("all", {}).get(args.version)
    m["consistency_with_training_report"] = cc

    # ---- figures
    m["figures"] = [
        fig_distribution(test, args.out_dir),
        fig_topk(m, args.out_dir),
        fig_timeseries(tables, best, test_seasons, args.out_dir),
        fig_significance(m["significance"], test_seasons, args.out_dir, tables, best),
        fig_breakdowns(m["breakdowns"], best, args.out_dir),
        fig_calibration(m["calibration"], args.out_dir),
        fig_importance(m["feature_importance"], args.out_dir),
        fig_groups(m["feature_importance"], args.out_dir),
        fig_ablations(m["ablations"], args.out_dir),
        fig_single_feature(la["single_feature_ranking"], args.out_dir),
        fig_error_patterns(m["error_analysis"], args.out_dir),
    ]
    lap("figures written")

    import xgboost
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    m["meta"] = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "git_commit": commit,
                 "python": platform.python_version(), "xgboost": xgboost.__version__, "pandas": pd.__version__,
                 "seed": SEED, "n_boot": args.n_boot, "runtime_seconds": time.time() - t0}
    m = _clean(m)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    with open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(build_readme(m))
    lap(f"wrote {args.out_dir}/metrics.json and README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
