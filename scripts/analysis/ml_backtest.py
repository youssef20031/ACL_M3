"""
Rolling-origin backtest of the v3 points model (experiment harness, runs in CI).

    python -m scripts.analysis.ml_backtest features --out feats.pkl.gz
    python -m scripts.analysis.ml_backtest run --feats feats.pkl.gz --variants base,d6 --out res.json
    python -m scripts.analysis.ml_backtest report res_*.json

For every test season S (2023-24 .. current) and every variant:
  * "es":    trained on seasons < S-1, early-stopped on S-1 (patience 100, as in train_points_v3)
  * "refit": refit on all seasons < S with the early-stopped tree count scaled by the extra rows,
             which is how the production model is built
Each is trained with several seeds; the seed-averaged prediction is scored per gameweek, and
the report compares every variant against "base" (the current v3) with paired bootstrap CIs
over gameweeks.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ml import points_model as pm  # noqa: E402

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
TEST_SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]
SEEDS = (0, 1, 2)
NMAX = 3000
PATIENCE = 100          # what train_points_v3 uses
PATIENCE_FIT = 400      # train further so the curve shows what happens after the stop
CURVE_POINTS = (50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000)

# ----------------------------------------------------------------------------- features

P90_STATS = ["total_points", "expected_goals", "expected_assists", "expected_goal_involvements",
             "threat", "creativity", "bps", "bonus", "defensive_contribution", "saves",
             "expected_goals_conceded", "goals_scored", "assists"]
P90_WINDOWS = (10, 30)
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}


def _rolling_sum_before_gameweek(frame, key, cols, windows):
    """Like pm._rolling_before_gameweek, but rolling sums."""
    frame = frame.sort_values([key, "kickoff_time"])
    out = pd.DataFrame(index=frame.index)
    shifted = frame.groupby(key, sort=False)[cols].shift(1)
    for w in windows:
        rolled = shifted.groupby(frame[key], sort=False).rolling(w, min_periods=1).sum()
        rolled = rolled.reset_index(level=0, drop=True).reindex(frame.index)
        out[[f"{c}_s{w}" for c in cols]] = rolled[cols].to_numpy()
    first_in_gw = frame.groupby([key, "season", "GW"], sort=False).cumcount() == 0
    out = out.where(first_in_gw)
    out[["_k", "_s", "_g"]] = frame[[key, "season", "GW"]].to_numpy()
    return out.groupby(["_k", "_s", "_g"], sort=False).transform("first")


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    known = df["minutes"].notna()

    # Per-90 rates over the last 10 / 30 matches, counting only minutes where the stat exists
    aux = {}
    for s in P90_STATS:
        has = df[s].notna() & known
        aux[f"n_{s}"] = df[s].where(has, 0.0).where(known)
        aux[f"m_{s}"] = df["minutes"].where(has, 0.0).where(known)
    tmp = pd.concat([df[["name", "kickoff_time", "season", "GW"]], pd.DataFrame(aux, index=df.index)], axis=1)
    sums = _rolling_sum_before_gameweek(tmp, "name", list(aux), P90_WINDOWS)
    for w in P90_WINDOWS:
        for s in P90_STATS:
            den = sums[f"m_{s}_s{w}"]
            df[f"{s}_p90_w{w}"] = (sums[f"n_{s}_s{w}"] / den * 90).where(den >= 90)
        df[f"minutes_sum_w{w}"] = sums[f"m_total_points_s{w}"]

    # Last match, and a longer 10-match window
    r1 = pm._rolling_before_gameweek(df, "name", ["minutes", "started", "total_points"], (1,))
    r10 = pm._rolling_before_gameweek(df, "name", ["minutes", "started", "total_points", "bps", "ict_index"], (10,))
    df = df.join(r1).join(r10)

    # FPL's defensive-contribution points (2025-26 rule): 2 points for 10+ CBIT (DEF) or 12+ CBIRT
    # (MID/FWD); GKs are not eligible. NaN where the action count isn't recorded (before 2025-26).
    dc = df["defensive_contribution"]
    thr = df["position"].map(DC_THRESHOLD)
    dcp = (2.0 * (dc >= thr)).where(dc.notna())
    df["dc_points"] = dcp.where(df["position"] != "GK", 0.0).where(dc.notna())
    df = df.join(pm._rolling_before_gameweek(df, "name", ["dc_points"], (3, 6, 10)))

    # Team / opponent strength over 20 matches
    cols = ["gf", "ga", "xgf", "xga"]
    tm = pm._team_match_table(df[~df["is_upcoming"]])
    tm = tm.join(pm._rolling_before_gameweek(tm, "team", cols, (20,)))
    keyed = tm.set_index(["season", "fixture", "team"])[[f"{c}_r20" for c in cols]]
    for team_col, prefix in (("team", "team"), ("opponent", "opp")):
        idx = pd.MultiIndex.from_frame(df[["season", "fixture", team_col]].rename(columns={team_col: "team"}))
        vals = keyed.reindex(idx).to_numpy()
        for j, c in enumerate(cols):
            col = f"{prefix}_{c}20"
            df[col] = vals[:, j]
            if c in ("gf", "ga"):
                df[col] = df[col].fillna(pm.NEW_TEAM_PRIOR[f"team_{c}"])
    return df


BASE = pm.feature_columns()
G_P90 = [f"{s}_p90_w{w}" for w in P90_WINDOWS for s in P90_STATS] + [f"minutes_sum_w{w}" for w in P90_WINDOWS]
G_R1 = ["minutes_r1", "started_r1", "total_points_r1"]
G_R10 = ["minutes_r10", "started_r10", "total_points_r10", "bps_r10", "ict_index_r10"]
G_TEAM20 = [f"{p}_{c}20" for p in ("team", "opp") for c in ("gf", "ga", "xgf", "xga")]
G_DC = ["dc_points_r3", "dc_points_r6", "dc_points_r10"]
KEEP = ["season", "GW", "name", "position", "team", "fixture", "kickoff_time", "total_points", "minutes", "xP"]


def cmd_features(args) -> None:
    t0 = time.time()
    live_csv, live_season = None, None
    meta_path = os.path.join("data", "live", "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        live_csv, live_season = os.path.join("data", "live", meta["files"]["season"]), meta["season"]
    history = pm.load_history(live_csv=live_csv, live_season=live_season)
    feats = pm.build_features(history)
    print(f"base features: {len(feats)} rows in {time.time() - t0:.0f}s", flush=True)
    feats = add_extra_features(feats)
    feats = feats[feats["total_points"].notna()]
    cols = list(dict.fromkeys(KEEP + BASE + G_P90 + G_R1 + G_R10 + G_TEAM20 + G_DC))
    feats = feats[cols].copy()
    for c in feats.columns:
        if feats[c].dtype == "float64":
            feats[c] = feats[c].astype("float32")
    feats.to_pickle(args.out)
    print(f"saved {args.out}: {feats.shape}, seasons {feats.groupby('season').size().to_dict()} "
          f"in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- variants

P = dict(pm.XGB_PARAMS)
NOCAREER = [c for c in BASE if c != "career_matches_before"]
VARIANTS = {
    "base": {},
    "d6": {"params": {**P, "max_depth": 6}},
    "d4": {"params": {**P, "max_depth": 4}},
    "mcw200": {"params": {**P, "min_child_weight": 200}},
    "hl1": {"halflife": 1.0},
    "hl2": {"halflife": 2.0},
    "last3": {"max_seasons": 3},
    "p90": {"features": BASE + G_P90},
    "r1": {"features": BASE + G_R1},
    "r10": {"features": BASE + G_R10},
    "team20": {"features": BASE + G_TEAM20},
    "nocareer": {"features": NOCAREER},
    "all": {"features": BASE + G_P90 + G_R1 + G_R10 + G_TEAM20},
    # round 2: combinations of the round-1 winners
    "r1t20": {"features": BASE + G_R1 + G_TEAM20},
    "r1t20nc": {"features": NOCAREER + G_R1 + G_TEAM20},
    "r1t20r10": {"features": BASE + G_R1 + G_TEAM20 + G_R10},
    "r1t20dc": {"features": BASE + G_R1 + G_TEAM20 + G_DC},
    "r1t20_d6": {"features": BASE + G_R1 + G_TEAM20, "params": {**P, "max_depth": 6}},
    "r1t20_mcw200": {"features": BASE + G_R1 + G_TEAM20, "params": {**P, "min_child_weight": 200}},
}


def _weights(df, halflife):
    if not halflife:
        return None
    idx = df["season"].map({s: k for k, s in enumerate(SEASONS)}).astype(float)
    w = 0.5 ** ((idx.max() - idx) / halflife)
    return (w / w.mean()).to_numpy()


def _patience_stop(curve, patience):
    best_i, best = 0, float("inf")
    for i, v in enumerate(curve):
        if v < best:
            best_i, best = i, v
        elif i - best_i >= patience:
            break
    return best_i


def gw_table(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    d = test[["season", "GW", "name", "position", "total_points"]].copy()
    d["pred"] = pred.astype(float)
    d["se"] = (d["pred"] - d["total_points"]) ** 2
    d["ae"] = (d["pred"] - d["total_points"]).abs()
    rows = d.groupby(["season", "GW"]).agg(se=("se", "sum"), ae=("ae", "sum"), rows=("se", "size")).reset_index()
    rows["GW"] = rows["GW"].astype(int)
    pg = d.groupby(["season", "GW", "name"], as_index=False)[["pred", "total_points"]].sum()
    out = []
    for (s, gw), g in pg.groupby(["season", "GW"]):
        r = g.sort_values("pred", ascending=False, kind="mergesort")
        out.append({"season": s, "GW": int(gw), "players": len(g),
                    "top20": float(r["total_points"].head(20).mean()),
                    "top50": float(r["total_points"].head(50).mean()),
                    "capt": float(r["total_points"].iloc[0]),
                    "spear": float(g["pred"].corr(g["total_points"], method="spearman"))})
    return pd.DataFrame(out).merge(rows, on=["season", "GW"])


def summarize(t: pd.DataFrame) -> dict:
    return {"rmse": float(np.sqrt(t["se"].sum() / t["rows"].sum())), "mae": float(t["ae"].sum() / t["rows"].sum()),
            "top20": float(t["top20"].mean()), "top50": float(t["top50"].mean()), "capt": float(t["capt"].mean()),
            "spear": float(t["spear"].mean()), "gws": int(len(t))}


def position_bias(test, pred) -> dict:
    d = test[["position", "total_points"]].copy()
    d["err"] = pred.astype(float) - d["total_points"]
    return {p: round(float(g["err"].mean()), 4) for p, g in d.groupby("position")}


def run_variant(feats: pd.DataFrame, name: str, spec: dict) -> dict:
    import xgboost as xgb

    cols = spec.get("features", BASE)
    params = spec.get("params", P)
    present = [s for s in SEASONS if s in set(feats["season"])]
    res = {"variant": name, "features": len(cols), "folds": {}}
    for test_season in TEST_SEASONS:
        if test_season not in present:
            continue
        pool = [s for s in present if s < test_season]
        if spec.get("max_seasons"):
            pool = pool[-spec["max_seasons"]:]
        valid_s, train_s = pool[-1], pool[:-1]
        tr = feats[feats["season"].isin(train_s)]
        va = feats[feats["season"] == valid_s]
        te = feats[feats["season"] == test_season]
        both = pd.concat([tr, va])
        preds = {"es": [], "refit": []}
        fold = {"train": train_s, "valid": valid_s, "rows": [len(tr), len(va), len(te)], "seeds": {}}
        for seed in SEEDS:
            t0 = time.time()
            p = {**params, "random_state": seed}
            m = xgb.XGBRegressor(n_estimators=NMAX, early_stopping_rounds=PATIENCE_FIT, **p)
            m.fit(tr[cols], tr["total_points"], sample_weight=_weights(tr, spec.get("halflife")),
                  eval_set=[(va[cols], va["total_points"])], verbose=False)
            curve = m.evals_result()["validation_0"]["rmse"]
            best = _patience_stop(curve, PATIENCE)
            pe = m.predict(te[cols], iteration_range=(0, best + 1))
            n_refit = min(int((best + 1) * len(both) / len(tr)), 4000)
            m2 = xgb.XGBRegressor(n_estimators=n_refit, **p)
            m2.fit(both[cols], both["total_points"], sample_weight=_weights(both, spec.get("halflife")), verbose=False)
            pr = m2.predict(te[cols])
            preds["es"].append(pe)
            preds["refit"].append(pr)
            info = {"best_iter": best + 1, "trained": len(curve), "n_refit": n_refit,
                    "valid_rmse_at_best": float(curve[best]), "valid_rmse_min": float(min(curve)),
                    "argmin_valid": int(np.argmin(curve)) + 1,
                    "es": summarize(gw_table(te, pe)), "refit": summarize(gw_table(te, pr)),
                    "secs": round(time.time() - t0, 1)}
            if seed == SEEDS[0]:
                curve_metrics = {}
                for k in CURVE_POINTS:
                    if k <= len(curve):
                        curve_metrics[k] = summarize(gw_table(te, m.predict(te[cols], iteration_range=(0, k))))
                        curve_metrics[k]["valid_rmse"] = float(curve[k - 1])
                info["test_curve"] = curve_metrics
                gain = m2.get_booster().get_score(importance_type="gain")
                info["top_gain"] = sorted(gain.items(), key=lambda kv: -kv[1])[:25]
            fold["seeds"][seed] = info
            print(f"  {name} {test_season} seed {seed}: best {best + 1} refit {n_refit} "
                  f"es top20 {info['es']['top20']:.3f} refit top20 {info['refit']['top20']:.3f} "
                  f"rmse {info['refit']['rmse']:.4f} ({info['secs']}s)", flush=True)
        for proto in ("es", "refit"):
            ens = np.mean(preds[proto], axis=0)
            t = gw_table(te, ens)
            fold[proto] = {"summary": summarize(t), "per_gw": t.to_dict(orient="list"),
                           "position_bias": position_bias(te, ens)}
        res["folds"][test_season] = fold
    for season in INSEASON:
        if season in present:
            res.setdefault("inseason", {})[season] = run_inseason(feats, spec, season, cols, params, present)
    return res


INSEASON: tuple = ()
INSEASON_EVERY = 5


def run_inseason(feats, spec, season, cols, params, present) -> dict:
    """Replays the weekly retrain through `season`: every INSEASON_EVERY gameweeks the model is
    refit on all earlier seasons plus this season's finished gameweeks and predicts the next block.
    The tree count comes from early stopping on the previous season, as in train_points_v3."""
    import xgboost as xgb

    pool = [s for s in present if s < season]
    if spec.get("max_seasons"):
        pool = pool[-spec["max_seasons"]:]
    valid_s, train_s = pool[-1], pool[:-1]
    tr = feats[feats["season"].isin(train_s)]
    va = feats[feats["season"] == valid_s]
    cur = feats[feats["season"] == season]
    gws = sorted(cur["GW"].unique())
    preds = []
    for seed in SEEDS:
        p = {**params, "random_state": seed}
        m = xgb.XGBRegressor(n_estimators=NMAX, early_stopping_rounds=PATIENCE, **p)
        m.fit(tr[cols], tr["total_points"], sample_weight=_weights(tr, spec.get("halflife")),
              eval_set=[(va[cols], va["total_points"])], verbose=False)
        best = int(m.best_iteration) + 1
        out = pd.Series(np.nan, index=cur.index)
        for start in range(int(gws[0]), int(gws[-1]) + 1, INSEASON_EVERY):
            seen = feats[feats["season"].isin(pool) | ((feats["season"] == season) & (feats["GW"] < start))]
            block = cur[(cur["GW"] >= start) & (cur["GW"] < start + INSEASON_EVERY)]
            if block.empty:
                continue
            n = min(int(best * len(seen) / len(tr)), 4000)
            m2 = xgb.XGBRegressor(n_estimators=n, **p)
            m2.fit(seen[cols], seen["total_points"], sample_weight=_weights(seen, spec.get("halflife")), verbose=False)
            out.loc[block.index] = m2.predict(block[cols])
        preds.append(out.to_numpy())
        print(f"  inseason {season} seed {seed}: best {best}", flush=True)
    ens = np.mean(preds, axis=0)
    t = gw_table(cur, ens)
    per_seed = [summarize(gw_table(cur, p_)) for p_ in preds]
    return {"summary": summarize(t), "per_gw": t.to_dict(orient="list"),
            "position_bias": position_bias(cur, ens), "seeds": per_seed}


def cmd_run(args) -> None:
    global INSEASON, TEST_SEASONS
    feats = pd.read_pickle(args.feats)
    if args.inseason:
        INSEASON = tuple(args.inseason.split(","))
    if args.folds is not None:
        TEST_SEASONS = [s for s in args.folds.split(",") if s]
    out = []
    for name in args.variants.split(","):
        t0 = time.time()
        print(f"== {name}", flush=True)
        out.append(run_variant(feats, name, VARIANTS[name]))
        print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)


# ----------------------------------------------------------------------------- report

def _boot(base: pd.DataFrame, var: pd.DataFrame, n=10000, seed=0) -> dict:
    m = base.merge(var, on=["season", "GW"], suffixes=("_b", "_v"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m), size=(n, len(m)))
    out = {}
    for k in ("top20", "top50", "capt", "spear"):
        d = (m[f"{k}_v"] - m[f"{k}_b"]).to_numpy()
        bs = d[idx].mean(axis=1)
        out[k] = (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    se_b, se_v, rows = m["se_b"].to_numpy(), m["se_v"].to_numpy(), m["rows_b"].to_numpy()
    full = np.sqrt(se_v.sum() / rows.sum()) - np.sqrt(se_b.sum() / rows.sum())
    bs = np.sqrt(se_v[idx].sum(1) / rows[idx].sum(1)) - np.sqrt(se_b[idx].sum(1) / rows[idx].sum(1))
    out["rmse"] = (float(full), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return out


def _fmt(t, digits=3):
    return f"{t[0]:+.{digits}f} [{t[1]:+.{digits}f}, {t[2]:+.{digits}f}]"


def cmd_report(args) -> None:
    results = {}
    for path in args.paths:
        for pattern in glob.glob(path):
            for r in json.load(open(pattern, encoding="utf-8")):
                results[r["variant"]] = r
    lines = []
    base = results[args.ref]

    def units(r, proto):
        if proto == "inseason":
            return r.get("inseason", {})
        return {s: f[proto] for s, f in r.get("folds", {}).items()}

    for proto in ("refit", "es", "inseason"):
        if not units(base, proto):
            continue
        lines.append(f"\n## Protocol: {proto}\n")
        lines.append(f"### {args.ref}, seed-averaged\n")
        lines.append("| fold | rmse | mae | top20 | top50 | capt | spear | bias GK/DEF/MID/FWD |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s, u in units(base, proto).items():
            sm, pb = u["summary"], u["position_bias"]
            lines.append(f"| {s} | {sm['rmse']:.4f} | {sm['mae']:.4f} | {sm['top20']:.3f} | {sm['top50']:.3f} | "
                         f"{sm['capt']:.2f} | {sm['spear']:.4f} | "
                         f"{'/'.join(f'{pb.get(p_, 0):+.3f}' for p_ in pm.POSITIONS)} |")
        lines.append("\n### Variant minus reference (pooled: mean [95% CI]; per fold: Δtop20 / Δrmse×1000)\n")
        lines.append("| variant | Δrmse pooled | Δtop20 pooled | Δtop50 pooled | Δcapt pooled | per fold |")
        lines.append("|---|---|---|---|---|---|")
        for name, r in results.items():
            ru, bu = units(r, proto), units(base, proto)
            common = [s_ for s_ in ru if s_ in bu]
            if name == args.ref or not common:
                continue
            b = _boot(pd.concat([pd.DataFrame(bu[s_]["per_gw"]) for s_ in common]),
                      pd.concat([pd.DataFrame(ru[s_]["per_gw"]) for s_ in common]))
            per = []
            for s_ in common:
                bs, vs = bu[s_]["summary"], ru[s_]["summary"]
                per.append(f"{s_[2:]}: {vs['top20'] - bs['top20']:+.3f}/{1000 * (vs['rmse'] - bs['rmse']):+.1f}")
            lines.append(f"| {name} | {_fmt(b['rmse'], 4)} | {_fmt(b['top20'])} | {_fmt(b['top50'])} | "
                         f"{_fmt(b['capt'], 2)} | {'; '.join(per)} |")
    # Seed noise and curves for the base model
    lines.append("\n## Seed spread (refit top20 / rmse per seed)\n")
    for name, r in results.items():
        cells = []
        for s, f in r.get("folds", {}).items():
            cells.append(f"{s[2:]}: " + ", ".join(f"{v['refit']['top20']:.3f}/{v['refit']['rmse']:.4f}"
                                                for v in f["seeds"].values()))
        lines.append(f"- **{name}**: " + "; ".join(cells))
    lines.append("\n## Test curve of the early-stopping fit (seed 0): trees -> valid rmse | test top20 | test rmse\n")
    for name, r in results.items():
        for s, f in r.get("folds", {}).items():
            seed0 = next(iter(f["seeds"].values()))
            pts = seed0.get("test_curve", {})
            cell = "; ".join(f"{k}: {v['valid_rmse']:.4f}|{v['top20']:.3f}|{v['rmse']:.4f}" for k, v in pts.items())
            lines.append(f"- {name} {s} (best {seed0['best_iter']}, argmin {seed0['argmin_valid']}): {cell}")
    lines.append("\n## Top gain features (refit, seed 0, last fold)\n")
    for name, r in results.items():
        if not r.get("folds"):
            continue
        last = list(r["folds"].values())[-1]
        seed0 = next(iter(last["seeds"].values()))
        lines.append(f"- **{name}**: " + ", ".join(k for k, _ in seed0.get("top_gain", [])[:20]))
    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--feats", required=True)
    r.add_argument("--variants", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--inseason", default="", help="seasons to replay with in-season retraining, e.g. 2025-26")
    r.add_argument("--folds", default=None, help="test seasons for the es/refit folds (default: all)")
    rep = sub.add_parser("report")
    rep.add_argument("paths", nargs="+")
    rep.add_argument("--out")
    rep.add_argument("--ref", default="base", help="variant the others are compared against")
    args = parser.parse_args()
    {"features": cmd_features, "run": cmd_run, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
