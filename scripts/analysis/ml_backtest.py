"""
Rolling-origin backtest for the v3 points model. Too heavy for a laptop: run it in CI with the
"ML backtest" workflow (.github/workflows/ml-backtest.yml), which calls

    python -m scripts.analysis.ml_backtest features --out feats.pkl.gz
    python -m scripts.analysis.ml_backtest run --feats feats.pkl.gz --variants current --seeds 0-4 --out chunk_a.json
    python -m scripts.analysis.ml_backtest score 'chunk_*.preds.parquet' --pool --reuse --out res.json
    python -m scripts.analysis.ml_backtest report res.json --ref current

For every test season S (2023-24 .. the current season) and every variant:
  * "es":       trained on seasons < S-1, early-stopped on S-1 (patience 100, as in train_points_v3)
  * "refit":    refit on all seasons < S with the early-stopped tree count scaled by the extra
                rows, which is how the production model is built
  * "inseason": (--inseason) the weekly retrain replayed through a season, refitting every 5 GWs
Each is trained with several seeds (default three). Every seed's predictions and their average are
scored per gameweek. Variants are compared with a reference variant by paired bootstrap CIs over
gameweeks (seed-averaged predictions), and single models by a two-level bootstrap that also
resamples seeds, so the seed lottery is inside the CI. The production model is one seed.

Squad metrics score what the Team of the Week ships: per gameweek, services/team_optimizer's
best_squad picks a 15-man squad (GBP 100m, 3 per club) from the predictions (pool: predicted
points > 0, each position's 60 best plus its 15 cheapest; per-fixture predictions floored at 0, as
in serving). sq_xi is the XI's actual points after FPL auto-subs with the captain counted once,
the least noisy squad metric; sq_fpl adds the captain's points again (the vice's when the captain
didn't play); sq_pred is the predicted XI plus captain, sq_dnp the picked starters who didn't play.
The top-20 metrics are unconstrained and mostly midfielders, so position picks (the top 4 GK, 10 DEF,
10 MID and 6 FWD by prediction) and pred20/pred_capt (the predicted side, for selection bias) are
reported too.

`run` writes <out>.json, <out>.gw.parquet (every per-gameweek table, per seed and seed-averaged)
and <out>.preds.parquet (row-level predictions). `score` rescores prediction files without
retraining; --pool merges files of the same variant with different seeds (the workflow splits a
variant's seeds into jobs of 5 and pools them), and --reuse keeps tables whose predictions haven't
changed.

Variants (VARIANTS): "current" is what ml.points_model trains, "previous" is v3 before the
2026-10 features; the others add candidate features computed only here (add_extra_features) or
change training settings. Subset metrics (promoted teams in their first 6 gameweeks, players new
from the Championship in their first 6, teams within 60 days of a manager change, clubs playing
in Europe, fixtures within 4 days of a cup or European tie, goalkeepers and defenders) show effects
too local to move the overall numbers. Pick metrics beyond the overall top 20: the mean points of
the top 20 picks among players of clubs in Europe and among goalkeepers and defenders, the
clean-sheet points of those GK/DEF picks (cs20_sub_gkdef), and the share of the overall top 20 who
didn't play (dnp20).

The non-Premier-League data (data/non_pl_*.csv, ml/other_competitions.py) comes from the
transfermarkt-datasets dump, which has no 2026-27 season: compare those variants with
`--folds 2023-24,2024-25,2025-26`.

Bookmaker odds (ml/odds.py, variants odds and odds_close) come from football-data.co.uk, whose
files aren't committed: run `python -m scripts.data.fetch_odds` first (the workflow does).

Round of 2026-10-22: fixture-strength products and opponent-adjusted team ratings
(ml/fixture_strength.py, variants matchup and ratings), transfer-market ranks and price changes
(ml/market.py, variant transfers), a Tweedie objective (tweedie) and a two-stage model that
multiplies P(plays) by the expected points if he plays (hurdle).
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ml import fixture_strength as fs  # noqa: E402
from ml import market as mk  # noqa: E402
from ml import odds as od  # noqa: E402
from ml import other_competitions as oc  # noqa: E402
from ml import points_model as pm  # noqa: E402
from utils.text import normalize_name  # noqa: E402


def _team_optimizer():
    """services/team_optimizer.py on its own: services/__init__ imports the Neo4j driver."""
    if "services.team_optimizer" in sys.modules:
        return sys.modules["services.team_optimizer"]
    spec = importlib.util.spec_from_file_location("team_optimizer", os.path.join(ROOT, "services", "team_optimizer.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


opt = _team_optimizer()

SEASONS =["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
TEST_SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]
SEEDS = (0, 1, 2)
NMAX = 3000
PATIENCE = 100          # what train_points_v3 uses
PATIENCE_FIT = 400      # train further so the curve shows what happens after the stop
CURVE_POINTS = (50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000)

# ----------------------------------------------------------------------------- candidate features

P90_STATS = ["total_points", "expected_goals", "expected_assists", "expected_goal_involvements",
             "threat", "creativity", "bps", "bonus", "defensive_contribution", "saves",
             "expected_goals_conceded", "goals_scored", "assists"]
P90_WINDOWS = (10, 30)
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}
MANAGER_CSV = os.path.join(ROOT, "data", "manager_changes.csv")
MGR_CAP_DAYS = 365
EFL_DIR = os.path.join(ROOT, "EFL_Data_All_Seasons")
EFL_TO_FPL = {"Hull": "Hull City", "Coventry": "Coventry City"}   # graph names since 2026-27
EFL_TEAM_MINUTES = 46 * 90


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


def _manager_days(df: pd.DataFrame) -> dict:
    """Days since the club's last manager change (vacancy or appointment, caretakers included),
    counting only changes dated before the gameweek's first kickoff; capped at MGR_CAP_DAYS."""
    ch = pd.read_csv(MANAGER_CSV)
    ev = pd.concat([ch[["team", "vacancy"]].rename(columns={"vacancy": "date"}),
                    ch[["team", "appointed"]].rename(columns={"appointed": "date"})]).dropna()
    events = {t: np.sort(pd.to_datetime(g["date"]).to_numpy().astype("datetime64[D]")) for t, g in ev.groupby("team")}
    start = df.groupby(["season", "GW"])["kickoff_time"].transform("min")
    start = start.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[D]")
    out = {}
    for col, team_col in (("team_mgr_days", "team"), ("opp_mgr_days", "opponent")):
        vals = np.full(len(df), float(MGR_CAP_DAYS))
        teams = df[team_col].to_numpy()
        for team, dates in events.items():
            pos = np.flatnonzero(teams == team)
            if not len(pos):
                continue
            i = np.searchsorted(dates, start[pos], side="left") - 1
            days = (start[pos] - dates[np.maximum(i, 0)]).astype("timedelta64[D]").astype(float)
            vals[pos] = np.where(i >= 0, np.minimum(days, MGR_CAP_DAYS), MGR_CAP_DAYS)
        out[col] = vals
    return out


def _team_rotation(df: pd.DataFrame) -> np.ndarray:
    """Mean number of changes to the starting XI over the team's previous 6 matches."""
    starters = df[~df["is_upcoming"] & (df["started"] == 1)]
    lineups = (starters.groupby(["team", "season", "fixture"])
               .agg(kickoff_time=("kickoff_time", "first"), GW=("GW", "first"), names=("name", frozenset))
               .reset_index().sort_values(["team", "kickoff_time"]).reset_index(drop=True))
    prev = lineups.groupby("team")["names"].shift(1)
    lineups["changes"] = [float(len(a - b)) if isinstance(b, frozenset) else np.nan
                          for a, b in zip(lineups["names"], prev)]
    lineups = lineups.drop(columns="names")
    lineups["rot"] = pm._rolling_before_gameweek(lineups, "team", ["changes"], (6,))["changes_r6"]
    keyed = lineups.set_index(["season", "fixture", "team"])["rot"]
    return keyed.reindex(pd.MultiIndex.from_frame(df[["season", "fixture", "team"]])).to_numpy()


ABSENCE_MIN_RUN = 3       # consecutive 0-minute matches before a run counts as an absence
ABSENCE_REGULAR = 2       # ... if he had started this many of his last 3 counted matches
ABSENCE_CAP = 10          # since_return for "no absence in the last 10 matches"
FORM_A = [f"{c}_r{w}" for w in pm.WINDOWS for c in pm.FORM_STATS] + [f"{c}_r1" for c in pm.LAST_MATCH_STATS]


def absence_form(df: pd.DataFrame, min_run: float = ABSENCE_MIN_RUN) -> pd.DataFrame:
    """The rolling form features (FORM_A, renamed *_a) with absences left out, plus absent_run
    (matches of the absence the player is in, 0 if none) and since_return (matches since the
    last absence ended, capped at ABSENCE_CAP).

    History has no pre-deadline FPL status, so an absence (injury, suspension, AFCON) is a run of
    `min_run`+ consecutive 0-minute matches that began while the player had started
    ABSENCE_REGULAR of his last 3 counted matches. Decided causally: at each match only earlier
    matches are known, so a run is left out only once it has reached `min_run` matches; shorter
    runs (rotation, a one-off benching) stay in the form as zeros. Like pm._rolling_before_gameweek,
    the values are frozen at each gameweek's first fixture; with min_run=inf they equal it.
    """
    frame = df.sort_values(["name", "kickoff_time"], kind="mergesort")
    stats = list(pm.FORM_STATS)
    x = frame[stats].to_numpy(dtype=float)
    minutes = frame["minutes"].to_numpy(dtype=float)
    started = frame["started"].to_numpy(dtype=float)
    last_idx = [stats.index(c) for c in pm.LAST_MATCH_STATS]
    n, wmax = len(frame), max(pm.WINDOWS)
    out = {w: np.full((n, len(stats)), np.nan) for w in pm.WINDOWS}
    r1 = np.full((n, len(last_idx)), np.nan)
    absent, since = np.zeros(n), np.full(n, float(ABSENCE_CAP))
    names = frame["name"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, names[1:] != names[:-1], True])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # nanmean of an all-NaN stat (no xG yet)
        for a, b in zip(bounds[:-1], bounds[1:]):
            kept, pending = [], []          # counted matches; trailing zeros not (yet) an absence
            in_absence, back, regular = 0, ABSENCE_CAP, False
            for j in range(a, b):
                eff = (kept[-wmax:] + pending)[-wmax:]
                if eff:
                    for w in pm.WINDOWS:
                        out[w][j] = np.nanmean(x[eff[-w:]], axis=0)
                    r1[j] = x[eff[-1], last_idx]
                absent[j], since[j] = in_absence, back
                m = minutes[j]
                if np.isnan(m):             # an upcoming fixture: no outcome to add
                    continue
                back = min(back + 1, ABSENCE_CAP)
                if m == 0:
                    if in_absence:
                        in_absence += 1
                        continue
                    if not pending:
                        regular = np.nansum(started[kept[-3:]]) >= ABSENCE_REGULAR if kept else False
                    pending.append(j)
                    if regular and len(pending) >= min_run:
                        in_absence, pending = len(pending), []
                else:
                    if in_absence:
                        in_absence, back = 0, 1
                    kept += pending + [j]
                    pending = []
    res = pd.DataFrame(index=frame.index)
    for w in pm.WINDOWS:
        res[[f"{c}_r{w}_a" for c in stats]] = out[w]
    res[[f"{c}_r1_a" for c in pm.LAST_MATCH_STATS]] = r1
    res["absent_run"], res["since_return"] = absent, since
    first_in_gw = frame.groupby(["name", "season", "GW"], sort=False).cumcount() == 0
    res = res.where(first_in_gw)
    res[["_k", "_s", "_g"]] = frame[["name", "season", "GW"]].to_numpy()
    res = res.groupby(["_k", "_s", "_g"], sort=False).transform("first")
    return res.reindex(df.index)


def _read_fbref(path: str) -> pd.DataFrame:
    """FBref table export; some files have a column-group row above the header."""
    with open(path, encoding="utf-8") as f:
        skip = next(i for i, line in enumerate(f) if line.startswith("Squad"))
    return pd.read_csv(path, sep="\t", skiprows=skip)


def _efl_team_table() -> pd.DataFrame:
    """Previous Championship season per promoted club, keyed by the Premier League season."""
    mapping = pd.read_csv(os.path.join(EFL_DIR, "squad_name_mapping.csv"))
    name = dict(zip(mapping["EFL_Name"], mapping["Normalized_Name"]))
    rows = []
    for d in sorted(glob.glob(os.path.join(EFL_DIR, "EFL-*"))):
        yrs = os.path.basename(d)[4:]
        std = _read_fbref(os.path.join(d, f"{yrs}_Standard_Teams.txt"))
        gk = _read_fbref(os.path.join(d, f"{yrs}_Goalkeeping_Teams.txt"))
        t = std[["Squad", "MP", "Gls"]].merge(gk[["Squad", "GA", "W", "D"]], on="Squad")
        nxt = int(yrs[:4]) + 1
        t["season"] = f"{nxt}-{str(nxt + 1)[2:]}"
        t["team"] = [EFL_TO_FPL.get(name.get(s, s), name.get(s, s)) for s in t["Squad"]]
        t["efl_gf_pg"] = t["Gls"] / t["MP"]
        t["efl_ga_pg"] = t["GA"] / t["MP"]
        t["efl_ppg"] = (3 * t["W"] + t["D"]) / t["MP"]
        rows.append(t[["season", "team", "efl_gf_pg", "efl_ga_pg", "efl_ppg"]])
    return pd.concat(rows, ignore_index=True)


def _efl_player_table() -> pd.DataFrame:
    """Previous Championship season per player (minutes-weighted over clubs), keyed by PL season."""
    e = pd.read_csv(os.path.join(EFL_DIR, "efl_combined_all_cleaned.csv"), low_memory=False)
    e["min"] = pd.to_numeric(e["Min"].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    e["key"] = e["Player"].map(normalize_name)
    nxt = e["Season"].str[:4].astype(int) + 1
    e["season"] = nxt.astype(str) + "-" + (nxt + 1).astype(str).str[2:]
    stats = {"efl_g90": "Gls", "efl_a90": "Ast", "efl_sh90": "Sh/90", "efl_ppm": "PPM", "efl_onga90": "onGA"}
    keys = [e["season"], e["key"]]
    out = pd.DataFrame({"efl_min_share": e["min"].groupby(keys).sum() / EFL_TEAM_MINUTES})
    for feat, src in stats.items():
        v = pd.to_numeric(e[src], errors="coerce")
        w = e["min"].where(v.notna(), 0.0)
        num = (v.fillna(0) * w).groupby(keys).sum()
        den = w.groupby(keys).sum()
        out[feat] = (num / den).where(den > 0)
    return out.reset_index()


def add_extra_features(df: pd.DataFrame, transfers: pd.DataFrame = None) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    known = df["minutes"].notna()

    # Matches outside the Premier League: congestion, minutes in cup ties, still in which cups
    matches, apps = oc.load_other_competitions()
    df = df.join(oc.other_competition_features(df, matches, apps))

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

    # A longer 10-match window
    df = df.join(pm._rolling_before_gameweek(df, "name", ["minutes", "started", "total_points", "bps", "ict_index"], (10,)))

    # FPL's defensive-contribution points (2025-26 rule): 2 points for 10+ CBIT (DEF) or 12+ CBIRT
    # (MID/FWD); GKs are not eligible. NaN where the action count isn't recorded (before 2025-26).
    dc = df["defensive_contribution"]
    dcp = (2.0 * (dc >= df["position"].map(DC_THRESHOLD))).where(dc.notna())
    df["dc_points"] = dcp.where(df["position"] != "GK", 0.0).where(dc.notna())
    df = df.join(pm._rolling_before_gameweek(df, "name", ["dc_points"], (3, 6, 10)))

    # Manager changes (data/manager_changes.csv, from Wikipedia) and lineup rotation
    for col, vals in _manager_days(df).items():
        df[col] = vals
    df["team_rotation_r6"] = _team_rotation(df)

    # Form that leaves out injuries, suspensions and tournaments (runs of 0-minute matches)
    df = df.join(absence_form(df))

    # Previous Championship season: promoted clubs (and their opponents), players new from it
    efl = _efl_team_table()
    stats = [c for c in efl.columns if c.startswith("efl_")]
    df = df.merge(efl.rename(columns={c: f"efl_team_{c[4:]}" for c in stats}), on=["season", "team"], how="left")
    df = df.merge(efl.rename(columns={"team": "opponent", **{c: f"opp_{c}" for c in stats}}),
                  on=["season", "opponent"], how="left")
    df["key"] = df["name"].map(normalize_name)
    df = df.merge(_efl_player_table(), on=["season", "key"], how="left").drop(columns="key")

    df["sub_promoted_early"] = (df["efl_team_gf_pg"].notna() & (df["GW"] <= 6)).astype(float)
    df["sub_efl_player_early"] = (df["efl_g90"].notna() & (df["GW"] <= 6)).astype(float)
    df["sub_new_manager"] = (df["team_mgr_days"] < 60).astype(float)
    df["sub_after_absence"] = (df["since_return"] <= 3).astype(float)

    # Bookmaker odds: pre-closing prices known at the deadline, and closing prices (an upper bound)
    df = df.join(od.odds_features(df, od.load_odds()))
    df["sub_gkdef"] = df["position"].isin(["GK", "DEF"]).astype(float)

    # Fixture-strength products, opponent-adjusted team ratings and transfer-market signals (2026-10-22)
    df = df.join(fs.matchup_features(df)).join(fs.ratings_features(df))
    if transfers is not None:
        df = df.join(mk.transfer_features(df, transfers))
    else:
        df[mk.FEATURES] = np.nan
    return df


CURRENT = pm.feature_columns()
NEW_2026_10 = [f"{c}_r1" for c in pm.LAST_MATCH_STATS] + [f"{p}_{c}20" for p in ("team", "opp") for c in pm.TEAM_STATS]
PREVIOUS = [c for c in CURRENT if c not in NEW_2026_10]
G_P90 = [f"{s}_p90_w{w}" for w in P90_WINDOWS for s in P90_STATS] + [f"minutes_sum_w{w}" for w in P90_WINDOWS]
G_R10 = ["minutes_r10", "started_r10", "total_points_r10", "bps_r10", "ict_index_r10"]
G_DC = ["dc_points_r3", "dc_points_r6", "dc_points_r10"]
G_MGR = ["team_mgr_days", "opp_mgr_days"]
G_ROT = ["team_rotation_r6"]
G_EFL_TEAM = ["efl_team_gf_pg", "efl_team_ga_pg", "efl_team_ppg", "opp_efl_gf_pg", "opp_efl_ga_pg", "opp_efl_ppg"]
G_EFL_PLAYER = ["efl_min_share", "efl_g90", "efl_a90", "efl_sh90", "efl_ppm", "efl_onga90"]
G_OC_SCHED = oc.SCHEDULE_FEATURES
G_OC_PLAYER = oc.PLAYER_FEATURES
G_OC_STATUS = oc.STATUS_FEATURES
G_ABS_FORM = [f"{c}_a" for c in FORM_A]
G_ABS_FLAGS = ["absent_run", "since_return"]
G_ODDS = od.FEATURES
G_ODDS_CLOSE = od.CLOSING_FEATURES
G_MATCHUP = fs.MATCHUP_FEATURES
G_RATINGS = fs.RATING_FEATURES
G_TRANSFERS = mk.FEATURES
SUBSETS = ["sub_promoted_early", "sub_efl_player_early", "sub_new_manager"] + oc.SUBSETS + ["sub_after_absence",
                                                                                            "sub_gkdef"]
PICK_SUBSETS = ["sub_euro_club", "sub_gkdef"]   # top-20 picks among these players only
CS_PICKS = "sub_gkdef"                          # ... and the clean-sheet points of these picks
PICK_METRICS = ["dnp20"] + [f"top20_{c}" for c in PICK_SUBSETS] + [f"cs20_{CS_PICKS}"]
POS_TOPK = {"GK": 4, "DEF": 10, "MID": 10, "FWD": 6}     # position picks: top k by prediction
POS_METRICS = [f"top_{p}" for p in POS_TOPK]             # ... their mean actual points
PRED_METRICS = ["pred20", "pred_capt"] + [f"pred_{p}" for p in POS_TOPK]   # ... and predicted
SQ_METRICS = ["sq_xi", "sq_fpl", "sq_pred", "sq_dnp"]
SQ_TOP_PER_POS = 60       # squad pool: each position's best predictions ...
SQ_CHEAPEST = 15          # ... plus its cheapest players, the same for every variant
SQ_TIME_LIMIT = 5.0       # seconds per squad solve
SQUADS = True             # run/score --no-squads skips the squad solves
ENS = -1                  # `seed` of a seed-averaged table in the .gw.parquet files
EXTRA =G_P90 + G_R10 + G_DC + G_MGR + G_ROT + G_EFL_TEAM + G_EFL_PLAYER + G_OC_SCHED + G_OC_PLAYER + G_OC_STATUS
EXTRA += G_ABS_FORM + G_ABS_FLAGS + G_ODDS + G_ODDS_CLOSE + G_MATCHUP + G_RATINGS + G_TRANSFERS
KEEP = ["season", "GW", "name", "position", "team", "fixture", "kickoff_time", "total_points", "minutes", "xP",
        "clean_sheets"]


def cmd_features(args) -> None:
    t0 = time.time()
    live_csv, live_season = None, None
    meta_path = os.path.join("data", "live", "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        live_csv, live_season = os.path.join("data", "live", meta["files"]["season"]), meta["season"]
    history = pm.load_history(live_csv=live_csv, live_season=live_season)
    feats = pm.build_features(history)
    print(f"model features: {len(feats)} rows in {time.time() - t0:.0f}s", flush=True)
    n = len(feats)
    feats = add_extra_features(feats, mk.load_transfers(live_csv=live_csv, live_season=live_season))
    assert len(feats) == n, "a merge duplicated rows"
    feats = feats[feats["total_points"].notna()]
    feats = feats[list(dict.fromkeys(KEEP + CURRENT + EXTRA + SUBSETS))].copy()
    for c in feats.columns:
        if feats[c].dtype == "float64":
            feats[c] = feats[c].astype("float32")
    feats.to_pickle(args.out)
    cover = {c: round(float(feats[c].notna().mean()), 3) for c in G_MGR + G_ROT + G_EFL_TEAM[:1] + G_EFL_PLAYER[:1]
             + ["oc_minutes_last", "ac_ga_p90_r8", "mu_xgf", "mu_pxg", "rt_xgf", "tr_bal_rank", "price_change_3"]}
    odds_cover = feats.groupby("season")[["odds_win", "odds_win_c"]].agg(lambda s: round(float(s.notna().mean()), 3))
    print(f"saved {args.out}: {feats.shape}, seasons {feats.groupby('season').size().to_dict()}, "
          f"subset rows {feats[SUBSETS].sum().astype(int).to_dict()}, coverage {cover}, "
          f"odds coverage (pre-closing, closing) {odds_cover.to_dict(orient='index')} "
          f"in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- variants

P = dict(pm.XGB_PARAMS)
VARIANTS = {
    "previous": {"features": PREVIOUS},
    "current": {"features": CURRENT},
    # candidates, relative to current
    "p90": {"features": CURRENT + G_P90},
    "r10": {"features": CURRENT + G_R10},
    "dc": {"features": CURRENT + G_DC},
    "nocareer": {"features": [c for c in CURRENT if c != "career_matches_before"]},
    "d6": {"params": {**P, "max_depth": 6}},
    "mcw200": {"params": {**P, "min_child_weight": 200}},
    "hl2": {"halflife": 2.0},
    "mgr": {"features": CURRENT + G_MGR},
    "rot": {"features": CURRENT + G_ROT},
    "mgr_rot": {"features": CURRENT + G_MGR + G_ROT},
    "efl_team": {"features": CURRENT + G_EFL_TEAM},
    "efl_player": {"features": CURRENT + G_EFL_PLAYER},
    "efl": {"features": CURRENT + G_EFL_TEAM + G_EFL_PLAYER},
    # matches outside the Premier League (2026-10; transfermarkt-datasets, so no EFL Cup and no 2026-27)
    "oc_sched": {"features": CURRENT + G_OC_SCHED},
    "oc_player": {"features": CURRENT + G_OC_PLAYER},
    "oc_status": {"features": CURRENT + G_OC_STATUS},
    "oc": {"features": CURRENT + G_OC_SCHED + G_OC_PLAYER + G_OC_STATUS},
    # form that leaves out absences (runs of 3+ 0-minute matches after regular starts), 2026-10
    "abs_flags": {"features": CURRENT + G_ABS_FLAGS},
    "abs_skip": {"features": [f"{c}_a" if c in FORM_A else c for c in CURRENT]},
    "abs_skip_flags": {"features": [f"{c}_a" if c in FORM_A else c for c in CURRENT] + G_ABS_FLAGS},
    # bookmaker odds (football-data.co.uk), 2026-10: prices known at the deadline, and closing
    # prices (after lineups) as an upper bound that can't be served
    "odds": {"features": CURRENT + G_ODDS},
    "odds_close": {"features": CURRENT + G_ODDS_CLOSE},
    # 2026-10-22: fixture-strength products of the model's own team stats, opponent-adjusted
    # team ratings, transfer-market ranks and price changes, a Tweedie objective (log link,
    # zero-inflated target; trained on points floored at 0, early-stopped on RMSE) and a
    # two-stage model, P(minutes > 0) x E[points | minutes > 0]
    "matchup": {"features": CURRENT + G_MATCHUP},
    "ratings": {"features": CURRENT + G_RATINGS},
    "transfers": {"features": CURRENT + G_TRANSFERS},
    "tweedie": {"params": {**P, "objective": "reg:tweedie", "tweedie_variance_power": 1.3, "eval_metric": "rmse"},
                "clip_target": 0.0},
    "hurdle": {"model": "hurdle"},
    "ratings_transfers": {"features": CURRENT + G_RATINGS + G_TRANSFERS},
    # which part of the transfer signal: the ranks (serving sees counts before the deadline) or the
    # price changes (known exactly at the deadline)
    "tr_ranks": {"features": CURRENT + G_TRANSFERS[:3]},
    "tr_price": {"features": CURRENT + G_TRANSFERS[3:]},
}


def _target(frame: pd.DataFrame, spec: dict) -> pd.Series:
    y = frame["total_points"]
    return y.clip(lower=spec["clip_target"]) if "clip_target" in spec else y


def _hurdle_es(train, valid, cols, p):
    """Early-stopped hurdle models, P(plays) and E[points | plays], with their tree counts."""
    import xgboost as xgb
    played, vplayed = train["minutes"] > 0, valid["minutes"] > 0
    cls = xgb.XGBClassifier(n_estimators=NMAX, early_stopping_rounds=PATIENCE,
                            **{**p, "objective": "binary:logistic", "eval_metric": "logloss"})
    cls.fit(train[cols], played.astype(int), eval_set=[(valid[cols], vplayed.astype(int))], verbose=False)
    reg = xgb.XGBRegressor(n_estimators=NMAX, early_stopping_rounds=PATIENCE, **p)
    reg.fit(train.loc[played, cols], train.loc[played, "total_points"],
            eval_set=[(valid.loc[vplayed, cols], valid.loc[vplayed, "total_points"])], verbose=False)
    return cls, reg, cls.best_iteration + 1, reg.best_iteration + 1


def _hurdle_fit(frame, cols, p, n_cls, n_reg):
    import xgboost as xgb
    played = frame["minutes"] > 0
    cls = xgb.XGBClassifier(n_estimators=n_cls, **{**p, "objective": "binary:logistic"})
    cls.fit(frame[cols], played.astype(int), verbose=False)
    reg = xgb.XGBRegressor(n_estimators=n_reg, **p)
    reg.fit(frame.loc[played, cols], frame.loc[played, "total_points"], verbose=False)
    return cls, reg


def _hurdle_predict(cls, reg, frame, cols) -> np.ndarray:
    return cls.predict_proba(frame[cols])[:, 1] * reg.predict(frame[cols])


def _hurdle_fold(tr, va, te, both, cols, p) -> tuple:
    """es and refit predictions of the hurdle model for one fold and seed, plus run info."""
    cls, reg, n_cls, n_reg = _hurdle_es(tr, va, cols, p)
    pe = _hurdle_predict(cls, reg, te, cols)
    valid_rmse = float(np.sqrt(np.mean((_hurdle_predict(cls, reg, va, cols) - va["total_points"]) ** 2)))
    scale = len(both) / len(tr)
    r_cls, r_reg = min(int(n_cls * scale), 4000), min(int(n_reg * scale), 4000)
    cls2, reg2 = _hurdle_fit(both, cols, p, r_cls, r_reg)
    pr = _hurdle_predict(cls2, reg2, te, cols)
    gain = reg2.get_booster().get_score(importance_type="gain")
    info = {"best_iter": n_reg, "best_iter_cls": n_cls, "trained": n_reg, "n_refit": r_reg, "n_refit_cls": r_cls,
            "valid_rmse_at_best": valid_rmse, "valid_rmse_min": valid_rmse, "argmin_valid": n_reg,
            "top_gain": sorted(gain.items(), key=lambda kv: -kv[1])[:25]}
    return pe, pr, info


def _xgb_fold(tr, va, te, both, cols, p, spec, curve: bool) -> tuple:
    """es and refit predictions of one fold and seed, plus run info; with `curve`, also the test
    metrics along the early-stopping fit and the refit's top features."""
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=NMAX, early_stopping_rounds=PATIENCE_FIT, **p)
    m.fit(tr[cols], _target(tr, spec), sample_weight=_weights(tr, spec.get("halflife")),
          eval_set=[(va[cols], va["total_points"])], verbose=False)
    valid = m.evals_result()["validation_0"]["rmse"]
    best = _patience_stop(valid, PATIENCE)
    pe = m.predict(te[cols], iteration_range=(0, best + 1))
    n_refit = min(int((best + 1) * len(both) / len(tr)), 4000)
    m2 = xgb.XGBRegressor(n_estimators=n_refit, **p)
    m2.fit(both[cols], _target(both, spec), sample_weight=_weights(both, spec.get("halflife")), verbose=False)
    pr = m2.predict(te[cols])
    info = {"best_iter": best + 1, "trained": len(valid), "n_refit": n_refit,
            "valid_rmse_at_best": float(valid[best]), "valid_rmse_min": float(min(valid)),
            "argmin_valid": int(np.argmin(valid)) + 1}
    if curve:
        curve_metrics = {}
        for k in CURVE_POINTS:
            if k <= len(valid):
                t = gw_table(te, m.predict(te[cols], iteration_range=(0, k)), squads=False)
                curve_metrics[k] = {**summarize(t), "valid_rmse": float(valid[k - 1])}
        info["test_curve"] = curve_metrics
        gain = m2.get_booster().get_score(importance_type="gain")
        info["top_gain"] = sorted(gain.items(), key=lambda kv: -kv[1])[:25]
    return pe, pr, info


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


def squad_pool(g: pd.DataFrame) -> list:
    """Candidates for one gameweek's squad from per-player rows (name, position, team, price in
    millions, pred_sq): predicted points > 0 and a price; each position's SQ_TOP_PER_POS best
    predictions plus its SQ_CHEAPEST cheapest players, which keeps the MILP small."""
    g = g[(g["pred_sq"] > 0) & g["price"].notna() & g["position"].isin(opt.POSITIONS)]
    keep = []
    for _, p in g.groupby("position", sort=False):
        best = p.sort_values(["pred_sq", "name"], ascending=[False, True], kind="mergesort").head(SQ_TOP_PER_POS)
        cheap = p.sort_values(["price", "pred_sq", "name"], ascending=[True, False, True],
                              kind="mergesort").head(SQ_CHEAPEST)
        keep.append(pd.concat([best, cheap]).drop_duplicates("name"))
    if not keep:
        return []
    return [opt.Candidate(id=str(r.name), position=str(r.position), team=str(r.team), points=float(r.pred_sq),
                          cost=int(round(float(r.price) * 10)))
            for r in pd.concat(keep).itertuples(index=False)]


def score_squad(sel, actual: dict) -> dict:
    """FPL score of a picked squad (an opt.Selection); `actual` maps each player to his
    (points, minutes) that gameweek. sq_xi: the XI's points after automatic substitutions, the
    captain counted once; sq_fpl: plus the captain's points again, or the vice-captain's when the
    captain didn't play; sq_pred: the predicted XI plus captain; sq_dnp: picked starters who didn't
    play."""
    def player(c):
        points, minutes = actual[c.id]
        return {"name": c.id, "position": c.position, "points": float(points), "played": bool(minutes > 0)}

    starters = [player(c) for c in sel.starters]
    xi, _, _ = opt.apply_auto_subs(starters, [player(c) for c in sel.bench])
    xi_points = sum(p["points"] for p in xi)
    armband = 0.0
    for c in (sel.captain, sel.vice_captain):
        if c is not None and actual[c.id][1] > 0:
            armband = float(actual[c.id][0])
            break
    return {"sq_xi": xi_points, "sq_fpl": xi_points + armband,
            "sq_pred": float(sum(c.points for c in sel.starters) + sel.captain.points),
            "sq_dnp": float(sum(not p["played"] for p in starters))}


def squad_metrics(g: pd.DataFrame) -> dict:
    """The squad best_squad picks from one gameweek's per-player rows, scored on actual points;
    NaN when no valid squad fits. sq_secs is the solve time."""
    t0 = time.perf_counter()
    try:
        sel = opt.best_squad(squad_pool(g), time_limit=SQ_TIME_LIMIT)
    except opt.InfeasibleError:
        return {**{k: np.nan for k in SQ_METRICS}, "sq_secs": time.perf_counter() - t0}
    actual = dict(zip(g["name"], zip(g["total_points"], g["minutes"])))
    return {**score_squad(sel, actual), "sq_secs": time.perf_counter() - t0}


def gw_table(test: pd.DataFrame, pred: np.ndarray, squads: bool = True) -> pd.DataFrame:
    """One row per gameweek: error sums (overall, per subset, per position), the unconstrained
    picks (top 20/50, captain), position picks and, with `squads`, the optimiser's squad."""
    subsets = [c for c in SUBSETS if c in test.columns]
    extra = ["clean_sheets"] if "clean_sheets" in test.columns else []
    info = [c for c in ("position", "team", "price") if c in test.columns]
    squads = squads and len(info) == 3
    d = test[["season", "GW", "name", "total_points", "minutes"] + info + extra + subsets].copy()
    d["pred"] = np.asarray(pred, dtype=float)
    d["err"] = d["pred"] - d["total_points"]
    d["se"] = d["err"] ** 2
    d["ae"] = d["err"].abs()
    agg = {"se": ("se", "sum"), "ae": ("ae", "sum"), "rows": ("se", "size")}
    for c in subsets:
        mask = d[c] > 0
        d[f"se_{c}"], d[f"err_{c}"], d[f"n_{c}"] = d["se"].where(mask, 0.0), d["err"].where(mask, 0.0), mask.astype(float)
        agg.update({f"se_{c}": (f"se_{c}", "sum"), f"err_{c}": (f"err_{c}", "sum"), f"n_{c}": (f"n_{c}", "sum")})
    if "position" in info:
        for p in opt.POSITIONS:
            mask = d["position"] == p
            d[f"perr_{p}"], d[f"pn_{p}"] = d["err"].where(mask, 0.0), mask.astype(float)
            agg.update({f"perr_{p}": (f"perr_{p}", "sum"), f"pn_{p}": (f"pn_{p}", "sum")})
    rows = d.groupby(["season", "GW"]).agg(**agg).reset_index()
    rows["GW"] = rows["GW"].astype(int)
    picks = [c for c in PICK_SUBSETS if c in subsets]
    d["pred_sq"] = d["pred"].clip(lower=0)   # serving floors each fixture at 0, then sums a double gameweek
    pg = d.groupby(["season", "GW", "name"], as_index=False).agg(
        pred=("pred", "sum"), pred_sq=("pred_sq", "sum"), total_points=("total_points", "sum"),
        minutes=("minutes", "sum"), **{c: (c, "first") for c in info},
        **{c: (c, "max") for c in picks}, **{c: (c, "sum") for c in extra})
    out = []
    for (s, gw), g in pg.groupby(["season", "GW"]):
        r = g.sort_values("pred", ascending=False, kind="mergesort")
        capt_row = 0 if r["minutes"].iloc[0] > 0 or len(r) < 2 else 1   # FPL: the vice if the captain didn't play
        row = {"season": s, "GW": int(gw), "players": len(g),
               "top20": float(r["total_points"].head(20).mean()),
               "top50": float(r["total_points"].head(50).mean()),
               "capt": float(r["total_points"].iloc[0]),
               "capt_fpl": float(r["total_points"].iloc[capt_row]),
               "pred20": float(r["pred"].head(20).mean()),
               "pred_capt": float(r["pred"].iloc[0]),
               "spear": float(g["pred"].corr(g["total_points"], method="spearman")),
               "dnp20": float((r["minutes"].head(20) == 0).mean())}
        for c in picks:
            sub = r[r[c] > 0]
            row[f"top20_{c}"] = float(sub["total_points"].head(20).mean()) if len(sub) >= 40 else np.nan
            if c == CS_PICKS and extra:   # GK/DEF score 4 points per clean sheet
                row[f"cs20_{c}"] = float(4 * sub["clean_sheets"].head(20).mean()) if len(sub) >= 40 else np.nan
        if "position" in info:
            for p, k in POS_TOPK.items():
                sub = r[r["position"] == p].head(k)
                full = len(sub) == k
                row[f"top_{p}"] = float(sub["total_points"].mean()) if full else np.nan
                row[f"pred_{p}"] = float(sub["pred"].mean()) if full else np.nan
        if squads:
            row.update(squad_metrics(g))
        out.append(row)
    return pd.DataFrame(out).merge(rows, on=["season", "GW"])


def summarize(t: pd.DataFrame) -> dict:
    out = {"rmse": float(np.sqrt(t["se"].sum() / t["rows"].sum())), "mae": float(t["ae"].sum() / t["rows"].sum()),
           "top20": float(t["top20"].mean()), "top50": float(t["top50"].mean()), "capt": float(t["capt"].mean()),
           "spear": float(t["spear"].mean()), "gws": int(len(t))}
    for k in PICK_METRICS + ["capt_fpl"] + PRED_METRICS + POS_METRICS + SQ_METRICS:
        if k in t and t[k].notna().any():
            out[k] = float(t[k].mean())
    for p in opt.POSITIONS:
        if f"pn_{p}" in t and t[f"pn_{p}"].sum() > 0:
            out[f"bias_{p}"] = float(t[f"perr_{p}"].sum() / t[f"pn_{p}"].sum())
    if "sq_secs" in t:
        out["sq_solves"], out["sq_failed"] = int(len(t)), int(t["sq_xi"].isna().sum())
        out["sq_secs_mean"], out["sq_secs_max"] = float(t["sq_secs"].mean()), float(t["sq_secs"].max())
    for c in SUBSETS:
        if f"n_{c}" in t and t[f"n_{c}"].sum() > 0:
            n = t[f"n_{c}"].sum()
            out[c] = {"rows": int(n), "rmse": float(np.sqrt(t[f"se_{c}"].sum() / n)),
                      "bias": float(t[f"err_{c}"].sum() / n)}
    return out


def position_bias(test, pred) -> dict:
    d = test[["position", "total_points"]].copy()
    d["err"] = pred.astype(float) - d["total_points"]
    return {p: round(float(g["err"].mean()), 4) for p, g in d.groupby("position")}


ROW_KEY = ["season", "GW", "fixture", "team", "name"]
ROW_COLUMNS = ROW_KEY + ["position", "price", "total_points", "minutes", "clean_sheets"] + SUBSETS
TABLE_META = ["variant", "proto", "fold", "seed", "seeds"]


class Sink:
    """Collects a run's per-gameweek tables (every seed and the seed average) and row-level
    predictions, saved as <out>.gw.parquet and <out>.preds.parquet."""

    def __init__(self, keep_preds: bool = True):
        self.keep_preds = keep_preds
        self.tables, self.preds, self.rows, self.units = [], {}, [], set()

    def add(self, variant, proto, fold, seed, table, rows=None, pred=None, seeds=None):
        t = table.copy()
        for i, (k, v) in enumerate(zip(TABLE_META, (variant, proto, fold, int(seed), seeds or str(seed)))):
            t.insert(i, k, v)
        self.tables.append(t)
        if pred is not None and self.keep_preds:
            col = f"pred|{variant}|{proto}|{seed}"
            s = pd.Series(np.asarray(pred, dtype="float32"), index=rows.index)
            self.preds[col] = pd.concat([self.preds[col], s]) if col in self.preds else s
            if (proto, fold) not in self.units:      # the same rows for every seed and variant
                self.units.add((proto, fold))
                self.rows.append(rows[[c for c in ROW_COLUMNS if c in rows.columns]])

    def save(self, out: str) -> None:
        stem = _stem(out)
        if self.tables:
            pd.concat(self.tables, ignore_index=True).to_parquet(f"{stem}.gw.parquet", index=False)
        if self.preds:
            rows = pd.concat(self.rows)
            rows = rows[~rows.index.duplicated()].sort_index()
            wide = pd.DataFrame({c: s.reindex(rows.index) for c, s in self.preds.items()})
            pd.concat([rows, wide], axis=1).to_parquet(f"{stem}.preds.parquet", index=False)


def _stem(path: str) -> str:
    for suffix in (".preds.parquet", ".gw.parquet", ".json"):
        if path.endswith(suffix):
            return path[:-len(suffix)]
    return path


def score_unit(rows: pd.DataFrame, seed_preds: dict, key: tuple, sink: Sink = None, reuse: dict = None) -> tuple:
    """Scores one protocol on one season: a per-gameweek table for each seed's predictions and
    one for their average. `key` is (variant, proto, season); `reuse` maps a seed, or
    (ENS, "0,1,2") for an average, to a table already computed from the same predictions.
    Returns (unit, {seed: summary}); the unit holds the seed-averaged summary, per-gameweek table
    and position bias, as in the results JSON."""
    reuse = reuse or {}
    seeds = sorted(seed_preds)
    tag = ",".join(map(str, seeds))
    t0, solved = time.time(), 0
    per_seed = {}
    for seed in seeds:
        t = reuse.get(seed)
        if t is None:
            t, solved = gw_table(rows, seed_preds[seed], SQUADS), solved + 1
        per_seed[seed] = summarize(t)
        if sink is not None:
            sink.add(*key, seed, t, rows, seed_preds[seed])
    ens = np.mean([seed_preds[s] for s in seeds], axis=0)
    t = reuse.get((ENS, tag))
    if t is None:
        t, solved = gw_table(rows, ens, SQUADS), solved + 1
    if sink is not None:
        sink.add(*key, ENS, t, seeds=tag)
    if solved:
        print(f"  scored {'/'.join(map(str, key))}: {solved} tables in {time.time() - t0:.0f}s", flush=True)
    unit = {"summary": summarize(t), "per_gw": t.to_dict(orient="list"), "position_bias": position_bias(rows, ens),
            "seed_list": seeds}
    return unit, per_seed


def _seed_line(name, season, seed, info) -> str:
    best = f"{info['best_iter']}/{info['best_iter_cls']}" if "best_iter_cls" in info else info["best_iter"]
    es, rf = info["es"], info["refit"]
    return (f"  {name} {season} seed {seed}: best {best} refit {info['n_refit']} es top20 {es['top20']:.3f} "
            f"refit top20 {rf['top20']:.3f} rmse {rf['rmse']:.4f} sq_fpl {rf.get('sq_fpl', float('nan')):.2f} "
            f"({info['secs']}s)")


def run_variant(feats: pd.DataFrame, name: str, spec: dict, sink: Sink = None) -> dict:
    cols = spec.get("features", CURRENT)
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
        preds = {"es": {}, "refit": {}}
        fold = {"train": train_s, "valid": valid_s, "rows": [len(tr), len(va), len(te)], "seeds": {}}
        for seed in SEEDS:
            t0 = time.time()
            p = {**params, "random_state": seed}
            if spec.get("model") == "hurdle":
                pe, pr, info = _hurdle_fold(tr, va, te, both, cols, p)
            else:
                pe, pr, info = _xgb_fold(tr, va, te, both, cols, p, spec, curve=seed == SEEDS[0])
            info["secs"] = round(time.time() - t0, 1)
            preds["es"][seed], preds["refit"][seed] = pe, pr
            fold["seeds"][seed] = info
        for proto in ("es", "refit"):
            fold[proto], per_seed = score_unit(te, preds[proto], (name, proto, test_season), sink)
            for seed, summary in per_seed.items():
                fold["seeds"][seed][proto] = summary
        for seed, info in fold["seeds"].items():
            print(_seed_line(name, test_season, seed, info), flush=True)
        res["folds"][test_season] = fold
    for season in INSEASON:
        if season in present:
            res.setdefault("inseason", {})[season] = run_inseason(feats, spec, season, cols, params, present,
                                                                  name, sink)
    return res


INSEASON: tuple = ()
INSEASON_EVERY = 5


def run_inseason(feats, spec, season, cols, params, present, name="", sink=None) -> dict:
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
    preds = {}
    for seed in SEEDS:
        p = {**params, "random_state": seed}
        if spec.get("model") == "hurdle":
            _, _, n_cls, n_reg = _hurdle_es(tr, va, cols, p)
            out = pd.Series(np.nan, index=cur.index)
            for start in range(int(gws[0]), int(gws[-1]) + 1, INSEASON_EVERY):
                seen = feats[feats["season"].isin(pool) | ((feats["season"] == season) & (feats["GW"] < start))]
                block = cur[(cur["GW"] >= start) & (cur["GW"] < start + INSEASON_EVERY)]
                if block.empty:
                    continue
                scale = len(seen) / len(tr)
                cls, reg = _hurdle_fit(seen, cols, p, min(int(n_cls * scale), 4000), min(int(n_reg * scale), 4000))
                out.loc[block.index] = _hurdle_predict(cls, reg, block, cols)
            preds[seed] = out.to_numpy()
            print(f"  inseason {season} seed {seed}: best {n_reg}/{n_cls}", flush=True)
            continue
        m = xgb.XGBRegressor(n_estimators=NMAX, early_stopping_rounds=PATIENCE, **p)
        m.fit(tr[cols], _target(tr, spec), sample_weight=_weights(tr, spec.get("halflife")),
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
            m2.fit(seen[cols], _target(seen, spec), sample_weight=_weights(seen, spec.get("halflife")), verbose=False)
            out.loc[block.index] = m2.predict(block[cols])
        preds[seed] = out.to_numpy()
        print(f"  inseason {season} seed {seed}: best {best}", flush=True)
    unit, per_seed = score_unit(cur, preds, (name, "inseason", season), sink)
    return {**unit, "seeds": [per_seed[s] for s in unit["seed_list"]]}


def parse_seeds(text: str) -> list:
    """"0-4,7" -> [0, 1, 2, 3, 4, 7]"""
    out = []
    for part in (text or "").replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    if len(set(out)) != len(out):
        raise SystemExit(f"repeated seeds in {text!r}")
    return out


def cmd_run(args) -> None:
    global INSEASON, TEST_SEASONS, SEEDS, SQUADS
    feats = pd.read_pickle(args.feats)
    if args.inseason:
        INSEASON = tuple(args.inseason.split(","))
    if args.seeds:
        SEEDS = tuple(parse_seeds(args.seeds))
    if args.folds is not None:
        TEST_SEASONS = [s for s in args.folds.split(",") if s]
    SQUADS = not args.no_squads
    names = args.variants.split(",")
    for name in names:
        # e.g. odds features when the odds weren't fetched (the workflow fetches them only for odds* variants)
        empty = [c for c in VARIANTS[name].get("features", CURRENT) if not feats[c].notna().any()]
        if empty:
            raise SystemExit(f"variant {name}: no values in {', '.join(empty)}; was its data fetched?")
    out, sink = [], Sink()
    for name in names:
        t0 = time.time()
        print(f"== {name}", flush=True)
        out.append(run_variant(feats, name, VARIANTS[name], sink))
        print(f"== {name} done in {time.time() - t0:.0f}s", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    sink.save(args.out)


# ----------------------------------------------------------------------------- score saved predictions

def _merge_preds(variant: str, frames: list) -> pd.DataFrame:
    """One variant's predictions from several files (runs with different seeds), joined on the
    row key; rows missing from any file are dropped, the first file's actuals are kept."""
    merged, seen = None, set()
    for df in frames:
        pcols = [c for c in df.columns if c.startswith(f"pred|{variant}|")]
        dup = seen & set(pcols)
        if dup:
            raise SystemExit(f"{variant}: {', '.join(sorted(dup))} in more than one file")
        seen |= set(pcols)
        part = df[[c for c in ROW_COLUMNS if c in df.columns] + pcols]
        part = part[part[pcols].notna().any(axis=1)].copy()
        part["_dup"] = part.groupby(ROW_KEY, sort=False).cumcount()
        if merged is None:
            merged = part
            continue
        n = (len(merged), len(part))
        merged = merged.merge(part[ROW_KEY + ["_dup"] + pcols], on=ROW_KEY + ["_dup"], how="inner")
        if len(merged) < max(n):
            print(f"warning: {variant}: {max(n) - len(merged)} rows are in only some of the files; "
                  f"scoring the {len(merged)} in all", flush=True)
    return merged.drop(columns="_dup").reset_index(drop=True)


def _training_info(variant: str, paths: list) -> dict:
    """The variant's results without scores (tree counts, test curves, top features), seeds of
    all files merged, from the results JSON beside each predictions file."""
    res = {"variant": variant, "folds": {}}
    for path in paths:
        js = _stem(path) + ".json"
        if not os.path.exists(js):
            continue
        for r in json.load(open(js, encoding="utf-8")):
            if r["variant"] != variant:
                continue
            res["features"] = r.get("features")
            for season, f in r.get("folds", {}).items():
                fold = res["folds"].setdefault(season, {k: v for k, v in f.items() if k not in ("seeds", "es", "refit")})
                fold.setdefault("seeds", {}).update(f.get("seeds", {}))
    return res


def _saved_tables(variant: str, paths: list) -> dict:
    """{(proto, season): {seed or (ENS, seeds): table}} from the .gw.parquet beside each file."""
    out = {}
    for path in paths:
        gw = _stem(path) + ".gw.parquet"
        if not os.path.exists(gw):
            continue
        t = pd.read_parquet(gw)
        for (proto, fold, seed, seeds), g in t[t["variant"] == variant].groupby(["proto", "fold", "seed", "seeds"]):
            key = (ENS, seeds) if seed == ENS else int(seed)
            out.setdefault((proto, fold), {})[key] = g.drop(columns=TABLE_META).reset_index(drop=True)
    return out


def score_variant(variant: str, rows: pd.DataFrame, base: dict, reuse: dict, sink: Sink) -> dict:
    """Results of a variant (as `run` writes them) from its row-level predictions."""
    cols = {}
    for c in rows.columns:
        if c.startswith(f"pred|{variant}|"):
            _, _, proto, seed = c.split("|")
            cols.setdefault(proto, {})[int(seed)] = c
    res = {**base, "variant": variant, "folds": base.get("folds", {})}
    for proto, by_seed in cols.items():
        for season in sorted(rows["season"].unique()):
            sel = rows[(rows["season"] == season) & rows[list(by_seed.values())].notna().all(axis=1)]
            if sel.empty:
                continue
            seed_preds = {s: sel[c].to_numpy(dtype=float) for s, c in by_seed.items()}
            unit, per_seed = score_unit(sel, seed_preds, (variant, proto, season), sink, reuse.get((proto, season)))
            if proto == "inseason":
                res.setdefault("inseason", {})[season] = {**unit, "seeds": [per_seed[s] for s in unit["seed_list"]]}
                continue
            fold = res["folds"].setdefault(season, {})
            fold[proto] = unit
            for s, summary in per_seed.items():
                fold.setdefault("seeds", {}).setdefault(str(s), {})[proto] = summary
    return res


def cmd_score(args) -> None:
    """Scores saved predictions (the .preds.parquet files `run` writes) without retraining."""
    global SQUADS
    SQUADS = not args.no_squads
    paths = sorted(dict.fromkeys(p for pattern in args.preds for p in glob.glob(pattern)))
    if not paths:
        raise SystemExit(f"no files match {args.preds}")
    files = {path: pd.read_parquet(path) for path in paths}
    by_variant = {}
    for path, df in files.items():
        for v in dict.fromkeys(c.split("|")[1] for c in df.columns if c.startswith("pred|")):
            by_variant.setdefault(v, []).append(path)
    out, sink = [], Sink(keep_preds=False)
    for variant, vpaths in by_variant.items():
        if len(vpaths) > 1 and not args.pool:
            raise SystemExit(f"{variant} is in {len(vpaths)} files; pass --pool to merge their seeds")
        print(f"== {variant}: {', '.join(vpaths)}", flush=True)
        rows = _merge_preds(variant, [files[p] for p in vpaths])
        reuse = _saved_tables(variant, vpaths) if args.reuse else {}
        out.append(score_variant(variant, rows, _training_info(variant, vpaths), reuse, sink))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    sink.save(args.out)


def cmd_matrix(args) -> None:
    """Prints the workflow's run matrix: one job per variant and chunk of at most --chunk seeds, so
    a 20-seed reference runs as four jobs that `score --pool` merges."""
    jobs = []
    for v in args.variants.split(","):
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}")
        seeds = parse_seeds(args.ref_seeds if v == args.ref and args.ref_seeds else args.seeds) or list(SEEDS)
        chunks = [seeds[i:i + args.chunk] for i in range(0, len(seeds), args.chunk)]
        for c in chunks:
            jobs.append({"variant": v, "seeds": ",".join(map(str, c)),
                         "tag": v if len(chunks) == 1 else f"{v}-s{c[0]}"})
    print("matrix=" + json.dumps(jobs))


# ----------------------------------------------------------------------------- report

def _boot(base: pd.DataFrame, var: pd.DataFrame, n=10000, seed=0) -> dict:
    m = base.merge(var, on=["season", "GW"], suffixes=("_b", "_v"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m), size=(n, len(m)))
    out = {}
    for k in ("top20", "top50", "capt", "spear", *PICK_METRICS, "capt_fpl", *SQ_METRICS, *POS_METRICS, *PRED_METRICS):
        if f"{k}_v" not in m or f"{k}_b" not in m:
            continue
        d = (m[f"{k}_v"] - m[f"{k}_b"]).to_numpy(dtype=float)
        if np.isnan(d).all():
            continue
        bs = np.nanmean(d[idx], axis=1)   # NaN: too few subset players that gameweek
        out[k] = (float(np.nanmean(d)), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)))
    se_b, se_v, rows = m["se_b"].to_numpy(), m["se_v"].to_numpy(), m["rows_b"].to_numpy()
    full = np.sqrt(se_v.sum() / rows.sum()) - np.sqrt(se_b.sum() / rows.sum())
    bs = np.sqrt(se_v[idx].sum(1) / rows[idx].sum(1)) - np.sqrt(se_b[idx].sum(1) / rows[idx].sum(1))
    out["rmse"] = (float(full), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    for c in SUBSETS:
        if f"n_{c}_b" not in m or f"se_{c}_v" not in m or m[f"n_{c}_b"].sum() == 0:
            continue
        sb, sv, nn = m[f"se_{c}_b"].to_numpy(), m[f"se_{c}_v"].to_numpy(), m[f"n_{c}_b"].to_numpy()
        full = np.sqrt(sv.sum() / nn.sum()) - np.sqrt(sb.sum() / nn.sum())
        den = np.maximum(nn[idx].sum(1), 1e-9)
        bs = np.sqrt(sv[idx].sum(1) / den) - np.sqrt(sb[idx].sum(1) / den)
        out[c] = (float(full), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    for p in opt.POSITIONS:
        if f"pn_{p}_b" not in m or f"perr_{p}_v" not in m or m[f"pn_{p}_b"].sum() == 0:
            continue
        eb, ev, nn = m[f"perr_{p}_b"].to_numpy(), m[f"perr_{p}_v"].to_numpy(), m[f"pn_{p}_b"].to_numpy()
        bs = (ev[idx].sum(1) - eb[idx].sum(1)) / np.maximum(nn[idx].sum(1), 1e-9)
        out[f"bias_{p}"] = (float((ev.sum() - eb.sum()) / nn.sum()), float(np.percentile(bs, 2.5)),
                            float(np.percentile(bs, 97.5)))
    return out


SINGLE_METRICS = ["rmse", "top20", "capt", "capt_fpl", "sq_xi", "sq_fpl", "sq_dnp"]
SINGLE_POS_METRICS = POS_METRICS + [f"bias_{p}" for p in opt.POSITIONS]
RATIOS = {"rmse": ("se", "rows", np.sqrt), **{f"bias_{p}": (f"perr_{p}", f"pn_{p}", None) for p in opt.POSITIONS}}


def _seed_matrix(t: pd.DataFrame, gws: pd.MultiIndex, col: str) -> np.ndarray:
    """Gameweeks x seeds of one column of per-seed tables."""
    return t.set_index(["season", "GW", "seed"])[col].unstack("seed").reindex(gws).to_numpy(dtype=float)


def _common_gws(*tables) -> pd.MultiIndex:
    keys = [t[["season", "GW"]].drop_duplicates() for t in tables]
    common = keys[0]
    for k in keys[1:]:
        common = common.merge(k)
    return pd.MultiIndex.from_frame(common.sort_values(["season", "GW"]))


def _per_seed(t: pd.DataFrame, gws: pd.MultiIndex, k: str) -> tuple:
    """(each seed's metric over the gameweeks, the seed-mean per gameweek or None for ratios)."""
    if k in RATIOS:
        num, den, f = RATIOS[k]
        v = _seed_matrix(t, gws, num).sum(0) / _seed_matrix(t, gws, den).sum(0)
        return (f(v) if f else v), None
    m = _seed_matrix(t, gws, k)
    m = m[~np.isnan(m).any(1)]
    return m.mean(0), m.mean(1)


def _boot2(base: pd.DataFrame, var: pd.DataFrame, n=10000, seed=0) -> dict:
    """Single models, variant minus reference, from per-seed per-gameweek tables: a two-level
    bootstrap that resamples gameweeks (iid, the same draw for both) and, independently on each
    side, seeds. The estimate is the difference of the seed means, so the CI covers both the
    gameweek sample and the seed lottery."""
    gws = _common_gws(base, var)
    rng = np.random.default_rng(seed)
    g = len(gws)
    counts = rng.multinomial(g, np.full(g, 1.0 / g), size=n).astype(float)        # replicates x gameweeks

    def seed_weights(t):
        s = t["seed"].nunique()
        return rng.multinomial(s, np.full(s, 1.0 / s), size=n) / s                 # replicates x seeds

    wb, wv = seed_weights(base), seed_weights(var)
    out = {}
    for k in SINGLE_METRICS + SINGLE_POS_METRICS:
        if k in RATIOS:
            num, den, f = RATIOS[k]
            if num not in base or num not in var:
                continue
            f = f or (lambda x: x)
            reps = []
            for t, w in ((base, wb), (var, wv)):
                ratio = (counts @ _seed_matrix(t, gws, num)) / np.maximum(counts @ _seed_matrix(t, gws, den), 1e-9)
                reps.append((f(ratio) * w).sum(1))
            bs = reps[1] - reps[0]
            point = _per_seed(var, gws, k)[0].mean() - _per_seed(base, gws, k)[0].mean()
        else:
            if k not in base or k not in var:
                continue
            mb, mv = _seed_matrix(base, gws, k), _seed_matrix(var, gws, k)
            ok = ~(np.isnan(mb).any(1) | np.isnan(mv).any(1))     # NaN: no squad, too few players
            if not ok.any():
                continue
            d = mv[ok] @ wv.T - mb[ok] @ wb.T                     # gameweeks x replicates
            c = counts[:, ok]
            total = c.sum(1)
            bs = np.where(total > 0, (c * d.T).sum(1) / np.maximum(total, 1), np.nan)
            point = float(np.mean(mv[ok].mean(1) - mb[ok].mean(1)))
        out[k] = (float(point), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)))
    return out


def _seed_spread(t: pd.DataFrame, k: str) -> tuple:
    """(mean over seeds, SD between seeds, SE over gameweeks of the seed mean) of a metric."""
    gws = _common_gws(t)
    if k in RATIOS:
        if RATIOS[k][0] not in t:
            return None
    elif k not in t:
        return None
    per_seed, per_gw = _per_seed(t, gws, k)
    if not len(per_seed):
        return None
    sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan")
    se = float(np.std(per_gw, ddof=1) / np.sqrt(len(per_gw))) if per_gw is not None and len(per_gw) > 1 else float("nan")
    return float(np.mean(per_seed)), sd, se


def _fmt(t, digits=3):
    return f"{t[0]:+.{digits}f} [{t[1]:+.{digits}f}, {t[2]:+.{digits}f}]"


def _digits(k: str) -> int:
    return 4 if k == "rmse" else 2 if k.startswith(("sq_", "capt")) else 3


def _squad_sections(lines, results, ref, proto, units, boots, tables) -> None:
    """Report sections on the squad, captaincy, position picks and position bias of one protocol."""
    nan = float("nan")
    base = units(results[ref], proto)

    def cell(sm, k, d=None):
        return f"{sm[k]:.{d or _digits(k)}f}" if k in sm else "-"

    if any("sq_xi" in u["summary"] or "top_DEF" in u["summary"] for u in base.values()):
        lines.append(f"\n### {ref}: the optimiser's squad, captaincy and position picks (seed-averaged)\n")
        lines.append("Actual points, predicted in brackets. sq_xi: the squad's XI after FPL auto-subs, captain "
                     "counted once; sq_fpl: its FPL score (captain doubled, the vice if he didn't play); sq_dnp: "
                     "picked starters who didn't play; capt_fpl: the top prediction as captain, the second as vice. "
                     "Position picks: mean of the top 4 GK, 10 DEF, 10 MID and 6 FWD by prediction.\n")
        lines.append("| fold | seeds | sq_xi | sq_fpl (pred) | sq_dnp | capt (pred) | capt_fpl | top20 (pred) | "
                     "GK top4 | DEF top10 | MID top10 | FWD top6 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s, u in base.items():
            sm = u["summary"]
            pos = " | ".join(f"{cell(sm, f'top_{p}')} ({cell(sm, f'pred_{p}', 3)})" for p in POS_TOPK)
            lines.append(f"| {s} | {len(u.get('seed_list', [])) or '-'} | {cell(sm, 'sq_xi')} | {cell(sm, 'sq_fpl')} "
                         f"({cell(sm, 'sq_pred')}) | {cell(sm, 'sq_dnp')} | {cell(sm, 'capt')} "
                         f"({cell(sm, 'pred_capt', 2)}) | {cell(sm, 'capt_fpl')} | {cell(sm, 'top20')} "
                         f"({cell(sm, 'pred20', 3)}) | {pos} |")
    squad_keys = ["sq_xi", "sq_fpl", "sq_dnp", "capt_fpl"] + POS_METRICS
    rows = [f"| {name} | " + " | ".join(_fmt(b[k], _digits(k)) if k in b else "-" for k in squad_keys) + " |"
            for name, b in boots.items() if any(k in b for k in squad_keys)]
    if rows:
        lines.append("\n#### Squad, captaincy and position picks, variant minus reference (seed-averaged; "
                     "gameweek bootstrap) [95% CI]\n")
        lines.append("| variant | Δsq_xi | Δsq_fpl | Δsq_dnp | Δcapt_fpl | Δtop4 GK | Δtop10 DEF | Δtop10 MID | Δtop6 FWD |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        lines += rows

    lines.append("\n#### Bias by position (pred - actual), every variant, seed-averaged: pooled over folds, "
                 "difference from the reference [95% CI], per fold GK/DEF/MID/FWD\n")
    lines.append("| variant | GK | DEF | MID | FWD | ΔGK | ΔDEF | ΔMID | ΔFWD | per fold |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        ru = units(r, proto)
        if not ru:
            continue
        per_gw = pd.concat([pd.DataFrame(u["per_gw"]) for u in ru.values()])
        pooled = [f"{per_gw[f'perr_{p}'].sum() / per_gw[f'pn_{p}'].sum():+.3f}"
                  if f"pn_{p}" in per_gw and per_gw[f"pn_{p}"].sum() > 0 else "-" for p in opt.POSITIONS]
        b = boots.get(name, {})
        delta = [_fmt(b[f"bias_{p}"]) if f"bias_{p}" in b else "-" for p in opt.POSITIONS]
        per_fold = "; ".join(f"{s[2:]}: " + "/".join(f"{u['position_bias'].get(p, nan):+.3f}" for p in opt.POSITIONS)
                             for s, u in ru.items())
        lines.append(f"| {name} | " + " | ".join(pooled + delta) + f" | {per_fold} |")

    tb = tables.get(ref)
    tb = tb[tb["proto"] == proto] if tb is not None else None
    if tb is None or tb.empty:
        return
    main_rows, pos_rows = [], []
    for name in results:
        tv = tables.get(name)
        if name == ref or tv is None:
            continue
        tv = tv[tv["proto"] == proto]
        common = sorted(set(tb["fold"]) & set(tv["fold"]))
        if not common:
            continue
        b2 = _boot2(tb[tb["fold"].isin(common)], tv[tv["fold"].isin(common)])
        seeds = f"{tv['seed'].nunique()}/{tb[tb['fold'].isin(common)]['seed'].nunique()}"
        main_rows.append(f"| {name} | {seeds} | " + " | ".join(_fmt(b2[k], _digits(k)) if k in b2 else "-"
                                                             for k in SINGLE_METRICS) + " |")
        pos_rows.append(f"| {name} | " + " | ".join(_fmt(b2[k]) if k in b2 else "-" for k in SINGLE_POS_METRICS) + " |")
    if main_rows:
        lines.append("\n#### Single models (what production ships): per-seed metrics, variant minus reference, "
                     "two-level bootstrap (gameweeks and seeds resampled) [95% CI]\n")
        lines.append("| variant | seeds (variant/ref) | Δrmse | Δtop20 | Δcapt | Δcapt_fpl | Δsq_xi | Δsq_fpl | Δsq_dnp |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        lines += main_rows
        lines.append("\n| variant | Δtop4 GK | Δtop10 DEF | Δtop10 MID | Δtop6 FWD | Δbias GK | Δbias DEF | Δbias MID "
                     "| Δbias FWD |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        lines += pos_rows


def _load_results(paths) -> tuple:
    """Results by variant from JSON files, and each variant's per-gameweek tables from the
    .gw.parquet beside a JSON (seed-averaged tables dropped), when there is one."""
    results, tables, solves = {}, {}, {}
    for path in paths:
        for pattern in sorted(glob.glob(path)):
            for r in json.load(open(pattern, encoding="utf-8")):
                results[r["variant"]] = r
            gw = _stem(pattern) + ".gw.parquet"
            if os.path.exists(gw):
                t = pd.read_parquet(gw)
                for v, g in t.groupby("variant"):
                    tables[v] = g[g["seed"] != ENS]
                    if "sq_secs" in g:
                        solves[v] = g["sq_secs"].to_numpy(), int(g["sq_xi"].isna().sum())
    return results, tables, solves


def cmd_report(args) -> None:
    results, tables, solves = _load_results(args.paths)
    lines = []
    base = results[args.ref]

    def units(r, proto):
        if proto == "inseason":
            return r.get("inseason", {})
        return {s: f[proto] for s, f in r.get("folds", {}).items() if proto in f}

    for proto in ("refit", "es", "inseason"):
        if not units(base, proto):
            continue
        lines.append(f"\n## Protocol: {proto}\n")
        lines.append(f"### {args.ref}, seed-averaged\n")
        lines.append("| fold | rmse | mae | top20 | top50 | capt | spear | dnp20 | top20 euro clubs | top20 GK/DEF "
                     "| CS pts top20 GK/DEF | bias GK/DEF/MID/FWD |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        nan = float("nan")
        for s, u in units(base, proto).items():
            sm, pb = u["summary"], u["position_bias"]
            lines.append(f"| {s} | {sm['rmse']:.4f} | {sm['mae']:.4f} | {sm['top20']:.3f} | {sm['top50']:.3f} | "
                         f"{sm['capt']:.2f} | {sm['spear']:.4f} | {sm.get('dnp20', nan):.3f} | "
                         f"{sm.get('top20_sub_euro_club', nan):.3f} | {sm.get('top20_sub_gkdef', nan):.3f} | "
                         f"{sm.get('cs20_sub_gkdef', nan):.3f} | "
                         f"{'/'.join(f'{pb.get(p_, 0):+.3f}' for p_ in pm.POSITIONS)} |")
        lines.append("\n### Variant minus reference (pooled: mean [95% CI]; per fold: Δtop20 / Δrmse×1000)\n")
        lines.append("| variant | Δrmse pooled | Δtop20 pooled | Δtop50 pooled | Δcapt pooled | per fold |")
        lines.append("|---|---|---|---|---|---|")
        subset_rows, pick_rows, boots, seed_counts = [], [], {}, {}
        for name, r in results.items():
            ru, bu = units(r, proto), units(base, proto)
            common = [s_ for s_ in ru if s_ in bu]
            seed_counts[name] = sorted({len(ru[s_].get("seed_list", [])) for s_ in common})
            if name == args.ref or not common:
                continue
            b = boots[name] = _boot(pd.concat([pd.DataFrame(bu[s_]["per_gw"]) for s_ in common]),
                                    pd.concat([pd.DataFrame(ru[s_]["per_gw"]) for s_ in common]))
            per = []
            for s_ in common:
                bs, vs = bu[s_]["summary"], ru[s_]["summary"]
                per.append(f"{s_[2:]}: {vs['top20'] - bs['top20']:+.3f}/{1000 * (vs['rmse'] - bs['rmse']):+.1f}")
            lines.append(f"| {name} | {_fmt(b['rmse'], 4)} | {_fmt(b['top20'])} | {_fmt(b['top50'])} | "
                         f"{_fmt(b['capt'], 2)} | {'; '.join(per)} |")
            subset_rows.append(f"| {name} | " + " | ".join(_fmt(b[c], 4) if c in b else "-" for c in SUBSETS) + " |")
            pick_rows.append(f"| {name} | " + " | ".join(_fmt(b[k]) if k in b else "-" for k in PICK_METRICS) + " |")
        counts = {n_: c for n_, c in seed_counts.items() if c and c != [0]}
        if len({tuple(c) for c in counts.values()}) > 1:
            lines.append("\nSeeds averaged: " + ", ".join(f"{n_} {'/'.join(map(str, c))}" for n_, c in counts.items())
                         + ". An average of more seeds predicts better, so the seed-averaged comparisons favour the "
                         "side with more seeds; compare single models below.")
        if pick_rows:
            lines.append("\n#### Pick subsets: share of the top 20 who didn't play (lower is better), top-20 points "
                         "among players of clubs in Europe and among GK/DEF, and the clean-sheet points of those "
                         "GK/DEF picks, variant minus reference [95% CI]\n")
            lines.append("| variant | Δdnp20 | Δtop20 euro clubs | Δtop20 GK/DEF | ΔCS pts top20 GK/DEF |")
            lines.append("|---|---|---|---|---|")
            lines += pick_rows
        ref_sub = []
        for c in SUBSETS:
            parts = [u["summary"][c] for u in units(base, proto).values() if c in u["summary"]]
            if parts:
                n = sum(x["rows"] for x in parts)
                rmse = np.sqrt(sum(x["rmse"] ** 2 * x["rows"] for x in parts) / n)
                bias = sum(x["bias"] * x["rows"] for x in parts) / n
                ref_sub.append(f"{c}: {n} rows, rmse {rmse:.3f}, bias {bias:+.3f}")
        if ref_sub:
            lines.append(f"\n#### Subsets ({args.ref}: " + "; ".join(ref_sub) + "): Δrmse [95% CI]\n")
            lines.append("| variant | " + " | ".join(SUBSETS) + " |")
            lines.append("|---|" + "---|" * len(SUBSETS))
            lines += subset_rows
        _squad_sections(lines, results, args.ref, proto, units, boots, tables)
    spread_keys = ["top20", "capt", "capt_fpl", "sq_xi", "sq_fpl", "rmse"]
    spread_rows = []
    for name, t in tables.items():
        for proto, tp in t.groupby("proto", sort=False):
            cells = []
            for k in spread_keys:
                s = _seed_spread(tp, k)
                d = _digits(k)
                cells.append(f"{s[0]:.{d}f} ± {s[1]:.{d}f} / {s[2]:.{d}f}" if s else "-")
            spread_rows.append(f"| {name} | {proto} | {tp['seed'].nunique()} | " + " | ".join(cells) + " |")
    if spread_rows:
        lines.append("\n## Seed variance of single models (pooled over folds): mean over seeds ± SD between seeds / "
                     "SE over gameweeks of the seed mean\n")
        lines.append("| variant | protocol | seeds | " + " | ".join(spread_keys) + " |")
        lines.append("|---|---|---|" + "---|" * len(spread_keys))
        lines += spread_rows
    if solves:
        lines.append(f"\n## Squad solves (time limit {SQ_TIME_LIMIT:.0f}s each)\n")
        for name, (secs, failed) in solves.items():
            lines.append(f"- **{name}**: {len(secs)} squads, {failed} infeasible, mean {np.nanmean(secs):.2f}s, "
                         f"max {np.nanmax(secs):.1f}s, {int((secs >= 0.98 * SQ_TIME_LIMIT).sum())} at the limit")
    lines.append("\n## Seed spread (refit top20 / rmse per seed)\n")
    for name, r in results.items():
        cells = []
        for s, f in r.get("folds", {}).items():
            cells.append(f"{s[2:]}: " + ", ".join(f"{v['refit']['top20']:.3f}/{v['refit']['rmse']:.4f}"
                                                for v in f.get("seeds", {}).values() if "refit" in v))
        lines.append(f"- **{name}**: " + "; ".join(cells))
    lines.append("\n## Test curve of the early-stopping fit (seed 0): trees -> valid rmse | test top20 | test rmse\n")
    for name, r in results.items():
        for s, f in r.get("folds", {}).items():
            seed0 = next(iter(f.get("seeds", {}).values()), {})
            if "best_iter" not in seed0:      # scored without its training info
                continue
            pts = seed0.get("test_curve", {})
            cell = "; ".join(f"{k}: {v['valid_rmse']:.4f}|{v['top20']:.3f}|{v['rmse']:.4f}" for k, v in pts.items())
            lines.append(f"- {name} {s} (best {seed0['best_iter']}, argmin {seed0['argmin_valid']}): {cell}")
    lines.append("\n## Top gain features (refit, seed 0, last fold)\n")
    for name, r in results.items():
        if not r.get("folds"):
            continue
        last = list(r["folds"].values())[-1]
        seed0 = next(iter(last.get("seeds", {}).values()), {})
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
    r.add_argument("--seeds", default="", help="training seeds, e.g. 3,4,5 or 0-19 (default: 0,1,2)")
    r.add_argument("--no-squads", action="store_true", help="skip the squad solves")
    sc = sub.add_parser("score", help="score saved predictions without retraining")
    sc.add_argument("preds", nargs="+", help=".preds.parquet files written by run (globs allowed)")
    sc.add_argument("--out", required=True)
    sc.add_argument("--pool", action="store_true", help="merge files of the same variant with different seeds")
    sc.add_argument("--reuse", action="store_true",
                    help="keep the tables in the .gw.parquet beside each file whose predictions are unchanged")
    sc.add_argument("--no-squads", action="store_true", help="skip the squad solves")
    mx = sub.add_parser("matrix", help="the workflow's run jobs: variants x chunks of seeds")
    mx.add_argument("--variants", required=True)
    mx.add_argument("--seeds", default="")
    mx.add_argument("--ref", default="current")
    mx.add_argument("--ref-seeds", default="", help="seeds for the reference variant, e.g. 0-19")
    mx.add_argument("--chunk", type=int, default=5, help="seeds per job")
    rep = sub.add_parser("report")
    rep.add_argument("paths", nargs="+")
    rep.add_argument("--out")
    rep.add_argument("--ref", default="current", help="variant the others are compared against")
    args = parser.parse_args()
    {"features": cmd_features, "run": cmd_run, "score": cmd_score, "matrix": cmd_matrix,
     "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
