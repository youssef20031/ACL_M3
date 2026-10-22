"""
Rolling-origin backtest for the v3 points model. Too heavy for a laptop: run it in CI with the
"ML backtest" workflow (.github/workflows/ml-backtest.yml), which calls

    python -m scripts.analysis.ml_backtest features --out feats.pkl.gz
    python -m scripts.analysis.ml_backtest run --feats feats.pkl.gz --variants current,mgr --out res.json
    python -m scripts.analysis.ml_backtest report res_*.json --ref current

For every test season S (2023-24 .. the current season) and every variant:
  * "es":       trained on seasons < S-1, early-stopped on S-1 (patience 100, as in train_points_v3)
  * "refit":    refit on all seasons < S with the early-stopped tree count scaled by the extra
                rows, which is how the production model is built
  * "inseason": (--inseason) the weekly retrain replayed through a season, refitting every 5 GWs
Each is trained with three seeds; the seed-averaged prediction is scored per gameweek, and every
variant is compared with a reference variant using paired bootstrap CIs over gameweeks.

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

SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
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
EXTRA = G_P90 + G_R10 + G_DC + G_MGR + G_ROT + G_EFL_TEAM + G_EFL_PLAYER + G_OC_SCHED + G_OC_PLAYER + G_OC_STATUS
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
            "es": summarize(gw_table(te, pe)), "refit": summarize(gw_table(te, pr)),
            "top_gain": sorted(gain.items(), key=lambda kv: -kv[1])[:25]}
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


def gw_table(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    subsets = [c for c in SUBSETS if c in test.columns]
    extra = ["clean_sheets"] if "clean_sheets" in test.columns else []
    d = test[["season", "GW", "name", "total_points", "minutes"] + extra + subsets].copy()
    d["pred"] = pred.astype(float)
    d["err"] = d["pred"] - d["total_points"]
    d["se"] = d["err"] ** 2
    d["ae"] = d["err"].abs()
    agg = {"se": ("se", "sum"), "ae": ("ae", "sum"), "rows": ("se", "size")}
    for c in subsets:
        mask = d[c] > 0
        d[f"se_{c}"], d[f"err_{c}"], d[f"n_{c}"] = d["se"].where(mask, 0.0), d["err"].where(mask, 0.0), mask.astype(float)
        agg.update({f"se_{c}": (f"se_{c}", "sum"), f"err_{c}": (f"err_{c}", "sum"), f"n_{c}": (f"n_{c}", "sum")})
    rows = d.groupby(["season", "GW"]).agg(**agg).reset_index()
    rows["GW"] = rows["GW"].astype(int)
    picks = [c for c in PICK_SUBSETS if c in subsets]
    pg = d.groupby(["season", "GW", "name"], as_index=False).agg(
        pred=("pred", "sum"), total_points=("total_points", "sum"), minutes=("minutes", "sum"),
        **{c: (c, "max") for c in picks}, **{c: (c, "sum") for c in extra})
    out = []
    for (s, gw), g in pg.groupby(["season", "GW"]):
        r = g.sort_values("pred", ascending=False, kind="mergesort")
        row = {"season": s, "GW": int(gw), "players": len(g),
               "top20": float(r["total_points"].head(20).mean()),
               "top50": float(r["total_points"].head(50).mean()),
               "capt": float(r["total_points"].iloc[0]),
               "spear": float(g["pred"].corr(g["total_points"], method="spearman")),
               "dnp20": float((r["minutes"].head(20) == 0).mean())}
        for c in picks:
            sub = r[r[c] > 0]
            row[f"top20_{c}"] = float(sub["total_points"].head(20).mean()) if len(sub) >= 40 else np.nan
            if c == CS_PICKS and extra:   # GK/DEF score 4 points per clean sheet
                row[f"cs20_{c}"] = float(4 * sub["clean_sheets"].head(20).mean()) if len(sub) >= 40 else np.nan
        out.append(row)
    return pd.DataFrame(out).merge(rows, on=["season", "GW"])


def summarize(t: pd.DataFrame) -> dict:
    out = {"rmse": float(np.sqrt(t["se"].sum() / t["rows"].sum())), "mae": float(t["ae"].sum() / t["rows"].sum()),
           "top20": float(t["top20"].mean()), "top50": float(t["top50"].mean()), "capt": float(t["capt"].mean()),
           "spear": float(t["spear"].mean()), "gws": int(len(t))}
    for k in PICK_METRICS:
        if k in t and t[k].notna().any():
            out[k] = float(t[k].mean())
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


def run_variant(feats: pd.DataFrame, name: str, spec: dict) -> dict:
    import xgboost as xgb

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
        preds = {"es": [], "refit": []}
        fold = {"train": train_s, "valid": valid_s, "rows": [len(tr), len(va), len(te)], "seeds": {}}
        for seed in SEEDS:
            t0 = time.time()
            p = {**params, "random_state": seed}
            if spec.get("model") == "hurdle":
                pe, pr, info = _hurdle_fold(tr, va, te, both, cols, p)
                info["secs"] = round(time.time() - t0, 1)
                preds["es"].append(pe)
                preds["refit"].append(pr)
                fold["seeds"][seed] = info
                print(f"  {name} {test_season} seed {seed}: best {info['best_iter']}/{info['best_iter_cls']} "
                      f"es top20 {info['es']['top20']:.3f} refit top20 {info['refit']['top20']:.3f} "
                      f"rmse {info['refit']['rmse']:.4f} ({info['secs']}s)", flush=True)
                continue
            m = xgb.XGBRegressor(n_estimators=NMAX, early_stopping_rounds=PATIENCE_FIT, **p)
            m.fit(tr[cols], _target(tr, spec), sample_weight=_weights(tr, spec.get("halflife")),
                  eval_set=[(va[cols], va["total_points"])], verbose=False)
            curve = m.evals_result()["validation_0"]["rmse"]
            best = _patience_stop(curve, PATIENCE)
            pe = m.predict(te[cols], iteration_range=(0, best + 1))
            n_refit = min(int((best + 1) * len(both) / len(tr)), 4000)
            m2 = xgb.XGBRegressor(n_estimators=n_refit, **p)
            m2.fit(both[cols], _target(both, spec), sample_weight=_weights(both, spec.get("halflife")), verbose=False)
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
            preds.append(out.to_numpy())
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
        preds.append(out.to_numpy())
        print(f"  inseason {season} seed {seed}: best {best}", flush=True)
    ens = np.mean(preds, axis=0)
    t = gw_table(cur, ens)
    per_seed = [summarize(gw_table(cur, p_)) for p_ in preds]
    return {"summary": summarize(t), "per_gw": t.to_dict(orient="list"),
            "position_bias": position_bias(cur, ens), "seeds": per_seed}


def cmd_run(args) -> None:
    global INSEASON, TEST_SEASONS, SEEDS
    feats = pd.read_pickle(args.feats)
    if args.inseason:
        INSEASON = tuple(args.inseason.split(","))
    if args.seeds:
        SEEDS = tuple(int(s) for s in args.seeds.split(","))
    if args.folds is not None:
        TEST_SEASONS = [s for s in args.folds.split(",") if s]
    names = args.variants.split(",")
    for name in names:
        # e.g. odds features when the odds weren't fetched (the workflow fetches them only for odds* variants)
        empty = [c for c in VARIANTS[name].get("features", CURRENT) if not feats[c].notna().any()]
        if empty:
            raise SystemExit(f"variant {name}: no values in {', '.join(empty)}; was its data fetched?")
    out = []
    for name in names:
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
    for k in ("top20", "top50", "capt", "spear", *PICK_METRICS):
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
        subset_rows, pick_rows = [], []
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
            subset_rows.append(f"| {name} | " + " | ".join(_fmt(b[c], 4) if c in b else "-" for c in SUBSETS) + " |")
            pick_rows.append(f"| {name} | " + " | ".join(_fmt(b[k]) if k in b else "-" for k in PICK_METRICS) + " |")
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
    r.add_argument("--seeds", default="", help="training seeds, e.g. 3,4,5 to replicate a run (default: 0,1,2)")
    rep = sub.add_parser("report")
    rep.add_argument("paths", nargs="+")
    rep.add_argument("--out")
    rep.add_argument("--ref", default="current", help="variant the others are compared against")
    args = parser.parse_args()
    {"features": cmd_features, "run": cmd_run, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
