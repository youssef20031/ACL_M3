"""
Points model (v3 and v4): predicts a player's FPL points for one fixture.

Replaces the v2 pipeline (ml/retrain_and_predict_csv.py + FeatureEngineer), whose
opponent-strength merges duplicated rows ~20x so the shifted target mostly landed on
a copy of the same match, and whose context features described the previous match.

Design:
- One row per player per fixture; the target is that fixture's total_points.
- Player features use only matches from earlier gameweeks (rolling means, season to
  date, the last match on its own). Within a double gameweek both fixtures share the
  pre-deadline features, which is exactly what's known when predicting.
- Team features come from a team-match table (one row per team per fixture, not per
  player), also rolled over earlier gameweeks only, over 6 and 20 matches.
- Context of the predicted fixture: home/away, opponent strength, rest days, fixtures
  in the gameweek, price, ownership rank. (FPL's xP is excluded; see feature_columns.)
- The same build_features() runs for training and for inference: inference appends
  "upcoming" rows with unknown outcomes, so features can't see the match they predict.

Model: one XGBoost regressor with position as a feature, saved in XGBoost's native
JSON format (no pickle/scikit-learn version coupling). See train_points_model().

v4 (2026-10-28, the served model; v3 is kept as a shadow scored on every prediction run): the
same rows and code with ExtraInputs (load_extra_inputs), which
  * groups a player's matches by his FPL code instead of the name string (player_identity), so a
    renamed player keeps his form and career; the `name` column stays the display name;
  * adds opponent-adjusted team ratings (ml/fixture_strength.RATING_FEATURES), transfer-market
    ranks and price changes (ml/market.FEATURES), pre-deadline bookmaker odds (ml/odds.FEATURES)
    and FPL's availability before the deadline (ml/snapshots.AVAIL_FEATURES), plus cop_factor
    (FPL's chance of playing as a multiplier; not a feature, production applies it on top).
Without ExtraInputs build_features is exactly v3's. feature_columns("v3") is V3_FEATURES.
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY_CSV = os.path.join(ROOT, "data", "cleaned_merged_seasons_cleaned.csv")
# (season, element) -> FPL code, the player's id across seasons (scripts/data/fetch_player_codes.py)
PLAYER_CODES_CSV = os.path.join(ROOT, "data", "player_codes.csv")

MODEL_VERSION = "v4"    # the served model
SHADOW_VERSION = "v3"   # scored alongside it on every prediction run, for comparison
# Bundled with the code; the weekly retrain (pipeline/retrain.py) writes newer copies to the
# live data volume, which take precedence (see active_model_paths)
MODEL_DIR = os.path.join("ml", "models")


def bundled_model_paths(version: str = MODEL_VERSION):
    """(model, meta) shipped with the code: ml/models/points_{version}.json and _meta.json."""
    return (os.path.join(MODEL_DIR, f"points_{version}.json"), os.path.join(MODEL_DIR, f"points_{version}_meta.json"))


# The v3 model's bundled files (train_points_v3's defaults); use bundled_model_paths(version)
MODEL_PATH, META_PATH = bundled_model_paths("v3")


def live_model_dir() -> str:
    return os.path.join(os.getenv("LIVE_DATA_DIR", os.path.join("data", "live")), "models")


def live_model_paths(version: str = MODEL_VERSION):
    """(model, meta) of a retrained copy in LIVE_DATA_DIR/models (they may not exist)."""
    d = live_model_dir()
    return os.path.join(d, f"points_{version}.json"), os.path.join(d, f"points_{version}_meta.json")


def _have(paths) -> bool:
    return all(os.path.exists(p) for p in paths)


def active_model_paths(version: Optional[str] = None):
    """(model, meta) of `version` to serve: the retrained copy in LIVE_DATA_DIR/models if present,
    else the bundled one. Without a version, the production model (MODEL_VERSION), or the shadow
    until a v4 model exists at all (it's trained in CI; the shadow's features are a subset of v4's,
    so it predicts from the same feature table)."""
    if version is None:
        version = MODEL_VERSION
        if not (_have(live_model_paths(version)) or _have(bundled_model_paths(version))):
            version = SHADOW_VERSION
    live = live_model_paths(version)
    return live if _have(live) else bundled_model_paths(version)


POSITIONS = ["GK", "DEF", "MID", "FWD"]

# Per-match stats that become rolling "form" features (never used for the same match)
FORM_STATS = [
    "total_points", "minutes", "started", "played", "goals_scored", "assists", "bonus", "bps",
    "ict_index", "influence", "creativity", "threat", "clean_sheets", "goals_conceded", "saves",
    "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
    "defensive_contribution",
]
WINDOWS = (3, 6)
# The previous match on its own (`*_r1`): whether a player started it is the strongest single
# signal for the next lineup (rotation, injuries, new signings), which the 3/6-match means blur.
LAST_MATCH_STATS = ["minutes", "started", "total_points"]
# Team / opponent strength over the last 6 matches (team_gf ...) and 20 (team_gf20 ...)
TEAM_WINDOWS = (6, 20)
TEAM_STATS = ["gf", "ga", "xgf", "xga"]

# Promoted/new teams have no rolling history: start them a bit below average
NEW_TEAM_PRIOR = {"team_gf": 1.1, "team_ga": 1.7, "team_xgf": 1.1, "team_xga": 1.7}

HISTORY_COLUMNS = [
    "season", "name", "position", "team", "fixture", "kickoff_time", "GW", "was_home",
    "team_h_score", "team_a_score", "total_points", "minutes", "starts", "goals_scored",
    "assists", "bonus", "bps", "ict_index", "influence", "creativity", "threat", "clean_sheets",
    "goals_conceded", "saves", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution", "value", "selected", "xP",
    # v4: the season's FPL element (player identity, snapshots) and the gameweek's transfers
    "element", "transfers_balance", "transfers_in", "transfers_out",
]


# --------------------------------------------------------------------------- loading

def _normalize(df: pd.DataFrame, season: Optional[str] = None) -> pd.DataFrame:
    df = df.rename(columns={"season_x": "season", "team_x": "team"})
    if season is not None:
        df["season"] = season
    df["position"] = df["position"].replace({"GKP": "GK"})
    df = df[df["position"].isin(POSITIONS)]
    for col in HISTORY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[HISTORY_COLUMNS].copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    df["was_home"] = df["was_home"].astype(bool)
    return df


def load_history(history_csv: str = HISTORY_CSV, live_csv: Optional[str] = None,
                 live_season: Optional[str] = None) -> pd.DataFrame:
    """Historical seasons plus (optionally) the current season's finished matches."""
    usecols = lambda c: c in set(HISTORY_COLUMNS) | {"season_x", "team_x"}
    frames = [_normalize(pd.read_csv(history_csv, usecols=usecols, low_memory=False))]
    if live_csv and os.path.exists(live_csv):
        frames.append(_normalize(pd.read_csv(live_csv, usecols=usecols), season=live_season))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["kickoff_time", "fixture", "name"]).reset_index(drop=True)


def load_player_codes(status_csv: Optional[str] = None, live_season: Optional[str] = None,
                      codes_csv: str = PLAYER_CODES_CSV) -> pd.DataFrame:
    """(season, element, code): data/player_codes.csv plus the current season's live status file."""
    frames = []
    if os.path.exists(codes_csv):
        frames.append(pd.read_csv(codes_csv, usecols=["season", "element", "code"]))
    if status_csv and os.path.exists(status_csv):
        frames.append(pd.read_csv(status_csv, usecols=["element", "code"]).assign(season=live_season))
    if not frames:
        return pd.DataFrame(columns=["season", "element", "code"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["season", "element"])


def player_identity(rows: pd.DataFrame, codes: pd.DataFrame) -> pd.Series:
    """One key per player for every row (season, element, name): his name in the latest season
    with an FPL code, plus " #code" when two players share it. Rows without a code keep their
    name. 173 of 1,980 players appear under several spellings in the history, and four names
    belong to two players (2026-10-27)."""
    left = rows[["season", "element", "name"]].assign(element=pd.to_numeric(rows["element"], errors="coerce"))
    right = codes[["season", "element", "code"]].assign(element=pd.to_numeric(codes["element"], errors="coerce"))
    keyed = left.merge(right.dropna(subset=["element"]), on=["season", "element"], how="left")
    keyed.index = rows.index
    has = keyed["code"].notna()
    latest = keyed[has].sort_values("season", kind="mergesort").groupby("code")["name"].last()
    shared = latest.duplicated(keep=False).to_numpy()
    latest[shared] = latest[shared] + " #" + latest.index[shared].astype(int).astype(str)
    out = keyed["name"].copy()
    out[has] = keyed.loc[has, "code"].map(latest)
    return out


@dataclass
class ExtraInputs:
    """What v4's features need beyond the match rows; load_extra_inputs builds it. Any part left
    None gives NaN features (or name identity, for codes)."""
    codes: Optional[pd.DataFrame] = None          # (season, element, code), load_player_codes
    odds: Optional[pd.DataFrame] = None           # ml.odds.load_odds: season files + fixtures.csv
    snapshots: Optional[pd.DataFrame] = None      # ml.snapshots.load_all_snapshots
    # Team-match table of the seasons before the rows' first one (ml.fixture_strength.team_match_history),
    # so a build from the last few seasons rates teams like one from the full history
    ratings_prior: Optional[pd.DataFrame] = None


def load_extra_inputs(live_dir: Optional[str] = None, season: Optional[str] = None,
                      status_csv: Optional[str] = None, history_csv: str = HISTORY_CSV,
                      ratings_prior: bool = True, odds_dir: Optional[str] = None,
                      snapshots_csv: Optional[str] = None) -> ExtraInputs:
    """ExtraInputs for training and serving alike.

    codes      data/player_codes.csv + the status file's codes for `season`
    odds       every football-data.co.uk file in `odds_dir` (default ml.odds.ODDS_DIR, env ODDS_DIR):
               season files and the upcoming fixtures.csv
    snapshots  data/fpl_snapshots.csv + `live_dir`/element_snapshots/<season>/gwNN.csv + the status
               file itself (captured at its fetched_at), which is what upcoming rows use
    ratings_prior  the history CSV's team-match table; training passes False (its rows start at
               the first season anyway)
    """
    from ml import fixture_strength as fs
    from ml import odds as od
    from ml import snapshots as sn

    status = pd.read_csv(status_csv) if status_csv and os.path.exists(status_csv) else None
    return ExtraInputs(
        codes=load_player_codes(status_csv, season),
        odds=od.load_odds(odds_dir or od.ODDS_DIR),
        snapshots=sn.load_all_snapshots(snapshots_csv or sn.SNAPSHOTS_CSV, live_dir,
                                        status=status, season=season),
        ratings_prior=fs.team_match_history(history_csv=history_csv) if ratings_prior else None,
    )


# --------------------------------------------------------------------------- features

def _team_match_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per fixture: goals and xG for/against, plus the opponent."""
    played = df[df["team_h_score"].notna()]
    first = played.groupby(["season", "fixture", "team"], as_index=False).agg(
        kickoff_time=("kickoff_time", "first"), GW=("GW", "first"), was_home=("was_home", "first"),
        team_h_score=("team_h_score", "first"), team_a_score=("team_a_score", "first"),
        xgf=("expected_goals", "sum"), has_xg=("expected_goals", "count"),
    )
    first["gf"] = np.where(first["was_home"], first["team_h_score"], first["team_a_score"])
    first["ga"] = np.where(first["was_home"], first["team_a_score"], first["team_h_score"])
    first.loc[first["has_xg"] == 0, "xgf"] = np.nan
    # Opponent's xG for = our xG against
    opp = first[["season", "fixture", "team", "xgf"]].rename(columns={"team": "opponent", "xgf": "xga"})
    paired = first.merge(opp, on=["season", "fixture"])
    paired = paired[paired["team"] != paired["opponent"]]
    # Keep a team's fixture even if the opponent has no player rows (xga unknown then)
    first = first.merge(paired[["season", "fixture", "team", "xga"]], on=["season", "fixture", "team"], how="left")
    return first.drop(columns=["has_xg"])


def _opponents(df: pd.DataFrame) -> pd.DataFrame:
    """(season, fixture, team) -> opponent, from the two teams that appear in each fixture."""
    pairs = df[["season", "fixture", "team"]].drop_duplicates()
    merged = pairs.merge(pairs.rename(columns={"team": "opponent"}), on=["season", "fixture"])
    return merged[merged["team"] != merged["opponent"]]


def _rolling_before_gameweek(frame: pd.DataFrame, key: str, cols: List[str], windows) -> pd.DataFrame:
    """Rolling means of `cols` over each `key`'s earlier rows, frozen at the first row of
    each (key, season, GW) so a double gameweek's second fixture can't see the first."""
    frame = frame.sort_values([key, "kickoff_time"])
    out = pd.DataFrame(index=frame.index)
    grouped = frame.groupby(key, sort=False)[cols]
    shifted = grouped.shift(1)
    for w in windows:
        rolled = shifted.groupby(frame[key], sort=False).rolling(w, min_periods=1).mean()
        rolled = rolled.reset_index(level=0, drop=True).reindex(frame.index)
        out[[f"{c}_r{w}" for c in cols]] = rolled[cols].to_numpy()
    first_in_gw = frame.groupby([key, "season", "GW"], sort=False).cumcount() == 0
    out = out.where(first_in_gw)
    out[["_k", "_s", "_g"]] = frame[[key, "season", "GW"]].to_numpy()
    out = out.groupby(["_k", "_s", "_g"], sort=False).transform("first")
    return out


PLAYER_KEY = "player_key"   # groups a player's matches: the name, or with ExtraInputs.codes his FPL code


def build_features(history: pd.DataFrame, upcoming: Optional[pd.DataFrame] = None,
                   extras: Optional[ExtraInputs] = None) -> pd.DataFrame:
    """Feature table for every history row (+ upcoming rows, whose outcomes are unknown).

    `upcoming` needs: season, name, position, team, opponent, fixture, kickoff_time, GW,
    was_home, value, and optionally selected_rank (0-1) and xP. With `extras` (v4) it also needs
    element and the gameweek's transfers_in / transfers_out so far (FPL's transfers_in_event /
    transfers_out_event; transfers_balance defaults to in - out), and the history needs element
    (load_history keeps it). Without `extras` the feature values are v3's.
    """
    df = history.copy()
    df["opponent"] = df.merge(_opponents(df), on=["season", "fixture", "team"], how="left")["opponent"].to_numpy()
    df["is_upcoming"] = False
    if upcoming is not None and len(upcoming):
        up = upcoming.copy()
        up["is_upcoming"] = True
        up["kickoff_time"] = pd.to_datetime(up["kickoff_time"], utc=True)
        df = pd.concat([df, up], ignore_index=True)
    df = df.reset_index(drop=True)
    key = PLAYER_KEY
    codes = extras.codes if extras is not None else None
    if codes is not None and "element" in df.columns:
        df[key] = player_identity(df, codes).to_numpy()
    else:
        df[key] = df["name"]

    df["played"] = (df["minutes"] > 0).astype(float).where(df["minutes"].notna())
    started = df["starts"].where(df["starts"].notna(), (df["minutes"] >= 60).astype(float))
    df["started"] = started.where(df["minutes"].notna())

    # ---- player form, earlier gameweeks only
    form = _rolling_before_gameweek(df, key, FORM_STATS, WINDOWS)
    last_match = _rolling_before_gameweek(df, key, LAST_MATCH_STATS, (1,))
    df = df.join(form).join(last_match)

    # Season to date: points per match played, appearances, minutes share
    df = df.sort_values([key, "kickoff_time"])
    g = df.groupby([key, "season"], sort=False)
    df["season_matches_before"] = g.cumcount()
    df["season_points_before"] = g["total_points"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    df["season_minutes_before"] = g["minutes"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    df["season_ppg"] = df["season_points_before"] / df["season_matches_before"].replace(0, np.nan)
    df["season_minutes_share"] = df["season_minutes_before"] / (90 * df["season_matches_before"].replace(0, np.nan))
    df["career_matches_before"] = df.groupby(key, sort=False).cumcount()
    df["days_rest"] = df.groupby(key, sort=False)["kickoff_time"].diff().dt.total_seconds() / 86400
    df["days_rest"] = df["days_rest"].clip(upper=30)
    # Freeze season-to-date at the gameweek's first fixture too
    first_in_gw = df.groupby([key, "season", "GW"], sort=False).cumcount() == 0
    for col in ["season_matches_before", "season_ppg", "season_minutes_share", "career_matches_before"]:
        df[col] = df[col].where(first_in_gw)
        df[col] = df.groupby([key, "season", "GW"], sort=False)[col].transform("first")

    # ---- team strength (own team and opponent), earlier gameweeks only
    tm = _team_match_table(df[~df["is_upcoming"]])
    team_form = _rolling_before_gameweek(tm, "team", TEAM_STATS, TEAM_WINDOWS)
    tm = tm.join(team_form)
    # Upcoming fixtures have no team-match rows: give each team its form after its last match
    last = tm.sort_values("kickoff_time").groupby("team").tail(1)
    latest = {}
    for row in last.itertuples(index=False):
        # Form including the last match = rolling over its last w incl. that one
        hist = tm[tm["team"] == row.team].sort_values("kickoff_time")
        latest[row.team] = {(c, w): hist[c].tail(w).mean() for w in TEAM_WINDOWS for c in TEAM_STATS}
    pairs = [(c, w) for w in TEAM_WINDOWS for c in TEAM_STATS]
    keyed = tm.set_index(["season", "fixture", "team"])[[f"{c}_r{w}" for c, w in pairs]]

    def team_feats(team_col: str, prefix: str) -> pd.DataFrame:
        idx = pd.MultiIndex.from_frame(df[["season", "fixture", team_col]].rename(columns={team_col: "team"}))
        vals = keyed.reindex(idx).to_numpy()
        feats = pd.DataFrame(vals, index=df.index, columns=[_team_feature(prefix, c, w) for c, w in pairs])
        up = df["is_upcoming"].to_numpy()
        for c, w in pairs:
            col = _team_feature(prefix, c, w)
            if up.any():
                fill = df.loc[up, team_col].map(lambda t: latest.get(t, {}).get((c, w), np.nan))
                feats.loc[up, col] = fill.astype(float).to_numpy()
            prior = NEW_TEAM_PRIOR[f"team_{c}"]
            feats[col] = feats[col].fillna(prior) if c in ("gf", "ga") else feats[col]
        return feats

    df = df.join(team_feats("team", "team")).join(team_feats("opponent", "opp"))

    # ---- fixture context
    fixtures_per_gw = df.drop_duplicates(["season", "fixture", "team"]).groupby(["season", "team", "GW"]).size()
    df["fixtures_in_gw"] = pd.MultiIndex.from_frame(df[["season", "team", "GW"]]).map(fixtures_per_gw).fillna(1).astype(float)
    df["was_home"] = df["was_home"].astype(float)
    df["price"] = df["value"] / 10.0
    if "selected_rank" not in df.columns:
        df["selected_rank"] = np.nan
    hist_mask = ~df["is_upcoming"]
    df.loc[hist_mask, "selected_rank"] = df[hist_mask].groupby(["season", "GW"])["selected"].rank(pct=True)
    for pos in POSITIONS:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(float)

    if extras is not None:
        df = _add_extra_features(df, extras)
    return df.sort_values(["kickoff_time", "fixture", "name"]).reset_index(drop=True)


def _add_extra_features(df: pd.DataFrame, extras: ExtraInputs) -> pd.DataFrame:
    """v4's columns on top of v3's (v4_extra_features, plus cop_factor), for history and upcoming
    rows alike: each module only reads earlier gameweeks or information from before the deadline."""
    from ml import fixture_strength as fs
    from ml import market as mk
    from ml import odds as od
    from ml import snapshots as sn

    for col in mk.TRANSFER_COLUMNS + ["element"]:
        if col not in df.columns:
            df[col] = np.nan
    # Upcoming rows carry FPL's transfers in and out since the last deadline
    df["transfers_balance"] = df["transfers_balance"].fillna(df["transfers_in"] - df["transfers_out"])
    odds = extras.odds if extras.odds is not None else od.empty_odds()
    snaps = extras.snapshots if extras.snapshots is not None else sn.empty_snapshots()
    snap = sn.snapshot_features(df, snaps)
    return df.join([
        fs.ratings_features(df, prior=extras.ratings_prior),
        mk.transfer_features(df, key=PLAYER_KEY),
        od.odds_features(df, odds)[od.FEATURES],
        snap[sn.AVAIL_FEATURES + [sn.COP]],
    ])


def _team_feature(prefix: str, stat: str, window: int) -> str:
    """team_gf for the 6-match window (its original name), team_gf20 for 20 matches."""
    return f"{prefix}_{stat}" if window == 6 else f"{prefix}_{stat}{window}"


def _v3_feature_columns() -> List[str]:
    cols = [f"{c}_r{w}" for w in WINDOWS for c in FORM_STATS]
    cols += ["season_matches_before", "season_ppg", "season_minutes_share", "career_matches_before", "days_rest"]
    cols += [_team_feature(p, c, 6) for p in ("team", "opp") for c in TEAM_STATS]
    # FPL's xP is deliberately NOT a feature: in the historical files it was recorded inside
    # the gameweek (2023-24/2024-25: ranking by xP alone picks ~7 pts/player vs ~4.4 for any
    # pre-deadline feature, i.e. it knows lineups) and is inconsistent in 2025-26, so it
    # doesn't match the pre-deadline ep_next available at prediction time.
    cols += ["fixtures_in_gw", "was_home", "price", "selected_rank"]
    cols += [f"pos_{p}" for p in POSITIONS]
    # Added 2026-10 (order kept so training reproduces the backtest exactly). Rolling-origin
    # backtest over 2023-24 .. 2026-27 (scripts/analysis/ml_backtest.py): RMSE -0.020 in every
    # season, top-50 picks +0.1 pts/player. Per-90 rates, a 10-match window, recency weighting,
    # DC-points features and shallower trees were tried too and didn't help reliably.
    cols += [f"{c}_r1" for c in LAST_MATCH_STATS]
    cols += [_team_feature(p, c, 20) for p in ("team", "opp") for c in TEAM_STATS]
    return cols


V3_FEATURES = _v3_feature_columns()   # the shadow model's, and the backtest's "current"


def v4_extra_features() -> List[str]:
    """What v4 adds to V3_FEATURES, in order (the backtest's odds_rt plus availability): team
    ratings, transfer ranks and price changes, pre-deadline odds, FPL availability. Not the
    set-piece flags (ml/snapshots.SP_FEATURES). In the backtest (2026-10-27) odds_rt beat v3 on
    top-20 under all three protocols on 2023-24 .. 2026-27 but not on the untouched 2022-23;
    availability was neutral on top of the chance-of-playing multiplier, and code identity changed
    no metric."""
    from ml import fixture_strength as fs
    from ml import market as mk
    from ml import odds as od
    from ml import snapshots as sn
    return fs.RATING_FEATURES + mk.FEATURES + od.FEATURES + sn.AVAIL_FEATURES


def feature_columns(version: str = MODEL_VERSION) -> List[str]:
    """The model's input columns: "v3" = V3_FEATURES, "v4" = V3_FEATURES + v4_extra_features()."""
    if version == "v3":
        return list(V3_FEATURES)
    if version == "v4":
        return list(V3_FEATURES) + v4_extra_features()
    raise ValueError(f"unknown model version {version!r}")


# --------------------------------------------------------------------------- evaluation

@dataclass
class Metrics:
    rows: int
    mae: float
    rmse: float
    spearman: float      # mean per-gameweek rank correlation (player-gameweek level)
    top20_points: float  # mean actual points of the top-20 predicted per gameweek
    captain_points: float  # actual points of the single top prediction per gameweek

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def evaluate(df: pd.DataFrame, pred_col: str, target_col: str = "total_points") -> Metrics:
    """Row-level errors plus pick quality per gameweek (double gameweeks summed per player)."""
    d = df[df[pred_col].notna() & df[target_col].notna()]
    err = d[pred_col] - d[target_col]
    pg = d.groupby(["season", "GW", "name"], as_index=False)[[pred_col, target_col]].sum()
    spear, top20, capt = [], [], []
    for _, gw in pg.groupby(["season", "GW"]):
        if len(gw) < 30:
            continue
        spear.append(gw[pred_col].corr(gw[target_col], method="spearman"))
        ranked = gw.sort_values(pred_col, ascending=False)
        top20.append(ranked[target_col].head(20).mean())
        capt.append(ranked[target_col].iloc[0])
    return Metrics(len(d), float(err.abs().mean()), float(np.sqrt((err ** 2).mean())),
                   float(np.nanmean(spear)), float(np.mean(top20)), float(np.mean(capt)))


# --------------------------------------------------------------------------- training

# Chosen on the 2024-25 validation season by top-20 pick quality (depth 4/6/8 x
# min_child_weight 10/40 were tried; differences were small)
XGB_PARAMS = dict(
    objective="reg:squarederror", tree_method="hist", learning_rate=0.03, max_depth=8,
    min_child_weight=40, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
)


def _xgb():
    import xgboost as xgb
    return xgb


def fit(train: pd.DataFrame, valid: Optional[pd.DataFrame] = None, n_estimators: int = 3000,
        params: Optional[dict] = None, cols: Optional[List[str]] = None):
    """XGBoost on `cols` (default: the production model's feature_columns())."""
    xgb = _xgb()
    cols = cols or feature_columns()
    model = xgb.XGBRegressor(n_estimators=n_estimators, early_stopping_rounds=100 if valid is not None else None,
                             **(params or XGB_PARAMS))
    eval_set = [(valid[cols], valid["total_points"])] if valid is not None else None
    model.fit(train[cols], train["total_points"], eval_set=eval_set, verbose=False)
    return model


def save_model(model, meta: dict, model_path: str = MODEL_PATH, meta_path: str = META_PATH,
               features: Optional[List[str]] = None) -> None:
    """Booster plus meta, whose "features" (default feature_columns(meta["version"])) predict reads."""
    model.get_booster().save_model(model_path)
    features = features or feature_columns(meta.get("version", MODEL_VERSION))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({**meta, "features": features}, f, indent=2)


def load_model(model_path: Optional[str] = None, meta_path: Optional[str] = None,
               version: Optional[str] = None):
    """(booster, meta) from the given paths, else active_model_paths(version): load_model(version="v3")
    is the shadow, load_model() the production model."""
    if model_path is None or meta_path is None:
        model_path, meta_path = active_model_paths(version)
    xgb = _xgb()
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return booster, meta


def predict(booster, meta: dict, features: pd.DataFrame) -> np.ndarray:
    xgb = _xgb()
    return booster.predict(xgb.DMatrix(features[meta["features"]]))
