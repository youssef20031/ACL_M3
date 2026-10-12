"""
Points model v3: predicts a player's FPL points for one fixture.

Replaces the v2 pipeline (ml/retrain_and_predict_csv.py + FeatureEngineer), whose
opponent-strength merges duplicated rows ~20x so the shifted target mostly landed on
a copy of the same match, and whose context features described the previous match.

Design:
- One row per player per fixture; the target is that fixture's total_points.
- Player features use only matches from earlier gameweeks (rolling means, season to
  date). Within a double gameweek both fixtures share the pre-deadline features, which
  is exactly what's known when predicting.
- Team features come from a team-match table (one row per team per fixture, not per
  player), also rolled over earlier gameweeks only.
- Context of the predicted fixture: home/away, opponent strength, rest days, fixtures
  in the gameweek, price, ownership rank. (FPL's xP is excluded; see feature_columns.)
- The same build_features() runs for training and for inference: inference appends
  "upcoming" rows with unknown outcomes, so features can't see the match they predict.

Model: one XGBoost regressor with position as a feature, saved in XGBoost's native
JSON format (no pickle/scikit-learn version coupling). See train_points_model().
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_CSV = "cleaned_merged_seasons_cleaned.csv"
# Bundled with the code; the weekly retrain (pipeline/retrain.py) writes newer copies to the
# live data volume, which take precedence (see active_model_paths)
MODEL_PATH = os.path.join("ml", "models", "points_v3.json")
META_PATH = os.path.join("ml", "models", "points_v3_meta.json")


def live_model_dir() -> str:
    return os.path.join(os.getenv("LIVE_DATA_DIR", os.path.join("data", "live")), "models")


def active_model_paths():
    """(model, meta) to serve: the retrained copy in LIVE_DATA_DIR/models if present, else bundled."""
    live = (os.path.join(live_model_dir(), "points_v3.json"), os.path.join(live_model_dir(), "points_v3_meta.json"))
    return live if all(os.path.exists(p) for p in live) else (MODEL_PATH, META_PATH)

POSITIONS = ["GK", "DEF", "MID", "FWD"]

# Per-match stats that become rolling "form" features (never used for the same match)
FORM_STATS = [
    "total_points", "minutes", "started", "played", "goals_scored", "assists", "bonus", "bps",
    "ict_index", "influence", "creativity", "threat", "clean_sheets", "goals_conceded", "saves",
    "expected_goals", "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
    "defensive_contribution",
]
WINDOWS = (3, 6)

# Promoted/new teams have no rolling history: start them a bit below average
NEW_TEAM_PRIOR = {"team_gf": 1.1, "team_ga": 1.7, "team_xgf": 1.1, "team_xga": 1.7}

HISTORY_COLUMNS = [
    "season", "name", "position", "team", "fixture", "kickoff_time", "GW", "was_home",
    "team_h_score", "team_a_score", "total_points", "minutes", "starts", "goals_scored",
    "assists", "bonus", "bps", "ict_index", "influence", "creativity", "threat", "clean_sheets",
    "goals_conceded", "saves", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "defensive_contribution", "value", "selected", "xP",
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


def build_features(history: pd.DataFrame, upcoming: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Feature table for every history row (+ upcoming rows, whose outcomes are unknown).

    `upcoming` needs: season, name, position, team, opponent, fixture, kickoff_time, GW,
    was_home, value, and optionally selected_rank (0-1) and xP.
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

    df["played"] = (df["minutes"] > 0).astype(float).where(df["minutes"].notna())
    started = df["starts"].where(df["starts"].notna(), (df["minutes"] >= 60).astype(float))
    df["started"] = started.where(df["minutes"].notna())

    # ---- player form, earlier gameweeks only
    form = _rolling_before_gameweek(df, "name", FORM_STATS, WINDOWS)
    df = df.join(form)

    # Season to date: points per match played, appearances, minutes share
    df = df.sort_values(["name", "kickoff_time"])
    g = df.groupby(["name", "season"], sort=False)
    df["season_matches_before"] = g.cumcount()
    df["season_points_before"] = g["total_points"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    df["season_minutes_before"] = g["minutes"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    df["season_ppg"] = df["season_points_before"] / df["season_matches_before"].replace(0, np.nan)
    df["season_minutes_share"] = df["season_minutes_before"] / (90 * df["season_matches_before"].replace(0, np.nan))
    df["career_matches_before"] = df.groupby("name", sort=False).cumcount()
    df["days_rest"] = df.groupby("name", sort=False)["kickoff_time"].diff().dt.total_seconds() / 86400
    df["days_rest"] = df["days_rest"].clip(upper=30)
    # Freeze season-to-date at the gameweek's first fixture too
    first_in_gw = df.groupby(["name", "season", "GW"], sort=False).cumcount() == 0
    for col in ["season_matches_before", "season_ppg", "season_minutes_share", "career_matches_before"]:
        df[col] = df[col].where(first_in_gw)
        df[col] = df.groupby(["name", "season", "GW"], sort=False)[col].transform("first")

    # ---- team strength (own team and opponent), earlier gameweeks only
    tm = _team_match_table(df[~df["is_upcoming"]])
    tm_cols = ["gf", "ga", "xgf", "xga"]
    team_form = _rolling_before_gameweek(tm, "team", tm_cols, (6,))
    tm = tm.join(team_form)
    # Upcoming fixtures have no team-match rows: give each team its form after its last match
    last = tm.sort_values("kickoff_time").groupby("team").tail(1)
    latest = {}
    for row in last.itertuples(index=False):
        # Form including the last match = rolling over its last 6 incl. that one
        hist = tm[tm["team"] == row.team].sort_values("kickoff_time").tail(6)
        latest[row.team] = {c: hist[c].mean() for c in tm_cols}
    keyed = tm.set_index(["season", "fixture", "team"])[[f"{c}_r6" for c in tm_cols]]

    def team_feats(team_col: str, prefix: str) -> pd.DataFrame:
        idx = pd.MultiIndex.from_frame(df[["season", "fixture", team_col]].rename(columns={team_col: "team"}))
        vals = keyed.reindex(idx).to_numpy()
        feats = pd.DataFrame(vals, index=df.index, columns=[f"{prefix}_{c}" for c in tm_cols])
        up = df["is_upcoming"].to_numpy()
        for c in tm_cols:
            col = f"{prefix}_{c}"
            if up.any():
                fill = df.loc[up, team_col].map(lambda t: latest.get(t, {}).get(c, np.nan))
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

    return df.sort_values(["kickoff_time", "fixture", "name"]).reset_index(drop=True)


def feature_columns() -> List[str]:
    cols = [f"{c}_r{w}" for w in WINDOWS for c in FORM_STATS]
    cols += ["season_matches_before", "season_ppg", "season_minutes_share", "career_matches_before", "days_rest"]
    cols += [f"{p}_{c}" for p in ("team", "opp") for c in ("gf", "ga", "xgf", "xga")]
    # FPL's xP is deliberately NOT a feature: in the historical files it was recorded inside
    # the gameweek (2023-24/2024-25: ranking by xP alone picks ~7 pts/player vs ~4.4 for any
    # pre-deadline feature, i.e. it knows lineups) and is inconsistent in 2025-26, so it
    # doesn't match the pre-deadline ep_next available at prediction time.
    cols += ["fixtures_in_gw", "was_home", "price", "selected_rank"]
    cols += [f"pos_{p}" for p in POSITIONS]
    return cols


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
        params: Optional[dict] = None):
    xgb = _xgb()
    cols = feature_columns()
    model = xgb.XGBRegressor(n_estimators=n_estimators, early_stopping_rounds=100 if valid is not None else None,
                             **(params or XGB_PARAMS))
    eval_set = [(valid[cols], valid["total_points"])] if valid is not None else None
    model.fit(train[cols], train["total_points"], eval_set=eval_set, verbose=False)
    return model


def save_model(model, meta: dict, model_path: str = MODEL_PATH, meta_path: str = META_PATH) -> None:
    model.get_booster().save_model(model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({**meta, "features": feature_columns()}, f, indent=2)


def load_model(model_path: Optional[str] = None, meta_path: Optional[str] = None):
    if model_path is None or meta_path is None:
        model_path, meta_path = active_model_paths()
    xgb = _xgb()
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return booster, meta


def predict(booster, meta: dict, features: pd.DataFrame) -> np.ndarray:
    xgb = _xgb()
    return booster.predict(xgb.DMatrix(features[meta["features"]]))
