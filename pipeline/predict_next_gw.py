"""
Regenerate next-gameweek point predictions from fresh data.

    python -m pipeline.predict_next_gw [--live-dir data/live] [--backtest GW]

Inputs:
  * FPL API: the next gameweek (bootstrap event with `is_next`) and its fixtures.
  * LIVE_DATA_DIR (written by `python -m pipeline.current_season`): this season's
    finished matches (FPL_YYYY_YYYY.csv) and every current player's status/news.
  * cleaned_merged_seasons_cleaned.csv: the last HISTORY_SEASONS historical seasons.
  * The production models the API loads: ml/models/xgboost_{gk,def,mid,fwd}_v2.pkl
    and their *_mappings.json, through ml.feature_engineering.FeatureEngineer.

Output: LIVE_DATA_DIR/next_gw_predictions.csv (+ .json), same schema as the old
ml/predictions/next_gw_predictions.csv plus a few transparency columns, written
atomically. services/predictions.py prefers this file over the bundled one.

How a prediction is built (see ml/README.md for the model itself):
  The v2 models score one player-match row. Their strongest inputs are that match's
  own stat line (influence, ICT, minutes, team goals), so for the upcoming match each
  player gets a synthetic row whose per-match stats are his average over his last
  RECENT_MATCHES matches, appended to his real match history so FeatureEngineer
  computes the history features (form, minutes_rolling5, points_per_90, home/away
  form) exactly as in training: from earlier rows only. Fixture context (home/away,
  gameweek, days of rest, double gameweek) comes from the real next fixture, and the
  team-level inputs come from both teams' recent goals for/against, so the opponent
  matters. Only matches from before the predicted gameweek are read (no leakage).

  * Double gameweek: one row per fixture, predictions summed.
  * Blank gameweek: the player is kept with 0 points and opponent "No fixture".
  * Availability: status u/n, or chance_of_playing_next_round 0 -> 0 points;
    otherwise points x chance/100 (the API's minutes model is only informational, so
    nothing else discounts availability).
  * No match rows at all (rare: e.g. a signing not yet in any squad list): the
    promoted-teams cold-start heuristic (position baseline) from
    ml/promoted_teams_handler.py. Promoted-team players with matches this season go
    through the model like everyone else.
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.current_season import LIVE_DATA_DIR, next_event, season_file_stem, season_label
from pipeline.team_names import graph_team_name

logger = logging.getLogger(__name__)

HISTORY_CSV = "cleaned_merged_seasons_cleaned.csv"
MODELS_DIR = os.path.join("ml", "models")
OUTPUT_NAME = "next_gw_predictions.csv"
POSITIONS = ("GK", "DEF", "MID", "FWD")
MODEL_METHOD = "xgboost_v2"
COLD_START_METHOD = "cold_start_heuristic"
NO_FIXTURE_METHOD = "no_fixture"
NO_FIXTURE_OPPONENT = "No fixture"

HISTORY_SEASONS = 2      # historical seasons read besides the live one
HISTORY_ROWS = 15        # most recent match rows kept per player (form windows are <= 5)
RECENT_MATCHES = 5       # matches averaged into the synthetic next-match stat line
TEAM_MATCHES = 5         # matches used for a team's goals for/against
DEFAULT_TEAM_GOALS = 1.4

# Per-match stats the models read directly; the synthetic row gets recent averages.
PER_MATCH_STATS = [
    "minutes", "creativity", "ict_index", "influence", "threat",
    "clearances_blocks_interceptions", "defensive_contribution", "recoveries", "tackles",
]
HISTORY_COLUMNS = [
    "season_x", "name", "position", "team_x", "kickoff_time", "GW", "was_home",
    "total_points", "value", "team_h_score", "team_a_score", "fixture",
] + PER_MATCH_STATS
OUTPUT_COLUMNS = [
    "name", "position", "team", "predicted_points", "value", "opponent", "is_home",
    "gameweek", "kickoff_time", "prediction_method", "is_promoted_team",
    # transparency columns (consumers ignore unknown columns)
    "fixtures", "status", "chance_of_playing_next_round", "news", "element",
]


# ------------------------------------------------------------------ pure helpers

def next_gameweek(bootstrap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The upcoming gameweek (FPL's `is_next` event), or None after the last deadline."""
    return next_event(bootstrap)


def team_name_map(bootstrap: Dict[str, Any]) -> Dict[int, str]:
    return {t["id"]: graph_team_name(t["name"]) for t in bootstrap["teams"]}


def gameweek_fixtures(fixtures: Iterable[Dict[str, Any]], event_id: int,
                      teams: Dict[int, str]) -> Dict[str, List[Dict[str, Any]]]:
    """Each team's fixtures in gameweek `event_id`, in kickoff order.

    Teams with two fixtures (double gameweek) get two entries; teams without one
    (blank gameweek) are absent. Unscheduled fixtures (event None) are ignored.
    """
    by_team: Dict[str, List[Dict[str, Any]]] = {}
    for f in fixtures:
        if f.get("event") != event_id:
            continue
        home, away = teams[f["team_h"]], teams[f["team_a"]]
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            by_team.setdefault(team, []).append({
                "fixture_id": f["id"], "opponent": opponent, "is_home": is_home,
                "kickoff_time": f.get("kickoff_time"),
            })
    for entries in by_team.values():
        entries.sort(key=lambda e: (e["kickoff_time"] or "", e["fixture_id"]))
    return by_team


def availability_factor(status: Any, chance_next: Any) -> float:
    """Multiplier for a player's predicted points from his FPL status.

    u (left / unavailable) and n (not in squad) -> 0. Otherwise FPL's
    chance_of_playing_next_round / 100 when set (0 for ruled out); injured or
    suspended players without a chance value are treated as out; else 1.
    """
    status = str(status or "a").lower()
    if status in ("u", "n"):
        return 0.0
    if chance_next is not None and not pd.isna(chance_next):
        return float(np.clip(float(chance_next) / 100.0, 0.0, 1.0))
    if status in ("i", "s"):
        return 0.0
    return 1.0


def describe_fixtures(entries: List[Dict[str, Any]]) -> Tuple[str, bool, str]:
    """(opponent, is_home, kickoff_time) columns for one player's gameweek.

    A double gameweek joins the opponents ("Chelsea & Everton"); home/away and
    kickoff are those of the first fixture.
    """
    first = entries[0]
    opponent = " & ".join(e["opponent"] for e in entries)
    return opponent, bool(first["is_home"]), first["kickoff_time"]


def congestion_risk(days_rest: float, matches_14d: float, minutes_rolling5: float) -> float:
    """FeatureEngineer._add_fixture_congestion_features, for one row."""
    risk = 0.4 * (days_rest <= 3) + 0.35 * min(matches_14d / 6.0, 1.0) + 0.25 * (minutes_rolling5 > 400)
    return float(min(max(risk, 0.0), 1.0))


def expected_goals(team: str, opponent: str, form: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """(goals for, goals against) expected for `team` vs `opponent`.

    Average of the team's recent scoring and the opponent's recent conceding, and
    vice versa; `form` maps team -> (avg scored, avg conceded).
    """
    t_for, t_against = form.get(team, (DEFAULT_TEAM_GOALS, DEFAULT_TEAM_GOALS))
    o_for, o_against = form.get(opponent, (DEFAULT_TEAM_GOALS, DEFAULT_TEAM_GOALS))
    return (t_for + o_against) / 2.0, (t_against + o_for) / 2.0


def team_goal_form(history: pd.DataFrame, season: str, n: int = TEAM_MATCHES) -> Dict[str, Tuple[float, float]]:
    """team -> (avg goals scored, avg conceded) over its last `n` matches.

    Uses this season's matches, topped up with the previous seasons' when a team
    has fewer than `n` so far.
    """
    df = history.dropna(subset=["team_h_score", "team_a_score"])
    if df.empty:
        return {}
    home = df["was_home"].astype(bool)
    matches = pd.DataFrame({
        "team": df["team_x"], "season": df["season_x"], "kickoff": df["kickoff_time"],
        "scored": np.where(home, df["team_h_score"], df["team_a_score"]),
        "conceded": np.where(home, df["team_a_score"], df["team_h_score"]),
    }).drop_duplicates(subset=["team", "season", "kickoff"])
    matches["current"] = matches["season"] == season
    matches = matches.sort_values(["team", "current", "kickoff"])
    recent = matches.groupby("team").tail(n)
    agg = recent.groupby("team")[["scored", "conceded"]].mean()
    return {team: (float(r.scored), float(r.conceded)) for team, r in agg.iterrows()}


def _atomic_write(path: str, write) -> None:
    tmp = f"{path}.tmp"
    write(tmp)
    os.replace(tmp, path)


def peak_memory_mb() -> Optional[float]:
    """Peak resident memory of this process so far (Linux: ru_maxrss; else psutil if present)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        pass
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / (1024.0 * 1024.0)
    except ImportError:
        return None


# ------------------------------------------------------------------ data loading

def live_paths(live_dir: str, season: str) -> Tuple[str, str]:
    """(season matches CSV, player status CSV) in `live_dir`, preferring meta.json's names."""
    files = {}
    meta_path = os.path.join(live_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("season") == season:
            files = meta.get("files") or {}
    season_csv = files.get("season") or f"FPL_{season_file_stem(season)}.csv"
    status_csv = files.get("status") or f"player_status_{season.replace('-', '_')}.csv"
    return os.path.join(live_dir, season_csv), os.path.join(live_dir, status_csv)


class History(NamedTuple):
    rows: pd.DataFrame            # current players' match rows, keyed by `name` = element id
    previous_teams: List[str]     # teams of the last historical season (for is_promoted_team)


def load_history(players: pd.DataFrame, live_csv: str, season: str, before_gw: int,
                 before_time: Optional[str], history_csv: str = HISTORY_CSV,
                 seasons: int = HISTORY_SEASONS) -> History:
    """Match rows of the current players, sorted by player then kickoff.

    This season's rows are matched by element id; historical rows by normalized
    name (only names unique among current players). Of this season only rows from
    before `before_gw` that kicked off before `before_time` are kept (leakage guard).
    """
    from utils.text import normalize_name

    live = pd.read_csv(live_csv, usecols=lambda c: c in set(HISTORY_COLUMNS) | {"element", "team"})
    live = live[live["GW"] < before_gw]
    if before_time is not None and len(live):
        live = live[pd.to_datetime(live["kickoff_time"], utc=True) < pd.to_datetime(before_time, utc=True)]
    live = live[live["element"].isin(players["element"])].rename(columns={"team": "team_x"})
    live["season_x"] = season
    live["name"] = live["element"].astype(int).astype(str)

    frames, previous_teams = [live], []
    if seasons > 0 and os.path.exists(history_csv):
        hist = pd.read_csv(history_csv, usecols=lambda c: c in set(HISTORY_COLUMNS))
        past = sorted(s for s in hist["season_x"].dropna().unique() if s < season)[-seasons:]
        hist = hist[hist["season_x"].isin(past)]
        if past:
            previous_teams = sorted(hist.loc[hist["season_x"] == past[-1], "team_x"].dropna().unique())
        norm = players["name"].map(normalize_name)
        unique = norm[~norm.duplicated(keep=False)]
        key_by_norm = dict(zip(unique, players.loc[unique.index, "element"].astype(int).astype(str)))
        hist["name"] = hist["name"].map(lambda n: key_by_norm.get(normalize_name(n)))
        frames.append(hist[hist["name"].notna()])

    history = pd.concat(frames, ignore_index=True)
    history = history[[c for c in HISTORY_COLUMNS if c in history.columns]]
    history["was_home"] = history["was_home"].astype(str).str.lower().eq("true")
    history["kickoff_time"] = pd.to_datetime(history["kickoff_time"], utc=True)
    history = history.sort_values(["name", "kickoff_time"]).reset_index(drop=True)
    return History(history, previous_teams)


# ------------------------------------------------------------------ models / features

class PositionModels:
    """The production v2 XGBoost models plus each one's FeatureEngineer mappings."""

    def __init__(self, models_dir: str = MODELS_DIR):
        from ml.feature_engineering import FeatureEngineer

        self.models: Dict[str, Any] = {}
        self.engineers: Dict[str, Any] = {}
        for pos in POSITIONS:
            path = os.path.join(models_dir, f"xgboost_{pos.lower()}_v2.pkl")
            with open(path, "rb") as f:
                self.models[pos] = pickle.load(f)
            fe = FeatureEngineer()
            fe.load_mappings(path.replace(".pkl", "_mappings.json"))
            self.engineers[pos] = fe

    def predict(self, pos: str, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.models[pos].predict(X[self.engineers[pos].feature_names]), 0, None)


def synthetic_rows(players: pd.DataFrame, history: pd.DataFrame, fixtures_by_team: Dict[str, List[Dict[str, Any]]],
                   season: str, gameweek: int) -> pd.DataFrame:
    """One upcoming-match row per player with a fixture, in the history's format.

    Per-match stats are the player's mean over his last RECENT_MATCHES rows; the
    context is his team's first fixture of the gameweek.
    """
    recent = history.groupby("name").tail(RECENT_MATCHES)
    means = recent.groupby("name")[PER_MATCH_STATS].mean()
    rows = []
    for p in players.itertuples(index=False):
        entries = fixtures_by_team.get(p.team)
        if not entries:
            continue
        key = str(int(p.element))
        first = entries[0]
        row = {
            "season_x": season, "name": key, "position": p.position, "team_x": p.team,
            "kickoff_time": pd.to_datetime(first["kickoff_time"], utc=True), "GW": gameweek,
            "was_home": bool(first["is_home"]), "total_points": 0.0, "value": float(p.now_cost),
            "team_h_score": np.nan, "team_a_score": np.nan, "fixture": first["fixture_id"],
        }
        stats = means.loc[key] if key in means.index else None
        for col in PER_MATCH_STATS:
            v = stats[col] if stats is not None else 0.0
            row[col] = 0.0 if pd.isna(v) else float(v)
        rows.append(row)
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def build_features(models: PositionModels, history: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the upcoming rows on top of each player's history.

    Returns one row per upcoming player (indexed by key) with the model feature
    columns. FeatureEngineer computes the history features from earlier rows only
    (shift(1)), so the synthetic row's own placeholder values never leak into them.
    Opponent columns are left out so FeatureEngineer skips its row-multiplying
    opponent merges; those team-level features are set per fixture afterwards.
    """
    fe = models.engineers["MID"]  # same feature list and team categories for every position
    frame = pd.concat([history.assign(_upcoming=False), upcoming.assign(_upcoming=True)], ignore_index=True)
    fe_logger = logging.getLogger("ml.feature_engineering")
    level = fe_logger.level
    fe_logger.setLevel(logging.ERROR)
    try:
        eng = fe.engineer_features(frame, is_training=False, lag_features=None)
        X, _ = fe.prepare_features(eng, include_target=False)
    finally:
        fe_logger.setLevel(level)
    mask = eng["_upcoming"].astype(bool)
    X = X.loc[mask].copy()
    X.index = eng.loc[mask, "name"].values
    return X


def predict_players(models: PositionModels, players: pd.DataFrame, history: pd.DataFrame,
                    fixtures_by_team: Dict[str, List[Dict[str, Any]]], season: str,
                    gameweek: int, team_form: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, Any]]:
    """key -> {'points': summed model points over the gameweek's fixtures, 'method'}."""
    history = history.groupby("name").tail(HISTORY_ROWS)  # form windows need <= 5 earlier rows
    upcoming = synthetic_rows(players, history, fixtures_by_team, season, gameweek)
    if upcoming.empty:
        return {}
    has_history = set(history["name"].unique())
    last_kickoff = history.groupby("name")["kickoff_time"].max()
    X = build_features(models, history, upcoming)
    position = dict(zip(upcoming["name"], upcoming["position"]))
    team = dict(zip(upcoming["name"], upcoming["team_x"]))

    rows, owners = [], []
    for key, base in X.iterrows():
        entries = fixtures_by_team[team[key]]
        n = len(entries)
        prev = last_kickoff.get(key)
        for e in entries:
            kickoff = pd.to_datetime(e["kickoff_time"], utc=True)
            days = 7.0 if prev is None or pd.isna(prev) else float((kickoff - prev).days)
            exp_for, exp_against = expected_goals(team[key], e["opponent"], team_form)
            r = base.copy()
            r["was_home"] = int(e["is_home"])
            r["GW"] = gameweek
            r["gw_in_season"] = gameweek / 38.0
            r["team_goals"] = exp_for
            r["opp_def_strength"] = exp_for        # what the opponent concedes
            r["team_def_strength"] = exp_against   # what the team concedes
            r["opp_off_strength"] = exp_against    # what the opponent scores
            r["fixtures_this_gw"] = n
            r["days_since_last_match"] = days
            r["matches_in_next_14_days"] = 4 if n >= 2 else 2
            r["fixture_congestion_risk"] = congestion_risk(days, r["matches_in_next_14_days"], r["minutes_rolling5"])
            rows.append(r)
            owners.append(key)
            prev = kickoff

    feats = pd.DataFrame(rows)
    feats["_key"] = owners
    out: Dict[str, Dict[str, Any]] = {}
    for pos, group in feats.groupby(feats["_key"].map(position)):
        preds = models.predict(pos, group.drop(columns="_key"))
        for key, pts in pd.Series(preds, index=group["_key"].values).groupby(level=0).sum().items():
            out[key] = {"points": float(pts), "method": MODEL_METHOD}

    # Players with no match rows at all: promoted-teams cold-start heuristic.
    cold = [k for k in out if k not in has_history]
    if cold:
        from ml.promoted_teams_handler import PromotedTeamsHandler
        handler = PromotedTeamsHandler()
        for key in cold:
            n = len(fixtures_by_team[team[key]])
            pts = handler.calculate_blended_heuristic({"position": position[key], "form": 0.0}, gameweek)
            out[key] = {"points": float(pts) * n, "method": COLD_START_METHOD}
    return out


# ------------------------------------------------------------------ assembly

def assemble_output(players: pd.DataFrame, predictions: Dict[str, Dict[str, Any]],
                    fixtures_by_team: Dict[str, List[Dict[str, Any]]], gameweek: int,
                    deadline: Optional[str], promoted_teams: Iterable[str]) -> pd.DataFrame:
    """Final table in the next_gw_predictions schema, one row per current player."""
    promoted = set(promoted_teams)
    rows = []
    for p in players.itertuples(index=False):
        key = str(int(p.element))
        entries = fixtures_by_team.get(p.team) or []
        if entries:
            opponent, is_home, kickoff = describe_fixtures(entries)
            pred = predictions.get(key, {"points": 0.0, "method": MODEL_METHOD})
            points = pred["points"] * availability_factor(p.status, p.chance_of_playing_next_round)
            method = pred["method"]
        else:
            opponent, is_home, kickoff, points, method = NO_FIXTURE_OPPONENT, False, deadline, 0.0, NO_FIXTURE_METHOD
        rows.append({
            "name": p.name, "position": p.position, "team": p.team,
            "predicted_points": round(float(points), 2), "value": int(p.now_cost),
            "opponent": opponent, "is_home": is_home, "gameweek": int(gameweek),
            "kickoff_time": kickoff, "prediction_method": method,
            "is_promoted_team": p.team in promoted, "fixtures": len(entries),
            "status": p.status, "chance_of_playing_next_round": p.chance_of_playing_next_round,
            "news": p.news if isinstance(p.news, str) else "", "element": int(p.element),
        })
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return df.sort_values(["predicted_points", "name"], ascending=[False, True]).reset_index(drop=True)


def promoted_teams(previous_teams: Iterable[str], teams: Iterable[str]) -> List[str]:
    """Teams this season that were not in the previous historical season."""
    previous = set(previous_teams)
    if not previous:
        return []
    return sorted(t for t in set(teams) if t not in previous)


def _load_players(status_csv: str) -> pd.DataFrame:
    players = pd.read_csv(status_csv)
    players = players[players["position"].isin(POSITIONS)].copy()
    players["news"] = players["news"].fillna("")
    return players.reset_index(drop=True)


def _context(client, live_dir: str, gameweek: Optional[int] = None):
    bootstrap = client.bootstrap()
    fixtures = client.fixtures()
    season = season_label(bootstrap)
    if gameweek is None:
        event = next_gameweek(bootstrap)
        if event is None:
            raise RuntimeError("No upcoming gameweek: the season is over")
    else:
        event = next(e for e in bootstrap["events"] if e["id"] == gameweek)
    fixtures_by_team = gameweek_fixtures(fixtures, event["id"], team_name_map(bootstrap))
    first_kickoff = min((e["kickoff_time"] for es in fixtures_by_team.values() for e in es if e["kickoff_time"]),
                        default=event["deadline_time"])
    return season, event, fixtures_by_team, first_kickoff


def predict_next_gameweek(live_dir: str = LIVE_DATA_DIR, client=None,
                          history_csv: str = HISTORY_CSV, models_dir: str = MODELS_DIR) -> Dict[str, Any]:
    """Predict every current player's points for the next gameweek and write the CSV/JSON.

    Returns a summary: gameweek, deadline, row counts, output paths, timing, memory.
    """
    started = time.perf_counter()
    own_client = client is None
    if own_client:
        from pipeline.fpl_api import FPLClient
        client = FPLClient()
    try:
        season, event, fixtures_by_team, first_kickoff = _context(client, live_dir)
    finally:
        if own_client:
            client.close()
    gameweek = int(event["id"])

    live_csv, status_csv = live_paths(live_dir, season)
    players = _load_players(status_csv)
    # Leakage guard: nothing from the predicted gameweek or later feeds the features.
    history, previous_teams = load_history(players, live_csv, season, before_gw=gameweek,
                                           before_time=first_kickoff, history_csv=history_csv)
    models = PositionModels(models_dir)
    predictions = predict_players(models, players, history, fixtures_by_team, season, gameweek,
                                  team_goal_form(history, season))
    promoted = promoted_teams(previous_teams, players["team"])
    out = assemble_output(players, predictions, fixtures_by_team, gameweek, event["deadline_time"], promoted)

    os.makedirs(live_dir, exist_ok=True)
    csv_path = os.path.join(live_dir, OUTPUT_NAME)
    json_path = csv_path[:-4] + ".json"
    _atomic_write(json_path, lambda p: out.to_json(p, orient="records", indent=2))
    _atomic_write(csv_path, lambda p: out.to_csv(p, index=False))  # CSV last: it is what the API watches

    has_fixture = out["fixtures"] > 0
    return {
        "season": season,
        "gameweek": gameweek,
        "deadline": event["deadline_time"],
        "players": int(len(out)),
        "players_with_fixture": int(has_fixture.sum()),
        "players_unavailable": int((has_fixture & (out["predicted_points"] == 0)
                                    & out["status"].isin(["i", "s", "u", "n"])).sum()),
        "double_gameweek_players": int((out["fixtures"] > 1).sum()),
        "blank_teams": sorted(set(players["team"]) - set(fixtures_by_team)),
        "promoted_teams": promoted,
        "cold_start_players": int((out["prediction_method"] == COLD_START_METHOD).sum()),
        "history_rows": int(len(history)),
        "history_last_kickoff": str(history["kickoff_time"].max()) if len(history) else None,
        "output": csv_path,
        "output_json": json_path,
        "seconds": round(time.perf_counter() - started, 1),
        "peak_memory_mb": (round(peak_memory_mb(), 1) if peak_memory_mb() is not None else None),
    }


def backtest(gameweek: int, live_dir: str = LIVE_DATA_DIR, client=None,
             history_csv: str = HISTORY_CSV, models_dir: str = MODELS_DIR) -> Dict[str, Any]:
    """Predict an already-played gameweek of this season from earlier data only and score it.

    No availability adjustment (past statuses are not stored). Compares against the
    naive "average of the last 5 matches" baseline.
    """
    own_client = client is None
    if own_client:
        from pipeline.fpl_api import FPLClient
        client = FPLClient()
    try:
        season, event, fixtures_by_team, first_kickoff = _context(client, live_dir, gameweek)
    finally:
        if own_client:
            client.close()
    live_csv, status_csv = live_paths(live_dir, season)
    players = _load_players(status_csv)
    history, _ = load_history(players, live_csv, season, before_gw=gameweek, before_time=first_kickoff,
                              history_csv=history_csv)
    preds = predict_players(PositionModels(models_dir), players, history, fixtures_by_team, season, gameweek,
                            team_goal_form(history, season))
    live = pd.read_csv(live_csv, usecols=["element", "GW", "total_points"])
    actual = live[live["GW"] == gameweek].groupby("element")["total_points"].sum()
    actual.index = actual.index.astype(str)
    naive = history.groupby("name").tail(RECENT_MATCHES).groupby("name")["total_points"].mean()
    df = pd.DataFrame({"pred": pd.Series({k: v["points"] for k, v in preds.items()})})
    df["actual"] = actual
    df["naive"] = naive
    df = df.dropna(subset=["actual"]).fillna({"naive": 0.0})
    return {
        "gameweek": gameweek,
        "players": int(len(df)),
        "history_last_kickoff": str(history["kickoff_time"].max()),
        "first_kickoff": first_kickoff,
        "model_mae": round(float((df.pred - df.actual).abs().mean()), 3),
        "naive_mae": round(float((df.naive - df.actual).abs().mean()), 3),
        "model_spearman": round(float(df.pred.corr(df.actual, method="spearman")), 3),
        "naive_spearman": round(float(df.naive.corr(df.actual, method="spearman")), 3),
        "model_pearson": round(float(df.pred.corr(df.actual)), 3),
        "naive_pearson": round(float(df.naive.corr(df.actual)), 3),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate next-gameweek predictions from fresh data")
    parser.add_argument("--live-dir", default=LIVE_DATA_DIR)
    parser.add_argument("--backtest", type=int, metavar="GW",
                        help="Score a played gameweek of this season instead of writing predictions")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.backtest:
        result = backtest(args.backtest, args.live_dir)
    else:
        result = predict_next_gameweek(args.live_dir)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
