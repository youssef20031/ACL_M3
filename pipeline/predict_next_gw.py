"""
Regenerate next-gameweek point predictions from fresh data.

    python -m pipeline.predict_next_gw [--live-dir data/live] [--backtest GW] [--no-odds-fetch]

Inputs:
  * FPL API: the next gameweek (bootstrap event with `is_next`) and its fixtures.
  * LIVE_DATA_DIR (written by `python -m pipeline.current_season`): this season's
    finished matches (FPL_YYYY_YYYY.csv) and every current player's status/news, FPL code
    and this gameweek's transfers in/out, plus our pre-deadline element_snapshots archive.
  * LIVE_DATA_DIR/odds: football-data.co.uk's odds, refreshed at the start of every run
    (pipeline.current_season.refresh_odds: this season's file and fixtures.csv, earlier
    seasons once). A failed download only leaves the odds features NaN.
  * data/cleaned_merged_seasons_cleaned.csv: all historical seasons (the last 3 for the
    features, plus the small team-match table that team ratings start from).
  * The served points model, v4 (ml.points_model.load_model(): LIVE_DATA_DIR/models/points_v4.json
    from the weekly retrain, else the bundled ml/models/points_v4.json; v3 until a v4 exists).

Output: LIVE_DATA_DIR/next_gw_predictions.csv (+ .json), same schema as the old
ml/predictions/next_gw_predictions.csv plus a few transparency columns, written
atomically. services/predictions.py prefers this file over the bundled one. Before the
deadline it also saves the gameweek's squad pick (services/model_review.save_picks) for
the Team of the Week page's "Model vs actual" review, and every run writes the model's
inputs for each upcoming row (next_gw_features.csv), which services/explain.py turns into
SHAP and LIME explanations for the /model page.

Shadow: the same run also scores the previous production model, v3 (LIVE_DATA_DIR/models/
points_v3.json if present, else the bundled one; no longer retrained), on the same feature rows
with the same chance-of-playing multiplier, and saves its squad pick before the deadline in
model_picks/<season>/shadow/ (same files and selection as the live pick). Only the live
predictions and pick reach the site; scripts/analysis/shadow_ledger.py compares the two picks
gameweek by gameweek, on gameweeks nobody selected on. A failing shadow never affects the live
predictions.

How a prediction is built: each player gets one row per fixture of the predicted
gameweek and goes through ml.points_model.build_features together with his real
match history, so features are computed by exactly the code used in training and
only from matches before the gameweek (rows from the predicted gameweek or later
are dropped before anything is computed).

  * Double gameweek: one row per fixture, predictions summed.
  * Blank gameweek: the player is kept with 0 points and opponent "No fixture".
  * Availability: status u/n, or chance_of_playing_next_round 0 -> 0 points;
    otherwise points x chance/100. v4 also sees FPL's availability before the deadline
    as features (snap_*, from the status file), but the history has few pre-deadline
    snapshots, so this multiplier stays on top, as in its backtest.
  * Players are keyed by FPL code (the status file's `code`), so a renamed player keeps
    his history; the shadow predicts from the same rows.
"""
import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml import points_model as pm
from pipeline.current_season import (
    LIVE_DATA_DIR, next_event, odds_dir, refresh_odds, season_file_stem, season_label,
)
from pipeline.team_names import graph_team_name
from services.season_service import chance_of_playing

logger = logging.getLogger(__name__)

HISTORY_CSV = pm.HISTORY_CSV
OUTPUT_NAME = "next_gw_predictions.csv"
FEATURES_NAME = "next_gw_features.csv"   # model inputs per upcoming row, for services/explain.py
FEATURE_ID_COLUMNS = ["element", "name", "position", "team", "opponent", "fixture", "kickoff_time", "GW", "season"]
POSITIONS = ("GK", "DEF", "MID", "FWD")
MODEL_METHOD = f"xgboost_{pm.MODEL_VERSION}"
NO_FIXTURE_METHOD = "no_fixture"
NO_FIXTURE_OPPONENT = "No fixture"
RECENT_MATCHES = 5       # matches averaged by the backtest's naive baseline
HISTORY_SEASONS = 3      # past seasons fed to the features at prediction time
# Share of upcoming rows with each v4 input, reported per run: odds are NaN before
# football-data.co.uk's Friday/Tuesday collection, snapshots come from the status file
COVERAGE_COLUMNS = {"odds": "odds_win", "snapshot": "snap_age_days", "transfers": "tr_bal_rank",
                    "ratings": "rt_att"}

PredictFn = Callable[[pd.DataFrame], np.ndarray]

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
    suspended players without a chance value are treated as out; else 1. The rule
    lives in season_service, whose captain pick uses it as the chance of playing.
    """
    return chance_of_playing(status, chance_next)


def describe_fixtures(entries: List[Dict[str, Any]]) -> Tuple[str, bool, str]:
    """(opponent, is_home, kickoff_time) columns for one player's gameweek.

    A double gameweek joins the opponents ("Chelsea & Everton"); home/away and
    kickoff are those of the first fixture.
    """
    first = entries[0]
    opponent = " & ".join(e["opponent"] for e in entries)
    return opponent, bool(first["is_home"]), first["kickoff_time"]


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


def load_history(live_csv: str, season: str, before_gw: int, before_time: Optional[str],
                 history_csv: str = HISTORY_CSV, past_seasons: int = HISTORY_SEASONS) -> pd.DataFrame:
    """The last `past_seasons` historical seasons plus this season's matches from before the
    predicted gameweek. (Measured against using every season: correlation 0.9998, same
    top 20, about half the time and memory.) v4's team ratings still start from every season,
    through load_extras' team-match table.

    Leakage guard: of this season only rows with GW < `before_gw` that kicked off before
    `before_time` (the predicted gameweek's first kickoff) are kept.
    """
    history = pm.load_history(history_csv, live_csv=live_csv, live_season=season)
    current = history["season"] == season
    past = sorted(s for s in history["season"].unique() if s < season)[-past_seasons:]
    history = history[history["season"].isin(past) | current]
    current = history["season"] == season
    late = current & (history["GW"] >= before_gw)
    if before_time is not None:
        late |= current & (history["kickoff_time"] >= pd.to_datetime(before_time, utc=True))
    return history[~late].reset_index(drop=True)


def load_extras(live_dir: str, season: str, status_csv: str, history_csv: str = HISTORY_CSV) -> pm.ExtraInputs:
    """v4's extra inputs (ml.points_model.load_extra_inputs): FPL codes, the odds in odds_dir,
    snapshots (vaastav's, our archive, and the status file as the upcoming gameweek's), and the
    full history's team-match table, so team ratings equal a build from every season. Odds files
    that can't be read are dropped with a warning (odds features NaN) instead of failing."""
    kwargs = dict(live_dir=live_dir, season=season, status_csv=status_csv, history_csv=history_csv)
    odds = odds_dir(live_dir)
    try:
        return pm.load_extra_inputs(odds_dir=odds, **kwargs)
    except Exception:
        logger.warning("Couldn't load the extra inputs with the odds in %s; predicting without odds",
                       odds, exc_info=True)
    return pm.load_extra_inputs(odds_dir=os.path.join(odds, "missing"), **kwargs)   # no folder, no odds


def previous_season_teams(history: pd.DataFrame, season: str) -> List[str]:
    past = sorted(s for s in history["season"].dropna().unique() if s < season)
    return sorted(history.loc[history["season"] == past[-1], "team"].dropna().unique()) if past else []


# ------------------------------------------------------------------ model rows

def _numeric(players: pd.DataFrame, col: str) -> pd.Series:
    if col not in players.columns:
        return pd.Series(np.nan, index=players.index)
    return pd.to_numeric(players[col], errors="coerce")


def upcoming_rows(players: pd.DataFrame, fixtures_by_team: Dict[str, List[Dict[str, Any]]],
                  season: str, gameweek: int) -> pd.DataFrame:
    """One row per player per fixture of the predicted gameweek, in the model's input format.

    Price and ownership are the current pre-deadline values; ownership becomes a rank
    within this gameweek, matching how training ranks `selected` per gameweek. Transfers in
    and out are FPL's counts since the last deadline (the status file's transfers_in_event /
    transfers_out_event), which v4 ranks within the gameweek like the history's counts.
    """
    rank = _numeric(players, "selected_by_percent").rank(pct=True)
    t_in, t_out = _numeric(players, "transfers_in_event"), _numeric(players, "transfers_out_event")
    rows = []
    for p, r, ti, to in zip(players.itertuples(index=False), rank, t_in, t_out):
        for e in fixtures_by_team.get(p.team) or []:
            rows.append({
                "season": season, "name": p.name, "position": p.position, "team": p.team,
                "opponent": e["opponent"], "fixture": e["fixture_id"],
                "kickoff_time": e["kickoff_time"], "GW": gameweek, "was_home": bool(e["is_home"]),
                "value": float(p.now_cost), "selected_rank": r, "element": int(p.element),
                "transfers_in": ti, "transfers_out": to,
            })
    return pd.DataFrame(rows)


def save_model_inputs(rows: pd.DataFrame, path: str) -> None:
    """Write the model's input row for every upcoming fixture, plus its raw output
    (model_points, before the floor at 0), atomically. Feature columns are
    ml.points_model.feature_columns() (v4's, which include v3's); services/explain.py checks
    them against the served model.
    Full precision on purpose: tree splits sit exactly on training values such as 1/3 (a
    3-match mean), and rounding to 6 digits moved 30% of players' predictions by up to 0.6."""
    cols = FEATURE_ID_COLUMNS + ["model_points"] + [c for c in pm.feature_columns() if c in rows.columns]
    out = rows[cols].sort_values(["element", "kickoff_time", "fixture"])
    _atomic_write(path, lambda p: out.to_csv(p, index=False))


def model_rows(players: pd.DataFrame, history: pd.DataFrame,
               fixtures_by_team: Dict[str, List[Dict[str, Any]]], season: str, gameweek: int,
               extras: Optional[pm.ExtraInputs] = None) -> pd.DataFrame:
    """The predicted gameweek's feature rows: one per player per fixture, with `element`.

    Features come from ml.points_model.build_features, the same code used in training;
    the upcoming rows have no outcomes, so their features only see earlier matches.
    `extras` (load_extras) adds v4's inputs and keys players by FPL code; without it the
    rows hold v3's features only.
    """
    upcoming = upcoming_rows(players, fixtures_by_team, season, gameweek)
    if upcoming.empty:
        return upcoming
    feats = pm.build_features(history, upcoming, extras=extras)
    return feats[feats["is_upcoming"]].reset_index(drop=True)


def input_coverage(rows: pd.DataFrame) -> Dict[str, float]:
    """Share of upcoming rows with each v4 input (COVERAGE_COLUMNS)."""
    return {k: round(float(rows[c].notna().mean()), 3) for k, c in COVERAGE_COLUMNS.items()
            if c in rows.columns and len(rows)}


def score_rows(rows: pd.DataFrame, predict_fn: PredictFn, method: str = MODEL_METHOD,
               shadow_fn: Optional[PredictFn] = None, features_path: Optional[str] = None,
               gameweek: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """element (str) -> {'points': model points summed over the gameweek's fixtures, 'method'},
    plus 'shadow_points' (the same from `shadow_fn`) when the shadow model ran.

    Each fixture's prediction is floored at 0 before a double gameweek is summed. With
    `features_path`, the model's inputs are saved there too (save_model_inputs). Neither a
    failure to save nor a failing shadow model is raised; both are logged.
    """
    if rows.empty:
        return {}
    rows = rows.copy()
    rows["model_points"] = predict_fn(rows)
    rows["points"] = np.clip(rows["model_points"], 0.0, None)
    if features_path:
        try:
            save_model_inputs(rows, features_path)
        except Exception:
            logger.exception("Couldn't save the model inputs for GW%s", gameweek)
    out = {str(int(k)): {"points": float(v), "method": method}
           for k, v in rows.groupby("element")["points"].sum().items()}
    if shadow_fn is not None:
        try:
            shadow = pd.Series(np.clip(np.asarray(shadow_fn(rows), dtype=float), 0.0, None), index=rows.index)
            for k, v in shadow.groupby(rows["element"]).sum().items():
                out[str(int(k))]["shadow_points"] = float(v)
        except Exception:
            logger.exception("The shadow model failed for GW%s; the live predictions are unaffected", gameweek)
    return out


def predict_players(players: pd.DataFrame, history: pd.DataFrame,
                    fixtures_by_team: Dict[str, List[Dict[str, Any]]], season: str, gameweek: int,
                    predict_fn: PredictFn, features_path: Optional[str] = None,
                    extras: Optional[pm.ExtraInputs] = None, shadow_fn: Optional[PredictFn] = None,
                    method: str = MODEL_METHOD) -> Dict[str, Dict[str, Any]]:
    """model_rows then score_rows: element (str) -> {'points', 'method'[, 'shadow_points']}.

    Players without any earlier matches still get a prediction from price, position,
    ownership and their team/opponent strength; blank-gameweek players get no entry.
    """
    rows = model_rows(players, history, fixtures_by_team, season, gameweek, extras)
    return score_rows(rows, predict_fn, method, shadow_fn, features_path, gameweek)


def load_model_fn(version: Optional[str] = None) -> Tuple[PredictFn, str]:
    """(predict function, model version): the served model (v4, or v3 until a v4 exists) by
    default, else `version`'s active copy (the retrained one in LIVE_DATA_DIR, else bundled).
    Each model reads its own feature list from its meta."""
    booster, meta = pm.load_model(version=version)
    return (lambda frame: pm.predict(booster, meta, frame)), str(meta.get("version") or version or pm.MODEL_VERSION)


def _shadow_fn(served_version: str) -> Tuple[Optional[PredictFn], str]:
    """(predict function, version) of the shadow model, or (None, why not): there's no v4
    model yet (the served model is the shadow), or the shadow's files don't load."""
    if served_version == pm.SHADOW_VERSION:
        return None, f"skipped: the served model is {served_version}"
    try:
        return load_model_fn(pm.SHADOW_VERSION)
    except Exception as exc:
        logger.exception("Couldn't load the shadow model %s", pm.SHADOW_VERSION)
        return None, f"failed to load: {exc}"[:200]


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


def save_shadow_pick(live_dir: str, season: str, event: Dict[str, Any], players: pd.DataFrame,
                     predictions: Dict[str, Dict[str, Any]], fixtures_by_team: Dict[str, List[Dict[str, Any]]],
                     promoted: Iterable[str], version: str) -> Optional[str]:
    """Assemble the shadow's predictions like the live ones (same availability multiplier and
    blank-gameweek rule) and save its squad pick next to the live pick (model_review.save_picks,
    variant SHADOW). Returns the JSON path, None after the deadline, or "failed"."""
    shadow = {k: {"points": v["shadow_points"], "method": f"xgboost_{version}"}
              for k, v in predictions.items() if "shadow_points" in v}
    if predictions and not shadow:
        return "failed"   # the shadow model raised (logged by score_rows)
    try:
        from services.model_review import SHADOW, save_picks
        gameweek, deadline = int(event["id"]), event["deadline_time"]
        out = assemble_output(players, shadow, fixtures_by_team, gameweek, deadline, promoted)
        return save_picks(live_dir, season, gameweek, deadline, out, fixtures_by_team, variant=SHADOW, model=version)
    except Exception:
        logger.exception("Couldn't save the shadow model's pick for GW%s", event["id"])
        return "failed"


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


def predict_next_gameweek(live_dir: str = LIVE_DATA_DIR, client=None, history_csv: str = HISTORY_CSV,
                          predict_fn: Optional[PredictFn] = None, shadow_fn: Optional[PredictFn] = None,
                          fetch_odds: bool = True) -> Dict[str, Any]:
    """Predict every current player's points for the next gameweek and write the CSV/JSON.

    Without `predict_fn` the served model predicts (load_model_fn) and the shadow model runs
    alongside it; an injected `predict_fn` runs alone unless `shadow_fn` is given too.
    `fetch_odds` first refreshes the odds files (refresh_odds; failures are only logged).
    Returns a summary: gameweek, deadline, model, odds, input coverage, row counts, output
    paths, the live and shadow picks, timing, memory.
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
    odds_status = refresh_odds(live_dir, season) if fetch_odds else "not fetched"

    live_csv, status_csv = live_paths(live_dir, season)
    players = _load_players(status_csv)
    # Leakage guard: nothing from the predicted gameweek or later feeds the features.
    history = load_history(live_csv, season, before_gw=gameweek, before_time=first_kickoff,
                           history_csv=history_csv)
    extras = load_extras(live_dir, season, status_csv, history_csv)
    if predict_fn is None:
        predict_fn, version = load_model_fn()
        if shadow_fn is None:
            shadow_fn, shadow_version = _shadow_fn(version)
        else:
            shadow_version = pm.SHADOW_VERSION
    else:
        version = pm.MODEL_VERSION
        shadow_version = pm.SHADOW_VERSION if shadow_fn is not None else "skipped: no shadow model given"

    os.makedirs(live_dir, exist_ok=True)
    features_path = os.path.join(live_dir, FEATURES_NAME)
    rows = model_rows(players, history, fixtures_by_team, season, gameweek, extras)
    coverage = input_coverage(rows)
    logger.info("GW%s: %d upcoming rows, v4 input coverage %s", gameweek, len(rows), coverage)
    predictions = score_rows(rows, predict_fn, f"xgboost_{version}", shadow_fn, features_path, gameweek)
    del rows, extras
    promoted = promoted_teams(previous_season_teams(history, season), players["team"])
    out = assemble_output(players, predictions, fixtures_by_team, gameweek, event["deadline_time"], promoted)

    csv_path = os.path.join(live_dir, OUTPUT_NAME)
    json_path = csv_path[:-4] + ".json"
    _atomic_write(json_path, lambda p: out.to_json(p, orient="records", indent=2))
    _atomic_write(csv_path, lambda p: out.to_csv(p, index=False))  # CSV last: it is what the API watches

    # Keep this gameweek's pick: the last run before the deadline leaves the final one.
    # A failure here mustn't cost the predictions themselves.
    try:
        from services.model_review import save_picks
        picks_path = save_picks(live_dir, season, gameweek, event["deadline_time"], csv_path, fixtures_by_team,
                                model=version)
    except Exception:
        logger.exception("Couldn't save the model's pick for GW%s", gameweek)
        picks_path = "failed"
    if shadow_fn is not None:
        shadow = {"model": shadow_version,
                  "pick": save_shadow_pick(live_dir, season, event, players, predictions, fixtures_by_team,
                                           promoted, shadow_version)}
    else:
        shadow = {"model": None, "pick": shadow_version}

    has_fixture = out["fixtures"] > 0
    return {
        "season": season,
        "gameweek": gameweek,
        "deadline": event["deadline_time"],
        "model": version,
        "odds": odds_status,
        "input_coverage": coverage,
        "players": int(len(out)),
        "players_with_fixture": int(has_fixture.sum()),
        "players_unavailable": int((has_fixture & (out["predicted_points"] == 0)
                                    & out["status"].isin(["i", "s", "u", "n"])).sum()),
        "double_gameweek_players": int((out["fixtures"] > 1).sum()),
        "blank_teams": sorted(set(players["team"]) - set(fixtures_by_team)),
        "promoted_teams": promoted,
        "history_rows": int(len(history)),
        "history_last_kickoff": str(history["kickoff_time"].max()) if len(history) else None,
        "output": csv_path,
        "output_json": json_path,
        "model_inputs": features_path if os.path.exists(features_path) else None,
        "model_pick": picks_path,
        "shadow": shadow,
        "seconds": round(time.perf_counter() - started, 1),
        "peak_memory_mb": (round(peak_memory_mb(), 1) if peak_memory_mb() is not None else None),
    }


def backtest(gameweek: int, live_dir: str = LIVE_DATA_DIR, client=None, history_csv: str = HISTORY_CSV,
             predict_fn: Optional[PredictFn] = None) -> Dict[str, Any]:
    """Predict an already-played gameweek of this season from earlier data only and score it.

    No availability multiplier. v4's inputs follow training's rules: the gameweek's transfer
    counts come from its rows in the season CSV, and odds and snapshots count only if captured
    before its deadline (so never the current status file). Compares against the naive
    "average of the last RECENT_MATCHES matches" baseline.
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
    history = load_history(live_csv, season, before_gw=gameweek, before_time=first_kickoff,
                           history_csv=history_csv)
    # The gameweek's transfers as its history rows carry them, not today's
    played = pd.read_csv(live_csv, usecols=["element", "GW", "transfers_in", "transfers_out"])
    played = played[played["GW"] == gameweek].drop_duplicates("element").set_index("element")
    for col in ("in", "out"):
        players[f"transfers_{col}_event"] = players["element"].map(played[f"transfers_{col}"])
    preds = predict_players(players, history, fixtures_by_team, season, gameweek,
                            predict_fn or load_model_fn()[0],
                            extras=load_extras(live_dir, season, status_csv, history_csv))
    live = pd.read_csv(live_csv, usecols=["element", "GW", "total_points"])
    actual = live[live["GW"] == gameweek].groupby("element")["total_points"].sum()
    actual.index = actual.index.astype(str)
    by_name = history.sort_values("kickoff_time").groupby("name").tail(RECENT_MATCHES)
    naive_by_name = by_name.groupby("name")["total_points"].mean()
    naive = players.set_index(players["element"].astype(int).astype(str))["name"].map(naive_by_name)
    df = pd.DataFrame({"pred": pd.Series({k: v["points"] for k, v in preds.items()})})
    df["actual"] = actual
    df["naive"] = naive
    df = df.dropna(subset=["actual"]).fillna({"naive": 0.0})
    top = lambda col: float(df.nlargest(30, col)["actual"].mean())
    return {
        "gameweek": gameweek,
        "players": int(len(df)),
        "history_last_kickoff": str(history.loc[history["season"] == season, "kickoff_time"].max()),
        "first_kickoff": first_kickoff,
        "model_mae": round(float((df.pred - df.actual).abs().mean()), 3),
        "naive_mae": round(float((df.naive - df.actual).abs().mean()), 3),
        "model_spearman": round(float(df.pred.corr(df.actual, method="spearman")), 3),
        "naive_spearman": round(float(df.naive.corr(df.actual, method="spearman")), 3),
        "model_top30_actual": round(top("pred"), 3),
        "naive_top30_actual": round(top("naive"), 3),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate next-gameweek predictions from fresh data")
    parser.add_argument("--live-dir", default=LIVE_DATA_DIR)
    parser.add_argument("--backtest", type=int, metavar="GW",
                        help="Score a played gameweek of this season instead of writing predictions")
    parser.add_argument("--no-odds-fetch", action="store_true",
                        help="Use the odds files already in LIVE_DATA_DIR/odds without downloading")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.backtest:
        result = backtest(args.backtest, args.live_dir)
    else:
        result = predict_next_gameweek(args.live_dir, fetch_odds=not args.no_odds_fetch)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
