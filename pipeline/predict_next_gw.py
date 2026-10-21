"""
Regenerate next-gameweek point predictions from fresh data.

    python -m pipeline.predict_next_gw [--live-dir data/live] [--backtest GW]

Inputs:
  * FPL API: the next gameweek (bootstrap event with `is_next`) and its fixtures.
  * LIVE_DATA_DIR (written by `python -m pipeline.current_season`): this season's
    finished matches (FPL_YYYY_YYYY.csv) and every current player's status/news.
  * data/cleaned_merged_seasons_cleaned.csv: all historical seasons.
  * The v3 points model (ml/models/points_v3.json, trained by `python -m ml.train_points_v3`).

Output: LIVE_DATA_DIR/next_gw_predictions.csv (+ .json), same schema as the old
ml/predictions/next_gw_predictions.csv plus a few transparency columns, written
atomically. services/predictions.py prefers this file over the bundled one. Before the
deadline it also saves the gameweek's squad pick (services/model_review.save_picks) for
the Team of the Week page's "Model vs actual" review.

How a prediction is built: each player gets one row per fixture of the predicted
gameweek and goes through ml.points_model.build_features together with his real
match history, so features are computed by exactly the code used in training and
only from matches before the gameweek (rows from the predicted gameweek or later
are dropped before anything is computed).

  * Double gameweek: one row per fixture, predictions summed.
  * Blank gameweek: the player is kept with 0 points and opponent "No fixture".
  * Availability: status u/n, or chance_of_playing_next_round 0 -> 0 points;
    otherwise points x chance/100. Historical availability isn't recorded, so the
    model itself can't know about injuries; this adjustment is the only place they count.
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
from pipeline.current_season import LIVE_DATA_DIR, next_event, season_file_stem, season_label
from pipeline.team_names import graph_team_name

logger = logging.getLogger(__name__)

HISTORY_CSV = pm.HISTORY_CSV
OUTPUT_NAME = "next_gw_predictions.csv"
POSITIONS = ("GK", "DEF", "MID", "FWD")
MODEL_METHOD = "xgboost_v3"
NO_FIXTURE_METHOD = "no_fixture"
NO_FIXTURE_OPPONENT = "No fixture"
RECENT_MATCHES = 5       # matches averaged by the backtest's naive baseline
HISTORY_SEASONS = 3      # past seasons fed to the features at prediction time

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
    top 20, about half the time and memory.)

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


def previous_season_teams(history: pd.DataFrame, season: str) -> List[str]:
    past = sorted(s for s in history["season"].dropna().unique() if s < season)
    return sorted(history.loc[history["season"] == past[-1], "team"].dropna().unique()) if past else []


# ------------------------------------------------------------------ model rows

def upcoming_rows(players: pd.DataFrame, fixtures_by_team: Dict[str, List[Dict[str, Any]]],
                  season: str, gameweek: int) -> pd.DataFrame:
    """One row per player per fixture of the predicted gameweek, in the model's input format.

    Price and ownership are the current pre-deadline values; ownership becomes a rank
    within this gameweek, matching how training ranks `selected` per gameweek.
    """
    if "selected_by_percent" in players.columns:
        rank = pd.to_numeric(players["selected_by_percent"], errors="coerce").rank(pct=True)
    else:
        rank = pd.Series(np.nan, index=players.index)
    rows = []
    for p, r in zip(players.itertuples(index=False), rank):
        for e in fixtures_by_team.get(p.team) or []:
            rows.append({
                "season": season, "name": p.name, "position": p.position, "team": p.team,
                "opponent": e["opponent"], "fixture": e["fixture_id"],
                "kickoff_time": e["kickoff_time"], "GW": gameweek, "was_home": bool(e["is_home"]),
                "value": float(p.now_cost), "selected_rank": r, "element": int(p.element),
            })
    return pd.DataFrame(rows)


def predict_players(players: pd.DataFrame, history: pd.DataFrame,
                    fixtures_by_team: Dict[str, List[Dict[str, Any]]], season: str, gameweek: int,
                    predict_fn: Callable[[pd.DataFrame], np.ndarray]) -> Dict[str, Dict[str, Any]]:
    """element (str) -> {'points': model points summed over the gameweek's fixtures, 'method'}.

    Features come from ml.points_model.build_features, the same code used in training;
    the upcoming rows have no outcomes, so their features only see earlier matches.
    Players without any earlier matches still get a prediction from price, position,
    ownership and their team/opponent strength.
    """
    upcoming = upcoming_rows(players, fixtures_by_team, season, gameweek)
    if upcoming.empty:
        return {}
    feats = pm.build_features(history, upcoming.drop(columns="element"))
    rows = feats[feats["is_upcoming"]].copy()
    # build_features re-sorts; map rows back to players by (name, fixture)
    rows = rows.merge(upcoming[["name", "fixture", "element"]], on=["name", "fixture"], how="left")
    rows["points"] = np.clip(predict_fn(rows), 0.0, None)
    totals = rows.groupby("element")["points"].sum()
    return {str(int(k)): {"points": float(v), "method": MODEL_METHOD} for k, v in totals.items()}


def _v3_predict_fn():
    booster, meta = pm.load_model()
    return lambda frame: pm.predict(booster, meta, frame)


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


def predict_next_gameweek(live_dir: str = LIVE_DATA_DIR, client=None, history_csv: str = HISTORY_CSV,
                          predict_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None) -> Dict[str, Any]:
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
    history = load_history(live_csv, season, before_gw=gameweek, before_time=first_kickoff,
                           history_csv=history_csv)
    predictions = predict_players(players, history, fixtures_by_team, season, gameweek,
                                  predict_fn or _v3_predict_fn())
    promoted = promoted_teams(previous_season_teams(history, season), players["team"])
    out = assemble_output(players, predictions, fixtures_by_team, gameweek, event["deadline_time"], promoted)

    os.makedirs(live_dir, exist_ok=True)
    csv_path = os.path.join(live_dir, OUTPUT_NAME)
    json_path = csv_path[:-4] + ".json"
    _atomic_write(json_path, lambda p: out.to_json(p, orient="records", indent=2))
    _atomic_write(csv_path, lambda p: out.to_csv(p, index=False))  # CSV last: it is what the API watches

    # Keep this gameweek's pick: the last run before the deadline leaves the final one.
    # A failure here mustn't cost the predictions themselves.
    try:
        from services.model_review import save_picks
        picks_path = save_picks(live_dir, season, gameweek, event["deadline_time"], csv_path, fixtures_by_team)
    except Exception:
        logger.exception("Couldn't save the model's pick for GW%s", gameweek)
        picks_path = "failed"

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
        "history_rows": int(len(history)),
        "history_last_kickoff": str(history["kickoff_time"].max()) if len(history) else None,
        "output": csv_path,
        "output_json": json_path,
        "model_pick": picks_path,
        "seconds": round(time.perf_counter() - started, 1),
        "peak_memory_mb": (round(peak_memory_mb(), 1) if peak_memory_mb() is not None else None),
    }


def backtest(gameweek: int, live_dir: str = LIVE_DATA_DIR, client=None, history_csv: str = HISTORY_CSV,
             predict_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None) -> Dict[str, Any]:
    """Predict an already-played gameweek of this season from earlier data only and score it.

    No availability adjustment (past statuses are not stored). Compares against the
    naive "average of the last RECENT_MATCHES matches" baseline.
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
    preds = predict_players(players, history, fixtures_by_team, season, gameweek,
                            predict_fn or _v3_predict_fn())
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
