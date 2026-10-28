"""
The model's pick for each gameweek, saved before the FPL deadline, scored like FPL.

Saving: pipeline.predict_next_gw calls save_picks() after every run that predicts a
gameweek whose deadline is still ahead, so the last run before the deadline leaves the
squad the Team of the Week page showed (season_service.predicted_selection, the same
function). Files:

  $LIVE_DATA_DIR/model_picks/<season>/gwNN.json             the 15-man squad as picked
  $LIVE_DATA_DIR/model_picks/<season>/gwNN_predictions.csv  every player's prediction
  $LIVE_DATA_DIR/model_picks/<season>/shadow/gwNN.json      the same for the shadow model (v3),
                                                            from the same run (variant SHADOW)

Each JSON names the model that made it (`model`; missing in picks from before v4). The site
shows only the live picks; shadow_comparison scores both the same way for
scripts/analysis/shadow_ledger.py.

Picks recorded by hand (GW7, from the FPL app, before saving existed) are committed in
data/model_picks/<season>/; a saved pick in LIVE_DATA_DIR wins over a committed one.

Scoring (score_picks, pure): actual points and minutes come from the live season CSV
(only finished fixtures), FPL's average and highest score from meta.json's `events`.
FPL's rules: a starter who didn't play is replaced by the first bench player (in order;
goalkeeper for goalkeeper) who played and keeps a valid formation (3+ DEF, 2+ MID,
1+ FWD); if the captain didn't play the vice-captain's points count double. While a
gameweek is live, a player whose matches haven't all finished counts as undecided: he
isn't replaced yet, and an undecided bench player blocks the ones behind him.

Nothing here makes network calls.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Union

from config.settings import CURRENT_SEASON
from services import season_service as ss
from services import team_optimizer as opt

logger = logging.getLogger(__name__)

PICKS_DIR = "model_picks"
SHADOW = "shadow"   # variant: the shadow model's picks, in model_picks/<season>/shadow/
COMMITTED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", PICKS_DIR)
# Auto-subs live in team_optimizer, shared with the ML backtest
FORMATION_MIN = opt.FORMATION_MIN
apply_auto_subs = opt.apply_auto_subs
_counts = opt._counts


# ------------------------------------------------------------------ files

def _pick_name(gameweek: int) -> str:
    return f"gw{int(gameweek):02d}.json"


def _dirs(season: str, variant: Optional[str] = None) -> List[str]:
    """Where picks live, highest priority first (a variant's picks are only ever saved, never committed)."""
    if variant:
        return [os.path.join(ss.live_dir(), PICKS_DIR, season, variant)]
    return [os.path.join(ss.live_dir(), PICKS_DIR, season), os.path.join(COMMITTED_DIR, season)]


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def build_picks(selection: opt.Selection, details: Dict[str, Dict[str, Any]], season: str, gameweek: int,
                deadline: Optional[str], fixture_ids: Dict[str, List[int]],
                archived_at: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """The JSON saved for a gameweek: starters in XI order, then the bench in priority
    order (goalkeeper first). `fixture_ids` maps a club to its fixture ids that gameweek."""
    def player(c: opt.Candidate, role: str) -> Dict[str, Any]:
        d = details[c.id]
        return {
            "element": d.get("element"), "name": c.id, "position": c.position, "team": d["team"],
            "cost": d.get("cost"), "predicted_points": round(float(c.points), 2), "role": role,
            "captain": c.id == selection.captain.id,
            "vice_captain": selection.vice_captain is not None and c.id == selection.vice_captain.id,
            "fixtures": list(fixture_ids.get(d["team"], [])), "opponents": d.get("opponents") or [],
            "status": d.get("status"), "chance_of_playing_next_round": d.get("chance_of_playing_next_round"),
            "news": d.get("news"),
        }

    return {
        "season": season, "gameweek": int(gameweek), "deadline": deadline,
        "archived_at": archived_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "pipeline", "note": None, "model": model,
        "players": [player(c, "starter") for c in selection.starters]
                   + [player(c, "bench") for c in selection.bench],
    }


def save_picks(live_dir: str, season: str, gameweek: int, deadline: Optional[str], predictions: Union[str, Any],
               fixtures_by_team: Dict[str, List[Dict[str, Any]]], now: Optional[datetime] = None,
               variant: Optional[str] = None, model: Optional[str] = None) -> Optional[str]:
    """Save the gameweek's pick and predictions, unless the deadline has passed (then the
    saved pick is final). `predictions` is the predictions CSV's path (the live pick reads
    exactly what the API reads) or a DataFrame in its schema (the shadow's). `variant` SHADOW
    saves into model_picks/<season>/shadow/; `model` is recorded in the JSON.
    Returns the JSON path, or None when nothing was written."""
    import pandas as pd

    now = now or datetime.now(timezone.utc)
    deadline_at = _parse_time(deadline)
    if deadline_at is not None and now >= deadline_at:
        logger.info("GW%s deadline has passed; keeping its saved %spick", gameweek, f"{variant} " if variant else "")
        return None
    df = pd.read_csv(predictions) if isinstance(predictions, str) else predictions
    opponents = {team: [{"team": e["opponent"], "home": bool(e["is_home"])} for e in entries]
                 for team, entries in fixtures_by_team.items()}
    selection, details = ss.predicted_selection(df.to_dict("records"), opponents)
    fixture_ids = {team: [int(e["fixture_id"]) for e in entries] for team, entries in fixtures_by_team.items()}
    picks = build_picks(selection, details, season, gameweek, deadline, fixture_ids,
                        archived_at=now.isoformat(timespec="seconds"), model=model)
    out_dir = os.path.join(live_dir, PICKS_DIR, season, *([variant] if variant else []))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _pick_name(gameweek))
    tmp = path[:-5] + "_predictions.csv.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path[:-5] + "_predictions.csv")
    with open(f"{path}.tmp", "w", encoding="utf-8") as f:   # JSON last: it marks the pick as saved
        json.dump(picks, f, indent=2)
    os.replace(f"{path}.tmp", path)
    return path


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    def load() -> Optional[Dict[str, Any]]:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None
    return ss._cached(("model-picks", path, ss._mtime(path)), load)


def load_picks(gameweek: int, season: str = CURRENT_SEASON, variant: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for d in _dirs(season, variant):
        picks = _read_json(os.path.join(d, _pick_name(gameweek)))
        if picks is not None:
            return picks
    return None


def reviewable_gameweeks(season: str = CURRENT_SEASON, now: Optional[datetime] = None,
                         variant: Optional[str] = None) -> List[int]:
    """Gameweeks with a saved pick (of `variant`) whose deadline has passed, ascending."""
    now = now or datetime.now(timezone.utc)
    found = set()
    for d in _dirs(season, variant):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if name.startswith("gw") and name.endswith(".json") and name[2:-5].isdigit():
                found.add(int(name[2:-5]))
    out = []
    for gw in sorted(found):
        picks = load_picks(gw, season, variant) or {}
        deadline = _parse_time(picks.get("deadline"))
        if deadline is not None and deadline <= now:
            out.append(gw)
    return out


def actual_by_element(gameweek: int, season: str = CURRENT_SEASON) -> Dict[int, Dict[str, int]]:
    """{element: {points, minutes}} for a gameweek from the live season CSV (finished
    fixtures only; double gameweeks summed)."""
    meta = ss.read_meta(season)
    name = (meta.get("files") or {}).get("season") or f"FPL_{season[:4]}_{int(season[:4]) + 1}.csv"
    path = os.path.join(ss.live_dir(), name)

    def load() -> Dict[int, Dict[int, Dict[str, int]]]:
        if not os.path.exists(path):
            return {}
        import pandas as pd
        df = pd.read_csv(path, usecols=["element", "GW", "minutes", "total_points"])
        sums = df.groupby(["GW", "element"])[["minutes", "total_points"]].sum()
        out: Dict[int, Dict[int, Dict[str, int]]] = {}
        for (gw, element), r in sums.iterrows():
            out.setdefault(int(gw), {})[int(element)] = {"points": int(r["total_points"]),
                                                         "minutes": int(r["minutes"])}
        return out

    return ss._cached(("model-review-actual", path, ss._mtime(path)), load).get(int(gameweek), {})


def event_info(gameweek: int, season: str = CURRENT_SEASON) -> Dict[str, Any]:
    for e in ss.read_meta(season).get("events") or []:
        if e.get("id") == gameweek:
            return e
    return {}


# ------------------------------------------------------------------ pure: scoring

def formation_of(players: Iterable[Dict[str, Any]]) -> str:
    c = _counts(players)
    return f"{c['DEF']}-{c['MID']}-{c['FWD']}"


def score_picks(picks: Dict[str, Any], actual: Dict[int, Dict[str, int]], event: Dict[str, Any],
                finished_fixtures: Iterable[int]) -> Dict[str, Any]:
    """Review payload for one saved pick (players keep `element`; photos are added later)."""
    finished = set(finished_fixtures)
    gw_finished = bool(event.get("finished"))
    players = []
    for p in picks["players"]:
        a = actual.get(p.get("element")) if p.get("element") is not None else None
        fixtures = p.get("fixtures") or []
        done = gw_finished or not fixtures or all(f in finished for f in fixtures)
        minutes = a["minutes"] if a else (0 if done else None)
        played = True if minutes else (False if done else None)
        players.append({**p, "points": a["points"] if a else (0 if done else None), "minutes": minutes,
                        "played": played})

    starters = [p for p in players if p["role"] == "starter"]
    bench = [p for p in players if p["role"] == "bench"]
    xi, final_bench, subs = apply_auto_subs(starters, bench)
    out_names = {o for o, _ in subs}
    in_names = {i for _, i in subs}

    captain = next((p for p in starters if p["captain"]), None)
    vice = next((p for p in starters if p["vice_captain"]), None)
    doubled = captain
    if captain is not None and captain["played"] is False:
        doubled = vice if vice is not None and vice["played"] is not False else None

    def row(p: Dict[str, Any], in_xi: bool) -> Dict[str, Any]:
        return {
            "element": p.get("element"), "name": p["name"], "team": p["team"], "position": p["position"],
            "cost": p.get("cost"), "predicted_points": p.get("predicted_points"),
            "points": p["points"], "minutes": p["minutes"],
            "multiplier": (2 if doubled is not None and p["name"] == doubled["name"] else 1) if in_xi else 0,
            "captain": bool(p["captain"]), "vice_captain": bool(p["vice_captain"]), "played": p["played"],
            "subbed_in": p["name"] in in_names, "subbed_out": p["name"] in out_names,
            "opponents": p.get("opponents") or [],
        }

    order = {pos: i for i, pos in enumerate(opt.POSITIONS)}
    xi_rows = sorted((row(p, True) for p in xi), key=lambda r: (order[r["position"]], r["subbed_in"]))
    bench_rows = [row(p, False) for p in final_bench]
    predicted = sum(p.get("predicted_points") or 0.0 for p in starters)
    predicted += (captain.get("predicted_points") or 0.0) if captain else 0.0
    costs = [p.get("cost") for p in players]
    return {
        "season": picks.get("season"), "gameweek": int(picks["gameweek"]),
        "status": "finished" if gw_finished else "live",
        "deadline": picks.get("deadline"), "archived_at": picks.get("archived_at"),
        "source": picks.get("source") or "pipeline", "note": picks.get("note"),
        "formation": formation_of(xi), "picked_formation": formation_of(starters),
        "predicted_points": round(predicted, 2),
        "points": sum((r["points"] or 0) * r["multiplier"] for r in xi_rows),
        "bench_points": sum(r["points"] or 0 for r in bench_rows),
        "fpl_average": event.get("average_entry_score") or None,
        "fpl_highest": event.get("highest_score") or None,
        "total_cost": round(sum(costs), 1) if costs and all(c is not None for c in costs) else None,
        "starters": xi_rows, "bench": bench_rows,
        "auto_subs": [{"out": o, "in": i} for o, i in subs],
    }


# ------------------------------------------------------------------ live vs shadow

def _elements(picks: Dict[str, Any]) -> set:
    return {p.get("element") for p in picks.get("players") or [] if p.get("element") is not None}


def compare_picks(live: Dict[str, Any], shadow: Dict[str, Any], actual: Dict[int, Dict[str, int]],
                  event: Dict[str, Any], finished_fixtures: Iterable[int]) -> Dict[str, Any]:
    """One gameweek of the live model's pick against the shadow's, both scored by score_picks
    (FPL auto-subs and captaincy): counted points, their difference (live - shadow), predicted
    points, and how many of the 15 players the squads share. `same_run` says whether both picks
    come from the same prediction run (pipeline.predict_next_gw saves the pair with one
    archived_at), False when one of them failed to save on a later run and the other is older,
    None when either has no archived_at."""
    finished = list(finished_fixtures)
    a = score_picks(live, actual, event, finished)
    b = score_picks(shadow, actual, event, finished)
    stamps = live.get("archived_at"), shadow.get("archived_at")
    return {
        "gameweek": a["gameweek"], "status": a["status"],
        "same_run": stamps[0] == stamps[1] if all(stamps) else None,
        "live_model": live.get("model"), "shadow_model": shadow.get("model"),
        "live_points": a["points"], "shadow_points": b["points"], "difference": a["points"] - b["points"],
        "live_predicted": a["predicted_points"], "shadow_predicted": b["predicted_points"],
        "shared_players": len(_elements(live) & _elements(shadow)),
        "fpl_average": a["fpl_average"],
    }


def shadow_comparison(season: str = CURRENT_SEASON, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """compare_picks for every gameweek with both a live and a shadow pick whose deadline has
    passed, ascending. A gameweek still being played is provisional (status "live")."""
    live_gws = set(reviewable_gameweeks(season, now))
    finished = ss.read_meta(season).get("finished_fixtures") or []
    out = []
    for gw in reviewable_gameweeks(season, now, variant=SHADOW):
        if gw not in live_gws:
            continue
        out.append(compare_picks(load_picks(gw, season), load_picks(gw, season, SHADOW),
                                 actual_by_element(gw, season), event_info(gw, season), finished))
    return out


# ------------------------------------------------------------------ endpoint payload

def get_model_review(gameweek: Optional[int] = None, photo_for: Optional[ss.PhotoFn] = None,
                     season: str = CURRENT_SEASON) -> Dict[str, Any]:
    gameweeks = reviewable_gameweeks(season)
    if not gameweeks:
        raise ss.NotFoundError("No saved model pick yet")
    if gameweek is None:
        gameweek = gameweeks[-1]
    elif gameweek not in gameweeks:
        raise ss.NotFoundError(f"No saved model pick for gameweek {gameweek}")
    picks = load_picks(gameweek, season)
    meta = ss.read_meta(season)
    payload = score_picks(picks, actual_by_element(gameweek, season), event_info(gameweek, season),
                          meta.get("finished_fixtures") or [])
    payload["season"] = payload["season"] or season
    payload["gameweeks"] = gameweeks
    payload["budget"] = ss.BUDGET / 10.0
    lookup = ss.status_lookup(season)
    for r in payload["starters"] + payload["bench"]:
        r["web_name"] = lookup["web_name"].get(r["name"])
        r["_code"] = lookup["code"].get(r.pop("element", None))
    return ss.with_photos(payload, photo_for)
