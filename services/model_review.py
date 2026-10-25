"""
The model's pick for each gameweek, saved before the FPL deadline, scored like FPL.

Saving: pipeline.predict_next_gw calls save_picks() after every run that predicts a
gameweek whose deadline is still ahead, so the last run before the deadline leaves the
squad the Team of the Week page showed (season_service.predicted_selection, the same
function). Files:

  $LIVE_DATA_DIR/model_picks/<season>/gwNN.json             the 15-man squad as picked
  $LIVE_DATA_DIR/model_picks/<season>/gwNN_safer.json       the safer pick (team_optimizer's
                                                            safer squad), when play60 exists
  $LIVE_DATA_DIR/model_picks/<season>/gwNN_predictions.csv  every player's prediction

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
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config.settings import CURRENT_SEASON
from services import season_service as ss
from services import team_optimizer as opt

logger = logging.getLogger(__name__)

PICKS_DIR = "model_picks"
COMMITTED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", PICKS_DIR)
FORMATION_MIN = {"DEF": 3, "MID": 2, "FWD": 1}


# ------------------------------------------------------------------ files

def _pick_name(gameweek: int, pick: str = "default") -> str:
    return f"gw{int(gameweek):02d}{'' if pick == 'default' else '_' + pick}.json"


def _pick_gameweek(name: str, pick: str = "default") -> Optional[int]:
    """The gameweek of a saved pick's file name for `pick`, else None."""
    suffix = ".json" if pick == "default" else f"_{pick}.json"
    middle = name[2:-len(suffix)] if name.startswith("gw") and name.endswith(suffix) else ""
    return int(middle) if middle.isdigit() else None


def _dirs(season: str) -> List[str]:
    """Where picks live, highest priority first."""
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
                archived_at: Optional[str] = None, pick: str = "default") -> Dict[str, Any]:
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
            "news": d.get("news"), "play60": d.get("play60"),
        }

    return {
        "season": season, "gameweek": int(gameweek), "pick": pick, "deadline": deadline,
        "archived_at": archived_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "pipeline", "note": None,
        "players": [player(c, "starter") for c in selection.starters]
                   + [player(c, "bench") for c in selection.bench],
    }


def save_picks(live_dir: str, season: str, gameweek: int, deadline: Optional[str], predictions_csv: str,
               fixtures_by_team: Dict[str, List[Dict[str, Any]]], now: Optional[datetime] = None) -> Optional[str]:
    """Save the gameweek's picks and predictions, unless the deadline has passed (then the
    saved picks are final): the default pick, and the safer one when the predictions carry
    play60. Returns the default pick's JSON path, or None when nothing was written."""
    import pandas as pd

    now = now or datetime.now(timezone.utc)
    deadline_at = _parse_time(deadline)
    if deadline_at is not None and now >= deadline_at:
        logger.info("GW%s deadline has passed; keeping its saved pick", gameweek)
        return None
    df = pd.read_csv(predictions_csv)      # exactly what the API reads
    rows = df.to_dict("records")
    opponents = {team: [{"team": e["opponent"], "home": bool(e["is_home"])} for e in entries]
                 for team, entries in fixtures_by_team.items()}
    fixture_ids = {team: [int(e["fixture_id"]) for e in entries] for team, entries in fixtures_by_team.items()}
    selection, details = ss.predicted_selection(rows, opponents)
    picks = {"default": build_picks(selection, details, season, gameweek, deadline, fixture_ids,
                                    archived_at=now.isoformat(timespec="seconds"))}
    try:
        selection, details = ss.predicted_selection(rows, opponents, safer=True)
        picks["safer"] = build_picks(selection, details, season, gameweek, deadline, fixture_ids,
                                     archived_at=now.isoformat(timespec="seconds"), pick="safer")
    except ss.UnavailableError as exc:
        logger.info("GW%s: no safer pick saved: %s", gameweek, exc)
    out_dir = os.path.join(live_dir, PICKS_DIR, season)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _pick_name(gameweek))
    tmp = path[:-5] + "_predictions.csv.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path[:-5] + "_predictions.csv")
    for pick, data in picks.items():                          # JSON last: it marks a pick as saved
        target = os.path.join(out_dir, _pick_name(gameweek, pick))
        with open(f"{target}.tmp", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(f"{target}.tmp", target)
    return path


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    def load() -> Optional[Dict[str, Any]]:
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None
    return ss._cached(("model-picks", path, ss._mtime(path)), load)


def load_picks(gameweek: int, season: str = CURRENT_SEASON, pick: str = "default") -> Optional[Dict[str, Any]]:
    for d in _dirs(season):
        picks = _read_json(os.path.join(d, _pick_name(gameweek, pick)))
        if picks is not None:
            return picks
    return None


def reviewable_gameweeks(season: str = CURRENT_SEASON, now: Optional[datetime] = None,
                         pick: str = "default") -> List[int]:
    """Gameweeks with a saved `pick` whose deadline has passed, ascending."""
    now = now or datetime.now(timezone.utc)
    found = set()
    for d in _dirs(season):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        found.update(gw for gw in (_pick_gameweek(name, pick) for name in names) if gw is not None)
    out = []
    for gw in sorted(found):
        picks = load_picks(gw, season, pick) or {}
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

def _counts(players: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {p: 0 for p in opt.POSITIONS}
    for pl in players:
        counts[pl["position"]] += 1
    return counts


def formation_of(players: Iterable[Dict[str, Any]]) -> str:
    c = _counts(players)
    return f"{c['DEF']}-{c['MID']}-{c['FWD']}"


def apply_auto_subs(starters: List[Dict[str, Any]], bench: List[Dict[str, Any]]
                    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, str]]]:
    """FPL automatic substitutions. Each player has `played`: True, False (his matches
    are over and he got no minutes) or None (undecided). Returns (final XI, final bench
    in priority order, [(out, in) names]). A replaced starter takes his replacement's
    bench slot, as the FPL app shows it."""
    xi, slots = list(starters), list(bench)
    subs: List[Tuple[str, str]] = []
    used = set()
    for out in [p for p in starters if p["played"] is False]:
        for i, cand in enumerate(slots):
            if i in used or (cand["position"] == "GK") != (out["position"] == "GK"):
                continue
            counts = _counts(xi)
            counts[out["position"]] -= 1
            counts[cand["position"]] += 1
            if any(counts[p] < n for p, n in FORMATION_MIN.items()):
                continue
            if cand["played"] is None:
                break                      # may still play: no sub until he's decided
            if cand["played"] is False:
                continue
            xi[xi.index(out)] = cand
            slots[i] = out
            used.add(i)
            subs.append((out["name"], cand["name"]))
            break
    return xi, slots, subs


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


# ------------------------------------------------------------------ endpoint payload

def get_model_review(gameweek: Optional[int] = None, photo_for: Optional[ss.PhotoFn] = None,
                     season: str = CURRENT_SEASON, pick: str = "default") -> Dict[str, Any]:
    if pick not in ss.PICKS:
        raise ValueError(f"pick must be one of {ss.PICKS}")
    what = "model pick" if pick == "default" else f"{pick} pick"
    gameweeks = reviewable_gameweeks(season, pick=pick)
    if not gameweeks:
        raise ss.NotFoundError(f"No saved {what} yet")
    if gameweek is None:
        gameweek = gameweeks[-1]
    elif gameweek not in gameweeks:
        raise ss.NotFoundError(f"No saved {what} for gameweek {gameweek}")
    picks = load_picks(gameweek, season, pick)
    meta = ss.read_meta(season)
    payload = score_picks(picks, actual_by_element(gameweek, season), event_info(gameweek, season),
                          meta.get("finished_fixtures") or [])
    payload["season"] = payload["season"] or season
    payload["pick"] = pick
    payload["gameweeks"] = gameweeks
    payload["budget"] = ss.BUDGET / 10.0
    lookup = ss.status_lookup(season)
    for r in payload["starters"] + payload["bench"]:
        r["web_name"] = lookup["web_name"].get(r["name"])
        r["_code"] = lookup["code"].get(r.pop("element", None))
    return ss.with_photos(payload, photo_for)
