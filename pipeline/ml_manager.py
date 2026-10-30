"""
Run the ML manager's gameweek cycle: the v4 model playing FPL from GW9 2026-27 under FPL's
rules (services/fpl_manager.py), planning with services/fpl_manager_planner.py. Called by
pipeline.daily after the predictions, since it plans from their projections.csv.

    python -m pipeline.ml_manager [--live-dir data/live]

Each run:
  1. Saves FPL's rules (bootstrap-static) for the season; the saved copy is used when the API
     is down.
  2. Creates the first entry (ML_MANAGER_START_GW) from the model's saved pick for it.
  3. Locks every planned gameweek whose deadline has passed. A gameweek whose deadline passed
     with no plan plays the previous team again (rollover), as FPL does.
  4. Scores the locked gameweeks FPL has finished and checked, and ranks the running total in
     FPL's "Gameweek N" league of the managers who started when the ML manager did. A failed
     rank lookup is retried on the next run.
  5. Plans the next gameweek from projections.csv, checks it against the rules and saves it as
     planned; every run before the deadline replaces it, so the last one counts.
  6. Checks what the next deadline will lock against FPL's latest status: no plan (the previous
     team would play again) or a captain who is out (a plan that replanning couldn't refresh) is
     listed under check.problems, which pipeline.daily turns into a failed healthcheck.
  7. Rebuilds state.json.
Never raises: each step's failure is logged and reported in the summary.
"""
import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import ML_MANAGER_SEASON, ML_MANAGER_START_GW
from pipeline.current_season import LIVE_DATA_DIR
from services import fpl_manager as fm

logger = logging.getLogger(__name__)

PROJECTIONS_NAME = "projections.csv"


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _status_frame(live_dir: str, meta: Dict[str, Any]) -> pd.DataFrame:
    name = (meta.get("files") or {}).get("status")
    path = os.path.join(live_dir, name) if name else None
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["element", "name", "web_name", "now_cost"])
    return pd.read_csv(path)


def _prices(status: pd.DataFrame) -> Dict[int, int]:
    rows = status.dropna(subset=["element", "now_cost"])
    return {int(e): int(c) for e, c in zip(rows["element"], rows["now_cost"])}


def _deadlines(meta: Dict[str, Any]) -> Dict[int, str]:
    return {int(e["id"]): e["deadline_time"] for e in meta.get("events") or [] if e.get("deadline_time")}


def _fpl_context(client=None):
    """(rules from bootstrap-static or None, {team: [{team, home}]} by gameweek getter)."""
    from pipeline.fpl_api import FPLClient
    from pipeline.predict_next_gw import gameweek_fixtures, team_name_map

    own = client is None
    client = client or FPLClient(max_retries=3)
    try:
        bootstrap = client.bootstrap()
        fixtures = client.fixtures()
    finally:
        if own:
            client.close()
    teams = team_name_map(bootstrap)

    def opponents(gameweek: int) -> Dict[str, List[Dict[str, Any]]]:
        by_team = gameweek_fixtures(fixtures, gameweek, teams)
        return {t: [{"team": e["opponent"], "home": bool(e["is_home"])} for e in entries]
                for t, entries in by_team.items()}

    return fm.Rules.from_bootstrap(bootstrap), opponents


def _add_opponents(entry: Dict[str, Any], opponents: Dict[str, List[Dict[str, Any]]]) -> None:
    for p in entry.get("squad") or []:
        p.setdefault("opponents", opponents.get(p.get("team"), []))


def _passed(deadline: Optional[str], now: datetime) -> bool:
    return fm.deadline_passed({"deadline": deadline}, now)


def _lock_and_roll_over(live_dir: str, season: str, rules: fm.Rules, deadlines: Dict[int, str],
                        prices: Dict[int, int], now: datetime) -> Dict[str, Any]:
    """Step 3: lock planned entries past their deadline, then add a rollover entry for every
    gameweek since the last entry whose deadline passed without a plan."""
    locked, rolled = [], []
    for e in fm.load_entries(live_dir, season):
        if e.get("status") == fm.PLANNED and fm.deadline_passed(e, now):
            fm.save_entry(live_dir, fm.lock_entry(e, now), now=now)
            locked.append(int(e["gameweek"]))
    entries = [e for e in fm.load_entries(live_dir, season) if e.get("status") != fm.PLANNED]
    if not entries:
        return {"locked": locked, "rolled_over": rolled}
    state = fm.replay(entries, rules, season=season)
    gw = int(entries[-1]["gameweek"]) + 1
    while gw in deadlines and _passed(deadlines[gw], now):
        entry = fm.rollover_entry(state, gw, deadlines[gw], prices=prices, rules=rules, now=now)
        entry["season"] = season
        fm.save_entry(live_dir, entry, now=now)
        state = fm.apply_entry(state, entry, rules)
        rolled.append(gw)
        gw += 1
    return {"locked": locked, "rolled_over": rolled}


def _score(live_dir: str, season: str, rules: fm.Rules, start_gw: int, now: datetime,
           league_rank=None) -> Dict[str, Any]:
    """Step 4: score every locked gameweek FPL has finished and checked, in order, and fill in
    the league rank where it's missing (also for gameweeks scored earlier)."""
    from services import fpl_manager_league as fl
    from services import model_review as mr

    league_rank = league_rank or fl.league_rank
    scored, ranked = [], []
    total = 0
    for e in fm.load_entries(live_dir, season):
        status, gw = e.get("status"), int(e["gameweek"])
        if status == fm.SCORED and e.get("result"):
            result = e["result"]
            if result.get("league") is None:
                league = league_rank(live_dir, gw, int(result["total"]), season, start_gameweek=start_gw)
                if league is not None:
                    fm.save_entry(live_dir, {**e, "result": {**result, "league": league}}, now=now)
                    ranked.append(gw)
            total = int(result["total"])
            continue
        if status != fm.LOCKED or not fm.ready_to_score(mr.event_info(gw, season)):
            break                                   # gameweeks are scored in order
        result = fm.score_live(e, season, previous_total=total)
        result["league"] = league_rank(live_dir, gw, int(result["total"]), season, start_gameweek=start_gw)
        fm.save_entry(live_dir, {**e, "status": fm.SCORED, "result": result}, now=now)
        total = int(result["total"])
        scored.append(gw)
    return {"scored": scored, "ranked": ranked}


def _plan_next(live_dir: str, season: str, rules: fm.Rules, meta: Dict[str, Any], prices: Dict[int, int],
               opponents_for, start_gw: int, now: datetime, planner=None) -> Dict[str, Any]:
    """Step 5: plan the next gameweek while its deadline is ahead."""
    gw, deadline = meta.get("next_gameweek"), meta.get("next_deadline")
    if gw is None or not deadline or _passed(deadline, now):
        return {"status": "skipped: no deadline ahead"}
    gw = int(gw)
    if gw <= start_gw:
        return {"status": f"skipped: GW{gw} uses the model's saved pick"}
    entries = [e for e in fm.load_entries(live_dir, season) if e.get("status") != fm.PLANNED]
    if not entries or int(entries[-1]["gameweek"]) != gw - 1:
        return {"status": f"skipped: GW{gw - 1} isn't locked yet"}
    path = os.path.join(live_dir, PROJECTIONS_NAME)
    if not os.path.exists(path):
        return {"status": "skipped: no projections.csv"}
    projections = pd.read_csv(path)

    if planner is None:
        from services.fpl_manager_planner import plan as planner
    state = fm.replay(entries, rules, season=season)
    current = fm.squad_at_prices(state, prices, rules)
    chips = fm.chip_deadlines(rules, state["chips_used"], gw)
    started = time.perf_counter()
    entry = planner(current, projections, rules.to_dict(), gw, chips, season=season, deadline=deadline, now=now)
    entry["season"] = season
    if opponents_for is not None:
        _add_opponents(entry, opponents_for(gw))
    errors = fm.validate_entry(state, entry, rules, prices)
    if errors:
        logger.error("The GW%s plan breaks the rules, keeping the previous plan: %s", gw, errors)
        return {"status": "rejected", "errors": errors[:10]}
    fm.save_entry(live_dir, entry, now=now)
    return {"status": "planned", "gameweek": gw, "chip": entry.get("chip"),
            "transfers": len(entry.get("transfers") or []), "hit_cost": entry.get("hit_cost"),
            "projected_points": entry.get("projected_points"), "optimal": entry.get("optimal"),
            "seconds": round(time.perf_counter() - started, 1)}


def _out_players(status: pd.DataFrame) -> Dict[int, str]:
    """element -> FPL status letter, for the players FPL gives no chance of playing."""
    from services.season_service import chance_of_playing

    if "status" not in status:
        return {}
    column = "chance_of_playing_next_round"
    chance = status[column] if column in status else [None] * len(status)
    return {int(e): str(s) for e, s, c in zip(status["element"], status["status"], chance)
            if pd.notna(e) and chance_of_playing(s, c) == 0.0}


def _check_next(live_dir: str, season: str, meta: Dict[str, Any], status: pd.DataFrame, start_gw: int,
                now: datetime) -> Dict[str, Any]:
    """Step 6: the entry the next deadline will lock, against FPL's latest status. `problems` are
    what pipeline.daily alerts on while there's still time to act."""
    gw, deadline = meta.get("next_gameweek"), meta.get("next_deadline")
    if gw is None or not deadline or _passed(deadline, now) or int(gw) <= start_gw:
        return {"problems": []}
    gw = int(gw)
    entry = fm.load_entry(live_dir, season, gw)
    if not entry or entry.get("status") != fm.PLANNED:
        return {"gameweek": gw, "problems": [f"GW{gw} has no plan: at its deadline ({deadline}) the previous "
                                             "team plays again"]}
    out = _out_players(status)
    squad = entry.get("squad") or []
    problems = [f"GW{gw}'s {'captain' if p.get('captain') else 'vice-captain'} {p.get('name')} is out "
                f"(status {out[int(p['element'])]}; plan from {entry.get('planned_at')})"
                for p in squad if (p.get("captain") or p.get("vice_captain")) and int(p["element"]) in out]
    starters_out = sum(p.get("role") == "starter" and int(p["element"]) in out for p in squad)
    return {"gameweek": gw, "planned_at": entry.get("planned_at"), "starters_out": starters_out,
            "problems": problems}


def run(live_dir: str = LIVE_DATA_DIR, client=None, now: Optional[datetime] = None,
        season: str = ML_MANAGER_SEASON, start_gameweek: int = ML_MANAGER_START_GW,
        fpl_context=None, planner=None, league_rank=None) -> Dict[str, Any]:
    """One pass of the cycle (see the module docstring). Never raises; returns a summary."""
    now = now or datetime.now(timezone.utc)
    meta = _read_json(os.path.join(live_dir, "meta.json"))
    if meta.get("season") != season:
        return {"status": f"skipped: the live season is {meta.get('season')}, the ML manager plays {season}"}
    summary: Dict[str, Any] = {"season": season}
    status = _status_frame(live_dir, meta)
    prices = _prices(status)
    deadlines = _deadlines(meta)

    opponents_for = None
    try:
        rules, opponents_for = (fpl_context or _fpl_context)(client)
        fm.save_rules(live_dir, season, rules)
    except Exception:
        logger.exception("Couldn't read FPL's rules; using the saved ones")
        rules = fm.load_rules(live_dir, season)
        summary["rules"] = "saved copy"

    steps = [
        ("start", lambda: fm.ensure_start_entry(
            live_dir, season, start_gameweek, now=now,
            web_names=dict(zip(status["name"], status["web_name"])) if "web_name" in status else None,
            rules=rules) is not None),
        ("lock", lambda: _lock_and_roll_over(live_dir, season, rules, deadlines, prices, now)),
        ("score", lambda: _score(live_dir, season, rules, start_gameweek, now, league_rank)),
        ("plan", lambda: _plan_next(live_dir, season, rules, meta, prices, opponents_for, start_gameweek,
                                    now, planner)),
        ("check", lambda: _check_next(live_dir, season, meta, status, start_gameweek, now)),
        ("state", lambda: {k: v for k, v in fm.rebuild_state(live_dir, season, rules).items()
                           if k in ("gameweek", "next_gameweek", "bank", "free_transfers", "totals", "problems")}),
    ]
    for name, step in steps:
        try:
            summary[name] = step()
        except Exception as exc:  # noqa: BLE001 - one step's failure mustn't stop the others
            logger.exception("ML manager step %s failed", name)
            summary[name] = f"failed: {exc}"[:300]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ML manager's gameweek cycle")
    parser.add_argument("--live-dir", default=LIVE_DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(args.live_dir), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
