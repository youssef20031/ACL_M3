"""
A player's profile for one season or his whole career, built from his PLAYED_IN rows (one per
match): totals, per-90 rates, where his FPL points came from, and his points match by match
(one season) or season by season (career). Everything but the two small Cypher reads is pure,
so it is tested without Neo4j.

FPL doesn't publish shots or penalties conceded, so the profile has neither; threat (FPL's
measure of shots and attacking positions) and xG stand in for shooting. Own goals and missed or
saved penalties are stored on PLAYED_IN since 2026-11 (a full reload fills earlier seasons);
until then they are null and count under "Other" in the points breakdown.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from config.settings import CURRENT_SEASON, SEASONS

PLAYER_QUERY = """
MATCH (p:Player {name: $player_name})
OPTIONAL MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team)
RETURN p.name AS name, pos.code AS position, t.name AS team, p.last_team AS last_team,
       p.status AS status, p.news AS news, p.chance_of_playing_next_round AS chance_of_playing_next_round,
       p.fpl_code AS fpl_code
LIMIT 1
"""

MATCHES_QUERY = """
MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)
WHERE $season IS NULL OR r.season = $season
OPTIONAL MATCH (p)-[st:SEASON_STATS]->(:Season {id: r.season})
OPTIONAL MATCH (f)-[:HOME_TEAM]->(ht:Team)
OPTIONAL MATCH (f)-[:AWAY_TEAM]->(at:Team)
RETURN r.season AS season, r.gameweek AS gameweek, f.kickoff_time AS kickoff_time,
       r.team AS team, r.was_home AS was_home, ht.name AS home_team, at.name AS away_team,
       f.home_score AS home_score, f.away_score AS away_score, st.position AS position,
       r.minutes AS minutes, r.total_points AS total_points, r.goals_scored AS goals,
       r.assists AS assists, r.clean_sheets AS clean_sheets, r.goals_conceded AS goals_conceded,
       r.saves AS saves, r.bonus AS bonus, r.bps AS bps, r.yellow_cards AS yellow_cards,
       r.red_cards AS red_cards, r.own_goals AS own_goals, r.penalties_missed AS penalties_missed,
       r.penalties_saved AS penalties_saved, r.influence AS influence, r.creativity AS creativity,
       r.threat AS threat, r.ict_index AS ict_index, r.value AS value, r.selected AS selected,
       r.starts AS starts, r.expected_goals AS xg, r.expected_assists AS xa,
       r.expected_goal_involvements AS xgi, r.expected_goals_conceded AS xgc,
       r.tackles AS tackles, r.recoveries AS recoveries,
       r.clearances_blocks_interceptions AS cbi, r.defensive_contribution AS defensive_contribution
ORDER BY r.season, r.gameweek, f.kickoff_time
"""

# FPL's scoring (the same rules scripts/analysis/xpts.py reproduces every score with).
GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}

# Stats summed over matches. Optional ones are null when no match has a value (e.g. xG before
# 2022-23, defensive contributions before 2025-26).
SUM_FIELDS = ["total_points", "minutes", "goals", "assists", "clean_sheets", "goals_conceded",
              "saves", "bonus", "bps", "yellow_cards", "red_cards", "influence", "creativity",
              "threat", "ict_index"]
OPTIONAL_SUM_FIELDS = ["starts", "own_goals", "penalties_missed", "penalties_saved", "xg", "xa", "xgi",
                       "xgc", "tackles", "recoveries", "cbi", "defensive_contribution"]
PER_90 = ["total_points", "goals", "xg", "assists", "xa", "xgi", "threat", "creativity", "influence",
          "bps", "saves", "goals_conceded", "xgc", "tackles", "recoveries", "cbi",
          "defensive_contribution"]
# Rates need a sample, or a 10-minute cameo with a goal shows 9 goals per 90.
MIN_MINUTES_FOR_RATES = 90

BREAKDOWN_LABELS = {
    "appearance": "Playing time",
    "goals": "Goals",
    "assists": "Assists",
    "clean_sheets": "Clean sheets",
    "saves": "Saves",
    "defensive": "Defensive contributions",
    "bonus": "Bonus",
    "conceded": "Goals conceded",
    "cards": "Cards",
    "other": "Other",
}


class NotFoundError(Exception):
    pass


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None      # NaN -> None


def _pos(value: Any) -> Optional[str]:
    code = str(value or "").upper()
    return {"GKP": "GK", "GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD", "AM": None}.get(code)


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(value, digits)


def match_points(row: Dict[str, Any], position: Optional[str]) -> Dict[str, float]:
    """Where one match's FPL points came from, by FPL's rules. What the stored stats don't
    explain (penalties and own goals before they were stored, rule changes) is "other"."""
    minutes = _num(row.get("minutes")) or 0.0
    pos = position or "MID"
    if minutes <= 0:
        parts = dict.fromkeys(BREAKDOWN_LABELS, 0.0)
        parts["other"] = _num(row.get("total_points")) or 0.0
        return parts
    goals = _num(row.get("goals")) or 0.0
    assists = _num(row.get("assists")) or 0.0
    conceded = _num(row.get("goals_conceded")) or 0.0
    dc = _num(row.get("defensive_contribution"))
    parts = {
        "appearance": 2.0 if minutes >= 60 else 1.0,
        "goals": goals * GOAL_POINTS[pos],
        "assists": 3.0 * assists,
        "clean_sheets": (_num(row.get("clean_sheets")) or 0.0) * CLEAN_SHEET_POINTS[pos] if minutes >= 60 else 0.0,
        "saves": float(int(_num(row.get("saves")) or 0) // 3) if pos == "GK" else 0.0,
        "defensive": 2.0 if pos in DC_THRESHOLD and dc is not None and dc >= DC_THRESHOLD[pos] else 0.0,
        "bonus": _num(row.get("bonus")) or 0.0,
        "conceded": -float(int(conceded) // 2) if pos in ("GK", "DEF") else 0.0,
        "cards": -(_num(row.get("yellow_cards")) or 0.0) - 3.0 * (_num(row.get("red_cards")) or 0.0),
    }
    parts["other"] = (_num(row.get("total_points")) or 0.0) - sum(parts.values())
    return parts


def _opponent(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    home, away, team = row.get("home_team"), row.get("away_team"), row.get("team")
    was_home = row.get("was_home")
    if isinstance(was_home, str):
        was_home = was_home.strip().lower() == "true"
    if was_home is None and team:
        was_home = team == home
    opponent = away if was_home else home
    if not opponent:
        return None
    hs, as_ = _num(row.get("home_score")), _num(row.get("away_score"))
    score = None
    if hs is not None and as_ is not None:
        own, other = (hs, as_) if was_home else (as_, hs)
        score = f"{int(own)}-{int(other)}"
    return {"team": opponent, "home": bool(was_home), "score": score}


def summarize(rows: Iterable[Dict[str, Any]], info: Dict[str, Any], season: Optional[str]) -> Dict[str, Any]:
    """The profile payload from PLAYED_IN rows (MATCHES_QUERY's) and the player's node."""
    rows = list(rows)
    played = [r for r in rows if (_num(r.get("minutes")) or 0) > 0]
    totals: Dict[str, Optional[float]] = {}
    for f in SUM_FIELDS:
        totals[f] = sum(_num(r.get(f)) or 0.0 for r in rows)
    for f in OPTIONAL_SUM_FIELDS:
        values = [_num(r.get(f)) for r in rows]
        known = [v for v in values if v is not None]
        totals[f] = sum(known) if known else None
    minutes = totals["minutes"] or 0.0
    appearances = len(played)

    per90 = {}
    for f in PER_90:
        value = totals.get(f)
        per90[f] = _round(value / minutes * 90) if value is not None and minutes >= MIN_MINUTES_FOR_RATES else None

    position = _pos(info.get("position"))
    breakdown = dict.fromkeys(BREAKDOWN_LABELS, 0.0)
    for r in rows:
        for k, v in match_points(r, _pos(r.get("position")) or position).items():
            breakdown[k] += v
    if rows and season is not None:
        # This season's position (players move between positions across seasons).
        position = _pos(rows[-1].get("position")) or position

    if season is not None:
        by_gw: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            gw = int(_num(r.get("gameweek")) or 0)
            entry = by_gw.setdefault(gw, {"gameweek": gw, "points": 0.0, "minutes": 0.0, "goals": 0.0,
                                          "assists": 0.0, "opponents": []})
            entry["points"] += _num(r.get("total_points")) or 0.0
            entry["minutes"] += _num(r.get("minutes")) or 0.0
            entry["goals"] += _num(r.get("goals")) or 0.0
            entry["assists"] += _num(r.get("assists")) or 0.0
            opp = _opponent(r)
            if opp:
                entry["opponents"].append(opp)
        timeline = [{**e, **{k: int(e[k]) for k in ("points", "minutes", "goals", "assists")}}
                    for _, e in sorted(by_gw.items())]
        seasons = []
    else:
        timeline = []
        by_season: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            s = str(r.get("season"))
            e = by_season.setdefault(s, {"season": s, "team": None, "points": 0.0, "minutes": 0.0,
                                         "appearances": 0, "goals": 0.0, "assists": 0.0})
            e["team"] = r.get("team") or e["team"]
            e["points"] += _num(r.get("total_points")) or 0.0
            e["minutes"] += _num(r.get("minutes")) or 0.0
            e["goals"] += _num(r.get("goals")) or 0.0
            e["assists"] += _num(r.get("assists")) or 0.0
            e["appearances"] += int((_num(r.get("minutes")) or 0) > 0)
        seasons = [{**e, **{k: int(e[k]) for k in ("points", "minutes", "goals", "assists")}}
                   for _, e in sorted(by_season.items())]

    last = rows[-1] if rows else {}
    value = _num(last.get("value"))
    team = (rows[-1].get("team") if rows and season is not None else None) or info.get("team") or info.get("last_team")
    is_current = season in (None, CURRENT_SEASON)

    def tidy(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        return int(v) if float(v).is_integer() else round(v, 2)

    return {
        "name": info.get("name"),
        "season": season or "All seasons",
        "position": position,
        "team": team,
        "status": info.get("status") if is_current else None,
        "news": info.get("news") if is_current else None,
        "chance_of_playing_next_round": info.get("chance_of_playing_next_round") if is_current else None,
        "matches": len(rows),
        "appearances": appearances,
        "price": _round(value / 10.0, 1) if value is not None else None,
        "selected": int(_num(last.get("selected"))) if _num(last.get("selected")) is not None else None,
        "points_per_game": _round((totals["total_points"] or 0) / appearances) if appearances else None,
        "minutes_per_game": _round(minutes / appearances, 0) if appearances else None,
        "totals": {k: tidy(v) for k, v in totals.items()},
        "per90": per90,
        "breakdown": [{"key": k, "label": BREAKDOWN_LABELS[k], "points": int(round(v))}
                      for k, v in breakdown.items() if round(v) != 0],
        "gameweeks": timeline,
        "seasons": seasons,
        "_code": info.get("fpl_code"),
    }


def get_player_profile(conn, name: str, season: Optional[str] = None,
                       photo_for: Optional[Callable[[str, Optional[str], Optional[int]], Optional[str]]] = None,
                       predicted_for: Optional[Callable[[str], Optional[float]]] = None) -> Dict[str, Any]:
    if season is not None and season not in SEASONS:
        raise NotFoundError(f"Unknown season {season}")
    info_rows = conn.execute_query(PLAYER_QUERY, {"player_name": name})
    if not info_rows:
        raise NotFoundError(f"No player called {name}")
    info = dict(info_rows[0])
    rows = conn.execute_query(MATCHES_QUERY, {"player_name": info["name"], "season": season})
    out = summarize(rows, info, season)
    code = out.pop("_code")
    out["photo"] = photo_for(out["name"], out["team"], code) if photo_for else None
    out["predicted_points"] = predicted_for(out["name"]) if predicted_for else None
    return out
