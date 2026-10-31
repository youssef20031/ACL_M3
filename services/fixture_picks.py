"""
"Who's expected to score most in each match next gameweek?": the predictions file grouped by
fixture, the top few players of each match (both sides together), for the assistant's context
and its `fixture_picks` card. Pure functions over the predictions rows (services/predictions).

A double-gameweek player's predicted points are the sum of both matches (that's how the file
stores them), so he is listed once, under his first match, marked `double`.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

DEFAULT_PER_FIXTURE = 3
MAX_PER_FIXTURE = 5
NO_FIXTURE = "No fixture"

# "for each match", "in every fixture", "per game", "match by match", "all the fixtures"
_PER_FIXTURE = re.compile(
    r"\b(?:each|every|per)\s+(?:single\s+)?(?:of\s+the\s+)?(?:matche?s?|fixtures?|games?|match-?ups?)\b"
    r"|\b(?:all|all\s+the)\s+(?:matches|fixtures|games)\b"
    r"|\b(?:match|fixture|game)[\s-]+by[\s-]+(?:match|fixture|game)\b"
    r"|\bby\s+(?:match|fixture)\b",
    re.I,
)
# ... about what's ahead, so "results of every match last season" isn't one.
_FORWARD = re.compile(
    r"\b(?:next|upcoming|this|coming)\s+(?:gw|gameweek|game\s*week|week|round)\b|\bgw\s*\d+\b"
    r"|\bexpected\b|\bpredict\w*|\bxpts?\b|\bforecast\w*|\bprojected\b|\blikely\b",
    re.I,
)
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_COUNT = re.compile(r"\b(?:top|best)\s+(\d|one|two|three|four|five)\b|\b(\d|one|two|three|four|five)\s+(?:best|top)?\s*players?\b", re.I)


def is_per_fixture_question(question: str) -> bool:
    """Whether a question asks for predicted players match by match."""
    q = question or ""
    return bool(_PER_FIXTURE.search(q)) and bool(_FORWARD.search(q))


def requested_count(question: str, default: int = DEFAULT_PER_FIXTURE) -> int:
    """How many players per match the question asks for ("top 2 in each game"), 1..5."""
    m = _COUNT.search(question or "")
    if not m:
        return default
    word = (m.group(1) or m.group(2)).lower()
    n = _WORD_NUMBERS.get(word) or int(word)
    return max(1, min(MAX_PER_FIXTURE, n))


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _str(v: Any) -> Optional[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    return s or None


def _bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def top_by_fixture(rows: Iterable[Dict[str, Any]], per_fixture: int = DEFAULT_PER_FIXTURE) -> List[Dict[str, Any]]:
    """[{home, away, kickoff_time, gameweek, players: [row...]}] by kickoff, each with its top
    `per_fixture` players by predicted points (players predicted 0, e.g. injured, left out)."""
    groups: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        team, opponents = _str(r.get("team")), _str(r.get("opponent"))
        points = _num(r.get("predicted_points"))
        if not team or not opponents or opponents == NO_FIXTURE or points is None:
            continue
        parts = [o.strip() for o in opponents.split(" & ") if o.strip()]
        first = parts[0]
        home = _bool(r.get("is_home"))
        key = (team, first) if home else (first, team)
        g = groups.setdefault(key, {"home": key[0], "away": key[1], "kickoff_time": _str(r.get("kickoff_time")),
                                    "gameweek": int(_num(r.get("gameweek")) or 0) or None, "players": []})
        if points <= 0:
            continue
        g["players"].append({**r, "predicted_points": points, "double": len(parts) > 1, "team": team})
    out = []
    for g in groups.values():
        g["players"] = sorted(g["players"], key=lambda p: (-p["predicted_points"], str(p.get("name"))))[:per_fixture]
        out.append(g)
    return sorted(out, key=lambda g: (g["kickoff_time"] or "9999", g["home"]))


def _kickoff_text(iso: Optional[str]) -> str:
    if not iso:
        return "time to be confirmed"
    from datetime import datetime
    try:
        when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    return when.strftime("%a %d %b %H:%M UTC")


def context(fixtures: List[Dict[str, Any]], per_fixture: int) -> str:
    """The assistant's context: every match, its top players with predicted points."""
    if not fixtures:
        return "There are no predictions for the next gameweek's matches yet."
    gw = next((f["gameweek"] for f in fixtures if f.get("gameweek")), None)
    lines = [f"ML predictions, gameweek {gw}: the {per_fixture} players with the most predicted points in each "
             f"of the {len(fixtures)} matches (both teams together). List every match, in this order:"]
    for f in fixtures:
        lines.append(f"{f['home']} v {f['away']} ({_kickoff_text(f.get('kickoff_time'))}):")
        if not f["players"]:
            lines.append("  - no player predicted to score")
        for i, p in enumerate(f["players"], 1):
            extra = []
            if p.get("double"):
                extra.append(f"double gameweek, total over two matches vs {p.get('opponent')}")
            status = _str(p.get("status"))
            if status and status != "a":
                chance = _num(p.get("chance_of_playing_next_round"))
                extra.append(f"flagged {status}" + (f", {int(chance)}% chance of playing" if chance is not None else ""))
            note = f" ({'; '.join(extra)})" if extra else ""
            lines.append(f"  {i}. {p.get('name')} ({p.get('position')}, {p['team']}): "
                         f"{p['predicted_points']:.1f} pts{note}")
    return "\n".join(lines)


def card(fixtures: List[Dict[str, Any]], per_fixture: int, *,
         player_ref: Callable[..., Dict[str, Any]],
         photo: Callable[[str, Optional[str], Optional[int]], Optional[str]],
         codes: Dict[int, int], deadline: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The `fixture_picks` answer card."""
    if not fixtures:
        return None
    gw = next((f["gameweek"] for f in fixtures if f.get("gameweek")), None)
    out = []
    for f in fixtures:
        players = []
        for p in f["players"]:
            element = _num(p.get("element"))
            code = codes.get(int(element)) if element is not None else None
            name = str(p.get("name"))
            players.append({
                "player": player_ref(name, p["team"], p.get("position"), photo(name, p["team"], code), p.get("status"),
                                     p.get("chance_of_playing_next_round"), p.get("news")),
                "points": round(p["predicted_points"], 1),
                "double": bool(p.get("double")),
            })
        out.append({"home": f["home"], "away": f["away"], "kickoff_time": f.get("kickoff_time"), "players": players})
    best = max((p for f in out for p in f["players"]), key=lambda p: p["points"], default=None)
    # (question, intent) pairs, as answer_cards' builders give them
    follow_ups = []
    if best:
        follow_ups.append((f"{best['player']['name']} stats this season", "player_stats"))
    follow_ups.append(("Who should I captain this gameweek?", "prediction"))
    return {
        "type": "fixture_picks",
        "id": f"fixture_picks:{gw}:{per_fixture}",
        "title": "Top predicted players in each match",
        "gameweek": gw,
        "deadline": deadline,
        "per_fixture": per_fixture,
        "fixtures": out,
        "_follow_ups": follow_ups,
    }
