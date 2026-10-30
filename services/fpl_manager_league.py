"""
The ML manager's rank among the managers who joined FPL in the same gameweek.

FPL puts every manager who registers during gameweek N into the system classic league
"Gameweek N". This season their ids are 275 + N (284 is "Gameweek 9", 283 "Gameweek 8");
gw_league checks the name and, if the offset doesn't hold, scans ids 260..320 (nearest
first).

rank_for_points: a league's standings come 50 a page, sorted by total desc
(leagues-classic/{id}/standings/?page_standings=N, with "has_next"). The rank of a total is
1 + the number of managers with a strictly higher total, so ties share a rank as FPL shows
them. The manager isn't in the league, so a total below everyone's ranks size + 1. The
boundary is found by binary search over the pages, and the size by galloping to the last
page, starting from the size found last time when there is one. That's about 30 requests
for a league of 300,000 the first time and about 15 after, with every page fetched at most
once per lookup.

league_rank wraps both with a cache, $LIVE_DATA_DIR/ml_manager/<season>/league_cache.json,
keyed by gameweek and total. It never raises: a failed lookup returns None and is tried again
on the next run. Ask only after FPL marks the gameweek checked (fpl_manager.ready_to_score),
since a cached rank is never fetched again.

Requests are sequential, with pipeline/fpl_api.FPLClient's timeout, User-Agent and retries
on 429/5xx, a short pause between requests and a request budget. The HTTP getter is
injected (anything with .get(path), or a callable), so tests use a fake.
"""
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.settings import CURRENT_SEASON, ML_MANAGER_START_GW

logger = logging.getLogger(__name__)

LEAGUE_ID_OFFSET = 275                 # "Gameweek N" is league 275 + N in 2026-27
SCAN_IDS = range(260, 321)
PAGE_SIZE = 50
REQUEST_DELAY = 0.5                    # seconds between requests
MAX_REQUESTS = 80                      # per lookup; a league of 300,000 takes ~30
MANAGER_DIR = "ml_manager"
CACHE_FILE = "league_cache.json"

Getter = Callable[[str], Any]


class LeagueError(RuntimeError):
    """The league or a rank couldn't be found."""


def standings_path(league_id: int, page: int = 1) -> str:
    return f"/leagues-classic/{int(league_id)}/standings/?page_standings={int(page)}"


def _getter(client: Any) -> Getter:
    get = getattr(client, "get", None)
    return get if callable(get) else client


class PoliteGetter:
    """Sequential GETs with a pause between them, a request budget, and each path fetched
    once (later calls get the first response)."""

    def __init__(self, client: Any, delay: float = REQUEST_DELAY, max_requests: int = MAX_REQUESTS,
                 sleep: Callable[[float], None] = time.sleep):
        self._get = _getter(client)
        self.delay = delay
        self.max_requests = max_requests
        self._sleep = sleep
        self.requests = 0
        self._memo: Dict[str, Any] = {}

    def __call__(self, path: str) -> Any:
        if path in self._memo:
            return self._memo[path]
        if self.requests >= self.max_requests:
            raise LeagueError(f"Stopped after {self.requests} requests")
        if self.requests and self.delay:
            self._sleep(self.delay)
        self.requests += 1
        data = self._get(path)
        self._memo[path] = data
        return data

    get = __call__


# ------------------------------------------------------------------ league lookup

def gw_league(client: Any, gameweek: int, scan: range = SCAN_IDS) -> Dict[str, Any]:
    """{id, name} of FPL's "Gameweek N" league: id 275 + N if its name matches, else the
    nearest id in `scan` with that name. LeagueError when none has it."""
    get = _getter(client)
    want = f"Gameweek {int(gameweek)}"
    guess = LEAGUE_ID_OFFSET + int(gameweek)
    for league_id in [guess] + sorted((i for i in scan if i != guess), key=lambda i: (abs(i - guess), i)):
        try:
            data = get(standings_path(league_id, 1)) or {}
        except LeagueError:
            raise
        except Exception as exc:  # an id with no league (404) or a failed request: try the next
            logger.debug("League %s: %s", league_id, exc)
            continue
        if (data.get("league") or {}).get("name") == want:
            return {"id": league_id, "name": want}
    raise LeagueError(f"No FPL league named {want!r}")


# ------------------------------------------------------------------ rank

Fetch = Callable[[int], Tuple[List[int], bool]]


def _last_page(fetch: Fetch, hint: Optional[int] = None) -> int:
    """The last page with rows (0 for an empty league). Gallops from page 2 doubling, or
    around `hint` in steps of 1, 2, 4, ..., then bisects."""
    rows, has_next = fetch(1)
    if not has_next:
        return 1 if rows else 0
    lo, hi = 1, None                       # lo has a next page; hi is past the end
    p = hint if hint and hint > 1 else 2
    step = 1 if hint and hint > 1 else p
    rows, has_next = fetch(p)
    if has_next:
        lo = p
        while hi is None:
            p = lo + step
            rows, has_next = fetch(p)
            if has_next:
                lo, step = p, step * 2
            elif rows:
                return p
            else:
                hi = p
    elif rows:
        return p
    else:
        hi = p
        while True:
            p = hi - step
            if p <= lo:
                break
            rows, has_next = fetch(p)
            if has_next:
                lo = p
                break
            if rows:
                return p
            hi, step = p, step * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        rows, has_next = fetch(mid)
        if has_next:
            lo = mid
        elif rows:
            return mid
        else:
            hi = mid
    return lo                              # has_next said otherwise; trust the rows


def _count_above(fetch: Fetch, last: int, points: int, known: Dict[int, Tuple[List[int], bool]],
                 page_size: int) -> int:
    """Managers with a total above `points`: the last page whose first total is above it,
    found by bisection (narrowed by the pages already fetched), then its rows."""
    lo, hi = 0, last + 1                   # page lo starts above `points` (0: none); page hi doesn't
    for p, (rows, _) in known.items():
        if 1 <= p <= last:
            if rows and rows[0] > points:
                lo = max(lo, p)
            else:
                hi = min(hi, p)
    if lo >= hi:                           # standings moved while we read them
        lo, hi = 0, last + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        rows, _ = fetch(mid)
        if rows and rows[0] > points:
            lo = mid
        else:
            hi = mid
    if lo == 0:
        return 0
    rows, _ = fetch(lo)
    return (lo - 1) * page_size + sum(t > points for t in rows)


def rank_for_points(client: Any, league_id: int, points: int,
                    size_hint: Optional[int] = None) -> Tuple[int, int]:
    """(rank, size): 1 + the number of the league's managers with a higher total than
    `points`, and how many managers it has. `size_hint` (the size found last time) makes
    finding the last page quick."""
    get = _getter(client)
    pages: Dict[int, Tuple[List[int], bool]] = {}

    def fetch(page: int) -> Tuple[List[int], bool]:
        if page not in pages:
            data = get(standings_path(league_id, page)) or {}
            standings = data.get("standings") or {}
            rows = [int(r.get("total") or 0) for r in standings.get("results") or []]
            pages[page] = (rows, bool(standings.get("has_next")) and bool(rows))
        return pages[page]

    hint = math.ceil(size_hint / PAGE_SIZE) if size_hint else None
    last = _last_page(fetch, hint)
    if last == 0:
        return 1, 0
    page_size = len(fetch(1)[0]) if last > 1 else PAGE_SIZE
    size = (last - 1) * page_size + len(fetch(last)[0])
    return _count_above(fetch, last, int(points), pages, page_size) + 1, size


def percentile(rank: int, size: int) -> Optional[float]:
    """Share of the league at or below the manager's total, in percent (100 = top)."""
    if not size:
        return None
    return round(100.0 * max(size - rank + 1, 0) / size, 1)


# ------------------------------------------------------------------ cached lookup

def cache_path(live_dir: str, season: str = CURRENT_SEASON) -> str:
    return os.path.join(live_dir, MANAGER_DIR, season, CACHE_FILE)


def _load_cache(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    for key in ("leagues", "sizes", "ranks"):
        if not isinstance(data.get(key), dict):
            data[key] = {}
    return data


def _save_cache(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def league_rank(live_dir: str, gameweek: int, points: int, season: str = CURRENT_SEASON,
                start_gameweek: int = ML_MANAGER_START_GW, client: Any = None,
                delay: float = REQUEST_DELAY, sleep: Callable[[float], None] = time.sleep,
                now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
    """The rank of `points` (the manager's total after `gameweek`) in FPL's
    "Gameweek <start_gameweek>" league: {id, name, rank, size, percentile, gameweek,
    fetched_at}, or None when the lookup fails. Cached per (gameweek, points); the league id
    and its size are cached too. `client` defaults to an FPLClient made (and closed) here."""
    path = cache_path(live_dir, season)
    cache = _load_cache(path)
    hit = cache["ranks"].get(str(int(gameweek)), {}).get(str(int(points)))
    if hit:
        return hit

    own = client is None
    try:
        if own:
            from pipeline.fpl_api import FPLClient
            client = FPLClient(max_retries=3)
        get = PoliteGetter(client, delay=delay, sleep=sleep)
        league = cache["leagues"].get(str(int(start_gameweek)))
        if not league:
            league = gw_league(get, start_gameweek)
            cache["leagues"][str(int(start_gameweek))] = league
            _save_cache(path, cache)
        rank, size = rank_for_points(get, league["id"], points, cache["sizes"].get(str(league["id"])))
        out = {"id": league["id"], "name": league["name"], "rank": rank, "size": size,
               "percentile": percentile(rank, size), "gameweek": int(gameweek),
               "fetched_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")}
        cache["ranks"].setdefault(str(int(gameweek)), {})[str(int(points))] = out
        cache["sizes"][str(league["id"])] = size
        _save_cache(path, cache)
        logger.info("%s: %s points after GW%s rank %s of %s (%d requests)",
                    league["name"], points, gameweek, rank, size, get.requests)
        return out
    except Exception as exc:  # never fails the pipeline; tried again next run
        logger.warning("League rank for %s points after GW%s failed: %s", points, gameweek, exc)
        return None
    finally:
        if own and client is not None:
            client.close()
