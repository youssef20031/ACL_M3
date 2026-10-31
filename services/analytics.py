"""Self-hosted, cookie-free page-view counts.

Privacy by design: nothing that identifies a visitor is stored (no IP, user agent, cookie or
session id), only aggregate counters per UTC day:
  * every view, consent or not: the normalised path and, when the visit came from another site,
    the referring host
  * views from visitors who accepted the consent banner add what their browser reports: its
    timezone (a valid IANA name, e.g. Africa/Cairo; the summary turns it into a country) and two
    flags it keeps in local storage, first visit ever and first view of the UTC day. The flags
    count visitors and new visitors without any identifier; the timezone is counted once per
    visitor per day.
  * views from the owner's browsers (the /stats page's "this browser is me") only count as
    owner views, so the rest are other people.

Counts live in memory and are written to $LIVE_DATA_DIR/analytics.json (a persistent volume on
the VM) with an atomic replace, at most every FLUSH_INTERVAL seconds and on shutdown.
"""
import functools
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import zoneinfo
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

ANALYTICS_FILE = "analytics.json"
RETENTION_DAYS = 400
MAX_KEYS_PER_DAY = 200  # distinct paths (and referrers) per day; the rest are pooled
FLUSH_INTERVAL = 10.0  # seconds

OTHER_PATH = "/other"
OTHER_REFERRER = "other"
UNKNOWN_ZONE = "unknown"   # a consented visitor whose browser sent no valid timezone
# Per-day scalar counters (the rest are {key: count} maps): consented views, visitors (first view
# of the UTC day), new visitors (first visit ever) and the owner's own views
COUNTERS = ("consented_views", "visitors", "new_visitors", "owner_views")
MAPS = ("paths", "referrers", "zones")
MAX_PATH_CHARS = 64
MAX_HOST_CHARS = 100

_PATH_RE = re.compile(r"/[a-z0-9\-/]*")
_HOST_RE = re.compile(r"[a-z0-9.\-]+")


# ------------------------------------------------------------------ normalisation

def normalize_path(raw: Optional[str]) -> str:
    """Route path to count: no query/hash, lowercase, no trailing slash, else "/other".

    Only short paths made of [a-z0-9-/] are kept, so free text (names, search terms, ids with
    odd characters) can't end up in the stats.
    """
    if not isinstance(raw, str):
        return OTHER_PATH
    path = raw.split("#", 1)[0].split("?", 1)[0].strip().lower()
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    if len(path) > MAX_PATH_CHARS or not _PATH_RE.fullmatch(path):
        return OTHER_PATH
    return path


def _bare_host(value: Optional[str]) -> Optional[str]:
    """Lowercase host of a URL (or of a Host header when no scheme), minus port, userinfo, www."""
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    try:
        host = urlparse(value if "//" in value else f"//{value}").hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host or None


def referrer_host(raw: Optional[str], own_host: Optional[str] = None) -> Optional[str]:
    """External referring host to count, or None.

    Drops empty or unparsable referrers, ones without a scheme (document.referrer always has
    one) and the site's own host (`own_host`, the request's Host header), so navigation inside
    the site isn't counted as a referral.
    """
    if not raw or not isinstance(raw, str) or "//" not in raw:
        return None
    host = _bare_host(raw)
    if not host or not _HOST_RE.fullmatch(host):
        return None
    if host == _bare_host(own_host):
        return None
    return host[:MAX_HOST_CHARS]


# ------------------------------------------------------------------ timezones

@functools.lru_cache(maxsize=1)
def _known_zones() -> frozenset:
    try:
        return frozenset(zoneinfo.available_timezones())
    except Exception:   # noqa: BLE001 - no tz database: no zones, nothing breaks
        return frozenset()


def normalize_zone(raw: Optional[str]) -> str:
    """A browser's IANA timezone if it's a real one (so no free text is stored), else UNKNOWN_ZONE."""
    if isinstance(raw, str) and raw.strip() in _known_zones():
        return raw.strip()
    return UNKNOWN_ZONE


@functools.lru_cache(maxsize=1)
def zone_countries() -> Tuple[Dict[str, str], Dict[str, str]]:
    """({zone: ISO 3166 code}, {code: country name}) from the tzdata package's zone.tab and
    iso3166.tab. A zone they don't list (an old alias such as Asia/Calcutta, which some browsers
    still report) takes the country of the listed zones with identical data, when that's one
    country. Empty without tzdata."""
    try:
        import tzdata
        base = os.path.join(os.path.dirname(tzdata.__file__), "zoneinfo")

        def table(name):
            with open(os.path.join(base, name), encoding="utf-8") as fh:
                return [line.rstrip("\n").split("\t") for line in fh if line.strip() and not line.startswith("#")]

        names = {row[0]: row[1] for row in table("iso3166.tab") if len(row) >= 2}
        listed = {row[2]: row[0] for row in table("zone.tab") if len(row) >= 3}

        def digest(zone):
            try:
                with open(os.path.join(base, *zone.split("/")), "rb") as fh:
                    return hashlib.sha1(fh.read()).hexdigest()
            except OSError:
                return None

        by_data: Dict[str, set] = {}
        for zone, code in listed.items():
            by_data.setdefault(digest(zone), set()).add(code)
        countries = dict(listed)
        for zone in _known_zones() - set(listed):
            codes = by_data.get(digest(zone)) or set()
            if len(codes) == 1 and not zone.startswith(("Etc/", "UTC", "GMT")):
                countries[zone] = next(iter(codes))
        return countries, names
    except Exception:   # noqa: BLE001
        logger.warning("No tzdata tables; visitors' countries will show as unknown", exc_info=True)
        return {}, {}


# ------------------------------------------------------------------ store

def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _bump(counts: Dict[str, int], key: str, overflow: str, cap: int) -> None:
    if key not in counts and len(counts) >= cap:
        key = overflow
    counts[key] = counts.get(key, 0) + 1


def _new_day() -> Dict[str, Any]:
    return {**{k: {} for k in MAPS}, **{k: 0 for k in COUNTERS}}


def _clean_counts(raw) -> Optional[Dict[str, int]]:
    if not isinstance(raw, dict):
        return None
    return {
        k: v for k, v in raw.items()
        if isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v > 0
    }


def _is_iso_day(value) -> bool:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat() == value
    except (TypeError, ValueError):
        return False


class PageViewStore:
    """Thread-safe daily counters: {"YYYY-MM-DD": {"paths": {path: n}, "referrers": {host: n},
    "zones": {zone: visitors}, "consented_views": n, "visitors": n, "new_visitors": n,
    "owner_views": n}}. Files from before visitors and zones (version 1) load with them at 0."""

    def __init__(
        self,
        path: str,
        retention_days: int = RETENTION_DAYS,
        max_keys_per_day: int = MAX_KEYS_PER_DAY,
        flush_interval: float = FLUSH_INTERVAL,
        today: Callable[[], date] = _utc_today,
        clock: Callable[[], float] = time.monotonic,
        load: bool = True,
    ):
        self.path = path
        self.retention_days = retention_days
        self.max_keys_per_day = max_keys_per_day
        self.flush_interval = flush_interval
        self._today = today
        self._clock = clock
        self._days: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._last_flush = float("-inf")
        self._lock = threading.Lock()  # guards the counters
        self._flush_lock = threading.Lock()  # serialises snapshot + write, so writes stay in order
        if load:
            self.load()

    # -- persistence ---------------------------------------------------------------------------

    def load(self) -> None:
        """Replace the counters with the file's; a missing or corrupt file means start empty."""
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            logger.info("No analytics file at %s; starting empty", self.path)
            return
        except (OSError, ValueError) as exc:
            self._set_aside_corrupt(f"unreadable ({exc})")
            return

        days_raw = raw.get("days") if isinstance(raw, dict) else None
        if not isinstance(days_raw, dict):
            self._set_aside_corrupt("unexpected structure")
            return

        days: Dict[str, Dict[str, Any]] = {}
        for day, entry in days_raw.items():
            if not _is_iso_day(day) or not isinstance(entry, dict):
                continue
            maps = {k: _clean_counts(entry.get(k, {})) for k in MAPS}
            if any(v is None for v in maps.values()):
                continue
            counters = {k: entry.get(k, 0) for k in COUNTERS}
            if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counters.values()):
                continue
            days[day] = {**maps, **counters}
        with self._lock:
            self._days = days
            self._trim_locked()
            self._dirty = False

    def _set_aside_corrupt(self, reason: str) -> None:
        """Keep a bad file for inspection instead of overwriting it on the next flush."""
        logger.warning("Analytics file %s is %s; starting empty", self.path, reason)
        try:
            os.replace(self.path, self.path + ".corrupt")
        except OSError:
            pass

    def flush(self, force: bool = True, blocking: bool = True) -> bool:
        """Write the counters if they changed (and, unless forced, the interval has passed).

        Returns True when the file was written. Errors are logged, never raised.
        """
        if not self._flush_lock.acquire(blocking=blocking):
            return False  # another thread is writing right now
        try:
            with self._lock:
                if not self._dirty:
                    return False
                now = self._clock()
                if not force and now - self._last_flush < self.flush_interval:
                    return False
                payload = json.dumps({"version": 2, "days": self._days}, sort_keys=True,
                                     separators=(",", ":"))
                self._dirty = False
                self._last_flush = now
            try:
                self._write(payload)
            except OSError as exc:
                logger.warning("Could not write analytics file %s: %s", self.path, exc)
                with self._lock:
                    self._dirty = True
                return False
            return True
        finally:
            self._flush_lock.release()

    def _write(self, payload: str) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".analytics-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    # -- counting ------------------------------------------------------------------------------

    def _trim_locked(self) -> None:
        cutoff = (self._today() - timedelta(days=self.retention_days - 1)).isoformat()
        for day in [d for d in self._days if d < cutoff]:
            del self._days[day]

    def record(self, path: str, referrer: Optional[str] = None, *, consented: bool = False,
               zone: Optional[str] = None, new_visitor: bool = False, first_today: bool = False,
               owner: bool = False, view: bool = True) -> None:
        """Count one view of an already-normalised path (and external referrer host, if any).

        An owner's view only adds to owner_views. A consented view also counts the browser's flags
        (first_today: a visitor, new_visitor: a new one) and, once per visitor per day, its
        normalised `zone`. Without consent the flags and zone are ignored. `view` False counts only
        the visitor: the frontend's signal when a visitor accepts on a page already counted."""
        day = self._today().isoformat()
        with self._lock:
            entry = self._days.get(day)
            if entry is None:
                entry = self._days[day] = _new_day()
                self._trim_locked()
            if owner:
                entry["owner_views"] += int(view)
            else:
                if view:
                    _bump(entry["paths"], path, OTHER_PATH, self.max_keys_per_day)
                    if referrer:
                        _bump(entry["referrers"], referrer, OTHER_REFERRER, self.max_keys_per_day)
                if consented:
                    entry["consented_views"] += int(view)
                    if first_today or new_visitor:   # a first visit ever is also the day's first view
                        entry["visitors"] += 1
                        _bump(entry["zones"], zone or UNKNOWN_ZONE, UNKNOWN_ZONE, self.max_keys_per_day)
                    if new_visitor:
                        entry["new_visitors"] += 1
            self._dirty = True
        self.flush(force=False, blocking=False)

    def summary(self, days: int = 30, top_n: int = 20) -> Dict:
        """Totals for the last `days` UTC days (today included), every day listed oldest first."""
        days = max(1, min(int(days), self.retention_days))
        start = self._today() - timedelta(days=days - 1)
        by_day = []
        paths: Counter = Counter()
        referrers: Counter = Counter()
        zones: Counter = Counter()
        totals: Counter = Counter()
        with self._lock:
            for offset in range(days):
                day = (start + timedelta(days=offset)).isoformat()
                entry = self._days.get(day) or _new_day()
                views = sum(entry["paths"].values())
                paths.update(entry["paths"])
                referrers.update(entry["referrers"])
                zones.update(entry["zones"])
                totals.update({k: entry[k] for k in COUNTERS})
                by_day.append({"date": day, "views": views, "visitors": entry["visitors"],
                               "owner_views": entry["owner_views"]})

        def top(counter: Counter, key: str, value: str = "views"):
            ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
            return [{key: name, value: n} for name, n in ranked]

        zone_country, names = zone_countries()
        countries: Counter = Counter()
        for zone, n in zones.items():
            countries[zone_country.get(zone, "")] += n
        top_countries = [{"code": code or None, "country": names.get(code, "Unknown"), "visitors": n}
                         for code, n in sorted(countries.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
        return {
            "days": days,
            "total_views": sum(d["views"] for d in by_day),
            "consented_views": totals["consented_views"],
            "visitors": totals["visitors"],
            "new_visitors": totals["new_visitors"],
            "owner_views": totals["owner_views"],
            "by_day": by_day,
            "top_paths": top(paths, "path"),
            "top_referrers": top(referrers, "host"),
            "top_countries": top_countries,
            "top_zones": top(zones, "zone", "visitors"),
        }


# ------------------------------------------------------------------ process-wide store

_store: Optional[PageViewStore] = None
_store_lock = threading.Lock()


def default_path() -> str:
    return os.path.join(os.getenv("LIVE_DATA_DIR", os.path.join("data", "live")), ANALYTICS_FILE)


def get_store() -> PageViewStore:
    """The process-wide store, loaded from default_path() on first use."""
    global _store
    with _store_lock:
        if _store is None:
            _store = PageViewStore(default_path())
        return _store


def set_store(store: Optional[PageViewStore]) -> None:
    """Swap the process-wide store (tests). None makes the next get_store() load from disk."""
    global _store
    with _store_lock:
        _store = store


def close_store() -> None:
    """Flush pending counts (app shutdown)."""
    with _store_lock:
        store = _store
    if store is not None:
        store.flush(force=True)


def record_pageview(raw_path: Optional[str], raw_referrer: Optional[str], own_host: Optional[str] = None,
                    *, consented: bool = False, raw_zone: Optional[str] = None, new_visitor: bool = False,
                    first_today: bool = False, owner: bool = False, view: bool = True) -> None:
    """Normalise one page view from the client and count it (PageViewStore.record)."""
    get_store().record(normalize_path(raw_path), referrer_host(raw_referrer, own_host), consented=consented,
                       zone=normalize_zone(raw_zone) if consented else None, new_visitor=new_visitor,
                       first_today=first_today, owner=owner, view=view)
