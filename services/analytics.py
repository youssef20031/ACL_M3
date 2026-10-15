"""Self-hosted, cookie-free page-view counts.

Privacy by design: nothing that identifies a visitor is stored (no IP, user agent, cookie or
session id). Each page view only bumps two aggregate counters for its UTC day: the normalised
path and, when the visit came from another site, the referring host. The frontend only reports
views after the visitor accepts the consent banner.

Counts live in memory and are written to $LIVE_DATA_DIR/analytics.json (a persistent volume on
the VM) with an atomic replace, at most every FLUSH_INTERVAL seconds and on shutdown.
"""
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

ANALYTICS_FILE = "analytics.json"
RETENTION_DAYS = 400
MAX_KEYS_PER_DAY = 200  # distinct paths (and referrers) per day; the rest are pooled
FLUSH_INTERVAL = 10.0  # seconds

OTHER_PATH = "/other"
OTHER_REFERRER = "other"
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


# ------------------------------------------------------------------ store

def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _bump(counts: Dict[str, int], key: str, overflow: str, cap: int) -> None:
    if key not in counts and len(counts) >= cap:
        key = overflow
    counts[key] = counts.get(key, 0) + 1


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
    """Thread-safe daily counters: {"YYYY-MM-DD": {"paths": {path: n}, "referrers": {host: n}}}."""

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
        self._days: Dict[str, Dict[str, Dict[str, int]]] = {}
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

        days: Dict[str, Dict[str, Dict[str, int]]] = {}
        for day, entry in days_raw.items():
            if not _is_iso_day(day) or not isinstance(entry, dict):
                continue
            paths = _clean_counts(entry.get("paths", {}))
            referrers = _clean_counts(entry.get("referrers", {}))
            if paths is None or referrers is None:
                continue
            days[day] = {"paths": paths, "referrers": referrers}
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
                payload = json.dumps({"version": 1, "days": self._days}, sort_keys=True,
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

    def record(self, path: str, referrer: Optional[str] = None) -> None:
        """Count one view of an already-normalised path (and external referrer host, if any)."""
        day = self._today().isoformat()
        with self._lock:
            entry = self._days.get(day)
            if entry is None:
                entry = self._days[day] = {"paths": {}, "referrers": {}}
                self._trim_locked()
            _bump(entry["paths"], path, OTHER_PATH, self.max_keys_per_day)
            if referrer:
                _bump(entry["referrers"], referrer, OTHER_REFERRER, self.max_keys_per_day)
            self._dirty = True
        self.flush(force=False, blocking=False)

    def summary(self, days: int = 30, top_n: int = 20) -> Dict:
        """Totals for the last `days` UTC days (today included), every day listed oldest first."""
        days = max(1, min(int(days), self.retention_days))
        start = self._today() - timedelta(days=days - 1)
        by_day = []
        paths: Counter = Counter()
        referrers: Counter = Counter()
        with self._lock:
            for offset in range(days):
                day = (start + timedelta(days=offset)).isoformat()
                entry = self._days.get(day)
                views = 0
                if entry:
                    views = sum(entry["paths"].values())
                    paths.update(entry["paths"])
                    referrers.update(entry["referrers"])
                by_day.append({"date": day, "views": views})

        def top(counter: Counter, key: str):
            ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
            return [{key: name, "views": n} for name, n in ranked]

        return {
            "days": days,
            "total_views": sum(d["views"] for d in by_day),
            "by_day": by_day,
            "top_paths": top(paths, "path"),
            "top_referrers": top(referrers, "host"),
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


def record_pageview(raw_path: Optional[str], raw_referrer: Optional[str],
                    own_host: Optional[str] = None) -> None:
    """Normalise one page view from the client and count it."""
    get_store().record(normalize_path(raw_path), referrer_host(raw_referrer, own_host))
