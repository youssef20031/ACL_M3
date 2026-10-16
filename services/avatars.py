"""
Player avatar lookup.

Resolution order, cheapest first:
  1. Official FPL headshot: built from the player's `code` in the FPL bootstrap
     feed (fetched once per process). The CDN allows hotlinking, so the browser
     loads it directly. Not every code has a photo (new signings often don't: the
     CDN answers 403), so after loading the feed every code is checked once with a
     HEAD request in the background, and codes without a photo fall through.
  2. Wikipedia page image via a single search request, kept only when the page's
     title matches the player's name.
  3. Google Custom Search, only if GOOGLE_CUSTOM_SEARCH_* is configured.

Results (including misses, which expire) are kept in memory and persisted to a
small JSON file so restarts don't repeat external lookups.
"""
import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urlparse

import httpx

from utils.text import normalize_name

logger = logging.getLogger(__name__)

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_PHOTO_URL = "https://resources.premierleague.com/premierleague25/photos/players/110x140/{code}.png"
_FPL_PHOTO_CODE = re.compile(re.escape(FPL_PHOTO_URL).replace(re.escape("{code}"), r"(\d+)") + "$")
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"
USER_AGENT = "FPL-FantasyTrivia/1.0 (https://github.com/youssef20031/ACL_M3)"

# Hosts the image proxy may fetch from. Everything else is rejected (SSRF guard).
PROXY_ALLOWED_HOSTS = ("resources.premierleague.com", "upload.wikimedia.org", "thumb.wikimedia.org")
PROXY_ALLOWED_SUFFIXES = (".gstatic.com",)  # Google image thumbnails

MISS_TTL_SECONDS = 7 * 24 * 3600
_GENERIC_IMAGE_MARKERS = (
    "image_created_with_a_mobile_phone", "no_image_available", "no_photo",
    "default", "placeholder", "silhouette", "logo", "crest", "kit",
)


def is_proxy_allowed(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        host in PROXY_ALLOWED_HOSTS or host.endswith(PROXY_ALLOWED_SUFFIXES)
    )


def build_proxy_image_url(source_url: str) -> str:
    return f"/api/images/proxy?url={quote_plus(source_url)}"


def _public_url(source_url: Optional[str]) -> Optional[str]:
    """FPL and Wikimedia images can be hotlinked; anything else goes via our proxy."""
    if not source_url:
        return None
    host = (urlparse(source_url).hostname or "").lower()
    if host in PROXY_ALLOWED_HOSTS:
        return source_url
    return build_proxy_image_url(source_url)


def _looks_generic(url: str) -> bool:
    lowered = url.lower()
    return any(marker in lowered for marker in _GENERIC_IMAGE_MARKERS)


def _fpl_code_of(url: Optional[str]) -> Optional[int]:
    match = _FPL_PHOTO_CODE.match(url or "")
    return int(match.group(1)) if match else None


def _title_matches(title: str, name: str) -> bool:
    """A Wikipedia result is about this player: one of the title (minus a "(footballer)"
    qualifier) and the name contains every word of the other, and at least two words match
    unless the player goes by a single name. "Ezri Konsa" matches "Ezri Konsa Ngoyo";
    the disambiguation page "Sidiki" doesn't match "Sidiki Cherif"."""
    title_words = normalize_name(re.sub(r"\(.*?\)", " ", title)).split()
    name_words = normalize_name(name).split()
    if not title_words or not name_words:
        return False
    shorter, longer = sorted((title_words, name_words), key=len)
    return set(shorter) <= set(longer) and (len(shorter) >= 2 or len(name_words) == 1)


class AvatarService:
    def __init__(
        self,
        cache_path: str = os.getenv("AVATAR_CACHE_PATH", os.path.join(".cache", "avatars.json")),
        google_api_key: str = "",
        google_cx: str = "",
        concurrency: int = 8,
    ):
        self.cache_path = cache_path
        self.google_api_key = google_api_key
        self.google_cx = google_cx
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(8.0, connect=4.0),
            headers={"User-Agent": USER_AGENT},
            follow_redirects=False,
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        # normalized name -> (source url or None, resolved_at)
        self._cache: Dict[str, Tuple[Optional[str], float]] = {}
        self._inflight: Dict[str, asyncio.Task] = {}
        self._fpl_by_name: Dict[str, int] = {}
        self._fpl_by_surname_team: Dict[Tuple[str, str], int] = {}
        self._fpl_ready = asyncio.Event()
        # FPL codes whose photo was checked on the CDN, and those that turned out to have none
        self._fpl_checked: Set[int] = set()
        self._fpl_missing: Set[int] = set()
        self._check_semaphore = asyncio.Semaphore(4)  # background checks; lookups keep _semaphore
        self._save_pending = False

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Load the persisted cache and kick off the FPL bootstrap fetch in the background."""
        self._load_cache_file()
        asyncio.create_task(self._load_fpl_codes())

    async def close(self) -> None:
        await self._save_cache_file()
        await self.client.aclose()

    async def refresh_fpl_codes(self) -> None:
        """Re-read the FPL bootstrap (new signings get photo codes during the season)."""
        await self._load_fpl_codes()

    def _load_cache_file(self) -> None:
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                raw = json.load(f)
            self._cache = {k: (v[0], float(v[1])) for k, v in raw.items()}
            logger.info("Loaded %d cached avatars from %s", len(self._cache), self.cache_path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Ignoring unreadable avatar cache %s: %s", self.cache_path, exc)

    async def _save_cache_file(self) -> None:
        self._save_pending = False
        data = {k: [url, ts] for k, (url, ts) in self._cache.items()}

        def write() -> None:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            tmp = f"{self.cache_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.cache_path)

        try:
            await asyncio.to_thread(write)
        except Exception as exc:
            logger.warning("Could not persist avatar cache: %s", exc)

    def _schedule_save(self) -> None:
        if self._save_pending:
            return
        self._save_pending = True

        async def later() -> None:
            await asyncio.sleep(5)  # batch writes from a burst of lookups
            await self._save_cache_file()

        asyncio.create_task(later())

    async def _load_fpl_codes(self) -> None:
        try:
            resp = await self.client.get(FPL_BOOTSTRAP_URL, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            teams = {t["id"]: t.get("name", "") for t in data.get("teams", [])}
            for el in data.get("elements", []):
                code = el.get("code")
                if not code:
                    continue
                first, second, web = el.get("first_name", ""), el.get("second_name", ""), el.get("web_name", "")
                for alias in (f"{first} {second}", f"{second} {first}", web):
                    key = normalize_name(alias)
                    if key:
                        self._fpl_by_name.setdefault(key, code)
                surname = normalize_name(second).split()
                team = teams.get(el.get("team"), "")
                if surname and team:
                    self._fpl_by_surname_team.setdefault((surname[-1], team), code)
            logger.info("FPL photo codes loaded for %d players", len(data.get("elements", [])))
            codes = {el["code"] for el in data.get("elements", []) if el.get("code")}
            asyncio.create_task(self._check_fpl_photos(codes))
        except Exception as exc:
            logger.warning("FPL bootstrap unavailable, falling back to other avatar sources: %s", exc)
        finally:
            self._fpl_ready.set()

    async def _check_fpl_photos(self, codes: Iterable[int]) -> None:
        """Find the codes the CDN has no photo for, so lookups and peek() skip them."""
        todo = [c for c in codes if c not in self._fpl_checked]
        await asyncio.gather(*(self._fpl_photo_exists(c) for c in todo))
        logger.info("FPL photos checked: %d codes, %d without a photo", len(todo), len(self._fpl_missing))

    async def _fpl_photo_exists(self, code: int) -> bool:
        """HEAD the CDN once per code. A 403/404 means no photo; on other errors assume there
        is one (and check again next time) rather than hide photos while the CDN is flaky."""
        if code in self._fpl_checked:
            return code not in self._fpl_missing
        async with self._check_semaphore:
            try:
                resp = await self.client.head(FPL_PHOTO_URL.format(code=code))
            except Exception as exc:
                logger.debug("FPL photo check failed for %s: %s", code, exc)
                return True
        if resp.status_code in (403, 404):
            self._fpl_missing.add(code)
        elif resp.status_code != 200:
            return True
        self._fpl_checked.add(code)
        return code not in self._fpl_missing

    # --------------------------------------------------------------- lookups

    def _fpl_code(self, name: str, team: Optional[str]) -> Optional[int]:
        key = normalize_name(name)
        code = self._fpl_by_name.get(key)
        if code is None and team:
            tokens = key.split()
            if tokens:
                code = self._fpl_by_surname_team.get((tokens[-1], team))
        return code

    def _fpl_photo(self, name: str, team: Optional[str]) -> Optional[str]:
        """FPL photo URL from the feed, unless the CDN is known to have no photo for the code."""
        code = self._fpl_code(name, team)
        return FPL_PHOTO_URL.format(code=code) if code and code not in self._fpl_missing else None

    def _known_missing(self, url: Optional[str]) -> bool:
        """An FPL photo URL (e.g. cached before the photo checks existed) that has no photo."""
        code = _fpl_code_of(url)
        return code is not None and code in self._fpl_missing

    async def _wikipedia(self, name: str) -> Optional[str]:
        params = {
            "action": "query", "format": "json", "formatversion": 2,
            "generator": "search", "gsrsearch": f"{name} footballer", "gsrlimit": 3,
            "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": 250,
        }
        resp = await self.client.get(WIKIPEDIA_API_URL, params=params)
        resp.raise_for_status()
        pages = sorted(resp.json().get("query", {}).get("pages", []), key=lambda p: p.get("index", 99))
        for page in pages:
            url = page.get("thumbnail", {}).get("source")
            if url and not _looks_generic(url) and _title_matches(page.get("title", ""), name):
                return url
        return None

    async def _google(self, name: str, team: Optional[str]) -> Optional[str]:
        if not (self.google_api_key and self.google_cx):
            return None
        params = {
            "key": self.google_api_key, "cx": self.google_cx, "searchType": "image",
            "num": 1, "safe": "active", "q": f"{name} {team or ''} footballer".strip(),
        }
        resp = await self.client.get(GOOGLE_CSE_URL, params=params)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if items:
            url = items[0].get("image", {}).get("thumbnailLink") or items[0].get("link")
            if url and not _looks_generic(url):
                return url
        return None

    async def _resolve(self, name: str, team: Optional[str]) -> Optional[str]:
        # Give the bootstrap fetch a moment on cold start; it's the cheapest source.
        try:
            await asyncio.wait_for(self._fpl_ready.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        code = self._fpl_code(name, team)
        if code and await self._fpl_photo_exists(code):
            return FPL_PHOTO_URL.format(code=code)
        async with self._semaphore:
            for source in (lambda: self._wikipedia(name), lambda: self._google(name, team)):
                try:
                    url = await source()
                except Exception as exc:
                    logger.debug("Avatar source failed for %s: %s", name, exc)
                    url = None
                if url:
                    return url
        return None

    async def get(self, name: str, team: Optional[str] = None) -> Optional[str]:
        """Public avatar URL for a player (direct CDN link or our proxy), or None."""
        key = normalize_name(name)
        if not key:
            return None

        cached = self._cache.get(key)
        if cached and self._known_missing(cached[0]):
            cached = None  # an FPL photo that turned out not to exist: look again
        if cached and (cached[0] or time.time() - cached[1] < MISS_TTL_SECONDS):
            # FPL codes can appear after a miss was cached (bootstrap was down)
            if cached[0] or not self._fpl_photo(name, team):
                return _public_url(cached[0])

        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(self._resolve(name, team))
            self._inflight[key] = task
            task.add_done_callback(lambda _t: self._inflight.pop(key, None))
        url = await task
        self._cache[key] = (url, time.time())
        self._schedule_save()
        return _public_url(url)

    def peek(self, name: str, team: Optional[str] = None, fpl_code: Optional[int] = None) -> Optional[str]:
        """Avatar URL only if it's known without a network call: a cache hit, else the FPL
        photo for the feed's code (or `fpl_code`, e.g. from the graph) unless it has none."""
        key = normalize_name(name)
        if not key:
            return None
        cached = self._cache.get(key)
        if cached and cached[0] and not self._known_missing(cached[0]):
            return _public_url(cached[0])
        url = self._fpl_photo(name, team)
        if url is None and fpl_code and not self._fpl_code(name, team) and fpl_code not in self._fpl_missing:
            url = FPL_PHOTO_URL.format(code=fpl_code)
        return _public_url(url)

    async def enrich(self, rows: Iterable[Dict[str, Any]], team_of=None) -> List[Dict[str, Any]]:
        """Return copies of rows with an `avatar` field, looked up concurrently."""
        rows = [dict(r) for r in rows]

        async def fill(row: Dict[str, Any]) -> None:
            name = row.get("player_name")
            if name and not row.get("avatar"):
                team = row.get("team_name") or (team_of(name) if team_of else None)
                row["avatar"] = await self.get(str(name), team)

        await asyncio.gather(*(fill(r) for r in rows))
        return rows


class ImageProxy:
    """Fetches allow-listed images and keeps a small in-memory LRU of the bytes."""

    def __init__(self, client: httpx.AsyncClient, max_bytes: int = 32 * 1024 * 1024):
        self.client = client
        self.max_bytes = max_bytes
        self._store: "Dict[str, Tuple[bytes, str]]" = {}
        self._size = 0

    async def fetch(self, url: str) -> Tuple[bytes, str]:
        if url in self._store:
            item = self._store.pop(url)
            self._store[url] = item  # move to most-recent
            return item

        current = url
        for _ in range(4):  # follow redirects manually so every hop is allow-listed
            if not is_proxy_allowed(current):
                raise PermissionError(current)
            resp = await self.client.get(current, headers={"Accept": "image/*,*/*;q=0.8"})
            if resp.is_redirect:
                current = str(resp.next_request.url) if resp.next_request else ""
                continue
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "image/jpeg")
            if not content_type.startswith("image/"):
                raise ValueError(f"not an image: {content_type}")
            item = (resp.content, content_type)
            self._remember(url, item)
            return item
        raise ValueError("too many redirects")

    def _remember(self, url: str, item: Tuple[bytes, str]) -> None:
        size = len(item[0])
        if size > self.max_bytes // 8:
            return
        self._store[url] = item
        self._size += size
        while self._size > self.max_bytes and self._store:
            _old_url, (data, _ct) = next(iter(self._store.items()))
            self._store.pop(_old_url)
            self._size -= len(data)
