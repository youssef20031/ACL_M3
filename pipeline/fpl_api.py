"""
Minimal client for the public (unofficial, keyless) FPL API.

Requests are retried with backoff on 429/5xx and network errors, and per-player
calls are spread over a few threads with a small delay so a full season fetch
(~700 requests) stays polite.
"""
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api"
USER_AGENT = "FPL-FantasyTrivia/1.0 (https://github.com/youssef20031/ACL_M3)"


class FPLApiError(RuntimeError):
    pass


class FPLClient:
    def __init__(self, timeout: float = 20.0, max_retries: int = 5, workers: int = 4, delay: float = 0.15):
        self.max_retries = max_retries
        self.workers = workers
        self.delay = delay
        self._client = httpx.Client(
            base_url=BASE_URL, timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FPLClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get(self, path: str) -> Any:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.get(path)
                if resp.status_code == 200:
                    return resp.json()
                retryable = resp.status_code == 429 or resp.status_code >= 500
                if not retryable:
                    raise FPLApiError(f"GET {path} -> HTTP {resp.status_code}")
                reason = f"HTTP {resp.status_code}"
            except (httpx.TransportError, ValueError) as exc:
                reason = str(exc)
            if attempt == self.max_retries:
                raise FPLApiError(f"GET {path} failed after {attempt} attempts: {reason}")
            wait = min(60.0, 2 ** attempt + random.random())
            logger.warning("GET %s: %s, retrying in %.0fs", path, reason, wait)
            time.sleep(wait)
        raise AssertionError("unreachable")

    # --------------------------------------------------------------- endpoints

    def bootstrap(self) -> Dict[str, Any]:
        """Players, teams, gameweeks (events) and their current status/news."""
        return self.get("/bootstrap-static/")

    def fixtures(self) -> List[Dict[str, Any]]:
        """Every fixture of the season, played or not."""
        return self.get("/fixtures/")

    def element_summary(self, element_id: int) -> Dict[str, Any]:
        """One player's per-fixture history this season plus their upcoming fixtures."""
        return self.get(f"/element-summary/{element_id}/")

    def element_summaries(self, element_ids: Iterable[int],
                          progress: Optional[Callable[[int, int], None]] = None) -> Dict[int, Dict[str, Any]]:
        ids = list(element_ids)
        results: Dict[int, Dict[str, Any]] = {}

        def fetch(element_id: int) -> None:
            time.sleep(self.delay)
            results[element_id] = self.element_summary(element_id)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for done, _ in enumerate(pool.map(fetch, ids), start=1):
                if progress and (done % 100 == 0 or done == len(ids)):
                    progress(done, len(ids))
        return results
