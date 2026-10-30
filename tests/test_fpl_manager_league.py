"""Tests for services.fpl_manager_league: finding FPL's "Gameweek N" league and the rank of a
points total in it, against a fake standings API (50 rows a page, totals descending)."""
import json
import os
import random
import re
import tempfile
import unittest

from services import fpl_manager_league as lg

PAGE = 50


class Totals:
    """A league's totals, descending, without building the list (300,000 managers)."""

    def __init__(self, n, total):
        self.n, self.total = n, total

    def __len__(self):
        return self.n

    def __getitem__(self, s):
        return [self.total(i) for i in range(*s.indices(self.n))]


class FakeFPL:
    """leagues-classic/{id}/standings/?page_standings=N for {id: (name, totals)}; unknown ids
    raise like FPLClient does on a 404."""

    def __init__(self, leagues):
        self.leagues = leagues
        self.paths = []

    def get(self, path):
        self.paths.append(path)
        m = re.fullmatch(r"/leagues-classic/(\d+)/standings/\?page_standings=(\d+)", path)
        league_id, page = int(m[1]), int(m[2])
        if league_id not in self.leagues:
            raise RuntimeError(f"GET {path} -> HTTP 404")
        name, totals = self.leagues[league_id]
        rows = totals[(page - 1) * PAGE:page * PAGE]
        return {"league": {"id": league_id, "name": name},
                "standings": {"has_next": page * PAGE < len(totals), "page": page,
                              "results": [{"total": t, "rank_sort": (page - 1) * PAGE + k + 1}
                                          for k, t in enumerate(rows)]}}


def brute(totals, points):
    return 1 + sum(t > points for t in totals), len(totals)


class TestLeagueLookup(unittest.TestCase):
    def test_offset(self):
        api = FakeFPL({283: ("Gameweek 8", []), 284: ("Gameweek 9", [])})
        self.assertEqual(lg.gw_league(api, 9), {"id": 284, "name": "Gameweek 9"})
        self.assertEqual(len(api.paths), 1)

    def test_scans_by_name_when_the_offset_moved(self):
        api = FakeFPL({284: ("Gameweek 10", []), 286: ("Gameweek 9", [])})
        self.assertEqual(lg.gw_league(api, 9)["id"], 286)
        self.assertEqual(len(api.paths), 5)                     # 284, then 283, 285, 282, 286 (nearest first)

    def test_missing(self):
        with self.assertRaises(lg.LeagueError):
            lg.gw_league(FakeFPL({}), 9, scan=range(280, 290))


class TestRank(unittest.TestCase):
    def rank(self, totals, points, hint=None):
        return lg.rank_for_points(FakeFPL({284: ("Gameweek 9", totals)}), 284, points, hint)

    def test_matches_brute_force(self):
        rnd = random.Random(7)
        for n in (1, 49, 50, 51, 100, 1234):
            totals = sorted((rnd.randint(0, 60) for _ in range(n)), reverse=True)
            for points in range(-1, 62):
                self.assertEqual(self.rank(totals, points), brute(totals, points), (n, points))

    def test_ties_share_a_rank(self):
        totals = [100] * 49 + [90] * 3 + [80] * 50 + [70]    # the 90s straddle pages 1 and 2
        self.assertEqual(self.rank(totals, 90), (50, 103))
        self.assertEqual(self.rank(totals, 95), (50, 103))
        self.assertEqual(self.rank(totals, 89), (53, 103))
        self.assertEqual(self.rank(totals, 100), (1, 103))
        self.assertEqual(self.rank(totals, 80), (53, 103))

    def test_first_and_last_page(self):
        totals = list(range(500, 0, -1))                       # 10 pages
        self.assertEqual(self.rank(totals, 600), (1, 500))
        self.assertEqual(self.rank(totals, 498), (3, 500))     # 500 and 499 above
        self.assertEqual(self.rank(totals, 1), (500, 500))
        self.assertEqual(self.rank(totals, 0), (501, 500))     # below everyone

    def test_empty_league(self):
        self.assertEqual(self.rank([], 50), (1, 0))
        self.assertIsNone(lg.percentile(1, 0))

    def test_big_league_takes_few_requests(self):
        totals = Totals(300_017, lambda i: 900 - i // 400)
        api = FakeFPL({284: ("Gameweek 9", totals)})
        get = lg.PoliteGetter(api, delay=0)
        self.assertEqual(lg.rank_for_points(get, 284, 850), (20_001, 300_017))
        self.assertLessEqual(len(api.paths), 40)
        for hint in (300_017, 299_000, 301_000, 250_000):       # last time's size, roughly right
            api.paths.clear()
            self.assertEqual(lg.rank_for_points(api, 284, 850, size_hint=hint), (20_001, 300_017))
            self.assertLessEqual(len(api.paths), 32, hint)
        api.paths.clear()
        lg.rank_for_points(api, 284, 850, size_hint=300_017)
        self.assertLessEqual(len(api.paths), 20)

    def test_percentile(self):
        self.assertEqual(lg.percentile(12_345, 300_000), 95.9)
        self.assertEqual(lg.percentile(1, 300_000), 100.0)
        self.assertEqual(lg.percentile(300_001, 300_000), 0.0)


class TestPoliteGetter(unittest.TestCase):
    def test_pauses_between_requests_and_fetches_once(self):
        sleeps = []
        api = FakeFPL({284: ("Gameweek 9", list(range(200, 0, -1)))})
        get = lg.PoliteGetter(api, delay=0.25, sleep=sleeps.append)
        lg.rank_for_points(get, 284, 120)
        self.assertEqual(len(api.paths), len(set(api.paths)))
        self.assertEqual(sleeps, [0.25] * (len(api.paths) - 1))

    def test_request_budget(self):
        api = FakeFPL({284: ("Gameweek 9", list(range(5000, 0, -1)))})
        with self.assertRaises(lg.LeagueError):
            lg.rank_for_points(lg.PoliteGetter(api, delay=0, max_requests=3), 284, 10)


class TestLeagueRank(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.totals = list(range(1000, 0, -1))
        self.api = FakeFPL({284: ("Gameweek 9", self.totals)})

    def tearDown(self):
        self.tmp.cleanup()

    def lookup(self, gameweek, points, api=None):
        return lg.league_rank(self.tmp.name, gameweek, points, season="2026-27", start_gameweek=9,
                              client=api or self.api, delay=0)

    def test_cached_per_gameweek_and_points(self):
        r = self.lookup(10, 900)
        self.assertEqual((r["id"], r["name"], r["rank"], r["size"], r["percentile"], r["gameweek"]),
                         (284, "Gameweek 9", 101, 1000, 90.0, 10))
        used = len(self.api.paths)
        self.assertEqual(self.lookup(10, 900), r)
        self.assertEqual(len(self.api.paths), used)            # from the cache
        self.lookup(11, 900)                                    # another gameweek: asked again
        self.assertGreater(len(self.api.paths), used)
        self.assertFalse(any(p.endswith("page_standings=1") and "/285/" in p for p in self.api.paths))
        with open(lg.cache_path(self.tmp.name, "2026-27"), encoding="utf-8") as f:
            cache = json.load(f)
        self.assertEqual(cache["leagues"]["9"], {"id": 284, "name": "Gameweek 9"})
        self.assertEqual(cache["sizes"]["284"], 1000)
        self.assertEqual(sorted(cache["ranks"]), ["10", "11"])

    def test_failure_returns_none_and_isnt_cached(self):
        broken = FakeFPL({})
        self.assertIsNone(lg.league_rank(self.tmp.name, 10, 900, season="2026-27", client=broken, delay=0))
        self.assertFalse(os.path.exists(lg.cache_path(self.tmp.name, "2026-27")))
        self.assertEqual(self.lookup(10, 900)["rank"], 101)


if __name__ == "__main__":
    unittest.main()
