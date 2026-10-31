"""FPL's live feed (services/live_points.py): who counts as started, caching and fallbacks."""
from datetime import datetime, timezone

from services import live_points as lp

LIVE = {"elements": [
    {"id": 1, "stats": {"total_points": 8, "minutes": 90}, "explain": [{"fixture": 10}]},
    {"id": 2, "stats": {"total_points": 0, "minutes": 0}, "explain": [{"fixture": 10}]},   # benched, match started
    {"id": 3, "stats": {"total_points": 0, "minutes": 0}, "explain": [{"fixture": 11}]},   # match not started
    {"id": 4, "stats": {"total_points": 0, "minutes": 0}, "explain": []},
]}
FIXTURES = [{"id": 10, "started": True, "finished": False, "finished_provisional": True},
            {"id": 11, "started": False, "finished": False}]


def test_parse_keeps_players_whose_match_started():
    out = lp.parse_live(LIVE, FIXTURES)
    assert out["actual"] == {1: {"points": 8, "minutes": 90}, 2: {"points": 0, "minutes": 0}}
    assert out["started"] == [10] and out["finished"] == [10]


def test_cached_for_a_minute_then_stale_on_failure():
    calls = []

    def fetch(path):
        calls.append(path)
        return LIVE if "live" in path else FIXTURES

    assert lp.gameweek_live(8, fetch, now=0.0)["actual"][1]["points"] == 8
    assert lp.gameweek_live(8, fetch, now=30.0) is not None and len(calls) == 2      # cached
    def down(path):
        raise OSError("down")
    assert lp.gameweek_live(8, down, now=120.0) is not None                           # stale, recent
    assert lp.gameweek_live(8, down, now=lp.STALE_SECONDS + 1) is None               # too old
    assert lp.gameweek_live(9, down, now=0.0) is None


def test_merge_prefers_live():
    assert lp.merge({1: {"points": 2, "minutes": 90}, 5: {"points": 1, "minutes": 10}},
                    {"actual": {1: {"points": 6, "minutes": 90}}}) == {1: {"points": 6, "minutes": 90},
                                                                       5: {"points": 1, "minutes": 10}}
    assert lp.merge({1: {"points": 2, "minutes": 1}}, None) == {1: {"points": 2, "minutes": 1}}


def test_is_live_between_deadline_and_finish():
    now = datetime(2026, 10, 24, 15, tzinfo=timezone.utc)
    assert lp.is_live({"finished": False}, "2026-10-24T10:00:00Z", now)
    assert not lp.is_live({"finished": False}, "2026-10-25T10:00:00Z", now)
    assert not lp.is_live({"finished": True}, "2026-10-24T10:00:00Z", now)
    assert not lp.is_live({}, None, now)
