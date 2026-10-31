"""Cookie-free page-view analytics: normalisation, the PageViewStore and the two routes."""
import json
import os
import sys
from datetime import date, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import api_main  # noqa: E402
from services import analytics  # noqa: E402
from services.analytics import PageViewStore, normalize_path, referrer_host  # noqa: E402
from services.security import RateLimiter  # noqa: E402


class FakeDay:
    def __init__(self, day=date(2026, 10, 15)):
        self.day = day

    def __call__(self):
        return self.day


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_store(tmp_path, **kwargs):
    kwargs.setdefault("today", FakeDay())
    kwargs.setdefault("clock", FakeClock())
    return PageViewStore(str(tmp_path / "analytics.json"), **kwargs)


# --- normalisation -----------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("/", "/"),
    ("/players", "/players"),
    ("/players?name=Salah", "/players"),
    ("/fixtures?gw=8#top", "/fixtures"),
    ("/table#standings", "/table"),
    ("/Team-Of-The-Week", "/team-of-the-week"),
    ("/players/", "/players"),
    ("/players//", "/players"),
    ("//", "/"),
    ("/?ref=x", "/"),
    ("/" + "a" * 63, "/" + "a" * 63),
    ("/" + "a" * 64, "/other"),
    ("/players/Mohamed Salah", "/other"),
    ("/players/%C3%98degaard", "/other"),
    ("/players/ødegaard", "/other"),
    ("/a_b", "/other"),
    ("/x.html", "/other"),
    ("players", "/other"),
    ("https://evil.example/players", "/other"),
    ("", "/other"),
    ("/players\n", "/players"),
    ("/pla\nyers", "/other"),
    (None, "/other"),
])
def test_normalize_path(raw, expected):
    assert normalize_path(raw) == expected


@pytest.mark.parametrize("raw,own,expected", [
    ("https://www.google.com/search?q=salah", "pitchme.app", "google.com"),
    ("https://News.YCombinator.com/item?id=1", "pitchme.app", "news.ycombinator.com"),
    ("https://user:pw@reddit.com:8443/r/fpl", "pitchme.app", "reddit.com"),
    ("android-app://com.google.android.gm/", "pitchme.app", "com.google.android.gm"),
    # the site's own host (any port, with or without www.) is internal navigation
    ("https://pitchme.app/players", "pitchme.app", None),
    ("https://www.pitchme.app/", "pitchme.app", None),
    ("https://pitchme.app/", "www.pitchme.app:443", None),
    ("http://localhost:3000/table", "localhost:8000", None),
    ("https://google.com/", None, "google.com"),
    (None, "pitchme.app", None),
    ("", "pitchme.app", None),
    ("   ", "pitchme.app", None),
    ("google.com", "pitchme.app", None),  # no scheme
    ("http://[::1", "pitchme.app", None),  # unparsable
    ("https:///path-only", "pitchme.app", None),
    ("https://exa<mple>.com/", "pitchme.app", None),
    ("https://bücher.de/", "pitchme.app", None),
])
def test_referrer_host(raw, own, expected):
    assert referrer_host(raw, own) == expected


def test_referrer_host_is_capped_at_100_chars():
    host = ("a" * 60 + ".") * 2 + "com"
    result = referrer_host(f"https://{host}/", "pitchme.app")
    assert result == host[:100]


# --- PageViewStore -----------------------------------------------------------------------------

def test_counts_are_aggregated_by_day(tmp_path):
    today = FakeDay(date(2026, 10, 13))
    store = make_store(tmp_path, today=today)
    store.record("/", "google.com")
    store.record("/players")
    today.day = date(2026, 10, 15)
    store.record("/")
    store.record("/", "google.com")
    store.record("/table", "reddit.com")

    s = store.summary(3)
    assert [{k: d[k] for k in ("date", "views")} for d in s["by_day"]] == [
        {"date": "2026-10-13", "views": 2},
        {"date": "2026-10-14", "views": 0},
        {"date": "2026-10-15", "views": 3},
    ]
    assert s["total_views"] == 5
    assert s["top_paths"][0] == {"path": "/", "views": 3}
    assert s["top_referrers"][0] == {"host": "google.com", "views": 2}

    assert store.summary(1)["total_views"] == 3  # today only


def test_summary_shape_zero_fill_and_ordering(tmp_path):
    store = make_store(tmp_path)
    assert store.summary(30) == {
        "days": 30,
        "total_views": 0, "consented_views": 0, "visitors": 0, "new_visitors": 0, "owner_views": 0,
        "by_day": [{"date": (date(2026, 10, 15) - timedelta(days=29 - i)).isoformat(), "views": 0,
                    "visitors": 0, "owner_views": 0} for i in range(30)],
        "top_paths": [],
        "top_referrers": [],
        "top_countries": [],
        "top_zones": [],
    }

    for i in range(25):
        for _ in range(i + 1):
            store.record(f"/p{i:02d}", f"site{i:02d}.com")
    store.record("/tie-b")
    store.record("/tie-a")

    s = store.summary(7)
    assert len(s["by_day"]) == 7 and s["by_day"][0]["date"] == "2026-10-09"
    assert s["by_day"][-1]["date"] == "2026-10-15"
    assert len(s["top_paths"]) == 20 and len(s["top_referrers"]) == 20
    assert s["top_paths"][0] == {"path": "/p24", "views": 25}
    views = [p["views"] for p in s["top_paths"]]
    assert views == sorted(views, reverse=True)
    assert s["top_referrers"][0] == {"host": "site24.com", "views": 25}

    store2 = make_store(tmp_path / "other")
    store2.record("/tie-b")
    store2.record("/tie-a")
    assert [p["path"] for p in store2.summary()["top_paths"]] == ["/tie-a", "/tie-b"]  # ties by name


def test_flush_and_reload_round_trip(tmp_path):
    store = make_store(tmp_path)
    store.record("/", "google.com")
    store.record("/fixtures")  # within the flush interval: only in memory so far
    assert store.flush(force=True) is True

    path = tmp_path / "analytics.json"
    assert path.exists()
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert data["days"]["2026-10-15"] == {"paths": {"/": 1, "/fixtures": 1}, "referrers": {"google.com": 1},
                                         "zones": {}, "consented_views": 0, "visitors": 0, "new_visitors": 0,
                                         "owner_views": 0}

    reloaded = make_store(tmp_path)
    assert reloaded.summary(30) == store.summary(30)


def test_flush_on_record_is_throttled(tmp_path):
    clock = FakeClock()
    store = make_store(tmp_path, clock=clock, flush_interval=10)
    path = tmp_path / "analytics.json"

    store.record("/")  # first write is due immediately
    assert json.loads(path.read_text())["days"]["2026-10-15"]["paths"] == {"/": 1}

    clock.t += 5
    store.record("/")  # within the interval: memory only
    assert json.loads(path.read_text())["days"]["2026-10-15"]["paths"] == {"/": 1}
    assert store.summary(1)["total_views"] == 2

    clock.t += 6
    store.record("/")
    assert json.loads(path.read_text())["days"]["2026-10-15"]["paths"] == {"/": 3}

    assert store.flush() is False  # nothing new to write


def test_close_store_flushes_pending_counts(tmp_path):
    clock = FakeClock()
    store = make_store(tmp_path, clock=clock)
    store.record("/")
    store.record("/table")  # not yet written (interval)
    analytics.set_store(store)
    try:
        analytics.close_store()
    finally:
        analytics.set_store(None)
    data = json.loads((tmp_path / "analytics.json").read_text())
    assert data["days"]["2026-10-15"]["paths"] == {"/": 1, "/table": 1}


def test_retention_keeps_last_400_days(tmp_path):
    today = FakeDay(date(2025, 1, 1))
    store = make_store(tmp_path, today=today)
    store.record("/old")
    today.day = date(2025, 1, 2)
    store.record("/kept")
    today.day = date(2025, 1, 1) + timedelta(days=400)  # 2025-01-01 is now 401 days ago
    store.record("/new")
    store.flush(force=True)

    data = json.loads((tmp_path / "analytics.json").read_text())
    assert sorted(data["days"]) == ["2025-01-02", today.day.isoformat()]
    assert store.summary(400)["total_views"] == 2


def test_retention_applied_when_loading(tmp_path):
    (tmp_path / "analytics.json").write_text(json.dumps({"version": 1, "days": {
        "2024-01-01": {"paths": {"/": 5}, "referrers": {}},
        "2026-10-14": {"paths": {"/": 2}, "referrers": {"google.com": 1}},
    }}))
    store = make_store(tmp_path)
    s = store.summary(400)
    assert s["total_views"] == 2
    assert s["top_referrers"] == [{"host": "google.com", "views": 1}]


def test_distinct_keys_per_day_are_capped(tmp_path):
    store = make_store(tmp_path, max_keys_per_day=2)
    for p, r in [("/a", "x.com"), ("/b", "y.com"), ("/c", "z.com"), ("/d", "w.com"), ("/a", "x.com")]:
        store.record(p, r)
    s = store.summary(1)
    assert {p["path"]: p["views"] for p in s["top_paths"]} == {"/a": 2, "/b": 1, "/other": 2}
    assert {r["host"]: r["views"] for r in s["top_referrers"]} == {"x.com": 2, "y.com": 1, "other": 2}


@pytest.mark.parametrize("content", ["{not json", "[]", '{"days": [1, 2]}', "\xff\xfe garbage"])
def test_corrupt_file_is_tolerated_and_set_aside(tmp_path, content):
    path = tmp_path / "analytics.json"
    path.write_text(content, encoding="latin-1")
    store = make_store(tmp_path)
    assert store.summary(30)["total_views"] == 0
    assert (tmp_path / "analytics.json.corrupt").exists()
    store.record("/")
    store.flush(force=True)
    assert json.loads(path.read_text())["days"]["2026-10-15"]["paths"] == {"/": 1}


def test_malformed_entries_are_skipped(tmp_path):
    (tmp_path / "analytics.json").write_text(json.dumps({"days": {
        "not-a-date": {"paths": {"/": 9}, "referrers": {}},
        "2026-10-14": {"paths": {"/": 2, "/neg": -3, "/bool": True, "/str": "4"}, "referrers": {}},
        "2026-10-13": "oops",
        "2026-10-12": {"paths": [], "referrers": {}},
    }}))
    s = make_store(tmp_path).summary(30)
    assert s["total_views"] == 2
    assert s["top_paths"] == [{"path": "/", "views": 2}]


def test_missing_file_starts_empty(tmp_path):
    store = PageViewStore(str(tmp_path / "nested" / "dir" / "analytics.json"), today=FakeDay())
    assert store.summary(5)["total_views"] == 0
    store.record("/")  # creates the directory on first write
    assert (tmp_path / "nested" / "dir" / "analytics.json").exists()


def test_default_path_uses_live_data_dir(tmp_path):
    with patch.dict(os.environ, {"LIVE_DATA_DIR": str(tmp_path)}):
        assert analytics.default_path() == os.path.join(str(tmp_path), "analytics.json")


# --- Through the real app (no lifespan) ----------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(api_main.app)


@pytest.fixture
def store(tmp_path):
    """A temp-dir store behind the routes, and a fresh, roomy rate limiter."""
    s = make_store(tmp_path)
    analytics.set_store(s)
    limiter = RateLimiter([(1000, 60)])
    with patch.object(api_main.pageview_limiter, "check", limiter.check):
        yield s
    analytics.set_store(None)


def test_pageview_returns_204_and_counts(client, store):
    resp = client.post("/api/analytics/pageview",
                       json={"path": "/Players/?q=Salah", "referrer": "https://www.google.com/"})
    assert resp.status_code == 204
    assert resp.content == b""
    client.post("/api/analytics/pageview", json={"path": "/table", "referrer": None})
    client.post("/api/analytics/pageview", json={"path": "/fixtures"})
    # TestClient's Host is "testserver": a referrer from there is internal navigation
    client.post("/api/analytics/pageview",
                json={"path": "/assistant", "referrer": "http://testserver/players"})

    s = store.summary(1)
    assert s["total_views"] == 4
    assert {p["path"] for p in s["top_paths"]} == {"/players", "/table", "/fixtures", "/assistant"}
    assert s["top_referrers"] == [{"host": "google.com", "views": 1}]


def test_pageview_stores_no_client_details(client, store, tmp_path):
    client.post("/api/analytics/pageview", json={"path": "/", "referrer": "https://google.com/"},
                headers={"X-Forwarded-For": "203.0.113.7", "User-Agent": "SecretBrowser/1.0"})
    store.flush(force=True)
    raw = (tmp_path / "analytics.json").read_text()
    assert "203.0.113.7" not in raw and "SecretBrowser" not in raw and "testclient" not in raw


@pytest.mark.parametrize("body", [
    {},
    {"referrer": "https://google.com/"},
    {"path": "/" + "a" * 200},
    {"path": "/", "referrer": "https://" + "a" * 500 + ".com/"},
    {"path": 123},
])
def test_pageview_rejects_junk_bodies(client, store, body):
    assert client.post("/api/analytics/pageview", json=body).status_code == 422
    assert store.summary(1)["total_views"] == 0


def test_pageview_is_rate_limited_per_client(client, store):
    limiter = RateLimiter([(3, 60)])
    with patch.object(api_main.pageview_limiter, "check", limiter.check):
        codes = [client.post("/api/analytics/pageview", json={"path": "/"},
                             headers={"X-Forwarded-For": "9.9.9.9"}).status_code for _ in range(4)]
        other = client.post("/api/analytics/pageview", json={"path": "/"},
                            headers={"X-Forwarded-For": "8.8.8.8"})
    assert codes == [204, 204, 204, 429]
    assert other.status_code == 204
    assert store.summary(1)["total_views"] == 4


def test_admin_analytics_hidden_without_admin_token_configured(client, store):
    with patch.dict(os.environ, {"ADMIN_TOKEN": ""}):
        assert client.get("/api/admin/analytics").status_code == 404
        assert client.get("/api/admin/analytics", headers={"X-Admin-Token": "x"}).status_code == 404


def test_admin_analytics_rejects_missing_or_wrong_token(client, store):
    with patch.dict(os.environ, {"ADMIN_TOKEN": "s3cret"}):
        assert client.get("/api/admin/analytics").status_code == 403
        assert client.get("/api/admin/analytics", headers={"X-Admin-Token": "nope"}).status_code == 403


def test_admin_analytics_returns_summary(client, store):
    store.record("/", "google.com")
    store.record("/table")
    with patch.dict(os.environ, {"ADMIN_TOKEN": "s3cret"}):
        resp = client.get("/api/admin/analytics?days=7", headers={"X-Admin-Token": "s3cret"})
        default = client.get("/api/admin/analytics", headers={"X-Admin-Token": "s3cret"})
        assert client.get("/api/admin/analytics?days=0",
                          headers={"X-Admin-Token": "s3cret"}).status_code == 422
        assert client.get("/api/admin/analytics?days=401",
                          headers={"X-Admin-Token": "s3cret"}).status_code == 422
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"days", "total_views", "consented_views", "visitors", "new_visitors", "owner_views",
                         "by_day", "top_paths", "top_referrers", "top_countries", "top_zones"}
    assert body["days"] == 7 and body["total_views"] == 2
    assert len(body["by_day"]) == 7
    assert body["by_day"][-1] == {"date": "2026-10-15", "views": 2, "visitors": 0, "owner_views": 0}
    assert body["top_paths"] == [{"path": "/", "views": 1}, {"path": "/table", "views": 1}]
    assert body["top_referrers"] == [{"host": "google.com", "views": 1}]
    assert default.json()["days"] == 30 and len(default.json()["by_day"]) == 30


# --- Visitors, countries and the owner's own views ---------------------------------------------------

def test_consented_views_count_visitors_new_visitors_and_their_timezones(tmp_path):
    store = make_store(tmp_path)
    store.record("/", consented=True, zone="Africa/Cairo", new_visitor=True, first_today=True)
    store.record("/players", consented=True, zone="Africa/Cairo")              # same visitor, same day
    store.record("/", consented=True, zone="Europe/London", first_today=True)  # a returning visitor
    store.record("/", consented=True, zone="Asia/Calcutta", first_today=True)  # an old alias
    store.record("/", consented=True, zone=analytics.UNKNOWN_ZONE, first_today=True)
    store.record("/table")                                                     # no consent: a view only

    s = store.summary(1)
    assert s["total_views"] == 6 and s["consented_views"] == 5
    assert s["visitors"] == 4 and s["new_visitors"] == 1 and s["owner_views"] == 0
    assert s["by_day"][-1] == {"date": "2026-10-15", "views": 6, "visitors": 4, "owner_views": 0}
    assert {z["zone"]: z["visitors"] for z in s["top_zones"]} == {
        "Africa/Cairo": 1, "Europe/London": 1, "Asia/Calcutta": 1, analytics.UNKNOWN_ZONE: 1}
    countries = {c["code"]: c for c in s["top_countries"]}
    assert set(countries) == {"EG", "GB", "IN", None}
    assert countries["EG"]["visitors"] == 1 and countries["EG"]["country"] == "Egypt"
    assert countries[None] == {"code": None, "country": "Unknown", "visitors": 1}


def test_without_consent_the_browsers_details_are_ignored(tmp_path):
    store = make_store(tmp_path)
    store.record("/", consented=False, zone="Africa/Cairo", new_visitor=True, first_today=True)
    s = store.summary(1)
    assert s["total_views"] == 1 and s["consented_views"] == 0 and s["visitors"] == 0
    assert s["top_zones"] == [] and s["top_countries"] == []


def test_the_owners_views_are_counted_apart(tmp_path):
    store = make_store(tmp_path)
    store.record("/stats", "google.com", owner=True, consented=True, zone="Africa/Cairo", first_today=True)
    store.record("/", "google.com")
    s = store.summary(1)
    assert s["owner_views"] == 1 and s["total_views"] == 1 and s["visitors"] == 0
    assert s["top_paths"] == [{"path": "/", "views": 1}]
    assert s["top_referrers"] == [{"host": "google.com", "views": 1}]
    assert s["by_day"][-1]["owner_views"] == 1


def test_version_1_files_load_with_the_new_counters_at_zero(tmp_path):
    (tmp_path / "analytics.json").write_text(json.dumps(
        {"version": 1, "days": {"2026-10-15": {"paths": {"/": 3}, "referrers": {"google.com": 1}}}}))
    store = make_store(tmp_path)
    s = store.summary(1)
    assert s["total_views"] == 3 and s["visitors"] == 0 and s["owner_views"] == 0
    store.record("/", consented=True, zone="Europe/London", first_today=True)
    assert store.summary(1)["visitors"] == 1


@pytest.mark.parametrize("raw,expected", [
    ("Africa/Cairo", "Africa/Cairo"),
    (" Europe/London ", "Europe/London"),
    ("Mars/Olympus_Mons", analytics.UNKNOWN_ZONE),
    ("<script>alert(1)</script>", analytics.UNKNOWN_ZONE),
    ("", analytics.UNKNOWN_ZONE),
    (None, analytics.UNKNOWN_ZONE),
])
def test_normalize_zone_keeps_only_real_timezones(raw, expected):
    assert analytics.normalize_zone(raw) == expected


def test_zone_countries_cover_canonical_zones_and_old_aliases():
    countries, names = analytics.zone_countries()
    assert countries["Africa/Cairo"] == "EG" and countries["America/New_York"] == "US"
    assert countries["Asia/Calcutta"] == "IN" and countries["US/Eastern"] == "US"
    assert "UTC" not in countries and "Etc/GMT+3" not in countries
    assert names["EG"] == "Egypt"


def test_pageview_with_consent_counts_a_visitor_and_their_country(client, store):
    body = {"path": "/", "consented": True, "timezone": "Africa/Cairo", "new_visitor": True, "first_today": True}
    assert client.post("/api/analytics/pageview", json=body).status_code == 204
    no_consent = {"path": "/", "timezone": "Europe/London", "first_today": True}
    assert client.post("/api/analytics/pageview", json=no_consent).status_code == 204
    assert client.post("/api/analytics/pageview", json={"path": "/stats", "owner": True}).status_code == 204
    s = store.summary(1)
    assert s["total_views"] == 2 and s["visitors"] == 1 and s["new_visitors"] == 1 and s["owner_views"] == 1
    assert [c["code"] for c in s["top_countries"]] == ["EG"]


def test_accepting_on_a_counted_page_adds_the_visitor_not_a_view(client, store):
    client.post("/api/analytics/pageview", json={"path": "/players"})
    client.post("/api/analytics/pageview", json={"path": "/players", "view": False, "consented": True,
                                                 "timezone": "Europe/London", "first_today": True,
                                                 "new_visitor": True})
    s = store.summary(1)
    assert s["total_views"] == 1 and s["consented_views"] == 0
    assert s["visitors"] == 1 and s["new_visitors"] == 1
    assert s["top_zones"] == [{"zone": "Europe/London", "visitors": 1}]


def test_pageview_rejects_an_overlong_timezone(client, store):
    body = {"path": "/", "consented": True, "timezone": "A" * 65}
    assert client.post("/api/analytics/pageview", json=body).status_code == 422


def test_timezone_is_stored_only_with_consent(client, store, tmp_path):
    client.post("/api/analytics/pageview", json={"path": "/", "timezone": "Pacific/Auckland", "first_today": True})
    store.flush(force=True)
    assert "Pacific/Auckland" not in (tmp_path / "analytics.json").read_text()
