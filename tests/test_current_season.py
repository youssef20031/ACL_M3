"""Offline tests for pipeline.current_season (no network)."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from pipeline.current_season import (
    CSV_COLUMNS, ELEMENT_SNAPSHOT_COLUMNS, build_season_rows, build_status, element_snapshot_path, run,
    season_file_stem, season_label, update_element_snapshots, update_xp_snapshots,
)


def _bootstrap(next_deadline: datetime):
    return {
        "events": [
            {"id": 1, "deadline_time": "2026-08-21T17:30:00Z", "is_next": False},
            {"id": 2, "deadline_time": next_deadline.strftime("%Y-%m-%dT%H:%M:%SZ"), "is_next": True},
        ],
        "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Ipswich Town"}, {"id": 3, "name": "Chelsea"}],
        "elements": [
            # Transferred from Ipswich (team 2) to Arsenal (team 1) after GW1
            {"id": 10, "code": 100, "first_name": "Test", "second_name": "Mover", "web_name": "Mover",
             "team": 1, "element_type": 3, "status": "a", "news": "", "now_cost": 55, "ep_next": "4.5",
             "transfers_in_event": 12000, "transfers_out_event": 3400, "cost_change_event": 1,
             "penalties_order": 1, "corners_and_indirect_freekicks_order": None, "direct_freekicks_order": 2},
            {"id": 11, "code": 101, "first_name": "Injured", "second_name": "Keeper", "web_name": "Keeper",
             "team": 3, "element_type": 1, "status": "i", "news": "Knee injury",
             "chance_of_playing_next_round": 0, "now_cost": 45, "ep_next": "0.0"},
            {"id": 12, "code": 102, "first_name": "Assistant", "second_name": "Manager", "web_name": "AM",
             "team": 3, "element_type": 5, "status": "a", "news": "", "now_cost": 10, "ep_next": "1.0"},
        ],
    }


FIXTURES = [
    {"id": 1, "team_h": 2, "team_a": 3, "finished": True},   # GW1 Ipswich v Chelsea
    {"id": 2, "team_h": 3, "team_a": 1, "finished": True},   # GW2 Chelsea v Arsenal
    {"id": 3, "team_h": 1, "team_a": 3, "finished": False},  # GW3, still provisional
]


def _history(fixture, rnd, was_home, minutes=90):
    return {"fixture": fixture, "round": rnd, "was_home": was_home, "minutes": minutes,
            "total_points": 2, "kickoff_time": f"2026-08-{20 + rnd}T14:00:00Z", "modified": False,
            "element": 10, "opponent_team": 3, "value": 55}


def test_season_label_and_stem():
    assert season_label(_bootstrap(datetime.now(timezone.utc))) == "2026-27"
    assert season_file_stem("2026-27") == "2026_2027"


def test_rows_use_team_from_fixture_and_skip_unfinished():
    bootstrap = _bootstrap(datetime.now(timezone.utc) + timedelta(days=2))
    summaries = {10: {"history": [_history(1, 1, True), _history(2, 2, False), _history(3, 3, True)]}}
    df = build_season_rows(bootstrap, FIXTURES, summaries)

    assert list(df.columns) == CSV_COLUMNS
    assert df["GW"].tolist() == [1, 2]                  # GW3 not finished yet
    assert df["team"].tolist() == ["Ipswich", "Arsenal"]  # team at the time, alias applied
    assert set(df["name"]) == {"Test Mover"} and set(df["position"]) == {"MID"}
    assert df["was_home"].dtype == bool


def test_xp_snapshot_before_deadline_is_merged():
    bootstrap = _bootstrap(datetime.now(timezone.utc) + timedelta(days=2))
    xp = update_xp_snapshots(bootstrap, None)
    assert set(xp["GW"]) == {2} and set(xp["element"]) == {10, 11}  # assistant manager skipped
    assert float(xp.loc[xp.element == 10, "xP"].iloc[0]) == 4.5

    summaries = {10: {"history": [_history(1, 1, True), _history(2, 2, False)]}}
    df = build_season_rows(bootstrap, FIXTURES, summaries, xp)
    assert pd.isna(df.loc[df.GW == 1, "xP"].iloc[0])  # no snapshot was taken for GW1
    assert df.loc[df.GW == 2, "xP"].iloc[0] == 4.5


def test_xp_snapshot_frozen_after_deadline():
    before = update_xp_snapshots(_bootstrap(datetime.now(timezone.utc) + timedelta(hours=1)), None)
    after = update_xp_snapshots(_bootstrap(datetime.now(timezone.utc) - timedelta(hours=1)), before)
    pd.testing.assert_frame_equal(before, after)


def test_status_includes_news_and_skips_assistant_managers():
    status = build_status(_bootstrap(datetime.now(timezone.utc)))
    assert set(status["element"]) == {10, 11}
    keeper = status[status.element == 11].iloc[0]
    assert keeper["status"] == "i" and keeper["news"] == "Knee injury" and keeper["position"] == "GK"


def test_status_includes_transfers_price_change_and_set_piece_orders():
    status = build_status(_bootstrap(datetime.now(timezone.utc)))
    mover = status[status.element == 10].iloc[0]
    assert (mover["transfers_in_event"], mover["transfers_out_event"], mover["cost_change_event"]) == (12000, 3400, 1)
    assert mover["penalties_order"] == 1 and mover["direct_freekicks_order"] == 2
    assert pd.isna(mover["corners_and_indirect_freekicks_order"])
    keeper = status[status.element == 11].iloc[0]
    assert pd.isna(keeper["transfers_in_event"]) and pd.isna(keeper["penalties_order"])  # fields absent


def test_element_snapshots_keep_every_capture_before_the_deadline(tmp_path):
    bootstrap = _bootstrap(datetime.now(timezone.utc) + timedelta(days=2))
    path = update_element_snapshots(bootstrap, str(tmp_path), "2026-27")
    assert path == element_snapshot_path(str(tmp_path), "2026-27", 2)
    assert path.endswith("gw02.csv")
    bootstrap["elements"][0]["transfers_in_event"] = 15000  # more transfers by the next run
    update_element_snapshots(bootstrap, str(tmp_path), "2026-27")

    snaps = pd.read_csv(path)
    assert list(snaps.columns) == ELEMENT_SNAPSHOT_COLUMNS
    assert set(snaps["GW"]) == {2} and set(snaps["element"]) == {10, 11}  # assistant manager skipped
    assert snaps.groupby("element").size().tolist() == [2, 2]
    assert snaps[snaps.element == 10]["transfers_in_event"].tolist() == [12000, 15000]
    keeper = snaps[snaps.element == 11].iloc[-1]
    assert keeper["status"] == "i" and keeper["news"] == "Knee injury" and keeper["chance_of_playing_next_round"] == 0


def test_element_snapshots_stop_at_the_deadline(tmp_path):
    bootstrap = _bootstrap(datetime.now(timezone.utc) - timedelta(minutes=1))
    assert update_element_snapshots(bootstrap, str(tmp_path), "2026-27") is None
    assert not (tmp_path / "element_snapshots").exists()


class _StatusOnlyClient:
    def __init__(self, bootstrap):
        self._bootstrap = bootstrap

    def bootstrap(self):
        return self._bootstrap


def test_status_only_run_writes_the_status_file_and_a_snapshot(tmp_path):
    bootstrap = _bootstrap(datetime.now(timezone.utc) + timedelta(days=1))
    run(str(tmp_path), status_only=True, client=_StatusOnlyClient(bootstrap))
    status = pd.read_csv(tmp_path / "player_status_2026_27.csv")
    assert {"transfers_in_event", "cost_change_event", "penalties_order"} <= set(status.columns)
    snaps = pd.read_csv(tmp_path / "element_snapshots" / "2026-27" / "gw02.csv")
    assert len(snaps) == 2


# ------------------------------------------------------------------ bookmaker odds

VALID_E0 = ("Div,Date,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5\n"
            "E0,16/08/2026,Arsenal,Chelsea,2.0,3.5,3.8,1.8,2.0\n")


class _FakeFootballData:
    """Stands in for ml.odds' downloads: writes `body` (a valid file by default) and counts calls."""

    def __init__(self, body=VALID_E0):
        self.body = body
        self.calls = []

    def _write(self, odds_dir, name):
        import os
        os.makedirs(odds_dir, exist_ok=True)
        path = os.path.join(odds_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.body)
        return path

    def season(self, season, odds_dir):
        from ml import odds as od
        self.calls.append(season)
        return self._write(odds_dir, f"E0_{od.season_code(season)}.csv")

    def fixtures(self, odds_dir):
        self.calls.append("fixtures")
        return self._write(odds_dir, "fixtures.csv")


def _fake_odds(monkeypatch, fake):
    from ml import odds as od
    monkeypatch.setattr(od, "fetch_season", fake.season)
    monkeypatch.setattr(od, "fetch_fixtures", fake.fixtures)
    monkeypatch.delenv("ODDS_DIR", raising=False)


def test_refresh_odds_fetches_past_seasons_once_and_skips_fresh_files(tmp_path, monkeypatch):
    from pipeline.current_season import odds_dir, refresh_odds
    fake = _FakeFootballData()
    _fake_odds(monkeypatch, fake)
    now = datetime.now(timezone.utc)

    first = refresh_odds(str(tmp_path), "2026-27", now=now)
    assert fake.calls == ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27", "fixtures"]
    assert set(first.values()) == {"fetched"} and len(first) == 8
    assert odds_dir(str(tmp_path)) == str(tmp_path / "odds")
    assert sorted(p.name for p in (tmp_path / "odds").iterdir())[-2:] == ["E0_2627.csv", "fixtures.csv"]

    fake.calls.clear()   # a run minutes later: nothing to download
    assert refresh_odds(str(tmp_path), "2026-27", now=now + timedelta(minutes=5)) == {
        "E0_2627.csv": "fresh", "fixtures.csv": "fresh"}
    assert fake.calls == []
    # The next run: this season's file and fixtures.csv only (two requests); past seasons are final
    assert refresh_odds(str(tmp_path), "2026-27", now=now + timedelta(hours=12)) == {
        "E0_2627.csv": "fetched", "fixtures.csv": "fetched"}
    assert fake.calls == ["2026-27", "fixtures"]


def test_refresh_odds_keeps_the_previous_file_when_a_download_fails(tmp_path, monkeypatch):
    import os
    from pipeline.current_season import refresh_odds
    fake = _FakeFootballData(body="<html>Service unavailable</html>\n")   # an error page, not a CSV
    _fake_odds(monkeypatch, fake)
    odds = tmp_path / "odds"
    odds.mkdir()
    (odds / "fixtures.csv").write_text(VALID_E0, encoding="utf-8")
    old = (datetime.now(timezone.utc) - timedelta(days=1)).timestamp()
    os.utime(odds / "fixtures.csv", (old, old))

    result = refresh_odds(str(tmp_path), "2021-22")
    assert result["fixtures.csv"].startswith("failed") and result["E0_2122.csv"].startswith("failed")
    assert (odds / "fixtures.csv").read_text(encoding="utf-8") == VALID_E0     # good copy kept
    assert sorted(p.name for p in odds.iterdir()) == ["fixtures.csv"]           # no staging left behind

    def offline(*args, **kwargs):
        raise OSError("network is unreachable")
    monkeypatch.setattr("ml.odds.fetch_fixtures", offline)
    assert refresh_odds(str(tmp_path), "2021-22", fixtures=True)["fixtures.csv"] == "failed: network is unreachable"


def test_refresh_odds_never_raises_when_the_odds_folder_cant_be_made(tmp_path, monkeypatch):
    from pipeline.current_season import refresh_odds
    fake = _FakeFootballData()
    _fake_odds(monkeypatch, fake)
    (tmp_path / "a_file").write_text("not a folder", encoding="utf-8")
    monkeypatch.setenv("ODDS_DIR", str(tmp_path / "a_file" / "odds"))     # e.g. a read-only mount
    result = refresh_odds(str(tmp_path), "2026-27")
    assert list(result) == ["odds_dir"] and result["odds_dir"].startswith("failed")
    assert fake.calls == []


def test_odds_dir_follows_the_environment(tmp_path, monkeypatch):
    from pipeline.current_season import odds_dir
    monkeypatch.setenv("ODDS_DIR", str(tmp_path / "elsewhere"))
    assert odds_dir(str(tmp_path)) == str(tmp_path / "elsewhere")
