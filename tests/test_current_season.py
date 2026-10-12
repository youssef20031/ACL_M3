"""Offline tests for pipeline.current_season (no network)."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from pipeline.current_season import (
    CSV_COLUMNS, build_season_rows, build_status, season_file_stem, season_label, update_xp_snapshots,
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
             "team": 1, "element_type": 3, "status": "a", "news": "", "now_cost": 55, "ep_next": "4.5"},
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
