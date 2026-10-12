"""Tests for pipeline.load_current (current-season graph load) with fake inputs."""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import pandas as pd

from pipeline import load_current as lc

SEASON = "2026-27"

TEAMS = [
    {"id": 1, "name": "Arsenal"},
    {"id": 7, "name": "Coventry City"},
    {"id": 12, "name": "Ipswich Town"},
]

FIXTURES = [
    {"id": 1, "event": 1, "team_h": 1, "team_a": 7, "team_h_score": 3, "team_a_score": 0,
     "kickoff_time": "2026-08-21T19:00:00Z", "finished": True, "started": True},
    {"id": 2, "event": 1, "team_h": 12, "team_a": 1, "team_h_score": 1, "team_a_score": 1,
     "kickoff_time": "2026-08-22T14:00:00Z", "finished": False, "started": True},
    {"id": 3, "event": 2, "team_h": 7, "team_a": 12, "team_h_score": None, "team_a_score": None,
     "kickoff_time": "2026-08-29T14:00:00Z", "finished": False, "started": False},
    {"id": 4, "event": None, "team_h": 7, "team_a": 1, "team_h_score": None, "team_a_score": None,
     "kickoff_time": None, "finished": False, "started": False},
]


def season_df() -> pd.DataFrame:
    base = {
        "xP": None, "assists": 0, "bonus": 0, "bps": 5, "clean_sheets": 0, "creativity": 1.0,
        "expected_assists": 0.0, "expected_goal_involvements": 0.0, "expected_goals": 0.0,
        "expected_goals_conceded": 0.5, "goals_conceded": 0, "goals_scored": 0, "ict_index": 1.0,
        "influence": 1.0, "kickoff_time": "2026-08-21T19:00:00Z", "minutes": 90, "modified": False,
        "own_goals": 0, "penalties_missed": 0, "penalties_saved": 0, "red_cards": 0, "round": 1,
        "saves": 0, "selected": 1000, "starts": 1, "team_a_score": 0, "team_h_score": 3,
        "threat": 1.0, "total_points": 2, "transfers_balance": 0, "transfers_in": 0,
        "transfers_out": 0, "value": 60, "yellow_cards": 0, "clearances_blocks_interceptions": 0,
        "defensive_contribution": 0, "recoveries": 0, "tackles": 0, "GW": 1, "fixture": 1,
    }
    rows = [
        {**base, "name": "Bukayo Saka", "position": "MID", "team": "Arsenal", "element": 16,
         "opponent_team": 7, "was_home": True, "goals_scored": 2, "total_points": 13},
        {**base, "name": "Coventry Keeper", "position": "GK", "team": "Coventry City", "element": 200,
         "opponent_team": 1, "was_home": False, "goals_conceded": 3, "total_points": 1},
        {**base, "name": "Old Boy", "position": "DEF", "team": "Coventry City", "element": 201,
         "opponent_team": 1, "was_home": False},
    ]
    return pd.DataFrame(rows)


def status_df() -> pd.DataFrame:
    rows = [
        {"element": 16, "code": 223340, "name": "Bukayo Saka", "web_name": "Saka", "team": "Arsenal",
         "position": "MID", "status": "a", "chance_of_playing_this_round": None,
         "chance_of_playing_next_round": None, "news": None, "news_added": None, "now_cost": 100,
         "selected_by_percent": 30.0, "form": 6.0, "ep_next": 6.5, "fetched_at": "2026-10-12T13:00:00+00:00"},
        {"element": 200, "code": 500, "name": "Coventry Keeper", "web_name": "Keeper", "team": "Coventry City",
         "position": "GK", "status": "d", "chance_of_playing_this_round": 75.0,
         "chance_of_playing_next_round": 50.0, "news": "Knock - 50% chance of playing",
         "news_added": "2026-10-10T10:00:00Z", "now_cost": 45, "selected_by_percent": 1.0,
         "form": 1.0, "ep_next": 1.0, "fetched_at": "2026-10-12T13:00:00+00:00"},
        {"element": 30, "code": 600, "name": "Loaned Out", "web_name": "Loaned", "team": "Arsenal",
         "position": "FWD", "status": "u", "chance_of_playing_this_round": 0.0,
         "chance_of_playing_next_round": 0.0, "news": "Has joined Hamburger SV on loan for the rest of the season",
         "news_added": "2026-09-01T10:00:00Z", "now_cost": 50, "selected_by_percent": 0.1,
         "form": 0.0, "ep_next": 0.0, "fetched_at": "2026-10-12T13:00:00+00:00"},
        {"element": 31, "code": 601, "name": "Not Registered", "web_name": "Unreg", "team": "Arsenal",
         "position": "DEF", "status": "u", "chance_of_playing_this_round": 0.0,
         "chance_of_playing_next_round": 0.0, "news": "not included in squad.",
         "news_added": "2026-09-01T10:00:00Z", "now_cost": 40, "selected_by_percent": 0.0,
         "form": 0.0, "ep_next": 0.0, "fetched_at": "2026-10-12T13:00:00+00:00"},
    ]
    return pd.DataFrame(rows)


class TestTeamsAndFixtures(unittest.TestCase):
    def test_team_names_from_bootstrap_are_mapped_to_graph_names(self):
        teams = lc.team_names_by_id(TEAMS)
        self.assertEqual(teams, {1: "Arsenal", 7: "Coventry City", 12: "Ipswich"})
        self.assertEqual(lc.team_names_by_id({"12": "Ipswich Town"}), {12: "Ipswich"})

    def test_team_names_inferred_from_season_csv(self):
        teams = lc.team_names_by_id(None, FIXTURES, season_df())
        self.assertEqual(teams, {1: "Arsenal", 7: "Coventry City"})

    def test_fixture_rows(self):
        rows = {r["fixture_id"]: r for r in lc.fixture_rows(FIXTURES, lc.team_names_by_id(TEAMS), SEASON)}
        self.assertEqual(set(rows), {1, 2, 3, 4})

        done = rows[1]
        self.assertEqual(done["id"], "1-2026-27")
        self.assertEqual(done["season"], SEASON)
        self.assertEqual((done["gameweek"], done["gw_id"]), (1, "2026-27-GW1"))
        self.assertEqual((done["home_team"], done["away_team"]), ("Arsenal", "Coventry City"))
        self.assertEqual((done["status"], done["home_score"], done["away_score"]), ("finished", 3, 0))
        self.assertEqual(done["kickoff_time"], "2026-08-21T19:00:00Z")

        # Provisional scores of a match in progress are not stored
        live = rows[2]
        self.assertEqual((live["status"], live["home_score"], live["away_score"]), ("in_progress", None, None))
        self.assertEqual(live["home_team"], "Ipswich")

        self.assertEqual(rows[3]["status"], "scheduled")
        self.assertIsNone(rows[3]["home_score"])

        # Postponed, not yet rescheduled: no gameweek
        self.assertIsNone(rows[4]["gameweek"])
        self.assertIsNone(rows[4]["gw_id"])
        self.assertIsNone(rows[4]["kickoff_time"])

    def test_fixture_with_unknown_team_is_skipped(self):
        rows = lc.fixture_rows(FIXTURES, {1: "Arsenal", 7: "Coventry City"}, SEASON)
        self.assertEqual({r["fixture_id"] for r in rows}, {1, 4})


class TestPlayers(unittest.TestCase):
    def test_keeps_club_rules(self):
        self.assertTrue(lc.keeps_club("a", ""))
        self.assertTrue(lc.keeps_club("i", "Has joined the injury list"))  # only "u" can lose the club
        self.assertTrue(lc.keeps_club("d", None))
        self.assertTrue(lc.keeps_club("u", "not included in squad."))
        self.assertTrue(lc.keeps_club("u", ""))
        for news in ("Has joined FC Barcelona permanently",
                     "Has joined Rangers on loan for the rest of the season",
                     "has departed the club as a free agent.",
                     "has returned to Getafe CF",
                     "Has left the club by mutual consent"):
            self.assertFalse(lc.keeps_club("u", news), news)

    def test_player_rows(self):
        rows = {r["name"]: r for r in lc.player_rows(status_df(), season_df(), SEASON)}
        self.assertEqual(set(rows), {"Bukayo Saka", "Coventry Keeper", "Loaned Out", "Not Registered", "Old Boy"})

        saka = rows["Bukayo Saka"]
        self.assertTrue(saka["plays_for"])
        self.assertEqual((saka["team"], saka["position"]), ("Arsenal", "MID"))
        props = saka["props"]
        self.assertEqual(props["element_id"], 16)
        self.assertEqual(props["fpl_code"], 223340)
        self.assertEqual(props["status"], "a")
        self.assertEqual(props["news"], "")
        self.assertIsNone(props["chance_of_playing_next_round"])
        self.assertEqual(props["name_normalized"], "bukayo saka")
        self.assertEqual((props["last_team"], props["last_season"]), ("Arsenal", SEASON))
        self.assertEqual(props["status_updated_at"], "2026-10-12T13:00:00+00:00")

        keeper = rows["Coventry Keeper"]["props"]
        self.assertEqual(keeper["chance_of_playing_next_round"], 50)
        self.assertIsInstance(keeper["chance_of_playing_next_round"], int)
        self.assertEqual(keeper["chance_of_playing_this_round"], 75)

        # Left the club: no PLAYS_FOR, but last_team is kept
        loaned = rows["Loaned Out"]
        self.assertFalse(loaned["plays_for"])
        self.assertEqual(loaned["props"]["last_team"], "Arsenal")
        # Unavailable for another reason: still in the squad
        self.assertTrue(rows["Not Registered"]["plays_for"])

        # In the season file but no longer in the FPL game: no current club, no status fields
        old = rows["Old Boy"]
        self.assertFalse(old["plays_for"])
        self.assertEqual((old["team"], old["position"]), ("Coventry City", "DEF"))
        self.assertNotIn("status", old["props"])
        self.assertEqual(old["props"]["element_id"], 201)

    def test_player_rows_without_status_file(self):
        rows = lc.player_rows(None, season_df(), SEASON)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not r["plays_for"] for r in rows))


class TestLiveFiles(unittest.TestCase):
    def test_find_live_files(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(lc.find_live_files(d))
            self.assertFalse(lc.has_live_data(d))
            season_df().to_csv(os.path.join(d, "FPL_2026_2027.csv"), index=False)
            files = lc.find_live_files(d)
            self.assertEqual(files["season"], SEASON)
            self.assertIsNone(files["status_csv"])
            status_df().to_csv(os.path.join(d, "player_status_2026_27.csv"), index=False)
            self.assertTrue(lc.find_live_files(d)["status_csv"].endswith("player_status_2026_27.csv"))

    def test_find_live_files_prefers_meta(self):
        with tempfile.TemporaryDirectory() as d:
            season_df().to_csv(os.path.join(d, "FPL_2026_2027.csv"), index=False)
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"season": "2026-27", "files": {"season": "FPL_2026_2027.csv",
                                                          "status": "player_status_2026_27.csv"}}, f)
            self.assertEqual(lc.find_live_files(d)["season"], SEASON)


class FakeConnection:
    """Records writes; answers the loader's read-back queries."""

    def __init__(self, played_in_rows=None):
        self.writes = []
        self.queries = []
        self.played_in_rows = played_in_rows

    def execute_write(self, query, params=None):
        self.writes.append((query, params or {}))
        return MagicMock()

    def execute_query(self, query, params=None):
        self.queries.append((query, params or {}))
        if "STARTS WITH $prefix" in query:
            return [{"deleted": 380}]
        if "IS :: STRING" in query:
            return [{"converted": 0}]
        if "AS dropped" in query:
            return [{"dropped": 5}]
        if "AS rows" in query:
            return [{"rows": self.played_in_rows}]
        return []

    def writes_matching(self, text):
        return [p for q, p in self.writes if text in q]


class TestLoadCurrentSeason(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        season_df().to_csv(os.path.join(self.dir.name, "FPL_2026_2027.csv"), index=False)
        status_df().to_csv(os.path.join(self.dir.name, "player_status_2026_27.csv"), index=False)

    def tearDown(self):
        self.dir.cleanup()

    def load(self, conn):
        return lc.load_current_season(conn, live_dir=self.dir.name, fpl_fixtures=FIXTURES, fpl_teams=TEAMS)

    def test_loads_every_step(self):
        conn = FakeConnection(played_in_rows=3)
        counts = self.load(conn)

        self.assertEqual(counts["season"], SEASON)
        self.assertEqual(counts["fixtures"], 4)
        self.assertEqual(counts["finished_fixtures"], 1)
        self.assertEqual(counts["gameweeks"], 2)
        self.assertEqual(counts["csv_schedule_fixtures_deleted"], 380)
        self.assertEqual(counts["players"], 5)
        self.assertEqual(counts["plays_for"], 3)
        self.assertEqual(counts["played_in"], 3)
        self.assertEqual(counts["season_stats"], 3)

        gw = conn.writes_matching("MERGE (gw:Gameweek")[0]
        self.assertEqual((gw["season"], gw["start_year"], gw["gameweeks"]), (SEASON, 2026, [1, 2]))

        fixture_batch = conn.writes_matching("MERGE (f:Fixture {id: fix.id})")[0]["batch"]
        self.assertEqual({f["id"] for f in fixture_batch}, {"1-2026-27", "2-2026-27", "3-2026-27", "4-2026-27"})

        # The old CSV schedule nodes are removed by their id prefix
        prefix_queries = [p for q, p in conn.queries if "STARTS WITH $prefix" in q]
        self.assertEqual(prefix_queries, [{"prefix": "fixture-2026-27-"}])

        # PLAYS_FOR is dropped for everyone not in the current squad
        current = [p for q, p in conn.queries if "AS dropped" in q][0]["current"]
        self.assertEqual(set(current), {"Bukayo Saka", "Coventry Keeper", "Not Registered"})

        perf_batch = conn.writes_matching("MERGE (p)-[r:PLAYED_IN]->(f)")[0]["batch"]
        saka = next(r for r in perf_batch if r["name"] == "Bukayo Saka")
        self.assertEqual((saka["season"], saka["fixture"], saka["GW"], saka["team_x"]), (SEASON, 1, 1, "Arsenal"))
        self.assertIs(saka["was_home"], True)
        self.assertIsNone(saka["xP"])
        self.assertEqual(saka["form"], 13.0)

        # SEASON_STATS: deleted for the season, then rebuilt from the CSV
        self.assertEqual(conn.writes_matching("DELETE st"), [{"season": SEASON}])
        stats = {r["name"]: r for r in conn.writes_matching("MERGE (p)-[st:SEASON_STATS]->(s)")[0]["batch"]}
        self.assertEqual((stats["Bukayo Saka"]["goals"], stats["Bukayo Saka"]["total_points"]), (2, 13))
        self.assertEqual(stats["Bukayo Saka"]["season"], SEASON)

        # Row counts matched, so PLAYED_IN was not rebuilt
        self.assertEqual(conn.writes_matching("DELETE r"), [])

    def test_rebuilds_played_in_when_graph_has_extra_rows(self):
        conn = FakeConnection(played_in_rows=4)
        self.load(conn)
        self.assertEqual(conn.writes_matching("DELETE r"), [{"season": SEASON}])
        self.assertEqual(len(conn.writes_matching("MERGE (p)-[r:PLAYED_IN]->(f)")), 2)

    def test_same_inputs_give_same_writes(self):
        first, second = FakeConnection(played_in_rows=3), FakeConnection(played_in_rows=3)
        self.load(first)
        self.load(second)
        self.assertEqual(first.writes, second.writes)

    def test_missing_live_data_raises(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                lc.load_current_season(FakeConnection(), live_dir=empty, fpl_fixtures=FIXTURES)


class TestCsvFallbackSkip(unittest.TestCase):
    def test_csv_schedule_is_skipped_when_api_fixtures_exist(self):
        from services.fixture_service import FixtureService

        conn = MagicMock()
        conn.execute_query.return_value = [{"api_fixtures": 380}]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fixtures.csv")
            pd.DataFrame([{"Matchday": 1, "Home Team": "Arsenal", "Away Team": "Coventry City",
                           "Date": "21/08/2026", "Time": "20:00"}]).to_csv(path, index=False)
            self.assertEqual(FixtureService(conn).load_fixtures_from_csv(path, season_id=SEASON), 0)
        conn.execute_write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
