"""Tests for services.season_service: pure computations plus the query layer with a
mocked Neo4j connection and a temporary LIVE_DATA_DIR."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from pipeline import load_current as lc
from services import season_service as ss

UTC = timezone.utc


def fx(id, gw, home, away, hs=None, as_=None, status=None, day=1, hour=15, **extra):
    """Raw fixture row as the FIXTURES_QUERY returns it."""
    if status is None:
        status = "finished" if hs is not None else "scheduled"
    return {"id": id, "gameweek": gw, "kickoff_time": datetime(2026, 8, day, hour, tzinfo=UTC),
            "status": status, "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_, **extra}


RAW_FIXTURES = [
    fx("1-2026-27", 1, "Arsenal", "Chelsea", 2, 0, day=1),
    fx("2-2026-27", 1, "Everton", "Fulham", 1, 1, day=1, hour=12),
    fx("3-2026-27", 2, "Chelsea", "Everton", 3, 1, day=8),
    fx("4-2026-27", 2, "Fulham", "Arsenal", 0, 1, day=8, hour=17),
    fx("5-2026-27", 3, "Arsenal", "Everton", status="in_progress", day=15),
    fx("6-2026-27", 3, "Fulham", "Chelsea", day=16),
    fx("7-2026-27", 4, "Chelsea", "Arsenal", day=22, home_difficulty=4, away_difficulty=3),
    fx("8-2026-27", 4, "Everton", "Brentford", day=22, hour=12),
]


class TestFixtureRows(unittest.TestCase):
    def test_clean_rows_sorted_and_scheduled_scores_dropped(self):
        rows = ss.clean_fixture_rows(RAW_FIXTURES + [
            fx("9-2026-27", 5, "Arsenal", "Fulham", 9, 9, status="scheduled", day=29),
            {"id": "10-2026-27", "gameweek": 5, "home_team": None, "away_team": "Fulham"},
        ])
        self.assertEqual([r["id"] for r in rows][:2], ["2-2026-27", "1-2026-27"])  # 12:00 before 15:00
        self.assertNotIn("10-2026-27", [r["id"] for r in rows])
        nine = next(r for r in rows if r["id"] == "9-2026-27")
        self.assertIsNone(nine["home_score"])

    def test_csv_schedule_dropped_when_api_fixtures_exist(self):
        csv = fx("fixture-2026-27-md1-ars-che", 1, "Arsenal", "Chelsea")
        rows = ss.clean_fixture_rows(RAW_FIXTURES + [csv])
        self.assertNotIn(csv["id"], [r["id"] for r in rows])
        # ...but used when it's all there is
        only_csv = ss.clean_fixture_rows([csv])
        self.assertEqual([r["id"] for r in only_csv], [csv["id"]])

    def test_kickoff_strings_and_neo4j_datetimes(self):
        class FakeNeo4jDateTime:
            def to_native(self):
                return datetime(2026, 8, 1, 15, tzinfo=UTC)
        rows = ss.clean_fixture_rows([
            {**fx("a", 1, "A", "B"), "kickoff_time": FakeNeo4jDateTime()},
            {**fx("b", 1, "C", "D"), "kickoff_time": "2026-08-01T12:00:00Z"},
        ])
        self.assertEqual([r["id"] for r in rows], ["b", "a"])
        self.assertEqual(ss.fixture_out(rows[0])["kickoff_time"], "2026-08-01T12:00:00+00:00")

    def test_fixtures_for_gameweek(self):
        rows = ss.clean_fixture_rows(RAW_FIXTURES)
        gw4 = ss.fixtures_for_gameweek(rows, 4)
        self.assertEqual([f["home_team"] for f in gw4], ["Everton", "Chelsea"])
        self.assertEqual(set(gw4[0]), {"id", "gameweek", "kickoff_time", "status", "home_team", "away_team",
                                       "home_score", "away_score", "home_difficulty", "away_difficulty"})
        self.assertEqual((gw4[1]["home_difficulty"], gw4[1]["away_difficulty"]), (4, 3))
        self.assertIsNone(gw4[0]["home_difficulty"])
        gw1 = ss.fixtures_for_gameweek(rows, 1)
        self.assertEqual((gw1[1]["home_score"], gw1[1]["away_score"]), (2, 0))

    def test_team_opponents_double_gameweek(self):
        rows = ss.clean_fixture_rows(RAW_FIXTURES + [fx("11-2026-27", 4, "Arsenal", "Fulham", day=25)])
        opp = ss.team_opponents(rows, 4)
        self.assertEqual(opp["Arsenal"], [{"team": "Chelsea", "home": False}, {"team": "Fulham", "home": True}])
        self.assertEqual(opp["Brentford"], [{"team": "Everton", "home": False}])
        self.assertNotIn("Leeds", opp)


class TestSeasonSummary(unittest.TestCase):
    def test_summary_from_fixtures_and_meta(self):
        rows = ss.clean_fixture_rows(RAW_FIXTURES)
        meta = {"next_gameweek": 4, "next_deadline": "2026-08-22T10:00:00Z",
                "status_fetched_at": "2026-08-20T08:00:00+00:00"}
        self.assertEqual(ss.summarize_season(rows, meta, "2026-27"), {
            "season": "2026-27", "current_gameweek": 3, "next_gameweek": 4,
            "next_deadline": "2026-08-22T10:00:00Z", "finished_gameweeks": [1, 2],
            "updated_at": "2026-08-20T08:00:00+00:00",
        })

    def test_summary_fallback_without_meta(self):
        rows = ss.clean_fixture_rows(RAW_FIXTURES)
        s = ss.summarize_season(rows, {}, "2026-27")
        self.assertEqual((s["next_gameweek"], s["next_deadline"], s["updated_at"]), (3, None, None))

    def test_summary_before_the_season(self):
        rows = ss.clean_fixture_rows([fx("1", 1, "A", "B"), fx("2", 2, "B", "A", day=8)])
        s = ss.summarize_season(rows, {}, "2026-27")
        self.assertEqual((s["current_gameweek"], s["next_gameweek"], s["finished_gameweeks"]), (None, 1, []))

    def test_default_gameweek(self):
        self.assertEqual(ss.default_gameweek({"next_gameweek": 4, "current_gameweek": 3}, [1, 2, 3, 4]), 4)
        self.assertEqual(ss.default_gameweek({"next_gameweek": None, "current_gameweek": 3}, [1, 2, 3]), 3)
        self.assertEqual(ss.default_gameweek({"next_gameweek": None, "current_gameweek": None}, [1, 2]), 1)
        self.assertEqual(ss.default_gameweek({}, []), 1)


class TestTable(unittest.TestCase):
    def setUp(self):
        self.rows = {r["team"]: r for r in ss.compute_table(ss.clean_fixture_rows(RAW_FIXTURES))}

    def test_results_and_points(self):
        ars = self.rows["Arsenal"]
        self.assertEqual((ars["played"], ars["won"], ars["drawn"], ars["lost"]), (2, 2, 0, 0))
        self.assertEqual((ars["goals_for"], ars["goals_against"], ars["goal_difference"], ars["points"]),
                         (3, 0, 3, 6))
        che = self.rows["Chelsea"]
        self.assertEqual((che["won"], che["lost"], che["goals_for"], che["goals_against"], che["points"]),
                         (1, 1, 3, 3, 3))
        eve = self.rows["Everton"]
        self.assertEqual((eve["drawn"], eve["lost"], eve["points"], eve["goal_difference"]), (1, 1, 1, -2))

    def test_in_progress_and_scheduled_not_counted(self):
        # Arsenal-Everton is in progress (no stored score) and must not count
        self.assertEqual(self.rows["Arsenal"]["played"], 2)
        self.assertEqual(self.rows["Everton"]["played"], 2)

    def test_zero_played_club_included(self):
        bre = self.rows["Brentford"]
        self.assertEqual((bre["played"], bre["points"], bre["form"]), (0, 0, []))
        self.assertEqual(bre["next_fixture"], {"opponent": "Everton", "home": False,
                                               "kickoff_time": "2026-08-22T12:00:00+00:00"})
        self.assertEqual(len(self.rows), 5)

    def test_form_oldest_first_and_capped(self):
        self.assertEqual(self.rows["Chelsea"]["form"], ["L", "W"])
        self.assertEqual(self.rows["Everton"]["form"], ["D", "L"])
        many = [fx(str(i), i, "A", "B", *(1, 0) if i % 2 else (0, 0), day=i) for i in range(1, 8)]
        a = next(r for r in ss.compute_table(ss.clean_fixture_rows(many)) if r["team"] == "A")
        self.assertEqual(a["form"], ["W", "D", "W", "D", "W"])  # games 3..7

    def test_next_fixture_skips_in_progress(self):
        # Arsenal's GW3 match is in progress; next is the scheduled GW4 game at Chelsea
        self.assertEqual(self.rows["Arsenal"]["next_fixture"]["opponent"], "Chelsea")
        self.assertFalse(self.rows["Arsenal"]["next_fixture"]["home"])

    def test_sort_order_and_positions(self):
        table = ss.compute_table(ss.clean_fixture_rows([
            fx("1", 1, "B", "A", 2, 1),        # B: 3 pts, GD +1, GF 2
            fx("2", 1, "C", "D", 3, 2, day=2),  # C: 3 pts, GD +1, GF 3
            fx("3", 1, "E", "F", 1, 0, day=3),  # E: 3 pts, GD +1, GF 1
            fx("4", 1, "H", "G", 1, 0, day=4),  # H: same as E -> name order puts E first
            fx("5", 2, "A", "D", 0, 0, day=9),
        ]))
        self.assertEqual([r["team"] for r in table], ["C", "B", "E", "H", "D", "A", "F", "G"])
        self.assertEqual([r["position"] for r in table], list(range(1, 9)))
        self.assertEqual(table[0]["goals_for"], 3)


class TestInjuries(unittest.TestCase):
    def test_status_order_then_newest_news_then_name(self):
        players = [
            {"name": "Zed", "status": "d", "news_added": "2026-10-01T00:00:00Z"},
            {"name": "Amy", "status": "n", "news_added": "2026-10-09T00:00:00Z"},
            {"name": "Bob", "status": "i", "news_added": "2026-09-01T00:00:00Z"},
            {"name": "Cal", "status": "i", "news_added": "2026-10-05T00:00:00Z"},
            {"name": "Dan", "status": "u", "news_added": None},
            {"name": "Eve", "status": "s", "news_added": "2026-10-02T00:00:00Z"},
            {"name": "Abe", "status": "i", "news_added": "2026-10-05T00:00:00Z"},
            {"name": "Fay", "status": "i", "news_added": None},
        ]
        self.assertEqual([p["name"] for p in ss.sort_injuries(players)],
                         ["Abe", "Cal", "Bob", "Fay", "Eve", "Zed", "Dan", "Amy"])

    def test_injury_rows(self):
        raw = [
            {"name": "Keeper", "team": "Arsenal", "position": "GKP", "status": "d",
             "chance_of_playing_next_round": 75.0, "news": "Knock", "news_added": "2026-10-01T10:00:00Z",
             "fpl_code": 123},
            {"name": "Fit", "team": "Arsenal", "position": "MID", "status": "a"},
            {"name": "Hurt", "team": "Leeds", "position": "FWD", "status": "i",
             "chance_of_playing_next_round": None, "news": "", "news_added": None, "fpl_code": None},
        ]
        rows = ss.injury_rows(raw, {"Keeper": "Kpr"})
        self.assertEqual([r["name"] for r in rows], ["Hurt", "Keeper"])
        keeper = rows[1]
        self.assertEqual((keeper["position"], keeper["web_name"], keeper["chance_of_playing_next_round"]),
                         ("GK", "Kpr", 75))
        self.assertEqual(keeper["news_added"], "2026-10-01T10:00:00+00:00")
        self.assertIsNone(rows[0]["news"])
        out = ss.with_photos(rows, lambda name, team, code: f"photo/{code}" if code else None)
        self.assertEqual(out[1]["photo"], "photo/123")
        self.assertIsNone(out[0]["photo"])
        self.assertNotIn("_code", out[1])


class TestLoadCurrentDifficulty(unittest.TestCase):
    def test_fixture_rows_carry_fpl_difficulty(self):
        fixtures = [{"id": 9, "event": 3, "team_h": 1, "team_a": 2, "finished": False, "started": False,
                     "kickoff_time": "2026-09-01T14:00:00Z", "team_h_difficulty": 2, "team_a_difficulty": 5},
                    {"id": 10, "event": 3, "team_h": 2, "team_a": 1, "finished": False, "started": False,
                     "kickoff_time": None}]
        rows = lc.fixture_rows(fixtures, {1: "Arsenal", 2: "Leeds"}, "2026-27")
        self.assertEqual((rows[0]["home_difficulty"], rows[0]["away_difficulty"]), (2, 5))
        self.assertEqual((rows[1]["home_difficulty"], rows[1]["away_difficulty"]), (None, None))
        self.assertIn("f.home_difficulty = fix.home_difficulty", lc.FIXTURES_QUERY)


# ------------------------------------------------------------------ query layer

def fake_conn(points_rows=None, injury_rows=None):
    conn = MagicMock()

    def execute_query(query, params=None):
        if query == ss.FIXTURES_QUERY:
            return [dict(r) for r in RAW_FIXTURES]
        if query == ss.GAMEWEEK_POINTS_QUERY:
            return [r for r in (points_rows or []) if r.get("_gw", params["gameweek"]) == params["gameweek"]]
        if query == ss.INJURIES_QUERY:
            return injury_rows or []
        raise AssertionError(f"unexpected query {query}")

    conn.execute_query.side_effect = execute_query
    return conn


class LiveDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"LIVE_DATA_DIR": self.tmp.name})
        self.env.start()
        ss.clear_cache()
        with open(os.path.join(self.tmp.name, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"season": "2026-27", "next_gameweek": 4, "next_deadline": "2026-08-22T10:00:00Z",
                       "status_fetched_at": "2026-08-20T08:00:00+00:00"}, f)

    def tearDown(self):
        ss.clear_cache()
        self.env.stop()
        self.tmp.cleanup()


class TestQueryLayer(LiveDirTestCase):
    def test_season_current_and_cache(self):
        conn = fake_conn()
        first = ss.get_season_current(conn, "2026-27")
        self.assertEqual((first["current_gameweek"], first["next_gameweek"]), (3, 4))
        ss.get_fixtures(conn, None, "2026-27")
        ss.get_table(conn, "2026-27")
        self.assertEqual(conn.execute_query.call_count, 1)    # fixtures fetched once
        ss.clear_cache()
        ss.get_season_current(conn, "2026-27")
        self.assertEqual(conn.execute_query.call_count, 2)

    def test_fixtures_default_and_unknown_gameweek(self):
        conn = fake_conn()
        res = ss.get_fixtures(conn, None, "2026-27")
        self.assertEqual((res["gameweek"], res["gameweeks"], res["season"]), (4, [1, 2, 3, 4], "2026-27"))
        self.assertEqual(len(res["fixtures"]), 2)
        with self.assertRaises(ss.NotFoundError):
            ss.get_fixtures(conn, 9, "2026-27")

    def test_table_payload(self):
        res = ss.get_table(fake_conn(), "2026-27")
        self.assertEqual(res["updated_at"], "2026-08-20T08:00:00+00:00")
        self.assertEqual(res["rows"][0]["team"], "Arsenal")

    def test_injuries_filter_and_web_names(self):
        pd.DataFrame([{"element": 1, "code": 55, "name": "Hurt", "web_name": "H."}]).to_csv(
            os.path.join(self.tmp.name, "player_status_2026_27.csv"), index=False)
        injuries = [
            {"name": "Hurt", "team": "Arsenal", "position": "DEF", "status": "i", "fpl_code": 55},
            {"name": "Banned", "team": "Leeds", "position": "MID", "status": "s", "fpl_code": None},
        ]
        conn = fake_conn(injury_rows=injuries)
        res = ss.get_injuries(conn, None, photo_for=lambda n, t, c: f"p{c}" if c else None, season="2026-27")
        self.assertEqual([p["name"] for p in res["players"]], ["Hurt", "Banned"])
        self.assertEqual((res["players"][0]["web_name"], res["players"][0]["photo"]), ("H.", "p55"))
        self.assertEqual(res["updated_at"], "2026-08-20T08:00:00+00:00")
        only = ss.get_injuries(conn, "leeds", season="2026-27")
        self.assertEqual([p["name"] for p in only["players"]], ["Banned"])
        self.assertIsNone(only["players"][0]["photo"])

    def test_actual_team_of_the_week(self):
        rows = []
        teams = ["Arsenal", "Chelsea", "Everton", "Fulham"]
        for k, pos in enumerate(["GK"] * 2 + ["DEF"] * 6 + ["MID"] * 6 + ["FWD"] * 4):
            rows.append({"name": f"{pos}{k}", "position": pos, "team": teams[k % 4], "total_points": k % 7,
                         "status": "a", "chance_of_playing_next_round": None, "news": "", "fpl_code": k,
                         "_gw": 2})
        # A second appearance in the gameweek (DGW) is summed
        rows.append({**rows[-1], "total_points": 10})
        conn = fake_conn(points_rows=rows)
        res = ss.get_team_of_the_week(conn, "actual", None, season="2026-27")
        self.assertEqual((res["mode"], res["gameweek"]), ("actual", 2))   # latest finished gameweek
        self.assertEqual((res["bench"], res["budget"], res["total_cost"]), ([], None, None))
        self.assertEqual(len(res["starters"]), 11)
        top = max(res["starters"], key=lambda p: p["points"])
        self.assertEqual(top["name"], "FWD17")
        self.assertEqual(top["points"], 13.0)          # 3 + 10
        self.assertTrue(top["captain"])
        self.assertEqual(sum(p["captain"] for p in res["starters"]), 1)
        self.assertEqual(sum(p["vice_captain"] for p in res["starters"]), 1)
        self.assertEqual(res["total_points"], sum(p["points"] for p in res["starters"]))  # no doubling
        self.assertEqual(res["starters"][0]["position"], "GK")
        fwd17 = top
        self.assertEqual(fwd17["team"], "Chelsea")
        self.assertEqual(fwd17["opponents"], [{"team": "Everton", "home": True}])
        self.assertIsNone(fwd17["cost"])
        self.assertIsNone(fwd17["photo"])              # no photo_for given
        with self.assertRaises(ss.NotFoundError):
            ss.get_team_of_the_week(conn, "actual", 4, season="2026-27")   # no points yet
        with self.assertRaises(ss.NotFoundError):
            ss.get_team_of_the_week(conn, "actual", 30, season="2026-27")  # unknown gameweek

    def _write_predictions(self, gameweek=4, play60=False):
        clubs = ["Arsenal", "Chelsea", "Everton", "Fulham", "Brentford", "Leeds", "Spurs", "Wolves"]
        rows = []
        for k, pos in enumerate((["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3) * 2):
            rows.append({"name": f"{pos}-{k}", "position": pos, "team": clubs[k % 8],
                         "predicted_points": round(1 + (k * 37 % 23) / 3, 2), "value": 40 + (k * 13) % 30,
                         "opponent": "Somebody", "is_home": True, "gameweek": gameweek,
                         "kickoff_time": "2026-08-22T14:00:00Z", "fixtures": 1, "status": "a",
                         "chance_of_playing_next_round": None, "news": None, "element": k})
        rows.append({**rows[0], "name": "Injured Star", "predicted_points": 50, "status": "i"})
        rows.append({**rows[0], "name": "Blank", "predicted_points": 0.0})
        rows.append({**rows[2], "name": "Doubtful Star", "position": "DEF", "predicted_points": 20,
                     "status": "d", "chance_of_playing_next_round": 75, "news": "Knock", "team": "Chelsea"})
        if play60:
            for k, r in enumerate(rows):
                r["play60"] = round(0.5 + (k * 29 % 50) / 100, 2)
        pd.DataFrame(rows).to_csv(os.path.join(self.tmp.name, "next_gw_predictions.csv"), index=False)

    def test_safer_pick(self):
        self._write_predictions(play60=True)
        default = ss.get_team_of_the_week(None, "predicted", None, season="2026-27")
        safer = ss.get_team_of_the_week(None, "predicted", None, season="2026-27", pick="safer")
        self.assertEqual((default["pick"], safer["pick"]), ("default", "safer"))
        players = safer["starters"] + safer["bench"]
        self.assertEqual(len(players), 15)
        self.assertTrue(all(0 <= p["play60"] <= 1 for p in players))
        self.assertLessEqual(safer["total_cost"], 100.0)
        # Each player shows his prediction either way; only the selection differs
        shown = {p["name"]: p["points"] for p in default["starters"] + default["bench"]}
        for p in players:
            if p["name"] in shown:
                self.assertEqual(p["points"], shown[p["name"]])
        with self.assertRaises(ValueError):
            ss.get_team_of_the_week(None, "actual", None, season="2026-27", pick="safer")

    def test_safer_pick_needs_play60(self):
        self._write_predictions()
        with self.assertRaises(ss.UnavailableError):
            ss.get_team_of_the_week(None, "predicted", None, season="2026-27", pick="safer")
        default = ss.get_team_of_the_week(None, "predicted", None, season="2026-27")
        self.assertEqual(default["pick"], "default")
        self.assertIsNone(default["starters"][0]["play60"])

    def test_predicted_team_of_the_week(self):
        self._write_predictions()
        res = ss.get_team_of_the_week(fake_conn(), "predicted", None,
                                      photo_for=lambda n, t, c: None, season="2026-27")
        self.assertEqual((res["mode"], res["gameweek"], res["budget"]), ("predicted", 4, 100.0))
        self.assertEqual((len(res["starters"]), len(res["bench"])), (11, 4))
        names = [p["name"] for p in res["starters"] + res["bench"]]
        self.assertNotIn("Injured Star", names)
        self.assertNotIn("Blank", names)
        star = next(p for p in res["starters"] if p["name"] == "Doubtful Star")
        self.assertTrue(star["captain"])
        self.assertEqual((star["status"], star["chance_of_playing_next_round"], star["news"]), ("d", 75, "Knock"))
        # Opponents come from the graph's gameweek-4 fixtures, not the CSV text
        self.assertEqual(star["opponents"], [{"team": "Arsenal", "home": True}])
        captain_pts = star["points"]
        self.assertAlmostEqual(res["total_points"],
                               round(sum(p["points"] for p in res["starters"]) + captain_pts, 2))
        squad_cost = sum(p["cost"] for p in res["starters"] + res["bench"])
        self.assertAlmostEqual(res["total_cost"], round(squad_cost, 1))
        self.assertLessEqual(res["total_cost"], 100.0)
        self.assertEqual(res["formation"].count("-"), 2)
        with self.assertRaises(ss.NotFoundError):
            ss.get_team_of_the_week(fake_conn(), "predicted", 5, season="2026-27")

    def test_predicted_without_graph_uses_prediction_opponents(self):
        self._write_predictions()
        res = ss.get_team_of_the_week(None, "predicted", 4, season="2026-27")
        self.assertEqual(res["starters"][0]["opponents"], [{"team": "Somebody", "home": True}])

    def test_opponents_from_prediction_double_gameweek(self):
        self.assertEqual(ss.opponents_from_prediction({"opponent": "Chelsea & Everton", "is_home": False}),
                         [{"team": "Chelsea", "home": False}, {"team": "Everton", "home": False}])
        self.assertEqual(ss.opponents_from_prediction({"opponent": "No fixture", "is_home": False}), [])


if __name__ == "__main__":
    unittest.main()
