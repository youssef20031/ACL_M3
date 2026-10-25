"""Tests for services.model_review: FPL auto-subs and captaincy, saving picks before the
deadline, the review payload and its endpoint, and the committed GW7 pick."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from services import model_review as mr
from services import season_service as ss


def pl(name, position, played=True, role="starter", captain=False, vice=False, element=None, predicted=1.0,
       fixtures=(1,)):
    return {"element": element, "name": name, "position": position, "team": "T", "cost": 5.0,
            "predicted_points": predicted, "role": role, "captain": captain, "vice_captain": vice,
            "fixtures": list(fixtures), "opponents": [], "played": played}


def xi_352(**played):
    """A 3-5-2 XI; `played` overrides a player's played flag by name."""
    names = [("GK1", "GK"), ("D1", "DEF"), ("D2", "DEF"), ("D3", "DEF"), ("M1", "MID"), ("M2", "MID"),
             ("M3", "MID"), ("M4", "MID"), ("M5", "MID"), ("F1", "FWD"), ("F2", "FWD")]
    return [pl(n, pos, played.get(n, True)) for n, pos in names]


def bench(*spec):
    return [pl(n, pos, p, role="bench") for n, pos, p in spec]


class TestAutoSubs(unittest.TestCase):
    def test_first_eligible_bench_player_comes_on(self):
        xi, b, subs = mr.apply_auto_subs(xi_352(M5=False),
                                         bench(("GK2", "GK", True), ("D4", "DEF", True), ("D5", "DEF", True),
                                               ("F3", "FWD", True)))
        self.assertEqual(subs, [("M5", "D4")])
        self.assertEqual(mr.formation_of(xi), "4-4-2")
        # The replaced starter takes his replacement's bench slot
        self.assertEqual([p["name"] for p in b], ["GK2", "M5", "D5", "F3"])

    def test_formation_is_kept(self):
        # A defender drops out of a 3-5-2: a midfielder would leave two defenders
        xi, _, subs = mr.apply_auto_subs(xi_352(D1=False),
                                         bench(("GK2", "GK", True), ("F3", "FWD", True), ("D4", "DEF", True),
                                               ("D5", "DEF", True)))
        self.assertEqual(subs, [("D1", "D4")])
        self.assertEqual(mr.formation_of(xi), "3-5-2")

    def test_goalkeeper_only_for_goalkeeper(self):
        _, _, subs = mr.apply_auto_subs(xi_352(GK1=False),
                                        bench(("GK2", "GK", False), ("D4", "DEF", True), ("D5", "DEF", True),
                                              ("F3", "FWD", True)))
        self.assertEqual(subs, [])
        _, _, subs = mr.apply_auto_subs(xi_352(GK1=False),
                                        bench(("GK2", "GK", True), ("D4", "DEF", True), ("D5", "DEF", True),
                                              ("F3", "FWD", True)))
        self.assertEqual(subs, [("GK1", "GK2")])

    def test_bench_player_who_didnt_play_is_skipped(self):
        _, _, subs = mr.apply_auto_subs(xi_352(M5=False),
                                        bench(("GK2", "GK", True), ("D4", "DEF", False), ("D5", "DEF", True),
                                              ("F3", "FWD", True)))
        self.assertEqual(subs, [("M5", "D5")])

    def test_undecided_bench_player_blocks_the_rest(self):
        _, _, subs = mr.apply_auto_subs(xi_352(M5=False),
                                        bench(("GK2", "GK", True), ("D4", "DEF", None), ("D5", "DEF", True),
                                              ("F3", "FWD", True)))
        self.assertEqual(subs, [])

    def test_undecided_starter_isnt_replaced(self):
        _, _, subs = mr.apply_auto_subs(xi_352(M5=None),
                                        bench(("GK2", "GK", True), ("D4", "DEF", True), ("D5", "DEF", True),
                                              ("F3", "FWD", True)))
        self.assertEqual(subs, [])

    def test_each_bench_player_used_once(self):
        _, _, subs = mr.apply_auto_subs(xi_352(M4=False, M5=False),
                                        bench(("GK2", "GK", True), ("D4", "DEF", True), ("D5", "DEF", False),
                                              ("F3", "FWD", True)))
        self.assertEqual(subs, [("M4", "D4"), ("M5", "F3")])


def picks_352(captain="F1", vice="M1"):
    starters = [("GK1", "GK"), ("D1", "DEF"), ("D2", "DEF"), ("D3", "DEF"), ("M1", "MID"), ("M2", "MID"),
                ("M3", "MID"), ("M4", "MID"), ("M5", "MID"), ("F1", "FWD"), ("F2", "FWD")]
    subs = [("GK2", "GK"), ("D4", "DEF"), ("D5", "DEF"), ("F3", "FWD")]
    players = []
    for k, (n, pos) in enumerate(starters + subs):
        p = pl(n, pos, role="starter" if k < 11 else "bench", captain=n == captain, vice=n == vice,
               element=k + 1, predicted=2.0, fixtures=(100 + k,))
        p.pop("played")
        players.append(p)
    return {"season": "2026-27", "gameweek": 5, "deadline": "2026-09-19T10:00:00Z", "players": players}


def actual(points=3, minutes=90, **over):
    """Everyone played 90 minutes for 3 points unless overridden as name=(points, minutes)."""
    out = {}
    for k, p in enumerate(picks_352()["players"]):
        pts, mins = over.get(p["name"], (points, minutes))
        out[k + 1] = {"points": pts, "minutes": mins}
    return out


FINISHED = {"finished": True, "average_entry_score": 52, "highest_score": 130}


class TestScorePicks(unittest.TestCase):
    def test_totals_and_captain(self):
        r = mr.score_picks(picks_352(), actual(F1=(10, 90)), FINISHED, [])
        self.assertEqual(r["points"], 10 * 2 + 3 * 10)         # captain 10 x2, ten others 3 each
        self.assertEqual(r["bench_points"], 12)
        self.assertEqual(r["predicted_points"], 24.0)          # 11 x 2.0, captain again
        self.assertEqual((r["fpl_average"], r["fpl_highest"], r["status"]), (52, 130, "finished"))
        f1 = next(p for p in r["starters"] if p["name"] == "F1")
        self.assertEqual(f1["multiplier"], 2)
        self.assertTrue(all(p["multiplier"] == 0 for p in r["bench"]))

    def test_vice_doubles_when_captain_didnt_play(self):
        r = mr.score_picks(picks_352(), actual(F1=(0, 0), M1=(7, 90)), FINISHED, [])
        m1 = next(p for p in r["starters"] if p["name"] == "M1")
        self.assertEqual(m1["multiplier"], 2)
        # One forward is enough, so the first bench outfielder comes on (4-5-1)
        self.assertEqual(r["auto_subs"], [{"out": "F1", "in": "D4"}])
        self.assertEqual(r["formation"], "4-5-1")
        self.assertEqual(r["points"], 7 * 2 + 3 * 10)

    def test_no_doubling_when_neither_played(self):
        r = mr.score_picks(picks_352(), actual(F1=(0, 0), M1=(0, 0)), FINISHED, [])
        self.assertTrue(all(p["multiplier"] in (0, 1) for p in r["starters"] + r["bench"]))

    def test_live_gameweek_waits_for_unfinished_matches(self):
        # Only GK1's fixture (100) has finished; nobody else has a row yet
        live = {1: {"points": 6, "minutes": 90}}
        r = mr.score_picks(picks_352(), live, {"finished": False}, [100])
        self.assertEqual(r["status"], "live")
        self.assertEqual(r["points"], 6)
        self.assertEqual(r["auto_subs"], [])
        d1 = next(p for p in r["starters"] if p["name"] == "D1")
        self.assertIsNone(d1["played"])
        self.assertIsNone(d1["points"])
        self.assertIsNone(r["fpl_average"])

    def test_finished_fixture_without_minutes_counts_as_didnt_play(self):
        live = {k: {"points": 2, "minutes": 90} for k in range(1, 16) if k != 9}   # M5 (element 9) has no row
        r = mr.score_picks(picks_352(), live, {"finished": False}, range(100, 115))
        self.assertEqual(r["auto_subs"], [{"out": "M5", "in": "D4"}])


GW7_POINTS = {   # element: (points, minutes), from the FPL app's GW7 points view
    1: (9, 90), 115: (6, 90), 112: (4, 90), 4: (8, 90), 397: (1, 13), 124: (5, 70), 455: (4, 90), 236: (2, 90),
    399: (0, 0), 411: (13, 90), 106: (4, 90), 497: (0, 0), 306: (1, 90), 175: (1, 90), 272: (0, 0),
}


class TestCommittedGW7(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(mr.COMMITTED_DIR, "2026-27", "gw07.json"), encoding="utf-8") as f:
            self.picks = json.load(f)

    def test_valid_squad(self):
        players = self.picks["players"]
        starters = [p for p in players if p["role"] == "starter"]
        bench_ = [p for p in players if p["role"] == "bench"]
        self.assertEqual((len(starters), len(bench_)), (11, 4))
        self.assertEqual(mr._counts(players), {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3})
        self.assertEqual(bench_[0]["position"], "GK")
        self.assertEqual([p["name"] for p in starters if p["captain"]], ["Erling Haaland"])
        self.assertEqual(len([p for p in starters if p["vice_captain"]]), 1)
        self.assertEqual(self.picks["source"], "manual")
        self.assertLessEqual(sum(p["cost"] for p in players), 100.0)

    def test_scores_like_the_fpl_app(self):
        act = {el: {"points": pts, "minutes": mins} for el, (pts, mins) in GW7_POINTS.items()}
        r = mr.score_picks(self.picks, act, {"finished": True, "average_entry_score": 59, "highest_score": 122}, [])
        self.assertEqual(r["points"], 70)
        self.assertEqual(r["auto_subs"], [{"out": "Rayan Cherki", "in": "Jacob Greaves"}])
        self.assertEqual((r["picked_formation"], r["formation"]), ("3-5-2", "4-4-2"))
        self.assertEqual([p["name"] for p in r["bench"]][:2], ["Martin Dubravka", "Rayan Cherki"])


def write_predictions(path, gameweek=8, play60=False):
    clubs = ["Arsenal", "Chelsea", "Everton", "Fulham", "Brentford", "Leeds", "Spurs", "Wolves"]
    rows = []
    for k, pos in enumerate((["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3) * 2):
        rows.append({"name": f"{pos}-{k}", "position": pos, "team": clubs[k % 8],
                     "predicted_points": round(1 + (k * 37 % 23) / 3, 2), "value": 40 + (k * 13) % 30,
                     "opponent": "Somebody", "is_home": True, "gameweek": gameweek,
                     "kickoff_time": "2026-10-24T14:00:00Z", "fixtures": 1, "status": "a",
                     "chance_of_playing_next_round": None, "news": None, "element": k + 1})
        if play60:
            rows[-1]["play60"] = round(0.5 + (k * 29 % 50) / 100, 2)
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


class TestSaveAndReview(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.committed = tempfile.TemporaryDirectory()
        self.patches = [patch.dict(os.environ, {"LIVE_DATA_DIR": self.tmp.name}),
                        patch.object(mr, "COMMITTED_DIR", self.committed.name)]
        for p in self.patches:
            p.start()
        ss.clear_cache()
        self.csv = os.path.join(self.tmp.name, "next_gw_predictions.csv")
        self.rows = write_predictions(self.csv)
        clubs = sorted({r["team"] for r in self.rows})
        self.fixtures = {t: [{"fixture_id": 500 + i, "opponent": "X", "is_home": True}] for i, t in enumerate(clubs)}

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        ss.clear_cache()
        self.tmp.cleanup()
        self.committed.cleanup()

    def save(self, now, deadline="2026-10-23T17:30:00Z", gameweek=8):
        return mr.save_picks(self.tmp.name, "2026-27", gameweek, deadline, self.csv, self.fixtures, now=now)

    def write_meta(self, events, finished=()):
        with open(os.path.join(self.tmp.name, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"season": "2026-27", "events": events, "finished_fixtures": list(finished),
                       "files": {"season": "FPL_2026_2027.csv"}}, f)

    def test_saves_the_page_team_before_the_deadline(self):
        path = self.save(datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc))
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(path.replace(".json", "_predictions.csv")))
        picks = json.load(open(path, encoding="utf-8"))
        self.assertEqual((picks["gameweek"], picks["source"], len(picks["players"])), (8, "pipeline", 15))
        page = ss.get_team_of_the_week(None, "predicted", None, season="2026-27")
        self.assertEqual([p["name"] for p in picks["players"]],
                         [p["name"] for p in page["starters"] + page["bench"]])
        self.assertTrue(all(p["fixtures"] for p in picks["players"]))
        # No play60 in these predictions: no safer pick
        self.assertFalse(os.path.exists(path.replace(".json", "_safer.json")))

    def test_saves_the_safer_pick_too(self):
        self.rows = write_predictions(self.csv, play60=True)
        path = self.save(datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc))
        safer = json.load(open(path.replace(".json", "_safer.json"), encoding="utf-8"))
        self.assertEqual((safer["gameweek"], safer["pick"], len(safer["players"])), (8, "safer", 15))
        self.assertEqual(json.load(open(path, encoding="utf-8"))["pick"], "default")
        page = ss.get_team_of_the_week(None, "predicted", None, season="2026-27", pick="safer")
        self.assertEqual([p["name"] for p in safer["players"]],
                         [p["name"] for p in page["starters"] + page["bench"]])
        self.assertTrue(all(0 <= p["play60"] <= 1 for p in safer["players"]))

    def test_review_of_the_safer_pick(self):
        self.rows = write_predictions(self.csv, play60=True)
        self.save(datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc))
        rows = [{"element": r["element"], "GW": 8, "minutes": 90, "total_points": 2} for r in self.rows]
        pd.DataFrame(rows).to_csv(os.path.join(self.tmp.name, "FPL_2026_2027.csv"), index=False)
        self.write_meta([{"id": 8, "finished": True, "average_entry_score": 55, "highest_score": 140}])
        with patch.object(mr, "datetime") as dt:
            dt.now.return_value = datetime(2026, 10, 27, tzinfo=timezone.utc)
            dt.fromisoformat = datetime.fromisoformat
            self.assertEqual(mr.reviewable_gameweeks(pick="safer"), [8])
            safer = mr.get_model_review(pick="safer")
            default = mr.get_model_review()
        self.assertEqual((safer["pick"], safer["gameweek"], safer["gameweeks"]), ("safer", 8, [8]))
        self.assertEqual(default["pick"], "default")
        self.assertEqual(len(safer["starters"]) + len(safer["bench"]), 15)

    def test_no_safer_review_without_a_saved_safer_pick(self):
        self.save(datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc))       # no play60: default only
        self.write_meta([])
        with patch.object(mr, "datetime") as dt:
            dt.now.return_value = datetime(2026, 10, 27, tzinfo=timezone.utc)
            dt.fromisoformat = datetime.fromisoformat
            self.assertEqual(mr.reviewable_gameweeks(), [8])
            self.assertEqual(mr.reviewable_gameweeks(pick="safer"), [])
            with self.assertRaises(ss.NotFoundError):
                mr.get_model_review(pick="safer")

    def test_keeps_the_pick_after_the_deadline(self):
        self.assertIsNone(self.save(datetime(2026, 10, 23, 17, 30, tzinfo=timezone.utc)))
        self.assertEqual(os.listdir(self.tmp.name), ["next_gw_predictions.csv"])

    def test_review_payload(self):
        self.save(datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc))
        picks = json.load(open(os.path.join(self.tmp.name, "model_picks", "2026-27", "gw08.json"), encoding="utf-8"))
        rows = [{"element": p["element"], "GW": 8, "minutes": 0 if k == 5 else 90, "total_points": 2}
                for k, p in enumerate(picks["players"])]
        pd.DataFrame(rows).to_csv(os.path.join(self.tmp.name, "FPL_2026_2027.csv"), index=False)
        self.write_meta([{"id": 8, "finished": True, "average_entry_score": 55, "highest_score": 140}])
        with patch.object(mr, "datetime") as dt:
            dt.now.return_value = datetime(2026, 10, 27, tzinfo=timezone.utc)
            dt.fromisoformat = datetime.fromisoformat
            r = mr.get_model_review(photo_for=lambda name, team, code: f"photo:{name}")
        self.assertEqual((r["gameweek"], r["gameweeks"], r["fpl_average"]), (8, [8], 55))
        self.assertEqual(len(r["auto_subs"]), 1)
        self.assertTrue(all(p["photo"].startswith("photo:") for p in r["starters"] + r["bench"]))
        self.assertEqual(r["budget"], 100.0)
        self.assertIn("web_name", r["starters"][0])
        self.assertNotIn("element", r["starters"][0])

    def test_no_review_before_the_deadline(self):
        self.save(datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc))
        self.write_meta([])
        with patch.object(mr, "datetime") as dt:
            dt.now.return_value = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
            dt.fromisoformat = datetime.fromisoformat
            self.assertEqual(mr.reviewable_gameweeks(), [])
            with self.assertRaises(ss.NotFoundError):
                mr.get_model_review()

    def test_saved_pick_wins_over_committed_one(self):
        os.makedirs(os.path.join(self.committed.name, "2026-27"))
        committed = {"season": "2026-27", "gameweek": 8, "deadline": "2026-10-23T17:30:00Z", "source": "manual",
                     "players": []}
        with open(os.path.join(self.committed.name, "2026-27", "gw08.json"), "w", encoding="utf-8") as f:
            json.dump(committed, f)
        self.assertEqual(mr.load_picks(8, "2026-27")["source"], "manual")
        self.save(datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc))
        self.assertEqual(mr.load_picks(8, "2026-27")["source"], "pipeline")

    def test_unknown_gameweek(self):
        self.write_meta([])
        with self.assertRaises(ss.NotFoundError):
            mr.get_model_review(3)


class TestEventSummaries(unittest.TestCase):
    def test_zero_scores_are_missing(self):
        from pipeline.current_season import event_summaries
        out = event_summaries({"events": [
            {"id": 1, "deadline_time": "t", "finished": True, "data_checked": True, "average_entry_score": 50,
             "highest_score": 131, "chip_plays": []},
            {"id": 2, "deadline_time": "u", "finished": False, "data_checked": False, "average_entry_score": 0,
             "highest_score": None},
        ]})
        self.assertEqual(out[0], {"id": 1, "deadline_time": "t", "finished": True, "data_checked": True,
                                  "average_entry_score": 50, "highest_score": 131})
        self.assertEqual((out[1]["average_entry_score"], out[1]["highest_score"]), (None, None))


class TestRoute(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        import api_main
        self.api = api_main
        self.client = TestClient(api_main.app)

    def test_payload_and_alias(self):
        payload = mr.score_picks(picks_352(), actual(F1=(0, 0)), FINISHED, [])
        payload.update(season="2026-27", gameweeks=[5])
        for p in payload["starters"] + payload["bench"]:
            p.pop("element")
            p["photo"] = None
        with patch.object(self.api.model_review, "get_model_review", return_value=payload) as get:
            resp = self.client.get("/api/model-review?gameweek=5")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["auto_subs"], [{"out": "F1", "in": "D4"}])
        self.assertEqual(get.call_args.args[0], 5)

    def test_not_found(self):
        with patch.object(self.api.model_review, "get_model_review", side_effect=ss.NotFoundError("none")):
            self.assertEqual(self.client.get("/api/model-review").status_code, 404)
        self.assertEqual(self.client.get("/api/model-review?gameweek=0").status_code, 422)

    def test_safer_pick(self):
        payload = mr.score_picks(picks_352(), actual(), FINISHED, [])
        payload.update(season="2026-27", gameweeks=[5], pick="safer")
        for p in payload["starters"] + payload["bench"]:
            p.pop("element")
            p["photo"] = None
        with patch.object(self.api.model_review, "get_model_review", return_value=payload) as get:
            resp = self.client.get("/api/model-review?gameweek=5&pick=safer")
        self.assertEqual((resp.status_code, resp.json()["pick"]), (200, "safer"))
        self.assertEqual(get.call_args.kwargs["pick"], "safer")
        self.assertEqual(self.client.get("/api/model-review?pick=bogus").status_code, 422)

    def test_team_of_the_week_pick(self):
        team = {"mode": "predicted", "pick": "safer", "gameweek": 9, "formation": "3-4-3", "total_points": 60.0,
                "total_cost": 99.5, "budget": 100.0, "starters": [], "bench": []}
        with patch.object(self.api.season_service, "get_team_of_the_week", return_value=team) as get:
            resp = self.client.get("/api/team-of-the-week?pick=safer")
        self.assertEqual((resp.status_code, resp.json()["pick"]), (200, "safer"))
        self.assertEqual(get.call_args.kwargs["pick"], "safer")
        self.assertEqual(self.client.get("/api/team-of-the-week?mode=actual&pick=safer").status_code, 422)
        self.assertEqual(self.client.get("/api/team-of-the-week?pick=bogus").status_code, 422)
        with patch.object(self.api.season_service, "get_team_of_the_week",
                          side_effect=ss.UnavailableError("no play60 yet")):
            self.assertEqual(self.client.get("/api/team-of-the-week?pick=safer").status_code, 503)


if __name__ == "__main__":
    unittest.main()
