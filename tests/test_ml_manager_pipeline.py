"""pipeline.ml_manager: the ML manager's gameweek cycle end to end on synthetic files
(no network: the FPL context and the league lookup are stubbed; the real planner runs on a
small pool)."""
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from pipeline import ml_manager
from services import fpl_manager as fm

SEASON = "2026-27"
TEAMS = ["ARS", "AVL", "BOU", "BRE", "BHA", "CHE", "CRY", "EVE", "FUL", "LIV"]
POSITIONS = ["GK"] * 4 + ["DEF"] * 8 + ["MID"] * 8 + ["FWD"] * 5     # elements 1..25
STARTERS = [1, 5, 6, 7, 8, 13, 14, 15, 16, 21, 22]                    # 4-4-2
BENCH = [2, 9, 17, 23]
CAPTAIN, VICE = 21, 13
COSTS = {1: 4.5, 2: 4.0, 5: 7.9, 6: 5.5, 7: 4.6, 8: 5.1, 9: 4.3, 13: 10.1, 14: 8.3, 15: 6.6, 16: 5.9,
         17: 5.4, 21: 14.0, 22: 7.7, 23: 5.6}                         # 99.5m
GW9_DEADLINE = datetime(2026, 10, 31, 11, 0, tzinfo=timezone.utc)
LEAGUE = {"id": 284, "name": "Gameweek 9", "rank": 10, "size": 100, "percentile": 90.0}


def deadline(gw):
    return (GW9_DEADLINE + timedelta(days=7 * (gw - 9))).isoformat().replace("+00:00", "Z")


def before(gw):
    return GW9_DEADLINE + timedelta(days=7 * (gw - 9), hours=-6)


def after(gw):
    return GW9_DEADLINE + timedelta(days=7 * (gw - 9), hours=1)


def price(k):
    return int(round(COSTS.get(k, 5.0) * 10))


def team(k):
    return TEAMS[(k - 1) % len(TEAMS)]


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_DATA_DIR", str(tmp_path))
    pd.DataFrame([{"element": k, "code": 1000 + k, "name": f"Player {k}", "web_name": f"P{k}", "team": team(k),
                   "position": pos, "status": "a", "now_cost": price(k)}
                  for k, pos in enumerate(POSITIONS, start=1)]).to_csv(tmp_path / "player_status_2026_27.csv",
                                                                       index=False)
    picks = tmp_path / "model_picks" / SEASON
    picks.mkdir(parents=True)
    order = STARTERS + [9, 2, 17, 23]
    pick = {"season": SEASON, "gameweek": 9, "deadline": deadline(9), "archived_at": before(9).isoformat(),
            "source": "pipeline", "model": "v4",
            "players": [{"element": k, "name": f"Player {k}", "position": POSITIONS[k - 1], "team": team(k),
                         "cost": COSTS[k], "predicted_points": 3.0, "role": "starter" if k in STARTERS else "bench",
                         "captain": k == CAPTAIN, "vice_captain": k == VICE, "fixtures": [100 + k],
                         "opponents": [{"team": "X", "home": True}]} for k in order]}
    (picks / "gw09.json").write_text(json.dumps(pick), encoding="utf-8")
    rows = []
    for gw in range(10, 15):
        for k, pos in enumerate(POSITIONS, start=1):
            pts = 6.0 if k == 25 else (2.0 if k == 23 else 3.0)   # FWD 25 (unowned) is worth buying for FWD 23
            rows.append({"element": k, "name": f"Player {k}", "web_name": f"P{k}", "position": pos, "team": team(k),
                         "now_cost": price(k), "gameweek": gw, "fixtures": 1, "opponents": "X",
                         "predicted_points": pts, "availability": 1.0, "status": "a"})
    pd.DataFrame(rows).to_csv(tmp_path / "projections.csv", index=False)
    return tmp_path


def write_meta(live, next_gw, finished=()):
    events = [{"id": gw, "deadline_time": deadline(gw), "finished": gw in finished, "data_checked": gw in finished,
               "average_entry_score": 45 if gw in finished else None,
               "highest_score": 120 if gw in finished else None} for gw in range(9, 16)]
    meta = {"season": SEASON, "files": {"season": "FPL_2026_2027.csv", "status": "player_status_2026_27.csv"},
            "events": events, "next_gameweek": next_gw, "next_deadline": deadline(next_gw),
            "finished_fixtures": []}
    (live / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def write_results(live, gws):
    pd.DataFrame([{"element": k, "name": f"Player {k}", "GW": gw, "fixture": 100 + k, "minutes": 90,
                   "total_points": 2 + (k == CAPTAIN) * 8}
                  for gw in gws for k in range(1, 26)]).to_csv(live / "FPL_2026_2027.csv", index=False)


def run(live, now):
    return ml_manager.run(str(live), now=now, fpl_context=lambda client: (fm.DEFAULT_RULES, lambda gw: {}),
                          league_rank=lambda *a, **k: dict(LEAGUE))


def entry(live, gw):
    return fm.load_entry(str(live), SEASON, gw)


def test_a_season_cycle(live):
    # Before GW9's deadline the start entry follows the model's pick and nothing else happens
    write_meta(live, 9)
    s = run(live, before(9))
    assert s["start"] is True and entry(live, 9)["status"] == fm.PLANNED
    assert s["plan"]["status"].startswith("skipped")

    # After it: GW9 is locked, GW10 is planned (one free transfer: FWD 23 -> FWD 25)
    write_meta(live, 10)
    s = run(live, after(9))
    assert entry(live, 9)["status"] == fm.LOCKED
    assert s["plan"]["status"] == "planned", s
    gw10 = entry(live, 10)
    assert gw10["status"] == fm.PLANNED and gw10["free_transfers"] == 1
    assert [(t["out"]["element"], t["in"]["element"]) for t in gw10["transfers"]] == [(23, 25)]
    assert gw10["hit_cost"] == 0

    # After GW10's deadline and GW9's results: GW10 locked, GW9 scored and ranked, GW11 planned
    write_meta(live, 11, finished={9})
    write_results(live, [9])
    s = run(live, after(10))
    assert s["score"]["scored"] == [9]
    gw9 = entry(live, 9)
    assert gw9["status"] == fm.SCORED and gw9["result"]["league"]["name"] == "Gameweek 9"
    assert gw9["result"]["points"] == 10 * 2 + 10 * 2       # 10 starters x 2, the captain's 10 doubled
    assert entry(live, 10)["status"] == fm.LOCKED and entry(live, 11)["status"] == fm.PLANNED

    # GW12's deadline passes with no run in between: GW11 is locked, GW12 rolls over
    write_meta(live, 13, finished={9})
    s = run(live, after(12))
    assert s["lock"]["locked"] == [11] and s["lock"]["rolled_over"] == [12]
    gw12 = entry(live, 12)
    assert gw12["status"] == fm.LOCKED and gw12["transfers"] == [] and gw12["chip"] is None
    assert entry(live, 13)["status"] == fm.PLANNED

    state = json.loads((live / "ml_manager" / SEASON / "state.json").read_text(encoding="utf-8"))
    assert state["problems"] == [] and state["totals"]["points"] == 40
    view = fm.get_manager_view(SEASON, now=after(12))
    json.dumps(view, allow_nan=False)                         # the API can serve it as JSON


def test_a_locked_plan_never_changes(live):
    write_meta(live, 10)
    run(live, after(9))
    planned = entry(live, 10)
    write_meta(live, 11)
    run(live, after(10))
    locked = entry(live, 10)
    assert locked["status"] == fm.LOCKED
    assert {k: v for k, v in locked.items() if k != "status"} == {k: v for k, v in planned.items() if k != "status"}
    # A later run after the deadline doesn't touch it either
    run(live, after(10) + timedelta(hours=6))
    assert entry(live, 10) == locked


def test_a_failing_planner_never_stops_the_cycle(live):
    write_meta(live, 10)

    def broken(*args, **kwargs):
        raise RuntimeError("solver crashed")

    s = ml_manager.run(str(live), now=after(9), fpl_context=lambda client: (fm.DEFAULT_RULES, None),
                       planner=broken, league_rank=lambda *a, **k: None)
    assert entry(live, 9)["status"] == fm.LOCKED
    assert s["plan"].startswith("failed") and "state" in s


def test_another_season_is_left_alone(live):
    write_meta(live, 10)
    meta = json.loads((live / "meta.json").read_text(encoding="utf-8"))
    meta["season"] = "2027-28"
    (live / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert run(live, after(9))["status"].startswith("skipped")
    assert not os.path.exists(live / "ml_manager")
