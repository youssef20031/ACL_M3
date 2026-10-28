"""Tests for pipeline/predict_next_gw.py and the live-file selection in services/predictions.py."""
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ml import odds as od
from ml import points_model as pm
from ml import snapshots as sn
from pipeline import predict_next_gw as png
from services import predictions as preds_service

TEAMS = {1: "Arsenal", 2: "Chelsea", 3: "Ipswich", 4: "Spurs", 5: "Hull City", 6: "Leeds"}


@pytest.fixture(autouse=True)
def hermetic_inputs(tmp_path, monkeypatch):
    """No real snapshot archive (vaastav's 2026-27 captures would join the test's elements),
    and odds only from the test's live dir."""
    monkeypatch.setattr(sn, "SNAPSHOTS_CSV", str(tmp_path / "no_snapshots.csv"))
    monkeypatch.delenv("ODDS_DIR", raising=False)


def fixture(fid, event, home, away, kickoff, finished=False):
    return {"id": fid, "event": event, "team_h": home, "team_a": away,
            "kickoff_time": kickoff, "finished": finished}


GW_FIXTURES = [
    fixture(1, 7, 1, 2, "2026-10-18T15:30:00Z"),
    fixture(2, 7, 3, 4, "2026-10-17T14:00:00Z"),
    fixture(3, 7, 1, 4, "2026-10-21T19:00:00Z"),   # Arsenal and Spurs double
    fixture(4, 8, 5, 6, "2026-10-25T14:00:00Z"),   # other gameweek
    fixture(5, None, 5, 2, None),                  # postponed / unscheduled
]


# ------------------------------------------------------------------ fixtures / gameweek

def test_next_gameweek_picks_is_next_event():
    bootstrap = {"events": [{"id": 6, "is_next": False}, {"id": 7, "is_next": True}]}
    assert png.next_gameweek(bootstrap)["id"] == 7
    assert png.next_gameweek({"events": [{"id": 38, "is_next": False}]}) is None


def test_gameweek_fixtures_single_double_and_blank():
    by_team = png.gameweek_fixtures(GW_FIXTURES, 7, TEAMS)

    assert [e["opponent"] for e in by_team["Arsenal"]] == ["Chelsea", "Spurs"]   # double, kickoff order
    assert [e["is_home"] for e in by_team["Arsenal"]] == [True, True]
    assert [e["opponent"] for e in by_team["Spurs"]] == ["Ipswich", "Arsenal"]
    assert [e["is_home"] for e in by_team["Spurs"]] == [False, False]
    assert by_team["Chelsea"] == [{"fixture_id": 1, "opponent": "Arsenal", "is_home": False,
                                   "kickoff_time": "2026-10-18T15:30:00Z"}]
    assert by_team["Ipswich"][0]["is_home"] is True and by_team["Ipswich"][0]["opponent"] == "Spurs"
    # blank: Hull City and Leeds only play in GW8 / an unscheduled fixture
    assert "Hull City" not in by_team and "Leeds" not in by_team


def test_describe_fixtures_joins_double_gameweek_opponents():
    by_team = png.gameweek_fixtures(GW_FIXTURES, 7, TEAMS)
    assert png.describe_fixtures(by_team["Arsenal"]) == ("Chelsea & Spurs", True, "2026-10-18T15:30:00Z")
    assert png.describe_fixtures(by_team["Chelsea"]) == ("Arsenal", False, "2026-10-18T15:30:00Z")


# ------------------------------------------------------------------ availability

@pytest.mark.parametrize("status, chance, expected", [
    ("a", np.nan, 1.0),
    ("a", None, 1.0),
    ("a", 100.0, 1.0),
    ("d", 75.0, 0.75),
    ("d", 25.0, 0.25),
    ("i", 0.0, 0.0),
    ("i", np.nan, 0.0),     # injured without a chance value: out
    ("s", np.nan, 0.0),
    ("u", np.nan, 0.0),     # left the club / unavailable
    ("u", 100.0, 0.0),
    ("n", np.nan, 0.0),
])
def test_availability_factor(status, chance, expected):
    assert png.availability_factor(status, chance) == pytest.approx(expected)


def test_promoted_teams():
    assert png.promoted_teams(["Arsenal", "Chelsea"], ["Arsenal", "Hull City", "Chelsea", "Hull City"]) == ["Hull City"]
    assert png.promoted_teams([], ["Arsenal"]) == []


# ------------------------------------------------------------------ output assembly

def players_frame():
    return pd.DataFrame([
        {"element": 1, "name": "Fit Gunner", "team": "Arsenal", "position": "MID", "status": "a",
         "chance_of_playing_next_round": np.nan, "news": "", "now_cost": 100},
        {"element": 2, "name": "Doubtful Blue", "team": "Chelsea", "position": "DEF", "status": "d",
         "chance_of_playing_next_round": 50.0, "news": "Knock - 50% chance of playing", "now_cost": 50},
        {"element": 3, "name": "Injured Gunner", "team": "Arsenal", "position": "DEF", "status": "i",
         "chance_of_playing_next_round": 0.0, "news": "Back injury", "now_cost": 60},
        {"element": 4, "name": "Blank Tiger", "team": "Hull City", "position": "FWD", "status": "a",
         "chance_of_playing_next_round": np.nan, "news": "", "now_cost": 55},
        {"element": 5, "name": "Gone Gunner", "team": "Arsenal", "position": "GK", "status": "u",
         "chance_of_playing_next_round": np.nan, "news": "Joined X on loan", "now_cost": 40},
    ]).assign(
        # the status file's FPL code, transfers since the last deadline and capture time
        code=lambda d: d["element"] + 9000,
        transfers_in_event=[50000, 100, 10, 2000, 0], transfers_out_event=[1000, 8000, 30000, 500, 9000],
        fetched_at="2026-10-16T12:00:00Z",
    )


def test_assemble_output_applies_availability_blank_and_schema():
    by_team = png.gameweek_fixtures(GW_FIXTURES, 7, TEAMS)
    predictions = {"1": {"points": 9.0, "method": "xgboost_v3"}, "2": {"points": 4.0, "method": "xgboost_v3"},
                   "3": {"points": 5.0, "method": "xgboost_v3"}, "5": {"points": 3.0, "method": "xgboost_v3"}}
    out = png.assemble_output(players_frame(), predictions, by_team, 7, "2026-10-17T10:00:00Z", ["Hull City"])
    rows = out.set_index("name")

    assert list(out.columns) == png.OUTPUT_COLUMNS
    assert out["predicted_points"].is_monotonic_decreasing
    assert rows.loc["Fit Gunner", "predicted_points"] == 9.0
    assert rows.loc["Fit Gunner", "opponent"] == "Chelsea & Spurs"
    assert rows.loc["Fit Gunner", "fixtures"] == 2
    assert rows.loc["Doubtful Blue", "predicted_points"] == 2.0          # 4 x 50%
    assert rows.loc["Injured Gunner", "predicted_points"] == 0.0
    assert rows.loc["Injured Gunner", "news"] == "Back injury"
    assert rows.loc["Gone Gunner", "predicted_points"] == 0.0
    blank = rows.loc["Blank Tiger"]
    assert blank["predicted_points"] == 0.0 and blank["fixtures"] == 0
    assert blank["opponent"] == png.NO_FIXTURE_OPPONENT
    assert blank["kickoff_time"] == "2026-10-17T10:00:00Z"              # never NaN for API consumers
    assert blank["prediction_method"] == png.NO_FIXTURE_METHOD
    assert bool(blank["is_promoted_team"]) is True
    assert set(out["gameweek"]) == {7}
    assert out[["opponent", "kickoff_time", "value", "gameweek"]].notna().all().all()


# ------------------------------------------------------------------ history + model rows

LIVE_COLUMNS = ["name", "position", "team", "element", "kickoff_time", "GW", "was_home", "total_points",
                "value", "team_h_score", "team_a_score", "fixture", "minutes", "creativity", "ict_index",
                "influence", "threat", "clearances_blocks_interceptions", "defensive_contribution",
                "recoveries", "tackles"]


def live_rows(element, name, team, gws, minutes=90, points=6):
    return [{
        "name": name, "position": "MID", "team": team, "element": element,
        "kickoff_time": f"2026-09-{gw * 3:02d}T15:00:00Z", "GW": gw, "was_home": gw % 2 == 0,
        "total_points": points + gw, "value": 80, "team_h_score": 2, "team_a_score": 1, "fixture": gw * 10 + element,
        "minutes": minutes, "creativity": 20.0, "ict_index": 8.0, "influence": 30.0, "threat": 25.0,
        "clearances_blocks_interceptions": 1, "defensive_contribution": 2, "recoveries": 3, "tackles": 1,
    } for gw in gws]


@pytest.fixture
def live_dir(tmp_path):
    rows = (live_rows(1, "Fit Gunner", "Arsenal", [1, 2, 3, 4, 5, 6, 7])
            + live_rows(2, "Doubtful Blue", "Chelsea", [1, 2, 3])
            + live_rows(3, "Injured Gunner", "Arsenal", [1, 2]))
    pd.DataFrame(rows, columns=LIVE_COLUMNS).to_csv(tmp_path / "FPL_2026_2027.csv", index=False)
    players_frame().to_csv(tmp_path / "player_status_2026_27.csv", index=False)
    hist = pd.DataFrame([{
        "season_x": "2025-26", "name": "Fit Gunner", "position": "MID", "team_x": "Arsenal",
        "kickoff_time": "2026-05-10T15:00:00Z", "GW": 37, "was_home": True, "total_points": 2, "value": 75,
        "team_h_score": 1, "team_a_score": 1, "fixture": 370, "minutes": 90, "creativity": 1.0,
        "ict_index": 1.0, "influence": 1.0, "threat": 1.0, "expected_goals": 1.2,
    }, {
        "season_x": "2025-26", "name": "Someone Else", "position": "FWD", "team_x": "Leeds",
        "kickoff_time": "2026-05-10T15:00:00Z", "GW": 37, "was_home": False, "total_points": 2, "value": 60,
        "team_h_score": 1, "team_a_score": 1, "fixture": 371, "minutes": 90, "creativity": 1.0,
        "ict_index": 1.0, "influence": 1.0, "threat": 1.0, "expected_goals": 0.8,
    }])
    hist.to_csv(tmp_path / "history.csv", index=False)
    return tmp_path


def test_load_history_excludes_predicted_gameweek_and_later(live_dir):
    history = png.load_history(str(live_dir / "FPL_2026_2027.csv"), "2026-27", before_gw=6,
                               before_time="2026-10-17T10:00:00Z", history_csv=str(live_dir / "history.csv"))
    current = history[history["season"] == "2026-27"]
    assert current["GW"].max() == 5                     # GW6/GW7 rows never feed features
    assert len(history[history["season"] == "2025-26"]) == 2
    assert png.previous_season_teams(history, "2026-27") == ["Arsenal", "Leeds"]

    # kickoff cutoff also applies (a later-round match played early would be dropped)
    history = png.load_history(str(live_dir / "FPL_2026_2027.csv"), "2026-27", before_gw=8,
                               before_time="2026-09-10T00:00:00Z", history_csv=str(live_dir / "history.csv"))
    assert history.loc[history["season"] == "2026-27", "GW"].max() == 3


class RecordingModel:
    """Stands in for the v3 booster: constant points, keeps the feature rows it saw."""

    def __init__(self, value):
        self.value = value
        self.seen = []

    def __call__(self, frame):
        self.seen.append(frame.copy())
        return np.full(len(frame), self.value)


def test_predict_players_sums_double_gameweek_and_uses_prior_matches_only(live_dir):
    players = players_frame()
    history = png.load_history(str(live_dir / "FPL_2026_2027.csv"), "2026-27", before_gw=7,
                               before_time=None, history_csv=str(live_dir / "history.csv"))
    by_team = png.gameweek_fixtures(GW_FIXTURES, 7, TEAMS)
    model = RecordingModel(2.5)
    out = png.predict_players(players, history, by_team, "2026-27", 7, model)

    assert out["1"]["points"] == pytest.approx(5.0)      # Arsenal double: 2 x 2.5
    assert out["2"]["points"] == pytest.approx(2.5)
    assert "4" not in out                                 # blank (Hull City): no model row
    assert out["5"]["points"] == pytest.approx(5.0)      # no match rows at all: still predicted
    assert {v["method"] for v in out.values()} == {png.MODEL_METHOD}

    rows = pd.concat(model.seen)
    fit = rows[rows["name"] == "Fit Gunner"].sort_values("kickoff_time")
    assert list(fit["was_home"]) == [1.0, 1.0]
    assert set(fit["fixtures_in_gw"]) == {2.0}
    assert set(fit["GW"]) == {7}
    # Form only from matches before GW7 (GW4-6 points are 10, 11, 12), identical for both
    # fixtures of the double gameweek
    assert list(fit["total_points_r3"]) == pytest.approx([11.0, 11.0])
    assert list(fit["opponent"]) == ["Chelsea", "Spurs"]


class FakeClient:
    def bootstrap(self):
        return {
            "events": [{"id": 6, "is_next": False, "deadline_time": "2026-08-21T17:30:00Z"},
                       {"id": 7, "is_next": True, "deadline_time": "2026-10-17T10:00:00Z"}],
            "teams": [{"id": i, "name": "Ipswich Town" if n == "Ipswich" else n} for i, n in TEAMS.items()],
        }

    def fixtures(self):
        return GW_FIXTURES

    def close(self):
        pass


def test_predict_next_gameweek_end_to_end(live_dir):
    # Real served model file (v4, or v3 until one exists), end to end; no odds download
    summary = png.predict_next_gameweek(str(live_dir), client=FakeClient(),
                                        history_csv=str(live_dir / "history.csv"), fetch_odds=False)
    out = pd.read_csv(live_dir / "next_gw_predictions.csv")
    records = json.loads((live_dir / "next_gw_predictions.json").read_text(encoding="utf-8"))

    assert summary["gameweek"] == 7 and summary["players"] == 5 == len(out) == len(records)
    assert summary["blank_teams"] == ["Hull City"]
    assert summary["double_gameweek_players"] == 3        # the three Arsenal players
    assert summary["history_last_kickoff"] < "2026-10-17"
    assert not any(p.endswith(".tmp") for p in os.listdir(live_dir))
    assert summary["model_pick"] is None                  # GW7's deadline has passed: its pick is final
    for col in ["name", "team", "position", "opponent", "is_home", "kickoff_time", "gameweek",
                "predicted_points", "value", "prediction_method"]:
        assert col in out.columns
    rows = out.set_index("name")
    assert rows.loc["Injured Gunner", "predicted_points"] == 0
    assert rows.loc["Gone Gunner", "predicted_points"] == 0
    assert rows.loc["Fit Gunner", "predicted_points"] > 0
    assert rows.loc["Fit Gunner", "opponent"] == "Chelsea & Spurs"

    # The model's inputs, for the /model page's explanations (services/explain.py)
    assert summary["model_inputs"] == str(live_dir / png.FEATURES_NAME)
    inputs = pd.read_csv(live_dir / png.FEATURES_NAME)
    assert set(png.FEATURE_ID_COLUMNS + ["model_points"] + pm.feature_columns()) <= set(inputs.columns)
    fit = inputs[inputs["name"] == "Fit Gunner"]
    assert list(fit["opponent"]) == ["Chelsea", "Spurs"]
    assert fit["model_points"].clip(lower=0).sum() == pytest.approx(rows.loc["Fit Gunner", "predicted_points"], abs=0.01)


# ------------------------------------------------------------------ v4 inputs, odds, shadow

# football-data.co.uk's fixtures.csv for GW7 (Tottenham is Spurs in the graph)
ODDS_FIXTURES_CSV = (
    "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5\n"
    "E0,17/10/2026,15:00,Ipswich,Tottenham,3.40,3.50,2.10,1.90,1.90\n"
    "E0,18/10/2026,16:30,Arsenal,Chelsea,1.80,3.80,4.50,1.70,2.10\n"
    "E0,21/10/2026,20:00,Arsenal,Tottenham,1.60,4.20,5.00,1.60,2.30\n"
)


def write_odds(live_dir, text=ODDS_FIXTURES_CSV):
    d = live_dir / "odds"
    d.mkdir(exist_ok=True)
    (d / od.FIXTURES_FILE).write_text(text, encoding="utf-8")
    return d


def test_model_rows_carry_every_v4_input(live_dir):
    write_odds(live_dir)
    status_csv = str(live_dir / "player_status_2026_27.csv")
    players = png._load_players(status_csv)
    history = png.load_history(str(live_dir / "FPL_2026_2027.csv"), "2026-27", before_gw=7,
                               before_time="2026-10-17T14:00:00Z", history_csv=str(live_dir / "history.csv"))
    extras = png.load_extras(str(live_dir), "2026-27", status_csv, str(live_dir / "history.csv"))
    rows = png.model_rows(players, history, png.gameweek_fixtures(GW_FIXTURES, 7, TEAMS), "2026-27", 7, extras)

    assert set(pm.feature_columns("v4")) <= set(rows.columns)
    assert len(rows) == 7 and set(rows["element"]) == {1, 2, 3, 5}      # Hull City blank
    r = rows.set_index(["name", "opponent"])
    # Odds count only if priced before the deadline (here Sunday's first kickoff - 90 min):
    # Sunday's match on Friday, the Wednesday match only on Tuesday
    assert 0.45 < r.loc[("Fit Gunner", "Chelsea"), "odds_win"] < 0.6
    assert r.loc[("Doubtful Blue", "Arsenal"), "odds_win"] < 0.3
    assert np.isnan(r.loc[("Fit Gunner", "Spurs"), "odds_win"])
    # The status file is the upcoming gameweek's availability snapshot
    assert r.loc[("Doubtful Blue", "Arsenal"), "snap_doubtful"] == 1
    assert r.loc[("Doubtful Blue", "Arsenal"), "snap_chance"] == 50
    assert r.loc[("Injured Gunner", "Chelsea"), "snap_unavailable"] == 1
    assert r.loc[("Fit Gunner", "Chelsea"), "snap_age_days"] == pytest.approx(2 + 2 / 24, abs=0.01)
    # Transfers since the last deadline, ranked within the gameweek; team ratings for everyone
    assert r.loc[("Fit Gunner", "Chelsea"), "tr_in_rank"] == 1.0
    assert rows[["tr_bal_rank", "rt_att", "rt_opp_def"]].notna().all().all()
    assert png.input_coverage(rows) == {"odds": round(4 / 7, 3), "snapshot": 1.0, "transfers": 1.0, "ratings": 1.0}


def test_odds_download_failure_degrades_to_nan_odds(live_dir, monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("network is unreachable")
    monkeypatch.setattr(od, "fetch_season", offline)
    monkeypatch.setattr(od, "fetch_fixtures", offline)
    # An unreadable leftover (an error page saved by hand) is skipped too, with a warning
    odds = write_odds(live_dir, "<html>Service unavailable</html>\n")
    old = (datetime.now() - timedelta(days=1)).timestamp()
    os.utime(odds / od.FIXTURES_FILE, (old, old))

    model = RecordingModel(2.0)
    summary = png.predict_next_gameweek(str(live_dir), client=FakeClient(),
                                        history_csv=str(live_dir / "history.csv"), predict_fn=model)
    assert summary["odds"]["fixtures.csv"].startswith("failed")
    assert summary["odds"]["E0_2627.csv"].startswith("failed")
    assert summary["input_coverage"]["odds"] == 0.0 and summary["input_coverage"]["snapshot"] == 1.0
    assert summary["shadow"] == {"model": None, "pick": "skipped: no shadow model given"}
    assert len(pd.read_csv(live_dir / png.OUTPUT_NAME)) == 5
    seen = pd.concat(model.seen)
    assert set(pm.feature_columns("v4")) <= set(seen.columns) and seen["odds_win"].isna().all()


SQUAD_CLUBS = ["Arsenal", "Chelsea", "Ipswich", "Spurs", "Hull City", "Leeds"]


def squad_frame():
    """Three players per club (GK or FWD, DEF, MID), enough for a 15-man squad, plus an injured keeper."""
    rows = []
    for k, team in enumerate(SQUAD_CLUBS):
        for j, pos in enumerate(["GK" if k < 3 else "FWD", "DEF", "MID"]):
            e = 100 + 3 * k + j
            rows.append({"element": e, "name": f"{team} {pos}", "team": team, "position": pos, "status": "a",
                         "chance_of_playing_next_round": np.nan, "news": "", "now_cost": 45 + (7 * e) % 20})
    rows.append({"element": 200, "name": "Crocked Keeper", "team": "Leeds", "position": "GK", "status": "i",
                 "chance_of_playing_next_round": 0.0, "news": "Knee injury", "now_cost": 40})
    fetched = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame(rows).assign(code=lambda d: d["element"] + 9000, transfers_in_event=100,
                                     transfers_out_event=50, fetched_at=fetched)


class UpcomingClient:
    """GW9, with every club playing, two days from now (its deadline is ahead)."""

    def __init__(self):
        now = datetime.now(timezone.utc).replace(microsecond=0)
        iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.deadline = iso(now + timedelta(days=2))
        kickoff = iso(now + timedelta(days=2, hours=2))
        self._fixtures = [fixture(90 + i, 9, h, a, kickoff) for i, (h, a) in enumerate([(1, 2), (3, 4), (5, 6)])]

    def bootstrap(self):
        return {
            "events": [{"id": 8, "is_next": False, "deadline_time": "2026-10-24T10:00:00Z"},
                       {"id": 9, "is_next": True, "deadline_time": self.deadline}],
            "teams": [{"id": i, "name": "Ipswich Town" if n == "Ipswich" else n} for i, n in TEAMS.items()],
        }

    def fixtures(self):
        return self._fixtures

    def close(self):
        pass


def test_shadow_pick_is_saved_next_to_the_live_pick(live_dir):
    squad_frame().to_csv(live_dir / "player_status_2026_27.csv", index=False)
    summary = png.predict_next_gameweek(
        str(live_dir), client=UpcomingClient(), history_csv=str(live_dir / "history.csv"), fetch_odds=False,
        predict_fn=lambda f: f["value"].to_numpy() / 20.0,          # the live model likes pricey players
        shadow_fn=lambda f: 6.0 - f["value"].to_numpy() / 20.0)     # the shadow likes cheap ones
    picks_dir = live_dir / "model_picks" / "2026-27"
    assert summary["model_pick"] == str(picks_dir / "gw09.json")
    assert summary["shadow"] == {"model": pm.SHADOW_VERSION, "pick": str(picks_dir / "shadow" / "gw09.json")}

    live_pick = json.loads((picks_dir / "gw09.json").read_text(encoding="utf-8"))
    shadow_pick = json.loads((picks_dir / "shadow" / "gw09.json").read_text(encoding="utf-8"))
    assert set(live_pick) == set(shadow_pick)
    assert [set(p) for p in live_pick["players"]] == [set(p) for p in shadow_pick["players"]]
    assert (live_pick["model"], shadow_pick["model"]) == (pm.MODEL_VERSION, pm.SHADOW_VERSION)
    assert len(shadow_pick["players"]) == 15
    assert {p["name"] for p in live_pick["players"]} != {p["name"] for p in shadow_pick["players"]}
    # Same availability multiplier for the shadow; the site keeps reading the live predictions
    shadow_preds = pd.read_csv(picks_dir / "shadow" / "gw09_predictions.csv").set_index("name")
    assert shadow_preds.loc["Crocked Keeper", "predicted_points"] == 0
    assert set(shadow_preds["prediction_method"]) == {f"xgboost_{pm.SHADOW_VERSION}"}
    assert set(pd.read_csv(live_dir / png.OUTPUT_NAME)["prediction_method"]) == {png.MODEL_METHOD}


def test_failing_shadow_never_touches_the_live_predictions(live_dir):
    squad_frame().to_csv(live_dir / "player_status_2026_27.csv", index=False)

    def broken(frame):
        raise RuntimeError("shadow model exploded")
    summary = png.predict_next_gameweek(str(live_dir), client=UpcomingClient(),
                                        history_csv=str(live_dir / "history.csv"), fetch_odds=False,
                                        predict_fn=lambda f: f["value"].to_numpy() / 20.0, shadow_fn=broken)
    assert summary["shadow"]["pick"] == "failed"
    assert os.path.exists(summary["model_pick"])
    assert not (live_dir / "model_picks" / "2026-27" / "shadow").exists()
    assert len(pd.read_csv(live_dir / png.OUTPUT_NAME)) == 19


# ------------------------------------------------------------------ services/predictions.py

@pytest.fixture
def prediction_files(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled.csv"
    pd.DataFrame([{"name": "Old Player", "predicted_points": 1.0}]).to_csv(bundled, index=False)
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    monkeypatch.setattr(preds_service, "PREDICTIONS_CSV", str(bundled))
    monkeypatch.setenv("LIVE_DATA_DIR", str(live_dir))
    monkeypatch.setattr(preds_service, "_cached", None)
    return bundled, live_dir / "next_gw_predictions.csv"


def test_predictions_fall_back_to_bundled_file(prediction_files):
    bundled, live = prediction_files
    assert preds_service.predictions_path() == str(bundled)
    df = preds_service.get_predictions_df()
    assert list(df["name_lower"]) == ["old player"]


def test_predictions_prefer_live_file_once_written(prediction_files):
    bundled, live = prediction_files
    assert list(preds_service.get_predictions_df()["name"]) == ["Old Player"]

    pd.DataFrame([{"name": "New Player", "predicted_points": 7.0}]).to_csv(live, index=False)
    assert preds_service.predictions_path() == str(live)
    assert list(preds_service.get_predictions_df()["name_lower"]) == ["new player"]   # no restart needed

    live.unlink()
    assert list(preds_service.get_predictions_df()["name"]) == ["Old Player"]


def test_predictions_cache_reuses_frame_until_mtime_changes(prediction_files):
    bundled, live = prediction_files
    pd.DataFrame([{"name": "A", "predicted_points": 1.0}]).to_csv(live, index=False)
    first = preds_service.get_predictions_df()
    assert preds_service.get_predictions_df() is first

    pd.DataFrame([{"name": "B", "predicted_points": 2.0}]).to_csv(live, index=False)
    os.utime(live, (os.path.getmtime(live) + 10, os.path.getmtime(live) + 10))
    assert list(preds_service.get_predictions_df()["name"]) == ["B"]


def test_predictions_missing_everywhere_returns_none(prediction_files):
    bundled, live = prediction_files
    bundled.unlink()
    assert preds_service.get_predictions_df() is None
