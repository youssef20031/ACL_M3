"""Tests for pipeline/predict_next_gw.py and the live-file selection in services/predictions.py."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from pipeline import predict_next_gw as png
from services import predictions as preds_service

TEAMS = {1: "Arsenal", 2: "Chelsea", 3: "Ipswich", 4: "Spurs", 5: "Hull City", 6: "Leeds"}


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


# ------------------------------------------------------------------ team strength

def test_expected_goals_mixes_team_and_opponent_form():
    form = {"Arsenal": (2.0, 0.6), "Ipswich": (0.8, 2.2)}
    gf, ga = png.expected_goals("Arsenal", "Ipswich", form)
    assert gf == pytest.approx((2.0 + 2.2) / 2)
    assert ga == pytest.approx((0.6 + 0.8) / 2)
    assert png.expected_goals("X", "Y", {}) == (png.DEFAULT_TEAM_GOALS, png.DEFAULT_TEAM_GOALS)


def test_team_goal_form_dedupes_players_and_prefers_current_season():
    rows = []
    # two players of the same match must count once
    for name in ("1", "2"):
        rows.append({"season_x": "2026-27", "team_x": "Arsenal", "kickoff_time": "2026-09-01T15:00:00Z",
                     "was_home": True, "team_h_score": 3, "team_a_score": 1, "name": name})
    rows.append({"season_x": "2026-27", "team_x": "Arsenal", "kickoff_time": "2026-09-08T15:00:00Z",
                 "was_home": False, "team_h_score": 2, "team_a_score": 0, "name": "1"})
    rows.append({"season_x": "2025-26", "team_x": "Arsenal", "kickoff_time": "2026-05-01T15:00:00Z",
                 "was_home": True, "team_h_score": 5, "team_a_score": 5, "name": "1"})
    df = pd.DataFrame(rows)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)

    assert png.team_goal_form(df, "2026-27", n=2)["Arsenal"] == pytest.approx(((3 + 0) / 2, (1 + 2) / 2))
    # fewer current matches than n: topped up with last season's
    assert png.team_goal_form(df, "2026-27", n=3)["Arsenal"] == pytest.approx(((3 + 0 + 5) / 3, (1 + 2 + 5) / 3))


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
    ])


def test_assemble_output_applies_availability_blank_and_schema():
    by_team = png.gameweek_fixtures(GW_FIXTURES, 7, TEAMS)
    predictions = {"1": {"points": 9.0, "method": "xgboost_v2"}, "2": {"points": 4.0, "method": "xgboost_v2"},
                   "3": {"points": 5.0, "method": "xgboost_v2"}, "5": {"points": 3.0, "method": "xgboost_v2"}}
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
        "ict_index": 1.0, "influence": 1.0, "threat": 1.0,
    }, {
        "season_x": "2025-26", "name": "Someone Else", "position": "FWD", "team_x": "Leeds",
        "kickoff_time": "2026-05-10T15:00:00Z", "GW": 37, "was_home": False, "total_points": 2, "value": 60,
        "team_h_score": 1, "team_a_score": 1, "fixture": 371, "minutes": 90, "creativity": 1.0,
        "ict_index": 1.0, "influence": 1.0, "threat": 1.0,
    }])
    hist.to_csv(tmp_path / "history.csv", index=False)
    return tmp_path


def test_load_history_excludes_predicted_gameweek_and_later(live_dir):
    players = players_frame()
    history, previous = png.load_history(players, str(live_dir / "FPL_2026_2027.csv"), "2026-27",
                                         before_gw=6, before_time="2026-10-17T10:00:00Z",
                                         history_csv=str(live_dir / "history.csv"))
    current = history[history["season_x"] == "2026-27"]
    assert current["GW"].max() == 5                     # GW6/GW7 rows never feed features
    assert set(history["name"]) == {"1", "2", "3"}      # keyed by element; non-current players dropped
    assert len(history[(history["name"] == "1") & (history["season_x"] == "2025-26")]) == 1  # name-matched
    assert previous == ["Arsenal", "Leeds"]
    assert history.groupby("name")["kickoff_time"].apply(lambda s: s.is_monotonic_increasing).all()

    # kickoff cutoff also applies (a later-round match played early would be dropped)
    history, _ = png.load_history(players, str(live_dir / "FPL_2026_2027.csv"), "2026-27",
                                  before_gw=8, before_time="2026-09-10T00:00:00Z",
                                  history_csv=str(live_dir / "history.csv"))
    assert history.loc[history["season_x"] == "2026-27", "GW"].max() == 3


class ConstantModel:
    def __init__(self, value):
        self.value = value
        self.seen = []

    def predict(self, X):
        self.seen.append(X.copy())
        return np.full(len(X), self.value)


def stub_models(value=2.5):
    from ml.feature_engineering import FeatureEngineer
    models = png.PositionModels.__new__(png.PositionModels)
    models.models, models.engineers = {}, {}
    for pos in png.POSITIONS:
        fe = FeatureEngineer()
        fe.load_mappings(os.path.join(png.MODELS_DIR, f"xgboost_{pos.lower()}_v2_mappings.json"))
        models.engineers[pos] = fe
        models.models[pos] = ConstantModel(value)
    return models


def test_predict_players_sums_double_gameweek_and_sets_context(live_dir):
    players = players_frame()
    history, _ = png.load_history(players, str(live_dir / "FPL_2026_2027.csv"), "2026-27", before_gw=7,
                                  before_time=None, history_csv=str(live_dir / "history.csv"))
    by_team = png.gameweek_fixtures(GW_FIXTURES, 7, TEAMS)
    models = stub_models(2.5)
    form = {"Arsenal": (2.0, 1.0), "Chelsea": (1.0, 1.0), "Spurs": (1.0, 2.0)}
    out = png.predict_players(models, players, history, by_team, "2026-27", 7, form)

    assert out["1"]["points"] == pytest.approx(5.0)      # Arsenal double: 2 x 2.5
    assert out["2"]["points"] == pytest.approx(2.5)
    assert "4" not in out                                 # blank (Hull City): no model row
    assert out["5"]["method"] == png.COLD_START_METHOD   # no match rows at all
    assert out["5"]["points"] > 0

    mid_rows = pd.concat(models.models["MID"].seen)       # Fit Gunner's two fixture rows
    assert list(mid_rows["was_home"]) == [1, 1]
    assert set(mid_rows["fixtures_this_gw"]) == {2}
    assert set(mid_rows["GW"]) == {7}
    # vs Chelsea: (2.0 + 1.0) / 2 for, (1.0 + 1.0) / 2 against; vs Spurs: (2.0 + 2.0) / 2, (1.0 + 1.0) / 2
    assert list(mid_rows["team_goals"]) == pytest.approx([1.5, 2.0])
    # history features come from earlier rows only: form = mean of the last 4 real points (9..12)
    assert list(mid_rows["form"]) == pytest.approx([np.mean([9, 10, 11, 12])] * 2)
    # synthetic per-match stats are recent averages, not zeros
    assert list(mid_rows["influence"]) == pytest.approx([30.0, 30.0])


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
    summary = png.predict_next_gameweek(str(live_dir), client=FakeClient(),
                                        history_csv=str(live_dir / "history.csv"))
    out = pd.read_csv(live_dir / "next_gw_predictions.csv")
    records = json.loads((live_dir / "next_gw_predictions.json").read_text(encoding="utf-8"))

    assert summary["gameweek"] == 7 and summary["players"] == 5 == len(out) == len(records)
    assert summary["blank_teams"] == ["Hull City"]
    assert summary["double_gameweek_players"] == 3        # the three Arsenal players
    assert summary["history_last_kickoff"] < "2026-10-17"
    assert not any(p.endswith(".tmp") for p in os.listdir(live_dir))
    for col in ["name", "team", "position", "opponent", "is_home", "kickoff_time", "gameweek",
                "predicted_points", "value", "prediction_method"]:
        assert col in out.columns
    rows = out.set_index("name")
    assert rows.loc["Injured Gunner", "predicted_points"] == 0
    assert rows.loc["Gone Gunner", "predicted_points"] == 0
    assert rows.loc["Fit Gunner", "predicted_points"] > 0
    assert rows.loc["Fit Gunner", "opponent"] == "Chelsea & Spurs"


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
