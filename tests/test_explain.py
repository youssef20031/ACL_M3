"""services/explain.py (SHAP + LIME over the saved model inputs) and ml/lime_tabular.py."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from ml import lime_tabular
from ml import points_model as pm
from pipeline import predict_next_gw as png
from services import explain
from services import predictions as preds_service

xgb = pytest.importorskip("xgboost")


# ------------------------------------------------------------------ LIME

def test_weighted_ridge_recovers_a_linear_function():
    rng = np.random.default_rng(0)
    z = rng.random((200, 3))
    y = 2.0 + z @ np.array([1.0, -3.0, 0.5])
    coef, intercept = lime_tabular.weighted_ridge(z, y, rng.random(200) + 0.1, alpha=1e-9)
    assert coef == pytest.approx([1.0, -3.0, 0.5], abs=1e-6)
    assert intercept == pytest.approx(2.0, abs=1e-6)


def test_lime_finds_the_features_the_model_uses():
    rng = np.random.default_rng(1)
    background = rng.random((400, 6))
    background[:, 5] = rng.integers(0, 2, 400)          # a flag: categorical

    def model(m):  # only features 0 (up) and 5 (down) matter
        return 3.0 * (m[:, 0] > 0.75) - 2.0 * m[:, 5]

    x = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 1.0])
    names = [f"f{j}" for j in range(6)]
    res = lime_tabular.explain(x, background, model, names, num_samples=3000, num_features=3, seed=0)
    assert {names[j] for j in res.features[:2]} == {"f0", "f5"}
    weights = dict(zip((names[j] for j in res.features), res.weights))
    assert weights["f0"] > 2.0 and weights["f5"] < -1.5
    assert res.model_prediction == pytest.approx(1.0)
    assert res.score > 0.9
    conditions = dict(zip((names[j] for j in res.features), res.conditions))
    assert conditions["f0"].startswith("f0 > ")       # top quartile
    assert conditions["f5"] == "f5 = 1"
    again = lime_tabular.explain(x, background, model, names, num_samples=3000, num_features=3, seed=0)
    assert again.weights == res.weights                # seeded: same explanation every time


def test_lime_discretizer_bins_and_missing_values():
    bg = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [np.nan]])
    d = lime_tabular.Discretizer(bg, ["x"])
    assert d.bins(np.array([[1.0], [2.0], [3.5], [9.0], [np.nan]]))[:, 0].tolist() == [0, 0, 2, 3, -1]
    assert d.condition(0, 0) == "x <= 2"
    assert d.condition(0, 2) == "3 < x <= 4"
    assert d.condition(0, -1) == "x is missing"


# ------------------------------------------------------------------ the service

ROWS = [  # element, name, team, position, opponent, fixture
    (1, "Fit Gunner", "Arsenal", "MID", "Chelsea", 71),
    (1, "Fit Gunner", "Arsenal", "MID", "Spurs", 72),     # double gameweek
    (2, "Doubtful Blue", "Chelsea", "FWD", "Arsenal", 71),
    (3, "Björn Defender", "Spurs", "DEF", "Arsenal", 72),
]


def _make_live(tmp_path, monkeypatch, version="v4", members=1):
    """A tiny `version` model on its feature columns in LIVE_DATA_DIR/models (no bundled models),
    its meta, saved inputs and a predictions file. One booster in the format from before ensembles
    (no "members" in the meta), or with `members` > 1 an ensemble saved by pm.save_model."""
    features = pm.feature_columns(version)
    rng = np.random.default_rng(0)
    x = rng.random((300, len(features)))
    y = 4 * x[:, features.index("minutes_r3")] + 2 * x[:, features.index("team_xgf")] + rng.normal(0, 0.1, 300)
    train = xgb.DMatrix(x, label=y, feature_names=features)
    models = tmp_path / "models"
    models.mkdir()
    scores = {"top20_points": 4.7, "captain_points": 6.6, "rmse": 1.9, "mae": 1.0, "spearman": 0.7}
    meta = {"version": version, "trained_at": "2026-10-20T00:00:00+00:00", "n_trees": 20, "seasons": ["2025-26"],
            "rows": 300, "features": features,
            "test_metrics": {version: scores, f"{version}_cop": scores, "last3_avg": {"top20_points": 3.9},
                             "fpl_xP": {"top20_points": 2.7}}}
    model_path, meta_path = models / f"points_{version}.json", models / f"points_{version}_meta.json"
    if members == 1:
        booster = xgb.train({"max_depth": 3, "eta": 0.3}, train, 20)
        booster.save_model(str(model_path))
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    else:
        booster = pm.Ensemble(xgb.train({"max_depth": 3, "eta": 0.3, "subsample": 0.7, "seed": k}, train, 20)
                              for k in range(members))
        pm.save_model(booster, meta, str(model_path), str(meta_path), features=features)
    monkeypatch.setenv("LIVE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pm, "MODEL_DIR", str(tmp_path / "bundled"))   # ignore the repo's model files

    inputs = pd.DataFrame(rng.random((len(ROWS), len(features))), columns=features)
    inputs["was_home"] = [1.0, 0.0, 0.0, 1.0]
    for col, values in zip(["element", "name", "team", "position", "opponent", "fixture"], zip(*ROWS)):
        inputs[col] = values
    inputs["kickoff_time"] = ["2026-10-24T14:00:00Z", "2026-10-27T19:30:00Z", "2026-10-24T14:00:00Z",
                              "2026-10-27T19:30:00Z"]
    inputs["GW"], inputs["season"] = 9, "2026-27"
    inputs["model_points"] = booster.predict(xgb.DMatrix(inputs[features], feature_names=features))
    png.save_model_inputs(inputs, str(tmp_path / explain.FEATURES_NAME))
    pd.DataFrame([
        {"name": "Fit Gunner", "element": 1, "predicted_points": 7.1, "status": "a",
         "chance_of_playing_next_round": None, "news": ""},
        {"name": "Doubtful Blue", "element": 2, "predicted_points": 1.2, "status": "d",
         "chance_of_playing_next_round": 50, "news": "Knock"},
        {"name": "Björn Defender", "element": 3, "predicted_points": 2.0, "status": "a",
         "chance_of_playing_next_round": None, "news": ""},
    ]).to_csv(tmp_path / "next_gw_predictions.csv", index=False)
    monkeypatch.setattr(preds_service, "_cached", None)
    explain.reset()
    return booster, inputs


@pytest.fixture
def live(tmp_path, monkeypatch):
    yield _make_live(tmp_path, monkeypatch)
    explain.reset()


def test_shap_adds_up_to_the_model_output(live):
    booster, inputs = live
    p = explain.player(1)
    assert [f["opponent"] for f in p["fixtures"]] == ["Chelsea", "Spurs"]
    for f, raw in zip(p["fixtures"], inputs["model_points"][:2]):
        assert f["base_value"] + sum(c["shap"] for c in f["contributions"]) == pytest.approx(raw, abs=1e-3)
        assert f["model_points"] == pytest.approx(raw, abs=1e-3)
        shaps = [abs(c["shap"]) for c in f["contributions"]]
        assert shaps == sorted(shaps, reverse=True)
        assert len(f["contributions"]) == len(pm.feature_columns())
        assert f["lime"]["num_samples"] == explain.LIME_SAMPLES and len(f["lime"]["weights"]) == explain.LIME_FEATURES
    assert p["fixtures"][0]["is_home"] is True and p["fixtures"][1]["is_home"] is False
    assert p["model_points"] == pytest.approx(float(np.clip(inputs["model_points"][:2], 0, None).sum()), abs=1e-3)
    assert p["predicted_points"] == 7.1 and p["availability_factor"] == 1.0


def test_an_ensemble_is_explained_by_its_members_mean(tmp_path, monkeypatch):
    model, inputs = _make_live(tmp_path, monkeypatch, members=3)
    try:
        assert isinstance(model, pm.Ensemble) and (tmp_path / "models" / "points_v4_m2.json").exists()
        p = explain.player(1)
        for f, raw in zip(p["fixtures"], inputs["model_points"][:2]):
            assert f["base_value"] + sum(c["shap"] for c in f["contributions"]) == pytest.approx(raw, abs=1e-3)
        assert explain.summary()["model"]["members"] == 3
    finally:
        explain.reset()


def test_player_view_applies_fpl_availability(live):
    p = explain.player(2)
    assert p["availability_factor"] == 0.77                       # a 50% doubt's fitted factor
    assert p["player"]["status"] == "d" and p["player"]["news"] == "Knock"
    assert p["player"]["chance_of_playing_next_round"] == 50
    with pytest.raises(KeyError):
        explain.player(999)


def test_player_view_when_a_feature_is_missing_for_every_player(live, tmp_path):
    """v4's odds are NaN for the whole gameweek until football-data.co.uk posts Friday's prices (and for
    gameweeks starting on a Tuesday): the player view must still explain, those features 'missing'."""
    from ml import odds as od

    booster, _ = live
    path = tmp_path / explain.FEATURES_NAME
    frame = pd.read_csv(path, float_precision="round_trip")
    frame[od.FEATURES] = np.nan
    features = pm.feature_columns()
    frame["model_points"] = booster.predict(xgb.DMatrix(frame[features], feature_names=features))
    frame.to_csv(path, index=False)
    explain.reset()
    p = explain.player(1)
    assert all(f["lime"]["num_samples"] == explain.LIME_SAMPLES for f in p["fixtures"])
    assert explain.summary()["rows"] == 4


def test_summary_ranks_features_and_players(live):
    s = explain.summary()
    assert s["gameweek"] == 9 and s["season"] == "2026-27" and s["rows"] == 4 and s["players"] == 3
    assert [f["feature"] for f in s["features"][:2]] == ["minutes_r3", "team_xgf"]  # what the model uses
    assert len(s["features"]) == len(pm.feature_columns())
    assert s["features"][0]["label"] == "Minutes, last 3 matches" and s["features"][0]["group"] == "Recent form"
    assert len(s["beeswarm"]) == explain.BEESWARM_FEATURES and len(s["beeswarm"][0]["points"]) == 4
    assert [g["group"] for g in s["groups"]][:2] == ["Recent form", "Team strength"]
    assert [p["element"] for p in s["top_players"]] == [1, 3, 2]          # by the site's predicted points
    assert s["model"]["version"] == "v4" and s["model"]["source"] == "live"
    assert set(s["model"]["test_metrics"]) == {"v4", "last3_avg"}   # the model's own, then the baselines
    groups = {f["group"] for f in s["features"]}
    assert {"Team ratings", "Transfers", "Bookmaker forecasts", "Availability"} <= groups


def test_explains_v3_until_a_v4_model_exists(tmp_path, monkeypatch):
    _make_live(tmp_path, monkeypatch, version="v3")
    try:
        s = explain.summary()
        assert s["model"]["version"] == "v3" and set(s["model"]["test_metrics"]) == {"v3", "last3_avg"}
        assert len(s["features"]) == len(pm.V3_FEATURES)
    finally:
        explain.reset()


def test_search_dependence_and_name_lookup(live):
    assert [p["name"] for p in explain.search("bjorn")] == ["Björn Defender"]   # accents ignored
    assert explain.find_element("bjorn defender") == 3
    assert explain.find_element("Nobody") is None
    d = explain.dependence("minutes_r3")
    assert d["label"] == "Minutes, last 3 matches" and len(d["points"]) == 4
    with pytest.raises(KeyError):
        explain.dependence("not_a_feature")


def test_unavailable_without_inputs_or_with_stale_columns(live, tmp_path):
    path = tmp_path / explain.FEATURES_NAME
    frame = pd.read_csv(path)
    frame.drop(columns=["minutes_r3"]).to_csv(path, index=False)
    explain.reset()
    with pytest.raises(explain.ExplainUnavailable):
        explain.summary()
    path.unlink()
    with pytest.raises(explain.ExplainUnavailable):
        explain.summary()


def test_every_feature_has_a_label():
    for f in pm.feature_columns("v4"):
        label, group, description = explain.feature_info(f)
        assert label != f and group != "Other" and description, f
        # Shown on the site: bookmakers' prices are forecasts, not betting
        text = f"{label} {group} {description}"
        assert not re.search(r"\b(bet|betting|gambl\w*|wager\w*|odds|punt\w*|stakes?)\b", text, re.I), f


def test_saved_inputs_round_trip_exactly(tmp_path):
    # Tree splits sit on training values like 1/3; a rounded copy would take the other branch
    rows = pd.DataFrame({c: [1 / 3, 2 / 3, 0.1 + 0.2] for c in pm.feature_columns()})
    for col, value in zip(png.FEATURE_ID_COLUMNS, [1, "A", "MID", "Arsenal", "Chelsea", 7, "2026-10-24", 9, "2026-27"]):
        rows[col] = value
    rows["element"] = [1, 2, 3]
    rows["model_points"] = 1 / 7
    path = tmp_path / "inputs.csv"
    png.save_model_inputs(rows, str(path))
    back = pd.read_csv(path, float_precision="round_trip").sort_values("element")
    assert np.array_equal(back[pm.feature_columns()].to_numpy(), rows[pm.feature_columns()].to_numpy())


def test_unavailable_when_the_model_changed_after_the_run(live, tmp_path):
    path = tmp_path / explain.FEATURES_NAME
    frame = pd.read_csv(path)
    frame["model_points"] += 0.5
    frame.to_csv(path, index=False)
    explain.reset()
    with pytest.raises(explain.ExplainUnavailable, match="model changed"):
        explain.summary()


# ------------------------------------------------------------------ routes

def test_explain_routes(live, tmp_path):
    from fastapi.testclient import TestClient
    import api_main

    client = TestClient(api_main.app)   # no lifespan: no Neo4j, no models loaded
    assert client.get("/api/ml/explain/summary").json()["gameweek"] == 9
    assert client.get("/api/ml/explain/player", params={"element": 1}).json()["player"]["name"] == "Fit Gunner"
    assert client.get("/api/ml/explain/player", params={"name": "bjorn defender"}).json()["player"]["element"] == 3
    assert client.get("/api/ml/explain/player", params={"name": "Doubtful"}).json()["player"]["element"] == 2
    assert client.get("/api/ml/explain/player", params={"name": "Nobody Here"}).status_code == 404
    assert client.get("/api/ml/explain/player", params={"element": 999}).status_code == 404
    assert client.get("/api/ml/explain/player").status_code == 422
    assert client.get("/api/ml/explain/dependence", params={"feature": "price"}).json()["label"] == "Price (£m)"
    assert client.get("/api/ml/explain/dependence", params={"feature": "nope"}).status_code == 404
    assert [p["element"] for p in client.get("/api/ml/explain/players", params={"q": "gunner"}).json()["players"]] == [1]
    (tmp_path / explain.FEATURES_NAME).unlink()
    assert client.get("/api/ml/explain/summary").status_code == 503
    assert client.get("/api/ml/explain/players").status_code == 503
