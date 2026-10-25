"""Tests for ml/play60_model.py: the chance of playing 60+ minutes."""
import json

import numpy as np
import pandas as pd
import pytest

from ml import play60_model
from ml import points_model as pm


def test_evaluate_auc_and_brier():
    y = [0, 0, 1, 1]
    perfect = play60_model.evaluate(y, [0.1, 0.2, 0.8, 0.9])
    assert perfect["auc"] == 1.0 and perfect["rate"] == 0.5
    assert perfect["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.04, 0.01]), abs=1e-4)
    assert perfect["base_rate_brier"] == 0.25
    assert play60_model.evaluate(y, [0.9, 0.8, 0.2, 0.1])["auc"] == 0.0
    assert play60_model.evaluate(y, [0.5, 0.5, 0.5, 0.5])["auc"] == 0.5     # ties count half


def test_target_is_sixty_minutes():
    assert list(play60_model.target(pd.DataFrame({"minutes": [0, 59, 60, 90]}))) == [0, 0, 1, 1]


def synthetic_features(n_per_season=2000, seed=0):
    """Feature rows where the last match's minutes decide the next one, most of the time.
    (Enough rows for min_child_weight 50 on a logistic loss: each row's hessian is <= 0.25.)"""
    rng = np.random.default_rng(seed)
    frames = []
    for season in ("2020-21", "2021-22", "2022-23"):
        f = pd.DataFrame(rng.random((n_per_season, len(pm.feature_columns()))), columns=pm.feature_columns())
        f["minutes_r1"] = rng.choice([0.0, 90.0], n_per_season)
        flip = rng.random(n_per_season) < 0.1
        f["minutes"] = np.where((f["minutes_r1"] > 0) ^ flip, 90.0, 0.0)
        f["season"] = season
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_train_save_load_predict(tmp_path, monkeypatch):
    monkeypatch.setitem(play60_model.PARAMS, "learning_rate", 0.3)     # converge (and stop) sooner
    model_path, meta_path = str(tmp_path / "play60.json"), str(tmp_path / "play60_meta.json")
    report = play60_model.train_and_save(synthetic_features(), ["2020-21"], ["2021-22"], model_path, meta_path)
    assert report["test_seasons"] == ["2022-23"]
    assert report["test"]["2022-23"]["auc"] > 0.8                      # learns minutes_r1
    meta = json.load(open(meta_path, encoding="utf-8"))
    assert meta["features"] == pm.feature_columns() and meta["target"] == "minutes >= 60"
    booster, meta = play60_model.load_model(model_path, meta_path)
    rows = synthetic_features(n_per_season=20, seed=1)
    p = play60_model.predict(booster, meta, rows)
    assert ((p >= 0) & (p <= 1)).all()
    assert p[rows["minutes_r1"] > 0].mean() > p[rows["minutes_r1"] == 0].mean()


def test_live_copy_wins_over_the_bundled_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_DATA_DIR", str(tmp_path))
    assert play60_model.active_model_paths() == (play60_model.MODEL_PATH, play60_model.META_PATH)
    (tmp_path / "models").mkdir()
    for name in ("play60_v1.json", "play60_v1_meta.json"):
        (tmp_path / "models" / name).write_text("{}")
    assert play60_model.active_model_paths()[0] == str(tmp_path / "models" / "play60_v1.json")


def test_bundled_model_matches_the_features():
    with open(play60_model.META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["features"] == pm.feature_columns()
    assert meta["test_metrics"]["auc"] > 0.9
