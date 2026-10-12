"""Tests for pipeline/retrain.py: schedule, guardrail, and promotion of the candidate model."""
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from ml import points_model as pm
from pipeline import retrain

MONDAY = datetime(2026, 10, 19, 6, 5, tzinfo=timezone.utc)


def metrics(v3, last3=3.8, last6=3.9, ppg=4.2):
    return {"v3": {"top20_points": v3, "mae": 1.0}, "last3_avg": {"top20_points": last3},
            "last6_avg": {"top20_points": last6}, "season_ppg": {"top20_points": ppg}}


@pytest.mark.parametrize("now, last, due", [
    (MONDAY, None, True),
    (MONDAY, MONDAY - timedelta(days=7), True),
    (MONDAY, MONDAY - timedelta(hours=12), False),     # the 18:00 run on the same Monday
    (MONDAY + timedelta(days=1), MONDAY - timedelta(days=8), False),  # Tuesday
])
def test_is_due(now, last, due):
    assert retrain.is_due(now, last) is due


def test_guardrail():
    assert retrain.passes_guardrail(metrics(4.6), metrics(4.5))[0]
    ok, reason = retrain.passes_guardrail(metrics(4.1), None)          # loses to season ppg 4.2
    assert not ok and "baseline" in reason
    ok, reason = retrain.passes_guardrail(metrics(4.3), metrics(4.9))  # > 10% below the active model
    assert not ok and "below" in reason
    assert retrain.passes_guardrail(metrics(4.5), metrics(4.9))[0]     # within 10%


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_DATA_DIR", str(tmp_path))
    return tmp_path


def fake_trainer(top20):
    def train_and_report(model_path, meta_path, report_path, eval_only=False):
        with open(model_path, "w") as f:
            f.write("{}")
        with open(meta_path, "w") as f:
            json.dump({"trained_at": MONDAY.isoformat(), "test_metrics": metrics(top20)}, f)
        report = {"test": {"all": metrics(top20)}, "production": {"rows": 1000}, "seconds": 1.0}
        with open(report_path, "w") as f:
            json.dump(report, f)
        return report
    return train_and_report


def test_accepted_candidate_becomes_active(live, monkeypatch):
    monkeypatch.setattr("ml.train_points_v3.train_and_report", fake_trainer(4.7))
    assert pm.active_model_paths() == (pm.MODEL_PATH, pm.META_PATH)   # bundled before

    result = retrain.maybe_retrain(force=True, now=MONDAY)
    assert result["status"] == "accepted"
    models = live / "models"
    assert pm.active_model_paths() == (str(models / "points_v3.json"), str(models / "points_v3_meta.json"))
    assert not any("candidate" in p for p in os.listdir(models))
    history = json.loads((models / "retrain_history.json").read_text())
    assert history[-1]["accepted"] is True and history[-1]["test_top20"] == 4.7


def test_rejected_candidate_leaves_active_model(live, monkeypatch):
    monkeypatch.setattr("ml.train_points_v3.train_and_report", fake_trainer(3.0))  # worse than baselines
    result = retrain.maybe_retrain(force=True, now=MONDAY)
    assert result["status"] == "rejected"
    assert pm.active_model_paths() == (pm.MODEL_PATH, pm.META_PATH)
    assert not any("candidate" in p for p in os.listdir(live / "models"))


def test_not_due_skips_without_training(live, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not train")
    monkeypatch.setattr("ml.train_points_v3.train_and_report", boom)
    assert retrain.maybe_retrain(now=MONDAY + timedelta(days=2))["status"] == "skipped"
