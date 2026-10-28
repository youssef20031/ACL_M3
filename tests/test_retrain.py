"""Tests for pipeline/retrain.py: schedule, guardrail, and promotion of the candidate (v4) model."""
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from ml import points_model as pm
from pipeline import retrain

MONDAY = datetime(2026, 10, 19, 6, 5, tzinfo=timezone.utc)


def metrics(top20, last3=3.8, last6=3.9, ppg=4.2, version=retrain.VERSION):
    return {version: {"top20_points": top20, "mae": 1.0}, "last3_avg": {"top20_points": last3},
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


def test_guardrail_against_the_v3_model_while_no_v4_exists():
    active = metrics(4.9, version="v3")
    assert retrain.VERSION == "v4"
    ok, reason = retrain.passes_guardrail(metrics(4.3), active, active_version="v3")
    assert not ok and "below" in reason
    assert retrain.passes_guardrail(metrics(4.5), active, active_version="v3")[0]


@pytest.fixture
def odds_calls(monkeypatch):
    """No downloads: prepare_odds records its calls."""
    calls = []
    monkeypatch.setattr(retrain, "prepare_odds", lambda live_dir, now: calls.append(now) or {"E0_2627.csv": "fetched"})
    return calls


@pytest.fixture
def live(tmp_path, monkeypatch, odds_calls):
    monkeypatch.setenv("LIVE_DATA_DIR", str(tmp_path))
    return tmp_path


def fake_trainer(top20, seen=None):
    def train_and_report(model_path, meta_path, report_path, eval_only=False, version="v3"):
        if seen is not None:
            seen.append(version)
        with open(model_path, "w") as f:
            f.write("{}")
        with open(meta_path, "w") as f:
            json.dump({"version": version, "trained_at": MONDAY.isoformat(), "test_metrics": metrics(top20)}, f)
        report = {"test": {"all": metrics(top20)}, "production": {"rows": 1000}, "seconds": 1.0,
                  "input_coverage": {"all": {"odds": 0.93}}}
        with open(report_path, "w") as f:
            json.dump(report, f)
        return report
    return train_and_report


def _served_from(live_dir) -> bool:
    return pm.active_model_paths()[0].startswith(str(live_dir))


def test_accepted_candidate_becomes_active(live, odds_calls, monkeypatch):
    seen = []
    monkeypatch.setattr("ml.train_points_v3.train_and_report", fake_trainer(4.7, seen))
    assert not _served_from(live)                                    # bundled before

    result = retrain.maybe_retrain(force=True, now=MONDAY, live_dir=str(live))
    assert result["status"] == "accepted" and seen == ["v4"] and odds_calls == [MONDAY]
    models = live / "models"
    assert pm.active_model_paths() == (str(models / "points_v4.json"), str(models / "points_v4_meta.json"))
    assert pm.active_model_paths("v3")[0] == pm.bundled_model_paths("v3")[0]   # the shadow isn't retrained
    assert not any("candidate" in p for p in os.listdir(models))
    history = json.loads((models / "retrain_history.json").read_text())
    assert history[-1]["accepted"] is True and history[-1]["test_top20"] == 4.7
    assert history[-1]["version"] == "v4" and history[-1]["input_coverage"] == {"odds": 0.93}
    assert history[-1]["odds"] == {"E0_2627.csv": "fetched"}


def test_rejected_candidate_leaves_active_model(live, monkeypatch):
    before = pm.active_model_paths()
    monkeypatch.setattr("ml.train_points_v3.train_and_report", fake_trainer(3.0))  # worse than baselines
    result = retrain.maybe_retrain(force=True, now=MONDAY, live_dir=str(live))
    assert result["status"] == "rejected"
    assert pm.active_model_paths() == before
    assert not any("candidate" in p for p in os.listdir(live / "models"))


def test_rejected_candidate_isnt_retrained_the_same_week(live, monkeypatch):
    """The 06:00 attempt is rejected: the 18:00 run (same Monday, same data) doesn't train again;
    next Monday does. A run that raised isn't logged, so it's retried."""
    calls = []

    def trainer(*args, **kwargs):
        calls.append(kwargs.get("version"))
        return fake_trainer(3.0)(*args, **kwargs)                       # worse than the baselines
    monkeypatch.setattr("ml.train_points_v3.train_and_report", trainer)
    monday = MONDAY + timedelta(days=14)                               # the bundled model is older than 5 days
    assert retrain.maybe_retrain(now=monday, live_dir=str(live))["status"] == "rejected"
    assert retrain.maybe_retrain(now=monday + timedelta(hours=12), live_dir=str(live))["status"] == "skipped"
    assert retrain.maybe_retrain(now=monday + timedelta(days=7), live_dir=str(live))["status"] == "rejected"
    assert calls == ["v4", "v4"]

    def boom(*args, **kwargs):
        raise RuntimeError("football-data.co.uk is down")
    monkeypatch.setattr("ml.train_points_v3.train_and_report", boom)
    with pytest.raises(RuntimeError):
        retrain.maybe_retrain(now=monday + timedelta(days=14), live_dir=str(live))
    monkeypatch.setattr("ml.train_points_v3.train_and_report", trainer)
    assert retrain.maybe_retrain(now=monday + timedelta(days=14, hours=12), live_dir=str(live))["status"] == "rejected"


def test_prepare_odds_points_training_at_the_live_odds(tmp_path, monkeypatch):
    from ml import odds as od
    monkeypatch.delenv("ODDS_DIR", raising=False)
    monkeypatch.setattr(od, "ODDS_DIR", od.ODDS_DIR)       # restored after the test
    calls = []
    monkeypatch.setattr(retrain, "refresh_odds",
                        lambda live_dir, season, fixtures=True, now=None: calls.append((season, fixtures)) or {})
    (tmp_path / "meta.json").write_text(json.dumps({"season": "2026-27"}))
    retrain.prepare_odds(str(tmp_path), MONDAY)
    assert calls == [("2026-27", False)] and od.ODDS_DIR == str(tmp_path / "odds")
    (tmp_path / "meta.json").unlink()                      # no meta yet: the season in progress
    retrain.prepare_odds(str(tmp_path), datetime(2027, 3, 1, tzinfo=timezone.utc))
    assert calls[-1] == ("2026-27", False)


def test_not_due_skips_without_training(live, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not train")
    monkeypatch.setattr("ml.train_points_v3.train_and_report", boom)
    assert retrain.maybe_retrain(now=MONDAY + timedelta(days=2))["status"] == "skipped"
