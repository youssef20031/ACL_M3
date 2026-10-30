"""pipeline.daily: run_problems (failures the pipeline steps report in their summaries instead of
raising, which the daily job turns into a failed healthcheck) and the --before-deadline gate."""
import json
from datetime import datetime, timezone

from pipeline import daily

# A Saturday 12:30 kickoff in November (UK time = UTC): the deadline is 11:00, and football-data.co.uk
# collects the weekend's prices by Friday 17:00.
DEADLINE = "2026-11-07T11:00:00Z"
FRIDAY_NOON = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
FRIDAY_EVENING = datetime(2026, 11, 6, 18, 0, tzinfo=timezone.utc)


def predictions(**overrides):
    out = {"gameweek": 11, "deadline": DEADLINE, "odds": {"fixtures.csv": "fetched"}, "extra_inputs": "ok",
           "input_coverage": {"odds": 0.97, "snapshot": 1.0}, "model_pick": "/data/model_picks/gw11.json",
           "projections": {"output": "/data/projections.csv", "gameweeks": [11, 12, 13, 14], "failed": None}}
    return {**out, **overrides}


def manager(**overrides):
    out = {"season": "2026-27", "start": False, "lock": {"locked": [], "rolled_over": []},
           "score": {"scored": [], "ranked": []},
           "plan": {"status": "planned", "gameweek": 11, "optimal": True},
           "check": {"gameweek": 11, "starters_out": 0, "problems": []}, "state": {}}
    return {**out, **overrides}


def test_a_clean_run_has_no_problems():
    assert daily.run_problems({"predictions": predictions(), "ml_manager": manager()}, FRIDAY_EVENING) == []
    # Skipped or failed steps that already raise are reported by run_daily, not here
    assert daily.run_problems({"predictions": "failed", "ml_manager": "failed"}, FRIDAY_EVENING) == []


def test_ml_manager_failures_and_checks_are_problems():
    summary = {"predictions": predictions(), "ml_manager": manager(
        score="failed: league lookup crashed",
        plan={"status": "rejected", "errors": ["15 players needed"]},
        check={"gameweek": 11, "problems": ["GW11's captain Player 3 is out (status i; plan from x)"]})}
    problems = daily.run_problems(summary, FRIDAY_EVENING)
    assert problems == ["ml_manager score: failed: league lookup crashed",
                        "ml_manager plan: rejected by the rules: ['15 players needed']",
                        "ml_manager: GW11's captain Player 3 is out (status i; plan from x)"]
    timed_out = manager(plan={"status": "planned", "gameweek": 11, "optimal": False})
    assert daily.run_problems({"ml_manager": timed_out}, FRIDAY_EVENING) == [
        "ml_manager plan: GW11's solve stopped at its time limit"]
    skipped = manager(plan={"status": "skipped: no deadline ahead"}, check={"problems": []})
    assert daily.run_problems({"ml_manager": skipped}, FRIDAY_EVENING) == []


def test_prediction_side_problems():
    summary = {"predictions": predictions(model_pick="failed", extra_inputs={"odds": "failed: bad file"},
                                          projections={"output": None, "failed": "failed: boom"})}
    assert daily.run_problems(summary, FRIDAY_EVENING) == [
        "predictions: the model's pick wasn't saved", "predictions: projections failed: failed: boom",
        "predictions: unreadable v4 inputs: {'odds': 'failed: bad file'}"]


def test_missing_odds_count_only_after_they_were_collected_and_before_the_deadline():
    none = predictions(input_coverage={"odds": 0.0})
    assert daily.run_problems({"predictions": none}, FRIDAY_NOON) == []        # not collected yet
    problems = daily.run_problems({"predictions": none}, FRIDAY_EVENING)
    assert len(problems) == 1 and problems[0].startswith("predictions: odds for 0% of GW11's rows")
    after_deadline = datetime(2026, 11, 7, 11, 30, tzinfo=timezone.utc)
    assert daily.run_problems({"predictions": none}, after_deadline) == []
    a_few_missing = predictions(input_coverage={"odds": 0.8})                  # a rearranged match or two
    assert daily.run_problems({"predictions": a_few_missing}, FRIDAY_EVENING) == []
    assert daily.run_problems({"predictions": predictions(input_coverage={})}, FRIDAY_EVENING) == []


# ----------------------------------------------------------------------------- --before-deadline

def write_live(tmp_path, deadlines, last_started=None):
    meta = {"events": [{"id": k, "deadline_time": d} for k, d in enumerate(deadlines, start=10)],
            "next_deadline": deadlines[0] if deadlines else None}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if last_started:
        (tmp_path / daily.RUNS_FILE).write_text(json.dumps({"last_started": last_started}), encoding="utf-8")
    return str(tmp_path)


def at(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def test_next_deadline_is_the_first_one_ahead():
    meta = {"events": [{"deadline_time": "2026-10-31T11:00:00Z"}, {"deadline_time": DEADLINE},
                       {"deadline_time": None}], "next_deadline": "2026-10-31T11:00:00Z"}
    assert daily.next_deadline(meta, FRIDAY_EVENING) == at(DEADLINE)
    assert daily.next_deadline(meta, at(DEADLINE)) is None
    assert daily.next_deadline({}, FRIDAY_EVENING) is None


def test_a_pre_deadline_run_starts_45_to_105_minutes_before_the_deadline(tmp_path):
    live = write_live(tmp_path, [DEADLINE])
    assert daily.pre_deadline_due(live, at("2026-11-07T09:10:00Z"))[0] is False     # 110 minutes before
    assert daily.pre_deadline_due(live, at("2026-11-07T09:15:00Z"))[0] is True
    assert daily.pre_deadline_due(live, at("2026-11-07T10:15:00Z"))[0] is True
    due, why = daily.pre_deadline_due(live, at("2026-11-07T10:20:00Z"))            # too close to finish
    assert due is False and "45-105 minutes" in why
    assert daily.pre_deadline_due(str(tmp_path / "missing"), at("2026-11-07T09:40:00Z"))[0] is False


def test_no_pre_deadline_run_after_a_recent_one(tmp_path):
    now = at("2026-11-07T09:40:00Z")
    assert daily.pre_deadline_due(write_live(tmp_path, [DEADLINE], "2026-11-07T06:05:00Z"), now)[0] is True
    due, why = daily.pre_deadline_due(write_live(tmp_path, [DEADLINE], "2026-11-07T08:30:00Z"), now)
    assert due is False and "a run started at" in why


def test_runs_record_their_start_and_pre_deadline_runs_skip_the_retrain(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(daily, "pre_deadline_due", lambda live, now: (False, "not near"))
    monkeypatch.setattr(daily, "run_daily", lambda *a, **k: calls.append(k) or {})
    monkeypatch.setattr(daily, "_ping", lambda *a, **k: calls.append("ping"))
    monkeypatch.setattr("sys.argv", ["daily", "--live-dir", str(tmp_path), "--before-deadline"])
    assert daily.main() == 0 and calls == [] and not (tmp_path / daily.RUNS_FILE).exists()
    monkeypatch.setattr(daily, "pre_deadline_due", lambda live, now: (True, "near"))
    assert daily.main() == 0
    assert calls[1]["allow_retrain"] is False and calls.count("ping") == 2
    recorded = json.loads((tmp_path / daily.RUNS_FILE).read_text(encoding="utf-8"))
    assert recorded["before_deadline"] is True and at(recorded["last_started"]) <= datetime.now(timezone.utc)
    calls.clear()
    monkeypatch.setattr("sys.argv", ["daily", "--live-dir", str(tmp_path)])
    assert daily.main() == 0 and calls[1]["allow_retrain"] is True
