"""Checks on deploy/install-daily-job.sh (the timer's schedule and the one lock around every pipeline
run) and deploy/deploy.sh, whose pipeline jobs take the same lock."""
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ml import odds as od

DEPLOY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy")
SCRIPT = os.path.join(DEPLOY_DIR, "install-daily-job.sh")
DEPLOY = os.path.join(DEPLOY_DIR, "deploy.sh")
UK = ZoneInfo("Europe/London")
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SCHEDULES = ("FPL_DAILY_SCHEDULE", "FPL_FRIDAY_SCHEDULE", "FPL_TUESDAY_SCHEDULE")


@pytest.fixture(scope="module")
def script():
    with open(SCRIPT, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def deploy_script():
    with open(DEPLOY, encoding="utf-8") as f:
        return f.read()


def default(script: str, variable: str) -> str:
    return re.search(rf'="\$\{{{variable}:-(.*?)\}}"', script).group(1)


CALENDAR = re.compile(r"^(?:(?P<days>[A-Za-z,]+) )?\*-\*-\* (?P<hours>[\d,]+):(?P<minute>\d\d):00 (?P<tz>\S+)$")


def run_starts(expressions, start: datetime, end: datetime):
    """UTC times in [start, end) at which the timer fires, before RandomizedDelaySec, for OnCalendar
    expressions of the forms the script uses: "[Tue,Fri ]*-*-* HH[,HH]:MM:00 <time zone>"."""
    out = set()
    for expression in expressions:
        m = CALENDAR.match(expression)
        assert m, f"unexpected OnCalendar form: {expression}"
        tz = ZoneInfo(m["tz"])
        days = m["days"].split(",") if m["days"] else WEEKDAYS
        day = start.astimezone(tz).date() - timedelta(days=1)
        while day <= end.astimezone(tz).date():
            if WEEKDAYS[day.weekday()] in days:
                for hour in m["hours"].split(","):
                    t = datetime(day.year, day.month, day.day, int(hour), int(m["minute"]), tzinfo=tz)
                    if start <= t.astimezone(timezone.utc) < end:
                        out.add(t.astimezone(timezone.utc))
            day += timedelta(days=1)
    return sorted(out)


def test_timer_runs_twice_a_day_and_before_tuesday_and_friday_deadlines(script):
    assert 'schedule="${FPL_DAILY_SCHEDULE:-*-*-* 06,18:00:00 UTC}"' in script
    assert 'friday_schedule="${FPL_FRIDAY_SCHEDULE:-Fri *-*-* 17:15:00 Europe/London}"' in script
    assert 'tuesday_schedule="${FPL_TUESDAY_SCHEDULE:-Tue *-*-* 16:00:00 Europe/London}"' in script
    assert "OnCalendar=${schedule}\nOnCalendar=${friday_schedule}\nOnCalendar=${tuesday_schedule}\n" in script
    assert "Persistent=true" in script


# A gameweek's first kickoff (UK time), in a week of British Summer Time and one of GMT
FIRST_KICKOFFS = [("Fri", "20:00"), ("Sat", "11:30"), ("Sat", "12:30"), ("Sun", "14:00"), ("Mon", "20:00"),
                  ("Tue", "19:30"), ("Tue", "19:45"), ("Tue", "20:00"), ("Wed", "19:30"), ("Thu", "19:30")]
MONDAYS = {"BST": datetime(2026, 9, 7), "GMT": datetime(2026, 11, 9)}
POSTED = timedelta(minutes=15)     # football-data collects "not later than" its hour: time to post the file
RUN_TIME = timedelta(minutes=20)   # a daily run without the Monday retrain, with room to spare


@pytest.mark.parametrize("clock", sorted(MONDAYS))
@pytest.mark.parametrize("day,time", FIRST_KICKOFFS)
def test_every_deadline_gets_a_run_with_the_odds(script, clock, day, time):
    """v4 trains on each match's pre-closing prices whenever they were collected before the deadline
    (ml.odds: Friday 17:00 UK for Friday-Monday matches, Tuesday 13:00 for Tuesday-Thursday ones),
    so serving needs a run that starts after that collection and ends before the deadline, or the
    gameweek is predicted with NaN odds. The 06:00/18:00 UTC runs alone miss every Tuesday-evening
    deadline, and in summer the only Friday run before an 18:30 deadline was at 17:00 UK time."""
    hours, minutes = map(int, time.split(":"))
    kickoff = (MONDAYS[clock] + timedelta(days=WEEKDAYS.index(day))).replace(hour=hours, minute=minutes, tzinfo=UK)
    kickoff = kickoff.astimezone(timezone.utc)
    deadline = kickoff - timedelta(minutes=od.DEADLINE_MINUTES)
    collected = od.collected_at(pd.Series([pd.Timestamp(kickoff)])).iloc[0].to_pydatetime()
    delay = timedelta(minutes=int(re.search(r"\nRandomizedDelaySec=(\d+)min\n", script).group(1)))
    latest_start = deadline - delay - RUN_TIME
    runs = run_starts([default(script, v) for v in SCHEDULES], collected + POSTED, latest_start + timedelta(seconds=1))
    assert runs, f"no run between {collected + POSTED:%a %H:%M} and {latest_start:%a %H:%M} UTC ({clock})"


def test_every_run_goes_through_one_lock(script):
    assert 'lock_file="${FPL_PIPELINE_LOCK:-/run/lock/fpl-pipeline.lock}"' in script
    runner = re.search(r'sudo tee "\$runner" >/dev/null <<EOF\n(.*?)\nEOF\n', script, re.S).group(1)
    exec_line = next(line for line in runner.splitlines() if line.startswith("exec "))
    assert exec_line.startswith('exec /usr/bin/flock --wait 2400 "${lock_file}" /usr/bin/docker compose')
    assert exec_line.endswith('run --rm backend python -m "\\$@"')
    # the timer's service runs the daily job through it, and nothing else calls docker directly
    assert "ExecStart=${runner} pipeline.daily" in script
    assert script.count("docker compose") == 1


def test_deploy_runs_pipeline_jobs_through_the_lock(deploy_script):
    """deploy.sh --run-daily (and the graph loads) wait for a running timer job instead of
    overlapping it: two daily runs at once share the odds staging folder and, on Mondays, would
    both retrain on a 4 GB VM."""
    helper = re.search(r"\npipeline_job\(\) \{\n(.*?)\n\}\n", deploy_script, re.S).group(1)
    assert '/usr/local/bin/fpl-pipeline "$@"' in helper
    lines = [line.strip() for line in deploy_script.splitlines()]
    for job in ("pipeline_job pipeline.daily", "pipeline_job load_data", "pipeline_job load_data --fixtures-only"):
        assert job in lines
    # the helper's fallback is the only one-off container
    assert deploy_script.count("run --rm backend") == 1 and "load_data.py" not in deploy_script


def test_install_is_idempotent(script):
    # files are overwritten (tee, not append) and the timer restarted to pick up a new schedule
    assert "tee -a" not in script and ">>" not in script
    assert "sudo systemctl daemon-reload" in script and "sudo systemctl restart fpl-daily.timer" in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("path", [SCRIPT, DEPLOY])
def test_scripts_parse(path):
    with open(path, "rb") as f:   # bytes: text-mode stdin would turn LF into CRLF on Windows
        subprocess.run(["bash", "-n"], input=f.read(), check=True)   # stdin: no path translation
