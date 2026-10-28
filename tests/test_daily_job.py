"""Checks on deploy/install-daily-job.sh: the Friday run and the one lock around every pipeline run."""
import os
import re
import shutil
import subprocess

import pytest

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "install-daily-job.sh")


@pytest.fixture(scope="module")
def script():
    with open(SCRIPT, encoding="utf-8") as f:
        return f.read()


def test_timer_runs_twice_a_day_and_on_friday_afternoon(script):
    assert 'schedule="${FPL_DAILY_SCHEDULE:-*-*-* 06,18:00:00 UTC}"' in script
    assert 'friday_schedule="${FPL_FRIDAY_SCHEDULE:-Fri *-*-* 16:00:00 UTC}"' in script
    assert "OnCalendar=${schedule}\nOnCalendar=${friday_schedule}\n" in script
    assert "Persistent=true" in script


def test_every_run_goes_through_one_lock(script):
    assert 'lock_file="${FPL_PIPELINE_LOCK:-/run/lock/fpl-pipeline.lock}"' in script
    runner = re.search(r'sudo tee "\$runner" >/dev/null <<EOF\n(.*?)\nEOF\n', script, re.S).group(1)
    exec_line = next(line for line in runner.splitlines() if line.startswith("exec "))
    assert exec_line.startswith('exec /usr/bin/flock --wait 2400 "${lock_file}" /usr/bin/docker compose')
    assert exec_line.endswith('run --rm backend python -m "\\$@"')
    # the timer's service runs the daily job through it, and nothing else calls docker directly
    assert "ExecStart=${runner} pipeline.daily" in script
    assert script.count("docker compose") == 1


def test_install_is_idempotent(script):
    # files are overwritten (tee, not append) and the timer restarted to pick up a new schedule
    assert "tee -a" not in script and ">>" not in script
    assert "sudo systemctl daemon-reload" in script and "sudo systemctl restart fpl-daily.timer" in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_script_parses(script):
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)   # stdin: no path translation
