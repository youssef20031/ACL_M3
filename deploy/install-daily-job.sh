#!/usr/bin/env bash
# Install (or update) the systemd timer that runs the current-season refresh
# (pipeline/daily.py) inside the backend container. Run from the repo root on the VM,
# again after pulling a change to this script (it's idempotent):
#
#   bash deploy/install-daily-job.sh
#
# Default schedule: 06:00 and 18:00 UTC. FPL finalizes results an hour or two after
# the last match of the day, so the morning run picks up the previous day's games;
# the evening run catches team news (injuries, suspensions) before deadlines.
# Also Fridays at 17:15 and Tuesdays at 16:00 UK time, so an evening deadline (18:00-18:30)
# gets that day's team news and bookmaker odds, an input of the v4 points model:
# football-data.co.uk collects the prices for Friday-Monday matches by 17:00 UK time on Friday
# and for Tuesday-Thursday matches by 13:00 on Tuesday (ml/odds.py). The 06:00/18:00 UTC runs
# miss Tuesday-evening deadlines (18:00 UTC is after them in summer and too close to them in
# winter), and in summer their only Friday run before an 18:30 deadline is at 07:00 UK time.
# tests/test_daily_job.py checks that every kind of deadline gets a run with the prices.
# Persistent=true runs a missed job after the VM was down.
#
# A second timer (fpl-deadline) fires every hour and calls the same job with --before-deadline,
# which runs only when the next deadline is 45-105 minutes away and no run started within 150
# minutes of it, so late team news reaches the predictions and the ML manager's plan; otherwise it
# exits at once (pipeline/daily.py). It never retrains, and it isn't Persistent: a pre-deadline
# run after the deadline would be useless.
#
# Every pipeline run goes through /usr/local/bin/fpl-pipeline, which holds one lock
# (flock on $lock_file) for the whole run, so the timer's runs, deploy.sh's jobs and manual
# runs (e.g. a forced retrain) never overlap; a run waits up to 40 minutes for the lock:
#
#   fpl-pipeline pipeline.daily
#   fpl-pipeline pipeline.retrain --force
set -euo pipefail

cd "$(dirname "$0")/.."
repo_dir="$(pwd)"
run_user="$(id -un)"
schedule="${FPL_DAILY_SCHEDULE:-*-*-* 06,18:00:00 UTC}"
friday_schedule="${FPL_FRIDAY_SCHEDULE:-Fri *-*-* 17:15:00 Europe/London}"
tuesday_schedule="${FPL_TUESDAY_SCHEDULE:-Tue *-*-* 16:00:00 Europe/London}"
deadline_schedule="${FPL_DEADLINE_SCHEDULE:-*-*-* *:40:00 UTC}"
lock_file="${FPL_PIPELINE_LOCK:-/run/lock/fpl-pipeline.lock}"
runner=/usr/local/bin/fpl-pipeline

sudo tee "$runner" >/dev/null <<EOF
#!/usr/bin/env bash
# Run a pipeline module in a one-off backend container, never alongside another pipeline run.
# Installed by deploy/install-daily-job.sh. Usage: fpl-pipeline pipeline.daily [args...]
set -euo pipefail
cd "${repo_dir}"
exec /usr/bin/flock --wait 2400 "${lock_file}" /usr/bin/docker compose -f deploy/docker-compose.yml run --rm backend python -m "\$@"
EOF
sudo chmod 755 "$runner"

sudo tee /etc/systemd/system/fpl-daily.service >/dev/null <<EOF
[Unit]
Description=Refresh current-season FPL data, graph and predictions
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=${run_user}
WorkingDirectory=${repo_dir}
ExecStart=${runner} pipeline.daily
TimeoutStartSec=90min
EOF

sudo tee /etc/systemd/system/fpl-daily.timer >/dev/null <<EOF
[Unit]
Description=Run fpl-daily on a schedule

[Timer]
OnCalendar=${schedule}
OnCalendar=${friday_schedule}
OnCalendar=${tuesday_schedule}
RandomizedDelaySec=10min
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo tee /etc/systemd/system/fpl-deadline.service >/dev/null <<EOF
[Unit]
Description=Refresh FPL data and predictions shortly before a deadline (when one is near)
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=${run_user}
WorkingDirectory=${repo_dir}
ExecStart=${runner} pipeline.daily --before-deadline
TimeoutStartSec=90min
EOF

sudo tee /etc/systemd/system/fpl-deadline.timer >/dev/null <<EOF
[Unit]
Description=Check every hour whether a deadline is near, and run fpl-deadline if so

[Timer]
OnCalendar=${deadline_schedule}
Persistent=false

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable fpl-daily.timer fpl-deadline.timer
sudo systemctl restart fpl-daily.timer fpl-deadline.timer   # picks up a changed schedule
echo "Installed. Next runs:"
systemctl list-timers fpl-daily.timer fpl-deadline.timer --no-pager
echo
echo "Run it now:        sudo systemctl start fpl-daily.service"
echo "Manual runs:       fpl-pipeline pipeline.retrain --force   (waits for a running job)"
echo "Logs of past runs: journalctl -u fpl-daily.service --no-pager | tail -50"
echo "Pre-deadline runs: journalctl -u fpl-deadline.service --no-pager | grep -i 'pre-deadline run' | tail"
