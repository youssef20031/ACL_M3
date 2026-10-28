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
# Fridays also at 16:00 UTC, so a Friday-evening deadline (18:30 UK time) gets Friday's
# team news and bookmaker odds: football-data.co.uk collects the weekend's prices on Friday
# afternoon (by 17:00 UK time), and the 18:00 UTC run is after that deadline during British
# Summer Time. In winter 16:00 UTC is 16:00 UK time, possibly before the prices are posted;
# FPL_FRIDAY_SCHEDULE="Fri *-*-* 17:15:00 Europe/London" would always follow them.
# Persistent=true runs a missed job after the VM was down.
#
# Every pipeline run goes through /usr/local/bin/fpl-pipeline, which holds one lock
# (flock on $lock_file) for the whole run, so the timer's runs, the Friday run and manual
# runs (e.g. a forced retrain) never overlap; a run waits up to 40 minutes for the lock:
#
#   fpl-pipeline pipeline.daily
#   fpl-pipeline pipeline.retrain --force
set -euo pipefail

cd "$(dirname "$0")/.."
repo_dir="$(pwd)"
run_user="$(id -un)"
schedule="${FPL_DAILY_SCHEDULE:-*-*-* 06,18:00:00 UTC}"
friday_schedule="${FPL_FRIDAY_SCHEDULE:-Fri *-*-* 16:00:00 UTC}"
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
RandomizedDelaySec=10min
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable fpl-daily.timer
sudo systemctl restart fpl-daily.timer   # picks up a changed schedule
echo "Installed. Next runs:"
systemctl list-timers fpl-daily.timer --no-pager
echo
echo "Run it now:        sudo systemctl start fpl-daily.service"
echo "Manual runs:       fpl-pipeline pipeline.retrain --force   (waits for a running job)"
echo "Logs of past runs: journalctl -u fpl-daily.service --no-pager | tail -50"
