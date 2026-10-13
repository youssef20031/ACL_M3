#!/usr/bin/env bash
# Install (or update) the systemd timer that runs the current-season refresh
# (pipeline/daily.py) inside the backend container. Run from the repo root on the VM:
#
#   bash deploy/install-daily-job.sh
#
# Default schedule: 06:00 and 18:00 UTC. FPL finalizes results an hour or two after
# the last match of the day, so the morning run picks up the previous day's games;
# the evening run catches team news (injuries, suspensions) before deadlines.
# Persistent=true runs a missed job after the VM was down.
set -euo pipefail

cd "$(dirname "$0")/.."
repo_dir="$(pwd)"
run_user="$(id -un)"
schedule="${FPL_DAILY_SCHEDULE:-*-*-* 06,18:00:00 UTC}"

sudo tee /etc/systemd/system/fpl-daily.service >/dev/null <<EOF
[Unit]
Description=Refresh current-season FPL data, graph and predictions
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=${run_user}
WorkingDirectory=${repo_dir}
ExecStart=/usr/bin/docker compose -f deploy/docker-compose.yml run --rm backend python -m pipeline.daily
TimeoutStartSec=45min
EOF

sudo tee /etc/systemd/system/fpl-daily.timer >/dev/null <<EOF
[Unit]
Description=Run fpl-daily on a schedule

[Timer]
OnCalendar=${schedule}
RandomizedDelaySec=10min
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now fpl-daily.timer
echo "Installed. Next runs:"
systemctl list-timers fpl-daily.timer --no-pager
echo
echo "Run it now:        sudo systemctl start fpl-daily.service"
echo "Logs of past runs: journalctl -u fpl-daily.service --no-pager | tail -50"
