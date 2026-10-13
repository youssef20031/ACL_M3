#!/usr/bin/env bash
# Build and (re)start the stack on the VM. Run from anywhere inside the repo:
#
#   bash deploy/deploy.sh                  # pull latest code, rebuild, restart
#   bash deploy/deploy.sh --no-pull        # rebuild what's checked out
#   bash deploy/deploy.sh --load-data      # also (re)load the graph from the CSVs
#   bash deploy/deploy.sh --load-fixtures  # only (re)load promoted teams + 2026-27 fixtures
#   bash deploy/deploy.sh --run-daily      # also run the current-season refresh now
set -euo pipefail

cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f deploy/docker-compose.yml)

# Pull first, then hand over to the freshly pulled copy of this script, so options
# added in the new version work on the first run.
if [[ " $* " != *" --no-pull "* ]]; then
  git pull --ff-only
  exec bash deploy/deploy.sh --no-pull "$@"
fi

load_data=0
load_fixtures=0
run_daily=0
for arg in "$@"; do
  case "$arg" in
    --no-pull) ;;
    --load-data) load_data=1 ;;
    --load-fixtures) load_fixtures=1 ;;
    --run-daily) run_daily=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -f deploy/.env ]; then
  echo "deploy/.env is missing — run deploy/setup-vm.sh first" >&2
  exit 1
fi

# Token the daily job uses to tell the API to reload (added to older .env files too)
if ! grep -q '^ADMIN_TOKEN=.' deploy/.env; then
  sed -i '/^ADMIN_TOKEN=/d' deploy/.env
  echo "ADMIN_TOKEN=$(openssl rand -hex 24)" >> deploy/.env
  echo "==> Added ADMIN_TOKEN to deploy/.env"
fi

"${COMPOSE[@]}" up -d --build --remove-orphans

if [ "$load_data" = 1 ]; then
  echo "==> Loading the graph (wipes Neo4j, takes a minute or two)"
  "${COMPOSE[@]}" run --rm backend python load_data.py
  # The running backend caches the player index and season list built at startup
  "${COMPOSE[@]}" restart backend
elif [ "$load_fixtures" = 1 ]; then
  echo "==> Loading promoted teams and fixtures"
  "${COMPOSE[@]}" run --rm backend python load_data.py --fixtures-only
  "${COMPOSE[@]}" restart backend
fi

if [ "$run_daily" = 1 ]; then
  echo "==> Running the current-season refresh"
  "${COMPOSE[@]}" run --rm backend python -m pipeline.daily
fi

docker image prune -f >/dev/null
"${COMPOSE[@]}" ps
