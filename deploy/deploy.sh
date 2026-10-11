#!/usr/bin/env bash
# Build and (re)start the stack on the VM. Run from anywhere inside the repo:
#
#   bash deploy/deploy.sh               # pull latest code, rebuild, restart
#   bash deploy/deploy.sh --no-pull     # rebuild what's checked out
#   bash deploy/deploy.sh --load-data   # also (re)load the graph from the CSVs
set -euo pipefail

cd "$(dirname "$0")/.."
COMPOSE=(docker compose -f deploy/docker-compose.yml)

pull=1
load_data=0
for arg in "$@"; do
  case "$arg" in
    --no-pull) pull=0 ;;
    --load-data) load_data=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -f deploy/.env ]; then
  echo "deploy/.env is missing — run deploy/setup-vm.sh first" >&2
  exit 1
fi

if [ "$pull" = 1 ]; then
  git pull --ff-only
fi

"${COMPOSE[@]}" up -d --build --remove-orphans

if [ "$load_data" = 1 ]; then
  echo "==> Loading the graph (wipes Neo4j, takes a minute or two)"
  "${COMPOSE[@]}" run --rm backend python load_data.py
  # The running backend caches the player index and season list built at startup
  "${COMPOSE[@]}" restart backend
fi

docker image prune -f >/dev/null
"${COMPOSE[@]}" ps
