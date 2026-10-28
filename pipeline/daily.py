"""
Daily refresh of the current season, run on the VM by a systemd timer
(deploy/install-daily-job.sh) inside the backend container:

    docker compose -f deploy/docker-compose.yml run --rm backend python -m pipeline.daily

Steps:
  1. pipeline.current_season: refresh news/availability and expected points; re-fetch match
     history only when FPL has finalized new results
  2. pipeline.load_current: apply new results, squads and availability to the graph
     (skipped when nothing new was fetched, unless --force)
  3. pipeline.retrain (Mondays): retrain the v4 points model on everything up to the weekend;
     the new model is only used if it passes the guardrail in pipeline/retrain.py
  4. pipeline.predict_next_gw: refresh the bookmaker odds (LIVE_DATA_DIR/odds), regenerate
     next-gameweek predictions (availability changes daily) and, before the deadline, save the
     live model's and the shadow model's (v3) squad picks
  5. POST /api/admin/reload so the running API rebuilds its caches without a restart
  6. Ping HEALTHCHECK_URL (e.g. healthchecks.io) on success, HEALTHCHECK_URL/fail on failure,
     so a silent failure turns into an email

The timer (06:00 and 18:00 UTC, plus Fridays at 16:00 UTC so a Friday-evening deadline gets
Friday's odds and team news) runs it under one flock with every other pipeline run, so two runs
never overlap (deploy/install-daily-job.sh).

Environment: LIVE_DATA_DIR, ODDS_DIR (optional, default LIVE_DATA_DIR/odds), BACKEND_URL (default http://backend:8000), ADMIN_TOKEN,
HEALTHCHECK_URL (optional), plus the usual NEO4J_* settings.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from typing import Any, Dict

import httpx

from pipeline.current_season import LIVE_DATA_DIR, run as refresh_current_season

logger = logging.getLogger("pipeline.daily")

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
HEALTHCHECK_URL = os.getenv("HEALTHCHECK_URL", "").rstrip("/")


def _ping(suffix: str = "", body: str = "") -> None:
    if not HEALTHCHECK_URL:
        return
    try:
        httpx.post(f"{HEALTHCHECK_URL}{suffix}", content=body[-10000:].encode(), timeout=10)
    except Exception as exc:  # monitoring must never break the job
        logger.warning("Healthcheck ping failed: %s", exc)


def _reload_backend() -> str:
    if not ADMIN_TOKEN:
        return "skipped (ADMIN_TOKEN not set)"
    try:
        resp = httpx.post(f"{BACKEND_URL}/api/admin/reload",
                          headers={"X-Admin-Token": ADMIN_TOKEN}, timeout=120)
        resp.raise_for_status()
        return "ok"
    except Exception as exc:
        # Not fatal: the data is already in place; the API picks it up on its next restart
        logger.warning("Backend reload failed: %s", exc)
        return f"failed: {exc}"


def _run_retrain(force: bool = False) -> Dict[str, Any]:
    """pipeline.retrain in a child process: training peaks around 0.8 GB, and a separate
    process guarantees that memory is returned before predictions run on the 4 GB VM."""
    cmd = [sys.executable, "-m", "pipeline.retrain"] + (["--force"] if force else [])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"pipeline.retrain exited {proc.returncode}:\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout)


def run_daily(live_dir: str = LIVE_DATA_DIR, force: bool = False, retrain: bool = False) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    started = time.time()

    meta = refresh_current_season(live_dir, full=force)
    summary["fetch"] = {k: meta.get(k) for k in ("season", "rows", "gameweeks", "next_gameweek", "refreshed")}

    if meta.get("refreshed") or force:
        from graph.connection import Neo4jConnection
        from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        from pipeline.load_current import load_current_season

        conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        try:
            summary["graph"] = load_current_season(conn, live_dir=live_dir)
        finally:
            conn.close()
    else:
        summary["graph"] = "skipped (no newly finished fixtures)"

    # Neither step may undo the refresh: on failure the active model / previous predictions
    # stay in place, the API still reloads, and the run is reported as failed so the
    # healthcheck alerts.
    errors = []
    try:
        summary["retrain"] = _run_retrain(force=retrain)
    except Exception:
        errors.append(("retrain", traceback.format_exc()))
        summary["retrain"] = "failed"
        logger.error("Retrain step failed:\n%s", errors[-1][1])

    try:
        from pipeline.predict_next_gw import predict_next_gameweek
        summary["predictions"] = predict_next_gameweek(live_dir=live_dir)
    except Exception:
        errors.append(("predictions", traceback.format_exc()))
        summary["predictions"] = "failed"
        logger.error("Prediction step failed:\n%s", errors[-1][1])

    summary["backend_reload"] = _reload_backend()
    summary["seconds"] = round(time.time() - started, 1)
    if errors:
        failed = ", ".join(step for step, _ in errors)
        details = "\n".join(tb for _, tb in errors)
        raise RuntimeError(f"{failed} failed (data refresh succeeded): {json.dumps(summary, default=str)}\n{details}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily current-season refresh")
    parser.add_argument("--live-dir", default=LIVE_DATA_DIR)
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch all history and reload the graph even if nothing changed")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model now, not just on Mondays")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _ping("/start")
    try:
        summary = run_daily(args.live_dir, force=args.force, retrain=args.retrain)
    except Exception:
        details = traceback.format_exc()
        logger.error("Daily refresh failed:\n%s", details)
        _ping("/fail", details)
        return 1
    report = json.dumps(summary, indent=2, default=str)
    print(report)
    _ping("", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
