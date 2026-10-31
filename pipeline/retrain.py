"""
Weekly retraining of the production points model (v4), run from pipeline/daily.py.

    python -m pipeline.retrain            # retrain if it's due (Mondays, at most every 5 days)
    python -m pipeline.retrain --force

The model is trained and evaluated with ml.train_points_v3.train_and_report(version="v4")
(held-out test seasons; ml/train_points_v4.py is the same entry point) and written to
LIVE_DATA_DIR/models/ as a candidate. Its extra inputs come from LIVE_DATA_DIR: the odds in
LIVE_DATA_DIR/odds (this season's file is refreshed first and missing past seasons fetched; v4
refuses to train when under half the rows have odds), our element_snapshots archive on top of
data/fpl_snapshots.csv, and the status file's FPL codes. It only replaces the active model when
it passes the guardrail:
  * on the held-out test seasons its top-20 picks beat every simple baseline, and
  * they are no more than MAX_REGRESSION worse than the active model's recorded result (the v3
    model's while no v4 exists yet).
ml.points_model.active_model_paths() serves the live copy when present, else the bundled
ml/models/points_v4.json. The shadow model (v3) is no longer retrained: it stays at its last
version, live copy or bundled. Every attempt is appended to LIVE_DATA_DIR/models/retrain_history.json,
with the child process's peak memory (v3 peaked around 0.8 GB; v4 has 96 features instead of 72).
The model is an ensemble (ml.points_model.fit_ensemble: 5 seeds, half the early-stopped trees),
fitted one member after another, so it takes about 2-3 times as long as one model but no more
memory; its member files are moved into place before the meta.
"""
import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from ml import points_model as pm
from pipeline.current_season import LIVE_DATA_DIR, odds_dir, refresh_odds

logger = logging.getLogger(__name__)

VERSION = pm.MODEL_VERSION          # the model retrained here
RETRAIN_WEEKDAY = 0                 # Monday: the morning run after the weekend's matches are final
MIN_INTERVAL = timedelta(days=5)    # the timer runs twice a day; retrain once per week
MAX_REGRESSION = 0.10               # reject a candidate >10% worse than the active model on top-20 picks
BASELINES = ("last3_avg", "last6_avg", "season_ppg")
HISTORY_LIMIT = 52


# ------------------------------------------------------------------ decisions (pure)

def is_due(now: datetime, last_trained: Optional[datetime]) -> bool:
    if now.weekday() != RETRAIN_WEEKDAY:
        return False
    return last_trained is None or now - last_trained >= MIN_INTERVAL


def passes_guardrail(candidate: Dict[str, Any], active: Optional[Dict[str, Any]], version: str = VERSION,
                     active_version: Optional[str] = None) -> Tuple[bool, str]:
    """candidate/active: test metrics as in report["test"]["all"] ({version: {...}, "last3_avg": {...}, ...}).
    `active_version` names the active model's entry (default `version`): v3's while no v4 exists."""
    top20 = candidate[version]["top20_points"]
    best_baseline = max((candidate[b]["top20_points"] for b in BASELINES if b in candidate), default=float("-inf"))
    if top20 <= best_baseline:
        return False, f"top-20 {top20:.3f} does not beat the best baseline ({best_baseline:.3f})"
    reference = (active or {}).get(active_version or version)
    if reference:
        floor = reference["top20_points"] * (1 - MAX_REGRESSION)
        if top20 < floor:
            return False, f"top-20 {top20:.3f} is more than {MAX_REGRESSION:.0%} below the active model's {reference['top20_points']:.3f}"
    return True, f"top-20 {top20:.3f} (best baseline {best_baseline:.3f})"


# ------------------------------------------------------------------ orchestration

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def last_trained_at() -> Optional[datetime]:
    """When the active model was trained (live copy, else the bundled one)."""
    _, meta_path = pm.active_model_paths()
    meta = _read_json(meta_path) or {}
    return _parse_time(meta.get("trained_at"))


def last_attempt_at(out_dir: str) -> Optional[datetime]:
    """When VERSION was last retrained, accepted or not (retrain_history.json). A rejected candidate
    leaves the active model's trained_at as it was, so without this the same Monday's evening run
    would train again, on the same data, at another ~1 GB peak."""
    history = _read_json(os.path.join(out_dir, "retrain_history.json")) or []
    times = [_parse_time(e.get("at")) for e in history if isinstance(e, dict) and e.get("version") == VERSION]
    return max((t for t in times if t is not None), default=None)


def _live_season(live_dir: str, now: datetime) -> str:
    """The season in LIVE_DATA_DIR/meta.json, else the one in progress (a new one from July)."""
    meta = _read_json(os.path.join(live_dir, "meta.json")) or {}
    if meta.get("season"):
        return meta["season"]
    start = now.year if now.month >= 7 else now.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def prepare_odds(live_dir: str, now: datetime) -> Dict[str, str]:
    """Refresh the odds files (this season's results and prices, missing past seasons) and point
    ml.odds at them, since train_and_report reads ml.odds.ODDS_DIR."""
    from ml import odds as od

    status = refresh_odds(live_dir, _live_season(live_dir, now), fixtures=False, now=now)
    od.ODDS_DIR = odds_dir(live_dir)
    return status


def maybe_retrain(force: bool = False, now: Optional[datetime] = None, live_dir: str = LIVE_DATA_DIR) -> Dict[str, Any]:
    from ml.train_points_v3 import train_and_report

    now = now or datetime.now(timezone.utc)
    out_dir = pm.live_model_dir()
    last = last_trained_at()
    # A failed run (an exception) isn't logged, so the next run tries again
    since = max((t for t in (last, last_attempt_at(out_dir)) if t is not None), default=None)
    if not force and not is_due(now, since):
        return {"status": "skipped", "reason": "not due", "last_trained": last.isoformat() if last else None}

    os.makedirs(out_dir, exist_ok=True)
    cand_model = os.path.join(out_dir, f"points_{VERSION}.candidate.json")
    cand_meta = os.path.join(out_dir, f"points_{VERSION}_meta.candidate.json")
    report_path = os.path.join(out_dir, f"points_{VERSION}_report.json")

    # The served model: v4, or v3 until a v4 exists (then the candidate must stay within 10% of v3)
    active_meta = _read_json(pm.active_model_paths()[1]) or {}
    odds = prepare_odds(live_dir, now)
    report = train_and_report(cand_model, cand_meta, report_path, version=VERSION)
    candidate = report["test"]["all"]
    active_version = active_meta.get("version") or VERSION
    ok, reason = passes_guardrail(candidate, active_meta.get("test_metrics"), VERSION, active_version)

    live_model, live_meta = pm.live_model_paths(VERSION)
    members = int((_read_json(cand_meta) or {}).get("members") or 1)
    candidates = pm.member_paths(cand_model, members)
    if ok:
        # The ensemble's other members, then its first file, then the meta: active_model_paths
        # needs the first file and the meta, and the meta marks completion
        for src, dst in list(zip(candidates, pm.member_paths(live_model, members)))[1:]:
            os.replace(src, dst)
        os.replace(cand_model, live_model)
        os.replace(cand_meta, live_meta)
    else:
        for path in candidates + [cand_meta]:
            if os.path.exists(path):
                os.remove(path)

    entry = {
        "at": now.isoformat(timespec="seconds"), "version": VERSION, "accepted": ok, "reason": reason,
        "rows": report.get("production", {}).get("rows"), "seconds": report.get("seconds"),
        "test_top20": candidate[VERSION]["top20_points"], "test_mae": candidate[VERSION]["mae"],
        "previous_version": active_version,
        "previous_test_top20": (active_meta.get("test_metrics") or {}).get(active_version, {}).get("top20_points"),
        "input_coverage": (report.get("input_coverage") or {}).get("all"), "odds": odds,
    }
    try:
        from pipeline.predict_next_gw import peak_memory_mb
        peak = peak_memory_mb()
        entry["peak_memory_mb"] = round(peak, 1) if peak is not None else None
    except Exception:  # noqa: BLE001 - reporting only
        pass
    history_path = os.path.join(out_dir, "retrain_history.json")
    history = (_read_json(history_path) or [])[-(HISTORY_LIMIT - 1):] + [entry]
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Retrain %s: %s", "accepted" if ok else "rejected", reason)
    return {"status": "accepted" if ok else "rejected", **entry}


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Weekly retraining of the {VERSION} points model")
    parser.add_argument("--force", action="store_true", help="Retrain now regardless of the schedule")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(maybe_retrain(force=args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
