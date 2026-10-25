"""
Weekly retraining of the v3 points model, run from pipeline/daily.py.

    python -m pipeline.retrain            # retrain if it's due (Mondays, at most every 5 days)
    python -m pipeline.retrain --force

The model is trained and evaluated with ml.train_points_v3 (held-out test seasons) and
written to LIVE_DATA_DIR/models/ as a candidate. It only replaces the active model when
it passes the guardrail:
  * on the held-out test seasons its top-20 picks beat every simple baseline, and
  * they are no more than MAX_REGRESSION worse than the active model's recorded result.
ml.points_model.active_model_paths() serves the live copy when present, else the bundled
ml/models/points_v3.json. Every attempt is appended to LIVE_DATA_DIR/models/retrain_history.json.

The chance-of-playing-60-minutes model (ml/play60_model.py) is trained in the same run and
promoted on its own check (play60_passes): held-out AUC of at least PLAY60_MIN_AUC and a Brier
score better than always predicting the base rate.
"""
import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from ml import play60_model
from ml import points_model as pm

logger = logging.getLogger(__name__)

RETRAIN_WEEKDAY = 0                 # Monday: the morning run after the weekend's matches are final
MIN_INTERVAL = timedelta(days=5)    # the timer runs twice a day; retrain once per week
MAX_REGRESSION = 0.10               # reject a candidate >10% worse than the active model on top-20 picks
BASELINES = ("last3_avg", "last6_avg", "season_ppg")
PLAY60_MIN_AUC = 0.85               # the play60 model scores 0.94-0.95 on held-out seasons
HISTORY_LIMIT = 52


# ------------------------------------------------------------------ decisions (pure)

def is_due(now: datetime, last_trained: Optional[datetime]) -> bool:
    if now.weekday() != RETRAIN_WEEKDAY:
        return False
    return last_trained is None or now - last_trained >= MIN_INTERVAL


def passes_guardrail(candidate: Dict[str, Any], active: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """candidate/active: test metrics as in report["test"]["all"] ({"v3": {...}, "last3_avg": {...}, ...})."""
    top20 = candidate["v3"]["top20_points"]
    best_baseline = max((candidate[b]["top20_points"] for b in BASELINES if b in candidate), default=float("-inf"))
    if top20 <= best_baseline:
        return False, f"top-20 {top20:.3f} does not beat the best baseline ({best_baseline:.3f})"
    if active and "v3" in active:
        floor = active["v3"]["top20_points"] * (1 - MAX_REGRESSION)
        if top20 < floor:
            return False, f"top-20 {top20:.3f} is more than {MAX_REGRESSION:.0%} below the active model's {active['v3']['top20_points']:.3f}"
    return True, f"top-20 {top20:.3f} (best baseline {best_baseline:.3f})"


def play60_passes(metrics: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """metrics: the play60 model's report["play60"]["test"]["all"] (auc, brier, base_rate_brier)."""
    if not metrics:
        return False, "no play60 test metrics"
    auc, brier, base = metrics.get("auc"), metrics.get("brier"), metrics.get("base_rate_brier")
    if auc is None or not auc >= PLAY60_MIN_AUC:
        return False, f"play60 AUC {auc} is below {PLAY60_MIN_AUC}"
    if brier is None or base is None or not brier < base:
        return False, f"play60 Brier {brier} is no better than the base rate's {base}"
    return True, f"play60 AUC {auc:.3f}, Brier {brier:.4f} (base rate {base:.4f})"


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


def _promote(ok: bool, candidate: Tuple[str, str], live: Tuple[str, str]) -> None:
    """Move a (model, meta) candidate over the live copy, or delete it. Model first, meta
    last: active_model_paths needs both, and the meta marks completion."""
    if ok and all(os.path.exists(p) for p in candidate):
        for src, dst in zip(candidate, live):
            os.replace(src, dst)
        return
    for path in candidate:
        if os.path.exists(path):
            os.remove(path)


def maybe_retrain(force: bool = False, now: Optional[datetime] = None) -> Dict[str, Any]:
    from ml.train_points_v3 import train_and_report

    now = now or datetime.now(timezone.utc)
    last = last_trained_at()
    if not force and not is_due(now, last):
        return {"status": "skipped", "reason": "not due", "last_trained": last.isoformat() if last else None}

    out_dir = pm.live_model_dir()
    os.makedirs(out_dir, exist_ok=True)
    cand_model = os.path.join(out_dir, "points_v3.candidate.json")
    cand_meta = os.path.join(out_dir, "points_v3_meta.candidate.json")
    report_path = os.path.join(out_dir, "points_v3_report.json")

    live60 = tuple(os.path.join(out_dir, os.path.basename(p))
                   for p in (play60_model.MODEL_PATH, play60_model.META_PATH))
    cand60_model, cand60_meta = (p[:-5] + ".candidate.json" for p in live60)

    active_meta = _read_json(pm.active_model_paths()[1]) or {}
    report = train_and_report(cand_model, cand_meta, report_path,
                              play60_model_path=cand60_model, play60_meta_path=cand60_meta)
    candidate = report["test"]["all"]
    ok, reason = passes_guardrail(candidate, active_meta.get("test_metrics"))
    _promote(ok, (cand_model, cand_meta),
             (os.path.join(out_dir, "points_v3.json"), os.path.join(out_dir, "points_v3_meta.json")))
    play60_metrics = ((report.get("play60") or {}).get("test") or {}).get("all")
    ok60, reason60 = play60_passes(play60_metrics)
    _promote(ok60, (cand60_model, cand60_meta), live60)

    entry = {
        "at": now.isoformat(timespec="seconds"), "accepted": ok, "reason": reason,
        "rows": report.get("production", {}).get("rows"), "seconds": report.get("seconds"),
        "test_top20": candidate["v3"]["top20_points"], "test_mae": candidate["v3"]["mae"],
        "previous_test_top20": (active_meta.get("test_metrics") or {}).get("v3", {}).get("top20_points"),
        "play60_accepted": ok60, "play60_reason": reason60, "play60_auc": (play60_metrics or {}).get("auc"),
    }
    history_path = os.path.join(out_dir, "retrain_history.json")
    history = (_read_json(history_path) or [])[-(HISTORY_LIMIT - 1):] + [entry]
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Retrain %s: %s", "accepted" if ok else "rejected", reason)
    return {"status": "accepted" if ok else "rejected", **entry}


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly retraining of the v3 points model")
    parser.add_argument("--force", action="store_true", help="Retrain now regardless of the schedule")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(maybe_retrain(force=args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
