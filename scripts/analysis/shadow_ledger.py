"""
Live model vs shadow model, gameweek by gameweek: the points each model's saved squad pick
scored under FPL's rules (services/model_review.shadow_comparison: auto-subs, captain and vice),
their difference (live - shadow), its running total, and a paired bootstrap CI of the mean
difference over the finished gameweeks.

    python -m scripts.analysis.shadow_ledger [--live-dir data/live] [--season 2026-27] [--json]

On the VM (scripts/ isn't in the backend image, so mount it read-only into a one-off container):

    docker compose -f deploy/docker-compose.yml run --rm --no-deps \\
        -v "$PWD/scripts:/app/scripts:ro" backend python -m scripts.analysis.shadow_ledger

Reads only the saved picks (LIVE_DATA_DIR/model_picks/<season>/ and its shadow/ folder), the
live season CSV and meta.json: no network and no model. pipeline.predict_next_gw saves both picks
on the same run before each deadline, from the same feature rows and with the same optimizer, so
each gameweek is out of sample for both models and was chosen by nobody. Since 2026-10-28 the
live model is v4 and the shadow v3; a gameweek where both picks came from the same model (no v4
file yet), or from different runs (one of them failed to save on the last run before the deadline),
is listed but left out of the totals; a gameweek with only one of the two picks is named below them.
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

N_BOOT = 10000
LEVEL = 0.95


def counted(row: Dict[str, Any]) -> bool:
    """Whether a gameweek counts towards the totals: finished, two different models, and both picks
    from the same prediction run (not when one failed to save on the last run before the deadline,
    leaving an earlier run's pick that saw less team news and fewer odds)."""
    same = row.get("live_model") is not None and row.get("live_model") == row.get("shadow_model")
    return row["status"] == "finished" and not same and row.get("same_run") is not False


def ledger(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The comparison rows plus `running`, the cumulative difference over counted gameweeks
    (None on the others)."""
    out, total = [], 0
    for row in rows:
        if counted(row):
            total += row["difference"]
            out.append({**row, "running": total})
        else:
            out.append({**row, "running": None})
    return out


def unpaired(live_gws, shadow_gws) -> List[int]:
    """Gameweeks since the first shadow pick with only one of the two picks (the other model failed to
    load or to save on every run before that deadline): not compared, but listed so a gap shows."""
    live_gws, shadow_gws = set(live_gws), set(shadow_gws)
    if not shadow_gws:
        return []
    first = min(shadow_gws)
    return sorted(g for g in live_gws ^ shadow_gws if g >= first)


def bootstrap_ci(diffs, n_boot: int = N_BOOT, level: float = LEVEL, seed: int = 0) -> Dict[str, Any]:
    """Mean of the per-gameweek differences with a percentile bootstrap CI, resampling gameweeks
    (each keeps its live and shadow scores together, so the comparison stays paired), and the
    share of resamples whose mean favours the live model."""
    d = np.asarray(list(diffs), dtype=float)
    n = len(d)
    out: Dict[str, Any] = {"gameweeks": n, "total": float(d.sum()) if n else 0.0,
                           "mean": float(d.mean()) if n else None, "lo": None, "hi": None, "p_live_ahead": None}
    if n < 2:
        return out
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    alpha = (1 - level) / 2
    out["lo"], out["hi"] = (float(x) for x in np.quantile(means, [alpha, 1 - alpha]))
    out["p_live_ahead"] = float((means > 0).mean())
    return out


def _fmt(value: Optional[float], signed: bool = False, digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def format_table(rows: List[Dict[str, Any]], summary: Dict[str, Any], missing: Optional[List[int]] = None) -> str:
    header = f"{'GW':>3}  {'status':<8}  {'live':>5}  {'shadow':>6}  {'diff':>5}  {'running':>7}  {'shared':>6}  models"
    lines = [header, "-" * len(header)]
    for r in rows:
        models = f"{r.get('live_model') or '?'} vs {r.get('shadow_model') or '?'}"
        if counted(r):
            note = ""
        elif r["status"] != "finished":
            note = "  (provisional)"
        else:
            note = "  (different runs, not counted)" if r.get("same_run") is False else "  (not counted)"
        lines.append(f"{r['gameweek']:>3}  {r['status']:<8}  {r['live_points']:>5}  {r['shadow_points']:>6}  "
                     f"{_fmt(r['difference'], True):>5}  {_fmt(r['running'], True):>7}  {r['shared_players']:>6}  "
                     f"{models}{note}")
    if not rows:
        lines.append("(no gameweek with both a live and a shadow pick whose deadline has passed)")
    if missing:
        lines.append(f"Only one of the two picks, not compared: GW {', '.join(str(g) for g in missing)}")
    s = summary
    lines.append("")
    if s["gameweeks"] == 0:
        lines.append("No finished gameweek with picks from two different models yet.")
    else:
        line = (f"{s['gameweeks']} finished gameweeks: live - shadow = {_fmt(s['total'], True)} points in total, "
                f"{_fmt(s['mean'], True, 2)} per gameweek")
        if s["lo"] is not None:
            line += (f", {LEVEL:.0%} CI [{_fmt(s['lo'], True, 2)}, {_fmt(s['hi'], True, 2)}] "
                     f"(paired bootstrap over gameweeks, {N_BOOT} resamples); "
                     f"live ahead in {s['p_live_ahead']:.0%} of resamples")
        lines.append(line + ".")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Live vs shadow model picks, gameweek by gameweek")
    parser.add_argument("--live-dir", help="LIVE_DATA_DIR (default: the environment's, else data/live)")
    parser.add_argument("--season", help="Season label, e.g. 2026-27 (default: the current season)")
    parser.add_argument("--json", action="store_true", help="Print the rows and summary as JSON")
    args = parser.parse_args(argv)
    if args.live_dir:
        os.environ["LIVE_DATA_DIR"] = args.live_dir

    from config.settings import CURRENT_SEASON
    from services.model_review import SHADOW, reviewable_gameweeks, shadow_comparison

    season = args.season or CURRENT_SEASON
    rows = ledger(shadow_comparison(season))
    missing = unpaired(reviewable_gameweeks(season), reviewable_gameweeks(season, variant=SHADOW))
    summary = bootstrap_ci(r["difference"] for r in rows if counted(r))
    if args.json:
        print(json.dumps({"rows": rows, "summary": summary, "unpaired": missing}, indent=2))
    else:
        print(format_table(rows, summary, missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
