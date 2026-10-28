"""
Train and evaluate the points model (ml/points_model.py): v3 here, v4 with ml/train_points_v4.py,
which calls train_and_report(version="v4").

    python -m ml.train_points_v3              # evaluate, then refit on everything and save
    python -m ml.train_points_v3 --eval-only

Temporal protocol:
  train  2020-21 .. 2023-24
  valid  2024-25             (early stopping)
  test   2025-26 and the current season so far (never seen while choosing anything)
Baselines on the same test rows: FPL's own expected points (xP, where present), the
player's last-3 and last-6 average points, and season-to-date points per game.
The production model is then refit on all seasons with the early-stopped tree count
scaled by the extra data.

v4 builds the same rows with pm.load_extra_inputs (player codes, odds from ODDS_DIR, FPL
snapshots, the status file in LIVE_DATA_DIR) and refuses to train without the odds files, which
aren't committed (python -m scripts.data.fetch_odds). Its report also scores the predictions times
FPL's chance of playing before each deadline ("v4_cop"), as production applies it; the snapshots
cover few gameweeks of 2025-26 and 2026-27, so that's close to "v4" on the test seasons.
"""
import argparse
import json
import logging
import os
import time
from typing import Optional

import pandas as pd

from ml import points_model as pm
from pipeline.current_season import LIVE_DATA_DIR

logger = logging.getLogger(__name__)

REPORT_PATH = os.path.join("ml", "models", "points_v3_report.json")
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
VALID_SEASONS = ["2024-25"]
# v4: share of rows with pre-deadline odds, overall and in every finished season (odds_gaps; about
# 0.87-0.91 per season to 2023-24 and 0.97+ since, with every file)
MIN_ODDS_COVERAGE = 0.5


def report_path_for(version: str) -> str:
    return os.path.join(pm.MODEL_DIR, f"points_{version}_report.json")


def _live_files() -> tuple:
    """(season CSV, status CSV, season) of LIVE_DATA_DIR, from its meta.json; Nones without one."""
    meta_path = os.path.join(LIVE_DATA_DIR, "meta.json")
    if not os.path.exists(meta_path):
        return None, None, None
    meta = json.load(open(meta_path, encoding="utf-8"))
    status = meta["files"].get("status")
    return (os.path.join(LIVE_DATA_DIR, meta["files"]["season"]),
            os.path.join(LIVE_DATA_DIR, status) if status else None, meta["season"])


def _live_csv() -> tuple:
    live_csv, _, season = _live_files()
    return live_csv, season


def input_coverage(feats: pd.DataFrame) -> dict:
    """Per season, the share of rows with each v4 input group (odds, snapshots, transfers, ratings)."""
    cols = {"odds": "odds_win", "snapshots": "snap_age_days", "transfers": "tr_bal_rank", "ratings": "rt_att"}
    have = pd.DataFrame({k: feats[c].notna() for k, c in cols.items() if c in feats})
    out = have.groupby(feats["season"]).mean().round(3).to_dict(orient="index")
    out["all"] = have.mean().round(3).to_dict()
    return out


def odds_gaps(coverage: dict) -> dict:
    """{season: share of rows with odds} of the seasons under MIN_ODDS_COVERAGE, from input_coverage,
    plus "all" when the whole table is. The latest season (the one in progress) is left out, since
    its file can lag at the start of a season; a finished one without odds means a failed download,
    which would otherwise train that season as if no prices existed."""
    seasons = sorted(s for s in coverage if s != "all")
    check = {s: coverage[s] for s in seasons[:-1]}
    check["all"] = coverage["all"]
    return {s: c["odds"] for s, c in check.items() if c.get("odds", 0.0) < MIN_ODDS_COVERAGE}


def load_training_features(version: str = "v3") -> pd.DataFrame:
    """Every finished match with the model's features: history CSV + the live season, and for v4
    its ExtraInputs (ratings need no prior: the rows start at the first season)."""
    live_csv, status_csv, live_season = _live_files()
    history = pm.load_history(live_csv=live_csv, live_season=live_season)
    extras = None
    if version != "v3":
        extras = pm.load_extra_inputs(live_dir=LIVE_DATA_DIR, season=live_season, status_csv=status_csv,
                                      ratings_prior=False)
    feats = pm.build_features(history, extras=extras)
    return feats[feats["total_points"].notna()]


def train_and_report(model_path: Optional[str] = None, meta_path: Optional[str] = None,
                     report_path: Optional[str] = None, eval_only: bool = False, version: str = "v3") -> dict:
    """Evaluate on the held-out seasons, then (unless eval_only) refit on everything and save.

    Returns the report dict; report["test"]["all"][version] holds the combined test metrics.
    Paths default to ml/models/points_{version}.json, _meta.json and _report.json.
    """
    default_model, default_meta = pm.bundled_model_paths(version)
    model_path, meta_path = model_path or default_model, meta_path or default_meta
    report_path = report_path or report_path_for(version)
    cols = pm.feature_columns(version)
    t0 = time.time()
    feats = load_training_features(version)
    logger.info("Feature table: %d rows x %d features in %.0fs", len(feats), len(cols), time.time() - t0)
    coverage = None
    if version != "v3":
        coverage = input_coverage(feats)
        logger.info("Input coverage: %s", coverage)
        gaps = odds_gaps(coverage)
        if gaps:
            raise RuntimeError(f"too few rows with odds ({', '.join(f'{s}: {c:.0%}' for s, c in gaps.items())}): "
                               f"fetch them into ODDS_DIR first (python -m scripts.data.fetch_odds)")
    # Only what training and the report use: the slices below copy the table, and the weekly
    # retrain runs next to the API and Neo4j on a 4 GB VM
    keep = dict.fromkeys(["season", "GW", pm.PLAYER_KEY, "total_points", "xP", "cop_factor"] + cols)
    feats = feats[[c for c in keep if c in feats.columns]]
    # Double gameweeks are summed per player (pm.evaluate groups by name): one key per player
    feats = feats.assign(name=feats[pm.PLAYER_KEY])

    test_seasons = sorted(set(feats["season"]) - set(TRAIN_SEASONS) - set(VALID_SEASONS))
    train = feats[feats["season"].isin(TRAIN_SEASONS)]
    valid = feats[feats["season"].isin(VALID_SEASONS)]
    test = feats[feats["season"].isin(test_seasons)].copy()

    model = pm.fit(train, valid, cols=cols)
    best_trees = int(model.best_iteration) + 1
    pred = f"pred_{version}"
    test[pred] = model.predict(test[cols])
    test["last3"] = test["total_points_r3"]
    test["last6"] = test["total_points_r6"]
    scored = [(version, pred), ("last3_avg", "last3"), ("last6_avg", "last6"), ("season_ppg", "season_ppg")]
    if "cop_factor" in test:
        test[f"{pred}_cop"] = test[pred] * test["cop_factor"]
        scored.insert(1, (f"{version}_cop", f"{pred}_cop"))

    report = {"version": version, "train_seasons": TRAIN_SEASONS, "valid_seasons": VALID_SEASONS,
              "test_seasons": test_seasons, "best_trees": best_trees,
              "rows": {"train": len(train), "valid": len(valid), "test": len(test)}, "test": {}}
    if coverage is not None:
        report["input_coverage"] = coverage
    for season in test_seasons + ["all"]:
        part = test if season == "all" else test[test["season"] == season]
        res = {name: pm.evaluate(part, col).as_dict() for name, col in scored}
        if part["xP"].notna().mean() > 0.5:
            both = part[part["xP"].notna()]
            res["fpl_xP"] = pm.evaluate(both, "xP").as_dict()
            res[f"{version}_same_rows_as_xP"] = pm.evaluate(both, pred).as_dict()
        report["test"][season] = res

    gain = model.get_booster().get_score(importance_type="gain")
    report["top_features"] = sorted(gain.items(), key=lambda kv: -kv[1])[:15]

    if not eval_only:
        # Refit on all seasons; scale trees with the extra data, capped
        n_trees = min(int(best_trees * len(feats) / len(train)), 4000)
        final = pm.fit(feats, None, n_estimators=n_trees, cols=cols)
        meta = {"version": version, "trained_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                "seasons": sorted(feats["season"].unique().tolist()), "rows": len(feats),
                "n_trees": n_trees, "params": pm.XGB_PARAMS, "test_metrics": report["test"].get("all")}
        if coverage is not None:
            meta["input_coverage"] = coverage["all"]
        pm.save_model(final, meta, model_path, meta_path, features=cols)
        report["production"] = {k: v for k, v in meta.items() if k != "test_metrics"}
        logger.info("Saved %s (%d trees, %d rows)", model_path, n_trees, len(feats))

    report["seconds"] = round(time.time() - t0, 1)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report: %s (%.0fs total)", report_path, time.time() - t0)
    return report


def main(version: str = "v3") -> int:
    parser = argparse.ArgumentParser(description=f"Train/evaluate the {version} points model")
    parser.add_argument("--eval-only", action="store_true", help="Don't refit/save the production model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = train_and_report(eval_only=args.eval_only, version=version)
    print(json.dumps(report["test"], indent=2))
    print("top features:", [k for k, _ in report["top_features"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
