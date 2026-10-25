"""
Train and evaluate the v3 points model (ml/points_model.py).

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

The chance-of-playing-60-minutes model (ml/play60_model.py) is trained and saved alongside,
on the same feature table and split; its test metrics are report["play60"].
"""
import argparse
import json
import logging
import os
import time

import pandas as pd

from ml import play60_model
from ml import points_model as pm
from pipeline.current_season import LIVE_DATA_DIR

logger = logging.getLogger(__name__)

REPORT_PATH = os.path.join("ml", "models", "points_v3_report.json")
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
VALID_SEASONS = ["2024-25"]


def _live_csv() -> tuple:
    meta_path = os.path.join(LIVE_DATA_DIR, "meta.json")
    if not os.path.exists(meta_path):
        return None, None
    meta = json.load(open(meta_path, encoding="utf-8"))
    return os.path.join(LIVE_DATA_DIR, meta["files"]["season"]), meta["season"]


def train_and_report(model_path: str = pm.MODEL_PATH, meta_path: str = pm.META_PATH,
                     report_path: str = REPORT_PATH, eval_only: bool = False,
                     play60_model_path: str = play60_model.MODEL_PATH,
                     play60_meta_path: str = play60_model.META_PATH) -> dict:
    """Evaluate on the held-out seasons, then (unless eval_only) refit on everything and save,
    for the points model and the play60 model.

    Returns the report dict; report["test"]["all"] holds the combined test metrics.
    """
    t0 = time.time()
    live_csv, live_season = _live_csv()
    history = pm.load_history(live_csv=live_csv, live_season=live_season)
    feats = pm.build_features(history)
    feats = feats[feats["total_points"].notna()]
    logger.info("Feature table: %d rows x %d features in %.0fs", len(feats), len(pm.feature_columns()), time.time() - t0)

    test_seasons = sorted(set(feats["season"]) - set(TRAIN_SEASONS) - set(VALID_SEASONS))
    train = feats[feats["season"].isin(TRAIN_SEASONS)]
    valid = feats[feats["season"].isin(VALID_SEASONS)]
    test = feats[feats["season"].isin(test_seasons)].copy()

    model = pm.fit(train, valid)
    best_trees = int(model.best_iteration) + 1
    test["pred_v3"] = model.predict(test[pm.feature_columns()])
    test["last3"] = test["total_points_r3"]
    test["last6"] = test["total_points_r6"]

    report = {"train_seasons": TRAIN_SEASONS, "valid_seasons": VALID_SEASONS, "test_seasons": test_seasons,
              "best_trees": best_trees, "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
              "test": {}}
    for season in test_seasons + ["all"]:
        part = test if season == "all" else test[test["season"] == season]
        res = {name: pm.evaluate(part, col).as_dict()
               for name, col in [("v3", "pred_v3"), ("last3_avg", "last3"), ("last6_avg", "last6"),
                                 ("season_ppg", "season_ppg")]}
        if part["xP"].notna().mean() > 0.5:
            both = part[part["xP"].notna()]
            res["fpl_xP"] = pm.evaluate(both, "xP").as_dict()
            res["v3_same_rows_as_xP"] = pm.evaluate(both, "pred_v3").as_dict()
        report["test"][season] = res

    gain = model.get_booster().get_score(importance_type="gain")
    report["top_features"] = sorted(gain.items(), key=lambda kv: -kv[1])[:15]

    if not eval_only:
        # Refit on all seasons; scale trees with the extra data, capped
        n_trees = min(int(best_trees * len(feats) / len(train)), 4000)
        final = pm.fit(feats, None, n_estimators=n_trees)
        meta = {"version": "v3", "trained_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
                "seasons": sorted(feats["season"].unique().tolist()), "rows": len(feats),
                "n_trees": n_trees, "params": pm.XGB_PARAMS, "test_metrics": report["test"].get("all")}
        pm.save_model(final, meta, model_path, meta_path)
        report["production"] = {k: v for k, v in meta.items() if k != "test_metrics"}
        logger.info("Saved %s (%d trees, %d rows)", model_path, n_trees, len(feats))

    report["play60"] = play60_model.train_and_save(feats, TRAIN_SEASONS, VALID_SEASONS, play60_model_path,
                                                   play60_meta_path, eval_only=eval_only)
    report["seconds"] = round(time.time() - t0, 1)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report: %s (%.0fs total)", report_path, time.time() - t0)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate the v3 points model")
    parser.add_argument("--eval-only", action="store_true", help="Don't refit/save the production model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = train_and_report(eval_only=args.eval_only)
    print(json.dumps(report["test"], indent=2))
    print("top features:", [k for k, _ in report["top_features"]])
    print("play60:", json.dumps(report["play60"]["test"].get("all")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
