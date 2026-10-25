"""
Chance of playing 60+ minutes in a fixture ("play60"): the risk input of the safer team of the
week (services/team_optimizer.best_squad(safer=True)).

60 minutes is what earns the 2 appearance points and a goalkeeper's or defender's clean sheet,
so it is what "nailed" means in FPL. The model reuses the points model's rows and features
(ml.points_model.build_features / feature_columns: earlier gameweeks only) with the target
minutes >= 60: XGBoost binary:logistic, early-stopped on the validation season, then refit on
every season with the tree count scaled by the extra rows, as ml.train_points_v3 does for points.

Trained by ml.train_points_v3 (so by the ML train workflow and the weekly retrain), or on its
own with `python -m ml.play60_model`. Measured on held-out seasons 2022-23 .. 2026-27, each
model trained on earlier seasons only: AUC 0.94-0.95, Brier 0.08-0.09 against 0.19-0.22 for
always predicting the base rate, and within two points of the actual rate in every probability
band. Even players who started each of their last 6 matches played 60+ only 85% of the time.

Not to be confused with ml/minutes_model.py, a legacy v2 model built on FeatureEngineer.
"""
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from ml import points_model as pm

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join("ml", "models", "play60_v1.json")
META_PATH = os.path.join("ml", "models", "play60_v1_meta.json")
PARAMS = dict(
    objective="binary:logistic", tree_method="hist", max_depth=5, learning_rate=0.05,
    min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss",
)


def target(df: pd.DataFrame) -> pd.Series:
    return (df["minutes"] >= 60).astype(int)


def active_model_paths() -> Tuple[str, str]:
    """(model, meta) to serve: the retrained copy in LIVE_DATA_DIR/models if present, else bundled."""
    live = tuple(os.path.join(pm.live_model_dir(), os.path.basename(p)) for p in (MODEL_PATH, META_PATH))
    return live if all(os.path.exists(p) for p in live) else (MODEL_PATH, META_PATH)


def fit(train: pd.DataFrame, valid: Optional[pd.DataFrame] = None, n_estimators: int = 3000):
    xgb = pm._xgb()
    cols = pm.feature_columns()
    model = xgb.XGBClassifier(n_estimators=n_estimators, early_stopping_rounds=100 if valid is not None else None,
                              random_state=0, **PARAMS)
    eval_set = [(valid[cols], target(valid))] if valid is not None else None
    model.fit(train[cols], target(train), eval_set=eval_set, verbose=False)
    return model


def evaluate(y: Iterable[float], p: Iterable[float]) -> Dict[str, Any]:
    """AUC (rank-based, ties averaged), Brier score, and the Brier score of always predicting
    the base rate."""
    y = np.asarray(list(y), dtype=float)
    p = np.asarray(list(p), dtype=float)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    ranks = pd.Series(p).rank(method="average").to_numpy()
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if n_pos and n_neg else float("nan")
    rate = float(y.mean()) if len(y) else float("nan")
    return {"rows": int(len(y)), "auc": round(float(auc), 4), "brier": round(float(np.mean((p - y) ** 2)), 4),
            "base_rate_brier": round(float(np.mean((rate - y) ** 2)), 4), "rate": round(rate, 4)}


def train_and_save(feats: pd.DataFrame, train_seasons: Iterable[str], valid_seasons: Iterable[str],
                   model_path: str = MODEL_PATH, meta_path: str = META_PATH, eval_only: bool = False) -> Dict[str, Any]:
    """Evaluate on the seasons outside train/valid, then (unless eval_only) refit on everything
    and save. `feats` is ml.points_model.build_features output with outcomes."""
    t0 = time.time()
    train_seasons, valid_seasons = list(train_seasons), list(valid_seasons)
    feats = feats[feats["minutes"].notna()]
    cols = pm.feature_columns()
    test_seasons = sorted(set(feats["season"]) - set(train_seasons) - set(valid_seasons))
    train = feats[feats["season"].isin(train_seasons)]
    model = fit(train, feats[feats["season"].isin(valid_seasons)])
    best_trees = int(model.best_iteration) + 1
    report: Dict[str, Any] = {"train_seasons": train_seasons, "valid_seasons": valid_seasons,
                              "test_seasons": test_seasons, "best_trees": best_trees, "test": {}}
    test = feats[feats["season"].isin(test_seasons)]
    if len(test):
        prob = model.predict_proba(test[cols])[:, 1]
        for season in test_seasons + ["all"]:
            mask = np.ones(len(test), dtype=bool) if season == "all" else (test["season"] == season).to_numpy()
            report["test"][season] = evaluate(target(test)[mask], prob[mask])

    if not eval_only:
        n_trees = min(int(best_trees * len(feats) / len(train)), 4000)
        final = fit(feats, None, n_estimators=n_trees)
        meta = {"version": "play60_v1", "target": "minutes >= 60",
                "trained_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                "seasons": sorted(feats["season"].unique().tolist()), "rows": len(feats), "n_trees": n_trees,
                "params": PARAMS, "test_metrics": report["test"].get("all"), "features": cols}
        final.get_booster().save_model(model_path)
        with open(meta_path, "w", encoding="utf-8") as f:   # meta last: it marks the model complete
            json.dump(meta, f, indent=2)
        report["production"] = {k: v for k, v in meta.items() if k not in ("test_metrics", "features")}
        logger.info("Saved %s (%d trees, %d rows)", model_path, n_trees, len(feats))
    report["seconds"] = round(time.time() - t0, 1)
    return report


def load_model(model_path: Optional[str] = None, meta_path: Optional[str] = None):
    if model_path is None or meta_path is None:
        model_path, meta_path = active_model_paths()
    xgb = pm._xgb()
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return booster, meta


def predict(booster, meta: dict, features: pd.DataFrame) -> np.ndarray:
    xgb = pm._xgb()
    return booster.predict(xgb.DMatrix(features[meta["features"]]))


def main() -> int:
    """Train and save only this model (ml.train_points_v3 trains it together with the points model)."""
    from ml.train_points_v3 import TRAIN_SEASONS, VALID_SEASONS, _live_csv

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    live_csv, live_season = _live_csv()
    feats = pm.build_features(pm.load_history(live_csv=live_csv, live_season=live_season))
    report = train_and_save(feats[feats["total_points"].notna()], TRAIN_SEASONS, VALID_SEASONS)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
