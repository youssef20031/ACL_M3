"""
Cached access to the precomputed next-gameweek predictions CSV.

Two locations, chosen on every call:
  1. LIVE_DATA_DIR/next_gw_predictions.csv, regenerated from fresh data by
     `python -m pipeline.predict_next_gw` (default data/live);
  2. ml/predictions/next_gw_predictions.csv, the file bundled with the repo, used
     when no live file exists.
Each file is parsed once and re-read only when its path or modification time
changes, so a newly written live file is picked up without a restart.
"""
import os
from threading import Lock
from typing import Optional, Tuple

import pandas as pd

PREDICTIONS_CSV = "ml/predictions/next_gw_predictions.csv"
LIVE_PREDICTIONS_NAME = "next_gw_predictions.csv"

_lock = Lock()
_cached: Optional[Tuple[str, float, pd.DataFrame]] = None


def live_predictions_path() -> str:
    return os.path.join(os.getenv("LIVE_DATA_DIR", os.path.join("data", "live")), LIVE_PREDICTIONS_NAME)


def predictions_path() -> str:
    """The live predictions file when it exists, else the bundled one."""
    live = live_predictions_path()
    return live if os.path.exists(live) else PREDICTIONS_CSV


def get_predictions_df(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Return the predictions DataFrame, or None if the file doesn't exist.

    `path` defaults to predictions_path(). Callers must not mutate the returned
    frame in place (use .assign/.copy).
    """
    global _cached
    path = path or predictions_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    with _lock:
        if _cached is None or _cached[0] != path or _cached[1] != mtime:
            df = pd.read_csv(path)
            df["name_lower"] = df["name"].str.lower()
            _cached = (path, mtime, df)
        return _cached[2]
