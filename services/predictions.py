"""
Cached access to the precomputed next-gameweek predictions CSV.

The file is parsed once and re-read only when its modification time changes,
so regenerating predictions on disk is picked up without a restart.
"""
import os
from threading import Lock
from typing import Optional, Tuple

import pandas as pd

PREDICTIONS_CSV = "ml/predictions/next_gw_predictions.csv"

_lock = Lock()
_cached: Optional[Tuple[float, pd.DataFrame]] = None


def get_predictions_df(path: str = PREDICTIONS_CSV) -> Optional[pd.DataFrame]:
    """Return the predictions DataFrame, or None if the file doesn't exist.

    Callers must not mutate the returned frame in place (use .assign/.copy).
    """
    global _cached
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    with _lock:
        if _cached is None or _cached[0] != mtime:
            df = pd.read_csv(path)
            df["name_lower"] = df["name"].str.lower()
            _cached = (mtime, df)
        return _cached[1]
