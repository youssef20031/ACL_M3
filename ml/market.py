"""
Candidate transfer-market features for the v3 points model (backtest only, 2026-10;
scripts/analysis/ml_backtest.py variant transfers): how FPL managers moved before the deadline.
Net transfers react to team news (injuries, suspensions, a dropped starter) that the match history
can't see, so they might cut the share of the model's picks who don't play.

  tr_bal_rank, tr_in_rank, tr_out_rank  percentile of the gameweek's net transfers, transfers in
                                        and transfers out among that gameweek's players. Ranks,
                                        because serving would see the counts a day or so before
                                        the deadline rather than the final ones.
  price_change_1, price_change_3        price now minus the price at the player's previous match,
                                        and minus the mean of his last 3 (this season's only)

Pre-deadline check: in the historical files a gameweek's net-transfer rank correlates with the
previous gameweek's points (Spearman 0.18-0.23 per season) more than with that gameweek's
(0.09-0.14) or the next one's (0.04-0.07), as expected of transfers made before the deadline in
reaction to the last match. Serving would need the next gameweek's transfers_in_event and
transfers_out_event from FPL's bootstrap, which pipeline/current_season.py doesn't store yet.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd

from ml import points_model as pm

TRANSFER_COLUMNS = ["transfers_balance", "transfers_in", "transfers_out"]
FEATURES = ["tr_bal_rank", "tr_in_rank", "tr_out_rank", "price_change_1", "price_change_3"]


def load_transfers(history_csv: str = pm.HISTORY_CSV, live_csv: Optional[str] = None,
                   live_season: Optional[str] = None) -> pd.DataFrame:
    """(season, name, fixture) -> transfer counts of that gameweek, history plus the live season."""
    keys = ["season_x", "name", "fixture"]
    frames = [pd.read_csv(history_csv, usecols=keys + TRANSFER_COLUMNS, low_memory=False)
              .rename(columns={"season_x": "season"})]
    if live_csv and os.path.exists(live_csv):
        live = pd.read_csv(live_csv, usecols=["name", "fixture"] + TRANSFER_COLUMNS)
        frames.append(live.assign(season=live_season))
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["season", "name", "fixture"])


def transfer_features(df: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """FEATURES for every row of a pm.build_features table (NaN where counts are missing, e.g.
    upcoming rows unless the caller adds them to `transfers`)."""
    d = df[["season", "name", "fixture", "GW"]].merge(transfers, on=["season", "name", "fixture"], how="left")
    d.index = df.index
    # One value per player-gameweek (a double gameweek's two rows carry the same counts)
    per_gw = d.dropna(subset=TRANSFER_COLUMNS).drop_duplicates(["season", "GW", "name"])
    ranks = per_gw.groupby(["season", "GW"])[TRANSFER_COLUMNS].rank(pct=True)
    ranks.columns = ["tr_bal_rank", "tr_in_rank", "tr_out_rank"]
    ranks[["season", "GW", "name"]] = per_gw[["season", "GW", "name"]]
    out = d[["season", "GW", "name"]].merge(ranks, on=["season", "GW", "name"], how="left")
    out.index = df.index
    out = out[["tr_bal_rank", "tr_in_rank", "tr_out_rank"]]

    prev = pm._rolling_before_gameweek(df, "name", ["value"], (1, 3)).reindex(df.index)
    played = df["season_matches_before"]
    out["price_change_1"] = ((df["value"] - prev["value_r1"]) / 10.0).where(played >= 1)
    out["price_change_3"] = ((df["value"] - prev["value_r3"]) / 10.0).where(played >= 3)
    return out.astype(float)
