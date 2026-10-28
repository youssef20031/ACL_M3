"""
Train and evaluate the v4 points model, the production model since 2026-10-28 (v3 is kept as a
shadow): ml/train_points_v3.train_and_report with version="v4", same protocol and baselines.

    python -m scripts.data.fetch_odds         # the odds files aren't committed
    python -m ml.train_points_v4              # evaluate, then refit on everything and save
    python -m ml.train_points_v4 --eval-only

Writes ml/models/points_v4.json, points_v4_meta.json and points_v4_report.json. Inputs: the
history CSV, LIVE_DATA_DIR (the live season, its status file and our element_snapshots archive),
data/player_codes.csv, data/fpl_snapshots.csv and the football-data.co.uk files in ODDS_DIR.
"""
from ml.train_points_v3 import main, train_and_report  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main("v4"))
