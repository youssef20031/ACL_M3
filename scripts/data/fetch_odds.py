"""
Download football-data.co.uk's Premier League results-and-odds files (the inputs of ml/odds.py)
into ODDS_DIR/E0_{yyyy}.csv, one per season from 2020-21 to the current one, plus the upcoming
matches' prices (fixtures.csv, Premier League rows only), which the v4 model's upcoming rows use.

    python -m scripts.data.fetch_odds [--out-dir data/odds] [--seasons 2020-21,2021-22] [--no-fixtures]

ODDS_DIR is data/odds unless the ODDS_DIR environment variable is set (production keeps the files
in $LIVE_DATA_DIR/odds). football-data.co.uk's data is free for private, non-commercial use and
not licensed for redistribution, so data/odds/ is gitignored: the ML workflows run this before
building features. Eight small files, fetched once per run.
"""
import argparse
import datetime as dt
import sys

import pandas as pd

from ml import odds

FIRST_SEASON = 2020


def default_seasons():
    """2020-21 .. the season in progress (a new one from July)."""
    today = dt.date.today()
    last = today.year if today.month >= 7 else today.year - 1
    return [f"{y}-{str(y + 1)[2:]}" for y in range(FIRST_SEASON, last + 1)]


def fetch(seasons, out_dir: str, fixtures: bool = True) -> int:
    failed = 0
    for season in seasons:
        try:
            path = odds.fetch_season(season, out_dir)
            print(f"{season}: {len(odds.load_odds(out_dir).query('season == @season'))} matches -> {path}")
        except Exception as exc:  # noqa: BLE001 - report every season, then fail
            failed += 1
            print(f"{season}: {odds.BASE_URL}/{odds.season_code(season)}/E0.csv failed: {exc}", file=sys.stderr)
    if fixtures:
        try:
            path = odds.fetch_fixtures(out_dir)
            upcoming = odds.parse_fixtures(pd.read_csv(path))
            print(f"fixtures: {len(upcoming)} upcoming matches, {int(upcoming['p_home'].notna().sum())} "
                  f"with prices -> {path}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"fixtures: {odds.FIXTURES_URL} failed: {exc}", file=sys.stderr)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out-dir", default=odds.ODDS_DIR)
    parser.add_argument("--seasons", default=",".join(default_seasons()))
    parser.add_argument("--no-fixtures", action="store_true", help="Skip the upcoming matches (fixtures.csv)")
    args = parser.parse_args()
    seasons = [s for s in args.seasons.split(",") if s]
    return 1 if fetch(seasons, args.out_dir, fixtures=not args.no_fixtures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
