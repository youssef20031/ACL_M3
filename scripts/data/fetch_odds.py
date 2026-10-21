"""
Download football-data.co.uk's Premier League results-and-odds files (the inputs of ml/odds.py)
into data/odds/E0_{yyyy}.csv, one per season from 2020-21 to the current one.

    python -m scripts.data.fetch_odds [--out-dir data/odds] [--seasons 2020-21,2021-22]

football-data.co.uk's data is free for private, non-commercial use and not licensed for
redistribution, so data/odds/ is gitignored: the ML backtest workflow runs this before building
features. Seven small files, fetched once per run.
"""
import argparse
import datetime as dt
import os
import sys
import urllib.request

from ml import odds

FIRST_SEASON = 2020


def default_seasons():
    """2020-21 .. the season in progress (a new one from July)."""
    today = dt.date.today()
    last = today.year if today.month >= 7 else today.year - 1
    return [f"{y}-{str(y + 1)[2:]}" for y in range(FIRST_SEASON, last + 1)]


def fetch(seasons, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    failed = 0
    for season in seasons:
        code = odds.season_code(season)
        url = f"{odds.BASE_URL}/{code}/E0.csv"
        path = os.path.join(out_dir, f"E0_{code}.csv")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pitch-me-backtest"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
            with open(path, "wb") as f:
                f.write(body)
            print(f"{season}: {len(odds.load_odds(out_dir).query('season == @season'))} matches -> {path}")
        except Exception as exc:  # noqa: BLE001 - report every season, then fail
            failed += 1
            print(f"{season}: {url} failed: {exc}", file=sys.stderr)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out-dir", default=odds.ODDS_DIR)
    parser.add_argument("--seasons", default=",".join(default_seasons()))
    args = parser.parse_args()
    return 1 if fetch(args.seasons.split(","), args.out_dir) else 0


if __name__ == "__main__":
    raise SystemExit(main())
