"""
Build data/player_codes.csv: FPL's per-season player id (`element`) mapped to its `code`, the
id that stays the same across seasons, for 2020-21 .. 2025-26.

    python -m scripts.data.fetch_player_codes [--out data/player_codes.csv] [--seasons 2020-21,2021-22]

The history CSV keys players by name, but 173 of its 1,980 players appear under more than one
spelling (David Raya Martin / Martín, Lukasz / Łukasz Fabiański), and four names belong to two
players (two Ben Davies). The backtest's code-keyed identity (scripts/analysis/ml_backtest.py)
reads this file; the current season's codes come from the live status file instead. Source:
vaastav/Fantasy-Premier-League's players_raw.csv, the same repository the history CSV comes from.
"""
import argparse
import io
import os
import sys
import urllib.request

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
OUT = os.path.join("data", "player_codes.csv")
COLUMNS = ["season", "element", "code", "first_name", "second_name", "web_name"]


def fetch(seasons) -> pd.DataFrame:
    frames = []
    for season in seasons:
        url = f"{BASE_URL}/{season}/players_raw.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "pitch-me-backtest"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = pd.read_csv(io.BytesIO(resp.read()), usecols=["id", "code", "first_name", "second_name", "web_name"])
        frames.append(raw.rename(columns={"id": "element"}).assign(season=season))
        print(f"{season}: {len(raw)} players")
    out = pd.concat(frames, ignore_index=True)[COLUMNS]
    if out.duplicated(["season", "element"]).any():
        raise ValueError("duplicate (season, element) rows")
    return out.sort_values(["season", "element"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--seasons", default=",".join(SEASONS))
    args = parser.parse_args()
    try:
        codes = fetch(args.seasons.split(","))
    except Exception as exc:  # noqa: BLE001
        print(f"failed: {exc}", file=sys.stderr)
        return 1
    codes.to_csv(args.out, index=False)
    print(f"{len(codes)} rows, {codes['code'].nunique()} players -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
