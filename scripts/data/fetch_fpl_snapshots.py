"""
Build data/fpl_snapshots.csv: every player's FPL status, chance of playing, team-news flag and
set-piece orders as they stood at each commit of vaastav's players_raw.csv, 2020-21 .. the current
season.

    python -m scripts.data.fetch_fpl_snapshots [--out data/fpl_snapshots.csv] [--seasons 2020-21,2021-22]
                                               [--raw-dir DIR]

The FPL API only shows current values, and our own pre-deadline archive (pipeline.current_season's
element_snapshots) starts in 2026-27. vaastav/Fantasy-Premier-League commits
data/<season>/players_raw.csv (FPL's bootstrap players) about once a gameweek, usually a day or
two after the gameweek ends, so its git history is a series of snapshots in between deadlines:
about 40 per season to 2024-25, 12 in 2025-26. ml/snapshots.py maps each gameweek to the latest one
captured between the previous deadline and its own, by commit time (UTC). The current season has
had a few commits too (four by 2026-10-27).

Columns: season, captured_at (commit time), sha (short), element, code, status (a/d/i/s/u/n),
chance_of_playing_next_round, news_flag (1 when the news text is non-empty; the text is left out
to keep the file small), penalties_order, corners_and_indirect_freekicks_order,
direct_freekicks_order.

Source: github.com/vaastav/Fantasy-Premier-League (MIT licence), the repository the history CSV
and data/player_codes.csv come from. The licence would allow committing this extract, but at 8 MB
(over 200 snapshots of 600-900 players) it's gitignored instead: the ML backtest workflow runs this
script for the variants that use it (about 40 s). The commit list comes from the GitHub API, with
GH_TOKEN or GITHUB_TOKEN when set (CI) and unauthenticated otherwise (60 requests an hour, about 7
needed); the files come from raw.githubusercontent.com. --raw-dir keeps each downloaded file, so a
rerun only fetches new commits.
"""
import argparse
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

REPO = "vaastav/Fantasy-Premier-League"
API_URL = f"https://api.github.com/repos/{REPO}/commits"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}"
FIRST_SEASON = 2020
OUT = os.path.join("data", "fpl_snapshots.csv")
ORDERS = ["penalties_order", "corners_and_indirect_freekicks_order", "direct_freekicks_order"]
RAW_COLUMNS = ["id", "code", "status", "chance_of_playing_next_round", "news"] + ORDERS
COLUMNS = ["season", "captured_at", "sha", "element", "code", "status", "chance_of_playing_next_round",
           "news_flag"] + ORDERS
WORKERS = 8
ATTEMPTS = 3


def default_seasons():
    """2020-21 .. the season in progress (a new one from July)."""
    today = dt.date.today()
    last = today.year if today.month >= 7 else today.year - 1
    return [f"{y}-{str(y + 1)[2:]}" for y in range(FIRST_SEASON, last + 1)]


def _get(url: str, api: bool = False) -> bytes:
    headers = {"User-Agent": "pitch-me-backtest"}
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if api:
        headers["Accept"] = "application/vnd.github+json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    # A few retries: one failed file out of ~240 fails the whole fetch (and the CI run's variants)
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except OSError:   # URLError, HTTPError, timeouts, resets
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)


def list_commits(season: str) -> list:
    """[(full sha, committer time)] of every commit that changed the season's players_raw.csv."""
    out, page = [], 1
    while True:
        url = f"{API_URL}?path=data/{season}/players_raw.csv&per_page=100&page={page}"
        batch = json.loads(_get(url, api=True))
        out += [(c["sha"], c["commit"]["committer"]["date"]) for c in batch]
        if len(batch) < 100:
            return out
        page += 1


def download(season: str, sha: str, raw_dir: str = None) -> bytes:
    path = os.path.join(raw_dir, season, f"{sha[:7]}.csv") if raw_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    body = _get(f"{RAW_URL}/{sha}/data/{season}/players_raw.csv")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(body)
    return body


def compact(body: bytes, season: str, sha: str, captured_at: str) -> pd.DataFrame:
    """COLUMNS from one players_raw.csv; columns a file lacks (set-piece orders before
    September 2020) are left empty."""
    raw = pd.read_csv(io.BytesIO(body), usecols=lambda c: c in RAW_COLUMNS, dtype={"news": str},
                      encoding="utf-8")
    raw = raw.reindex(columns=RAW_COLUMNS)
    out = pd.DataFrame({"season": season, "captured_at": captured_at, "sha": sha[:7],
                        "element": raw["id"].astype(int), "code": raw["code"].astype(int),
                        "status": raw["status"].fillna("").astype(str).str.strip().str.lower()})
    for c in ["chance_of_playing_next_round"] + ORDERS:
        out[c] = pd.to_numeric(raw[c], errors="coerce").astype("Int64")
    out["news_flag"] = raw["news"].fillna("").str.strip().ne("").astype(int)
    return out[COLUMNS]


def fetch(seasons, raw_dir: str = None) -> pd.DataFrame:
    frames, failed = [], 0
    for season in seasons:
        commits = list_commits(season)

        def one(c):
            sha, at = c
            try:
                return compact(download(season, sha, raw_dir), season, sha, at)
            except Exception as exc:  # noqa: BLE001 - report every commit, then fail
                print(f"{season} {sha[:7]}: failed: {exc}", file=sys.stderr)
                return None

        with ThreadPoolExecutor(WORKERS) as pool:
            got = list(pool.map(one, commits))
        failed += sum(g is None for g in got)
        got = [g for g in got if g is not None]
        frames += got
        times = sorted(c[1] for c in commits)
        print(f"{season}: {len(got)} of {len(commits)} snapshots ({times[0] if times else '-'} .. "
              f"{times[-1] if times else '-'}), {sum(map(len, got))} rows")
    if failed:
        raise RuntimeError(f"{failed} snapshots failed")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["season", "sha", "element"])
    return out.sort_values(["season", "captured_at", "element"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--seasons", default=",".join(default_seasons()))
    parser.add_argument("--raw-dir", default=None, help="keep (and reuse) the downloaded files here")
    args = parser.parse_args()
    try:
        snaps = fetch(args.seasons.split(","), args.raw_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"failed: {exc}", file=sys.stderr)
        return 1
    snaps.to_csv(args.out, index=False)
    print(f"{len(snaps)} rows, {snaps['sha'].nunique()} snapshots -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
