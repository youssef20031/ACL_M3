"""
Pre-match bookmaker odds as candidate team features for the v3 points model: each team's
implied chance of winning, expected goals for and against, and clean-sheet probability.

Not used by the production model. The ML backtest (scripts/analysis/ml_backtest.py, variants
odds and odds_close) found top-20 gains that were positive in every protocol but reliable only
under refit, and no reliable captain gain (2026-10, see CLAUDE.md). Serving would also need
upcoming fixtures' prices before each deadline.

Source: football-data.co.uk's Premier League files (mmz4281/{yyyy}/E0.csv, one per season, with
the current season updated twice a week). The data is free for private, non-commercial use and
isn't licensed for redistribution, so the CSVs are not committed: scripts/data/fetch_odds.py
downloads them into ODDS_DIR (gitignored), and the ML backtest workflow does that in CI.

Each match has two sets of prices (market averages, Bet365 where an average is missing):
  pre-closing  collected on Friday afternoon (by 17:00 UK time) for Friday-Monday matches and on
               Tuesday (by 13:00) for Tuesday-Thursday matches; FEATURES
  closing      at kickoff, so after the FPL deadline and after lineups are out; CLOSING_FEATURES
               are an upper bound only and can't be served

Leakage rule (tests/test_points_model.py): a pre-closing price counts only if its collection time
(collected_at) is no later than the FPL deadline, 90 minutes before the gameweek's first kickoff.
In a normal gameweek it always is. A midweek match inside a weekend gameweek (a rearranged match
in a double gameweek) gets NaN, which is also all that serving would have at the deadline.

Goals come from a Poisson fit: the total from the over/under 2.5 price, the split between the two
teams from the 1X2 prices (P(home) - P(away)), each with the bookmaker margin removed.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd

ODDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "odds")
BASE_URL = "https://www.football-data.co.uk/mmz4281"

# football-data.co.uk name -> graph (FPL) name; the rest are the same
FD_TO_GRAPH = {
    "Man United": "Man Utd", "Tottenham": "Spurs", "Sheffield United": "Sheffield Utd",
    "Hull": "Hull City", "Coventry": "Coventry City",
}

# Price columns: (home, draw, away, over 2.5, under 2.5), first choice then fallback
PRE_CLOSING = [("AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"), ("B365H", "B365D", "B365A", "B365>2.5", "B365<2.5")]
CLOSING = [("AvgCH", "AvgCD", "AvgCA", "AvgC>2.5", "AvgC<2.5"),
           ("B365CH", "B365CD", "B365CA", "B365C>2.5", "B365C<2.5")]

# When the pre-closing prices are collected (football-data.co.uk/matches.php): weekday of the
# match (Mon=0) -> (days back to the collection day, latest collection hour, UK time)
COLLECTION = {0: (3, 17), 1: (0, 13), 2: (1, 13), 3: (2, 13), 4: (0, 17), 5: (1, 17), 6: (2, 17)}
DEADLINE_MINUTES = 90
UK = "Europe/London"

FEATURES = ["odds_win", "odds_draw", "odds_gf", "odds_ga", "odds_cs"]
CLOSING_FEATURES = [f"{c}_c" for c in FEATURES]

MAX_GOALS = 15
_K = np.arange(MAX_GOALS + 1)
_LOG_FACT = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, MAX_GOALS + 1)))])


def season_code(season: str) -> str:
    """'2020-21' -> '2021', the folder name in football-data.co.uk's URLs."""
    return season[2:4] + season[5:7]


def graph_team(name: str) -> str:
    return FD_TO_GRAPH.get(name, name)


def parse_odds(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per match from a football-data.co.uk file (a season's E0.csv, or fixtures.csv
    filtered to Div E0): season, home, away, date, and per price set the margin-free
    probabilities p_home/p_draw/p_away/p_over (suffix _c for closing)."""
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    out = pd.DataFrame({"season": season, "home": raw["HomeTeam"].map(graph_team).to_numpy(),
                        "away": raw["AwayTeam"].map(graph_team).to_numpy(),
                        "date": pd.to_datetime(raw["Date"], dayfirst=True).dt.date.to_numpy()})
    for suffix, sets in (("", PRE_CLOSING), ("_c", CLOSING)):
        prices = np.full((len(raw), 5), np.nan)
        for cols in reversed(sets):   # the first choice wins where it has a price
            got = raw.reindex(columns=list(cols)).apply(pd.to_numeric, errors="coerce").to_numpy()
            ok = np.isfinite(got) & (got > 1.0)
            prices = np.where(ok, got, prices)
        inv = 1.0 / prices
        x12 = inv[:, :3] / inv[:, :3].sum(axis=1, keepdims=True)
        out[f"p_home{suffix}"], out[f"p_draw{suffix}"], out[f"p_away{suffix}"] = x12.T
        out[f"p_over{suffix}"] = inv[:, 3] / (inv[:, 3] + inv[:, 4])
    return out.drop_duplicates(["season", "home", "away"], keep="last").reset_index(drop=True)


def load_odds(odds_dir: str = ODDS_DIR) -> pd.DataFrame:
    """Every season file in `odds_dir` (E0_2021.csv, ...; see scripts/data/fetch_odds.py)."""
    frames = []
    if os.path.isdir(odds_dir):
        for fname in sorted(os.listdir(odds_dir)):
            if fname.startswith("E0_") and fname.endswith(".csv"):
                code = fname[3:7]
                season = f"20{code[:2]}-{code[2:]}"
                frames.append(parse_odds(pd.read_csv(os.path.join(odds_dir, fname), encoding="utf-8-sig"), season))
    if not frames:
        return parse_odds(pd.DataFrame(columns=["HomeTeam", "AwayTeam", "Date"]), "")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- Poisson fit

def _pmf(lam: np.ndarray) -> np.ndarray:
    lam = np.maximum(lam, 1e-9)[:, None]
    return np.exp(_K * np.log(lam) - lam - _LOG_FACT)


def _home_minus_away(lam_h: np.ndarray, lam_a: np.ndarray) -> np.ndarray:
    ph, pa = _pmf(lam_h), _pmf(lam_a)
    cdf_h, cdf_a = np.cumsum(ph, axis=1), np.cumsum(pa, axis=1)
    # P(home > away) = sum_i P(h = i) P(a <= i - 1), and the same the other way round
    win_h = (ph[:, 1:] * cdf_a[:, :-1]).sum(axis=1)
    win_a = (pa[:, 1:] * cdf_h[:, :-1]).sum(axis=1)
    return win_h - win_a


def _bisect(f, lo: np.ndarray, hi: np.ndarray, target: np.ndarray, iters: int = 60) -> np.ndarray:
    """x in [lo, hi] with f(x) = target, for f increasing; clipped to the ends."""
    lo, hi = lo.copy(), hi.copy()
    for _ in range(iters):
        mid = (lo + hi) / 2
        below = f(mid) < target
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


def fit_goals(p_home, p_away, p_over):
    """Expected goals (home, away) of independent Poisson scores whose total matches P(over 2.5)
    and whose P(home win) - P(away win) matches the 1X2 prices. NaN where a price is missing."""
    p_home, p_away, p_over = (np.asarray(x, dtype=float) for x in (p_home, p_away, p_over))
    ok = np.isfinite(p_home) & np.isfinite(p_away) & np.isfinite(p_over)
    lam_h, lam_a = np.full(len(p_home), np.nan), np.full(len(p_home), np.nan)
    if not ok.any():
        return lam_h, lam_a
    over, diff = p_over[ok], (p_home - p_away)[ok]
    n = int(ok.sum())
    # P(total > 2.5) = 1 - e^-mu (1 + mu + mu^2 / 2), increasing in mu
    total = _bisect(lambda mu: 1 - np.exp(-mu) * (1 + mu + mu ** 2 / 2), np.full(n, 0.05), np.full(n, 10.0), over)
    sup = _bisect(lambda s: _home_minus_away((total + s) / 2, (total - s) / 2), -total * 0.999, total * 0.999, diff)
    lam_h[ok], lam_a[ok] = (total + sup) / 2, (total - sup) / 2
    return lam_h, lam_a


# --------------------------------------------------------------------------- features

def collected_at(kickoff: pd.Series) -> pd.Series:
    """Latest time football-data.co.uk collects a match's pre-closing prices (UTC)."""
    local = pd.to_datetime(kickoff, utc=True).dt.tz_convert(UK)
    back = local.dt.weekday.map(lambda d: COLLECTION[d][0])
    hour = local.dt.weekday.map(lambda d: COLLECTION[d][1])
    day = local.dt.normalize().dt.tz_localize(None) - pd.to_timedelta(back, unit="D")
    at = (day + pd.to_timedelta(hour, unit="h")).dt.tz_localize(UK, ambiguous="NaT", nonexistent="shift_forward")
    return at.dt.tz_convert("UTC")


def odds_features(df: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """FEATURES and CLOSING_FEATURES for every row of build_features' frame `df` (season, GW,
    team, opponent, was_home, kickoff_time), on the same index, from the team's side."""
    out = pd.DataFrame(np.nan, index=df.index, columns=FEATURES + CLOSING_FEATURES)
    if odds.empty:
        return out
    home_side = df["was_home"].astype(bool).to_numpy()
    home = np.where(home_side, df["team"], df["opponent"])
    away = np.where(home_side, df["opponent"], df["team"])
    key = pd.MultiIndex.from_arrays([df["season"].to_numpy(), home, away])
    kickoff = pd.to_datetime(df["kickoff_time"], utc=True)
    deadline = kickoff.groupby([df["season"], df["GW"]]).transform("min") - pd.Timedelta(minutes=DEADLINE_MINUTES)
    in_time = (collected_at(kickoff) <= deadline).to_numpy()

    odds = odds.drop_duplicates(["season", "home", "away"], keep="last")   # e.g. a season file + fixtures.csv
    for suffix, names in (("", FEATURES), ("_c", CLOSING_FEATURES)):
        o = odds.copy()
        o["lam_h"], o["lam_a"] = fit_goals(o[f"p_home{suffix}"], o[f"p_away{suffix}"], o[f"p_over{suffix}"])
        got = o.set_index(["season", "home", "away"]).reindex(key)
        ph, pd_, pa = (got[f"p_{c}{suffix}"].to_numpy() for c in ("home", "draw", "away"))
        lam_h, lam_a = got["lam_h"].to_numpy(), got["lam_a"].to_numpy()
        gf, ga = np.where(home_side, lam_h, lam_a), np.where(home_side, lam_a, lam_h)
        vals = np.column_stack([np.where(home_side, ph, pa), pd_, gf, ga, np.exp(-ga)])
        if not suffix:
            vals[~in_time] = np.nan
        out[names] = vals
    return out
