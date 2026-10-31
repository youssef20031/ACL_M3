"""
Expected FPL points of a match from what the player did, not from what went in: realized xG and
xA in place of goals and assists, and the xG his team conceded while he was on the pitch in place
of clean sheets and goals conceded. The backtests score picks on these as well as on actual
points: finishing luck makes a week's actual points noisy, so differences between variants that
are too small to show in actual points can show here.

Per match, for a player's position (FPL's rules, which reproduce every actual score 2020-21..2025-26):
  appearance   actual: 1 point for 1-59 minutes, 2 for 60+
  goals        xG x 6 (GK, DEF), 5 (MID), 4 (FWD)
  assists      xA x 3
  clean sheet  60+ minutes: 4 (GK, DEF) or 1 (MID) x exp(-xGC), Poisson's chance of no goal
  conceded     GK, DEF: -E[floor(goals conceded / 2)], goals conceded ~ Poisson(xGC)
  saves        actual: 1 point per 3 (GK)
  defensive    actual: 2 points for 10+ (DEF) or 12+ (MID, FWD) defensive contributions (2025-26 on)
  bonus        a least-squares fit of actual bonus on the terms above, clipped to 0..3
  the rest     cards, own goals and penalty saves and misses: their average, as luck
Each season and position is calibrated on its own players who played: the goals, assists, clean
sheet and conceded terms are scaled so their totals match the actual points those terms scored
(FPL counts more assists than xA does, such as penalties won, and exp(-xGC) overstates clean
sheets), the bonus is fitted there, and "the rest" is its mean there. So expected points keep each
term's level and only redistribute it by the underlying numbers. A player who didn't play scores
0; one who played in a season without expected stats (before 2023-24) is NaN.
"""
import numpy as np
import pandas as pd

GOAL_POINTS = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
CS_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}
DC_POINTS = 2
SCALED = ["goals", "assists", "clean_sheet", "conceded"]     # expected terms scaled to the actual level
EXACT = ["appearance", "saves", "defensive"]                 # terms taken from the actual stats
BONUS_MIN_ROWS = 200        # rows a season and position need for its own bonus fit
X_COLUMNS = ["expected_goals", "expected_assists", "expected_goals_conceded"]
_MAX_GOALS = 20             # Poisson terms summed for the goals-conceded penalty


def conceded_penalty(xgc) -> np.ndarray:
    """E[floor(G / 2)] for G ~ Poisson(xgc), elementwise."""
    lam = np.clip(np.asarray(xgc, dtype=float), 0, None)
    pmf = np.exp(-lam)
    out = np.zeros_like(lam)
    for k in range(1, _MAX_GOALS + 1):
        pmf = pmf * lam / k
        out += (k // 2) * pmf
    return out


def dc_points(position: pd.Series, defensive_contribution: pd.Series) -> np.ndarray:
    """FPL's defensive-contribution points from the actual count (0 before 2025-26: no count)."""
    threshold = position.map(DC_THRESHOLD).to_numpy(dtype=float)   # NaN for GK: never reached
    count = pd.to_numeric(defensive_contribution, errors="coerce").fillna(0).to_numpy(dtype=float)
    return np.where(count >= threshold, DC_POINTS, 0)


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def terms(df: pd.DataFrame) -> tuple:
    """(expected, actual): each scoring term per row, from the underlying numbers and from what
    happened. The actual frame also has "rest" (cards, own goals, penalties) and "bonus"."""
    pos = df["position"].astype(str)
    gk, gk_def = (pos == "GK").to_numpy(), pos.isin(["GK", "DEF"]).to_numpy()
    goal_pts = pos.map(GOAL_POINTS).to_numpy(dtype=float)
    cs_pts = pos.map(CS_POINTS).to_numpy(dtype=float)
    minutes = _num(df, "minutes")
    xg, xa, xgc = (_num(df, c) for c in X_COLUMNS)
    exact = pd.DataFrame({
        "appearance": np.where(minutes >= 60, 2.0, np.where(minutes > 0, 1.0, 0.0)),
        "saves": np.where(gk, np.nan_to_num(_num(df, "saves")) // 3, 0.0),
        "defensive": dc_points(pos, df["defensive_contribution"] if "defensive_contribution" in df
                               else pd.Series(np.nan, index=df.index)).astype(float),
    }, index=df.index)
    expected = exact.assign(
        goals=xg * goal_pts,
        assists=3.0 * xa,
        clean_sheet=(minutes >= 60) * cs_pts * np.exp(-np.clip(xgc, 0, None)),
        conceded=np.where(gk_def, -conceded_penalty(np.nan_to_num(xgc)), 0.0),
    )
    actual = exact.assign(
        goals=np.nan_to_num(_num(df, "goals_scored")) * goal_pts,
        assists=3.0 * np.nan_to_num(_num(df, "assists")),
        clean_sheet=np.nan_to_num(_num(df, "clean_sheets")) * cs_pts,
        conceded=np.where(gk_def, -(np.nan_to_num(_num(df, "goals_conceded")) // 2), 0.0),
        bonus=np.nan_to_num(_num(df, "bonus")),
    )
    actual["rest"] = _num(df, "total_points") - actual[EXACT + SCALED + ["bonus"]].sum(axis=1)
    return expected, actual


def _lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def expected_points(df: pd.DataFrame) -> pd.Series:
    """Expected points per row (see the module docstring). Needs season, position (GK/DEF/MID/
    FWD), minutes, total_points, the X_COLUMNS and the actual goals_scored, assists, clean_sheets,
    goals_conceded, saves and bonus; defensive_contribution is optional."""
    expected, actual = terms(df)
    played = _num(df, "minutes") > 0
    has_x = np.isfinite(expected[["goals", "assists"]].to_numpy()).all(axis=1) & np.isfinite(
        _num(df, "expected_goals_conceded"))
    fit = played & has_x
    groups = (df["season"].astype(str) + "|" + df["position"].astype(str)).to_numpy()
    out = np.full(len(df), np.nan)
    out[~played] = 0.0
    bonus_cols = ["goals", "assists", "clean_sheet", "saves", "defensive"]
    for g in pd.unique(groups[fit]):
        rows = fit & (groups == g)
        e = expected[rows].copy()
        for t in SCALED:
            denominator = e[t].sum()
            if abs(denominator) > 1e-9:
                e[t] *= actual.loc[rows, t].sum() / denominator
        x = np.column_stack([np.ones(rows.sum()), e[bonus_cols].to_numpy(), (e["appearance"] == 2).astype(float)])
        y = actual.loc[rows, "bonus"].to_numpy()
        bonus = np.clip(x @ _lstsq(x, y), 0, 3) if rows.sum() >= BONUS_MIN_ROWS else np.full(rows.sum(), y.mean())
        rest = np.nanmean(actual.loc[rows, "rest"].to_numpy())
        out[rows] = e[EXACT + SCALED].sum(axis=1).to_numpy() + bonus + (0.0 if np.isnan(rest) else rest)
    return pd.Series(out, index=df.index, name="xpts")
