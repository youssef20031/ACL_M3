"""
Fixture-strength features: the ratings are part of the v4 points model (ml/points_model.py, since
2026-10-28); matchup stays a backtest candidate (scripts/analysis/ml_backtest.py variants matchup
and ratings).

  matchup  multiplicative expectations for the predicted fixture, built from the model's own
           rolling team stats: expected xG for and against (own attack x opponent defence, with a
           home factor) and their difference, the player's expected goals and assists (his share
           of the team's xG over the last 6 matches times the fixture's expected xG for), and a
           starter's clean-sheet chance (share of recent matches started x P(0 goals against)).
           A tree splits on one feature at a time, so it can only approximate such products with
           many splits.
  ratings  opponent-adjusted team ratings: attack and defence on a log scale, updated after every
           gameweek from each match's xG (blended with goals) against what the ratings expected,
           so a team that has just played the top clubs isn't rated like a weak one. The rolling
           team features are plain averages over whoever the team played. The same expected-xG,
           clean-sheet and involvement features are then built from the ratings.

Both use only matches from earlier gameweeks: matchup is a function of features that already
satisfy that (ml.points_model), and the ratings for a gameweek are recorded before its matches are
added (tests/test_points_model.py).

The ratings are recursive from the first season they see, so serving, which builds features from
the last few seasons only (pipeline/predict_next_gw.HISTORY_SEASONS), passes the earlier seasons'
team-match table as `prior` (team_match_history): the ratings then equal a full-history build's
(tests/test_points_model_v4.py).
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ml import points_model as pm

LEAGUE_XG = 1.35          # a Premier League team's xG per match, roughly (2022-23 .. 2025-26)
HOME_FACTOR = 1.08        # home sides create ~17% more xG than away sides (1.08 / 0.92)
AWAY_FACTOR = 0.92
MIN_TEAM_XG = 0.3         # floor for the denominator of a player's share of his team's xG

MATCHUP_FEATURES = ["mu_xgf", "mu_xga", "mu_xgd", "mu_pxg", "mu_pxa", "mu_cs"]

RATING_STEP = 0.05        # log-rating change per unit of (observed - expected) xG
RATING_SEASON_SHRINK = 0.7  # ratings regress towards average over the summer
RATING_NEW_TEAM = -0.15   # promoted clubs start below average in attack and defence
RATING_XG_WEIGHT = 0.7    # match "goals" = 0.7 xG + 0.3 goals (goals alone before xG was recorded)
HOME_LOG = float(np.log(HOME_FACTOR))
RATING_FEATURES = ["rt_att", "rt_def", "rt_opp_att", "rt_opp_def", "rt_xgf", "rt_xga", "rt_cs", "rt_pxg", "rt_pxa"]


def _strength(df: pd.DataFrame, prefix: str, stat: str) -> pd.Series:
    """Mean of the 6- and 20-match rolling xG values; goals where xG isn't recorded (before 2022-23)."""
    xg = df[[pm._team_feature(prefix, f"x{stat}", w) for w in pm.TEAM_WINDOWS]].mean(axis=1)
    goals = df[[pm._team_feature(prefix, stat, w) for w in pm.TEAM_WINDOWS]].mean(axis=1)
    return xg.fillna(goals)


def _player_shares(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """The player's share of his team's xG and of its assists' xG over the last 6 matches
    (goals/assists where xG isn't recorded)."""
    team = df["team_xgf"].fillna(df["team_gf"]).clip(lower=MIN_TEAM_XG)
    xg = df["expected_goals_r6"].fillna(df["goals_scored_r6"])
    xa = df["expected_assists_r6"].fillna(df["assists_r6"])
    return xg / team, xa / team


def _home_factors(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    home = df["was_home"].astype(float).to_numpy() > 0
    return np.where(home, HOME_FACTOR, AWAY_FACTOR), np.where(home, AWAY_FACTOR, HOME_FACTOR)


def matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """MATCHUP_FEATURES for every row of a pm.build_features table."""
    own, other = _home_factors(df)
    xgf = _strength(df, "team", "gf") * _strength(df, "opp", "ga") / LEAGUE_XG * own
    xga = _strength(df, "opp", "gf") * _strength(df, "team", "ga") / LEAGUE_XG * other
    share_g, share_a = _player_shares(df)
    return pd.DataFrame({
        "mu_xgf": xgf, "mu_xga": xga, "mu_xgd": xgf - xga,
        "mu_pxg": share_g * xgf, "mu_pxa": share_a * xgf,
        "mu_cs": df["started_r6"] * np.exp(-xga),
    }, index=df.index)


# Columns of the history CSV that pm._team_match_table reads (plus position, for load_history's filter)
TEAM_MATCH_COLUMNS = ["season", "team", "fixture", "kickoff_time", "GW", "was_home", "team_h_score",
                      "team_a_score", "expected_goals"]


def team_match_history(history: Optional[pd.DataFrame] = None, history_csv: str = pm.HISTORY_CSV) -> pd.DataFrame:
    """pm._team_match_table of `history`, or of the history CSV read with only the columns it needs
    (rows filtered like pm.load_history), for team_ratings' `prior`."""
    if history is None:
        wanted = {"season_x", "team_x", "position"} | set(TEAM_MATCH_COLUMNS)
        raw = pd.read_csv(history_csv, usecols=lambda c: c in wanted, low_memory=False)
        raw = raw.rename(columns={"season_x": "season", "team_x": "team"})
        history = raw[raw["position"].replace({"GKP": "GK"}).isin(pm.POSITIONS)].copy()
        history["kickoff_time"] = pd.to_datetime(history["kickoff_time"], utc=True)
        history["was_home"] = history["was_home"].astype(bool)
    return pm._team_match_table(history[TEAM_MATCH_COLUMNS])


def team_ratings(history: pd.DataFrame, prior: Optional[pd.DataFrame] = None
                 ) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]], str]:
    """Pre-gameweek attack/defence ratings of both teams for every (season, fixture, team).

    Returns (table with columns season, fixture, team, att, dfn, opp_att, opp_dfn; the ratings
    after the last gameweek {team: (att, dfn)}; the last season seen). Gameweeks are processed in
    order of their first kickoff; every match of a gameweek is rated with the ratings from before
    it, and only then are the gameweek's results added (a double gameweek adds two updates).
    `prior` (team_match_history) adds the seasons before `history`'s first one, which only
    move the starting ratings; the table keeps them too.
    """
    tm = pm._team_match_table(history)
    if prior is not None and len(prior):
        first = tm["season"].min() if len(tm) else None
        earlier = prior if first is None else prior[prior["season"] < first]
        tm = pd.concat([earlier[tm.columns], tm], ignore_index=True)
    y = (RATING_XG_WEIGHT * tm["xgf"] + (1 - RATING_XG_WEIGHT) * tm["gf"]).fillna(tm["gf"])
    tm = tm.assign(y=y)
    opp = pm._opponents(tm[["season", "fixture", "team"]])
    # Left join: a fixture whose opponent has no player rows keeps its team's ratings, unpaired
    pairs = tm.merge(opp, on=["season", "fixture", "team"], how="left")
    order = pairs.groupby(["season", "GW"])["kickoff_time"].min().sort_values()

    att: Dict[str, float] = {}
    dfn: Dict[str, float] = {}
    first_season = order.index[0][0] if len(order) else None
    season = first_season
    rows = []
    for (s, gw) in order.index:
        if s != season:
            for t in att:
                att[t] *= RATING_SEASON_SHRINK
                dfn[t] *= RATING_SEASON_SHRINK
            season = s
        block = pairs[(pairs["season"] == s) & (pairs["GW"] == gw)]
        for t in pd.unique(block[["team", "opponent"]].to_numpy().ravel()):
            if isinstance(t, str) and t not in att:
                start = 0.0 if s == first_season else RATING_NEW_TEAM
                att[t], dfn[t] = start, start
        updates: Dict[str, list] = {}
        for r in block.itertuples(index=False):
            paired = isinstance(r.opponent, str)
            rows.append((r.season, r.fixture, r.team, att[r.team], dfn[r.team],
                         att[r.opponent] if paired else np.nan, dfn[r.opponent] if paired else np.nan))
            if not paired or pd.isna(r.y):
                continue
            home = HOME_LOG if r.was_home else -HOME_LOG
            err = r.y - LEAGUE_XG * np.exp(att[r.team] - dfn[r.opponent] + home)
            updates.setdefault(r.team, [0.0, 0.0])[0] += RATING_STEP * err
            updates.setdefault(r.opponent, [0.0, 0.0])[1] -= RATING_STEP * err
        for t, (da, dd) in updates.items():
            att[t] += da
            dfn[t] += dd
    table = pd.DataFrame(rows, columns=["season", "fixture", "team", "att", "dfn", "opp_att", "opp_dfn"])
    return table, {t: (att[t], dfn[t]) for t in att}, season


def ratings_features(df: pd.DataFrame, prior: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """RATING_FEATURES for every row of a pm.build_features table (upcoming rows get the ratings
    after the last finished gameweek, regressed if their season is a new one). `prior`: see
    team_ratings."""
    upcoming = df["is_upcoming"].to_numpy(dtype=bool) if "is_upcoming" in df else np.zeros(len(df), bool)
    table, latest, last_season = team_ratings(df[~upcoming], prior)
    keyed = table.drop_duplicates(["season", "fixture", "team"]).set_index(["season", "fixture", "team"])
    idx = pd.MultiIndex.from_frame(df[["season", "fixture", "team"]])
    vals = keyed.reindex(idx)[["att", "dfn", "opp_att", "opp_dfn"]].to_numpy()
    out = pd.DataFrame(vals, index=df.index, columns=["rt_att", "rt_def", "rt_opp_att", "rt_opp_def"])
    if upcoming.any():
        up = df[upcoming]
        shrink = np.where(up["season"].to_numpy() == last_season, 1.0, RATING_SEASON_SHRINK)

        def rating(team_col: str, k: int) -> np.ndarray:
            # A club without a rated match (promoted) starts at RATING_NEW_TEAM, which team_ratings
            # gives it after the summer's regression, so it isn't shrunk
            teams = up[team_col]
            known = teams.map(lambda t: t in latest).to_numpy(dtype=bool)
            vals = teams.map(lambda t: latest.get(t, (RATING_NEW_TEAM, RATING_NEW_TEAM))[k]).to_numpy(dtype=float)
            return np.where(known, shrink * vals, vals)

        out.loc[upcoming, "rt_att"] = rating("team", 0)
        out.loc[upcoming, "rt_def"] = rating("team", 1)
        out.loc[upcoming, "rt_opp_att"] = rating("opponent", 0)
        out.loc[upcoming, "rt_opp_def"] = rating("opponent", 1)
    home = np.where(df["was_home"].astype(float).to_numpy() > 0, HOME_LOG, -HOME_LOG)
    out["rt_xgf"] = LEAGUE_XG * np.exp(out["rt_att"] - out["rt_opp_def"] + home)
    out["rt_xga"] = LEAGUE_XG * np.exp(out["rt_opp_att"] - out["rt_def"] - home)
    out["rt_cs"] = df["started_r6"] * np.exp(-out["rt_xga"])
    share_g, share_a = _player_shares(df)
    out["rt_pxg"] = share_g * out["rt_xgf"]
    out["rt_pxa"] = share_a * out["rt_xgf"]
    return out
