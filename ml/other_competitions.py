"""
Matches outside the Premier League (European cups, FA Cup, EFL Cup, super cups, Club World Cup)
as candidate features for the v3 points model: congestion around midweek ties, a player's
minutes in his club's latest matches in any competition, and whether the club is still in
each competition.

Inputs (load_other_competitions): two tables, the history in data/ (built from the
transfermarkt-datasets dump by scripts/data/fetch_transfermarkt_non_pl.py) and, when present, the
current season in LIVE_DATA_DIR.
  matches  one row per Premier League club per match: match_id, season, competition (a key of
           COMPETITIONS), kickoff_time (UTC; empty for a next-round tie that isn't scheduled
           yet), team (graph name), opponent, status ("finished" or "scheduled")
  apps     one row per player in a finished match's matchday squad: match_id, team, name (the
           FPL name; empty when unmatched), started, minutes, goals, assists

Leakage rules (tests/test_points_model.py):
  * Outcome features (minutes, starts, goals) use only matches that kicked off at least
    CUTOFF_HOURS before the gameweek's first kickoff (FPL's deadline is 90 minutes before it),
    so both fixtures of a double gameweek see the same values, as the PL form features do.
  * Schedule features (days to the previous/next tie, matches in the last 14 days, which cups
    the club is still in) use only kickoff dates, which are published weeks ahead.
"""
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

COMPETITIONS = {
    "UCL": "UEFA Champions League", "UCLQ": "UEFA Champions League qualifying",
    "UEL": "UEFA Europa League", "UELQ": "UEFA Europa League qualifying",
    "UECL": "UEFA Conference League", "UECLQ": "UEFA Conference League qualifying",
    "FAC": "FA Cup", "EFLC": "EFL Cup", "CS": "Community Shield", "USC": "UEFA Super Cup",
    "CWC": "FIFA Club World Cup",
}
EURO_TIER = {"UCL": 3, "UCLQ": 3, "UEL": 2, "UELQ": 2, "UECL": 1, "UECLQ": 1}
DOMESTIC_CUPS = ("FAC", "EFLC")
# How much a tie is worth resting players for: European ties by tier, everything else 1
TIE_TIER = {**{c: t + 1 for c, t in EURO_TIER.items()}, **{c: 1 for c in ("FAC", "EFLC", "CS", "USC", "CWC")}}

CAP_DAYS = 30.0        # "no match for a month" and "none at all" look the same
NEXT_TIE_DAYS = 10.0   # oc_next_tier only looks this far ahead
LAST_K = 3             # oc_*_r3: the club's last 3 non-PL matches this season
GA_WINDOW = 8          # ac_ga_p90_r8: the player's last 8 appearances in any competition
UNSCHEDULED = 1e9      # days: a next-round tie without a date sorts after everything
CUTOFF_HOURS = 2.0     # outcomes must be known this long before the gameweek's first kickoff

HISTORY_MATCHES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "non_pl_matches.csv")
HISTORY_APPS = HISTORY_MATCHES.replace("non_pl_matches", "non_pl_appearances")
LIVE_MATCHES = "non_pl_matches.csv"
LIVE_APPS = "non_pl_appearances.csv"

# Schedule around the predicted fixture (team level, dates only)
SCHEDULE_FEATURES = ["oc_prev_days", "oc_next_days", "oc_next_tier", "ac_days_since", "ac_matches_14d"]
# The player's recent matches in any competition, before the gameweek's first kickoff
PLAYER_FEATURES = ["oc_minutes_last", "oc_started_last", "oc_squad_last", "oc_minutes_r3", "oc_started_r3",
                   "ac_minutes_last", "ac_started_last", "ac_minutes_7d", "ac_minutes_14d", "ac_starts_14d",
                   "ac_ga_p90_r8"]
# Which competitions the club is still in at the gameweek's first kickoff
STATUS_FEATURES = ["oc_euro_tier", "oc_cups_alive"]
FEATURES = SCHEDULE_FEATURES + PLAYER_FEATURES + STATUS_FEATURES
# Subsets for the backtest (they may use the whole season, unlike features)
SUBSETS = ["sub_euro_club", "sub_cup_week"]

_EPOCH = pd.Timestamp("2000-01-01", tz="UTC")


def _days(ts: pd.Series) -> pd.Series:
    """Kickoff times as float days, so merge_asof never trips over time zones or units."""
    return (pd.to_datetime(ts, utc=True) - _EPOCH) / pd.Timedelta(days=1)


def load_other_competitions(live_dir: Optional[str] = None, history_matches: str = HISTORY_MATCHES,
                            history_apps: str = HISTORY_APPS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(matches, apps): the committed history plus, when present, the current season from
    `live_dir`. A match in both comes from the live copy."""
    frames = {"matches": [], "apps": []}
    for m_path, a_path in ((history_matches, history_apps),
                           (os.path.join(live_dir, LIVE_MATCHES) if live_dir else None,
                            os.path.join(live_dir, LIVE_APPS) if live_dir else None)):
        if m_path and os.path.exists(m_path):
            frames["matches"].append(pd.read_csv(m_path, dtype={"match_id": str}))
            if a_path and os.path.exists(a_path):
                frames["apps"].append(pd.read_csv(a_path, dtype={"match_id": str}))
    matches = (pd.concat(frames["matches"], ignore_index=True) if frames["matches"]
               else pd.DataFrame(columns=["match_id", "season", "competition", "kickoff_time", "team", "opponent", "status"]))
    apps = (pd.concat(frames["apps"], ignore_index=True) if frames["apps"]
            else pd.DataFrame(columns=["match_id", "team", "name", "started", "minutes", "goals", "assists"]))
    matches = matches.drop_duplicates(["match_id", "team"], keep="last").reset_index(drop=True)
    apps = apps.drop_duplicates(["match_id", "team", "name", "source_id"] if "source_id" in apps else
                                ["match_id", "team", "name"], keep="last").reset_index(drop=True)
    matches["kickoff_time"] = pd.to_datetime(matches["kickoff_time"], utc=True)
    return matches, apps


def _asof(left: pd.DataFrame, right: pd.DataFrame, on: str, by, direction: str, exact: bool) -> pd.DataFrame:
    """pd.merge_asof that returns rows in `left`'s order and index. `on` is a float column
    present in both; rows of `left` where it is NaN get NaN."""
    ok = left[on].notna()
    lhs = left[ok].assign(_i=np.flatnonzero(ok.to_numpy())).sort_values(on, kind="mergesort")
    rhs = right.dropna(subset=[on]).sort_values(on, kind="mergesort")
    rhs = rhs.rename(columns={on: f"{on}_r"})
    merged = pd.merge_asof(lhs, rhs, left_on=on, right_on=f"{on}_r", by=by, direction=direction,
                           allow_exact_matches=exact)
    cols = [c for c in rhs.columns if c not in ([by] if isinstance(by, str) else list(by or []))]
    out = pd.DataFrame(np.nan, index=np.arange(len(left)), columns=cols, dtype=object)
    out.iloc[merged["_i"].to_numpy()] = merged[cols].to_numpy()
    out.index = left.index
    return out.infer_objects()


def _count_before(left_t: pd.Series, keys: pd.DataFrame, events: pd.DataFrame, by) -> pd.Series:
    """Number of `events` (same `by` values) strictly before each time in `left_t`."""
    ev = events.sort_values(["t"], kind="mergesort").copy()
    ev["n"] = ev.groupby(by, sort=False).cumcount() + 1
    left = keys.assign(t=left_t)
    return _asof(left, ev[([by] if isinstance(by, str) else list(by)) + ["t", "n"]], "t", by,
                 "backward", False)["n"].astype(float).fillna(0.0)


def _cumulative_before(left_t: pd.Series, names: pd.Series, events: pd.DataFrame, cols) -> pd.DataFrame:
    """Per name, the sum of `cols` over events strictly before each time in `left_t`."""
    ev = events.sort_values(["name", "t"], kind="mergesort").copy()
    ev[cols] = ev.groupby("name", sort=False)[cols].cumsum()
    got = _asof(pd.DataFrame({"name": names, "t": left_t}), ev[["name", "t"] + cols], "t", "name",
                "backward", False)
    return got[cols].astype(float).fillna(0.0)


def other_competition_features(df: pd.DataFrame, matches: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    """FEATURES and SUBSETS for every row of build_features' frame `df`, on the same index.

    `df` needs season, GW, name, team, fixture, kickoff_time, is_upcoming, minutes, started,
    goals_scored, assists.
    """
    out = pd.DataFrame(index=df.index)
    t = _days(df["kickoff_time"])
    # The cut-off for outcomes: the FPL deadline, 90 minutes before the gameweek's first kickoff
    # (never after the row's own kickoff); CUTOFF_HOURS leaves a margin
    first = t.groupby([df["season"], df["GW"]]).transform("min")
    cut = np.minimum(first, t) - CUTOFF_HOURS / 24
    keys = df[["season", "team", "name"]]

    m = matches.copy()
    m["t"] = _days(m["kickoff_time"])
    m["tier"] = m["competition"].map(TIE_TIER).fillna(1).astype(float)
    played_oc = m[(m["status"] == "finished") & m["t"].notna()]

    # ---- schedule around the fixture (dates only)
    sched_oc = m.loc[m["t"].notna(), ["team", "t", "tier"]]
    prev = _asof(keys.assign(t=t), sched_oc[["team", "t"]], "t", "team", "backward", False)["t_r"]
    nxt = _asof(keys.assign(t=t), sched_oc, "t", "team", "forward", False)
    out["oc_prev_days"] = (t - prev.astype(float)).clip(upper=CAP_DAYS).fillna(CAP_DAYS)
    out["oc_next_days"] = (nxt["t_r"].astype(float) - t).clip(upper=CAP_DAYS).fillna(CAP_DAYS)
    out["oc_next_tier"] = nxt["tier"].astype(float).where(out["oc_next_days"] <= NEXT_TIE_DAYS, 0.0).fillna(0.0)

    pl_sched = df.drop_duplicates(["season", "fixture", "team"])[["team"]].assign(
        t=t[df.drop_duplicates(["season", "fixture", "team"]).index])
    sched_all = pd.concat([pl_sched, sched_oc[["team", "t"]]], ignore_index=True)
    prev_any = _asof(keys.assign(t=t), sched_all, "t", "team", "backward", False)["t_r"]
    out["ac_days_since"] = (t - prev_any.astype(float)).clip(upper=CAP_DAYS).fillna(CAP_DAYS)
    out["ac_matches_14d"] = (_count_before(t, keys[["team"]], sched_all, "team")
                             - _count_before(t - 14, keys[["team"]], sched_all, "team"))

    # ---- the player's matches before the cut-off
    hist = df[~df["is_upcoming"].astype(bool) & df["minutes"].notna()]
    pl_club = hist.drop_duplicates(["season", "fixture", "team"])
    pl_club = pd.DataFrame({"season": pl_club["season"], "team": pl_club["team"], "t": t[pl_club.index],
                            "key": "PL:" + pl_club["season"].astype(str) + ":" + pl_club["fixture"].astype(str)})
    oc_club = pd.DataFrame({"season": played_oc["season"], "team": played_oc["team"], "t": played_oc["t"],
                            "key": "OC:" + played_oc["match_id"].astype(str)})
    pl_player = pd.DataFrame({
        "name": hist["name"], "key": "PL:" + hist["season"].astype(str) + ":" + hist["fixture"].astype(str),
        "t": t[hist.index], "minutes": hist["minutes"].astype(float), "started": hist["started"].astype(float),
        "squad": 1.0, "ga": (hist["goals_scored"].fillna(0) + hist["assists"].fillna(0)).astype(float)})
    a = apps[apps["name"].notna()].merge(played_oc[["match_id", "team", "t"]], on=["match_id", "team"])
    oc_player = pd.DataFrame({
        "name": a["name"], "key": "OC:" + a["match_id"].astype(str), "t": a["t"],
        "minutes": a["minutes"].astype(float), "started": a["started"].astype(float), "squad": 1.0,
        "ga": (a["goals"].fillna(0) + a["assists"].fillna(0)).astype(float)})
    player = pd.concat([pl_player, oc_player], ignore_index=True)
    by_key = player.drop_duplicates(["name", "key"]).set_index(["name", "key"])[["minutes", "started", "squad"]]
    # A match without any squad data says nothing about who played (not "everyone was rested")
    with_squads = set(pl_player["key"]) | set("OC:" + apps["match_id"].astype(str))

    def club_last(club: pd.DataFrame, k: int) -> pd.Series:
        """Key of the club's k-th last match (0 = latest) this season before the cut-off."""
        club = club.sort_values("t", kind="mergesort").copy()
        club["idx"] = club.groupby(["team", "season"], sort=False).cumcount()
        last = _asof(keys.assign(t=cut), club[["season", "team", "t", "idx"]], "t", ["season", "team"],
                     "backward", False)["idx"].astype(float)
        if k:
            last = last - k
        want = pd.DataFrame({"season": keys["season"], "team": keys["team"], "idx": last})
        lookup = club.set_index(["season", "team", "idx"])["key"]
        idx = pd.MultiIndex.from_frame(want.fillna({"idx": -1}).astype({"idx": int}))
        return pd.Series(lookup.reindex(idx).to_numpy(), index=df.index)

    def player_in(key: pd.Series) -> pd.DataFrame:
        """The row's player in match `key`: minutes/started/squad, 0 if absent, NaN if there is
        no such match or no squad data for it."""
        idx = pd.MultiIndex.from_arrays([df["name"], key.fillna("")])
        got = by_key.reindex(idx).set_axis(df.index)
        return got.fillna(0.0).where(key.isin(with_squads))

    last_oc = player_in(club_last(oc_club, 0))
    out["oc_minutes_last"], out["oc_started_last"], out["oc_squad_last"] = (
        last_oc["minutes"], last_oc["started"], last_oc["squad"])
    recent = [last_oc] + [player_in(club_last(oc_club, k)) for k in range(1, LAST_K)]
    out["oc_minutes_r3"] = pd.concat([r["minutes"] for r in recent], axis=1).mean(axis=1)
    out["oc_started_r3"] = pd.concat([r["started"] for r in recent], axis=1).mean(axis=1)

    last_any = player_in(club_last(pd.concat([pl_club, oc_club], ignore_index=True), 0))
    out["ac_minutes_last"], out["ac_started_last"] = last_any["minutes"], last_any["started"]

    for days in (7, 14):
        now = _cumulative_before(cut, df["name"], player, ["minutes", "started"])
        then = _cumulative_before(cut - days, df["name"], player, ["minutes", "started"])
        out[f"ac_minutes_{days}d"] = now["minutes"] - then["minutes"]
        if days == 14:
            out["ac_starts_14d"] = now["started"] - then["started"]

    apps_played = player[player["minutes"] > 0].sort_values(["name", "t"], kind="mergesort").copy()
    roll = apps_played.groupby("name", sort=False)[["ga", "minutes"]].rolling(GA_WINDOW, min_periods=1).sum()
    apps_played[["ga_w", "min_w"]] = roll.reset_index(level=0, drop=True).reindex(apps_played.index).to_numpy()
    form = _asof(pd.DataFrame({"name": df["name"], "t": cut}), apps_played[["name", "t", "ga_w", "min_w"]],
                 "t", "name", "backward", False)
    out["ac_ga_p90_r8"] = (form["ga_w"].astype(float) / form["min_w"].astype(float) * 90).where(
        form["min_w"].astype(float) >= 90)

    # ---- which competitions the club is still in (has a tie on or after the cut-off)
    m["t_open"] = m["t"].fillna(UNSCHEDULED)
    euro = m[m["competition"].isin(EURO_TIER)].assign(euro_tier=lambda x: x["competition"].map(EURO_TIER))
    nxt_euro = _asof(keys.assign(t_open=cut), euro[["season", "team", "t_open", "euro_tier"]], "t_open",
                     ["season", "team"], "forward", True)
    out["oc_euro_tier"] = nxt_euro["euro_tier"].astype(float).fillna(0.0)
    alive = np.zeros(len(df))
    for cup in DOMESTIC_CUPS:
        c = m[m["competition"] == cup][["season", "team", "t_open"]]
        hit = _asof(keys.assign(t_open=cut), c.assign(x=1.0), "t_open", ["season", "team"], "forward", True)["x"]
        alive += hit.astype(float).fillna(0.0).to_numpy()
    out["oc_cups_alive"] = alive

    # ---- backtest subsets
    euro_clubs = set(map(tuple, euro[["season", "team"]].drop_duplicates().to_numpy()))
    out["sub_euro_club"] = [float((s, tm) in euro_clubs) for s, tm in zip(df["season"], df["team"])]
    out["sub_cup_week"] = ((out["oc_prev_days"] <= 4) | (out["oc_next_days"] <= 4)).astype(float)
    return out
