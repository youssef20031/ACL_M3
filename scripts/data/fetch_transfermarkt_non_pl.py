"""
Build data/non_pl_matches.csv and data/non_pl_appearances.csv (the inputs of
ml/other_competitions.py) from the transfermarkt-datasets dump
(github.com/dcaribou/transfermarkt-datasets, CC0-1.0): Premier League clubs' matches in the
Champions League, Europa League, Conference League (with their qualifying rounds), FA Cup,
Community Shield and UEFA Super Cup, with each matchday squad's starts, minutes, goals and assists.

    python -m scripts.data.fetch_transfermarkt_non_pl [--cache-dir .cache/transfermarkt]

Limits: the dump has no EFL Cup and no Club World Cup, gives match dates without kickoff times
(KICKOFF_HOUR is assumed), and stopped updating in July 2026, so it has no 2026-27 data. It feeds
the ML backtest only; serving would need a source that covers the current season.

Mapping to the FPL names in data/cleaned_merged_seasons_cleaned.csv:
  * Clubs: each Transfermarkt club id is the FPL team that played at home on the same dates in
    the Premier League (GB1 games vs the FPL fixtures).
  * Players: within a club and season, each Transfermarkt player is paired with the FPL player
    whose match-by-match league minutes agree, helped by name similarity (normalize_name +
    rapidfuzz). Players without league minutes (cup-only youngsters) need a near-exact name.
"""
import argparse
import os
import sys
import urllib.request
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from ml import points_model as pm
from utils.text import normalize_name

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ("games", "appearances", "game_lineups")
OUT_MATCHES = os.path.join("data", "non_pl_matches.csv")
OUT_APPS = os.path.join("data", "non_pl_appearances.csv")

# Transfermarkt competition id -> ml.other_competitions code
COMPETITIONS = {"CL": "UCL", "CLQ": "UCLQ", "EL": "UEL", "ELQ": "UELQ", "UCOL": "UECL", "ECLQ": "UECLQ",
                "FAC": "FAC", "GBCS": "CS", "USC": "USC"}
KICKOFF_HOUR = 20       # UTC; the dump has dates only. Late on purpose: an outcome is never
                        # assumed known earlier than it was (weekend cup ties are earlier)
FIRST_SEASON = 2020     # 2020-21, the first season of the FPL history
# 2019-20 finished in August 2020 (European finals, FA Cup final): keep those matches for the
# congestion features of early 2020-21
LATE_2019_20 = "2020-07-01"
MINUTES_TOLERANCE = 5   # league minutes that "agree" between the two sources
MIN_AGREEMENT = 0.5     # share of league appearances that must agree
MIN_NAME_WITH_MINUTES = 40
MIN_NAME_ONLY = 90


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[2:]}"


def download(cache_dir: str) -> Dict[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for table in TABLES:
        path = os.path.join(cache_dir, f"{table}.csv.gz")
        if not os.path.exists(path):
            print(f"downloading {table}", flush=True)
            urllib.request.urlretrieve(f"{BASE_URL}/{table}.csv.gz", path)
        paths[table] = path
    return paths


def read_filtered(path: str, keep, usecols=None) -> pd.DataFrame:
    """Read a large gzipped CSV in chunks, keeping rows where keep(chunk) is True."""
    parts = [c[keep(c)] for c in pd.read_csv(path, usecols=usecols, chunksize=500_000, low_memory=False)]
    return pd.concat(parts, ignore_index=True)


def london_date(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")


def club_map(games: pd.DataFrame, fpl: pd.DataFrame) -> Dict[int, str]:
    """Transfermarkt club id -> FPL team, by who was at home on the same Premier League dates."""
    fpl_home = fpl[fpl["was_home"]].drop_duplicates(["season", "fixture"])[["date", "team"]]
    tm_home = games.loc[(games["competition_id"] == "GB1") & games["date"].isin(set(fpl["date"])), ["date", "home_club_id"]]
    votes = tm_home.merge(fpl_home, on="date").groupby(["home_club_id", "team"]).size().rename("n").reset_index()
    homes = tm_home.groupby("home_club_id").size()
    best = votes.sort_values("n", ascending=False).drop_duplicates("home_club_id")
    # The right team was at home on every one of the club's home dates; a wrong one on a few
    best = best[best["n"] >= 0.8 * best["home_club_id"].map(homes)]
    dup = best[best["team"].duplicated(keep=False)]
    if len(dup):
        raise SystemExit(f"two clubs mapped to one FPL team:\n{dup}")
    return dict(zip(best["home_club_id"].astype(int), best["team"]))


def match_players(tm: pd.DataFrame, fpl: pd.DataFrame) -> Dict[Tuple[str, str, int], str]:
    """(season, team, Transfermarkt player id) -> FPL name.

    tm:  season, team, player_id, player_name, date, minutes (league rows; cup-only players
         have one row with date NaN)
    fpl: season, team, name, date, minutes
    """
    out = {}
    for (season, team), t in tm.groupby(["season", "team"]):
        f = fpl[(fpl["season"] == season) & (fpl["team"] == team)]
        f_min = {n: dict(zip(g["date"], g["minutes"])) for n, g in f[f["minutes"] > 0].groupby("name")}
        f_names = {n: normalize_name(n) for n in f["name"].unique()}
        cands = []
        for pid, g in t.groupby("player_id"):
            q_min = dict(zip(g["date"].dropna(), g.loc[g["date"].notna(), "minutes"]))
            q_name = normalize_name(g["player_name"].iloc[0])
            name_only = []
            for name, norm in f_names.items():
                sim = fuzz.token_set_ratio(q_name, norm)
                if sim < MIN_NAME_ONLY and q_name.split()[-1:] == norm.split()[-1:]:
                    sim = max(sim, MIN_NAME_ONLY)   # same surname: "Tino Anjorin", "Aji Alese"
                p_min = f_min.get(name, {})
                agree = sum(abs(p_min[d] - m) <= MINUTES_TOLERANCE for d, m in q_min.items() if d in p_min)
                if q_min and p_min:
                    share = agree / max(len(q_min), len(p_min))
                    # Teammates who both play every minute agree too: the name has to fit as
                    # well, unless the minutes alone are conclusive ("Jorginho" is "Jorge Luiz
                    # Frello Filho" in FPL)
                    if agree >= 2 and share >= MIN_AGREEMENT and (sim >= MIN_NAME_WITH_MINUTES or
                                                                  (share >= 0.8 and agree >= 5)):
                        cands.append((1, 100 * share + sim, pid, name))
                        continue
                # A name alone, when the league minutes (if both have any) don't contradict it
                if sim >= MIN_NAME_ONLY and (not (q_min and p_min) or agree >= 1):
                    name_only.append((sim, name))
            # Without minutes, only an unambiguous name ("Gabriel" fits three Arsenal players)
            best = [x for x in name_only if x[0] == max(name_only)[0]] if name_only else []
            if len(best) == 1:
                cands.append((0, best[0][0], pid, best[0][1]))
        # Greedy one-to-one: minutes-backed pairs first, best combined score first
        used_p, used_n = set(), set()
        for _, _, pid, name in sorted(cands, reverse=True):
            if pid not in used_p and name not in used_n:
                out[(season, team, int(pid))] = name
                used_p.add(pid)
                used_n.add(name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--cache-dir", default=os.path.join(".cache", "transfermarkt"))
    args = parser.parse_args()
    paths = download(args.cache_dir)

    hist = pm.load_history()
    hist["date"] = london_date(hist["kickoff_time"])
    seasons = sorted(hist["season"].unique())
    pl_teams = {s: set(hist.loc[hist["season"] == s, "team"]) for s in seasons}

    games = pd.read_csv(paths["games"], usecols=[
        "game_id", "competition_id", "season", "round", "date", "home_club_id", "away_club_id",
        "home_club_goals", "away_club_goals", "home_club_name", "away_club_name"])
    games = games[games["season"] >= FIRST_SEASON - 1]
    clubs = club_map(games, hist)
    print(f"clubs mapped: {len(clubs)} -> {sorted(set(clubs.values()))}")

    # ---- matches: one row per Premier League club per non-PL game
    g = games[games["competition_id"].isin(COMPETITIONS)].copy()
    g = g[(g["season"] >= FIRST_SEASON) | (g["date"] >= LATE_2019_20)]
    g["label"] = g["season"].map(season_label)
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        part = g.copy()
        part["team"] = part[f"{side}_club_id"].map(clubs)
        # 2019-20's late matches count for the clubs of 2020-21
        teams_of = part["label"].map(lambda s: pl_teams.get(s if s in pl_teams else season_label(FIRST_SEASON), set()))
        part = part[[t in ts for t, ts in zip(part["team"], teams_of)]]
        part = part.assign(club_id=part[f"{side}_club_id"], opponent=part[f"{other}_club_name"],
                           was_home=side == "home", goals_for=part[f"{side}_club_goals"],
                           goals_against=part[f"{other}_club_goals"])
        rows.append(part)
    m = pd.concat(rows, ignore_index=True)
    m["match_id"] = "tm" + m["game_id"].astype(str)
    m["competition"] = m["competition_id"].map(COMPETITIONS)
    m["kickoff_time"] = pd.to_datetime(m["date"]).dt.tz_localize("UTC") + pd.Timedelta(hours=KICKOFF_HOUR)
    m["status"] = "finished"
    m = m.drop(columns="season").rename(columns={"label": "season"}).sort_values(["kickoff_time", "match_id", "team"])
    print("matches per season/competition:\n" + m.groupby(["season", "competition"]).size().unstack(fill_value=0).to_string())

    # ---- squads: lineups (who started / sat on the bench) + appearances (minutes, goals, assists)
    ids = set(m["game_id"])
    lu = read_filtered(paths["game_lineups"], lambda c: c["game_id"].isin(ids),
                       usecols=["game_id", "player_id", "club_id", "player_name", "type"])
    lu = lu.drop_duplicates(["game_id", "player_id"])
    ap = read_filtered(paths["appearances"], lambda c: c["game_id"].isin(ids) | (c["competition_id"] == "GB1"),
                       usecols=["game_id", "player_id", "player_club_id", "player_name", "competition_id",
                                "date", "goals", "assists", "minutes_played"])
    cup_ap = ap[ap["game_id"].isin(ids)].drop_duplicates(["game_id", "player_id"]).rename(
        columns={"player_club_id": "club_id_ap", "player_name": "name_ap"})
    squads = lu.merge(cup_ap[["game_id", "player_id", "club_id_ap", "name_ap", "minutes_played", "goals", "assists"]],
                      on=["game_id", "player_id"], how="outer")
    # Games without lineups: the players who played, "started" as in build_features' fallback
    squads["club_id"] = squads["club_id"].fillna(squads["club_id_ap"]).astype(int)
    squads["player_name"] = squads["player_name"].fillna(squads["name_ap"])
    squads["minutes"] = squads["minutes_played"].fillna(0)
    squads["started"] = np.where(squads["type"].notna(), (squads["type"] == "starting_lineup").astype(float),
                                 (squads["minutes"] >= 60).astype(float))
    squads[["goals", "assists"]] = squads[["goals", "assists"]].fillna(0)
    squads = squads.merge(m[["game_id", "club_id", "match_id", "team", "season"]], on=["game_id", "club_id"])
    print(f"squad rows {len(squads)}; matches with squads {squads['match_id'].nunique()} of {m['match_id'].nunique()}")

    # ---- player names
    gb1 = ap[ap["competition_id"] == "GB1"].merge(games[["game_id", "season"]], on="game_id")
    gb1 = gb1.assign(season=gb1["season"].map(season_label), team=gb1["player_club_id"].map(clubs),
                     minutes=gb1["minutes_played"])
    gb1 = gb1[gb1["team"].notna()]
    cup_only = squads[["season", "team", "player_id", "player_name"]].drop_duplicates().assign(date=np.nan, minutes=np.nan)
    cup_only["season"] = cup_only["season"].replace({season_label(FIRST_SEASON - 1): season_label(FIRST_SEASON)})
    tm = pd.concat([gb1[["season", "team", "player_id", "player_name", "date", "minutes"]],
                    cup_only[~cup_only.set_index(["season", "team", "player_id"]).index.isin(
                        gb1.set_index(["season", "team", "player_id"]).index)]], ignore_index=True)
    names = match_players(tm, hist[["season", "team", "name", "date", "minutes"]])
    by_pid = {}
    for (season, _, pid), name in sorted(names.items()):
        by_pid[pid] = name   # latest season wins: the fallback for a squad outside its PL club
    key_season = squads["season"].replace({season_label(FIRST_SEASON - 1): season_label(FIRST_SEASON)})
    squads["name"] = [names.get((s, t, int(p)), by_pid.get(int(p))) for s, t, p in
                      zip(key_season, squads["team"], squads["player_id"])]

    played = squads[squads["minutes"] > 0]
    share = played["name"].notna().groupby(played["season"]).mean().round(3).to_dict()
    mins = (played["minutes"] * played["name"].notna()).groupby(played["season"]).sum() / played.groupby("season")["minutes"].sum()
    print(f"appearances with a FPL name, by season: {share}; share of minutes: {mins.round(3).to_dict()}")
    miss = played[played["name"].isna()].groupby(["team", "player_name"])["minutes"].sum().nlargest(15)
    print("most minutes without a FPL name:\n" + miss.to_string())

    os.makedirs("data", exist_ok=True)
    m[["match_id", "season", "competition", "round", "kickoff_time", "team", "opponent", "was_home",
       "goals_for", "goals_against", "status"]].to_csv(OUT_MATCHES, index=False)
    apps = squads.rename(columns={"player_id": "source_id", "player_name": "source_name"})
    apps["source_id"] = apps["source_id"].astype(int)
    apps = apps.astype({"started": int, "minutes": int, "goals": int, "assists": int})
    apps.sort_values(["match_id", "team", "started", "minutes"], ascending=[True, True, False, False])[
        ["match_id", "team", "name", "source_id", "source_name", "started", "minutes", "goals", "assists"]
    ].to_csv(OUT_APPS, index=False)
    print(f"wrote {OUT_MATCHES} ({len(m)} rows) and {OUT_APPS} ({len(apps)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
