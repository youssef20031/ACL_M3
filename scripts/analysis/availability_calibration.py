"""
How much of a flagged player's points FPL's chance of playing really leaves. Production multiplies
each prediction by chance / 100 (services.season_service.chance_of_playing); this measures what
the multiplier should be for each flag (25/50/75% doubts, and out: i/s/u/n), from the snapshot
before each deadline (data/fpl_snapshots.csv, ml/snapshots.py's timing rule) and what happened.

    python -m scripts.analysis.availability_calibration [--boot 300] [--out res.json]

A flag's factor is indirectly standardized: its players' actual points (or appearances) over
what unflagged players (status a) of the same stratum averaged. Strata are known before the
deadline: position, fixtures in the gameweek, minutes in the player's previous gameweek (none, 0,
1-59, 60+) and in his previous three (0, 1-134, 135-269, 270+). So a doubtful regular is compared
with available regulars, and a doubt returning from injury with players who also didn't play
last week. Rows: player-gameweeks (a double gameweek's fixtures summed), seasons with snapshots.
The CI resamples gameweeks.

Splits: "regulars" (60+ minutes the gameweek before, the players the model picks), early vs late
seasons (stability), and the snapshot's age at the deadline: vaastav's commits are a median 1.5-2
days old, while production's status file is hours old, so "fresh" (< 48 h) is the closer match.
"""
import argparse
import json

import numpy as np
import pandas as pd

from ml import points_model as pm
from ml import snapshots as sn

FLAGS = ["d75", "d50", "d25", "out"]
LITERAL = {"d75": 0.75, "d50": 0.5, "d25": 0.25, "out": 0.0}
MEASURES = {"points": "total_points", "played": "played", "min60": "min60"}


def player_gameweeks(history_csv: str = pm.HISTORY_CSV, snapshots_csv: str = sn.SNAPSHOTS_CSV) -> pd.DataFrame:
    """One row per player-gameweek with a snapshot before its deadline: outcome, flag, stratum."""
    h = pd.read_csv(history_csv, low_memory=False, usecols=["season_x", "element", "GW", "kickoff_time", "position",
                                                            "minutes", "total_points"])
    h = h.rename(columns={"season_x": "season"})
    h = h[h["position"].isin(["GK", "GKP", "DEF", "MID", "FWD"])].replace({"position": {"GKP": "GK"}})
    win = sn.snapshot_windows(sn.gameweek_deadlines(h), sn.load_snapshots(snapshots_csv))
    pg = (h.groupby(["season", "GW", "element"])
          .agg(position=("position", "first"), fixtures=("minutes", "size"), minutes=("minutes", "sum"),
               total_points=("total_points", "sum"))
          .reset_index().sort_values(["season", "element", "GW"], kind="mergesort"))
    by = pg.groupby(["season", "element"])["minutes"]
    pg["prev1"] = by.shift(1)
    pg["prev3"] = by.transform(lambda m: m.shift(1).rolling(3, min_periods=1).sum())
    pg["played"] = (pg["minutes"] > 0).astype(float)
    pg["min60"] = (pg["minutes"] >= 60).astype(float)

    snaps = sn.load_snapshots(snapshots_csv)[["season", "sha", "element", "status", "chance_of_playing_next_round"]]
    snaps = snaps.drop_duplicates(["season", "sha", "element"])
    rows = pg.merge(win[["season", "GW", "sha", "deadline", "captured_at"]].dropna(subset=["sha"]), on=["season", "GW"])
    rows = rows.merge(snaps, on=["season", "sha", "element"])
    rows["age_h"] = (rows["deadline"] - rows["captured_at"]).dt.total_seconds() / 3600
    factor = sn.cop_factor(rows["status"], rows["chance_of_playing_next_round"])
    status = rows["status"].fillna("a").str.lower()
    rows["flag"] = np.select([factor == 0, factor == 0.25, factor == 0.5, factor == 0.75, status == "a"],
                             ["out", "d25", "d50", "d75", "a"], default="other")
    rows["stratum"] = (rows["position"] + "|" + rows["fixtures"].astype(str) + "|"
                       + pd.cut(rows["prev1"].fillna(-1), [-2, -0.5, 0.5, 59.5, 1e9],
                                labels=["none", "0", "1-59", "60+"]).astype(str) + "|"
                       + pd.cut(rows["prev3"].fillna(0), [-1, 0.5, 134.5, 269.5, 1e9],
                                labels=["0", "1-134", "135-269", "270+"]).astype(str))
    return rows


def factors(rows: pd.DataFrame) -> dict:
    """{flag: {measure: actual / expected-if-unflagged, "n": rows}} for one set of rows."""
    base = rows[rows["flag"] == "a"].groupby("stratum")[list(MEASURES.values())].mean()
    out = {}
    for flag in FLAGS:
        f = rows[rows["flag"] == flag]
        expected = base.reindex(f["stratum"]).to_numpy()
        keep = ~np.isnan(expected).any(axis=1)
        res = {"n": int(keep.sum())}
        for i, (name, col) in enumerate(MEASURES.items()):
            denominator = expected[keep, i].sum()
            res[name] = float(f[col].to_numpy()[keep].sum() / denominator) if denominator > 0 else None
        out[flag] = res
    return out


def bootstrap(rows: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    """95% CIs of each flag's points factor, resampling (season, GW) clusters."""
    rng = np.random.default_rng(seed)
    gws = rows[["season", "GW"]].drop_duplicates().reset_index(drop=True)
    groups = dict(tuple(rows.groupby(["season", "GW"])))
    draws = {flag: [] for flag in FLAGS}
    for _ in range(n_boot):
        pick = gws.iloc[rng.integers(0, len(gws), len(gws))]
        sample = pd.concat([groups[(s, g)] for s, g in pick.itertuples(index=False)], ignore_index=True)
        for flag, res in factors(sample).items():
            if res["points"] is not None:
                draws[flag].append(res["points"])
    return {flag: [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)] if d else None
            for flag, d in draws.items()}


def report(rows: pd.DataFrame, n_boot: int) -> dict:
    seasons = sorted(rows["season"].unique())
    half = len(seasons) // 2
    splits = {
        "all": rows,
        "regulars (60+ the gameweek before)": rows[rows["prev1"] >= 60],
        f"seasons {seasons[0]}..{seasons[half - 1]}": rows[rows["season"].isin(seasons[:half])],
        f"seasons {seasons[half]}..{seasons[-1]}": rows[rows["season"].isin(seasons[half:])],
        "fresh snapshot (< 48 h before the deadline)": rows[rows["age_h"] < 48],
        "older snapshot (48 h+)": rows[rows["age_h"] >= 48],
    }
    out = {name: factors(part) for name, part in splits.items()}
    if n_boot:
        out["all"]["ci_points"] = bootstrap(rows, n_boot)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--boot", type=int, default=300, help="bootstrap resamples (0 to skip)")
    parser.add_argument("--out", help="also write the results as JSON")
    args = parser.parse_args()
    rows = player_gameweeks()
    res = report(rows, args.boot)
    print(f"{len(rows):,} player-gameweeks with a snapshot, {rows['season'].nunique()} seasons\n")
    for name, per_flag in res.items():
        print(name)
        for flag in FLAGS:
            r = per_flag[flag]
            ci = per_flag.get("ci_points", {}).get(flag)
            fmt = lambda v: "  n/a" if v is None else f"{v:5.2f}"  # noqa: E731
            print(f"  {flag:4} literal {LITERAL[flag]:.2f}  points {fmt(r['points'])}"
                  + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
                  + f"  played {fmt(r['played'])}  60+ min {fmt(r['min60'])}  n={r['n']:,}")
        print()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
