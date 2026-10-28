"""
Build the current season's per-match player file from the FPL API.

    python -m pipeline.current_season            # refresh if new results exist, else status only
    python -m pipeline.current_season --full     # always re-fetch every player's history
    python -m pipeline.current_season --status-only

Writes to LIVE_DATA_DIR (default data/live):
  FPL_YYYY_YYYY.csv           one row per player per finished fixture, same 46 columns and
                              conventions as the historical FPL_2025_2026.csv
  player_status_YYYY_YY.csv   current availability: status, news, chance of playing, price, form,
                              this gameweek's transfers in/out and price change, set-piece orders
  xp_snapshots.csv            FPL's expected points per player per gameweek, captured before
                              each deadline (the API only exposes the upcoming gameweek's value)
  element_snapshots/<season>/gwNN.csv
                              every capture before gameweek NN's deadline of each player's
                              availability, transfer, price and set-piece state; the API only
                              shows current values, so this is the only record of what was
                              known before past deadlines
  meta.json                   season, finished fixtures included, timestamps, and per gameweek
                              its deadline, whether it's finished and FPL's average and
                              highest manager score (`events`)
  odds/                       football-data.co.uk's bookmaker odds for the v4 points model
                              (refresh_odds, called by pipeline.predict_next_gw and
                              pipeline.retrain, not by run(); $ODDS_DIR overrides the folder)

Only fixtures FPL marks `finished` (final, bonus confirmed) are included, so rows never
change after they're written.
"""
import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from pipeline.fpl_api import FPLClient
from pipeline.team_names import graph_team_name

logger = logging.getLogger(__name__)

LIVE_DATA_DIR = os.getenv("LIVE_DATA_DIR", os.path.join("data", "live"))

# Column order of the historical season files (e.g. FPL_2025_2026.csv)
CSV_COLUMNS = [
    "name", "position", "team", "xP", "assists", "bonus", "bps", "clean_sheets", "creativity",
    "element", "expected_assists", "expected_goal_involvements", "expected_goals",
    "expected_goals_conceded", "fixture", "goals_conceded", "goals_scored", "ict_index",
    "influence", "kickoff_time", "minutes", "modified", "opponent_team", "own_goals",
    "penalties_missed", "penalties_saved", "red_cards", "round", "saves", "selected", "starts",
    "team_a_score", "team_h_score", "threat", "total_points", "transfers_balance",
    "transfers_in", "transfers_out", "value", "was_home", "yellow_cards",
    "clearances_blocks_interceptions", "defensive_contribution", "recoveries", "tackles", "GW",
]

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}  # 5 = assistant-manager chip, skipped

# Transfers in/out and price change since the last deadline, i.e. for the upcoming gameweek
MARKET_COLUMNS = ["transfers_in_event", "transfers_out_event", "cost_change_event"]
# 1 = first choice; empty when FPL lists no order for the player
SET_PIECE_COLUMNS = ["penalties_order", "corners_and_indirect_freekicks_order", "direct_freekicks_order"]

STATUS_COLUMNS = [
    "element", "code", "name", "web_name", "team", "position", "status",
    "chance_of_playing_this_round", "chance_of_playing_next_round", "news", "news_added",
    "now_cost", "selected_by_percent", "form", "ep_next", *MARKET_COLUMNS, *SET_PIECE_COLUMNS,
    "fetched_at",
]

ELEMENT_SNAPSHOT_DIR = "element_snapshots"
ELEMENT_SNAPSHOT_COLUMNS = [
    "element", "code", "GW", "captured_at", "team", "position", "status",
    "chance_of_playing_this_round", "chance_of_playing_next_round", "news", "news_added",
    "now_cost", "selected_by_percent", "ep_next", *MARKET_COLUMNS, *SET_PIECE_COLUMNS,
]


# ------------------------------------------------------------------ helpers

def season_label(bootstrap: Dict[str, Any]) -> str:
    """'2026-27' from the first gameweek's deadline."""
    year = int(bootstrap["events"][0]["deadline_time"][:4])
    return f"{year}-{str(year + 1)[2:]}"


def season_file_stem(season: str) -> str:
    start = int(season[:4])
    return f"{start}_{start + 1}"  # FPL_2026_2027.csv


def player_name(element: Dict[str, Any]) -> str:
    return f"{element['first_name']} {element['second_name']}".strip()


def next_event(bootstrap: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return next((e for e in bootstrap["events"] if e.get("is_next")), None)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _deadline_passed(event: Dict[str, Any]) -> bool:
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    return datetime.now(timezone.utc) >= deadline


EVENT_FIELDS = ("id", "deadline_time", "finished", "data_checked", "average_entry_score", "highest_score")


def event_summaries(bootstrap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per gameweek: deadline, finished flags and FPL's average/highest manager score
    (0 until FPL computes them, which is stored as None)."""
    out = []
    for e in bootstrap["events"]:
        row = {k: e.get(k) for k in EVENT_FIELDS}
        for k in ("average_entry_score", "highest_score"):
            row[k] = row[k] or None
        out.append(row)
    return out


# ------------------------------------------------------------------ builders

def _element_state(el: Dict[str, Any], teams: Dict[int, str]) -> Dict[str, Any]:
    """One player's availability, price, transfer and set-piece fields from the bootstrap."""
    return {
        "element": el["id"], "code": el["code"], "team": teams.get(el["team"]),
        "position": POSITIONS[el["element_type"]], "status": el["status"],
        "chance_of_playing_this_round": el.get("chance_of_playing_this_round"),
        "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
        "news": el.get("news") or "", "news_added": el.get("news_added"),
        "now_cost": el["now_cost"], "selected_by_percent": el.get("selected_by_percent"),
        "ep_next": el.get("ep_next"),
        **{col: el.get(col) for col in MARKET_COLUMNS + SET_PIECE_COLUMNS},
    }


def build_status(bootstrap: Dict[str, Any]) -> pd.DataFrame:
    teams = {t["id"]: graph_team_name(t["name"]) for t in bootstrap["teams"]}
    fetched_at = _now()
    rows = [
        {**_element_state(el, teams), "name": player_name(el), "web_name": el["web_name"],
         "form": el.get("form"), "fetched_at": fetched_at}
        for el in bootstrap["elements"] if el["element_type"] in POSITIONS
    ]
    return pd.DataFrame(rows, columns=STATUS_COLUMNS).sort_values(["team", "position", "name"])


def element_snapshot_path(out_dir: str, season: str, gameweek: int) -> str:
    return os.path.join(out_dir, ELEMENT_SNAPSHOT_DIR, season, f"gw{gameweek:02d}.csv")


def update_element_snapshots(bootstrap: Dict[str, Any], out_dir: str, season: str) -> Optional[str]:
    """Append this run's capture of every player's state to the upcoming gameweek's snapshot
    file. Every capture is kept, so the state at any time before the deadline (such as the last
    prediction run's) can be recovered. Returns the file's path, or None after the deadline."""
    event = next_event(bootstrap)
    if event is None or _deadline_passed(event):
        return None
    teams = {t["id"]: graph_team_name(t["name"]) for t in bootstrap["teams"]}
    captured_at = _now()
    fresh = pd.DataFrame([
        {**_element_state(el, teams), "GW": event["id"], "captured_at": captured_at}
        for el in bootstrap["elements"] if el["element_type"] in POSITIONS
    ], columns=ELEMENT_SNAPSHOT_COLUMNS)
    path = element_snapshot_path(out_dir, season, event["id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = _read_csv(path)
    _write_csv(fresh if existing is None else pd.concat([existing, fresh], ignore_index=True), path)
    return path


def update_xp_snapshots(bootstrap: Dict[str, Any], existing: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Record ep_next for the upcoming gameweek; the last capture before its deadline wins."""
    cols = ["element", "GW", "xP", "captured_at"]
    snapshots = existing if existing is not None else pd.DataFrame(columns=cols)
    event = next_event(bootstrap)
    if event is None:
        return snapshots
    if _deadline_passed(event):
        return snapshots  # too late: this gameweek's value is already frozen
    fresh = pd.DataFrame([
        {"element": el["id"], "GW": event["id"], "xP": float(el.get("ep_next") or 0.0), "captured_at": _now()}
        for el in bootstrap["elements"] if el["element_type"] in POSITIONS
    ], columns=cols)
    keep = snapshots[snapshots["GW"] != event["id"]] if len(snapshots) else snapshots
    return pd.concat([keep, fresh], ignore_index=True)


def build_season_rows(bootstrap: Dict[str, Any], fixtures: List[Dict[str, Any]],
                      summaries: Dict[int, Dict[str, Any]],
                      xp: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per player per finished fixture, in the historical CSV format."""
    teams = {t["id"]: graph_team_name(t["name"]) for t in bootstrap["teams"]}
    elements = {el["id"]: el for el in bootstrap["elements"]}
    fixtures_by_id = {f["id"]: f for f in fixtures}
    finished = {f["id"] for f in fixtures if f.get("finished")}
    xp_lookup = {}
    if xp is not None and len(xp):
        xp_lookup = {(int(r.element), int(r.GW)): float(r.xP) for r in xp.itertuples(index=False)}

    rows = []
    for element_id, summary in summaries.items():
        el = elements.get(element_id)
        if el is None or el["element_type"] not in POSITIONS:
            continue
        for h in summary.get("history", []):
            fixture = fixtures_by_id.get(h["fixture"])
            if h["fixture"] not in finished or fixture is None:
                continue
            row = {col: h.get(col) for col in CSV_COLUMNS if col in h}
            # The player's team in *that* match (correct for mid-season transfers)
            row["team"] = teams.get(fixture["team_h"] if h["was_home"] else fixture["team_a"])
            row["name"] = player_name(el)
            row["position"] = POSITIONS[el["element_type"]]
            row["GW"] = h["round"]
            row["xP"] = xp_lookup.get((element_id, h["round"]))
            rows.append(row)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    if df.empty:
        return df
    for col in ("was_home", "modified"):
        df[col] = df[col].astype(bool)
    return df.sort_values(["GW", "kickoff_time", "fixture", "element"]).reset_index(drop=True)


# ------------------------------------------------------------------ orchestration

def _read_csv(path: str) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if os.path.exists(path) else None


def _write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run(out_dir: str = LIVE_DATA_DIR, full: bool = False, status_only: bool = False,
        client: Optional[FPLClient] = None) -> Dict[str, Any]:
    """Refresh the live files. Returns the new meta dict (with `refreshed` = history re-fetched)."""
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}

    own_client = client is None
    client = client or FPLClient()
    try:
        bootstrap = client.bootstrap()
        season = season_label(bootstrap)
        stem = season_file_stem(season)
        season_csv = os.path.join(out_dir, f"FPL_{stem}.csv")
        status_csv = os.path.join(out_dir, f"player_status_{season.replace('-', '_')}.csv")
        xp_csv = os.path.join(out_dir, "xp_snapshots.csv")

        _write_csv(build_status(bootstrap), status_csv)
        xp = update_xp_snapshots(bootstrap, _read_csv(xp_csv))
        _write_csv(xp, xp_csv)
        update_element_snapshots(bootstrap, out_dir, season)

        refreshed = False
        if not status_only:
            fixtures = client.fixtures()
            finished_ids = sorted(f["id"] for f in fixtures if f.get("finished"))
            new_results = finished_ids != meta.get("finished_fixtures") or meta.get("season") != season
            if full or new_results or not os.path.exists(season_csv):
                ids = [el["id"] for el in bootstrap["elements"] if el["element_type"] in POSITIONS]
                logger.info("Fetching match history for %d players", len(ids))
                summaries = client.element_summaries(
                    ids, progress=lambda d, n: logger.info("  %d/%d players", d, n))
                season_df = build_season_rows(bootstrap, fixtures, summaries, xp)
                _write_csv(season_df, season_csv)
                meta.update(
                    season=season, finished_fixtures=finished_ids,
                    rows=len(season_df), gameweeks=sorted(season_df["GW"].unique().tolist()) if len(season_df) else [],
                    history_fetched_at=_now(),
                )
                refreshed = True
            else:
                logger.info("No newly finished fixtures; history unchanged")

        event = next_event(bootstrap)
        meta.update(
            season=season, status_fetched_at=_now(),
            next_gameweek=event["id"] if event else None,
            next_deadline=event["deadline_time"] if event else None,
            events=event_summaries(bootstrap),
            files={"season": os.path.basename(season_csv), "status": os.path.basename(status_csv),
                   "xp": os.path.basename(xp_csv)},
        )
        tmp = f"{meta_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, meta_path)
        return {**meta, "refreshed": refreshed}
    finally:
        if own_client:
            client.close()


# ------------------------------------------------------------------ bookmaker odds (v4 model)

ODDS_FIRST_SEASON = 2020                # the first season the v4 model trains on
ODDS_MIN_AGE = timedelta(minutes=30)    # a file fetched more recently isn't downloaded again


def odds_dir(live_dir: str = LIVE_DATA_DIR) -> str:
    """Where football-data.co.uk's files live: $ODDS_DIR if set, else LIVE_DATA_DIR/odds (on the
    VM's data volume, next to the files they're used with)."""
    return os.getenv("ODDS_DIR") or os.path.join(live_dir, "odds")


def odds_seasons(season: str) -> List[str]:
    """2020-21 .. `season`."""
    return [f"{y}-{str(y + 1)[2:]}" for y in range(ODDS_FIRST_SEASON, int(season[:4]) + 1)]


def _fetch_checked(fetch: Callable[[str], str], directory: str) -> str:
    """Run one of ml.odds' fetchers into a staging folder and move its file into `directory` only
    if ml.odds can read it, so an error page or a changed format never replaces a good file."""
    from ml import odds as od

    staging = os.path.join(directory, ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        path = fetch(staging)
        od.load_odds(staging)       # raises on a file it can't parse
        final = os.path.join(directory, os.path.basename(path))
        os.replace(path, final)
        return final
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def refresh_odds(live_dir: str, season: str, fixtures: bool = True,
                 now: Optional[datetime] = None) -> Dict[str, str]:
    """Bring odds_dir(live_dir) up to date for the v4 points model: the current season's file and
    the upcoming fixtures.csv on every call unless fetched within ODDS_MIN_AGE, and each earlier
    season's file once (again if that copy predates the season's end, July of its second year).
    So a normal run makes two small requests, and the first one also fetches the past seasons,
    which the weekly retrain needs.

    Never raises: a failed download is logged and keeps the previous file (or none: the odds
    features are then NaN). Returns {file: "fetched" | "fresh" | "failed: ..."}; past seasons
    that are already final aren't listed."""
    from ml import odds as od

    now = now or datetime.now(timezone.utc)
    directory = odds_dir(live_dir)
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:   # e.g. $ODDS_DIR on a read-only mount: predict with what's there
        logger.warning("Can't create the odds folder %s: %s", directory, exc)
        return {"odds_dir": f"failed: {exc}"[:200]}
    jobs = []   # (file name, fetcher, keep the copy if modified after this time)
    for s in odds_seasons(season):
        if s == season:
            keep_after = now - ODDS_MIN_AGE
        else:
            keep_after = datetime(int(s[:4]) + 1, 7, 1, tzinfo=timezone.utc)
        jobs.append((f"E0_{od.season_code(s)}.csv", lambda d, s=s: od.fetch_season(s, d), keep_after, s == season))
    if fixtures:
        jobs.append((od.FIXTURES_FILE, od.fetch_fixtures, now - ODDS_MIN_AGE, True))

    result: Dict[str, str] = {}
    for name, fetch, keep_after, report in jobs:
        path = os.path.join(directory, name)
        if os.path.exists(path) and os.path.getmtime(path) >= keep_after.timestamp():
            if report:
                result[name] = "fresh"
            continue
        try:
            _fetch_checked(fetch, directory)
            result[name] = "fetched"
        except Exception as exc:  # noqa: BLE001 - odds are optional inputs
            logger.warning("Couldn't fetch the odds file %s: %s", name, exc)
            result[name] = f"failed: {exc}"[:200]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh current-season FPL data from the FPL API")
    parser.add_argument("--out-dir", default=LIVE_DATA_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="Re-fetch every player's history")
    mode.add_argument("--status-only", action="store_true", help="Only refresh news/availability and xP")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta = run(args.out_dir, full=args.full, status_only=args.status_only)
    print(json.dumps({k: v for k, v in meta.items() if k != "finished_fixtures"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
