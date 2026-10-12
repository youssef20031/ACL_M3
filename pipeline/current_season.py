"""
Build the current season's per-match player file from the FPL API.

    python -m pipeline.current_season            # refresh if new results exist, else status only
    python -m pipeline.current_season --full     # always re-fetch every player's history
    python -m pipeline.current_season --status-only

Writes to LIVE_DATA_DIR (default data/live):
  FPL_YYYY_YYYY.csv           one row per player per finished fixture, same 46 columns and
                              conventions as the historical FPL_2025_2026.csv
  player_status_YYYY_YY.csv   current availability: status, news, chance of playing, price, form
  xp_snapshots.csv            FPL's expected points per player per gameweek, captured before
                              each deadline (the API only exposes the upcoming gameweek's value)
  meta.json                   season, finished fixtures included, timestamps

Only fixtures FPL marks `finished` (final, bonus confirmed) are included, so rows never
change after they're written.
"""
import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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

STATUS_COLUMNS = [
    "element", "code", "name", "web_name", "team", "position", "status",
    "chance_of_playing_this_round", "chance_of_playing_next_round", "news", "news_added",
    "now_cost", "selected_by_percent", "form", "ep_next", "fetched_at",
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


# ------------------------------------------------------------------ builders

def build_status(bootstrap: Dict[str, Any]) -> pd.DataFrame:
    teams = {t["id"]: graph_team_name(t["name"]) for t in bootstrap["teams"]}
    fetched_at = _now()
    rows = [
        {
            "element": el["id"], "code": el["code"], "name": player_name(el),
            "web_name": el["web_name"], "team": teams.get(el["team"]),
            "position": POSITIONS[el["element_type"]], "status": el["status"],
            "chance_of_playing_this_round": el.get("chance_of_playing_this_round"),
            "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
            "news": el.get("news") or "", "news_added": el.get("news_added"),
            "now_cost": el["now_cost"], "selected_by_percent": el.get("selected_by_percent"),
            "form": el.get("form"), "ep_next": el.get("ep_next"), "fetched_at": fetched_at,
        }
        for el in bootstrap["elements"] if el["element_type"] in POSITIONS
    ]
    return pd.DataFrame(rows, columns=STATUS_COLUMNS).sort_values(["team", "position", "name"])


def update_xp_snapshots(bootstrap: Dict[str, Any], existing: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Record ep_next for the upcoming gameweek; the last capture before its deadline wins."""
    cols = ["element", "GW", "xP", "captured_at"]
    snapshots = existing if existing is not None else pd.DataFrame(columns=cols)
    event = next_event(bootstrap)
    if event is None:
        return snapshots
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) >= deadline:
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
