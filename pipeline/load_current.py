"""
Load the current season (data/live/, written by pipeline.current_season) into Neo4j.

    python -m pipeline.load_current            # uses the NEO4J_* credentials in .env

Incremental and idempotent: history is never touched (apart from a one-time
conversion of old string kickoff times to DateTimes) and running it twice gives the
same graph. Steps:

  1. Fixtures from the FPL API, one node per match keyed f"{fpl_id}-{season}", with
     both relationship styles: PLAYS_HOME/PLAYS_AWAY (fixture service, ML) and
     HOME_TEAM/AWAY_TEAM + PART_OF -> Gameweek -> Season (historical-style queries).
     The schedule nodes loaded from FPL_2026_2027_fixtures_fixed.csv
     ("fixture-2026-27-...") are then deleted.
  2. Players from the status file and the season CSV: FPL ids, photo code,
     availability, current position (PLAYS_POSITION) and current club (PLAYS_FOR).
  3. PLAYED_IN rows for every finished match in the season CSV.
  4. SEASON_STATS for the season, rebuilt from the CSV.
"""
import glob
import json
import logging
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from graph.data_loader import FPLDataLoader
from pipeline.current_season import LIVE_DATA_DIR
from pipeline.team_names import graph_team_name
from utils.text import normalize_name

logger = logging.getLogger(__name__)

# "u" (unavailable) players whose news says they've left the club: transferred,
# loaned out, released, or a loanee gone back to his parent club.
LEFT_CLUB = re.compile(
    r"\b(has joined|joined|on loan|loan|has departed|departed|has left|left the club|"
    r"released|returned to|has returned|transferred|permanently|contract terminated|"
    r"mutual consent)\b",
    re.IGNORECASE,
)

STATUS_FIELDS = (
    "status", "news", "news_added",
    "chance_of_playing_this_round", "chance_of_playing_next_round",
)


# ------------------------------------------------------------------ input files

def _season_from_filename(path: str) -> str:
    """'.../FPL_2026_2027.csv' -> '2026-27'."""
    m = re.search(r"FPL_(\d{4})_(\d{4})\.csv$", os.path.basename(path))
    if not m:
        raise ValueError(f"Not a season file: {path}")
    return f"{m.group(1)}-{m.group(2)[2:]}"


def find_live_files(live_dir: str = LIVE_DATA_DIR) -> Optional[Dict[str, str]]:
    """Season label and the season/status file paths in live_dir, or None if absent."""
    meta_path = os.path.join(live_dir, "meta.json")
    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    files = meta.get("files") or {}
    season_csv = os.path.join(live_dir, files["season"]) if files.get("season") else None
    if not season_csv or not os.path.exists(season_csv):
        candidates = sorted(glob.glob(os.path.join(live_dir, "FPL_*_*.csv")))
        if not candidates:
            return None
        season_csv = candidates[-1]
    season = meta.get("season") or _season_from_filename(season_csv)
    status_csv = os.path.join(
        live_dir, files.get("status") or f"player_status_{season.replace('-', '_')}.csv")
    return {"season": season, "season_csv": season_csv,
            "status_csv": status_csv if os.path.exists(status_csv) else None}


def has_live_data(live_dir: str = LIVE_DATA_DIR) -> bool:
    return find_live_files(live_dir) is not None


# ------------------------------------------------------------------ pure builders

def _clean(value: Any) -> Any:
    """NaN -> None and numpy scalars -> Python, for Neo4j parameters."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def team_names_by_id(fpl_teams: Optional[Iterable[Any]] = None,
                     fixtures: Optional[List[Dict[str, Any]]] = None,
                     season_df: Optional[pd.DataFrame] = None) -> Dict[int, str]:
    """FPL team id -> graph team name.

    From the bootstrap `teams` list (or a ready {id: name} dict) when given, otherwise
    inferred from the season CSV: a row's `team` is the home side's name when
    `was_home`, else the away side's, for that row's fixture.
    """
    if fpl_teams is not None:
        if isinstance(fpl_teams, dict):
            return {int(k): graph_team_name(v) for k, v in fpl_teams.items()}
        return {int(t["id"]): graph_team_name(t["name"]) for t in fpl_teams}
    mapping: Dict[int, str] = {}
    if fixtures is None or season_df is None or season_df.empty:
        return mapping
    by_id = {f["id"]: f for f in fixtures}
    for row in season_df[["fixture", "was_home", "team"]].drop_duplicates().itertuples(index=False):
        fix = by_id.get(int(row.fixture))
        if fix is None:
            continue
        home = str(row.was_home).lower() in ("true", "1")
        mapping[int(fix["team_h"] if home else fix["team_a"])] = row.team
    return mapping


def fixture_rows(fixtures: List[Dict[str, Any]], teams: Dict[int, str], season: str) -> List[Dict[str, Any]]:
    """One property map per FPL fixture. Scores are kept only once FPL marks the match
    finished (bonus confirmed), matching the rows in the season CSV."""
    rows = []
    for fix in fixtures:
        home, away = teams.get(fix.get("team_h")), teams.get(fix.get("team_a"))
        if home is None or away is None:
            logger.warning("Fixture %s: unknown team id(s) %s/%s, skipped",
                           fix.get("id"), fix.get("team_h"), fix.get("team_a"))
            continue
        finished = bool(fix.get("finished"))
        status = "finished" if finished else ("in_progress" if fix.get("started") else "scheduled")
        event = fix.get("event")
        rows.append({
            "id": f"{fix['id']}-{season}",
            "fixture_id": int(fix["id"]),
            "season": season,
            "gameweek": int(event) if event is not None else None,
            "gw_id": f"{season}-GW{int(event)}" if event is not None else None,
            "kickoff_time": fix.get("kickoff_time"),
            "status": status,
            "home_score": fix.get("team_h_score") if finished else None,
            "away_score": fix.get("team_a_score") if finished else None,
            "home_team": home,
            "away_team": away,
            # FPL's fixture difficulty rating (1-5) for each side
            "home_difficulty": fix.get("team_h_difficulty"),
            "away_difficulty": fix.get("team_a_difficulty"),
        })
    return rows


def keeps_club(status: Optional[str], news: Optional[str]) -> bool:
    """Whether a player in the FPL status file still belongs to his listed club.

    Only "u" (unavailable) players whose news says they've gone lose it; "u" for other
    reasons (e.g. "not included in squad") still means registered with the club.
    """
    if status != "u":
        return True
    return not LEFT_CLUB.search(news or "")


def player_rows(status_df: Optional[pd.DataFrame], season_df: pd.DataFrame,
                season: str) -> List[Dict[str, Any]]:
    """One row per player for the player MERGE.

    Status-file players carry their FPL fields and `plays_for` (keep/replace PLAYS_FOR
    with `team`, or drop it). Players only in the season CSV (no longer in the FPL
    game) get position/team from their latest match and no current club.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    if status_df is not None:
        for r in status_df.to_dict("records"):
            name = r["name"]
            props = {field: _clean(r.get(field)) for field in STATUS_FIELDS}
            for field in ("chance_of_playing_this_round", "chance_of_playing_next_round"):
                if props[field] is not None:
                    props[field] = int(props[field])
            props["news"] = props["news"] or ""
            props.update(
                element_id=int(r["element"]),
                fpl_code=_clean(r.get("code")),
                status_updated_at=_clean(r.get("fetched_at")),
                name_normalized=normalize_name(name),
                last_team=r["team"],
                last_season=season,
            )
            if props["fpl_code"] is not None:
                props["fpl_code"] = int(props["fpl_code"])
            rows[name] = {
                "name": name, "position": r["position"], "team": r["team"],
                "plays_for": keeps_club(props["status"], props["news"]),
                "props": props,
            }

    if season_df is not None and not season_df.empty:
        latest = (season_df.sort_values(["GW", "kickoff_time"])
                  .groupby("name", sort=False).tail(1))
        for r in latest.to_dict("records"):
            if r["name"] in rows:
                continue
            rows[r["name"]] = {
                "name": r["name"], "position": r["position"], "team": r["team"],
                "plays_for": False,
                "props": {
                    "element_id": int(r["element"]),
                    "name_normalized": normalize_name(r["name"]),
                    "last_team": r["team"],
                    "last_season": season,
                },
            }
    return list(rows.values())


# ------------------------------------------------------------------ Cypher

GAMEWEEKS_QUERY = """
MERGE (s:Season {id: $season})
SET s.start_year = $start_year, s.end_year = $start_year + 1
WITH s
UNWIND $gameweeks AS n
MERGE (gw:Gameweek {id: $season + '-GW' + toString(n)})
SET gw.number = n, gw.season_id = $season
MERGE (gw)-[:IN_SEASON]->(s)
"""

FIXTURES_QUERY = """
UNWIND $batch AS fix
MERGE (f:Fixture {id: fix.id})
SET f.fixture_id = fix.fixture_id,
    f.season_id = fix.season,
    f.gameweek = fix.gameweek,
    f.matchday = fix.gameweek,
    f.kickoff_time = CASE WHEN fix.kickoff_time IS NULL THEN null ELSE datetime(fix.kickoff_time) END,
    f.status = fix.status,
    f.home_score = fix.home_score,
    f.away_score = fix.away_score,
    f.home_difficulty = fix.home_difficulty,
    f.away_difficulty = fix.away_difficulty
MERGE (home:Team {name: fix.home_team})
MERGE (away:Team {name: fix.away_team})
MERGE (f)-[:PLAYS_HOME]->(home)
MERGE (f)-[:PLAYS_AWAY]->(away)
MERGE (f)-[:HOME_TEAM]->(home)
MERGE (f)-[:AWAY_TEAM]->(away)
WITH f, fix
// A postponed match moves to another gameweek (or none, until it is rescheduled)
CALL (f, fix) {
  OPTIONAL MATCH (f)-[old:PART_OF]->(g:Gameweek)
  WHERE fix.gw_id IS NULL OR g.id <> fix.gw_id
  DELETE old
}
WITH f, fix
WHERE fix.gw_id IS NOT NULL
MATCH (gw:Gameweek {id: fix.gw_id})
MERGE (f)-[:PART_OF]->(gw)
"""

DELETE_CSV_SCHEDULE_QUERY = """
MATCH (f:Fixture) WHERE f.id STARTS WITH $prefix
DETACH DELETE f
RETURN count(f) AS deleted
"""

PLAYERS_QUERY = """
UNWIND $batch AS row
MERGE (p:Player {name: row.name})
SET p += row.props
WITH p, row
CALL (p, row) {
  OPTIONAL MATCH (p)-[old:PLAYS_POSITION]->(op:Position)
  WHERE op.code <> row.position
  DELETE old
}
MATCH (pos:Position {code: row.position})
MERGE (p)-[:PLAYS_POSITION]->(pos)
WITH p, row
CALL (p, row) {
  OPTIONAL MATCH (p)-[old:PLAYS_FOR]->(ot:Team)
  WHERE NOT row.plays_for OR ot.name <> row.team
  DELETE old
}
WITH p, row
WHERE row.plays_for
MERGE (t:Team {name: row.team})
MERGE (p)-[r:PLAYS_FOR]->(t)
SET r.season = $season
"""

# PLAYS_FOR means "in the current squad": drop it for everyone the FPL game no longer lists
DROP_STALE_PLAYS_FOR_QUERY = """
MATCH (p:Player)-[r:PLAYS_FOR]->(:Team)
WHERE NOT p.name IN $current
DELETE r
RETURN count(r) AS dropped
"""

COUNT_PLAYED_IN_QUERY = """
MATCH (:Fixture {season_id: $season})<-[r:PLAYED_IN]-(:Player)
RETURN count(r) AS rows
"""

DELETE_PLAYED_IN_QUERY = """
MATCH (:Fixture {season_id: $season})<-[r:PLAYED_IN]-(:Player)
DELETE r
"""


# ------------------------------------------------------------------ orchestration

def _fetch_from_api() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from pipeline.fpl_api import FPLClient

    with FPLClient() as client:
        return client.fixtures(), client.bootstrap()["teams"]


def load_current_season(conn, live_dir: str = LIVE_DATA_DIR,
                        fpl_fixtures: Optional[List[Dict[str, Any]]] = None,
                        fpl_teams: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """Load the current season from live_dir into the graph. Returns counts.

    fpl_fixtures is FPL's /fixtures/ list; when omitted it (and the team list) is
    fetched from the API. fpl_teams is the bootstrap `teams` list or an {id: name}
    dict; when omitted with fixtures given, team ids are inferred from the season CSV.
    """
    started = time.perf_counter()
    files = find_live_files(live_dir)
    if files is None:
        raise FileNotFoundError(f"No FPL_*.csv season file in {live_dir}")
    season = files["season"]

    raw_df = pd.read_csv(files["season_csv"], low_memory=False)
    status_df = pd.read_csv(files["status_csv"]) if files["status_csv"] else None
    if status_df is None:
        logger.warning("No player status file in %s; PLAYS_FOR is left unchanged", live_dir)

    # Fetch everything before writing, so an API failure leaves the graph untouched
    if fpl_fixtures is None:
        fpl_fixtures, api_teams = _fetch_from_api()
        fpl_teams = fpl_teams if fpl_teams is not None else api_teams
    teams = team_names_by_id(fpl_teams, fpl_fixtures, raw_df)
    fixtures = fixture_rows(fpl_fixtures, teams, season)
    if not fixtures:
        raise ValueError("No fixtures to load (empty FPL fixture list or unknown teams)")

    loader = FPLDataLoader(conn)
    counts: Dict[str, Any] = {"season": season}

    # 1. Season, gameweeks, fixtures
    gameweeks = sorted({f["gameweek"] for f in fixtures if f["gameweek"] is not None})
    conn.execute_write(GAMEWEEKS_QUERY, {"season": season, "start_year": int(season[:4]),
                                         "gameweeks": gameweeks})
    loader._run_batches(FIXTURES_QUERY, fixtures, "Fixtures")
    deleted = conn.execute_query(DELETE_CSV_SCHEDULE_QUERY, {"prefix": f"fixture-{season}-"})
    counts.update(
        fixtures=len(fixtures),
        finished_fixtures=sum(f["status"] == "finished" for f in fixtures),
        gameweeks=len(gameweeks),
        csv_schedule_fixtures_deleted=deleted[0]["deleted"] if deleted else 0,
        kickoff_strings_converted=loader.convert_kickoff_strings(),
    )

    # 2. Players
    players = player_rows(status_df, raw_df, season)
    _run_players(conn, players, season)
    counts["players"] = len(players)
    counts["plays_for"] = sum(p["plays_for"] for p in players)
    if status_df is not None:
        dropped = conn.execute_query(DROP_STALE_PLAYS_FOR_QUERY,
                                     {"current": [p["name"] for p in players if p["plays_for"]]})
        counts["plays_for_dropped_not_in_squad"] = dropped[0]["dropped"] if dropped else 0

    # 3. PLAYED_IN. MERGE is idempotent; if the graph holds rows the CSV doesn't
    # (e.g. a player renamed by FPL), rebuild the season's rows from scratch.
    df = loader.transform_columns(loader.season_file_frame(raw_df, season))
    performances = loader.performance_records(df)
    loader.create_performance_relationships(performances)
    in_graph = conn.execute_query(COUNT_PLAYED_IN_QUERY, {"season": season})
    if in_graph and in_graph[0]["rows"] != len(performances):
        logger.info("PLAYED_IN rows in graph (%s) != CSV rows (%s); rebuilding %s",
                    in_graph[0]["rows"], len(performances), season)
        conn.execute_write(DELETE_PLAYED_IN_QUERY, {"season": season})
        loader.create_performance_relationships(performances)
    counts["played_in"] = len(performances)

    # 4. SEASON_STATS
    season_stats = loader.build_season_stats(df) if len(df) else []
    loader.replace_season_stats(season, season_stats)
    counts["season_stats"] = len(season_stats)

    counts["seconds"] = round(time.perf_counter() - started, 1)
    logger.info("Loaded %s: %s", season, counts)
    return counts


def _run_players(conn, players: List[Dict[str, Any]], season: str, batch_size: int = 1000) -> None:
    for i in range(0, len(players), batch_size):
        conn.execute_write(PLAYERS_QUERY, {"batch": players[i:i + batch_size], "season": season})


def main() -> int:
    import argparse

    from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from graph.connection import Neo4jConnection

    parser = argparse.ArgumentParser(description="Load the current season from data/live into Neo4j")
    parser.add_argument("--live-dir", default=LIVE_DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        counts = load_current_season(conn, args.live_dir)
    finally:
        conn.close()
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
