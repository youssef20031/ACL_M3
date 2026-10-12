"""
Load FPL data into Neo4j using the credentials in .env.

    python load_data.py                  # wipe and reload everything, including 2026-27 fixtures
    python load_data.py --no-fixtures
    python load_data.py --fixtures-only  # (re)load promoted teams + 2026-27 fixtures, keep the rest
"""
import argparse
import logging
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_PATH
from graph.connection import Neo4jConnection
from graph.data_loader import FPLDataLoader
from services.fixture_service import FixtureService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FIXTURES_CSV = "FPL_2026_2027_fixtures_fixed.csv"
FIXTURES_SEASON = "2026-27"
# Championship table for the season before FIXTURES_SEASON: the source of promoted clubs
EFL_TEAMS_FILE = "EFL_Data_All_Seasons/EFL-2025-2026/2025-2026_Standard_Teams.txt"


def add_promoted_teams(conn: Neo4jConnection) -> List[str]:
    """Create Team nodes for fixture-list clubs that have no Premier League history.

    A club counts as promoted when it appears in FIXTURES_CSV, isn't in the graph yet,
    and matches a squad in the previous Championship table (exactly, or as a prefix
    such as "Ipswich" for "Ipswich Town"). Without these nodes their fixtures are skipped.
    """
    fixtures = pd.read_csv(FIXTURES_CSV)
    fixture_teams = set(fixtures["Home Team"]) | set(fixtures["Away Team"])
    known = {r["name"] for r in conn.execute_query("MATCH (t:Team) RETURN t.name AS name")}
    efl_squads = pd.read_csv(EFL_TEAMS_FILE, sep="\t")["Squad"].dropna().astype(str).tolist()

    promoted = []
    for team in sorted(fixture_teams - known):
        efl_name = next((s for s in efl_squads if s == team or s.startswith(f"{team} ")), None)
        if efl_name is None:
            logger.warning(f"'{team}' is in {FIXTURES_CSV} but not in the graph or the Championship table")
            continue
        conn.execute_write(
            "MERGE (t:Team {name: $name}) SET t.promoted_season = $season, t.efl_name = $efl_name",
            {"name": team, "season": FIXTURES_SEASON, "efl_name": efl_name},
        )
        promoted.append(team)
    logger.info(f"Promoted teams added for {FIXTURES_SEASON}: {promoted or 'none'}")
    return promoted


def load_fixtures(conn: Neo4jConnection) -> None:
    add_promoted_teams(conn)
    loaded = FixtureService(conn).load_fixtures_from_csv(FIXTURES_CSV, season_id=FIXTURES_SEASON)
    logger.info(f"Loaded {loaded} {FIXTURES_SEASON} fixtures from {FIXTURES_CSV}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Load FPL data into Neo4j")
    parser.add_argument("--csv", default=DATA_PATH, help="Historical per-match CSV")
    parser.add_argument("--keep-existing", action="store_true", help="Don't wipe the database first")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-fixtures", action="store_true", help=f"Skip loading {FIXTURES_SEASON} fixtures")
    group.add_argument("--fixtures-only", action="store_true",
                       help=f"Only (re)load promoted teams and {FIXTURES_SEASON} fixtures")
    args = parser.parse_args()

    conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        if not args.fixtures_only:
            FPLDataLoader(conn).load_all(args.csv, clear_existing=not args.keep_existing)
        if not args.no_fixtures:
            load_fixtures(conn)

        stats = conn.get_database_stats()
        print(f"\nTotal nodes: {stats['total_nodes']:,}")
        print(f"Total relationships: {stats['total_relationships']:,}")
        for label, count in stats['node_labels'].items():
            print(f"  {label}: {count:,}")
        for rel_type, count in stats['relationship_types'].items():
            print(f"  [{rel_type}]: {count:,}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
