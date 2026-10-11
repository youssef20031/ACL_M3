"""
Load FPL data into Neo4j using the credentials in .env.

    python load_data.py            # wipe and reload everything, including 2026-27 fixtures
    python load_data.py --no-fixtures
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_PATH
from graph.connection import Neo4jConnection
from graph.data_loader import FPLDataLoader
from services.fixture_service import FixtureService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FIXTURES_CSV = "FPL_2026_2027_fixtures_fixed.csv"
FIXTURES_SEASON = "2026-27"


def main() -> int:
    parser = argparse.ArgumentParser(description="Load FPL data into Neo4j")
    parser.add_argument("--csv", default=DATA_PATH, help="Historical per-match CSV")
    parser.add_argument("--keep-existing", action="store_true", help="Don't wipe the database first")
    parser.add_argument("--no-fixtures", action="store_true", help=f"Skip loading {FIXTURES_SEASON} fixtures")
    args = parser.parse_args()

    conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        stats = FPLDataLoader(conn).load_all(args.csv, clear_existing=not args.keep_existing)

        if not args.no_fixtures:
            loaded = FixtureService(conn).load_fixtures_from_csv(FIXTURES_CSV, season_id=FIXTURES_SEASON)
            logger.info(f"Loaded {loaded} {FIXTURES_SEASON} fixtures from {FIXTURES_CSV}")
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
