#!/usr/bin/env python3
"""
Load 2026-27 Season Fixtures into Neo4j
Migration script to load fixture data from CSV.
"""
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.connection import Neo4jConnection
from services.fixture_service import FixtureService
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Load fixtures from CSV into Neo4j."""
    parser = argparse.ArgumentParser(
        description='Load Premier League fixture data into Neo4j',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load fixtures from default path
  python load_2026_27_fixtures.py
  
  # Load fixtures from custom CSV path
  python load_2026_27_fixtures.py --csv-path /path/to/fixtures.csv
  
  # Load fixtures for different season
  python load_2026_27_fixtures.py --season-id 2027-28
  
  # Clear existing fixtures before loading
  python load_2026_27_fixtures.py --clear-existing
        """
    )
    
    parser.add_argument(
        '--csv-path',
        type=str,
        default='FPL_2026_2027_fixtures.csv',
        help='Path to CSV file containing fixture data (default: FPL_2026_2027_fixtures.csv)'
    )
    
    parser.add_argument(
        '--season-id',
        type=str,
        default='2026-27',
        help='Season identifier (default: 2026-27)'
    )
    
    parser.add_argument(
        '--clear-existing',
        action='store_true',
        help='Clear existing fixtures for the season before loading'
    )
    
    parser.add_argument(
        '--neo4j-uri',
        type=str,
        default=NEO4J_URI,
        help=f'Neo4j URI (default: {NEO4J_URI})'
    )
    
    parser.add_argument(
        '--neo4j-user',
        type=str,
        default=NEO4J_USER,
        help=f'Neo4j username (default: {NEO4J_USER})'
    )
    
    parser.add_argument(
        '--neo4j-password',
        type=str,
        default=NEO4J_PASSWORD,
        help='Neo4j password (default: from environment)'
    )
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    if not os.path.exists(args.csv_path):
        logger.error(f"CSV file not found: {args.csv_path}")
        return 1
    
    logger.info("=" * 80)
    logger.info("Loading Premier League Fixtures into Neo4j")
    logger.info("=" * 80)
    logger.info(f"CSV Path: {args.csv_path}")
    logger.info(f"Season ID: {args.season_id}")
    logger.info(f"Clear Existing: {args.clear_existing}")
    logger.info(f"Neo4j URI: {args.neo4j_uri}")
    logger.info("=" * 80)
    
    try:
        # Connect to Neo4j
        logger.info("Connecting to Neo4j...")
        neo4j_conn = Neo4jConnection(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password
        )
        
        # Test connection
        neo4j_conn.test_connection()
        logger.info("✅ Connected to Neo4j successfully")
        
        # Create fixture service
        fixture_service = FixtureService(neo4j_conn)
        
        # Clear existing fixtures if requested
        if args.clear_existing:
            logger.info(f"Clearing existing fixtures for season {args.season_id}...")
            clear_query = """
            MATCH (f:Fixture {season_id: $season_id})
            DETACH DELETE f
            """
            neo4j_conn.execute_write(clear_query, {"season_id": args.season_id})
            logger.info("✅ Existing fixtures cleared")
        
        # Load fixtures from CSV
        logger.info(f"Loading fixtures from {args.csv_path}...")
        fixtures_loaded = fixture_service.load_fixtures_from_csv(
            csv_path=args.csv_path,
            season_id=args.season_id
        )
        
        logger.info("=" * 80)
        logger.info(f"✅ SUCCESS: Loaded {fixtures_loaded} fixtures for season {args.season_id}")
        logger.info("=" * 80)
        
        # Display summary statistics
        summary_query = """
        MATCH (f:Fixture {season_id: $season_id})
        WITH f.matchday AS matchday, count(*) AS fixture_count
        ORDER BY matchday
        RETURN matchday, fixture_count
        """
        results = neo4j_conn.execute_query(summary_query, {"season_id": args.season_id})
        
        if results:
            logger.info("\nFixtures by Matchday:")
            logger.info("-" * 40)
            total = 0
            for result in results:
                matchday = result["matchday"]
                count = result["fixture_count"]
                total += count
                logger.info(f"  Matchday {matchday:2d}: {count} fixtures")
            logger.info("-" * 40)
            logger.info(f"  Total: {total} fixtures")
        
        # Close connection
        neo4j_conn.close()
        
        return 0
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ ERROR: Failed to load fixtures")
        logger.error(f"Error: {str(e)}")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
