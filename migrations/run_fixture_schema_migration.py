"""
Migration Runner: Add Fixture Schema
=====================================
This script executes the add_fixture_schema.cypher migration against the Neo4j database.

Usage:
    python migrations/run_fixture_schema_migration.py

Requirements:
    - Neo4j database must be running
    - Environment variables must be set (.env file)
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graph.connection import Neo4jConnection
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_migration_file(filepath: str) -> list:
    """
    Load and parse Cypher migration file.
    
    Args:
        filepath: Path to .cypher file
    
    Returns:
        List of Cypher statements
    """
    logger.info(f"Loading migration file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by semicolon and filter out comments and empty lines
    statements = []
    for statement in content.split(';'):
        # Remove comments
        lines = []
        for line in statement.split('\n'):
            # Remove line comments
            if '//' in line:
                line = line[:line.index('//')]
            lines.append(line)
        
        statement = '\n'.join(lines).strip()
        
        # Skip empty statements
        if statement and not statement.startswith('//'):
            statements.append(statement)
    
    logger.info(f"Found {len(statements)} statements to execute")
    return statements


def run_migration(conn: Neo4jConnection, migration_file: str) -> bool:
    """
    Execute migration statements.
    
    Args:
        conn: Neo4j connection
        migration_file: Path to migration file
    
    Returns:
        True if migration succeeded
    """
    try:
        statements = load_migration_file(migration_file)
        
        logger.info("Starting migration execution...")
        for i, statement in enumerate(statements, 1):
            logger.info(f"Executing statement {i}/{len(statements)}...")
            logger.debug(f"Statement: {statement[:100]}...")
            
            try:
                conn.execute_write(statement)
                logger.info(f"✓ Statement {i} executed successfully")
            except Exception as e:
                # Some statements may fail if already exists - this is expected
                if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                    logger.info(f"✓ Statement {i} - already exists (skipped)")
                else:
                    logger.error(f"✗ Statement {i} failed: {e}")
                    raise
        
        logger.info("Migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def verify_schema(conn: Neo4jConnection):
    """
    Verify that the schema was created correctly.
    
    Args:
        conn: Neo4j connection
    """
    logger.info("\n" + "="*60)
    logger.info("Verifying Schema Creation")
    logger.info("="*60)
    
    try:
        # Check constraints
        logger.info("\nConstraints:")
        constraints = conn.execute_query("SHOW CONSTRAINTS")
        fixture_constraints = [c for c in constraints if 'Fixture' in str(c) or 'Matchday' in str(c)]
        for constraint in fixture_constraints:
            logger.info(f"  ✓ {constraint}")
        
        # Check indexes
        logger.info("\nIndexes:")
        indexes = conn.execute_query("SHOW INDEXES")
        fixture_indexes = [idx for idx in indexes if 'fixture' in str(idx).lower() or 'matchday' in str(idx).lower()]
        for index in fixture_indexes:
            logger.info(f"  ✓ {index}")
        
        # Check if Fixture nodes exist
        fixture_count = conn.get_node_count("Fixture")
        logger.info(f"\nFixture nodes in database: {fixture_count}")
        
        matchday_count = conn.get_node_count("Matchday")
        logger.info(f"Matchday nodes in database: {matchday_count}")
        
        logger.info("\n" + "="*60)
        logger.info("Schema verification complete!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Schema verification failed: {e}")


def main():
    """Main migration execution function."""
    logger.info("="*60)
    logger.info("Fixture Schema Migration")
    logger.info("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j connection details
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not neo4j_password:
        logger.error("NEO4J_PASSWORD not set in environment variables!")
        sys.exit(1)
    
    logger.info(f"Connecting to Neo4j at {neo4j_uri}...")
    
    try:
        # Connect to Neo4j
        conn = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
        
        # Test connection
        if not conn.test_connection():
            logger.error("Failed to connect to Neo4j database!")
            sys.exit(1)
        
        logger.info("✓ Connected to Neo4j successfully\n")
        
        # Run migration
        migration_file = Path(__file__).parent / "add_fixture_schema.cypher"
        success = run_migration(conn, str(migration_file))
        
        if success:
            # Verify schema
            verify_schema(conn)
            logger.info("\n✓ Migration completed successfully!")
            return 0
        else:
            logger.error("\n✗ Migration failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
