"""Create PLAYS_FOR relationships between Players and Teams"""
import pandas as pd
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_player_team_relationships():
    """Create PLAYS_FOR relationships from the most recent season data."""
    
    conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Read the most recent season's data (2025-26)
        logger.info("Reading data/FPL_2025_2026.csv...")
        df = pd.read_csv('data/FPL_2025_2026.csv')
        
        # Get unique player-team combinations (most recent gameweek for each player)
        logger.info("Finding latest team for each player...")
        latest_teams = df.sort_values('GW', ascending=False).groupby('name').first()[['team']].reset_index()
        
        logger.info(f"Found {len(latest_teams)} unique players")
        
        # Get existing teams in Neo4j
        team_results = conn.execute_query('MATCH (t:Team) RETURN t.name as name')
        neo4j_teams = {r['name'] for r in team_results}
        logger.info(f"Found {len(neo4j_teams)} teams in Neo4j: {sorted(neo4j_teams)}")
        
        # Create PLAYS_FOR relationships
        relationships_created = 0
        players_not_found = 0
        teams_not_found = 0
        
        for _, row in latest_teams.iterrows():
            player_name = row['name']
            team_name = row['team']
            
            # Check if team exists in Neo4j
            if team_name not in neo4j_teams:
                if teams_not_found < 5:  # Only log first 5
                    logger.warning(f"Team '{team_name}' not found in Neo4j for player {player_name}")
                teams_not_found += 1
                continue
            
            # Create or update PLAYS_FOR relationship
            query = """
            MATCH (p:Player {name: $player_name})
            MATCH (t:Team {name: $team_name})
            MERGE (p)-[:PLAYS_FOR]->(t)
            RETURN p.name as player, t.name as team
            """
            
            try:
                result = conn.execute_write(query, {
                    "player_name": player_name,
                    "team_name": team_name
                })
                
                if result:
                    relationships_created += 1
                    if relationships_created % 100 == 0:
                        logger.info(f"Created {relationships_created} relationships...")
                else:
                    players_not_found += 1
                    
            except Exception as e:
                logger.error(f"Error creating relationship for {player_name} -> {team_name}: {e}")
        
        logger.info("=" * 80)
        logger.info(f"✅ Created {relationships_created} PLAYS_FOR relationships")
        logger.info(f"⚠️  Players not found in Neo4j: {players_not_found}")
        logger.info(f"⚠️  Teams not found in Neo4j: {teams_not_found}")
        logger.info("=" * 80)
        
        # Verify a few relationships
        logger.info("\nVerifying sample relationships:")
        verify_query = """
        MATCH (p:Player)-[:PLAYS_FOR]->(t:Team)
        RETURN p.name as player, t.name as team
        LIMIT 10
        """
        results = conn.execute_query(verify_query)
        for r in results:
            logger.info(f"  {r['player']} -> {r['team']}")
        
        return relationships_created
        
    finally:
        conn.close()


if __name__ == "__main__":
    create_player_team_relationships()
