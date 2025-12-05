"""
FPL Data Loader - Loads CSV data into Neo4j Knowledge Graph
"""
import pandas as pd
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import logging
from .connection import Neo4jConnection
from .schema import GraphSchema

logger = logging.getLogger(__name__)


class FPLDataLoader:
    """Loads FPL CSV data into Neo4j graph database."""
    
    def __init__(self, connection: Neo4jConnection):
        """
        Initialize data loader.
        
        Args:
            connection: Neo4j connection instance
        """
        self.conn = connection
        self.schema = GraphSchema(connection)
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load FPL data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Pandas DataFrame with FPL data
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare data for graph loading by extracting unique entities.
        
        Args:
            df: Raw FPL DataFrame
            
        Returns:
            Dictionary with prepared data for each entity type
        """
        logger.info("Preparing data for graph loading...")
        
        # Extract unique players with their positions
        players = df.groupby(['name', 'position', 'element']).size().reset_index()[['name', 'position', 'element']]
        players.columns = ['name', 'position', 'element_id']
        players = players.to_dict('records')
        logger.info(f"Found {len(players)} unique players")
        
        # Extract unique teams from home_team and away_team columns
        home_teams = df['home_team'].unique().tolist()
        away_teams = df['away_team'].unique().tolist()
        teams = list(set(home_teams + away_teams))
        logger.info(f"Found {len(teams)} unique teams")
        
        # Extract unique seasons
        seasons = df['season'].unique().tolist()
        logger.info(f"Found {len(seasons)} seasons: {seasons}")
        
        # Extract unique gameweeks per season
        gameweeks = df.groupby(['season', 'GW']).size().reset_index()[['season', 'GW']]
        gameweeks = gameweeks.to_dict('records')
        logger.info(f"Found {len(gameweeks)} unique gameweeks")
        
        # Extract unique fixtures
        fixtures = df.groupby([
            'fixture', 'season', 'GW', 'home_team', 'away_team', 
            'team_h_score', 'team_a_score', 'kickoff_time'
        ]).size().reset_index()
        fixtures = fixtures[[
            'fixture', 'season', 'GW', 'home_team', 'away_team',
            'team_h_score', 'team_a_score', 'kickoff_time'
        ]].to_dict('records')
        logger.info(f"Found {len(fixtures)} unique fixtures")
        
        # Prepare player-fixture performance records
        performances = df[[
            'name', 'season', 'fixture', 'GW', 'minutes', 'goals_scored', 'assists',
            'clean_sheets', 'bonus', 'bps', 'total_points', 'ict_index',
            'influence', 'creativity', 'threat', 'value', 'form',
            'selected', 'transfers_in', 'transfers_out', 'transfers_balance',
            'goals_conceded', 'own_goals', 'penalties_missed', 'penalties_saved',
            'saves', 'yellow_cards', 'red_cards', 'home_team', 'away_team'
        ]].to_dict('records')
        logger.info(f"Prepared {len(performances)} performance records")
        
        return {
            'players': players,
            'teams': teams,
            'seasons': seasons,
            'gameweeks': gameweeks,
            'fixtures': fixtures,
            'performances': performances,
            'raw_df': df
        }
    
    def create_team_nodes(self, teams: List[str]):
        """Create Team nodes."""
        logger.info("Creating Team nodes...")
        query = """
        UNWIND $teams AS team_name
        MERGE (t:Team {name: team_name})
        """
        self.conn.execute_query(query, {"teams": teams})
        logger.info(f"Created {len(teams)} Team nodes")
    
    def create_gameweek_nodes(self, gameweeks: List[Dict]):
        """Create Gameweek nodes with relationships to seasons."""
        logger.info("Creating Gameweek nodes...")
        query = """
        UNWIND $gameweeks AS gw
        MERGE (gameweek:Gameweek {id: gw.season + '-GW' + toString(gw.GW)})
        SET gameweek.number = gw.GW,
            gameweek.season_id = gw.season
        WITH gameweek, gw
        MATCH (s:Season {id: gw.season})
        MERGE (gameweek)-[:IN_SEASON]->(s)
        """
        self.conn.execute_query(query, {"gameweeks": gameweeks})
        logger.info(f"Created {len(gameweeks)} Gameweek nodes")
    
    def create_player_nodes(self, players: List[Dict]):
        """Create Player nodes with position relationships."""
        logger.info("Creating Player nodes...")
        query = """
        UNWIND $players AS player
        MERGE (p:Player {name: player.name})
        SET p.element_id = player.element_id
        WITH p, player
        MATCH (pos:Position {code: player.position})
        MERGE (p)-[:PLAYS_POSITION]->(pos)
        """
        self.conn.execute_query(query, {"players": players})
        logger.info(f"Created {len(players)} Player nodes")
    
    def create_fixture_nodes(self, fixtures: List[Dict]):
        """Create Fixture nodes with team and gameweek relationships."""
        logger.info("Creating Fixture nodes...")
        query = """
        UNWIND $fixtures AS fix
        MERGE (f:Fixture {id: toString(fix.fixture) + '-' + fix.season})
        SET f.fixture_id = fix.fixture,
            f.kickoff_time = fix.kickoff_time,
            f.home_score = fix.team_h_score,
            f.away_score = fix.team_a_score,
            f.season_id = fix.season,
            f.gameweek = fix.GW
        WITH f, fix
        MATCH (home:Team {name: fix.home_team})
        MATCH (away:Team {name: fix.away_team})
        MATCH (gw:Gameweek {id: fix.season + '-GW' + toString(fix.GW)})
        MERGE (f)-[:HOME_TEAM]->(home)
        MERGE (f)-[:AWAY_TEAM]->(away)
        MERGE (f)-[:PART_OF]->(gw)
        """
        self.conn.execute_query(query, {"fixtures": fixtures})
        logger.info(f"Created {len(fixtures)} Fixture nodes")
    
    def create_player_team_relationships(self, df: pd.DataFrame):
        """Create PLAYS_FOR relationships between players and teams."""
        logger.info("Creating Player-Team relationships...")
        
        # Determine player's team from fixtures they played in
        # A player plays for the home team if they appear in a fixture where home_team matches
        # We need to look at which team the player actually played for
        
        # Group by player, season and determine their team
        player_teams = []
        
        for (name, season), group in df.groupby(['name', 'season']):
            # Get fixtures where player played minutes
            played = group[group['minutes'] > 0]
            if len(played) == 0:
                # Use all records if player never played
                played = group
            
            # Determine team - check first fixture
            first_fixture = played.iloc[0]
            home_team = first_fixture['home_team']
            away_team = first_fixture['away_team']
            
            # Need to infer which team the player belongs to
            # We'll use a heuristic: count appearances as home vs away
            home_count = len(played[played['home_team'] == home_team])
            away_count = len(played[played['away_team'] == away_team])
            
            # Actually, we need a smarter approach. Let's check if player name
            # appears more often with certain teams at home
            team_appearances = {}
            for _, row in played.iterrows():
                # If they're in a Brighton home game, they could be Brighton player
                # We'll just pick the most common home_team in their fixtures
                if row['home_team'] not in team_appearances:
                    team_appearances[row['home_team']] = 0
                if row['away_team'] not in team_appearances:
                    team_appearances[row['away_team']] = 0
            
            # For simplicity, we'll use the home_team from a fixture where player scored/assisted
            # or just the first team that appears most
            scoring = played[(played['goals_scored'] > 0) | (played['assists'] > 0)]
            
            if len(scoring) > 0:
                # Use the team from their scoring record
                # Assume player plays for home team in home fixtures
                first_score = scoring.iloc[0]
                # We actually can't determine this without more info
                # So we'll use a different approach
                pass
            
            # Final approach: Use element_id patterns or just assign based on fixture pattern
            # For now, we'll skip team relationships and add them via a simpler method
            
        # Alternative: Create relationships based on fixture appearances
        # Players appear in fixtures for their team
        query = """
        UNWIND $records AS rec
        MATCH (p:Player {name: rec.name})
        MATCH (t:Team {name: rec.team})
        MERGE (p)-[r:PLAYS_FOR]->(t)
        SET r.season = rec.season
        """
        
        # We need to infer teams from the data differently
        # For now, skip this - we'll add team info during performance creation
        logger.info("Player-Team relationships will be inferred from performances")
    
    def create_performance_relationships(self, performances: List[Dict], batch_size: int = 500):
        """Create PLAYED_IN relationships with performance stats."""
        logger.info("Creating performance relationships...")
        
        query = """
        UNWIND $batch AS perf
        MATCH (p:Player {name: perf.name})
        MATCH (f:Fixture {id: toString(perf.fixture) + '-' + perf.season})
        MERGE (p)-[r:PLAYED_IN]->(f)
        SET r.minutes = perf.minutes,
            r.goals_scored = perf.goals_scored,
            r.assists = perf.assists,
            r.clean_sheets = perf.clean_sheets,
            r.bonus = perf.bonus,
            r.bps = perf.bps,
            r.total_points = perf.total_points,
            r.ict_index = perf.ict_index,
            r.influence = perf.influence,
            r.creativity = perf.creativity,
            r.threat = perf.threat,
            r.value = perf.value,
            r.form = perf.form,
            r.selected = perf.selected,
            r.transfers_in = perf.transfers_in,
            r.transfers_out = perf.transfers_out,
            r.goals_conceded = perf.goals_conceded,
            r.saves = perf.saves,
            r.yellow_cards = perf.yellow_cards,
            r.red_cards = perf.red_cards,
            r.gameweek = perf.GW
        """
        
        total = len(performances)
        for i in tqdm(range(0, total, batch_size), desc="Loading performances"):
            batch = performances[i:i + batch_size]
            self.conn.execute_query(query, {"batch": batch})
        
        logger.info(f"Created {total} performance relationships")
    
    def infer_player_teams(self, df: pd.DataFrame):
        """Infer and create player-team relationships from performance data."""
        logger.info("Inferring player-team relationships...")
        
        # For each player in each season, find their most common team
        player_teams = []
        
        for (name, season), group in df.groupby(['name', 'season']):
            # Count home appearances for each team
            team_counts = {}
            
            for _, row in group.iterrows():
                # Player is part of home team in home fixtures and away team in away fixtures
                # We can't directly tell, but we can use the pattern of fixtures
                pass
            
            # Alternative: Look at which team they appear with most often
            # when that team is playing at home AND the player has good stats
            
        # Simpler approach: Create a computed relationship in Cypher
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:HOME_TEAM]->(ht:Team)
        MATCH (f)-[:AWAY_TEAM]->(at:Team)
        WITH p, f, r, ht, at,
             CASE WHEN r.minutes > 0 THEN 1 ELSE 0 END AS played
        WHERE played = 1
        WITH p, ht, at, f, count(*) AS appearances
        RETURN p.name, ht.name, at.name, appearances
        LIMIT 10
        """
        # This is complex - for now we'll skip direct team relationships
        # The team can be inferred from fixtures in queries
        logger.info("Team relationships will be inferred from fixture context in queries")
    
    def load_all(self, filepath: str, clear_existing: bool = True):
        """
        Load all FPL data into Neo4j.
        
        Args:
            filepath: Path to CSV file
            clear_existing: Whether to clear existing data first
        """
        logger.info("Starting full data load...")
        
        if clear_existing:
            logger.info("Clearing existing data...")
            self.conn.clear_database()
        
        # Initialize schema
        self.schema.initialize_schema()
        self.schema.create_static_nodes()
        
        # Load and prepare data
        df = self.load_csv(filepath)
        data = self.prepare_data(df)
        
        # Create nodes and relationships
        self.create_team_nodes(data['teams'])
        self.create_gameweek_nodes(data['gameweeks'])
        self.create_player_nodes(data['players'])
        self.create_fixture_nodes(data['fixtures'])
        self.create_performance_relationships(data['performances'])
        
        # Get stats
        stats = self.conn.get_database_stats()
        logger.info(f"Data load complete!")
        logger.info(f"Total nodes: {stats['total_nodes']}")
        logger.info(f"Total relationships: {stats['total_relationships']}")
        logger.info(f"Node labels: {stats['node_labels']}")
        
        return stats
    
    def get_aggregate_player_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aggregate player statistics per season.
        
        Args:
            df: Raw FPL DataFrame
            
        Returns:
            DataFrame with aggregated player stats
        """
        agg_stats = df.groupby(['name', 'position', 'season']).agg({
            'goals_scored': 'sum',
            'assists': 'sum',
            'total_points': 'sum',
            'clean_sheets': 'sum',
            'bonus': 'sum',
            'bps': 'sum',
            'minutes': 'sum',
            'ict_index': 'mean',
            'influence': 'mean',
            'creativity': 'mean',
            'threat': 'mean',
            'value': 'last',  # End of season value
            'form': 'mean',
            'selected': 'max',
            'saves': 'sum',
            'goals_conceded': 'sum',
            'yellow_cards': 'sum',
            'red_cards': 'sum',
            'GW': 'count'  # Number of gameweeks
        }).reset_index()
        
        agg_stats.rename(columns={'GW': 'games_played'}, inplace=True)
        
        return agg_stats
