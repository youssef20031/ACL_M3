"""
Neo4j Graph Schema Definition and Creation
"""
from typing import List
from .connection import Neo4jConnection
import logging

logger = logging.getLogger(__name__)


class GraphSchema:
    """Defines and creates the FPL Knowledge Graph schema in Neo4j."""
    
    # Node Labels
    PLAYER = "Player"
    TEAM = "Team"
    POSITION = "Position"
    SEASON = "Season"
    GAMEWEEK = "Gameweek"
    FIXTURE = "Fixture"
    
    # Relationship Types
    PLAYS_POSITION = "PLAYS_POSITION"
    PLAYS_FOR = "PLAYS_FOR"
    PLAYED_IN = "PLAYED_IN"
    HOME_TEAM = "HOME_TEAM"
    AWAY_TEAM = "AWAY_TEAM"
    PART_OF = "PART_OF"
    IN_SEASON = "IN_SEASON"
    
    def __init__(self, connection: Neo4jConnection):
        """
        Initialize schema manager.
        
        Args:
            connection: Neo4j connection instance
        """
        self.conn = connection
    
    def create_constraints(self):
        """Create uniqueness constraints for node identifiers."""
        constraints = [
            # Player constraint
            f"CREATE CONSTRAINT player_name IF NOT EXISTS FOR (p:{self.PLAYER}) REQUIRE p.name IS UNIQUE",
            # Team constraint
            f"CREATE CONSTRAINT team_name IF NOT EXISTS FOR (t:{self.TEAM}) REQUIRE t.name IS UNIQUE",
            # Position constraint
            f"CREATE CONSTRAINT position_code IF NOT EXISTS FOR (pos:{self.POSITION}) REQUIRE pos.code IS UNIQUE",
            # Season constraint
            f"CREATE CONSTRAINT season_id IF NOT EXISTS FOR (s:{self.SEASON}) REQUIRE s.id IS UNIQUE",
            # Gameweek constraint (composite)
            f"CREATE CONSTRAINT gameweek_id IF NOT EXISTS FOR (gw:{self.GAMEWEEK}) REQUIRE gw.id IS UNIQUE",
            # Fixture constraint
            f"CREATE CONSTRAINT fixture_id IF NOT EXISTS FOR (f:{self.FIXTURE}) REQUIRE f.id IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                self.conn.execute_write(constraint)
                logger.info(f"Created constraint: {constraint[:50]}...")
            except Exception as e:
                # Constraint might already exist
                logger.warning(f"Constraint may already exist: {e}")
    
    def create_indexes(self):
        """Create indexes for frequently queried properties."""
        indexes = [
            # Player indexes
            f"CREATE INDEX player_element IF NOT EXISTS FOR (p:{self.PLAYER}) ON (p.element_id)",
            
            # Gameweek indexes
            f"CREATE INDEX gameweek_number IF NOT EXISTS FOR (gw:{self.GAMEWEEK}) ON (gw.number)",
            f"CREATE INDEX gameweek_season IF NOT EXISTS FOR (gw:{self.GAMEWEEK}) ON (gw.season_id)",
            
            # Fixture indexes
            f"CREATE INDEX fixture_kickoff IF NOT EXISTS FOR (f:{self.FIXTURE}) ON (f.kickoff_time)",
        ]
        
        for index in indexes:
            try:
                self.conn.execute_write(index)
                logger.info(f"Created index: {index[:50]}...")
            except Exception as e:
                logger.warning(f"Index may already exist: {e}")
    
    def create_vector_index(self, embedding_dim: int = 384):
        """
        Create vector index for embedding-based similarity search.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        # Vector index for player embeddings
        vector_index_query = f"""
        CREATE VECTOR INDEX player_embedding IF NOT EXISTS
        FOR (p:{self.PLAYER})
        ON p.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {embedding_dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        
        try:
            self.conn.execute_write(vector_index_query)
            logger.info(f"Created vector index with dimension {embedding_dim}")
        except Exception as e:
            logger.warning(f"Vector index creation failed (may need Neo4j 5.11+): {e}")
    
    def initialize_schema(self, embedding_dim: int = 384):
        """
        Initialize complete database schema.
        
        Args:
            embedding_dim: Dimension for vector embeddings
        """
        logger.info("Initializing database schema...")
        self.create_constraints()
        self.create_indexes()
        self.create_vector_index(embedding_dim)
        logger.info("Schema initialization complete")
    
    def create_static_nodes(self):
        """Create static nodes for positions and seasons."""
        # Create Position nodes
        positions = [
            {"code": "GK", "name": "Goalkeeper"},
            {"code": "DEF", "name": "Defender"},
            {"code": "MID", "name": "Midfielder"},
            {"code": "FWD", "name": "Forward"}
        ]
        
        position_query = f"""
        UNWIND $positions AS pos
        MERGE (p:{self.POSITION} {{code: pos.code}})
        SET p.name = pos.name
        """
        self.conn.execute_query(position_query, {"positions": positions})
        logger.info("Created Position nodes")
        
        # Create Season nodes
        seasons = [
            {"id": "2020-21", "start_year": 2020, "end_year": 2021},
            {"id": "2021-22", "start_year": 2021, "end_year": 2022},
            {"id": "2022-23", "start_year": 2022, "end_year": 2023}
        ]
        
        season_query = f"""
        UNWIND $seasons AS s
        MERGE (season:{self.SEASON} {{id: s.id}})
        SET season.start_year = s.start_year,
            season.end_year = s.end_year
        """
        self.conn.execute_query(season_query, {"seasons": seasons})
        logger.info("Created Season nodes")
    
    def get_schema_info(self) -> dict:
        """
        Get information about the current schema.
        
        Returns:
            Dictionary with schema information
        """
        schema_info = {
            "node_labels": [
                self.PLAYER, self.TEAM, self.POSITION, 
                self.SEASON, self.GAMEWEEK, self.FIXTURE
            ],
            "relationship_types": [
                self.PLAYS_POSITION, self.PLAYS_FOR, self.PLAYED_IN,
                self.HOME_TEAM, self.AWAY_TEAM, self.PART_OF, self.IN_SEASON
            ],
            "constraints": [],
            "indexes": []
        }
        
        # Get constraints
        try:
            constraints = self.conn.execute_query("SHOW CONSTRAINTS")
            schema_info["constraints"] = constraints
        except:
            pass
        
        # Get indexes
        try:
            indexes = self.conn.execute_query("SHOW INDEXES")
            schema_info["indexes"] = indexes
        except:
            pass
        
        return schema_info
