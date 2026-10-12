"""
Fixture Service for Managing Match Fixture Data
Provides functionality to determine next matches based on current date.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import logging
import pandas as pd

from graph.connection import Neo4jConnection
from services.cache import PredictionCache
from services.exceptions import (
    FixtureServiceError,
    FixtureNotFoundError,
    InvalidFixtureDataError,
    TeamNotFoundError,
    InvalidDateFormatError,
    CSVLoadError
)

logger = logging.getLogger(__name__)


@dataclass
class FixtureContext:
    """Fixture context for ML predictions."""
    opponent_team: str
    home_away: str  # "home" or "away"
    match_date: datetime
    matchday: int
    fixture_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        data = asdict(self)
        # Convert datetime to ISO format string
        # Handle both Python datetime and Neo4j DateTime objects
        match_date = data['match_date']
        if match_date is not None:
            # Check if it's a Neo4j DateTime object (has to_native method)
            if hasattr(match_date, 'to_native'):
                # Convert Neo4j DateTime to Python datetime first
                match_date = match_date.to_native()
            # Now convert Python datetime to ISO string
            if isinstance(match_date, datetime):
                data['match_date'] = match_date.isoformat()
            else:
                # Fallback: try to call isoformat() directly
                data['match_date'] = str(match_date)
        return data


@dataclass
class FixtureData:
    """Raw fixture data from CSV or API."""
    id: str
    home_team: str
    away_team: str
    kickoff_time: datetime
    matchday: int
    season_id: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str = "scheduled"


@dataclass
class OpponentStrength:
    """Opponent strength metrics."""
    defensive_strength: float  # Goals conceded average (lower is better defense)
    offensive_strength: float  # Goals scored average (higher is better offense)
    matches_played: int


class FixtureService:
    """Service for managing fixture data and determining next matches."""
    
    def __init__(self, neo4j_conn: Neo4jConnection, cache: Optional[PredictionCache] = None):
        """
        Initialize FixtureService.
        
        Args:
            neo4j_conn: Neo4j connection instance
            cache: Optional PredictionCache instance for invalidating predictions
        """
        self.conn = neo4j_conn
        self.cache = cache
        logger.info("FixtureService initialized")
    
    def get_next_match(
        self, 
        player_name: str, 
        current_date: Optional[datetime] = None
    ) -> Optional[FixtureContext]:
        """
        Get the next upcoming fixture for a player.
        
        Args:
            player_name: Player's full name
            current_date: Reference date (defaults to datetime.now(UTC))
        
        Returns:
            FixtureContext object or None if no upcoming matches
        
        Raises:
            InvalidFixtureDataError: If player_name is empty or invalid
        """
        import time
        start_time = time.time()
        
        # Validate input
        if not player_name or not player_name.strip():
            raise InvalidFixtureDataError("Player name cannot be empty")
        
        if current_date is None:
            current_date = datetime.now(timezone.utc)
        
        # Ensure current_date is timezone-aware (UTC)
        if current_date.tzinfo is None:
            current_date = current_date.replace(tzinfo=timezone.utc)
        
        query = """
        MATCH (p:Player {name: $player_name})-[:PLAYS_FOR]->(t:Team)
        MATCH (f:Fixture)
        WHERE ((f)-[:PLAYS_HOME]->(t) OR (f)-[:PLAYS_AWAY]->(t))
          AND f.kickoff_time > $current_date
        WITH f, t,
          CASE 
            WHEN (f)-[:PLAYS_HOME]->(t) THEN 'home'
            ELSE 'away'
          END AS home_away,
          CASE
            WHEN (f)-[:PLAYS_HOME]->(t) 
            THEN [(f)-[:PLAYS_AWAY]->(opp:Team) | opp.name][0]
            ELSE [(f)-[:PLAYS_HOME]->(opp:Team) | opp.name][0]
          END AS opponent
        ORDER BY f.kickoff_time ASC
        LIMIT 1
        RETURN f.id AS fixture_id,
               f.kickoff_time AS match_date,
               f.matchday AS matchday,
               opponent AS opponent_team,
               home_away
        """
        
        try:
            results = self.conn.execute_query(
                query,
                {
                    "player_name": player_name,
                    "current_date": current_date
                }
            )
            
            query_time_ms = (time.time() - start_time) * 1000
            
            if not results:
                logger.info(
                    f"No upcoming fixtures found for player: {player_name}",
                    extra={
                        "player_name": player_name,
                        "query_time_ms": query_time_ms,
                        "result": "no_fixture"
                    }
                )
                return None
            
            result = results[0]
            
            # Create FixtureContext from result
            fixture_context = FixtureContext(
                opponent_team=result["opponent_team"],
                home_away=result["home_away"],
                match_date=result["match_date"],
                matchday=result["matchday"],
                fixture_id=result["fixture_id"]
            )
            
            logger.info(
                f"Next match for {player_name}: {fixture_context.opponent_team} ({fixture_context.home_away})",
                extra={
                    "player_name": player_name,
                    "fixture_id": fixture_context.fixture_id,
                    "opponent_team": fixture_context.opponent_team,
                    "home_away": fixture_context.home_away,
                    "query_time_ms": query_time_ms,
                    "result": "success"
                }
            )
            return fixture_context
            
        except Exception as e:
            query_time_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Error getting next match for {player_name}: {e}",
                extra={
                    "player_name": player_name,
                    "query_time_ms": query_time_ms,
                    "error": str(e)
                }
            )
            raise
    
    def get_team_next_match(
        self,
        team_name: str,
        current_date: Optional[datetime] = None
    ) -> Optional[FixtureContext]:
        """
        Get the next fixture for a specific team.
        
        Args:
            team_name: Team name
            current_date: Reference date (defaults to datetime.now(UTC))
        
        Returns:
            FixtureContext object or None if no upcoming matches
        """
        if current_date is None:
            current_date = datetime.now(timezone.utc)
        
        # Ensure current_date is timezone-aware (UTC)
        if current_date.tzinfo is None:
            current_date = current_date.replace(tzinfo=timezone.utc)
        
        # Anchored on the team's own fixtures rather than scanning every Fixture
        query = """
        MATCH (t:Team {name: $team_name})<-[side:PLAYS_HOME|PLAYS_AWAY]-(f:Fixture)
        WHERE f.kickoff_time > $current_date
        WITH f, t,
          CASE type(side) WHEN 'PLAYS_HOME' THEN 'home' ELSE 'away' END AS home_away,
          CASE type(side)
            WHEN 'PLAYS_HOME' THEN [(f)-[:PLAYS_AWAY]->(opp:Team) | opp.name][0]
            ELSE [(f)-[:PLAYS_HOME]->(opp:Team) | opp.name][0]
          END AS opponent
        ORDER BY f.kickoff_time ASC
        LIMIT 1
        RETURN f.id AS fixture_id,
               f.kickoff_time AS match_date,
               f.matchday AS matchday,
               opponent AS opponent_team,
               home_away
        """
        
        try:
            results = self.conn.execute_query(
                query,
                {
                    "team_name": team_name,
                    "current_date": current_date
                }
            )
            
            if not results:
                logger.info(f"No upcoming fixtures found for team: {team_name}")
                return None
            
            result = results[0]
            
            # Create FixtureContext from result
            fixture_context = FixtureContext(
                opponent_team=result["opponent_team"],
                home_away=result["home_away"],
                match_date=result["match_date"],
                matchday=result["matchday"],
                fixture_id=result["fixture_id"]
            )
            
            logger.info(f"Next match for {team_name}: {fixture_context.opponent_team} ({fixture_context.home_away})")
            return fixture_context
            
        except Exception as e:
            logger.error(f"Error getting next match for team {team_name}: {e}")
            raise
    
    def is_fixture_past(
        self,
        fixture_id: str,
        current_date: Optional[datetime] = None
    ) -> bool:
        """
        Check if a fixture has already occurred.
        
        Args:
            fixture_id: Fixture identifier
            current_date: Reference date (defaults to datetime.now(UTC))
        
        Returns:
            True if fixture is in the past, False otherwise
        """
        if current_date is None:
            current_date = datetime.now(timezone.utc)
        
        # Ensure current_date is timezone-aware (UTC)
        if current_date.tzinfo is None:
            current_date = current_date.replace(tzinfo=timezone.utc)
        
        query = """
        MATCH (f:Fixture {id: $fixture_id})
        RETURN f.kickoff_time AS kickoff_time
        """
        
        try:
            results = self.conn.execute_query(query, {"fixture_id": fixture_id})
            
            if not results:
                logger.warning(f"Fixture not found: {fixture_id}")
                return False
            
            kickoff_time = results[0]["kickoff_time"]
            
            # Compare times (both should be in UTC)
            is_past = kickoff_time <= current_date
            
            logger.debug(f"Fixture {fixture_id} is_past: {is_past}")
            return is_past
            
        except Exception as e:
            logger.error(f"Error checking if fixture is past: {e}")
            raise
    
    def _api_fixture_count(self, season_id: str) -> int:
        """Fixtures of a season loaded from the FPL API (they carry fixture_id)."""
        rows = self.conn.execute_query(
            "MATCH (f:Fixture {season_id: $season_id}) WHERE f.fixture_id IS NOT NULL "
            "RETURN count(f) AS api_fixtures",
            {"season_id": season_id},
        )
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            count = rows[0].get("api_fixtures")
            return count if isinstance(count, int) else 0
        return 0

    def load_fixtures_from_csv(
        self,
        csv_path: str,
        season_id: str = "2026-27"
    ) -> int:
        """
        Load fixture data from CSV into Neo4j.
        
        Args:
            csv_path: Path to CSV file
            season_id: Season identifier
        
        Returns:
            Number of fixtures loaded
        """
        logger.info(f"Loading fixtures from {csv_path} for season {season_id}")
        
        try:
            # Parse CSV file
            df = pd.read_csv(csv_path)
            required_columns = {"Matchday", "Home Team", "Away Team", "Date"}
            
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Missing required columns: {missing}")

            # This CSV is the offline fallback. Once the season's fixtures come from the
            # FPL API (pipeline/load_current.py, keyed by FPL fixture id) loading it too
            # would create a second node per match.
            api_fixtures = self._api_fixture_count(season_id)
            if api_fixtures:
                logger.info(
                    f"Skipping {csv_path}: {api_fixtures} {season_id} fixtures from the FPL API are already loaded"
                )
                return 0

            # Get valid team names from Neo4j
            valid_teams_query = "MATCH (t:Team) RETURN t.name AS name"
            valid_teams_results = self.conn.execute_query(valid_teams_query)
            valid_teams = {team["name"] for team in valid_teams_results}
            
            logger.info(f"Found {len(valid_teams)} valid teams in database")
            
            fixtures_loaded = 0
            
            for idx, row in df.iterrows():
                home_team = row["Home Team"]
                away_team = row["Away Team"]
                
                # Validate teams exist
                if home_team not in valid_teams:
                    logger.warning(f"Skipping fixture at row {idx}: Unknown home team '{home_team}'")
                    continue
                if away_team not in valid_teams:
                    logger.warning(f"Skipping fixture at row {idx}: Unknown away team '{away_team}'")
                    continue
                
                # Parse date and time
                date_str = row["Date"]
                time_str = row.get("Time", "15:00")  # Default to 3pm if Time column missing
                
                # Try different date formats
                try:
                    # Try DD/MM/YYYY format first
                    kickoff = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
                except ValueError:
                    try:
                        # Try YYYY-MM-DD format
                        kickoff = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                    except ValueError:
                        logger.warning(f"Skipping fixture at row {idx}: Invalid date format '{date_str}'")
                        continue
                
                # Convert to UTC
                kickoff_utc = kickoff.replace(tzinfo=timezone.utc)
                
                # Generate unique fixture ID
                matchday = int(row["Matchday"])
                # Create more robust abbreviations
                home_abbr = ''.join(c for c in home_team[:3] if c.isalnum()).lower()
                away_abbr = ''.join(c for c in away_team[:3] if c.isalnum()).lower()
                fixture_id = f"fixture-{season_id}-md{matchday}-{home_abbr}-{away_abbr}"
                
                # Create fixture and relationships using MERGE for idempotency
                fixture_query = """
                MERGE (f:Fixture {id: $fixture_id})
                SET f.kickoff_time = $kickoff,
                    f.matchday = $matchday,
                    f.season_id = $season_id,
                    f.status = 'scheduled'
                
                MERGE (ht:Team {name: $home_team})
                MERGE (at:Team {name: $away_team})
                
                MERGE (f)-[:PLAYS_HOME]->(ht)
                MERGE (f)-[:PLAYS_AWAY]->(at)
                
                MERGE (s:Season {id: $season_id})
                MERGE (f)-[:IN_SEASON]->(s)
                """
                
                self.conn.execute_write(
                    fixture_query,
                    {
                        "fixture_id": fixture_id,
                        "kickoff": kickoff_utc,
                        "matchday": matchday,
                        "season_id": season_id,
                        "home_team": home_team,
                        "away_team": away_team
                    }
                )
                
                fixtures_loaded += 1
                
                if fixtures_loaded % 50 == 0:
                    logger.info(f"Loaded {fixtures_loaded} fixtures...")
            
            logger.info(f"✅ Successfully loaded {fixtures_loaded} fixtures for season {season_id}")
            return fixtures_loaded
            
        except Exception as e:
            logger.error(f"Error loading fixtures from CSV: {e}")
            raise
    
    def update_fixture(
        self,
        fixture_id: str,
        updates: Dict[str, Any],
        user: Optional[str] = None
    ) -> bool:
        """
        Update an existing fixture's details.
        
        Args:
            fixture_id: Fixture identifier
            updates: Dictionary of fields to update
            user: User performing the update (for audit logging)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Query existing fixture first
            check_query = "MATCH (f:Fixture {id: $fixture_id}) RETURN f"
            results = self.conn.execute_query(check_query, {"fixture_id": fixture_id})
            
            if not results:
                logger.warning(f"Fixture not found: {fixture_id}")
                return False
            
            # Store original fixture state for audit logging
            original_fixture = results[0]["f"]
            
            # Build SET clause from updates dictionary
            set_clauses = []
            params = {"fixture_id": fixture_id}
            changed_fields = {}
            
            for key, value in updates.items():
                # Prevent updating certain fields
                if key in ["id", "fixture_id"]:
                    continue
                set_clauses.append(f"f.{key} = ${key}")
                params[key] = value
                
                # Track changed fields for audit log
                original_value = original_fixture.get(key)
                if original_value != value:
                    changed_fields[key] = {
                        "old": original_value,
                        "new": value
                    }
            
            if not set_clauses:
                logger.warning("No valid fields to update")
                return False
            
            set_clause = ", ".join(set_clauses)
            update_query = f"""
            MATCH (f:Fixture {{id: $fixture_id}})
            SET {set_clause}
            RETURN f
            """
            
            # Perform update
            self.conn.execute_write(update_query, params)
            
            # Invalidate cached predictions for this fixture
            if self.cache:
                cache_pattern = f"prediction:*:{fixture_id}:*"
                self.cache.invalidate_pattern(cache_pattern)
                logger.info(f"Invalidated cached predictions for fixture {fixture_id} using pattern: {cache_pattern}")
            else:
                logger.debug(f"No cache configured, skipping invalidation for fixture {fixture_id}")
            
            # Audit logging: Log fixture update with timestamp, user, and changed fields
            audit_timestamp = datetime.now(timezone.utc).isoformat()
            logger.info(
                f"AUDIT: Fixture updated",
                extra={
                    "audit_type": "fixture_update",
                    "timestamp": audit_timestamp,
                    "user": user or "system",
                    "fixture_id": fixture_id,
                    "changed_fields": changed_fields,
                    "update_count": len(changed_fields)
                }
            )
            
            logger.info(f"Updated fixture {fixture_id} with {len(changed_fields)} field(s)")
            return True
            
        except Exception as e:
            logger.error(f"Error updating fixture: {e}")
            return False
    
    def calculate_opponent_strength(
        self,
        team_name: str,
        current_date: Optional[datetime] = None,
        num_matches: int = 5
    ) -> Dict[str, float]:
        """
        Calculate opponent strength metrics based on recent performance.
        
        Calculates rolling averages for:
        - Defensive strength: average goals conceded in last N matches
        - Offensive strength: average goals scored in last N matches
        
        Args:
            team_name: Team name to calculate strength for
            current_date: Reference date (defaults to datetime.now(UTC))
            num_matches: Number of recent matches to consider (default: 5)
        
        Returns:
            Dictionary with defensive_strength and offensive_strength keys.
            Returns league average (1.5 goals) when no data available.
        """
        if current_date is None:
            current_date = datetime.now(timezone.utc)
        
        # Ensure current_date is timezone-aware (UTC)
        if current_date.tzinfo is None:
            current_date = current_date.replace(tzinfo=timezone.utc)
        
        # Default league averages
        LEAGUE_AVG_GOALS = 1.5
        
        query = """
        MATCH (t:Team {name: $team_name})
        MATCH (f:Fixture)
        WHERE ((f)-[:PLAYS_HOME]->(t) OR (f)-[:PLAYS_AWAY]->(t))
          AND f.kickoff_time <= $current_date
          AND f.home_score IS NOT NULL
          AND f.away_score IS NOT NULL
        WITH f, t,
          CASE 
            WHEN (f)-[:PLAYS_HOME]->(t) THEN f.away_score
            ELSE f.home_score
          END AS goals_conceded,
          CASE
            WHEN (f)-[:PLAYS_HOME]->(t) THEN f.home_score
            ELSE f.away_score
          END AS goals_scored
        ORDER BY f.kickoff_time DESC
        LIMIT $num_matches
        RETURN collect(goals_conceded) AS conceded_list,
               collect(goals_scored) AS scored_list,
               count(*) AS matches_played
        """
        
        try:
            results = self.conn.execute_query(
                query,
                {
                    "team_name": team_name,
                    "current_date": current_date,
                    "num_matches": num_matches
                }
            )
            
            if not results or results[0]["matches_played"] == 0:
                logger.info(f"No historical data for {team_name}, using league averages")
                return {
                    "defensive_strength": LEAGUE_AVG_GOALS,
                    "offensive_strength": LEAGUE_AVG_GOALS
                }
            
            result = results[0]
            conceded_list = result["conceded_list"]
            scored_list = result["scored_list"]
            matches_played = result["matches_played"]
            
            # Calculate averages
            defensive_strength = sum(conceded_list) / len(conceded_list) if conceded_list else LEAGUE_AVG_GOALS
            offensive_strength = sum(scored_list) / len(scored_list) if scored_list else LEAGUE_AVG_GOALS
            
            logger.info(
                f"Opponent strength for {team_name} (last {matches_played} matches): "
                f"defensive={defensive_strength:.2f}, offensive={offensive_strength:.2f}"
            )
            
            return {
                "defensive_strength": defensive_strength,
                "offensive_strength": offensive_strength
            }
            
        except Exception as e:
            logger.error(f"Error calculating opponent strength for {team_name}: {e}")
            # Return league averages on error
            return {
                "defensive_strength": LEAGUE_AVG_GOALS,
                "offensive_strength": LEAGUE_AVG_GOALS
            }

    
    def validate_fixture_relationships(self, fixture_id: str) -> List[str]:
        """
        Validate that a fixture has exactly one PLAYS_HOME and one PLAYS_AWAY relationship.
        
        Args:
            fixture_id: Fixture identifier to validate
        
        Returns:
            List of validation error messages (empty list if valid)
        
        **Validates: Requirements 9.2**
        """
        errors = []
        
        try:
            # Count PLAYS_HOME relationships
            home_query = """
            MATCH (f:Fixture {id: $fixture_id})-[r:PLAYS_HOME]->(:Team)
            RETURN count(r) AS home_count
            """
            home_results = self.conn.execute_query(home_query, {"fixture_id": fixture_id})
            
            if not home_results:
                errors.append(f"Fixture {fixture_id} not found")
                return errors
            
            home_count = home_results[0]["home_count"]
            
            if home_count == 0:
                errors.append(f"Fixture {fixture_id} missing PLAYS_HOME relationship")
            elif home_count > 1:
                errors.append(f"Fixture {fixture_id} has {home_count} PLAYS_HOME relationships (expected 1)")
            
            # Count PLAYS_AWAY relationships
            away_query = """
            MATCH (f:Fixture {id: $fixture_id})-[r:PLAYS_AWAY]->(:Team)
            RETURN count(r) AS away_count
            """
            away_results = self.conn.execute_query(away_query, {"fixture_id": fixture_id})
            away_count = away_results[0]["away_count"]
            
            if away_count == 0:
                errors.append(f"Fixture {fixture_id} missing PLAYS_AWAY relationship")
            elif away_count > 1:
                errors.append(f"Fixture {fixture_id} has {away_count} PLAYS_AWAY relationships (expected 1)")
            
            if not errors:
                logger.debug(f"Fixture {fixture_id} relationships are valid")
            else:
                logger.warning(f"Fixture {fixture_id} validation errors: {errors}")
            
            return errors
            
        except Exception as e:
            logger.error(f"Error validating fixture relationships: {e}")
            return [f"Validation error: {str(e)}"]
    
    def delete_fixture(self, fixture_id: str) -> bool:
        """
        Delete a fixture and all its relationships (cascade delete).
        
        Args:
            fixture_id: Fixture identifier to delete
        
        Returns:
            True if successful, False if fixture not found
        
        **Validates: Requirements 9.5**
        """
        try:
            # Check if fixture exists
            check_query = "MATCH (f:Fixture {id: $fixture_id}) RETURN f"
            results = self.conn.execute_query(check_query, {"fixture_id": fixture_id})
            
            if not results:
                logger.warning(f"Fixture not found for deletion: {fixture_id}")
                return False
            
            # Delete fixture and all relationships using DETACH DELETE
            # DETACH DELETE automatically removes all relationships
            delete_query = """
            MATCH (f:Fixture {id: $fixture_id})
            DETACH DELETE f
            """
            
            self.conn.execute_write(delete_query, {"fixture_id": fixture_id})
            
            # Invalidate cached predictions for this fixture
            if self.cache:
                cache_pattern = f"prediction:*:{fixture_id}:*"
                self.cache.invalidate_pattern(cache_pattern)
                logger.info(f"Invalidated cached predictions for deleted fixture {fixture_id}")
            
            logger.info(f"Successfully deleted fixture {fixture_id} and all relationships")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting fixture {fixture_id}: {e}")
            return False
