"""
FPL Data Loader - Loads CSV data into Neo4j Knowledge Graph
"""
import pandas as pd
from typing import Dict, List, Any
from tqdm import tqdm
import logging
from .connection import Neo4jConnection
from .schema import GraphSchema
from utils.text import normalize_name

logger = logging.getLogger(__name__)

# Rows per UNWIND batch. Large batches keep round trips to a remote (Aura) instance low.
BATCH_SIZE = 5000

# Per-match columns summed into SEASON_STATS. Averages are derived as sum / games
# at query time so they stay correct when several seasons are added together.
SEASON_SUM_COLUMNS = {
    "total_points": "total_points",
    "goals_scored": "goals",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "bonus": "bonus",
    "bps": "bps",
    "minutes": "minutes",
    "saves": "saves",
    "goals_conceded": "goals_conceded",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "ict_index": "ict_sum",
    "influence": "influence_sum",
    "creativity": "creativity_sum",
    "threat": "threat_sum",
    "value": "value_sum",
    "form": "form_sum",
}


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
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

        # Transform columns to match expected schema
        df = self.transform_columns(df)

        return df

    def transform_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform CSV columns to match expected schema.
        Handles column renaming and creates home_team/away_team columns.

        Args:
            df: Raw DataFrame from CSV

        Returns:
            Transformed DataFrame with standardized column names
        """
        logger.info("Transforming columns to match expected schema...")

        if 'season_x' in df.columns and 'season' not in df.columns:
            df = df.rename(columns={'season_x': 'season'})

        # Normalize position codes (e.g., GKP -> GK). "AM" rows are the 2024-25
        # assistant-manager chip — managers, not players — so they are not loaded.
        if 'position' in df.columns:
            df['position'] = df['position'].replace({'GKP': 'GK'})
            dropped = int((df['position'] == 'AM').sum())
            if dropped:
                df = df[df['position'] != 'AM'].copy()
                logger.info(f"Dropped {dropped} assistant-manager (AM) rows")

        # When was_home is True team_x is the home side, otherwise opp_team_name is
        if 'team_x' in df.columns and 'opp_team_name' in df.columns and 'was_home' in df.columns:
            home = df['was_home'].astype(bool)
            if 'home_team' not in df.columns:
                df['home_team'] = df['team_x'].where(home, df['opp_team_name'])
            if 'away_team' not in df.columns:
                df['away_team'] = df['opp_team_name'].where(home, df['team_x'])

        # Add form column if missing: rolling mean of the last 5 gameweeks' points
        if 'form' not in df.columns:
            df = df.sort_values(['name', 'season', 'GW', 'kickoff_time'])
            df['form'] = (
                df.groupby(['name', 'season'])['total_points']
                .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
                .round(1)
            )

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

        # A player's identity is their name; position/team/element come from their
        # most recent appearance so each player gets exactly one PLAYS_POSITION.
        latest = (
            df.sort_values(['season', 'GW', 'kickoff_time'])
            .groupby('name', sort=False)
            .tail(1)
        )
        latest_season = df['season'].max()
        players = [
            {
                'name': row.name_,
                'name_normalized': normalize_name(row.name_),
                'position': row.position,
                'element_id': int(row.element),
                'team': row.team_x,
                'last_season': row.season,
                'is_current': row.season == latest_season,
            }
            for row in latest.rename(columns={'name': 'name_'}).itertuples(index=False)
        ]
        logger.info(f"Found {len(players)} unique players")

        teams = sorted(set(df['home_team']) | set(df['away_team']))
        logger.info(f"Found {len(teams)} unique teams")

        seasons = sorted(df['season'].unique().tolist())
        logger.info(f"Found {len(seasons)} seasons: {seasons}")

        gameweeks = df[['season', 'GW']].drop_duplicates().to_dict('records')
        logger.info(f"Found {len(gameweeks)} unique gameweeks")

        fixtures = df[[
            'fixture', 'season', 'GW', 'home_team', 'away_team',
            'team_h_score', 'team_a_score', 'kickoff_time'
        ]].drop_duplicates(subset=['fixture', 'season']).to_dict('records')
        logger.info(f"Found {len(fixtures)} unique fixtures")

        performances = self.performance_records(df)
        logger.info(f"Prepared {len(performances)} performance records")

        return {
            'players': players,
            'teams': teams,
            'seasons': seasons,
            'gameweeks': gameweeks,
            'fixtures': fixtures,
            'performances': performances,
            'season_stats': self.build_season_stats(df),
        }

    @staticmethod
    def season_file_frame(df: pd.DataFrame, season: str) -> pd.DataFrame:
        """Adapt a single-season file (FPL_YYYY_YYYY.csv layout: `team`, no `season`)
        to the merged-CSV layout the rest of the loader expects."""
        df = df.copy()
        if 'team_x' not in df.columns and 'team' in df.columns:
            df = df.rename(columns={'team': 'team_x'})
        df['season'] = season
        return df

    @staticmethod
    def performance_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """One PLAYED_IN property map per (player, fixture) row, NaN -> None."""
        perf_columns = [
            'name', 'season', 'fixture', 'GW', 'minutes', 'goals_scored', 'assists',
            'clean_sheets', 'bonus', 'bps', 'total_points', 'ict_index',
            'influence', 'creativity', 'threat', 'value', 'form',
            'selected', 'transfers_in', 'transfers_out', 'transfers_balance',
            'goals_conceded', 'own_goals', 'penalties_missed', 'penalties_saved',
            'saves', 'yellow_cards', 'red_cards', 'team_x', 'was_home',
        ]
        # Optional enrichment columns from newer seasons — include only if present
        optional_perf_columns = [
            'xP', 'expected_goals', 'expected_assists',
            'expected_goal_involvements', 'expected_goals_conceded',
            'starts', 'tackles', 'recoveries',
            'clearances_blocks_interceptions', 'defensive_contribution',
        ]
        perf_columns += [c for c in optional_perf_columns if c in df.columns]
        perf_df = df[perf_columns].astype(object).where(df[perf_columns].notna(), None)
        return perf_df.to_dict('records')

    def build_season_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Aggregate per-match rows into one record per (player, season)."""
        grouped = df.groupby(['name', 'season'], sort=False)
        stats = grouped[list(SEASON_SUM_COLUMNS)].sum().rename(columns=SEASON_SUM_COLUMNS)
        stats['games'] = grouped.size()
        stats['max_value'] = grouped['value'].max()
        stats['max_selected'] = grouped['selected'].max()

        played = df[df['minutes'] > 0].groupby(['name', 'season'], sort=False)
        stats['appearances'] = played.size()
        stats['ict_played_sum'] = played['ict_index'].sum()
        stats['influence_played_sum'] = played['influence'].sum()
        stats['creativity_played_sum'] = played['creativity'].sum()
        stats['threat_played_sum'] = played['threat'].sum()

        # Team and position the player had most often that season
        stats['team'] = grouped['team_x'].agg(lambda s: s.mode().iat[0])
        stats['position'] = grouped['position'].agg(lambda s: s.mode().iat[0])

        stats = stats.fillna(0).reset_index()
        stats['appearances'] = stats['appearances'].astype(int)
        records = stats.to_dict('records')
        for rec in records:
            for key, val in rec.items():
                if hasattr(val, 'item'):  # numpy scalar -> python
                    rec[key] = val.item()
        logger.info(f"Built {len(records)} player-season aggregates")
        return records

    def _run_batches(self, query: str, rows: List[Dict[str, Any]], desc: str, batch_size: int = BATCH_SIZE):
        for i in tqdm(range(0, len(rows), batch_size), desc=desc):
            self.conn.execute_write(query, {"batch": rows[i:i + batch_size]})

    def create_team_nodes(self, teams: List[str]):
        """Create Team nodes."""
        query = """
        UNWIND $teams AS team_name
        MERGE (t:Team {name: team_name})
        """
        self.conn.execute_write(query, {"teams": teams})
        logger.info(f"Created {len(teams)} Team nodes")

    def create_gameweek_nodes(self, gameweeks: List[Dict]):
        """Create Gameweek nodes with relationships to seasons."""
        query = """
        UNWIND $batch AS gw
        MERGE (gameweek:Gameweek {id: gw.season + '-GW' + toString(gw.GW)})
        SET gameweek.number = gw.GW,
            gameweek.season_id = gw.season
        WITH gameweek, gw
        MATCH (s:Season {id: gw.season})
        MERGE (gameweek)-[:IN_SEASON]->(s)
        """
        self._run_batches(query, gameweeks, "Gameweeks")

    def create_player_nodes(self, players: List[Dict]):
        """Create Player nodes with position and current-team relationships."""
        query = """
        UNWIND $batch AS player
        MERGE (p:Player {name: player.name})
        SET p.element_id = player.element_id,
            p.name_normalized = player.name_normalized,
            p.last_team = player.team,
            p.last_season = player.last_season
        WITH p, player
        MATCH (pos:Position {code: player.position})
        MERGE (p)-[:PLAYS_POSITION]->(pos)
        WITH p, player
        // PLAYS_FOR means "current team", so only players from the latest season get one
        WHERE player.is_current
        MATCH (t:Team {name: player.team})
        MERGE (p)-[r:PLAYS_FOR]->(t)
        SET r.season = player.last_season
        """
        self._run_batches(query, players, "Players")

    def create_fixture_nodes(self, fixtures: List[Dict]):
        """Create Fixture nodes with team and gameweek relationships."""
        query = """
        UNWIND $batch AS fix
        MERGE (f:Fixture {id: toString(fix.fixture) + '-' + fix.season})
        SET f.fixture_id = fix.fixture,
            // Stored as a DateTime (as 2026-27 fixtures are) so ordering and
            // comparisons with datetime parameters work across every season
            f.kickoff_time = CASE WHEN fix.kickoff_time IS NULL THEN null ELSE datetime(fix.kickoff_time) END,
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
        self._run_batches(query, fixtures, "Fixtures")

    def create_performance_relationships(self, performances: List[Dict]):
        """Create PLAYED_IN relationships with performance stats."""
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
            r.gameweek = perf.GW,
            r.season = perf.season,
            r.team = perf.team_x,
            r.was_home = perf.was_home,
            // Optional enrichment columns (null when absent for older seasons)
            r.xP = perf.xP,
            r.expected_goals = perf.expected_goals,
            r.expected_assists = perf.expected_assists,
            r.expected_goal_involvements = perf.expected_goal_involvements,
            r.expected_goals_conceded = perf.expected_goals_conceded,
            r.starts = perf.starts,
            r.tackles = perf.tackles,
            r.recoveries = perf.recoveries,
            r.clearances_blocks_interceptions = perf.clearances_blocks_interceptions,
            r.defensive_contribution = perf.defensive_contribution
        """
        self._run_batches(query, performances, "Performances")
        logger.info(f"Created {len(performances)} performance relationships")

    def create_season_stats(self, season_stats: List[Dict]):
        """Create precomputed (Player)-[:SEASON_STATS]->(Season) aggregates."""
        query = """
        UNWIND $batch AS row
        MATCH (p:Player {name: row.name})
        MATCH (s:Season {id: row.season})
        MERGE (p)-[st:SEASON_STATS]->(s)
        SET st += row
        REMOVE st.name
        """
        self._run_batches(query, season_stats, "Season stats")

    def replace_season_stats(self, season: str, season_stats: List[Dict]):
        """Drop one season's SEASON_STATS and write them again (idempotent refresh)."""
        self.conn.execute_write(
            "MATCH (:Player)-[st:SEASON_STATS]->(:Season {id: $season}) DELETE st",
            {"season": season},
        )
        self.create_season_stats(season_stats)

    def convert_kickoff_strings(self) -> int:
        """Turn Fixture.kickoff_time strings (graphs loaded before kickoff times were
        DateTimes) into DateTimes. A no-op once converted."""
        rows = self.conn.execute_query(
            """
            MATCH (f:Fixture) WHERE f.kickoff_time IS :: STRING
            SET f.kickoff_time = datetime(f.kickoff_time)
            RETURN count(f) AS converted
            """
        )
        return rows[0]["converted"] if rows else 0

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

        self.schema.initialize_schema()
        self.schema.create_static_nodes()

        df = self.load_csv(filepath)
        data = self.prepare_data(df)

        self.create_team_nodes(data['teams'])
        self.create_gameweek_nodes(data['gameweeks'])
        self.create_player_nodes(data['players'])
        self.create_fixture_nodes(data['fixtures'])
        self.create_performance_relationships(data['performances'])
        self.create_season_stats(data['season_stats'])

        stats = self.conn.get_database_stats()
        logger.info("Data load complete!")
        logger.info(f"Total nodes: {stats['total_nodes']}")
        logger.info(f"Total relationships: {stats['total_relationships']}")
        logger.info(f"Node labels: {stats['node_labels']}")

        return stats
