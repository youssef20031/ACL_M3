"""
Cypher Query Library for FPL Knowledge Graph
Contains 15+ parameterized query templates for various FPL operations

Season-level questions read the precomputed (Player)-[:SEASON_STATS]->(Season)
aggregates (one edge per player-season, built by the data loader) instead of
summing every (Player)-[:PLAYED_IN]->(Fixture) match row. SEASON_STATS stores sums;
averages are sum / games so they stay correct when seasons are added together.

Gameweek-level questions start from the matching Fixture nodes (which carry
season_id/gameweek) and expand to their PLAYED_IN rows. On Aura this is several
times faster than a relationship-property index seek.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from config.settings import CURRENT_SEASON
from utils.text import normalize_name


@dataclass
class QueryResult:
    """Container for query results with metadata."""
    query: str
    parameters: Dict[str, Any]
    results: List[Dict]
    description: str


POSITION_ORDER = "CASE position WHEN 'GK' THEN 0 WHEN 'DEF' THEN 1 WHEN 'MID' THEN 2 ELSE 3 END"


POSITIONS = ("GK", "DEF", "MID", "FWD")


def _season_node(season: Optional[str]) -> str:
    return "(s:Season {id: $season})" if season else "(s:Season)"


def _with_season(params: Dict[str, Any], season: Optional[str]) -> Dict[str, Any]:
    if season:
        params["season"] = season
    return params


def _player_totals(season: Optional[str], gameweek: Optional[int]) -> str:
    """MATCH + WITH producing per-player total_points/goals/assists/bonus/games."""
    if gameweek:
        season_prop = ", season_id: $season" if season else ""
        return f"""
        MATCH (:Fixture {{gameweek: $gameweek{season_prop}}})<-[r:PLAYED_IN]-(p:Player)
        WITH p, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals,
             SUM(r.assists) AS assists, SUM(r.bonus) AS bonus, COUNT(r) AS games
        """
    return f"""
        MATCH (p:Player)-[st:SEASON_STATS]->{_season_node(season)}
        WITH p, SUM(st.total_points) AS total_points, SUM(st.goals) AS goals,
             SUM(st.assists) AS assists, SUM(st.bonus) AS bonus, SUM(st.games) AS games
        """


class CypherQueries:
    """Library of Cypher queries for FPL Graph-RAG system."""

    # ===========================================
    # PLAYER STATISTICS QUERIES
    # ===========================================

    @staticmethod
    def get_top_scorers_by_season(season: str = None, limit: int = 10) -> tuple:
        """
        Query 1: Get top goal scorers in a specific season or all seasons.

        Args:
            season: Season ID (e.g., '2022-23') - if None, returns from all seasons
            limit: Number of results to return
        """
        query = f"""
        MATCH (p:Player)-[st:SEASON_STATS]->{_season_node(season)}
        WITH p, SUM(st.goals) AS total_goals, SUM(st.total_points) AS total_points
        WHERE total_goals > 0
        RETURN p.name AS player_name, total_goals, total_points
        ORDER BY total_goals DESC, total_points DESC, player_name
        LIMIT $limit
        """
        return query, _with_season({"limit": limit}, season)

    @staticmethod
    def get_top_assisters_by_season(season: str = None, limit: int = 10) -> tuple:
        """
        Query 2: Get top assist providers in a specific season or all seasons.
        """
        query = f"""
        MATCH (p:Player)-[st:SEASON_STATS]->{_season_node(season)}
        WITH p, SUM(st.assists) AS total_assists, SUM(st.total_points) AS total_points
        WHERE total_assists > 0
        RETURN p.name AS player_name, total_assists, total_points
        ORDER BY total_assists DESC, total_points DESC, player_name
        LIMIT $limit
        """
        return query, _with_season({"limit": limit}, season)

    @staticmethod
    def get_top_points_by_position(position: str = None, season: str = None, gameweek: int = None, sort_by: str = "total_points", limit: int = 10) -> tuple:
        """
        Query 3: Get top scoring players by position in a season or all seasons.
        Supports dynamic sorting by the following metrics:
        - 'total_points' / 'points' (default)
        - 'goals' / 'goals_scored'
        - 'assists'
        - 'bonus' / 'bps'

        Args:
            position: Position code (GK, DEF, MID, FWD) - if None, returns top 10 overall
            season: Season ID - if None, returns from all seasons
            gameweek: Gameweek number (1-38) - if specified, filter to that specific gameweek
            sort_by: Metric to sort by (e.g., 'goals', 'assists', 'bonus').
            limit: Number of results
        """
        valid_sorts = {
            "total_points": "total_points",
            "points": "total_points",
            "points_per_game": "points_per_game",
            "goals": "goals",
            "goals_scored": "goals",
            "assists": "assists",
            "bonus": "bonus",
            "bps": "bonus"
        }
        sort_field = valid_sorts.get(sort_by, "total_points")

        params: Dict[str, Any] = {"limit": limit}
        _with_season(params, season)
        code = position.upper() if position and position.upper() in POSITIONS else None
        if code:
            params["position"] = code

        if gameweek:
            # Per-match rows: the position filter uses the player's current position.
            params["gameweek"] = gameweek
            season_prop = ", season_id: $season" if season else ""
            pos_props = " {code: $position}" if code else ""
            totals = f"""
        MATCH (:Fixture {{gameweek: $gameweek{season_prop}}})<-[r:PLAYED_IN]-(p:Player)
        WITH p, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals,
             SUM(r.assists) AS assists, SUM(r.bonus) AS bonus, COUNT(r) AS games,
             head(collect(r.team)) AS team
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position{pos_props})
        WITH p, pos.code AS position, team, total_points, goals, assists, bonus, games
        """
        else:
            # SEASON_STATS carries the team and position the player had that season, so
            # "midfielders in 2023-24" means midfielders then, not today's position.
            pos_filter = "WHERE st.position = $position" if code else ""
            totals = f"""
        MATCH (p:Player)-[st:SEASON_STATS]->{_season_node(season)}
        {pos_filter}
        WITH p, SUM(st.total_points) AS total_points, SUM(st.goals) AS goals,
             SUM(st.assists) AS assists, SUM(st.bonus) AS bonus, SUM(st.games) AS games,
             collect(st.team) AS teams, collect(st.position) AS positions
        OPTIONAL MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        WITH p, total_points, goals, assists, bonus, games,
             CASE WHEN size(teams) = 1 THEN teams[0] ELSE coalesce(p.last_team, last(teams)) END AS team,
             CASE WHEN size(positions) = 1 THEN positions[0] ELSE {"$position" if code else "pos.code"} END AS position
        """

        query = f"""
        {totals}
        WITH p, position, team, total_points, goals, assists, bonus, games,
             total_points * 1.0 / CASE WHEN games = 0 THEN 1 ELSE games END AS points_per_game
        RETURN p.name AS player_name, position, team, total_points, goals, assists, bonus, games,
               round(points_per_game, 2) AS points_per_game
        ORDER BY {sort_field} DESC, total_points DESC, player_name
        LIMIT $limit
        """
        return query, params

    @staticmethod
    def get_top_players_all_positions(season: str = None, gameweek: int = None, limit_per_position: int = 5) -> tuple:
        """
        Query 3b: Get top scoring players for ALL positions in a season or all seasons.

        Args:
            season: Season ID - if None, returns from all seasons
            gameweek: Gameweek number (1-38) - if specified, filter to that specific gameweek
            limit_per_position: Number of results per position
        """
        query = f"""
        {_player_totals(season, gameweek)}
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        WITH pos.code AS position, p.name AS player_name, total_points, goals, assists, bonus, games
        ORDER BY total_points DESC
        WITH position, collect({{
            player_name: player_name, total_points: total_points, goals: goals,
            assists: assists, bonus: bonus, games: games
        }})[..$limit_per_position] AS top
        UNWIND top AS row
        RETURN position, row.player_name AS player_name, row.total_points AS total_points,
               row.goals AS goals, row.assists AS assists, row.bonus AS bonus, row.games AS games,
               round(row.total_points * 1.0 / CASE WHEN row.games = 0 THEN 1 ELSE row.games END, 2) AS points_per_game
        ORDER BY {POSITION_ORDER}, total_points DESC
        """
        params: Dict[str, Any] = {"limit_per_position": limit_per_position}
        _with_season(params, season)
        if gameweek:
            params["gameweek"] = gameweek
        return query, params

    @staticmethod
    def get_player_season_stats(player_name: str, season: str) -> tuple:
        """
        Query 4: Get comprehensive stats for a specific player in a season.
        """
        query = """
        MATCH (p:Player {name: $player_name})-[st:SEASON_STATS]->(s:Season {id: $season})
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        RETURN p.name AS player_name, pos.code AS position,
               st.total_points AS total_points, st.goals AS goals, st.assists AS assists,
               st.clean_sheets AS clean_sheets, st.bonus AS bonus, st.minutes AS minutes,
               st.ict_sum * 1.0 / st.games AS avg_ict,
               st.influence_sum * 1.0 / st.games AS avg_influence,
               st.creativity_sum * 1.0 / st.games AS avg_creativity,
               st.threat_sum * 1.0 / st.games AS avg_threat,
               st.max_value AS max_value, st.max_selected AS max_selected, st.games AS games
        """
        return query, {"player_name": player_name, "season": season}

    @staticmethod
    def get_player_all_seasons_stats(player_name: str) -> tuple:
        """
        Query 4b: Get comprehensive stats for a specific player across all seasons.
        Returns stats broken down by each season.
        """
        query = """
        MATCH (p:Player {name: $player_name})-[st:SEASON_STATS]->(s:Season)
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        RETURN p.name AS player_name, pos.code AS position, s.id AS season,
               st.total_points AS total_points, st.goals AS goals, st.assists AS assists,
               st.clean_sheets AS clean_sheets, st.bonus AS bonus, st.minutes AS minutes,
               round(st.ict_sum * 1.0 / st.games, 2) AS avg_ict,
               round(st.influence_sum * 1.0 / st.games, 2) AS avg_influence,
               round(st.creativity_sum * 1.0 / st.games, 2) AS avg_creativity,
               round(st.threat_sum * 1.0 / st.games, 2) AS avg_threat,
               st.max_value AS max_value, st.max_selected AS max_selected, st.games AS games
        ORDER BY s.id
        """
        return query, {"player_name": player_name}

    @staticmethod
    def get_player_career_stats(player_name: str) -> tuple:
        """
        Query 4c: A player's combined stats across every season.
        """
        query = """
        MATCH (p:Player {name: $player_name})-[st:SEASON_STATS]->(:Season)
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        WITH p, pos,
             SUM(st.total_points) AS total_points, SUM(st.goals) AS goals,
             SUM(st.assists) AS assists, SUM(st.clean_sheets) AS clean_sheets,
             SUM(st.bonus) AS bonus, SUM(st.minutes) AS minutes,
             SUM(st.ict_sum) AS ict_sum, MAX(st.max_value) AS max_value,
             MAX(st.max_selected) AS max_selected, SUM(st.games) AS games
        RETURN p.name AS player_name, pos.code AS position, 'All seasons' AS season,
               total_points, goals, assists, clean_sheets, bonus, minutes,
               round(ict_sum * 1.0 / games, 2) AS avg_ict_index,
               round(max_value / 10.0, 2) AS avg_value_millions,
               max_selected, games
        """
        return query, {"player_name": player_name}

    @staticmethod
    def get_player_gameweek_performance(player_name: str, season: str, gameweek: int) -> tuple:
        """
        Query 5: Get a player's performance in a specific gameweek.
        """
        query = """
        MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)
        WHERE r.season = $season AND r.gameweek = $gameweek
        MATCH (f)-[:HOME_TEAM]->(ht:Team)
        MATCH (f)-[:AWAY_TEAM]->(at:Team)
        RETURN p.name AS player_name, r.gameweek AS gameweek,
               r.total_points AS points, r.goals_scored AS goals, r.assists AS assists,
               r.minutes AS minutes, r.bonus AS bonus, r.bps AS bps,
               r.ict_index AS ict_index, r.clean_sheets AS clean_sheets,
               ht.name AS home_team, at.name AS away_team,
               f.home_score AS home_score, f.away_score AS away_score
        """
        return query, {"player_name": player_name, "season": season, "gameweek": gameweek}

    # ===========================================
    # TEAM ANALYSIS QUERIES
    # ===========================================

    @staticmethod
    def get_team_top_performers(team_name: str, season: str = None, limit: int = 5) -> tuple:
        """
        Query 6: Get the top performing players who played for a team in a season or all seasons.

        Args:
            team_name: Team name
            season: Season ID (e.g., '2022-23') - if None, returns from all seasons
            limit: Number of results to return
        """
        query = f"""
        MATCH (p:Player)-[st:SEASON_STATS]->{_season_node(season)}
        WHERE st.team = $team_name
        WITH p, SUM(st.total_points) AS total_points, SUM(st.goals) AS goals, SUM(st.assists) AS assists
        RETURN p.name AS player_name, total_points, goals, assists
        ORDER BY total_points DESC, player_name
        LIMIT $limit
        """
        return query, _with_season({"team_name": team_name, "limit": limit}, season)

    @staticmethod
    def get_team_squad(team_name: str, season: str = None, position: str = None, limit: int = 40) -> tuple:
        """
        A club's squad, optionally one position: "Who is the Coventry City goalkeeper?",
        "Arsenal defenders", "Who plays for Villa?".

        For the current season (or no season) it is today's squad, (Player)-[:PLAYS_FOR]->(Team)
        with the player's current position, FPL status and news, plus his SEASON_STATS for this
        season (zeros for someone yet to play). For a past season it is everyone with SEASON_STATS for the club
        that season, in the position he had then.

        Args:
            team_name: Team name as in the graph
            season: Season ID (e.g., '2022-23'); None means the current season
            position: GK, DEF, MID or FWD; None for the whole squad
            limit: Number of results to return
        """
        season = season or CURRENT_SEASON
        if season == CURRENT_SEASON:
            pos_filter = "WHERE pos.code = $position" if position else ""
            query = f"""
        MATCH (p:Player)-[:PLAYS_FOR]->(:Team {{name: $team_name}})
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        {pos_filter}
        OPTIONAL MATCH (p)-[st:SEASON_STATS]->(:Season {{id: $season}})
        WITH p, pos.code AS position, st, p.status AS status, p.news AS news
        """
        else:
            pos_filter = "AND st.position = $position" if position else ""
            query = f"""
        MATCH (p:Player)-[st:SEASON_STATS]->(:Season {{id: $season}})
        WHERE st.team = $team_name {pos_filter}
        WITH p, st.position AS position, st, null AS status, null AS news
        """
        query += f"""
        RETURN p.name AS player_name, position, $team_name AS team,
               coalesce(st.total_points, 0) AS total_points, coalesce(st.minutes, 0) AS minutes,
               coalesce(st.appearances, st.games, 0) AS games, coalesce(st.goals, 0) AS goals,
               coalesce(st.assists, 0) AS assists, status, news
        ORDER BY {POSITION_ORDER}, total_points DESC, minutes DESC, player_name
        LIMIT $limit
        """
        params = {"team_name": team_name, "season": season, "limit": limit}
        if position:
            params["position"] = position
        return query, params

    @staticmethod
    def get_fixture_results(team_name: str, season: str = None) -> tuple:
        """
        Query 7: Get all fixture results for a team in a season or all seasons.

        Args:
            team_name: Team name
            season: Season ID (e.g., '2022-23') - if None, returns from all seasons
        """
        # Played matches only: the current season's unplayed fixtures have no score
        season_filter = "AND f.season_id = $season" if season else ""
        query = f"""
        MATCH (:Team {{name: $team_name}})<-[:HOME_TEAM|AWAY_TEAM]-(f:Fixture)
        WHERE f.home_score IS NOT NULL {season_filter}
        MATCH (f)-[:HOME_TEAM]->(ht:Team)
        MATCH (f)-[:AWAY_TEAM]->(at:Team)
        RETURN f.season_id AS season, f.gameweek AS gameweek, ht.name AS home_team, at.name AS away_team,
               f.home_score AS home_score, f.away_score AS away_score,
               toString(f.kickoff_time) AS kickoff_time
        ORDER BY season, gameweek
        """
        return query, _with_season({"team_name": team_name}, season)

    @staticmethod
    def get_head_to_head(team1: str, team2: str) -> tuple:
        """
        Query 8: Get head-to-head results between two teams.
        """
        query = """
        MATCH (:Team {name: $team1})<-[:HOME_TEAM|AWAY_TEAM]-(f:Fixture)-[:HOME_TEAM|AWAY_TEAM]->(:Team {name: $team2})
        WHERE f.home_score IS NOT NULL
        MATCH (f)-[:HOME_TEAM]->(ht:Team)
        MATCH (f)-[:AWAY_TEAM]->(at:Team)
        RETURN f.season_id AS season, f.gameweek AS gameweek,
               ht.name AS home_team, at.name AS away_team,
               f.home_score AS home_score, f.away_score AS away_score
        ORDER BY season, gameweek
        """
        return query, {"team1": team1, "team2": team2}

    # ===========================================
    # VALUE & TRANSFER ANALYSIS QUERIES
    # ===========================================

    @staticmethod
    def get_best_value_players(season: str, position: Optional[str] = None, limit: int = 10) -> tuple:
        """
        Query 9: Get best value players (points per million).
        """
        pos_node = "(pos:Position {code: $position})" if position else "(pos:Position)"
        query = f"""
        MATCH (p:Player)-[st:SEASON_STATS]->(s:Season {{id: $season}})
        WHERE st.games > 0 AND st.value_sum > 0
        MATCH (p)-[:PLAYS_POSITION]->{pos_node}
        WITH p, pos, st.total_points AS total_points, st.value_sum * 1.0 / st.games AS avg_value
        WITH p, pos, total_points, avg_value, (total_points * 10.0 / avg_value) AS points_per_million
        RETURN p.name AS player_name, pos.code AS position, total_points,
               avg_value / 10.0 AS value_millions, round(points_per_million, 2) AS points_per_million
        ORDER BY points_per_million DESC
        LIMIT $limit
        """
        params: Dict[str, Any] = {"season": season, "limit": limit}
        if position:
            params["position"] = position
        return query, params

    @staticmethod
    def get_most_transferred_players(season: str, gameweek: int, direction: str = "in", limit: int = 10) -> tuple:
        """
        Query 10: Get most transferred in/out players in a gameweek.

        Args:
            direction: 'in' for transfers in, 'out' for transfers out
        """
        transfer_field = "transfers_in" if direction == "in" else "transfers_out"
        query = f"""
        MATCH (:Fixture {{season_id: $season, gameweek: $gameweek}})<-[r:PLAYED_IN]-(p:Player)
        RETURN p.name AS player_name, r.{transfer_field} AS transfers, r.total_points AS points
        ORDER BY r.{transfer_field} DESC
        LIMIT $limit
        """
        return query, {"season": season, "gameweek": gameweek, "limit": limit}

    # ===========================================
    # PERFORMANCE METRICS QUERIES
    # ===========================================

    @staticmethod
    def get_bonus_point_leaders(season: str, limit: int = 10) -> tuple:
        """
        Query 11: Get players with most bonus points in a season.
        """
        query = """
        MATCH (p:Player)-[st:SEASON_STATS]->(s:Season {id: $season})
        WHERE st.bonus > 0
        RETURN p.name AS player_name, st.bonus AS total_bonus, st.bps AS total_bps, st.games AS games,
               round(st.bonus * 1.0 / st.games, 2) AS bonus_per_game
        ORDER BY total_bonus DESC, total_bps DESC, player_name
        LIMIT $limit
        """
        return query, {"season": season, "limit": limit}

    @staticmethod
    def get_clean_sheet_leaders(season: str = None, position: str = None, limit: int = 10) -> tuple:
        """
        Query 12: Get players (DEF/GK) with most clean sheets.
        """
        params: Dict[str, Any] = {"limit": limit}
        if position and position.upper() in ['GK', 'DEF']:
            params["positions"] = [position.upper()]
        else:
            params["positions"] = ['GK', 'DEF']
        _with_season(params, season)

        query = f"""
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
        WHERE pos.code IN $positions
        MATCH (p)-[st:SEASON_STATS]->{_season_node(season)}
        WITH p, pos, SUM(st.clean_sheets) AS total_clean_sheets, SUM(st.total_points) AS total_points,
             SUM(st.games) AS games
        WHERE total_clean_sheets > 0
        RETURN p.name AS player_name, pos.code AS position, total_clean_sheets, total_points, games,
               round(total_clean_sheets * 1.0 / CASE WHEN games = 0 THEN 1 ELSE games END, 2) AS clean_sheets_per_game
        ORDER BY total_clean_sheets DESC, total_points DESC, player_name
        LIMIT $limit
        """
        return query, params

    @staticmethod
    def get_ict_index_leaders(season: str, limit: int = 10) -> tuple:
        """
        Query 13: Get players with highest average ICT index (per appearance, min. 10 appearances).
        """
        query = """
        MATCH (p:Player)-[st:SEASON_STATS]->(s:Season {id: $season})
        WHERE st.appearances >= 10
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        WITH p, pos, st, st.ict_played_sum * 1.0 / st.appearances AS avg_ict
        RETURN p.name AS player_name, pos.code AS position,
               round(avg_ict, 2) AS avg_ict_index,
               round(st.influence_played_sum * 1.0 / st.appearances, 2) AS avg_influence,
               round(st.creativity_played_sum * 1.0 / st.appearances, 2) AS avg_creativity,
               round(st.threat_played_sum * 1.0 / st.appearances, 2) AS avg_threat,
               st.total_points AS total_points, st.appearances AS games
        ORDER BY avg_ict DESC
        LIMIT $limit
        """
        return query, {"season": season, "limit": limit}

    @staticmethod
    def get_most_selected_players(season: str, gameweek: int, limit: int = 10) -> tuple:
        """
        Query 14: Get most selected players in a specific gameweek.
        """
        query = """
        MATCH (:Fixture {season_id: $season, gameweek: $gameweek})<-[r:PLAYED_IN]-(p:Player)
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        RETURN p.name AS player_name, pos.code AS position, r.selected AS selected,
               r.value / 10.0 AS value_millions, r.total_points AS points
        ORDER BY r.selected DESC
        LIMIT $limit
        """
        return query, {"season": season, "gameweek": gameweek, "limit": limit}

    # ===========================================
    # COMPARISON QUERIES
    # ===========================================

    @staticmethod
    def compare_players(player1: str, player2: str, season: str = None) -> tuple:
        """
        Query 15: Compare two players' season statistics.
        """
        query = f"""
        MATCH (p:Player)-[st:SEASON_STATS]->{_season_node(season)}
        WHERE p.name IN [$player1, $player2]
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team)
        WITH p, pos, coalesce(t.name, p.last_team) AS team_name,
             SUM(st.total_points) AS total_points,
             SUM(st.goals) AS goals,
             SUM(st.assists) AS assists,
             SUM(st.clean_sheets) AS clean_sheets,
             SUM(st.bonus) AS bonus,
             SUM(st.minutes) AS minutes,
             SUM(st.ict_sum) AS ict_sum,
             SUM(st.form_sum) AS form_sum,
             SUM(st.value_sum) AS value_sum,
             SUM(st.games) AS games
        RETURN p.name AS player_name, pos.code AS position, team_name,
               total_points, goals, assists, clean_sheets, bonus, minutes,
               round(ict_sum * 1.0 / games, 2) AS avg_ict_index,
               round(form_sum * 1.0 / games, 2) AS avg_form,
               round(value_sum / 10.0 / games, 2) AS avg_value_millions,
               games
        ORDER BY total_points DESC
        """
        return query, _with_season({"player1": player1, "player2": player2}, season)

    @staticmethod
    def get_player_form_history(player_name: str, season: str = None) -> tuple:
        """
        Query 16: Get a player's form progression throughout the season(s).
        """
        season_filter = "WHERE r.season = $season" if season else ""
        query = f"""
        MATCH (p:Player {{name: $player_name}})-[r:PLAYED_IN]->(:Fixture)
        {season_filter}
        RETURN r.season AS season, r.gameweek AS gameweek, r.total_points AS points, r.form AS form,
               r.goals_scored AS goals, r.assists AS assists, r.minutes AS minutes
        ORDER BY season, gameweek
        """
        return query, _with_season({"player_name": player_name}, season)

    @staticmethod
    def find_similar_players_kg(player_name: str, season: str = None, limit: int = 10) -> tuple:
        """
        Query: Find similar players based on knowledge graph data.
        Matches players in the same position with similar stats.
        """
        season_node = _season_node(season)
        query = f"""
        MATCH (ref:Player {{name: $player_name}})-[:PLAYS_POSITION]->(refPos:Position)
        MATCH (ref)-[rRef:SEASON_STATS]->{season_node}
        WITH refPos,
             SUM(rRef.total_points) AS ref_points,
             SUM(rRef.goals) AS ref_goals,
             SUM(rRef.assists) AS ref_assists

        MATCH (p:Player)-[:PLAYS_POSITION]->(refPos)
        WHERE p.name <> $player_name
        MATCH (p)-[st:SEASON_STATS]->{season_node.replace('s:', 's2:')}
        WITH ref_points, ref_goals, ref_assists, p, refPos,
             SUM(st.total_points) AS total_points,
             SUM(st.goals) AS goals,
             SUM(st.assists) AS assists,
             SUM(st.bonus) AS bonus,
             SUM(st.games) AS games

        // Similarity score (lower is more similar)
        WITH p, refPos.code AS position, total_points, goals, assists, bonus, games,
             abs(total_points - ref_points) + abs(goals - ref_goals) * 10 + abs(assists - ref_assists) * 5 AS diff_score

        RETURN p.name AS player_name, position, total_points, goals, assists, bonus, games, diff_score
        ORDER BY diff_score ASC
        LIMIT $limit
        """
        return query, _with_season({"player_name": player_name, "limit": limit}, season)

    # ===========================================
    # SEARCH QUERIES
    # ===========================================

    @staticmethod
    def search_players_by_name(name_pattern: str, limit: int = 20) -> tuple:
        """
        Query 17: Search players by name (accent/case-insensitive partial match).
        """
        query = """
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
        WHERE p.name_normalized CONTAINS $name_pattern
        OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team)
        RETURN p.name AS player_name, pos.code AS position, p.element_id AS element_id,
               coalesce(t.name, p.last_team) AS team_name
        ORDER BY p.name
        LIMIT $limit
        """
        return query, {"name_pattern": normalize_name(name_pattern), "limit": limit}

    @staticmethod
    def get_all_player_names() -> tuple:
        """
        Query 17b: Get all player names for fuzzy matching.
        """
        query = """
        MATCH (p:Player)
        RETURN DISTINCT p.name AS player_name
        ORDER BY p.name
        """
        return query, {}

    @staticmethod
    def get_all_players_by_position(position: str) -> tuple:
        """
        Query 18: Get all players in a specific position.
        """
        query = """
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: $position})
        RETURN p.name AS player_name, p.element_id AS element_id
        ORDER BY p.name
        """
        return query, {"position": position}

    @staticmethod
    def get_all_teams() -> tuple:
        """
        Query 19: Get all teams in the database.
        """
        query = """
        MATCH (t:Team)
        RETURN t.name AS team_name
        ORDER BY t.name
        """
        return query, {}

    @staticmethod
    def get_season_summary(season: str) -> tuple:
        """
        Query 20: Get overall season summary statistics.
        """
        query = """
        MATCH (s:Season {id: $season})
        // Played fixtures only, so a season in progress reports what has happened so far
        OPTIONAL MATCH (f:Fixture {season_id: s.id})-[:PART_OF]->(gw:Gameweek)
        WHERE f.home_score IS NOT NULL
        WITH s, COUNT(DISTINCT gw) AS total_gameweeks, COUNT(f) AS total_fixtures,
             SUM(f.home_score + f.away_score) AS total_goals
        WHERE total_fixtures > 0
        OPTIONAL MATCH (:Player)-[st:SEASON_STATS]->(s)
        WITH s, total_gameweeks, total_fixtures, total_goals,
             SUM(st.total_points) AS total_fpl_points, COUNT(st) AS total_players
        RETURN s.id AS season, total_gameweeks, total_fixtures, total_goals,
               total_fpl_points, total_players,
               round(total_goals * 1.0 / total_fixtures, 2) AS goals_per_game
        """
        return query, {"season": season}

    @staticmethod
    def get_all_seasons_summary() -> tuple:
        """
        Query 20b: Get overall summary statistics for all seasons.
        """
        query = """
        MATCH (s:Season)
        // Played fixtures only, so a season in progress reports what has happened so far
        OPTIONAL MATCH (f:Fixture {season_id: s.id})-[:PART_OF]->(gw:Gameweek)
        WHERE f.home_score IS NOT NULL
        WITH s, COUNT(DISTINCT gw) AS total_gameweeks, COUNT(f) AS total_fixtures,
             SUM(f.home_score + f.away_score) AS total_goals
        WHERE total_fixtures > 0
        OPTIONAL MATCH (:Player)-[st:SEASON_STATS]->(s)
        WITH s, total_gameweeks, total_fixtures, total_goals,
             SUM(st.total_points) AS total_fpl_points, COUNT(st) AS total_players
        RETURN s.id AS season, total_gameweeks, total_fixtures, total_goals,
               total_fpl_points, total_players,
               round(total_goals * 1.0 / total_fixtures, 2) AS goals_per_game
        ORDER BY s.id
        """
        return query, {}

    # ===========================================
    # TRIVIA-SPECIFIC QUERIES
    # ===========================================

    @staticmethod
    def get_highest_single_gameweek_score(season: str) -> tuple:
        """
        Query for trivia: Get the highest single gameweek score.
        """
        query = """
        MATCH (:Fixture {season_id: $season})<-[r:PLAYED_IN]-(p:Player)
        WHERE r.total_points > 0
        RETURN p.name AS player_name, r.gameweek AS gameweek, r.total_points AS points,
               r.goals_scored AS goals, r.assists AS assists, r.bonus AS bonus
        ORDER BY r.total_points DESC
        LIMIT 1
        """
        return query, {"season": season}

    @staticmethod
    def get_player_with_most_cards(season: str, card_type: str = "yellow") -> tuple:
        """
        Query for trivia: Get player with most yellow/red cards.
        """
        card_field = "yellow_cards" if card_type == "yellow" else "red_cards"
        query = f"""
        MATCH (p:Player)-[st:SEASON_STATS]->(s:Season {{id: $season}})
        WHERE st.{card_field} > 0
        RETURN p.name AS player_name, st.{card_field} AS total_cards
        ORDER BY total_cards DESC, player_name
        LIMIT 5
        """
        return query, {"season": season}

    @staticmethod
    def get_goalkeeper_saves_leader(season: str) -> tuple:
        """
        Query for trivia: Get goalkeeper with most saves.
        """
        query = """
        MATCH (p:Player)-[:PLAYS_POSITION]->(:Position {code: 'GK'})
        MATCH (p)-[st:SEASON_STATS]->(s:Season {id: $season})
        WHERE st.saves > 0
        RETURN p.name AS player_name, st.saves AS total_saves, st.clean_sheets AS clean_sheets
        ORDER BY total_saves DESC, player_name
        LIMIT 5
        """
        return query, {"season": season}

    @staticmethod
    def get_highest_scoring_fixture(season: str) -> tuple:
        """
        Query for trivia: Get the highest scoring fixture.
        """
        query = """
        MATCH (f:Fixture {season_id: $season})-[:HOME_TEAM]->(ht:Team)
        WHERE f.home_score IS NOT NULL
        MATCH (f)-[:AWAY_TEAM]->(at:Team)
        WITH f, ht, at, (f.home_score + f.away_score) AS total_goals
        RETURN ht.name AS home_team, at.name AS away_team,
               f.home_score AS home_score, f.away_score AS away_score,
               total_goals, f.gameweek AS gameweek
        ORDER BY total_goals DESC, gameweek, home_team
        LIMIT 5
        """
        return query, {"season": season}

    @staticmethod
    def get_player_embeddings_data() -> tuple:
        """
        Query for building player embeddings: Get aggregated player stats per season.
        Returns player performance data suitable for embedding generation.
        """
        query = """
        MATCH (p:Player)-[st:SEASON_STATS]->(s:Season)
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        RETURN p.name AS name,
               COALESCE(s.name, s.id) AS season,
               COALESCE(pos.code, 'Unknown') AS position,
               st.total_points AS total_points, st.goals AS goals_scored, st.assists AS assists,
               st.clean_sheets AS clean_sheets, st.bonus AS bonus, st.bps AS bps,
               st.minutes AS minutes,
               st.ict_sum * 1.0 / st.games AS ict_index,
               st.influence_sum * 1.0 / st.games AS influence,
               st.creativity_sum * 1.0 / st.games AS creativity,
               st.threat_sum * 1.0 / st.games AS threat,
               st.value_sum * 1.0 / st.games AS value,
               st.max_selected AS selected, st.saves AS saves,
               st.goals_conceded AS goals_conceded,
               st.yellow_cards AS yellow_cards, st.red_cards AS red_cards,
               st.games AS games,
               [st.team] AS teams
        """
        return query, {}


class QueryExecutor:
    """Executes Cypher queries and formats results."""

    def __init__(self, connection):
        """
        Initialize query executor.

        Args:
            connection: Neo4j connection instance
        """
        self.conn = connection
        self.queries = CypherQueries()

    def execute(self, query: str, parameters: Dict[str, Any]) -> List[Dict]:
        """
        Execute a Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result dictionaries
        """
        return self.conn.execute_query(query, parameters)

    def execute_query_method(self, method_name: str, **kwargs) -> QueryResult:
        """
        Execute a query method from CypherQueries class.

        Args:
            method_name: Name of the query method
            **kwargs: Arguments for the query method

        Returns:
            QueryResult with query, parameters, and results
        """
        query_method = getattr(self.queries, method_name)
        query, params = query_method(**kwargs)
        results = self.execute(query, params)

        return QueryResult(
            query=query,
            parameters=params,
            results=results,
            description=query_method.__doc__ or ""
        )
