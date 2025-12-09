"""
Cypher Query Library for FPL Knowledge Graph
Contains 15+ parameterized query templates for various FPL operations
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class QueryResult:
    """Container for query results with metadata."""
    query: str
    parameters: Dict[str, Any]
    results: List[Dict]
    description: str


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
        if season:
            query = """
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
            WITH p, SUM(r.goals_scored) AS total_goals, SUM(r.total_points) AS total_points
            WHERE total_goals > 0
            RETURN p.name AS player_name, total_goals, total_points
            ORDER BY total_goals DESC
            LIMIT $limit
            """
            return query, {"season": season, "limit": limit}
        else:
            query = """
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
            WITH p, SUM(r.goals_scored) AS total_goals, SUM(r.total_points) AS total_points
            WHERE total_goals > 0
            RETURN p.name AS player_name, total_goals, total_points
            ORDER BY total_goals DESC
            LIMIT $limit
            """
            return query, {"limit": limit}
    
    @staticmethod
    def get_top_assisters_by_season(season: str = None, limit: int = 10) -> tuple:
        """
        Query 2: Get top assist providers in a specific season or all seasons.
        """
        if season:
            query = """
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
            WITH p, SUM(r.assists) AS total_assists, SUM(r.total_points) AS total_points
            WHERE total_assists > 0
            RETURN p.name AS player_name, total_assists, total_points
            ORDER BY total_assists DESC
            LIMIT $limit
            """
            return query, {"season": season, "limit": limit}
        else:
            query = """
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
            WITH p, SUM(r.assists) AS total_assists, SUM(r.total_points) AS total_points
            WHERE total_assists > 0
            RETURN p.name AS player_name, total_assists, total_points
            ORDER BY total_assists DESC
            LIMIT $limit
            """
            return query, {"limit": limit}
    
    @staticmethod
    def get_top_points_by_position(position: str = None, season: str = None, limit: int = 10) -> tuple:
        """
        Query 3: Get top scoring players by position in a season or all seasons.
        
        Args:
            position: Position code (GK, DEF, MID, FWD) - if None, returns top 10 overall
            season: Season ID - if None, returns from all seasons
            limit: Number of results
        """
        season_filter = "MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})" if season else "MATCH (gw)-[:IN_SEASON]->(s:Season)"
        
        if position and position.upper() in ['GK', 'DEF', 'MID', 'FWD']:
            query = f"""
            MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {{code: $position}})
            MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
            {season_filter}
            WITH p, pos, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals, 
                 SUM(r.assists) AS assists, SUM(r.bonus) AS bonus
            RETURN p.name AS player_name, pos.code AS position, total_points, goals, assists, bonus
            ORDER BY total_points DESC
            LIMIT $limit
            """
            params = {"position": position.upper(), "limit": limit}
            if season:
                params["season"] = season
            return query, params
        else:
            # Return top players overall with their positions
            query = f"""
            MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
            MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
            {season_filter}
            WITH p, pos, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals, 
                 SUM(r.assists) AS assists, SUM(r.bonus) AS bonus
            RETURN pos.code AS position, p.name AS player_name, total_points, goals, assists, bonus
            ORDER BY total_points DESC
            LIMIT $limit
            """
            params = {"limit": limit}
            if season:
                params["season"] = season
            return query, params

    @staticmethod
    def get_top_players_all_positions(season: str = None, limit_per_position: int = 5) -> tuple:
        """
        Query 3b: Get top scoring players for ALL positions in a season or all seasons.
        Uses UNION to combine results from each position.
        """
        season_filter = "MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})" if season else "MATCH (gw)-[:IN_SEASON]->(s:Season)"
        
        query = f"""
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {{code: 'GK'}})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
        {season_filter}
        WITH 'GK' AS position, p.name AS player_name, 
             SUM(r.total_points) AS total_points, 
             SUM(r.goals_scored) AS goals, 
             SUM(r.assists) AS assists, 
             SUM(r.bonus) AS bonus
        ORDER BY total_points DESC
        LIMIT $limit_per_position
        RETURN position, player_name, total_points, goals, assists, bonus
        
        UNION ALL
        
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {{code: 'DEF'}})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
        {season_filter}
        WITH 'DEF' AS position, p.name AS player_name, 
             SUM(r.total_points) AS total_points, 
             SUM(r.goals_scored) AS goals, 
             SUM(r.assists) AS assists, 
             SUM(r.bonus) AS bonus
        ORDER BY total_points DESC
        LIMIT $limit_per_position
        RETURN position, player_name, total_points, goals, assists, bonus
        
        UNION ALL
        
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {{code: 'MID'}})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
        {season_filter}
        WITH 'MID' AS position, p.name AS player_name, 
             SUM(r.total_points) AS total_points, 
             SUM(r.goals_scored) AS goals, 
             SUM(r.assists) AS assists, 
             SUM(r.bonus) AS bonus
        ORDER BY total_points DESC
        LIMIT $limit_per_position
        RETURN position, player_name, total_points, goals, assists, bonus
        
        UNION ALL
        
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {{code: 'FWD'}})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
        {season_filter}
        WITH 'FWD' AS position, p.name AS player_name, 
             SUM(r.total_points) AS total_points, 
             SUM(r.goals_scored) AS goals, 
             SUM(r.assists) AS assists, 
             SUM(r.bonus) AS bonus
        ORDER BY total_points DESC
        LIMIT $limit_per_position
        RETURN position, player_name, total_points, goals, assists, bonus
        """
        params = {"limit_per_position": limit_per_position}
        if season:
            params["season"] = season
        return query, params
    
    @staticmethod
    def get_player_season_stats(player_name: str, season: str) -> tuple:
        """
        Query 4: Get comprehensive stats for a specific player in a season.
        """
        query = """
        MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        WITH p, pos, 
             SUM(r.total_points) AS total_points,
             SUM(r.goals_scored) AS goals,
             SUM(r.assists) AS assists,
             SUM(r.clean_sheets) AS clean_sheets,
             SUM(r.bonus) AS bonus,
             SUM(r.minutes) AS minutes,
             AVG(r.ict_index) AS avg_ict,
             AVG(r.influence) AS avg_influence,
             AVG(r.creativity) AS avg_creativity,
             AVG(r.threat) AS avg_threat,
             MAX(r.value) AS max_value,
             MAX(r.selected) AS max_selected,
             COUNT(f) AS games
        RETURN p.name AS player_name, pos.code AS position,
               total_points, goals, assists, clean_sheets, bonus,
               minutes, avg_ict, avg_influence, avg_creativity, avg_threat,
               max_value, max_selected, games
        """
        return query, {"player_name": player_name, "season": season}
    
    @staticmethod
    def get_player_all_seasons_stats(player_name: str) -> tuple:
        """
        Query 4b: Get comprehensive stats for a specific player across all seasons.
        Returns stats broken down by each season.
        """
        query = """
        MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
        MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        WITH p, pos, s,
             SUM(r.total_points) AS total_points,
             SUM(r.goals_scored) AS goals,
             SUM(r.assists) AS assists,
             SUM(r.clean_sheets) AS clean_sheets,
             SUM(r.bonus) AS bonus,
             SUM(r.minutes) AS minutes,
             AVG(r.ict_index) AS avg_ict,
             AVG(r.influence) AS avg_influence,
             AVG(r.creativity) AS avg_creativity,
             AVG(r.threat) AS avg_threat,
             MAX(r.value) AS max_value,
             MAX(r.selected) AS max_selected,
             COUNT(f) AS games
        RETURN p.name AS player_name, pos.code AS position, s.id AS season,
               total_points, goals, assists, clean_sheets, bonus,
               minutes, round(avg_ict, 2) AS avg_ict, round(avg_influence, 2) AS avg_influence, 
               round(avg_creativity, 2) AS avg_creativity, round(avg_threat, 2) AS avg_threat,
               max_value, max_selected, games
        ORDER BY s.id
        """
        return query, {"player_name": player_name}
    
    @staticmethod
    def get_player_gameweek_performance(player_name: str, season: str, gameweek: int) -> tuple:
        """
        Query 5: Get a player's performance in a specific gameweek.
        """
        query = """
        MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek {number: $gameweek})
        MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
        MATCH (f)-[:HOME_TEAM]->(ht:Team)
        MATCH (f)-[:AWAY_TEAM]->(at:Team)
        RETURN p.name AS player_name, gw.number AS gameweek,
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
        Query 6: Get top performing players from fixtures involving a team in a season or all seasons.
        
        Args:
            team_name: Team name
            season: Season ID (e.g., '2022-23') - if None, returns from all seasons
            limit: Number of results to return
        """
        if season:
            query = """
            MATCH (t:Team {name: $team_name})
            MATCH (f:Fixture)-[:HOME_TEAM|AWAY_TEAM]->(t)
            MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            WITH p, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals, SUM(r.assists) AS assists
            RETURN p.name AS player_name, total_points, goals, assists
            ORDER BY total_points DESC
            LIMIT $limit
            """
            return query, {"team_name": team_name, "season": season, "limit": limit}
        else:
            query = """
            MATCH (t:Team {name: $team_name})
            MATCH (f:Fixture)-[:HOME_TEAM|AWAY_TEAM]->(t)
            MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
            MATCH (p:Player)-[r:PLAYED_IN]->(f)
            WITH p, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals, SUM(r.assists) AS assists
            RETURN p.name AS player_name, total_points, goals, assists
            ORDER BY total_points DESC
            LIMIT $limit
            """
            return query, {"team_name": team_name, "limit": limit}
    
    @staticmethod
    def get_fixture_results(team_name: str, season: str = None) -> tuple:
        """
        Query 7: Get all fixture results for a team in a season or all seasons.
        
        Args:
            team_name: Team name
            season: Season ID (e.g., '2022-23') - if None, returns from all seasons
        """
        if season:
            query = """
            MATCH (t:Team {name: $team_name})
            MATCH (f:Fixture)-[:HOME_TEAM]->(ht:Team)
            MATCH (f)-[:AWAY_TEAM]->(at:Team)
            WHERE ht.name = $team_name OR at.name = $team_name
            MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
            RETURN s.id AS season, gw.number AS gameweek, ht.name AS home_team, at.name AS away_team,
                   f.home_score AS home_score, f.away_score AS away_score,
                   f.kickoff_time AS kickoff_time
            ORDER BY gw.number
            """
            return query, {"team_name": team_name, "season": season}
        else:
            query = """
            MATCH (t:Team {name: $team_name})
            MATCH (f:Fixture)-[:HOME_TEAM]->(ht:Team)
            MATCH (f)-[:AWAY_TEAM]->(at:Team)
            WHERE ht.name = $team_name OR at.name = $team_name
            MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
            RETURN s.id AS season, gw.number AS gameweek, ht.name AS home_team, at.name AS away_team,
                   f.home_score AS home_score, f.away_score AS away_score,
                   f.kickoff_time AS kickoff_time
            ORDER BY s.id, gw.number
            """
            return query, {"team_name": team_name}
    
    @staticmethod
    def get_head_to_head(team1: str, team2: str) -> tuple:
        """
        Query 8: Get head-to-head results between two teams.
        """
        query = """
        MATCH (f:Fixture)-[:HOME_TEAM]->(ht:Team)
        MATCH (f)-[:AWAY_TEAM]->(at:Team)
        WHERE (ht.name = $team1 AND at.name = $team2) OR (ht.name = $team2 AND at.name = $team1)
        MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
        RETURN s.id AS season, gw.number AS gameweek, 
               ht.name AS home_team, at.name AS away_team,
               f.home_score AS home_score, f.away_score AS away_score
        ORDER BY s.id, gw.number
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
        if position:
            query = """
            MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: $position})
            MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
            WITH p, pos, SUM(r.total_points) AS total_points, AVG(r.value) AS avg_value
            WHERE avg_value > 0
            WITH p, pos, total_points, avg_value, (total_points * 10.0 / avg_value) AS points_per_million
            RETURN p.name AS player_name, pos.code AS position, total_points, 
                   avg_value / 10.0 AS value_millions, round(points_per_million, 2) AS points_per_million
            ORDER BY points_per_million DESC
            LIMIT $limit
            """
            return query, {"season": season, "position": position, "limit": limit}
        else:
            query = """
            MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
            MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
            WITH p, pos, SUM(r.total_points) AS total_points, AVG(r.value) AS avg_value
            WHERE avg_value > 0
            WITH p, pos, total_points, avg_value, (total_points * 10.0 / avg_value) AS points_per_million
            RETURN p.name AS player_name, pos.code AS position, total_points,
                   avg_value / 10.0 AS value_millions, round(points_per_million, 2) AS points_per_million
            ORDER BY points_per_million DESC
            LIMIT $limit
            """
            return query, {"season": season, "limit": limit}
    
    @staticmethod
    def get_most_transferred_players(season: str, gameweek: int, direction: str = "in", limit: int = 10) -> tuple:
        """
        Query 10: Get most transferred in/out players in a gameweek.
        
        Args:
            direction: 'in' for transfers in, 'out' for transfers out
        """
        transfer_field = "transfers_in" if direction == "in" else "transfers_out"
        query = f"""
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek {{number: $gameweek}})
        MATCH (gw)-[:IN_SEASON]->(s:Season {{id: $season}})
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
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
        WITH p, SUM(r.bonus) AS total_bonus, SUM(r.bps) AS total_bps, COUNT(f) AS games
        WHERE total_bonus > 0
        RETURN p.name AS player_name, total_bonus, total_bps, games,
               round(total_bonus * 1.0 / games, 2) AS bonus_per_game
        ORDER BY total_bonus DESC
        LIMIT $limit
        """
        return query, {"season": season, "limit": limit}
    
    @staticmethod
    def get_clean_sheet_leaders(season: str, limit: int = 10) -> tuple:
        """
        Query 12: Get players (DEF/GK) with most clean sheets.
        """
        query = """
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
        WHERE pos.code IN ['GK', 'DEF']
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
        WITH p, pos, SUM(r.clean_sheets) AS total_clean_sheets, SUM(r.total_points) AS total_points
        WHERE total_clean_sheets > 0
        RETURN p.name AS player_name, pos.code AS position, total_clean_sheets, total_points
        ORDER BY total_clean_sheets DESC
        LIMIT $limit
        """
        return query, {"season": season, "limit": limit}
    
    @staticmethod
    def get_ict_index_leaders(season: str, limit: int = 10) -> tuple:
        """
        Query 13: Get players with highest average ICT index.
        """
        query = """
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
        WHERE r.minutes > 0
        WITH p, pos, AVG(r.ict_index) AS avg_ict, AVG(r.influence) AS avg_influence,
             AVG(r.creativity) AS avg_creativity, AVG(r.threat) AS avg_threat,
             SUM(r.total_points) AS total_points, COUNT(f) AS games
        WHERE games >= 10
        RETURN p.name AS player_name, pos.code AS position,
               round(avg_ict, 2) AS avg_ict_index,
               round(avg_influence, 2) AS avg_influence,
               round(avg_creativity, 2) AS avg_creativity,
               round(avg_threat, 2) AS avg_threat,
               total_points, games
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
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek {number: $gameweek})
        MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
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
        if season:
            query = """
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
            WHERE p.name IN [$player1, $player2]
            MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
            WITH p, pos,
                 SUM(r.total_points) AS total_points,
                 SUM(r.goals_scored) AS goals,
                 SUM(r.assists) AS assists,
                 SUM(r.clean_sheets) AS clean_sheets,
                 SUM(r.bonus) AS bonus,
                 SUM(r.minutes) AS minutes,
                 AVG(r.ict_index) AS avg_ict,
                 AVG(r.value) AS avg_value,
                 COUNT(f) AS games
            RETURN p.name AS player_name, pos.code AS position,
                   total_points, goals, assists, clean_sheets, bonus, minutes,
                   round(avg_ict, 2) AS avg_ict_index, 
                   round(avg_value / 10.0, 2) AS avg_value_millions,
                   games
            ORDER BY total_points DESC
            """
            return query, {"player1": player1, "player2": player2, "season": season}
        else:
            query = """
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
            WHERE p.name IN [$player1, $player2]
            MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
            WITH p, pos,
                 SUM(r.total_points) AS total_points,
                 SUM(r.goals_scored) AS goals,
                 SUM(r.assists) AS assists,
                 SUM(r.clean_sheets) AS clean_sheets,
                 SUM(r.bonus) AS bonus,
                 SUM(r.minutes) AS minutes,
                 AVG(r.ict_index) AS avg_ict,
                 AVG(r.value) AS avg_value,
                 COUNT(f) AS games
            RETURN p.name AS player_name, pos.code AS position,
                   total_points, goals, assists, clean_sheets, bonus, minutes,
                   round(avg_ict, 2) AS avg_ict_index, 
                   round(avg_value / 10.0, 2) AS avg_value_millions,
                   games
            ORDER BY total_points DESC
            """
            return query, {"player1": player1, "player2": player2}
    
    @staticmethod
    def get_player_form_history(player_name: str, season: str = None) -> tuple:
        """
        Query 16: Get a player's form progression throughout the season(s).
        """
        if season:
            query = """
            MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
            MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
            RETURN gw.number AS gameweek, r.total_points AS points, r.form AS form,
                   r.goals_scored AS goals, r.assists AS assists, r.minutes AS minutes, s.id AS season
            ORDER BY gw.number
            """
            return query, {"player_name": player_name, "season": season}
        else:
            query = """
            MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
            MATCH (gw)-[:IN_SEASON]->(s:Season)
            RETURN s.id AS season, gw.number AS gameweek, r.total_points AS points, r.form AS form,
                   r.goals_scored AS goals, r.assists AS assists, r.minutes AS minutes
            ORDER BY s.id, gw.number
            """
            return query, {"player_name": player_name}
    
    @staticmethod
    def find_similar_players_kg(player_name: str, season: str = None, limit: int = 10) -> tuple:
        """
        Query: Find similar players based on knowledge graph data.
        Matches players in the same position with similar stats.
        """
        if season:
            query = """
            // First get the reference player's stats
            MATCH (ref:Player {name: $player_name})-[:PLAYS_POSITION]->(refPos:Position)
            MATCH (ref)-[rRef:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
            WITH ref, refPos, 
                 SUM(rRef.total_points) AS ref_points,
                 SUM(rRef.goals_scored) AS ref_goals,
                 SUM(rRef.assists) AS ref_assists
            
            // Find other players in same position with similar stats
            MATCH (p:Player)-[:PLAYS_POSITION]->(refPos)
            WHERE p.name <> $player_name
            MATCH (p)-[r:PLAYED_IN]->(f2:Fixture)-[:PART_OF]->(gw2:Gameweek)-[:IN_SEASON]->(s2:Season {id: $season})
            WITH ref, ref_points, ref_goals, ref_assists, p, refPos,
                 SUM(r.total_points) AS total_points,
                 SUM(r.goals_scored) AS goals,
                 SUM(r.assists) AS assists,
                 SUM(r.bonus) AS bonus,
                 COUNT(r) AS games
            
            // Calculate similarity score (lower is more similar)
            WITH p, refPos.code AS position, total_points, goals, assists, bonus, games,
                 abs(total_points - ref_points) + abs(goals - ref_goals) * 10 + abs(assists - ref_assists) * 5 AS diff_score
            
            RETURN p.name AS player_name, position, total_points, goals, assists, bonus, games, diff_score
            ORDER BY diff_score ASC
            LIMIT $limit
            """
            return query, {"player_name": player_name, "season": season, "limit": limit}
        else:
            query = """
            // First get the reference player's stats across all seasons
            MATCH (ref:Player {name: $player_name})-[:PLAYS_POSITION]->(refPos:Position)
            MATCH (ref)-[rRef:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
            WITH ref, refPos, 
                 SUM(rRef.total_points) AS ref_points,
                 SUM(rRef.goals_scored) AS ref_goals,
                 SUM(rRef.assists) AS ref_assists
            
            // Find other players in same position with similar stats
            MATCH (p:Player)-[:PLAYS_POSITION]->(refPos)
            WHERE p.name <> $player_name
            MATCH (p)-[r:PLAYED_IN]->(f2:Fixture)-[:PART_OF]->(gw2:Gameweek)-[:IN_SEASON]->(s2:Season)
            WITH ref, ref_points, ref_goals, ref_assists, p, refPos,
                 SUM(r.total_points) AS total_points,
                 SUM(r.goals_scored) AS goals,
                 SUM(r.assists) AS assists,
                 SUM(r.bonus) AS bonus,
                 COUNT(r) AS games
            
            // Calculate similarity score (lower is more similar)
            WITH p, refPos.code AS position, total_points, goals, assists, bonus, games,
                 abs(total_points - ref_points) + abs(goals - ref_goals) * 10 + abs(assists - ref_assists) * 5 AS diff_score
            
            RETURN p.name AS player_name, position, total_points, goals, assists, bonus, games, diff_score
            ORDER BY diff_score ASC
            LIMIT $limit
            """
            return query, {"player_name": player_name, "limit": limit}
    
    # ===========================================
    # SEARCH QUERIES
    # ===========================================
    
    @staticmethod
    def search_players_by_name(name_pattern: str, limit: int = 20) -> tuple:
        """
        Query 17: Search players by name (case-insensitive partial match).
        """
        query = """
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
        WHERE toLower(p.name) CONTAINS toLower($name_pattern)
        RETURN p.name AS player_name, pos.code AS position, p.element_id AS element_id
        ORDER BY p.name
        LIMIT $limit
        """
        return query, {"name_pattern": name_pattern, "limit": limit}
    
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
        MATCH (gw:Gameweek)-[:IN_SEASON]->(s)
        WITH s, COUNT(DISTINCT gw) AS total_gameweeks
        MATCH (f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s)
        WITH s, total_gameweeks, COUNT(DISTINCT f) AS total_fixtures,
             SUM(f.home_score + f.away_score) AS total_goals
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(:Gameweek)-[:IN_SEASON]->(s)
        WITH s, total_gameweeks, total_fixtures, total_goals,
             SUM(r.total_points) AS total_fpl_points, COUNT(DISTINCT p) AS total_players
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
        MATCH (gw:Gameweek)-[:IN_SEASON]->(s)
        WITH s, COUNT(DISTINCT gw) AS total_gameweeks
        MATCH (f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s)
        WITH s, total_gameweeks, COUNT(DISTINCT f) AS total_fixtures,
             SUM(f.home_score + f.away_score) AS total_goals
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(:Gameweek)-[:IN_SEASON]->(s)
        WITH s, total_gameweeks, total_fixtures, total_goals,
             SUM(r.total_points) AS total_fpl_points, COUNT(DISTINCT p) AS total_players
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
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
        WHERE r.total_points > 0
        RETURN p.name AS player_name, gw.number AS gameweek, r.total_points AS points,
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
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {{id: $season}})
        WITH p, SUM(r.{card_field}) AS total_cards
        WHERE total_cards > 0
        RETURN p.name AS player_name, total_cards
        ORDER BY total_cards DESC
        LIMIT 5
        """
        return query, {"season": season}
    
    @staticmethod
    def get_goalkeeper_saves_leader(season: str) -> tuple:
        """
        Query for trivia: Get goalkeeper with most saves.
        """
        query = """
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: 'GK'})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
        WITH p, SUM(r.saves) AS total_saves, SUM(r.clean_sheets) AS clean_sheets
        WHERE total_saves > 0
        RETURN p.name AS player_name, total_saves, clean_sheets
        ORDER BY total_saves DESC
        LIMIT 5
        """
        return query, {"season": season}
    
    @staticmethod
    def get_highest_scoring_fixture(season: str) -> tuple:
        """
        Query for trivia: Get the highest scoring fixture.
        """
        query = """
        MATCH (f:Fixture)-[:HOME_TEAM]->(ht:Team)
        MATCH (f)-[:AWAY_TEAM]->(at:Team)
        MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
        WITH f, ht, at, gw, (f.home_score + f.away_score) AS total_goals
        RETURN ht.name AS home_team, at.name AS away_team, 
               f.home_score AS home_score, f.away_score AS away_score,
               total_goals, gw.number AS gameweek
        ORDER BY total_goals DESC
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
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
        OPTIONAL MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
        WITH p, s, pos,
             SUM(r.total_points) as total_points,
             SUM(r.goals_scored) as goals_scored,
             SUM(r.assists) as assists,
             SUM(r.clean_sheets) as clean_sheets,
             SUM(r.bonus) as bonus,
             SUM(r.minutes) as minutes,
             AVG(r.ict_index) as ict_index,
             AVG(r.influence) as influence,
             AVG(r.creativity) as creativity,
             AVG(r.threat) as threat,
             AVG(r.value) as value,
             MAX(r.selected) as selected,
             COUNT(r) as games
        RETURN p.name as name, 
               COALESCE(s.name, s.id) as season,
               COALESCE(pos.code, 'Unknown') as position,
               total_points, goals_scored, assists, clean_sheets,
               bonus, minutes, ict_index, influence, creativity,
               threat, value, selected, games
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
