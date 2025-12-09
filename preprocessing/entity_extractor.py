"""
Entity Extraction for FPL Query Processing
"""
import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class ExtractedEntities:
    """Container for extracted entities from a query."""
    players: List[str] = field(default_factory=list)
    teams: List[str] = field(default_factory=list)
    positions: List[str] = field(default_factory=list)
    seasons: List[str] = field(default_factory=list)
    gameweeks: List[int] = field(default_factory=list)
    stats: List[str] = field(default_factory=list)
    numbers: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "players": self.players,
            "teams": self.teams,
            "positions": self.positions,
            "seasons": self.seasons,
            "gameweeks": self.gameweeks,
            "stats": self.stats,
            "numbers": self.numbers
        }
    
    def has_entities(self) -> bool:
        """Check if any entities were extracted."""
        return any([
            self.players, self.teams, self.positions,
            self.seasons, self.gameweeks, self.stats, self.numbers
        ])


class EntityExtractor:
    """
    Extracts FPL-specific entities from user queries.
    Uses pattern matching and known entity lists.
    """
    
    # Known teams in Premier League (2021-2023)
    TEAMS = {
        # Full names and common variations
        "arsenal": "Arsenal",
        "aston villa": "Aston Villa",
        "villa": "Aston Villa",
        "bournemouth": "Bournemouth",
        "brentford": "Brentford",
        "brighton": "Brighton",
        "brighton & hove albion": "Brighton",
        "burnley": "Burnley",
        "chelsea": "Chelsea",
        "crystal palace": "Crystal Palace",
        "palace": "Crystal Palace",
        "everton": "Everton",
        "fulham": "Fulham",
        "leeds": "Leeds",
        "leeds united": "Leeds",
        "leicester": "Leicester",
        "leicester city": "Leicester",
        "liverpool": "Liverpool",
        "manchester city": "Man City",
        "man city": "Man City",
        "city": "Man City",
        "manchester united": "Man Utd",
        "man utd": "Man Utd",
        "man united": "Man Utd",
        "united": "Man Utd",
        "newcastle": "Newcastle",
        "newcastle united": "Newcastle",
        "nottingham forest": "Nott'm Forest",
        "forest": "Nott'm Forest",
        "norwich": "Norwich",
        "norwich city": "Norwich",
        "southampton": "Southampton",
        "spurs": "Spurs",
        "tottenham": "Spurs",
        "tottenham hotspur": "Spurs",
        "watford": "Watford",
        "west ham": "West Ham",
        "west ham united": "West Ham",
        "wolves": "Wolves",
        "wolverhampton": "Wolves",
        "wolverhampton wanderers": "Wolves",
    }
    
    # Position variations
    POSITIONS = {
        "goalkeeper": "GK",
        "gk": "GK",
        "keeper": "GK",
        "goalie": "GK",
        "defender": "DEF",
        "def": "DEF",
        "defenders": "DEF",
        "defence": "DEF",
        "defense": "DEF",
        "cb": "DEF",
        "rb": "DEF",
        "lb": "DEF",
        "fullback": "DEF",
        "centre back": "DEF",
        "midfielder": "MID",
        "mid": "MID",
        "midfielders": "MID",
        "midfield": "MID",
        "cm": "MID",
        "cam": "MID",
        "cdm": "MID",
        "winger": "MID",
        "forward": "FWD",
        "fwd": "FWD",
        "forwards": "FWD",
        "striker": "FWD",
        "strikers": "FWD",
        "attacker": "FWD",
        "attackers": "FWD",
        "cf": "FWD",
        "st": "FWD",
    }
    
    # Stats variations
    STATS = {
        "goals": "goals_scored",
        "goal": "goals_scored",
        "scored": "goals_scored",
        "assists": "assists",
        "assist": "assists",
        "points": "total_points",
        "total points": "total_points",
        "fpl points": "total_points",
        "clean sheets": "clean_sheets",
        "clean sheet": "clean_sheets",
        "cleansheets": "clean_sheets",
        "cs": "clean_sheets",
        "bonus": "bonus",
        "bonus points": "bonus",
        "bps": "bps",
        "minutes": "minutes",
        "mins": "minutes",
        "ict": "ict_index",
        "ict index": "ict_index",
        "influence": "influence",
        "creativity": "creativity",
        "threat": "threat",
        "value": "value",
        "price": "value",
        "cost": "value",
        "form": "form",
        "saves": "saves",
        "save": "saves",
        "yellow cards": "yellow_cards",
        "yellows": "yellow_cards",
        "red cards": "red_cards",
        "reds": "red_cards",
        "cards": "yellow_cards",
        "transfers": "transfers_in",
        "transfers in": "transfers_in",
        "transfers out": "transfers_out",
        "selected": "selected",
        "ownership": "selected",
        "owned": "selected",
    }
    
    # Season patterns
    SEASON_PATTERNS = [
        r"20(\d{2})[-/](\d{2})",  # 2021-22, 2021/22
        r"20(\d{2})",  # 2021, 2022, 2023
        r"(\d{2})[-/](\d{2})\s+season",  # 21-22 season
        r"last season",
        r"this season",
        r"previous season",
    ]
    
    def __init__(self, known_players: Optional[Set[str]] = None):
        """
        Initialize entity extractor.
        
        Args:
            known_players: Optional set of known player names for better extraction
        """
        self.known_players = known_players or set()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        # Team pattern (case insensitive)
        team_names = "|".join(re.escape(t) for t in self.TEAMS.keys())
        self.team_pattern = re.compile(rf"\b({team_names})\b", re.IGNORECASE)
        
        # Position pattern
        position_names = "|".join(re.escape(p) for p in self.POSITIONS.keys())
        self.position_pattern = re.compile(rf"\b({position_names})\b", re.IGNORECASE)
        
        # Stats pattern
        stat_names = "|".join(re.escape(s) for s in self.STATS.keys())
        self.stats_pattern = re.compile(rf"\b({stat_names})\b", re.IGNORECASE)
        
        # Gameweek pattern
        self.gameweek_pattern = re.compile(
            r"\b(?:gw|gameweek|game week|week)\s*(\d{1,2})\b",
            re.IGNORECASE
        )
        
        # Number pattern
        self.number_pattern = re.compile(r"\b(\d+)\b")
        
        # Season pattern
        self.season_pattern = re.compile(
            r"\b20(\d{2})[-/](\d{2})\b|" +
            r"\b(2021|2022|2023)\b|" +
            r"\b(last|this|previous)\s+season\b",
            re.IGNORECASE
        )
    
    def set_known_players(self, players: Set[str]):
        """
        Set the known players list for better extraction.
        
        Args:
            players: Set of known player names
        """
        self.known_players = players
    
    def extract(self, query: str) -> ExtractedEntities:
        """
        Extract all entities from a query.
        
        Args:
            query: User's input query
            
        Returns:
            ExtractedEntities with all found entities
        """
        entities = ExtractedEntities()
        
        # Extract teams
        entities.teams = self._extract_teams(query)
        
        # Extract positions
        entities.positions = self._extract_positions(query)
        
        # Extract seasons
        entities.seasons = self._extract_seasons(query)
        
        # Extract gameweeks
        entities.gameweeks = self._extract_gameweeks(query)
        
        # Extract stats
        entities.stats = self._extract_stats(query)
        
        # Extract numbers (after gameweeks to avoid duplication)
        entities.numbers = self._extract_numbers(query, entities.gameweeks)
        
        # Extract players (do this last, using other entities for context)
        entities.players = self._extract_players(query, entities)
        
        return entities
    
    def _extract_teams(self, query: str) -> List[str]:
        """Extract team names from query."""
        matches = self.team_pattern.findall(query)
        teams = []
        for match in matches:
            normalized = self.TEAMS.get(match.lower())
            if normalized and normalized not in teams:
                teams.append(normalized)
        return teams
    
    def _extract_positions(self, query: str) -> List[str]:
        """Extract positions from query."""
        matches = self.position_pattern.findall(query)
        positions = []
        for match in matches:
            normalized = self.POSITIONS.get(match.lower())
            if normalized and normalized not in positions:
                positions.append(normalized)
        return positions
    
    def _extract_seasons(self, query: str) -> List[str]:
        """Extract seasons from query."""
        seasons = []
        
        # Check for explicit season patterns
        matches = self.season_pattern.findall(query)
        for match in matches:
            if match[0] and match[1]:  # 2021-22 format
                season = f"20{match[0]}-{match[1]}"
                if season in ["2020-21", "2021-22", "2022-23"]:
                    seasons.append(season)
            elif match[2]:  # Single year
                year = int(match[2])
                if year == 2020:
                    seasons.append("2020-21")
                elif year == 2021:
                    seasons.append("2021-22")
                elif year == 2022:
                    seasons.append("2022-23")
                elif year == 2023:
                    seasons.append("2022-23")
            elif match[3]:  # last/this/previous season
                # Note: Since we removed season selector, relative terms
                # are handled by querying all seasons in the app layer
                # We don't return a specific season here
                pass
        
        # Return empty list if no specific season mentioned
        # App will query all seasons by default
        return list(set(seasons))
    
    def _extract_gameweeks(self, query: str) -> List[int]:
        """Extract gameweek numbers from query."""
        matches = self.gameweek_pattern.findall(query)
        gameweeks = []
        for match in matches:
            gw = int(match)
            if 1 <= gw <= 38 and gw not in gameweeks:
                gameweeks.append(gw)
        return gameweeks
    
    def _extract_stats(self, query: str) -> List[str]:
        """Extract stat names from query."""
        matches = self.stats_pattern.findall(query)
        stats = []
        for match in matches:
            normalized = self.STATS.get(match.lower())
            if normalized and normalized not in stats:
                stats.append(normalized)
        return stats
    
    def _extract_numbers(self, query: str, exclude_gws: List[int]) -> List[int]:
        """Extract numbers from query (excluding gameweeks)."""
        matches = self.number_pattern.findall(query)
        numbers = []
        for match in matches:
            num = int(match)
            # Exclude gameweeks and years
            if num not in exclude_gws and num not in [2021, 2022, 2023]:
                if num not in numbers:
                    numbers.append(num)
        return numbers
    
    def _extract_players(self, query: str, entities: ExtractedEntities) -> List[str]:
        """
        Extract player names from query.
        Uses known players list and heuristics.
        """
        players = []
        
        # First, check against known players
        if self.known_players:
            query_lower = query.lower()
            for player in self.known_players:
                if player.lower() in query_lower:
                    players.append(player)
        
        # If no known players found, try to extract names heuristically
        if not players:
            # Remove known entities from query
            clean_query = query
            for team in entities.teams:
                clean_query = re.sub(re.escape(team), "", clean_query, flags=re.IGNORECASE)
            for pos in self.POSITIONS.keys():
                clean_query = re.sub(rf"\b{re.escape(pos)}\b", "", clean_query, flags=re.IGNORECASE)
            for stat in self.STATS.keys():
                clean_query = re.sub(rf"\b{re.escape(stat)}\b", "", clean_query, flags=re.IGNORECASE)
            
            # Look for capitalized words that might be names
            # Pattern: Two consecutive capitalized words
            name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
            potential_names = name_pattern.findall(query)
            
            # Filter out common non-name words
            non_names = {
                "Premier", "League", "Season", "Gameweek", "Week",
                "Top", "Best", "Most", "Points", "Goals", "Assists",
                "Who", "What", "When", "Where", "How", "Which",
                "The", "And", "For", "With", "From", "This", "That",
                "Get", "Show", "Find", "List", "Tell", "Give", "Compare",
                "Stats", "Statistics", "Performance", "Data", "Info",
                "Comprehensive", "Detailed", "All", "Seasons", "Overall",
                "Player", "Players", "Team", "Teams", "Position", "Positions"
            }
            
            for name in potential_names:
                if name not in non_names and len(name) > 2:
                    players.append(name)
        
        return players[:2]  # Limit to 2 players for comparisons
    
    def get_query_parameters(self, entities: ExtractedEntities) -> Dict:
        """
        Convert extracted entities to query parameters.
        
        Args:
            entities: Extracted entities
            
        Returns:
            Dictionary of parameters for Cypher queries
        """
        params = {}
        
        if entities.players:
            if len(entities.players) >= 2:
                params["player1"] = entities.players[0]
                params["player2"] = entities.players[1]
            else:
                params["player_name"] = entities.players[0]
        
        if entities.teams:
            if len(entities.teams) >= 2:
                params["team1"] = entities.teams[0]
                params["team2"] = entities.teams[1]
            else:
                params["team_name"] = entities.teams[0]
        
        if entities.positions:
            params["position"] = entities.positions[0]
        
        # Only set season if explicitly mentioned - don't set a default
        # This allows "all seasons" queries to work properly
        if entities.seasons:
            params["season"] = entities.seasons[0]
        
        if entities.gameweeks:
            params["gameweek"] = entities.gameweeks[0]
        
        if entities.numbers:
            # Use first number as limit if reasonable
            if entities.numbers[0] <= 50:
                params["limit"] = entities.numbers[0]
        
        return params
