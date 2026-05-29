"""
Entity Extraction for FPL Query Processing using spaCy
"""
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import re


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
        "goalkeepers": "GK",
        "gk": "GK",
        "keeper": "GK",
        "keepers": "GK",
        "goalie": "GK",
        "goalies": "GK",
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
        Initialize entity extractor with spaCy.
        
        Args:
            known_players: Optional set of known player names for better extraction
        """
        self.known_players = known_players or set()
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Attempting to download...")
            try:
                import subprocess
                import sys
                # Use -q flag for quiet download to reduce output
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                               check=True, capture_output=True, timeout=120)  # 2 minute timeout
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"Error downloading spaCy model: {e}")
                # Fallback to a blank model if download fails
                self.nlp = spacy.blank("en")
                print("Using blank English model as fallback.")

                self.nlp = spacy.blank("en")
        
        # Initialize matcher for patterns
        self.matcher = Matcher(self.nlp.vocab)
        self._add_patterns()
    
    def _add_patterns(self):
        """Add spaCy matcher patterns for FPL entities."""
        # Gameweek patterns
        gameweek_patterns = [
            [{"LOWER": {"IN": ["gw", "gameweek"]}}, {"IS_DIGIT": True}],
            [{"LOWER": "game"}, {"LOWER": "week"}, {"IS_DIGIT": True}],
            [{"LOWER": "week"}, {"IS_DIGIT": True}]
        ]
        for pattern in gameweek_patterns:
            self.matcher.add("GAMEWEEK", [pattern])
        
        # Season patterns
        season_patterns = [
            [{"SHAPE": "dddd"}, {"TEXT": "-"}, {"SHAPE": "dd"}],  # 2021-22
            [{"SHAPE": "dddd"}, {"TEXT": "/"}, {"SHAPE": "dd"}],  # 2021/22
            [{"SHAPE": "dddd"}, {"TEXT": "-"}, {"SHAPE": "dddd"}],  # 2021-2022
            [{"LOWER": {"IN": ["last", "this", "previous"]}}, {"LOWER": "season"}]
        ]
        for pattern in season_patterns:
            self.matcher.add("SEASON", [pattern])
    
    def set_known_players(self, players: Set[str]):
        """
        Set the known players list for better extraction.
        
        Args:
            players: Set of known player names
        """
        self.known_players = players
    
    def extract(self, query: str) -> ExtractedEntities:
        """
        Extract all entities from a query using spaCy.
        
        Args:
            query: User's input query
            
        Returns:
            ExtractedEntities with all found entities
        """
        entities = ExtractedEntities()
        
        # Process with spaCy
        doc = self.nlp(query)
        
        # Extract teams
        entities.teams = self._extract_teams(doc)
        
        # Extract positions
        entities.positions = self._extract_positions(doc)
        
        # Extract seasons
        entities.seasons = self._extract_seasons(doc)
        
        # Extract gameweeks
        entities.gameweeks = self._extract_gameweeks(doc)
        
        # Extract stats
        entities.stats = self._extract_stats(doc)
        
        # Extract numbers (after gameweeks to avoid duplication)
        entities.numbers = self._extract_numbers(doc, entities.gameweeks)
        
        # Extract players (do this last, using other entities for context)
        entities.players = self._extract_players(doc, entities)
        
        return entities
    
    def _extract_teams(self, doc: Doc) -> List[str]:
        """Extract team names from query using spaCy."""
        teams = []
        text_lower = doc.text.lower()
        
        # Check for team names using spaCy's ORG entities
        for ent in doc.ents:
            if ent.label_ == "ORG":
                normalized = self.TEAMS.get(ent.text.lower())
                if normalized and normalized not in teams:
                    teams.append(normalized)
        
        # Also check token sequences for team names
        for team_key, team_name in self.TEAMS.items():
            if team_key in text_lower:
                if team_name not in teams:
                    teams.append(team_name)
        
        return teams
    
    def _extract_positions(self, doc: Doc) -> List[str]:
        """Extract positions from query using spaCy tokens."""
        positions = []
        
        for token in doc:
            normalized = self.POSITIONS.get(token.text.lower())
            if normalized and normalized not in positions:
                positions.append(normalized)
        
        # Check for multi-word positions
        text_lower = doc.text.lower()
        for pos_key, pos_name in self.POSITIONS.items():
            if " " in pos_key and pos_key in text_lower:
                if pos_name not in positions:
                    positions.append(pos_name)
        
        return positions
    
    def _extract_seasons(self, doc: Doc) -> List[str]:
        """Extract seasons from query using spaCy matcher."""
        seasons = []
        text_lower = doc.text.lower()

        # Handle phrases like "season that ended in 2022" before generic year parsing.
        # This should map to the previous FPL season (2021-22), not 2022-23.
        ended_year_matches = re.findall(r"ended\s+in\s+(20\d{2})", text_lower)
        for year_text in ended_year_matches:
            year = int(year_text)
            if year > 2020:
                prev_year = year - 1
                season = f"{prev_year}-{str(year)[2:]}"
                if season in ["2020-21", "2021-22", "2022-23"] and season not in seasons:
                    seasons.append(season)

        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "SEASON":
                span = doc[start:end]
                season_text = span.text
                
                # Parse season format
                # Full year format: 2021-2022 or 2021/2022
                if re.match(r"(20\d{2})[-/](20\d{2})", season_text):
                    years = re.findall(r"20\d{2}", season_text)
                    if len(years) == 2:
                        year1, year2 = int(years[0]), int(years[1])
                        if year2 == year1 + 1:
                            season = f"{year1}-{str(year2)[2:]}"
                            if season in ["2020-21", "2021-22", "2022-23"]:
                                seasons.append(season)
                # Short format: 2021-22 or 2021/22
                elif re.match(r"20(\d{2})[-/](\d{2})", season_text):
                    match = re.match(r"20(\d{2})[-/](\d{2})", season_text)
                    season = f"20{match.group(1)}-{match.group(2)}"
                    if season in ["2020-21", "2021-22", "2022-23"]:
                        seasons.append(season)
        
        # Check for single year mentions using DATE entities
        for ent in doc.ents:
            if ent.label_ == "DATE" and ent.text.isdigit():
                year = int(ent.text)
                # Skip if this year already belongs to an "ended in YEAR" phrase.
                if re.search(rf"ended\s+in\s+{year}", text_lower):
                    continue
                if year == 2020:
                    seasons.append("2020-21")
                elif year == 2021:
                    seasons.append("2021-22")
                elif year == 2022:
                    seasons.append("2022-23")
                elif year == 2023:
                    seasons.append("2022-23")
        
        return list(set(seasons))
    
    def _extract_gameweeks(self, doc: Doc) -> List[int]:
        """Extract gameweek numbers from query using spaCy matcher."""
        gameweeks = []
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "GAMEWEEK":
                span = doc[start:end]
                # Extract the number from the span
                for token in span:
                    if token.like_num:
                        gw = int(token.text)
                        if 1 <= gw <= 38 and gw not in gameweeks:
                            gameweeks.append(gw)
        
        return gameweeks
    
    def _extract_stats(self, doc: Doc) -> List[str]:
        """Extract stat names from query using spaCy tokens."""
        stats = []
        text_lower = doc.text.lower()
        
        # Check for stats in tokens and multi-word phrases
        for stat_key, stat_name in self.STATS.items():
            if stat_key in text_lower:
                if stat_name not in stats:
                    stats.append(stat_name)
        
        return stats
    
    def _extract_numbers(self, doc: Doc, exclude_gws: List[int]) -> List[int]:
        """Extract numbers from query using spaCy (excluding gameweeks)."""
        numbers = []
        
        for token in doc:
            if token.like_num and token.text.isdigit():
                num = int(token.text)
                # Exclude gameweeks and years
                if num not in exclude_gws and num not in [2020, 2021, 2022, 2023]:
                    if num not in numbers:
                        numbers.append(num)
        
        # Also check CARDINAL entities
        for ent in doc.ents:
            if ent.label_ == "CARDINAL" and ent.text.isdigit():
                num = int(ent.text)
                if num not in exclude_gws and num not in [2020, 2021, 2022, 2023]:
                    if num not in numbers:
                        numbers.append(num)
        
        return numbers
    
    def _extract_players(self, doc: Doc, entities: ExtractedEntities) -> List[str]:
        """
        Extract player names from query using spaCy NER.
        Uses known players list and spaCy's PERSON entities.
        Supports partial matching (e.g., "Salah" matches "Mohamed Salah").
        """
        players = []
        text_lower = doc.text.lower()
        
        # First, check against known players (exact full name match)
        if self.known_players:
            for player in self.known_players:
                if player.lower() in text_lower:
                    players.append(player)
        
        # If no exact match, try partial matching (last name or common nicknames)
        if not players and self.known_players:
            query_words = set(text_lower.split())
            # Common words to exclude from matching
            common_words = {
                'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or',
                'how', 'did', 'do', 'does', 'was', 'were', 'is', 'are', 'what', 'who',
                'most', 'best', 'top', 'all', 'get', 'show', 'find', 'gameweek', 'season',
                'stats', 'points', 'goals', 'assists', 'scored', 'performance'
            }
            query_words = query_words - common_words
            
            for player in self.known_players:
                player_parts = player.lower().split()
                # Check if any part of the player name (especially last name) matches query words
                for part in player_parts:
                    if len(part) >= 4 and part in query_words:  # Minimum 4 chars to avoid false positives
                        players.append(player)
                        break
                if players:
                    break  # Found a match, stop searching
        
        # If no known players found, use spaCy's PERSON entity recognition
        if not players:
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
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Check if not a team or common word
                    if (ent.text not in entities.teams and 
                        ent.text not in non_names and
                        len(ent.text) > 2):
                        players.append(ent.text)
        
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
        
        # Derive sort_by from stats
        if entities.stats:
            stat_map = {
                "goals_scored": "goals",
                "assists": "assists",
                "total_points": "total_points",
                "bonus": "bonus",
                "ict_index": "ict_index",
                "clean_sheets": "clean_sheets",
                "transfers_in": "transfers_in",
                "transfers_out": "transfers_out",
                "selected": "selected",
                "value": "value"
            }
            # Use the first relevant stat found
            for stat in entities.stats:
                if stat in stat_map:
                    params["sort_by"] = stat_map[stat]
                    break
                    
        return params
