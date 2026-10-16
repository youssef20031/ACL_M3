"""
Entity Extraction for FPL Query Processing using spaCy
"""
from config.settings import SEASONS
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from typing import Dict, Iterable, List, Optional, Set
from dataclasses import dataclass, field
import re

# A season written as "2022-23", "2022/23", "2022-2023" or "22/23".
SEASON_TEXT_RE = re.compile(r"\b(?:19|20)?\d{2}\s*[-/–]\s*(?:19|20)?\d{2}\b")


def _keyword_regex(keys: Iterable[str]) -> "re.Pattern[str]":
    """Match any of the (lowercase) keys as whole words.

    Longer keys are tried first, so the longest key at a position wins and the words it
    covers can't match again: "coventry city" beats "city", "bonus points" beats "points".
    Whole words keep "goal" out of "goalkeepers" and "cs" out of "statistics".
    """
    alternatives = sorted(keys, key=len, reverse=True)
    body = "|".join(r"\s+".join(re.escape(word) for word in key.split()) for key in alternatives)
    return re.compile(rf"\b(?:{body})\b")


def _find_keywords(regex: "re.Pattern[str]", table: Dict[str, str], text: str) -> List[str]:
    """The table's values for the keys found in the text, in order of appearance, no repeats."""
    found: List[str] = []
    for match in regex.finditer(text.lower()):
        value = table.get(" ".join(match.group().split()))
        if value and value not in found:
            found.append(value)
    return found


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
    
    # Every club in the graph (2020-21 onwards, including the 2026-27 promoted clubs), keyed
    # by lowercase name or nickname and mapped to the graph's Team name (FPL short names:
    # "Ipswich", "Man Utd"). Matched as whole words, longest key first (find_teams), so a
    # full name like "Coventry City" or "Leeds United" wins over the nicknames "city"/"united".
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
        "coventry": "Coventry City",
        "coventry city": "Coventry City",
        "crystal palace": "Crystal Palace",
        "palace": "Crystal Palace",
        "everton": "Everton",
        "fulham": "Fulham",
        "hull": "Hull City",
        "hull city": "Hull City",
        "ipswich": "Ipswich",
        "ipswich town": "Ipswich",
        "leeds": "Leeds",
        "leeds united": "Leeds",
        "leicester": "Leicester",
        "leicester city": "Leicester",
        "liverpool": "Liverpool",
        "luton": "Luton",
        "luton town": "Luton",
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
        "nott'm forest": "Nott'm Forest",
        "forest": "Nott'm Forest",
        "norwich": "Norwich",
        "norwich city": "Norwich",
        "sheffield united": "Sheffield Utd",
        "sheffield utd": "Sheffield Utd",
        "southampton": "Southampton",
        "spurs": "Spurs",
        "sunderland": "Sunderland",
        "tottenham": "Spurs",
        "tottenham hotspur": "Spurs",
        "watford": "Watford",
        "west brom": "West Brom",
        "west bromwich": "West Brom",
        "west bromwich albion": "West Brom",
        "west ham": "West Ham",
        "west ham united": "West Ham",
        "wolves": "Wolves",
        "wolverhampton": "Wolves",
        "wolverhampton wanderers": "Wolves",
    }
    _TEAM_RE = _keyword_regex(TEAMS)
    
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
    
    # Stats variations, matched as whole words like TEAMS (a key doesn't match inside a
    # longer word, so word forms worth catching are listed: "goalscorers", "assisters")
    STATS = {
        "goals": "goals_scored",
        "goal": "goals_scored",
        "scored": "goals_scored",
        "goalscorer": "goals_scored",
        "goalscorers": "goals_scored",
        "goalscoring": "goals_scored",
        "assists": "assists",
        "assist": "assists",
        "assisted": "assists",
        "assister": "assists",
        "assisters": "assists",
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
    _STAT_RE = _keyword_regex(STATS)
    
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
        # Optional services.player_index.PlayerIndex: accent/typo-tolerant matching that
        # prefers popular players for ambiguous surnames. Set by the API after startup.
        self.player_index = None
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
    
    def extract(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> ExtractedEntities:
        """
        Extract all entities from a query using spaCy.
        
        Args:
            query: User's input query
            chat_history: Optional conversation history for pronoun resolution
            
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
        
        # If no players found but query has pronouns, check chat history
        if not entities.players and chat_history:
            query_lower = query.lower()
            has_pronoun = bool(re.search(r'\b(he|she|they|him|her|his|hers|their)\b', query_lower))
            if has_pronoun:
                # Extract players from recent conversation
                history_players = self._extract_players_from_history(chat_history)
                if history_players:
                    entities.players = history_players
        
        return entities
    
    def _extract_players_from_history(self, chat_history: List[Dict[str, str]]) -> List[str]:
        """
        Extract player names from recent chat history for pronoun resolution.
        
        Args:
            chat_history: Conversation history
            
        Returns:
            List of player names found in recent messages
        """
        players = []
        # Check last 3 messages for player names
        recent_messages = chat_history[-3:] if len(chat_history) >= 3 else chat_history

        if self.player_index is not None:
            for msg in reversed(recent_messages):  # most recent mention wins
                found = self.player_index.mentions(msg.get('content', ''), limit=1)
                if found:
                    return found
            return []
        
        # Common words to exclude
        common_words = {
            'premier', 'league', 'season', 'gameweek', 'week', 'top', 'best', 'most',
            'points', 'goals', 'assists', 'player', 'team', 'stats', 'performance',
            'who', 'what', 'when', 'where', 'how', 'which', 'the', 'and', 'for',
            'with', 'from', 'this', 'that', 'would', 'should', 'could', 'like',
            'next', 'upcoming', 'forecast', 'predict', 'earn', 'score', 'get'
        }
        
        for msg in recent_messages:
            content = msg.get('content', '')
            # Look for capitalized words (likely proper names)
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            for word in words:
                if word.lower() not in common_words and len(word) > 3:
                    # Check if it looks like a player name (at least one capital letter after first word)
                    parts = word.split()
                    if len(parts) >= 1:  # Single name like "Haaland" or full name
                        players.append(word)
        
        return players[:1]  # Return most recent player mentioned
    
    @classmethod
    def find_teams(cls, text: str) -> List[str]:
        """Graph team names mentioned in the text, in order ("Hull City vs Man City" ->
        ["Hull City", "Man City"]; "Leeds United" -> ["Leeds"], not Man Utd as well)."""
        return _find_keywords(cls._TEAM_RE, cls.TEAMS, text)

    def _extract_teams(self, doc: Doc) -> List[str]:
        """Extract team names from the query, in the order they're mentioned.

        Substring matching used to read "Coventry City" as Man City ("city") and "Leeds
        United" as Leeds and Man Utd ("united"). spaCy's ORG entities added nothing the
        whole-word match doesn't find, so they're no longer consulted.
        """
        return self.find_teams(doc.text)
    
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
        """Extract seasons ("2024-25", "2024/25", "2024-2025", "2023", "last season").

        Only seasons present in config.settings.SEASONS are returned.
        """
        seasons: List[str] = []
        text_lower = doc.text.lower()

        def add(start_year: int) -> bool:
            season = f"{start_year}-{str(start_year + 1)[2:]}"
            if season in SEASONS and season not in seasons:
                seasons.append(season)
                return True
            return False

        # "season that ended in 2022" -> the season that finished that year (2021-22)
        ended_years = set()
        for year_text in re.findall(r"ended\s+in\s+(20\d{2})", text_lower):
            ended_years.add(year_text)
            add(int(year_text) - 1)

        # Explicit ranges: 2021-22, 2021/22, 2021-2022, 2021 - 22
        spans = []
        for m in re.finditer(r"\b(20\d{2})\s*[-/–]\s*(20\d{2}|\d{2})\b", text_lower):
            start_year, end_text = int(m.group(1)), m.group(2)
            end_year = int(end_text) if len(end_text) == 4 else 2000 + int(end_text)
            if end_year == start_year + 1:
                add(start_year)
                spans.append(m.span())

        # Relative phrases: SEASONS[-1] is the season in progress, so "this/current
        # season" is that one and "last/previous season" the one before it.
        relative = []
        if re.search(r"\b(this|current)\s+season\b", text_lower):
            relative.append(SEASONS[-1])
        if re.search(r"\b(last|previous)\s+season\b", text_lower):
            relative.append(SEASONS[-2] if len(SEASONS) > 1 else SEASONS[-1])
        for season in relative:
            if season not in seasons:
                seasons.append(season)

        # A lone year: the season starting that year, else the one ending in it
        for m in re.finditer(r"\b(20\d{2})\b", text_lower):
            if m.group(1) in ended_years or any(a <= m.start() < b for a, b in spans):
                continue
            year = int(m.group(1))
            if not add(year):
                add(year - 1)

        return seasons
    
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
        """Extract stat names from the query, in the order they're mentioned.

        Whole words only: substring matching found "goal" in "goalkeepers", so "top 3
        goalkeepers by points" ranked keepers by goals. A multi-word key covers its words,
        so "bonus points" is bonus alone, not total points too.
        """
        return _find_keywords(self._STAT_RE, self.STATS, doc.text)
    
    def _extract_numbers(self, doc: Doc, exclude_gws: List[int]) -> List[int]:
        """Extract numbers from query using spaCy (excluding gameweeks)."""
        numbers = []
        # Digits inside a season ("2022-23", "2023/24", "22/23") are not counts: spaCy splits
        # "2022-23" into "2022", "-", "23", and the "23" used to become the result limit.
        season_spans = [m.span() for m in SEASON_TEXT_RE.finditer(doc.text)]

        def usable(text: str, start: int) -> Optional[int]:
            if not text.isdigit() or any(a <= start < b for a, b in season_spans):
                return None
            num = int(text)
            if num in exclude_gws or 1900 <= num <= 2100:  # gameweeks and years
                return None
            return num

        for token in doc:
            if token.like_num:
                num = usable(token.text, token.idx)
                if num is not None and num not in numbers:
                    numbers.append(num)

        # Also check CARDINAL entities
        for ent in doc.ents:
            if ent.label_ == "CARDINAL":
                num = usable(ent.text, ent.start_char)
                if num is not None and num not in numbers:
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

        if self.player_index is not None:
            players = self.player_index.mentions(doc.text, limit=2)
        
        # First, check against known players (exact full name match)
        elif self.known_players:
            for player in self.known_players:
                if player.lower() in text_lower:
                    players.append(player)
        
        # If no exact match, try partial matching (last name or common nicknames)
        if not players and self.known_players and self.player_index is None:
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
            # What a player search (search_players_by_name) looks for. Without it "Who is
            # Bukayo Saka?" missed a required param and got the top-players fallback.
            params["name_pattern"] = entities.players[0]
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
