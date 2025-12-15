import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import re

@dataclass
class ExtractedEntities:
    players: List[str] = field(default_factory=list)
    teams: List[str] = field(default_factory=list)
    positions: List[str] = field(default_factory=list)
    seasons: List[str] = field(default_factory=list)
    gameweeks: List[int] = field(default_factory=list)
    stats: List[str] = field(default_factory=list)
    numbers: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
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
        return any([
            self.players, self.teams, self.positions,
            self.seasons, self.gameweeks, self.stats, self.numbers
        ])

class EntityExtractor:
    TEAMS = {
        "arsenal": "Arsenal", "aston villa": "Aston Villa", "villa": "Aston Villa",
        "bournemouth": "Bournemouth", "brentford": "Brentford", "brighton": "Brighton",
        "brighton & hove albion": "Brighton", "burnley": "Burnley", "chelsea": "Chelsea",
        "crystal palace": "Crystal Palace", "palace": "Crystal Palace", "everton": "Everton",
        "fulham": "Fulham", "leeds": "Leeds", "leeds united": "Leeds",
        "leicester": "Leicester", "leicester city": "Leicester", "liverpool": "Liverpool",
        "manchester city": "Man City", "man city": "Man City", "city": "Man City",
        "manchester united": "Man Utd", "man utd": "Man Utd", "man united": "Man Utd",
        "united": "Man Utd", "newcastle": "Newcastle", "newcastle united": "Newcastle",
        "nottingham forest": "Nott'm Forest", "forest": "Nott'm Forest", "norwich": "Norwich",
        "norwich city": "Norwich", "southampton": "Southampton", "spurs": "Spurs",
        "tottenham": "Spurs", "tottenham hotspur": "Spurs", "watford": "Watford",
        "west ham": "West Ham", "west ham united": "West Ham", "wolves": "Wolves",
        "wolverhampton": "Wolves", "wolverhampton wanderers": "Wolves",
    }
    
    POSITIONS = {
        "goalkeeper": "GK", "gk": "GK", "keeper": "GK", "goalie": "GK",
        "defender": "DEF", "def": "DEF", "defenders": "DEF", "defence": "DEF",
        "defense": "DEF", "cb": "DEF", "rb": "DEF", "lb": "DEF", "fullback": "DEF",
        "centre back": "DEF", "midfielder": "MID", "mid": "MID", "midfielders": "MID",
        "midfield": "MID", "cm": "MID", "cam": "MID", "cdm": "MID", "winger": "MID",
        "forward": "FWD", "fwd": "FWD", "forwards": "FWD", "striker": "FWD",
        "strikers": "FWD", "attacker": "FWD", "attackers": "FWD", "cf": "FWD",
        "st": "FWD",
    }
    
    STATS = {
        "goals": "goals_scored", "goal": "goals_scored", "scored": "goals_scored",
        "assists": "assists", "assist": "assists", "points": "total_points",
        "total points": "total_points", "fpl points": "total_points",
        "clean sheets": "clean_sheets", "clean sheet": "clean_sheets",
        "cleansheets": "clean_sheets", "cs": "clean_sheets", "bonus": "bonus",
        "bonus points": "bonus", "bps": "bps", "minutes": "minutes", "mins": "minutes",
        "ict": "ict_index", "ict index": "ict_index", "influence": "influence",
        "creativity": "creativity", "threat": "threat", "value": "value",
        "price": "value", "cost": "value", "form": "form", "saves": "saves",
        "save": "saves", "yellow cards": "yellow_cards", "yellows": "yellow_cards",
        "red cards": "red_cards", "reds": "red_cards", "cards": "yellow_cards",
        "transfers": "transfers_in", "transfers in": "transfers_in",
        "transfers out": "transfers_out", "selected": "selected",
        "ownership": "selected", "owned": "selected",
    }
    
    SEASON_PATTERNS = [
        r"20(\d{2})[-/](\d{2})", r"20(\d{2})", r"(\d{2})[-/](\d{2})\s+season",
        r"last season", r"this season", r"previous season",
    ]
    
    def __init__(self, known_players: Optional[Set[str]] = None):
        self.known_players = known_players or set()
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        self.matcher = Matcher(self.nlp.vocab)
        self._add_patterns()
    
    def _add_patterns(self):
        gameweek_patterns = [
            [{"LOWER": {"IN": ["gw", "gameweek"]}}, {"IS_DIGIT": True}],
            [{"LOWER": "game"}, {"LOWER": "week"}, {"IS_DIGIT": True}],
            [{"LOWER": "week"}, {"IS_DIGIT": True}]
        ]
        for pattern in gameweek_patterns:
            self.matcher.add("GAMEWEEK", [pattern])
        
        season_patterns = [
            [{"SHAPE": "dddd"}, {"TEXT": "-"}, {"SHAPE": "dd"}],
            [{"SHAPE": "dddd"}, {"TEXT": "/"}, {"SHAPE": "dd"}],
            [{"SHAPE": "dddd"}, {"TEXT": "-"}, {"SHAPE": "dddd"}],
            [{"SHAPE": "dddd"}, {"TEXT": ","}, {"SHAPE": "dddd"}],
            [{"LOWER": {"IN": ["last", "this", "previous"]}}, {"LOWER": "season"}]
        ]
        for pattern in season_patterns:
            self.matcher.add("SEASON", [pattern])
    
    def set_known_players(self, players: Set[str]):
        self.known_players = players
    
    def extract(self, query: str) -> ExtractedEntities:
        entities = ExtractedEntities()
        doc = self.nlp(query)
        
        entities.teams = self._extract_teams(doc)
        entities.positions = self._extract_positions(doc)
        entities.seasons = self._extract_seasons(doc)
        entities.gameweeks = self._extract_gameweeks(doc)
        entities.stats = self._extract_stats(doc)
        entities.numbers = self._extract_numbers(doc, entities.gameweeks)
        entities.players = self._extract_players(doc, entities)
        
        return entities
    
    def _extract_teams(self, doc: Doc) -> List[str]:
        teams = []
        text_lower = doc.text.lower()
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                normalized = self.TEAMS.get(ent.text.lower())
                if normalized and normalized not in teams:
                    teams.append(normalized)
        
        for team_key, team_name in self.TEAMS.items():
            if team_key in text_lower:
                if team_name not in teams:
                    teams.append(team_name)
        
        return teams
    
    def _extract_positions(self, doc: Doc) -> List[str]:
        positions = []
        
        for token in doc:
            normalized = self.POSITIONS.get(token.text.lower())
            if normalized and normalized not in positions:
                positions.append(normalized)
        
        text_lower = doc.text.lower()
        for pos_key, pos_name in self.POSITIONS.items():
            if " " in pos_key and pos_key in text_lower:
                if pos_name not in positions:
                    positions.append(pos_name)
        
        return positions
    
    def _extract_seasons(self, doc: Doc) -> List[str]:
        seasons = []
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "SEASON":
                span = doc[start:end]
                season_text = span.text
                
                if re.match(r"(20\d{2})[-/](20\d{2})", season_text):
                    years = re.findall(r"20\d{2}", season_text)
                    if len(years) == 2:
                        year1, year2 = int(years[0]), int(years[1])
                        if year2 == year1 + 1:
                            season = f"{year1}-{str(year2)[2:]}"
                            if season in ["2020-21", "2021-22", "2022-23"]:
                                seasons.append(season)
                elif re.match(r"20(\d{2})[-/](\d{2})", season_text):
                    match = re.match(r"20(\d{2})[-/](\d{2})", season_text)
                    season = f"20{match.group(1)}-{match.group(2)}"
                    if season in ["2020-21", "2021-22", "2022-23"]:
                        seasons.append(season)
        
        for ent in doc.ents:
            if ent.label_ == "DATE" and ent.text.isdigit():
                year = int(ent.text)
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
        gameweeks = []
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "GAMEWEEK":
                span = doc[start:end]
                for token in span:
                    if token.like_num:
                        gw = int(token.text)
                        if 1 <= gw <= 38 and gw not in gameweeks:
                            gameweeks.append(gw)
        
        return gameweeks
    
    def _extract_stats(self, doc: Doc) -> List[str]:
        stats = []
        text_lower = doc.text.lower()
        
        for stat_key, stat_name in self.STATS.items():
            if stat_key in text_lower:
                if stat_name not in stats:
                    stats.append(stat_name)
        
        return stats
    
    def _extract_numbers(self, doc: Doc, exclude_gws: List[int]) -> List[int]:
        numbers = []
        
        for token in doc:
            if token.like_num and token.text.isdigit():
                num = int(token.text)
                if num not in exclude_gws and num not in [2020, 2021, 2022, 2023]:
                    if num not in numbers:
                        numbers.append(num)
        
        for ent in doc.ents:
            if ent.label_ == "CARDINAL" and ent.text.isdigit():
                num = int(ent.text)
                if num not in exclude_gws and num not in [2020, 2021, 2022, 2023]:
                    if num not in numbers:
                        numbers.append(num)
        
        return numbers
    
    def _extract_players(self, doc: Doc, entities: ExtractedEntities) -> List[str]:
        players = []
        text_lower = doc.text.lower()
        
        if self.known_players:
            for player in self.known_players:
                if player.lower() in text_lower:
                    players.append(player)
        
        if not players:
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
                    if (ent.text not in entities.teams and 
                        ent.text not in non_names and
                        len(ent.text) > 2):
                        players.append(ent.text)
        
        return players[:2]
    
    def get_query_parameters(self, entities: ExtractedEntities) -> Dict:
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
        
        if entities.seasons:
            params["season"] = entities.seasons[0]
        
        if entities.gameweeks:
            params["gameweek"] = entities.gameweeks[0]
        
        if entities.numbers:
            if entities.numbers[0] <= 50:
                params["limit"] = entities.numbers[0]
        
        return params