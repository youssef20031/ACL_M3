"""
Intent Classification for FPL Query Processing
"""
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Intent(Enum):
    """Supported query intents."""
    # Player-related intents
    PLAYER_STATS = "player_stats"
    PLAYER_COMPARISON = "player_comparison"
    PLAYER_SEARCH = "player_search"
    
    # Rankings and leaders
    TOP_SCORERS = "top_scorers"
    TOP_ASSISTERS = "top_assisters"
    TOP_POINTS = "top_points"
    BEST_VALUE = "best_value"
    
    # Team-related intents
    TEAM_ANALYSIS = "team_analysis"
    HEAD_TO_HEAD = "head_to_head"
    FIXTURE_RESULTS = "fixture_results"
    
    # Performance metrics
    CLEAN_SHEETS = "clean_sheets"
    BONUS_POINTS = "bonus_points"
    ICT_INDEX = "ict_index"
    TRANSFERS = "transfers"
    MOST_SELECTED = "most_selected"
    
    # ML Predictions
    PREDICTION = "prediction"

    # Best XI: the predicted squad for the next gameweek, or a finished gameweek's dream team
    TEAM_OF_THE_WEEK = "team_of_the_week"

    # Trivia
    TRIVIA = "trivia"
    
    # Recommendations
    RECOMMENDATION = "recommendation"
    
    # General
    GENERAL_QUESTION = "general_question"
    SEASON_SUMMARY = "season_summary"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: Intent
    confidence: float
    matched_patterns: List[str]
    extracted_keywords: List[str]


class IntentClassifier:
    """
    Rule-based intent classifier for FPL queries.
    Uses keyword matching and pattern recognition.
    """
    
    def __init__(self):
        """Initialize intent classifier with patterns."""
        self.intent_patterns = self._build_patterns()
    
    def _build_patterns(self) -> Dict[Intent, List[str]]:
        """Build regex patterns for each intent."""
        return {
            # First, so it wins ties: "best team for next gameweek" also matches PREDICTION's
            # "next gameweek", and "what was the dream team" GENERAL_QUESTION
            Intent.TEAM_OF_THE_WEEK: [
                r"team of the (?:week|gameweek|gw)",
                r"\btotw\b",
                r"dream ?team",
                r"best (?:starting )?(?:xi|11|eleven|line-?up)\b",
                r"best (?:possible |fpl )?squad\b",
                r"best (?:possible |fpl )?team (?:for |this |in )?(?:the )?(?:next |this |coming )?(?:gw|gameweek|week)",
                r"(?:optimal|ideal|perfect) (?:fpl )?(?:team|squad|xi|line-?up)\b",
                r"\b(?:pick|build|choose|make|give)(?: me)? (?:a |an |my |the )?(?:best )?(?:fpl )?(?:team|squad|xi|line-?up)\b",
                r"wildcard (?:team|squad|draft)",
            ],

            Intent.PLAYER_STATS: [
                r"(?:stats?|statistics?|performance|how did|how was)\s+(?:for\s+)?(\w+(?:\s+\w+)?)",
                r"(\w+(?:\s+\w+)?)\s+(?:stats?|statistics?|performance)",
                r"tell me about\s+(\w+(?:\s+\w+)?)",
                r"show me\s+(\w+(?:\s+\w+)?)",
                r"what (?:are|were) (\w+(?:\s+\w+)?)'?s?\s+(?:stats?|numbers?)",
            ],
            
            Intent.PLAYER_COMPARISON: [
                r"compare\s+(\w+(?:\s+\w+)?)\s+(?:and|vs?\.?|with|to)\s+(\w+(?:\s+\w+)?)",
                r"(\w+(?:\s+\w+)?)\s+(?:vs?\.?|versus|or|against)\s+(\w+(?:\s+\w+)?)",
                r"who(?:'s| is) better[,:]?\s+(\w+(?:\s+\w+)?)\s+(?:or|vs?\.?)\s+(\w+(?:\s+\w+)?)",
                r"difference between\s+(\w+(?:\s+\w+)?)\s+and\s+(\w+(?:\s+\w+)?)",
            ],
            
            Intent.PLAYER_SEARCH: [
                r"(?:find|search|look(?:ing)? for|who is)\s+(?:player\s+)?(\w+(?:\s+\w+)?)",
                r"players? (?:named?|called)\s+(\w+(?:\s+\w+)?)",
            ],
            
            Intent.TOP_SCORERS: [
                r"top\s+(?:\d+\s+)?(?:goal\s*)?scorers?",
                r"most goals?",
                r"who scored (?:the )?most",
                r"leading (?:goal\s*)?scorers?",
                r"highest (?:goal\s*)?scorers?",
                r"goals? leaders?",
                r"top\s+(?:goal\s*)?scoring\s+(?:players?|gk|defenders?|midfielders?|forwards?)",
            ],
            
            Intent.TOP_ASSISTERS: [
                r"top\s+(?:\d+\s+)?assist(?:er)?s?",
                r"most assists?",
                r"who (?:had|got|provided) (?:the )?most assists?",
                r"leading assist",
                r"assist leaders?",
                r"playmakers?",
            ],
            
            Intent.TOP_POINTS: [
                r"top\s+(?:\d+\s+)?(?:fpl\s+)?(?:points?|scorers?)",
                r"most (?:fpl\s+)?points",
                r"highest (?:fpl\s+)?points",
                r"(?:fpl\s+)?points? leaders?",
                r"best (?:performing|fpl)?\s+players?",
                # "top midfielders", "top 5 midfielders", "top 10 performing defenders"
                r"top\s+(?:\d+\s+)?(?:performing\s+)?(?:gks?|goalkeepers?|keepers?|defs?|defenders?|mids?|midfielders?|fwds?|forwards?|strikers?)\b",
                r"top\s+\d+\s+(?:fpl\s+)?players?\b",
                # A count makes "best/leading N ..." a ranking (plain "best midfielders for
                # next gameweek" stays a prediction question)
                r"(?:best|leading|highest[\s-]scoring)\s+\d+\s+(?:gks?|goalkeepers?|keepers?|defs?|defenders?|mids?|midfielders?|fwds?|forwards?|strikers?|players?)\b",
                r"top\s+(?:scoring\s+)?players?\s+(?:by|per|for|in)\s*(?:each\s+)?position",
                r"best\s+players?\s+(?:by|per|for|in)\s*(?:each\s+)?position",
                r"top\s+players?\s+(?:all|each|every)\s+positions?",
                r"(?:get|show|list)\s+top\s+(?:scoring\s+)?players?",
                r"top\s+(?:scoring\s+)?(?:gk|defenders?|midfielders?|forwards?)\s+(?:by|per|for|in)\s*(?:points|fpl points)?",
            ],
            
            Intent.BEST_VALUE: [
                r"best value",
                r"value picks?",
                r"points? per (?:million|£|pound)",
                r"budget (?:picks?|options?|players?)",
                r"cheap(?:est)? (?:good\s+)?(?:picks?|options?|players?)",
                r"undervalued",
                r"differential",
            ],
            
            Intent.TEAM_ANALYSIS: [
                r"(?:team|club)\s+(?:stats?|analysis|performance)",
                r"how (?:did|is|was)\s+(arsenal|chelsea|liverpool|man\s*(?:city|utd|united)|spurs|tottenham|\w+)\s+(?:do|doing|perform)",
                r"(arsenal|chelsea|liverpool|man\s*(?:city|utd|united)|spurs|tottenham|\w+)\s+(?:team\s+)?(?:stats?|analysis)",
                r"players? (?:from|in|at)\s+(\w+)",
                r"(arsenal|chelsea|liverpool|man\s*(?:city|utd|united)|spurs|tottenham|\w+)(?:'s)?\s+(?:best|top|leading|highest\s+scoring)\s+players?",
                r"(?:best|top|leading)\s+players?\s+(?:for|in|from|at)\s+(arsenal|chelsea|liverpool|man\s*(?:city|utd|united)|spurs|tottenham|\w+)",
            ],
            
            Intent.HEAD_TO_HEAD: [
                r"head[\s-]*to[\s-]*head",
                r"h2h",
                r"(\w+)\s+(?:vs?\.?|versus|against)\s+(\w+)",
                r"(?:between|results?\s+(?:for|between))\s+(\w+)\s+(?:and|vs?\.?)\s+(\w+)",
            ],
            
            Intent.FIXTURE_RESULTS: [
                r"fixture(?:s)?\s+(?:results?|scores?)",
                r"match(?:es)?\s+(?:results?|scores?)",
                r"(?:game|match)\s+(?:between|results?)",
                r"score(?:s|line)?\s+(?:for|in)",
            ],
            
            Intent.CLEAN_SHEETS: [
                r"clean\s*sheets?",
                r"most clean\s*sheets?",
                r"best (?:defenders?|goalkeepers?|gk)",
                r"(?:defenders?|goalkeepers?|gk) (?:with\s+)?most (?:clean\s*sheets?|points)",
            ],
            
            Intent.BONUS_POINTS: [
                r"bonus\s*(?:points?)?",
                r"bps",
                r"most bonus",
                r"bonus (?:leaders?|magnets?)",
            ],
            
            Intent.ICT_INDEX: [
                r"ict\s*(?:index)?",
                r"influence|creativity|threat",
                r"highest ict",
                r"best ict",
            ],
            
            Intent.TRANSFERS: [
                r"transfer(?:s|red)?(?:\s+in|\s+out)?",
                r"most transfer(?:s|red)?",
                r"who (?:was|is|got)\s+transferred",
                r"popular transfers?",
            ],
            
            Intent.MOST_SELECTED: [
                r"most selected",
                r"most (?:owned|popular)",
                r"ownership",
                r"template",
                r"essential",
            ],

            Intent.PREDICTION: [
                r"predict(?:ions?)?",
                r"forecast(?:ed)?",
                r"who will score",
                r"who (?:should|could|might) (?:i|we) pick",
                r"next (?:gw|gameweek|week)",
                r"upcoming (?:gw|gameweek|week|match|fixture)",
                r"expected (?:points|performance|score)",
                r"who (?:should|is likely to|will|might) perform",
                r"who to (?:pick|choose|select|captain)",
                r"(?:good|best) (?:picks?|choice|option|captain) for (?:next|upcoming)",
                r"captain(?:cy)? (?:pick|choice|option)",
                # The captain is always chosen for the next deadline, so any captaincy question
                # is a prediction (it used to fall to RECOMMENDATION -> an all-time ranking)
                r"\bcaptain(?:cy|s|ing)?\b",
                r"who should (?:i|we) (?:captain|start|bench|play)\b",
                r"points? (?:for|in) (?:next|upcoming) (?:gw|gameweek)",
                r"(?:how many|what) points? (?:will|do you think|is|are|can|might) (?:he|she|they|him|her|\w+)?\s*(?:gonna|going to)? (?:earn|get|score|make)",
                r"(?:will|do you think|can|might) (?:he|she|they|\w+)\s*(?:get|score|earn|make) (?:any\s+)?points?",
            ],
            
            Intent.TRIVIA: [
                r"trivia",
                r"quiz",
                r"fun fact",
                r"did you know",
                r"interesting",
                r"random (?:fact|stat)",
                r"test (?:me|my knowledge)",
            ],
            
            Intent.RECOMMENDATION: [
                r"recommend",
                r"suggest",
                r"who should (?:i|we)\s+(?:pick|get|buy|transfer)",
                r"(?:good|best)\s+(?:picks?|options?|choices?)",
                r"who to (?:pick|get)",
            ],
            
            Intent.SEASON_SUMMARY: [
                r"season\s+(?:summary|overview|stats?)",
                r"overall\s+(?:stats?|summary)",
                r"(?:20\d{2}-\d{2})\s+(?:season|summary|overview)",
            ],
            
            Intent.GENERAL_QUESTION: [
                r"^(?:what|who|when|where|why|how|which|is|are|was|were|did|do|does)",
            ],
        }
    
    def classify(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> IntentResult:
        """
        Classify the intent of a user query.
        
        Args:
            query: User's input query
            chat_history: Optional conversation history for pronoun resolution
            
        Returns:
            IntentResult with classified intent and metadata
        """
        query_lower = query.lower().strip()
        
        # Enhanced prediction detection for pronoun-based queries
        # Check if query has prediction-related keywords + pronouns
        has_pronoun = bool(re.search(r'\b(he|she|they|him|her|his|hers|their)\b', query_lower))
        has_points_question = bool(re.search(r'\b(points?|earn|score|get|make)\b', query_lower))
        has_future_tense = bool(re.search(r'\b(will|gonna|going to|do you think|can|might)\b', query_lower))
        
        # If this looks like a prediction query with pronouns, check chat history for player context
        if has_pronoun and has_points_question and has_future_tense and chat_history:
            # Look for player names in recent conversation
            recent_players = self._extract_players_from_history(chat_history)
            if recent_players:
                # This is likely a prediction query about a previously mentioned player
                # Upgrade to PREDICTION intent
                return IntentResult(
                    intent=Intent.PREDICTION,
                    confidence=0.8,
                    matched_patterns=["pronoun_reference_with_prediction_context"],
                    extracted_keywords=recent_players
                )
        
        best_intent = Intent.UNKNOWN
        best_confidence = 0.0
        matched_patterns = []
        extracted_keywords = []
        
        for intent, patterns in self.intent_patterns.items():
            matches = 0
            intent_matched_patterns = []
            
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    matches += 1
                    intent_matched_patterns.append(pattern)
                    # Extract captured groups as keywords
                    if match.groups():
                        extracted_keywords.extend([g for g in match.groups() if g])
            
            if matches > 0:
                # Calculate confidence based on number of matches and pattern specificity
                confidence = min(matches * 0.3 + 0.5, 1.0)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_intent = intent
                    matched_patterns = intent_matched_patterns
        
        # Default to general question if nothing matched but it starts with question word
        if best_intent == Intent.UNKNOWN and re.match(r'^(?:what|who|when|where|why|how|which)', query_lower):
            best_intent = Intent.GENERAL_QUESTION
            best_confidence = 0.3
        
        return IntentResult(
            intent=best_intent,
            confidence=best_confidence,
            matched_patterns=matched_patterns,
            extracted_keywords=list(set(extracted_keywords))
        )
    
    def _extract_players_from_history(self, chat_history: List[Dict[str, str]]) -> List[str]:
        """
        Extract player names from recent chat history.
        
        Args:
            chat_history: Conversation history
            
        Returns:
            List of player names found in recent messages
        """
        players = []
        # Check last 3 messages for player names
        recent_messages = chat_history[-3:] if len(chat_history) >= 3 else chat_history
        
        # Common player name patterns (capitalized words that aren't common words)
        common_words = {
            'premier', 'league', 'season', 'gameweek', 'week', 'top', 'best', 'most',
            'points', 'goals', 'assists', 'player', 'team', 'stats', 'performance',
            'who', 'what', 'when', 'where', 'how', 'which', 'the', 'and', 'for',
            'with', 'from', 'this', 'that', 'erling', 'mohamed', 'harry', 'son'
        }
        
        for msg in recent_messages:
            content = msg.get('content', '')
            # Look for capitalized words (likely names)
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            for word in words:
                if word.lower() not in common_words and len(word) > 3:
                    players.append(word)
        
        return players[:1]  # Return most recent player mentioned
    
    def get_query_type_for_intent(self, intent: Intent) -> str:
        """
        Map intent to corresponding Cypher query method.
        
        Args:
            intent: Classified intent
            
        Returns:
            Name of the Cypher query method to use
        """
        intent_to_query = {
            Intent.PLAYER_STATS: "get_player_season_stats",
            Intent.PLAYER_COMPARISON: "compare_players",
            Intent.PLAYER_SEARCH: "search_players_by_name",
            Intent.TOP_SCORERS: "get_top_points_by_position",  # Reroute to generic top points/stats query
            Intent.TOP_ASSISTERS: "get_top_points_by_position", # Reroute to generic top points/stats query
            Intent.TOP_POINTS: "get_top_points_by_position",
            Intent.BEST_VALUE: "get_best_value_players",
            Intent.TEAM_ANALYSIS: "get_team_top_performers",
            Intent.HEAD_TO_HEAD: "get_head_to_head",
            Intent.FIXTURE_RESULTS: "get_fixture_results",
            Intent.CLEAN_SHEETS: "get_clean_sheet_leaders",
            Intent.BONUS_POINTS: "get_bonus_point_leaders",
            Intent.ICT_INDEX: "get_ict_index_leaders",
            Intent.TRANSFERS: "get_most_transferred_players",
            Intent.MOST_SELECTED: "get_most_selected_players",
            Intent.SEASON_SUMMARY: "get_season_summary",
            Intent.RECOMMENDATION: "get_best_value_players",  # Default recommendation strategy
            Intent.TRIVIA: "get_highest_single_gameweek_score", # Default trivia
            Intent.PREDICTION: None,  # ML predictions handled separately in api_main.py, not via Cypher
            Intent.TEAM_OF_THE_WEEK: None,  # services/season_service.get_team_of_the_week, in api_main.py
        }
        
        return intent_to_query.get(intent, "")
    
    def get_supported_intents(self) -> List[Dict[str, str]]:
        """Get list of supported intents with descriptions."""
        descriptions = {
            Intent.PLAYER_STATS: "Get detailed statistics for a specific player",
            Intent.PLAYER_COMPARISON: "Compare two players' performance",
            Intent.PLAYER_SEARCH: "Search for players by name",
            Intent.TOP_SCORERS: "Find top goal scorers",
            Intent.TOP_ASSISTERS: "Find top assist providers",
            Intent.TOP_POINTS: "Find players with most FPL points",
            Intent.BEST_VALUE: "Find best value players (points per million)",
            Intent.TEAM_ANALYSIS: "Analyze team performance",
            Intent.HEAD_TO_HEAD: "Compare head-to-head results between teams",
            Intent.FIXTURE_RESULTS: "Get fixture/match results",
            Intent.CLEAN_SHEETS: "Find players with most clean sheets",
            Intent.BONUS_POINTS: "Find players with most bonus points",
            Intent.ICT_INDEX: "Find players with highest ICT index",
            Intent.TRANSFERS: "Find most transferred players",
            Intent.MOST_SELECTED: "Find most selected players",
            Intent.PREDICTION: "Get ML model predictions for future gameweeks",
            Intent.TEAM_OF_THE_WEEK: "Pick the best XI: predicted for the next gameweek, or a past gameweek's dream team",
            Intent.TRIVIA: "Play FPL trivia",
            Intent.RECOMMENDATION: "Get player recommendations",
            Intent.SEASON_SUMMARY: "Get season overview and statistics",
        }
        
        return [
            {"intent": intent.value, "description": desc}
            for intent, desc in descriptions.items()
        ]