
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class Intent(Enum):
    """Supported query intents."""
    PLAYER_STATS = "player_stats"
    PLAYER_COMPARISON = "player_comparison"
    PLAYER_SEARCH = "player_search"
    TOP_SCORERS = "top_scorers"
    TOP_ASSISTERS = "top_assisters"
    TOP_POINTS = "top_points"
    BEST_VALUE = "best_value"
    TEAM_ANALYSIS = "team_analysis"
    HEAD_TO_HEAD = "head_to_head"
    FIXTURE_RESULTS = "fixture_results"
    CLEAN_SHEETS = "clean_sheets"
    BONUS_POINTS = "bonus_points"
    ICT_INDEX = "ict_index"
    TRANSFERS = "transfers"
    MOST_SELECTED = "most_selected"
    TRIVIA = "trivia"
    RECOMMENDATION = "recommendation"
    GENERAL_QUESTION = "general_question"
    SEASON_SUMMARY = "season_summary"
    UNKNOWN = "unknown"

@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    matched_patterns: List[str]
    extracted_keywords: List[str]

class IntentClassifier:
    def __init__(self):
        self.intent_patterns = self._build_patterns()
    
    def _build_patterns(self) -> Dict[Intent, List[str]]:
        return {
            Intent.TOP_SCORERS: [
                r"top\s+(?:\d+\s+)?(?:goal\s*)?scorers?",
                r"most goals?",
                r"who scored (?:the )?most",
                r"leading (?:goal\s*)?scorers?",
                r"highest (?:goal\s*)?scorers?",
                r"goals? leaders?",
                r"top\s+(?:goal\s*)?scoring\s+(?:players?|gk|defenders?|midfielders?|forwards?)",
            ],
            Intent.TOP_POINTS: [
                r"top\s+(?:\d+\s+)?(?:fpl\s+)?(?:points?|scorers?)",
                r"most (?:fpl\s+)?points",
                r"highest (?:fpl\s+)?points",
                r"(?:fpl\s+)?points? leaders?",
                r"best (?:performing|fpl)?\s+players?",
                r"top (?:performing\s+)?(?:gk|goalkeeper|def|defender|mid|midfielder|fwd|forward|striker)s?",
                r"top\s+(?:scoring\s+)?players?\s+(?:by|per|for|in)\s*(?:each\s+)?position",
                r"best\s+players?\s+(?:by|per|for|in)\s*(?:each\s+)?position",
                r"top\s+players?\s+(?:all|each|every)\s+positions?",
                r"(?:get|show|list)\s+top\s+(?:scoring\s+)?players?",
                r"top\s+(?:scoring\s+)?(?:gk|defenders?|midfielders?|forwards?)\s+(?:by|per|for|in)\s*(?:points|fpl points)?",
            ],
            # Minimal other intents for context
             Intent.PLAYER_STATS: [
                r"(?:stats?|statistics?|performance|how did|how was)\s+(?:for\s+)?(\w+(?:\s+\w+)?)",
                r"(\w+(?:\s+\w+)?)\s+(?:stats?|statistics?|performance)",
            ],
        }
    
    def classify(self, query: str) -> IntentResult:
        query_lower = query.lower().strip()
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
            
            if matches > 0:
                confidence = min(matches * 0.3 + 0.5, 1.0)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_intent = intent
                    matched_patterns = intent_matched_patterns
        
        return IntentResult(best_intent, best_confidence, matched_patterns, extracted_keywords)

if __name__ == "__main__":
    classifier = IntentClassifier()
    # Test cases
    queries = [
        "Get the top goal scoring defenders in 2022-2023",
        "Get top scoring players by position in a season",
        "top goal scorers",
        "who scored the most goals"
    ]
    
    for q in queries:
        res = classifier.classify(q)
        print(f"Query: '{q}' -> {res.intent} (Conf: {res.confidence})")
