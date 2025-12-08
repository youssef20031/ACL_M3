"""
FPL FantasyTrivia Generator
Generates trivia questions from the Knowledge Graph
"""
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TriviaCategory(Enum):
    """Categories of trivia questions."""
    TOP_SCORERS = "top_scorers"
    PLAYER_STATS = "player_stats"
    TEAM_FACTS = "team_facts"
    RECORDS = "records"
    COMPARISONS = "comparisons"
    TRUE_FALSE = "true_false"
    MULTIPLE_CHOICE = "multiple_choice"


class Difficulty(Enum):
    """Difficulty levels for trivia."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TriviaQuestion:
    """Container for a trivia question."""
    question: str
    correct_answer: str
    options: List[str]  # For multiple choice
    category: TriviaCategory
    difficulty: Difficulty
    explanation: str
    source_query: str  # Cypher query used to generate
    metadata: Dict[str, Any]


class TriviaGenerator:
    """
    Generates FPL trivia questions from the Knowledge Graph.
    Uses Cypher queries to fetch real data for questions.
    """
    
    def __init__(self, graph_connection):
        """
        Initialize trivia generator.
        
        Args:
            graph_connection: Neo4j connection instance
        """
        self.conn = graph_connection
        self.question_templates = self._build_templates()
    
    def _get_random_season(self) -> str:
        """Get a random season from the database."""
        query = "MATCH (s:Season) RETURN s.id AS season ORDER BY s.id"
        try:
            results = self.conn.execute_query(query, {})
            if results:
                seasons = [r["season"] for r in results]
                return random.choice(seasons)
            return "2022-23"  # Fallback
        except:
            return "2022-23"  # Fallback
    
    def _build_templates(self) -> Dict[TriviaCategory, List[Dict]]:
        """Build question templates for each category."""
        return {
            TriviaCategory.TOP_SCORERS: [
                {
                    "difficulty": Difficulty.EASY,
                    "template": "Who was the top goal scorer in the {season} season?",
                    "query": """
                        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                        WITH p, SUM(r.goals_scored) AS goals
                        RETURN p.name AS answer, goals
                        ORDER BY goals DESC
                        LIMIT 5
                    """,
                    "answer_field": "answer",
                    "explanation_template": "{answer} scored {goals} goals in the {season} season."
                },
                {
                    "difficulty": Difficulty.MEDIUM,
                    "template": "Who scored the most goals among {position}s in {season}?",
                    "query": """
                        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: $position})
                        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                        WITH p, SUM(r.goals_scored) AS goals
                        RETURN p.name AS answer, goals
                        ORDER BY goals DESC
                        LIMIT 5
                    """,
                    "answer_field": "answer",
                    "explanation_template": "{answer} was the top scoring {position} with {goals} goals in {season}."
                },
            ],
            
            TriviaCategory.PLAYER_STATS: [
                {
                    "difficulty": Difficulty.EASY,
                    "template": "How many FPL points did {player} score in {season}?",
                    "query": """
                        MATCH (p:Player {name: $player})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                        RETURN SUM(r.total_points) AS answer
                    """,
                    "answer_field": "answer",
                    "explanation_template": "{player} accumulated {answer} FPL points throughout the {season} season.",
                    "numeric": True,
                    "tolerance": 5  # Accept answers within ±5
                },
                {
                    "difficulty": Difficulty.MEDIUM,
                    "template": "How many assists did {player} provide in {season}?",
                    "query": """
                        MATCH (p:Player {name: $player})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                        RETURN SUM(r.assists) AS answer
                    """,
                    "answer_field": "answer",
                    "explanation_template": "{player} provided {answer} assists in the {season} season.",
                    "numeric": True
                },
            ],
            
            TriviaCategory.RECORDS: [
                {
                    "difficulty": Difficulty.HARD,
                    "template": "What was the highest single gameweek score in {season}?",
                    "query": """
                        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                        RETURN p.name AS player, gw.number AS gameweek, r.total_points AS answer
                        ORDER BY r.total_points DESC
                        LIMIT 1
                    """,
                    "answer_field": "answer",
                    "explanation_template": "{player} scored {answer} points in GW{gameweek} - the highest single gameweek haul of {season}.",
                    "numeric": True
                },
                {
                    "difficulty": Difficulty.HARD,
                    "template": "Which goalkeeper made the most saves in {season}?",
                    "query": """
                        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: 'GK'})
                        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                        WITH p, SUM(r.saves) AS saves
                        RETURN p.name AS answer, saves
                        ORDER BY saves DESC
                        LIMIT 5
                    """,
                    "answer_field": "answer",
                    "explanation_template": "{answer} made {saves} saves throughout the {season} season."
                },
            ],
            
            TriviaCategory.TRUE_FALSE: [
                {
                    "difficulty": Difficulty.EASY,
                    "template": "True or False: {player} scored more than {threshold} goals in {season}.",
                    "query": """
                        MATCH (p:Player {name: $player})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                        RETURN SUM(r.goals_scored) AS goals
                    """,
                    "answer_field": "goals",
                    "comparison": "threshold",
                    "explanation_template": "{player} actually scored {goals} goals in {season}."
                },
                {
                    "difficulty": Difficulty.MEDIUM,
                    "template": "True or False: {player} had more assists than {player2} in {season}.",
                    "query": """
                        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                        WHERE p.name IN [$player, $player2]
                        WITH p.name AS name, SUM(r.assists) AS assists
                        RETURN name, assists
                        ORDER BY assists DESC
                    """,
                    "answer_field": "comparison",
                    "explanation_template": "The actual assist counts were compared to determine the answer."
                },
            ],
            
            TriviaCategory.MULTIPLE_CHOICE: [
                {
                    "difficulty": Difficulty.EASY,
                    "template": "Which position does {player} play?",
                    "query": """
                        MATCH (p:Player {name: $player})-[:PLAYS_POSITION]->(pos:Position)
                        RETURN pos.code AS answer, pos.name AS full_name
                    """,
                    "answer_field": "full_name",
                    "options": ["Goalkeeper", "Defender", "Midfielder", "Forward"],
                    "explanation_template": "{player} plays as a {full_name}."
                },
                {
                    "difficulty": Difficulty.MEDIUM,
                    "template": "Which player scored the most FPL points in {season}?",
                    "query": """
                        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                        WITH p, SUM(r.total_points) AS points
                        RETURN p.name AS answer, points
                        ORDER BY points DESC
                        LIMIT 4
                    """,
                    "answer_field": "answer",
                    "explanation_template": "{answer} topped the FPL charts with {points} points in {season}."
                },
            ],
            
            TriviaCategory.COMPARISONS: [
                {
                    "difficulty": Difficulty.MEDIUM,
                    "template": "Who had more FPL points in {season}: {player} or {player2}?",
                    "query": """
                        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                        WHERE p.name IN [$player, $player2]
                        WITH p.name AS name, SUM(r.total_points) AS points
                        RETURN name, points
                        ORDER BY points DESC
                    """,
                    "answer_field": "name",
                    "explanation_template": "The points comparison showed the winner clearly."
                },
            ],
            
            TriviaCategory.TEAM_FACTS: [
                {
                    "difficulty": Difficulty.MEDIUM,
                    "template": "What was the highest-scoring fixture in {season}?",
                    "query": """
                        MATCH (f:Fixture)-[:HOME_TEAM]->(ht:Team)
                        MATCH (f)-[:AWAY_TEAM]->(at:Team)
                        MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                        WITH f, ht, at, (f.home_score + f.away_score) AS total_goals
                        RETURN ht.name AS home, at.name AS away, f.home_score AS home_score, 
                               f.away_score AS away_score, total_goals
                        ORDER BY total_goals DESC
                        LIMIT 1
                    """,
                    "answer_field": "formatted",
                    "explanation_template": "{home} {home_score} - {away_score} {away} was the highest-scoring game with {total_goals} goals."
                },
            ],
        }
    
    def _get_random_players(self, season: str, count: int = 2, position: Optional[str] = None) -> List[str]:
        """Get random player names from the database."""
        if position:
            query = """
                MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: $position})
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                WITH p, SUM(r.total_points) AS points
                WHERE points > 50
                RETURN p.name AS name
                ORDER BY rand()
                LIMIT $count
            """
            params = {"season": season, "position": position, "count": count}
        else:
            query = """
                MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
                WITH p, SUM(r.total_points) AS points
                WHERE points > 100
                RETURN p.name AS name
                ORDER BY rand()
                LIMIT $count
            """
            params = {"season": season, "count": count}
        
        try:
            results = self.conn.execute_query(query, params)
            return [r["name"] for r in results]
        except Exception as e:
            logger.error(f"Error getting random players: {e}")
            return []
    
    def _get_top_players(self, season: str, limit: int = 10) -> List[Dict]:
        """Get top players for generating meaningful questions."""
        query = """
            MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
            MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
            WITH p, pos, SUM(r.total_points) AS points, SUM(r.goals_scored) AS goals, SUM(r.assists) AS assists
            WHERE points > 100
            RETURN p.name AS name, pos.code AS position, points, goals, assists
            ORDER BY points DESC
            LIMIT $limit
        """
        try:
            return self.conn.execute_query(query, {"season": season, "limit": limit})
        except Exception as e:
            logger.error(f"Error getting top players: {e}")
            return []
    
    def generate_question(
        self, 
        category: Optional[TriviaCategory] = None,
        difficulty: Optional[Difficulty] = None,
        season: str = None
    ) -> Optional[TriviaQuestion]:
        """
        Generate a trivia question.
        
        Args:
            category: Specific category (random if None)
            difficulty: Specific difficulty (random if None)
            season: Season to use for questions (auto-selected if None)
            
        Returns:
            Generated TriviaQuestion or None if generation fails
        """
        # Auto-select season if not provided
        if season is None:
            season = self._get_random_season()
        
        # Select category and difficulty
        if category is None:
            category = random.choice(list(TriviaCategory))
        
        templates = self.question_templates.get(category, [])
        if not templates:
            return None
        
        # Filter by difficulty if specified
        if difficulty:
            templates = [t for t in templates if t["difficulty"] == difficulty]
            if not templates:
                templates = self.question_templates[category]
        
        template = random.choice(templates)
        difficulty = template["difficulty"]
        
        # Get parameters for the question
        params = {"season": season}
        
        # Get player names if needed
        if "{player}" in template["template"]:
            players = self._get_random_players(season, count=2)
            if not players:
                return None
            params["player"] = players[0]
            if len(players) > 1:
                params["player2"] = players[1]
        
        # Get position if needed
        if "{position}" in template["template"]:
            params["position"] = random.choice(["GK", "DEF", "MID", "FWD"])
        
        # Add threshold for true/false
        if "threshold" in str(template.get("comparison", "")):
            params["threshold"] = random.randint(5, 20)
        
        # Execute query
        try:
            results = self.conn.execute_query(template["query"], params)
            if not results:
                return None
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None
        
        # Extract answer
        answer_field = template["answer_field"]
        
        if answer_field == "formatted":
            # Special formatting for complex answers
            r = results[0]
            correct_answer = f"{r.get('home', '')} vs {r.get('away', '')}"
        elif answer_field == "comparison":
            # True/False comparison
            if "threshold" in params:
                actual_value = results[0].get("goals", 0)
                correct_answer = "True" if actual_value > params["threshold"] else "False"
            else:
                # Player comparison
                if len(results) >= 2:
                    correct_answer = "True" if results[0]["name"] == params.get("player") else "False"
                else:
                    return None
        else:
            correct_answer = str(results[0].get(answer_field, "Unknown"))
        
        # Generate options for multiple choice
        if category == TriviaCategory.MULTIPLE_CHOICE:
            if "options" in template:
                options = template["options"].copy()
            else:
                # Use query results as options
                options = [str(r.get(answer_field, "")) for r in results[:4]]
            random.shuffle(options)
        elif category == TriviaCategory.TRUE_FALSE:
            options = ["True", "False"]
        else:
            # Generate plausible wrong answers
            if template.get("numeric"):
                try:
                    correct_num = int(correct_answer)
                    options = [
                        str(correct_num),
                        str(correct_num + random.randint(5, 20)),
                        str(max(0, correct_num - random.randint(5, 20))),
                        str(correct_num + random.randint(-10, 10))
                    ]
                except:
                    options = [correct_answer]
            else:
                # Use other results as options
                options = [str(r.get(answer_field, "")) for r in results[:4]]
            
            if correct_answer not in options:
                options[0] = correct_answer
            random.shuffle(options)
        
        # Format question
        question_text = template["template"].format(**params)
        
        # Format explanation
        explanation_params = {**params, **results[0]} if results else params
        explanation_params["answer"] = correct_answer
        explanation = template["explanation_template"].format(**explanation_params)
        
        return TriviaQuestion(
            question=question_text,
            correct_answer=correct_answer,
            options=options,
            category=category,
            difficulty=difficulty,
            explanation=explanation,
            source_query=template["query"],
            metadata={
                "season": season,
                "params": params,
                "raw_results": results[:3] if results else []
            }
        )
    
    def generate_quiz(
        self, 
        num_questions: int = 5,
        difficulty: Optional[Difficulty] = None,
        season: str = "2022-23"
    ) -> List[TriviaQuestion]:
        """
        Generate a quiz with multiple questions.
        
        Args:
            num_questions: Number of questions to generate
            difficulty: Difficulty level (mixed if None)
            season: Season to use
            
        Returns:
            List of TriviaQuestion objects
        """
        questions = []
        categories = list(TriviaCategory)
        attempts = 0
        max_attempts = num_questions * 3
        
        while len(questions) < num_questions and attempts < max_attempts:
            attempts += 1
            category = random.choice(categories)
            
            question = self.generate_question(
                category=category,
                difficulty=difficulty,
                season=season
            )
            
            if question:
                # Avoid duplicate questions
                existing_texts = [q.question for q in questions]
                if question.question not in existing_texts:
                    questions.append(question)
        
        return questions
    
    def check_answer(
        self, 
        question: TriviaQuestion, 
        user_answer: str
    ) -> Tuple[bool, str]:
        """
        Check if user's answer is correct.
        
        Args:
            question: The trivia question
            user_answer: User's answer
            
        Returns:
            Tuple of (is_correct, feedback_message)
        """
        correct = question.correct_answer.lower().strip()
        user = user_answer.lower().strip()
        
        # Check for numeric tolerance
        if question.metadata.get("numeric"):
            try:
                correct_num = float(correct)
                user_num = float(user)
                tolerance = question.metadata.get("tolerance", 0)
                is_correct = abs(correct_num - user_num) <= tolerance
            except:
                is_correct = correct == user
        else:
            is_correct = correct == user
        
        if is_correct:
            feedback = f"✅ Correct! {question.explanation}"
        else:
            feedback = f"❌ Wrong! The correct answer was: {question.correct_answer}. {question.explanation}"
        
        return is_correct, feedback
    
    def get_categories(self) -> List[Dict[str, str]]:
        """Get available trivia categories with descriptions."""
        descriptions = {
            TriviaCategory.TOP_SCORERS: "Questions about top goal scorers and point leaders",
            TriviaCategory.PLAYER_STATS: "Questions about individual player statistics",
            TriviaCategory.TEAM_FACTS: "Questions about team performances and fixtures",
            TriviaCategory.RECORDS: "Questions about FPL records and achievements",
            TriviaCategory.COMPARISONS: "Head-to-head player comparisons",
            TriviaCategory.TRUE_FALSE: "True or False statements about FPL",
            TriviaCategory.MULTIPLE_CHOICE: "Multiple choice questions",
        }
        
        return [
            {"category": cat.value, "description": desc}
            for cat, desc in descriptions.items()
        ]
