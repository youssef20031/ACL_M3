"""
LLM Model Evaluation Script for FPL Graph-RAG System
Evaluates all available LLM models with quantitative and qualitative metrics,
replicating the test cases from the website functionality.
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import numpy as np
import inspect
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, HUGGINGFACE_API_TOKEN
from graph.connection import Neo4jConnection
from graph.queries import CypherQueries
from llm.llm_manager import LLMManager, LLMResponse, PromptBuilder
from llm.prompts import PromptTemplates
from embeddings.embedding_manager import EmbeddingManager

load_dotenv()


@dataclass
class QualitativeScore:
    """Human evaluation scores for qualitative assessment."""
    quality: int  # 1-5: Overall response quality
    relevance: int  # 1-5: How relevant is the answer to the question
    naturalness: int  # 1-5: How natural/fluent is the language
    correctness: int  # 1-5: Factual accuracy based on KG data


@dataclass
class TestCase:
    """Test case definition."""
    id: str
    category: str
    question: str
    expected_keywords: List[str]  # Keywords that should appear in a good answer
    context_query: str  # Which query method to use for context
    context_params: Dict[str, Any]  # Parameters for the query
    expected_answer_hint: str  # Hint about what the answer should contain
    retrieval_mode: str = "cypher"  # "cypher", "embedding", or "hybrid"


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model-testcase pair."""
    model_name: str
    test_case_id: str
    question: str
    response_text: str
    response_time: float
    tokens_used: int
    success: bool
    error: Optional[str]
    qualitative_scores: Optional[QualitativeScore]
    keyword_match_score: float  # Percentage of expected keywords found
    

class LLMEvaluator:
    """Evaluates LLM models on FPL Q&A tasks."""
    
    def __init__(self, neo4j_conn: Neo4jConnection, llm_manager: LLMManager, embedding_manager: Optional[EmbeddingManager] = None, human_evaluation: bool = False, llm_judge: bool = False, judge_model: str = "qwen-2.5-72b", quantitative_only: bool = False, rate_limit_delay: float = 3.0):
        self.neo4j_conn = neo4j_conn
        self.llm_manager = llm_manager
        self.embedding_manager = embedding_manager
        self.human_evaluation = human_evaluation
        self.llm_judge = llm_judge
        self.judge_model = judge_model
        self.quantitative_only = quantitative_only
        self.rate_limit_delay = rate_limit_delay
        self.results: List[EvaluationResult] = []
        self.output_dir = os.path.join(os.path.dirname(__file__), "evaluation_results")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_human_score(self, prompt: str, min_val: int = 1, max_val: int = 5) -> int:
        """Get a score from a human evaluator with validation."""
        while True:
            try:
                score = input(f"  {prompt} ({min_val}-{max_val}): ").strip()
                if score.lower() == 'q':
                    raise KeyboardInterrupt("User quit evaluation")
                score = int(score)
                if min_val <= score <= max_val:
                    return score
                print(f"    ⚠ Please enter a number between {min_val} and {max_val}")
            except ValueError:
                print(f"    ⚠ Invalid input. Please enter a number between {min_val}-{max_val}, or 'q' to quit.")
    
    def human_score_qualitative(self, response: str, test_case: TestCase, success: bool) -> QualitativeScore:
        """
        Prompt a human evaluator to score the response quality.
        Displays the question, expected answer hint, and the model's response.
        """
        if not success or not response:
            print("  [Response failed or empty - assigning minimum scores]")
            return QualitativeScore(quality=1, relevance=1, naturalness=1, correctness=1)
        
        print("\n" + "="*70)
        print("🧑‍💼 HUMAN EVALUATION REQUIRED")
        print("="*70)
        print(f"\n📋 QUESTION: {test_case.question}")
        print(f"\n💡 EXPECTED ANSWER HINT: {test_case.expected_answer_hint}")
        print(f"\n🔑 EXPECTED KEYWORDS: {', '.join(test_case.expected_keywords)}")
        print(f"\n🤖 MODEL RESPONSE:")
        print("-"*50)
        print(response[:1000] if len(response) > 1000 else response)
        if len(response) > 1000:
            print(f"\n... [Response truncated, {len(response)} chars total]")
        print("-"*50)
        
        print("\n📊 Please rate the response on the following criteria (1-5 scale):")
        print("   1 = Very Poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent")
        print("   Enter 'q' at any time to quit evaluation.\n")
        
        quality = self.get_human_score("Quality (overall response quality)")
        relevance = self.get_human_score("Relevance (how relevant is the answer to the question)")
        naturalness = self.get_human_score("Naturalness (how natural/fluent is the language)")
        correctness = self.get_human_score("Correctness (factual accuracy based on expected answer)")
        
        print(f"\n✅ Recorded scores: Quality={quality}, Relevance={relevance}, Naturalness={naturalness}, Correctness={correctness}")
        
        return QualitativeScore(
            quality=quality,
            relevance=relevance,
            naturalness=naturalness,
            correctness=correctness
        )
    
    def llm_score_qualitative(self, response: str, test_case: TestCase, success: bool) -> QualitativeScore:
        """
        Use an LLM as a judge to evaluate the response quality.
        This implements the LLM-as-a-judge pattern for automated evaluation.
        """
        if not success or not response:
            return QualitativeScore(quality=1, relevance=1, naturalness=1, correctness=1)
        
        # Build the evaluation prompt
        eval_prompt = f"""You are an expert evaluator assessing the quality of an AI assistant's response to a question about Fantasy Premier League (FPL) data.

QUESTION: {test_case.question}

EXPECTED ANSWER HINT: {test_case.expected_answer_hint}

EXPECTED KEYWORDS: {', '.join(test_case.expected_keywords)}

MODEL RESPONSE TO EVALUATE:
{response[:1500]}

Please evaluate the response on the following criteria, scoring each from 1-5:
- Quality (1-5): Overall response quality - is it well-structured, complete, and professional?
- Relevance (1-5): How relevant is the answer to the question asked?
- Naturalness (1-5): How natural and fluent is the language?
- Correctness (1-5): Is it factually accurate based on what was expected?

IMPORTANT: You MUST respond with ONLY a JSON object in this exact format, nothing else:
{{"quality": <1-5>, "relevance": <1-5>, "naturalness": <1-5>, "correctness": <1-5>}}

Do not include any explanation, just the JSON object."""

        try:
            # Use the judge model to evaluate
            judge_response = self.llm_manager.generate(eval_prompt, model_key=self.judge_model)
            
            if judge_response.success and judge_response.text:
                # Parse the JSON response
                import re
                # Try to find JSON in the response
                json_match = re.search(r'\{[^}]+\}', judge_response.text)
                if json_match:
                    scores = json.loads(json_match.group())
                    return QualitativeScore(
                        quality=max(1, min(5, int(scores.get('quality', 3)))),
                        relevance=max(1, min(5, int(scores.get('relevance', 3)))),
                        naturalness=max(1, min(5, int(scores.get('naturalness', 3)))),
                        correctness=max(1, min(5, int(scores.get('correctness', 3))))
                    )
        except Exception as e:
            print(f"    ⚠ LLM judge error: {e}, falling back to auto-scoring")
        
        # Fallback to auto-scoring if LLM judge fails
        return self.auto_score_qualitative(response, test_case, success)
        
    def get_test_cases(self) -> List[TestCase]:
        """
        Define test cases based on numbered queries from notes/queries folder.
        Excludes Query_18 (All Players By Position) and Query_19 (All Teams) as they are internal.
        """
        return [
            # Query 1: Top Scorers
            TestCase(
                id="Q01",
                category="Player Stats",
                question="Who are the top goal scorers in 2022-23?",
                expected_keywords=["goals", "scorer", "points", "Haaland"],
                context_query="get_top_scorers_by_season",
                context_params={"season": "2022-23", "limit": 10},
                expected_answer_hint="Should mention Erling Haaland as top scorer with 36 goals"
            ),
            # Query 2: Top Assisters
            TestCase(
                id="Q02",
                category="Player Stats",
                question="Top assisters all time",
                expected_keywords=["assists", "provider", "De Bruyne"],
                context_query="get_top_assisters_by_season",
                context_params={"season": None, "limit": 10},
                expected_answer_hint="Should mention Kevin De Bruyne as top assist provider"
            ),
            # Query 3: Top Players by Position
            TestCase(
                id="Q03",
                category="Player Stats",
                question="Top defenders by points in 2022-23",
                expected_keywords=["defender", "DEF", "points", "Trippier"],
                context_query="get_top_points_by_position",
                context_params={"position": "DEF", "season": "2022-23", "limit": 10},
                expected_answer_hint="Should list top scoring defenders"
            ),
            # Query 4: Player Season Stats
            TestCase(
                id="Q04",
                category="Player Stats",
                question="Mohamed Salah 2022-23 stats",
                expected_keywords=["Salah", "points", "goals", "assists"],
                context_query="get_player_season_stats",
                context_params={"player_name": "Mohamed Salah", "season": "2022-23"},
                expected_answer_hint="Should show Salah's comprehensive stats for 2022-23"
            ),
            # Query 5: Player Gameweek Performance
            TestCase(
                id="Q05",
                category="Player Stats",
                question="How did Haaland perform in gameweek 10 of 2022-23?",
                expected_keywords=["Haaland", "gameweek", "points", "goals"],
                context_query="get_player_gameweek_performance",
                context_params={"player_name": "Erling Haaland", "season": "2022-23", "gameweek": 10},
                expected_answer_hint="Should show Haaland's performance in GW10"
            ),
            # Query 6: Team Top Performers
            TestCase(
                id="Q06",
                category="Team Analysis",
                question="Arsenal team analysis 2022-23",
                expected_keywords=["Arsenal", "points", "performer", "Saka"],
                context_query="get_team_top_performers",
                context_params={"team_name": "Arsenal", "season": "2022-23", "limit": 5},
                expected_answer_hint="Should list Arsenal's best FPL performers"
            ),
            # Query 7: Fixture Results
            TestCase(
                id="Q07",
                category="Team Analysis",
                question="Liverpool fixture results 2022-23",
                expected_keywords=["Liverpool", "fixtures", "score", "win", "loss"],
                context_query="get_fixture_results",
                context_params={"team_name": "Liverpool", "season": "2022-23"},
                expected_answer_hint="Should summarize Liverpool's match results"
            ),
            # Query 8: Head to Head (no embedding search - team query)
            TestCase(
                id="Q08",
                category="Team Analysis",
                question="Arsenal vs Spurs head to head",
                expected_keywords=["Arsenal", "Spurs", "score", "match"],
                context_query="get_head_to_head",
                context_params={"team1": "Arsenal", "team2": "Spurs"},
                expected_answer_hint="Should show historical results between Arsenal and Spurs"
            ),
            # Query 9: Best Value Players
            TestCase(
                id="Q09",
                category="Value Analysis",
                question="Best value midfielders in 2022-23",
                expected_keywords=["value", "midfielder", "MID", "points per million"],
                context_query="get_best_value_players",
                context_params={"season": "2022-23", "position": "MID", "limit": 10},
                expected_answer_hint="Should list midfielders with best points per million"
            ),
            # Query 10: Most Transferred Players
            TestCase(
                id="Q10",
                category="Transfers",
                question="Most transferred in players gameweek 5 2022-23",
                expected_keywords=["transfer", "gameweek", "players"],
                context_query="get_most_transferred_players",
                context_params={"season": "2022-23", "gameweek": 5, "direction": "in", "limit": 10},
                expected_answer_hint="Should list most transferred in players for GW5"
            ),
            # Query 11: Bonus Point Leaders
            TestCase(
                id="Q11",
                category="Performance",
                question="Most bonus points 2022-23",
                expected_keywords=["bonus", "points", "BPS"],
                context_query="get_bonus_point_leaders",
                context_params={"season": "2022-23", "limit": 10},
                expected_answer_hint="Should list top bonus point earners"
            ),
            # Query 12: Clean Sheet Leaders
            TestCase(
                id="Q12",
                category="Performance",
                question="Most clean sheets 2022-23",
                expected_keywords=["clean sheet", "goalkeeper", "defender"],
                context_query="get_clean_sheet_leaders",
                context_params={"season": "2022-23", "limit": 10},
                expected_answer_hint="Should list GKs/DEFs with most clean sheets"
            ),
            # Query 13: ICT Index Leaders
            TestCase(
                id="Q13",
                category="Performance",
                question="Highest ICT index 2022-23",
                expected_keywords=["ICT", "influence", "creativity", "threat"],
                context_query="get_ict_index_leaders",
                context_params={"season": "2022-23", "limit": 10},
                expected_answer_hint="Should list players with highest ICT index"
            ),
            # Query 14: Most Selected Players
            TestCase(
                id="Q14",
                category="Ownership",
                question="Most selected players gameweek 1 2022-23",
                expected_keywords=["selected", "ownership", "players"],
                context_query="get_most_selected_players",
                context_params={"season": "2022-23", "gameweek": 1, "limit": 10},
                expected_answer_hint="Should list most owned players in GW1"
            ),
            # Query 15: Compare Players
            TestCase(
                id="Q15",
                category="Comparison",
                question="Compare Mohamed Salah and Harry Kane",
                expected_keywords=["Salah", "Kane", "goals", "points", "compare"],
                context_query="compare_players",
                context_params={"player1": "Mohamed Salah", "player2": "Harry Kane"},
                expected_answer_hint="Should compare stats of both players"
            ),
            # Query 16: Player Form History
            TestCase(
                id="Q16",
                category="Player Stats",
                question="Show me Kevin De Bruyne's form history in 2022-23",
                expected_keywords=["De Bruyne", "form", "gameweek", "points"],
                context_query="get_player_form_history",
                context_params={"player_name": "Kevin De Bruyne", "season": "2022-23"},
                expected_answer_hint="Should show KDB's performance across gameweeks"
            ),
            # Query 17: Search Players
            TestCase(
                id="Q17",
                category="Search",
                question="Search for player Raya",
                expected_keywords=["Raya", "player", "position"],
                context_query="search_players_by_name",
                context_params={"name_pattern": "Raya", "limit": 10},
                expected_answer_hint="Should find David Raya and show his position"
            ),
            # Query 20: Season Summary
            TestCase(
                id="Q20",
                category="Summary",
                question="2022-23 season summary",
                expected_keywords=["season", "fixtures", "goals", "players"],
                context_query="get_season_summary",
                context_params={"season": "2022-23"},
                expected_answer_hint="Should summarize the 2022-23 season stats"
            ),
        ]
    
    def get_context_for_test_case(self, test_case: TestCase) -> tuple:
        """
        Execute the appropriate query and format context based on retrieval mode.
        Returns: (kg_context, embedding_context)
        """
        kg_context = ""
        embedding_context = ""
        cypher_results = []
        
        # ALWAYS Run Cypher first (for all modes) to get results/seed data
        try:
            # Get the method
            query_method = getattr(CypherQueries, test_case.context_query)
            
            # Filter parameters to only those accepted by the method
            sig = inspect.signature(query_method)
            valid_params = {k: v for k, v in test_case.context_params.items() if k in sig.parameters}
            
            # Execute query
            query, params = query_method(**valid_params)
            cypher_results = self.neo4j_conn.execute_query(query, params)
            
            # Only set kg_context if NOT in embedding-only mode
            if test_case.retrieval_mode != "embedding":
                kg_context = PromptBuilder.format_kg_context(cypher_results)
            
        except Exception as e:
            print(f"Error getting KG context for {test_case.id}: {e}")
            if test_case.retrieval_mode != "embedding":
                kg_context = f"Error retrieving KG context: {e}"
        
        # Get embedding context if needed
        if test_case.retrieval_mode in ["embedding", "hybrid"] and self.embedding_manager:
            try:
                player_name = test_case.context_params.get("player_name", "")
                season = test_case.context_params.get("season", "2022-23")
                top_k = test_case.context_params.get("top_k", 5)
                
                # Logic: Use Cypher results to find player to seed embeddings if possible
                similar_players = []
                
                # Check if we have players in cypher results
                found_cypher_player = False
                if cypher_results:
                    # Look for player_name in results
                    candidates = [r.get('player_name') for r in cypher_results if r.get('player_name')]
                    if candidates:
                        # Use the first player found in Cypher results
                        target_player = candidates[0]
                        similar_players = self.embedding_manager.find_similar_to_player(
                            target_player, season=season, top_k=top_k, exclude_self=False
                        )
                        found_cypher_player = True
                
                # Fallback to params or question if no cypher results
                if not found_cypher_player:
                    if player_name:
                        # Find similar players using param
                        similar_players = self.embedding_manager.find_similar_to_player(
                            player_name, season=season, top_k=top_k, exclude_self=False
                        )
                    else:
                        # Use query text for similarity search
                        similar_players = self.embedding_manager.find_similar_players(
                            test_case.question, top_k=top_k
                        )
                
                embedding_context = PromptBuilder.format_embedding_context(similar_players)
                
            except Exception as e:
                print(f"Error getting embedding context for {test_case.id}: {e}")
                embedding_context = f"Error retrieving embedding context: {e}"
        
        return kg_context, embedding_context
    
    def calculate_keyword_score(self, response: str, keywords: List[str]) -> float:
        """Calculate what percentage of expected keywords are in the response."""
        if not response or not keywords:
            return 0.0
        
        response_lower = response.lower()
        matches = sum(1 for kw in keywords if kw.lower() in response_lower)
        return matches / len(keywords)
    
    def auto_score_qualitative(self, response: str, test_case: TestCase, success: bool) -> QualitativeScore:
        """
        Automatically estimate qualitative scores based on response characteristics.
        This is a heuristic approach - real human evaluation would be more accurate.
        """
        if not success or not response:
            return QualitativeScore(quality=1, relevance=1, naturalness=1, correctness=1)
        
        # Quality: Based on response length and structure
        quality = 3
        if len(response) > 100:
            quality += 1
        if len(response) > 300:
            quality += 1
        if len(response) < 20:
            quality = 2
            
        # Relevance: Based on keyword matching
        keyword_score = self.calculate_keyword_score(response, test_case.expected_keywords)
        relevance = max(1, min(5, int(keyword_score * 5) + 1))
        
        # Naturalness: Based on sentence structure (simple heuristic)
        naturalness = 3
        sentences = response.split('.')
        if len(sentences) >= 2:
            naturalness += 1
        if any(word in response.lower() for word in ['however', 'additionally', 'therefore', 'moreover']):
            naturalness += 1
        if response.count('  ') > 3:  # Double spaces indicate formatting issues
            naturalness -= 1
            
        # Correctness: Based on having numbers/stats in the response
        correctness = 3
        if any(char.isdigit() for char in response):
            correctness += 1
        if any(kw.lower() in response.lower() for kw in test_case.expected_keywords[:2]):
            correctness += 1
            
        return QualitativeScore(
            quality=max(1, min(5, quality)),
            relevance=max(1, min(5, relevance)),
            naturalness=max(1, min(5, naturalness)),
            correctness=max(1, min(5, correctness))
        )
    
    def evaluate_model(self, model_key: str, test_cases: List[TestCase]) -> List[EvaluationResult]:
        """Evaluate a single model on all test cases."""
        results = []
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_key}")
        print(f"{'='*60}")
        
        for test_case in test_cases:
            mode_label = f"[{test_case.retrieval_mode.upper()}]"
            print(f"\nTest {test_case.id} {mode_label}: {test_case.question[:45]}...")
            
            # Get context based on retrieval mode
            kg_context, embedding_context = self.get_context_for_test_case(test_case)
            
            # Build prompt with appropriate context
            prompt = PromptTemplates.qa_template(
                question=test_case.question,
                kg_context=kg_context if kg_context else "No KG context available.",
                embedding_context=embedding_context if embedding_context else None,
                data_scope="2022-23 season" if "2022-23" in test_case.question else "all seasons"
            )
            
            # Generate response with retry and exponential backoff
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                response = self.llm_manager.generate(prompt, model_key=model_key, timeout=60)
                
                # Check for rate limit errors (402, 429)
                if response.error and ("402" in str(response.error) or "429" in str(response.error) or "rate" in str(response.error).lower()):
                    wait_time = self.rate_limit_delay * (2 ** attempt)  # Exponential backoff
                    print(f"    ⏳ Rate limited. Waiting {wait_time:.0f}s before retry ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    break  # Success or non-rate-limit error
            
            # Calculate scores
            keyword_score = self.calculate_keyword_score(response.text, test_case.expected_keywords)
            
            # Use human evaluation, LLM judge, automatic heuristics, or skip qualitative entirely
            if self.quantitative_only:
                qualitative = None  # Skip qualitative scoring
            elif self.human_evaluation:
                qualitative = self.human_score_qualitative(response.text, test_case, response.success)
            elif self.llm_judge:
                qualitative = self.llm_score_qualitative(response.text, test_case, response.success)
            else:
                qualitative = self.auto_score_qualitative(response.text, test_case, response.success)
            
            result = EvaluationResult(
                model_name=model_key,
                test_case_id=test_case.id,
                question=test_case.question,
                response_text=response.text[:500] if response.text else "",  # Truncate for storage
                response_time=response.response_time,
                tokens_used=response.tokens_used,
                success=response.success,
                error=response.error,
                qualitative_scores=qualitative,
                keyword_match_score=keyword_score
            )
            results.append(result)
            
            status = "✓" if response.success else "✗"
            print(f"  {status} Time: {response.response_time:.2f}s | Tokens: {response.tokens_used} | Keywords: {keyword_score*100:.0f}%")
            
            # Rate limiting delay between requests
            time.sleep(self.rate_limit_delay)
        
        return results
    
    def run_evaluation(self) -> Dict[str, List[EvaluationResult]]:
        """Run evaluation on all models and all test cases."""
        test_cases = self.get_test_cases()
        all_results = {}
        
        print("\n" + "="*60)
        print("FPL Graph-RAG LLM Model Evaluation")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Cases: {len(test_cases)}")
        print(f"Models: {list(self.llm_manager.MODELS.keys())}")
        print("="*60)
        
        for model_key in self.llm_manager.MODELS.keys():
            model_results = self.evaluate_model(model_key, test_cases)
            all_results[model_key] = model_results
            self.results.extend(model_results)
        
        return all_results
    
    def calculate_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Calculate aggregate metrics per model."""
        metrics = {}
        
        for model_key in self.llm_manager.MODELS.keys():
            model_results = [r for r in self.results if r.model_name == model_key]
            successful = [r for r in model_results if r.success]
            # Filter for results with qualitative scores (None in quantitative-only mode)
            with_qual_scores = [r for r in successful if r.qualitative_scores is not None]
            
            if not model_results:
                continue
            
            metrics[model_key] = {
                "display_name": self.llm_manager.MODELS[model_key]["display_name"],
                "total_tests": len(model_results),
                "successful_tests": len(successful),
                "success_rate": len(successful) / len(model_results) * 100,
                "avg_response_time": np.mean([r.response_time for r in successful]) if successful else 0,
                "total_tokens": sum(r.tokens_used for r in successful),
                "avg_tokens": np.mean([r.tokens_used for r in successful]) if successful else 0,
                "avg_keyword_score": np.mean([r.keyword_match_score for r in model_results]) * 100,
                "avg_quality": np.mean([r.qualitative_scores.quality for r in with_qual_scores]) if with_qual_scores else 0,
                "avg_relevance": np.mean([r.qualitative_scores.relevance for r in with_qual_scores]) if with_qual_scores else 0,
                "avg_naturalness": np.mean([r.qualitative_scores.naturalness for r in with_qual_scores]) if with_qual_scores else 0,
                "avg_correctness": np.mean([r.qualitative_scores.correctness for r in with_qual_scores]) if with_qual_scores else 0,
                # Estimated cost (rough approximation: $0.0001 per token for reference)
                "estimated_cost": sum(r.tokens_used for r in successful) * 0.0001,
            }
        
        return metrics
    
    def create_visualizations(self, metrics: Dict[str, Dict[str, Any]]):
        """Create matplotlib visualizations for the evaluation results."""
        models = list(metrics.keys())
        display_names = [metrics[m]["display_name"] for m in models]
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        if self.quantitative_only:
            # QUANTITATIVE-ONLY MODE: 2x3 grid with quantitative graphs only
            fig = plt.figure(figsize=(20, 12))
            
            # 1. Response Time Comparison
            ax1 = fig.add_subplot(2, 3, 1)
            response_times = [metrics[m]["avg_response_time"] for m in models]
            bars1 = ax1.bar(display_names, response_times, color=colors)
            ax1.set_ylabel('Average Response Time (seconds)')
            ax1.set_title('Response Time Comparison', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars1, response_times):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
            
            # 2. Token Usage Comparison
            ax2 = fig.add_subplot(2, 3, 2)
            token_usage = [metrics[m]["avg_tokens"] for m in models]
            bars2 = ax2.bar(display_names, token_usage, color=colors)
            ax2.set_ylabel('Average Tokens Used')
            ax2.set_title('Token Usage Comparison', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars2, token_usage):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)
            
            # 3. Success Rate Comparison
            ax3 = fig.add_subplot(2, 3, 3)
            success_rates = [metrics[m]["success_rate"] for m in models]
            bars3 = ax3.bar(display_names, success_rates, color=colors)
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
            ax3.set_ylim(0, 105)
            ax3.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars3, success_rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
            
            # 4. Keyword Match Score Comparison
            ax4 = fig.add_subplot(2, 3, 4)
            keyword_scores = [metrics[m]["avg_keyword_score"] for m in models]
            bars4 = ax4.bar(display_names, keyword_scores, color=colors)
            ax4.set_ylabel('Keyword Match Score (%)')
            ax4.set_title('Keyword Match Accuracy', fontsize=12, fontweight='bold')
            ax4.set_ylim(0, 105)
            ax4.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars4, keyword_scores):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
            
            # 5. Radar Chart - Quantitative Metrics Only
            ax5 = fig.add_subplot(2, 3, 5, projection='polar')
            categories = ['Response\nSpeed', 'Token\nEfficiency', 'Success\nRate', 'Keyword\nMatch']
            
            for i, m in enumerate(models):
                values = [
                    1 - min(metrics[m]["avg_response_time"] / 15, 1),  # Inverse
                    1 - min(metrics[m]["avg_tokens"] / 800, 1),  # Inverse
                    metrics[m]["success_rate"] / 100,
                    metrics[m]["avg_keyword_score"] / 100,
                ]
                
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]
                angles += angles[:1]
                
                ax5.plot(angles, values, 'o-', linewidth=2, label=display_names[i], color=colors[i])
                ax5.fill(angles, values, alpha=0.1, color=colors[i])
            
            ax5.set_xticks(np.linspace(0, 2 * np.pi, len(categories), endpoint=False))
            ax5.set_xticklabels(categories, fontsize=8)
            ax5.set_ylim(0, 1)
            ax5.set_title('Quantitative Metrics Radar', fontsize=12, fontweight='bold', pad=20)
            ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
            
            # 6. Summary Table (Quantitative Only)
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.axis('off')
            
            table_data = []
            headers = ['Model', 'Time(s)', 'Tokens', 'Success%', 'Keywords%', 'Cost($)']
            for m in models:
                table_data.append([
                    metrics[m]["display_name"],
                    f'{metrics[m]["avg_response_time"]:.2f}',
                    f'{metrics[m]["avg_tokens"]:.0f}',
                    f'{metrics[m]["success_rate"]:.0f}%',
                    f'{metrics[m]["avg_keyword_score"]:.0f}%',
                    f'{metrics[m]["estimated_cost"]:.4f}',
                ])
            
            table = ax6.table(cellText=table_data, colLabels=headers, 
                             cellLoc='center', loc='center',
                             colColours=['#2196F3']*len(headers))
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax6.set_title('Quantitative Summary', fontsize=12, fontweight='bold', pad=20)
            
            plt.suptitle('FPL Graph-RAG LLM Evaluation (Quantitative Metrics)', 
                        fontsize=16, fontweight='bold', y=1.02)
        else:
            # FULL MODE: Include qualitative graphs
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Response Time Comparison
            ax1 = fig.add_subplot(2, 3, 1)
            response_times = [metrics[m]["avg_response_time"] for m in models]
            bars1 = ax1.bar(display_names, response_times, color=colors)
            ax1.set_ylabel('Average Response Time (seconds)')
            ax1.set_title('Response Time Comparison', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars1, response_times):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
            
            # 2. Token Usage Comparison
            ax2 = fig.add_subplot(2, 3, 2)
            token_usage = [metrics[m]["avg_tokens"] for m in models]
            bars2 = ax2.bar(display_names, token_usage, color=colors)
            ax2.set_ylabel('Average Tokens Used')
            ax2.set_title('Token Usage Comparison', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars2, token_usage):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)
            
            # 3. Success Rate Comparison
            ax3 = fig.add_subplot(2, 3, 3)
            success_rates = [metrics[m]["success_rate"] for m in models]
            bars3 = ax3.bar(display_names, success_rates, color=colors)
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
            ax3.set_ylim(0, 105)
            ax3.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars3, success_rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
            
            # 4. Qualitative Scores Heatmap
            ax4 = fig.add_subplot(2, 3, 4)
            qual_categories = ['Quality', 'Relevance', 'Naturalness', 'Correctness']
            qual_data = np.array([
                [metrics[m]["avg_quality"], metrics[m]["avg_relevance"], 
                 metrics[m]["avg_naturalness"], metrics[m]["avg_correctness"]]
                for m in models
            ])
            im = ax4.imshow(qual_data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)
            ax4.set_xticks(np.arange(len(qual_categories)))
            ax4.set_yticks(np.arange(len(display_names)))
            ax4.set_xticklabels(qual_categories)
            ax4.set_yticklabels(display_names)
            ax4.set_title('Qualitative Scores (1-5)', fontsize=12, fontweight='bold')
            
            for i in range(len(display_names)):
                for j in range(len(qual_categories)):
                    text = ax4.text(j, i, f'{qual_data[i, j]:.1f}',
                                   ha="center", va="center", color="black", fontsize=10)
            
            plt.colorbar(im, ax=ax4, shrink=0.8)
            
            # 5. Radar Chart - Full Comparison
            ax5 = fig.add_subplot(2, 3, 5, projection='polar')
            categories = ['Response\nSpeed', 'Token\nEfficiency', 'Success\nRate', 
                         'Keyword\nMatch', 'Overall\nQuality']
            
            for i, m in enumerate(models):
                values = [
                    1 - min(metrics[m]["avg_response_time"] / 10, 1),
                    1 - min(metrics[m]["avg_tokens"] / 500, 1),
                    metrics[m]["success_rate"] / 100,
                    metrics[m]["avg_keyword_score"] / 100,
                    (metrics[m]["avg_quality"] + metrics[m]["avg_relevance"] + 
                     metrics[m]["avg_naturalness"] + metrics[m]["avg_correctness"]) / 20,
                ]
                
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]
                angles += angles[:1]
                
                ax5.plot(angles, values, 'o-', linewidth=2, label=display_names[i], color=colors[i])
                ax5.fill(angles, values, alpha=0.1, color=colors[i])
            
            ax5.set_xticks(np.linspace(0, 2 * np.pi, len(categories), endpoint=False))
            ax5.set_xticklabels(categories, fontsize=8)
            ax5.set_ylim(0, 1)
            ax5.set_title('Overall Model Comparison', fontsize=12, fontweight='bold', pad=20)
            ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
            
            # 6. Summary Table
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.axis('off')
            
            table_data = []
            headers = ['Model', 'Time(s)', 'Tokens', 'Success%', 'Quality', 'Cost($)']
            for m in models:
                table_data.append([
                    metrics[m]["display_name"],
                    f'{metrics[m]["avg_response_time"]:.2f}',
                    f'{metrics[m]["avg_tokens"]:.0f}',
                    f'{metrics[m]["success_rate"]:.0f}%',
                    f'{((metrics[m]["avg_quality"] + metrics[m]["avg_relevance"] + metrics[m]["avg_naturalness"] + metrics[m]["avg_correctness"]) / 4):.1f}',
                    f'{metrics[m]["estimated_cost"]:.4f}',
                ])
            
            table = ax6.table(cellText=table_data, colLabels=headers, 
                             cellLoc='center', loc='center',
                             colColours=['#4CAF50']*len(headers))
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax6.set_title('Summary Table', fontsize=12, fontweight='bold', pad=20)
            
            plt.suptitle('FPL Graph-RAG LLM Model Evaluation Results', 
                        fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'evaluation_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\n✓ Visualization saved to: {output_path}")
        
        # Create individual charts for better detail
        self._create_individual_charts(metrics, models, display_names, colors)
    
    def _create_individual_charts(self, metrics, models, display_names, colors):
        """Create individual detailed charts based on evaluation mode."""
        
        if self.quantitative_only:
            # QUANTITATIVE CHARTS
            
            # 1. Response Time per Test Case
            fig, ax = plt.subplots(figsize=(14, 8))
            test_cases = list(set(r.test_case_id for r in self.results))
            test_cases.sort()
            
            x = np.arange(len(test_cases))
            width = 0.15
            
            for i, m in enumerate(models):
                model_results = [r for r in self.results if r.model_name == m]
                response_times = []
                for tc in test_cases:
                    tc_result = next((r for r in model_results if r.test_case_id == tc), None)
                    response_times.append(tc_result.response_time if tc_result else 0)
                
                ax.bar(x + i * width, response_times, width, label=display_names[i], color=colors[i])
            
            ax.set_xlabel('Test Case')
            ax.set_ylabel('Response Time (seconds)')
            ax.set_title('Response Time per Test Case', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(test_cases, rotation=45)
            ax.legend()
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'response_time_per_testcase.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ Response time breakdown saved to: {output_path}")
            
            # 2. Token Usage per Test Case
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for i, m in enumerate(models):
                model_results = [r for r in self.results if r.model_name == m]
                token_counts = []
                for tc in test_cases:
                    tc_result = next((r for r in model_results if r.test_case_id == tc), None)
                    token_counts.append(tc_result.tokens_used if tc_result else 0)
                
                ax.bar(x + i * width, token_counts, width, label=display_names[i], color=colors[i])
            
            ax.set_xlabel('Test Case')
            ax.set_ylabel('Tokens Used')
            ax.set_title('Token Usage per Test Case', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(test_cases, rotation=45)
            ax.legend()
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'tokens_per_testcase.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ Token usage breakdown saved to: {output_path}")
            
            # 3. Keyword Match per Test Case
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for i, m in enumerate(models):
                model_results = [r for r in self.results if r.model_name == m]
                keyword_scores = []
                for tc in test_cases:
                    tc_result = next((r for r in model_results if r.test_case_id == tc), None)
                    keyword_scores.append(tc_result.keyword_match_score * 100 if tc_result else 0)
                
                ax.bar(x + i * width, keyword_scores, width, label=display_names[i], color=colors[i])
            
            ax.set_xlabel('Test Case')
            ax.set_ylabel('Keyword Match Score (%)')
            ax.set_title('Keyword Match per Test Case', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(test_cases, rotation=45)
            ax.legend()
            ax.set_ylim(0, 110)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'keyword_match_per_testcase.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ Keyword match breakdown saved to: {output_path}")
            
            # 4. Keyword Match by Category
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Group test cases by category
            test_cases_list = self.get_test_cases()
            categories = list(set(tc.category for tc in test_cases_list))
            categories.sort()
            
            category_scores = {m: [] for m in models}
            for cat in categories:
                cat_tests = [tc.id for tc in test_cases_list if tc.category == cat]
                for m in models:
                    model_results = [r for r in self.results if r.model_name == m and r.test_case_id in cat_tests]
                    if model_results:
                        avg_score = np.mean([r.keyword_match_score * 100 for r in model_results])
                    else:
                        avg_score = 0
                    category_scores[m].append(avg_score)
            
            x = np.arange(len(categories))
            for i, m in enumerate(models):
                ax.bar(x + i * width, category_scores[m], width, label=display_names[i], color=colors[i])
            
            ax.set_xlabel('Category')
            ax.set_ylabel('Average Keyword Match (%)')
            ax.set_title('Keyword Match by Query Category', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 110)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'keyword_by_category.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ Category performance saved to: {output_path}")
            
        else:
            # QUALITATIVE CHARTS (original)
            
            # Detailed Qualitative Comparison
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            qual_metrics = ['avg_quality', 'avg_relevance', 'avg_naturalness', 'avg_correctness']
            qual_titles = ['Quality Score', 'Relevance Score', 'Naturalness Score', 'Correctness Score']
            
            for ax, metric, title in zip(axes.flat, qual_metrics, qual_titles):
                values = [metrics[m][metric] for m in models]
                bars = ax.bar(display_names, values, color=colors)
                ax.set_ylabel('Score (1-5)')
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_ylim(0, 5.5)
                ax.tick_params(axis='x', rotation=45)
                ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Neutral (3)')
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           f'{val:.2f}', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle('Qualitative Metrics Breakdown', fontsize=14, fontweight='bold')
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'qualitative_breakdown.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ Qualitative breakdown saved to: {output_path}")
            
            # Per-Test-Case Performance
            fig, ax = plt.subplots(figsize=(14, 8))
            test_cases = list(set(r.test_case_id for r in self.results))
            test_cases.sort()
            
            x = np.arange(len(test_cases))
            width = 0.15
            
            for i, m in enumerate(models):
                model_results = [r for r in self.results if r.model_name == m]
                keyword_scores = []
                for tc in test_cases:
                    tc_result = next((r for r in model_results if r.test_case_id == tc), None)
                    keyword_scores.append(tc_result.keyword_match_score * 100 if tc_result else 0)
                
                ax.bar(x + i * width, keyword_scores, width, label=display_names[i], color=colors[i])
            
            ax.set_xlabel('Test Case')
            ax.set_ylabel('Keyword Match Score (%)')
            ax.set_title('Per-Test-Case Keyword Match Performance', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(test_cases, rotation=45)
            ax.legend()
            ax.set_ylim(0, 110)
            
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'per_testcase_performance.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ Per-test-case performance saved to: {output_path}")
    
    def save_results(self, metrics: Dict[str, Dict[str, Any]]):
        """Save evaluation results to JSON."""
        output = {
            "evaluation_date": datetime.now().isoformat(),
            "num_test_cases": len(self.get_test_cases()),
            "models_evaluated": list(metrics.keys()),
            "metrics": metrics,
            "detailed_results": [
                {
                    "model_name": r.model_name,
                    "test_case_id": r.test_case_id,
                    "question": r.question,
                    "response_text": r.response_text,
                    "response_time": r.response_time,
                    "tokens_used": r.tokens_used,
                    "success": r.success,
                    "error": r.error,
                    "keyword_match_score": r.keyword_match_score,
                    "qualitative_scores": asdict(r.qualitative_scores) if r.qualitative_scores else None,
                }
                for r in self.results
            ]
        }
        
        output_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved to: {output_path}")
    
    def print_summary(self, metrics: Dict[str, Dict[str, Any]]):
        """Print a summary of the evaluation results."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        # Find best model for each category
        best_speed = min(metrics.items(), key=lambda x: x[1]["avg_response_time"] if x[1]["success_rate"] > 0 else float('inf'))
        best_success = max(metrics.items(), key=lambda x: x[1]["success_rate"])
        best_quality = max(metrics.items(), key=lambda x: (x[1]["avg_quality"] + x[1]["avg_relevance"] + x[1]["avg_naturalness"] + x[1]["avg_correctness"]) / 4)
        best_efficiency = min(metrics.items(), key=lambda x: x[1]["avg_tokens"] if x[1]["success_rate"] > 0 else float('inf'))
        
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"   Fastest Response Time: {best_speed[1]['display_name']} ({best_speed[1]['avg_response_time']:.2f}s)")
        print(f"   Highest Success Rate:  {best_success[1]['display_name']} ({best_success[1]['success_rate']:.0f}%)")
        print(f"   Best Quality Scores:   {best_quality[1]['display_name']}")
        print(f"   Most Token Efficient:  {best_efficiency[1]['display_name']} ({best_efficiency[1]['avg_tokens']:.0f} tokens)")
        
        print(f"\n📊 DETAILED METRICS PER MODEL:")
        print("-"*80)
        print(f"{'Model':<20} {'Time(s)':<10} {'Tokens':<10} {'Success':<10} {'Qual.':<10} {'Cost($)':<10}")
        print("-"*80)
        
        for model, m in metrics.items():
            avg_qual = (m["avg_quality"] + m["avg_relevance"] + m["avg_naturalness"] + m["avg_correctness"]) / 4
            print(f"{m['display_name']:<20} {m['avg_response_time']:<10.2f} {m['avg_tokens']:<10.0f} {m['success_rate']:<10.0f} {avg_qual:<10.2f} {m['estimated_cost']:<10.4f}")
        
        print("-"*80)
        
        # Overall recommendation
        print(f"\n💡 RECOMMENDATION:")
        overall_scores = {}
        for model, m in metrics.items():
            # Calculate overall score (weighted)
            if m["success_rate"] == 0:
                overall_scores[model] = 0
            else:
                overall_scores[model] = (
                    (1 - min(m["avg_response_time"] / 10, 1)) * 0.2 +  # Speed (20%)
                    (m["success_rate"] / 100) * 0.3 +  # Reliability (30%)
                    ((m["avg_quality"] + m["avg_relevance"] + m["avg_naturalness"] + m["avg_correctness"]) / 20) * 0.4 +  # Quality (40%)
                    (1 - min(m["estimated_cost"] / 0.01, 1)) * 0.1  # Cost (10%)
                )
        
        best_overall = max(overall_scores.items(), key=lambda x: x[1])
        print(f"   Based on weighted scoring (Speed 20%, Reliability 30%, Quality 40%, Cost 10%):")
        print(f"   → Best Overall Model: {metrics[best_overall[0]]['display_name']} (Score: {best_overall[1]:.2f})")
        
        print("\n" + "="*80)


def main():
    """Main function to run the evaluation."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='FPL Graph-RAG LLM Model Evaluation')
    parser.add_argument('--human', action='store_true', 
                        help='Enable human evaluation mode for qualitative scoring')
    parser.add_argument('--llm-judge', action='store_true',
                        help='Use an LLM as a judge to automatically rate responses')
    parser.add_argument('--judge-model', type=str, default='qwen-2.5-72b',
                        help='Model to use as judge (default: qwen-2.5-72b)')
    parser.add_argument('--quantitative-only', action='store_true',
                        help='Run only quantitative tests (skip qualitative scoring)')
    parser.add_argument('--rate-limit', type=float, default=3.0,
                        help='Delay in seconds between API calls (default: 3.0)')
    args = parser.parse_args()
    
    human_mode = args.human
    llm_judge_mode = args.llm_judge
    judge_model = args.judge_model
    quantitative_mode = args.quantitative_only
    rate_limit = args.rate_limit
    
    # Validate mutually exclusive options
    if sum([human_mode, llm_judge_mode, quantitative_mode]) > 1:
        print("❌ Error: --human, --llm-judge, and --quantitative-only cannot be used together.")
        return
    
    print("Initializing FPL Graph-RAG LLM Evaluation...")
    print(f"Rate limit: {rate_limit}s between requests (with exponential backoff on errors)")
    
    if quantitative_mode:
        print("="*60)
        print("📊 QUANTITATIVE-ONLY MODE ENABLED")
        print("="*60)
        print("Testing: Response Time, Token Usage, Keyword Matching, Success Rate")
        print("Skipping: Qualitative evaluation (Quality, Relevance, Naturalness, Correctness)")
        print("="*60)
    elif human_mode:
        print("="*60)
        print("🧑‍💼 HUMAN EVALUATION MODE ENABLED")
        print("="*60)
        print("You will be prompted to rate each model response on:")
        print("  - Quality (1-5): Overall response quality")
        print("  - Relevance (1-5): How relevant is the answer to the question")
        print("  - Naturalness (1-5): How natural/fluent is the language")
        print("  - Correctness (1-5): Factual accuracy based on expected answer")
        print("\nPress Enter to continue or Ctrl+C to cancel...")
        input()
    elif llm_judge_mode:
        print("="*60)
        print("🤖 LLM-AS-JUDGE MODE ENABLED")
        print("="*60)
        print(f"Using model '{judge_model}' to automatically evaluate responses.")
        print("The judge will rate each response on Quality, Relevance, Naturalness, and Correctness.")
        print("="*60)
    
    # Initialize connections
    try:
        neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        if not neo4j_conn.test_connection():
            print("❌ Failed to connect to Neo4j. Please ensure the database is running.")
            return
        print("✓ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Neo4j connection error: {e}")
        return
    
    # Initialize LLM Manager
    if not HUGGINGFACE_API_TOKEN:
        print("❌ HUGGINGFACE_API_TOKEN not set. Please set it in your .env file.")
        return
    
    llm_manager = LLMManager(api_token=HUGGINGFACE_API_TOKEN)
    print(f"✓ LLM Manager initialized with {len(llm_manager.MODELS)} models")
    
    # Initialize Embedding Manager
    print("\nInitializing Embedding Manager...")
    embedding_manager = None
    try:
        embedding_manager = EmbeddingManager(model_key="minilm")
        print(f"✓ Embedding model loaded: {embedding_manager.model_info['name']}")
        
        # Build embeddings from Neo4j data
        print("Building player embeddings from Neo4j data...")
        query, query_params = CypherQueries.get_player_embeddings_data()
        results = neo4j_conn.execute_query(query, query_params)
        
        if results:
            embedding_manager.build_player_embeddings(results)
            print(f"✓ Built {len(embedding_manager.player_embeddings)} player embeddings")
        else:
            print("⚠️ No player data found for embeddings. Embedding tests will be skipped.")
            embedding_manager = None
    except Exception as e:
        print(f"⚠️ Embedding initialization failed: {e}")
        print("   Embedding-based tests will be skipped.")
        embedding_manager = None
    
    # Run evaluation with the appropriate evaluation mode
    evaluator = LLMEvaluator(
        neo4j_conn, 
        llm_manager, 
        embedding_manager, 
        human_evaluation=human_mode,
        llm_judge=llm_judge_mode,
        judge_model=judge_model,
        quantitative_only=quantitative_mode,
        rate_limit_delay=rate_limit
    )
    
    try:
        # Run all evaluations
        evaluator.run_evaluation()
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics()
        
        # Create visualizations
        print("\nGenerating visualizations...")
        evaluator.create_visualizations(metrics)
        
        # Save results
        evaluator.save_results(metrics)
        
        # Print summary
        evaluator.print_summary(metrics)
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        neo4j_conn.close()
        print("\n✓ Neo4j connection closed")


if __name__ == "__main__":
    main()

