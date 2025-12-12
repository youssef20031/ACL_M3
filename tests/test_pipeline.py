import unittest
import sys
import os
import inspect

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.intent_classifier import IntentClassifier, Intent
from preprocessing.entity_extractor import EntityExtractor
from graph.queries import CypherQueries

class TestAppPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = IntentClassifier()
        # Initialize with dummy known player to ensure extraction works for generic names if needed
        cls.extractor = EntityExtractor(known_players={"Mohamed Salah", "Harry Kane", "Haaland", "Saka", "Raya"})

    def verify_pipeline(self, prompt, expected_intent, expected_method, expected_params_subset=None, expected_cypher_strings=None):
        """
        Generic verification method for the pipeline.
        ...
        expected_cypher_strings: List of strings that MUST be present in the generated query
        """
        print(f"\nTesting Prompt: '{prompt}'")
        
        # 1. Intent Classification
        intent_result = self.classifier.classify(prompt)
        self.assertEqual(intent_result.intent, expected_intent, 
                         f"Intent mismatch. Got {intent_result.intent}, expected {expected_intent}")
        print(f"  [Pass] Intent: {intent_result.intent}")

        # 2. Entity Extraction
        entities = self.extractor.extract(prompt)
        params = self.extractor.get_query_parameters(entities)
        
        if expected_params_subset:
            for k, v in expected_params_subset.items():
                self.assertIn(k, params, f"Missing expected param: {k}")
                self.assertEqual(str(params[k]), str(v), f"Param '{k}' mismatch. Got {params[k]}, expected {v}")
        print(f"  [Pass] Params: {params}")

        # 3. Method Resolution
        mapped_method_name = self.classifier.get_query_type_for_intent(intent_result.intent)
        self.assertEqual(mapped_method_name, expected_method,
                         f"Method mismatch. Got {mapped_method_name}, expected {expected_method}")
        print(f"  [Pass] Method: {mapped_method_name}")

        # 4. Query Generation
        if hasattr(CypherQueries, mapped_method_name):
            # Special fallback logic mimicking app.py
            if mapped_method_name == "get_player_season_stats" and "season" not in params:
                 mapped_method_name = "get_player_all_seasons_stats"
                 print(f"  [Info] Switching to fallback method: {mapped_method_name}")
            
            # Special mapping for search
            if mapped_method_name == "search_players_by_name" and "name_pattern" not in params:
                 if "player_name" in params:
                     params["name_pattern"] = params["player_name"]

            method = getattr(CypherQueries, mapped_method_name)
            
            # Filter params to only those accepted by the function
            sig = inspect.signature(method)
            valid_params = {k: v for k, v in params.items() if k in sig.parameters}
            
            try:
                query, q_params = method(**valid_params)
                self.assertIsInstance(query, str)
                self.assertTrue(len(query) > 0)
                
                # Check for expected substrings
                if expected_cypher_strings:
                    for s in expected_cypher_strings:
                        self.assertIn(s, query, f"Expected substring '{s}' not found in query.")
                
                print(f"  [Pass] Query Generated Successfully and Verified")
            except Exception as e:
                self.fail(f"Query generation failed: {str(e)}")
        else:
            self.fail(f"Method {mapped_method_name} not found in CypherQueries")
            
        return params

    # ===========================================
    # TEST CASES
    # ===========================================

    def test_query_1_top_scorers_rerouted(self):
        # "Top scorers" should now route to get_top_points_by_position with sort_by='goals'
        self.verify_pipeline(
            prompt="Who are the top goal scorers in 2022-23?",
            expected_intent=Intent.TOP_SCORERS,
            expected_method="get_top_points_by_position",
            expected_params_subset={"season": "2022-23", "sort_by": "goals"},
            expected_cypher_strings=["ORDER BY goals DESC", "season"]
        )

    def test_query_2_top_assisters_rerouted(self):
        # "Top assisters" should route to get_top_points_by_position with sort_by='assists'
        self.verify_pipeline(
            prompt="most assists in 2021-22",
            expected_intent=Intent.TOP_ASSISTERS,
            expected_method="get_top_points_by_position",
            expected_params_subset={"season": "2021-22", "sort_by": "assists"},
            expected_cypher_strings=["ORDER BY assists DESC"]
        )

    def test_query_3_top_points_position(self):
        self.verify_pipeline(
            prompt="Top defenders by points in 2022-23",
            expected_intent=Intent.TOP_POINTS,
            expected_method="get_top_points_by_position",
            expected_params_subset={"season": "2022-23", "position": "DEF"},
            expected_cypher_strings=["pos:Position {code: $position}", "ORDER BY total_points DESC"]
        )
    
    def test_query_3_dynamic_sort_position(self):
        # The specific bug fix case
        self.verify_pipeline(
            prompt="get top goal scoring defenders",
            expected_intent=Intent.TOP_SCORERS, 
            expected_method="get_top_points_by_position",
            expected_params_subset={"position": "DEF", "sort_by": "goals"},
            expected_cypher_strings=["pos:Position {code: $position}", "ORDER BY goals DESC"]
        )

    def test_query_4_player_stats(self):
        self.verify_pipeline(
            prompt="Mohamed Salah 2022-23 stats",
            expected_intent=Intent.PLAYER_STATS,
            expected_method="get_player_season_stats",
            expected_params_subset={"player_name": "Mohamed Salah", "season": "2022-23"},
            expected_cypher_strings=["p:Player {name: $player_name}"]
        )

    def test_query_4b_player_all_seasons(self):
        self.verify_pipeline(
            prompt="how did Mohamed Salah perform",
            expected_intent=Intent.PLAYER_STATS,
            expected_method="get_player_season_stats", # Logic checks fallback to all_seasons
            expected_params_subset={"player_name": "Mohamed Salah"},
            expected_cypher_strings=["MATCH (p:Player {name: $player_name})", "ORDER BY s.id"]
        )

    def test_query_5_gameweek_perf(self):
        pass

    def test_query_6_team_performers(self):
        self.verify_pipeline(
            prompt="Arsenal team analysis 2022-23",
            expected_intent=Intent.TEAM_ANALYSIS,
            expected_method="get_team_top_performers",
            expected_params_subset={"team_name": "Arsenal", "season": "2022-23"},
            expected_cypher_strings=["MATCH (t:Team {name: $team_name})", "ORDER BY total_points DESC"]
        )

    def test_query_7_fixture_results(self):
        self.verify_pipeline(
            prompt="Liverpool fixture results 2022-23",
            expected_intent=Intent.FIXTURE_RESULTS,
            expected_method="get_fixture_results",
            expected_params_subset={"team_name": "Liverpool", "season": "2022-23"},
            expected_cypher_strings=["MATCH (t:Team {name: $team_name})", "ORDER BY gw.number"]
        )

    def test_query_8_head_to_head(self):
        self.verify_pipeline(
            prompt="Arsenal vs Spurs head to head",
            expected_intent=Intent.HEAD_TO_HEAD,
            expected_method="get_head_to_head",
            expected_params_subset={"team1": "Arsenal", "team2": "Spurs"},
            expected_cypher_strings=["WHERE (ht.name = $team1 AND at.name = $team2)"]
        )

    def test_query_9_best_value(self):
        self.verify_pipeline(
            prompt="best value midfielders in 2022-23",
            expected_intent=Intent.BEST_VALUE,
            expected_method="get_best_value_players",
            expected_params_subset={"season": "2022-23", "position": "MID"},
            expected_cypher_strings=["ORDER BY points_per_million DESC", "pos:Position {code: $position}"]
        )

    def test_query_10_transfers(self):
        self.verify_pipeline(
            prompt="most transferred in gameweek 5 2022-23",
            expected_intent=Intent.TRANSFERS,
            expected_method="get_most_transferred_players",
            expected_params_subset={"season": "2022-23", "gameweek": 5},
            expected_cypher_strings=["ORDER BY r.transfers_in DESC"]
        )

    def test_query_11_bonus(self):
        self.verify_pipeline(
            prompt="most bonus points 2022-23",
            expected_intent=Intent.BONUS_POINTS,
            expected_method="get_bonus_point_leaders",
            expected_params_subset={"season": "2022-23"},
            expected_cypher_strings=["ORDER BY total_bonus DESC"]
        )

    def test_query_12_clean_sheets(self):
        self.verify_pipeline(
            prompt="most clean sheets 2022-23",
            expected_intent=Intent.CLEAN_SHEETS,
            expected_method="get_clean_sheet_leaders",
            expected_params_subset={"season": "2022-23"},
            expected_cypher_strings=["ORDER BY total_clean_sheets DESC"]
        )

    def test_query_13_ict(self):
        self.verify_pipeline(
            prompt="highest ict index 2022-23",
            expected_intent=Intent.ICT_INDEX,
            expected_method="get_ict_index_leaders",
            expected_params_subset={"season": "2022-23"},
            expected_cypher_strings=["ORDER BY avg_ict DESC"]
        )

    def test_query_14_most_selected(self):
        self.verify_pipeline(
            prompt="most selected players gameweek 1 2022-23",
            expected_intent=Intent.MOST_SELECTED,
            expected_method="get_most_selected_players",
            expected_params_subset={"season": "2022-23"},
            expected_cypher_strings=["ORDER BY r.selected DESC"]
        )

    def test_query_15_compare(self):
        # Use full names to ensure known_players matching matches logic if spacy falls back
        params = self.verify_pipeline(
            prompt="Compare Mohamed Salah and Harry Kane",
            expected_intent=Intent.PLAYER_COMPARISON,
            expected_method="compare_players",
            expected_cypher_strings=["WHERE p.name IN [$player1, $player2]"]
            # expected_params_subset handled manually due to extraction order
        )
        self.assertEqual({params.get("player1"), params.get("player2")}, 
                         {"Mohamed Salah", "Harry Kane"},
                         f"Players mismatch. Got {params.get('player1')}, {params.get('player2')}")
        
    def test_query_17_search(self):
        self.verify_pipeline(
            prompt="search for player Raya",
            expected_intent=Intent.PLAYER_SEARCH,
            expected_method="search_players_by_name",
            expected_cypher_strings=["WHERE toLower(p.name) CONTAINS toLower($name_pattern)"]
        )
        
    def test_query_20_season_summary(self):
        self.verify_pipeline(
            prompt="2022-23 season summary",
            expected_intent=Intent.SEASON_SUMMARY,
            expected_method="get_season_summary",
            expected_params_subset={"season": "2022-23"},
            expected_cypher_strings=["MATCH (s:Season {id: $season})"]
        )

if __name__ == "__main__":
    unittest.main()
