import unittest
import sys
import os
import csv
import inspect

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import FIXTURES_FALLBACK_CSV
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

            method = getattr(CypherQueries, mapped_method_name)

            # Filter params to only those accepted by the function
            sig = inspect.signature(method)
            # The handler runs get_top_players_all_positions instead when a required param is
            # missing, so the extracted params alone must cover the mapped query
            missing = [p.name for p in sig.parameters.values()
                       if p.default is inspect.Parameter.empty and p.name not in params]
            self.assertEqual(missing, [], f"{mapped_method_name} lacks {missing}; the handler would fall back")
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
            # Season queries filter on the position the player had that season
            expected_cypher_strings=["WHERE st.position = $position", "ORDER BY total_points DESC"]
        )
    
    def test_query_3_dynamic_sort_position(self):
        # The specific bug fix case
        self.verify_pipeline(
            prompt="get top goal scoring defenders",
            expected_intent=Intent.TOP_SCORERS, 
            expected_method="get_top_points_by_position",
            expected_params_subset={"position": "DEF", "sort_by": "goals"},
            expected_cypher_strings=["WHERE st.position = $position", "ORDER BY goals DESC"]
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
            expected_cypher_strings=["WHERE st.team = $team_name", "ORDER BY total_points DESC"]
        )

    def test_query_7_fixture_results(self):
        self.verify_pipeline(
            prompt="Liverpool fixture results 2022-23",
            expected_intent=Intent.FIXTURE_RESULTS,
            expected_method="get_fixture_results",
            expected_params_subset={"team_name": "Liverpool", "season": "2022-23"},
            expected_cypher_strings=["MATCH (:Team {name: $team_name})<-[:HOME_TEAM|AWAY_TEAM]-(f:Fixture)", "ORDER BY season, gameweek"]
        )

    def test_query_8_head_to_head(self):
        self.verify_pipeline(
            prompt="Arsenal vs Spurs head to head",
            expected_intent=Intent.HEAD_TO_HEAD,
            expected_method="get_head_to_head",
            expected_params_subset={"team1": "Arsenal", "team2": "Spurs"},
            expected_cypher_strings=["MATCH (:Team {name: $team1})<-[:HOME_TEAM|AWAY_TEAM]-(f:Fixture)-[:HOME_TEAM|AWAY_TEAM]->(:Team {name: $team2})"]
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
            expected_cypher_strings=["WHERE p.name_normalized CONTAINS $name_pattern"]
        )
        
    def test_query_20_season_summary(self):
        self.verify_pipeline(
            prompt="2022-23 season summary",
            expected_intent=Intent.SEASON_SUMMARY,
            expected_method="get_season_summary",
            expected_params_subset={"season": "2022-23"},
            expected_cypher_strings=["MATCH (s:Season {id: $season})"]
        )

    # ----- Regression: "top N <position>" questions used to classify as UNKNOWN, so no
    # query ran and the assistant said it had no data.

    def test_top_n_position_by_points_with_teams(self):
        self.verify_pipeline(
            prompt="List the top 5 midfielders by total points in 2023-24 with their teams",
            expected_intent=Intent.TOP_POINTS,
            expected_method="get_top_points_by_position",
            expected_params_subset={"position": "MID", "season": "2023-24", "limit": 5, "sort_by": "total_points"},
            expected_cypher_strings=["WHERE st.position = $position", "AS team", "ORDER BY total_points DESC"],
        )

    def test_best_n_position_is_a_ranking(self):
        self.verify_pipeline(
            prompt="best 3 defenders in 2021/22",
            expected_intent=Intent.TOP_POINTS,
            expected_method="get_top_points_by_position",
            expected_params_subset={"position": "DEF", "season": "2021-22", "limit": 3},
        )

    def test_top_n_position_in_gameweek_uses_current_position(self):
        self.verify_pipeline(
            prompt="top 3 defenders in gameweek 5 2025-26",
            expected_intent=Intent.TOP_POINTS,
            expected_method="get_top_points_by_position",
            expected_params_subset={"position": "DEF", "gameweek": 5, "limit": 3},
            expected_cypher_strings=["pos:Position {code: $position}", "head(collect(r.team)) AS team"],
        )

    def test_best_position_for_next_gameweek_stays_prediction(self):
        self.assertEqual(
            self.classifier.classify("Who are the best midfielders for next gameweek?").intent,
            Intent.PREDICTION,
        )

    def test_season_digits_are_not_a_limit(self):
        for prompt in ("Who scored the most goals in 2022-23?", "most assists in 2023/24",
                       "top scorers 22/23", "top scorers in 2021-2022"):
            params = self.extractor.get_query_parameters(self.extractor.extract(prompt))
            self.assertNotIn("limit", params, f"{prompt!r} -> {params}")
        params = self.extractor.get_query_parameters(self.extractor.extract("top 7 forwards in 2024-25"))
        self.assertEqual(params.get("limit"), 7)

    # ----- Regression: captain questions were RECOMMENDATION -> best-value query without a
    # season -> all-time fallback, which suggested players who have left the league.

    def test_captain_questions_are_predictions(self):
        for prompt in ("Who should I captain this gameweek?", "Who should I captain?",
                       "who to captain in gw8", "captaincy options this week"):
            self.assertEqual(self.classifier.classify(prompt).intent, Intent.PREDICTION, prompt)

    def test_recommendations_without_captain_unchanged(self):
        for prompt in ("recommend a cheap defender", "suggest some midfielders"):
            self.assertEqual(self.classifier.classify(prompt).intent, Intent.RECOMMENDATION, prompt)

    # ----- Regression: stats were matched as substrings, so "goalkeepers" contained "goal"
    # and "Top 3 goalkeepers by points" ranked keepers by goals (all 0).

    def stats_of(self, prompt):
        return self.extractor.extract(prompt).stats

    def sort_of(self, prompt):
        return self.extractor.get_query_parameters(self.extractor.extract(prompt)).get("sort_by")

    def test_goalkeepers_by_points_rank_by_points(self):
        self.verify_pipeline(
            prompt="Top 3 goalkeepers by points in 2023-24",
            expected_intent=Intent.TOP_POINTS,
            expected_method="get_top_points_by_position",
            expected_params_subset={"position": "GK", "season": "2023-24", "limit": 3, "sort_by": "total_points"},
            expected_cypher_strings=["WHERE st.position = $position", "ORDER BY total_points DESC"],
        )

    def test_stats_match_whole_words_only(self):
        # "goal" in goalkeepers, "cs" in statistics, "ict" in predict, "form" in perform,
        # "assist" in assistant
        for prompt in ("top 5 goalkeepers in 2024-25", "Show me statistics for Mohamed Salah",
                       "predict the best defenders", "how did Mohamed Salah perform",
                       "ask the assistant about Arsenal"):
            self.assertEqual(self.stats_of(prompt), [], prompt)

    def test_multi_word_stats_cover_their_words(self):
        cases = {
            "top midfielders by bonus points in 2023-24": ["bonus"],
            "most clean sheets 2022-23": ["clean_sheets"],
            "most red cards 2022-23": ["red_cards"],
            "List the top 5 midfielders by total points": ["total_points"],
            "highest ict index 2022-23": ["ict_index"],
        }
        for prompt, stats in cases.items():
            self.assertEqual(self.stats_of(prompt), stats, prompt)
        self.assertEqual(self.sort_of("top midfielders by bonus points in 2023-24"), "bonus")

    def test_stat_word_forms_set_the_sort(self):
        cases = {
            "top 5 goalscorers 2023-24": "goals",
            "most goal-scoring defenders": "goals",
            "top assisters in 2023-24": "assists",
            # the first stat mentioned decides
            "top midfielders by points and goals": "total_points",
            "top midfielders by goals and points": "goals",
        }
        for prompt, sort_by in cases.items():
            self.assertEqual(self.sort_of(prompt), sort_by, prompt)

    # ----- Regression: teams were matched as substrings ("city" in "Coventry City" -> Man
    # City, "united" in "Leeds United" -> Man Utd), and the promoted clubs were missing.

    def teams_of(self, prompt):
        return self.extractor.extract(prompt).teams

    def test_promoted_club_fixture_results(self):
        self.verify_pipeline(
            prompt="Coventry City fixture results",
            expected_intent=Intent.FIXTURE_RESULTS,
            expected_method="get_fixture_results",
            expected_params_subset={"team_name": "Coventry City"},
            expected_cypher_strings=["MATCH (:Team {name: $team_name})<-[:HOME_TEAM|AWAY_TEAM]-(f:Fixture)"],
        )

    def test_united_in_a_club_name_is_not_man_utd(self):
        params = self.verify_pipeline(
            prompt="Leeds United team analysis 2025-26",
            expected_intent=Intent.TEAM_ANALYSIS,
            expected_method="get_team_top_performers",
            expected_params_subset={"team_name": "Leeds", "season": "2025-26"},
            expected_cypher_strings=["WHERE st.team = $team_name"],
        )
        self.assertNotIn("team2", params)

    def test_head_to_head_between_two_city_clubs(self):
        self.verify_pipeline(
            prompt="Hull City vs Man City head to head",
            expected_intent=Intent.HEAD_TO_HEAD,
            expected_method="get_head_to_head",
            expected_params_subset={"team1": "Hull City", "team2": "Man City"},
        )

    def test_teams_match_whole_words_longest_first(self):
        cases = {
            "Ipswich Town team analysis": ["Ipswich"],
            "Sunderland fixture results": ["Sunderland"],
            "Newcastle United vs Manchester United": ["Newcastle", "Man Utd"],
            "Leicester City results in 2022-23": ["Leicester"],
            "Sheffield United players in 2023-24": ["Sheffield Utd"],
            "players from Nott'm Forest": ["Nott'm Forest"],
            "Arsenal's top scorer": ["Arsenal"],
            # nicknames still work on their own
            "How did City do in 2023-24?": ["Man City"],
            "how many points did United get": ["Man Utd"],
            "Who plays for Villa?": ["Aston Villa"],
            # and never inside another word
            "Who is Marc Albrighton?": [],
            "Villarreal results": [],
        }
        for prompt, teams in cases.items():
            self.assertEqual(self.teams_of(prompt), teams, prompt)

    def test_official_club_names(self):
        cases = {
            "AFC Bournemouth": "Bournemouth", "Brighton & Hove Albion": "Brighton",
            "Brighton and Hove Albion": "Brighton", "Coventry City": "Coventry City",
            "Hull City": "Hull City", "Ipswich Town": "Ipswich", "Leeds United": "Leeds",
            "Leicester City": "Leicester", "Luton Town": "Luton", "Manchester City": "Man City",
            "Manchester United": "Man Utd", "Newcastle United": "Newcastle",
            "Norwich City": "Norwich", "Nottingham Forest": "Nott'm Forest",
            "Sheffield United": "Sheffield Utd", "Tottenham Hotspur": "Spurs",
            "West Bromwich Albion": "West Brom", "West Ham United": "West Ham",
            "Wolverhampton Wanderers": "Wolves",
        }
        for name, team in cases.items():
            self.assertEqual(EntityExtractor.find_teams(name), [team], name)

    def test_every_club_name_resolves_to_itself(self):
        # So a question written with a graph Team name (a card's follow-up, say) is about
        # that club alone
        for team in sorted(set(EntityExtractor.TEAMS.values())):
            self.assertEqual(EntityExtractor.find_teams(team), [team], team)

    def test_current_season_clubs_are_known(self):
        # The 2026-27 fixture list uses the graph's club names, promoted clubs included
        with open(FIXTURES_FALLBACK_CSV, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        clubs = {row["Home Team"] for row in rows} | {row["Away Team"] for row in rows}
        self.assertEqual(len(clubs), 20)
        for club in sorted(clubs):
            self.assertEqual(EntityExtractor.find_teams(club), [club], club)

    # ----- Regression: PLAYER_SEARCH maps to search_players_by_name(name_pattern), but the
    # extracted player was never passed as name_pattern, so "Who is Bukayo Saka?" got the
    # top-players fallback.

    def test_who_is_player_runs_the_player_search(self):
        params = self.verify_pipeline(
            prompt="Who is Bukayo Saka?",
            expected_intent=Intent.PLAYER_SEARCH,
            expected_method="search_players_by_name",
            expected_params_subset={"name_pattern": "Saka"},
            expected_cypher_strings=["WHERE p.name_normalized CONTAINS $name_pattern"],
        )
        _, q_params = CypherQueries.search_players_by_name(params["name_pattern"])
        self.assertEqual(q_params["name_pattern"], "saka")

    def test_no_player_no_name_pattern(self):
        params = self.extractor.get_query_parameters(self.extractor.extract("Who is the top scorer in 2023-24?"))
        self.assertNotIn("name_pattern", params)


class TestDefaultRequiredSeason(unittest.TestCase):
    """The /api/query handler fills a missing required season with the latest one."""

    @classmethod
    def setUpClass(cls):
        from api_main import default_required_season
        cls.fill = staticmethod(default_required_season)

    def test_fills_required_season(self):
        params = {"position": "DEF"}
        self.fill(CypherQueries.get_best_value_players, params, "2026-27")
        self.assertEqual(params["season"], "2026-27")
        query, q_params = CypherQueries.get_best_value_players(**params)
        self.assertEqual(q_params.get("season"), "2026-27")

    def test_keeps_season_from_question(self):
        params = {"season": "2023-24"}
        self.fill(CypherQueries.get_best_value_players, params, "2026-27")
        self.assertEqual(params["season"], "2023-24")

    def test_optional_season_stays_all_seasons(self):
        # get_top_points_by_position takes season=None to mean all seasons; leave it alone
        params = {}
        self.fill(CypherQueries.get_top_points_by_position, params, "2026-27")
        self.assertNotIn("season", params)


class TestTeamQuestionIntents(unittest.TestCase):
    """Player and team questions share shapes ("How did X do?", "X vs Y"). The team patterns
    used to take one word, and on a tie the player intent won by coming first, so team
    questions ran no query or the top-players fallback."""

    verify_pipeline = TestAppPipeline.verify_pipeline

    @classmethod
    def setUpClass(cls):
        from config.settings import CURRENT_SEASON
        cls.current_season = CURRENT_SEASON
        cls.classifier = IntentClassifier()
        cls.extractor = EntityExtractor(known_players={"Mohamed Salah", "Harry Kane", "Haaland", "Saka", "Palmer"})

    def intent_of(self, prompt):
        return self.classifier.classify(prompt).intent

    def test_how_is_a_club_doing(self):
        # The rich cards' team follow-up; "Coventry City" or "Aston Villa" didn't fit the
        # one-word slot, so it was a general question
        for club in sorted(set(EntityExtractor.TEAMS.values())):
            with self.subTest(club=club):
                self.verify_pipeline(
                    prompt=f"How is {club} doing this season?",
                    expected_intent=Intent.TEAM_ANALYSIS,
                    expected_method="get_team_top_performers",
                    expected_params_subset={"team_name": club, "season": self.current_season},
                )

    def test_how_did_a_club_nickname_do(self):
        # Tied with PLAYER_STATS, which lacked a player and fell back to the top players
        self.verify_pipeline(
            prompt="How did City do in 2023-24?",
            expected_intent=Intent.TEAM_ANALYSIS,
            expected_method="get_team_top_performers",
            expected_params_subset={"team_name": "Man City", "season": "2023-24"},
            expected_cypher_strings=["WHERE st.team = $team_name"],
        )

    def test_club_vs_club_without_head_to_head_words(self):
        # Was PLAYER_COMPARISON (compare_players without players)
        self.verify_pipeline(
            prompt="Newcastle United vs Manchester United",
            expected_intent=Intent.HEAD_TO_HEAD,
            expected_method="get_head_to_head",
            expected_params_subset={"team1": "Newcastle", "team2": "Man Utd"},
        )

    def test_other_team_question_shapes(self):
        cases = {
            "How are Arsenal doing?": Intent.TEAM_ANALYSIS,
            "How did Wolves get on last season?": Intent.TEAM_ANALYSIS,
            "How are Arsenal players doing?": Intent.TEAM_ANALYSIS,
            "Arsenal stats": Intent.TEAM_ANALYSIS,
            "Who is Hull City's best player?": Intent.TEAM_ANALYSIS,
            "find players from Aston Villa": Intent.TEAM_ANALYSIS,
            "Best players at Coventry City this season": Intent.TEAM_ANALYSIS,
            "Aston Villa versus Crystal Palace": Intent.HEAD_TO_HEAD,
            "Man City v Man Utd": Intent.HEAD_TO_HEAD,
            "Compare Aston Villa and Crystal Palace": Intent.HEAD_TO_HEAD,
            "results between Coventry City and Hull City": Intent.HEAD_TO_HEAD,
            "How did Arsenal do against Chelsea?": Intent.HEAD_TO_HEAD,
            "Head to head Hull City vs Nott'm Forest": Intent.HEAD_TO_HEAD,
        }
        for prompt, intent in cases.items():
            self.assertEqual(self.intent_of(prompt), intent, prompt)

    def test_player_questions_keep_their_intents(self):
        cases = {
            "how did Mohamed Salah perform": Intent.PLAYER_STATS,
            "How did Haaland do for City?": Intent.PLAYER_STATS,
            "how did Salah perform against Arsenal": Intent.PLAYER_STATS,
            "Haaland stats this season": Intent.PLAYER_STATS,
            # a club word in a player's name, but not in the team pattern's slot
            "Conor Coventry 2022-23 stats": Intent.PLAYER_STATS,
            "Compare Mohamed Salah and Harry Kane": Intent.PLAYER_COMPARISON,
            "who is better Saka or Palmer": Intent.PLAYER_COMPARISON,
            "Salah vs Kane": Intent.PLAYER_COMPARISON,
            "Who is Bukayo Saka?": Intent.PLAYER_SEARCH,
        }
        for prompt, intent in cases.items():
            self.assertEqual(self.intent_of(prompt), intent, prompt)

    def test_how_is_a_player_doing(self):
        # A one-word name fit the old team slot (TEAM_ANALYSIS); a full name was a general
        # question
        for prompt in ("How is Salah doing this season?", "How is Kevin De Bruyne doing?",
                       "How has Trent Alexander-Arnold done this season?"):
            self.assertEqual(self.intent_of(prompt), Intent.PLAYER_STATS, prompt)
        self.verify_pipeline(
            prompt="How is Mohamed Salah doing in 2023-24?",
            expected_intent=Intent.PLAYER_STATS,
            expected_method="get_player_season_stats",
            expected_params_subset={"player_name": "Mohamed Salah", "season": "2023-24"},
        )
        self.assertEqual(self.intent_of("How are you doing?"), Intent.GENERAL_QUESTION)

    def test_head_to_head_words_dont_make_players_clubs(self):
        # "head to head" plus "X vs Y" used to outscore the comparison, then lack team1/team2
        params = self.verify_pipeline(
            prompt="Mohamed Salah vs Harry Kane head to head",
            expected_intent=Intent.PLAYER_COMPARISON,
            expected_method="compare_players",
        )
        self.assertEqual({params["player1"], params["player2"]}, {"Mohamed Salah", "Harry Kane"})

    # ----- /api/query's refine_with_entities: a player intent that names clubs but no player
    # ran the top-players fallback ("Arsenal stats 2023-24" outscores the team pattern)

    def refined(self, prompt):
        """The handler's intent, method and params for the prompt."""
        entities = self.extractor.extract(prompt)
        intent = self.classifier.refine_with_entities(
            self.classifier.classify(prompt).intent, entities.players, entities.teams)
        params = self.extractor.get_query_parameters(entities)
        method = getattr(CypherQueries, self.classifier.get_query_type_for_intent(intent))
        missing = [p.name for p in inspect.signature(method).parameters.values()
                   if p.default is inspect.Parameter.empty and p.name not in params]
        self.assertEqual(missing, [], f"{prompt!r}: {method.__name__} lacks {missing}")
        return intent, params

    def test_club_only_player_questions_become_team_questions(self):
        cases = {
            "Arsenal stats 2023-24": (Intent.TEAM_ANALYSIS, {"team_name": "Arsenal", "season": "2023-24"}),
            "Aston Villa stats this season": (Intent.TEAM_ANALYSIS, {"team_name": "Aston Villa"}),
            "Tell me about Arsenal": (Intent.TEAM_ANALYSIS, {"team_name": "Arsenal"}),
            "Show me Coventry City": (Intent.TEAM_ANALYSIS, {"team_name": "Coventry City"}),
            "Arsenal stats and Chelsea stats": (Intent.HEAD_TO_HEAD, {"team1": "Arsenal", "team2": "Chelsea"}),
        }
        for prompt, (intent, params) in cases.items():
            got_intent, got_params = self.refined(prompt)
            self.assertEqual(got_intent, intent, prompt)
            self.assertEqual({k: got_params.get(k) for k in params}, params, prompt)

    def test_refinement_needs_clubs_and_no_player(self):
        refine = self.classifier.refine_with_entities
        self.assertEqual(refine(Intent.PLAYER_STATS, ["Harry Kane"], ["Spurs"]), Intent.PLAYER_STATS)
        self.assertEqual(refine(Intent.PLAYER_STATS, [], []), Intent.PLAYER_STATS)
        self.assertEqual(refine(Intent.PLAYER_SEARCH, [], ["Coventry City"]), Intent.TEAM_ANALYSIS)
        self.assertEqual(refine(Intent.PLAYER_COMPARISON, [], ["Arsenal", "Chelsea"]), Intent.HEAD_TO_HEAD)
        # only player intents change
        self.assertEqual(refine(Intent.TOP_SCORERS, [], ["Arsenal"]), Intent.TOP_SCORERS)
        self.assertEqual(refine(Intent.FIXTURE_RESULTS, [], ["Arsenal"]), Intent.FIXTURE_RESULTS)
        kane = self.extractor.extract("Kane stats for Spurs")
        self.assertEqual(refine(self.classifier.classify("Kane stats for Spurs").intent, kane.players, kane.teams),
                         Intent.PLAYER_STATS)


if __name__ == "__main__":
    unittest.main()
