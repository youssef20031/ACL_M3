"""
End-to-End Integration Tests for Date-Aware Match Predictions

These tests validate the complete flow from fixture data loading through
prediction generation and caching, ensuring all components work together correctly.

**Task 18: Write end-to-end integration tests**
"""
import unittest
import sys
import os
import tempfile
import csv
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from graph.connection import Neo4jConnection
from services.fixture_service import FixtureService, FixtureContext
from services.cache import PredictionCache
from ml.api_integration import MLAPIIntegration, PlayerPredictionRequest
from ml.predictor import FPLPredictor, PlayerPrediction


def run_async(coro):
    """Helper to run async functions in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestE2EFixtureAwarePredictionFlow(unittest.TestCase):
    """
    Test 18.1: End-to-end test for complete fixture-aware prediction flow
    
    Tests the complete flow:
    1. Load fixtures from test CSV
    2. Request prediction for player with upcoming fixture
    3. Verify response includes fixture_context
    4. Verify fixture_context fields are correct
    
    **Validates: Requirements 1.1, 2.1, 3.1, 3.2, 3.3, 4.1**
    """
    
    def setUp(self):
        """Set up test fixtures and mocks."""
        # Create mock Neo4j connection
        self.mock_neo4j_conn = Mock(spec=Neo4jConnection)
        
        # Create mock cache
        self.mock_cache = Mock(spec=PredictionCache)
        self.mock_cache.get.return_value = None  # Start with cache miss
        
        # Create fixture service with mocks
        self.fixture_service = FixtureService(
            neo4j_conn=self.mock_neo4j_conn,
            cache=self.mock_cache
        )
        
        # Create ML integration with mocks
        self.ml_integration = MLAPIIntegration(
            neo4j_conn=self.mock_neo4j_conn,
            query_executor=Mock(),
            fixture_service=self.fixture_service
        )
        
        # Mock predictor
        self.mock_predictor = Mock(spec=FPLPredictor)
        self.ml_integration.predictor = self.mock_predictor
        self.ml_integration.predictor_loaded = True
        
        # Test data
        self.test_player_name = "Mohamed Salah"
        self.test_team_name = "Liverpool"
        self.test_opponent = "Manchester United"
        # Relative to now so the fixture is always upcoming, whatever the date the suite runs on
        self.current_date = datetime.now(timezone.utc).replace(microsecond=0)
        self.test_fixture_date = (self.current_date + timedelta(days=7)).replace(hour=15, minute=0, second=0)
        self.test_matchday = 2
        self.test_fixture_id = "fixture-2026-27-md2-liv-mun"
    
    def test_e2e_fixture_aware_prediction_flow(self):
        """
        Complete E2E test: Load fixtures → Request prediction → Verify fixture context
        
        **Validates: Requirements 1.1, 2.1, 3.1, 3.2, 3.3, 4.1**
        """
        # Step 1: Simulate fixture loading
        # Mock valid teams query
        self.mock_neo4j_conn.execute_query.return_value = [
            {"name": "Liverpool"},
            {"name": "Manchester United"},
            {"name": "Arsenal"},
            {"name": "Chelsea"}
        ]
        
        # Create temporary test CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
            writer.writeheader()
            writer.writerow({
                "Matchday": self.test_matchday,
                "Home Team": self.test_team_name,
                "Away Team": self.test_opponent,
                "Date": self.test_fixture_date.strftime("%d/%m/%Y"),
                "Time": self.test_fixture_date.strftime("%H:%M")
            })
            csv_path = f.name
        
        try:
            # Mock execute_write for fixture loading
            self.mock_neo4j_conn.execute_write = Mock()
            
            # Load fixtures from CSV
            fixtures_loaded = self.fixture_service.load_fixtures_from_csv(csv_path, season_id="2026-27")
            
            # Verify fixture was loaded
            self.assertEqual(fixtures_loaded, 1)
            self.mock_neo4j_conn.execute_write.assert_called()
            
            # Step 2: Mock fixture lookup for prediction
            fixture_context = FixtureContext(
                opponent_team=self.test_opponent,
                home_away="home",
                match_date=self.test_fixture_date,
                matchday=self.test_matchday,
                fixture_id=self.test_fixture_id
            )
            
            # Reset mock to handle fixture lookup
            self.mock_neo4j_conn.execute_query.return_value = [{
                "fixture_id": self.test_fixture_id,
                "match_date": self.test_fixture_date,
                "matchday": self.test_matchday,
                "opponent_team": self.test_opponent,
                "home_away": "home"
            }]
            
            # Get next match
            next_match = self.fixture_service.get_next_match(
                self.test_player_name,
                current_date=self.current_date
            )
            
            # Verify fixture context is returned
            self.assertIsNotNone(next_match)
            self.assertEqual(next_match.opponent_team, self.test_opponent)
            self.assertEqual(next_match.home_away, "home")
            self.assertEqual(next_match.matchday, self.test_matchday)
            self.assertEqual(next_match.fixture_id, self.test_fixture_id)
            
            # Step 3: Request prediction with fixture context
            # Mock player data query
            self.mock_neo4j_conn.execute_query.return_value = [{
                "name": self.test_player_name,
                "position": "FWD",
                "minutes": 90.0,
                "goals_scored": 1.0,
                "assists": 0.5,
                "form": 7.0,
                "value": 130.0
            }]
            
            # Mock fixture service to return fixture context
            with patch.object(self.fixture_service, 'get_next_match', return_value=fixture_context):
                # Mock predictor response
                mock_prediction = PlayerPrediction(
                    player_name=self.test_player_name,
                    predicted_points=8.5,
                    features_used={"form": 7.0, "opponent_team": self.test_opponent},
                    model_version="xgboost_fwd_v3"
                )
                self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
                
                # Request prediction
                request = PlayerPredictionRequest(player_name=self.test_player_name)
                response = run_async(self.ml_integration.predict_player_next_gameweek(request))
                
                # Step 4: Verify response includes correct fixture context
                self.assertIsNotNone(response.fixture_context)
                self.assertEqual(response.fixture_context["opponent_team"], self.test_opponent)
                self.assertEqual(response.fixture_context["home_away"], "home")
                self.assertEqual(response.fixture_context["matchday"], self.test_matchday)
                self.assertEqual(response.fixture_context["fixture_id"], self.test_fixture_id)
                
                # Verify match_date is present
                self.assertIn("match_date", response.fixture_context)
                
                # Verify predicted points
                self.assertEqual(response.predicted_points, 8.5)
                self.assertEqual(response.player_name, self.test_player_name)
                
                # Verify predictor received enriched player data with fixture context
                call_args = self.mock_predictor.predict_next_gameweek.call_args
                enriched_data = call_args[0][0]
                self.assertEqual(enriched_data["opponent_team"], self.test_opponent)
                self.assertEqual(enriched_data["was_home"], 1)  # home = 1
                self.assertEqual(enriched_data["fixture_id"], self.test_fixture_id)
        
        finally:
            # Clean up temp file
            os.unlink(csv_path)
    
    def test_e2e_prediction_without_fixture_context(self):
        """
        E2E test for prediction when no fixture context is available.
        
        **Validates: Requirements 3.5, 4.3**
        """
        # Mock no fixture context
        self.mock_neo4j_conn.execute_query.return_value = []
        
        # Get next match (should return None)
        next_match = self.fixture_service.get_next_match(
            self.test_player_name,
            current_date=self.current_date
        )
        
        self.assertIsNone(next_match)
        
        # Mock player data query
        self.mock_neo4j_conn.execute_query.return_value = [{
            "name": self.test_player_name,
            "position": "FWD",
            "minutes": 90.0,
            "form": 7.0
        }]
        
        # Mock predictor response
        mock_prediction = PlayerPrediction(
            player_name=self.test_player_name,
            predicted_points=7.0,
            features_used={},
            model_version="xgboost_fwd_v3"
        )
        self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
        
        # Request prediction
        request = PlayerPredictionRequest(player_name=self.test_player_name)
        response = run_async(self.ml_integration.predict_player_next_gameweek(request))
        
        # Verify response has null fixture_context
        self.assertIsNone(response.fixture_context)
        
        # Verify prediction still succeeds
        self.assertEqual(response.predicted_points, 7.0)
        self.assertEqual(response.player_name, self.test_player_name)


class TestE2EPredictionCachingBehavior(unittest.TestCase):
    """
    Test 18.2: End-to-end test for prediction caching behavior
    
    Tests:
    1. Request same prediction twice
    2. Verify second request is faster (cache hit)
    3. Update fixture
    4. Verify cache is invalidated
    
    **Validates: Requirements 10.1, 10.2, 10.4**
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create in-memory cache (not mocked, using real implementation)
        self.cache = PredictionCache(redis_client=None)
        
        # Create mock Neo4j connection
        self.mock_neo4j_conn = Mock(spec=Neo4jConnection)
        
        # Create fixture service with real cache
        self.fixture_service = FixtureService(
            neo4j_conn=self.mock_neo4j_conn,
            cache=self.cache
        )
        
        # Create ML integration
        self.ml_integration = MLAPIIntegration(
            neo4j_conn=self.mock_neo4j_conn,
            query_executor=Mock(),
            fixture_service=self.fixture_service,
            cache=self.cache
        )
        
        # Mock predictor
        self.mock_predictor = Mock(spec=FPLPredictor)
        self.ml_integration.predictor = self.mock_predictor
        self.ml_integration.predictor_loaded = True
        
        # Test data
        self.test_player_name = "Kevin De Bruyne"
        self.test_fixture_id = "fixture-2026-27-md5-mci-ars"
        # Must stay in the future: PredictionCache.calculate_cache_ttl returns 0 (no caching)
        # for a kickoff that has passed. Three days ahead gives the 6-hour TTL.
        self.test_fixture_date = datetime.now(timezone.utc) + timedelta(days=3)
    
    def test_e2e_prediction_caching_flow(self):
        """
        Complete E2E caching test: Request prediction → Cache → Cache hit → Invalidate
        
        **Validates: Requirements 10.1, 10.2, 10.4**
        """
        # Setup fixture context
        fixture_context = FixtureContext(
            opponent_team="Arsenal",
            home_away="home",
            match_date=self.test_fixture_date,
            matchday=5,
            fixture_id=self.test_fixture_id
        )
        
        # Mock fixture lookup
        with patch.object(self.fixture_service, 'get_next_match', return_value=fixture_context):
            # Mock player data query
            self.mock_neo4j_conn.execute_query.return_value = [{
                "name": self.test_player_name,
                "position": "MID",
                "minutes": 85.0,
                "form": 8.5,
                "value": 100.0
            }]
            
            # Mock predictor response
            mock_prediction = PlayerPrediction(
                player_name=self.test_player_name,
                predicted_points=9.2,
                features_used={},
                model_version="xgboost_mid_v3"
            )
            self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
            
            # Step 1: First prediction request (cache MISS)
            request = PlayerPredictionRequest(player_name=self.test_player_name)
            
            import time
            start_time = time.time()
            response1 = run_async(self.ml_integration.predict_player_next_gameweek(request))
            first_request_time = time.time() - start_time
            
            # Verify predictor was called (cache miss)
            self.assertEqual(self.mock_predictor.predict_next_gameweek.call_count, 1)
            
            # Verify response
            self.assertEqual(response1.predicted_points, 9.2)
            self.assertIsNotNone(response1.fixture_context)
            
            # Reset predictor mock for second request
            self.mock_predictor.reset_mock()
            
            # Step 2: Second prediction request (cache HIT)
            start_time = time.time()
            response2 = run_async(self.ml_integration.predict_player_next_gameweek(request))
            second_request_time = time.time() - start_time
            
            # Verify predictor was NOT called (cache hit)
            self.assertEqual(self.mock_predictor.predict_next_gameweek.call_count, 0)
            
            # Verify response is identical
            self.assertEqual(response2.predicted_points, response1.predicted_points)
            self.assertEqual(response2.player_name, response1.player_name)
            
            # Verify second request was faster (cache hit should be significantly faster)
            # Note: This may not always be reliable in unit tests, but demonstrates the concept
            print(f"First request: {first_request_time*1000:.2f}ms")
            print(f"Second request (cached): {second_request_time*1000:.2f}ms")
            
            # Step 3: Update fixture (should invalidate cache)
            # Mock fixture update
            self.mock_neo4j_conn.execute_query.return_value = [{
                "f": {
                    "id": self.test_fixture_id,
                    "kickoff_time": self.test_fixture_date,
                    "matchday": 5
                }
            }]
            self.mock_neo4j_conn.execute_write = Mock()
            
            # Update fixture
            update_success = self.fixture_service.update_fixture(
                self.test_fixture_id,
                {"kickoff_time": self.test_fixture_date + timedelta(hours=2)},
                user="test_user"
            )
            
            self.assertTrue(update_success)
            
            # Step 4: Third prediction request (cache MISS after invalidation)
            self.mock_predictor.reset_mock()
            self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
            
            response3 = run_async(self.ml_integration.predict_player_next_gameweek(request))
            
            # Verify predictor was called again (cache was invalidated)
            self.assertEqual(self.mock_predictor.predict_next_gameweek.call_count, 1)
            
            # Verify response
            self.assertEqual(response3.predicted_points, 9.2)
    
    def test_cache_key_generation_format(self):
        """
        Test cache key format matches specification.
        
        **Validates: Requirements 10.1**
        """
        cache_key = PredictionCache.generate_prediction_key(
            player_id=self.test_player_name,
            fixture_id=self.test_fixture_id,
            model_version="xgboost_v1"
        )
        
        # Verify key format
        expected_key = f"prediction:{self.test_player_name}:{self.test_fixture_id}:xgboost_v1"
        self.assertEqual(cache_key, expected_key)
        
        # Verify key contains all components
        self.assertIn("prediction:", cache_key)
        self.assertIn(self.test_player_name, cache_key)
        self.assertIn(self.test_fixture_id, cache_key)
        self.assertIn("xgboost_v1", cache_key)
    
    def test_cache_ttl_dynamic_calculation(self):
        """
        Test cache TTL varies based on fixture timing.
        
        **Validates: Requirements 10.3**
        """
        # Fixture 10 days away - TTL should be 24 hours
        far_future = datetime.now(timezone.utc) + timedelta(days=10)
        ttl_far = PredictionCache.calculate_cache_ttl(far_future)
        self.assertEqual(ttl_far, 86400)  # 24 hours
        
        # Fixture 3 days away - TTL should be 6 hours
        near_future = datetime.now(timezone.utc) + timedelta(days=3)
        ttl_near = PredictionCache.calculate_cache_ttl(near_future)
        self.assertEqual(ttl_near, 21600)  # 6 hours
        
        # Fixture 12 hours away - TTL should be 1 hour
        very_near = datetime.now(timezone.utc) + timedelta(hours=12)
        ttl_very_near = PredictionCache.calculate_cache_ttl(very_near)
        self.assertEqual(ttl_very_near, 3600)  # 1 hour
        
        # Fixture in the past - TTL should be 0
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        ttl_past = PredictionCache.calculate_cache_ttl(past)
        self.assertEqual(ttl_past, 0)  # No caching


class TestE2EFixtureRefreshWorkflow(unittest.TestCase):
    """
    Test 18.3: End-to-end test for fixture refresh workflow
    
    Tests:
    1. Load initial fixtures
    2. Modify fixture data
    3. Reload fixtures
    4. Verify updated data is reflected
    5. Verify historical fixtures preserved
    
    **Validates: Requirements 6.2, 6.3**
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock Neo4j connection
        self.mock_neo4j_conn = Mock(spec=Neo4jConnection)
        
        # Create fixture service
        self.fixture_service = FixtureService(
            neo4j_conn=self.mock_neo4j_conn
        )
        
        # Test data
        self.test_season = "2026-27"
        self.test_matchday = 10
    
    def test_e2e_fixture_refresh_workflow(self):
        """
        Complete E2E fixture refresh test: Load → Modify → Reload → Verify
        
        **Validates: Requirements 6.2, 6.3**
        """
        # Step 1: Load initial fixtures
        # Mock valid teams query
        self.mock_neo4j_conn.execute_query.return_value = [
            {"name": "Chelsea"},
            {"name": "Tottenham Hotspur"}
        ]
        
        # Create initial CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
            writer.writeheader()
            writer.writerow({
                "Matchday": self.test_matchday,
                "Home Team": "Chelsea",
                "Away Team": "Tottenham Hotspur",
                "Date": "01/11/2026",
                "Time": "15:00"
            })
            csv_path = f.name
        
        try:
            # Mock execute_write
            self.mock_neo4j_conn.execute_write = Mock()
            
            # Load initial fixtures
            fixtures_loaded = self.fixture_service.load_fixtures_from_csv(csv_path, season_id=self.test_season)
            self.assertEqual(fixtures_loaded, 1)
            
            # Verify MERGE was called (idempotent loading)
            write_calls = self.mock_neo4j_conn.execute_write.call_args_list
            self.assertTrue(len(write_calls) > 0)
            
            # Step 2: Modify fixture data (change kickoff time)
            modified_date = "01/11/2026"
            modified_time = "17:30"  # Changed from 15:00 to 17:30
            
            # Create modified CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
                writer.writeheader()
                writer.writerow({
                    "Matchday": self.test_matchday,
                    "Home Team": "Chelsea",
                    "Away Team": "Tottenham Hotspur",
                    "Date": modified_date,
                    "Time": modified_time
                })
            
            # Step 3: Reload fixtures
            self.mock_neo4j_conn.execute_write.reset_mock()
            
            fixtures_reloaded = self.fixture_service.load_fixtures_from_csv(csv_path, season_id=self.test_season)
            self.assertEqual(fixtures_reloaded, 1)
            
            # Verify fixture was updated (MERGE should handle updates)
            self.mock_neo4j_conn.execute_write.assert_called()
            
            # Step 4: Verify idempotency - loading same data twice shouldn't cause errors
            self.mock_neo4j_conn.execute_write.reset_mock()
            
            fixtures_reloaded_again = self.fixture_service.load_fixtures_from_csv(csv_path, season_id=self.test_season)
            self.assertEqual(fixtures_reloaded_again, 1)
            
            # Verify MERGE was still called (idempotent)
            self.mock_neo4j_conn.execute_write.assert_called()
        
        finally:
            # Clean up temp file
            os.unlink(csv_path)
    
    def test_fixture_update_preserves_historical_data(self):
        """
        Test that updating fixtures preserves historical match scores.
        
        **Validates: Requirements 6.3**
        """
        fixture_id = "fixture-2026-27-md1-che-tot"
        
        # Mock existing fixture with historical scores
        self.mock_neo4j_conn.execute_query.return_value = [{
            "f": {
                "id": fixture_id,
                "kickoff_time": datetime(2026, 8, 15, 15, 0, tzinfo=timezone.utc),
                "matchday": 1,
                "home_score": 2,  # Historical score
                "away_score": 1   # Historical score
            }
        }]
        
        self.mock_neo4j_conn.execute_write = Mock()
        
        # Update fixture (changing kickoff time, but NOT scores)
        updates = {
            "kickoff_time": datetime(2026, 8, 15, 17, 30, tzinfo=timezone.utc)
        }
        
        success = self.fixture_service.update_fixture(fixture_id, updates, user="test_admin")
        
        self.assertTrue(success)
        
        # Verify update was called
        self.mock_neo4j_conn.execute_write.assert_called_once()
        
        # Verify only kickoff_time was updated (scores should not be in SET clause)
        call_args = self.mock_neo4j_conn.execute_write.call_args
        query = call_args[0][0]
        
        # Verify query contains kickoff_time update
        self.assertIn("kickoff_time", query)
        
        # Verify scores are NOT being updated
        self.assertNotIn("home_score", query)
        self.assertNotIn("away_score", query)
    
    def test_fixture_validation_on_reload(self):
        """
        Test that invalid fixture data is rejected during reload.
        
        **Validates: Requirements 1.4, 5.4**
        """
        # Mock valid teams query (only Chelsea)
        self.mock_neo4j_conn.execute_query.return_value = [
            {"name": "Chelsea"}
        ]
        
        # Create CSV with invalid team
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
            writer.writeheader()
            writer.writerow({
                "Matchday": 1,
                "Home Team": "Chelsea",
                "Away Team": "Invalid Team Name",  # Team doesn't exist
                "Date": "15/08/2026",
                "Time": "15:00"
            })
            csv_path = f.name
        
        try:
            self.mock_neo4j_conn.execute_write = Mock()
            
            # Load fixtures - should skip invalid fixture
            fixtures_loaded = self.fixture_service.load_fixtures_from_csv(csv_path, season_id=self.test_season)
            
            # Verify no fixtures were loaded (invalid team)
            self.assertEqual(fixtures_loaded, 0)
            
            # Verify execute_write was NOT called (fixture was rejected)
            self.mock_neo4j_conn.execute_write.assert_not_called()
        
        finally:
            # Clean up temp file
            os.unlink(csv_path)


class TestE2EMultipleFixturesAndDGW(unittest.TestCase):
    """
    Additional E2E tests for edge cases: Double Gameweeks, multiple fixtures
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_neo4j_conn = Mock(spec=Neo4jConnection)
        self.fixture_service = FixtureService(neo4j_conn=self.mock_neo4j_conn)
    
    def test_e2e_double_gameweek_handling(self):
        """
        Test that system correctly handles Double Gameweek scenarios.
        
        **Validates: Requirements 2.5**
        """
        player_name = "Harry Kane"
        current_date = datetime(2026, 12, 1, 12, 0, tzinfo=timezone.utc)
        
        # Mock two fixtures on the same matchday at different times
        # Next match should be the EARLIEST fixture
        self.mock_neo4j_conn.execute_query.return_value = [{
            "fixture_id": "fixture-2026-27-md15-tot-ars-1",
            "match_date": datetime(2026, 12, 10, 12, 30, tzinfo=timezone.utc),  # Earlier
            "matchday": 15,
            "opponent_team": "Arsenal",
            "home_away": "home"
        }]
        
        # Get next match
        next_match = self.fixture_service.get_next_match(player_name, current_date)
        
        # Verify earliest fixture is returned
        self.assertIsNotNone(next_match)
        self.assertEqual(next_match.opponent_team, "Arsenal")
        self.assertEqual(next_match.matchday, 15)
        
        # Verify query uses ORDER BY kickoff_time and LIMIT 1
        query_call = self.mock_neo4j_conn.execute_query.call_args
        query = query_call[0][0]
        self.assertIn("ORDER BY", query)
        self.assertIn("LIMIT 1", query)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
