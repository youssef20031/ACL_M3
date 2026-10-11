"""
Property-Based Tests and Unit Tests for ML API Integration with Fixture Context
Tests the integration of FixtureService with ML prediction system.
"""
import unittest
import sys
import os
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.api_integration import MLAPIIntegration, PlayerPredictionRequest, PredictionResponse
from services.fixture_service import FixtureService, FixtureContext
from ml.predictor import PlayerPrediction


def run_async(coro):
    """Helper to run async functions in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

try:
    from hypothesis import given, strategies as st, assume, settings, HealthCheck
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    print("WARNING: hypothesis not installed. Property-based tests will be skipped.")
    print("Install with: pip install hypothesis")


# Helper strategies for generating test data
if HYPOTHESIS_AVAILABLE:
    @composite
    def valid_player_name(draw):
        """Generate valid player names."""
        first_names = ["Mohamed", "Harry", "Kevin", "Bruno", "Erling"]
        last_names = ["Salah", "Kane", "De Bruyne", "Fernandes", "Haaland"]
        return f"{draw(st.sampled_from(first_names))} {draw(st.sampled_from(last_names))}"
    
    @composite
    def valid_fixture_context(draw):
        """Generate valid fixture context for property tests."""
        opponent_teams = ["Manchester United", "Liverpool", "Arsenal", "Chelsea", "Manchester City"]
        home_away = draw(st.sampled_from(["home", "away"]))
        
        # Generate future date
        days_ahead = draw(st.integers(min_value=1, max_value=30))
        match_date = datetime.now(timezone.utc) + timedelta(days=days_ahead)
        
        matchday = draw(st.integers(min_value=1, max_value=38))
        
        return FixtureContext(
            opponent_team=draw(st.sampled_from(opponent_teams)),
            home_away=home_away,
            match_date=match_date,
            matchday=matchday,
            fixture_id=f"fixture-2026-27-md{matchday}-test"
        )
    
    @composite
    def valid_player_data(draw):
        """Generate valid player stats data."""
        return {
            "name": draw(valid_player_name()),
            "position": draw(st.sampled_from(["GK", "DEF", "MID", "FWD"])),
            "minutes": draw(st.floats(min_value=0, max_value=90)),
            "goals_scored": draw(st.floats(min_value=0, max_value=5)),
            "assists": draw(st.floats(min_value=0, max_value=5)),
            "form": draw(st.floats(min_value=0, max_value=15)),
            "bps": draw(st.floats(min_value=0, max_value=100)),
            "ict_index": draw(st.floats(min_value=0, max_value=50)),
            "value": draw(st.floats(min_value=40, max_value=150))
        }


@unittest.skipIf(not HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestMLAPIIntegrationFixtureContextProperties(unittest.TestCase):
    """Property-based tests for ML API Integration with Fixture Context."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_neo4j_conn = Mock()
        self.mock_query_executor = Mock()
        self.mock_fixture_service = Mock(spec=FixtureService)
        
        # Create ML API integration with mocked fixture service
        self.ml_integration = MLAPIIntegration(
            self.mock_neo4j_conn,
            self.mock_query_executor,
            self.mock_fixture_service
        )
        
        # Mock predictor as loaded
        self.mock_predictor = Mock()
        self.ml_integration.predictor = self.mock_predictor
        self.ml_integration.predictor_loaded = True
    
    # Property 8: Prediction Response Fixture Context
    # **Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.2**
    @given(
        player_name=valid_player_name(),
        fixture_ctx=valid_fixture_context(),
        player_data=valid_player_data()
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    def test_property_8_prediction_response_includes_fixture_context(
        self, player_name, fixture_ctx, player_data
    ):
        """
        Property 8: Prediction Response Fixture Context
        
        For any player with an upcoming fixture, the prediction response SHALL 
        include fixture_context with fields: opponent_team, home_away, match_date, 
        matchday, fixture_id.
        
        **Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.2**
        """
        # Reset mocks for each hypothesis example
        self.mock_fixture_service.reset_mock()
        self.mock_neo4j_conn.reset_mock()
        self.mock_predictor.reset_mock()
        
        # Mock fixture service returns fixture context
        self.mock_fixture_service.get_next_match.return_value = fixture_ctx
        
        # Mock player data query
        player_data["name"] = player_name
        self.mock_neo4j_conn.execute_query.return_value = [player_data]
        
        # Mock predictor response
        mock_prediction = PlayerPrediction(
            player_name=player_name,
            predicted_points=7.5,
            features_used={"form": 5.0},
            model_version="xgboost_v1"
        )
        self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
        
        # Create request
        request = PlayerPredictionRequest(player_name=player_name)
        
        # Get prediction (run async function synchronously)
        response = run_async(self.ml_integration.predict_player_next_gameweek(request))
        
        # Verify fixture_context is included
        self.assertIsNotNone(response.fixture_context, "fixture_context should be present")
        
        # Verify all required fields are present
        self.assertIn("opponent_team", response.fixture_context)
        self.assertIn("home_away", response.fixture_context)
        self.assertIn("match_date", response.fixture_context)
        self.assertIn("matchday", response.fixture_context)
        self.assertIn("fixture_id", response.fixture_context)
        
        # Verify field values match fixture context
        self.assertEqual(response.fixture_context["opponent_team"], fixture_ctx.opponent_team)
        self.assertEqual(response.fixture_context["home_away"], fixture_ctx.home_away)
        self.assertEqual(response.fixture_context["matchday"], fixture_ctx.matchday)
        self.assertEqual(response.fixture_context["fixture_id"], fixture_ctx.fixture_id)
        
        # Verify fixture service was called
        self.mock_fixture_service.get_next_match.assert_called_once_with(
            player_name,
            current_date=unittest.mock.ANY
        )
    
    # Property 9: Prediction Features Include Fixture Context
    # **Validates: Requirements 3.4, 8.3**
    @given(
        player_name=valid_player_name(),
        fixture_ctx=valid_fixture_context(),
        player_data=valid_player_data()
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
    def test_property_9_prediction_features_include_fixture_context(
        self, player_name, fixture_ctx, player_data
    ):
        """
        Property 9: Prediction Features Include Fixture Context
        
        For any player with an upcoming fixture, the player_data passed to the 
        ML predictor SHALL include opponent_team and was_home fields derived 
        from fixture context.
        
        **Validates: Requirements 3.4, 8.3**
        """
        # Reset mocks for each hypothesis example
        self.mock_fixture_service.reset_mock()
        self.mock_neo4j_conn.reset_mock()
        self.mock_predictor.reset_mock()
        
        # Mock fixture service returns fixture context
        self.mock_fixture_service.get_next_match.return_value = fixture_ctx
        
        # Mock player data query
        player_data["name"] = player_name
        self.mock_neo4j_conn.execute_query.return_value = [player_data]
        
        # Mock predictor response
        mock_prediction = PlayerPrediction(
            player_name=player_name,
            predicted_points=7.5,
            features_used={},
            model_version="xgboost_v1"
        )
        self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
        
        # Create request
        request = PlayerPredictionRequest(player_name=player_name)
        
        # Get prediction (run async function synchronously)
        run_async(self.ml_integration.predict_player_next_gameweek(request))
        
        # Verify predictor was called with enriched data
        self.mock_predictor.predict_next_gameweek.assert_called_once()
        
        # Get the player_data that was passed to predictor
        call_args = self.mock_predictor.predict_next_gameweek.call_args
        enriched_player_data = call_args[0][0]
        
        # Verify fixture context fields are present
        self.assertIn("opponent_team", enriched_player_data)
        self.assertIn("was_home", enriched_player_data)
        self.assertIn("fixture_id", enriched_player_data)
        self.assertIn("matchday", enriched_player_data)
        
        # Verify opponent_team matches
        self.assertEqual(enriched_player_data["opponent_team"], fixture_ctx.opponent_team)
        
        # Verify was_home is correctly set based on home_away
        expected_was_home = 1 if fixture_ctx.home_away == "home" else 0
        self.assertEqual(enriched_player_data["was_home"], expected_was_home)
        
        # Verify fixture_id and matchday match
        self.assertEqual(enriched_player_data["fixture_id"], fixture_ctx.fixture_id)
        self.assertEqual(enriched_player_data["matchday"], fixture_ctx.matchday)


class TestMLAPIIntegrationNullFixtureHandling(unittest.TestCase):
    """Unit tests for null fixture handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_neo4j_conn = Mock()
        self.mock_query_executor = Mock()
        self.mock_fixture_service = Mock(spec=FixtureService)
        
        # Create ML API integration with mocked fixture service
        self.ml_integration = MLAPIIntegration(
            self.mock_neo4j_conn,
            self.mock_query_executor,
            self.mock_fixture_service
        )
        
        # Mock predictor as loaded
        self.mock_predictor = Mock()
        self.ml_integration.predictor = self.mock_predictor
        self.ml_integration.predictor_loaded = True
    
    def test_prediction_when_player_has_no_team(self):
        """
        Test prediction when player has no team (no PLAYS_FOR relationship).
        
        The system should return a prediction with null fixture_context.
        
        **Validates: Requirements 3.5, 4.3**
        """
        player_name = "Test Player"
        
        # Mock fixture service returns None (no team = no fixtures)
        self.mock_fixture_service.get_next_match.return_value = None
        
        # Mock player data query
        player_data = {
            "name": player_name,
            "position": "MID",
            "minutes": 80.0,
            "form": 6.0,
            "value": 85.0
        }
        self.mock_neo4j_conn.execute_query.return_value = [player_data]
        
        # Mock predictor response
        mock_prediction = PlayerPrediction(
            player_name=player_name,
            predicted_points=5.0,
            features_used={},
            model_version="xgboost_v1"
        )
        self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
        
        # Create request
        request = PlayerPredictionRequest(player_name=player_name)
        
        # Get prediction (run async function synchronously)
        response = run_async(self.ml_integration.predict_player_next_gameweek(request))
        
        # Verify response includes null fixture_context
        self.assertIsNone(response.fixture_context, "fixture_context should be None when player has no team")
        
        # Verify prediction still succeeds
        self.assertEqual(response.player_name, player_name)
        self.assertEqual(response.predicted_points, 5.0)
        
        # Verify player_data passed to predictor has default fixture values
        call_args = self.mock_predictor.predict_next_gameweek.call_args
        enriched_player_data = call_args[0][0]
        
        self.assertEqual(enriched_player_data["opponent_team"], None)
        self.assertEqual(enriched_player_data["was_home"], 0)
        self.assertEqual(enriched_player_data["fixture_id"], None)
        self.assertEqual(enriched_player_data["matchday"], None)
    
    def test_prediction_when_team_has_no_upcoming_fixtures(self):
        """
        Test prediction when team has no upcoming fixtures (end of season).
        
        The system should return a prediction with null fixture_context.
        
        **Validates: Requirements 3.5, 4.3**
        """
        player_name = "End Season Player"
        
        # Mock fixture service returns None (no upcoming fixtures)
        self.mock_fixture_service.get_next_match.return_value = None
        
        # Mock player data query
        player_data = {
            "name": player_name,
            "position": "FWD",
            "minutes": 90.0,
            "form": 8.0,
            "value": 110.0
        }
        self.mock_neo4j_conn.execute_query.return_value = [player_data]
        
        # Mock predictor response
        mock_prediction = PlayerPrediction(
            player_name=player_name,
            predicted_points=7.0,
            features_used={},
            model_version="xgboost_v1"
        )
        self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
        
        # Create request
        request = PlayerPredictionRequest(player_name=player_name)
        
        # Get prediction (run async function synchronously)
        response = run_async(self.ml_integration.predict_player_next_gameweek(request))
        
        # Verify response includes null fixture_context
        self.assertIsNone(response.fixture_context, "fixture_context should be None when no upcoming fixtures")
        
        # Verify prediction still succeeds
        self.assertEqual(response.player_name, player_name)
        self.assertEqual(response.predicted_points, 7.0)
    
    def test_prediction_graceful_degradation_on_fixture_service_error(self):
        """
        Test graceful degradation when FixtureService raises an error.
        
        The system should log a warning and continue with prediction 
        without fixture context.
        
        **Validates: Requirements 3.5, 4.3**
        """
        player_name = "Error Test Player"
        
        # Mock fixture service raises an exception
        self.mock_fixture_service.get_next_match.side_effect = Exception("Database connection error")
        
        # Mock player data query
        player_data = {
            "name": player_name,
            "position": "DEF",
            "minutes": 85.0,
            "form": 5.5,
            "value": 65.0
        }
        self.mock_neo4j_conn.execute_query.return_value = [player_data]
        
        # Mock predictor response
        mock_prediction = PlayerPrediction(
            player_name=player_name,
            predicted_points=4.5,
            features_used={},
            model_version="xgboost_v1"
        )
        self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
        
        # Create request
        request = PlayerPredictionRequest(player_name=player_name)
        
        # Get prediction - should not raise exception (run async function synchronously)
        response = run_async(self.ml_integration.predict_player_next_gameweek(request))
        
        # Verify response includes null fixture_context (graceful degradation)
        self.assertIsNone(response.fixture_context, "fixture_context should be None on error")
        
        # Verify prediction still succeeds
        self.assertEqual(response.player_name, player_name)
        self.assertEqual(response.predicted_points, 4.5)
    
    def test_prediction_with_provided_player_data_enriched_with_fixture(self):
        """
        Test that when player_data is provided in request, it still gets 
        enriched with fixture context.
        
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        player_name = "Provided Data Player"
        
        # Mock fixture context
        fixture_ctx = FixtureContext(
            opponent_team="Chelsea",
            home_away="home",
            match_date=datetime.now(timezone.utc) + timedelta(days=5),
            matchday=15,
            fixture_id="fixture-2026-27-md15-test"
        )
        self.mock_fixture_service.get_next_match.return_value = fixture_ctx
        
        # Provide player_data in request
        provided_player_data = {
            "name": player_name,
            "position": "MID",
            "minutes": 75.0,
            "form": 6.5,
            "value": 90.0
        }
        
        # Mock predictor response
        mock_prediction = PlayerPrediction(
            player_name=player_name,
            predicted_points=6.8,
            features_used={},
            model_version="xgboost_v1"
        )
        self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
        
        # Create request with provided player_data
        request = PlayerPredictionRequest(
            player_name=player_name,
            player_data=provided_player_data
        )
        
        # Get prediction (run async function synchronously)
        response = run_async(self.ml_integration.predict_player_next_gameweek(request))
        
        # Verify fixture_context is included
        self.assertIsNotNone(response.fixture_context)
        self.assertEqual(response.fixture_context["opponent_team"], "Chelsea")
        self.assertEqual(response.fixture_context["home_away"], "home")
        
        # Verify player_data passed to predictor has fixture enrichment
        call_args = self.mock_predictor.predict_next_gameweek.call_args
        enriched_player_data = call_args[0][0]
        
        self.assertEqual(enriched_player_data["opponent_team"], "Chelsea")
        self.assertEqual(enriched_player_data["was_home"], 1)
        self.assertEqual(enriched_player_data["fixture_id"], "fixture-2026-27-md15-test")
    
    def test_fixture_service_dependency_injection(self):
        """
        Test that FixtureService can be injected or created automatically.
        
        **Validates: Architecture pattern**
        """
        # Test with explicit fixture_service
        explicit_fixture_service = Mock(spec=FixtureService)
        ml_integration = MLAPIIntegration(
            self.mock_neo4j_conn,
            self.mock_query_executor,
            explicit_fixture_service
        )
        
        # Verify the explicit fixture service is used
        self.assertIs(ml_integration.fixture_service, explicit_fixture_service)
        
        # Test with None (should create FixtureService automatically)
        with patch('ml.api_integration.FixtureService') as mock_fixture_service_class:
            mock_auto_service = Mock()
            mock_fixture_service_class.return_value = mock_auto_service
            
            ml_integration_auto = MLAPIIntegration(
                self.mock_neo4j_conn,
                self.mock_query_executor,
                None
            )
            
            # Verify FixtureService was created with neo4j_conn
            mock_fixture_service_class.assert_called_once_with(self.mock_neo4j_conn)
            self.assertIs(ml_integration_auto.fixture_service, mock_auto_service)


if __name__ == '__main__':
    # Run tests
    unittest.main()
