"""
Unit tests for MLAPIIntegration caching functionality
"""
import unittest
import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, AsyncMock, patch

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml.api_integration import MLAPIIntegration, PlayerPredictionRequest, PredictionResponse
from services.fixture_service import FixtureService, FixtureContext
from services.cache import PredictionCache
from ml.predictor import FPLPredictor, PlayerPrediction


class TestMLAPIIntegrationCache(unittest.IsolatedAsyncioTestCase):
    """Test cases for MLAPIIntegration caching integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_conn = Mock()
        self.mock_query_executor = Mock()
        self.mock_fixture_service = Mock(spec=FixtureService)
        self.mock_cache = Mock(spec=PredictionCache)
        
        # Create MLAPIIntegration with mocked dependencies
        self.ml_integration = MLAPIIntegration(
            neo4j_conn=self.mock_conn,
            query_executor=self.mock_query_executor,
            fixture_service=self.mock_fixture_service,
            cache=self.mock_cache
        )
        
        # Mock predictor
        self.mock_predictor = Mock(spec=FPLPredictor)
        self.ml_integration.predictor = self.mock_predictor
        self.ml_integration.predictor_loaded = True
    
    def test_initialization_with_cache(self):
        """Test MLAPIIntegration initializes with cache correctly."""
        self.assertIsNotNone(self.ml_integration.cache)
        self.assertEqual(self.ml_integration.cache, self.mock_cache)
    
    def test_initialization_creates_cache_if_not_provided(self):
        """Test MLAPIIntegration creates cache if not provided."""
        ml_integration = MLAPIIntegration(
            neo4j_conn=self.mock_conn,
            query_executor=self.mock_query_executor
        )
        
        self.assertIsNotNone(ml_integration.cache)
        self.assertIsInstance(ml_integration.cache, PredictionCache)
    
    @patch('ml.api_integration.datetime')
    async def test_predict_checks_cache_before_prediction(self, mock_datetime):
        """Test that predict_player_next_gameweek checks cache before generating prediction."""
        # Setup
        now = datetime.now(timezone.utc)
        mock_datetime.now.return_value = now
        
        # Mock fixture context (relative to now: PredictionCache.calculate_cache_ttl uses the real clock)
        match_date = now + timedelta(days=3)
        fixture_context = FixtureContext(
            opponent_team="Manchester United",
            home_away="away",
            match_date=match_date,
            matchday=2,
            fixture_id="fixture-2026-27-md2-liv-mun"
        )
        self.mock_fixture_service.get_next_match.return_value = fixture_context
        
        # Mock cached prediction (cache HIT)
        cached_response = PredictionResponse(
            player_name="Mohamed Salah",
            predicted_points=7.5,
            features_used={},
            model_version="xgboost_v1",
            fixture_context=fixture_context.to_dict()
        )
        self.mock_cache.get.return_value = cached_response
        
        # Create request
        request = PlayerPredictionRequest(player_name="Mohamed Salah")
        
        # Call method
        result = await self.ml_integration.predict_player_next_gameweek(request)
        
        # Assertions
        self.assertEqual(result, cached_response)
        
        # Verify cache was checked with correct key
        self.mock_cache.get.assert_called_once()
        call_args = self.mock_cache.get.call_args[0]
        cache_key = call_args[0]
        self.assertIn("prediction:", cache_key)
        self.assertIn("Mohamed Salah", cache_key)
        self.assertIn("fixture-2026-27-md2-liv-mun", cache_key)
        
        # Verify predictor was NOT called (cache hit)
        self.mock_predictor.predict_next_gameweek.assert_not_called()
    
    @patch('ml.api_integration.datetime')
    async def test_predict_stores_in_cache_on_miss(self, mock_datetime):
        """Test that prediction is stored in cache on cache miss."""
        # Setup
        now = datetime.now(timezone.utc)
        mock_datetime.now.return_value = now
        
        # Mock fixture context (relative to now: PredictionCache.calculate_cache_ttl uses the real clock)
        match_date = now + timedelta(days=3)
        fixture_context = FixtureContext(
            opponent_team="Manchester United",
            home_away="away",
            match_date=match_date,
            matchday=2,
            fixture_id="fixture-2026-27-md2-liv-mun"
        )
        self.mock_fixture_service.get_next_match.return_value = fixture_context
        
        # Mock cache miss
        self.mock_cache.get.return_value = None
        
        # Mock player data query
        self.ml_integration.get_player_data_for_prediction = AsyncMock(return_value={
            "name": "Mohamed Salah",
            "position": "FWD",
            "minutes": 90.0,
            "goals_scored": 1.0,
            "assists": 0.5,
            "form": 7.0
        })
        
        # Mock predictor response
        mock_prediction = PlayerPrediction(
            player_name="Mohamed Salah",
            predicted_points=7.2,
            features_used={},
            model_version="xgboost_fwd_v3"
        )
        self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
        
        # Create request
        request = PlayerPredictionRequest(player_name="Mohamed Salah")
        
        # Call method
        result = await self.ml_integration.predict_player_next_gameweek(request)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.player_name, "Mohamed Salah")
        self.assertEqual(result.predicted_points, 7.2)
        
        # Verify cache.set was called
        self.mock_cache.set.assert_called_once()
        set_call_args = self.mock_cache.set.call_args
        cache_key = set_call_args[0][0]
        cached_value = set_call_args[0][1]
        ttl = set_call_args[0][2]
        
        # Verify cache key format
        self.assertIn("prediction:", cache_key)
        self.assertIn("Mohamed Salah", cache_key)
        self.assertIn("fixture-2026-27-md2-liv-mun", cache_key)
        
        # Verify TTL is calculated (3 days away = 6 hours = 21600 seconds)
        self.assertEqual(ttl, 21600)
        
        # Verify cached value is the response
        self.assertIsInstance(cached_value, PredictionResponse)
    
    @patch('ml.api_integration.datetime')
    async def test_predict_skips_cache_when_no_fixture_context(self, mock_datetime):
        """Test that caching is skipped when no fixture context is available."""
        # Setup
        mock_datetime.now.return_value = datetime.now(timezone.utc)
        
        # Mock no fixture context
        self.mock_fixture_service.get_next_match.return_value = None
        
        # Mock player data query
        self.ml_integration.get_player_data_for_prediction = AsyncMock(return_value={
            "name": "Mohamed Salah",
            "position": "FWD",
            "minutes": 90.0,
            "goals_scored": 1.0,
            "assists": 0.5,
            "form": 7.0
        })
        
        # Mock predictor response
        mock_prediction = PlayerPrediction(
            player_name="Mohamed Salah",
            predicted_points=7.2,
            features_used={},
            model_version="xgboost_fwd_v3"
        )
        self.mock_predictor.predict_next_gameweek.return_value = mock_prediction
        
        # Create request
        request = PlayerPredictionRequest(player_name="Mohamed Salah")
        
        # Call method
        result = await self.ml_integration.predict_player_next_gameweek(request)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsNone(result.fixture_context)
        
        # Verify cache was NOT checked or set (no fixture_id)
        self.mock_cache.get.assert_not_called()
        self.mock_cache.set.assert_not_called()
    
    def test_cache_key_generation(self):
        """Test cache key generation format."""
        cache_key = PredictionCache.generate_prediction_key(
            player_id="Mohamed Salah",
            fixture_id="fixture-2026-27-md2-liv-mun",
            model_version="xgboost_v1"
        )
        
        expected_key = "prediction:Mohamed Salah:fixture-2026-27-md2-liv-mun:xgboost_v1"
        self.assertEqual(cache_key, expected_key)
    
    def test_cache_ttl_calculation_far_future(self):
        """Test cache TTL calculation for fixtures > 7 days away."""
        # Fixture 10 days away
        fixture_kickoff = datetime.now(timezone.utc).replace(microsecond=0) + \
                         __import__('datetime').timedelta(days=10)
        
        ttl = PredictionCache.calculate_cache_ttl(fixture_kickoff)
        
        # Should be 24 hours = 86400 seconds
        self.assertEqual(ttl, 86400)
    
    def test_cache_ttl_calculation_near_future(self):
        """Test cache TTL calculation for fixtures 1-7 days away."""
        # Fixture 3 days away
        fixture_kickoff = datetime.now(timezone.utc).replace(microsecond=0) + \
                         __import__('datetime').timedelta(days=3)
        
        ttl = PredictionCache.calculate_cache_ttl(fixture_kickoff)
        
        # Should be 6 hours = 21600 seconds
        self.assertEqual(ttl, 21600)
    
    def test_cache_ttl_calculation_very_near_future(self):
        """Test cache TTL calculation for fixtures < 1 day away."""
        # Fixture 12 hours away
        fixture_kickoff = datetime.now(timezone.utc).replace(microsecond=0) + \
                         __import__('datetime').timedelta(hours=12)
        
        ttl = PredictionCache.calculate_cache_ttl(fixture_kickoff)
        
        # Should be 1 hour = 3600 seconds
        self.assertEqual(ttl, 3600)
    
    def test_cache_ttl_calculation_past_fixture(self):
        """Test cache TTL calculation for past fixtures."""
        # Fixture in the past
        fixture_kickoff = datetime.now(timezone.utc) - __import__('datetime').timedelta(hours=1)
        
        ttl = PredictionCache.calculate_cache_ttl(fixture_kickoff)
        
        # Should be 0 (no caching)
        self.assertEqual(ttl, 0)


if __name__ == "__main__":
    unittest.main()
