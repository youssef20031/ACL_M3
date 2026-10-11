"""
Unit Tests for Opponent Strength Calculation
Tests the calculate_opponent_strength() method in FixtureService.
"""
import unittest
import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.fixture_service import FixtureService, OpponentStrength
from graph.connection import Neo4jConnection


class TestOpponentStrength(unittest.TestCase):
    """Test cases for opponent strength calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_conn = Mock(spec=Neo4jConnection)
        self.fixture_service = FixtureService(self.mock_conn)
    
    def test_calculate_opponent_strength_with_sufficient_data(self):
        """Test opponent strength calculation with 5+ matches."""
        # Mock 5 completed fixtures
        mock_result = {
            "conceded_list": [2, 1, 0, 3, 1],  # Avg = 1.4
            "scored_list": [3, 2, 1, 2, 2],    # Avg = 2.0
            "matches_played": 5
        }
        self.mock_conn.execute_query.return_value = [mock_result]
        
        # Calculate strength
        result = self.fixture_service.calculate_opponent_strength("Liverpool")
        
        # Verify calculations
        self.assertAlmostEqual(result["defensive_strength"], 1.4, places=2)
        self.assertAlmostEqual(result["offensive_strength"], 2.0, places=2)
    
    def test_calculate_opponent_strength_with_fewer_matches(self):
        """Test opponent strength calculation with fewer than 5 matches."""
        # Mock only 2 completed fixtures
        mock_result = {
            "conceded_list": [1, 2],  # Avg = 1.5
            "scored_list": [2, 3],    # Avg = 2.5
            "matches_played": 2
        }
        self.mock_conn.execute_query.return_value = [mock_result]
        
        # Calculate strength
        result = self.fixture_service.calculate_opponent_strength("Newly Promoted Team")
        
        # Should use available matches
        self.assertAlmostEqual(result["defensive_strength"], 1.5, places=2)
        self.assertAlmostEqual(result["offensive_strength"], 2.5, places=2)
    
    def test_calculate_opponent_strength_no_data_returns_league_average(self):
        """Test that no historical data returns league average."""
        # Mock no completed fixtures
        mock_result = {
            "conceded_list": [],
            "scored_list": [],
            "matches_played": 0
        }
        self.mock_conn.execute_query.return_value = [mock_result]
        
        # Calculate strength
        result = self.fixture_service.calculate_opponent_strength("New Team")
        
        # Should return league average (1.5)
        self.assertEqual(result["defensive_strength"], 1.5)
        self.assertEqual(result["offensive_strength"], 1.5)
    
    def test_calculate_opponent_strength_empty_result_returns_league_average(self):
        """Test that empty query result returns league average."""
        # Mock empty result
        self.mock_conn.execute_query.return_value = []
        
        # Calculate strength
        result = self.fixture_service.calculate_opponent_strength("Unknown Team")
        
        # Should return league average (1.5)
        self.assertEqual(result["defensive_strength"], 1.5)
        self.assertEqual(result["offensive_strength"], 1.5)
    
    def test_calculate_opponent_strength_uses_correct_date(self):
        """Test that current_date parameter is used correctly."""
        mock_result = {
            "conceded_list": [1, 2],
            "scored_list": [2, 3],
            "matches_played": 2
        }
        self.mock_conn.execute_query.return_value = [mock_result]
        
        # Use specific date
        test_date = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
        result = self.fixture_service.calculate_opponent_strength(
            "Arsenal",
            current_date=test_date
        )
        
        # Verify query was called with correct date
        call_args = self.mock_conn.execute_query.call_args
        params = call_args[0][1]
        self.assertEqual(params["current_date"], test_date)
    
    def test_calculate_opponent_strength_custom_num_matches(self):
        """Test that num_matches parameter is respected."""
        mock_result = {
            "conceded_list": [1, 2, 1],
            "scored_list": [2, 3, 2],
            "matches_played": 3
        }
        self.mock_conn.execute_query.return_value = [mock_result]
        
        # Request last 3 matches instead of default 5
        result = self.fixture_service.calculate_opponent_strength(
            "Chelsea",
            num_matches=3
        )
        
        # Verify query was called with correct limit
        call_args = self.mock_conn.execute_query.call_args
        params = call_args[0][1]
        self.assertEqual(params["num_matches"], 3)
    
    def test_calculate_opponent_strength_error_returns_league_average(self):
        """Test that database errors return league average gracefully."""
        # Mock database error
        self.mock_conn.execute_query.side_effect = Exception("Database connection failed")
        
        # Calculate strength - should not raise exception
        result = self.fixture_service.calculate_opponent_strength("Error Team")
        
        # Should return league average (1.5)
        self.assertEqual(result["defensive_strength"], 1.5)
        self.assertEqual(result["offensive_strength"], 1.5)


if __name__ == "__main__":
    unittest.main()
