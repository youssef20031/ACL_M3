"""
Unit tests for FixtureService
"""
import unittest
import sys
import os
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.fixture_service import FixtureService, FixtureContext, FixtureData
from graph.connection import Neo4jConnection


class TestFixtureService(unittest.TestCase):
    """Test cases for FixtureService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock Neo4j connection
        self.mock_conn = Mock(spec=Neo4jConnection)
        self.fixture_service = FixtureService(self.mock_conn)
    
    def test_fixture_service_initialization(self):
        """Test that FixtureService initializes correctly."""
        self.assertIsNotNone(self.fixture_service)
        self.assertEqual(self.fixture_service.conn, self.mock_conn)
    
    def test_fixture_context_creation(self):
        """Test FixtureContext dataclass creation."""
        match_date = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)
        context = FixtureContext(
            opponent_team="Manchester United",
            home_away="away",
            match_date=match_date,
            matchday=2,
            fixture_id="fixture-2026-27-md2-liv-mun"
        )
        
        self.assertEqual(context.opponent_team, "Manchester United")
        self.assertEqual(context.home_away, "away")
        self.assertEqual(context.matchday, 2)
        self.assertEqual(context.fixture_id, "fixture-2026-27-md2-liv-mun")
    
    def test_fixture_context_to_dict(self):
        """Test FixtureContext conversion to dictionary."""
        match_date = datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)
        context = FixtureContext(
            opponent_team="Manchester United",
            home_away="away",
            match_date=match_date,
            matchday=2,
            fixture_id="fixture-2026-27-md2-liv-mun"
        )
        
        context_dict = context.to_dict()
        
        self.assertIsInstance(context_dict, dict)
        self.assertEqual(context_dict["opponent_team"], "Manchester United")
        self.assertEqual(context_dict["home_away"], "away")
        self.assertIsInstance(context_dict["match_date"], str)  # Should be ISO format string
        self.assertEqual(context_dict["matchday"], 2)
    
    def test_get_next_match_returns_fixture_context(self):
        """Test get_next_match returns correct FixtureContext."""
        # Mock the query result
        mock_result = [{
            "fixture_id": "fixture-2026-27-md2-liv-mun",
            "match_date": datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc),
            "matchday": 2,
            "opponent_team": "Manchester United",
            "home_away": "away"
        }]
        self.mock_conn.execute_query.return_value = mock_result
        
        # Call method
        current_date = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
        result = self.fixture_service.get_next_match("Mohamed Salah", current_date)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, FixtureContext)
        self.assertEqual(result.opponent_team, "Manchester United")
        self.assertEqual(result.home_away, "away")
        self.assertEqual(result.matchday, 2)
        
        # Verify query was called
        self.mock_conn.execute_query.assert_called_once()
    
    def test_get_next_match_returns_none_when_no_fixtures(self):
        """Test get_next_match returns None when no upcoming fixtures."""
        # Mock empty result
        self.mock_conn.execute_query.return_value = []
        
        # Call method
        current_date = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
        result = self.fixture_service.get_next_match("Mohamed Salah", current_date)
        
        # Assertions
        self.assertIsNone(result)
    
    def test_get_next_match_uses_current_time_when_date_not_provided(self):
        """Test get_next_match uses current UTC time when current_date is None."""
        # Mock result
        mock_result = [{
            "fixture_id": "fixture-2026-27-md2-liv-mun",
            "match_date": datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc),
            "matchday": 2,
            "opponent_team": "Manchester United",
            "home_away": "away"
        }]
        self.mock_conn.execute_query.return_value = mock_result
        
        # Call method without current_date
        result = self.fixture_service.get_next_match("Mohamed Salah")
        
        # Should still work and return result
        self.assertIsNotNone(result)
        self.assertIsInstance(result, FixtureContext)
    
    def test_get_team_next_match_returns_fixture_context(self):
        """Test get_team_next_match returns correct FixtureContext."""
        # Mock the query result
        mock_result = [{
            "fixture_id": "fixture-2026-27-md1-liv-ars",
            "match_date": datetime(2026, 8, 15, 15, 0, tzinfo=timezone.utc),
            "matchday": 1,
            "opponent_team": "Arsenal",
            "home_away": "home"
        }]
        self.mock_conn.execute_query.return_value = mock_result
        
        # Call method
        current_date = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)
        result = self.fixture_service.get_team_next_match("Liverpool", current_date)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, FixtureContext)
        self.assertEqual(result.opponent_team, "Arsenal")
        self.assertEqual(result.home_away, "home")
        self.assertEqual(result.matchday, 1)
    
    def test_is_fixture_past_returns_true_for_past_fixture(self):
        """Test is_fixture_past returns True for past fixtures."""
        # Mock fixture with past kickoff time
        mock_result = [{
            "kickoff_time": datetime(2026, 8, 15, 15, 0, tzinfo=timezone.utc)
        }]
        self.mock_conn.execute_query.return_value = mock_result
        
        # Call with future current_date
        current_date = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
        result = self.fixture_service.is_fixture_past("fixture-2026-27-md1-liv-ars", current_date)
        
        # Should return True (fixture is in the past)
        self.assertTrue(result)
    
    def test_is_fixture_past_returns_false_for_future_fixture(self):
        """Test is_fixture_past returns False for future fixtures."""
        # Mock fixture with future kickoff time
        mock_result = [{
            "kickoff_time": datetime(2026, 8, 22, 15, 0, tzinfo=timezone.utc)
        }]
        self.mock_conn.execute_query.return_value = mock_result
        
        # Call with past current_date
        current_date = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
        result = self.fixture_service.is_fixture_past("fixture-2026-27-md2-liv-mun", current_date)
        
        # Should return False (fixture is in the future)
        self.assertFalse(result)
    
    def test_is_fixture_past_returns_false_for_nonexistent_fixture(self):
        """Test is_fixture_past returns False when fixture doesn't exist."""
        # Mock empty result
        self.mock_conn.execute_query.return_value = []
        
        # Call method
        current_date = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
        result = self.fixture_service.is_fixture_past("nonexistent-fixture", current_date)
        
        # Should return False (fixture not found)
        self.assertFalse(result)
    
    def test_update_fixture_returns_true_on_success(self):
        """Test update_fixture returns True on successful update."""
        # Mock fixture exists
        self.mock_conn.execute_query.return_value = [{"f": {}}]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Call method
        updates = {"status": "completed", "home_score": 2, "away_score": 1}
        result = self.fixture_service.update_fixture("fixture-2026-27-md1-liv-ars", updates)
        
        # Should return True
        self.assertTrue(result)
        self.mock_conn.execute_write.assert_called_once()
    
    def test_update_fixture_returns_false_when_fixture_not_found(self):
        """Test update_fixture returns False when fixture doesn't exist."""
        # Mock fixture not found
        self.mock_conn.execute_query.return_value = []
        
        # Call method
        updates = {"status": "completed"}
        result = self.fixture_service.update_fixture("nonexistent-fixture", updates)
        
        # Should return False
        self.assertFalse(result)
    
    def test_update_fixture_ignores_id_field(self):
        """Test update_fixture doesn't update id field."""
        # Mock fixture exists
        self.mock_conn.execute_query.return_value = [{"f": {}}]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Call method with id in updates (should be ignored)
        updates = {"id": "new-id", "status": "completed"}
        result = self.fixture_service.update_fixture("fixture-2026-27-md1-liv-ars", updates)
        
        # Verify execute_write was called
        self.mock_conn.execute_write.assert_called_once()
        
        # Get the actual query that was called
        call_args = self.mock_conn.execute_write.call_args
        query = call_args[0][0]
        
        # Verify 'id' is not in the SET clause
        self.assertNotIn("f.id =", query)
        self.assertIn("f.status =", query)


class TestFixtureServiceLoadCSV(unittest.TestCase):
    """Test cases for CSV loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_conn = Mock(spec=Neo4jConnection)
        self.fixture_service = FixtureService(self.mock_conn)
    
    @patch('services.fixture_service.pd.read_csv')
    def test_load_fixtures_from_csv_validates_columns(self, mock_read_csv):
        """Test that load_fixtures_from_csv validates required columns."""
        import pandas as pd
        
        # Mock CSV with missing columns
        mock_df = pd.DataFrame({
            "Matchday": [1],
            "Home Team": ["Liverpool"]
            # Missing "Away Team" and "Date"
        })
        mock_read_csv.return_value = mock_df
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.fixture_service.load_fixtures_from_csv("test.csv")
        
        self.assertIn("Missing required columns", str(context.exception))
    
    @patch('services.fixture_service.pd.read_csv')
    def test_load_fixtures_from_csv_validates_team_names(self, mock_read_csv):
        """Test that load_fixtures_from_csv validates team names."""
        import pandas as pd
        
        # Mock CSV with valid columns
        mock_df = pd.DataFrame({
            "Matchday": [1],
            "Home Team": ["Liverpool"],
            "Away Team": ["Arsenal"],
            "Date": ["15/08/2026"],
            "Time": ["15:00"]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock valid teams query - returns only Liverpool (Arsenal missing)
        self.mock_conn.execute_query.return_value = [{"name": "Liverpool"}]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Call method
        count = self.fixture_service.load_fixtures_from_csv("test.csv")
        
        # Should skip invalid fixture and return 0
        self.assertEqual(count, 0)
    
    @patch('services.fixture_service.pd.read_csv')
    def test_load_fixtures_from_csv_success(self, mock_read_csv):
        """Test successful CSV loading."""
        import pandas as pd
        
        # Mock CSV with valid data
        mock_df = pd.DataFrame({
            "Matchday": [1, 2],
            "Home Team": ["Liverpool", "Arsenal"],
            "Away Team": ["Arsenal", "Chelsea"],
            "Date": ["15/08/2026", "22/08/2026"],
            "Time": ["15:00", "17:30"]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock valid teams query - all teams exist
        self.mock_conn.execute_query.return_value = [
            {"name": "Liverpool"},
            {"name": "Arsenal"},
            {"name": "Chelsea"}
        ]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Call method
        count = self.fixture_service.load_fixtures_from_csv("test.csv", season_id="2026-27")
        
        # Should load both fixtures
        self.assertEqual(count, 2)
        
        # Verify execute_write was called twice (once per fixture)
        self.assertEqual(self.mock_conn.execute_write.call_count, 2)


if __name__ == "__main__":
    unittest.main()
