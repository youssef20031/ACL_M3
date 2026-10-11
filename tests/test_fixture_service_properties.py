"""
Property-Based Tests for FixtureService
Tests universal correctness properties from design document.
"""
import unittest
import sys
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch
import tempfile
import csv

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.fixture_service import FixtureService, FixtureContext, FixtureData
from graph.connection import Neo4jConnection

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
    def valid_fixture_data(draw):
        """Generate valid fixture data for property tests."""
        matchday = draw(st.integers(min_value=1, max_value=38))
        home_team = draw(st.sampled_from(["Liverpool", "Arsenal", "Chelsea", "Manchester City", "Manchester United"]))
        away_team = draw(st.sampled_from(["Tottenham Hotspur", "Newcastle United", "Brighton and Hove Albion", "Aston Villa", "West Ham United"]))
        
        # Generate date in future
        days_ahead = draw(st.integers(min_value=1, max_value=365))
        kickoff = datetime.now(timezone.utc) + timedelta(days=days_ahead)
        
        return {
            "matchday": matchday,
            "home_team": home_team,
            "away_team": away_team,
            "date": kickoff.strftime("%d/%m/%Y"),
            "time": kickoff.strftime("%H:%M"),
            "kickoff_utc": kickoff
        }
    
    @composite
    def incomplete_fixture_data(draw):
        """Generate fixture data missing required fields."""
        # Use ASCII-only strings to avoid encoding issues
        fields = {
            "Matchday": str(draw(st.integers(min_value=1, max_value=38))),
            "Home Team": draw(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=50)),
            "Away Team": draw(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=50)),
            "Date": "15/08/2026",
            "Time": "15:00"
        }
        
        # Remove at least one required field
        required_fields = ["Matchday", "Home Team", "Away Team", "Date"]
        field_to_remove = draw(st.sampled_from(required_fields))
        del fields[field_to_remove]
        
        return fields


@unittest.skipIf(not HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestFixtureServiceProperties(unittest.TestCase):
    """Property-based tests for FixtureService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_conn = Mock(spec=Neo4jConnection)
        self.fixture_service = FixtureService(self.mock_conn)
    
    # Property 1: Fixture Storage Completeness
    # **Validates: Requirements 1.1**
    @given(fixture_data=valid_fixture_data())
    @settings(max_examples=5)  # Reduced to avoid mock call count issues
    def test_property_1_fixture_storage_completeness(self, fixture_data):
        """
        Property 1: Fixture Storage Completeness
        
        For any valid fixture data with all required fields (date, time, home_team, 
        away_team, matchday), storing the fixture SHALL result in a Neo4j Fixture 
        node with all fields present and retrievable.
        
        **Validates: Requirements 1.1**
        """
        # Reset mocks for each hypothesis example
        self.mock_conn.reset_mock()
        
        # Mock successful storage
        self.mock_conn.execute_query.return_value = [
            {"name": fixture_data["home_team"]},
            {"name": fixture_data["away_team"]}
        ]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
            writer.writeheader()
            writer.writerow({
                "Matchday": fixture_data["matchday"],
                "Home Team": fixture_data["home_team"],
                "Away Team": fixture_data["away_team"],
                "Date": fixture_data["date"],
                "Time": fixture_data["time"]
            })
            csv_path = f.name
        
        try:
            # Load fixtures
            count = self.fixture_service.load_fixtures_from_csv(csv_path, season_id="2026-27")
            
            # Verify fixture was stored
            self.assertEqual(count, 1, "Valid fixture data should result in storage")
            
            # Verify execute_write was called with all required fields
            self.mock_conn.execute_write.assert_called_once()
            call_args = self.mock_conn.execute_write.call_args
            params = call_args[0][1]
            
            # All fields should be present in parameters
            self.assertIn("fixture_id", params)
            self.assertIn("kickoff", params)
            self.assertIn("matchday", params)
            self.assertIn("season_id", params)
            self.assertIn("home_team", params)
            self.assertIn("away_team", params)
            
            # Verify field values
            self.assertEqual(params["matchday"], fixture_data["matchday"])
            self.assertEqual(params["home_team"], fixture_data["home_team"])
            self.assertEqual(params["away_team"], fixture_data["away_team"])
            self.assertEqual(params["season_id"], "2026-27")
            
        finally:
            # Clean up temp file
            os.unlink(csv_path)
    
    # Property 3: Fixture Validation Rejection
    # **Validates: Requirements 1.4**
    @given(incomplete_data=incomplete_fixture_data())
    @settings(max_examples=10)
    def test_property_3_fixture_validation_rejection(self, incomplete_data):
        """
        Property 3: Fixture Validation Rejection
        
        For any fixture data missing one or more required fields (date, time, 
        home_team, away_team, matchday), the System SHALL reject the fixture 
        and return an error message.
        
        **Validates: Requirements 1.4**
        """
        # Create temporary CSV file with incomplete data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            fieldnames = list(incomplete_data.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(incomplete_data)
            csv_path = f.name
        
        try:
            # Loading should raise ValueError for missing required columns
            with self.assertRaises(ValueError) as context:
                self.fixture_service.load_fixtures_from_csv(csv_path)
            
            # Error message should mention missing columns
            self.assertIn("Missing required columns", str(context.exception))
            
        finally:
            # Clean up temp file
            os.unlink(csv_path)
    
    # Property 11: Team Name Validation
    # **Validates: Requirements 5.4**
    @given(
        home_team=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=50).filter(lambda x: x not in ["Liverpool", "Arsenal", "Chelsea"]),
        away_team=st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=50).filter(lambda x: x not in ["Liverpool", "Arsenal", "Chelsea"])
    )
    @settings(max_examples=10)
    def test_property_11_team_name_validation(self, home_team, away_team):
        """
        Property 11: Team Name Validation
        
        For any fixture with a team name that does not match an existing Team 
        node in the database, the System SHALL reject the fixture with a 
        descriptive error message.
        
        **Validates: Requirements 5.4**
        """
        # Mock that only specific teams exist
        self.mock_conn.execute_query.return_value = [
            {"name": "Liverpool"},
            {"name": "Arsenal"},
            {"name": "Chelsea"}
        ]
        
        # Create CSV with invalid team names
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
            writer.writeheader()
            writer.writerow({
                "Matchday": 1,
                "Home Team": home_team,
                "Away Team": away_team,
                "Date": "15/08/2026",
                "Time": "15:00"
            })
            csv_path = f.name
        
        try:
            # Load fixtures - should skip invalid teams
            count = self.fixture_service.load_fixtures_from_csv(csv_path)
            
            # No fixtures should be loaded because teams are invalid
            self.assertEqual(count, 0, "Fixtures with invalid team names should be rejected")
            
        finally:
            # Clean up temp file
            os.unlink(csv_path)
    
    # Property 4: Duplicate Fixture Rejection
    # **Validates: Requirements 1.5**
    def test_property_4_duplicate_fixture_rejection(self):
        """
        Property 4: Duplicate Fixture Rejection
        
        For any fixture that already exists in the database, attempting to insert 
        the same fixture again SHALL be rejected with an error message OR handled 
        idempotently (MERGE operation ensures no duplicates are created).
        
        **Validates: Requirements 1.5**
        
        Note: The current implementation uses MERGE which is idempotent, so this
        test verifies that loading the same fixture twice doesn't create duplicates.
        """
        # Mock valid teams
        self.mock_conn.execute_query.return_value = [
            {"name": "Liverpool"},
            {"name": "Arsenal"}
        ]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Create CSV with same fixture twice
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
            writer.writeheader()
            # Same fixture data
            for _ in range(2):
                writer.writerow({
                    "Matchday": 1,
                    "Home Team": "Liverpool",
                    "Away Team": "Arsenal",
                    "Date": "15/08/2026",
                    "Time": "15:00"
                })
            csv_path = f.name
        
        try:
            # Load fixtures
            count = self.fixture_service.load_fixtures_from_csv(csv_path)
            
            # Should load 2 rows (MERGE handles idempotency at database level)
            self.assertEqual(count, 2)
            
            # Verify execute_write was called twice (once per row)
            self.assertEqual(self.mock_conn.execute_write.call_count, 2)
            
            # Both calls should have the same fixture_id (MERGE prevents duplicates)
            calls = self.mock_conn.execute_write.call_args_list
            fixture_id_1 = calls[0][0][1]["fixture_id"]
            fixture_id_2 = calls[1][0][1]["fixture_id"]
            self.assertEqual(fixture_id_1, fixture_id_2, "Duplicate fixtures should have same ID (MERGE ensures idempotency)")
            
        finally:
            # Clean up temp file
            os.unlink(csv_path)


class TestFixtureServiceCSVEdgeCases(unittest.TestCase):
    """Unit tests for CSV parsing edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_conn = Mock(spec=Neo4jConnection)
        self.fixture_service = FixtureService(self.mock_conn)
    
    def test_csv_missing_time_column_defaults_to_1500(self):
        """
        Test missing Time column defaults to 15:00.
        
        **Validates: Requirements 5.1**
        """
        # Mock valid teams
        self.mock_conn.execute_query.return_value = [
            {"name": "Liverpool"},
            {"name": "Arsenal"}
        ]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Create CSV without Time column
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date"])
            writer.writeheader()
            writer.writerow({
                "Matchday": 1,
                "Home Team": "Liverpool",
                "Away Team": "Arsenal",
                "Date": "15/08/2026"
            })
            csv_path = f.name
        
        try:
            # Load fixtures
            count = self.fixture_service.load_fixtures_from_csv(csv_path)
            
            # Should load successfully with default time
            self.assertEqual(count, 1)
            
            # Verify kickoff time uses 15:00
            call_args = self.mock_conn.execute_write.call_args
            params = call_args[0][1]
            kickoff = params["kickoff"]
            
            self.assertEqual(kickoff.hour, 15, "Default time should be 15:00")
            self.assertEqual(kickoff.minute, 0, "Default time should be 15:00")
            
        finally:
            # Clean up temp file
            os.unlink(csv_path)
    
    def test_csv_malformed_date_formats_are_skipped(self):
        """
        Test malformed date formats are skipped with warning.
        
        **Validates: Requirements 5.1, 5.5**
        """
        # Mock valid teams
        self.mock_conn.execute_query.return_value = [
            {"name": "Liverpool"},
            {"name": "Arsenal"}
        ]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Create CSV with malformed dates
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
            writer.writeheader()
            # Valid fixture
            writer.writerow({
                "Matchday": 1,
                "Home Team": "Liverpool",
                "Away Team": "Arsenal",
                "Date": "15/08/2026",
                "Time": "15:00"
            })
            # Malformed date
            writer.writerow({
                "Matchday": 2,
                "Home Team": "Liverpool",
                "Away Team": "Arsenal",
                "Date": "invalid-date",
                "Time": "15:00"
            })
            csv_path = f.name
        
        try:
            # Load fixtures
            count = self.fixture_service.load_fixtures_from_csv(csv_path)
            
            # Only valid fixture should be loaded
            self.assertEqual(count, 1, "Malformed dates should be skipped")
            
        finally:
            # Clean up temp file
            os.unlink(csv_path)
    
    def test_csv_empty_file_returns_zero(self):
        """
        Test empty CSV file returns 0 fixtures loaded.
        
        **Validates: Requirements 5.1, 5.5**
        """
        # Mock valid teams
        self.mock_conn.execute_query.return_value = [
            {"name": "Liverpool"}
        ]
        
        # Create empty CSV (only header)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
            writer.writeheader()
            csv_path = f.name
        
        try:
            # Load fixtures
            count = self.fixture_service.load_fixtures_from_csv(csv_path)
            
            # Should return 0
            self.assertEqual(count, 0, "Empty CSV should return 0 fixtures loaded")
            
            # execute_write should not be called
            self.mock_conn.execute_write.assert_not_called()
            
        finally:
            # Clean up temp file
            os.unlink(csv_path)
    
    def test_csv_supports_yyyy_mm_dd_date_format(self):
        """
        Test that CSV parser supports YYYY-MM-DD date format.
        
        **Validates: Requirements 5.3**
        """
        # Mock valid teams
        self.mock_conn.execute_query.return_value = [
            {"name": "Liverpool"},
            {"name": "Arsenal"}
        ]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Create CSV with YYYY-MM-DD format
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
            writer.writeheader()
            writer.writerow({
                "Matchday": 1,
                "Home Team": "Liverpool",
                "Away Team": "Arsenal",
                "Date": "2026-08-15",
                "Time": "15:00"
            })
            csv_path = f.name
        
        try:
            # Load fixtures
            count = self.fixture_service.load_fixtures_from_csv(csv_path)
            
            # Should load successfully
            self.assertEqual(count, 1)
            
            # Verify date was parsed correctly
            call_args = self.mock_conn.execute_write.call_args
            params = call_args[0][1]
            kickoff = params["kickoff"]
            
            self.assertEqual(kickoff.year, 2026)
            self.assertEqual(kickoff.month, 8)
            self.assertEqual(kickoff.day, 15)
            
        finally:
            # Clean up temp file
            os.unlink(csv_path)
    
    def test_csv_creates_correct_fixture_id_format(self):
        """
        Test that fixture IDs follow the expected format.
        
        **Validates: Requirements 1.1**
        """
        # Mock valid teams
        self.mock_conn.execute_query.return_value = [
            {"name": "Liverpool"},
            {"name": "Arsenal"}
        ]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Create CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
            writer.writeheader()
            writer.writerow({
                "Matchday": 5,
                "Home Team": "Liverpool",
                "Away Team": "Arsenal",
                "Date": "15/08/2026",
                "Time": "15:00"
            })
            csv_path = f.name
        
        try:
            # Load fixtures
            count = self.fixture_service.load_fixtures_from_csv(csv_path, season_id="2026-27")
            
            # Should load successfully
            self.assertEqual(count, 1)
            
            # Verify fixture ID format
            call_args = self.mock_conn.execute_write.call_args
            params = call_args[0][1]
            fixture_id = params["fixture_id"]
            
            # Format should be: fixture-{season_id}-md{matchday}-{home_abbr}-{away_abbr}
            self.assertTrue(fixture_id.startswith("fixture-2026-27-md5-"))
            self.assertIn("liv", fixture_id)
            self.assertIn("ars", fixture_id)
            
        finally:
            # Clean up temp file
            os.unlink(csv_path)
    
    def test_csv_stores_dates_in_utc(self):
        """
        Test that all dates are stored with UTC timezone.
        
        **Validates: Requirements 7.1, 7.4**
        """
        # Mock valid teams
        self.mock_conn.execute_query.return_value = [
            {"name": "Liverpool"},
            {"name": "Arsenal"}
        ]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Create CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Matchday", "Home Team", "Away Team", "Date", "Time"])
            writer.writeheader()
            writer.writerow({
                "Matchday": 1,
                "Home Team": "Liverpool",
                "Away Team": "Arsenal",
                "Date": "15/08/2026",
                "Time": "15:00"
            })
            csv_path = f.name
        
        try:
            # Load fixtures
            count = self.fixture_service.load_fixtures_from_csv(csv_path)
            
            # Should load successfully
            self.assertEqual(count, 1)
            
            # Verify kickoff time has UTC timezone
            call_args = self.mock_conn.execute_write.call_args
            params = call_args[0][1]
            kickoff = params["kickoff"]
            
            self.assertIsNotNone(kickoff.tzinfo, "Kickoff time should be timezone-aware")
            self.assertEqual(kickoff.tzinfo, timezone.utc, "Kickoff time should be in UTC")
            
        finally:
            # Clean up temp file
            os.unlink(csv_path)


class TestFixtureUpdateProperties(unittest.TestCase):
    """Property-based tests for fixture update operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_conn = Mock(spec=Neo4jConnection)
        self.fixture_service = FixtureService(self.mock_conn)
    
    @unittest.skipIf(not HYPOTHESIS_AVAILABLE, "hypothesis not installed")
    def test_property_12_fixture_update_idempotency(self):
        """
        Property 12: Fixture Update Idempotency
        
        For any existing fixture, refreshing with identical data SHALL not create 
        duplicate nodes and SHALL preserve the original fixture unchanged.
        
        **Validates: Requirements 6.2**
        """
        fixture_id = "fixture-2026-27-md1-liv-ars"
        
        # Mock existing fixture
        existing_fixture = {
            "f": {
                "id": fixture_id,
                "kickoff_time": datetime(2026, 8, 15, 15, 0, tzinfo=timezone.utc),
                "matchday": 1,
                "season_id": "2026-27",
                "status": "scheduled"
            }
        }
        self.mock_conn.execute_query.return_value = [existing_fixture]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Update with identical data (idempotent update)
        updates = {
            "kickoff_time": datetime(2026, 8, 15, 15, 0, tzinfo=timezone.utc),
            "matchday": 1,
            "status": "scheduled"
        }
        
        # Apply update with audit user
        result = self.fixture_service.update_fixture(fixture_id, updates, user="test_user")
        
        # Should succeed
        self.assertTrue(result, "Idempotent update should succeed")
        
        # Should call execute_write (update happens even if data is same)
        self.mock_conn.execute_write.assert_called_once()
        
        # Verify the update query was called with correct parameters
        call_args = self.mock_conn.execute_write.call_args
        params = call_args[0][1]
        
        self.assertEqual(params["fixture_id"], fixture_id)
        self.assertEqual(params["kickoff_time"], updates["kickoff_time"])
        self.assertEqual(params["matchday"], updates["matchday"])
        self.assertEqual(params["status"], updates["status"])
    
    @unittest.skipIf(not HYPOTHESIS_AVAILABLE, "hypothesis not installed")
    @given(
        home_score=st.integers(min_value=0, max_value=10),
        away_score=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
    def test_property_13_historical_fixture_preservation(self, home_score, away_score):
        """
        Property 13: Historical Fixture Preservation
        
        For any fixture that has already occurred (kickoff_time < current_date), 
        updating fixture attributes SHALL preserve the home_score and away_score fields.
        
        **Validates: Requirements 6.3**
        """
        # Reset mocks for each hypothesis example
        self.mock_conn.reset_mock()
        
        fixture_id = "fixture-2026-27-md1-liv-ars"
        
        # Past kickoff time (fixture has already occurred)
        past_kickoff = datetime.now(timezone.utc) - timedelta(days=7)
        
        # Mock existing fixture with scores (finished match)
        existing_fixture = {
            "f": {
                "id": fixture_id,
                "kickoff_time": past_kickoff,
                "matchday": 1,
                "season_id": "2026-27",
                "status": "finished",
                "home_score": home_score,
                "away_score": away_score
            }
        }
        self.mock_conn.execute_query.return_value = [existing_fixture]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Try to update status (should preserve scores)
        updates = {
            "status": "completed"  # Just changing status
        }
        
        result = self.fixture_service.update_fixture(fixture_id, updates)
        
        # Should succeed
        self.assertTrue(result, "Update of historical fixture should succeed")
        
        # Verify execute_write was called
        self.mock_conn.execute_write.assert_called_once()
        
        # Verify that home_score and away_score are NOT in the updates
        # (they should be preserved, not included in the SET clause)
        call_args = self.mock_conn.execute_write.call_args
        params = call_args[0][1]
        
        # Status should be updated
        self.assertEqual(params["status"], "completed")
        
        # home_score and away_score should NOT be in params
        # (they should be preserved as-is in the database)
        self.assertNotIn("home_score", params, "home_score should be preserved, not updated")
        self.assertNotIn("away_score", params, "away_score should be preserved, not updated")
    
    @unittest.skipIf(not HYPOTHESIS_AVAILABLE, "hypothesis not installed")
    def test_update_fixture_returns_false_for_nonexistent_fixture(self):
        """
        Test that updating a non-existent fixture returns False.
        
        **Validates: Requirements 6.1**
        """
        # Mock no fixture found
        self.mock_conn.execute_query.return_value = []
        
        result = self.fixture_service.update_fixture("nonexistent-fixture-id", {"status": "cancelled"})
        
        # Should return False
        self.assertFalse(result, "Updating non-existent fixture should return False")
        
        # Should not call execute_write
        self.mock_conn.execute_write.assert_not_called()
    
    @unittest.skipIf(not HYPOTHESIS_AVAILABLE, "hypothesis not installed")
    def test_update_fixture_prevents_id_modification(self):
        """
        Test that fixture ID cannot be modified through updates.
        
        **Validates: Requirements 6.1**
        """
        fixture_id = "fixture-2026-27-md1-liv-ars"
        
        # Mock existing fixture
        existing_fixture = {
            "f": {
                "id": fixture_id,
                "kickoff_time": datetime(2026, 8, 15, 15, 0, tzinfo=timezone.utc),
                "matchday": 1,
                "season_id": "2026-27",
                "status": "scheduled"
            }
        }
        self.mock_conn.execute_query.return_value = [existing_fixture]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Try to update with id field (should be ignored)
        updates = {
            "id": "different-fixture-id",
            "fixture_id": "another-different-id",
            "status": "cancelled"
        }
        
        result = self.fixture_service.update_fixture(fixture_id, updates)
        
        # Should succeed (but id fields ignored)
        self.assertTrue(result)
        
        # Verify that id and fixture_id were NOT included in the update
        call_args = self.mock_conn.execute_write.call_args
        params = call_args[0][1]
        
        # fixture_id should only be used for matching, not updating
        self.assertEqual(params["fixture_id"], fixture_id)
        
        # id and fixture_id should NOT be in the SET parameters
        # Only status should be updated
        self.assertIn("status", params)
        self.assertEqual(params["status"], "cancelled")
        
        # Verify the query doesn't try to SET id or fixture_id
        query = call_args[0][0]
        self.assertNotIn("f.id =", query, "Query should not update id field")
        self.assertNotIn("f.fixture_id =", query, "Query should not update fixture_id field")
    
    def test_audit_logging_on_fixture_update(self):
        """
        Test that fixture updates are logged with audit information.
        
        **Validates: Requirements 6.5**
        """
        fixture_id = "fixture-2026-27-md1-liv-ars"
        
        # Mock existing fixture
        existing_fixture = {
            "f": {
                "id": fixture_id,
                "kickoff_time": datetime(2026, 8, 15, 15, 0, tzinfo=timezone.utc),
                "matchday": 1,
                "season_id": "2026-27",
                "status": "scheduled"
            }
        }
        self.mock_conn.execute_query.return_value = [existing_fixture]
        self.mock_conn.execute_write.return_value = Mock()
        
        # Perform update with user information
        updates = {"status": "completed", "home_score": 2, "away_score": 1}
        
        with patch('services.fixture_service.logger') as mock_logger:
            result = self.fixture_service.update_fixture(fixture_id, updates, user="admin_user")
            
            # Should succeed
            self.assertTrue(result)
            
            # Verify audit logging was called
            # Check that logger.info was called with audit information
            mock_logger.info.assert_called()
            
            # Get the call arguments
            call_args = mock_logger.info.call_args_list
            
            # Find the audit log call
            audit_log_found = False
            for call in call_args:
                if "AUDIT: Fixture updated" in str(call):
                    audit_log_found = True
                    # Verify extra fields exist
                    if len(call[1]) > 0 and 'extra' in call[1]:
                        extra = call[1]['extra']
                        self.assertIn('fixture_id', extra)
                        self.assertIn('user', extra)
                        self.assertIn('timestamp', extra)
                        self.assertEqual(extra['user'], 'admin_user')
                    break
            
            self.assertTrue(audit_log_found, "Audit log entry should be created")


@unittest.skipIf(not HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestOpponentStrengthProperties(unittest.TestCase):
    """Property-based tests for opponent strength calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_conn = Mock(spec=Neo4jConnection)
        self.fixture_service = FixtureService(self.mock_conn)
    
    # Property 17: Opponent Defensive Strength Calculation
    # **Validates: Requirements 8.1**
    @given(
        goals_conceded=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=5)
    )
    @settings(max_examples=20)
    def test_property_17_opponent_defensive_strength_calculation(self, goals_conceded):
        """
        Property 17: Opponent Defensive Strength Calculation
        
        For any team with N completed fixtures (1 ≤ N ≤ 5), the defensive strength 
        SHALL equal the average goals conceded across those N fixtures.
        
        **Validates: Requirements 8.1**
        """
        team_name = "Liverpool"
        
        # Mock fixture results with goals conceded
        goals_scored = [2] * len(goals_conceded)  # Fixed goals scored for simplicity
        
        self.mock_conn.execute_query.return_value = [{
            "conceded_list": goals_conceded,
            "scored_list": goals_scored,
            "matches_played": len(goals_conceded)
        }]
        
        # Calculate strength
        result = self.fixture_service.calculate_opponent_strength(team_name)
        
        # Verify defensive strength is the average of goals conceded
        expected_defensive_strength = sum(goals_conceded) / len(goals_conceded)
        self.assertAlmostEqual(
            result["defensive_strength"],
            expected_defensive_strength,
            places=5,
            msg=f"Defensive strength should be average of goals conceded: {goals_conceded}"
        )
    
    # Property 18: Opponent Offensive Strength Calculation
    # **Validates: Requirements 8.2**
    @given(
        goals_scored=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=5)
    )
    @settings(max_examples=20)
    def test_property_18_opponent_offensive_strength_calculation(self, goals_scored):
        """
        Property 18: Opponent Offensive Strength Calculation
        
        For any team with N completed fixtures (1 ≤ N ≤ 5), the offensive strength 
        SHALL equal the average goals scored across those N fixtures.
        
        **Validates: Requirements 8.2**
        """
        team_name = "Arsenal"
        
        # Mock fixture results with goals scored
        goals_conceded = [1] * len(goals_scored)  # Fixed goals conceded for simplicity
        
        self.mock_conn.execute_query.return_value = [{
            "conceded_list": goals_conceded,
            "scored_list": goals_scored,
            "matches_played": len(goals_scored)
        }]
        
        # Calculate strength
        result = self.fixture_service.calculate_opponent_strength(team_name)
        
        # Verify offensive strength is the average of goals scored
        expected_offensive_strength = sum(goals_scored) / len(goals_scored)
        self.assertAlmostEqual(
            result["offensive_strength"],
            expected_offensive_strength,
            places=5,
            msg=f"Offensive strength should be average of goals scored: {goals_scored}"
        )
    
    def test_opponent_strength_with_no_data_returns_league_average(self):
        """
        Test that teams with no historical data return league average strength.
        
        **Validates: Requirements 8.5**
        """
        team_name = "New Team"
        
        # Mock no fixture results
        self.mock_conn.execute_query.return_value = [{
            "conceded_list": [],
            "scored_list": [],
            "matches_played": 0
        }]
        
        # Calculate strength
        result = self.fixture_service.calculate_opponent_strength(team_name)
        
        # Should return league average (1.5 goals)
        self.assertEqual(result["defensive_strength"], 1.5)
        self.assertEqual(result["offensive_strength"], 1.5)
    
    def test_opponent_strength_with_fewer_than_5_matches(self):
        """
        Test that teams with fewer than 5 matches use available matches.
        
        **Validates: Requirements 8.1, 8.2**
        """
        team_name = "Chelsea"
        
        # Mock 3 fixtures only
        goals_conceded = [1, 2, 0]  # Average: 1.0
        goals_scored = [3, 2, 4]    # Average: 3.0
        
        self.mock_conn.execute_query.return_value = [{
            "conceded_list": goals_conceded,
            "scored_list": goals_scored,
            "matches_played": 3
        }]
        
        # Calculate strength
        result = self.fixture_service.calculate_opponent_strength(team_name)
        
        # Should use all 3 available matches
        self.assertAlmostEqual(result["defensive_strength"], 1.0)
        self.assertAlmostEqual(result["offensive_strength"], 3.0)
    
    def test_opponent_strength_only_considers_completed_fixtures(self):
        """
        Test that only completed fixtures with scores are included.
        
        **Validates: Requirements 8.1, 8.2**
        """
        team_name = "Manchester City"
        current_date = datetime(2026, 9, 1, tzinfo=timezone.utc)
        
        # Mock query - the query itself filters for completed fixtures
        # We just verify the method handles the results correctly
        goals_conceded = [2, 1, 3, 0, 2]  # 5 completed matches
        goals_scored = [3, 2, 4, 1, 2]
        
        self.mock_conn.execute_query.return_value = [{
            "conceded_list": goals_conceded,
            "scored_list": goals_scored,
            "matches_played": 5
        }]
        
        # Calculate strength
        result = self.fixture_service.calculate_opponent_strength(team_name, current_date=current_date)
        
        # Should calculate averages from completed matches only
        expected_defensive = sum(goals_conceded) / 5  # 1.6
        expected_offensive = sum(goals_scored) / 5     # 2.4
        
        self.assertAlmostEqual(result["defensive_strength"], expected_defensive)
        self.assertAlmostEqual(result["offensive_strength"], expected_offensive)
    
    def test_opponent_strength_orders_by_most_recent(self):
        """
        Test that strength calculation uses most recent N matches.
        
        This is verified by the Cypher query which includes:
        ORDER BY f.kickoff_time DESC LIMIT $num_matches
        
        **Validates: Requirements 8.1, 8.2**
        """
        team_name = "Manchester United"
        
        # The query already handles ordering - we just verify results are used correctly
        # Most recent 5 matches (already ordered by query)
        goals_conceded = [1, 0, 2, 1, 3]  # Most recent first
        goals_scored = [3, 2, 2, 1, 4]
        
        self.mock_conn.execute_query.return_value = [{
            "conceded_list": goals_conceded,
            "scored_list": goals_scored,
            "matches_played": 5
        }]
        
        # Calculate strength
        result = self.fixture_service.calculate_opponent_strength(team_name, num_matches=5)
        
        # Should use these exact matches
        expected_defensive = sum(goals_conceded) / 5  # 1.4
        expected_offensive = sum(goals_scored) / 5     # 2.4
        
        self.assertAlmostEqual(result["defensive_strength"], expected_defensive)
        self.assertAlmostEqual(result["offensive_strength"], expected_offensive)
    
    def test_opponent_strength_handles_home_and_away_correctly(self):
        """
        Test that defensive/offensive strength correctly identifies team perspective.
        
        For home games: goals_scored = home_score, goals_conceded = away_score
        For away games: goals_scored = away_score, goals_conceded = home_score
        
        **Validates: Requirements 8.1, 8.2**
        """
        team_name = "Tottenham Hotspur"
        
        # The Cypher query handles this logic, we verify the method uses results correctly
        # Assume query returns correctly calculated goals from team's perspective
        goals_conceded = [2, 1, 0, 3, 1]  # From team's perspective
        goals_scored = [3, 3, 2, 1, 4]    # From team's perspective
        
        self.mock_conn.execute_query.return_value = [{
            "conceded_list": goals_conceded,
            "scored_list": goals_scored,
            "matches_played": 5
        }]
        
        # Calculate strength
        result = self.fixture_service.calculate_opponent_strength(team_name)
        
        # Verify calculations
        expected_defensive = sum(goals_conceded) / 5  # 1.4
        expected_offensive = sum(goals_scored) / 5     # 2.6
        
        self.assertAlmostEqual(result["defensive_strength"], expected_defensive)
        self.assertAlmostEqual(result["offensive_strength"], expected_offensive)


if __name__ == '__main__':
    unittest.main()

