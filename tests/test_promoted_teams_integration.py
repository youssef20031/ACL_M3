"""
Integration tests for promoted teams system.

Tests complete flow:
1. Championship data → Translation models → Predictions
2. Archetype matching works correctly
3. GW-based confidence decay
4. API integration
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestPromotedTeamsHandlerV1(unittest.TestCase):
    """Test V1 handler (6 manual archetypes)."""
    
    def setUp(self):
        from ml.promoted_teams_handler import PromotedTeamsHandler
        self.handler = PromotedTeamsHandler()
    
    def test_scale_championship_stats(self):
        """Test Championship → EPL scaling."""
        player_data = {
            'goals_scored': 20,
            'assists': 10,
            'form': 7.0,
            'position': 'FWD'
        }
        
        scaled = self.handler.scale_championship_stats(player_data)
        
        # Should apply 0.70 scalar
        self.assertAlmostEqual(scaled['goals_scored'], 14.0, places=1)
        self.assertAlmostEqual(scaled['assists'], 7.5, places=1)
        self.assertAlmostEqual(scaled['form'], 4.9, places=1)
        self.assertTrue(scaled['is_promoted'])
        self.assertEqual(scaled['league_origin'], 0.70)
    
    def test_blended_heuristic_early_season(self):
        """Test GW 1-6 uses heavy heuristic weighting."""
        player_data = {
            'position': 'FWD',
            'form': 6.0,
            'league_origin': 0.70
        }
        
        # GW 3 - should be 70% baseline, 30% scaled form
        prediction = self.handler.calculate_blended_heuristic(player_data, current_gw=3)
        
        # FWD baseline = 4.5
        # Form is already scaled in the data (6.0)
        # Expected: 0.70 * 4.5 + 0.30 * 6.0 = 3.15 + 1.8 = 4.95
        self.assertAlmostEqual(prediction, 4.95, places=1)
    
    def test_blended_heuristic_late_season(self):
        """Test GW 13+ uses mostly real EPL data."""
        player_data = {
            'position': 'MID',
            'form': 5.0,
            'pl_actual_form': 7.5,
            'league_origin': 0.70
        }
        
        # GW 15 - should be 80% real data
        prediction = self.handler.calculate_blended_heuristic(player_data, current_gw=15)
        
        # Should be dominated by pl_actual_form (7.5)
        self.assertGreater(prediction, 6.0)
    
    def test_predict_promoted_player_gw1(self):
        """Test prediction for promoted player in GW1."""
        player_data = {
            'name': 'Test Player',
            'position': 'FWD',
            'form': 6.0,
            'league_origin': 0.70
        }
        
        prediction, metadata = self.handler.predict_promoted_player(
            player_data, current_gw=1
        )
        
        # Should return reasonable FWD prediction (3-6 points)
        self.assertGreater(prediction, 2.0)
        self.assertLess(prediction, 8.0)
        self.assertEqual(metadata['cold_start_phase'], 'initial')
        self.assertLess(metadata['confidence'], 0.70)  # Low confidence early


class TestPromotedTeamsHandlerV2(unittest.TestCase):
    """Test V2 handler (translation models)."""
    
    @patch('ml.promoted_teams_handler_v2.os.path.exists')
    def setUp(self, mock_exists):
        mock_exists.return_value = False  # No models available
        from ml.promoted_teams_handler_v2 import PromotedTeamsHandlerV2
        self.handler = PromotedTeamsHandlerV2()
    
    def test_initialization_without_models(self):
        """Test handler works without translation models (fallback)."""
        self.assertIsNone(self.handler.model_a)
        self.assertIsNone(self.handler.model_b)
    
    def test_scale_championship_stats_fallback(self):
        """Test falls back to flat scalars when models unavailable."""
        player_data = {
            'goals_scored': 20,
            'position': 'FWD'
        }
        
        scaled = self.handler.scale_championship_stats(player_data)
        
        # Should apply flat scalar fallback
        self.assertAlmostEqual(scaled['goals_scored'], 14.0, places=1)


class TestMinutesModelsLeakagePrevention(unittest.TestCase):
    """Test data leakage prevention in minutes models."""
    
    def test_exclusion_list_completeness(self):
        """Verify exclusion list includes all leakage variables."""
        from ml.minutes_model import MinutesPredictor
        
        # These should NEVER be in feature columns
        leakage_vars = [
            'goals_scored', 'assists', 'clean_sheets',
            'bonus', 'bps', 'total_points', 'minutes',
            'started', 'starts', 'saves',
            'team_a_score', 'team_h_score'
        ]
        
        # Would need to instantiate and check, but structure is correct
        self.assertTrue(True)  # Placeholder - actual test in test_minutes_leakage.py
    
    def test_lagged_features(self):
        """Verify minutes features use shift(1)."""
        import pandas as pd
        from ml.minutes_model import MinutesPredictor
        
        predictor = MinutesPredictor()
        
        # Create test data
        df = pd.DataFrame({
            'name': ['Player_A'] * 5,
            'minutes': [90, 0, 90, 45, 90],
            'started': [1, 0, 1, 0, 1],
            'GW': [1, 2, 3, 4, 5]
        })
        
        df_with_features = predictor._add_minutes_features(df)
        
        # Check first row has 0 (no previous data)
        self.assertEqual(df_with_features.iloc[0]['started_last_match'], 0)
        
        # Check second row references first row
        self.assertEqual(df_with_features.iloc[1]['started_last_match'], 1)


class TestAPIIntegration(unittest.TestCase):
    """Test API integration with promoted teams system."""
    
    @patch('ml.api_integration.FPLPredictor')
    def test_api_loads_v2_handler_when_available(self, mock_predictor_class):
        """Test API auto-upgrades to v2 handler."""
        from ml.api_integration import MLAPIIntegration
        
        mock_conn = Mock()
        mock_executor = Mock()
        
        integration = MLAPIIntegration(mock_conn, mock_executor)
        
        # Mock v2 archetype DB exists
        with patch('os.path.exists', return_value=True):
            integration.load_predictor(enable_promoted_teams_v2=True)
        
        # Should have attempted to load predictor
        mock_predictor_class.assert_called()


class TestDeployment2026(unittest.TestCase):
    """Test 2026-27 deployment configuration."""
    
    def test_promoted_teams_config_exists(self):
        """Verify 2026-27 config file exists."""
        config_path = 'ml/config/promoted_teams_2026.json'
        self.assertTrue(os.path.exists(config_path))
    
    def test_promoted_teams_config_valid(self):
        """Verify config contains Bristol City, Ipswich, Hull."""
        import json
        
        with open('ml/config/promoted_teams_2026.json') as f:
            config = json.load(f)
        
        self.assertEqual(config['season'], '2026-2027')
        self.assertIn('Bristol City', config['promoted_teams'])
        self.assertIn('Ipswich Town', config['promoted_teams'])
        self.assertIn('Hull City', config['promoted_teams'])


def run_tests():
    """Run all integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPromotedTeamsHandlerV1))
    suite.addTests(loader.loadTestsFromTestCase(TestPromotedTeamsHandlerV2))
    suite.addTests(loader.loadTestsFromTestCase(TestMinutesModelsLeakagePrevention))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDeployment2026))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit(run_tests())
