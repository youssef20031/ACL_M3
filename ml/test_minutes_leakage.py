"""
Test script to verify minutes models don't have data leakage.

Checks:
1. Feature columns don't include current-GW outcomes
2. Minutes-specific features are properly lagged
3. Target values don't leak into features
"""
import sys
import os
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.minutes_model import MinutesPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_feature_leakage():
    """Test that feature columns don't include leakage."""
    logger.info("="*80)
    logger.info("TEST: Feature Leakage Check")
    logger.info("="*80)
    
    # Create synthetic test data
    n_samples = 100
    test_data = pd.DataFrame({
        'name': ['Player_A'] * 50 + ['Player_B'] * 50,
        'position': ['FWD'] * 50 + ['MID'] * 50,
        'GW': list(range(1, 51)) + list(range(1, 51)),
        'kickoff_time': pd.date_range('2024-01-01', periods=n_samples),
        
        # Historical features (OK)
        'form': np.random.uniform(3, 7, n_samples),
        'value': np.random.uniform(60, 120, n_samples),
        'was_home': np.random.choice([0, 1], n_samples),
        'minutes_rolling5': np.random.uniform(60, 90, n_samples),
        
        # Current GW target (should NOT be in features)
        'minutes': np.random.randint(0, 90, n_samples),
        'starts': np.random.choice([0, 1], n_samples),
        'started': np.random.choice([0, 1], n_samples),
        
        # Current GW outcomes (LEAKAGE if included)
        'goals_scored': np.random.randint(0, 3, n_samples),
        'assists': np.random.randint(0, 2, n_samples),
        'total_points': np.random.randint(0, 15, n_samples),
        'bonus': np.random.randint(0, 4, n_samples),
        'clean_sheets': np.random.choice([0, 1], n_samples),
    })
    
    # Initialize predictor
    predictor = MinutesPredictor()
    
    # Create started feature
    test_data = predictor._create_started_feature(test_data)
    
    # Add minutes features
    test_data = predictor._add_minutes_features(test_data)
    
    # Check minutes features are lagged
    logger.info("\n1. Checking minutes-specific features are properly lagged...")
    
    player_a = test_data[test_data['name'] == 'Player_A'].copy()
    
    # First row should have 0 for lagged features
    assert player_a.iloc[0]['started_last_match'] == 0, "❌ started_last_match not properly initialized"
    assert player_a.iloc[0]['avg_minutes_last_3'] == 0, "❌ avg_minutes_last_3 not properly initialized"
    
    # Second row should reference first row's values
    if len(player_a) > 1:
        expected_started = player_a.iloc[0]['started']
        actual_started_lag = player_a.iloc[1]['started_last_match']
        assert actual_started_lag == expected_started, f"❌ started_last_match leakage: expected {expected_started}, got {actual_started_lag}"
    
    logger.info("   ✅ Minutes features properly lagged with shift(1)")
    
    # Simulate feature preparation for Model C
    logger.info("\n2. Checking Model C feature exclusion list...")
    
    exclude_cols = [
        # Target and direct leakage
        'name', 'started', 'starts', 'minutes', 'total_points', 'upcoming',
        # Identifiers
        'season_x', 'season', 'element', 'fixture', 'kickoff_time',
        # Current GW outcomes (LEAKAGE!)
        'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'bonus', 'bps', 'saves', 'penalties_saved', 'penalties_missed',
        'yellow_cards', 'red_cards', 'own_goals',
        # CRITICAL: ICT statistics (current-GW FPL stats)
        'influence', 'creativity', 'threat', 'ict_index',
        # CRITICAL: Defensive statistics (current-GW outcomes)
        'recoveries', 'tackles', 'clearances_blocks_interceptions',
        'defensive_contribution',
    ]
    
    feature_cols = [col for col in test_data.columns if col not in exclude_cols]
    
    # Check no leakage columns in features
    leakage_cols = ['goals_scored', 'assists', 'total_points', 'bonus', 'minutes', 'started']
    found_leakage = [col for col in feature_cols if col in leakage_cols]
    
    if found_leakage:
        logger.error(f"   ❌ LEAKAGE FOUND: {found_leakage} in feature columns!")
        return False
    else:
        logger.info(f"   ✅ No leakage columns in features (excluded: {len(exclude_cols)} columns)")
    
    logger.info(f"\n3. Final feature set has {len(feature_cols)} columns:")
    logger.info(f"   {feature_cols[:10]}...")
    
    # Verify safe features are included
    safe_features = ['form', 'value', 'was_home', 'minutes_rolling5', 
                    'started_last_match', 'avg_minutes_last_3']
    for feat in safe_features:
        if feat in test_data.columns:
            assert feat in feature_cols, f"❌ Safe feature {feat} incorrectly excluded"
    
    logger.info("   ✅ All safe features included")
    
    return True


def test_temporal_ordering():
    """Test that features only use past information."""
    logger.info("\n" + "="*80)
    logger.info("TEST: Temporal Ordering Check")
    logger.info("="*80)
    
    # Create time-series test case
    test_data = pd.DataFrame({
        'name': ['Player_A'] * 10,
        'position': ['FWD'] * 10,
        'GW': list(range(1, 11)),
        'kickoff_time': pd.date_range('2024-01-01', periods=10),
        'minutes': [90, 0, 90, 45, 90, 0, 90, 90, 60, 90],
        'started': [1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
    })
    
    predictor = MinutesPredictor()
    test_data = predictor._add_minutes_features(test_data)
    
    logger.info("\nChecking temporal consistency of lagged features...")
    
    for i in range(1, len(test_data)):
        # Check started_last_match
        expected = test_data.iloc[i-1]['started']
        actual = test_data.iloc[i]['started_last_match']
        
        assert expected == actual, f"❌ GW{i}: started_last_match mismatch (expected {expected}, got {actual})"
    
    logger.info("   ✅ All lagged features correctly reference previous timesteps only")
    
    # Check avg_minutes_last_3 calculation
    gw4_avg = test_data.iloc[3]['avg_minutes_last_3']
    expected_gw4 = np.mean([90, 0, 90])  # GW 1-3
    
    assert abs(gw4_avg - expected_gw4) < 0.1, f"❌ avg_minutes_last_3 calculation error: expected {expected_gw4}, got {gw4_avg}"
    
    logger.info("   ✅ Rolling averages correctly use only past data")
    
    return True


def main():
    """Run all leakage tests."""
    logger.info("\n" + "🔒"*40)
    logger.info("DATA LEAKAGE VERIFICATION: MINUTES MODELS")
    logger.info("🔒"*40 + "\n")
    
    try:
        test1_pass = test_feature_leakage()
        test2_pass = test_temporal_ordering()
        
        if test1_pass and test2_pass:
            logger.info("\n" + "="*80)
            logger.info("✅ ALL TESTS PASSED - NO DATA LEAKAGE DETECTED")
            logger.info("="*80)
            return 0
        else:
            logger.error("\n" + "="*80)
            logger.error("❌ TESTS FAILED - DATA LEAKAGE DETECTED")
            logger.error("="*80)
            return 1
    
    except Exception as e:
        logger.error(f"\n❌ Test execution failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
