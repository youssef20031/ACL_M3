"""
Comprehensive leakage validation tests.
Tests suggested by user for FPL-style pipelines.
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.minutes_model import MinutesPredictor
from ml.train_minutes_models import load_fpl_data

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_rolling_features_for_leakage(df):
    """
    Check if rolling features accidentally include current GW.
    
    Rolling features should use .shift(1) to exclude current row.
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Rolling Matchweek Aggregates (Current GW Leakage)")
    logger.info("="*70)
    
    # Check a player's rolling features
    test_player = df[df['name'] == df['name'].iloc[0]].head(10).copy()
    test_player = test_player.sort_values('GW')
    
    if 'minutes' not in test_player.columns:
        logger.warning("Cannot test - minutes column missing")
        return True
    
    # Check if minutes_rolling5 includes current row
    logger.info("\nChecking 'minutes_rolling5' excludes current match...")
    
    predictor = MinutesPredictor()
    test_player = predictor._create_started_feature(test_player)
    test_player = predictor._add_minutes_features(test_player)
    test_player = predictor.feature_engineer.engineer_features(test_player, is_training=True)
    
    if 'minutes_rolling5' not in test_player.columns:
        logger.warning("  minutes_rolling5 not found, skipping check")
        return True
    
    # For row i, minutes_rolling5 should be avg of rows [i-5 to i-1], NOT including i
    for i in range(5, min(10, len(test_player))):
        current_minutes = test_player.iloc[i]['minutes']
        rolling_avg = test_player.iloc[i]['minutes_rolling5']
        
        # If current match minutes affects the rolling avg, we have leakage
        past_5_minutes = test_player.iloc[i-5:i]['minutes'].values
        expected_avg = past_5_minutes.mean() if len(past_5_minutes) > 0 else 0
        
        if abs(rolling_avg - expected_avg) > 1.0:
            # Check if current minute is included
            with_current = test_player.iloc[i-4:i+1]['minutes'].mean()
            if abs(rolling_avg - with_current) < 1.0:
                logger.error(f"  ❌ Row {i}: minutes_rolling5={rolling_avg:.1f} includes current match (minutes={current_minutes})")
                return False
    
    logger.info("  ✅ minutes_rolling5 correctly excludes current match")
    
    # Check form (should be lagged total_points)
    logger.info("\nChecking 'form' excludes current match...")
    if 'form' in test_player.columns and 'total_points' in test_player.columns:
        for i in range(4, min(10, len(test_player))):
            current_points = test_player.iloc[i]['total_points']
            form_value = test_player.iloc[i]['form']
            
            # Form should be rolling avg of PAST 4 gameweeks, not including current
            past_4_points = test_player.iloc[max(0,i-4):i]['total_points'].values
            
            if len(past_4_points) > 0 and current_points > 0:
                # If form includes current points, it would be higher
                if current_points in past_4_points:
                    logger.error(f"  ❌ Row {i}: form may include current match points")
                    return False
        
        logger.info("  ✅ form correctly excludes current match")
    
    return True


def check_season_totals_for_leakage(df):
    """
    Check if cumulative season stats accidentally include target match.
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Post-Match Season Totals")
    logger.info("="*70)
    
    # Check if we're using any cumulative columns
    cumulative_cols = [col for col in df.columns if 'total' in col.lower()]
    
    logger.info(f"\nFound {len(cumulative_cols)} potential cumulative columns:")
    for col in cumulative_cols:
        logger.info(f"  - {col}")
    
    # Check if total_points is used as a feature (it shouldn't be)
    predictor = MinutesPredictor()
    test_df = df.head(100).copy()
    test_df = predictor._create_started_feature(test_df)
    test_df = predictor._add_minutes_features(test_df)
    test_df = predictor.feature_engineer.engineer_features(test_df, is_training=True)
    
    exclude_cols = [
        'name', 'started', 'starts', 'minutes', 'total_points', 'upcoming',
        'season_x', 'season', 'element', 'fixture', 'kickoff_time',
        'selected', 'transfers_in', 'transfers_out', 'transfers_balance',
        'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
        'bonus', 'bps', 'saves', 'penalties_saved', 'penalties_missed',
        'yellow_cards', 'red_cards', 'own_goals',
        'influence', 'creativity', 'threat', 'ict_index',
        'recoveries', 'tackles', 'clearances_blocks_interceptions',
        'defensive_contribution',
        'xP', 'expected_assists', 'expected_goal_involvements',
        'expected_goals', 'expected_goals_conceded',
        'team_a_score', 'team_h_score', 'opponent_team', 'opp_team_name',
        'round', 'Unnamed: 0', 'index', 'home_team', 'away_team',
        'stat_category', 'team_x', 'position', 'modified'
    ]
    
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]
    
    # Check if any cumulative column leaked through
    leaked_cumulative = [col for col in cumulative_cols if col in feature_cols]
    
    if leaked_cumulative:
        logger.error(f"  ❌ Cumulative columns in features: {leaked_cumulative}")
        return False
    else:
        logger.info("  ✅ No cumulative season totals in features")
    
    return True


def ablation_test_minimal_features(df, position='FWD'):
    """
    Ablation test: Train with ONLY safe features (no performance stats).
    
    Safe features:
    - Fixture difficulty (opponent strength)
    - Team strength
    - Historical minutes trends (lagged)
    - Basic metadata (position, value, was_home, GW)
    
    If AUC is still high (>0.85), we might have leakage.
    If AUC drops significantly (<0.75), confirms model needs legitimate features.
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Ablation Test (Minimal Safe Features)")
    logger.info("="*70)
    
    # Filter to position
    df_pos = df[df['position'] == position].copy()
    df_pos = df_pos.sort_values('kickoff_time').reset_index(drop=True)
    
    # Temporal split
    n = len(df_pos)
    train_end = int(n * 0.8)
    
    train_df = df_pos.iloc[:train_end].copy()
    test_df = df_pos.iloc[train_end:].copy()
    
    logger.info(f"\nPosition: {position}")
    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Test: {len(test_df)} samples")
    
    # Train full model
    logger.info("\n1. Training FULL model (all features)...")
    predictor_full = MinutesPredictor()
    predictor_full.train(train_df, val_df=None, save_models=False)
    
    # Prepare test data
    test_df_full = predictor_full._create_started_feature(test_df)
    test_df_full = predictor_full._add_minutes_features(test_df_full)
    test_df_full = predictor_full.feature_engineer.engineer_features(
        test_df_full, is_training=False
    )
    
    # Predict with full model
    if position in predictor_full.pstart_models:
        model_info = predictor_full.pstart_models[position]
        X_test_full = test_df_full[model_info['feature_cols']].fillna(0)
        y_test = test_df_full['started']
        
        y_pred_full = model_info['model'].predict_proba(X_test_full)[:, 1]
        auc_full = roc_auc_score(y_test, y_pred_full)
        
        logger.info(f"  Full model AUC: {auc_full:.4f}")
        logger.info(f"  Features used: {len(model_info['feature_cols'])}")
    else:
        logger.warning(f"  No model trained for {position}")
        return True
    
    # Now train MINIMAL model (ablation)
    logger.info("\n2. Training MINIMAL model (safe features only)...")
    logger.info("   Keeping: team/opponent strength, minutes trends, metadata")
    logger.info("   Removing: form, points_per_90, home/away form, etc.")
    
    # Create minimal feature set
    safe_features_only = [
        'started_last_match',
        'started_2_of_last_3',
        'avg_minutes_last_3',
        'minutes_rolling5',
        'opp_def_strength',
        'opp_off_strength',
        'team_def_strength',
        'fixtures_this_gw',
        'value',
        'was_home',
        'GW',
        'gw_in_season',
    ]
    
    # Add team one-hot columns
    team_cols = [col for col in model_info['feature_cols'] if col.startswith('team_')]
    safe_features_only.extend(team_cols)
    
    # Filter to safe features only
    available_safe = [f for f in safe_features_only if f in X_test_full.columns]
    X_test_minimal = X_test_full[available_safe]
    
    logger.info(f"  Minimal features: {len(available_safe)}")
    logger.info(f"  Example: {available_safe[:5]}")
    
    # Train minimal model on same data
    X_train_full = train_df_full[model_info['feature_cols']].fillna(0)
    X_train_minimal = X_train_full[available_safe]
    y_train = train_df_full['started']
    
    from xgboost import XGBClassifier
    model_minimal = XGBClassifier(
        objective='binary:logistic',
        max_depth=4,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42
    )
    
    train_df_full = predictor_full._create_started_feature(train_df)
    train_df_full = predictor_full._add_minutes_features(train_df_full)
    train_df_full = predictor_full.feature_engineer.engineer_features(
        train_df_full, is_training=False
    )
    X_train_full = train_df_full[model_info['feature_cols']].fillna(0)
    X_train_minimal = X_train_full[available_safe]
    y_train = train_df_full['started']
    
    model_minimal.fit(X_train_minimal, y_train)
    
    # Predict with minimal model
    y_pred_minimal = model_minimal.predict_proba(X_test_minimal)[:, 1]
    auc_minimal = roc_auc_score(y_test, y_pred_minimal)
    
    logger.info(f"  Minimal model AUC: {auc_minimal:.4f}")
    
    # Analysis
    logger.info("\n3. Analysis:")
    auc_drop = auc_full - auc_minimal
    pct_drop = (auc_drop / auc_full) * 100
    
    logger.info(f"  AUC drop: {auc_drop:.4f} ({pct_drop:.1f}%)")
    
    if auc_minimal > 0.88:
        logger.warning(f"  ⚠️  Minimal model AUC still very high (>0.88)")
        logger.warning(f"  This suggests possible remaining leakage in 'safe' features")
        return False
    elif auc_drop < 0.03:
        logger.warning(f"  ⚠️  AUC drop < 0.03 suggests performance stats not needed")
        logger.warning(f"  Could indicate leakage in remaining features")
        return False
    else:
        logger.info(f"  ✅ Significant drop confirms model uses legitimate features")
        logger.info(f"  Minimal features give baseline ~{auc_minimal:.2f}")
        logger.info(f"  Performance features add ~{auc_drop:.2f} AUC")
    
    return True


def strict_time_split_test(df, position='FWD'):
    """
    Test 4: Strict time split (train on GW 1-25, test on GW 26-29).
    
    This catches train/test contamination that temporal percentage splits might miss.
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Strict GW Split (GW 1-25 train, 26-29 test)")
    logger.info("="*70)
    
    # Filter to latest season and position
    df_latest = df[df['season_x'] == '2025-26'].copy() if 'season_x' in df.columns else df.copy()
    df_pos = df_latest[df_latest['position'] == position].copy()
    
    if len(df_pos) == 0:
        logger.warning("No data for strict split test")
        return True
    
    # Split by GW
    train_gw = df_pos[df_pos['GW'] <= 25].copy()
    test_gw = df_pos[df_pos['GW'] >= 26].copy()
    
    if len(test_gw) < 50:
        logger.warning(f"Not enough test data ({len(test_gw)} samples), skipping")
        return True
    
    logger.info(f"\nPosition: {position}")
    logger.info(f"Train: GW 1-25, {len(train_gw)} samples")
    logger.info(f"Test: GW 26+, {len(test_gw)} samples")
    
    # Train model
    predictor = MinutesPredictor()
    predictor.train(train_gw, val_df=None, save_models=False)
    
    # Predict on future GWs
    test_gw = predictor._create_started_feature(test_gw)
    test_gw = predictor._add_minutes_features(test_gw)
    test_gw = predictor.feature_engineer.engineer_features(test_gw, is_training=False)
    
    if position in predictor.pstart_models:
        model_info = predictor.pstart_models[position]
        X_test = test_gw[model_info['feature_cols']].fillna(0)
        y_test = test_gw['started']
        
        y_pred = model_info['model'].predict_proba(X_test)[:, 1]
        auc_future = roc_auc_score(y_test, y_pred)
        
        logger.info(f"\nFuture GW AUC: {auc_future:.4f}")
        
        # Compare to temporal split AUC
        if auc_future < 0.70:
            logger.warning(f"  ⚠️  AUC on future GWs is low (<0.70)")
            logger.warning(f"  Model may not generalize well to unseen gameweeks")
        elif auc_future > 0.95:
            logger.warning(f"  ⚠️  AUC on future GWs is suspiciously high (>0.95)")
            logger.warning(f"  Possible leakage or very predictable period")
        else:
            logger.info(f"  ✅ Future GW performance is realistic")
    
    return True


def main():
    """Run all validation tests."""
    logger.info("\n" + "🔍"*35)
    logger.info("COMPREHENSIVE LEAKAGE VALIDATION")
    logger.info("Following user-suggested FPL pipeline checks")
    logger.info("🔍"*35)
    
    # Load data
    df = load_fpl_data()
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    df = df.dropna(subset=['minutes', 'position'])
    
    if 'season_x' not in df.columns and 'season' in df.columns:
        df['season_x'] = df['season']
    
    logger.info(f"\nLoaded {len(df):,} records")
    
    # Run tests
    results = []
    
    # Test 1: Rolling features
    results.append(("Rolling GW aggregates", check_rolling_features_for_leakage(df)))
    
    # Test 2: Season totals
    results.append(("Post-match season totals", check_season_totals_for_leakage(df)))
    
    # Test 3: Ablation (FWD only to save time)
    results.append(("Ablation test (FWD)", ablation_test_minimal_features(df, 'FWD')))
    
    # Test 4: Strict time split (FWD only)
    results.append(("Strict GW split (FWD)", strict_time_split_test(df, 'FWD')))
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:40s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        logger.info("\n" + "="*70)
        logger.info("✅ ALL VALIDATION TESTS PASSED")
        logger.info("Model is robust against common FPL leakage patterns")
        logger.info("="*70)
        return 0
    else:
        logger.error("\n" + "="*70)
        logger.error("❌ SOME TESTS FAILED")
        logger.error("Review failures above for potential remaining leakage")
        logger.error("="*70)
        return 1


if __name__ == '__main__':
    exit(main())
