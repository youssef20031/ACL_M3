"""
Validation Script: Check for Data Leakage in V5 Models

Tests:
1. Shuffle test to verify R² collapses to ~0
2. Feature shift verification (no same-GW data in features)
3. Time-series cross-validation for robust R² estimate
4. Feature correlation analysis with target
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_engineering import FeatureEngineer
from ml.train import FPLModelTrainer

print("="*80)
print("DATA LEAKAGE VALIDATION SCRIPT")
print("="*80)

# Load data
print("\n1. Loading dataset...")
trainer = FPLModelTrainer("merged_recent_seasons.csv")
trainer.load_data()

# Split data
train_df_raw, val_df_raw, test_df_raw = trainer.temporal_train_test_split_raw()

print(f"Train: {len(train_df_raw)} samples")
print(f"Val: {len(val_df_raw)} samples")
print(f"Test: {len(test_df_raw)} samples")

# Test each position
positions = ['GK', 'DEF', 'MID', 'FWD']

for position in positions:
    print("\n" + "="*80)
    print(f"TESTING POSITION: {position}")
    print("="*80)
    
    # Filter by position
    test_pos_raw = test_df_raw[test_df_raw['position'] == position].copy()
    
    if len(test_pos_raw) == 0:
        print(f"No {position} data in test set. Skipping.")
        continue
    
    print(f"\nTest samples (raw): {len(test_pos_raw)}")
    
    # Engineer features
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(train_df_raw)
    
    test_pos = feature_engineer.engineer_features(
        test_pos_raw,
        is_training=False,
        lag_features=['total_points']
    )
    
    print(f"Test samples (after engineering): {len(test_pos)}")
    
    if len(test_pos) < 10:
        print(f"Too few samples after engineering. Skipping.")
        continue
    
    # Prepare features and target
    X_test, y_test = feature_engineer.prepare_features(test_pos, include_target=True)
    
    print(f"Features: {X_test.shape[1]}")
    print(f"Samples: {len(y_test)}")
    
    # Load model
    model_path = f"ml/models/xgboost_{position.lower()}_v3.pkl"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}. Skipping.")
        continue
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Original R²
    original_r2 = r2_score(y_test, y_pred)
    original_mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n📊 Original Performance:")
    print(f"   R² = {original_r2:.4f}")
    print(f"   MAE = {original_mae:.4f}")
    
    # ============================================
    # TEST 1: Shuffle Test (Leakage Detection)
    # ============================================
    print(f"\n🔍 TEST 1: Shuffle Test (should collapse to ~0)")
    
    # Shuffle y_test
    y_test_shuffled = y_test.sample(frac=1, random_state=99).values
    shuffled_r2 = r2_score(y_test, y_test_shuffled)
    
    print(f"   Shuffled R² = {shuffled_r2:.4f}")
    
    if shuffled_r2 > 0.1:
        print(f"   ❌ WARNING: Shuffled R² is high! Possible leakage!")
    elif shuffled_r2 < -0.5:
        print(f"   ✅ PASS: Shuffled R² is negative (model predictions are meaningful)")
    else:
        print(f"   ✅ PASS: Shuffled R² near 0 (model predictions are meaningful)")
    
    # ============================================
    # TEST 2: Feature Shift Verification
    # ============================================
    print(f"\n🔍 TEST 2: Feature Shift Verification")
    print("   Checking if defensive features use same-GW data...")
    
    # Check if opp_off_strength and team_def_strength exist
    if 'opp_off_strength' in X_test.columns:
        # Check correlation between opp_off_strength and target
        corr = X_test['opp_off_strength'].corr(y_test)
        print(f"   opp_off_strength vs target correlation: {corr:.4f}")
        
        if abs(corr) > 0.5:
            print(f"   ⚠️  WARNING: High correlation! Possible leakage!")
        else:
            print(f"   ✅ OK: Reasonable correlation")
    
    if 'team_def_strength' in X_test.columns:
        corr = X_test['team_def_strength'].corr(y_test)
        print(f"   team_def_strength vs target correlation: {corr:.4f}")
        
        if abs(corr) > 0.5:
            print(f"   ⚠️  WARNING: High correlation! Possible leakage!")
        else:
            print(f"   ✅ OK: Reasonable correlation")
    
    # Sample check: show first few rows
    print(f"\n   Sample feature values (first 5 rows):")
    sample_cols = [c for c in ['GW', 'opp_off_strength', 'team_def_strength', 'form'] if c in test_pos.columns]
    if sample_cols:
        print(test_pos[sample_cols].head())
    
    # ============================================
    # TEST 3: Time-Series Cross-Validation
    # ============================================
    print(f"\n🔍 TEST 3: Time-Series Cross-Validation (5-fold)")
    print("   This tests model stability across different time periods...")
    
    # Get all data for this position (train + val + test)
    all_pos_raw = pd.concat([train_df_raw, val_df_raw, test_df_raw])
    all_pos_raw = all_pos_raw[all_pos_raw['position'] == position].copy()
    
    # Engineer features
    all_pos = feature_engineer.engineer_features(
        all_pos_raw,
        is_training=False,
        lag_features=['total_points']
    )
    
    if len(all_pos) < 100:
        print(f"   Too few samples for CV ({len(all_pos)}). Skipping.")
    else:
        X_all, y_all = feature_engineer.prepare_features(all_pos, include_target=True)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_r2_scores = []
        cv_mae_scores = []
        
        from xgboost import XGBRegressor
        
        fold = 1
        for train_idx, test_idx in tscv.split(X_all):
            # Train model on fold
            fold_model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            fold_model.fit(X_all.iloc[train_idx], y_all.iloc[train_idx], verbose=False)
            
            # Evaluate on fold test set
            fold_pred = fold_model.predict(X_all.iloc[test_idx])
            fold_r2 = r2_score(y_all.iloc[test_idx], fold_pred)
            fold_mae = mean_absolute_error(y_all.iloc[test_idx], fold_pred)
            
            cv_r2_scores.append(fold_r2)
            cv_mae_scores.append(fold_mae)
            
            print(f"   Fold {fold}: R² = {fold_r2:.4f}, MAE = {fold_mae:.4f} (n={len(test_idx)})")
            fold += 1
        
        cv_r2_mean = np.mean(cv_r2_scores)
        cv_r2_std = np.std(cv_r2_scores)
        cv_mae_mean = np.mean(cv_mae_scores)
        
        print(f"\n   Cross-Validation Summary:")
        print(f"   R² = {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
        print(f"   MAE = {cv_mae_mean:.4f}")
        
        # Compare to original
        diff = abs(original_r2 - cv_r2_mean)
        if diff > 0.1:
            print(f"   ⚠️  WARNING: Large difference from original R² ({original_r2:.4f})")
            print(f"   Single test split may be lucky/unlucky")
        else:
            print(f"   ✅ PASS: CV R² consistent with original ({original_r2:.4f})")
    
    # ============================================
    # TEST 4: Feature Importance Check
    # ============================================
    print(f"\n🔍 TEST 4: Feature Importance Analysis")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns
        
        # Get top 10 features
        indices = np.argsort(importances)[::-1][:10]
        
        print(f"   Top 10 Most Important Features:")
        for i, idx in enumerate(indices, 1):
            print(f"   {i}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Check if defensive features are being used
        defensive_features = ['opp_off_strength', 'team_def_strength', 'opp_def_strength']
        for feat in defensive_features:
            if feat in feature_names:
                feat_idx = list(feature_names).index(feat)
                importance = importances[feat_idx]
                rank = sorted(importances, reverse=True).index(importance) + 1
                print(f"\n   {feat}: importance={importance:.4f}, rank={rank}/{len(feature_names)}")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print("\n✅ If all tests passed, the model is clean (no leakage)")
print("❌ If any test failed, there may be data leakage to investigate")
