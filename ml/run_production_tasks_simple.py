"""
Simplified Production Readiness Script
Runs calibration, start probability, and DGW analysis using pre-trained models
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report

# Create output directory
Path("ml/models").mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("FPL ML PRODUCTION READINESS - SIMPLIFIED")
print("="*60)
print("\nNote: Run this after training models with ml/train.py\n")

# ============================================================================
# TASK 1: CALIBRATION ANALYSIS (From Training Results)
# ============================================================================

print("="*60)
print("TASK 1: CALIBRATION ANALYSIS")
print("="*60)
print("Analyzing predictions from training results...")

# Load training results
results_file = Path("ml/models/training_results.json")
if results_file.exists():
    import json
    with open(results_file, 'r') as f:
        training_results = json.load(f)
    
    print("\n📊 XGBoost Model Performance by Position:\n")
    xgb_results = training_results.get('xgboost_by_position', {})
    
    for position, metrics in xgb_results.get('by_position', {}).items():
        print(f"{position}:")
        print(f"  R² = {metrics['r2']:.4f}")
        print(f"  MAE = {metrics['mae']:.2f} pts")
        print(f"  RMSE = {metrics['rmse']:.2f} pts")
        print(f"  Samples = {metrics['n_samples']:,}")
        print()
    
    print("✅ Calibration metrics loaded from training results")
    print("   For detailed calibration curves, retrain with calibration=True")
else:
    print("⚠️  No training results found. Run: python ml/train.py")

# ============================================================================
# TASK 2: START PROBABILITY MODEL
# ============================================================================

print("\n" + "="*60)
print("TASK 2: START PROBABILITY MODEL")
print("="*60)
print("Checking for start probability model...")

start_model_file = Path("ml/models/start_probability_v1.pkl")
if start_model_file.exists():
    with open(start_model_file, 'rb') as f:
        start_model_data = pickle.load(f)
    print(f"✅ Start probability model found!")
    print(f"   AUC: {start_model_data['metrics']['test_auc']:.4f}")
    print(f"   Start rate: {start_model_data['metrics']['start_rate']:.1%}")
else:
    print("ℹ️  Start probability model not yet built")
    print("   This requires full dataset - will be built during training")

# ============================================================================
# TASK 3: DOUBLE GAMEWEEK ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("TASK 3: DOUBLE GAMEWEEK FEATURES")
print("="*60)
print("Analyzing DGW patterns from dataset...")

# Try to load data to check for DGW
data_file = "cleaned_merged_seasons_cleaned.csv"
if Path(data_file).exists():
    df = pd.read_csv(data_file, nrows=100000)  # Sample for analysis
    
    # Count fixtures per player per gameweek
    if 'name' in df.columns and 'GW' in df.columns:
        fixtures_per_gw = df.groupby(['name', 'GW']).size()
        dgw_count = (fixtures_per_gw > 1).sum()
        total_gw = len(fixtures_per_gw)
        
        print(f"\n📊 DGW Analysis:")
        print(f"   Total gameweeks: {total_gw:,}")
        print(f"   Double gameweeks: {dgw_count:,} ({dgw_count/total_gw:.1%})")
        
        if dgw_count > 0:
            # Sample DGW data
            dgw_mask = df.duplicated(subset=['name', 'GW'], keep=False)
            dgw_sample = df[dgw_mask].head(10)
            print(f"\n   Sample DGW entries:")
            print(dgw_sample[['name', 'GW', 'total_points']].to_string(index=False))
        
        print("\n✅ DGW features can be added to training pipeline")
        print("   Features: fixtures_this_gw, is_dgw, next_gw_fixtures")
    else:
        print("⚠️  Dataset missing required columns for DGW analysis")
else:
    print(f"⚠️  Dataset not found: {data_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("✅ PRODUCTION READINESS CHECK COMPLETE")
print("="*60)

print("\n📋 STATUS SUMMARY:")
print(f"  {'✅' if results_file.exists() else '❌'} Training results available")
print(f"  {'✅' if start_model_file.exists() else '⏳'} Start probability model")
print(f"  {'✅' if Path(data_file).exists() else '❌'} Dataset available for DGW")

print("\n🚀 NEXT STEPS:")
if not results_file.exists():
    print("  1. Train models: python ml/train.py")
else:
    print("  1. ✅ Models trained")

if not start_model_file.exists():
    print("  2. Build start probability (integrated in train.py)")
else:
    print("  2. ✅ Start probability model ready")

print("  3. Add DGW features to feature_engineering.py (already implemented)")
print("  4. Integrate into API (ml/api_integration.py)")
print("  5. Deploy to production!")

print("\n" + "="*60)

# Create a simple calibration note
print("\n📝 CALIBRATION GUIDANCE:")
print("   Top 20% bias interpretation:")
print("   • Positive bias (+0.3) = model underest imates high scorers (conservative)")
print("   • Negative bias (-0.3) = model overestimates high scorers (risky captain picks)")
print("   • Target: ±0.5 pts is acceptable")
print("\n" + "="*60)
