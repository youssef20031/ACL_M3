"""
Run Production Readiness Tasks on Trained Models
Executes calibration, start probability, and DGW analysis
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml.production_readiness import ProductionReadiness
from ml.feature_engineering import FeatureEngineer


def load_models():
    """Load trained XGBoost models."""
    models_dir = Path("ml/models")
    position_models = {}
    
    for position in ['gk', 'def', 'mid', 'fwd']:
        model_path = models_dir / f'xgboost_{position}_v3.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                position_models[position.upper()] = pickle.load(f)
            print(f"✅ Loaded {position.upper()} model")
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    return position_models


def load_and_prepare_data():
    """Load and prepare dataset for analysis."""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load merged dataset
    data_files = [
        "cleaned_merged_seasons_cleaned.csv",
        "merged_recent_seasons.csv",
        "FPL_2023_2024.csv"
    ]
    
    df = None
    for data_file in data_files:
        if Path(data_file).exists():
            print(f"Loading {data_file}...")
            df = pd.read_csv(data_file)
            break
    
    if df is None:
        raise FileNotFoundError("No dataset found. Please ensure data files exist.")
    
    print(f"Loaded {len(df):,} rows")
    
    # Basic preprocessing
    if 'kickoff_time' in df.columns:
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')
        df = df.sort_values('kickoff_time').reset_index(drop=True)
    
    # Filter to recent seasons (same as training)
    if 'season_x' in df.columns:
        recent_seasons = ['2023-24', '2024-25', '2025-26']
        df = df[df['season_x'].isin(recent_seasons)]
        print(f"Filtered to {len(df):,} rows from seasons: {recent_seasons}")
    
    # Engineer features
    print("\nEngineering features...")
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(df)
    
    # Keep original df with metadata before one-hot encoding
    df_original = df.copy()
    
    # Engineer features (this modifies df in-place)
    df_engineered = feature_engineer.engineer_features(df, is_training=True, lag_features=['total_points'])
    
    print(f"Final dataset: {len(df_engineered):,} rows")
    
    # df_original has the metadata we need (position, name, etc.)
    # df_engineered has the features after engineering
    # They should have same length (just columns changed)
    
    return df_original, feature_engineer


def main():
    """Main execution."""
    print("\n" + "="*60)
    print("FPL ML PRODUCTION READINESS TASKS")
    print("="*60)
    print("\nThis script will:")
    print("1. Analyze calibration of predictions (captain pick accuracy)")
    print("2. Build start probability model (rotation risk)")
    print("3. Add double gameweek features (highest FPL edge)")
    print("\nEstimated time: 2-3 minutes\n")
    
    # Load models
    print("="*60)
    print("LOADING MODELS")
    print("="*60)
    position_models = load_models()
    
    if not position_models:
        print("❌ No models found. Please train models first:")
        print("   python ml/train.py")
        return
    
    # Load data
    df, feature_engineer = load_and_prepare_data()
    
    # Prepare position-specific data and models
    position_data = {}
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        pos_mask = df['position'] == position
        position_data[position] = {
            'df': df[pos_mask].copy(),
            'model': position_models.get(position)
        }
    
    # ========================================================================
    # TASK 1: CALIBRATION ANALYSIS
    # ========================================================================
    print("\n" + "="*60)
    print("TASK 1: CALIBRATION ANALYSIS")
    print("="*60)
    print("Checking if predictions are well-calibrated...")
    print("Key question: Are high predictions (captain picks) trustworthy?\n")
    
    # Initialize production readiness
    prod_ready = ProductionReadiness(models_dir="ml/models")
    
    calibration_results = {}
    
    for position, data in position_data.items():
        if data['model'] is None:
            continue
        
        pos_df = data['df']
        model = data['model']
        
        # Need to engineer features for this position's data
        pos_df_eng = feature_engineer.engineer_features(pos_df.copy(), is_training=False, lag_features=['total_points'])
        X_pos, y_pos = feature_engineer.prepare_features(pos_df_eng, include_target=True)
        
        if len(X_pos) < 100:
            print(f"⚠️  Skipping {position} (insufficient data: {len(X_pos)})")
            continue
        
        # Split data temporally
        split_idx = int(len(X_pos) * 0.8)
        X_test = X_pos.iloc[split_idx:]
        y_test = y_pos.iloc[split_idx:] if y_pos is not None else None
        
        if y_test is None or len(y_test) < 50:
            print(f"⚠️  Skipping {position} (insufficient test data)")
            continue
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calibration analysis
        cal_result = prod_ready.calibration_analysis(y_test.values, y_pred, position, 
                                                     n_bins=10, save_plot=True)
        calibration_results[position] = cal_result
    
    # ========================================================================
    # TASK 2: START PROBABILITY MODEL
    # ========================================================================
    print("\n" + "="*60)
    print("TASK 2: START PROBABILITY MODEL")
    print("="*60)
    print("Building model to predict rotation/benching risk...")
    print("Critical for avoiding 0pt predictions on benched players\n")
    
    # Build start probability model on full dataset
    # Need to engineer features first
    df_for_start = feature_engineer.engineer_features(df.copy(), is_training=False, lag_features=['total_points'])
    X_full, _ = feature_engineer.prepare_features(df_for_start, include_target=False)
    
    # Add back necessary columns
    df_for_start_model = X_full.copy()
    df_for_start_model['minutes'] = df['minutes'].values[:len(X_full)]
    df_for_start_model['position'] = df['position'].values[:len(X_full)]
    
    start_model = prod_ready.build_start_probability_model(df_for_start_model, minutes_threshold=60)
    
    # ========================================================================
    # TASK 3: DOUBLE GAMEWEEK FEATURES
    # ========================================================================
    print("\n" + "="*60)
    print("TASK 3: DOUBLE GAMEWEEK FEATURES")
    print("="*60)
    print("Adding DGW features (highest edge in FPL)...")
    print("Players with 2 fixtures typically score 1.7x points\n")
    
    df_with_dgw = prod_ready.add_dgw_features(df)
    
    # Save enhanced dataset
    output_path = "ml/data_with_dgw_features.csv"
    df_with_dgw.to_csv(output_path, index=False)
    print(f"\n✅ Saved enhanced dataset to {output_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("✅ ALL TASKS COMPLETE")
    print("="*60)
    
    print("\n📊 CALIBRATION SUMMARY:")
    for position, results in calibration_results.items():
        top20_bias = results['top_20_pct']['bias']
        status = "✅" if abs(top20_bias) < 0.5 else "⚠️ "
        print(f"  {position}: Top 20% bias = {top20_bias:+.2f} pts {status}")
    
    print("\n📁 FILES CREATED:")
    print("  - ml/models/calibration_*.png (calibration plots)")
    print("  - ml/models/start_probability_v1.pkl (rotation model)")
    print("  - ml/models/production_readiness_results.json (metrics)")
    print("  - ml/data_with_dgw_features.csv (enhanced dataset)")
    
    print("\n🚀 NEXT STEPS:")
    print("  1. Review calibration plots to check prediction quality")
    print("  2. Retrain models with DGW features:")
    print("     python ml/train.py --with-dgw")
    print("  3. Integrate start probability into prediction API")
    print("  4. Deploy to production!")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
