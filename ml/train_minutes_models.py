"""
Training script for Minutes Models (Model C & D)
Trains P(start) and E[minutes|start] models per position
"""
import os
import sys
import pandas as pd
import json
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.minutes_model import MinutesPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_fpl_data() -> pd.DataFrame:
    """
    Load FPL data from CSV files.
    
    Returns:
        Combined dataframe from all seasons
    """
    data_files = [
        'data/FPL_2023_2024.csv',
        'data/FPL_2024_2025.csv',
        'data/FPL_2025_2026.csv'
    ]
    
    dfs = []
    for file in data_files:
        if os.path.exists(file):
            logger.info(f"Loading {file}...")
            df = pd.read_csv(file)
            dfs.append(df)
        else:
            logger.warning(f"File not found: {file}")
    
    if not dfs:
        raise FileNotFoundError("No data files found. Please ensure FPL CSV files exist.")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} total records from {len(dfs)} files")
    
    return combined


def temporal_split(df: pd.DataFrame, 
                   train_pct: float = 0.72,
                   val_pct: float = 0.08) -> tuple:
    """
    Split data temporally (chronological order).
    
    Args:
        df: Input dataframe
        train_pct: Percentage for training (default 72%)
        val_pct: Percentage for validation (default 8%)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Sort by kickoff time
    df = df.sort_values('kickoff_time').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Temporal split:")
    logger.info(f"  Train: {len(train_df):,} samples ({train_pct:.0%})")
    logger.info(f"  Val:   {len(val_df):,} samples ({val_pct:.0%})")
    logger.info(f"  Test:  {len(test_df):,} samples ({1-train_pct-val_pct:.0%})")
    
    if 'kickoff_time' in df.columns:
        logger.info(f"  Train period: {train_df['kickoff_time'].min()} to {train_df['kickoff_time'].max()}")
        logger.info(f"  Val period:   {val_df['kickoff_time'].min()} to {val_df['kickoff_time'].max()}")
        logger.info(f"  Test period:  {test_df['kickoff_time'].min()} to {test_df['kickoff_time'].max()}")
    
    return train_df, val_df, test_df


def main():
    """Main training pipeline."""
    logger.info("="*80)
    logger.info("TRAINING MINUTES MODELS (Model C & D)")
    logger.info("="*80)
    
    # Load data
    logger.info("\n1. Loading FPL data...")
    df = load_fpl_data()
    
    # Basic cleaning
    logger.info("\n2. Cleaning data...")
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    df = df.dropna(subset=['minutes', 'position'])
    
    # Add season_x if missing (for feature engineering compatibility)
    if 'season_x' not in df.columns and 'season' in df.columns:
        df['season_x'] = df['season']
    
    logger.info(f"   {len(df):,} records after cleaning")
    
    # Temporal split
    logger.info("\n3. Splitting data temporally...")
    train_df, val_df, test_df = temporal_split(df)
    
    # Initialize predictor
    logger.info("\n4. Initializing MinutesPredictor...")
    predictor = MinutesPredictor(models_dir='ml/models')
    
    # Train models
    logger.info("\n5. Training Model C (P(start)) and Model D (E[minutes|start])...")
    logger.info("   This will train 8 models total (4 positions × 2 models)")
    
    metrics = predictor.train(
        train_df=train_df,
        val_df=val_df,
        save_models=True
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    
    for position, pos_metrics in metrics.items():
        logger.info(f"\n{position}:")
        
        if 'model_c' in pos_metrics:
            mc = pos_metrics['model_c']
            logger.info(f"  Model C (P(start)):")
            logger.info(f"    Train AUC: {mc.get('train_auc', 0):.4f}")
            if 'val_auc' in mc:
                logger.info(f"    Val AUC:   {mc['val_auc']:.4f}")
            logger.info(f"    Start rate: {mc.get('start_rate', 0):.2%}")
        
        if 'model_d' in pos_metrics:
            md = pos_metrics['model_d']
            logger.info(f"  Model D (E[minutes|start]):")
            logger.info(f"    Train RMSE: {md.get('train_rmse', 0):.2f} min")
            if 'val_rmse' in md:
                logger.info(f"    Val RMSE:   {md['val_rmse']:.2f} min")
            logger.info(f"    Mean minutes: {md.get('mean_minutes', 0):.1f}")
    
    # Save metrics
    metrics_path = 'ml/models/minutes_models_metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        serializable_metrics = {}
        for pos, pos_met in metrics.items():
            serializable_metrics[pos] = {}
            for model_type, model_met in pos_met.items():
                serializable_metrics[pos][model_type] = {
                    k: float(v) if isinstance(v, (float, int)) else v
                    for k, v in model_met.items()
                }
        
        json.dump(serializable_metrics, f, indent=2)
    
    logger.info(f"\n✅ Metrics saved to {metrics_path}")
    logger.info(f"✅ Models saved to ml/models/minutes_*.pkl")
    
    # Test inference on a sample
    logger.info("\n6. Testing inference on sample player...")
    sample_player = train_df[train_df['position'] == 'FWD'].iloc[0].to_dict()
    
    try:
        prediction = predictor.predict(sample_player)
        logger.info(f"   Sample prediction:")
        logger.info(f"     P(start): {prediction['p_start']:.3f}")
        logger.info(f"     E[minutes|start]: {prediction['e_minutes_given_start']:.1f}")
        logger.info(f"     Expected minutes: {prediction['expected_minutes']:.1f}")
    except Exception as e:
        logger.error(f"   Inference test failed: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("DONE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
