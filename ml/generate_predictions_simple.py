"""
Generate Predictions Using Retrained Models
Simple version that generates sample predictions
"""
import pandas as pd
import numpy as np
import pickle
import os
import logging
from ml.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate sample predictions."""
    logger.info("="*80)
    logger.info("GENERATING PREDICTIONS WITH RETRAINED MODELS")
    logger.info("="*80)
    
    # Load trained models
    models = {}
    feature_engineer = FeatureEngineer()
    
    positions = ['GK', 'DEF', 'MID', 'FWD']
    for pos in positions:
        model_path = f"ml/models/xgboost_{pos.lower()}_v2.pkl"
        mappings_path = f"ml/models/xgboost_{pos.lower()}_v2_mappings.json"
        
        with open(model_path, 'rb') as f:
            models[pos] = pickle.load(f)
        
        if pos == 'GK':  # Load mappings once
            feature_engineer.load_mappings(mappings_path)
        
        logger.info(f"✅ Loaded {pos} model")
    
    # Load historical data to get latest player stats
    df = pd.read_csv('data/cleaned_merged_seasons_cleaned.csv')
    
    # Get latest season data (2025-26)
    latest_season = df[df['season_x'] == '2025-26'].copy()
    
    # Get last game for each player
    latest_season = latest_season.sort_values('kickoff_time')
    last_games = latest_season.groupby('name').last().reset_index()
    
    logger.info(f"\nGenerating predictions for {len(last_games)} players...")
    
    predictions = []
    
    for _, row in last_games.iterrows():
        try:
            position = row['position']
            if position not in models:
                continue
            
            # Prepare single player dataframe
            player_df = pd.DataFrame([row])
            
            # Engineer features
            player_engineered = feature_engineer.engineer_features(
                player_df,
                is_training=False,
                lag_features=None
            )
            
            # Prepare features
            X, _ = feature_engineer.prepare_features(
                player_engineered,
                include_target=False
            )
            
            # Predict
            prediction = models[position].predict(X)[0]
            prediction = max(0, prediction)
            
            predictions.append({
                'name': row['name'],
                'position': position,
                'team': row.get('team_x', 'Unknown'),
                'predicted_points': round(prediction, 2),
                'last_gw_points': row.get('total_points', 0),
                'form': row.get('form', 0) if not pd.isna(row.get('form')) else 0,
                'minutes': row.get('minutes', 0)
            })
            
        except Exception as e:
            logger.warning(f"Error for {row.get('name')}: {e}")
            continue
    
    # Create DataFrame
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values('predicted_points', ascending=False)
    
    # Save predictions
    os.makedirs('ml/predictions', exist_ok=True)
    csv_path = 'ml/predictions/next_gw_predictions_v2.csv'
    predictions_df.to_csv(csv_path, index=False)
    
    json_path = 'ml/predictions/next_gw_predictions_v2.json'
    predictions_df.to_json(json_path, orient='records', indent=2)
    
    logger.info(f"\n✅ Saved {len(predictions_df)} predictions")
    logger.info(f"   CSV: {csv_path}")
    logger.info(f"   JSON: {json_path}")
    
    # Display top predictions by position
    logger.info("\n" + "="*80)
    logger.info("TOP 10 PREDICTED PERFORMERS BY POSITION")
    logger.info("="*80)
    
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        pos_df = predictions_df[predictions_df['position'] == position].head(10)
        logger.info(f"\n{position}:")
        for _, player in pos_df.iterrows():
            logger.info(
                f"  {player['predicted_points']:5.2f} pts | "
                f"{player['name']:25s} | {player['team']:20s} | "
                f"Form: {player['form']:.1f}"
            )
    
    logger.info("\n" + "="*80)
    logger.info("✅ PREDICTIONS COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
