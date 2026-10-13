"""
Quantile Regression Models for Uncertainty Quantification

Predicts P10, P50, P90 percentiles for player points.
Provides confidence intervals and risk assessment.

Why Quantile Regression:
- Standard models predict mean/expected value
- Quantiles capture full distribution
- Useful for risk-averse/aggressive strategies
- Identify high-ceiling vs consistent players
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import pickle
import json
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_pinball_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantilePredictor:
    """
    Quantile regression for uncertainty quantification.
    
    Trains 3 models per position:
    - P10 (10th percentile): Pessimistic scenario
    - P50 (50th percentile): Median prediction
    - P90 (90th percentile): Optimistic scenario
    """
    
    QUANTILES = [0.10, 0.50, 0.90]
    POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    
    def __init__(self, models_dir: str = 'ml/models'):
        """
        Initialize quantile predictor.
        
        Args:
            models_dir: Directory to save/load models
        """
        self.models_dir = models_dir
        self.feature_engineer = FeatureEngineer()
        
        # Store models: {position: {quantile: model}}
        self.models = {}
        for pos in self.POSITIONS:
            self.models[pos] = {}
        
        self.is_fitted = False
    
    def train(self,
             train_df: pd.DataFrame,
             val_df: Optional[pd.DataFrame] = None,
             save_models: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Train quantile models for all positions and quantiles.
        
        Args:
            train_df: Training dataframe
            val_df: Optional validation dataframe
            save_models: Whether to save trained models
            
        Returns:
            Dictionary of metrics per position and quantile
        """
        logger.info("Training quantile regression models...")
        logger.info(f"Quantiles: {self.QUANTILES}")
        logger.info(f"Total models: {len(self.POSITIONS)} positions × {len(self.QUANTILES)} quantiles = {len(self.POSITIONS) * len(self.QUANTILES)}")
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Engineer features
        train_df = self.feature_engineer.engineer_features(
            train_df,
            is_training=True,
            lag_features=['total_points']
        )
        
        if val_df is not None:
            val_df = self.feature_engineer.engineer_features(
                val_df,
                is_training=False,
                lag_features=['total_points']
            )
        
        metrics = {}
        
        for position in self.POSITIONS:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training models for position: {position}")
            logger.info(f"{'='*60}")
            
            # Filter to position
            train_pos = train_df[train_df['position'] == position].copy()
            val_pos = val_df[val_df['position'] == position].copy() if val_df is not None else None
            
            if len(train_pos) == 0:
                logger.warning(f"No training data for {position}, skipping")
                continue
            
            # Prepare features
            X_train, y_train = self.feature_engineer.prepare_features(train_pos, include_target=True)
            
            if val_pos is not None and len(val_pos) > 0:
                X_val, y_val = self.feature_engineer.prepare_features(val_pos, include_target=True)
            else:
                X_val, y_val = None, None
            
            logger.info(f"Training data: {len(X_train)} samples, {len(X_train.columns)} features")
            
            position_metrics = {}
            
            # Train model for each quantile
            for quantile in self.QUANTILES:
                quantile_metrics = self._train_quantile_model(
                    position, quantile,
                    X_train, y_train,
                    X_val, y_val
                )
                position_metrics[f'q{int(quantile*100)}'] = quantile_metrics
            
            metrics[position] = position_metrics
            
            # Save models for this position
            if save_models:
                self._save_position_models(position)
        
        self.is_fitted = True
        
        # Save feature mappings
        if save_models:
            mappings_path = os.path.join(self.models_dir, 'quantile_model_mappings.json')
            self.feature_engineer.save_mappings(mappings_path)
            logger.info(f"Saved feature mappings to {mappings_path}")
        
        # Save overall metrics
        if save_models:
            metrics_path = os.path.join(self.models_dir, 'quantile_models_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_path}")
        
        return metrics
    
    def _train_quantile_model(self,
                             position: str,
                             quantile: float,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_val: Optional[pd.DataFrame],
                             y_val: Optional[pd.Series]) -> Dict[str, float]:
        """
        Train a single quantile model.
        
        Args:
            position: Position (GK/DEF/MID/FWD)
            quantile: Quantile to predict (0.10, 0.50, 0.90)
            X_train, y_train: Training data
            X_val, y_val: Optional validation data
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"\n  Training Q{int(quantile*100)} model...")
        
        # XGBoost with quantile objective
        params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': quantile,
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',
            'enable_categorical': False,
            'early_stopping_rounds': 30 if X_val is not None else None
        }
        
        model = xgb.XGBRegressor(**params)
        
        eval_set = [(X_train.values, y_train.values)]
        if X_val is not None:
            eval_set.append((X_val.values, y_val.values))
        
        model.fit(
            X_train.values, y_train.values,
            eval_set=eval_set,
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_train.values)
        train_pinball = mean_pinball_loss(y_train, y_pred, alpha=quantile)
        train_mae = mean_absolute_error(y_train, y_pred)
        
        metrics = {
            'quantile': quantile,
            'train_pinball_loss': float(train_pinball),
            'train_mae': float(train_mae),
            'train_samples': len(X_train)
        }
        
        if X_val is not None:
            y_val_pred = model.predict(X_val.values)
            val_pinball = mean_pinball_loss(y_val, y_val_pred, alpha=quantile)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            
            metrics['val_pinball_loss'] = float(val_pinball)
            metrics['val_mae'] = float(val_mae)
            metrics['val_samples'] = len(X_val)
            
            logger.info(f"    Train Pinball Loss: {train_pinball:.4f}, Val: {val_pinball:.4f}")
            logger.info(f"    MAE: {train_mae:.3f}, Val MAE: {val_mae:.3f}")
        else:
            logger.info(f"    Train Pinball Loss: {train_pinball:.4f}, MAE: {train_mae:.3f}")
        
        # Store model
        self.models[position][quantile] = model
        
        return metrics
    
    def predict(self, player_data: Dict) -> Dict[str, float]:
        """
        Predict quantiles for a single player.
        
        Args:
            player_data: Player feature dictionary
            
        Returns:
            Dictionary with p10, p50, p90 predictions
        """
        if not self.is_fitted:
            raise ValueError("Models not trained. Call train() or load() first.")
        
        position = player_data.get('position', 'MID')
        
        if position not in self.models or not self.models[position]:
            logger.warning(f"No models for position {position}, using fallback")
            # Fallback to position averages
            baselines = {'GK': 3.2, 'DEF': 3.8, 'MID': 4.1, 'FWD': 4.5}
            baseline = baselines.get(position, 4.0)
            return {
                'p10': baseline * 0.5,
                'p50': baseline,
                'p90': baseline * 1.8,
                'uncertainty': baseline * 0.65
            }
        
        # Convert to dataframe
        df = pd.DataFrame([player_data])
        
        # Engineer features
        df = self.feature_engineer.engineer_features(df, is_training=False, lag_features=None)
        
        # Prepare features
        X, _ = self.feature_engineer.prepare_features(df, include_target=False)
        
        # Predict each quantile
        predictions = {}
        for quantile in self.QUANTILES:
            if quantile in self.models[position]:
                model = self.models[position][quantile]
                pred = model.predict(X.values)[0]
                predictions[f'p{int(quantile*100)}'] = float(max(0, pred))
        
        # Calculate uncertainty (P90 - P10)
        if 'p10' in predictions and 'p90' in predictions:
            predictions['uncertainty'] = predictions['p90'] - predictions['p10']
        
        # Add interpretation
        if 'p50' in predictions and 'uncertainty' in predictions:
            # High uncertainty: wide range of outcomes
            # Low uncertainty: consistent performer
            predictions['risk_profile'] = (
                'high_risk' if predictions['uncertainty'] > 6 else
                'moderate_risk' if predictions['uncertainty'] > 3 else
                'low_risk'
            )
        
        return predictions
    
    def predict_batch(self, players_data: List[Dict]) -> pd.DataFrame:
        """
        Predict quantiles for multiple players.
        
        Args:
            players_data: List of player dictionaries
            
        Returns:
            DataFrame with columns: name, position, p10, p50, p90, uncertainty
        """
        results = []
        
        for player_data in players_data:
            preds = self.predict(player_data)
            preds['name'] = player_data.get('name', 'Unknown')
            preds['position'] = player_data.get('position', 'MID')
            results.append(preds)
        
        return pd.DataFrame(results)
    
    def _save_position_models(self, position: str):
        """Save quantile models for a position."""
        for quantile, model in self.models[position].items():
            filename = f'quantile_{position.lower()}_q{int(quantile*100)}_v1.pkl'
            path = os.path.join(self.models_dir, filename)
            
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"  Saved {filename}")
    
    def load(self, position: Optional[str] = None):
        """
        Load trained quantile models.
        
        Args:
            position: Specific position to load, or None for all
        """
        positions_to_load = [position] if position else self.POSITIONS
        
        for pos in positions_to_load:
            for quantile in self.QUANTILES:
                filename = f'quantile_{pos.lower()}_q{int(quantile*100)}_v1.pkl'
                path = os.path.join(self.models_dir, filename)
                
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.models[pos][quantile] = pickle.load(f)
                    logger.info(f"Loaded {filename}")
        
        # Load feature mappings
        mappings_path = os.path.join(self.models_dir, 'quantile_model_mappings.json')
        if os.path.exists(mappings_path):
            self.feature_engineer.load_mappings(mappings_path)
            logger.info("Loaded feature mappings")
        
        self.is_fitted = True


def main():
    """Train and save quantile models."""
    logger.info("="*80)
    logger.info("QUANTILE REGRESSION MODELS TRAINING")
    logger.info("="*80)
    
    # Load FPL data
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
    
    if not dfs:
        logger.error("No data files found")
        return 1
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} records")
    
    # Clean data
    combined['kickoff_time'] = pd.to_datetime(combined['kickoff_time'])
    combined = combined.dropna(subset=['minutes', 'position', 'total_points'])
    combined = combined.sort_values('kickoff_time').reset_index(drop=True)
    
    # Temporal split
    n = len(combined)
    train_end = int(n * 0.72)
    val_end = int(n * 0.80)
    
    train_df = combined.iloc[:train_end].copy()
    val_df = combined.iloc[train_end:val_end].copy()
    test_df = combined.iloc[val_end:].copy()
    
    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_df):,} samples")
    logger.info(f"  Val:   {len(val_df):,} samples")
    logger.info(f"  Test:  {len(test_df):,} samples")
    
    # Train models
    predictor = QuantilePredictor()
    
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
        for quantile_name, q_metrics in pos_metrics.items():
            logger.info(f"  {quantile_name}:")
            logger.info(f"    Pinball Loss: {q_metrics.get('train_pinball_loss', 0):.4f}")
            logger.info(f"    MAE: {q_metrics.get('train_mae', 0):.3f}")
    
    logger.info(f"\n✅ Saved {len(metrics) * 3} models to ml/models/")
    logger.info(f"✅ Models: quantile_{{position}}_q{{10|50|90}}_v1.pkl")
    
    return 0


if __name__ == '__main__':
    try:
        exit(main())
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        exit(1)
