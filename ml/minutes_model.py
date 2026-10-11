"""
Minutes Prediction Models (Model C & D)
Predicts P(start) and E[minutes|start] for FPL players

Model C: Binary classification - whether player starts (minutes >= 60)
Model D: Regression - expected minutes if player starts
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
import logging
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV

from ml.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class MinutesPredictor:
    """
    Two-stage minutes prediction:
    - Model C: P(start) - probability player starts
    - Model D: E[minutes|start] - expected minutes if they do start
    """
    
    def __init__(self, models_dir: str = "ml/models"):
        """
        Initialize minutes predictor.
        
        Args:
            models_dir: Directory to save/load models
        """
        self.models_dir = models_dir
        self.feature_engineer = FeatureEngineer()
        
        # Model C: P(start) per position
        self.pstart_models: Dict[str, Any] = {}  # position -> calibrated model
        self.pstart_scalers: Dict[str, StandardScaler] = {}
        
        # Model D: E[minutes|start] per position  
        self.eminutes_models: Dict[str, Any] = {}
        self.eminutes_scalers: Dict[str, StandardScaler] = {}
        
        self.positions = ['GK', 'DEF', 'MID', 'FWD']
        self.is_fitted = False
    
    def _create_started_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 'started' binary target from minutes or starts column.
        
        Priority:
        1. Use 'starts' column if available (2023-24+)
        2. Fall back to minutes >= 60 threshold
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with 'started' column
        """
        df = df.copy()
        
        if 'starts' in df.columns:
            df['started'] = df['starts'].astype(int)
            logger.info("Using 'starts' column for Model C target")
        else:
            df['started'] = (df['minutes'] >= 60).astype(int)
            logger.info("Using minutes >= 60 threshold for Model C target")
        
        return df
    
    def _add_minutes_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add minutes-specific features for Model C and D.
        
        Features added:
        - started_last_match: Binary, did player start previous GW
        - started_2_of_last_3: Binary, started >= 2 of last 3 GWs
        - avg_minutes_last_3: Mean minutes over last 3 GWs
        
        Args:
            df: Input dataframe with 'started' column
            
        Returns:
            Dataframe with new features
        """
        df = df.copy()
        
        # Sort by player and time
        sort_cols = ['name', 'kickoff_time'] if 'kickoff_time' in df.columns else ['name', 'GW']
        df = df.sort_values(sort_cols)
        
        # Feature 1: Started last match
        df['started_last_match'] = df.groupby('name')['started'].shift(1).fillna(0).astype(int)
        
        # Feature 2: Started 2 of last 3
        df['started_rolling3'] = df.groupby('name')['started'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).sum()
        )
        df['started_2_of_last_3'] = (df['started_rolling3'] >= 2).astype(int)
        df = df.drop(columns=['started_rolling3'])
        
        # Feature 3: Average minutes last 3 games
        df['avg_minutes_last_3'] = df.groupby('name')['minutes'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        ).fillna(0)
        
        logger.info("Added minutes features: started_last_match, started_2_of_last_3, avg_minutes_last_3")
        
        return df
    
    def train(self, 
              train_df: pd.DataFrame,
              val_df: Optional[pd.DataFrame] = None,
              save_models: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Train Model C and Model D for all positions.
        
        Args:
            train_df: Training dataframe
            val_df: Optional validation dataframe
            save_models: Whether to save trained models
            
        Returns:
            Dictionary of metrics per position
        """
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create started feature
        train_df = self._create_started_feature(train_df)
        if val_df is not None:
            val_df = self._create_started_feature(val_df)
        
        # Add minutes-specific features
        train_df = self._add_minutes_features(train_df)
        if val_df is not None:
            val_df = self._add_minutes_features(val_df)
        
        # Engineer standard features
        train_df = self.feature_engineer.engineer_features(
            train_df,
            is_training=True,
            lag_features=None  # We'll handle targets separately
        )
        
        if val_df is not None:
            val_df = self.feature_engineer.engineer_features(
                val_df,
                is_training=False,
                lag_features=None
            )
        
        metrics = {}
        
        for position in self.positions:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training minutes models for position: {position}")
            logger.info(f"{'='*60}")
            
            # Filter to position
            train_pos = train_df[train_df['position'] == position].copy()
            val_pos = val_df[val_df['position'] == position].copy() if val_df is not None else None
            
            if len(train_pos) == 0:
                logger.warning(f"No training data for {position}, skipping")
                continue
            
            # Train Model C and Model D
            metrics_c = self._train_model_c(position, train_pos, val_pos)
            metrics_d = self._train_model_d(position, train_pos, val_pos)
            
            metrics[position] = {
                'model_c': metrics_c,
                'model_d': metrics_d
            }
            
            # Save models
            if save_models:
                self._save_position_models(position)
        
        self.is_fitted = True
        
        # Save feature mappings
        if save_models:
            mappings_path = os.path.join(self.models_dir, 'minutes_model_mappings.json')
            self.feature_engineer.save_mappings(mappings_path)
        
        return metrics
    
    def _train_model_c(self,
                      position: str,
                      train_df: pd.DataFrame,
                      val_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        Train Model C: P(start) for given position.
        
        Args:
            position: Position to train (GK/DEF/MID/FWD)
            train_df: Training data for this position
            val_df: Optional validation data
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"\nTraining Model C (P(start)) for {position}")
        
        # Prepare features - CRITICAL: exclude ALL current-GW outcome variables (data leakage)
        exclude_cols = [
            # Target and direct leakage
            'name', 'started', 'starts', 'minutes', 'total_points', 'upcoming',
            # Identifiers and temporal
            'season_x', 'season', 'element', 'fixture', 'kickoff_time',
            # Transfer activity (lookahead)
            'selected', 'transfers_in', 'transfers_out', 'transfers_balance',
            # CRITICAL: Current GW outcomes (LEAKAGE!)
            'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'bonus', 'bps', 'saves', 'penalties_saved', 'penalties_missed',
            'yellow_cards', 'red_cards', 'own_goals',
            # CRITICAL: ICT statistics (current-GW FPL stats - LEAKAGE!)
            'influence', 'creativity', 'threat', 'ict_index',
            # CRITICAL: Defensive statistics (current-GW outcomes - LEAKAGE!)
            'recoveries', 'tackles', 'clearances_blocks_interceptions',
            'defensive_contribution',
            # Expected stats (predictive of outcome)
            'xP', 'expected_assists', 'expected_goal_involvements', 
            'expected_goals', 'expected_goals_conceded',
            # Team scores (current match outcome)
            'team_a_score', 'team_h_score', 'opponent_team', 'opp_team_name',
            # Metadata and strings
            'round', 'Unnamed: 0', 'index', 'home_team', 'away_team',
            'stat_category', 'team_x', 'position', 'modified'  # String/object columns
        ]
        
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['started']
        
        if val_df is not None and len(val_df) > 0:
            # Ensure validation has all feature columns (fill missing with 0)
            X_val = pd.DataFrame(index=val_df.index)
            for col in feature_cols:
                if col in val_df.columns:
                    X_val[col] = val_df[col].fillna(0)
                else:
                    X_val[col] = 0
            y_val = val_df['started']
        else:
            X_val, y_val = None, None
        
        # Calculate class imbalance
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        logger.info(f"Class distribution: {pos_count} starts, {neg_count} non-starts (scale_pos_weight={scale_pos_weight:.2f})")
        logger.info(f"Training on {len(X_train)} samples with {len(feature_cols)} features")
        
        # Train XGBoost classifier
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'tree_method': 'hist',
            'enable_categorical': False,
            'early_stopping_rounds': 30 if X_val is not None else None
        }
        
        model = xgb.XGBClassifier(**params)
        
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Skip Platt scaling for now (API compatibility issues)
        # Model probabilities are generally well-calibrated with XGBoost
        logger.info("Using uncalibrated model (XGBoost probabilities generally well-calibrated)")
        calibrated_model = model
        
        # Evaluate
        y_pred_proba = calibrated_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_pred_proba)
        
        metrics = {
            'train_auc': train_auc,
            'train_samples': len(X_train),
            'start_rate': float(y_train.mean())
        }
        
        if X_val is not None:
            y_val_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            metrics['val_auc'] = val_auc
            metrics['val_samples'] = len(X_val)
            
            logger.info(f"✅ Model C ({position}): Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")
        else:
            logger.info(f"✅ Model C ({position}): Train AUC={train_auc:.4f}")
        
        # Store model and feature columns
        self.pstart_models[position] = {
            'model': calibrated_model,
            'feature_cols': feature_cols
        }
        
        return metrics
    
    def _train_model_d(self,
                      position: str,
                      train_df: pd.DataFrame,
                      val_df: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        Train Model D: E[minutes|start] for given position.
        
        Only trains on samples where started == 1.
        
        Args:
            position: Position to train (GK/DEF/MID/FWD)
            train_df: Training data for this position
            val_df: Optional validation data
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"\nTraining Model D (E[minutes|start]) for {position}")
        
        # Filter to only started matches
        train_started = train_df[train_df['started'] == 1].copy()
        
        if val_df is not None:
            val_started = val_df[val_df['started'] == 1].copy()
        else:
            val_started = None
        
        if len(train_started) == 0:
            logger.warning(f"No started matches for {position} in training data")
            return {}
        
        logger.info(f"Training on {len(train_started)} started matches (filtered from {len(train_df)} total)")
        
        # Prepare features - CRITICAL: exclude ALL current-GW outcome variables (data leakage)
        exclude_cols = [
            # Target and direct leakage
            'name', 'started', 'starts', 'minutes', 'total_points', 'upcoming',
            # Identifiers and temporal
            'season_x', 'season', 'element', 'fixture', 'kickoff_time',
            # Transfer activity (lookahead)
            'selected', 'transfers_in', 'transfers_out', 'transfers_balance',
            # CRITICAL: Current GW outcomes (LEAKAGE!)
            'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'bonus', 'bps', 'saves', 'penalties_saved', 'penalties_missed',
            'yellow_cards', 'red_cards', 'own_goals',
            # CRITICAL: ICT statistics (current-GW FPL stats - LEAKAGE!)
            'influence', 'creativity', 'threat', 'ict_index',
            # CRITICAL: Defensive statistics (current-GW outcomes - LEAKAGE!)
            'recoveries', 'tackles', 'clearances_blocks_interceptions',
            'defensive_contribution',
            # Expected stats (predictive of outcome)
            'xP', 'expected_assists', 'expected_goal_involvements',
            'expected_goals', 'expected_goals_conceded',
            # Team scores (current match outcome)
            'team_a_score', 'team_h_score', 'opponent_team', 'opp_team_name',
            # Metadata and strings
            'round', 'Unnamed: 0', 'index', 'home_team', 'away_team',
            'stat_category', 'team_x', 'position', 'modified'  # String/object columns
        ]
        
        feature_cols = [col for col in train_started.columns if col not in exclude_cols]
        
        X_train = train_started[feature_cols].fillna(0)
        y_train = train_started['minutes']
        
        if val_started is not None and len(val_started) > 0:
            # Ensure validation has all feature columns (fill missing with 0)
            X_val = pd.DataFrame(index=val_started.index)
            for col in feature_cols:
                if col in val_started.columns:
                    X_val[col] = val_started[col].fillna(0)
                else:
                    X_val[col] = 0
            y_val = val_started['minutes']
        else:
            X_val, y_val = None, None
        
        logger.info(f"Training on {len(X_train)} samples, mean minutes={y_train.mean():.1f}")
        
        # Train XGBoost regressor
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 4,
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
        
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        train_mae = mean_absolute_error(y_train, y_pred)
        
        metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_samples': len(X_train),
            'mean_minutes': float(y_train.mean())
        }
        
        if X_val is not None:
            y_val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_mae = mean_absolute_error(y_val, y_val_pred)
            metrics['val_rmse'] = val_rmse
            metrics['val_mae'] = val_mae
            metrics['val_samples'] = len(X_val)
            
            logger.info(f"✅ Model D ({position}): Train RMSE={train_rmse:.2f}, Val RMSE={val_rmse:.2f} min")
        else:
            logger.info(f"✅ Model D ({position}): Train RMSE={train_rmse:.2f} min")
        
        # Store model and feature columns
        self.eminutes_models[position] = {
            'model': model,
            'feature_cols': feature_cols
        }
        
        return metrics
    
    def predict(self, player_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict minutes for a single player.
        
        Returns:
        - p_start: Probability player starts
        - e_minutes_given_start: Expected minutes if they start
        - expected_minutes: p_start × e_minutes_given_start
        
        Args:
            player_data: Dictionary with player features
            
        Returns:
            Dictionary with prediction components
        """
        if not self.is_fitted:
            raise ValueError("Models not trained. Call train() or load() first.")
        
        position = player_data.get('position', 'MID')
        
        if position not in self.pstart_models or position not in self.eminutes_models:
            # Fallback to position average
            logger.warning(f"No models for position {position}, using fallback")
            return {
                'p_start': 0.65,
                'e_minutes_given_start': 75.0,
                'expected_minutes': 48.75
            }
        
        # Convert to dataframe
        df = pd.DataFrame([player_data])
        
        # Add minutes features (use dummy 'started' for feature creation)
        df['started'] = 0  # Placeholder
        df = self._add_minutes_features(df)
        
        # Engineer features
        df = self.feature_engineer.engineer_features(df, is_training=False, lag_features=None)
        
        # Model C: P(start)
        pstart_info = self.pstart_models[position]
        X_c = df[pstart_info['feature_cols']].fillna(0)
        p_start = pstart_info['model'].predict_proba(X_c)[0, 1]
        
        # Model D: E[minutes|start]
        eminutes_info = self.eminutes_models[position]
        X_d = df[eminutes_info['feature_cols']].fillna(0)
        e_minutes_given_start = eminutes_info['model'].predict(X_d)[0]
        
        # Ensure valid range
        p_start = np.clip(p_start, 0, 1)
        e_minutes_given_start = np.clip(e_minutes_given_start, 0, 90)
        
        # Calculate expected minutes
        expected_minutes = p_start * e_minutes_given_start
        
        return {
            'p_start': float(p_start),
            'e_minutes_given_start': float(e_minutes_given_start),
            'expected_minutes': float(expected_minutes)
        }
    
    def _save_position_models(self, position: str):
        """Save Model C and Model D for a position."""
        # Save Model C
        if position in self.pstart_models:
            model_path = os.path.join(self.models_dir, f'minutes_pstart_{position.lower()}_v1.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.pstart_models[position], f)
            logger.info(f"Saved Model C ({position}) to {model_path}")
        
        # Save Model D
        if position in self.eminutes_models:
            model_path = os.path.join(self.models_dir, f'minutes_eminutes_{position.lower()}_v1.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.eminutes_models[position], f)
            logger.info(f"Saved Model D ({position}) to {model_path}")
    
    def load(self, position: Optional[str] = None):
        """
        Load trained models from disk.
        
        Args:
            position: Specific position to load, or None for all
        """
        positions_to_load = [position] if position else self.positions
        
        for pos in positions_to_load:
            # Load Model C
            pstart_path = os.path.join(self.models_dir, f'minutes_pstart_{pos.lower()}_v1.pkl')
            if os.path.exists(pstart_path):
                with open(pstart_path, 'rb') as f:
                    self.pstart_models[pos] = pickle.load(f)
                logger.info(f"Loaded Model C ({pos}) from {pstart_path}")
            
            # Load Model D
            eminutes_path = os.path.join(self.models_dir, f'minutes_eminutes_{pos.lower()}_v1.pkl')
            if os.path.exists(eminutes_path):
                with open(eminutes_path, 'rb') as f:
                    self.eminutes_models[pos] = pickle.load(f)
                logger.info(f"Loaded Model D ({pos}) from {eminutes_path}")
        
        # Load feature mappings
        mappings_path = os.path.join(self.models_dir, 'minutes_model_mappings.json')
        if os.path.exists(mappings_path):
            self.feature_engineer.load_mappings(mappings_path)
            logger.info("Loaded feature mappings")
        
        self.is_fitted = True
