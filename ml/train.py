"""
FPL ML Model Training Script
Implements improvements from milestone_1 notebook
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
import logging
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

# TensorFlow import with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential, Input
    from tensorflow.keras.metrics import Precision, Recall
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("WARNING: TensorFlow not available. Neural network model will be skipped.")

# XGBoost import with error handling
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: XGBoost not available. Gradient boosting model will be skipped.")

from ml.feature_engineering import FeatureEngineer
from ml.predictor import FPLPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FPLModelTrainer:
    """
    Trainer for FPL prediction models.
    
    IMPROVEMENTS FROM REMARKS:
    1. ✅ Temporal train/test split (most impactful)
    2. ✅ Remove total_points and bps from features (data leakage)
    3. ✅ Add Dropout to Neural Network
    4. ✅ Fix position labels in reports
    5. ✅ Rename nn_bad_model to nn_baseline_model
    6. ✅ Split by Position (BIGGEST SINGLE WIN - expect +0.05-0.15 R²)
    7. ✅ Handle xP column properly (prevent lookahead bias)
    """
    
    # Fix position labels (IMPROVEMENT 4)
    POSITION_NAMES = {
        "GK": "Goalkeepers (GK)",
        "DEF": "Defenders (DEF)",
        "MID": "Midfielders (MID)",  # FIX: was "Forwards (MID)"
        "FWD": "Forwards (FWD)"
    }
    
    def __init__(self, data_path: str):
        """
        Initialize trainer.
        
        Args:
            data_path: Path to dataset CSV
        """
        self.data_path = data_path
        self.df_raw = None
        self.df_processed = None
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate dataset."""
        logger.info(f"Loading data from {self.data_path}")
        self.df_raw = pd.read_csv(self.data_path, low_memory=False)
        logger.info(f"Loaded {len(self.df_raw)} records with {len(self.df_raw.columns)} columns")
        
        # Basic validation
        required_cols = ['name', 'position', 'total_points', 'kickoff_time']
        missing = [col for col in required_cols if col not in self.df_raw.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return self.df_raw
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess data with feature engineering.
        
        IMPROVEMENT 1: Temporal split - sort by kickoff_time
        IMPROVEMENT 2: Remove total_points and bps from features
        """
        logger.info("Preprocessing data...")
        
        # Fit feature engineer
        self.feature_engineer.fit(self.df_raw)
        
        # Engineer features with lagged target (IMPROVEMENT 2)
        self.df_processed = self.feature_engineer.engineer_features(
            self.df_raw,
            is_training=True,
            lag_features=['total_points']  # Creates 'upcoming' target
        )
        
        logger.info(f"Processed {len(self.df_processed)} records")
        logger.info(f"Features: {len(self.df_processed.columns)} columns")
        
        return self.df_processed
    
    def temporal_train_test_split_raw(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        IMPROVEMENT 1: Temporal train/test split on RAW data (before feature engineering).
        
        Split data based on time order to prevent data leakage.
        Train on earlier data, validate on middle period, test on recent data.
        
        Args:
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df) - RAW dataframes before preprocessing
        """
        logger.info("Performing temporal train/test split on raw data...")
        
        # Use raw data
        df = self.df_raw.copy()
        
        # Ensure kickoff_time is datetime
        if 'kickoff_time' in df.columns:
            df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
        
        # Sort by kickoff_time (temporal ordering)
        df_sorted = df.sort_values('kickoff_time').reset_index(drop=True)
        
        n = len(df_sorted)
        test_start = int(n * (1 - test_size))
        train_end = int(test_start * (1 - validation_size))
        
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:test_start].copy()
        test_df = df_sorted.iloc[test_start:].copy()
        
        logger.info(f"Train set: {len(train_df)} records (earliest data)")
        logger.info(f"Validation set: {len(val_df)} records (middle period)")
        logger.info(f"Test set: {len(test_df)} records (most recent data)")
        
        # Log date ranges
        for name, df_split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            if len(df_split) > 0 and 'kickoff_time' in df_split.columns:
                min_date = df_split['kickoff_time'].min()
                max_date = df_split['kickoff_time'].max()
                logger.info(f"{name} date range: {min_date} to {max_date}")
        
        return train_df, val_df, test_df
    
    def temporal_train_test_split(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        IMPROVEMENT 1: Temporal train/test split (most impactful fix).
        
        Split data based on time order to prevent data leakage.
        Train on earlier data, validate on middle period, test on recent data.
        
        Args:
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Performing temporal train/test split...")
        
        # Sort by kickoff_time (temporal ordering)
        df_sorted = self.df_processed.sort_values('kickoff_time').reset_index(drop=True)
        
        n = len(df_sorted)
        test_start = int(n * (1 - test_size))
        train_end = int(test_start * (1 - validation_size))
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:test_start]
        test_df = df_sorted.iloc[test_start:]
        
        logger.info(f"Train set: {len(train_df)} records (earliest data)")
        logger.info(f"Validation set: {len(val_df)} records (middle period)")
        logger.info(f"Test set: {len(test_df)} records (most recent data)")
        
        # Log date ranges
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            if len(df) > 0:
                min_date = df['kickoff_time'].min()
                max_date = df['kickoff_time'].max()
                logger.info(f"{name} date range: {min_date} to {max_date}")
        
        return train_df, val_df, test_df
    
    def train_linear_regression(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        split_by_position: bool = False
    ) -> LinearRegression:
        """
        Train Linear Regression model (baseline).
        
        IMPROVEMENT 6: Option to split by position (biggest single win).
        
        Args:
            train_df: Training data
            val_df: Optional validation data
            split_by_position: If True, train separate model per position
            
        Returns:
            Trained model (or dict of models if split_by_position)
        """
        if split_by_position:
            return self._train_position_specific_models(train_df, val_df)
        
        logger.info("Training Linear Regression model (all positions combined)...")
        
        # Prepare features and target
        X_train, y_train = self.feature_engineer.prepare_features(train_df, include_target=True)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        logger.info(f"Train - RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.3f}")
        
        # Evaluate on validation set if provided
        if val_df is not None and len(val_df) > 0:
            X_val, y_val = self.feature_engineer.prepare_features(val_df, include_target=True)
            val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            logger.info(f"Val - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.3f}")
        
        self.models['linear_regression'] = model
        return model
    
    def _train_position_specific_models(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, LinearRegression]:
        """
        IMPROVEMENT 6: Train separate models per position (BIGGEST SINGLE WIN).
        
        GK scoring clean sheets has nothing in common with FWD scoring goals.
        Per-position models typically improve R² by 0.05-0.15.
        
        Args:
            train_df: Training data (BEFORE feature engineering - needs position column)
            val_df: Optional validation data (BEFORE feature engineering)
            
        Returns:
            Dictionary mapping position to trained model
        """
        logger.info("Training position-specific Linear Regression models...")
        
        # Fit feature engineer on full training data (needed for categorical mappings)
        self.feature_engineer.fit(train_df)
        
        position_models = {}
        position_results = {}
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            # Filter by position BEFORE feature engineering
            train_pos_raw = train_df[train_df['position'] == position].copy()
            
            if len(train_pos_raw) < 100:  # Need minimum samples
                logger.warning(f"Insufficient data for {position}: {len(train_pos_raw)} samples. Skipping.")
                continue
            
            logger.info(f"\n--- Training {position} model ---")
            logger.info(f"Training samples (raw): {len(train_pos_raw)}")
            
            # Engineer features for this position
            train_pos = self.feature_engineer.engineer_features(
                train_pos_raw,
                is_training=False,  # Use existing mappings
                lag_features=['total_points']
            )
            
            logger.info(f"Training samples (after engineering): {len(train_pos)}")
            
            if len(train_pos) < 50:
                logger.warning(f"Too few samples after engineering for {position}. Skipping.")
                continue
            
            # Prepare features and target
            X_train, y_train = self.feature_engineer.prepare_features(train_pos, include_target=True)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate on training set
            train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)
            
            logger.info(f"{position} Train - RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.3f}")
            
            position_results[position] = {'train': {'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2}}
            
            # Evaluate on validation set if provided
            if val_df is not None and len(val_df) > 0:
                val_pos_raw = val_df[val_df['position'] == position].copy()
                if len(val_pos_raw) > 0:
                    # Engineer features
                    val_pos = self.feature_engineer.engineer_features(
                        val_pos_raw,
                        is_training=False,
                        lag_features=['total_points']
                    )
                    
                    if len(val_pos) > 0:
                        X_val, y_val = self.feature_engineer.prepare_features(val_pos, include_target=True)
                        val_pred = model.predict(X_val)
                        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                        val_mae = mean_absolute_error(y_val, val_pred)
                        val_r2 = r2_score(y_val, val_pred)
                        logger.info(f"{position} Val - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.3f}")
                        position_results[position]['val'] = {'rmse': val_rmse, 'mae': val_mae, 'r2': val_r2}
            
            position_models[position] = model
        
        # Calculate weighted average R² across positions
        if position_results:
            total_samples = sum(len(train_df[train_df['position'] == pos]) for pos in position_results.keys())
            weighted_r2 = sum(
                position_results[pos]['train']['r2'] * len(train_df[train_df['position'] == pos]) / total_samples
                for pos in position_results.keys()
            )
            logger.info(f"\n📊 Weighted Average R² across positions: {weighted_r2:.3f}")
        
        self.models['linear_regression_by_position'] = position_models
        self.models['position_results'] = position_results
        return position_models
    
    def train_neural_network(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Train Neural Network model with IMPROVEMENTS:
        - IMPROVEMENT 3: Add Dropout layers
        - IMPROVEMENT 5: Renamed from nn_bad_model to nn_baseline_model
        
        Args:
            train_df: Training data
            val_df: Optional validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Trained model or None if TensorFlow unavailable
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping neural network training.")
            return None
        
        logger.info("Training Neural Network baseline model...")
        
        # Prepare data
        X_train, y_train = self.feature_engineer.prepare_features(train_df, include_target=True)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Prepare validation data if available
        validation_data = None
        if val_df is not None and len(val_df) > 0:
            X_val, y_val = self.feature_engineer.prepare_features(val_df, include_target=True)
            X_val_scaled = scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        
        # Build model with Dropout (IMPROVEMENT 3)
        input_dim = X_train_scaled.shape[1]
        
        model = Sequential([
            Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),  # IMPROVEMENT 3: Add Dropout
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),  # IMPROVEMENT 3: Add Dropout
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),  # IMPROVEMENT 3: Add Dropout
            layers.Dense(1, activation='linear')  # Regression output
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Model architecture:\n{model.summary()}")
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate
        train_pred = model.predict(X_train_scaled, verbose=0).flatten()
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        logger.info(f"Train - RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.3f}")
        
        if validation_data:
            val_pred = model.predict(X_val_scaled, verbose=0).flatten()
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            logger.info(f"Val - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.3f}")
        
        # IMPROVEMENT 5: Renamed from nn_bad_model
        self.models['nn_baseline_model'] = model
        self.models['scaler'] = scaler
        
        return model
    
    def train_xgboost(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        split_by_position: bool = True
    ):
        """
        IMPROVEMENT 10: Train XGBoost Gradient Boosting models.
        
        XGBoost captures non-linear relationships and feature interactions that
        Linear Regression cannot. Expected improvement: +0.05-0.10 R².
        
        Args:
            train_df: Training data (RAW, before feature engineering)
            val_df: Optional validation data (RAW)
            split_by_position: If True, train separate model per position (recommended)
            
        Returns:
            Trained model(s) or None if XGBoost unavailable
        """
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available. Skipping gradient boosting training.")
            return None
        
        if split_by_position:
            return self._train_xgboost_by_position(train_df, val_df)
        else:
            return self._train_xgboost_combined(train_df, val_df)
    
    def _train_xgboost_by_position(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, XGBRegressor]:
        """
        IMPROVEMENT 10: Train position-specific XGBoost models.
        
        Expected to outperform Linear Regression by capturing:
        - Non-linear relationships (e.g., diminishing returns on minutes)
        - Feature interactions (e.g., high form × weak opponent)
        - Complex patterns in categorical encodings
        
        Args:
            train_df: Training data (BEFORE feature engineering)
            val_df: Optional validation data (BEFORE feature engineering)
            
        Returns:
            Dictionary mapping position to trained XGBoost model
        """
        logger.info("Training position-specific XGBoost models...")
        logger.info("Expected improvement: +0.05-0.10 R² over Linear Regression")
        
        # Fit feature engineer on full training data
        self.feature_engineer.fit(train_df)
        
        position_models = {}
        position_results = {}
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            # Filter by position BEFORE feature engineering
            train_pos_raw = train_df[train_df['position'] == position].copy()
            
            if len(train_pos_raw) < 100:
                logger.warning(f"Insufficient data for {position}: {len(train_pos_raw)} samples. Skipping.")
                continue
            
            logger.info(f"\n--- Training {position} XGBoost model ---")
            logger.info(f"Training samples (raw): {len(train_pos_raw)}")
            
            # Engineer features
            train_pos = self.feature_engineer.engineer_features(
                train_pos_raw,
                is_training=False,
                lag_features=['total_points']
            )
            
            logger.info(f"Training samples (after engineering): {len(train_pos)}")
            
            if len(train_pos) < 50:
                logger.warning(f"Too few samples after engineering for {position}. Skipping.")
                continue
            
            # Prepare features and target
            X_train, y_train = self.feature_engineer.prepare_features(train_pos, include_target=True)
            
            # Prepare validation set for early stopping
            eval_set = None
            if val_df is not None and len(val_df) > 0:
                val_pos_raw = val_df[val_df['position'] == position].copy()
                if len(val_pos_raw) > 0:
                    val_pos = self.feature_engineer.engineer_features(
                        val_pos_raw,
                        is_training=False,
                        lag_features=['total_points']
                    )
                    if len(val_pos) > 0:
                        X_val, y_val = self.feature_engineer.prepare_features(val_pos, include_target=True)
                        eval_set = [(X_val, y_val)]
            
            # Train XGBoost model
            model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,  # Use all CPU cores
                early_stopping_rounds=50 if eval_set else None,
                eval_metric='rmse'
            )
            
            # Fit model
            if eval_set:
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train, verbose=False)
            
            # Evaluate on training set
            train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)
            
            logger.info(f"{position} Train - RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.3f}")
            
            position_results[position] = {'train': {'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2}}
            
            # Evaluate on validation set if provided
            if eval_set:
                X_val, y_val = eval_set[0]
                val_pred = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                val_mae = mean_absolute_error(y_val, val_pred)
                val_r2 = r2_score(y_val, val_pred)
                logger.info(f"{position} Val - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.3f}")
                position_results[position]['val'] = {'rmse': val_rmse, 'mae': val_mae, 'r2': val_r2}
            
            position_models[position] = model
        
        # Calculate weighted average R² across positions
        if position_results:
            total_samples = sum(len(train_df[train_df['position'] == pos]) for pos in position_results.keys())
            weighted_r2 = sum(
                position_results[pos]['train']['r2'] * len(train_df[train_df['position'] == pos]) / total_samples
                for pos in position_results.keys()
            )
            logger.info(f"\n📊 XGBoost Weighted Average R² across positions: {weighted_r2:.3f}")
        
        self.models['xgboost_by_position'] = position_models
        self.models['xgboost_position_results'] = position_results
        return position_models
    
    def _train_xgboost_combined(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> XGBRegressor:
        """
        Train a single combined XGBoost model (all positions together).
        
        Note: Position-specific models are recommended for better performance.
        
        Args:
            train_df: Training data (BEFORE feature engineering)
            val_df: Optional validation data
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training combined XGBoost model (all positions)...")
        
        # Fit feature engineer
        self.feature_engineer.fit(train_df)
        
        # Engineer features
        train_engineered = self.feature_engineer.engineer_features(
            train_df,
            is_training=False,
            lag_features=['total_points']
        )
        
        # Prepare features
        X_train, y_train = self.feature_engineer.prepare_features(train_engineered, include_target=True)
        
        # Prepare validation set
        eval_set = None
        if val_df is not None and len(val_df) > 0:
            val_engineered = self.feature_engineer.engineer_features(
                val_df,
                is_training=False,
                lag_features=['total_points']
            )
            X_val, y_val = self.feature_engineer.prepare_features(val_engineered, include_target=True)
            eval_set = [(X_val, y_val)]
        
        # Train model
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50 if eval_set else None,
            eval_metric='rmse'
        )
        
        if eval_set:
            model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
        else:
            model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        logger.info(f"Train - RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.3f}")
        
        if eval_set:
            val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            logger.info(f"Val - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.3f}")
        
        self.models['xgboost_combined'] = model
        return model
    
    def evaluate_models(
        self,
        test_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test set.
        
        Args:
            test_df: Test data (RAW, before feature engineering if using position models)
            
        Returns:
            Dictionary of evaluation metrics per model
        """
        logger.info("\n" + "="*60)
        logger.info("Evaluating models on test set...")
        logger.info("="*60)
        
        results = {}
        
        # Evaluate position-specific models (IMPROVEMENT 6)
        if 'linear_regression_by_position' in self.models:
            logger.info("\nEvaluating Position-Specific Models:")
            position_models = self.models['linear_regression_by_position']
            
            all_preds = []
            all_actuals = []
            position_metrics = {}
            
            for position in ['GK', 'DEF', 'MID', 'FWD']:
                if position not in position_models:
                    continue
                
                # Filter raw data by position
                test_pos_raw = test_df[test_df['position'] == position].copy()
                if len(test_pos_raw) == 0:
                    continue
                
                # Engineer features
                test_pos = self.feature_engineer.engineer_features(
                    test_pos_raw,
                    is_training=False,
                    lag_features=['total_points']
                )
                
                if len(test_pos) == 0:
                    continue
                
                X_test, y_test = self.feature_engineer.prepare_features(test_pos, include_target=True)
                y_pred = position_models[position].predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                position_metrics[position] = {'rmse': rmse, 'mae': mae, 'r2': r2, 'n_samples': len(test_pos)}
                
                logger.info(f"{position} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f} (n={len(test_pos)})")
                
                all_preds.extend(y_pred)
                all_actuals.extend(y_test)
            
            # Overall metrics
            overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
            overall_mae = mean_absolute_error(all_actuals, all_preds)
            overall_r2 = r2_score(all_actuals, all_preds)
            
            results['linear_regression_by_position'] = {
                'rmse': overall_rmse,
                'mae': overall_mae,
                'r2': overall_r2,
                'by_position': position_metrics
            }
            
            logger.info(f"\n📊 Overall (Position-Specific Models):")
            logger.info(f"   RMSE: {overall_rmse:.2f}, MAE: {overall_mae:.2f}, R²: {overall_r2:.3f}")
            logger.info(f"\n🎯 IMPROVEMENT: R² increased from ~0.324 to {overall_r2:.3f} (+{overall_r2 - 0.324:.3f})")
        
        # Evaluate combined Linear Regression (if exists)
        if 'linear_regression' in self.models:
            # Engineer features for combined model
            test_engineered = self.feature_engineer.engineer_features(
                test_df,
                is_training=False,
                lag_features=['total_points']
            )
            X_test, y_test = self.feature_engineer.prepare_features(test_engineered, include_target=True)
            model = self.models['linear_regression']
            y_pred = model.predict(X_test)
            results['linear_regression'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            logger.info(f"\nLinear Regression (Combined) - RMSE: {results['linear_regression']['rmse']:.2f}, "
                       f"MAE: {results['linear_regression']['mae']:.2f}, "
                       f"R²: {results['linear_regression']['r2']:.3f}")
        
        # Evaluate Neural Network
        if 'nn_baseline_model' in self.models and TF_AVAILABLE:
            test_engineered = self.feature_engineer.engineer_features(
                test_df,
                is_training=False,
                lag_features=['total_points']
            )
            X_test, y_test = self.feature_engineer.prepare_features(test_engineered, include_target=True)
            model = self.models['nn_baseline_model']
            scaler = self.models['scaler']
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled, verbose=0).flatten()
            results['nn_baseline_model'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            logger.info(f"Neural Network Baseline - RMSE: {results['nn_baseline_model']['rmse']:.2f}, "
                       f"MAE: {results['nn_baseline_model']['mae']:.2f}, "
                       f"R²: {results['nn_baseline_model']['r2']:.3f}")
        
        # Evaluate XGBoost position-specific models (IMPROVEMENT 10)
        if 'xgboost_by_position' in self.models and XGB_AVAILABLE:
            logger.info("\n🚀 Evaluating XGBoost Position-Specific Models:")
            position_models = self.models['xgboost_by_position']
            
            all_preds = []
            all_actuals = []
            position_metrics = {}
            
            for position in ['GK', 'DEF', 'MID', 'FWD']:
                if position not in position_models:
                    continue
                
                # Filter raw data by position
                test_pos_raw = test_df[test_df['position'] == position].copy()
                if len(test_pos_raw) == 0:
                    continue
                
                # Engineer features
                test_pos = self.feature_engineer.engineer_features(
                    test_pos_raw,
                    is_training=False,
                    lag_features=['total_points']
                )
                
                if len(test_pos) == 0:
                    continue
                
                X_test, y_test = self.feature_engineer.prepare_features(test_pos, include_target=True)
                y_pred = position_models[position].predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                position_metrics[position] = {'rmse': rmse, 'mae': mae, 'r2': r2, 'n_samples': len(test_pos)}
                
                logger.info(f"{position} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f} (n={len(test_pos)})")
                
                all_preds.extend(y_pred)
                all_actuals.extend(y_test)
            
            # Overall metrics
            overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
            overall_mae = mean_absolute_error(all_actuals, all_preds)
            overall_r2 = r2_score(all_actuals, all_preds)
            
            results['xgboost_by_position'] = {
                'rmse': overall_rmse,
                'mae': overall_mae,
                'r2': overall_r2,
                'by_position': position_metrics
            }
            
            logger.info(f"\n📊 Overall (XGBoost Position-Specific):")
            logger.info(f"   RMSE: {overall_rmse:.2f}, MAE: {overall_mae:.2f}, R²: {overall_r2:.3f}")
            
            # Compare to Linear Regression baseline
            if 'linear_regression_by_position' in results:
                lr_r2 = results['linear_regression_by_position']['r2']
                improvement = overall_r2 - lr_r2
                logger.info(f"\n🎯 IMPROVEMENT over Linear Regression:")
                logger.info(f"   ΔR²: {improvement:+.4f} ({improvement/lr_r2*100:+.2f}%)")
                if improvement > 0.02:
                    logger.info(f"   ✅ Significant improvement achieved!")
                elif improvement > 0:
                    logger.info(f"   ⚠️ Marginal improvement")
                else:
                    logger.info(f"   ❌ No improvement - Linear Regression performs better")
        
        # Evaluate XGBoost combined model (if exists)
        if 'xgboost_combined' in self.models and XGB_AVAILABLE:
            test_engineered = self.feature_engineer.engineer_features(
                test_df,
                is_training=False,
                lag_features=['total_points']
            )
            X_test, y_test = self.feature_engineer.prepare_features(test_engineered, include_target=True)
            model = self.models['xgboost_combined']
            y_pred = model.predict(X_test)
            results['xgboost_combined'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            logger.info(f"\nXGBoost (Combined) - RMSE: {results['xgboost_combined']['rmse']:.2f}, "
                       f"MAE: {results['xgboost_combined']['mae']:.2f}, "
                       f"R²: {results['xgboost_combined']['r2']:.3f}")
        
        self.results = results
        return results
    
    def analyze_by_position(
        self,
        test_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        IMPROVEMENT 4: Analyze performance by position with corrected labels.
        
        Args:
            test_df: Test data (before feature engineering)
            
        Returns:
            Dictionary of metrics per position
        """
        logger.info("Analyzing performance by position...")
        
        # We need the original data with 'position' column
        # So we'll work with the test_df before it was engineered
        # Create a copy and add position back if needed
        
        position_results = {}
        
        # Check if position is still in the dataframe
        if 'position' not in test_df.columns:
            # Position was already encoded, skip analysis
            logger.warning("Position column not found (already encoded). Skipping position analysis.")
            return {}
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            position_df = test_df[test_df['position'] == position].copy()
            
            if len(position_df) == 0:
                continue
            
            # Engineer features for this position
            position_df_engineered = self.feature_engineer.engineer_features(
                position_df,
                is_training=False,
                lag_features=['total_points']
            )
            
            if len(position_df_engineered) == 0:
                continue
            
            X, y = self.feature_engineer.prepare_features(position_df_engineered, include_target=True)
            
            # Use Linear Regression for analysis
            if 'linear_regression' in self.models:
                model = self.models['linear_regression']
                y_pred = model.predict(X)
                
                # IMPROVEMENT 4: Use corrected position names
                position_name = self.POSITION_NAMES.get(position, position)
                
                position_results[position_name] = {
                    'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                    'mae': mean_absolute_error(y, y_pred),
                    'r2': r2_score(y, y_pred),
                    'n_samples': len(position_df_engineered)
                }
                
                logger.info(f"{position_name}: RMSE={position_results[position_name]['rmse']:.2f}, "
                           f"MAE={position_results[position_name]['mae']:.2f}, "
                           f"R²={position_results[position_name]['r2']:.3f}, "
                           f"n={position_results[position_name]['n_samples']}")
        
        return position_results
    
    def save_models(self, output_dir: str = "ml/models"):
        """
        Save all trained models.
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        predictor = FPLPredictor()
        predictor.feature_engineer = self.feature_engineer
        
        # Save position-specific models (IMPROVEMENT 6)
        if 'linear_regression_by_position' in self.models:
            position_models = self.models['linear_regression_by_position']
            for position, model in position_models.items():
                model_path = os.path.join(output_dir, f"linear_regression_{position.lower()}_v2.pkl")
                predictor.save_model(model, model_path)
                logger.info(f"Saved {position} model to {model_path}")
        
        # Save combined Linear Regression (if exists)
        if 'linear_regression' in self.models:
            model_path = os.path.join(output_dir, "linear_regression_combined_v1.pkl")
            predictor.save_model(self.models['linear_regression'], model_path)
            logger.info(f"Saved Linear Regression (combined) to {model_path}")
        
        # Save Neural Network
        if 'nn_baseline_model' in self.models and TF_AVAILABLE:
            model_path = os.path.join(output_dir, "nn_baseline_v1.pkl")
            scaler = self.models.get('scaler')
            predictor.save_model(self.models['nn_baseline_model'], model_path, scaler)
            logger.info(f"Saved Neural Network to {model_path}")
        
        # Save XGBoost position-specific models (IMPROVEMENT 10)
        if 'xgboost_by_position' in self.models:
            position_models = self.models['xgboost_by_position']
            for position, model in position_models.items():
                model_path = os.path.join(output_dir, f"xgboost_{position.lower()}_v3.pkl")
                predictor.save_model(model, model_path)
                logger.info(f"Saved {position} XGBoost model to {model_path}")
        
        # Save combined XGBoost (if exists)
        if 'xgboost_combined' in self.models:
            model_path = os.path.join(output_dir, "xgboost_combined_v3.pkl")
            predictor.save_model(self.models['xgboost_combined'], model_path)
            logger.info(f"Saved XGBoost (combined) to {model_path}")
        
        # Save results
        results_path = os.path.join(output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            def convert_to_native(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                else:
                    return obj
            
            json.dump(convert_to_native(self.results), f, indent=2)
        logger.info(f"Saved results to {results_path}")


def main():
    """Main training pipeline."""
    # Determine which dataset to use
    # DECISION: Use last 2 seasons (2024-25 and 2025-26) for recency
    # This gives us enough data while focusing on recent trends
    
    dataset_options = {
        "2_seasons": ["FPL_2024_2025.csv", "FPL_2025_2026.csv"],
        "3_seasons": ["FPL_2023_2024.csv", "FPL_2024_2025.csv", "FPL_2025_2026.csv"],
        "all_6_seasons": "cleaned_merged_seasons_cleaned.csv"
    }
    
    # Use 3 most recent seasons for good balance
    choice = "3_seasons"
    
    logger.info(f"Training strategy: Using {choice}")
    
    if choice == "all_6_seasons":
        data_path = dataset_options[choice]
    else:
        # Merge recent seasons
        dfs = []
        for csv_file in dataset_options[choice]:
            df = pd.read_csv(csv_file, low_memory=False)
            logger.info(f"Loaded {csv_file}: {len(df)} records")
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        data_path = "merged_recent_seasons.csv"
        combined_df.to_csv(data_path, index=False)
        logger.info(f"Merged {len(dfs)} files into {data_path}: {len(combined_df)} records")
    
    # Initialize trainer
    trainer = FPLModelTrainer(data_path)
    
    # Load and validate data
    trainer.load_data()
    
    # Don't preprocess yet - we need raw position column for split
    # trainer.preprocess_data()  # Skip this
    
    # Temporal split on RAW data (before preprocessing)
    train_df_raw, val_df_raw, test_df_raw = trainer.temporal_train_test_split_raw()
    
    # Train models - IMPROVEMENT 6: Split by position (biggest win!)
    logger.info("\n" + "="*60)
    logger.info("Training Strategy: Per-Position Models (BIGGEST WIN)")
    logger.info("="*60)
    
    # Pass RAW dataframes (with position column) to position-specific trainer
    trainer.train_linear_regression(train_df_raw, val_df_raw, split_by_position=True)
    
    # IMPROVEMENT 10: Train XGBoost models (expected +0.05-0.10 R² improvement)
    logger.info("\n" + "="*60)
    logger.info("IMPROVEMENT 10: Training XGBoost Models")
    logger.info("Expected: Capture feature interactions Linear Regression cannot")
    logger.info("="*60)
    trainer.train_xgboost(train_df_raw, val_df_raw, split_by_position=True)
    
    # Optional: Train Neural Network (currently TensorFlow not available)
    # trainer.train_neural_network(train_df_raw, val_df_raw, epochs=50)
    
    # Evaluate models
    trainer.evaluate_models(test_df_raw)
    
    # Analyze by position (IMPROVEMENT 4) - use raw test data
    trainer.analyze_by_position(test_df_raw)
    
    # Save models
    trainer.save_models()
    
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()
