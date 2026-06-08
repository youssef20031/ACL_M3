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
        val_df: Optional[pd.DataFrame] = None
    ) -> LinearRegression:
        """
        Train Linear Regression model (baseline).
        
        Args:
            train_df: Training data
            val_df: Optional validation data
            
        Returns:
            Trained model
        """
        logger.info("Training Linear Regression model...")
        
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
    
    def evaluate_models(
        self,
        test_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test set.
        
        Args:
            test_df: Test data
            
        Returns:
            Dictionary of evaluation metrics per model
        """
        logger.info("Evaluating models on test set...")
        
        X_test, y_test = self.feature_engineer.prepare_features(test_df, include_target=True)
        
        results = {}
        
        # Evaluate Linear Regression
        if 'linear_regression' in self.models:
            model = self.models['linear_regression']
            y_pred = model.predict(X_test)
            results['linear_regression'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            logger.info(f"Linear Regression - RMSE: {results['linear_regression']['rmse']:.2f}, "
                       f"MAE: {results['linear_regression']['mae']:.2f}, "
                       f"R²: {results['linear_regression']['r2']:.3f}")
        
        # Evaluate Neural Network
        if 'nn_baseline_model' in self.models and TF_AVAILABLE:
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
        
        # Save Linear Regression
        if 'linear_regression' in self.models:
            model_path = os.path.join(output_dir, "linear_regression_v1.pkl")
            predictor.save_model(self.models['linear_regression'], model_path)
            logger.info(f"Saved Linear Regression to {model_path}")
        
        # Save Neural Network
        if 'nn_baseline_model' in self.models and TF_AVAILABLE:
            model_path = os.path.join(output_dir, "nn_baseline_v1.pkl")
            scaler = self.models.get('scaler')
            predictor.save_model(self.models['nn_baseline_model'], model_path, scaler)
            logger.info(f"Saved Neural Network to {model_path}")
        
        # Save results
        results_path = os.path.join(output_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
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
    
    # Load and preprocess data
    trainer.load_data()
    trainer.preprocess_data()
    
    # Temporal split (IMPROVEMENT 1)
    train_df, val_df, test_df = trainer.temporal_train_test_split()
    
    # Save a copy of test_df with original positions for analysis
    test_df_original = test_df.copy()
    
    # Train models
    trainer.train_linear_regression(train_df, val_df)
    trainer.train_neural_network(train_df, val_df, epochs=50)
    
    # Evaluate models
    trainer.evaluate_models(test_df)
    
    # Analyze by position (IMPROVEMENT 4) - use original test data
    trainer.analyze_by_position(test_df_original)
    
    # Save models
    trainer.save_models()
    
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()
