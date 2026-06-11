"""
FPL ML Predictor
Handles model loading, inference, and predictions
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from ml.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class PlayerPrediction:
    """Container for player prediction results."""
    player_name: str
    predicted_points: float
    confidence_interval: Optional[Tuple[float, float]] = None
    features_used: Optional[Dict[str, Any]] = None
    model_version: str = "v1"


class FPLPredictor:
    """
    FPL Player Performance Predictor.
    Uses trained ML models to predict next gameweek points.
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "linear"):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model file (.pkl)
            model_type: Type of model ('linear' or 'neural')
        """
        self.model = None
        self.scaler = None
        self.feature_engineer = FeatureEngineer()
        self.model_type = model_type
        self.model_loaded = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Load trained model and scaler.
        
        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file (optional, for neural networks)
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Loaded model from {model_path}")
            
            # Load scaler if provided (for neural networks)
            if scaler_path and os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Loaded scaler from {scaler_path}")
            
            # Load feature mappings
            mappings_path = model_path.replace('.pkl', '_mappings.json')
            if os.path.exists(mappings_path):
                self.feature_engineer.load_mappings(mappings_path)
            
            self.model_loaded = True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def save_model(self, model, model_path: str, scaler=None):
        """
        Save trained model and scaler.
        
        Args:
            model: Trained model
            model_path: Path to save model
            scaler: Optional scaler for neural networks
        """
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model to {model_path}")
            
            if scaler:
                scaler_path = model_path.replace('.pkl', '_scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                logger.info(f"Saved scaler to {scaler_path}")
            
            # Save feature mappings
            mappings_path = model_path.replace('.pkl', '_mappings.json')
            self.feature_engineer.save_mappings(mappings_path)
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def predict_next_gameweek(
        self, 
        player_data: Dict[str, Any]
    ) -> PlayerPrediction:
        """
        Predict points for a single player's next gameweek.
        
        Args:
            player_data: Dictionary with player's recent stats and fixture info
            
        Returns:
            PlayerPrediction object
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to dataframe for feature engineering
        df = pd.DataFrame([player_data])
        
        # Engineer features
        df_engineered = self.feature_engineer.engineer_features(
            df, 
            is_training=False,
            lag_features=None  # No lagging needed for inference
        )
        
        # Prepare features
        X, _ = self.feature_engineer.prepare_features(df_engineered, include_target=False)
        
        # Scale if using neural network
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        # Ensure non-negative prediction
        prediction = max(0, prediction)
        
        return PlayerPrediction(
            player_name=player_data.get('name', 'Unknown'),
            predicted_points=float(prediction),
            features_used={
                'form': player_data.get('form', 0),
                'minutes_avg': player_data.get('minutes', 0),
                'goals_scored': player_data.get('goals_scored', 0),
                'assists': player_data.get('assists', 0)
            },
            model_version="v1"
        )
    
    def predict_top_performers(
        self,
        players_data: List[Dict[str, Any]],
        position: Optional[str] = None,
        top_k: int = 10
    ) -> List[PlayerPrediction]:
        """
        Predict top K performers for next gameweek.
        
        Args:
            players_data: List of player data dictionaries
            position: Filter by position (GK/DEF/MID/FWD)
            top_k: Number of top predictions to return
            
        Returns:
            List of PlayerPrediction objects sorted by predicted points
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Filter by position if specified
        if position:
            players_data = [p for p in players_data if p.get('position') == position]
        
        # Convert to dataframe
        df = pd.DataFrame(players_data)
        
        if len(df) == 0:
            return []
        
        # Engineer features
        df_engineered = self.feature_engineer.engineer_features(
            df,
            is_training=False,
            lag_features=None
        )
        
        # Prepare features
        X, _ = self.feature_engineer.prepare_features(df_engineered, include_target=False)
        
        # Scale if needed
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Create results
        results = []
        for i, pred in enumerate(predictions):
            player_data = players_data[i]
            results.append(PlayerPrediction(
                player_name=player_data.get('name', 'Unknown'),
                predicted_points=max(0, float(pred)),
                features_used={
                    'form': player_data.get('form', 0),
                    'position': player_data.get('position', ''),
                    'value': player_data.get('value', 0)
                }
            ))
        
        # Sort by predicted points and return top K
        results.sort(key=lambda x: x.predicted_points, reverse=True)
        return results[:top_k]
    
    def predict_best_value(
        self,
        players_data: List[Dict[str, Any]],
        position: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Predict best value players (predicted points per £1m).
        
        Args:
            players_data: List of player data dictionaries
            position: Filter by position
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries with player info and value metrics
        """
        predictions = self.predict_top_performers(
            players_data,
            position=position,
            top_k=len(players_data)  # Get all predictions
        )
        
        # Calculate value score
        value_results = []
        for pred in predictions:
            # Find matching player data
            player = next((p for p in players_data if p.get('name') == pred.player_name), None)
            if player and player.get('value', 0) > 0:
                value = player['value'] / 10.0  # Convert to millions
                points_per_million = pred.predicted_points / value if value > 0 else 0
                
                value_results.append({
                    'name': pred.player_name,
                    'predicted_points': pred.predicted_points,
                    'value': value,
                    'points_per_million': points_per_million,
                    'position': player.get('position', ''),
                    'form': player.get('form', 0)
                })
        
        # Sort by points per million
        value_results.sort(key=lambda x: x['points_per_million'], reverse=True)
        return value_results[:top_k]
    
    def evaluate_model(
        self,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test dataframe with ground truth
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Engineer features
        df_engineered = self.feature_engineer.engineer_features(
            test_data,
            is_training=False,
            lag_features=['total_points']  # Include target
        )
        
        # Prepare features and target
        X, y = self.feature_engineer.prepare_features(df_engineered, include_target=True)
        
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predict
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mean_actual': float(y.mean()),
            'mean_predicted': float(y_pred.mean())
        }
        
        logger.info(f"Model evaluation: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R2={metrics['r2']:.3f}")
        
        return metrics
