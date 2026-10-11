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
from ml.promoted_teams_handler import PromotedTeamsHandler

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
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "linear",
                 enable_promoted_teams_handler: bool = True,
                 enable_minutes_models: bool = True):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model file (.pkl)
            model_type: Type of model ('linear' or 'neural')
            enable_promoted_teams_handler: Enable cold start handling for promoted teams
            enable_minutes_models: Enable minutes prediction models (Model C & D)
        """
        self.model = None
        self.scaler = None
        self.feature_engineer = FeatureEngineer()
        self.model_type = model_type
        self.model_loaded = False
        
        # Initialize promoted teams handler
        self.promoted_teams_handler = None
        if enable_promoted_teams_handler:
            # Try v2 archetype database first (with EFL data), then fallback to v1
            archetype_db_path = 'ml/models/promoted_archetypes_v2.pkl'
            if not os.path.exists(archetype_db_path):
                archetype_db_path = 'ml/models/promoted_archetypes.pkl'
            
            if os.path.exists(archetype_db_path):
                self.promoted_teams_handler = PromotedTeamsHandler(
                    archetype_db_path=archetype_db_path
                )
                logger.info(f"Promoted teams handler enabled with {archetype_db_path}")
            else:
                self.promoted_teams_handler = PromotedTeamsHandler()
                logger.info("Promoted teams handler enabled without archetype database")
        
        # Initialize minutes models (Model C & D)
        self.minutes_predictor = None
        if enable_minutes_models:
            try:
                from ml.minutes_model import MinutesPredictor
                self.minutes_predictor = MinutesPredictor(models_dir='ml/models')
                # Try to load existing models
                if os.path.exists('ml/models/minutes_pstart_fwd_v1.pkl'):
                    self.minutes_predictor.load()
                    logger.info("✅ Minutes models (Model C & D) loaded")
                else:
                    logger.info("⚠️ Minutes models not found - predictions will use fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize minutes models: {e}")
                self.minutes_predictor = None
        
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
        
        Automatically handles promoted teams with cold start handling.
        Uses minutes models (Model C & D) if available.
        
        Args:
            player_data: Dictionary with player's recent stats and fixture info
            
        Returns:
            PlayerPrediction object
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Check if player is from promoted team
        is_promoted = player_data.get('is_promoted', False)
        current_gw = player_data.get('GW', 1)
        
        # Use promoted teams handler for cold start (GW 1-12)
        if is_promoted and current_gw <= 12 and self.promoted_teams_handler:
            logger.info(f"Using promoted teams handler for {player_data.get('name')} (GW{current_gw})")
            
            try:
                prediction, metadata = self.promoted_teams_handler.predict_promoted_player(
                    player_data,
                    current_gw=current_gw,
                    use_archetypes=True
                )
                
                return PlayerPrediction(
                    player_name=player_data.get('name', 'Unknown'),
                    predicted_points=float(prediction),
                    features_used={
                        'method': metadata.get('method'),
                        'cold_start_phase': metadata.get('cold_start_phase'),
                        'confidence': metadata.get('confidence'),
                        'form': player_data.get('form', 0),
                        'position': player_data.get('position', '')
                    },
                    model_version=f"v1_promoted_{metadata.get('method')}"
                )
            except Exception as e:
                logger.warning(f"Promoted teams handler failed: {e}. Falling back to standard prediction.")
        
        # Standard prediction for non-promoted or post-cold-start
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
        
        # Predict points using existing model
        # NOTE: Existing models predict total_points directly, NOT points-per-minute
        # We keep this as-is until models are retrained with points-per-90 targets
        raw_prediction = self.model.predict(X_scaled)[0]
        
        # Optionally adjust prediction based on minutes models
        # This provides better context but doesn't use the flawed division approach
        minutes_info = {}
        if self.minutes_predictor and self.minutes_predictor.is_fitted:
            try:
                minutes_pred = self.minutes_predictor.predict(player_data)
                minutes_info = minutes_pred
                
                # Store minutes prediction for context
                # Do NOT apply points_per_minute = raw_points / 66.5 (mathematically invalid)
                # The existing model already accounts for typical playing time
                # in its training data distribution
                
                logger.debug(f"Minutes prediction: {minutes_pred['expected_minutes']:.1f} min "
                           f"(P(start)={minutes_pred['p_start']:.2f})")
                
            except Exception as e:
                logger.warning(f"Minutes prediction failed: {e}")
        
        # Use raw prediction as-is
        # TODO: Retrain points models with points-per-90 targets for proper integration
        prediction = max(0, raw_prediction)
        
        return PlayerPrediction(
            player_name=player_data.get('name', 'Unknown'),
            predicted_points=float(prediction),
            features_used={
                'form': player_data.get('form', 0),
                'minutes_avg': player_data.get('minutes', 0),
                'goals_scored': player_data.get('goals_scored', 0),
                'assists': player_data.get('assists', 0),
                **minutes_info  # Include minutes prediction context
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

        # engineer_features re-sorts rows (by kickoff_time/name), so the
        # predictions follow df_engineered's row order, not players_data's.
        # Map each prediction back to its player by name; indexing
        # players_data[i] attributed predictions to the wrong players.
        ordered_players = players_data
        if 'name' in df_engineered.columns and len(df_engineered) == len(players_data):
            by_name = {}
            for p in players_data:
                by_name.setdefault(p.get('name'), p)
            ordered_players = [by_name.get(n, {'name': n}) for n in df_engineered['name'].tolist()]

        # Create results
        results = []
        for i, pred in enumerate(predictions):
            player_data = ordered_players[i]
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
        players_by_name = {}
        for p in players_data:
            players_by_name.setdefault(p.get('name'), p)  # first match wins, as before
        for pred in predictions:
            # Find matching player data
            player = players_by_name.get(pred.player_name)
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
