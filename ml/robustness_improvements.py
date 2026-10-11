"""
Real-world Robustness Improvements for FPL ML Model
Implements LIST 1 requirements for production-ready predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr
import json
import pickle
import os

logger = logging.getLogger(__name__)


# ==================== 1. GAMEWEEK-BY-GAMEWEEK BACKTEST ====================

class WalkForwardValidator:
    """
    Implements walk-forward validation with Top-K hit rate and rank correlation.
    
    This is the honest performance signal - not aggregate R²/RMSE.
    """
    
    def __init__(self, model, feature_engineer):
        """
        Initialize validator.
        
        Args:
            model: Trained model
            feature_engineer: FeatureEngineer instance
        """
        self.model = model
        self.feature_engineer = feature_engineer
        self.gw_results = []
    
    def backtest_gameweek(self, 
                         train_data: pd.DataFrame,
                         test_gw_data: pd.DataFrame,
                         k: int = 10) -> Dict[str, Any]:
        """
        Backtest single gameweek with Top-K metrics.
        
        Args:
            train_data: Historical training data up to this GW
            test_gw_data: Single gameweek to predict
            k: Top K players to evaluate
            
        Returns:
            Dictionary with Top-K hit rate, rank correlation, etc.
        """
        # Train model on historical data
        X_train, y_train = self.feature_engineer.prepare_features(train_data, include_target=True)
        self.model.fit(X_train, y_train)
        
        # Predict test gameweek
        X_test, y_test = self.feature_engineer.prepare_features(test_gw_data, include_target=True)
        y_pred = self.model.predict(X_test)
        
        # Get actual top K performers
        actual_top_k_idx = np.argsort(y_test)[-k:][::-1]
        actual_top_k_points = y_test.iloc[actual_top_k_idx].values
        
        # Get predicted top K performers
        predicted_top_k_idx = np.argsort(y_pred)[-k:][::-1]
        predicted_top_k_points = y_test.iloc[predicted_top_k_idx].values  # Actual points of predicted top K
        
        # Calculate Top-K hit rate (overlap between predicted and actual top K)
        hit_rate = len(set(actual_top_k_idx) & set(predicted_top_k_idx)) / k
        
        # Calculate rank correlation (Spearman's rho)
        rank_corr, _ = spearmanr(y_test, y_pred)
        
        # Calculate NDCG@K (normalized discounted cumulative gain)
        y_true_binary = np.zeros(len(y_test))
        y_true_binary[actual_top_k_idx] = 1
        ndcg = ndcg_score([y_true_binary], [y_pred], k=k)
        
        # Points captured by predicted top K (% of optimal)
        optimal_points = actual_top_k_points.sum()
        captured_points = predicted_top_k_points.sum()
        points_capture_rate = captured_points / optimal_points if optimal_points > 0 else 0
        
        # RMSE and MAE for comparison
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae = np.mean(np.abs(y_test - y_pred))
        
        results = {
            'top_k_hit_rate': hit_rate,
            'rank_correlation': rank_corr,
            'ndcg_at_k': ndcg,
            'points_capture_rate': points_capture_rate,
            'optimal_points': optimal_points,
            'captured_points': captured_points,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(y_test)
        }
        
        logger.info(f"GW Backtest: Hit Rate={hit_rate:.2%}, Rank Corr={rank_corr:.3f}, "
                   f"NDCG@{k}={ndcg:.3f}, Points Captured={captured_points:.1f}/{optimal_points:.1f}")
        
        return results
    
    def walk_forward_validation(self,
                               data: pd.DataFrame,
                               start_gw: int,
                               end_gw: int,
                               k: int = 10) -> pd.DataFrame:
        """
        Perform walk-forward validation across multiple gameweeks.
        
        Args:
            data: Full dataset sorted by gameweek
            start_gw: Starting gameweek for validation
            end_gw: Ending gameweek for validation
            k: Top K players to evaluate
            
        Returns:
            DataFrame with per-GW results
        """
        results = []
        
        for gw in range(start_gw, end_gw + 1):
            logger.info(f"Backtesting GW {gw}/{end_gw}...")
            
            # Split: train on all data before this GW, test on this GW
            train_data = data[data['GW'] < gw]
            test_data = data[data['GW'] == gw]
            
            if len(train_data) == 0 or len(test_data) == 0:
                logger.warning(f"Skipping GW {gw} - insufficient data")
                continue
            
            gw_results = self.backtest_gameweek(train_data, test_data, k=k)
            gw_results['gameweek'] = gw
            results.append(gw_results)
        
        results_df = pd.DataFrame(results)
        
        # Summary statistics
        logger.info("\n" + "="*60)
        logger.info("WALK-FORWARD VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Avg Top-{k} Hit Rate: {results_df['top_k_hit_rate'].mean():.2%}")
        logger.info(f"Avg Rank Correlation: {results_df['rank_correlation'].mean():.3f}")
        logger.info(f"Avg NDCG@{k}: {results_df['ndcg_at_k'].mean():.3f}")
        logger.info(f"Avg Points Capture: {results_df['points_capture_rate'].mean():.2%}")
        logger.info(f"Avg RMSE: {results_df['rmse'].mean():.2f}")
        
        return results_df


# ==================== 2. DATA DRIFT DETECTION ====================

class DataDriftDetector:
    """
    Monitors feature distributions and detects data drift using PSI.
    
    Population Stability Index (PSI):
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change, monitor
    - PSI >= 0.2: Significant change, retrain required
    """
    
    def __init__(self, reference_data: pd.DataFrame, bins: int = 10):
        """
        Initialize detector with reference distribution.
        
        Args:
            reference_data: Training data to use as reference
            bins: Number of bins for PSI calculation
        """
        self.bins = bins
        self.reference_distributions = {}
        self.drift_log = []
        
        # Calculate reference distributions for numeric features
        for col in reference_data.select_dtypes(include=[np.number]).columns:
            if col not in ['upcoming', 'total_points']:  # Exclude target
                self.reference_distributions[col] = self._calculate_distribution(
                    reference_data[col]
                )
    
    def _calculate_distribution(self, data: pd.Series) -> Dict[str, Any]:
        """Calculate binned distribution for a feature."""
        # Handle NaN values
        data_clean = data.dropna()
        
        if len(data_clean) == 0:
            return {'bins': [], 'counts': [], 'percentages': []}
        
        # Create bins
        percentiles = np.linspace(0, 100, self.bins + 1)
        bin_edges = np.percentile(data_clean, percentiles)
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        # Count values in each bin
        counts, _ = np.histogram(data_clean, bins=bin_edges)
        percentages = counts / len(data_clean)
        
        return {
            'bins': bin_edges.tolist(),
            'counts': counts.tolist(),
            'percentages': percentages.tolist()
        }
    
    def calculate_psi(self, 
                     current_data: pd.DataFrame,
                     feature: str) -> float:
        """
        Calculate Population Stability Index for a feature.
        
        Args:
            current_data: Current data to compare
            feature: Feature name
            
        Returns:
            PSI value
        """
        if feature not in self.reference_distributions:
            logger.warning(f"Feature {feature} not in reference distributions")
            return 0.0
        
        ref_dist = self.reference_distributions[feature]
        if len(ref_dist['bins']) == 0:
            return 0.0
        
        # Calculate current distribution using reference bins
        current_clean = current_data[feature].dropna()
        if len(current_clean) == 0:
            return 0.0
        
        current_counts, _ = np.histogram(current_clean, bins=ref_dist['bins'])
        current_percentages = current_counts / len(current_clean)
        
        # Calculate PSI
        psi = 0.0
        ref_pct = np.array(ref_dist['percentages'])
        
        for i in range(len(current_percentages)):
            # Add small epsilon to avoid log(0)
            expected = ref_pct[i] + 1e-10
            actual = current_percentages[i] + 1e-10
            psi += (actual - expected) * np.log(actual / expected)
        
        return psi
    
    def detect_drift(self, 
                    current_data: pd.DataFrame,
                    gameweek: int) -> Dict[str, Any]:
        """
        Detect data drift across all features.
        
        Args:
            current_data: Current gameweek data
            gameweek: Current gameweek number
            
        Returns:
            Dictionary with drift status and metrics
        """
        drift_features = []
        psi_values = {}
        
        for feature in self.reference_distributions.keys():
            if feature not in current_data.columns:
                continue
            
            psi = self.calculate_psi(current_data, feature)
            psi_values[feature] = psi
            
            # Check thresholds
            if psi >= 0.2:
                drift_features.append(feature)
                logger.warning(f"⚠️ SIGNIFICANT DRIFT in {feature}: PSI={psi:.3f}")
            elif psi >= 0.1:
                logger.info(f"⚡ Moderate drift in {feature}: PSI={psi:.3f}")
        
        drift_status = {
            'gameweek': gameweek,
            'timestamp': datetime.now().isoformat(),
            'has_significant_drift': len(drift_features) > 0,
            'drift_features': drift_features,
            'psi_values': psi_values,
            'max_psi': max(psi_values.values()) if psi_values else 0.0,
            'avg_psi': np.mean(list(psi_values.values())) if psi_values else 0.0,
            'retrain_recommended': len(drift_features) >= 3  # 3+ features with PSI > 0.2
        }
        
        self.drift_log.append(drift_status)
        
        if drift_status['retrain_recommended']:
            logger.error(f"🚨 RETRAINING REQUIRED: {len(drift_features)} features with significant drift")
        
        return drift_status
    
    def save_drift_log(self, filepath: str):
        """Save drift detection log."""
        with open(filepath, 'w') as f:
            json.dump(self.drift_log, f, indent=2)
        logger.info(f"Saved drift log to {filepath}")


# ==================== 3. RETRAINING CADENCE ====================

class RetrainingScheduler:
    """
    Manages model retraining on rolling 3-season window.
    
    Retrains at minimum every international break (3-4 weeks).
    """
    
    def __init__(self, 
                 model,
                 feature_engineer,
                 rolling_seasons: int = 3):
        """
        Initialize retraining scheduler.
        
        Args:
            model: Model to retrain
            feature_engineer: FeatureEngineer instance
            rolling_seasons: Number of seasons to include in training
        """
        self.model = model
        self.feature_engineer = feature_engineer
        self.rolling_seasons = rolling_seasons
        self.last_retrain_date = None
        self.retrain_history = []
    
    def should_retrain(self, 
                      current_date: datetime,
                      drift_status: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if model should be retrained.
        
        Args:
            current_date: Current date
            drift_status: Optional drift detection results
            
        Returns:
            True if retraining is recommended
        """
        # First retrain
        if self.last_retrain_date is None:
            return True
        
        days_since_retrain = (current_date - self.last_retrain_date).days
        
        # Minimum 21 days between retrains (international break cadence)
        if days_since_retrain < 21:
            return False
        
        # Force retrain after 28 days (monthly)
        if days_since_retrain >= 28:
            logger.info(f"Retraining required: {days_since_retrain} days since last retrain")
            return True
        
        # Retrain if significant drift detected
        if drift_status and drift_status.get('retrain_recommended', False):
            logger.info("Retraining required: Significant data drift detected")
            return True
        
        return False
    
    def retrain_model(self,
                     all_data: pd.DataFrame,
                     current_season: str,
                     model_save_path: str) -> Dict[str, Any]:
        """
        Retrain model on rolling 3-season window.
        
        Args:
            all_data: All available historical data
            current_season: Current season (e.g., "2026-27")
            model_save_path: Path to save retrained model
            
        Returns:
            Retraining results
        """
        # Get seasons for rolling window
        current_year = int(current_season.split('-')[0])
        seasons_to_include = [
            f"{current_year - i}-{str(current_year - i + 1)[-2:]}"
            for i in range(self.rolling_seasons)
        ]
        
        # Filter data to rolling window
        train_data = all_data[all_data['season_x'].isin(seasons_to_include)]
        
        if len(train_data) == 0:
            logger.error("No training data available for rolling window")
            return {'success': False, 'error': 'No data'}
        
        logger.info(f"Retraining on {self.rolling_seasons} seasons: {seasons_to_include}")
        logger.info(f"Training samples: {len(train_data):,}")
        
        # Engineer features
        train_engineered = self.feature_engineer.engineer_features(
            train_data,
            is_training=True,
            lag_features=['total_points']
        )
        
        X_train, y_train = self.feature_engineer.prepare_features(
            train_engineered,
            include_target=True
        )
        
        # Retrain model
        self.model.fit(X_train, y_train)
        
        # Save model
        with open(model_save_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        self.last_retrain_date = datetime.now()
        
        retrain_info = {
            'success': True,
            'timestamp': self.last_retrain_date.isoformat(),
            'seasons_included': seasons_to_include,
            'n_samples': len(train_data),
            'model_path': model_save_path
        }
        
        self.retrain_history.append(retrain_info)
        
        logger.info(f"✅ Model retrained successfully: {len(train_data):,} samples")
        
        return retrain_info


# ==================== 4. INJURY/NEWS INGESTION ====================

class InjuryNewsHandler:
    """
    Handles injury and availability status from FPL API.
    
    Uses chance_of_playing and news flags to discount predictions.
    """
    
    def __init__(self):
        """Initialize injury handler."""
        self.injury_cache = {}
    
    def parse_fpl_availability(self, player_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse FPL API availability fields.
        
        Args:
            player_data: Player data from FPL API
            
        Returns:
            Availability status
        """
        chance_of_playing = player_data.get('chance_of_playing_next_round')
        news = player_data.get('news', '')
        news_added = player_data.get('news_added')
        
        # Determine availability status
        if chance_of_playing is None:
            status = 'available'
            confidence_discount = 1.0
        elif chance_of_playing == 0:
            status = 'ruled_out'
            confidence_discount = 0.0
        elif chance_of_playing <= 25:
            status = 'doubtful'
            confidence_discount = 0.25
        elif chance_of_playing <= 50:
            status = 'questionable'
            confidence_discount = 0.50
        elif chance_of_playing <= 75:
            status = 'probable'
            confidence_discount = 0.75
        else:
            status = 'available'
            confidence_discount = 1.0
        
        # Check for suspension/ban keywords
        suspension_keywords = ['suspended', 'banned', 'red card', 'sent off']
        injury_keywords = ['injured', 'injury', 'knock', 'strain', 'fitness']
        
        if any(kw in news.lower() for kw in suspension_keywords):
            status = 'suspended'
            confidence_discount = 0.0
        elif any(kw in news.lower() for kw in injury_keywords) and confidence_discount == 1.0:
            # Injury mentioned but no chance_of_playing set
            confidence_discount = 0.75
        
        return {
            'player_id': player_data.get('id'),
            'player_name': player_data.get('web_name', 'Unknown'),
            'status': status,
            'chance_of_playing': chance_of_playing,
            'confidence_discount': confidence_discount,
            'news': news,
            'news_added': news_added
        }
    
    def apply_availability_discount(self,
                                   predictions: List[Dict[str, Any]],
                                   availability_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply confidence discount to predictions based on availability.
        
        Args:
            predictions: List of prediction dictionaries
            availability_data: List of availability status dictionaries
            
        Returns:
            Updated predictions with discounted values
        """
        # Create availability lookup
        availability_map = {
            item['player_id']: item 
            for item in availability_data
        }
        
        discounted_predictions = []
        flagged_count = 0
        
        for pred in predictions:
            player_id = pred.get('player_id')
            availability = availability_map.get(player_id)
            
            if availability and availability['confidence_discount'] < 1.0:
                # Apply discount
                original_points = pred.get('predicted_points', 0)
                discount = availability['confidence_discount']
                
                pred['predicted_points'] = original_points * discount
                pred['availability_status'] = availability['status']
                pred['availability_discount'] = discount
                pred['original_prediction'] = original_points
                pred['availability_news'] = availability['news']
                
                flagged_count += 1
                
                if discount == 0.0:
                    logger.warning(f"⚠️ Zeroed out {pred.get('player_name')} - {availability['status']}")
            
            discounted_predictions.append(pred)
        
        # Re-sort by discounted predictions
        discounted_predictions.sort(
            key=lambda x: x.get('predicted_points', 0),
            reverse=True
        )
        
        logger.info(f"Applied availability discount to {flagged_count} players")
        
        return discounted_predictions


# ==================== 5. PRICE & OWNERSHIP CHANGE HANDLING ====================

class PriceOwnershipTracker:
    """
    Tracks nightly price changes and ownership for differential signals.
    """
    
    def __init__(self, history_path: str = 'ml/models/price_ownership_history.json'):
        """
        Initialize tracker.
        
        Args:
            history_path: Path to save price/ownership history
        """
        self.history_path = history_path
        self.history = self._load_history()
    
    def _load_history(self) -> Dict[str, List[Dict]]:
        """Load historical price/ownership data."""
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_history(self):
        """Save historical data."""
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f)
    
    def update_player_data(self,
                          player_id: str,
                          current_price: float,
                          ownership_pct: float,
                          timestamp: datetime):
        """
        Update player price and ownership.
        
        Args:
            player_id: Player ID
            current_price: Current price in £0.1m
            ownership_pct: Ownership percentage (0-100)
            timestamp: Update timestamp
        """
        if player_id not in self.history:
            self.history[player_id] = []
        
        self.history[player_id].append({
            'timestamp': timestamp.isoformat(),
            'price': current_price,
            'ownership': ownership_pct
        })
        
        # Keep only last 30 days
        cutoff = timestamp - timedelta(days=30)
        self.history[player_id] = [
            entry for entry in self.history[player_id]
            if datetime.fromisoformat(entry['timestamp']) > cutoff
        ]
    
    def calculate_price_velocity(self,
                                player_id: str,
                                lookback_days: int = 7) -> float:
        """
        Calculate price change velocity (£ per day).
        
        Args:
            player_id: Player ID
            lookback_days: Days to look back
            
        Returns:
            Price velocity
        """
        if player_id not in self.history or len(self.history[player_id]) < 2:
            return 0.0
        
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_entries = [
            entry for entry in self.history[player_id]
            if datetime.fromisoformat(entry['timestamp']) > cutoff
        ]
        
        if len(recent_entries) < 2:
            return 0.0
        
        # Calculate daily change
        first_price = recent_entries[0]['price']
        last_price = recent_entries[-1]['price']
        days_elapsed = (
            datetime.fromisoformat(recent_entries[-1]['timestamp']) -
            datetime.fromisoformat(recent_entries[0]['timestamp'])
        ).days
        
        if days_elapsed == 0:
            return 0.0
        
        velocity = (last_price - first_price) / days_elapsed
        return velocity
    
    def get_ownership_differential_signal(self,
                                         player_id: str,
                                         ownership_pct: float) -> str:
        """
        Calculate ownership differential signal.
        
        Args:
            player_id: Player ID
            ownership_pct: Current ownership percentage
            
        Returns:
            Differential signal: 'template', 'popular', 'differential', 'ultra_differential'
        """
        if ownership_pct >= 50:
            return 'template'  # Template player (>50% owned)
        elif ownership_pct >= 20:
            return 'popular'   # Popular (20-50%)
        elif ownership_pct >= 5:
            return 'differential'  # Differential (5-20%)
        else:
            return 'ultra_differential'  # Ultra differential (<5%)
    
    def enhance_value_predictions(self,
                                 predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance value endpoint with price velocity and ownership signals.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Enhanced predictions
        """
        for pred in predictions:
            player_id = str(pred.get('player_id', ''))
            ownership = pred.get('ownership_pct', 0)
            
            # Add price velocity
            price_velocity = self.calculate_price_velocity(player_id)
            pred['price_velocity_7d'] = price_velocity
            
            # Add ownership signal
            pred['ownership_signal'] = self.get_ownership_differential_signal(
                player_id, ownership
            )
            
            # Boost differential players in value ranking
            if pred['ownership_signal'] == 'differential':
                pred['differential_boost'] = 1.1  # 10% boost
            elif pred['ownership_signal'] == 'ultra_differential':
                pred['differential_boost'] = 1.2  # 20% boost
            else:
                pred['differential_boost'] = 1.0
            
            # Apply differential boost to value score
            original_value = pred.get('points_per_million', 0)
            pred['differential_value'] = original_value * pred['differential_boost']
        
        # Re-sort by differential value
        predictions.sort(
            key=lambda x: x.get('differential_value', 0),
            reverse=True
        )
        
        return predictions


# ==================== 6. FIXTURE CONGESTION & ROTATION RISK ====================

class FixtureCongestionAnalyzer:
    """
    Analyzes fixture congestion and rotation risk.
    
    Adds features:
    - days_since_last_match
    - matches_in_next_14_days
    - ucl_cup_involvement (European/cup fixtures)
    """
    
    def __init__(self):
        """Initialize analyzer."""
        pass
    
    def calculate_days_since_last_match(self,
                                       player_data: Dict[str, Any],
                                       team_fixtures: pd.DataFrame) -> int:
        """
        Calculate days since player's team last played.
        
        Args:
            player_data: Player data
            team_fixtures: Team's fixture history
            
        Returns:
            Days since last match
        """
        team = player_data.get('team')
        current_date = datetime.now()
        
        # Get most recent fixture
        past_fixtures = team_fixtures[
            (team_fixtures['team'] == team) &
            (pd.to_datetime(team_fixtures['kickoff_time']) < current_date)
        ].sort_values('kickoff_time', ascending=False)
        
        if len(past_fixtures) == 0:
            return 7  # Default to 1 week
        
        last_match = pd.to_datetime(past_fixtures.iloc[0]['kickoff_time'])
        days_since = (current_date - last_match).days
        
        return days_since
    
    def calculate_upcoming_fixture_density(self,
                                          player_data: Dict[str, Any],
                                          team_fixtures: pd.DataFrame,
                                          lookback_days: int = 14) -> int:
        """
        Calculate number of matches in next N days.
        
        Args:
            player_data: Player data
            team_fixtures: Team's fixture schedule
            lookback_days: Days to look ahead
            
        Returns:
            Number of upcoming matches
        """
        team = player_data.get('team')
        current_date = datetime.now()
        future_date = current_date + timedelta(days=lookback_days)
        
        upcoming_fixtures = team_fixtures[
            (team_fixtures['team'] == team) &
            (pd.to_datetime(team_fixtures['kickoff_time']) >= current_date) &
            (pd.to_datetime(team_fixtures['kickoff_time']) <= future_date)
        ]
        
        return len(upcoming_fixtures)
    
    def check_european_involvement(self,
                                  team: str,
                                  season: str) -> bool:
        """
        Check if team is in European competition.
        
        Args:
            team: Team name
            season: Season (e.g., "2026-27")
            
        Returns:
            True if team in UCL/UEL/UECL
        """
        # Simplified: Top 7 typically in Europe
        # In production, this should query actual competition data
        top_european_teams = [
            'Man City', 'Arsenal', 'Liverpool', 'Aston Villa',
            'Spurs', 'Chelsea', 'Man Utd', 'Newcastle'
        ]
        
        return team in top_european_teams
    
    def calculate_rotation_risk_score(self,
                                     player_data: Dict[str, Any],
                                     team_fixtures: pd.DataFrame) -> float:
        """
        Calculate rotation risk score (0-1, higher = more risk).
        
        Factors:
        - Fixture density
        - European involvement
        - Recent minutes
        - Days since last match
        
        Args:
            player_data: Player data
            team_fixtures: Team fixture data
            
        Returns:
            Rotation risk score (0-1)
        """
        # Factor 1: Fixture density
        matches_in_14d = self.calculate_upcoming_fixture_density(
            player_data, team_fixtures, lookback_days=14
        )
        density_risk = min(matches_in_14d / 6, 1.0)  # 6+ matches = max risk
        
        # Factor 2: European involvement
        team = player_data.get('team')
        season = player_data.get('season', '2026-27')
        european_risk = 0.3 if self.check_european_involvement(team, season) else 0.0
        
        # Factor 3: Recent heavy minutes
        minutes_rolling5 = player_data.get('minutes_rolling5', 0)
        heavy_minutes_risk = 0.3 if minutes_rolling5 > 400 else 0.0  # >400 mins in 5 games
        
        # Factor 4: Short rest
        days_since = self.calculate_days_since_last_match(player_data, team_fixtures)
        short_rest_risk = 0.4 if days_since <= 3 else 0.0
        
        # Combined risk score (weighted)
        rotation_risk = (
            0.35 * density_risk +
            0.25 * european_risk +
            0.20 * heavy_minutes_risk +
            0.20 * short_rest_risk
        )
        
        return min(rotation_risk, 1.0)
    
    def enhance_predictions_with_congestion(self,
                                           predictions: List[Dict[str, Any]],
                                           team_fixtures: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Add fixture congestion features to predictions.
        
        Args:
            predictions: List of prediction dictionaries
            team_fixtures: Team fixture data
            
        Returns:
            Enhanced predictions
        """
        for pred in predictions:
            # Calculate congestion features
            days_since = self.calculate_days_since_last_match(pred, team_fixtures)
            matches_14d = self.calculate_upcoming_fixture_density(pred, team_fixtures)
            rotation_risk = self.calculate_rotation_risk_score(pred, team_fixtures)
            
            # Add to prediction
            pred['days_since_last_match'] = days_since
            pred['matches_in_next_14_days'] = matches_14d
            pred['rotation_risk_score'] = rotation_risk
            pred['european_involvement'] = self.check_european_involvement(
                pred.get('team'), pred.get('season', '2026-27')
            )
            
            # Apply rotation discount to prediction
            original_points = pred.get('predicted_points', 0)
            rotation_discount = 1.0 - (rotation_risk * 0.3)  # Max 30% discount
            pred['rotation_adjusted_points'] = original_points * rotation_discount
            
            if rotation_risk > 0.5:
                logger.info(f"⚠️ High rotation risk for {pred.get('player_name')}: "
                          f"{rotation_risk:.1%} (discount: {(1-rotation_discount)*100:.0f}%)")
        
        return predictions


# ==================== INTEGRATION CLASS ====================

class RobustnessManager:
    """
    Main manager for all robustness improvements.
    Integrates all 6 components.
    """
    
    def __init__(self,
                 model,
                 feature_engineer,
                 reference_data: pd.DataFrame):
        """
        Initialize robustness manager.
        
        Args:
            model: ML model
            feature_engineer: FeatureEngineer instance
            reference_data: Reference training data
        """
        self.validator = WalkForwardValidator(model, feature_engineer)
        self.drift_detector = DataDriftDetector(reference_data)
        self.retrain_scheduler = RetrainingScheduler(model, feature_engineer)
        self.injury_handler = InjuryNewsHandler()
        self.price_tracker = PriceOwnershipTracker()
        self.congestion_analyzer = FixtureCongestionAnalyzer()
        
        logger.info("✅ RobustnessManager initialized with all 6 components")
