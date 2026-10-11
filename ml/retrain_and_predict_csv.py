"""
Retrain ML Model and Generate Predictions using CSV files
No Neo4j dependency - uses historical data + fixtures CSV
"""
import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_engineering import FeatureEngineer
from ml.promoted_teams_handler import PromotedTeamsHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelRetrainer:
    """Retrain models with all new features."""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.positions = ['GK', 'DEF', 'MID', 'FWD']
    
    def fetch_training_data(self) -> pd.DataFrame:
        """Load historical data from CSV file."""
        logger.info("Loading training data from CSV...")
        
        csv_path = 'cleaned_merged_seasons_cleaned.csv'
        df = pd.read_csv(csv_path)
        
        # Rename columns to match expected names
        if 'season' in df.columns and 'season_x' not in df.columns:
            df['season_x'] = df['season']
        
        logger.info(f"✅ Loaded {len(df):,} records from {df['season_x'].nunique()} seasons")
        logger.info(f"   Seasons: {sorted(df['season_x'].unique())}")
        
        return df
    
    def train_position_model(self, 
                            data: pd.DataFrame, 
                            position: str) -> dict:
        """Train model for specific position."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {position} Model")
        logger.info(f"{'='*80}")
        
        # Filter by position
        pos_data = data[data['position'] == position].copy()
        logger.info(f"Position data: {len(pos_data):,} samples")
        
        if len(pos_data) < 100:
            logger.warning(f"Insufficient data for {position} - skipping")
            return None
        
        # Engineer features
        logger.info("Engineering features...")
        self.feature_engineer.fit(pos_data)
        pos_data_engineered = self.feature_engineer.engineer_features(
            pos_data,
            is_training=True,
            lag_features=['total_points']
        )
        
        # Prepare features
        X, y = self.feature_engineer.prepare_features(
            pos_data_engineered,
            include_target=True
        )
        
        logger.info(f"Features: {X.shape[1]}, Samples: {len(X):,}")
        
        # Temporal train/validation/test split (72/8/20)
        n_samples = len(X)
        train_end = int(n_samples * 0.72)
        val_end = int(n_samples * 0.80)
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        # Train XGBoost model
        logger.info("Training XGBoost model...")
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"\n{position} Model Performance:")
        logger.info(f"  RMSE: {rmse:.3f}")
        logger.info(f"  MAE:  {mae:.3f}")
        logger.info(f"  R²:   {r2:.3f}")
        
        # Save model
        model_path = f"ml/models/xgboost_{position.lower()}_v2.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save feature mappings
        mappings_path = model_path.replace('.pkl', '_mappings.json')
        self.feature_engineer.save_mappings(mappings_path)
        
        logger.info(f"✅ Saved model to {model_path}")
        
        return {
            'position': position,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X.shape[1],
            'model_path': model_path
        }
    
    def retrain_all_models(self):
        """Retrain all position-specific models."""
        logger.info("\n" + "="*80)
        logger.info("RETRAINING ALL ML MODELS WITH NEW FEATURES")
        logger.info("="*80)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Fetch data
        data = self.fetch_training_data()
        
        # Train each position
        results = []
        for position in self.positions:
            result = self.train_position_model(data, position)
            if result:
                results.append(result)
                self.models[position] = result
        
        # Save training results
        results_df = pd.DataFrame(results)
        results_path = 'ml/models/training_results_v2.json'
        results_df.to_json(results_path, orient='records', indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        print(results_df.to_string(index=False))
        
        logger.info(f"\n✅ All models retrained successfully")
        logger.info(f"Results saved to: {results_path}")
        
        return results


class PredictionGenerator:
    """Generate predictions for all players' next games using CSV data."""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.promoted_handler = None
        if os.path.exists('ml/models/promoted_archetypes.pkl'):
            self.promoted_handler = PromotedTeamsHandler(
                archetype_db_path='ml/models/promoted_archetypes.pkl'
            )
        self.models = {}
        self.historical_data = None
        self.fixtures = None
        self.load_models()
        self.load_data()
    
    def load_models(self):
        """Load trained models."""
        positions = ['GK', 'DEF', 'MID', 'FWD']
        
        for pos in positions:
            model_path = f"ml/models/xgboost_{pos.lower()}_v2.pkl"
            mappings_path = model_path.replace('.pkl', '_mappings.json')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[pos] = pickle.load(f)
                
                # Load feature mappings (once)
                if os.path.exists(mappings_path) and pos == 'GK':
                    self.feature_engineer.load_mappings(mappings_path)
                
                logger.info(f"✅ Loaded {pos} model")
            else:
                logger.warning(f"⚠️ Model not found: {model_path}")
    
    def load_data(self):
        """Load historical player data and fixtures."""
        logger.info("Loading historical data and fixtures...")
        
        # Load historical data
        self.historical_data = pd.read_csv('cleaned_merged_seasons_cleaned.csv')
        
        # Load fixtures
        self.fixtures = pd.read_csv('FPL_2026_2027_fixtures_processed.csv')
        
        logger.info(f"✅ Loaded {len(self.historical_data):,} historical records")
        logger.info(f"✅ Loaded {len(self.fixtures):,} fixtures for 2026-27")
    
    def get_player_recent_stats(self, player_name: str, team: str) -> dict:
        """Get player's recent stats from historical data."""
        # Filter for this player
        player_data = self.historical_data[
            (self.historical_data['name'] == player_name) & 
            (self.historical_data['team_x'] == team)
        ].copy()
        
        if len(player_data) == 0:
            # No history for this player - return defaults
            return {
                'minutes': 0,
                'goals_scored': 0,
                'assists': 0,
                'total_points': 0,
                'clean_sheets': 0,
                'goals_conceded': 0,
                'saves': 0,
                'bonus': 0,
                'bps': 0,
                'influence': 0,
                'creativity': 0,
                'threat': 0,
                'ict_index': 0
            }
        
        # Sort by date (most recent first)
        player_data['kickoff_time'] = pd.to_datetime(player_data['kickoff_time'])
        player_data = player_data.sort_values('kickoff_time', ascending=False)
        
        # Get last 5 games
        recent_games = player_data.head(5)
        
        # Calculate averages
        avg_stats = {
            'minutes': recent_games['minutes'].mean(),
            'goals_scored': recent_games['goals_scored'].mean(),
            'assists': recent_games['assists'].mean(),
            'total_points': recent_games['total_points'].mean(),
            'clean_sheets': recent_games['clean_sheets'].mean(),
            'goals_conceded': recent_games['goals_conceded'].mean(),
            'saves': recent_games['saves'].mean(),
            'bonus': recent_games['bonus'].mean(),
            'bps': recent_games['bps'].mean(),
            'influence': recent_games['influence'].mean(),
            'creativity': recent_games['creativity'].mean(),
            'threat': recent_games['threat'].mean(),
            'ict_index': recent_games['ict_index'].mean()
        }
        
        return avg_stats
    
    def get_unique_players(self) -> pd.DataFrame:
        """Get unique players from historical data - only active players from latest GW29."""
        # Filter for latest season's GW29 (current active players)
        df = self.historical_data.copy()
        
        # Get latest season
        latest_season = df['season_x'].max()
        logger.info(f"Latest season: {latest_season}")
        
        # Get players from GW29 of latest season (most recent complete GW with all players)
        active_players = df[(df['season_x'] == latest_season) & (df['GW'] == 29)].copy()
        
        logger.info(f"Found {active_players['name'].nunique()} active players from GW29 {latest_season}")
        
        # Keep only unique players (in case of multiple games in same GW)
        unique_players = active_players.drop_duplicates(subset=['name'], keep='first')
        
        return unique_players[['name', 'position', 'team_x', 'value']].reset_index(drop=True)
    
    def predict_all_players(self) -> pd.DataFrame:
        """Generate predictions for all players' next games."""
        logger.info("\n" + "="*80)
        logger.info("GENERATING PREDICTIONS FOR ALL PLAYERS")
        logger.info("="*80)
        
        # Get unique players
        players_df = self.get_unique_players()
        logger.info(f"Found {len(players_df):,} unique players")
        
        # Get next gameweek fixtures (GW 1 of 2026-27)
        next_gw_fixtures = self.fixtures[self.fixtures['gameweek'] == 1].copy()
        logger.info(f"Found {len(next_gw_fixtures):,} fixtures in GW1")
        
        predictions = []
        
        for idx, player_row in players_df.iterrows():
            player_name = player_row['name']
            team = player_row['team_x']
            position = player_row['position']
            
            # Find fixture for this team
            fixture = next_gw_fixtures[
                (next_gw_fixtures['home_team'] == team) | 
                (next_gw_fixtures['away_team'] == team)
            ]
            
            if len(fixture) == 0:
                continue  # No fixture for this team
            
            fixture = fixture.iloc[0]
            
            try:
                # Get player stats
                recent_stats = self.get_player_recent_stats(player_name, team)
                
                # Determine home/away
                is_home = fixture['home_team'] == team
                opponent = fixture['away_team'] if is_home else fixture['home_team']
                
                # Build feature dict
                player_features = {
                    'name': player_name,
                    'position': position,
                    'team': team,
                    'team_x': team,
                    'value': player_row['value'],
                    'GW': int(fixture['gameweek']),
                    'season_x': '2026-27',
                    'was_home': is_home,
                    'opp_team_name': opponent,
                    'kickoff_time': fixture['kickoff_time'],
                    **recent_stats
                }
                
                # Convert to DataFrame for feature engineering
                player_df = pd.DataFrame([player_features])
                
                # Engineer features
                player_engineered = self.feature_engineer.engineer_features(
                    player_df,
                    is_training=False,
                    lag_features=None
                )
                
                # Prepare features for model
                X, _ = self.feature_engineer.prepare_features(
                    player_engineered,
                    include_target=False
                )
                
                # Get position-specific model
                if position not in self.models:
                    continue
                
                model = self.models[position]
                
                # Predict
                prediction = model.predict(X)[0]
                prediction = max(0, prediction)  # Ensure non-negative
                
                # Check if promoted team (for cold start handling)
                promoted_teams = ['Ipswich Town', 'Leicester City', 'Southampton']
                is_promoted = team in promoted_teams
                
                # Apply promoted team adjustment if needed
                if is_promoted and player_features['GW'] <= 12 and self.promoted_handler:
                    try:
                        adjusted_pred, metadata = self.promoted_handler.predict_promoted_player(
                            player_features,
                            current_gw=player_features['GW'],
                            use_archetypes=True
                        )
                        prediction = adjusted_pred
                        prediction_method = metadata.get('method', 'promoted_adjusted')
                    except:
                        prediction_method = 'xgboost_v2'
                else:
                    prediction_method = 'xgboost_v2'
                
                predictions.append({
                    'name': player_name,
                    'position': position,
                    'team': team,
                    'predicted_points': round(prediction, 2),
                    'value': player_row['value'],
                    'opponent': opponent,
                    'is_home': is_home,
                    'gameweek': int(fixture['gameweek']),
                    'kickoff_time': fixture['kickoff_time'],
                    'prediction_method': prediction_method,
                    'is_promoted_team': is_promoted
                })
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(players_df)} players...")
                    
            except Exception as e:
                logger.error(f"Error predicting for {player_name}: {e}")
                continue
        
        predictions_df = pd.DataFrame(predictions)
        
        # Sort by predicted points
        predictions_df = predictions_df.sort_values('predicted_points', ascending=False)
        
        logger.info(f"\n✅ Generated {len(predictions_df):,} predictions")
        
        return predictions_df
    
    def save_predictions(self, predictions_df: pd.DataFrame):
        """Save predictions to file."""
        # Save to CSV
        csv_path = 'ml/predictions/next_gw_predictions.csv'
        os.makedirs('ml/predictions', exist_ok=True)
        predictions_df.to_csv(csv_path, index=False)
        logger.info(f"✅ Saved predictions to {csv_path}")
        
        # Save to JSON
        json_path = 'ml/predictions/next_gw_predictions.json'
        predictions_df.to_json(json_path, orient='records', indent=2)
        logger.info(f"✅ Saved predictions to {json_path}")
        
        # Display top predictions by position
        logger.info("\n" + "="*80)
        logger.info("TOP 10 PREDICTED PERFORMERS BY POSITION")
        logger.info("="*80)
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_df = predictions_df[predictions_df['position'] == position].head(10)
            logger.info(f"\n{position}:")
            for _, row in pos_df.iterrows():
                home_indicator = 'H' if row['is_home'] else 'A'
                logger.info(
                    f"  {row['predicted_points']:5.2f} pts | "
                    f"{row['name']:25s} | {row['team']:20s} | "
                    f"vs {row['opponent']:20s} ({home_indicator})"
                )


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("FPL ML MODEL - COMPLETE RETRAINING & PREDICTION PIPELINE (CSV-BASED)")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Retrain models
    logger.info("STEP 1: Retraining all position-specific models...")
    retrainer = ModelRetrainer()
    training_results = retrainer.retrain_all_models()
    
    # Step 2: Generate predictions
    logger.info("\n\nSTEP 2: Generating predictions for all players...")
    predictor = PredictionGenerator()
    predictions_df = predictor.predict_all_players()
    
    # Step 3: Save predictions
    logger.info("\n\nSTEP 3: Saving predictions...")
    predictor.save_predictions(predictions_df)
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nPredictions saved to: ml/predictions/next_gw_predictions.csv")
    print(f"Total predictions: {len(predictions_df):,}")
    print(f"\nModel Performance Summary:")
    for result in training_results:
        print(f"  {result['position']}: R²={result['r2']:.3f}, RMSE={result['rmse']:.3f}")


if __name__ == '__main__':
    main()
