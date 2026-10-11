"""
Retrain ML Model with All New Features and Generate Predictions
Includes: Promoted teams handling, fixture congestion, all robustness improvements
"""
import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
from neo4j import GraphDatabase
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_engineering import FeatureEngineer
from ml.promoted_teams_handler import PromotedTeamsHandler
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

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
        logger.info(f"   Columns: {list(df.columns)}")
        
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
    """Generate predictions for all players' next games."""
    
    def __init__(self):
        self.db = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        self.feature_engineer = FeatureEngineer()
        self.promoted_handler = PromotedTeamsHandler(
            archetype_db_path='ml/models/promoted_archetypes.pkl'
        )
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models."""
        positions = ['GK', 'DEF', 'MID', 'FWD']
        
        for pos in positions:
            model_path = f"ml/models/xgboost_{pos.lower()}_v2.pkl"
            mappings_path = model_path.replace('.pkl', '_mappings.json')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[pos] = pickle.load(f)
                
                # Load feature mappings
                if os.path.exists(mappings_path):
                    if pos == list(self.models.keys())[0]:  # Load once
                        self.feature_engineer.load_mappings(mappings_path)
                
                logger.info(f"✅ Loaded {pos} model")
            else:
                logger.warning(f"⚠️ Model not found: {model_path}")
    
    def fetch_current_players_with_fixtures(self) -> pd.DataFrame:
        """Fetch all active players with their next fixtures."""
        logger.info("Fetching current players and upcoming fixtures...")
        
        query = """
        // Get all players
        MATCH (p:Player)
        WHERE p.season = '2026-27'
        
        // Get their recent performance (last 5 games)
        OPTIONAL MATCH (p)-[r:PLAYED_IN]->(g:Game)-[:IN_GAMEWEEK]->(gw:Gameweek)
        WHERE gw.season = '2026-27'
        WITH p, r, g, gw
        ORDER BY g.kickoff_time DESC
        WITH p, collect({
            minutes: r.minutes,
            goals_scored: r.goals_scored,
            assists: r.assists,
            total_points: r.total_points,
            clean_sheets: r.clean_sheets,
            goals_conceded: r.goals_conceded,
            saves: r.saves,
            bonus: r.bonus,
            bps: r.bps,
            influence: r.influence,
            creativity: r.creativity,
            threat: r.threat,
            ict_index: r.ict_index,
            kickoff_time: g.kickoff_time,
            gw: gw.id,
            was_home: CASE WHEN (g)-[:HOME_TEAM]->(:Team {name: p.team}) THEN true ELSE false END
        })[0..5] as recent_games
        
        // Get next fixture
        OPTIONAL MATCH (next_fixture:Fixture)
        WHERE next_fixture.season = '2026-27'
          AND (next_fixture.home_team = p.team OR next_fixture.away_team = p.team)
          AND datetime(next_fixture.kickoff_time) > datetime()
        WITH p, recent_games, next_fixture
        ORDER BY next_fixture.kickoff_time
        WITH p, recent_games, collect(next_fixture)[0] as next_fixture
        
        WHERE next_fixture IS NOT NULL
        
        RETURN 
            p.id as player_id,
            p.name as name,
            p.position as position,
            p.team as team,
            p.value as value,
            p.selected_by_percent as ownership_pct,
            recent_games,
            next_fixture.id as fixture_id,
            next_fixture.kickoff_time as next_kickoff,
            next_fixture.home_team as next_home_team,
            next_fixture.away_team as next_away_team,
            next_fixture.gameweek as next_gw
        ORDER BY p.name
        """
        
        results = self.db.execute_query(query)
        df = pd.DataFrame(results)
        logger.info(f"✅ Fetched {len(df):,} players with upcoming fixtures")
        
        return df
    
    def prepare_player_features(self, player_row: pd.Series) -> dict:
        """Prepare features for a single player prediction."""
        # Extract recent game stats
        recent_games = player_row.get('recent_games', [])
        
        if not recent_games or len(recent_games) == 0:
            # New player or no history - use defaults
            avg_stats = {
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
        else:
            # Calculate averages from recent games
            avg_stats = {
                'minutes': np.mean([g.get('minutes', 0) for g in recent_games]),
                'goals_scored': np.mean([g.get('goals_scored', 0) for g in recent_games]),
                'assists': np.mean([g.get('assists', 0) for g in recent_games]),
                'total_points': np.mean([g.get('total_points', 0) for g in recent_games]),
                'clean_sheets': np.mean([g.get('clean_sheets', 0) for g in recent_games]),
                'goals_conceded': np.mean([g.get('goals_conceded', 0) for g in recent_games]),
                'saves': np.mean([g.get('saves', 0) for g in recent_games]),
                'bonus': np.mean([g.get('bonus', 0) for g in recent_games]),
                'bps': np.mean([g.get('bps', 0) for g in recent_games]),
                'influence': np.mean([g.get('influence', 0) for g in recent_games]),
                'creativity': np.mean([g.get('creativity', 0) for g in recent_games]),
                'threat': np.mean([g.get('threat', 0) for g in recent_games]),
                'ict_index': np.mean([g.get('ict_index', 0) for g in recent_games])
            }
        
        # Determine home/away
        is_home = player_row['team'] == player_row['next_home_team']
        opponent = player_row['next_away_team'] if is_home else player_row['next_home_team']
        
        # Build feature dict
        features = {
            'name': player_row['name'],
            'player_id': player_row['player_id'],
            'position': player_row['position'],
            'team': player_row['team'],
            'team_x': player_row['team'],
            'value': player_row['value'],
            'GW': int(player_row['next_gw']),
            'season_x': '2026-27',
            'was_home': is_home,
            'opp_team_name': opponent,
            'kickoff_time': player_row['next_kickoff'],
            **avg_stats
        }
        
        return features
    
    def predict_all_players(self) -> pd.DataFrame:
        """Generate predictions for all players' next games."""
        logger.info("\n" + "="*80)
        logger.info("GENERATING PREDICTIONS FOR ALL PLAYERS")
        logger.info("="*80)
        
        # Fetch players
        players_df = self.fetch_current_players_with_fixtures()
        
        if len(players_df) == 0:
            logger.error("No players with upcoming fixtures found!")
            return pd.DataFrame()
        
        predictions = []
        
        for idx, player_row in players_df.iterrows():
            try:
                # Prepare features
                player_features = self.prepare_player_features(player_row)
                
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
                position = player_features['position']
                if position not in self.models:
                    logger.warning(f"No model for position {position}")
                    continue
                
                model = self.models[position]
                
                # Predict
                prediction = model.predict(X)[0]
                prediction = max(0, prediction)  # Ensure non-negative
                
                # Check if promoted team (for cold start handling)
                promoted_teams = ['Ipswich Town', 'Leicester City', 'Southampton']
                is_promoted = player_features['team'] in promoted_teams
                
                # Apply promoted team adjustment if needed
                if is_promoted and player_features['GW'] <= 12:
                    adjusted_pred, metadata = self.promoted_handler.predict_promoted_player(
                        player_features,
                        current_gw=player_features['GW'],
                        use_archetypes=True
                    )
                    prediction = adjusted_pred
                    prediction_method = metadata.get('method', 'promoted_adjusted')
                else:
                    prediction_method = 'xgboost_v2'
                
                predictions.append({
                    'player_id': player_features['player_id'],
                    'name': player_features['name'],
                    'position': position,
                    'team': player_features['team'],
                    'predicted_points': round(prediction, 2),
                    'value': player_features['value'],
                    'opponent': player_features['opp_team_name'],
                    'is_home': player_features['was_home'],
                    'gameweek': player_features['GW'],
                    'kickoff_time': player_features['kickoff_time'],
                    'prediction_method': prediction_method,
                    'is_promoted_team': is_promoted
                })
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(players_df)} players...")
                    
            except Exception as e:
                logger.error(f"Error predicting for {player_row.get('name', 'Unknown')}: {e}")
                continue
        
        predictions_df = pd.DataFrame(predictions)
        
        # Sort by predicted points
        predictions_df = predictions_df.sort_values('predicted_points', ascending=False)
        
        logger.info(f"\n✅ Generated {len(predictions_df):,} predictions")
        
        return predictions_df
    
    def save_predictions(self, predictions_df: pd.DataFrame):
        """Save predictions to file and database."""
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
    print("FPL ML MODEL - COMPLETE RETRAINING & PREDICTION PIPELINE")
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


if __name__ == '__main__':
    main()
