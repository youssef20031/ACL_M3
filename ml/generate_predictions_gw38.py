"""
Generate predictions using active players from GW38 (latest complete GW)
Uses existing XGBoost v2 models with updated 2025-26 data
"""
import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_engineering import FeatureEngineer
from ml.promoted_teams_handler import PromotedTeamsHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuickPredictor:
    """Generate predictions for active players using existing models."""
    
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
        """Load trained v2 models."""
        positions = ['GK', 'DEF', 'MID', 'FWD']
        
        for pos in positions:
            model_path = f"ml/models/xgboost_{pos.lower()}_v2.pkl"
            mappings_path = model_path.replace('.pkl', '_mappings.json')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[pos] = pickle.load(f)
                
                # Load feature mappings (once for first position)
                if os.path.exists(mappings_path) and not hasattr(self.feature_engineer, '_mappings_loaded'):
                    self.feature_engineer.load_mappings(mappings_path)
                    self.feature_engineer._mappings_loaded = True
                
                logger.info(f"✅ Loaded {pos} model (v2)")
            else:
                logger.error(f"⚠️ Model not found: {model_path}")
    
    def load_data(self):
        """Load historical player data and fixtures."""
        logger.info("Loading historical data and fixtures...")
        
        # Load historical data
        self.historical_data = pd.read_csv('cleaned_merged_seasons_cleaned.csv')
        
        # Load fixtures
        self.fixtures = pd.read_csv('FPL_2026_2027_fixtures_processed.csv')
        
        logger.info(f"✅ Loaded {len(self.historical_data):,} historical records")
        logger.info(f"✅ Loaded {len(self.fixtures):,} fixtures for 2026-27")
    
    def get_active_players(self) -> pd.DataFrame:
        """Get active players from latest season GW38 (now the latest complete GW)."""
        df = self.historical_data.copy()
        
        # Determine column names
        season_col = 'season_x' if 'season_x' in df.columns else 'season'
        team_col = 'team_x' if 'team_x' in df.columns else 'team'
        
        # Get latest season
        latest_season = df[season_col].max()
        logger.info(f"Latest season: {latest_season}")
        
        # Get latest GW
        latest_gw = df[df[season_col] == latest_season]['GW'].max()
        logger.info(f"Latest GW: {latest_gw}")
        
        # Get players from latest GW (current active players)
        active = df[(df[season_col] == latest_season) & (df['GW'] == latest_gw)].copy()
        
        logger.info(f"Found {len(active):,} records for GW{latest_gw}")
        logger.info(f"Unique players: {active['name'].nunique()}")
        
        # Keep only needed columns and drop duplicates
        players = active[['name', 'position', team_col, 'value']].drop_duplicates(subset=['name'])
        players = players.rename(columns={team_col: 'team'})
        
        logger.info(f"Active players to predict: {len(players)}")
        
        return players.reset_index(drop=True)
    
    def get_player_recent_stats(self, player_name: str, team: str) -> dict:
        """Get player's recent 5-game stats."""
        team_col = 'team_x' if 'team_x' in self.historical_data.columns else 'team'
        
        player_data = self.historical_data[
            (self.historical_data['name'] == player_name) & 
            (self.historical_data[team_col] == team)
        ].copy()
        
        if len(player_data) == 0:
            return self._default_stats()
        
        # Sort by date and get last 5 games
        player_data['kickoff_time'] = pd.to_datetime(player_data['kickoff_time'])
        player_data = player_data.sort_values('kickoff_time', ascending=False).head(5)
        
        # Calculate averages
        stats = {
            'minutes': player_data['minutes'].mean(),
            'goals_scored': player_data['goals_scored'].mean(),
            'assists': player_data['assists'].mean(),
            'total_points': player_data['total_points'].mean(),
            'clean_sheets': player_data['clean_sheets'].mean(),
            'goals_conceded': player_data['goals_conceded'].mean(),
            'saves': player_data['saves'].mean() if 'saves' in player_data else 0,
            'bonus': player_data['bonus'].mean(),
            'bps': player_data['bps'].mean(),
            'influence': player_data['influence'].mean(),
            'creativity': player_data['creativity'].mean(),
            'threat': player_data['threat'].mean(),
            'ict_index': player_data['ict_index'].mean()
        }
        
        return stats
    
    def _default_stats(self) -> dict:
        """Return default stats for players with no history."""
        return {
            'minutes': 0, 'goals_scored': 0, 'assists': 0, 'total_points': 0,
            'clean_sheets': 0, 'goals_conceded': 0, 'saves': 0, 'bonus': 0,
            'bps': 0, 'influence': 0, 'creativity': 0, 'threat': 0, 'ict_index': 0
        }
    
    def predict_all(self) -> pd.DataFrame:
        """Generate predictions for all active players."""
        logger.info("\\n" + "="*80)
        logger.info("GENERATING PREDICTIONS FOR ACTIVE PLAYERS")
        logger.info("="*80)
        
        # Get active players
        players_df = self.get_active_players()
        
        # Get GW1 fixtures
        next_gw = self.fixtures[self.fixtures['gameweek'] == 1].copy()
        logger.info(f"GW1 fixtures: {len(next_gw)}")
        
        predictions = []
        season_col = 'season_x' if 'season_x' in self.historical_data.columns else 'season'
        
        for idx, player in players_df.iterrows():
            player_name = player['name']
            team = player['team']
            position = player['position']
            
            # Find fixture
            fixture = next_gw[
                (next_gw['home_team'] == team) | 
                (next_gw['away_team'] == team)
            ]
            
            if len(fixture) == 0:
                continue
            
            fixture = fixture.iloc[0]
            
            try:
                # Get recent stats
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
                    'value': player['value'],
                    'GW': 1,
                    season_col: '2026-27',
                    'was_home': is_home,
                    'opp_team_name': opponent,
                    'kickoff_time': fixture['kickoff_time'],
                    **recent_stats
                }
                
                # Convert to DataFrame
                player_df = pd.DataFrame([player_features])
                
                # Engineer features
                player_engineered = self.feature_engineer.engineer_features(
                    player_df,
                    is_training=False,
                    lag_features=None
                )
                
                # Prepare for model
                X, _ = self.feature_engineer.prepare_features(
                    player_engineered,
                    include_target=False
                )
                
                # Get model and predict
                if position not in self.models:
                    continue
                
                model = self.models[position]
                prediction = max(0, model.predict(X)[0])
                
                # Check for promoted teams
                promoted_teams = ['Ipswich Town', 'Leicester City', 'Southampton']
                is_promoted = team in promoted_teams
                
                predictions.append({
                    'name': player_name,
                    'position': position,
                    'team': team,
                    'predicted_points': round(prediction, 2),
                    'value': player['value'],
                    'opponent': opponent,
                    'is_home': is_home,
                    'gameweek': 1,
                    'kickoff_time': fixture['kickoff_time'],
                    'prediction_method': 'xgboost_v2',
                    'is_promoted_team': is_promoted
                })
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1}/{len(players_df)} players...")
                    
            except Exception as e:
                logger.error(f"Error for {player_name}: {e}")
                continue
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('predicted_points', ascending=False)
        
        logger.info(f"\\n✅ Generated {len(predictions_df):,} predictions")
        
        return predictions_df
    
    def save_predictions(self, predictions_df: pd.DataFrame):
        """Save predictions to files."""
        os.makedirs('ml/predictions', exist_ok=True)
        
        # Save CSV
        csv_path = 'ml/predictions/next_gw_predictions.csv'
        predictions_df.to_csv(csv_path, index=False)
        logger.info(f"✅ Saved to {csv_path}")
        
        # Save JSON
        json_path = 'ml/predictions/next_gw_predictions.json'
        predictions_df.to_json(json_path, orient='records', indent=2)
        logger.info(f"✅ Saved to {json_path}")
        
        # Show top picks by position
        logger.info("\\n" + "="*80)
        logger.info("TOP 10 PREDICTED PERFORMERS BY POSITION")
        logger.info("="*80)
        
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_df = predictions_df[predictions_df['position'] == position].head(10)
            logger.info(f"\\n{position}:")
            for _, row in pos_df.iterrows():
                home_marker = 'H' if row['is_home'] else 'A'
                logger.info(
                    f"  {row['predicted_points']:5.2f} pts | "
                    f"{row['name']:30s} | {row['team']:20s} | "
                    f"vs {row['opponent']:20s} ({home_marker})"
                )


def main():
    print("\\n" + "="*80)
    print("GENERATING PREDICTIONS FOR ACTIVE PLAYERS (GW38 -> GW1)")
    print("Using updated 2025-26 data with GW 30-38")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
    
    predictor = QuickPredictor()
    predictions_df = predictor.predict_all()
    predictor.save_predictions(predictions_df)
    
    print("\\n" + "="*80)
    print("✅ PREDICTIONS COMPLETE")
    print("="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\\nTotal predictions: {len(predictions_df):,}")
    print(f"Saved to: ml/predictions/next_gw_predictions.csv")


if __name__ == '__main__':
    main()
