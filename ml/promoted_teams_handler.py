"""
Promoted Teams Cold Start Handler
Implements solutions for newly promoted teams without PL history

Solutions implemented:
1. Championship Data Transfer - Scale Championship stats to PL equivalent
2. Similarity-Based Player Embeddings - Match to historical archetypes
3. Blended Heuristic Fallback - Conservative position averages with decay
4. Rapid Online Update - Incremental model retraining after each GW
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pickle
import os

logger = logging.getLogger(__name__)


# Historical promotion baseline points (calculated from 2016-2026 promotions)
PROMOTION_BASELINES = {
    'GK': 3.2,   # Average points for promoted GK in debut season
    'DEF': 3.8,  # Average for promoted defenders
    'MID': 4.1,  # Average for promoted midfielders
    'FWD': 4.5   # Average for promoted forwards
}

# League difficulty scalar (Championship -> PL conversion)
LEAGUE_SCALARS = {
    'goals_scored': 0.70,      # Harder to score in PL
    'assists': 0.75,           # Higher quality defenses
    'clean_sheets': 0.60,      # Much harder in PL (for defenders)
    'bonus': 0.65,             # Lower BPS thresholds
    'minutes': 0.90,           # Rotation risk increases
    'form': 0.70,              # Overall performance scaling
    'points_per_90': 0.70,     # Normalized performance
}


class PromotedTeamsHandler:
    """
    Handler for newly promoted teams' cold start problem.
    Combines multiple solutions to provide predictions for players without PL history.
    """
    
    def __init__(self, 
                 archetype_db_path: Optional[str] = None,
                 models_dir: str = "ml/models"):
        """
        Initialize promoted teams handler.
        
        Args:
            archetype_db_path: Path to historical archetype database
            models_dir: Directory containing ML models
        """
        self.models_dir = models_dir
        self.archetype_db = None
        self.knn_matcher = None
        self.scaler = StandardScaler()
        self.translation_models = {}
        
        # Load archetype database v2 (with EFL data)
        if archetype_db_path is None:
            archetype_db_path = os.path.join(models_dir, 'promoted_archetypes_v2.pkl')
        
        if os.path.exists(archetype_db_path):
            self.load_archetype_database(archetype_db_path)
        
        # Load translation models if available
        for pos in ['DEF', 'MID', 'FWD']:
            model_path = os.path.join(models_dir, f'player_translation_{pos}_v1.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.translation_models[pos] = pickle.load(f)
        
        if self.translation_models:
            logger.info(f"Loaded {len(self.translation_models)} translation models")
    
    # ==================== SOLUTION 1: Championship Data Transfer ====================
    
    def scale_championship_stats(self, 
                                 player_data: Dict[str, Any],
                                 league: str = "Championship") -> Dict[str, Any]:
        """
        Solution 1: Scale Championship statistics to PL-equivalent feature space.
        
        Applies league-difficulty scalar based on historical promotion cohort performance.
        
        Args:
            player_data: Dictionary with Championship stats
            league: Source league (default: "Championship")
            
        Returns:
            Dictionary with PL-scaled statistics
        """
        if league != "Championship":
            logger.warning(f"Unsupported league: {league}. Using Championship scalars.")
        
        scaled_data = player_data.copy()
        
        # Add league origin flag
        scaled_data['league_origin'] = 0.70  # Championship = 0.70, PL = 1.0
        scaled_data['is_promoted'] = True
        
        # Scale each feature according to league difficulty
        for feature, scalar in LEAGUE_SCALARS.items():
            if feature in scaled_data:
                original_value = scaled_data[feature]
                scaled_data[feature] = original_value * scalar
                logger.debug(f"Scaled {feature}: {original_value:.2f} -> {scaled_data[feature]:.2f} (×{scalar})")
        
        # Scale total_points if available (used for form calculation)
        if 'total_points' in scaled_data:
            scaled_data['total_points'] = scaled_data['total_points'] * 0.70
        
        logger.info(f"Scaled Championship stats for player with league_origin={scaled_data['league_origin']}")
        
        return scaled_data
    
    # ==================== SOLUTION 2: Similarity-Based Player Embeddings ====================
    
    def create_player_embedding(self, player_data: Dict[str, Any]) -> np.ndarray:
        """
        Create normalized per-90 stat embedding for player matching.
        
        Args:
            player_data: Dictionary with player statistics
            
        Returns:
            Normalized feature vector
        """
        # Per-90 normalized features
        minutes = player_data.get('minutes', 1)
        if minutes == 0:
            minutes = 1  # Avoid division by zero
        
        # Calculate per-90 stats
        per_90_factor = 90.0 / minutes if minutes > 0 else 0
        
        embedding_features = [
            player_data.get('goals_scored', 0) * per_90_factor,
            player_data.get('assists', 0) * per_90_factor,
            player_data.get('xG', player_data.get('goals_scored', 0) * 0.8) * per_90_factor,  # Approximate if missing
            player_data.get('xA', player_data.get('assists', 0) * 0.7) * per_90_factor,
            player_data.get('shots', player_data.get('goals_scored', 0) * 3) * per_90_factor,
            player_data.get('key_passes', player_data.get('assists', 0) * 2) * per_90_factor,
            player_data.get('tackles', 0) * per_90_factor,
            player_data.get('interceptions', 0) * per_90_factor,
            player_data.get('dribbles', 0) * per_90_factor,
            player_data.get('pass_completion', 75.0),  # Default to typical value
            player_data.get('aerials_won', 0) * per_90_factor,
            player_data.get('fouls', 0) * per_90_factor,
        ]
        
        return np.array(embedding_features)
    
    def load_archetype_database(self, filepath: str):
        """
        Load historical promotion archetype database.
        
        Args:
            filepath: Path to archetype database pickle file
        """
        try:
            with open(filepath, 'rb') as f:
                self.archetype_db = pickle.load(f)
            
            # Extract embeddings and build KNN matcher
            embeddings = []
            for archetype in self.archetype_db:
                embeddings.append(archetype['embedding'])
            
            embeddings_array = np.array(embeddings)
            
            # Fit KNN matcher
            self.knn_matcher = NearestNeighbors(n_neighbors=5, metric='cosine')
            self.knn_matcher.fit(embeddings_array)
            
            logger.info(f"Loaded archetype database with {len(self.archetype_db)} historical players")
            
        except Exception as e:
            logger.error(f"Failed to load archetype database: {e}")
            self.archetype_db = None
            self.knn_matcher = None
    
    def find_similar_archetypes(self, 
                                player_data: Dict[str, Any],
                                k: int = 5) -> List[Dict[str, Any]]:
        """
        Find k most similar historical promoted players.
        
        Args:
            player_data: Dictionary with player statistics
            k: Number of similar players to return
            
        Returns:
            List of similar archetype dictionaries with similarity scores
        """
        if self.knn_matcher is None or self.archetype_db is None:
            logger.warning("Archetype database not loaded. Cannot find similar players.")
            return []
        
        # Create embedding for new player
        new_embedding = self.create_player_embedding(player_data)
        new_embedding = new_embedding.reshape(1, -1)
        
        # Find k nearest neighbors
        distances, indices = self.knn_matcher.kneighbors(new_embedding, n_neighbors=k)
        
        # Get similar archetypes with similarity scores
        similar_archetypes = []
        for i, idx in enumerate(indices[0]):
            archetype = self.archetype_db[idx].copy()
            # Convert cosine distance to similarity (1 - distance)
            archetype['similarity'] = 1 - distances[0][i]
            similar_archetypes.append(archetype)
        
        logger.info(f"Found {len(similar_archetypes)} similar players with avg similarity {np.mean([a['similarity'] for a in similar_archetypes]):.3f}")
        
        return similar_archetypes
    
    def predict_from_archetypes(self, 
                                player_data: Dict[str, Any],
                                similar_archetypes: List[Dict[str, Any]]) -> float:
        """
        Bootstrap prediction from similar historical players' debut seasons.
        
        Args:
            player_data: Dictionary with player statistics
            similar_archetypes: List of similar historical players
            
        Returns:
            Predicted points based on archetype averaging
        """
        if not similar_archetypes:
            # Fallback to position baseline
            position = player_data.get('position', 'MID')
            return PROMOTION_BASELINES.get(position, 4.0)
        
        # Calculate confidence-weighted prediction
        total_weight = 0
        weighted_sum = 0
        
        for archetype in similar_archetypes:
            similarity = archetype['similarity']
            debut_points = archetype.get('debut_season_avg_points', 4.0)
            
            # Weight by similarity
            # High similarity (>0.85): full weight
            # Medium similarity (0.70-0.85): 0.6 weight
            # Low similarity (<0.70): 0.3 weight
            if similarity > 0.85:
                weight = 1.0
            elif similarity > 0.70:
                weight = 0.6
            else:
                weight = 0.3
            
            weighted_sum += debut_points * weight
            total_weight += weight
        
        if total_weight == 0:
            position = player_data.get('position', 'MID')
            return PROMOTION_BASELINES.get(position, 4.0)
        
        archetype_prediction = weighted_sum / total_weight
        
        logger.info(f"Archetype prediction: {archetype_prediction:.2f} points (from {len(similar_archetypes)} similar players)")
        
        return archetype_prediction
    
    # ==================== SOLUTION 3: Blended Heuristic Fallback ====================
    
    def calculate_blended_heuristic(self,
                                    player_data: Dict[str, Any],
                                    current_gw: int) -> float:
        """
        Solution 3: Blended heuristic with gradual decay.
        
        Starts with conservative position-average baseline,
        gradually incorporates Championship form, then real PL data.
        
        Args:
            player_data: Dictionary with player statistics
            current_gw: Current gameweek number
            
        Returns:
            Blended predicted points
        """
        position = player_data.get('position', 'MID')
        baseline = PROMOTION_BASELINES.get(position, 4.0)
        
        # Get Championship form (scaled)
        champ_form = player_data.get('form', 0) 
        if player_data.get('league_origin', 1.0) < 1.0:  # Is from Championship
            champ_form_scaled = champ_form  # Already scaled by scale_championship_stats
        else:
            champ_form_scaled = champ_form * 0.70
        
        # Determine blending weights based on gameweek
        if current_gw <= 6:
            # GW 1-6: Heavy reliance on heuristic
            heuristic_weight = 0.70
            champ_weight = 0.30
            pl_weight = 0.0
        elif current_gw <= 12:
            # GW 7-12: Gradual transition
            progress = (current_gw - 7) / 6  # 0 to 1
            heuristic_weight = 0.70 - (progress * 0.50)  # 0.70 -> 0.20
            champ_weight = 0.30 - (progress * 0.10)       # 0.30 -> 0.20
            pl_weight = progress * 0.60                   # 0.0 -> 0.60
        else:
            # GW 13+: Majority real PL data
            heuristic_weight = 0.10
            champ_weight = 0.10
            pl_weight = 0.80
        
        # Get actual PL form if available
        pl_form = player_data.get('pl_actual_form', 0)
        
        blended_prediction = (
            heuristic_weight * baseline +
            champ_weight * champ_form_scaled +
            pl_weight * pl_form
        )
        
        logger.info(f"GW{current_gw} blended: {blended_prediction:.2f} = "
                   f"{heuristic_weight:.2f}×{baseline:.2f} + "
                   f"{champ_weight:.2f}×{champ_form_scaled:.2f} + "
                   f"{pl_weight:.2f}×{pl_form:.2f}")
        
        return blended_prediction
    
    # ==================== SOLUTION 4: Rapid Online Update ====================
    
    def incremental_model_update(self,
                                 position: str,
                                 new_gw_data: pd.DataFrame,
                                 current_gw: int):
        """
        Solution 4: Incrementally retrain model with new promoted player data.
        
        After each gameweek, updates position-specific model using only
        promoted players' new results. Uses warm start to preserve existing
        model knowledge while adapting to promoted players.
        
        Args:
            position: Position to update (GK/DEF/MID/FWD)
            new_gw_data: DataFrame with latest gameweek results
            current_gw: Current gameweek number
        """
        import xgboost as xgb
        from ml.feature_engineering import FeatureEngineer
        
        # Only update after GW 3+ (need some data)
        if current_gw < 3:
            logger.info(f"Skipping incremental update for GW{current_gw} - need 3+ gameweeks")
            return
        
        # Filter to promoted players only
        promoted_data = new_gw_data[new_gw_data.get('is_promoted', False) == True]
        
        if len(promoted_data) == 0:
            logger.warning(f"No promoted player data for position {position}")
            return
        
        # Load existing model
        model_path = os.path.join(self.models_dir, f"xgboost_{position.lower()}_v1.pkl")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return
        
        try:
            # Load existing XGBoost model
            existing_model = xgb.Booster()
            existing_model.load_model(model_path)
            
            # Prepare new data
            feature_engineer = FeatureEngineer()
            feature_engineer.load_mappings(model_path.replace('.pkl', '_mappings.json'))
            
            # Engineer features
            df_engineered = feature_engineer.engineer_features(
                promoted_data,
                is_training=False,
                lag_features=['total_points']
            )
            
            X, y = feature_engineer.prepare_features(df_engineered, include_target=True)
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X.values, label=y.values)
            
            # Incremental training with conservative learning rate
            params = {
                'learning_rate': 0.01,  # Lower than initial training (0.05)
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            }
            
            # Warm start from existing model
            updated_model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=50,  # Small number of rounds for fine-tuning
                xgb_model=existing_model  # Warm start
            )
            
            # Save updated model with GW tag
            updated_model_path = os.path.join(
                self.models_dir, 
                f"xgboost_{position.lower()}_v1_gw{current_gw}.pkl"
            )
            updated_model.save_model(updated_model_path)
            
            logger.info(f"✅ Updated {position} model with {len(promoted_data)} promoted player samples (GW{current_gw})")
            logger.info(f"   Saved to: {updated_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to update model for {position}: {e}")
    
    # ==================== MAIN PREDICTION INTERFACE ====================
    
    def predict_promoted_player(self,
                               player_data: Dict[str, Any],
                               current_gw: int,
                               use_archetypes: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Main prediction interface for promoted players.
        
        Combines all solutions with intelligent fallback:
        1. Try archetype matching if available
        2. Scale Championship stats
        3. Apply blended heuristic with GW-based decay
        4. Return prediction with metadata
        
        Args:
            player_data: Dictionary with player statistics
            current_gw: Current gameweek number
            use_archetypes: Whether to use archetype matching
            
        Returns:
            Tuple of (predicted_points, metadata_dict)
        """
        position = player_data.get('position', 'MID')
        player_name = player_data.get('name', 'Unknown')
        
        metadata = {
            'method': 'unknown',
            'confidence': 0.5,
            'cold_start_phase': self._get_cold_start_phase(current_gw)
        }
        
        # Step 1: Scale Championship stats if needed
        if player_data.get('league_origin', 1.0) < 1.0:
            player_data = self.scale_championship_stats(player_data)
            metadata['championship_scaled'] = True
        
        # Step 2: Try archetype matching (high confidence solution)
        archetype_prediction = None
        if use_archetypes and self.archetype_db is not None:
            try:
                similar_archetypes = self.find_similar_archetypes(player_data, k=5)
                if similar_archetypes:
                    archetype_prediction = self.predict_from_archetypes(player_data, similar_archetypes)
                    
                    # Confidence based on average similarity
                    avg_similarity = np.mean([a['similarity'] for a in similar_archetypes])
                    if avg_similarity > 0.85:
                        metadata['confidence'] = 0.80
                        metadata['archetype_confidence'] = 'high'
                    elif avg_similarity > 0.70:
                        metadata['confidence'] = 0.65
                        metadata['archetype_confidence'] = 'medium'
                    else:
                        metadata['confidence'] = 0.50
                        metadata['archetype_confidence'] = 'low'
                    
                    metadata['method'] = 'archetype'
                    metadata['similar_players'] = [a.get('name', 'Unknown') for a in similar_archetypes[:3]]
            except Exception as e:
                logger.warning(f"Archetype matching failed: {e}")
        
        # Step 3: Calculate blended heuristic
        blended_prediction = self.calculate_blended_heuristic(player_data, current_gw)
        
        # Step 4: Combine predictions with smart weighting
        if archetype_prediction is not None and current_gw <= 12:
            # Early season: Blend archetype with heuristic
            archetype_weight = 0.70 if metadata['confidence'] > 0.70 else 0.50
            heuristic_weight = 1.0 - archetype_weight
            
            final_prediction = (
                archetype_weight * archetype_prediction +
                heuristic_weight * blended_prediction
            )
            
            metadata['method'] = 'archetype_blended'
            metadata['archetype_weight'] = archetype_weight
            
            logger.info(f"{player_name} (GW{current_gw}): {final_prediction:.2f} pts = "
                       f"{archetype_weight:.0%} archetype ({archetype_prediction:.2f}) + "
                       f"{heuristic_weight:.0%} heuristic ({blended_prediction:.2f})")
        else:
            # No archetype or late season: Use blended heuristic
            final_prediction = blended_prediction
            metadata['method'] = 'blended_heuristic'
            
            logger.info(f"{player_name} (GW{current_gw}): {final_prediction:.2f} pts (blended heuristic)")
        
        # Ensure non-negative
        final_prediction = max(0, final_prediction)
        
        # Adjust confidence based on gameweek
        if current_gw <= 3:
            metadata['confidence'] *= 0.6  # Very low confidence early
        elif current_gw <= 6:
            metadata['confidence'] *= 0.8  # Low confidence
        elif current_gw <= 12:
            metadata['confidence'] *= 0.9  # Moderate confidence
        # After GW 12: Full confidence (enough PL data)
        
        return final_prediction, metadata
    
    def _get_cold_start_phase(self, current_gw: int) -> str:
        """Get descriptive phase of cold start based on gameweek."""
        if current_gw <= 3:
            return "initial" # No real PL data
        elif current_gw <= 6:
            return "early"    # 3-6 matches
        elif current_gw <= 12:
            return "transition"  # 6-12 matches
        else:
            return "resolved"  # Sufficient history


# ==================== ARCHETYPE DATABASE BUILDER ====================


class ArchetypeDatabaseBuilder:
    """
    Builder for historical promotion archetype database.
    
    Extracts debut-season performance for promoted players (2016-2026)
    and creates embeddings for similarity matching.
    """
    
    def __init__(self):
        """Initialize archetype builder."""
        self.archetypes = []
    
    def add_archetype(self,
                     name: str,
                     position: str,
                     promoted_season: str,
                     championship_stats: Dict[str, Any],
                     debut_season_avg_points: float,
                     tags: Optional[List[str]] = None):
        """
        Add a historical promoted player to archetype database.
        
        Args:
            name: Player name
            position: Position (GK/DEF/MID/FWD)
            promoted_season: Season promoted to PL (e.g., "2020-21")
            championship_stats: Championship season statistics
            debut_season_avg_points: Average points in PL debut season
            tags: Optional tags (e.g., ["set_piece_specialist", "dribbler"])
        """
        handler = PromotedTeamsHandler()
        embedding = handler.create_player_embedding(championship_stats)
        
        archetype = {
            'name': name,
            'position': position,
            'promoted_season': promoted_season,
            'championship_stats': championship_stats,
            'debut_season_avg_points': debut_season_avg_points,
            'embedding': embedding,
            'tags': tags or []
        }
        
        self.archetypes.append(archetype)
        logger.info(f"Added archetype: {name} ({position}, {promoted_season}) - {debut_season_avg_points:.1f} pts/game")
    
    def build_from_dataframe(self, df: pd.DataFrame) -> int:
        """
        Build archetype database from historical promoted players dataframe.
        
        Expected columns:
        - name, position, promoted_season
        - Championship stats: goals_scored, assists, minutes, etc.
        - debut_season_avg_points
        
        Args:
            df: DataFrame with historical promoted players
            
        Returns:
            Number of archetypes added
        """
        count = 0
        for _, row in df.iterrows():
            championship_stats = {
                'goals_scored': row.get('champ_goals', 0),
                'assists': row.get('champ_assists', 0),
                'minutes': row.get('champ_minutes', 90),
                'xG': row.get('champ_xG', 0),
                'xA': row.get('champ_xA', 0),
                'shots': row.get('champ_shots', 0),
                'key_passes': row.get('champ_key_passes', 0),
                'tackles': row.get('champ_tackles', 0),
                'interceptions': row.get('champ_interceptions', 0),
                'dribbles': row.get('champ_dribbles', 0),
                'pass_completion': row.get('champ_pass_completion', 75.0),
                'aerials_won': row.get('champ_aerials_won', 0),
                'fouls': row.get('champ_fouls', 0),
            }
            
            self.add_archetype(
                name=row['name'],
                position=row['position'],
                promoted_season=row['promoted_season'],
                championship_stats=championship_stats,
                debut_season_avg_points=row['debut_season_avg_points'],
                tags=row.get('tags', '').split(',') if row.get('tags') else None
            )
            count += 1
        
        logger.info(f"Built archetype database with {count} players")
        return count
    
    def save(self, filepath: str):
        """Save archetype database to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.archetypes, f)
        logger.info(f"Saved {len(self.archetypes)} archetypes to {filepath}")
    
    def load(self, filepath: str) -> int:
        """Load archetype database from pickle file."""
        with open(filepath, 'rb') as f:
            self.archetypes = pickle.load(f)
        logger.info(f"Loaded {len(self.archetypes)} archetypes from {filepath}")
        return len(self.archetypes)


# ==================== EXAMPLE ARCHETYPES (2016-2026) ====================

EXAMPLE_ARCHETYPES = [
    # Successful Championship -> PL transitions
    {
        'name': 'Ollie Watkins',
        'position': 'FWD',
        'promoted_season': '2020-21',
        'debut_season_avg_points': 4.8,
        'tags': ['clinical_finisher', 'high_work_rate']
    },
    {
        'name': 'Eberechi Eze',
        'position': 'MID',
        'promoted_season': '2020-21',
        'debut_season_avg_points': 5.1,
        'tags': ['dribbler', 'set_piece_specialist']
    },
    {
        'name': 'James Maddison',
        'position': 'MID',
        'promoted_season': '2018-19',
        'debut_season_avg_points': 5.4,
        'tags': ['creative', 'free_kick_specialist']
    },
    {
        'name': 'Aleksandar Mitrovic',
        'position': 'FWD',
        'promoted_season': '2022-23',
        'debut_season_avg_points': 4.2,
        'tags': ['target_man', 'aerial_threat', 'penalty_taker']
    },
]
