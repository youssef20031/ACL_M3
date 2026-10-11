"""
Promoted Teams Handler V2

Integrates translation models (Model A & B) with archetype database v2.
Replaces flat scalars with learned translations.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class PromotedTeamsHandlerV2:
    """
    V2 handler with translation models integrated.
    """
    
    def __init__(self, 
                 archetype_db_path: str = 'ml/models/promoted_archetypes_v2.pkl',
                 model_a_path: str = 'ml/models/translation_model_a_v1.pkl',
                 model_b_path: str = 'ml/models/translation_model_b_v1.pkl'):
        """
        Initialize V2 handler.
        
        Args:
            archetype_db_path: Path to expanded archetype database
            model_a_path: Path to Model A (team translation)
            model_b_path: Path to Model B (player translation)
        """
        self.archetype_db = None
        self.model_a = None
        self.model_b = None
        
        # Load archetype database
        if os.path.exists(archetype_db_path):
            with open(archetype_db_path, 'rb') as f:
                self.archetype_db = pickle.load(f)
            logger.info(f"Loaded {len(self.archetype_db)} archetypes (v2)")
        
        # Load translation models
        if os.path.exists(model_a_path):
            from ml.translation.model_a_team import TeamTranslationModel
            self.model_a = TeamTranslationModel()
            self.model_a.load()
            logger.info("Loaded Model A (team translation)")
        
        if os.path.exists(model_b_path):
            from ml.translation.model_b_player import PlayerTranslationModel
            self.model_b = PlayerTranslationModel()
            self.model_b.load()
            logger.info("Loaded Model B (player translation)")
    
    def scale_championship_stats(self,
                                 player_data: Dict[str, Any],
                                 team_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        V2 scaling using Model A & B instead of flat scalars.
        
        Args:
            player_data: Championship player stats
            team_data: Championship team stats (optional, for Model A)
            
        Returns:
            EPL-scaled statistics
        """
        scaled_data = player_data.copy()
        
        # Model A: Translate team strength
        if self.model_a and team_data:
            try:
                team_translation = self.model_a.predict(team_data)
                scaled_data['translated_attack_strength'] = team_translation['epl_attack_strength']
                scaled_data['translated_defensive_strength'] = team_translation['epl_defensive_strength']
                scaled_data['translated_team_ppg'] = team_translation['epl_team_ppg']
                logger.debug(f"Model A: attack={team_translation['epl_attack_strength']:.2f}, defense={team_translation['epl_defensive_strength']:.2f}")
            except Exception as e:
                logger.warning(f"Model A failed: {e}")
        
        # Model B: Translate player per-90 stats
        if self.model_b:
            try:
                player_translation = self.model_b.predict(player_data)
                
                # Replace Championship stats with EPL-equivalent
                for stat, value in player_translation.items():
                    # Map EPL stat names to FPL stat names
                    fpl_stat = stat.replace('epl_', '').replace('_per90', '')
                    scaled_data[fpl_stat] = value
                
                logger.debug(f"Model B translated {len(player_translation)} stats")
            except Exception as e:
                logger.warning(f"Model B failed: {e}")
        
        # Fallback to flat scalars if models unavailable
        if not self.model_b:
            LEAGUE_SCALARS = {
                'goals_scored': 0.70,
                'assists': 0.75,
                'form': 0.70
            }
            for feature, scalar in LEAGUE_SCALARS.items():
                if feature in scaled_data:
                    scaled_data[feature] = scaled_data[feature] * scalar
        
        scaled_data['is_promoted'] = True
        scaled_data['league_origin'] = 0.70
        
        return scaled_data
    
    def find_similar_archetypes(self,
                                player_data: Dict[str, Any],
                                k: int = 5) -> List[Dict[str, Any]]:
        """
        Find k similar archetypes using expanded database.
        
        Args:
            player_data: Player stats
            k: Number of neighbors
            
        Returns:
            List of similar archetypes
        """
        if not self.archetype_db:
            logger.warning("Archetype database not loaded")
            return []
        
        from sklearn.neighbors import NearestNeighbors
        from ml.promoted_teams_handler import PromotedTeamsHandler
        
        # Create embedding for new player
        handler = PromotedTeamsHandler()
        new_embedding = handler.create_player_embedding(player_data)
        
        # Extract embeddings from database
        embeddings = np.array([arch['embedding'] for arch in self.archetype_db])
        
        # KNN search
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(embeddings)
        
        distances, indices = knn.kneighbors([new_embedding])
        
        # Return similar archetypes
        similar = []
        for i, idx in enumerate(indices[0]):
            archetype = self.archetype_db[idx].copy()
            archetype['similarity'] = 1 - distances[0][i]
            similar.append(archetype)
        
        return similar
    
    def predict_promoted_player(self,
                               player_data: Dict[str, Any],
                               team_data: Optional[Dict[str, Any]] = None,
                               current_gw: int = 1) -> Tuple[float, Dict[str, Any]]:
        """
        Predict points for promoted player using V2 pipeline.
        
        Args:
            player_data: Player stats
            team_data: Team stats (for Model A)
            current_gw: Current gameweek
            
        Returns:
            Tuple of (predicted_points, metadata)
        """
        # Scale stats using Models A & B
        scaled_data = self.scale_championship_stats(player_data, team_data)
        
        # Find similar archetypes (expanded DB)
        similar = self.find_similar_archetypes(scaled_data, k=10)
        
        # Predict from archetypes
        if similar:
            weights = [arch['similarity'] for arch in similar]
            points = [arch.get('epl_avg_points', 4.0) for arch in similar]
            
            weighted_pred = np.average(points, weights=weights)
        else:
            # Fallback to position baseline
            position_baselines = {'GK': 3.2, 'DEF': 3.8, 'MID': 4.1, 'FWD': 4.5}
            weighted_pred = position_baselines.get(player_data.get('position', 'MID'), 4.0)
        
        # Apply GW-based confidence decay
        if current_gw <= 6:
            confidence = 0.60
        elif current_gw <= 12:
            confidence = 0.75
        else:
            confidence = 0.90
        
        metadata = {
            'method': 'translation_models_v2',
            'confidence': confidence,
            'archetypes_used': len(similar),
            'model_a_available': self.model_a is not None,
            'model_b_available': self.model_b is not None
        }
        
        return float(weighted_pred), metadata
