"""
Archetype Database Builder v2 (Automated)

Builds expanded archetype database from historical promoted players (2016-2026).
Expands from 6 manual archetypes to 80-100+ players with k-means clustering.
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArchetypeDatabaseBuilderV2:
    """
    Automated builder for promoted player archetype database.
    """
    
    def __init__(self, 
                 championship_data_path: str = 'ml/data/championship_player_stats.csv',
                 epl_data_paths: List[str] = None):
        """
        Initialize builder.
        
        Args:
            championship_data_path: Path to Championship player CSV
            epl_data_paths: List of EPL season CSV paths
        """
        self.champ_path = championship_data_path
        self.epl_paths = epl_data_paths or [
            'FPL_2023_2024.csv',
            'FPL_2024_2025.csv',
            'FPL_2025_2026.csv'
        ]
        self.archetypes = []
        self.scaler = StandardScaler()
    
    def load_championship_data(self) -> pd.DataFrame:
        """Load Championship player stats."""
        if not os.path.exists(self.champ_path):
            raise FileNotFoundError(f"Championship data not found: {self.champ_path}")
        
        df = pd.read_csv(self.champ_path)
        logger.info(f"Loaded {len(df):,} Championship player-season records")
        return df
    
    def load_epl_data(self) -> pd.DataFrame:
        """Load EPL player stats."""
        dfs = []
        for path in self.epl_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                dfs.append(df)
                logger.info(f"Loaded {len(df):,} records from {path}")
        
        if not dfs:
            raise FileNotFoundError("No EPL data files found")
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined EPL data: {len(combined):,} records")
        return combined
    
    def create_per90_embedding(self, row: pd.Series) -> np.ndarray:
        """
        Create 14-dimensional per-90 embedding vector.
        
        Args:
            row: Player stats row
            
        Returns:
            14-dim numpy array
        """
        # Extract per-90 stats (or calculate if totals provided)
        minutes = max(row.get('minutes', row.get('90s', 1) * 90), 1)
        per90_factor = 90.0 / minutes
        
        embedding = [
            row.get('goals_per90', row.get('goals', 0) * per90_factor),
            row.get('assists_per90', row.get('assists', 0) * per90_factor),
            row.get('xg_per90', row.get('xg', 0) * per90_factor),
            row.get('xa_per90', row.get('xa', 0) * per90_factor),
            row.get('shots_per90', row.get('shots', 0) * per90_factor),
            row.get('key_passes_per90', row.get('key_passes', 0) * per90_factor),
            row.get('tackles_per90', row.get('tackles', 0) * per90_factor),
            row.get('interceptions_per90', row.get('interceptions', 0) * per90_factor),
            row.get('dribbles_per90', row.get('dribbles_completed', 0) * per90_factor),
            row.get('pass_completion_pct', row.get('pass_completion', 75.0)),
            row.get('aerials_won_per90', row.get('aerials_won', 0) * per90_factor),
            row.get('progressive_carries_per90', row.get('progressive_carries', 0) * per90_factor),
            row.get('sca_per90', row.get('sca', 0) * per90_factor),
            row.get('gca_per90', row.get('gca', 0) * per90_factor),
        ]
        
        return np.array(embedding, dtype=float)
    
    def match_promoted_players(self, 
                               champ_df: pd.DataFrame,
                               epl_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match Championship promoted players to their EPL debut seasons.
        
        Args:
            champ_df: Championship data (with was_promoted flags)
            epl_df: EPL data
            
        Returns:
            Matched dataframe with both Championship and EPL stats
        """
        # Filter to promoted players only
        promoted = champ_df[champ_df['was_promoted'] == True].copy()
        logger.info(f"Found {len(promoted):,} promoted player-season records")
        
        # Filter to sufficient minutes (≥ 900 Championship minutes)
        promoted = promoted[promoted['minutes'] >= 900]
        logger.info(f"Filtered to {len(promoted):,} with ≥900 Championship minutes")
        
        matched_records = []
        
        for _, champ_row in promoted.iterrows():
            player_name = champ_row.get('player', '')
            champ_season = champ_row.get('season', '')
            
            # Calculate EPL debut season (year after promotion)
            try:
                year = int(champ_season.split('-')[0])
                epl_debut_season = f"{year+1}-{year+2}"
            except:
                continue
            
            # Find player in EPL debut season
            epl_matches = epl_df[
                (epl_df['name'].str.contains(player_name, case=False, na=False)) &
                (epl_df.get('season', epl_df.get('season_x', '')).str.contains(str(year+1), na=False))
            ]
            
            if len(epl_matches) == 0:
                continue
            
            # Aggregate EPL debut season stats
            epl_stats = epl_matches.groupby('name').agg({
                'total_points': 'sum',
                'minutes': 'sum',
                'goals_scored': 'sum',
                'assists': 'sum',
                'starts': 'sum' if 'starts' in epl_matches.columns else 'count'
            }).iloc[0]
            
            # Filter: ≥ 900 EPL minutes
            if epl_stats['minutes'] < 900:
                continue
            
            # Create matched record
            matched_records.append({
                'player_name': player_name,
                'position': champ_row.get('position', 'MID'),
                'age': champ_row.get('age', 25),
                'promoted_season': champ_season,
                'team': champ_row.get('squad', ''),
                'team_finish_position': champ_row.get('team_finish_position', 0),
                
                # Championship stats
                'champ_minutes': champ_row.get('minutes', 0),
                **{f'champ_{k}': v for k, v in champ_row.items() 
                   if k not in ['player', 'squad', 'season', 'position', 'age']},
                
                # EPL debut stats
                'epl_minutes_total': epl_stats['minutes'],
                'epl_total_points': epl_stats['total_points'],
                'epl_avg_points': epl_stats['total_points'] / (epl_stats['minutes'] / 90) if epl_stats['minutes'] > 0 else 0,
                'epl_goals': epl_stats['goals_scored'],
                'epl_assists': epl_stats['assists'],
                'epl_starts': epl_stats['starts'],
            })
        
        matched_df = pd.DataFrame(matched_records)
        logger.info(f"Matched {len(matched_df)} promoted players with ≥900 EPL debut minutes")
        
        return matched_df
    
    def cluster_archetypes(self, 
                          matched_df: pd.DataFrame,
                          n_clusters: int = 8) -> pd.DataFrame:
        """
        Apply k-means clustering to assign archetype labels.
        
        Args:
            matched_df: Matched player dataframe
            n_clusters: Number of clusters (archetypes)
            
        Returns:
            Dataframe with archetype labels
        """
        logger.info(f"Clustering into {n_clusters} archetypes...")
        
        # Create embeddings
        embeddings = []
        for _, row in matched_df.iterrows():
            emb = self.create_per90_embedding(row)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Standardize
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        matched_df['archetype_cluster'] = kmeans.fit_predict(embeddings_scaled)
        
        # Calculate silhouette score
        sil_score = silhouette_score(embeddings_scaled, matched_df['archetype_cluster'])
        logger.info(f"Silhouette score: {sil_score:.3f}")
        
        # Label clusters based on characteristics
        cluster_labels = self._label_clusters(matched_df, kmeans.cluster_centers_)
        matched_df['archetype'] = matched_df['archetype_cluster'].map(cluster_labels)
        
        # Store embeddings
        matched_df['embedding'] = list(embeddings)
        
        return matched_df
    
    def _label_clusters(self, df: pd.DataFrame, centroids: np.ndarray) -> Dict[int, str]:
        """
        Label clusters based on centroid characteristics.
        
        Args:
            df: Dataframe with clusters
            centroids: Cluster centroids (scaled)
            
        Returns:
            Dictionary mapping cluster_id -> label
        """
        # Unscale centroids
        centroids_original = self.scaler.inverse_transform(centroids)
        
        labels = {}
        
        for i, centroid in enumerate(centroids_original):
            goals, assists, xg, xa, shots, kp, tackles, inter, drib, pass_pct, aerials, prog, sca, gca = centroid
            
            # Classify based on dominant characteristics
            if goals > 0.4 and shots > 2.5:
                label = 'clinical_finisher'
            elif assists > 0.25 and kp > 1.5:
                label = 'creative_playmaker'
            elif goals > 0.2 and assists > 0.15 and tackles > 1.0:
                label = 'box_to_box'
            elif tackles > 2.0 and inter > 1.5:
                label = 'defensive_anchor'
            elif drib > 1.5 and prog > 2.0:
                label = 'ball_carrying_threat'
            elif aerials > 2.0:
                label = 'target_man'
            elif gca > 0.6:
                label = 'set_piece_specialist'
            elif pass_pct > 80 and prog > 1.5:
                label = 'ball_playing_defender'
            else:
                label = f'archetype_{i}'
            
            labels[i] = label
        
        return labels
    
    def build_database(self, save_path: str = 'ml/models/promoted_archetypes_v2.pkl') -> int:
        """
        Full pipeline: load → match → cluster → save.
        
        Args:
            save_path: Path to save archetype database
            
        Returns:
            Number of archetypes created
        """
        logger.info("Building archetype database v2...")
        
        # Load data
        champ_df = self.load_championship_data()
        epl_df = self.load_epl_data()
        
        # Match promoted players to EPL debuts
        matched_df = self.match_promoted_players(champ_df, epl_df)
        
        if len(matched_df) < 20:
            logger.warning(f"Only {len(matched_df)} matched players - database will be small")
        
        # Cluster archetypes
        matched_df = self.cluster_archetypes(matched_df, n_clusters=min(8, len(matched_df) // 10))
        
        # Convert to archetype format
        self.archetypes = matched_df.to_dict('records')
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.archetypes, f)
        
        logger.info(f"✅ Saved {len(self.archetypes)} archetypes to {save_path}")
        
        # Print summary
        logger.info("\nArchetype distribution:")
        for archetype, count in matched_df['archetype'].value_counts().items():
            logger.info(f"  {archetype}: {count}")
        
        return len(self.archetypes)


def main():
    """Run archetype database builder."""
    builder = ArchetypeDatabaseBuilderV2()
    
    try:
        count = builder.build_database()
        logger.info(f"\n✅ Built archetype database with {count} players")
        return 0
    except Exception as e:
        logger.error(f"❌ Failed to build database: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
