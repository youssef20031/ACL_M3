"""
Feature Engineering for FPL ML Models
Replicates notebook preprocessing with improvements
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for FPL player predictions.
    Ensures consistency between training and inference.
    """
    
    def __init__(self):
        """Initialize feature engineer with default configurations."""
        self.feature_names: List[str] = []
        self.categorical_mappings: Dict[str, List[str]] = {}
        self.numeric_features: List[str] = []
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the feature engineer on training data.
        Stores categorical mappings and feature names.
        
        Args:
            df: Training dataframe
            
        Returns:
            self
        """
        # Store categorical mappings
        if 'position' in df.columns:
            self.categorical_mappings['position'] = sorted(df['position'].dropna().unique())
        
        # Handle team column (could be 'team' or 'team_x')
        team_col = 'team' if 'team' in df.columns else 'team_x' if 'team_x' in df.columns else None
        if team_col:
            self.categorical_mappings['team'] = sorted(df[team_col].dropna().astype(str).unique())
        
        # Handle opponent team column (could be various names)
        opp_col = None
        for col_name in ['opp_team_name', 'opponent_team', 'opp_team']:
            if col_name in df.columns:
                opp_col = col_name
                break
        if opp_col:
            self.categorical_mappings['opponent'] = sorted(df[opp_col].dropna().astype(str).unique())
        
        # Define numeric features (excluding target and leakage-prone features)
        self.numeric_features = [
            'minutes', 'goals_scored', 'assists', 'bps', 
            'ict_index', 'influence', 'creativity', 'threat',
            'clean_sheets', 'bonus', 'goals_conceded', 'saves',
            'yellow_cards', 'red_cards', 'penalties_missed', 'penalties_saved',
            'own_goals', 'value', 'was_home', 'GW',
            'form', 'team_goals'  # Engineered features
        ]
        
        self.is_fitted = True
        logger.info(f"Feature engineer fitted with {len(self.categorical_mappings)} categorical features")
        
        return self
    
    def engineer_features(
        self, 
        df: pd.DataFrame, 
        is_training: bool = True,
        lag_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply feature engineering to dataframe.
        
        IMPROVEMENTS FROM REMARKS:
        1. Remove total_points and bps from features (data leakage)
        2. Lag total_points if needed for supervised learning
        3. Add team_goals feature
        4. Sort by temporal order before splitting
        
        Args:
            df: Input dataframe
            is_training: If True, fit categorical encoders
            lag_features: Features to lag (e.g., ['total_points'] for target)
            
        Returns:
            Engineered dataframe
        """
        df = df.copy()
        
        # Ensure temporal ordering (IMPROVEMENT: temporal split)
        if 'kickoff_time' in df.columns:
            df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
            df = df.sort_values(['kickoff_time', 'name']).reset_index(drop=True)
        
        # IMPROVEMENT: Remove total_points and bps from features (data leakage)
        # We'll lag them or use them as targets only
        
        # Calculate form (4-game rolling average of total_points)
        if 'total_points' in df.columns:
            df = df.sort_values(['name', 'kickoff_time'])
            df['form'] = df.groupby('name')['total_points'].transform(
                lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
            )
            # Fill NaN with 0 for players with < 4 games
            df['form'] = df['form'].fillna(0)
        else:
            df['form'] = 0
        
        # Calculate team_goals (conditional on home/away)
        if 'team_h_score' in df.columns and 'team_a_score' in df.columns:
            df['team_goals'] = df.apply(
                lambda row: row['team_h_score'] if row.get('was_home', False) else row['team_a_score'],
                axis=1
            )
        else:
            df['team_goals'] = 0
        
        # IMPROVEMENT: Lag total_points for supervised learning (target = next gameweek points)
        if lag_features and 'total_points' in lag_features:
            df = df.sort_values(['name', 'kickoff_time'])
            df['upcoming'] = df.groupby('name')['total_points'].shift(-1)
            # Drop rows where upcoming is NaN (last gameweek for each player)
            df = df[df['upcoming'].notna()]
        
        # Handle missing values in numeric features
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Convert was_home to int if it's boolean
        if 'was_home' in df.columns:
            df['was_home'] = df['was_home'].astype(int)
        
        # One-hot encode categorical features
        if is_training or self.is_fitted:
            df = self._encode_categorical(df, is_training)
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        One-hot encode categorical features with consistent column ordering.
        
        Args:
            df: Input dataframe
            is_training: If True, update mappings
            
        Returns:
            Dataframe with one-hot encoded columns
        """
        encoded_dfs = [df]
        
        # Map actual columns to stored mappings
        column_mapping = {}
        if 'position' in df.columns:
            column_mapping['position'] = 'position'
        
        team_col = 'team' if 'team' in df.columns else 'team_x' if 'team_x' in df.columns else None
        if team_col and 'team' in self.categorical_mappings:
            column_mapping[team_col] = 'team'
        
        for opp_col in ['opp_team_name', 'opponent_team', 'opp_team']:
            if opp_col in df.columns and 'opponent' in self.categorical_mappings:
                column_mapping[opp_col] = 'opponent'
                break
        
        for actual_col, mapping_key in column_mapping.items():
            categories = self.categorical_mappings[mapping_key]
            
            # Create one-hot encoding
            if is_training:
                dummies = pd.get_dummies(df[actual_col], prefix=mapping_key, dtype=int)
                # Ensure all expected categories are present
                for cat in categories:
                    col_name = f"{mapping_key}_{cat}"
                    if col_name not in dummies.columns:
                        dummies[col_name] = 0
            else:
                # Use fixed categories from training
                dummies = pd.DataFrame(index=df.index)
                for cat in categories:
                    col_name = f"{mapping_key}_{cat}"
                    dummies[col_name] = (df[actual_col].astype(str) == str(cat)).astype(int)
            
            # Sort columns for consistency
            dummies = dummies[[c for c in sorted(dummies.columns) if c.startswith(f"{mapping_key}_")]]
            encoded_dfs.append(dummies)
        
        result = pd.concat(encoded_dfs, axis=1)
        
        # Drop original categorical columns
        for actual_col in column_mapping.keys():
            if actual_col in result.columns:
                result = result.drop(columns=[actual_col])
        
        return result
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        include_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features for model training/inference.
        
        Args:
            df: Engineered dataframe
            include_target: If True, return target variable
            
        Returns:
            Tuple of (features_df, target_series or None)
        """
        # Columns to exclude (leakage-prone and identifiers)
        exclude_cols = [
            'name', 'season_x', 'season', 'element', 'fixture', 'kickoff_time',
            'selected', 'transfers_in', 'transfers_out', 'transfers_balance',
            'round', 'total_points', 'bps',  # IMPROVEMENT: Removed from features
            'team_a_score', 'team_h_score', 'opponent_team', 'xP',
            'home_team', 'away_team', 'Unnamed: 0', 'index',
            'expected_assists', 'expected_goal_involvements', 'expected_goals',
            'expected_goals_conceded'  # These are also predictive of target
        ]
        
        # Get target if available
        target = None
        if include_target and 'upcoming' in df.columns:
            target = df['upcoming']
            exclude_cols.append('upcoming')
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        features = df[feature_cols].copy()
        
        # Ensure numeric types
        for col in features.columns:
            if features[col].dtype == 'object':
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except:
                    pass
        
        # Fill any remaining NaN
        features = features.fillna(0)
        
        # Store feature names for inference
        if not self.feature_names:
            self.feature_names = list(features.columns)
        else:
            # Ensure columns match training
            for col in self.feature_names:
                if col not in features.columns:
                    features[col] = 0
            features = features[self.feature_names]
        
        return features, target
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after encoding."""
        return self.feature_names.copy()
    
    def save_mappings(self, filepath: str):
        """Save categorical mappings to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'categorical_mappings': self.categorical_mappings,
                'numeric_features': self.numeric_features,
                'feature_names': self.feature_names
            }, f, indent=2)
        logger.info(f"Saved feature mappings to {filepath}")
    
    def load_mappings(self, filepath: str):
        """Load categorical mappings from file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.categorical_mappings = data['categorical_mappings']
            self.numeric_features = data['numeric_features']
            self.feature_names = data['feature_names']
            self.is_fitted = True
        logger.info(f"Loaded feature mappings from {filepath}")
