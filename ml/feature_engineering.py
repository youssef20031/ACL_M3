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
            'form', 'team_goals',  # Engineered features
            # IMPROVEMENT 8: High-signal features
            'minutes_rolling5', 'points_per_90', 'home_form', 'away_form',
            'gw_in_season',
            # IMPROVEMENT 9: Opponent strength (for attackers)
            'opp_def_strength',
            # IMPROVEMENT 11: Defensive features (for GK/DEF clean sheets)
            'opp_off_strength', 'team_def_strength',
            # IMPROVEMENT 12: Double gameweek feature
            'fixtures_this_gw'
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
        
        # IMPROVEMENT 8: Add high-signal features
        df = self._add_high_signal_features(df)
        
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
        
        # IMPROVEMENT 11: Skip opponent one-hot encoding when using continuous opponent strength features
        # This removes collinearity between opp_team_name_Manchester_City and opp_off_strength
        # Only encode opponent if we're NOT using continuous strength features
        skip_opponent_encoding = False
        for opp_col in ['opp_team_name', 'opponent_team', 'opp_team']:
            if opp_col in df.columns:
                # Check if opponent strength features exist
                if 'opp_off_strength' in df.columns or 'opp_def_strength' in df.columns:
                    skip_opponent_encoding = True
                    logger.info(f"Skipping opponent one-hot encoding (using continuous strength features instead)")
                elif 'opponent' in self.categorical_mappings:
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
            'team_a_score', 'team_h_score', 'opponent_team', 'xP',  # IMPROVEMENT 7: xP is lookahead
            'home_team', 'away_team', 'Unnamed: 0', 'index',
            'expected_assists', 'expected_goal_involvements', 'expected_goals',
            'expected_goals_conceded',  # These are also predictive of target
            # CRITICAL: Remove current GW outcome variables (LEAKAGE!)
            'clean_sheets',  # Current GW clean sheet = direct leakage
            'starts',  # Whether started this game = leakage
            'goals_scored',  # Current GW goals = leakage
            'assists',  # Current GW assists = leakage
            'bonus',  # Current GW bonus = leakage
            'goals_conceded',  # Current GW goals conceded = leakage
            'saves',  # Current GW saves = leakage
            'penalties_saved',  # Current GW penalties saved = leakage
            'penalties_missed',  # Current GW penalties missed = leakage
            'yellow_cards',  # Current GW yellows = leakage
            'red_cards',  # Current GW reds = leakage
            'own_goals',  # Current GW own goals = leakage
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
    
    def _add_high_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IMPROVEMENT 8: Add high-signal features for better predictions.
        
        Features added:
        1. minutes_rolling5 - Rolling 5-GW avg of minutes (rotation risk)
        2. points_per_90 - Points normalized per 90 minutes
        3. home_form - Rolling avg points at home
        4. away_form - Rolling avg points away
        5. gw_in_season - Normalized gameweek (fixture congestion)
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with new features
        """
        sort_cols = ['name', 'kickoff_time'] if 'kickoff_time' in df.columns else ['name']
        df = df.sort_values(sort_cols).copy()
        
        # 1. Minutes rolling average (captures rotation risk)
        df['minutes_rolling5'] = df.groupby('name')['minutes'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        ).fillna(0)
        
        # 2. Points per 90 minutes (normalizes for subs/benchings)
        df['points_per_90'] = df.apply(
            lambda row: (row['total_points'] / row['minutes'] * 90) if row['minutes'] > 0 else 0,
            axis=1
        )
        # Shift to avoid leakage
        df['points_per_90'] = df.groupby('name')['points_per_90'].shift(1).fillna(0)
        
        # 3. Home/Away form (position-dependent home advantage)
        # Calculate home form
        home_mask = df['was_home'] == True
        df.loc[home_mask, 'home_form_temp'] = df.loc[home_mask, 'total_points']
        df['home_form'] = df.groupby('name')['home_form_temp'].transform(
            lambda x: x.rolling(4, min_periods=1).mean().shift(1)
        )
        df['home_form'] = df.groupby('name')['home_form'].ffill().fillna(0)
        
        # Calculate away form
        away_mask = df['was_home'] == False
        df.loc[away_mask, 'away_form_temp'] = df.loc[away_mask, 'total_points']
        df['away_form'] = df.groupby('name')['away_form_temp'].transform(
            lambda x: x.rolling(4, min_periods=1).mean().shift(1)
        )
        df['away_form'] = df.groupby('name')['away_form'].ffill().fillna(0)
        
        # Clean up temp columns
        df = df.drop(columns=['home_form_temp', 'away_form_temp'], errors='ignore')
        
        # 4. GW in season (normalized 0-1, captures fixture congestion)
        if 'GW' in df.columns:
            df['gw_in_season'] = df['GW'] / 38.0
        else:
            df['gw_in_season'] = 0
        
        # 5. Opponent defensive strength (IMPROVEMENT 9) - helps attackers
        df = self._add_opponent_defensive_strength(df)
        
        # 6 & 7. Defensive features (IMPROVEMENT 11) - helps GK/DEF with clean sheets
        df = self._add_defensive_features(df)
        
        # 8. Double Gameweek (DGW) feature (IMPROVEMENT 12) - captures biggest FPL edge
        df = self._add_dgw_feature(df)
        
        logger.info("Added 9 high-signal features: minutes_rolling5, points_per_90, home_form, away_form, gw_in_season, opp_def_strength, opp_off_strength, team_def_strength, fixtures_this_gw")
        
        return df
    
    def _add_opponent_defensive_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IMPROVEMENT 9: Add opponent defensive strength feature.
        
        Compresses 20 sparse opponent one-hot columns into 1 dense signal:
        rolling 5-game average of goals conceded by opponent.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with opp_def_strength feature
        """
        # Identify team column name
        team_col = 'team' if 'team' in df.columns else 'team_x'
        opp_col = None
        for col in ['opp_team_name', 'opponent_team', 'opp_team']:
            if col in df.columns:
                opp_col = col
                break
        
        if team_col not in df.columns or opp_col not in df.columns or 'goals_conceded' not in df.columns:
            logger.warning("Cannot compute opponent defensive strength - missing required columns")
            df['opp_def_strength'] = 0
            return df
        
        # Calculate goals conceded by each team per match
        # Group by team and calculate their defensive record
        df_sorted = df.sort_values(['season_x', team_col, 'kickoff_time']).copy() if 'season_x' in df.columns else df.sort_values([team_col, 'kickoff_time']).copy()
        
        # Ensure consistent data types - convert both to string to avoid merge issues
        df_sorted[team_col] = df_sorted[team_col].astype(str)
        df_sorted[opp_col] = df_sorted[opp_col].astype(str)
        
        # Rolling average of goals conceded per team
        group_cols = ['season_x', team_col] if 'season_x' in df.columns else [team_col]
        df_sorted['team_def_strength'] = df_sorted.groupby(group_cols)['goals_conceded'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        
        # Create a mapping of team -> defensive strength per gameweek
        merge_cols = ['season_x', team_col, 'GW'] if 'season_x' in df.columns else [team_col, 'GW']
        team_defense = df_sorted[merge_cols + ['team_def_strength']].drop_duplicates()
        
        # Ensure df also has consistent types
        df[opp_col] = df[opp_col].astype(str)
        
        # Merge as opponent's defensive strength
        df = df.merge(
            team_defense.rename(columns={team_col: opp_col, 'team_def_strength': 'opp_def_strength'}),
            on=[col for col in merge_cols if col != team_col] + [opp_col],
            how='left'
        )
        
        df['opp_def_strength'] = df['opp_def_strength'].fillna(1.0)  # Default to league average ~1 goal/game
        
        logger.info("Added opponent defensive strength (rolling 5-game goals conceded)")
        
        return df
    
    def _add_defensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IMPROVEMENT 11: Add defensive features for GK/DEF clean sheet prediction.
        
        Adds TWO critical features:
        1. opp_off_strength: How many goals opponent SCORES (their attack)
        2. team_def_strength: How many goals own team CONCEDES (their defense)
        
        Both are crucial for predicting clean sheets:
        - Strong opponent attack → less likely to keep clean sheet
        - Strong own defense → more likely to keep clean sheet
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with opp_off_strength and team_def_strength features
        """
        # Identify column names
        team_col = 'team' if 'team' in df.columns else 'team_x'
        opp_col = None
        for col in ['opp_team_name', 'opponent_team', 'opp_team']:
            if col in df.columns:
                opp_col = col
                break
        
        if team_col not in df.columns or opp_col not in df.columns:
            logger.warning("Cannot compute defensive features - missing team/opponent columns")
            df['opp_off_strength'] = 1.5  # League average goals scored
            df['team_def_strength'] = 1.0  # League average goals conceded
            return df
        
        # Ensure consistent data types
        df[team_col] = df[team_col].astype(str)
        df[opp_col] = df[opp_col].astype(str)
        
        # Calculate ACTUAL GOALS (not just stats)
        # For a player: goals scored AGAINST them = opponent's offensive output
        if 'team_h_score' in df.columns and 'team_a_score' in df.columns:
            # Goals scored BY opponent (what we face)
            df['opp_goals_scored'] = df.apply(
                lambda row: row['team_a_score'] if row.get('was_home', False) else row['team_h_score'],
                axis=1
            )
            
            # Goals conceded BY own team (our defensive weakness)
            df['own_goals_conceded'] = df['opp_goals_scored'].copy()
        else:
            logger.warning("Missing team_h_score/team_a_score - using fallback")
            df['opp_goals_scored'] = df.get('goals_conceded', 0)
            df['own_goals_conceded'] = df.get('goals_conceded', 0)
        
        # Sort for temporal rolling averages
        group_cols = ['season_x'] if 'season_x' in df.columns else []
        df = df.sort_values(group_cols + [team_col, 'kickoff_time']).copy()
        
        # === FEATURE 1: Opponent Offensive Strength (goals they score per game) ===
        # Group by opponent team and calculate their rolling goals scored
        opp_group = group_cols + [opp_col]
        
        # Create temp df with opponent goals
        opp_df = df[opp_group + ['GW', 'opp_goals_scored', team_col]].copy()
        
        # For each opponent, calculate rolling average of goals they score
        opp_df = opp_df.sort_values(opp_group + ['GW'])
        opp_df['opp_off_strength_temp'] = opp_df.groupby(opp_col)['opp_goals_scored'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )
        
        # Merge back (match opponent team to get their offensive strength)
        merge_cols = opp_group + ['GW']
        opp_strength_map = opp_df[merge_cols + ['opp_off_strength_temp']].drop_duplicates()
        df = df.merge(
            opp_strength_map,
            on=merge_cols,
            how='left',
            suffixes=('', '_drop')
        )
        df['opp_off_strength'] = df['opp_off_strength_temp'].fillna(1.5)  # League avg
        df = df.drop(columns=['opp_off_strength_temp'], errors='ignore')
        
        # === FEATURE 2: Team Defensive Strength (goals own team concedes per game) ===
        # Group by own team and calculate rolling goals conceded
        team_group = group_cols + [team_col]
        
        df = df.sort_values(team_group + ['GW'])
        df['team_def_strength'] = df.groupby(team_col)['own_goals_conceded'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean()
        )
        df['team_def_strength'] = df['team_def_strength'].fillna(1.0)  # League avg
        
        # Clean up temporary columns
        df = df.drop(columns=['opp_goals_scored', 'own_goals_conceded'], errors='ignore')
        
        logger.info("Added defensive features (opp_off_strength & team_def_strength) - critical for GK/DEF clean sheet prediction")
        
        return df


    
    def _add_dgw_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IMPROVEMENT 12: Add Double Gameweek (DGW) feature.
        
        In FPL, some gameweeks have teams playing multiple fixtures (double/triple gameweeks).
        This is the BIGGEST edge in FPL as players can score points in multiple matches.
        
        Feature: fixtures_this_gw = number of fixtures player has in this gameweek
        - Normal gameweek: 1 fixture
        - Double gameweek: 2 fixtures (points × 1.5-1.8 typical)
        - Triple gameweek: 3 fixtures (rare, points × 2.0-2.5)
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with fixtures_this_gw feature
        """
        # Check if we have the required columns
        required_cols = ['name', 'GW']
        if 'season_x' in df.columns:
            required_cols.append('season_x')
        
        if not all(col in df.columns for col in ['name', 'GW']):
            logger.warning("Cannot compute DGW feature - missing name or GW column")
            df['fixtures_this_gw'] = 1
            return df
        
        # Count fixtures per player per gameweek
        # Group by player + GW (+ season if available) and count rows
        group_cols = ['name', 'GW']
        if 'season_x' in df.columns:
            group_cols.append('season_x')
        
        df['fixtures_this_gw'] = df.groupby(group_cols)['name'].transform('count')
        
        # Log statistics - calculate BEFORE duplication per fixture
        fixture_counts = df.groupby(group_cols).size()
        single_gw_count = (fixture_counts == 1).sum()
        dgw_count = (fixture_counts == 2).sum()
        tgw_count = (fixture_counts >= 3).sum()
        total_player_gw_combinations = len(fixture_counts)
        
        dgw_pct = dgw_count / total_player_gw_combinations * 100 if total_player_gw_combinations > 0 else 0
        tgw_pct = tgw_count / total_player_gw_combinations * 100 if total_player_gw_combinations > 0 else 0
        
        max_fixtures = df['fixtures_this_gw'].max()
        
        logger.info(f"Added DGW feature: {single_gw_count:,} single GWs ({100-dgw_pct-tgw_pct:.1f}%), {dgw_count:,} DGWs ({dgw_pct:.1f}%), {tgw_count:,} TGWs ({tgw_pct:.1f}%) - max {max_fixtures} fixtures/GW")
        
        return df
