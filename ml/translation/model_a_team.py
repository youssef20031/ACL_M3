"""
Phase 4: Model A - Team Strength Translation
Translates Championship team stats into EPL-equivalent team strength metrics
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Promoted teams per season
PROMOTED_TEAMS = {
    "2019-2020": ["Leeds", "West Brom", "Fulham"],
    "2020-2021": ["Norwich", "Watford", "Brentford"],
    "2021-2022": ["Fulham", "Bournemouth", "Nott'm Forest"],
    "2022-2023": ["Burnley", "Sheffield Utd", "Luton"],
    "2023-2024": ["Leicester", "Ipswich", "Southampton"],
}


def load_data():
    """Load Championship and EPL team-level data"""
    print("Loading team-level data...")
    
    # Load archetype database (has team info)
    arch_path = Path('C:/ACL2/FPL/ACL_M3/ml/models/promoted_archetypes_v2.pkl')
    with open(arch_path, 'rb') as f:
        archetypes = pickle.load(f)
    
    # Aggregate team-level stats from archetype database
    teams = {}
    for arch in archetypes:
        team_key = (arch['championship_team'], arch['promoted_season'])
        if team_key not in teams:
            teams[team_key] = {
                'champ_team': arch['championship_team'],
                'epl_team': arch['epl_team'],
                'season': arch['promoted_season'],
                'champ_goals_total': 0,
                'champ_assists_total': 0,
                'champ_minutes_total': 0,
                'epl_goals_total': 0,
                'epl_assists_total': 0,
                'epl_minutes_total': 0,
                'epl_points_total': 0,
                'player_count': 0,
            }
        
        teams[team_key]['champ_goals_total'] += arch['champ_goals_per90'] * (arch['champ_minutes_total'] / 90)
        teams[team_key]['champ_assists_total'] += arch['champ_assists_per90'] * (arch['champ_minutes_total'] / 90)
        teams[team_key]['champ_minutes_total'] += arch['champ_minutes_total']
        teams[team_key]['epl_goals_total'] += arch['epl_goals_per90'] * (arch['epl_minutes_total'] / 90)
        teams[team_key]['epl_assists_total'] += arch['epl_assists_per90'] * (arch['epl_minutes_total'] / 90)
        teams[team_key]['epl_minutes_total'] += arch['epl_minutes_total']
        teams[team_key]['epl_points_total'] += arch['epl_total_fpl_points']
        teams[team_key]['player_count'] += 1
    
    df = pd.DataFrame(list(teams.values()))
    
    # Calculate per-game rates
    df['champ_goals_per_match'] = df['champ_goals_total'] / df['player_count']
    df['champ_assists_per_match'] = df['champ_assists_total'] / df['player_count']
    df['epl_goals_per_match'] = df['epl_goals_total'] / df['player_count']
    df['epl_assists_per_match'] = df['epl_assists_total'] / df['player_count']
    df['epl_points_per_player'] = df['epl_points_total'] / df['player_count']
    
    print(f"Loaded {len(df)} promoted teams")
    return df


def train_team_translation_model(df):
    """Train Ridge regression model for team strength translation"""
    print("\nTraining Team Translation Model (Model A)...")
    
    # Features: Championship team aggregates
    feature_cols = [
        'champ_goals_per_match',
        'champ_assists_per_match',
        'champ_minutes_total',
        'player_count',
    ]
    
    X = df[feature_cols].values
    
    # Targets: EPL team performance
    target_cols = [
        'epl_goals_per_match',
        'epl_assists_per_match',
        'epl_points_per_player',
    ]
    
    y = df[target_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Ridge regression with LOOCV
    loo = LeaveOneOut()
    model = Ridge(alpha=1.0)
    
    predictions = np.zeros_like(y)
    
    print(f"\nRunning Leave-One-Out Cross-Validation on {len(X)} teams...")
    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        predictions[test_idx] = model.predict(X_test)
    
    # Calculate metrics
    print("\nLOOCV Performance:")
    for i, target_name in enumerate(target_cols):
        mae = mean_absolute_error(y[:, i], predictions[:, i])
        r2 = r2_score(y[:, i], predictions[:, i])
        print(f"  {target_name}: MAE={mae:.3f}, R2={r2:.3f}")
    
    # Train final model on all data
    model.fit(X_scaled, y)
    
    # Save model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
    }
    
    output_path = Path('C:/ACL2/FPL/ACL_M3/ml/models/team_translation_v1.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n[OK] Model saved to {output_path}")
    return model_data


def main():
    print("="*80)
    print("PHASE 4: TEAM STRENGTH TRANSLATION MODEL (MODEL A)")
    print("="*80)
    
    df = load_data()
    model_data = train_team_translation_model(df)
    
    print("\n[OK] PHASE 4 COMPLETE")


if __name__ == '__main__':
    main()
