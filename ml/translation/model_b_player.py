"""
Phase 5: Model B - Player Per-90 Translation
Translates Championship player per-90 stats into EPL-equivalent per-90 stats
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def load_archetypes():
    """Load archetype database"""
    print("Loading archetype database...")
    arch_path = Path('C:/ACL2/FPL/ACL_M3/ml/models/promoted_archetypes_v2.pkl')
    with open(arch_path, 'rb') as f:
        archetypes = pickle.load(f)
    
    df = pd.DataFrame(archetypes)
    print(f"Loaded {len(df)} player transitions")
    print(f"  By position: {df['position'].value_counts().to_dict()}")
    return df


def train_player_translation_model(df, position):
    """Train Ridge regression for a specific position"""
    print(f"\n{'='*60}")
    print(f"Training Player Translation Model: {position}")
    print(f"{'='*60}")
    
    df_pos = df[df['position'] == position].copy()
    print(f"  Training samples: {len(df_pos)}")
    
    if len(df_pos) < 5:
        print(f"  [SKIP] Not enough samples for {position}")
        return None
    
    # Features: Championship per-90 stats
    feature_cols = [
        'champ_goals_per90',
        'champ_assists_per90',
        'champ_xg_per90',
        'champ_xa_per90',
        'champ_shots_per90',
        'champ_key_passes_per90',
        'champ_tackles_per90',
        'champ_interceptions_per90',
        'age_at_promotion',
    ]
    
    X = df_pos[feature_cols].values
    
    # Targets: EPL per-90 stats
    target_cols = [
        'epl_goals_per90',
        'epl_assists_per90',
    ]
    
    y = df_pos[target_cols].values
    
    # Split data
    if len(df_pos) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        # Too small for train/test split, use all for training
        X_train, X_test = X, X
        y_train, y_test = y, y
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Ridge regression
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    print("  Performance:")
    for i, target in enumerate(target_cols):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i]) if len(df_pos) >= 20 else 0
        print(f"    {target}: MAE={mae:.3f}, R2={r2:.3f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'position': position,
    }


def main():
    print("="*80)
    print("PHASE 5: PLAYER PER-90 TRANSLATION MODEL (MODEL B)")
    print("="*80)
    
    df = load_archetypes()
    
    # Train position-specific models
    positions = ['DEF', 'MID', 'FWD']  # Skip GK (only 11 samples)
    models = {}
    
    for pos in positions:
        model_data = train_player_translation_model(df, pos)
        if model_data:
            models[pos] = model_data
            
            # Save individual model
            output_path = Path(f'C:/ACL2/FPL/ACL_M3/ml/models/player_translation_{pos}_v1.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"  [OK] Saved {output_path.name}")
    
    print(f"\n[OK] PHASE 5 COMPLETE - Trained {len(models)} position-specific models")


if __name__ == '__main__':
    main()
