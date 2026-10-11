"""
Extract and analyze feature importances from Model C to diagnose data leakage.
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_analyze_model(position: str):
    """Load Model C for position and extract top features."""
    model_path = Path(f"ml/models/minutes_pstart_{position.lower()}_v1.pkl")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING MODEL C - {position} (P(start))")
    print(f"{'='*80}\n")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Display top 30 features
    print("TOP 30 FEATURES (by importance):")
    print("-" * 80)
    
    for i, row in importance_df.head(30).iterrows():
        feature = row['feature']
        importance = row['importance']
        
        # Categorize feature for leakage detection
        if 'opp_' in feature or 'opponent' in feature:
            category = "⚠️ OPPONENT"
        elif 'team_' in feature and feature != 'team_def_strength':
            category = "⚠️ TEAM"
        elif feature in ['was_home', 'GW', 'gw_in_season']:
            category = "⚠️ MATCH"
        elif 'rolling' in feature or 'form' in feature or 'last' in feature:
            category = "✅ LAGGED"
        elif feature in ['value', 'minutes_rolling5', 'points_per_90']:
            category = "✅ PERSISTENT"
        else:
            category = "❓ OTHER"
        
        print(f"{i+1:2d}. {category:15s} {feature:40s} {importance:8.5f}")
    
    # Summary statistics
    print("\n" + "-" * 80)
    print("LEAKAGE INDICATOR SUMMARY:")
    print("-" * 80)
    
    # Count opponent features in top 30
    top30 = importance_df.head(30)
    opponent_features = top30[top30['feature'].str.contains('opp_|opponent', case=False, regex=True)]
    team_features = top30[top30['feature'].str.contains('team_', case=False) & 
                         ~top30['feature'].str.contains('team_def_strength', case=False)]
    match_features = top30[top30['feature'].isin(['was_home', 'GW', 'gw_in_season'])]
    
    print(f"Opponent features in top 30: {len(opponent_features)} "
          f"(total importance: {opponent_features['importance'].sum():.3f})")
    print(f"Team features in top 30: {len(team_features)} "
          f"(total importance: {team_features['importance'].sum():.3f})")
    print(f"Match context in top 30: {len(match_features)} "
          f"(total importance: {match_features['importance'].sum():.3f})")
    
    leakage_total = (opponent_features['importance'].sum() + 
                     team_features['importance'].sum() + 
                     match_features['importance'].sum())
    
    print(f"\nCombined leakage importance: {leakage_total:.3f}")
    print(f"Percentage of top-30 importance from leaky features: {leakage_total/top30['importance'].sum()*100:.1f}%")
    
    # Diagnosis
    print("\n" + "=" * 80)
    if leakage_total > 0.3:
        print("🚨 DIAGNOSIS: SEVERE DATA LEAKAGE DETECTED")
        print(f"   {leakage_total:.1%} of model decisions based on current match context")
        print("   Model is 'nowcasting' not 'forecasting'")
    elif leakage_total > 0.15:
        print("⚠️ DIAGNOSIS: MODERATE DATA LEAKAGE")
        print(f"   {leakage_total:.1%} of model decisions based on current match context")
    else:
        print("✅ DIAGNOSIS: Minimal leakage")
        print(f"   {leakage_total:.1%} of model decisions based on current match context")
    print("=" * 80)


def main():
    """Run analysis for all positions."""
    print("\n" + "🔍" * 40)
    print("FEATURE IMPORTANCE ANALYSIS - DATA LEAKAGE DIAGNOSIS")
    print("🔍" * 40)
    
    positions = ['GK', 'DEF', 'MID', 'FWD']
    
    for position in positions:
        load_and_analyze_model(position)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNEXT STEPS:")
    print("1. If leakage detected: Refactor to remove opponent/match-specific features")
    print("2. Shift target to next gameweek (GW+1) instead of current GW")
    print("3. Retrain with forward-looking setup")
    print("4. Validate with walk-forward testing")


if __name__ == '__main__':
    main()
