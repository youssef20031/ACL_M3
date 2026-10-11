"""
Analyze actual Championship → EPL transition sample size
Critical for determining model architecture (tree-based vs deep learning)
"""

import pandas as pd
import sys
sys.path.append('EFL_Data_All_Seasons')
from promoted_teams_mapping import PROMOTED_TEAMS, add_promotion_features

print("="*80)
print("CHAMPIONSHIP → EPL TRANSITION SAMPLE SIZE ANALYSIS")
print("="*80)

# Load Championship data
champ_df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_all_cleaned.csv')
print(f"\nChampionship dataset: {len(champ_df)} rows")

# Add promotion features
champ_df = add_promotion_features(champ_df)
promoted_only = champ_df[champ_df['Was_Promoted'] == 1]
print(f"Promoted player records: {len(promoted_only)}")

# ============================================================================
# STEP 1: Identify Promoted Players by Season
# ============================================================================
print("\n" + "="*80)
print("STEP 1: PROMOTED PLAYERS BY SEASON")
print("="*80)

promoted_by_season = {}
for season in PROMOTED_TEAMS.keys():
    season_promoted = promoted_only[promoted_only['Season'] == season]
    unique_players = season_promoted['Player_Key'].nunique()
    promoted_by_season[season] = {
        'total_records': len(season_promoted),
        'unique_players': unique_players,
        'teams': PROMOTED_TEAMS[season]['teams']
    }
    print(f"\n{season}:")
    print(f"  Teams: {', '.join(PROMOTED_TEAMS[season]['teams'])}")
    print(f"  Total records: {len(season_promoted)}")
    print(f"  Unique players: {unique_players}")

total_unique_promoted = promoted_only['Player_Key'].nunique()
print(f"\n[TOTAL] Unique players from promoted teams: {total_unique_promoted}")

# ============================================================================
# STEP 2: Estimate EPL Transition Sample (requires EPL data check)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: THEORETICAL TRANSITION SAMPLE")
print("="*80)

print("\nTheoretical maximum transitions (if ALL promoted players stayed in England):")
print(f"  Total promoted player-seasons: {len(promoted_only)}")
print(f"  Unique players: {total_unique_promoted}")

# Estimate realistic transitions based on typical patterns
print("\n\nRealistic transition estimate:")
print("  Typical patterns:")
print("    - ~70-80% of promoted squad stays with promoted team in EPL")
print("    - ~10-15% move to other EPL clubs")
print("    - ~10-15% leave England or retire")
print("    - ~60-70% log 'meaningful' minutes (500+ min in EPL)")

realistic_stayed = int(total_unique_promoted * 0.75)  # 75% stay in England
realistic_meaningful = int(realistic_stayed * 0.65)   # 65% get meaningful minutes

print(f"\n  Estimated stayed in England: {realistic_stayed}")
print(f"  Estimated meaningful EPL minutes: {realistic_meaningful}")

print("\n[WARNING] These are ESTIMATES. Actual count requires EPL dataset join.")

# ============================================================================
# STEP 3: Trajectory Feature Availability
# ============================================================================
print("\n" + "="*80)
print("STEP 3: TRAJECTORY FEATURE AVAILABILITY")
print("="*80)

# Find promoted players with multi-season Championship history
promoted_players = promoted_only['Player_Key'].unique()
multi_season_promoted = []

for player in promoted_players:
    player_history = champ_df[champ_df['Player_Key'] == player]
    if len(player_history) > 1:
        multi_season_promoted.append({
            'player_key': player,
            'player_name': player_history['Player'].iloc[0],
            'num_seasons': len(player_history),
            'seasons': sorted(player_history['Season'].unique()),
            'promoted_season': player_history[player_history['Was_Promoted'] == 1]['Season'].iloc[0]
        })

print(f"\nPromoted players with 2+ Championship seasons:")
print(f"  Total: {len(multi_season_promoted)} out of {total_unique_promoted} ({100*len(multi_season_promoted)/total_unique_promoted:.1f}%)")

print("\n\nSample trajectory-trackable promoted players (first 15):")
for player in multi_season_promoted[:15]:
    print(f"  - {player['player_name']:<25} {player['num_seasons']} seasons: {player['seasons']}")
    print(f"    → Promoted in {player['promoted_season']}")

# Show detailed example
if len(multi_season_promoted) > 0:
    example = multi_season_promoted[0]
    print(f"\n\nDetailed trajectory example: {example['player_name']}")
    player_data = champ_df[champ_df['Player_Key'] == example['player_key']].sort_values('Season')
    
    if 'Gls' in player_data.columns and '90s' in player_data.columns:
        for _, row in player_data.iterrows():
            promoted = " [PROMOTED]" if row['Was_Promoted'] == 1 else ""
            gls_90 = row['Gls'] / row['90s'] if row['90s'] > 0 else 0
            print(f"    {row['Season']}: {row['Gls']:.0f} goals in {row['90s']:.1f} 90s ({gls_90:.2f} per 90){promoted}")

# ============================================================================
# STEP 4: Create Sample Translation Dataset Structure
# ============================================================================
print("\n" + "="*80)
print("STEP 4: TRANSLATION DATASET STRUCTURE (TEMPLATE)")
print("="*80)

print("\nRecommended columns for Championship → EPL translation dataset:")
print("""
Core Identifiers:
  - Player_Key               (normalized player identifier)
  - Player_Name              (original name for reference)
  - Champ_Season             (e.g., "2019-2020")
  - EPL_Season               (next season, e.g., "2020-2021")
  - Age                      (age in promotion season)
  - Position                 (GK/DF/MF/FW)

Championship Stats (Pre-Promotion):
  - Champ_Minutes            (total minutes)
  - Champ_90s                (minutes/90)
  - Champ_Gls_90             (goals per 90)
  - Champ_Ast_90             (assists per 90)
  - Champ_Sh_90              (shots per 90)
  - Champ_SoT_90             (shots on target per 90)
  - Champ_SoT_Pct            (shot accuracy)
  - Champ_Pass_Cmp_Pct       (passing accuracy)
  
Trajectory Features (if 2+ seasons available):
  - Champ_Gls_90_Growth      (year-over-year change)
  - Champ_Ast_90_Growth
  - Champ_Minutes_Growth     (playing time trend)
  - Num_Champ_Seasons        (experience indicator)
  
Context Features:
  - Promotion_Type           (champions/runners_up/playoff_winners)
  - Promotion_Type_Code      (1/2/3)
  - Squad                    (team name)
  - Squad_Finish_Position    (1st/2nd/playoffs)

EPL Outcomes (Target Variables):
  - EPL_Minutes              (total EPL minutes next season)
  - EPL_90s                  
  - EPL_Gls_90               (target: goals per 90)
  - EPL_Ast_90               (target: assists per 90)
  - EPL_Appeared             (binary: played any EPL minutes)
  - EPL_Regular              (binary: 1000+ minutes)
""")

# ============================================================================
# STEP 5: Next Actions Required
# ============================================================================
print("\n" + "="*80)
print("STEP 5: NEXT ACTIONS TO BUILD TRANSLATION DATASET")
print("="*80)

print("""
ACTION 1: Obtain EPL player-season dataset
  - Same structure as Championship data (player × season)
  - Covering seasons: 2020-2021 through 2025-2026
  - Required columns: Player name, Squad, Season, Minutes, Goals, Assists, etc.
  
ACTION 2: Normalize player names for matching
  - Use same Player_Key generation (accent stripping, lowercasing)
  - Handle name variations (nicknames, shortened names)
  - Manual review of promoted player matches (quality check)
  
ACTION 3: Join Championship → EPL
  - Link: Championship 2019-20 promoted players → EPL 2020-21 outcomes
  - Link: Championship 2020-21 promoted players → EPL 2021-22 outcomes
  - ... etc.
  - Filter: Only include players with EPL_Minutes > threshold (e.g., 500 min)
  
ACTION 4: Compute trajectory features
  - For players with 2+ Championship seasons
  - Calculate year-over-year growth rates
  - Add "acceleration" vs "deceleration" flags
  
ACTION 5: Split into train/validation/test
  - Temporal split (NOT random):
    - Train: 2019-2022 promotions
    - Validation: 2022-2023 promotions
    - Test: 2023-2024 promotions
  - This simulates real-world: predict future from past
""")

# ============================================================================
# STEP 6: Model Architecture Recommendation
# ============================================================================
print("\n" + "="*80)
print("STEP 6: MODEL ARCHITECTURE RECOMMENDATION")
print("="*80)

print(f"\nCurrent sample size estimate:")
print(f"  Theoretical max: {total_unique_promoted} players")
print(f"  Realistic transitions: ~{realistic_meaningful} players with meaningful EPL minutes")

print("\n\nRECOMMENDED APPROACH:")
if realistic_meaningful < 150:
    print("""
  Sample size: <150 → Use TREE-BASED MODELS
  
  Recommended: LightGBM or XGBoost
  
  Reasons:
    - Works well with small samples (50-200 examples)
    - Handles missing features gracefully
    - Captures non-linear interactions without overfitting
    - Provides feature importance (interpretability)
    - Fast to train and tune
    
  Architecture:
    - Separate models per position (GK, DF, MF, FW)
    - Ensemble: Average predictions from multiple models
    - Hyperparameter tuning via cross-validation
""")
else:
    print("""
  Sample size: 150-300 → Use TREE-BASED OR SHALLOW NEURAL NETWORK
  
  Option A: LightGBM/XGBoost (safer choice)
  Option B: Neural network with 1-2 hidden layers
  
  If using neural network:
    - Keep it shallow (1-2 hidden layers, 32-64 neurons)
    - Use dropout (0.3-0.5) to prevent overfitting
    - Early stopping on validation set
    - L2 regularization
""")

print("\n\nFEATURE PRIORITIZATION (based on prior research):")
print("""
  High signal:
    ✓ Per-90 rates (Gls/90, Ast/90, Sh/90) — better than raw counts
    ✓ Trajectory (growth rates) — trend > snapshot
    ✓ Age — younger players adapt better
    ✓ Minutes — playing time indicates trust
    ✓ Promotion_Type — context matters
    
  Medium signal:
    ~ Shot accuracy (SoT%)
    ~ Playing time share (Min%)
    ~ Position
    
  Weak signal (Championship ≠ EPL):
    ✗ Tackles, interceptions — EPL teams defend differently
    ✗ Raw goal/assist counts — league quality differs
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Dataset Status:
  ✓ Championship data cleaned: {len(champ_df)} rows, 7 seasons
  ✓ Promoted players identified: {len(promoted_only)} records ({total_unique_promoted} unique)
  ✓ Trajectory-trackable: {len(multi_season_promoted)} players ({100*len(multi_season_promoted)/total_unique_promoted:.1f}%)
  
Next Critical Step:
  → Obtain EPL player-season data (2020-2026)
  → Join to Championship promoted players
  → Confirm actual transition sample size
  
Expected Outcome:
  Realistic sample: ~{realistic_meaningful} Championship → EPL transitions
  → Sufficient for LightGBM/XGBoost
  → Marginal for neural networks
  → Excellent for tree-based ensemble
  
Model Recommendation:
  Primary: LightGBM with position-specific models
  Backup: XGBoost ensemble
  Not recommended: Deep learning (insufficient sample)
  
Key Insight:
  "{len(multi_season_promoted)} players with trajectory data" means you can build
  growth features (xG_growth, minutes_growth) which often predict better
  than single-season snapshots.
""")

print("="*80)
