"""
CRITICAL LEAKAGE CHECK: Verify clean_sheets feature

The clean_sheets feature at 56% importance for DEF is suspicious.
We need to verify it's NOT the current gameweek's clean sheet (which would be direct leakage).

This script checks:
1. What does clean_sheets contain? (current GW or historical?)
2. Is it being used in features without lagging?
3. Check correlation between clean_sheets and target
4. Inspect actual values in the dataset
"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

print("="*80)
print("CLEAN SHEETS LEAKAGE VERIFICATION")
print("="*80)

# Load raw data
print("\n1. Loading raw dataset...")
df = pd.read_csv("merged_recent_seasons.csv", low_memory=False)
print(f"Loaded {len(df)} records")

# Focus on defenders
df_def = df[df['position'] == 'DEF'].copy()
print(f"Defenders: {len(df_def)} records")

# Sort by player and time
df_def = df_def.sort_values(['name', 'kickoff_time'])

print("\n" + "="*80)
print("TEST 1: What does clean_sheets contain?")
print("="*80)

# Sample a few players and inspect clean_sheets values
sample_players = df_def['name'].unique()[:3]

for player in sample_players:
    player_data = df_def[df_def['name'] == player][['GW', 'clean_sheets', 'total_points', 'goals_conceded']].head(10)
    print(f"\n{player}:")
    print(player_data.to_string(index=False))

print("\n" + "="*80)
print("TEST 2: Is clean_sheets cumulative or binary?")
print("="*80)

# Check if clean_sheets increases monotonically (cumulative) or is 0/1 (per-GW)
sample = df_def.groupby('name')['clean_sheets'].apply(list).head(5)
print("\nFirst 5 players - clean_sheets values across GWs:")
for name, values in sample.items():
    print(f"{name}: {values[:15]}")  # First 15 GWs
    # Check if monotonically increasing
    is_cumulative = all(values[i] <= values[i+1] for i in range(len(values)-1) if len(values) > 1)
    print(f"  Monotonically increasing (cumulative)? {is_cumulative}")

print("\n" + "="*80)
print("TEST 3: Correlation Analysis")
print("="*80)

# Create lagged versions
df_def = df_def.sort_values(['name', 'kickoff_time'])
df_def['clean_sheets_current'] = df_def['clean_sheets']
df_def['clean_sheets_lagged'] = df_def.groupby('name')['clean_sheets'].shift(1)

# Correlations with total_points
corr_current = df_def['clean_sheets_current'].corr(df_def['total_points'])
corr_lagged = df_def['clean_sheets_lagged'].corr(df_def['total_points'])

print(f"clean_sheets (current) vs total_points correlation: {corr_current:.4f}")
print(f"clean_sheets (lagged) vs total_points correlation: {corr_lagged:.4f}")

if corr_current > 0.5:
    print("❌ WARNING: Very high correlation! Likely data leakage!")
elif corr_current > 0.3:
    print("⚠️  CAUTION: Moderate correlation - investigate further")
else:
    print("✅ OK: Reasonable correlation")

print("\n" + "="*80)
print("TEST 4: Check goals_conceded relationship")
print("="*80)

# In FPL, clean sheet = 0 goals conceded
# If clean_sheets is current GW, it should be perfectly correlated with goals_conceded=0
sample = df_def[['name', 'GW', 'goals_conceded', 'clean_sheets', 'total_points']].head(20)
print("\nSample data:")
print(sample.to_string(index=False))

# Check if clean_sheets == (goals_conceded == 0)
if 'goals_conceded' in df_def.columns:
    df_def['is_clean_sheet_this_gw'] = (df_def['goals_conceded'] == 0).astype(int)
    
    # Compare with clean_sheets column
    match_rate = (df_def['clean_sheets'] == df_def['is_clean_sheet_this_gw']).mean()
    print(f"\nMatch rate between clean_sheets and (goals_conceded==0): {match_rate:.2%}")
    
    if match_rate > 0.9:
        print("❌ CRITICAL: clean_sheets IS the current GW clean sheet!")
        print("   This is DIRECT LEAKAGE - it's an outcome of the match being predicted!")
    elif match_rate < 0.1:
        print("✅ PASS: clean_sheets appears to be cumulative/historical, not current GW")
    else:
        print("⚠️  UNCLEAR: Partial match - needs investigation")

print("\n" + "="*80)
print("TEST 5: Predictive Power Test")
print("="*80)

# If clean_sheets is leakage, a simple model using ONLY clean_sheets should have very high R²
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Remove NaN
df_test = df_def[['clean_sheets', 'total_points']].dropna()

if len(df_test) > 100:
    X = df_test[['clean_sheets']].values
    y = df_test['total_points'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"R² using ONLY clean_sheets: {r2:.4f}")
    
    if r2 > 0.3:
        print("❌ CRITICAL: Single feature R² > 0.3 indicates severe leakage!")
    elif r2 > 0.15:
        print("⚠️  WARNING: Single feature R² > 0.15 is suspiciously high")
    else:
        print("✅ OK: Single feature predictive power is reasonable")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("""
If clean_sheets is CURRENT GW clean sheet (goals_conceded == 0):
  ❌ This is DIRECT LEAKAGE
  ❌ It's an outcome variable, not a predictor
  ❌ Model is essentially predicting: "if they kept clean sheet, points are high"
  
  FIX:
  1. Remove clean_sheets from features entirely, OR
  2. Create clean_sheets_rolling5 = rolling avg of historical clean sheets (lagged)
  
If clean_sheets is CUMULATIVE season total:
  ⚠️  Still problematic - includes current GW's result up to match time
  
  FIX:
  1. Lag by 1: clean_sheets_prev_gw = clean_sheets.shift(1)
  2. Or use rolling average of historical clean sheets
  
If clean_sheets is ALREADY historical/lagged:
  ✅ Good! Model is clean.
  ✅ High importance (56%) makes sense for DEF prediction
""")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
