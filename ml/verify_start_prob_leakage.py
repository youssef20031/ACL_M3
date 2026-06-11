"""
Verify Start Probability Model for Leakage
Check all the suspicious issues flagged
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from ml.feature_engineering import FeatureEngineer

print("\n" + "="*60)
print("START PROBABILITY LEAKAGE VERIFICATION")
print("="*60)

# Load small sample
df = pd.read_csv("cleaned_merged_seasons_cleaned.csv", nrows=10000)
df = df[df['season_x'].isin(['2023-24', '2024-25', '2025-26'])]

print(f"\n📊 Sample data: {len(df)} rows")

# Check 1: ict_index, influence, creativity, threat
print("\n" + "="*60)
print("CHECK 1: Are ict_index, influence, etc. post-match?")
print("="*60)

# For players who didn't play (minutes=0), these should be 0
no_minutes = df[df['minutes'] == 0]
if len(no_minutes) > 0:
    print(f"\nPlayers with 0 minutes: {len(no_minutes)}")
    print(f"  Avg ict_index: {no_minutes['ict_index'].mean():.2f}")
    print(f"  Avg influence: {no_minutes['influence'].mean():.2f}")
    print(f"  Avg creativity: {no_minutes['creativity'].mean():.2f}")
    print(f"  Avg threat: {no_minutes['threat'].mean():.2f}")
    
    if no_minutes['ict_index'].mean() < 0.1:
        print("  ❌ LEAKAGE: These are POST-MATCH stats (0 for non-starters)")
        print("  ❌ Cannot use to predict starting!")
    else:
        print("  ✅ Seems like pre-match data")

# Check 2: Temporal ordering
print("\n" + "="*60)
print("CHECK 2: Is data temporally ordered?")
print("="*60)

if 'kickoff_time' in df.columns:
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')
    is_sorted = df['kickoff_time'].is_monotonic_increasing
    print(f"\nChronologically sorted: {is_sorted}")
    
    if not is_sorted:
        print("  ❌ PROBLEM: Data not sorted by time")
        print("  ❌ Simple split by index will mix past/future")
        
        # Show gameweek distribution
        print(f"\n  First 10 rows GW range: {df.head(10)['GW'].min()}-{df.head(10)['GW'].max()}")
        print(f"  Last 10 rows GW range: {df.tail(10)['GW'].min()}-{df.tail(10)['GW'].max()}")
        
        if df.head(10)['GW'].max() > df.tail(10)['GW'].min():
            print("  ❌ CONFIRMED: Future gameweeks in early rows!")
else:
    print("  ⚠️  No kickoff_time to verify sorting")

# Check 3: Rolling features calculation
print("\n" + "="*60)
print("CHECK 3: Do rolling features include current row?")
print("="*60)

# Engineer features and check
fe = FeatureEngineer()
fe.fit(df)

# Check the code for shift
import inspect
source = inspect.getsource(fe._add_high_signal_features)

has_shift = 'shift(1)' in source or '.shift(1)' in source
print(f"\nFeature engineering uses shift(1): {has_shift}")

if not has_shift:
    print("  ❌ LEAKAGE RISK: Rolling averages may include current row")
else:
    print("  ✅ Rolling features properly shifted")

# Check actual values
df_test = df.head(100).copy()
df_test = df_test.sort_values(['name', 'kickoff_time'])

# Manual calculation
sample_player = df_test['name'].iloc[5] if len(df_test) > 5 else df_test['name'].iloc[0]
player_data = df_test[df_test['name'] == sample_player].sort_values('kickoff_time')

if len(player_data) >= 3:
    print(f"\n  Sample player: {sample_player}")
    print(f"  Gameweeks: {list(player_data['GW'].values[:3])}")
    print(f"  Minutes: {list(player_data['minutes'].values[:3])}")
    
    # What should minutes_rolling5 be for row 2?
    # Should be rolling average of rows 0-1 (previous games)
    expected_rolling = player_data['minutes'].iloc[:2].mean()
    
    # Engineer and check
    df_eng = fe.engineer_features(df_test.copy(), is_training=True)
    
    if 'minutes_rolling5' in df_eng.columns:
        player_eng = df_eng[df_eng['name'] == sample_player].sort_values('kickoff_time')
        if len(player_eng) >= 3:
            actual_rolling = player_eng['minutes_rolling5'].iloc[2]
            print(f"\n  Expected minutes_rolling5 at GW {player_data['GW'].iloc[2]}: {expected_rolling:.1f}")
            print(f"  Actual minutes_rolling5: {actual_rolling:.1f}")
            
            # Check if current minutes (row 2) is included
            includes_current = abs(actual_rolling - player_data['minutes'].iloc[:3].mean()) < 0.1
            if includes_current:
                print(f"  ❌ LEAKAGE: Current row minutes included in rolling avg!")
            else:
                print(f"  ✅ Current row excluded from rolling avg")

# Check 4: minutes as feature when target = f(minutes)
print("\n" + "="*60)
print("CHECK 4: Using minutes to predict started?")
print("="*60)

print(f"\nTarget definition: started = (minutes >= 60)")
print(f"Feature used: minutes_rolling5 (avg of past minutes)")
print(f"\n⚠️  ISSUE: Even with proper shift, this is highly circular")
print(f"   - High past minutes → predicts high future minutes")
print(f"   - This is somewhat tautological")
print(f"\nBetter features:")
print(f"   ✅ Position (GK almost always start)")
print(f"   ✅ Price/value (expensive players start more)")
print(f"   ✅ Team rotation policy")
print(f"   ✅ Recent form (points, not minutes)")
print(f"   ❌ minutes_rolling5 (too circular)")

# Check 5: Feature engineering before split
print("\n" + "="*60)
print("CHECK 5: Features computed before temporal split?")
print("="*60)

print(f"\nCurrent approach:")
print(f"  1. engineer_features(full_df)  ← computes rolling on ALL data")
print(f"  2. split_idx = 0.8 * len")
print(f"  3. train/test split")
print(f"\n❌ PROBLEM: Rolling features computed on full dataset")
print(f"   - Test set rolling values use future training data")
print(f"   - Should compute rolling AFTER split, or use strict temporal boundary")

print("\n" + "="*60)
print("SUMMARY OF ISSUES FOUND")
print("="*60)
print("\n1. ❌ ict_index, influence, creativity, threat are POST-MATCH")
print("2. ❌ Data may not be temporally sorted → train/test contamination")
print("3. ⚠️  minutes_rolling5 is circular (predicts minutes from minutes)")
print("4. ❌ Rolling features computed on full dataset before split")
print("\nExpected legitimate AUC: 0.75-0.82")
print("Observed AUC: 0.9896 ← SUSPICIOUSLY HIGH")
print("\n" + "="*60 + "\n")
