"""
Debug script to investigate why player matching is so low
"""
import pandas as pd
import numpy as np

# Load EFL data
efl_df = pd.read_csv('C:/ACL2/FPL/ACL_M3/EFL_Data_All_Seasons/efl_combined_all_cleaned.csv')

# Convert numeric columns
numeric_cols = ['Min', 'Gls', 'Ast', 'Age', 'MP', 'Starts']
for col in numeric_cols:
    if col in efl_df.columns:
        efl_df[col] = pd.to_numeric(efl_df[col], errors='coerce')

# Load EPL data
epl_df = pd.read_csv('C:/ACL2/FPL/ACL_M3/cleaned_merged_seasons_cleaned.csv')

# Test case: Brentford 2020-21 Championship -> 2021-22 EPL
print("="*80)
print("DEBUG: Brentford 2020-21 (Championship) -> 2021-22 (EPL)")
print("="*80)

# Get Brentford Championship players
brentford_champ = efl_df[(efl_df['Season'] == '2020-2021') & (efl_df['Squad'] == 'Brentford')].copy()
print(f"\nChampionship: Found {len(brentford_champ)} Brentford players in 2020-2021")

# Filter by minutes
brentford_champ_450 = brentford_champ[brentford_champ['Min'] >= 450].copy()
print(f"  With >=450 minutes: {len(brentford_champ_450)} players")

print("\nChampionship players with >=450 minutes:")
for idx, row in brentford_champ_450[['Player', 'Pos', 'Min', 'MP']].iterrows():
    print(f"  {row['Player']:30s} {row['Pos']:10s} {row['Min']:6.0f} min ({row['MP']} matches)")

# Get Brentford EPL players
brentford_epl = epl_df[(epl_df['season_x'] == '2021-22') & (epl_df['team_x'] == 'Brentford')].copy()
print(f"\nEPL: Found {len(brentford_epl)} Brentford player-GW records in 2021-22")

# Aggregate by player
brentford_epl_agg = brentford_epl.groupby('name').agg({
    'minutes': 'sum',
    'goals_scored': 'sum',
    'assists': 'sum',
    'GW': 'count'
}).reset_index()

print(f"  Unique players: {len(brentford_epl_agg)}")

brentford_epl_450 = brentford_epl_agg[brentford_epl_agg['minutes'] >= 450].copy()
print(f"  With >=450 minutes: {len(brentford_epl_450)} players")

print("\nEPL players with >=450 minutes:")
for idx, row in brentford_epl_450[['name', 'minutes', 'goals_scored', 'assists']].iterrows():
    print(f"  {row['name']:30s} {row['minutes']:6.0f} min  G:{row['goals_scored']:2.0f} A:{row['assists']:2.0f}")

# Now try to match
print("\n" + "="*80)
print("MATCHING ANALYSIS")
print("="*80)

champ_names = set(brentford_champ_450['Player'].str.lower().str.strip())
epl_names = set(brentford_epl_450['name'].str.lower().str.strip())

print(f"\nChampionship players (normalized): {len(champ_names)}")
print(sorted(champ_names))

print(f"\nEPL players (normalized): {len(epl_names)}")
print(sorted(epl_names))

# Exact matches
exact_matches = champ_names & epl_names
print(f"\n✓ EXACT MATCHES: {len(exact_matches)}")
if exact_matches:
    print(sorted(exact_matches))

# Last name matching
print("\n" + "="*80)
print("LAST NAME MATCHING")
print("="*80)

champ_last_names = {}
for name in brentford_champ_450['Player']:
    last_name = name.split()[-1].lower().strip() if ' ' in name else name.lower().strip()
    if last_name not in champ_last_names:
        champ_last_names[last_name] = []
    champ_last_names[last_name].append(name)

epl_last_names = {}
for name in brentford_epl_450['name']:
    last_name = name.split()[-1].lower().strip() if ' ' in str(name) else str(name).lower().strip()
    if last_name not in epl_last_names:
        epl_last_names[last_name] = []
    epl_last_names[last_name].append(name)

print("\nMatching by last name:")
matches_count = 0
for last_name, champ_players in champ_last_names.items():
    if last_name in epl_last_names:
        matches_count += 1
        print(f"\n  Last name '{last_name}':")
        print(f"    Championship: {champ_players}")
        print(f"    EPL:          {epl_last_names[last_name]}")

print(f"\n✓ LAST NAME MATCHES: {matches_count}")

# Players only in Championship
print("\n" + "="*80)
print("PLAYERS ONLY IN CHAMPIONSHIP (>=450 min)")
print("="*80)
for name in sorted(champ_names - epl_names):
    print(f"  {name}")

# Players only in EPL
print("\n" + "="*80)
print("PLAYERS ONLY IN EPL (>=450 min)")
print("="*80)
for name in sorted(epl_names - champ_names):
    print(f"  {name}")

# Check a specific player: Ivan Toney
print("\n" + "="*80)
print("SPECIFIC CHECK: Ivan Toney")
print("="*80)

toney_champ = brentford_champ[brentford_champ['Player'].str.contains('Toney', case=False, na=False)]
print(f"\nChampionship: Found {len(toney_champ)} matches")
for idx, row in toney_champ.iterrows():
    print(f"  Name: '{row['Player']}'  Minutes: {row['Min']}")

toney_epl = brentford_epl_450[brentford_epl_450['name'].str.contains('Toney', case=False, na=False)]
print(f"\nEPL: Found {len(toney_epl)} matches")
for idx, row in toney_epl.iterrows():
    print(f"  Name: '{row['name']}'  Minutes: {row['minutes']}")

print("\n" + "="*80)
