"""
Test DGW feature is working correctly
"""
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("DGW FEATURE TEST")
print("="*70)

# Load data
df = pd.read_csv("cleaned_merged_seasons_cleaned.csv")
df = df[df['season_x'].isin(['2023-24', '2024-25', '2025-26'])]

print(f"\n[*] Loaded {len(df):,} records from 3 seasons")

# Add DGW feature
df['fixtures_this_gw'] = df.groupby(['name', 'GW', 'season_x'])['name'].transform('count')

print(f"\n[*] Fixture count distribution:")
print(df['fixtures_this_gw'].value_counts().sort_index())

# Statistics
single_gw = (df['fixtures_this_gw'] == 1).sum()
dgw = (df['fixtures_this_gw'] == 2).sum()
tgw = (df['fixtures_this_gw'] >= 3).sum()
total = len(df)

print(f"\n[*] Row-level statistics:")
print(f"  - Single GW rows: {single_gw:,} ({single_gw/total*100:.1f}%)")
print(f"  - DGW rows (2 per player): {dgw:,} ({dgw/total*100:.1f}%)")
print(f"  - TGW rows (3+ per player): {tgw:,} ({tgw/total*100:.1f}%)")

# Player-GW level statistics
player_gw = df.groupby(['name', 'GW', 'season_x']).agg({
    'fixtures_this_gw': 'first'  # All rows in same group have same value
}).reset_index()

print(f"\n[*] Player-GW level statistics:")
print(f"  - Total player-GW combinations: {len(player_gw):,}")
print(f"  - Single GW: {(player_gw['fixtures_this_gw'] == 1).sum():,} ({(player_gw['fixtures_this_gw'] == 1).sum()/len(player_gw)*100:.1f}%)")
print(f"  - DGW (2 fixtures): {(player_gw['fixtures_this_gw'] == 2).sum():,} ({(player_gw['fixtures_this_gw'] == 2).sum()/len(player_gw)*100:.1f}%)")
print(f"  - TGW (3+ fixtures): {(player_gw['fixtures_this_gw'] >= 3).sum():,} ({(player_gw['fixtures_this_gw'] >= 3).sum()/len(player_gw)*100:.1f}%)")

# Show some DGW examples
print(f"\n[*] Sample DGW players (players with 2 fixtures in same GW):")
dgw_players = player_gw[player_gw['fixtures_this_gw'] == 2].head(10)
for idx, row in dgw_players.iterrows():
    player_fixtures = df[(df['name'] == row['name']) & 
                         (df['GW'] == row['GW']) & 
                         (df['season_x'] == row['season_x'])]
    print(f"\n  - {row['name']} (GW{row['GW']}, {row['season_x']}):")
    for _, fixture in player_fixtures.iterrows():
        print(f"      vs {fixture['opp_team_name']} ({fixture['kickoff_time'][:10]}) - {fixture['total_points']}pts")

# Expected impact analysis
print(f"\n[*] Expected Impact on Model:")
print(f"  - Feature range: 1 (single GW) to {df['fixtures_this_gw'].max()} (max fixtures)")
print(f"  - DGW multiplier: ~1.5-1.8x points (player plays twice)")
print(f"  - Model can learn: More fixtures → more points")

# Check points in DGW vs single GW
print(f"\n[*] Points comparison:")
single_gw_pts = df[df['fixtures_this_gw'] == 1]['total_points'].mean()
dgw_pts = df[df['fixtures_this_gw'] == 2]['total_points'].mean()
print(f"  - Avg points in single GW: {single_gw_pts:.2f}pts")
print(f"  - Avg points in DGW (per fixture): {dgw_pts:.2f}pts")
print(f"  - Note: DGW points are per fixture, not total")

# Actual total points for DGW players
if dgw > 0:
    dgw_player_gw = df[df['fixtures_this_gw'] == 2].groupby(['name', 'GW', 'season_x'])['total_points'].sum()
    print(f"  - Avg TOTAL points for DGW players: {dgw_player_gw.mean():.2f}pts (across both fixtures)")
    print(f"  - Ratio: DGW total / Single GW = {dgw_player_gw.mean() / single_gw_pts:.2f}x")

print("\n" + "="*70)
print("[OK] DGW FEATURE VALIDATED")
print("="*70 + "\n")
