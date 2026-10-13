"""
Sanity checks on EFL Championship dataset
Based on advisor feedback
"""

import pandas as pd

print("="*80)
print("EFL CHAMPIONSHIP DATASET SANITY CHECKS")
print("="*80)

df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_all_cleaned.csv')
print(f"\nDataset: {len(df)} rows, {df.shape[1]} columns")

# ============================================================================
# CHECK 1: Mid-Season Transfer Integrity (No Accidental Duplicates)
# ============================================================================
print("\n" + "="*80)
print("CHECK 1: MID-SEASON TRANSFER INTEGRITY")
print("="*80)

print("\nChecking for accidental duplicate rows...")
key_counts = df.groupby(["Player_Key", "Season", "Squad"]).size()
print(f"\nComposite key uniqueness:")
print(key_counts.value_counts().to_dict())

duplicates = key_counts[key_counts > 1]
if len(duplicates) > 0:
    print(f"\n[WARNING] Found {len(duplicates)} duplicate keys:")
    for (player, season, squad), count in duplicates.head(10).items():
        print(f"  - {player} | {squad} | {season}: {count} rows")
else:
    print("\n[OK] Every (Player_Key, Season, Squad) appears exactly once")

# Verify mid-season transfers are handled correctly
print("\n\nMid-season transfers (same player, same season, different squads):")
transfer_cases = df.groupby(["Player_Key", "Season"]).filter(lambda x: len(x) > 1)
if len(transfer_cases) > 0:
    transfer_summary = transfer_cases.groupby(["Player_Key", "Season"])["Squad"].apply(list)
    print(f"\nFound {len(transfer_summary)} cases of mid-season transfers:")
    for (player, season), squads in transfer_summary.head(10).items():
        print(f"  - {player} | {season}: {squads}")
else:
    print("\n[INFO] No mid-season transfers detected")

# ============================================================================
# CHECK 2: Team Name Consistency
# ============================================================================
print("\n" + "="*80)
print("CHECK 2: TEAM NAME CONSISTENCY")
print("="*80)

print("\nAll unique squad names in dataset:")
unique_squads = sorted(df['Squad'].unique())
print(f"Total: {len(unique_squads)} unique squads\n")

# Check for potential inconsistencies
potential_issues = {
    "Nott'm Forest": ["Nottingham", "Nottingham Forest", "Nott'm Forest"],
    "West Brom": ["West Bromwich Albion", "WBA", "West Brom"],
    "QPR": ["Queens Park Rangers", "QPR"],
    "Sheffield Wed": ["Sheffield Wednesday", "Sheff Wed", "Sheffield Weds"],
    "Sheffield Utd": ["Sheffield United", "Sheff Utd", "Sheffield Utd"],
}

inconsistencies_found = []
for canonical, variants in potential_issues.items():
    found_variants = [v for v in variants if v in unique_squads]
    if len(found_variants) > 1:
        inconsistencies_found.append((canonical, found_variants))

if inconsistencies_found:
    print("[WARNING] Potential inconsistencies detected:")
    for canonical, variants in inconsistencies_found:
        print(f"  - {canonical}: found {variants}")
else:
    print("[OK] No obvious naming inconsistencies")

# Show squads that might need mapping for EPL datasets
print("\n\nSquads likely to need EPL mapping:")
squads_to_map = [
    ("Nott'm Forest", "Nottingham Forest"),
    ("Leeds", "Leeds United"),
    ("West Brom", "West Bromwich Albion"),
    ("Sheffield Utd", "Sheffield United"),
    ("Sheffield Wed", "Sheffield Wednesday"),
]

for champ_name, potential_epl_name in squads_to_map:
    if champ_name in unique_squads:
        print(f"  ✓ {champ_name:<20} → EPL likely uses: {potential_epl_name}")

# ============================================================================
# CHECK 3: Promotion Mapping
# ============================================================================
print("\n" + "="*80)
print("CHECK 3: PROMOTION MAPPING")
print("="*80)

# Define known promotions (based on historical data)
PROMOTED_TEAMS = {
    "2019-2020": ["Leeds", "West Brom", "Fulham"],
    "2020-2021": ["Norwich", "Watford", "Brentford"],
    "2021-2022": ["Fulham", "Bournemouth", "Nott'm Forest"],
    "2022-2023": ["Burnley", "Sheffield Utd", "Luton"],
    "2023-2024": ["Leicester", "Ipswich", "Southampton"],
    # 2024-2025 and 2025-2026 not yet decided (ongoing/future)
}

print("\nKnown promoted teams by season:")
for season, teams in PROMOTED_TEAMS.items():
    print(f"\n{season}:")
    for team in teams:
        # Check if team exists in dataset for that season
        exists = ((df['Season'] == season) & (df['Squad'] == team)).any()
        player_count = len(df[(df['Season'] == season) & (df['Squad'] == team)])
        status = "✓" if exists else "✗"
        print(f"  {status} {team:<20} ({player_count} players)")

# Create promotion feature
print("\n\nCreating promotion indicator...")
df['Was_Promoted'] = 0
for season, teams in PROMOTED_TEAMS.items():
    mask = (df['Season'] == season) & (df['Squad'].isin(teams))
    df['Was_Promoted'] = df['Was_Promoted'].where(~mask, 1)

promoted_count = df['Was_Promoted'].sum()
print(f"[OK] Tagged {promoted_count} player records from promoted teams")

# Show distribution
print(f"\nPromoted vs Non-Promoted:")
print(df['Was_Promoted'].value_counts())

# ============================================================================
# CHECK 4: Data Completeness by Season
# ============================================================================
print("\n" + "="*80)
print("CHECK 4: DATA COMPLETENESS BY SEASON")
print("="*80)

print("\nRecords per season:")
print(df['Season'].value_counts().sort_index())

print("\n\nKey metrics completeness by season:")
key_metrics = ['90s', 'Min', 'Gls', 'Ast', 'xG', 'xAG', 'Sh', 'SoT']
for metric in key_metrics:
    if metric in df.columns:
        by_season = df.groupby('Season')[metric].apply(lambda x: x.notna().sum())
        print(f"\n{metric}:")
        for season, count in by_season.items():
            pct = 100 * count / len(df[df['Season'] == season])
            print(f"  {season}: {count:>4} ({pct:>5.1f}%)")

# ============================================================================
# CHECK 5: Player-Level Statistics Summary
# ============================================================================
print("\n" + "="*80)
print("CHECK 5: PLAYER-LEVEL STATISTICS")
print("="*80)

print("\nUnique players across all seasons:")
unique_players = df['Player_Key'].nunique()
print(f"Total: {unique_players} unique players")

print("\n\nPlayers with multiple seasons (trajectory trackable):")
player_season_counts = df.groupby('Player_Key')['Season'].nunique()
multi_season = player_season_counts[player_season_counts > 1]
print(f"Players with 2+ seasons: {len(multi_season)} ({100*len(multi_season)/unique_players:.1f}%)")
print(f"Players with 3+ seasons: {len(player_season_counts[player_season_counts > 2])} ({100*len(player_season_counts[player_season_counts > 2])/unique_players:.1f}%)")

print("\n\nSample multi-season players:")
for player_key, num_seasons in multi_season.head(10).items():
    player_name = df[df['Player_Key'] == player_key]['Player'].iloc[0]
    seasons = sorted(df[df['Player_Key'] == player_key]['Season'].unique())
    print(f"  - {player_name:<25} {num_seasons} seasons: {seasons}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SANITY CHECK SUMMARY")
print("="*80)

checks = [
    ("Composite key uniqueness", len(duplicates) == 0),
    ("Mid-season transfers tracked", len(transfer_cases) > 0),
    ("Team names consistent", len(inconsistencies_found) == 0),
    ("Promotion mapping complete", promoted_count > 0),
    ("Multi-season tracking possible", len(multi_season) > 0),
]

print("\nResults:")
for check_name, passed in checks:
    status = "[PASS]" if passed else "[WARN]"
    print(f"  {status} {check_name}")

print("\n" + "="*80)
print("DATASET READY FOR MODEL E: CHAMPIONSHIP → EPL TRANSLATION")
print("="*80)
