"""
Phase 3: Archetype Database Expansion using EFL data
Builds comprehensive promoted player database with Championship -> EPL translation pairs
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Import promoted teams mapping
import sys
sys.path.append('C:/ACL2/FPL/ACL_M3/EFL_Data_All_Seasons')
from promoted_teams_mapping import PROMOTED_TEAMS

# Known promoted teams per season (for EPL lookup)
PROMOTED_TEAMS_HISTORY = {
    '2019-20': ['Leeds United', 'West Bromwich Albion', 'Fulham'],
    '2020-21': ['Norwich City', 'Watford', 'Brentford'],
    '2021-22': ['Fulham', 'Bournemouth', 'Nottingham Forest'],
    '2022-23': ['Burnley', 'Sheffield United', 'Luton Town'],
    '2023-24': ['Leicester City', 'Ipswich Town', 'Southampton'],
}

# Archetype labels based on playing style
ARCHETYPE_LABELS = {
    'clinical_finisher': 'High goals/90, high xG/90, high shots/90',
    'creative_playmaker': 'High assists/90, high key passes/90, high xA/90',
    'box_to_box': 'Balanced goals + assists + tackles',
    'defensive_anchor': 'High tackles/90, high interceptions/90, low goals/90',
    'ball_carrying_threat': 'High dribbles/90, high progressive carries/90',
    'target_man': 'High aerials won/90, moderate goals/90',
    'set_piece_specialist': 'High GCA/90 with moderate other stats',
    'ball_playing_defender': 'High pass completion, high progressive passes/90',
    'shot_stopper': 'High saves/90, high PSxG+/- (GK only)',
    'sweeper_keeper': 'High pass completion, moderate saves/90 (GK only)',
}


def load_squad_name_mapping():
    """Load EFL squad name mapping"""
    mapping_path = Path('C:/ACL2/FPL/ACL_M3/EFL_Data_All_Seasons/squad_name_mapping.csv')
    df = pd.read_csv(mapping_path)
    # Create dict: Normalized_Name -> list of possible EFL names
    mapping = {}
    for _, row in df.iterrows():
        normalized = row['Normalized_Name']
        efl_name = row['EFL_Name']
        if normalized not in mapping:
            mapping[normalized] = []
        mapping[normalized].append(efl_name)
    return mapping


def load_efl_data():
    """Load EFL Championship data"""
    print("Loading EFL Championship data...")
    efl_path = Path('C:/ACL2/FPL/ACL_M3/EFL_Data_All_Seasons/efl_combined_all_cleaned.csv')
    df = pd.read_csv(efl_path)
    
    # Convert numeric columns - handle comma-separated thousands
    numeric_cols = ['Min', 'Gls', 'Ast', 'Age', 'MP', 'Starts']
    for col in numeric_cols:
        if col in df.columns:
            # Remove commas first, then convert
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Loaded {len(df)} player records")
    print(f"Seasons: {sorted(df['Season'].unique())}")
    print(f"Sample squads: {df['Squad'].unique()[:10].tolist()}")
    print(f"Min column - Non-null: {df['Min'].notna().sum()} / {len(df)}")
    return df


def load_epl_data():
    """Load EPL data for debut seasons"""
    print("\nLoading EPL data for promoted teams' debut seasons...")
    
    all_dfs = []
    
    # Load individual season files (2023-24 onwards - have team column)
    epl_files = [
        ('data/FPL_2023_2024.csv', 'team'),
        ('data/FPL_2024_2025.csv', 'team'),
        ('data/FPL_2025_2026.csv', 'team'),
    ]
    
    for fname, team_col in epl_files:
        fpath = Path('C:/ACL2/FPL/ACL_M3') / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            # Extract season year from filename
            season_start = fname.replace('FPL_', '').replace('.csv', '').split('_')[0]
            df['season_year'] = int(season_start)
            df['team'] = df[team_col]  # Standardize team column name
            all_dfs.append(df)
            print(f"  Loaded {fname}: {len(df)} records")
    
    # Load cleaned_merged_seasons_cleaned for historical data (2020-21, 2021-22, 2022-23)
    merged_path = Path('C:/ACL2/FPL/ACL_M3/data/cleaned_merged_seasons_cleaned.csv')
    if merged_path.exists():
        df_merged = pd.read_csv(merged_path)
        
        # Map season format: '2020-21' -> 2020
        def extract_season_year(season_str):
            if isinstance(season_str, str) and '-' in season_str:
                return int('20' + season_str.split('-')[0] if len(season_str.split('-')[0]) == 2 else season_str.split('-')[0])
            return None
        
        df_merged['season_year'] = df_merged['season_x'].apply(extract_season_year)
        df_merged['team'] = df_merged['team_x']  # Standardize team column name
        
        all_dfs.append(df_merged)
        print(f"  Loaded data/cleaned_merged_seasons_cleaned.csv: {len(df_merged)} records")
        print(f"    Seasons: {sorted(df_merged['season_x'].unique())}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n[OK] Total EPL records loaded: {len(combined)}")
        print(f"   Seasons covered: {sorted(combined['season_year'].unique())}")
        return combined
    
    return None


def standardize_team_names(team_name):
    """Standardize team names between EFL and EPL datasets"""
    mappings = {
        # EFL/Normalized name -> EPL FPL name (exact as it appears in FPL data)
        'Leeds': 'Leeds',
        'Leeds United': 'Leeds',
        'West Brom': 'West Brom',
        'West Bromwich Albion': 'West Brom',
        'Norwich': 'Norwich',
        'Norwich City': 'Norwich',
        "Nott'm Forest": "Nott'm Forest",
        'Nottingham': "Nott'm Forest",
        'Nottingham Forest': "Nott'm Forest",
        'Sheffield Utd': 'Sheffield Utd',
        'Sheffield United': 'Sheffield Utd',
        'Luton': 'Luton',
        'Luton Town': 'Luton',
        'Leicester': 'Leicester',
        'Leicester City': 'Leicester',
        'Ipswich': 'Ipswich',
        'Ipswich Town': 'Ipswich',
        'Fulham': 'Fulham',
        'Brentford': 'Brentford',
        'Watford': 'Watford',
        'Bournemouth': 'Bournemouth',
        'Southampton': 'Southampton',
        'Burnley': 'Burnley',
    }
    
    # Direct match
    if team_name in mappings:
        return mappings[team_name]
    
    # Return as-is if not in mappings
    return team_name


def identify_promoted_players(efl_df, epl_df):
    """
    Match EFL players to their EPL debut seasons
    Returns DataFrame with both Championship and EPL stats
    """
    print("\nIdentifying promoted players with Championship -> EPL transitions...")
    
    archetypes = []
    
    # Iterate through promoted teams using the EFL Season format
    for season_key, season_data in PROMOTED_TEAMS.items():
        teams = season_data['teams']
        
        # Convert season key (e.g., "2019-2020") to match format
        season_years = season_key.split('-')
        season_str = f"{season_years[0][-2:]}-{season_years[1][-2:]}"  # e.g., "19-20"
        
        print(f"\n{season_str} ({season_key}): {', '.join(teams)}")
        
        for efl_team in teams:
            # Find players from this team in EFL data for this season
            efl_season_mask = efl_df['Season'] == season_key
            efl_team_mask = efl_df['Squad'] == efl_team
            efl_players = efl_df[efl_season_mask & efl_team_mask].copy()
            
            if len(efl_players) == 0:
                print(f"  No EFL data found for {efl_team} in {season_key}")
                continue
            
            # Standardize team name for EPL lookup
            epl_team = standardize_team_names(efl_team)
            
            # Get next season year for EPL debut
            next_season_start = int(season_years[1])
            
            print(f"  {efl_team} -> {epl_team} (EPL debut: {next_season_start}-{str(next_season_start + 1)[-2:]})")
            print(f"  Found {len(efl_players)} EFL players in Championship season")
            
            # Find EPL data for this team in debut season
            if epl_df is not None:
                # Match using both season_year (for individual files) and season_x (for merged file)
                # EPL debut season format: "2020-21" for 2020
                epl_season_str = f"{next_season_start}-{str(next_season_start + 1)[-2:]}"
                
                # Try matching with season_year first (for 2023+ data)
                epl_team_mask = epl_df['team'].str.contains(epl_team, case=False, na=False, regex=False)
                epl_season_mask = (epl_df['season_year'] == next_season_start)
                
                # Also try matching with season_x for the merged file
                if 'season_x' in epl_df.columns:
                    epl_season_mask = epl_season_mask | (epl_df['season_x'] == epl_season_str)
                
                epl_team_data = epl_df[epl_team_mask & epl_season_mask].copy()
                
                print(f"  Found {len(epl_team_data)} EPL player-GW records in debut season")
                
                # Aggregate EPL data by player (sum across gameweeks)
                if len(epl_team_data) > 0:
                    epl_players = epl_team_data.groupby('name').agg({
                        'minutes': 'sum',
                        'goals_scored': 'sum',
                        'assists': 'sum',
                        'total_points': 'sum',
                        'GW': 'count',  # Number of gameweeks played
                    }).reset_index()
                    
                    # Add starts count (approximate: minutes >= 60)
                    epl_team_data['started'] = epl_team_data['minutes'] >= 60
                    starts_count = epl_team_data.groupby('name')['started'].sum().reset_index()
                    epl_players = epl_players.merge(starts_count, on='name', how='left')
                    epl_players = epl_players.rename(columns={'started': 'starts'})
                    
                    print(f"  Aggregated to {len(epl_players)} unique EPL players")
                
                    # Match players by name
                    matched_count = 0
                    for _, efl_player in efl_players.iterrows():
                        # Filter: minimum 450 minutes in Championship (relaxed threshold)
                        champ_minutes = efl_player.get('Min', 0)
                        if pd.isna(champ_minutes) or champ_minutes < 450:
                            continue
                        
                        player_name = efl_player['Player']
                        
                        # Try multiple matching strategies
                        epl_match = None
                        
                        # Strategy 1: Match on last name
                        player_last_name = player_name.split()[-1] if ' ' in player_name else player_name
                        if len(player_last_name) > 3:  # Avoid matching very short names
                            epl_match_candidates = epl_players[epl_players['name'].str.contains(player_last_name, case=False, na=False, regex=False)]
                            if len(epl_match_candidates) > 0:
                                epl_match = epl_match_candidates.iloc[0]
                        
                        # Strategy 2: Match on first + last name
                        if epl_match is None and ' ' in player_name:
                            name_parts = player_name.split()
                            if len(name_parts) >= 2:
                                first_name = name_parts[0]
                                # Try "FirstName LastName" pattern
                                full_name_pattern = f"{first_name}.*{player_last_name}"
                                epl_match_candidates = epl_players[epl_players['name'].str.contains(full_name_pattern, case=False, na=False, regex=True)]
                                if len(epl_match_candidates) > 0:
                                    epl_match = epl_match_candidates.iloc[0]
                        
                        if epl_match is not None:
                            # Filter: minimum 450 EPL debut minutes (relaxed threshold)
                            epl_minutes = epl_match.get('minutes', 0)
                            if pd.isna(epl_minutes) or epl_minutes < 450:
                                continue
                            
                            # Build archetype entry
                            archetype = build_archetype_entry(
                                efl_player, epl_match, efl_team, epl_team, 
                                season_str, season_data.get('promotion_type', 'unknown')
                            )
                            if archetype:
                                archetypes.append(archetype)
                                matched_count += 1
                    
                    print(f"  [OK] Matched {matched_count} players with >=450 Championship & EPL minutes")
                else:
                    print(f"  [WARN]️  No EPL data found for {epl_team} in {next_season_start}")
            else:
                print(f"  [ERR] EPL dataframe is None")
    
    print(f"\n{'='*80}")
    print(f"[OK] Total archetypes created: {len(archetypes)}")
    print(f"{'='*80}")
    return archetypes


def build_archetype_entry(efl_player, epl_stats, champ_team, epl_team, promoted_season, promotion_type='unknown'):
    """Build a single archetype entry with Championship -> EPL stats"""
    
    try:
        # Extract position
        pos = efl_player.get('Pos', 'MID')
        if pd.isna(pos):
            pos = efl_player.get('Pos_Primary', 'MID')
        
        if 'GK' in str(pos):
            position = 'GK'
        elif 'DF' in str(pos) or 'DEF' in str(pos) or 'CB' in str(pos) or 'LB' in str(pos) or 'RB' in str(pos):
            position = 'DEF'
        elif 'FW' in str(pos) or 'FWD' in str(pos) or 'ST' in str(pos) or 'CF' in str(pos):
            position = 'FWD'
        else:
            position = 'MID'
        
        # Calculate per-90 stats for Championship
        champ_minutes = float(efl_player.get('Min', 0))
        champ_90s = champ_minutes / 90.0 if champ_minutes > 0 else 1.0
        
        # Safely get numeric values
        def get_numeric(val, default=0):
            try:
                return float(val) if not pd.isna(val) else default
            except:
                return default
        
        # Build embedding features (14-dimensional)
        embedding = np.array([
            get_numeric(efl_player.get('Gls', 0)) / champ_90s,  # goals_per90
            get_numeric(efl_player.get('Ast', 0)) / champ_90s,  # assists_per90
            get_numeric(efl_player.get('xG', 0)) / champ_90s,  # xG_per90
            get_numeric(efl_player.get('xAG', 0)) / champ_90s,  # xA_per90
            get_numeric(efl_player.get('Sh', 0)) / champ_90s,  # shots_per90
            get_numeric(efl_player.get('KP', 0)) / champ_90s,  # key_passes_per90
            get_numeric(efl_player.get('Tkl', 0)) / champ_90s,  # tackles_per90
            get_numeric(efl_player.get('Int', 0)) / champ_90s,  # interceptions_per90
            get_numeric(efl_player.get('Succ', 0)) / champ_90s,  # dribbles_per90
            get_numeric(efl_player.get('Cmp%', 0)),  # pass_completion_pct
            get_numeric(efl_player.get('Won', 0)) / champ_90s,  # aerials_won_per90
            get_numeric(efl_player.get('PrgC', 0)) / champ_90s,  # progressive_carries_per90
            get_numeric(efl_player.get('SCA', 0)) / champ_90s,  # sca_per90
            get_numeric(efl_player.get('GCA', 0)) / champ_90s,  # gca_per90
        ])
        
        # EPL debut stats
        epl_minutes = float(epl_stats.get('minutes', 0))
        epl_90s = epl_minutes / 90.0 if epl_minutes > 0 else 1.0
        epl_gws = int(epl_stats.get('GW', 1))
        
        archetype = {
            'player_name': str(efl_player['Player']),
            'position': position,
            'promoted_season': promoted_season,
            'promotion_type': promotion_type,
            'age_at_promotion': int(get_numeric(efl_player.get('Age', 25))),
            'championship_team': champ_team,
            'epl_team': epl_team,
            
            # Championship per-90 stats
            'champ_goals_per90': float(embedding[0]),
            'champ_assists_per90': float(embedding[1]),
            'champ_xg_per90': float(embedding[2]),
            'champ_xa_per90': float(embedding[3]),
            'champ_shots_per90': float(embedding[4]),
            'champ_key_passes_per90': float(embedding[5]),
            'champ_tackles_per90': float(embedding[6]),
            'champ_interceptions_per90': float(embedding[7]),
            'champ_dribbles_per90': float(embedding[8]),
            'champ_pass_completion_pct': float(embedding[9]),
            'champ_aerials_won_per90': float(embedding[10]),
            'champ_progressive_carries_per90': float(embedding[11]),
            'champ_sca_per90': float(embedding[12]),
            'champ_gca_per90': float(embedding[13]),
            'champ_minutes_total': int(champ_minutes),
            'champ_starts': int(get_numeric(efl_player.get('Starts', 0))),
            'champ_matches': int(get_numeric(efl_player.get('MP', 0))),
            
            # EPL debut season actuals
            'epl_goals_per90': float(epl_stats.get('goals_scored', 0)) / epl_90s,
            'epl_assists_per90': float(epl_stats.get('assists', 0)) / epl_90s,
            'epl_minutes_total': int(epl_minutes),
            'epl_starts': int(epl_stats.get('starts', 0)),
            'epl_gameweeks_played': epl_gws,
            'epl_avg_fpl_points': float(epl_stats.get('total_points', 0)) / max(epl_gws, 1),
            'epl_total_fpl_points': int(epl_stats.get('total_points', 0)),
            
            # Embedding for KNN matching
            'embedding': embedding,
            
            # Will be assigned by clustering
            'archetype': None,
            'tags': [],
        }
        
        return archetype
    
    except Exception as e:
        print(f"  [WARN]️  Error building archetype for {efl_player.get('Player', 'Unknown')}: {e}")
        return None


def assign_archetypes(archetypes_list):
    """Use k-means clustering to assign archetype labels"""
    print("\nAssigning archetype labels using k-means clustering...")
    
    if len(archetypes_list) < 8:
        print(f"[WARN]️  Only {len(archetypes_list)} archetypes - too few for clustering")
        for arch in archetypes_list:
            arch['archetype'] = 'general'
            arch['tags'] = ['general']
        return archetypes_list
    
    # Extract embeddings
    embeddings = np.array([arch['embedding'] for arch in archetypes_list])
    
    # Standardize features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Find optimal k using silhouette score
    best_k = 8
    best_score = -1
    
    for k in range(5, min(12, len(archetypes_list) // 3)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_scaled)
        score = silhouette_score(embeddings_scaled, labels)
        print(f"  k={k}: silhouette score = {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"\n[OK] Selected k={best_k} (silhouette score: {best_score:.3f})")
    
    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(embeddings_scaled)
    
    # Assign labels based on cluster centroids
    archetype_names = list(ARCHETYPE_LABELS.keys())
    
    for i, arch in enumerate(archetypes_list):
        cluster_id = labels[i]
        
        # Assign archetype name (cycle through if more clusters than names)
        arch['archetype'] = archetype_names[cluster_id % len(archetype_names)]
        arch['tags'] = [arch['archetype']]
        arch['cluster_id'] = int(cluster_id)
    
    # Print cluster distribution
    print("\nCluster distribution:")
    for i in range(best_k):
        count = np.sum(labels == i)
        archetype_name = archetype_names[i % len(archetype_names)]
        print(f"  Cluster {i} ({archetype_name}): {count} players")
    
    return archetypes_list


def save_archetype_database(archetypes_list, output_path):
    """Save archetype database as pickle file"""
    print(f"\nSaving archetype database to {output_path}...")
    
    # Create models directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(archetypes_list, f)
    
    print(f"[OK] Saved {len(archetypes_list)} archetypes")
    
    # Save as CSV for inspection
    csv_path = output_path.with_suffix('.csv')
    df = pd.DataFrame(archetypes_list)
    df = df.drop(columns=['embedding'], errors='ignore')  # Drop array column for CSV
    df.to_csv(csv_path, index=False)
    print(f"[OK] Also saved as CSV: {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("ARCHETYPE DATABASE SUMMARY")
    print("="*80)
    print(f"\nTotal Players: {len(archetypes_list)}")
    
    # By position
    positions = {}
    for arch in archetypes_list:
        pos = arch['position']
        positions[pos] = positions.get(pos, 0) + 1
    
    print("\nBy Position:")
    for pos, count in sorted(positions.items()):
        print(f"  {pos}: {count}")
    
    # By archetype
    archetypes = {}
    for arch in archetypes_list:
        arch_type = arch.get('archetype', 'unknown')
        archetypes[arch_type] = archetypes.get(arch_type, 0) + 1
    
    print("\nBy Archetype:")
    for arch_type, count in sorted(archetypes.items()):
        print(f"  {arch_type}: {count}")
    
    # By season
    seasons = {}
    for arch in archetypes_list:
        season = arch['promoted_season']
        seasons[season] = seasons.get(season, 0) + 1
    
    print("\nBy Promoted Season:")
    for season, count in sorted(seasons.items()):
        print(f"  {season}: {count}")
    
    print("\n" + "="*80)


def main():
    """Main execution function"""
    print("="*80)
    print("PHASE 3: ARCHETYPE DATABASE BUILDER")
    print("Building promoted player database from EFL -> EPL transitions")
    print("="*80)
    
    # Load data
    efl_df = load_efl_data()
    epl_df = load_epl_data()
    
    if epl_df is None:
        print("\n[ERR] ERROR: Could not load EPL data")
        print("Please ensure EPL season CSV files exist")
        return
    
    # Identify promoted players
    archetypes_list = identify_promoted_players(efl_df, epl_df)
    
    if len(archetypes_list) == 0:
        print("\n[ERR] ERROR: No archetypes created")
        print("Check team name mappings and data availability")
        return
    
    # Assign archetype labels using clustering
    archetypes_list = assign_archetypes(archetypes_list)
    
    # Save to file
    output_path = Path('C:/ACL2/FPL/ACL_M3/ml/models/promoted_archetypes_v2.pkl')
    save_archetype_database(archetypes_list, output_path)
    
    print("\n[OK] PHASE 3 COMPLETE")
    print(f"Archetype database saved: {output_path}")
    print(f"Total archetypes: {len(archetypes_list)}")


if __name__ == '__main__':
    main()
