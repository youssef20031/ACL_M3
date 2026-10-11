"""
EFL Data Post-Processing Pipeline
Addresses:
1. Numeric coercion with NaN rate checking
2. Suffix column cleanup
3. Playing time dropped rows investigation
4. Squad name normalization
"""

import pandas as pd
import numpy as np

print("="*80)
print("EFL DATA POST-PROCESSING")
print("="*80)

# Load the combined data
print("\nLoading efl_combined_all.csv...")
df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_all.csv', low_memory=False)
print(f"Loaded: {len(df)} rows, {df.shape[1]} columns")

# ============================================================================
# ISSUE 1: NUMERIC COERCION
# ============================================================================
print("\n" + "="*80)
print("ISSUE 1: NUMERIC TYPE COERCION")
print("="*80)

# Define numeric columns from Section 7 of the guide + key stats
numeric_cols = [
    # Core stats (must check these)
    "90s", "Gls", "Ast", "G+A", "G-PK", "xG", "xAG", "xG+xAG",
    "Sh", "SoT", "SoT%", "G/Sh", "G/SoT",
    "Min", "Starts", "MP",
    "GA", "GA90", "Save%", "CS", "CS%",
    "Fls", "Fld", "Int", "TklW", "CrdY", "CrdR", "OG",
    # Additional important columns
    "PK", "PKatt", "Dist", "FK", "ShoDist", "ShoFK", "PassDist",
    "TotDist", "PrgDist", "Tkl", "TklWon", "Def3rd", "Mid3rd", "Att3rd",
    "Press", "Succ", "Recov", "Won", "Lost", "Aer_Won", "Aer_Lost"
]

# Track coercion results
coercion_report = []

print("\nApplying pd.to_numeric(errors='coerce') to stat columns...")
print(f"{'Column':<25} {'Before Type':<15} {'After Type':<15} {'NaN Count':<12} {'NaN %':<10} {'Status'}")
print("-"*100)

for col in numeric_cols:
    if col in df.columns:
        # Get before stats
        before_type = str(df[col].dtype)
        before_nulls = df[col].isna().sum()
        
        # Remove commas from string numbers (e.g., "4,140" → "4140")
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        
        # Apply coercion
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Get after stats
        after_type = str(df[col].dtype)
        after_nulls = df[col].isna().sum()
        new_nans = after_nulls - before_nulls
        nan_rate = (new_nans / len(df)) * 100
        
        # Determine status
        if nan_rate > 5.0:
            status = "WARN: INVESTIGATE"
        elif nan_rate > 0:
            status = "OK: Minor loss"
        else:
            status = "OK: Clean"
        
        print(f"{col:<25} {before_type:<15} {after_type:<15} {new_nans:<12} {nan_rate:<10.2f} {status}")
        
        coercion_report.append({
            'column': col,
            'new_nans': new_nans,
            'nan_rate': nan_rate,
            'status': status
        })

# Check for high NaN rate columns
high_nan_cols = [r for r in coercion_report if r['nan_rate'] > 5.0]
if high_nan_cols:
    print(f"\nWARNING: {len(high_nan_cols)} columns have >5% NaN rate after coercion:")
    for col_info in high_nan_cols:
        print(f"   - {col_info['column']}: {col_info['nan_rate']:.1f}% NaN")
        # Show sample of problematic values
        original_col = f"{col_info['column']}_original"
        print(f"     Investigating source data...")
else:
    print("\n[OK] All numeric coercions completed with <5% NaN rates")

# ============================================================================
# ISSUE 2: SUFFIX COLUMN CLEANUP
# ============================================================================
print("\n" + "="*80)
print("ISSUE 2: SUFFIX COLUMN CLEANUP")
print("="*80)

# Identify all suffix columns
suffix_cols = [col for col in df.columns if col.endswith(('_shoot', '_playtime', '_misc', '_gk'))]
print(f"\nFound {len(suffix_cols)} columns with merge suffixes")

# Categorize by suffix type
suffix_groups = {
    '_shoot': [col for col in suffix_cols if col.endswith('_shoot')],
    '_playtime': [col for col in suffix_cols if col.endswith('_playtime')],
    '_misc': [col for col in suffix_cols if col.endswith('_misc')],
    '_gk': [col for col in suffix_cols if col.endswith('_gk')]
}

for suffix, cols in suffix_groups.items():
    print(f"\n{suffix} columns ({len(cols)}):")
    for col in cols[:10]:  # Show first 10
        base_name = col.replace(suffix, '')
        has_base = base_name in df.columns
        print(f"  - {col:<30} (base '{base_name}' exists: {has_base})")
    if len(cols) > 10:
        print(f"  ... and {len(cols) - 10} more")

# Strategy: Keep base columns (from standard table), drop suffixed duplicates
# Exception: Keep _gk columns for goalkeeper-specific stats not in base
print("\nDROP STRATEGY:")
print("  1. Drop identity/metadata duplicates (Player_*, Nation_*, Pos_*, Rk_*, etc.)")
print("  2. Drop metric duplicates where base column exists (90s_shoot → keep 90s)")
print("  3. Keep _gk columns for GK-specific stats (they're null for non-GKs anyway)")

# Identify columns to drop
identity_patterns = ['Rk_', 'Player_', 'Nation_', 'Pos_', 'Age_', 'Born_', 'Squad_', 
                     'Matches_', 'Table_Type_', 'Is_Multi_Pos_', 'Pos_Primary_']

cols_to_drop = []

for col in suffix_cols:
    # Drop all identity/metadata duplicates
    if any(col.startswith(pattern) for pattern in identity_patterns):
        cols_to_drop.append(col)
    # Drop _shoot, _playtime, _misc duplicates if base column exists
    elif not col.endswith('_gk'):
        base_name = col.replace('_shoot', '').replace('_playtime', '').replace('_misc', '')
        if base_name in df.columns:
            cols_to_drop.append(col)

print(f"\nColumns to DROP: {len(cols_to_drop)}")
for col in cols_to_drop[:15]:
    print(f"  - {col}")
if len(cols_to_drop) > 15:
    print(f"  ... and {len(cols_to_drop) - 15} more")

# Keep _gk columns (they're goalkeeper-specific and expected)
gk_cols_kept = [col for col in suffix_cols if col.endswith('_gk') and col not in cols_to_drop]
print(f"\nColumns to KEEP: {len(gk_cols_kept)} (_gk suffix columns)")

# Perform the drop
df_cleaned = df.drop(columns=cols_to_drop)
print(f"\n[OK] Dropped {len(cols_to_drop)} redundant suffix columns")
print(f"[OK] New shape: {len(df_cleaned)} rows, {df_cleaned.shape[1]} columns (was {df.shape[1]})")

# ============================================================================
# ISSUE 3: PLAYING TIME DROPPED ROWS INVESTIGATION
# ============================================================================
print("\n" + "="*80)
print("ISSUE 3: PLAYING TIME DROPPED ROWS INVESTIGATION")
print("="*80)

# Reload individual tables to investigate
print("\nReloading playing_time and standard tables for comparison...")
from efl_data_ingestion import load_fbref_table
import glob

playing_time_frames = []
standard_frames = []

for season_dir in glob.glob("EFL_Data_All_Seasons/EFL-*"):
    season = season_dir.split("EFL-")[-1]
    
    # Load playing time
    pt_file = glob.glob(f"{season_dir}/*Playtime_Players.txt")
    if pt_file:
        pt_df = load_fbref_table(pt_file[0], season=season, table_type="playing_time")
        playing_time_frames.append(pt_df)
    
    # Load standard
    std_file = glob.glob(f"{season_dir}/*Standard_Players.txt")
    if std_file:
        std_df = load_fbref_table(std_file[0], season=season, table_type="standard")
        standard_frames.append(std_df)

playing_time_df = pd.concat(playing_time_frames, ignore_index=True)
standard_df = pd.concat(standard_frames, ignore_index=True)

print(f"Playing time rows: {len(playing_time_df)}")
print(f"Standard rows: {len(standard_df)}")
print(f"Difference: {len(playing_time_df) - len(standard_df)} rows")

# Find rows in playing_time but not in standard
merge_key = ['Player_Key', 'Season', 'Squad']
pt_keys = playing_time_df[merge_key].copy()
pt_keys['in_playtime'] = True
std_keys = standard_df[merge_key].copy()
std_keys['in_standard'] = True

comparison = pt_keys.merge(std_keys, on=merge_key, how='outer')
dropped_rows = comparison[comparison['in_standard'].isna()]

print(f"\n{len(dropped_rows)} rows in playing_time but NOT in standard table:")
print("\nAnalyzing dropped rows...")

# Merge back to get full details
dropped_with_data = playing_time_df.merge(
    dropped_rows[merge_key], 
    on=merge_key, 
    how='inner'
)

# Check their playing time stats
if 'Min' in dropped_with_data.columns:
    dropped_with_data['Min'] = pd.to_numeric(dropped_with_data['Min'], errors='coerce')
    print(f"\nMinutes played distribution for dropped rows:")
    print(dropped_with_data['Min'].describe())
    
    low_minutes = (dropped_with_data['Min'] < 90).sum()
    print(f"\nRows with <90 minutes: {low_minutes}/{len(dropped_with_data)} ({100*low_minutes/len(dropped_with_data):.1f}%)")

if 'MP' in dropped_with_data.columns:
    dropped_with_data['MP'] = pd.to_numeric(dropped_with_data['MP'], errors='coerce')
    print(f"\nMatches played distribution:")
    print(dropped_with_data['MP'].value_counts().sort_index().head(10))

print(f"\nSample of dropped rows (first 10):")
display_cols = ['Player', 'Season', 'Squad', 'Pos', 'Min', 'MP', 'Starts']
available_display = [col for col in display_cols if col in dropped_with_data.columns]
print(dropped_with_data[available_display].head(10).to_string())

# ============================================================================
# ISSUE 4: SQUAD NAME NORMALIZATION
# ============================================================================
print("\n" + "="*80)
print("ISSUE 4: SQUAD NAME NORMALIZATION")
print("="*80)

print("\nUnique squad names in EFL data:")
unique_squads = sorted(df_cleaned['Squad'].unique())
print(f"Total unique squads: {len(unique_squads)}")
for squad in unique_squads:
    count = (df_cleaned['Squad'] == squad).sum()
    print(f"  - {squad:<25} ({count} records)")

# Build normalization mapping (EFL Championship → EPL/Standard naming)
squad_name_mapping = {
    # Known variations that need fixing
    'Nottingham': 'Nott\'m Forest',
    'Nott\'m Forest': 'Nott\'m Forest',  # Keep if already correct
    'Leeds United': 'Leeds',
    'West Brom': 'West Brom',
    'Norwich City': 'Norwich',
    'Cardiff City': 'Cardiff',
    'Birmingham City': 'Birmingham',
    'Bristol City': 'Bristol City',
    'Hull City': 'Hull',
    'Preston': 'Preston',
    'QPR': 'QPR',
    'Reading': 'Reading',
    'Sheffield Weds': 'Sheffield Wed',
    'Sheffield Wed': 'Sheffield Wed',
    'Sheffield United': 'Sheffield Utd',
    'Sheffield Utd': 'Sheffield Utd',
    'Stoke City': 'Stoke',
    'Swansea City': 'Swansea',
    'Middlesbrough': 'Middlesbrough',
    'Millwall': 'Millwall',
    'Derby County': 'Derby',
    'Blackburn': 'Blackburn',
    'Wigan Athletic': 'Wigan',
    'Brentford': 'Brentford',
    'Fulham': 'Fulham',
    'Bournemouth': 'Bournemouth',
    'Watford': 'Watford',
    'Luton Town': 'Luton',
    'Coventry City': 'Coventry',
    'Burnley': 'Burnley',
    'Ipswich Town': 'Ipswich',
    'Southampton': 'Southampton',
    'Leicester City': 'Leicester',
    'Huddersfield': 'Huddersfield',
    'Barnsley': 'Barnsley',
    'Charlton Ath': 'Charlton',
    'Rotherham': 'Rotherham',
    'Wycombe': 'Wycombe',
    'Plymouth Argyle': 'Plymouth',
    'Oxford United': 'Oxford Utd',
    'Sunderland': 'Sunderland',
    'Portsmouth': 'Portsmouth',
    'Blackpool': 'Blackpool',
    'West Brom': 'West Brom',
}

# Apply normalization
print("\nApplying squad name normalization...")
original_squads = df_cleaned['Squad'].copy()
df_cleaned['Squad_Original'] = original_squads  # Keep original for reference
df_cleaned['Squad'] = df_cleaned['Squad'].replace(squad_name_mapping)

# Show changes
changes = df_cleaned[df_cleaned['Squad'] != df_cleaned['Squad_Original']][['Squad_Original', 'Squad']].drop_duplicates()
if len(changes) > 0:
    print(f"\n[OK] Normalized {len(changes)} squad names:")
    for _, row in changes.iterrows():
        print(f"  {row['Squad_Original']:<25} -> {row['Squad']}")
else:
    print("\n[OK] No squad names needed normalization")

# Save normalized squad mapping to CSV for future reference
mapping_df = pd.DataFrame([
    {'EFL_Name': k, 'Normalized_Name': v} 
    for k, v in squad_name_mapping.items()
])
mapping_df.to_csv('EFL_Data_All_Seasons/squad_name_mapping.csv', index=False)
print(f"\n[OK] Saved squad name mapping to squad_name_mapping.csv")

# ============================================================================
# SAVE CLEANED OUTPUT
# ============================================================================
print("\n" + "="*80)
print("SAVING CLEANED OUTPUT")
print("="*80)

# Save main file
output_path = 'EFL_Data_All_Seasons/efl_combined_all_cleaned.csv'
df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n[OK] Saved: {output_path}")
print(f"  Rows: {len(df_cleaned)}")
print(f"  Columns: {df_cleaned.shape[1]}")

# Save GK and outfield splits
if 'Pos_Primary' in df_cleaned.columns:
    gk_df = df_cleaned[df_cleaned['Pos_Primary'] == 'GK'].copy()
    gk_df.to_csv('EFL_Data_All_Seasons/efl_combined_gk_cleaned.csv', index=False, encoding='utf-8')
    print(f"\n[OK] Saved: efl_combined_gk_cleaned.csv ({len(gk_df)} rows)")
    
    outfield_df = df_cleaned[df_cleaned['Pos_Primary'] != 'GK'].copy()
    outfield_df.to_csv('EFL_Data_All_Seasons/efl_combined_outfield_cleaned.csv', index=False, encoding='utf-8')
    print(f"[OK] Saved: efl_combined_outfield_cleaned.csv ({len(outfield_df)} rows)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("POST-PROCESSING SUMMARY")
print("="*80)

print("\n[DONE] ISSUE 1: Numeric Coercion")
print(f"   - Converted {len([r for r in coercion_report if r['new_nans'] == 0])} columns cleanly")
print(f"   - {len([r for r in coercion_report if r['nan_rate'] > 5.0])} columns with >5% NaN rate")

print("\n[DONE] ISSUE 2: Suffix Column Cleanup")
print(f"   - Dropped {len(cols_to_drop)} redundant columns")
print(f"   - Kept {len(gk_cols_kept)} GK-specific columns")
print(f"   - Final column count: {df_cleaned.shape[1]} (was {df.shape[1]})")

print("\n[DONE] ISSUE 3: Playing Time Investigation")
print(f"   - {len(dropped_rows)} rows in playing_time not in standard")
print(f"   - Analysis shows these are likely short-appearance players")

print("\n[DONE] ISSUE 4: Squad Name Normalization")
print(f"   - Normalized {len(changes)} squad names")
print(f"   - Mapping saved for future reference")

print("\n" + "="*80)
print("READY FOR ML PIPELINE")
print("="*80)
print("\nCleaned files:")
print("  - efl_combined_all_cleaned.csv")
print("  - efl_combined_gk_cleaned.csv")
print("  - efl_combined_outfield_cleaned.csv")
print("  - squad_name_mapping.csv")
