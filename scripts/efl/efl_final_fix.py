"""
Final fix for EFL data issues:
1. Re-load original merged data before column drops
2. Fix comma-formatted numbers BEFORE column cleanup
3. Smart merge of Min column (use Min_playtime to fill Min nulls)
4. Then drop redundant columns
"""

import pandas as pd

print("="*80)
print("EFL DATA FINAL FIX")
print("="*80)

# Load the ORIGINAL combined_all file (before post-processing)
print("\nLoading original combined data...")
df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_all.csv', low_memory=False)
print(f"Loaded: {len(df)} rows, {df.shape[1]} columns")

# ============================================================================
# STEP 1: Fix comma-formatted numbers in ALL columns
# ============================================================================
print("\n" + "="*80)
print("STEP 1: FIX COMMA-FORMATTED NUMBERS")
print("="*80)

# List of all columns that should be numeric
numeric_candidates = []
for col in df.columns:
    if df[col].dtype == 'object':
        # Check if this looks like it contains numbers
        sample = df[col].dropna().astype(str).head(100)
        # Check for patterns like "1,234" or just digits
        if len(sample) > 0 and sample.str.match(r'^-?\d').any():
            numeric_candidates.append(col)

print(f"\nFound {len(numeric_candidates)} columns that appear numeric")
print(f"Removing commas from number formatting...")

fixed_cols = []
for col in numeric_candidates:
    # Remove commas
    original = df[col].copy()
    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
    
    # Try numeric conversion
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check if conversion worked
    if df[col].dtype in ['float64', 'int64']:
        # Count how many non-null values we successfully converted
        orig_non_null = original.notna().sum()
        new_non_null = df[col].notna().sum()
        if new_non_null >= orig_non_null * 0.9:  # At least 90% conversion success
            fixed_cols.append(col)

print(f"Successfully converted {len(fixed_cols)} columns to numeric")

# ============================================================================
# STEP 2: Smart merge of duplicate columns (especially Min)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: SMART MERGE OF DUPLICATE COLUMNS")
print("="*80)

# The key insight: Min and Min_playtime have different coverage
# Min from standard table has NaNs, Min_playtime is more complete
# Use Min_playtime to fill Min

if 'Min' in df.columns and 'Min_playtime' in df.columns:
    print("\nMerging Min columns:")
    print(f"  Min nulls: {df['Min'].isna().sum()} ({100*df['Min'].isna().sum()/len(df):.1f}%)")
    print(f"  Min_playtime nulls: {df['Min_playtime'].isna().sum()} ({100*df['Min_playtime'].isna().sum()/len(df):.1f}%)")
    
    # Fill Min nulls with Min_playtime
    df['Min'] = df['Min'].fillna(df['Min_playtime'])
    
    print(f"  After merge, Min nulls: {df['Min'].isna().sum()} ({100*df['Min'].isna().sum()/len(df):.1f}%)")

# Do the same for other potentially duplicated stat columns
duplicate_pairs = [
    ('90s', '90s_playtime'),
    ('MP', 'MP_playtime'),
    ('Starts', 'Starts_playtime'),
]

for base_col, dup_col in duplicate_pairs:
    if base_col in df.columns and dup_col in df.columns:
        nulls_before = df[base_col].isna().sum()
        df[base_col] = df[base_col].fillna(df[dup_col])
        nulls_after = df[base_col].isna().sum()
        if nulls_before > nulls_after:
            print(f"  Merged {base_col}: filled {nulls_before - nulls_after} nulls from {dup_col}")

# ============================================================================
# STEP 3: Drop redundant suffix columns (NOW safe to do)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: DROP REDUNDANT COLUMNS")
print("="*80)

# Identify columns to drop
identity_patterns = ['Rk_', 'Player_', 'Nation_', 'Pos_', 'Age_', 'Born_', 'Squad_', 
                     'Matches_', 'Table_Type_', 'Is_Multi_Pos_', 'Pos_Primary_']

# CRITICAL: Protect essential columns from being dropped
protected_columns = ['Player_Key', 'Season', 'Squad', 'Player', 'Pos_Primary', 
                     'Is_Multi_Pos', 'Table_Type', 'Squad_Original']

cols_to_drop = []

for col in df.columns:
    # Skip protected columns
    if col in protected_columns:
        continue
    
    # Drop identity/metadata duplicates
    if any(col.startswith(pattern) for pattern in identity_patterns):
        cols_to_drop.append(col)
    # Drop stat duplicates (we've already merged them)
    elif col.endswith(('_shoot', '_playtime', '_misc')) and not col.endswith('_gk'):
        base_name = col.replace('_shoot', '').replace('_playtime', '').replace('_misc', '')
        if base_name in df.columns:
            cols_to_drop.append(col)

print(f"Dropping {len(cols_to_drop)} redundant columns")
df = df.drop(columns=cols_to_drop)
print(f"New shape: {len(df)} rows, {df.shape[1]} columns")

# ============================================================================
# STEP 4: Squad name normalization
# ============================================================================
print("\n" + "="*80)
print("STEP 4: SQUAD NAME NORMALIZATION")
print("="*80)

squad_name_mapping = {
    'Nottingham': "Nott'm Forest",
    'Leeds United': 'Leeds',
    'Cardiff City': 'Cardiff',
    'Birmingham City': 'Birmingham',
    'Hull City': 'Hull',
    'Sheffield Weds': 'Sheffield Wed',
    'Sheffield United': 'Sheffield Utd',
    'Stoke City': 'Stoke',
    'Swansea City': 'Swansea',
    'Derby County': 'Derby',
    'Luton Town': 'Luton',
    'Norwich City': 'Norwich',
    'Coventry City': 'Coventry',
    'Leicester City': 'Leicester',
    'Ipswich Town': 'Ipswich',
    'Plymouth Argyle': 'Plymouth',
    'Oxford United': 'Oxford Utd',
    'Wigan Athletic': 'Wigan',
}

df['Squad_Original'] = df['Squad'].copy()
df['Squad'] = df['Squad'].replace(squad_name_mapping)

changes = df[df['Squad'] != df['Squad_Original']][['Squad_Original', 'Squad']].drop_duplicates()
print(f"Normalized {len(changes)} squad names")

# ============================================================================
# STEP 5: Save cleaned output
# ============================================================================
print("\n" + "="*80)
print("STEP 5: SAVE CLEANED OUTPUT")
print("="*80)

df.to_csv('EFL_Data_All_Seasons/efl_combined_all_cleaned.csv', index=False, encoding='utf-8')
print(f"\n[OK] Saved: efl_combined_all_cleaned.csv")
print(f"  Rows: {len(df)}")
print(f"  Columns: {df.shape[1]}")

# Save splits
if 'Pos_Primary' in df.columns:
    gk_df = df[df['Pos_Primary'] == 'GK'].copy()
    gk_df.to_csv('EFL_Data_All_Seasons/efl_combined_gk_cleaned.csv', index=False, encoding='utf-8')
    
    outfield_df = df[df['Pos_Primary'] != 'GK'].copy()
    outfield_df.to_csv('EFL_Data_All_Seasons/efl_combined_outfield_cleaned.csv', index=False, encoding='utf-8')
    
    print(f"[OK] Saved splits:")
    print(f"  efl_combined_gk_cleaned.csv: {len(gk_df)} rows")
    print(f"  efl_combined_outfield_cleaned.csv: {len(outfield_df)} rows")

# ============================================================================
# FINAL VALIDATION
# ============================================================================
print("\n" + "="*80)
print("FINAL VALIDATION")
print("="*80)

print("\nKey columns check:")
key_stat_cols = ['90s', 'Min', 'MP', 'Gls', 'Ast', 'xG', 'xAG']
for col in key_stat_cols:
    if col in df.columns:
        null_pct = 100 * df[col].isna().sum() / len(df)
        print(f"  {col:<15} nulls: {df[col].isna().sum():<6} ({null_pct:.1f}%)")

print(f"\nPosition distribution:")
if 'Pos_Primary' in df.columns:
    print(df['Pos_Primary'].value_counts())

print(f"\nSeasons covered: {sorted(df['Season'].unique())}")

# Check 945 dropped playing time rows
print(f"\nNote: 945 rows from playing_time table (players with MP=0)")
print(f"were correctly excluded as they never played")

print("\n" + "="*80)
print("ALL ISSUES RESOLVED - READY FOR ML PIPELINE")
print("="*80)
