"""
Validation script to check the quality of EFL ingestion output.
"""
import pandas as pd

print("="*80)
print("EFL DATA QUALITY VALIDATION")
print("="*80)

# Load the main output file
df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_all.csv')

print(f"\n1. SHAPE AND BASIC INFO")
print(f"   Total rows: {len(df)}")
print(f"   Total columns: {df.shape[1]}")

print(f"\n2. SEASONS COVERED")
seasons = sorted(df['Season'].unique())
print(f"   Seasons: {seasons}")
for season in seasons:
    count = (df['Season'] == season).sum()
    print(f"   - {season}: {count} player records")

print(f"\n3. POSITION DISTRIBUTION")
print(df['Pos_Primary'].value_counts())

print(f"\n4. SAMPLE PLAYER DATA (first 10 rows)")
print(df[['Player', 'Player_Key', 'Season', 'Squad', 'Pos_Primary', 'Age']].head(10).to_string())

print(f"\n5. NULL VALUE ANALYSIS")
nulls = df.isnull().sum()
nulls_present = nulls[nulls > 0].sort_values(ascending=False)
if len(nulls_present) > 0:
    print(f"   Columns with null values (top 20):")
    for col, count in nulls_present.head(20).items():
        pct = (count / len(df)) * 100
        print(f"   - {col}: {count} ({pct:.1f}%)")
else:
    print("   No null values found!")

print(f"\n6. ENCODING CHECK")
# Check for encoding artifacts
if 'Player' in df.columns:
    garbled = df['Player'].str.contains("Ã|Â|â€", na=False, regex=True)
    if garbled.any():
        print(f"   WARNING: {garbled.sum()} player names with encoding artifacts")
        print(f"   Examples:")
        print(df[garbled][['Player', 'Season']].head())
    else:
        print("   ✓ No encoding artifacts detected in Player names")

print(f"\n7. MERGE KEY INTEGRITY")
print(f"   Player_Key nulls: {df['Player_Key'].isna().sum()}")
print(f"   Season nulls: {df['Season'].isna().sum()}")
print(f"   Squad nulls: {df['Squad'].isna().sum()}")

print(f"\n8. COLUMN SUFFIX CHECK (merge artifacts)")
suffix_cols = [col for col in df.columns if col.endswith(('_shoot', '_playtime', '_misc', '_gk'))]
if suffix_cols:
    print(f"   Found {len(suffix_cols)} columns with merge suffixes:")
    for col in suffix_cols[:15]:
        print(f"   - {col}")
else:
    print("   ✓ No unexpected merge suffix columns")

print(f"\n9. NUMERIC COLUMN SAMPLE")
numeric_cols = ['90s', 'Gls', 'Ast', 'xG', 'xAG']
available = [col for col in numeric_cols if col in df.columns]
if available:
    print(df[available].describe())

print(f"\n10. GOALKEEPER-SPECIFIC CHECK")
gk_df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_gk.csv')
print(f"    GK file rows: {len(gk_df)}")
print(f"    GK file columns: {gk_df.shape[1]}")
# Check for GK-specific columns
gk_specific = ['GA', 'GA90', 'Save%', 'CS', 'CS%']
gk_cols_present = [col for col in gk_specific if col in gk_df.columns]
print(f"    GK-specific columns present: {gk_cols_present}")

print(f"\n11. OUTFIELD PLAYER CHECK")
outfield_df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_outfield.csv')
print(f"    Outfield file rows: {len(outfield_df)}")
print(f"    Outfield file columns: {outfield_df.shape[1]}")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
