"""
Fix numeric columns in EFL data by removing commas before coercion.
This script re-processes the cleaned data to fix the Min column issue.
"""

import pandas as pd

print("="*80)
print("FIX NUMERIC COLUMNS IN EFL DATA")
print("="*80)

# Load the cleaned file
df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_all_cleaned.csv')
print(f"\nLoaded cleaned data: {len(df)} rows, {df.shape[1]} columns")

# Identify all numeric columns (any column that should contain numbers)
# This includes all stat columns
potential_numeric_cols = df.columns[df.columns.str.match(r'^[A-Z]|^[0-9]|^\d|^x[GA]|^%|Per|Poss|Tkl|Press|Aer|Won|Lost|Fls|Fld|Off|Crs|Int|Recov')]

print(f"\nChecking {len(potential_numeric_cols)} potential numeric columns for comma formatting...")

fixed_count = 0
for col in df.columns:
    if df[col].dtype == 'object':  # Only process string columns
        # Check if column contains comma-formatted numbers
        sample = df[col].dropna().head(100)
        if sample.astype(str).str.match(r'^\d{1,3}(,\d{3})+$').any():
            print(f"  Fixing: {col}")
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            fixed_count += 1

print(f"\n[DONE] Fixed {fixed_count} columns with comma formatting")

# Double-check Min column specifically
print(f"\nMin column status:")
print(f"  Dtype: {df['Min'].dtype}")
print(f"  Null count: {df['Min'].isna().sum()} / {len(df)} ({100*df['Min'].isna().sum()/len(df):.1f}%)")
print(f"  Non-null range: {df['Min'].min():.0f} to {df['Min'].max():.0f}")

# Save the fixed version
df.to_csv('EFL_Data_All_Seasons/efl_combined_all_cleaned.csv', index=False, encoding='utf-8')
print(f"\n[OK] Saved updated file: efl_combined_all_cleaned.csv")

# Update splits
if 'Pos_Primary' in df.columns:
    gk_df = df[df['Pos_Primary'] == 'GK'].copy()
    gk_df.to_csv('EFL_Data_All_Seasons/efl_combined_gk_cleaned.csv', index=False, encoding='utf-8')
    
    outfield_df = df[df['Pos_Primary'] != 'GK'].copy()
    outfield_df.to_csv('EFL_Data_All_Seasons/efl_combined_outfield_cleaned.csv', index=False, encoding='utf-8')
    print(f"[OK] Updated splits: gk_cleaned.csv ({len(gk_df)} rows), outfield_cleaned.csv ({len(outfield_df)} rows)")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
