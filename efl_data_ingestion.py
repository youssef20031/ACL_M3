"""
EFL Championship Data Ingestion and Cleaning Pipeline
Follows EFL_FBREF_INGESTION_GUIDE.md exactly.
"""

import pandas as pd
import unicodedata
import os
import glob
from typing import Dict, List, Tuple

# ============================================================================
# SECTION 1: Core Functions from the Guide
# ============================================================================

def normalize_name(name: str) -> str:
    """
    Strip accents and lowercase for safe cross-dataset player matching.
    André Ayew → andre ayew
    """
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    return name.encode("ascii", "ignore").decode("ascii").lower().strip()


def load_fbref_table(filepath: str, season: str, table_type: str) -> pd.DataFrame:
    """
    Load a single FBref EFL Championship table.

    Args:
        filepath:   path to the .txt or .csv file
        season:     season label, e.g. "2021-22"
        table_type: category label, e.g. "standard", "shooting", "goalkeeping"

    Returns:
        Clean DataFrame with metadata columns appended.
    """
    print(f"\n[LOADING] {os.path.basename(filepath)}")
    
    # Step 1: Load with tab delimiter and UTF-8 encoding
    df = pd.read_csv(filepath, sep="\t", encoding="utf-8", low_memory=False)
    initial_rows = len(df)
    print(f"  Initial rows: {initial_rows}")

    # Step 2: Remove repeated FBref header rows
    if "Player" in df.columns:
        header_rows = (df["Player"] == "Player").sum()
        df = df[df["Player"] != "Player"]
        if header_rows > 0:
            print(f"  Removed {header_rows} repeated header rows")
    
    if "Rk" in df.columns:
        rk_header_rows = (df["Rk"] == "Rk").sum()
        df = df[df["Rk"] != "Rk"]
        if rk_header_rows > 0:
            print(f"  Removed {rk_header_rows} 'Rk' header rows")
    
    df = df.reset_index(drop=True)
    rows_after_cleaning = len(df)
    print(f"  Rows after header removal: {rows_after_cleaning} (removed {initial_rows - rows_after_cleaning})")

    # Step 3: Normalize position field
    if "Pos" in df.columns:
        df["Pos"] = df["Pos"].str.strip().str.replace(" ", "")
        df["Pos_Primary"] = df["Pos"].str.split(",").str[0]
        df["Is_Multi_Pos"] = df["Pos"].str.contains(",", na=False)

    # Step 4: Normalize player name for cross-table merging
    if "Player" in df.columns:
        df["Player_Key"] = df["Player"].apply(normalize_name)

    # Step 5: Attach metadata
    df["Season"] = season
    df["Table_Type"] = table_type

    return df


def validate_table(df: pd.DataFrame, filepath: str) -> List[str]:
    """
    Run validation checks on a loaded table.
    Returns list of issues found (empty list = passed).
    """
    issues = []

    # 1. Check for unsplit rows (entire row in column 1)
    if df.shape[1] < 5:
        issues.append(f"CRITICAL: Only {df.shape[1]} columns — likely parsed with wrong delimiter")

    # 2. Check for surviving header rows
    if "Player" in df.columns and (df["Player"] == "Player").any():
        issues.append("WARNING: Repeated header rows not fully removed")

    # 3. Check for encoding artifacts
    if "Player" in df.columns:
        garbled = df["Player"].str.contains("Ã|Â|â€", na=False, regex=True)
        if garbled.any():
            issues.append(f"WARNING: {garbled.sum()} player names with encoding artifacts")

    # 4. Check Player_Key populated
    if "Player_Key" not in df.columns or df["Player_Key"].isna().all():
        issues.append("WARNING: Player_Key column missing or empty")

    return issues


# ============================================================================
# SECTION 2: File Discovery and Inventory
# ============================================================================

def discover_files(base_dir: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Task 1: Inventory all files in the EFL data directory.
    
    Returns:
        Dict mapping season -> list of (filepath, table_type, player_or_team)
    """
    print("\n" + "="*80)
    print("TASK 1: FILE INVENTORY")
    print("="*80)
    
    seasons = [
        "2019-2020", "2020-2021", "2021-2022", 
        "2022-2023", "2023-2024", "2024-2025", "2025-2026"
    ]
    
    files_by_season = {}
    all_files = []
    invalid_files = []
    
    for season in seasons:
        season_dir = os.path.join(base_dir, f"EFL-{season}")
        if not os.path.exists(season_dir):
            print(f"\n[WARNING] Season directory not found: {season_dir}")
            continue
        
        season_files = []
        
        # Find all .txt files in this season's directory
        txt_files = glob.glob(os.path.join(season_dir, "*.txt"))
        
        for filepath in txt_files:
            filename = os.path.basename(filepath)
            
            # Check if file is readable
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    f.read(1)  # Try reading one character
                readable = True
            except Exception as e:
                readable = False
                print(f"\n[ERROR] Cannot read {filename}: {e}")
                invalid_files.append(filename)
                continue
            
            # Parse filename: {season}_{table_type}_{Players/Teams}.txt
            # Handle variations like "2022_2023" vs "2022-2023"
            parts = filename.replace(".txt", "").split("_")
            
            if len(parts) >= 3:
                table_type = parts[-2].lower()  # Second-to-last part
                player_or_team = parts[-1].lower()  # Last part
                
                # Normalize table type names
                if table_type == "playtime":
                    table_type = "playing_time"
                elif table_type == "misc":
                    table_type = "misc"
                elif table_type == "standard":
                    table_type = "standard"
                elif table_type == "shooting":
                    table_type = "shooting"
                elif table_type == "goalkeeping":
                    table_type = "goalkeeping"
                
                # Only process player files (not team files)
                if "player" in player_or_team:
                    season_files.append((filepath, table_type, player_or_team))
                    all_files.append((filepath, season, table_type))
                    print(f"  ✓ {filename} → Season: {season}, Type: {table_type}, Entity: {player_or_team}")
                else:
                    print(f"  ⊗ {filename} → Skipped (team file, not player file)")
            else:
                print(f"  ✗ {filename} → Invalid filename pattern (expected: season_type_entity.txt)")
                invalid_files.append(filename)
        
        if season_files:
            files_by_season[season] = season_files
    
    print(f"\n[SUMMARY] Found {len(all_files)} valid player files across {len(files_by_season)} seasons")
    if invalid_files:
        print(f"[SUMMARY] {len(invalid_files)} invalid/unreadable files: {invalid_files}")
    
    return files_by_season, all_files, invalid_files


# ============================================================================
# SECTION 3: Load and Validate Each File
# ============================================================================

def load_and_validate_all(files_by_season: Dict) -> Tuple[List, List]:
    """
    Task 2: Load and validate each file using the canonical pipeline.
    
    Returns:
        (successful_dataframes, failed_files)
    """
    print("\n" + "="*80)
    print("TASK 2: LOAD AND VALIDATE ALL FILES")
    print("="*80)
    
    successful_dfs = []
    failed_files = []
    
    for season, file_list in files_by_season.items():
        print(f"\n--- Processing Season: {season} ---")
        
        for filepath, table_type, entity_type in file_list:
            try:
                # Load using canonical pipeline
                df = load_fbref_table(filepath, season=season, table_type=table_type)
                
                # Validate
                issues = validate_table(df, filepath)
                
                if issues:
                    print(f"  [VALIDATION FAILED] {os.path.basename(filepath)}")
                    for issue in issues:
                        print(f"    - {issue}")
                    failed_files.append((filepath, issues))
                else:
                    print(f"  [VALIDATION PASSED] {os.path.basename(filepath)} → {len(df)} rows, {df.shape[1]} columns")
                    successful_dfs.append(df)
                    
            except Exception as e:
                print(f"  [LOAD FAILED] {os.path.basename(filepath)}: {e}")
                failed_files.append((filepath, [f"Exception during load: {e}"]))
    
    print(f"\n[SUMMARY] Successfully loaded: {len(successful_dfs)} files")
    print(f"[SUMMARY] Failed: {len(failed_files)} files")
    
    return successful_dfs, failed_files


# ============================================================================
# SECTION 4: Combine by Table Type
# ============================================================================

def combine_by_table_type(successful_dfs: List[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Task 3: Concatenate all successfully loaded files into per-type DataFrames.
    """
    print("\n" + "="*80)
    print("TASK 3: COMBINE BY TABLE TYPE")
    print("="*80)
    
    # Separate by table type
    standard_frames = [df for df in successful_dfs if df["Table_Type"].iloc[0] == "standard"]
    shooting_frames = [df for df in successful_dfs if df["Table_Type"].iloc[0] == "shooting"]
    goalkeeping_frames = [df for df in successful_dfs if df["Table_Type"].iloc[0] == "goalkeeping"]
    misc_frames = [df for df in successful_dfs if df["Table_Type"].iloc[0] == "misc"]
    playing_time_frames = [df for df in successful_dfs if df["Table_Type"].iloc[0] == "playing_time"]
    
    combined = {}
    
    if standard_frames:
        combined["standard"] = pd.concat(standard_frames, ignore_index=True)
        print(f"\n[standard_df] {len(combined['standard'])} rows from {len(standard_frames)} files")
        print(f"  Seasons: {sorted(combined['standard']['Season'].unique())}")
        print(f"  Player_Key nulls: {combined['standard']['Player_Key'].isna().sum()}")
    
    if shooting_frames:
        combined["shooting"] = pd.concat(shooting_frames, ignore_index=True)
        print(f"\n[shooting_df] {len(combined['shooting'])} rows from {len(shooting_frames)} files")
        print(f"  Seasons: {sorted(combined['shooting']['Season'].unique())}")
        print(f"  Player_Key nulls: {combined['shooting']['Player_Key'].isna().sum()}")
    
    if goalkeeping_frames:
        combined["goalkeeping"] = pd.concat(goalkeeping_frames, ignore_index=True)
        print(f"\n[goalkeeping_df] {len(combined['goalkeeping'])} rows from {len(goalkeeping_frames)} files")
        print(f"  Seasons: {sorted(combined['goalkeeping']['Season'].unique())}")
        print(f"  Player_Key nulls: {combined['goalkeeping']['Player_Key'].isna().sum()}")
    
    if misc_frames:
        combined["misc"] = pd.concat(misc_frames, ignore_index=True)
        print(f"\n[misc_df] {len(combined['misc'])} rows from {len(misc_frames)} files")
        print(f"  Seasons: {sorted(combined['misc']['Season'].unique())}")
        print(f"  Player_Key nulls: {combined['misc']['Player_Key'].isna().sum()}")
    
    if playing_time_frames:
        combined["playing_time"] = pd.concat(playing_time_frames, ignore_index=True)
        print(f"\n[playing_time_df] {len(combined['playing_time'])} rows from {len(playing_time_frames)} files")
        print(f"  Seasons: {sorted(combined['playing_time']['Season'].unique())}")
        print(f"  Player_Key nulls: {combined['playing_time']['Player_Key'].isna().sum()}")
    
    return combined


# ============================================================================
# SECTION 5: Merge Across Table Types
# ============================================================================

def merge_all_tables(combined: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Task 4: Merge using composite key ["Player_Key", "Season", "Squad"]
    """
    print("\n" + "="*80)
    print("TASK 4: MERGE ACROSS TABLE TYPES")
    print("="*80)
    
    merge_key = ["Player_Key", "Season", "Squad"]
    
    # Start with standard as base
    if "standard" not in combined:
        raise ValueError("Cannot merge: standard_df is missing!")
    
    merged = combined["standard"].copy()
    print(f"\n[BASE] standard_df: {len(merged)} rows, {merged.shape[1]} columns")
    
    # Left join shooting
    if "shooting" in combined:
        before_cols = merged.shape[1]
        merged = merged.merge(
            combined["shooting"], 
            on=merge_key, 
            how="left", 
            suffixes=("", "_shoot")
        )
        print(f"[+] shooting_df: {len(merged)} rows, {merged.shape[1]} columns (added {merged.shape[1] - before_cols} cols)")
    
    # Left join playing_time
    if "playing_time" in combined:
        before_cols = merged.shape[1]
        merged = merged.merge(
            combined["playing_time"], 
            on=merge_key, 
            how="left", 
            suffixes=("", "_playtime")
        )
        print(f"[+] playing_time_df: {len(merged)} rows, {merged.shape[1]} columns (added {merged.shape[1] - before_cols} cols)")
    
    # Left join misc
    if "misc" in combined:
        before_cols = merged.shape[1]
        merged = merged.merge(
            combined["misc"], 
            on=merge_key, 
            how="left", 
            suffixes=("", "_misc")
        )
        print(f"[+] misc_df: {len(merged)} rows, {merged.shape[1]} columns (added {merged.shape[1] - before_cols} cols)")
    
    # Check for column name collisions
    suffix_cols = [col for col in merged.columns if col.endswith(("_shoot", "_playtime", "_misc", "_gk"))]
    if suffix_cols:
        print(f"\n[WARNING] Found {len(suffix_cols)} columns with merge suffixes:")
        for col in suffix_cols[:10]:  # Show first 10
            print(f"  - {col}")
    
    print(f"\n[MERGED] Final merged_df (before GK split): {len(merged)} rows, {merged.shape[1]} columns")
    
    # Now handle goalkeeping separately
    if "goalkeeping" in combined and "Pos_Primary" in merged.columns:
        gk_mask = merged["Pos_Primary"] == "GK"
        gk_count = gk_mask.sum()
        print(f"\n[GK MERGE] Found {gk_count} goalkeeper rows")
        
        if gk_count > 0:
            # Merge goalkeeping stats only for GK rows
            gk_data = merged[gk_mask].copy()
            gk_merged = gk_data.merge(
                combined["goalkeeping"],
                on=merge_key,
                how="left",
                suffixes=("", "_gk")
            )
            
            # Replace GK rows in main dataframe
            merged = pd.concat([
                merged[~gk_mask],  # Non-GK rows
                gk_merged           # GK rows with goalkeeping stats
            ], ignore_index=True)
            
            print(f"[GK MERGE] Merged goalkeeping stats for {gk_count} goalkeepers")
    
    return merged


# ============================================================================
# SECTION 6: Output
# ============================================================================

def save_outputs(merged_df: pd.DataFrame, output_dir: str):
    """
    Task 5: Save three output files.
    """
    print("\n" + "="*80)
    print("TASK 5: SAVE OUTPUT FILES")
    print("="*80)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # File 1: All data
    all_path = os.path.join(output_dir, "efl_combined_all.csv")
    merged_df.to_csv(all_path, index=False, encoding="utf-8")
    print(f"\n[SAVED] efl_combined_all.csv")
    print(f"  Rows: {len(merged_df)}, Columns: {merged_df.shape[1]}")
    
    # File 2: Goalkeepers only
    if "Pos_Primary" in merged_df.columns:
        gk_df = merged_df[merged_df["Pos_Primary"] == "GK"].copy()
        gk_path = os.path.join(output_dir, "efl_combined_gk.csv")
        gk_df.to_csv(gk_path, index=False, encoding="utf-8")
        print(f"\n[SAVED] efl_combined_gk.csv")
        print(f"  Rows: {len(gk_df)}, Columns: {gk_df.shape[1]}")
    else:
        print("\n[SKIPPED] efl_combined_gk.csv (Pos_Primary column not found)")
    
    # File 3: Outfield players only
    if "Pos_Primary" in merged_df.columns:
        outfield_df = merged_df[merged_df["Pos_Primary"] != "GK"].copy()
        outfield_path = os.path.join(output_dir, "efl_combined_outfield.csv")
        outfield_df.to_csv(outfield_path, index=False, encoding="utf-8")
        print(f"\n[SAVED] efl_combined_outfield.csv")
        print(f"  Rows: {len(outfield_df)}, Columns: {outfield_df.shape[1]}")
    else:
        print("\n[SKIPPED] efl_combined_outfield.csv (Pos_Primary column not found)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Execute all tasks in order.
    """
    print("\n" + "="*80)
    print("EFL CHAMPIONSHIP DATA INGESTION PIPELINE")
    print("="*80)
    
    base_dir = r"C:\ACL2\FPL\ACL_M3\EFL_Data_All_Seasons"
    output_dir = r"C:\ACL2\FPL\ACL_M3\EFL_Data_All_Seasons"
    
    # Task 1: Inventory
    files_by_season, all_files, invalid_files = discover_files(base_dir)
    
    if not all_files:
        print("\n[CRITICAL ERROR] No valid files found. Stopping.")
        return
    
    # Task 2: Load and validate
    successful_dfs, failed_files = load_and_validate_all(files_by_season)
    
    if not successful_dfs:
        print("\n[CRITICAL ERROR] No files passed validation. Stopping.")
        return
    
    # Task 3: Combine by table type
    combined = combine_by_table_type(successful_dfs)
    
    if not combined:
        print("\n[CRITICAL ERROR] No combined dataframes created. Stopping.")
        return
    
    # Task 4: Merge across table types
    try:
        merged_df = merge_all_tables(combined)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Merge failed: {e}")
        return
    
    if len(merged_df) == 0:
        print("\n[CRITICAL ERROR] Merged dataframe has zero rows. Stopping.")
        return
    
    # Task 5: Save outputs
    save_outputs(merged_df, output_dir)
    
    # Final Summary
    print("\n" + "="*80)
    print("=== INGESTION SUMMARY ===")
    print("="*80)
    print(f"\nFiles attempted:        {len(all_files)}")
    print(f"Files passed validation: {len(successful_dfs)}")
    print(f"Files failed validation: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for filepath, issues in failed_files:
            print(f"  - {os.path.basename(filepath)}")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"      {issue}")
    
    print(f"\nSeasons processed:      {sorted(files_by_season.keys())}")
    print(f"Table types processed:  {list(combined.keys())}")
    
    print("\nOutput files:")
    print(f"  efl_combined_all.csv       → {len(merged_df)} rows, {merged_df.shape[1]} columns")
    
    if "Pos_Primary" in merged_df.columns:
        gk_count = (merged_df["Pos_Primary"] == "GK").sum()
        outfield_count = (merged_df["Pos_Primary"] != "GK").sum()
        print(f"  efl_combined_gk.csv        → {gk_count} rows, {merged_df.shape[1]} columns")
        print(f"  efl_combined_outfield.csv  → {outfield_count} rows, {merged_df.shape[1]} columns")
    
    # Check for warnings
    warnings = []
    
    if invalid_files:
        warnings.append(f"{len(invalid_files)} files had invalid naming or were unreadable")
    
    if failed_files:
        warnings.append(f"{len(failed_files)} files failed validation checks")
    
    # Check for encoding artifacts in final output
    if "Player" in merged_df.columns:
        garbled = merged_df["Player"].str.contains("Ã|Â|â€", na=False, regex=True)
        if garbled.any():
            warnings.append(f"{garbled.sum()} player names still contain encoding artifacts")
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nWarnings: None")
    
    print("\n" + "="*80)
    print("=== END SUMMARY ===")
    print("="*80)


if __name__ == "__main__":
    main()
