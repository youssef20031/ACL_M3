"""
Update the 2025-26 season data in cleaned_merged_seasons_cleaned.csv
with the newly added GW 30-38 data from FPL_2025_2026.csv
"""
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def update_season_data():
    """Remove old 2025-26 data and replace with updated version"""
    
    # 1. Load master file
    logger.info("Step 1: Loading master CSV...")
    master_df = pd.read_csv('cleaned_merged_seasons_cleaned.csv')
    logger.info(f"  Master has {len(master_df):,} rows")
    logger.info(f"  Seasons: {sorted(master_df['season_x'].unique())}")
    
    # Check 2025-26 data before removal
    season_2025_26_old = master_df[master_df['season_x'] == '2025-26']
    logger.info(f"  Current 2025-26 data: {len(season_2025_26_old):,} rows")
    if len(season_2025_26_old) > 0:
        logger.info(f"    GWs: {sorted(season_2025_26_old['GW'].unique())}")
        logger.info(f"    Max GW: {season_2025_26_old['GW'].max()}")
    
    # 2. Remove old 2025-26 data
    logger.info("\nStep 2: Removing old 2025-26 data...")
    master_without_2025_26 = master_df[master_df['season_x'] != '2025-26'].copy()
    logger.info(f"  After removal: {len(master_without_2025_26):,} rows")
    
    # 3. Load updated 2025-26 data
    logger.info("\nStep 3: Loading updated FPL_2025_2026.csv...")
    new_2025_26 = pd.read_csv('FPL_2025_2026.csv')
    logger.info(f"  New 2025-26 data: {len(new_2025_26):,} rows")
    logger.info(f"    GWs: {sorted(new_2025_26['GW'].unique())}")
    logger.info(f"    Max GW: {new_2025_26['GW'].max()}")
    
    # 4. Normalize the new data to match master schema
    logger.info("\nStep 4: Normalizing new 2025-26 data...")
    
    # Add season_x column
    new_2025_26['season_x'] = '2025-26'
    
    # Rename team to team_x if needed
    if 'team' in new_2025_26.columns and 'team_x' not in new_2025_26.columns:
        new_2025_26 = new_2025_26.rename(columns={'team': 'team_x'})
    
    # Build opp_team_name from fixtures
    logger.info("  Building opp_team_name...")
    fixture_teams = new_2025_26.groupby("fixture")["team_x"].apply(lambda s: s.unique().tolist()).to_dict()
    
    def get_opponent(row):
        teams = fixture_teams.get(row["fixture"], [])
        others = [t for t in teams if t != row["team_x"]]
        return others[0] if others else None
    
    new_2025_26["opp_team_name"] = new_2025_26.apply(get_opponent, axis=1)
    
    # Add opponent_team as None (numeric ID not available)
    if "opponent_team" not in new_2025_26.columns:
        new_2025_26["opponent_team"] = None
    
    # Add round column (same as GW)
    if "round" not in new_2025_26.columns and "GW" in new_2025_26.columns:
        new_2025_26["round"] = new_2025_26["GW"]
    
    # Ensure all master columns exist (fill with None if missing)
    master_columns = master_df.columns.tolist()
    for col in master_columns:
        if col not in new_2025_26.columns:
            new_2025_26[col] = None
    
    # Keep only master columns in same order
    new_2025_26 = new_2025_26[master_columns]
    
    logger.info(f"  Normalized to {len(new_2025_26.columns)} columns")
    
    # 5. Combine
    logger.info("\nStep 5: Combining data...")
    combined_df = pd.concat([master_without_2025_26, new_2025_26], ignore_index=True)
    logger.info(f"  Combined: {len(combined_df):,} rows")
    logger.info(f"  Seasons: {sorted(combined_df['season_x'].unique())}")
    
    # Verify 2025-26 data
    season_2025_26_new = combined_df[combined_df['season_x'] == '2025-26']
    logger.info(f"\nVerification - New 2025-26 data:")
    logger.info(f"  Rows: {len(season_2025_26_new):,}")
    logger.info(f"  GWs: {sorted(season_2025_26_new['GW'].unique())}")
    logger.info(f"  Max GW: {season_2025_26_new['GW'].max()}")
    logger.info(f"  Players: {season_2025_26_new['name'].nunique()}")
    
    # 6. Save
    logger.info("\nStep 6: Saving updated master CSV...")
    combined_df.to_csv('cleaned_merged_seasons_cleaned.csv', index=False)
    logger.info("  ✅ Saved to cleaned_merged_seasons_cleaned.csv")
    
    # 7. Also update merged_recent_seasons.csv (if it exists)
    try:
        logger.info("\nStep 7: Updating merged_recent_seasons.csv...")
        recent_df = pd.read_csv('merged_recent_seasons.csv', low_memory=False)
        
        # Check which season column name is used
        season_col = 'season_x' if 'season_x' in recent_df.columns else 'season'
        
        if season_col in recent_df.columns:
            recent_without_2025_26 = recent_df[recent_df[season_col] != '2025-26'].copy()
            
            # Make sure new data has the same season column name
            if season_col == 'season' and season_col not in new_2025_26.columns:
                new_2025_26_recent = new_2025_26.copy()
                new_2025_26_recent['season'] = new_2025_26_recent['season_x']
            else:
                new_2025_26_recent = new_2025_26
            
            recent_combined = pd.concat([recent_without_2025_26, new_2025_26_recent], ignore_index=True)
            recent_combined.to_csv('merged_recent_seasons.csv', index=False)
            logger.info("  ✅ Updated merged_recent_seasons.csv")
        else:
            logger.info("  merged_recent_seasons.csv has unexpected structure, skipping")
    except FileNotFoundError:
        logger.info("  merged_recent_seasons.csv not found, skipping")
    except Exception as e:
        logger.warning(f"  Could not update merged_recent_seasons.csv: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ UPDATE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total rows: {len(combined_df):,}")
    logger.info(f"2025-26 rows: {len(season_2025_26_new):,}")
    logger.info(f"GW range for 2025-26: 1-{season_2025_26_new['GW'].max()}")

if __name__ == "__main__":
    update_season_data()
