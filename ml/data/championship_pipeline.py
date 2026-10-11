"""
Championship Data Ingestion Pipeline Orchestrator

Coordinates scraping, cleaning, and merging of Championship data:
1. Scrape player stats from FBref (all categories)
2. Scrape team stats from FBref
3. Clean and normalize data
4. Merge into final CSVs
5. Add promoted team flags and finish positions
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data.fbref_scraper import FBrefScraper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Promoted teams history (2016-2026)
PROMOTED_TEAMS_HISTORY = {
    '2016-2017': ['Newcastle', 'Brighton', 'Huddersfield'],
    '2017-2018': ['Wolverhampton', 'Cardiff', 'Fulham'],
    '2018-2019': ['Norwich', 'Sheffield United', 'Aston Villa'],
    '2019-2020': ['Leeds', 'West Brom', 'Fulham'],
    '2020-2021': ['Norwich', 'Watford', 'Brentford'],
    '2021-2022': ['Fulham', 'Bournemouth', 'Nottingham Forest'],
    '2022-2023': ['Burnley', 'Sheffield United', 'Luton'],
    '2023-2024': ['Leicester', 'Ipswich', 'Southampton'],
    '2024-2025': ['Leeds', 'Burnley', 'Sunderland'],
    '2025-2026': ['Bristol City', 'Ipswich', 'Hull City'],  # Upcoming season
}


class ChampionshipPipeline:
    """
    Orchestrates Championship data ingestion and processing.
    """
    
    def __init__(self, output_dir: str = 'ml/data'):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Directory to save output CSVs
        """
        self.output_dir = output_dir
        self.scraper = FBrefScraper(rate_limit_delay=4.0)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def scrape_all_seasons(self, 
                          start_year: int = 2019,
                          end_year: int = 2026) -> Dict[str, pd.DataFrame]:
        """
        Scrape all Championship seasons.
        
        Args:
            start_year: Starting year (2019 = 2019-2020 season)
            end_year: Ending year (2026 = 2025-2026 season)
            
        Returns:
            Dictionary mapping season -> combined dataframe
        """
        logger.info(f"\nScraping Championship data from {start_year}-{end_year}...")
        
        seasons = self.scraper.get_seasons_list(start_year, end_year)
        all_season_data = {}
        
        for season in seasons:
            logger.info(f"\n{'='*80}")
            logger.info(f"SEASON: {season}")
            logger.info(f"{'='*80}")
            
            try:
                # Scrape all categories for this season
                season_data = self.scraper.scrape_season_all_categories(season)
                
                if season_data:
                    all_season_data[season] = season_data
                    logger.info(f"✅ Scraped {len(season_data)} categories for {season}")
                else:
                    logger.warning(f"⚠️ No data scraped for {season}")
            
            except Exception as e:
                logger.error(f"❌ Failed to scrape {season}: {e}")
        
        logger.info(f"\n✅ Completed scraping {len(all_season_data)} seasons")
        return all_season_data
    
    def merge_player_stats(self, season_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple stat categories into single player dataframe.
        
        Args:
            season_data: Dictionary with category dataframes
            
        Returns:
            Merged dataframe with all player stats
        """
        logger.info("Merging player stat categories...")
        
        # Start with standard stats as base
        if 'stats' not in season_data:
            logger.error("No standard stats found!")
            return pd.DataFrame()
        
        base_df = season_data['stats'].copy()
        
        # Identify player key columns (typically: Player, Squad, Age, etc.)
        # FBref uses 'Player' and 'Squad' as primary identifiers
        merge_keys = []
        for col in ['Player', 'Squad', 'Nation', 'Pos', 'Age']:
            if col in base_df.columns:
                merge_keys.append(col)
        
        if not merge_keys:
            logger.warning("Could not identify merge keys, using index")
            return base_df
        
        logger.info(f"Using merge keys: {merge_keys}")
        
        # Merge other categories
        for category in ['shooting', 'passing', 'defense', 'possession', 'gca', 'playingtime']:
            if category in season_data:
                cat_df = season_data[category]
                
                # Merge
                base_df = base_df.merge(
                    cat_df,
                    on=merge_keys,
                    how='left',
                    suffixes=('', f'_{category}')
                )
                logger.info(f"  Merged {category}: {len(base_df)} rows")
        
        # Handle keepers separately (different columns)
        if 'keepers' in season_data:
            keeper_df = season_data['keepers']
            keeper_keys = [k for k in merge_keys if k in keeper_df.columns]
            if keeper_keys:
                base_df = base_df.merge(
                    keeper_df,
                    on=keeper_keys,
                    how='left',
                    suffixes=('', '_keeper')
                )
                logger.info(f"  Merged keepers: {len(base_df)} rows")
        
        logger.info(f"✅ Merged to {len(base_df)} rows with {len(base_df.columns)} columns")
        return base_df
    
    def clean_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize player data.
        
        Args:
            df: Raw player dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Cleaning player data...")
        
        df = df.copy()
        
        # Standardize column names (lowercase, underscores)
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') 
                     for col in df.columns]
        
        # Map FBref positions to FPL positions
        if 'pos' in df.columns:
            position_map = {
                'GK': 'GK',
                'DF': 'DEF', 'DF,MF': 'DEF', 'DF,FW': 'DEF',
                'MF': 'MID', 'MF,DF': 'MID', 'MF,FW': 'MID',
                'FW': 'FWD', 'FW,MF': 'FWD', 'FW,DF': 'FWD'
            }
            df['position'] = df['pos'].map(lambda x: position_map.get(x, 'MID') if pd.notna(x) else 'MID')
        
        # Convert numeric columns
        numeric_cols = df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            if col not in ['player', 'squad', 'nation', 'pos', 'position', 'season', 'stat_category']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # Fill NaN with 0 for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove duplicate rows
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            logger.info(f"  Removed {before - after} duplicate rows")
        
        logger.info(f"✅ Cleaned data: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def add_promoted_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add promoted team flags and finish positions.
        
        Args:
            df: Player dataframe with season and squad
            
        Returns:
            Dataframe with was_promoted and team_finish_position columns
        """
        logger.info("Adding promoted team flags...")
        
        df = df.copy()
        
        # Initialize columns
        df['was_promoted'] = False
        df['team_finish_position'] = 0
        
        for season, teams in PROMOTED_TEAMS_HISTORY.items():
            # Match season format
            season_mask = df['season'] == season
            
            for i, team in enumerate(teams):
                team_mask = df['squad'].str.contains(team, case=False, na=False)
                df.loc[season_mask & team_mask, 'was_promoted'] = True
                df.loc[season_mask & team_mask, 'team_finish_position'] = i + 1  # 1st, 2nd, 3rd
        
        promoted_count = df['was_promoted'].sum()
        logger.info(f"✅ Flagged {promoted_count} player-season records from promoted teams")
        
        return df
    
    def process_all_seasons(self, 
                           start_year: int = 2019,
                           end_year: int = 2026,
                           scrape: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline: scrape → clean → merge → save.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            scrape: If False, load from cached files
            
        Returns:
            Tuple of (player_df, team_df)
        """
        logger.info("\n" + "="*80)
        logger.info("CHAMPIONSHIP DATA PIPELINE")
        logger.info("="*80)
        
        all_players = []
        all_teams = []
        
        if scrape:
            # Scrape all seasons
            all_season_data = self.scrape_all_seasons(start_year, end_year)
            
            # Process each season
            for season, season_data in all_season_data.items():
                logger.info(f"\nProcessing {season}...")
                
                # Merge and clean player stats
                player_df = self.merge_player_stats(season_data)
                if not player_df.empty:
                    player_df = self.clean_player_data(player_df)
                    all_players.append(player_df)
                
                # Extract team stats
                if 'team_stats' in season_data:
                    team_df = season_data['team_stats'].copy()
                    team_df = self.clean_player_data(team_df)  # Same cleaning
                    all_teams.append(team_df)
        
        # Combine all seasons
        if all_players:
            logger.info("\nCombining all seasons...")
            combined_players = pd.concat(all_players, ignore_index=True)
            combined_players = self.add_promoted_flags(combined_players)
        else:
            combined_players = pd.DataFrame()
        
        if all_teams:
            combined_teams = pd.concat(all_teams, ignore_index=True)
        else:
            combined_teams = pd.DataFrame()
        
        # Save outputs
        player_output = os.path.join(self.output_dir, 'championship_player_stats.csv')
        team_output = os.path.join(self.output_dir, 'championship_team_stats.csv')
        
        if not combined_players.empty:
            combined_players.to_csv(player_output, index=False)
            logger.info(f"\n✅ Saved player stats: {player_output}")
            logger.info(f"   {len(combined_players):,} rows × {len(combined_players.columns)} columns")
        
        if not combined_teams.empty:
            combined_teams.to_csv(team_output, index=False)
            logger.info(f"✅ Saved team stats: {team_output}")
            logger.info(f"   {len(combined_teams):,} rows × {len(combined_teams.columns)} columns")
        
        # Save promoted teams history
        history_output = os.path.join(self.output_dir, 'promoted_teams_history.json')
        with open(history_output, 'w') as f:
            json.dump(PROMOTED_TEAMS_HISTORY, f, indent=2)
        logger.info(f"✅ Saved promoted teams history: {history_output}")
        
        return combined_players, combined_teams


def main():
    """Run Championship data pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Championship data ingestion pipeline')
    parser.add_argument('--start-year', type=int, default=2019, help='Start year (default: 2019)')
    parser.add_argument('--end-year', type=int, default=2026, help='End year (default: 2026)')
    parser.add_argument('--no-scrape', action='store_true', help='Skip scraping, use cached data')
    parser.add_argument('--test', action='store_true', help='Test mode: scrape only one season')
    
    args = parser.parse_args()
    
    pipeline = ChampionshipPipeline(output_dir='ml/data')
    
    if args.test:
        logger.info("TEST MODE: Scraping 2023-2024 season only")
        scraper = FBrefScraper()
        data = scraper.scrape_season_all_categories('2023-2024')
        logger.info(f"✅ Test scrape successful: {len(data)} categories")
        return 0
    
    try:
        player_df, team_df = pipeline.process_all_seasons(
            start_year=args.start_year,
            end_year=args.end_year,
            scrape=not args.no_scrape
        )
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Player records: {len(player_df):,}")
        logger.info(f"Team records: {len(team_df):,}")
        
        return 0
    
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
