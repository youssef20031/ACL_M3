"""
Aggregate EPL team-level statistics from FPL player data.

Creates data/epl_team_stats.csv required for Model A training.
Aggregates goals, clean sheets, and calculates team-level metrics.
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def aggregate_team_stats(epl_data: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Aggregate team-level statistics from player data.
    
    Args:
        epl_data: Combined EPL player data
        season: Season label (e.g., '2023-24')
        
    Returns:
        Team-level statistics dataframe
    """
    logger.info(f"Aggregating team statistics for {season}...")
    
    # Identify team column
    team_col = 'team' if 'team' in epl_data.columns else 'team_x'
    
    # Group by team
    team_stats = []
    
    for team, group in epl_data.groupby(team_col):
        # Count unique gameweeks (matches played)
        gw_col = 'GW' if 'GW' in group.columns else 'round'
        matches = group[gw_col].nunique()
        
        # Sum goals and assists (team attack)
        total_goals = group['goals_scored'].sum()
        total_assists = group['assists'].sum()
        
        # Sum goals conceded (divide by number of matches, not players)
        # Approximate: use defenders' clean sheets
        defenders = group[group['position'] == 'DEF']
        if len(defenders) > 0:
            # Clean sheets: take max (all defenders get same clean sheet)
            clean_sheets = defenders.groupby(gw_col)['clean_sheets'].max().sum()
            goals_conceded = matches - clean_sheets
        else:
            goals_conceded = 0
        
        # Calculate points (rough estimate from results)
        # Wins ≈ clean sheets, draws ≈ goals == goals_conceded, losses ≈ remainder
        wins_estimate = clean_sheets
        losses_estimate = max(0, matches - clean_sheets - (matches - wins_estimate) // 3)
        draws_estimate = matches - wins_estimate - losses_estimate
        points = wins_estimate * 3 + draws_estimate
        
        team_stats.append({
            'season': season,
            'team': team,
            'matches_played': matches,
            'goals_scored': total_goals,
            'goals_conceded': goals_conceded,
            'goal_difference': total_goals - goals_conceded,
            'clean_sheets': clean_sheets,
            'points': points,
            'wins_est': wins_estimate,
            'draws_est': draws_estimate,
            'losses_est': losses_estimate
        })
    
    team_df = pd.DataFrame(team_stats)
    
    # Calculate per-game rates
    team_df['goals_per_game'] = team_df['goals_scored'] / team_df['matches_played']
    team_df['conceded_per_game'] = team_df['goals_conceded'] / team_df['matches_played']
    team_df['points_per_game'] = team_df['points'] / team_df['matches_played']
    
    logger.info(f"Aggregated {len(team_df)} team-season records")
    
    return team_df


def main():
    """Aggregate EPL team stats from player data."""
    logger.info("="*80)
    logger.info("EPL TEAM STATISTICS AGGREGATION")
    logger.info("="*80)
    
    # Load EPL player data
    epl_files = [
        ('data/FPL_2023_2024.csv', '2023-24'),
        ('data/FPL_2024_2025.csv', '2024-25'),
        ('data/FPL_2025_2026.csv', '2025-26')
    ]
    
    all_team_stats = []
    
    for file, season in epl_files:
        if os.path.exists(file):
            logger.info(f"\nProcessing {file} ({season})...")
            df = pd.read_csv(file)
            logger.info(f"  Loaded {len(df):,} player records")
            
            # Aggregate team stats for this season
            team_stats = aggregate_team_stats(df, season)
            all_team_stats.append(team_stats)
    
    if not all_team_stats:
        logger.error("No EPL data files found!")
        return 1
    
    # Combine all seasons
    combined_stats = pd.concat(all_team_stats, ignore_index=True)
    
    # Combine all seasons
    combined_stats = pd.concat(all_team_stats, ignore_index=True)
    
    # Save
    output_path = 'data/epl_team_stats.csv'
    combined_stats.to_csv(output_path, index=False)
    logger.info(f"\n✅ Saved team statistics to {output_path}")
    
    # Display summary
    logger.info("\nSample team stats:")
    logger.info(combined_stats.head(10).to_string())
    
    logger.info(f"\nTeams per season:")
    logger.info(combined_stats.groupby('season')['team'].count().to_string())
    
    return 0


if __name__ == '__main__':
    try:
        exit(main())
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        exit(1)
