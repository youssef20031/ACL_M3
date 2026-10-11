"""
Phase 10: Walk-Forward Validation for Promoted Teams
Historical validation of promoted team predictions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def load_archetypes():
    """Load archetype database"""
    arch_path = Path('C:/ACL2/FPL/ACL_M3/ml/models/promoted_archetypes_v2.pkl')
    with open(arch_path, 'rb') as f:
        archetypes = pickle.load(f)
    return pd.DataFrame(archetypes)


def calculate_baseline_predictions(df, position):
    """Calculate baseline predictions (simple position average)"""
    pos_df = df[df['position'] == position]
    if len(pos_df) == 0:
        return {'goals': 0, 'assists': 0, 'points': 0}
    
    return {
        'goals': pos_df['epl_goals_per90'].mean(),
        'assists': pos_df['epl_assists_per90'].mean(), 
        'points': pos_df['epl_avg_fpl_points'].mean(),
    }


def walk_forward_validation(df):
    """
    Walk-forward validation through historical promoted teams
    """
    print("="*80)
    print("WALK-FORWARD VALIDATION: PROMOTED TEAMS")
    print("="*80)
    
    seasons = sorted(df['promoted_season'].unique())
    print(f"\nSeasons: {seasons}")
    print(f"Total players: {len(df)}\n")
    
    results_by_season = []
    
    for i, test_season in enumerate(seasons):
        print(f"\n{'='*60}")
        print(f"TEST SEASON: {test_season}")
        print(f"{'='*60}")
        
        # Train on all previous seasons
        if i == 0:
            print("  Skipping first season (no training data)")
            continue
        
        train_seasons = seasons[:i]
        train_df = df[df['promoted_season'].isin(train_seasons)]
        test_df = df[df['promoted_season'] == test_season]
        
        print(f"  Train seasons: {train_seasons}")
        print(f"  Train players: {len(train_df)}")
        print(f"  Test players: {len(test_df)}")
        
        # For each position, calculate baseline from training data
        position_results = {}
        
        for position in ['DEF', 'MID', 'FWD', 'GK']:
            test_pos = test_df[test_df['position'] == position]
            if len(test_pos) == 0:
                continue
            
            # Baseline: use training data average for this position
            baseline = calculate_baseline_predictions(train_df, position)
            
            # Actual values
            actual_goals = test_pos['epl_goals_per90'].values
            actual_assists = test_pos['epl_assists_per90'].values
            actual_points = test_pos['epl_avg_fpl_points'].values
            
            # Predictions (baseline)
            pred_goals = np.full(len(test_pos), baseline['goals'])
            pred_assists = np.full(len(test_pos), baseline['assists'])
            pred_points = np.full(len(test_pos), baseline['points'])
            
            # Calculate MAE
            mae_goals = mean_absolute_error(actual_goals, pred_goals)
            mae_assists = mean_absolute_error(actual_assists, pred_assists)
            mae_points = mean_absolute_error(actual_points, pred_points)
            
            position_results[position] = {
                'n_players': len(test_pos),
                'mae_goals': mae_goals,
                'mae_assists': mae_assists,
                'mae_points': mae_points,
            }
            
            print(f"\n  {position} (n={len(test_pos)}):")
            print(f"    MAE Goals/90:   {mae_goals:.3f}")
            print(f"    MAE Assists/90: {mae_assists:.3f}")
            print(f"    MAE Pts/game:   {mae_points:.3f}")
        
        # Overall season metrics
        if position_results:
            weighted_mae_points = np.average(
                [r['mae_points'] for r in position_results.values()],
                weights=[r['n_players'] for r in position_results.values()]
            )
            
            results_by_season.append({
                'season': test_season,
                'n_players': len(test_df),
                'weighted_mae_points': weighted_mae_points,
                'position_results': position_results,
            })
            
            print(f"\n  Overall MAE Points: {weighted_mae_points:.3f}")
    
    # Summary across all seasons
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if results_by_season:
        avg_mae = np.mean([r['weighted_mae_points'] for r in results_by_season])
        print(f"\nAverage MAE across seasons: {avg_mae:.3f} pts/game")
        
        print("\nBy Season:")
        for result in results_by_season:
            print(f"  {result['season']}: {result['weighted_mae_points']:.3f} pts (n={result['n_players']})")
        
        # By position aggregate
        print("\nBy Position (aggregate):")
        all_positions = {}
        for result in results_by_season:
            for pos, metrics in result['position_results'].items():
                if pos not in all_positions:
                    all_positions[pos] = []
                all_positions[pos].append(metrics['mae_points'])
        
        for pos, mae_list in sorted(all_positions.items()):
            avg_mae_pos = np.mean(mae_list)
            print(f"  {pos}: {avg_mae_pos:.3f} pts")
        
        # Assessment
        print("\n" + "="*80)
        print("ASSESSMENT")
        print("="*80)
        if avg_mae < 1.5:
            print("[OK] MAE < 1.5 pts - Good performance")
        elif avg_mae < 2.0:
            print("[WARN] MAE 1.5-2.0 pts - Acceptable but could improve")
        else:
            print("[WARN] MAE > 2.0 pts - Needs improvement")
    
    return results_by_season


def main():
    print("="*80)
    print("PHASE 10: WALK-FORWARD VALIDATION")
    print("="*80)
    
    df = load_archetypes()
    results = walk_forward_validation(df)
    
    # Save results
    output_path = Path('C:/ACL2/FPL/ACL_M3/ml/validation/walk_forward_results.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n[OK] Results saved to {output_path}")
    print("\n[OK] PHASE 10 COMPLETE")


if __name__ == '__main__':
    main()
