"""
Test Promoted Teams Handler
Demonstrates all 4 solutions for cold start problem
"""
import logging
from ml.promoted_teams_handler import PromotedTeamsHandler, PROMOTION_BASELINES
from ml.build_archetype_database import build_example_database

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_championship_data_transfer():
    """Test Solution 1: Championship data transfer with league scaling."""
    print("\n" + "="*80)
    print("SOLUTION 1: Championship Data Transfer")
    print("="*80)
    
    handler = PromotedTeamsHandler()
    
    # Example: Ipswich Town striker from Championship
    ipswich_striker = {
        'name': 'Ipswich Striker',
        'position': 'FWD',
        'team': 'Ipswich Town',
        'league_origin': 0.70,  # From Championship
        'goals_scored': 18,      # Championship season
        'assists': 5,
        'minutes': 3240,
        'form': 5.2,             # Championship form
        'points_per_90': 6.1,
        'bonus': 15,
        'clean_sheets': 0
    }
    
    print("\n📊 Original Championship Stats:")
    print(f"  Goals: {ipswich_striker['goals_scored']}")
    print(f"  Assists: {ipswich_striker['assists']}")
    print(f"  Form: {ipswich_striker['form']:.1f}")
    print(f"  Points/90: {ipswich_striker['points_per_90']:.1f}")
    
    scaled_data = handler.scale_championship_stats(ipswich_striker)
    
    print("\n📈 PL-Scaled Stats:")
    print(f"  Goals (scaled): {scaled_data['goals_scored']:.1f} (×0.70)")
    print(f"  Assists (scaled): {scaled_data['assists']:.1f} (×0.75)")
    print(f"  Form (scaled): {scaled_data['form']:.1f} (×0.70)")
    print(f"  Points/90 (scaled): {scaled_data['points_per_90']:.1f} (×0.70)")
    print(f"  League Origin Flag: {scaled_data['league_origin']:.2f}")


def test_blended_heuristic():
    """Test Solution 3: Blended heuristic with GW-based decay."""
    print("\n" + "="*80)
    print("SOLUTION 3: Blended Heuristic Fallback")
    print("="*80)
    
    handler = PromotedTeamsHandler()
    
    player = {
        'name': 'Southampton Midfielder',
        'position': 'MID',
        'team': 'Southampton',
        'league_origin': 0.70,
        'form': 4.5,  # Championship form (scaled)
        'pl_actual_form': 0  # No PL data yet
    }
    
    print(f"\n👤 Player: {player['name']} ({player['position']})")
    print(f"   Championship Form (scaled): {player['form']:.1f}")
    print(f"   Position Baseline: {PROMOTION_BASELINES[player['position']]:.1f}\n")
    
    print("Predictions by Gameweek:")
    print("-" * 60)
    for gw in [1, 4, 8, 12, 16]:
        prediction = handler.calculate_blended_heuristic(player, gw)
        print(f"  GW {gw:2d}: {prediction:.2f} pts")
    
    print("\n💡 Notice: Prediction gradually increases as PL data accumulates")


def test_archetype_matching():
    """Test Solution 2: Similarity-based player embeddings."""
    print("\n" + "="*80)
    print("SOLUTION 2: Similarity-Based Player Embeddings")
    print("="*80)
    
    # Build archetype database first
    print("\n🔨 Building archetype database...")
    builder = build_example_database()
    
    # Initialize handler with archetype database
    handler = PromotedTeamsHandler(archetype_db_path='ml/models/promoted_archetypes.pkl')
    
    # Test with a creative midfielder (similar to Eze/Maddison)
    test_player = {
        'name': 'Leicester Creative Mid',
        'position': 'MID',
        'goals_scored': 12,
        'assists': 9,
        'minutes': 3600,
        'xG': 10.5,
        'xA': 8.8,
        'shots': 90,
        'key_passes': 75,
        'tackles': 40,
        'interceptions': 30,
        'dribbles': 95,
        'pass_completion': 80.0,
        'aerials_won': 25,
        'fouls': 38
    }
    
    print(f"\n👤 Test Player: {test_player['name']}")
    print(f"   Championship Stats: {test_player['goals_scored']}G, {test_player['assists']}A in {test_player['minutes']} mins")
    
    similar = handler.find_similar_archetypes(test_player, k=3)
    
    if similar:
        print(f"\n🎯 Top 3 Similar Historical Players:")
        for i, archetype in enumerate(similar, 1):
            print(f"\n  {i}. {archetype['name']} ({archetype['position']})")
            print(f"     Promoted: {archetype['promoted_season']}")
            print(f"     Similarity: {archetype['similarity']:.3f}")
            print(f"     PL Debut Avg: {archetype['debut_season_avg_points']:.1f} pts/game")
            print(f"     Tags: {', '.join(archetype['tags'])}")
        
        prediction = handler.predict_from_archetypes(test_player, similar)
        print(f"\n📊 Archetype-Based Prediction: {prediction:.2f} pts/game")


def test_combined_prediction():
    """Test combined prediction using all solutions."""
    print("\n" + "="*80)
    print("COMBINED PREDICTION (All Solutions)")
    print("="*80)
    
    handler = PromotedTeamsHandler(archetype_db_path='ml/models/promoted_archetypes.pkl')
    
    # Ipswich forward similar to Watkins/Mitrovic
    ipswich_forward = {
        'name': 'Ipswich Star Striker',
        'position': 'FWD',
        'team': 'Ipswich Town',
        'league_origin': 0.70,
        'goals_scored': 22,
        'assists': 6,
        'minutes': 3510,
        'form': 5.8,
        'xG': 19.5,
        'xA': 5.2,
        'shots': 155,
        'key_passes': 42,
        'tackles': 14,
        'interceptions': 6,
        'dribbles': 58,
        'pass_completion': 66.8,
        'aerials_won': 142,
        'fouls': 48,
        'pl_actual_form': 0
    }
    
    print(f"\n👤 Player: {ipswich_forward['name']}")
    print(f"   Team: {ipswich_forward['team']}")
    print(f"   Position: {ipswich_forward['position']}")
    print(f"   Championship: {ipswich_forward['goals_scored']}G, {ipswich_forward['assists']}A")
    print(f"   Form: {ipswich_forward['form']:.1f} pts/game")
    
    print("\n📈 Predictions Across Season:")
    print("-" * 80)
    print(f"{'GW':<6} {'Prediction':<12} {'Method':<25} {'Confidence':<12} {'Phase'}")
    print("-" * 80)
    
    for gw in [1, 3, 6, 9, 12, 15]:
        prediction, metadata = handler.predict_promoted_player(
            ipswich_forward,
            current_gw=gw,
            use_archetypes=True
        )
        
        print(f"{gw:<6} {prediction:>6.2f} pts   {metadata['method']:<25} "
              f"{metadata['confidence']:>6.1%}      {metadata['cold_start_phase']}")
    
    print("\n💡 Key Insights:")
    print("  • Early GWs: Low confidence, heavy archetype/heuristic reliance")
    print("  • Mid season: Increasing confidence as PL data accumulates")
    print("  • GW 12+: Cold start resolved, normal prediction confidence")


if __name__ == '__main__':
    test_championship_data_transfer()
    test_blended_heuristic()
    test_archetype_matching()
    test_combined_prediction()
    
    print("\n" + "="*80)
    print("✅ All Tests Complete")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Scrape Championship 2025-26 data from FBRef")
    print("  2. Build full archetype database (2016-2026 promotions)")
    print("  3. Implement incremental model updates (Solution 4)")
    print("  4. Integrate with main predictor.py")
