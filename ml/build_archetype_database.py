"""
Build Historical Archetype Database for Promoted Players
Creates embeddings from historical Championship->PL promotions (2016-2026)
"""
import pandas as pd
import logging
from ml.promoted_teams_handler import ArchetypeDatabaseBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_example_database():
    """
    Build example archetype database with known successful promotions.
    
    This is a starter database with manually curated examples.
    In production, this would be built from FBRef/Opta data.
    """
    builder = ArchetypeDatabaseBuilder()
    
    # Example 1: Ollie Watkins (Brentford 2019-20 -> Aston Villa 2020-21)
    builder.add_archetype(
        name='Ollie Watkins',
        position='FWD',
        promoted_season='2020-21',
        championship_stats={
            'goals_scored': 25,  # Championship season
            'assists': 7,
            'minutes': 3420,
            'xG': 22.1,
            'xA': 6.3,
            'shots': 140,
            'key_passes': 45,
            'tackles': 15,
            'interceptions': 8,
            'dribbles': 85,
            'pass_completion': 68.5,
            'aerials_won': 95,
            'fouls': 38
        },
        debut_season_avg_points=4.8,  # PL 2020-21 average
        tags=['clinical_finisher', 'high_work_rate', 'striker']
    )
    
    # Example 2: Eberechi Eze (QPR 2019-20 -> Crystal Palace 2020-21)
    builder.add_archetype(
        name='Eberechi Eze',
        position='MID',
        promoted_season='2020-21',
        championship_stats={
            'goals_scored': 14,
            'assists': 8,
            'minutes': 3780,
            'xG': 11.2,
            'xA': 9.1,
            'shots': 95,
            'key_passes': 78,
            'tackles': 42,
            'interceptions': 25,
            'dribbles': 145,
            'pass_completion': 78.2,
            'aerials_won': 32,
            'fouls': 45
        },
        debut_season_avg_points=5.1,
        tags=['dribbler', 'set_piece_specialist', 'creative']
    )
    
    # Example 3: James Maddison (Norwich 2017-18 -> Leicester 2018-19)
    builder.add_archetype(
        name='James Maddison',
        position='MID',
        promoted_season='2018-19',
        championship_stats={
            'goals_scored': 15,
            'assists': 11,
            'minutes': 3960,
            'xG': 12.8,
            'xA': 11.5,
            'shots': 105,
            'key_passes': 92,
            'tackles': 38,
            'interceptions': 28,
            'dribbles': 88,
            'pass_completion': 82.1,
            'aerials_won': 18,
            'fouls': 32
        },
        debut_season_avg_points=5.4,
        tags=['creative', 'free_kick_specialist', 'playmaker']
    )
    
    # Example 4: Aleksandar Mitrovic (Fulham 2022-23)
    builder.add_archetype(
        name='Aleksandar Mitrovic',
        position='FWD',
        promoted_season='2022-23',
        championship_stats={
            'goals_scored': 43,  # Record-breaking Championship season
            'assists': 7,
            'minutes': 3690,
            'xG': 31.5,
            'xA': 5.8,
            'shots': 185,
            'key_passes': 38,
            'tackles': 12,
            'interceptions': 5,
            'dribbles': 42,
            'pass_completion': 65.3,
            'aerials_won': 215,  # Dominant in the air
            'fouls': 58
        },
        debut_season_avg_points=4.2,
        tags=['target_man', 'aerial_threat', 'penalty_taker']
    )
    
    # Example 5: Ben White (Brighton 2019-20 - via loan, then permanent)
    builder.add_archetype(
        name='Ben White',
        position='DEF',
        promoted_season='2019-20',
        championship_stats={
            'goals_scored': 1,
            'assists': 1,
            'minutes': 4050,
            'xG': 0.8,
            'xA': 2.1,
            'shots': 12,
            'key_passes': 18,
            'tackles': 78,
            'interceptions': 92,
            'dribbles': 35,
            'pass_completion': 88.5,
            'aerials_won': 125,
            'fouls': 22
        },
        debut_season_avg_points=4.1,
        tags=['ball_playing_defender', 'composed', 'clean_sheet_merchant']
    )
    
    # Example 6: John Lundstram (Sheffield United 2019-20)
    builder.add_archetype(
        name='John Lundstram',
        position='MID',
        promoted_season='2019-20',
        championship_stats={
            'goals_scored': 5,
            'assists': 3,
            'minutes': 2970,
            'xG': 4.2,
            'xA': 3.8,
            'shots': 48,
            'key_passes': 32,
            'tackles': 88,
            'interceptions': 65,
            'dribbles': 22,
            'pass_completion': 79.2,
            'aerials_won': 58,
            'fouls': 42
        },
        debut_season_avg_points=5.8,  # Breakout season (OOP as DEF)
        tags=['box_to_box', 'defensive_midfielder', 'surprise_attacker']
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("Built Example Archetype Database")
    logger.info(f"{'='*60}")
    logger.info(f"Total archetypes: {len(builder.archetypes)}")
    
    # Save to file
    output_path = 'ml/models/promoted_archetypes.pkl'
    builder.save(output_path)
    
    logger.info(f"\n✅ Saved archetype database to {output_path}")
    logger.info("\nTo use in production:")
    logger.info("  handler = PromotedTeamsHandler(archetype_db_path='ml/models/promoted_archetypes.pkl')")
    
    return builder


if __name__ == '__main__':
    build_example_database()
