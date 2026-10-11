"""
Deployment script for 2026-27 promoted teams.

Sets up predictions for: Bristol City, Ipswich Town, Hull City

Steps:
1. Scrape 2025-26 Championship data for these teams
2. Run Model A to predict EPL strength
3. Add key players to archetype DB
4. Configure promoted teams handler
"""
import os
import sys
import logging
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROMOTED_TEAMS_2026 = {
    'Bristol City': {'championship_finish': 1, 'key_players': []},
    'Ipswich Town': {'championship_finish': 2, 'key_players': []},
    'Hull City': {'championship_finish': 3, 'key_players': []}
}


def scrape_championship_data():
    """Scrape 2025-26 Championship data for promoted teams."""
    logger.info("Step 1: Scraping 2025-26 Championship data...")
    
    from ml.data.fbref_scraper import FBrefScraper
    
    scraper = FBrefScraper()
    
    season = '2025-2026'
    season_data = scraper.scrape_season_all_categories(season)
    
    if not season_data:
        logger.error("Failed to scrape Championship data")
        return None
    
    logger.info(f"✅ Scraped {len(season_data)} categories for {season}")
    
    # Filter to promoted teams
    player_stats = season_data.get('stats')
    if player_stats is not None:
        for team in PROMOTED_TEAMS_2026.keys():
            team_players = player_stats[
                player_stats['Squad'].str.contains(team, case=False, na=False)
            ]
            logger.info(f"  {team}: {len(team_players)} players found")
    
    return season_data


def predict_team_strength():
    """Run Model A to predict EPL strength for promoted teams."""
    logger.info("\nStep 2: Predicting EPL team strength...")
    
    from ml.translation.model_a_team import TeamTranslationModel
    
    # Load Model A
    model = TeamTranslationModel()
    
    if not os.path.exists('ml/models/translation_model_a_v1.pkl'):
        logger.warning("Model A not trained yet. Run: python ml/translation/model_a_team.py")
        return
    
    model.load()
    
    # Predict for each team (would need actual Championship stats)
    for team, info in PROMOTED_TEAMS_2026.items():
        # Placeholder - would use actual scraped stats
        team_stats = {
            'xg': 60,
            'xga': 45,
            'possession_pct': 52,
            'points': 85 if info['championship_finish'] == 1 else 80,
            'finish_position': info['championship_finish']
        }
        
        try:
            prediction = model.predict(team_stats)
            logger.info(f"\n{team}:")
            logger.info(f"  Attack strength: {prediction['epl_attack_strength']:.2f}")
            logger.info(f"  Defense strength: {prediction['epl_defensive_strength']:.2f}")
            logger.info(f"  Points/game: {prediction['epl_team_ppg']:.2f}")
            
            info['predicted_strength'] = prediction
        
        except Exception as e:
            logger.error(f"Failed to predict for {team}: {e}")


def configure_handler():
    """Configure promoted teams handler for 2026-27."""
    logger.info("\nStep 3: Configuring promoted teams handler...")
    
    config = {
        'season': '2026-2027',
        'promoted_teams': list(PROMOTED_TEAMS_2026.keys()),
        'team_info': PROMOTED_TEAMS_2026
    }
    
    config_path = 'ml/config/promoted_teams_2026.json'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Saved configuration to {config_path}")


def main():
    """Run deployment setup."""
    logger.info("="*80)
    logger.info("PROMOTED TEAMS 2026-27 DEPLOYMENT")
    logger.info("="*80)
    logger.info("\nPromoted Teams:")
    for team, info in PROMOTED_TEAMS_2026.items():
        logger.info(f"  {info['championship_finish']}. {team}")
    
    # Step 1: Scrape Championship data
    logger.info("\n⚠️  Step 1 (scraping) commented out to avoid rate limiting")
    logger.info("   Run manually if needed: uncomment scrape_championship_data() call")
    # season_data = scrape_championship_data()
    
    # Step 2: Predict team strength
    logger.info("\n⚠️  Step 2 requires trained Model A")
    logger.info("   Train first: python ml/translation/model_a_team.py")
    # predict_team_strength()
    
    # Step 3: Configure handler
    configure_handler()
    
    logger.info("\n" + "="*80)
    logger.info("DEPLOYMENT CHECKLIST")
    logger.info("="*80)
    logger.info("[ ] 1. Scrape 2025-26 Championship data")
    logger.info("[ ] 2. Train Model A (team translation)")
    logger.info("[ ] 3. Train Model B (player translation)")
    logger.info("[✅] 4. Configure promoted teams handler")
    logger.info("[ ] 5. Update api_main.py to load v2 handler")
    logger.info("[ ] 6. Test predictions for sample players")
    logger.info("[ ] 7. Monitor first 6 gameweeks, adjust if needed")
    
    return 0


if __name__ == '__main__':
    exit(main())
