"""
Master training script for all promoted teams models.

Runs complete pipeline:
1. Championship data scraping (Phase 1)
2. Minutes models training (Phase 2)
3. Archetype database building (Phase 3)
4. Translation models training (Phases 4-5)
5. Validation (Phase 10)
"""
import os
import sys
import logging
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: str, description: str) -> bool:
    """
    Run shell command and log output.
    
    Args:
        cmd: Command to run
        description: Human-readable description
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"{description}")
    logger.info(f"{'='*80}")
    logger.info(f"Running: {cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Run complete training pipeline."""
    logger.info("\n" + "🚀"*40)
    logger.info("PROMOTED TEAMS MODELS - COMPLETE TRAINING PIPELINE")
    logger.info("🚀"*40 + "\n")
    
    # Check if we should skip data collection (already have CSVs)
    champ_data_exists = os.path.exists('ml/data/championship_player_stats.csv')
    
    steps = []
    
    # Phase 1: Championship data collection
    if not champ_data_exists:
        steps.append((
            "python ml/data/championship_pipeline.py --test",
            "Phase 1: Test Championship scraper"
        ))
        
        logger.info("\n⚠️  Full Championship scraping will take ~2 hours (rate limiting)")
        logger.info("   Run manually: python ml/data/championship_pipeline.py")
        logger.info("   Skipping full scrape for now...\n")
    else:
        logger.info("✅ Championship data already exists, skipping Phase 1")
    
    # Phase 2: Minutes models
    steps.append((
        "python ml/train_minutes_models.py",
        "Phase 2: Train minutes models (Model C & D)"
    ))
    
    # Phase 3: Archetype database
    if champ_data_exists:
        steps.append((
            "python ml/data/build_archetype_database_v2.py",
            "Phase 3: Build archetype database v2"
        ))
    else:
        logger.info("⚠️  Skipping Phase 3 (requires Championship data)")
    
    # Phase 4-5: Translation models
    # Skipping for now - requires manual EPL team stats aggregation
    logger.info("\n⚠️  Phase 4-5 (translation models) require manual setup:")
    logger.info("   1. Create data/epl_team_stats.csv (aggregate from FPL data)")
    logger.info("   2. Run: python ml/translation/model_a_team.py")
    logger.info("   3. Run: python ml/translation/model_b_player.py")
    
    # Run steps
    success_count = 0
    for cmd, description in steps:
        if run_command(cmd, description):
            success_count += 1
        else:
            logger.error(f"\n❌ Pipeline halted at: {description}")
            break
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING PIPELINE SUMMARY")
    logger.info("="*80)
    logger.info(f"Completed: {success_count}/{len(steps)} steps")
    
    if success_count == len(steps):
        logger.info("\n✅ All steps completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Create data/epl_team_stats.csv for Model A training")
        logger.info("2. Train translation models (Phases 4-5)")
        logger.info("3. Run validation: python ml/validation/walk_forward_promoted.py")
        return 0
    else:
        logger.error("\n❌ Some steps failed. Check logs above.")
        return 1


if __name__ == '__main__':
    exit(main())
