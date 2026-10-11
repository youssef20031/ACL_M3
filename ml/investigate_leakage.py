"""
Investigate potential data leakage in minutes models.
High AUC (0.99-1.00) suggests possible leakage.

Checks:
1. Feature importance rankings
2. Feature correlation with target
3. Feature value distributions
4. Walk-forward validation
5. Exact feature list used
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.minutes_model import MinutesPredictor
from ml.train_minutes_models import load_fpl_data, temporal_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_features(position='FWD'):
    """Load a trained model and extract feature names."""
    model_path = f'ml/models/minutes_pstart_{position.lower()}_v1.pkl'
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None, None
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    return model, feature_cols


def get_feature_importance(model, feature_cols):
    """Extract feature importance from XGBoost model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        logger.warning("Model doesn't have feature_importances_")
        return None
    
    # Create dataframe of feature importances
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feat_imp


def check_feature_correlations(df, feature_cols, target_col='started'):
    """Calculate correlation between features and target."""
    correlations = []
    
    for feat in feature_cols:
        if feat in df.columns:
            corr = df[[feat, target_col]].corr().iloc[0, 1]
            correlations.append({
                'feature': feat,
                'correlation': abs(corr)
            })
    
    corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    return corr_df


def walk_forward_validation(predictor, df, position, n_splits=5):
    """
    Perform walk-forward validation to check if AUC is consistent.
    If leakage exists, AUC should be much higher on training than test.
    """
    df_pos = df[df['position'] == position].copy()
    df_pos = df_pos.sort_values('kickoff_time').reset_index(drop=True)
    
    n = len(df_pos)
    split_size = n // (n_splits + 1)
    
    aucs = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Walk-Forward Validation: {position}")
    logger.info(f"{'='*60}")
    
    for i in range(n_splits):
        train_end = (i + 1) * split_size
        test_start = train_end
        test_end = min(test_start + split_size, n)
        
        train_fold = df_pos.iloc[:train_end].copy()
        test_fold = df_pos.iloc[test_start:test_end].copy()
        
        if len(test_fold) < 100:
            continue
        
        # Train mini model on this fold
        temp_predictor = MinutesPredictor()
        temp_predictor.train(
            train_df=train_fold,
            val_df=None,
            save_models=False
        )
        
        # Predict on test fold
        test_fold = temp_predictor._create_started_feature(test_fold)
        test_fold = temp_predictor._add_minutes_features(test_fold)
        test_fold = temp_predictor.feature_engineer.engineer_features(
            test_fold, is_training=False
        )
        
        # Get features
        model_info = temp_predictor.pstart_models.get(position)
        if model_info is None:
            continue
        
        feature_cols = model_info['feature_cols']
        X_test = test_fold[feature_cols].fillna(0)
        y_test = test_fold['started']
        
        # Predict
        y_pred = model_info['model'].predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        aucs.append(auc)
        
        logger.info(f"  Fold {i+1}: Train size={len(train_fold)}, Test size={len(test_fold)}, AUC={auc:.4f}")
    
    logger.info(f"\n  Mean AUC across folds: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    logger.info(f"  Min AUC: {np.min(aucs):.4f}, Max AUC: {np.max(aucs):.4f}")
    
    return aucs


def main():
    """Main investigation pipeline."""
    logger.info("="*80)
    logger.info("INVESTIGATING POTENTIAL DATA LEAKAGE")
    logger.info("="*80)
    
    # Load data
    logger.info("\n1. Loading FPL data...")
    df = load_fpl_data()
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    df = df.dropna(subset=['minutes', 'position'])
    
    if 'season_x' not in df.columns and 'season' in df.columns:
        df['season_x'] = df['season']
    
    train_df, val_df, test_df = temporal_split(df)
    
    # Check each position
    positions = ['GK', 'FWD', 'MID', 'DEF']
    
    for position in positions:
        logger.info("\n" + "="*80)
        logger.info(f"POSITION: {position}")
        logger.info("="*80)
        
        # Load model
        logger.info(f"\n2. Loading trained {position} model...")
        model, feature_cols = load_model_and_features(position)
        
        if model is None:
            continue
        
        logger.info(f"   Model has {len(feature_cols)} features")
        
        # Feature importance
        logger.info(f"\n3. Feature Importance (Top 20):")
        feat_imp = get_feature_importance(model, feature_cols)
        
        if feat_imp is not None:
            print(feat_imp.head(20).to_string(index=False))
            
            # Check for suspicious features
            top_features = feat_imp.head(10)['feature'].tolist()
            suspicious = []
            
            for feat in top_features:
                # Check if feature name contains current-GW keywords
                keywords = ['minutes', 'goals', 'assists', 'points', 'bonus', 
                           'clean_sheet', 'saves', 'cards']
                
                # But exclude safe rolling/lagged features
                safe_keywords = ['rolling', 'last', 'avg', 'form', 'home_form', 
                                'away_form', 'per_90', 'strength']
                
                is_suspicious = any(kw in feat.lower() for kw in keywords)
                is_safe = any(kw in feat.lower() for kw in safe_keywords)
                
                if is_suspicious and not is_safe:
                    suspicious.append(feat)
            
            if suspicious:
                logger.warning(f"\n   ⚠️  POTENTIALLY SUSPICIOUS FEATURES: {suspicious}")
            else:
                logger.info(f"\n   ✅ No obviously suspicious features in top 10")
        
        # Feature correlations with target
        logger.info(f"\n4. Feature Correlations with 'started' (Top 20):")
        
        # Prepare data for correlation check
        predictor = MinutesPredictor()
        train_pos = train_df[train_df['position'] == position].copy()
        train_pos = predictor._create_started_feature(train_pos)
        train_pos = predictor._add_minutes_features(train_pos)
        train_pos = predictor.feature_engineer.engineer_features(
            train_pos, is_training=True
        )
        
        corr_df = check_feature_correlations(train_pos, feature_cols)
        print(corr_df.head(20).to_string(index=False))
        
        # Check for suspiciously high correlations
        high_corr = corr_df[corr_df['correlation'] > 0.8]
        if len(high_corr) > 0:
            logger.warning(f"\n   ⚠️  VERY HIGH CORRELATIONS (>0.8):")
            print(high_corr.to_string(index=False))
        else:
            logger.info(f"\n   ✅ No suspiciously high correlations (>0.8)")
        
        # Target distribution
        logger.info(f"\n5. Target Distribution:")
        start_rate = train_pos['started'].mean()
        logger.info(f"   Positive class rate (started=1): {start_rate:.2%}")
        logger.info(f"   Total samples: {len(train_pos):,}")
        
        # Walk-forward validation (only for one position to save time)
        if position == 'FWD':
            logger.info(f"\n6. Walk-Forward Validation (checking temporal stability):")
            aucs = walk_forward_validation(predictor, df, position, n_splits=3)
            
            # Check if AUC is consistently high
            if all(auc > 0.95 for auc in aucs):
                logger.info(f"\n   ⚠️  AUC consistently >0.95 across all folds")
                logger.info(f"   This suggests either:")
                logger.info(f"   - Very predictable target (e.g., GK always starts)")
                logger.info(f"   - Strong legitimate features (form, recent starts)")
                logger.info(f"   - OR potential leakage")
            else:
                logger.info(f"\n   ✅ AUC varies across folds (normal)")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("INVESTIGATION COMPLETE")
    logger.info("="*80)
    
    logger.info("\nKey Questions to Answer:")
    logger.info("1. Are the top features legitimate lagged/historical features?")
    logger.info("2. Are any features derived from current match outcomes?")
    logger.info("3. Is the high AUC due to easy targets (e.g., GK always starts)?")
    logger.info("4. Does walk-forward validation show consistent performance?")
    
    logger.info("\nNext Steps:")
    logger.info("- Review top features for each position")
    logger.info("- Check if 'started' is easy to predict (e.g., GK rotation is rare)")
    logger.info("- Verify all rolling features use .shift(1)")
    logger.info("- Consider if AUC 0.99 is realistic for this problem")


if __name__ == '__main__':
    main()
