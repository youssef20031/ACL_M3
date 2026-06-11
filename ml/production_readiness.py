"""
Production Readiness Tasks for FPL ML Models
Implements Priority 1 tasks: calibration, start probability, DGW features
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, classification_report, r2_score, mean_absolute_error
import pickle
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionReadiness:
    """
    Implements production readiness tasks for FPL ML models.
    """
    
    def __init__(self, models_dir: str = "ml/models"):
        """Initialize with models directory."""
        self.models_dir = Path(models_dir)
        self.position_models = {}
        self.start_model = None
        
    # ============================================================================
    # TASK 1: CALIBRATION ANALYSIS
    # ============================================================================
    
    def calibration_analysis(
        self, 
        y_test: np.ndarray, 
        y_pred: np.ndarray, 
        position: str, 
        n_bins: int = 10,
        save_plot: bool = True
    ):
        """
        Analyze calibration of predictions vs actual outcomes.
        
        Most FPL-relevant: Are high predictions (captain picks) trustworthy?
        
        Args:
            y_test: Actual points
            y_pred: Predicted points
            position: Position name (GK/DEF/MID/FWD)
            n_bins: Number of bins for calibration curve
            save_plot: Whether to save plot to file
            
        Returns:
            Dict with calibration metrics
        """
        logger.info(f"Running calibration analysis for {position}...")
        
        # Bin predictions into percentiles
        bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(y_pred, bins[1:-1])
        
        mean_predicted, mean_actual, counts = [], [], []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                mean_predicted.append(y_pred[mask].mean())
                mean_actual.append(y_test[mask].mean())
                counts.append(mask.sum())
        
        # Plot calibration
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Calibration curve
        ax1.plot(mean_predicted, mean_actual, 'o-', linewidth=2, markersize=8, label='Model')
        ax1.plot([0, max(mean_predicted)], [0, max(mean_predicted)],
                'r--', linewidth=2, label='Perfect calibration')
        ax1.set_xlabel('Mean Predicted Points', fontsize=12)
        ax1.set_ylabel('Mean Actual Points', fontsize=12)
        ax1.set_title(f'{position} Calibration Curve', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        
        # Bias by range (most actionable for FPL)
        bias = np.array(mean_actual) - np.array(mean_predicted)
        colors = ['red' if b < 0 else 'green' for b in bias]
        ax2.bar(range(n_bins), bias, color=colors, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='--', linewidth=2)
        ax2.set_xlabel('Prediction Decile (low → high)', fontsize=12)
        ax2.set_ylabel('Actual - Predicted (bias)', fontsize=12)
        ax2.set_title(f'{position} Prediction Bias by Range', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.models_dir / f'calibration_{position.lower()}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved calibration plot to {plot_path}")
        
        plt.close()
        
        # Key FPL insight: are high predictions trustworthy?
        # This affects captain picks (highest predicted points)
        high_pred_mask = y_pred >= np.percentile(y_pred, 80)
        top20_predicted = y_pred[high_pred_mask].mean()
        top20_actual = y_test[high_pred_mask].mean()
        top20_bias = (y_test[high_pred_mask] - y_pred[high_pred_mask]).mean()
        
        # Also check low predictions (bench decisions)
        low_pred_mask = y_pred <= np.percentile(y_pred, 20)
        bottom20_predicted = y_pred[low_pred_mask].mean()
        bottom20_actual = y_test[low_pred_mask].mean()
        bottom20_bias = (y_test[low_pred_mask] - y_pred[low_pred_mask]).mean()
        
        # Overall calibration metrics
        mae_by_bin = np.mean(np.abs(bias))
        max_bias = np.max(np.abs(bias))
        
        results = {
            'position': position,
            'n_bins': n_bins,
            'overall_mae': mean_absolute_error(y_test, y_pred),
            'overall_r2': r2_score(y_test, y_pred),
            'calibration_mae': float(mae_by_bin),
            'max_bias': float(max_bias),
            'top_20_pct': {
                'mean_predicted': float(top20_predicted),
                'mean_actual': float(top20_actual),
                'bias': float(top20_bias),
                'interpretation': 'positive = underpredict (conservative), negative = overpredict (risky)'
            },
            'bottom_20_pct': {
                'mean_predicted': float(bottom20_predicted),
                'mean_actual': float(bottom20_actual),
                'bias': float(bottom20_bias)
            },
            'bin_stats': {
                'mean_predicted': [float(x) for x in mean_predicted],
                'mean_actual': [float(x) for x in mean_actual],
                'bias': [float(x) for x in bias],
                'counts': [int(x) for x in counts]
            }
        }
        
        # Print key insights
        print(f"\n{'='*60}")
        print(f"{position} CALIBRATION ANALYSIS")
        print(f"{'='*60}")
        print(f"Overall R²: {results['overall_r2']:.4f}")
        print(f"Overall MAE: {results['overall_mae']:.2f} pts")
        print(f"Calibration MAE: {results['calibration_mae']:.2f} pts")
        print(f"Max bias: {results['max_bias']:.2f} pts")
        print(f"\n📊 TOP 20% PREDICTIONS (Captain Picks):")
        print(f"  Mean predicted: {top20_predicted:.2f} pts")
        print(f"  Mean actual:    {top20_actual:.2f} pts")
        print(f"  Bias:           {top20_bias:+.2f} pts")
        if top20_bias > 0.5:
            print(f"  ⚠️  Model UNDERESTIMATES high scorers (conservative)")
        elif top20_bias < -0.5:
            print(f"  ⚠️  Model OVERESTIMATES high scorers (risky for captain picks!)")
        else:
            print(f"  ✅ Well calibrated for captain picks")
        
        print(f"\n📊 BOTTOM 20% PREDICTIONS (Bench Decisions):")
        print(f"  Mean predicted: {bottom20_predicted:.2f} pts")
        print(f"  Mean actual:    {bottom20_actual:.2f} pts")
        print(f"  Bias:           {bottom20_bias:+.2f} pts")
        
        return results
    
    # ============================================================================
    # TASK 2: START PROBABILITY MODEL
    # ============================================================================
    
    def build_start_probability_model(
        self,
        df: pd.DataFrame,
        minutes_threshold: int = 60
    ):
        """
        Build classifier to predict whether a player will start.
        
        This is CRITICAL because predicting 5pts for a player who doesn't
        start is the biggest source of MAE in practice.
        
        Args:
            df: Training dataframe with features
            minutes_threshold: Minutes to consider "started" (default 60)
            
        Returns:
            Trained XGBClassifier
        """
        logger.info("Building start probability model...")
        
        # Target: did the player start (minutes >= threshold)?
        df['started'] = (df['minutes'] >= minutes_threshold).astype(int)
        
        # Features — only pre-match information
        # These capture rotation risk
        start_features = [
            'minutes_rolling5',      # recent playing time trend
            'form',                  # form affects selection
            'was_home',              # home/away rotation patterns
            'gw_in_season',          # managers rotate more in busy periods
            'value',                 # expensive players usually start
            'ict_index',             # recent performance
            'influence', 'creativity', 'threat',  # role in team
        ]
        
        # Add position and team one-hot features
        position_cols = [c for c in df.columns if c.startswith('position_')]
        team_cols = [c for c in df.columns if c.startswith('team_')]
        
        all_features = start_features + position_cols + team_cols
        available_features = [f for f in all_features if f in df.columns]
        
        logger.info(f"Using {len(available_features)} features for start probability")
        
        X_start = df[available_features].fillna(0)
        y_start = df['started']
        
        # Temporal split
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X_start.iloc[:split_idx], X_start.iloc[split_idx:]
        y_train, y_test = y_start.iloc[:split_idx], y_start.iloc[split_idx:]
        
        # Train classifier
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric='logloss'
        )
        
        clf.fit(X_train, y_train)
        
        # Evaluate
        train_proba = clf.predict_proba(X_train)[:, 1]
        test_proba = clf.predict_proba(X_test)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)
        
        print(f"\n{'='*60}")
        print(f"START PROBABILITY MODEL")
        print(f"{'='*60}")
        print(f"Train AUC: {train_auc:.4f}")
        print(f"Test AUC:  {test_auc:.4f}")
        print(f"\nClassification Report (threshold=0.5):")
        print(classification_report(y_test, test_proba >= 0.5, 
                                     target_names=['Benched', 'Started']))
        
        # Key insight: impact on expected points
        print(f"\n📊 IMPACT ON PREDICTIONS:")
        start_rate = y_test.mean()
        print(f"Actual start rate: {start_rate:.1%}")
        print(f"Mean predicted probability: {test_proba.mean():.1%}")
        
        # Save model
        model_path = self.models_dir / 'start_probability_v1.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': clf,
                'features': available_features,
                'threshold': minutes_threshold,
                'metrics': {
                    'train_auc': float(train_auc),
                    'test_auc': float(test_auc),
                    'start_rate': float(start_rate)
                }
            }, f)
        logger.info(f"Saved start probability model to {model_path}")
        
        self.start_model = clf
        return clf
    
    def apply_start_probability(
        self,
        df: pd.DataFrame,
        predicted_points: np.ndarray,
        start_features: list
    ) -> np.ndarray:
        """
        Apply start probability to discount predictions.
        
        Args:
            df: Dataframe with features
            predicted_points: Raw point predictions
            start_features: Feature names for start model
            
        Returns:
            Expected points (predicted_points * start_probability)
        """
        if self.start_model is None:
            raise ValueError("Start model not loaded. Call build_start_probability_model() first.")
        
        X_start = df[start_features].fillna(0)
        start_prob = self.start_model.predict_proba(X_start)[:, 1]
        
        expected_points = predicted_points * start_prob
        
        # Report impact
        mae_before = np.mean(np.abs(predicted_points - df['total_points'].values))
        mae_after = np.mean(np.abs(expected_points - df['total_points'].values))
        improvement = mae_before - mae_after
        
        print(f"\n📊 START PROBABILITY IMPACT:")
        print(f"MAE before adjustment: {mae_before:.3f}")
        print(f"MAE after adjustment:  {mae_after:.3f}")
        print(f"Improvement: {improvement:+.3f} pts")
        
        return expected_points
    
    # ============================================================================
    # TASK 3: DOUBLE GAMEWEEK FEATURES
    # ============================================================================
    
    def add_dgw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add double gameweek (DGW) features.
        
        This is the HIGHEST-EDGE feature for actual FPL decisions.
        Players with 2 fixtures in a gameweek typically score ~1.7x points.
        
        Args:
            df: Dataframe with player data
            
        Returns:
            Dataframe with DGW features added
        """
        logger.info("Adding double gameweek features...")
        
        # Count fixtures per player per gameweek
        # In a DGW, a player appears twice (or more) for the same GW number
        fixtures_per_gw = (
            df.groupby(['name', 'season_x', 'GW'])
            .size()
            .reset_index(name='fixtures_this_gw')
        )
        
        df = df.merge(fixtures_per_gw, on=['name', 'season_x', 'GW'], how='left')
        df['fixtures_this_gw'] = df['fixtures_this_gw'].fillna(1).astype(int)
        df['is_dgw'] = (df['fixtures_this_gw'] > 1).astype(int)
        
        # UPCOMING DGW signal (next GW has 2+ fixtures)
        # This is what FPL managers actually want to transfer INTO
        # Note: shift(-1) looks like leakage but ISN'T — fixture schedule is public
        df = df.sort_values(['name', 'season_x', 'kickoff_time'])
        df['next_gw_fixtures'] = (
            df.groupby(['name', 'season_x'])['fixtures_this_gw']
            .shift(-1)
            .fillna(1)
            .astype(int)
        )
        df['next_gw_is_dgw'] = (df['next_gw_fixtures'] > 1).astype(int)
        
        # Analyze DGW impact
        dgw_mask = df['is_dgw'] == 1
        if dgw_mask.sum() > 0:
            avg_points_sgw = df[~dgw_mask]['total_points'].mean()
            avg_points_dgw = df[dgw_mask]['total_points'].mean()
            dgw_multiplier = avg_points_dgw / avg_points_sgw if avg_points_sgw > 0 else 1.0
            
            print(f"\n{'='*60}")
            print(f"DOUBLE GAMEWEEK ANALYSIS")
            print(f"{'='*60}")
            print(f"Single GW avg points: {avg_points_sgw:.2f}")
            print(f"Double GW avg points: {avg_points_dgw:.2f}")
            print(f"DGW multiplier: {dgw_multiplier:.2f}x")
            print(f"DGW samples: {dgw_mask.sum():,} / {len(df):,} ({dgw_mask.mean():.1%})")
            
            # By position
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                pos_mask = df['position'] == pos
                pos_dgw_mask = pos_mask & dgw_mask
                if pos_dgw_mask.sum() > 10:  # At least 10 samples
                    pos_sgw_avg = df[pos_mask & ~dgw_mask]['total_points'].mean()
                    pos_dgw_avg = df[pos_dgw_mask]['total_points'].mean()
                    pos_mult = pos_dgw_avg / pos_sgw_avg if pos_sgw_avg > 0 else 1.0
                    print(f"  {pos}: {pos_sgw_avg:.2f} → {pos_dgw_avg:.2f} ({pos_mult:.2f}x)")
        else:
            logger.warning("No DGW samples found in dataset")
        
        logger.info(f"Added DGW features: fixtures_this_gw, is_dgw, next_gw_fixtures, next_gw_is_dgw")
        
        return df
    
    # ============================================================================
    # MAIN WORKFLOW
    # ============================================================================
    
    def run_all_production_tasks(
        self,
        df: pd.DataFrame,
        position_models: dict,
        save_results: bool = True
    ):
        """
        Run all 3 production readiness tasks.
        
        Args:
            df: Full dataframe with features and targets
            position_models: Dict of {position: model} for each position
            save_results: Whether to save results to JSON
            
        Returns:
            Dict with all results
        """
        results = {
            'calibration': {},
            'start_probability': {},
            'dgw_analysis': {}
        }
        
        # Task 1: Calibration Analysis per position
        logger.info("\n" + "="*60)
        logger.info("TASK 1: CALIBRATION ANALYSIS")
        logger.info("="*60)
        
        for position, model in position_models.items():
            pos_mask = df['position'] == position
            pos_df = df[pos_mask]
            
            # Need to get predictions from saved model
            # For this demo, we'll use test split
            split_idx = int(len(pos_df) * 0.8)
            X_test = pos_df.iloc[split_idx:]
            y_test = X_test['total_points'].values if 'total_points' in X_test else X_test['upcoming'].values
            
            # Get features (excluding target and metadata)
            feature_cols = [c for c in X_test.columns 
                           if c not in ['total_points', 'upcoming', 'name', 'season_x', 
                                       'kickoff_time', 'GW', 'element', 'fixture']]
            X_test_features = X_test[feature_cols].fillna(0)
            
            # Predict
            y_pred = model.predict(X_test_features)
            
            # Analyze calibration
            cal_results = self.calibration_analysis(y_test, y_pred, position)
            results['calibration'][position] = cal_results
        
        # Task 2: Start Probability Model
        logger.info("\n" + "="*60)
        logger.info("TASK 2: START PROBABILITY MODEL")
        logger.info("="*60)
        
        start_model = self.build_start_probability_model(df)
        results['start_probability']['model_saved'] = True
        
        # Task 3: DGW Features
        logger.info("\n" + "="*60)
        logger.info("TASK 3: DOUBLE GAMEWEEK FEATURES")
        logger.info("="*60)
        
        df_with_dgw = self.add_dgw_features(df.copy())
        results['dgw_analysis']['features_added'] = True
        results['dgw_analysis']['new_features'] = [
            'fixtures_this_gw', 'is_dgw', 'next_gw_fixtures', 'next_gw_is_dgw'
        ]
        
        # Save results
        if save_results:
            results_path = self.models_dir / 'production_readiness_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nSaved results to {results_path}")
        
        print("\n" + "="*60)
        print("✅ ALL PRODUCTION READINESS TASKS COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Review calibration plots in ml/models/")
        print("2. Integrate start probability into prediction pipeline")
        print("3. Add DGW features to feature engineering")
        print("4. Retrain models with DGW features")
        print("5. Deploy to production!")
        
        return results, df_with_dgw


# Example usage
if __name__ == "__main__":
    # This would be called from train.py after models are trained
    print("Production Readiness Module")
    print("Import this module and call run_all_production_tasks()")
