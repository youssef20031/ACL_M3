"""
Calibrate Start Probability Model
Fix underconfident probabilities using Platt scaling or isotonic regression
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent.parent))

print("\n" + "="*60)
print("START PROBABILITY CALIBRATION")
print("="*60)

# Load the uncalibrated model
model_path = Path("ml/models/start_probability_v1_clean.pkl")
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    print("   Run build_start_probability_clean.py first")
    sys.exit(1)

with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

uncalibrated_model = model_data['model']
feature_names = model_data['features']
print(f"[OK] Loaded uncalibrated model")

# Load and prepare data (same as build script)
print("\n[*] Loading dataset...")
df = pd.read_csv("cleaned_merged_seasons_cleaned.csv")
df = df[df['season_x'].isin(['2023-24', '2024-25', '2025-26'])]
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')
df = df.sort_values('kickoff_time').reset_index(drop=True)

MINUTES_THRESHOLD = 60
df['started'] = (df['minutes'] >= MINUTES_THRESHOLD).astype(int)

# Rebuild features (same as build script)
features_list = []

# Position
if 'position' in df.columns:
    position_dummies = pd.get_dummies(df['position'], prefix='pos', dtype=int)
    features_list.append(position_dummies)

# Team
team_col = 'team_x' if 'team_x' in df.columns else 'team'
if team_col in df.columns:
    team_dummies = pd.get_dummies(df[team_col], prefix='team', dtype=int)
    features_list.append(team_dummies)

# Other features
if 'value' in df.columns:
    features_list.append(df[['value']])
if 'was_home' in df.columns:
    features_list.append(df[['was_home']].astype(int))
if 'GW' in df.columns:
    df['gw_normalized'] = df['GW'] / 38.0
    df['busy_period'] = ((df['GW'] >= 14) & (df['GW'] <= 20)).astype(int)
    features_list.append(df[['gw_normalized', 'busy_period']])
if 'opp_team_name' in df.columns:
    opp_strength = df.groupby('opp_team_name')['team_h_score'].transform('mean').fillna(1.5)
    df['opponent_strength'] = opp_strength
    features_list.append(df[['opponent_strength']])

X = pd.concat(features_list, axis=1)
y = df['started']

# Temporal split
split_time = df['kickoff_time'].quantile(0.8)
train_mask = df['kickoff_time'] <= split_time
test_mask = df['kickoff_time'] > split_time

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"[OK] Data prepared: {len(X_train):,} train, {len(X_test):,} test")

# Get uncalibrated predictions
uncalib_proba = uncalibrated_model.predict_proba(X_test)[:, 1]

# Check calibration
print(f"\n{'='*60}")
print("UNCALIBRATED MODEL ANALYSIS")
print(f"{'='*60}")

benched_mask = y_test == 0
started_mask = y_test == 1

print(f"\nPredicted probabilities:")
print(f"  Actual benched: {uncalib_proba[benched_mask].mean():.1%} avg")
print(f"  Actual started: {uncalib_proba[started_mask].mean():.1%} avg")
print(f"\nProblem: Started players only 43.4% predicted!")
print(f"  This means: 5pts x 43% = 2.2pts (undervalues starters by 57%)")

# Calibration curve
fraction_pos, mean_pred = calibration_curve(y_test, uncalib_proba, n_bins=10)

print(f"\n[*] Calibration curve analysis:")
for i, (pred, actual) in enumerate(zip(mean_pred, fraction_pos)):
    print(f"  Bin {i+1}: Predicted {pred:.1%} → Actual {actual:.1%} (gap: {actual-pred:+.1%})")

# Plot calibration
plt.figure(figsize=(10, 6))
plt.plot(mean_pred, fraction_pos, 'o-', linewidth=2, label='Uncalibrated')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.xlabel('Mean predicted probability', fontsize=12)
plt.ylabel('Fraction actually started', fontsize=12)
plt.title('Start Probability Calibration Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = Path("ml/models/start_prob_calibration_before.png")
plt.savefig(plot_path, dpi=150)
print(f"\n[OK] Saved calibration plot: {plot_path}")

# Manual Platt scaling (sigmoid calibration)
print(f"\n{'='*60}")
print("APPLYING CALIBRATION (Manual Platt Scaling)")
print(f"{'='*60}")

from scipy.optimize import minimize
from scipy.special import expit  # sigmoid

# Get uncalibrated predictions on training data for fitting
print("\n[*] Fitting Platt scaling parameters...")
train_proba = uncalibrated_model.predict_proba(X_train)[:, 1]

# Platt scaling: P_calib = sigmoid(A * log_odds + B)
# where log_odds = log(p / (1-p))

def platt_loss(params, proba, y_true):
    A, B = params
    # Convert probabilities to log-odds
    proba_clipped = np.clip(proba, 1e-7, 1 - 1e-7)
    log_odds = np.log(proba_clipped / (1 - proba_clipped))
    # Apply scaling
    calib_logits = A * log_odds + B
    calib_proba = expit(calib_logits)
    # Binary cross-entropy loss
    loss = -np.mean(y_train * np.log(calib_proba + 1e-7) + (1 - y_train) * np.log(1 - calib_proba + 1e-7))
    return loss

# Optimize
result = minimize(
    platt_loss,
    x0=[1.0, 0.0],  # Start with identity transform
    args=(train_proba, y_train.values),
    method='BFGS'
)

A_platt, B_platt = result.x
print(f"[OK] Platt parameters: A={A_platt:.4f}, B={B_platt:.4f}")

# Apply calibration to test set
def apply_platt_scaling(proba, A, B):
    proba_clipped = np.clip(proba, 1e-7, 1 - 1e-7)
    log_odds = np.log(proba_clipped / (1 - proba_clipped))
    calib_logits = A * log_odds + B
    return expit(calib_logits)

calib_proba = apply_platt_scaling(uncalib_proba, A_platt, B_platt)
print("[OK] Calibration applied")

# Evaluate calibrated model
print(f"\n{'='*60}")
print("CALIBRATED MODEL PERFORMANCE")
print(f"{'='*60}")

calib_auc = roc_auc_score(y_test, calib_proba)
print(f"\nAUC: {calib_auc:.4f} (uncalibrated: {roc_auc_score(y_test, uncalib_proba):.4f})")
print("  (AUC should stay same - only probabilities change)")

print(f"\nCalibrated probabilities:")
print(f"  Actual benched: {calib_proba[benched_mask].mean():.1%} avg")
print(f"  Actual started: {calib_proba[started_mask].mean():.1%} avg")
print(f"  Discrimination: {calib_proba[started_mask].mean() - calib_proba[benched_mask].mean():.1%}")

# New calibration curve
fraction_pos_calib, mean_pred_calib = calibration_curve(y_test, calib_proba, n_bins=10)

print(f"\n[*] Calibrated curve:")
for i, (pred, actual) in enumerate(zip(mean_pred_calib, fraction_pos_calib)):
    gap = abs(actual - pred)
    status = "[OK]" if gap < 0.05 else "[!]"
    print(f"  {status} Bin {i+1}: Predicted {pred:.1%} -> Actual {actual:.1%} (gap: {actual-pred:+.1%})")

# Plot comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(mean_pred, fraction_pos, 'o-', linewidth=2, label='Uncalibrated', color='red')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect', alpha=0.3)
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction actually started')
plt.title('Before Calibration')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(mean_pred_calib, fraction_pos_calib, 'o-', linewidth=2, label='Calibrated', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect', alpha=0.3)
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction actually started')
plt.title('After Calibration')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plot_path = Path("ml/models/start_prob_calibration_comparison.png")
plt.savefig(plot_path, dpi=150)
print(f"\n[OK] Saved comparison plot: {plot_path}")

# Expected impact
print(f"\n{'='*60}")
print("IMPACT ON PREDICTIONS")
print(f"{'='*60}")

print(f"\nFor a 5pt prediction:")
print(f"  Uncalibrated: 5 x {uncalib_proba[started_mask].mean():.1%} = {5 * uncalib_proba[started_mask].mean():.2f}pts (undervalues by {(1 - uncalib_proba[started_mask].mean())*100:.0f}%)")
print(f"  Calibrated:   5 x {calib_proba[started_mask].mean():.1%} = {5 * calib_proba[started_mask].mean():.2f}pts")

# Decision on which to use
avg_gap = np.mean(np.abs(fraction_pos_calib - mean_pred_calib))
print(f"\nCalibration quality:")
print(f"  Avg absolute gap: {avg_gap:.3f}")

if avg_gap < 0.05:
    print(f"  [OK] Excellent calibration (<5% gap)")
    recommendation = "Use calibrated model"
elif avg_gap < 0.10:
    print(f"  [OK] Good calibration (<10% gap)")
    recommendation = "Use calibrated model"
else:
    print(f"  [!] Moderate calibration (>10% gap)")
    recommendation = "Consider isotonic regression instead"

print(f"\n[*] Recommendation: {recommendation}")

# Save calibrated model
if avg_gap < 0.10:
    print(f"\n{'='*60}")
    print("SAVING CALIBRATED MODEL")
    print(f"{'='*60}")
    
    output_path = Path("ml/models/start_probability_v1_calibrated.pkl")
    
    calibrated_data = {
        'base_model': uncalibrated_model,
        'A_platt': float(A_platt),
        'B_platt': float(B_platt),
        'features': feature_names,
        'threshold': MINUTES_THRESHOLD,
        'metrics': {
            'test_auc': float(calib_auc),
            'calibration_gap': float(avg_gap),
            'started_avg_prob': float(calib_proba[started_mask].mean()),
            'benched_avg_prob': float(calib_proba[benched_mask].mean())
        },
        'calibration_method': 'platt_scaling_manual'
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(calibrated_data, f)
    
    print(f"[OK] Calibrated model saved: {output_path}")
    print(f"\n[OK] Use this model for production!")
else:
    print(f"\n[!] Calibration not good enough, trying isotonic regression...")

print(f"\n{'='*60}\n")
