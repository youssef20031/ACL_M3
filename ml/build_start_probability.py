"""
Build Start Probability Model (v6.1 Update)
Creates a classifier to predict rotation/benching risk
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.feature_engineering import FeatureEngineer

print("\n" + "="*60)
print("START PROBABILITY MODEL BUILDER (V6.1)")
print("="*60)

# Load data
print("\n📂 Loading dataset...")
data_file = "cleaned_merged_seasons_cleaned.csv"
if not Path(data_file).exists():
    print(f"❌ Dataset not found: {data_file}")
    sys.exit(1)

df = pd.read_csv(data_file)
print(f"✅ Loaded {len(df):,} rows")

# Filter to recent seasons
if 'season_x' in df.columns:
    recent_seasons = ['2023-24', '2024-25', '2025-26']
    df = df[df['season_x'].isin(recent_seasons)]
    print(f"✅ Filtered to {len(df):,} rows from {recent_seasons}")

# Engineer features
print("\n⚙️  Engineering features...")

# Save minutes BEFORE engineering (will be transformed)
minutes_original = df['minutes'].copy()

feature_engineer = FeatureEngineer()
feature_engineer.fit(df)
df_eng = feature_engineer.engineer_features(df, is_training=True)

# Now minutes is still in df_eng, just use it from there
# Prepare features (without target since we're not predicting points)
X, _ = feature_engineer.prepare_features(df_eng, include_target=False)

# Minutes should still be in X as it's in the base features
# But let's check and handle if needed
if 'minutes' not in X.columns:
    print("❌ Missing 'minutes' column after feature engineering")
    # Try to get it from df_eng if available
    if 'minutes' in df_eng.columns and len(df_eng) == len(X):
        X['minutes'] = df_eng['minutes'].values
    else:
        print("❌ Cannot recover minutes column")
        sys.exit(1)

print(f"✅ Features ready: {X.shape[0]} samples, {X.shape[1]-1} features")

# Create target: started if minutes >= 60
print("\n🎯 Creating target variable...")
MINUTES_THRESHOLD = 60
X['started'] = (X['minutes'] >= MINUTES_THRESHOLD).astype(int)

start_rate = X['started'].mean()
print(f"✅ Target created: {start_rate:.1%} start rate (minutes >= {MINUTES_THRESHOLD})")

# Define features for start probability
print("\n📋 Selecting features...")
start_features = [
    'minutes_rolling5',      # Recent playing time
    'form',                  # Player form
    'was_home',              # Home/away
    'gw_in_season',          # Fixture congestion
    'value',                 # Price (expensive players start more)
    'ict_index',             # Recent performance
    'influence', 'creativity', 'threat',  # Role indicators
]

# Add position and team one-hot
position_cols = [c for c in X.columns if c.startswith('position_')]
team_cols = [c for c in X.columns if c.startswith('team_')]

all_features = start_features + position_cols + team_cols
available_features = [f for f in all_features if f in X.columns]

print(f"✅ Using {len(available_features)} features:")
print(f"   - Base features: {len([f for f in start_features if f in X.columns])}")
print(f"   - Position one-hot: {len(position_cols)}")
print(f"   - Team one-hot: {len(team_cols)}")

# Prepare data
X_features = X[available_features].fillna(0)
y = X['started']

# Temporal split
split_idx = int(len(X_features) * 0.8)
X_train = X_features.iloc[:split_idx]
X_test = X_features.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"\n📊 Data split:")
print(f"   Train: {len(X_train):,} samples ({y_train.mean():.1%} start rate)")
print(f"   Test:  {len(X_test):,} samples ({y_test.mean():.1%} start rate)")

# Train model
print("\n🤖 Training XGBoost classifier...")
clf = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

clf.fit(X_train, y_train, verbose=False)
print("✅ Training complete")

# Evaluate
print("\n📈 Evaluating performance...")
train_proba = clf.predict_proba(X_train)[:, 1]
test_proba = clf.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_proba)
test_auc = roc_auc_score(y_test, test_proba)

train_acc = accuracy_score(y_train, train_proba >= 0.5)
test_acc = accuracy_score(y_test, test_proba >= 0.5)

print(f"\n{'='*60}")
print("PERFORMANCE METRICS")
print(f"{'='*60}")
print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")
print(f"Train Accuracy: {train_acc:.1%}")
print(f"Test Accuracy:  {test_acc:.1%}")

print(f"\n📊 Classification Report (threshold=0.5):")
print(classification_report(y_test, test_proba >= 0.5, 
                           target_names=['Benched', 'Started'],
                           digits=3))

# Feature importance
print(f"\n🔍 Top 10 Features:")
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:.4f}")

# Impact analysis
print(f"\n{'='*60}")
print("IMPACT ON PREDICTIONS")
print(f"{'='*60}")

# Simulate applying to point predictions
mean_start_prob = test_proba.mean()
print(f"Mean start probability: {mean_start_prob:.1%}")

# Estimate MAE impact (rough approximation)
# Players who don't start score 0-2 pts (assume ~1pt average)
# If we predict 5pts but they score 1pt, error = 4pts
# Start probability adjustment: 5 * 0.3 = 1.5pts → error = 0.5pts
benched_mask = y_test == 0
if benched_mask.sum() > 0:
    avg_prob_benched = test_proba[benched_mask].mean()
    avg_prob_started = test_proba[~benched_mask].mean()
    
    print(f"\nActual benched players:")
    print(f"   Count: {benched_mask.sum():,} ({benched_mask.mean():.1%})")
    print(f"   Avg predicted prob: {avg_prob_benched:.1%}")
    
    print(f"\nActual started players:")
    print(f"   Count: {(~benched_mask).sum():,} ({(~benched_mask).mean():.1%})")
    print(f"   Avg predicted prob: {avg_prob_started:.1%}")
    
    print(f"\nModel discrimination:")
    discrimination = avg_prob_started - avg_prob_benched
    print(f"   Difference: {discrimination:.1%}")
    if discrimination > 0.3:
        print(f"   ✅ Excellent discrimination")
    elif discrimination > 0.2:
        print(f"   ✅ Good discrimination")
    else:
        print(f"   ⚠️  Weak discrimination")

# Expected MAE improvement
print(f"\n💡 Expected Impact:")
print(f"   Without start prob: Predict 5pts for benched player → MAE +4pts")
print(f"   With start prob: Predict 5pts × {avg_prob_benched:.0%} = {5*avg_prob_benched:.1f}pts → MAE +{abs(1-5*avg_prob_benched):.1f}pts")
print(f"   Estimated MAE reduction: 0.05-0.10 pts overall")

# Save model
print(f"\n{'='*60}")
print("SAVING MODEL")
print(f"{'='*60}")

output_dir = Path("ml/models")
output_dir.mkdir(exist_ok=True)

model_data = {
    'model': clf,
    'features': available_features,
    'threshold': MINUTES_THRESHOLD,
    'metrics': {
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'start_rate': float(start_rate),
        'mean_start_prob': float(mean_start_prob)
    },
    'feature_importance': feature_importance.to_dict('records')
}

model_path = output_dir / 'start_probability_v1.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"✅ Model saved to: {model_path}")

# Save feature list
features_path = output_dir / 'start_probability_features.txt'
with open(features_path, 'w') as f:
    for feat in available_features:
        f.write(f"{feat}\n")
print(f"✅ Features saved to: {features_path}")

# Save metrics
metrics_path = output_dir / 'start_probability_metrics.json'
import json
with open(metrics_path, 'w') as f:
    json.dump(model_data['metrics'], f, indent=2)
print(f"✅ Metrics saved to: {metrics_path}")

print(f"\n{'='*60}")
print("✅ START PROBABILITY MODEL COMPLETE (V6.1)")
print(f"{'='*60}")
print(f"\nModel ready for integration into prediction pipeline!")
print(f"Test AUC: {test_auc:.4f} (target: >0.75)")
print(f"\nUsage:")
print(f"  from ml.production_readiness import ProductionReadiness")
print(f"  prod = ProductionReadiness()")
print(f"  with open('{model_path}', 'rb') as f:")
print(f"      start_model_data = pickle.load(f)")
print(f"      prod.start_model = start_model_data['model']")
print(f"\n{'='*60}\n")
