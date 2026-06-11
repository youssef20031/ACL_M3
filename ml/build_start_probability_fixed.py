"""
Build Start Probability Model (LEAKAGE-FREE V6.1)
Fixed version removing all identified leakage sources
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import sys

sys.path.append(str(Path(__file__).parent.parent))

print("\n" + "="*60)
print("START PROBABILITY MODEL (LEAKAGE-FREE)")
print("="*60)

# Load data
print("\n📂 Loading dataset...")
df = pd.read_csv("cleaned_merged_seasons_cleaned.csv")
print(f"✅ Loaded {len(df):,} rows")

# Filter and sort temporally
if 'season_x' in df.columns:
    df = df[df['season_x'].isin(['2023-24', '2024-25', '2025-26'])]
    print(f"✅ Filtered to {len(df):,} rows")

if 'kickoff_time' in df.columns:
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')
    df = df.sort_values(['kickoff_time', 'name']).reset_index(drop=True)
    print(f"✅ Sorted temporally")

# Target: started if minutes >= 60
MINUTES_THRESHOLD = 60
df['started'] = (df['minutes'] >= MINUTES_THRESHOLD).astype(int)
print(f"✅ Target created: {df['started'].mean():.1%} start rate")

# CRITICAL: Remove ALL post-match and circular features
print("\n🔒 Selecting ONLY pre-match features...")

# Safe pre-match features
safe_features = [
    'value',            # Price (known before GW)
    'was_home',         # Home/away (known before GW)
    'GW',              # GW number (temporal context)
]

# Add lagged performance (from PREVIOUS games only)
df = df.sort_values(['name', 'kickoff_time'])

# Lag total_points from previous games (NOT minutes!)
df['prev_form'] = df.groupby('name')['total_points'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
).fillna(0)

# Lag appearances (did they start last 5 games?)
df['prev_start_rate'] = df.groupby('name')['started'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
).fillna(0.5)  # Default 50% for new players

# Position (structural - GK almost always start)
position_dummies = pd.get_dummies(df['position'], prefix='position', dtype=int)

# Team (rotation policies differ by team)
team_dummies = pd.get_dummies(df['team_x' if 'team_x' in df.columns else 'team'], prefix='team', dtype=int)

safe_features.extend(['prev_form', 'prev_start_rate'])

print(f"✅ Safe features:")
print(f"   - Base: {len(safe_features)}")
print(f"   - Position: {len(position_dummies.columns)}")
print(f"   - Team: {len(team_dummies.columns)}")

# Combine features
X = df[safe_features].copy()
X = pd.concat([X, position_dummies, team_dummies], axis=1)
y = df['started']

print(f"\n📊 Feature matrix: {X.shape}")

# CRITICAL: Temporal split (by date, not random index)
# Find the 80% timepoint
if 'kickoff_time' in df.columns:
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    sorted_times = df['kickoff_time'].sort_values()
    split_time = sorted_times.iloc[int(len(sorted_times) * 0.8)]
    
    train_mask = df['kickoff_time'] <= split_time
    test_mask = df['kickoff_time'] > split_time
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"\n⏰ Temporal split at {split_time.date()}")
    print(f"   Train GW range: {df[train_mask]['GW'].min()}-{df[train_mask]['GW'].max()}")
    print(f"   Test GW range: {df[test_mask]['GW'].min()}-{df[test_mask]['GW'].max()}")
else:
    # Fallback to index split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"\n⚠️  Using index split (no temporal column)")

print(f"   Train: {len(X_train):,} ({y_train.mean():.1%} start rate)")
print(f"   Test: {len(X_test):,} ({y_test.mean():.1%} start rate)")

# Train model
print("\n🤖 Training XGBoost...")
clf = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train, verbose=False)

# Evaluate
train_proba = clf.predict_proba(X_train)[:, 1]
test_proba = clf.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_proba)
test_auc = roc_auc_score(y_test, test_proba)

print(f"\n{'='*60}")
print("PERFORMANCE (LEAKAGE-FREE)")
print(f"{'='*60}")
print(f"Train AUC: {train_auc:.4f}")
print(f"Test AUC:  {test_auc:.4f}")
print(f"Train Acc: {accuracy_score(y_train, train_proba >= 0.5):.1%}")
print(f"Test Acc:  {accuracy_score(y_test, test_proba >= 0.5):.1%}")

if test_auc > 0.90:
    print(f"\n⚠️  WARNING: AUC still suspiciously high ({test_auc:.4f})")
    print(f"   Check for remaining leakage!")
elif test_auc >= 0.75:
    print(f"\n✅ AUC in expected range (0.75-0.85)")
    print(f"   Model appears legitimate!")
else:
    print(f"\n⚠️  AUC lower than expected ({test_auc:.4f})")
    print(f"   May need better features")

print(f"\n📊 Classification Report:")
print(classification_report(y_test, test_proba >= 0.5,
                           target_names=['Benched', 'Started'],
                           digits=3))

# Feature importance
print(f"\n🔍 Top 10 Features:")
feature_names = list(X.columns)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance_df.head(10).iterrows():
    print(f"   {row['feature']:30s}: {row['importance']:.4f}")

# Save
output_dir = Path("ml/models")
model_path = output_dir / 'start_probability_v1_fixed.pkl'

model_data = {
    'model': clf,
    'features': feature_names,
    'threshold': MINUTES_THRESHOLD,
    'metrics': {
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'start_rate': float(df['started'].mean())
    }
}

with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✅ Saved to: {model_path}")
print(f"\n{'='*60}\n")
