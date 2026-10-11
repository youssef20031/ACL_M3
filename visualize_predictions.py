"""Quick visualization of model predictions"""
import pandas as pd
import json

# Load predictions
df = pd.read_csv('ml/predictions/next_gw_predictions.csv')

print("="*80)
print("FPL ML MODEL - PREDICTION ANALYSIS")
print("="*80)
print(f"\nTotal Predictions: {len(df):,}")
print(f"Average Predicted Points: {df['predicted_points'].mean():.2f}")
print(f"Median Predicted Points: {df['predicted_points'].median():.2f}")
print(f"Max Predicted Points: {df['predicted_points'].max():.2f} ({df.loc[df['predicted_points'].idxmax(), 'name']})")
print(f"Min Predicted Points: {df['predicted_points'].min():.2f}")

print("\n" + "="*80)
print("PREDICTIONS BY POSITION")
print("="*80)
for pos in ['GK', 'DEF', 'MID', 'FWD']:
    pos_df = df[df['position'] == pos]
    print(f"\n{pos}:")
    print(f"  Count: {len(pos_df):,}")
    print(f"  Avg Points: {pos_df['predicted_points'].mean():.2f}")
    print(f"  Top Player: {pos_df.iloc[0]['name']} ({pos_df.iloc[0]['predicted_points']:.2f} pts)")

print("\n" + "="*80)
print("VALUE PICKS (Points per £1m)")
print("="*80)
df['points_per_million'] = df['predicted_points'] / (df['value'] / 10)
df_sorted = df.sort_values('points_per_million', ascending=False)

for pos in ['GK', 'DEF', 'MID', 'FWD']:
    pos_df = df_sorted[df_sorted['position'] == pos].head(5)
    print(f"\n{pos} - Best Value:")
    for idx, row in pos_df.iterrows():
        print(f"  {row['predicted_points']:5.2f} pts | £{row['value']/10:.1f}m | {row['points_per_million']:.2f} pts/£m | {row['name']:30s} | {row['team']}")

print("\n" + "="*80)
print("PROMOTED TEAMS ANALYSIS")
print("="*80)
promoted = df[df['is_promoted_team'] == True]
print(f"Players from promoted teams: {len(promoted)}")
if len(promoted) > 0:
    print(f"Average predicted points: {promoted['predicted_points'].mean():.2f}")
    print(f"Top 5 promoted team players:")
    for idx, row in promoted.nlargest(5, 'predicted_points').iterrows():
        print(f"  {row['predicted_points']:5.2f} pts | {row['name']:30s} | {row['team']} | {row['position']}")

print("\n" + "="*80)
print("HOME VS AWAY PERFORMANCE")
print("="*80)
home_df = df[df['is_home'] == True]
away_df = df[df['is_home'] == False]
print(f"Home players: {len(home_df):,} | Avg: {home_df['predicted_points'].mean():.2f} pts")
print(f"Away players: {len(away_df):,} | Avg: {away_df['predicted_points'].mean():.2f} pts")
print(f"Home advantage: {(home_df['predicted_points'].mean() - away_df['predicted_points'].mean()):.2f} pts")

# Load training results
with open('ml/models/training_results_v2.json', 'r') as f:
    training_results = json.load(f)

print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)
for result in training_results:
    print(f"\n{result['position']}:")
    print(f"  R² Score: {result['r2']:.3f} ({result['r2']*100:.1f}% variance explained)")
    print(f"  RMSE: {result['rmse']:.3f} points")
    print(f"  MAE: {result['mae']:.3f} points")
    print(f"  Training samples: {result['n_train']:,}")
    print(f"  Test samples: {result['n_test']:,}")
    print(f"  Features: {result['n_features']}")

avg_r2 = sum(r['r2'] for r in training_results) / len(training_results)
print(f"\n{'='*80}")
print(f"🏆 OVERALL MODEL QUALITY: R² = {avg_r2:.3f} ({avg_r2*100:.1f}% variance explained)")
print(f"{'='*80}")
print("\nInterpretation:")
if avg_r2 >= 0.85:
    print("✅ EXCELLENT - Model has very high predictive accuracy")
elif avg_r2 >= 0.75:
    print("✅ GOOD - Model has strong predictive power")
elif avg_r2 >= 0.65:
    print("⚠️ FAIR - Model has moderate predictive power")
else:
    print("❌ POOR - Model needs improvement")
