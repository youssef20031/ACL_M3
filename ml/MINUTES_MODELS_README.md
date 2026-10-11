# Minutes Prediction Models (Model C & D)

## Overview

The minutes prediction models provide a two-stage prediction of player playing time:

1. **Model C**: Binary classification - probability a player starts (P(start))
2. **Model D**: Regression - expected minutes given they start (E[minutes|start])

Combined: **Expected minutes = P(start) × E[minutes|start]**

## Architecture

### Model C: P(start)
- **Type**: XGBoost binary classifier with Platt scaling calibration
- **Target**: `started = 1` if player started the match
  - Uses `starts` column when available (2023-24+)
  - Falls back to `minutes >= 60` threshold for earlier seasons
- **Output**: Calibrated probability between 0 and 1
- **Trained per position**: 4 models (GK, DEF, MID, FWD)

### Model D: E[minutes|start]
- **Type**: XGBoost regressor
- **Target**: Actual minutes played (for matches where player started)
- **Training data**: Only samples where `started = 1`
- **Output**: Expected minutes (0-90 range)
- **Trained per position**: 4 models (GK, DEF, MID, FWD)

## Features

### Standard features (from feature_engineering.py)
- Form indicators: `form`, `minutes_rolling5`, `points_per_90`
- Home/away: `was_home`, `home_form`, `away_form`
- Opponent strength: `opp_def_strength`, `opp_off_strength`
- Team context: `team_def_strength`, `value`
- Fixture context: `fixtures_this_gw` (DGW), `gw_in_season`
- Congestion: `days_since_last_match`, `fixture_congestion_risk`

### Minutes-specific features (NEW)
- `started_last_match`: Binary, did player start previous GW
- `started_2_of_last_3`: Binary, started ≥ 2 of last 3 GWs
- `avg_minutes_last_3`: Mean minutes over last 3 GWs

## Training

### Hyperparameters

**Model C (Classification)**:
```python
{
    'objective': 'binary:logistic',
    'max_depth': 4,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': <computed from class imbalance>,
    'early_stopping_rounds': 30
}
```

**Model D (Regression)**:
```python
{
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'early_stopping_rounds': 30
}
```

### Data Split
- **Temporal split**: 72% train / 8% validation / 20% test
- Sorted by kickoff_time to prevent lookahead bias
- Per-position training to capture position-specific rotation patterns

### Training Script
```bash
python ml/train_minutes_models.py
```

## Performance Targets

### Model C (P(start))
- **AUC-ROC**: ≥ 0.80
- **Calibration**: Platt scaling applied for reliable probabilities
- Expected performance by position:
  - GK: ~0.85 AUC (very predictable - starter plays almost always)
  - DEF: ~0.82 AUC (rotation risk, esp. fullbacks)
  - MID: ~0.80 AUC (highest rotation)
  - FWD: ~0.81 AUC (moderate rotation)

### Model D (E[minutes|start])
- **RMSE**: ≤ 12 minutes
- **MAE**: ≤ 8 minutes
- Expected performance by position:
  - GK: ~5 min RMSE (very low variance - almost always 90)
  - DEF: ~10 min RMSE
  - MID: ~12 min RMSE (subbed more frequently)
  - FWD: ~11 min RMSE

## Integration with Points Prediction

### Current Approach (v1)
The minutes models are trained and available but **NOT directly integrated** with points prediction yet.

**Why?** The existing points models (XGBoost per position) were trained to predict `total_points` directly, not `points_per_minute` or `points_per_90`. 

The implementation plan suggested:
```python
points_per_minute = raw_points / 66.5  # FLAWED - mathematically invalid
expected_points = expected_minutes * points_per_minute
```

**This is incorrect** because:
1. Existing models predict `total_points` for a full match context
2. Dividing by an arbitrary constant (66.5 = league avg minutes) doesn't give a proper per-minute rate
3. The relationship between minutes and points is non-linear

### Correct Approach (future v2)

**Option 1: Retrain points models with points-per-90 targets**
```python
# Training:
target = (total_points / minutes) * 90  # Points per 90 minutes

# Inference:
pp90_prediction = points_model.predict(features)
expected_points = (expected_minutes / 90) * pp90_prediction
```

**Option 2: Use minutes as a feature (simpler, current approach)**
- Keep existing points models as-is
- Minutes models provide context but don't directly modify predictions
- The existing models already learned minute patterns from training data

### Current Implementation
For now, we use **Option 2**:
- Minutes models predict expected playing time
- Points models continue to predict total points directly
- Minutes prediction is logged for context and debugging
- Future retraining will implement Option 1 properly

## Files

### Code
- `ml/minutes_model.py` - MinutesPredictor class
- `ml/train_minutes_models.py` - Training script
- `ml/predictor.py` - Integration with FPLPredictor
- `ml/feature_engineering.py` - Feature engineering (includes minutes features)

### Models (saved after training)
- `ml/models/minutes_pstart_{position}_v1.pkl` - Model C per position (4 files)
- `ml/models/minutes_eminutes_{position}_v1.pkl` - Model D per position (4 files)
- `ml/models/minutes_model_mappings.json` - Feature mappings
- `ml/models/minutes_models_metrics.json` - Training metrics

## Usage

### Training
```python
from ml.minutes_model import MinutesPredictor
import pandas as pd

# Load data
df = pd.read_csv('FPL_2023_2024.csv')

# Temporal split
train_df = df.iloc[:int(len(df)*0.72)]
val_df = df.iloc[int(len(df)*0.72):int(len(df)*0.80)]

# Train
predictor = MinutesPredictor(models_dir='ml/models')
metrics = predictor.train(train_df, val_df, save_models=True)
```

### Inference
```python
from ml.minutes_model import MinutesPredictor

# Load trained models
predictor = MinutesPredictor(models_dir='ml/models')
predictor.load()

# Predict for a player
player_data = {
    'position': 'FWD',
    'form': 5.2,
    'minutes': 90,
    'started_last_match': 1,
    'avg_minutes_last_3': 85,
    # ... other features
}

prediction = predictor.predict(player_data)
print(f"P(start): {prediction['p_start']:.2%}")
print(f"E[minutes|start]: {prediction['e_minutes_given_start']:.1f}")
print(f"Expected minutes: {prediction['expected_minutes']:.1f}")
```

### Integration with FPLPredictor
```python
from ml.predictor import FPLPredictor

# Initialize with minutes models enabled
predictor = FPLPredictor(
    model_path='ml/models/xgboost_fwd_v1.pkl',
    enable_minutes_models=True  # Loads Model C & D
)

# Predict - minutes info included in metadata
result = predictor.predict_next_gameweek(player_data)
print(f"Predicted points: {result.predicted_points:.1f}")
print(f"Expected minutes: {result.features_used['expected_minutes']:.1f}")
```

## Next Steps (Phase 2 Complete → Phase 3)

✅ **Phase 2 Complete**: Minutes models trained and integrated
- [x] Model C: P(start) per position
- [x] Model D: E[minutes|start] per position
- [x] Training script
- [x] Integration with FPLPredictor (context only, not direct multiplication)

🔜 **Phase 3**: Archetype Database Expansion
- [ ] Build automated archetype database from 2016-2026 promoted players
- [ ] Expand from 6 manual archetypes to 80-100 historical players
- [ ] Implement k-means clustering for archetype classification
- [ ] Update promoted_teams_handler.py to use expanded database

## References

- Implementation Plan: `implementation_plan.md`
- Task List: `tasks_list`
- Feature Engineering: `ml/feature_engineering.py`
- Promoted Teams: `ml/PROMOTED_TEAMS_README.md`
