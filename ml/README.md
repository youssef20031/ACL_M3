# FPL ML Prediction Module

## Overview

This module integrates machine learning predictions into the FPL RAG system. It predicts player performance (FPL points) for the next gameweek based on recent form, historical statistics, and match context.

## Improvements Implemented

Based on the milestone_1 notebook remarks, the following improvements have been implemented:

### 1. ✅ Temporal Train/Test Split (Most Impactful Fix)
- **Problem**: Random split causes data leakage - model sees future data during training
- **Solution**: Sort data by `kickoff_time` and split chronologically
- **Impact**: Train on past data, validate on middle period, test on most recent data
- **Implementation**: `FPLModelTrainer.temporal_train_test_split()`

### 2. ✅ Remove/Lag Features (Data Leakage Prevention)
- **Problem**: `total_points` and `bps` are target-derived features that leak information
- **Solution**: 
  - Removed from feature set
  - Lag `total_points` by 1 gameweek to create `upcoming` target
  - Use 4-game rolling average as `form` feature instead
- **Implementation**: `FeatureEngineer.engineer_features(lag_features=['total_points'])`

### 3. ✅ Add Dropout to Neural Network
- **Problem**: Original "nn_bad_model" prone to overfitting
- **Solution**: Added Dropout layers (0.3, 0.2, 0.1) after Dense layers
- **Implementation**: `FPLModelTrainer.train_neural_network()`
```python
Sequential([
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='linear')
])
```

### 4. ✅ Fix Position Labels in Reports
- **Problem**: "Forwards (MID)" label was incorrect for midfielders
- **Solution**: Created `POSITION_NAMES` mapping with correct labels
- **Implementation**: 
```python
POSITION_NAMES = {
    "GK": "Goalkeepers (GK)",
    "DEF": "Defenders (DEF)",
    "MID": "Midfielders (MID)",  # FIXED
    "FWD": "Forwards (FWD)"
}
```

### 5. ✅ Rename Model from nn_bad_model to nn_baseline_model
- **Problem**: Poor naming convention
- **Solution**: Renamed throughout codebase to `nn_baseline_model`
- **Impact**: Better communicates that it's a baseline for comparison

## Architecture

```
ml/
├── __init__.py                 # Module exports
├── feature_engineering.py      # Feature pipeline (form, encoding, lagging)
├── predictor.py               # Model inference and predictions
├── train.py                   # Training script with all improvements
├── api_integration.py         # FastAPI endpoints for predictions
├── models/                    # Trained model storage
│   ├── .gitkeep
│   ├── linear_regression_v1.pkl
│   ├── linear_regression_v1_mappings.json
│   ├── nn_baseline_v1.pkl (optional)
│   └── nn_baseline_v1_scaler.pkl
└── README.md                  # This file
```

## Dataset Strategy

Training uses the **3 most recent seasons** (2023-24, 2024-25, 2025-26):
- ✅ Sufficient data volume (~15,000+ gameweeks)
- ✅ Recent trends and meta changes
- ✅ Avoids outdated patterns from 2016-17 era

Alternative: Use all 6 seasons from `cleaned_merged_seasons_cleaned.csv` for maximum data.

## Models

### Linear Regression (Primary Model)
- **Pros**: Fast, interpretable, no overfitting
- **Cons**: Cannot capture non-linear patterns
- **Use Case**: Production predictions, baseline

### Neural Network Baseline (Secondary Model)
- **Pros**: Can learn complex patterns
- **Cons**: Slower, requires more data, needs scaling
- **Use Case**: Experimentation, comparison

## Features

### Engineered Features
- `form`: 4-game rolling average of total_points (lagged by 1)
- `team_goals`: Conditional on home/away status

### Numeric Features (~20)
minutes, goals_scored, assists, bps, ict_index, influence, creativity, threat, clean_sheets, bonus, goals_conceded, saves, yellow_cards, red_cards, penalties_missed, penalties_saved, own_goals, value, was_home, GW

### Categorical Features (One-Hot Encoded)
- `position`: GK, DEF, MID, FWD (4 columns)
- `team_x`: ~20 teams
- `opp_team_name`: ~20 opponent teams

**Total Features**: ~150 after encoding

### Excluded Features (Data Leakage Prevention)
- `total_points` (target-derived)
- `bps` (target-derived)
- `selected`, `transfers_in`, `transfers_out` (depends on predictions)
- `name`, `season`, `element`, `fixture` (identifiers)

## Training

### Quick Start

```bash
# Install dependencies
pip install scikit-learn tensorflow pandas numpy

# Train models (creates ml/models/*.pkl files)
python ml/train.py
```

### Training Options

Edit `ml/train.py` to choose dataset:
```python
dataset_options = {
    "2_seasons": ["FPL_2024_2025.csv", "FPL_2025_2026.csv"],
    "3_seasons": ["FPL_2023_2024.csv", "FPL_2024_2025.csv", "FPL_2025_2026.csv"],
    "all_6_seasons": "cleaned_merged_seasons_cleaned.csv"
}

choice = "3_seasons"  # Change here
```

### Expected Performance

Based on temporal split:
- **Linear Regression**: RMSE ~2.5-3.5, MAE ~1.8-2.2, R² ~0.35-0.45
- **Neural Network**: RMSE ~2.3-3.2, MAE ~1.7-2.0, R² ~0.40-0.50

Performance varies by position:
- **Forwards**: Best predictions (high variance in points)
- **Goalkeepers**: Consistent but narrow range
- **Midfielders/Defenders**: Moderate variance

## API Integration

### Startup

Add to `api_main.py` lifespan:
```python
from ml.api_integration import MLAPIIntegration, register_ml_routes

ml_integration = MLAPIIntegration(neo4j_conn, query_executor)
ml_integration.load_predictor("ml/models/linear_regression_v1.pkl")

register_ml_routes(app, ml_integration)
```

### Endpoints

#### 1. Predict Single Player
```bash
POST /api/ml/predict/player
{
  "player_name": "Mohamed Salah",
  "player_data": {}  # Optional, fetches from Neo4j if empty
}

Response:
{
  "player_name": "Mohamed Salah",
  "predicted_points": 8.5,
  "features_used": {"form": 7.2, "goals_scored": 1.25},
  "model_version": "v1"
}
```

#### 2. Predict Top Performers
```bash
POST /api/ml/predict/top-performers
{
  "position": "FWD",  # Optional: GK/DEF/MID/FWD
  "top_k": 10,
  "season": "2025-26"  # Optional
}

Response:
{
  "predictions": [
    {"player_name": "Erling Haaland", "predicted_points": 9.2, ...},
    {"player_name": "Harry Kane", "predicted_points": 8.8, ...}
  ],
  "metadata": {"total_players_analyzed": 150}
}
```

#### 3. Predict Best Value
```bash
POST /api/ml/predict/best-value
{
  "position": "MID",
  "max_price": 8.0,  # £8.0m max
  "top_k": 5
}

Response:
[
  {
    "name": "Player X",
    "predicted_points": 6.5,
    "value": 6.5,
    "points_per_million": 1.0
  }
]
```

#### 4. ML Status
```bash
GET /api/ml/status

Response:
{
  "predictor_loaded": true,
  "model_type": "linear",
  "endpoints": [...]
}
```

## Frontend Integration

### Example: Display Predictions in Q&A

In `QAAssistant.tsx`:
```typescript
const fetchMLPredictions = async (position: string) => {
  const response = await fetch('/api/ml/predict/top-performers', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ position, top_k: 5 })
  });
  const data = await response.json();
  return data.predictions;
};
```

### Example: Settings Page Toggle

Add ML predictions toggle in Settings:
```typescript
<Toggle 
  label="Show ML Predictions" 
  checked={showMLPredictions}
  onChange={setShowMLPredictions}
/>
```

## Retraining

### When to Retrain
- **Weekly**: After each gameweek to include latest results
- **Monthly**: For major meta changes
- **Season Start**: With new season data

### Retraining Script
```bash
# Fetch latest data from Neo4j
python scripts/export_latest_data.py

# Retrain models
python ml/train.py

# Restart API to load new models
```

### Automated Retraining (Future)
- Schedule weekly training job
- Deploy new models via CI/CD
- A/B test model versions

## Testing

Before pushing to production:

```bash
# Test feature engineering
python -c "from ml.feature_engineering import FeatureEngineer; fe = FeatureEngineer(); print('✅ Features OK')"

# Test predictor loading
python -c "from ml.predictor import FPLPredictor; pred = FPLPredictor('ml/models/linear_regression_v1.pkl'); print('✅ Predictor OK')"

# Test API endpoints (requires running server)
curl -X POST http://localhost:8000/api/ml/status
```

## Limitations

1. **Cold Start**: New players have no historical data → use league averages
2. **Injuries/Rotation**: Model doesn't know about team news
3. **Fixture Difficulty**: Opponent strength not fully captured
4. **Variance**: High variance in low-scoring positions (GK)
5. **Sample Size**: Predictions improve with more data per player

## Future Improvements

- [ ] Add fixture difficulty rating (FDR) as feature
- [ ] Incorporate team news/injuries from external API
- [ ] Ensemble models (combine Linear + Neural Network)
- [ ] Confidence intervals with quantile regression
- [ ] Transfer recommendations (predict price changes)
- [ ] Captain picks (highest ceiling vs floor)
- [ ] Bench order optimization

## References

- Original notebook: `.github/workflows/ML/milestone_1.ipynb`
- scikit-learn docs: https://scikit-learn.org/
- TensorFlow/Keras: https://www.tensorflow.org/

## Contact

For questions or improvements, see the project's main README.
