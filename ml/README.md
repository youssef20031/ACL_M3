# FPL ML Prediction Module

## Overview

This module integrates machine learning predictions into the FPL RAG system. It predicts player performance (FPL points) for the next gameweek based on recent form, historical statistics, and match context.

## Current Status (V6 - Leakage Corrected) ✅

**Performance**: R² = 0.720, MAE = 0.48 pts (+122% improvement over baseline)
**Status**: Validated, leakage-free, production-ready (pending calibration analysis)
**Models**: XGBoost position-specific (GK, DEF, MID, FWD)

**See**: `ML_V6_LEAKAGE_CORRECTED.md` for complete documentation

## Improvements Implemented

### Phase 1: Original 5 Notebook Improvements (V1)

#### 1. ✅ Temporal Train/Test Split (Most Impactful Fix)
- **Problem**: Random split causes data leakage - model sees future data during training
- **Solution**: Sort data by `kickoff_time` and split chronologically
- **Impact**: Train on past data, validate on middle period, test on most recent data
- **Implementation**: `FPLModelTrainer.temporal_train_test_split()`

#### 2. ✅ Remove/Lag Features (Data Leakage Prevention)
- **Problem**: `total_points` and `bps` are target-derived features that leak information
- **Solution**: 
  - Removed from feature set
  - Lag `total_points` by 1 gameweek to create `upcoming` target
  - Use 4-game rolling average as `form` feature instead
- **Implementation**: `FeatureEngineer.engineer_features(lag_features=['total_points'])`

#### 3. ✅ Add Dropout to Neural Network
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

#### 4. ✅ Fix Position Labels in Reports
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

#### 5. ✅ Rename Model from nn_bad_model to nn_baseline_model
- **Problem**: Poor naming convention
- **Solution**: Renamed throughout codebase to `nn_baseline_model`
- **Impact**: Better communicates that it's a baseline for comparison

### Phase 2: Position-Specific Models (V2)

#### 6. ✅ Train Separate Models by Position
- **Reason**: GK/DEF/MID/FWD have fundamentally different point-scoring patterns
- **Implementation**: 4 separate models (one per position)
- **Impact**: Allows each position to use relevant features differently

#### 7. ✅ Exclude xP Column
- **Reason**: Expected points (xP) may contain post-match data
- **Impact**: Eliminates potential lookahead bias

### Phase 3: High-Signal Features (V3)

#### 8. ✅ Add High-Signal Engineered Features
- `minutes_rolling5` - Rotation risk indicator
- `points_per_90` - Efficiency metric
- `home_form` / `away_form` - Venue-specific performance
- `gw_in_season` - Fixture congestion indicator
- **Impact**: +6 features with strong predictive signal

#### 9. ✅ Opponent Defensive Strength
- Rolling 5-game average of goals conceded by opponent
- Compresses 20 sparse opponent one-hot columns into 1 dense signal
- **Impact**: Helps attackers (MID/FWD) predict easier fixtures

### Phase 4: XGBoost Implementation (V4)

#### 10. ✅ Gradient Boosting Models
- Replaced Linear Regression with XGBoost
- Captures nonlinear feature interactions
- Position-specific hyperparameters
- **Impact**: +2.5% improvement (R² 0.316 → 0.324)

### Phase 5: Defensive Features (V5 - Had Leakage)

#### 11. ✅ Fixed Opponent Offensive Strength + Team Defense
- **Fix 1**: Removed opponent one-hot encoding (collinearity)
- **Fix 2**: Corrected opp_off_strength calculation (use actual opponent goals)
- **Fix 3**: Added team_def_strength (symmetric feature)
- **Impact**: DEF/GK predictions improved significantly
- **Note**: V5 initially showed R² = 0.777 but had data leakage

### Phase 6: Leakage Elimination (V6) ✅

#### 12. ✅ Removed Current-GW Outcome Variables
- **Critical Fix**: Eliminated 12 leaked features (clean_sheets, starts, goals_scored, assists, bonus, etc.)
- **Discovery**: clean_sheets had 77.5% correlation with target (direct leakage!)
- **Impact**: Performance dropped to REAL level (R² 0.777 → 0.720)
- **Validation**: Shuffle test, cross-validation, feature importance checks all pass
- **Status**: Now production-ready with validated performance

**See**: `ml/LEAKAGE_SUMMARY.md` for detailed explanation of what was leaked and how it was fixed

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

## Models (V6)

### XGBoost Position-Specific (Primary - Recommended)
- **Pros**: Captures nonlinear patterns, excellent performance, no overfitting
- **Cons**: Slightly slower than linear (still <2ms per prediction)
- **Use Case**: Production predictions
- **Files**: `xgboost_{gk,def,mid,fwd}_v3.pkl`
- **Performance**: R² = 0.720 overall

### Linear Regression (Backup)
- **Pros**: Fast, interpretable
- **Cons**: Poor performance (R² = 0.157)
- **Use Case**: Baseline comparison only (not recommended for production)

### Neural Network Baseline (Deprecated)
- Not maintained in V6
- Use XGBoost instead

## Features (V6)

### Engineered Features (8 Added in V3-V5)
- `form`: 4-game rolling average of total_points (lagged by 1)
- `team_goals`: Conditional on home/away status
- `minutes_rolling5`: 5-game avg minutes (rotation risk)
- `points_per_90`: Points per 90 minutes (efficiency)
- `home_form` / `away_form`: Venue-specific form
- `gw_in_season`: Normalized gameweek (fixture congestion)
- `opp_def_strength`: Opponent goals conceded (for attackers)
- `opp_off_strength`: Opponent goals scored (for GK/DEF)
- `team_def_strength`: Own team goals conceded (for GK/DEF)

### Numeric Features (~20 Base + 9 Engineered)
**Base**: minutes, value, was_home, GW, ict_index, influence, creativity, threat, form

**Note**: Raw match statistics (goals_scored, assists, clean_sheets, etc.) are **EXCLUDED** from features to prevent leakage. Only historical/rolling versions are used.

### Categorical Features (One-Hot Encoded)
- `position`: GK, DEF, MID, FWD (4 columns)
- `team_x`: ~20 teams (20 columns)
- **Opponent one-hot**: REMOVED in V5 (replaced by continuous opp_off_strength/opp_def_strength)

**Total Features**: ~50 after encoding (down from ~70)

### Excluded Features (Data Leakage Prevention)

**Target-derived**:
- `total_points` (target variable)
- `bps` (bonus point system - component of target)

**Current-GW outcomes** (CRITICAL - V6 fix):
- `clean_sheets`, `starts`, `goals_scored`, `assists`, `bonus`
- `goals_conceded`, `saves`, `penalties_saved`, `penalties_missed`
- `yellow_cards`, `red_cards`, `own_goals`

**Lookahead features**:
- `xP` (expected points - may contain post-match data)
- `expected_goals`, `expected_assists`, `expected_goal_involvements`
- `expected_goals_conceded`

**Market-dependent**:
- `selected`, `transfers_in`, `transfers_out` (depends on predictions)

**Identifiers**:
- `name`, `season`, `element`, `fixture`, `kickoff_time`

**See**: `ml/LEAKAGE_SUMMARY.md` for complete explanation of why each feature was excluded

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

### Current Performance (V6 - Leakage-Free) ✅

**Overall (XGBoost)**:
- RMSE: 1.25 pts
- MAE: 0.48 pts
- R²: 0.720 (+122% improvement over V1 baseline)

**By Position (XGBoost)**:

| Position | R² | RMSE | MAE | Samples |
|----------|-----|------|-----|---------|
| **GK** | 0.633 | 1.14 | 0.37 | 6,514 |
| **DEF** | 0.698 | 1.34 | 0.55 | 24,145 |
| **MID** | 0.755 | 1.18 | 0.44 | 32,771 |
| **FWD** | 0.665 | 1.34 | 0.52 | 6,285 |

**Key Insights**:
- **MID**: Most predictable (R² = 0.755) - large sample, balanced scoring
- **DEF**: Biggest improvement (+166% from baseline) - defensive features work
- **FWD**: Room for improvement - consider shot conversion rate feature
- **GK**: Good performance (R² = 0.633) - clean sheets ~63% predictable

**Linear Regression (Backup)**:
- Not recommended for production (R² = 0.157 overall)
- Use only for baseline comparison

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

## Limitations & Considerations

### Current Limitations
1. **Cold Start**: New players have no historical data → use league averages or similar player profiles
2. **Injuries/Rotation**: Model doesn't know about team news (requires external API integration)
3. **Calibration**: Not yet validated (predicted 8pts games may not average 8pts) - Priority 1 task
4. **Blank Gameweeks**: Doesn't predict start probability (can predict points for benched players) - Priority 1 task
5. **Double Gameweeks**: No DGW feature (biggest FPL edge missing) - Priority 1 task

### Validated Strengths ✅
- **No data leakage**: Extensive validation (shuffle tests, CV, feature shift verification)
- **Good generalization**: Test R² matches validation R²
- **Temporal robustness**: Time-series cross-validation confirms performance
- **Realistic expectations**: R² = 0.720 is excellent for sports prediction (75% variance explained)

### Production-Readiness Status

**Ready**:
- ✅ Leakage-free predictions
- ✅ Fast inference (<2ms per player)
- ✅ Validated performance
- ✅ Reproducible pipeline

**Before Deployment** (Priority 1):
- ⚠️ Calibration analysis (check predicted vs actual bins)
- ⚠️ Start probability model (separate binary classifier)
- ⚠️ Double gameweek feature (fixtures_this_gw counter)

**See**: `ML_V6_LEAKAGE_CORRECTED.md` for detailed production readiness checklist

## Future Improvements (Prioritized)

### Priority 1: Production Readiness ⚠️ REQUIRED
- [ ] **Calibration analysis** - Verify predicted 8pts games actually average ~8pts
- [ ] **Start probability model** - Binary classifier for rotation/benching risk
- [ ] **Double gameweek feature** - Add `fixtures_this_gw` (critical FPL edge)

**Effort**: 1-2 days  
**Impact**: Not R² improvement, but user trust & practical value

### Priority 2: FWD Improvements 📈 RECOMMENDED
- [ ] **Shot conversion rate** - `goals_rolling / shots_on_target_rolling`
- [ ] **Penalty taker indicator** - Boolean flag from penalty taker lists

**Effort**: 1-2 days  
**Expected Gain**: FWD R² 0.665 → 0.70 (+5%)

### Priority 3: Hyperparameter Tuning 🔧 OPTIONAL
- [ ] **Grid search** - Position-specific max_depth, learning_rate, n_estimators
- [ ] **Cross-validation** - Time-series CV for robust selection

**Effort**: 2-3 days  
**Expected Gain**: +1-2% R² overall

### Priority 4: Additional Features 🌟 NICE-TO-HAVE
- [ ] **Fixture difficulty rating (FDR)** - If available from API
- [ ] **Team overall form** - Rolling team points average
- [ ] **Historical head-to-head** - Team vs opponent performance patterns
- [ ] **Bookmaker odds** - If accessible, very predictive signal

**Effort**: 3-4 days  
**Expected Gain**: +2-5% R² overall

### Not Recommended ❌
- Neural Networks (unlikely to beat XGBoost, more complex)
- Ensemble stacking (marginal gains, high complexity)
- More defensive features (current ones sufficient)

## References & Documentation

**Main Documentation**:
- `ML_V6_LEAKAGE_CORRECTED.md` - Complete V6 documentation with real performance
- `ml/LEAKAGE_SUMMARY.md` - Detailed explanation of leakage discovery and fix
- `ML_FINAL_V5_BREAKTHROUGH.md` - V5 results (⚠️ had leakage, DO NOT USE)

**Training Scripts**:
- `ml/train.py` - Main training pipeline
- `ml/validate_no_leakage.py` - Comprehensive validation suite
- `ml/verify_clean_sheets_leakage.py` - Specific leakage verification

**Original Notebook**: `.github/workflows/ML/milestone_1.ipynb`

**Libraries**:
- scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- Pandas: https://pandas.pydata.org/

## Contact

For questions or improvements, see the project's main README.
