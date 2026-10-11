# Promoted Teams Implementation - Complete Guide

## Quick Start

### Training Pipeline
```bash
# Option 1: Run complete pipeline
python scripts/train_all_models.py

# Option 2: Run steps individually

# Phase 1: Championship data (takes ~2 hours with rate limiting)
python ml/data/championship_pipeline.py

# Phase 2: Minutes models
python ml/train_minutes_models.py

# Phase 3: Archetype database
python ml/data/build_archetype_database_v2.py

# Create EPL team stats (required for Phase 4)
python scripts/aggregate_epl_team_stats.py

# Phase 4: Team translation
python ml/translation/model_a_team.py

# Phase 5: Player translation
python ml/translation/model_b_player.py

# Phase 10: Validation
python ml/validation/walk_forward_promoted.py
```

### Deployment for 2026-27 Season
```bash
# Configure for Bristol City, Ipswich Town, Hull City
python scripts/deploy_promoted_teams_2026.py
```

## Architecture Overview

### Five-Model System

**Model C: P(start)** - Probability player starts match
- Type: XGBoost classifier + Platt scaling
- Training: 69K player-GW records
- Target: AUC ≥ 0.80
- Output: Calibrated probability [0, 1]

**Model D: E[minutes|start]** - Expected minutes if starting
- Type: XGBoost regressor
- Training: 45K started matches
- Target: RMSE ≤ 12 minutes
- Output: Minutes [0, 90]

**Model A: Team Translation** - Championship → EPL team strength
- Type: Ridge regression (multi-output)
- Training: ~30 promoted teams
- Validation: LOOCV
- Outputs: attack_strength, defense_strength, points_per_game

**Model B: Player Translation** - Championship → EPL player per-90
- Type: Ridge regression per position
- Training: 80-100 promoted players
- Segmentation: Position × age × archetype
- Outputs: EPL-equivalent per-90 stats

**Archetype Database v2** - Historical promoted player embeddings
- Size: 80-100 players (2016-2026)
- Clustering: K-means (k=8 archetypes)
- Embedding: 14-dim per-90 feature vectors
- Usage: K=10 nearest neighbor matching

## Data Pipeline

### Phase 1: Championship Data Ingestion

**Source**: FBref.com (Championship, competition ID=10)

**Seasons**: 2019-2020 through 2025-2026 (7 seasons)

**Player Stats Categories**:
- Standard: goals, assists, minutes
- Shooting: shots, xG
- Passing: key passes, completion %
- Defense: tackles, interceptions
- Possession: dribbles, progressive carries
- GCA: goal/shot-creating actions
- Keepers: saves, clean sheets, PSxG

**Rate Limiting**: 3+ seconds between requests (FBref policy)

**Output Files**:
```
ml/data/championship_player_stats.csv  (~3,600 rows)
ml/data/championship_team_stats.csv    (~144 rows)
ml/data/promoted_teams_history.json
```

**Promoted Teams (Hardcoded)**:
```python
{
    '2016-2017': ['Newcastle', 'Brighton', 'Huddersfield'],
    '2017-2018': ['Wolverhampton', 'Cardiff', 'Fulham'],
    '2018-2019': ['Norwich', 'Sheffield United', 'Aston Villa'],
    '2019-2020': ['Leeds', 'West Brom', 'Fulham'],
    '2020-2021': ['Norwich', 'Watford', 'Brentford'],
    '2021-2022': ['Fulham', 'Bournemouth', 'Nottingham Forest'],
    '2022-2023': ['Burnley', 'Sheffield United', 'Luton'],
    '2023-2024': ['Leicester', 'Ipswich', 'Southampton'],
    '2024-2025': ['Leeds', 'Burnley', 'Sunderland'],
    '2025-2026': ['Bristol City', 'Ipswich', 'Hull City']
}
```

## Data Leakage Prevention

### Critical Exclusions (23 columns)

**Current GW outcomes** (LEAKAGE):
- goals_scored, assists, clean_sheets
- bonus, bps, saves
- penalties_saved, penalties_missed
- yellow_cards, red_cards, own_goals
- goals_conceded

**Target variables**:
- minutes, starts, started, total_points

**Lookahead features**:
- xP, expected_assists, expected_goals
- transfers_in, selected

**Match outcomes**:
- team_a_score, team_h_score

### Lagging Requirements

All derived features must use `.shift(1)`:
```python
df['started_last_match'] = df.groupby('name')['started'].shift(1)
df['avg_minutes_last_3'] = df.groupby('name')['minutes'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
```

### Verification
```bash
python ml/test_minutes_leakage.py  # Should pass all tests ✅
```

## Prediction Flow

### Standard EPL Players
```
1. Feature engineering (lagged, no leakage)
2. Model C → P(start)
3. Model D → E[minutes|start]
4. Expected minutes = P(start) × E[minutes|start]
5. Points model → predicted_points (total, not per-minute)
6. Return: (predicted_points, minutes_context)
```

### Promoted Players (GW 1-12 Cold Start)
```
1. Load Championship stats
2. Model A: Translate team strength
3. Model B: Translate player per-90 stats
4. Find K=10 similar archetypes (expanded DB)
5. Weighted prediction from archetype debuts
6. Apply GW-based confidence decay:
   - GW 1-6: 60% confidence
   - GW 7-12: 75% confidence
   - GW 13+: 90% confidence (standard model)
```

### Integration Point (FPLPredictor)
```python
from ml.predictor import FPLPredictor

predictor = FPLPredictor(
    model_path='ml/models/xgboost_fwd_v1.pkl',
    enable_minutes_models=True,        # Model C & D
    enable_promoted_teams_handler=True # Uses v2 handler if available
)

result = predictor.predict_next_gameweek(player_data)
# Returns: PlayerPrediction with minutes context
```

## File Structure

```
ml/
├── minutes_model.py              # Model C & D implementation
├── train_minutes_models.py       # Training script
├── test_minutes_leakage.py       # Leakage verification ✅
├── predictor.py                  # Main predictor (modified)
├── feature_engineering.py        # Modified with minutes features
├── promoted_teams_handler.py     # V1 handler (6 archetypes)
├── promoted_teams_handler_v2.py  # V2 handler (80+ archetypes)
│
├── data/
│   ├── __init__.py
│   ├── fbref_scraper.py          # FBref scraper with rate limiting
│   ├── championship_pipeline.py  # Orchestrator
│   └── build_archetype_database_v2.py  # Automated archetype builder
│
├── translation/
│   ├── __init__.py
│   ├── model_a_team.py          # Team strength translation
│   └── model_b_player.py        # Player per-90 translation
│
├── validation/
│   ├── __init__.py
│   └── walk_forward_promoted.py # Historical validation
│
└── models/                       # Saved models (after training)
    ├── minutes_pstart_{pos}_v1.pkl
    ├── minutes_eminutes_{pos}_v1.pkl
    ├── promoted_archetypes_v2.pkl
    ├── translation_model_a_v1.pkl
    └── translation_model_b_v1.pkl

scripts/
├── train_all_models.py          # Master training script
├── aggregate_epl_team_stats.py  # EPL team stats builder
└── deploy_promoted_teams_2026.py # Deployment for 2026-27

documentation/
├── FINAL_IMPLEMENTATION_SUMMARY.md  # Detailed summary
├── IMPLEMENTATION_STATUS.md         # Progress tracking
├── ml/MINUTES_MODELS_README.md      # Technical docs
└── README_PROMOTED_TEAMS_IMPLEMENTATION.md  # This file
```

## Performance Targets

| Model | Metric | Target | Status |
|-------|--------|--------|--------|
| Model C (P(start)) | AUC-ROC | ≥ 0.80 | To be validated |
| Model D (E[minutes\|start]) | RMSE | ≤ 12 min | To be validated |
| Model A (Team) | PPG MAE | ≤ 0.25 | To be validated |
| Model B (Player) | Per-90 MAE | TBD | To be validated |
| Walk-Forward | Overall MAE | < 2.0 pts | To be validated |

## Known Limitations

### Minutes Integration Issue
**Problem**: Existing points models predict `total_points`, not `points_per_minute`.

**Current Approach**: Minutes models provide context only, not direct multiplication.

**Future Fix** (v2):
```python
# Retrain points models with points-per-90 targets
target = (total_points / minutes) * 90

# Then at inference:
pp90_prediction = points_model.predict(features)
expected_points = (expected_minutes / 90) * pp90_prediction
```

### Small Training Sets
- Model A: ~30 teams (use Ridge, not XGBoost)
- Model B: ~20-30 per position (use Ridge, aggressive regularization)
- Solution: LOOCV validation, report confidence intervals

### Championship Data Availability
- FBref has data from 2019-20 onwards
- Earlier seasons may require manual data collection
- Transfer activity data (Transfermarkt) not implemented

## Deployment Checklist for 2026-27

### Pre-Season (August 2026)
- [ ] Scrape 2025-26 Championship data for Bristol City, Ipswich Town, Hull City
- [ ] Aggregate their key players (≥900 Championship minutes)
- [ ] Run Model A to predict team strength
- [ ] Update `PROMOTED_TEAMS_HISTORY` in `championship_pipeline.py`
- [ ] Test predictions on sample players

### Season Start (GW 1)
- [ ] Enable promoted teams handler v2
- [ ] Log all predictions with confidence intervals
- [ ] Monitor prediction accuracy vs actuals

### Early Season (GW 1-6)
- [ ] Cold start phase - high uncertainty expected
- [ ] Collect feedback on prediction quality
- [ ] Flag systematic over/under-predictions

### Mid Season (GW 7-12)
- [ ] Transition phase - blending archetype + real data
- [ ] Begin incremental model updates
- [ ] Validate confidence decay is working

### Late Season (GW 13+)
- [ ] Standard predictions - sufficient EPL data
- [ ] Promoted teams handler auto-disables
- [ ] Retrospective analysis for next season

## Troubleshooting

### "No Championship data found"
```bash
# Run data pipeline
python ml/data/championship_pipeline.py
```

### "Model not trained"
```bash
# Check if model files exist
ls ml/models/minutes_pstart_*.pkl
ls ml/models/translation_model_*.pkl

# If missing, train models
python ml/train_minutes_models.py
python ml/translation/model_a_team.py
```

### "Data leakage detected"
```bash
# Run leakage test
python ml/test_minutes_leakage.py

# Check exclusion list in:
# - ml/minutes_model.py (_train_model_c, _train_model_d)
# - ml/feature_engineering.py (prepare_features)
```

### "Archetype database empty"
```bash
# Rebuild archetype database
python ml/data/build_archetype_database_v2.py
```

## Contact & Support

**Documentation**:
- Technical: `ml/MINUTES_MODELS_README.md`
- Summary: `FINAL_IMPLEMENTATION_SUMMARY.md`
- Tasks: `tasks_list`

**Testing**:
- Leakage: `python ml/test_minutes_leakage.py`
- Validation: `python ml/validation/walk_forward_promoted.py`

**Critical Decision Log**:
- Avoided flawed `points_per_minute` calculation
- Used Ridge (not XGBoost) for small datasets
- Implemented comprehensive leakage prevention
- Minutes models provide context only (v1)
