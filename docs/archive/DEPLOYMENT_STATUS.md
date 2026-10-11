# FPL Promoted Teams - Deployment Status

**Date**: June 22, 2026
**Status**: IN PROGRESS - Training Execution Phase

---

## Summary

Implementation of promoted teams system is **complete** (7 of 10 phases). Currently executing training pipeline with bug fixes applied.

---

## Completed Work

### ✅ Phase 1-7, 9-10: Code Implementation
- All 37 files created and ready
- 11/11 integration tests passing
- Data leakage verification passing
- Documentation comprehensive

### ✅ Pre-Training Setup
- EPL team stats aggregated successfully (`epl_team_stats.csv`)
- Training data files present (87,087 records across 3 seasons)
- Feature engineering bugs fixed:
  - Categorical mappings now build during training
  - Position column preserved (not one-hot encoded)
  - Validation feature alignment fixed
  - XGBoost API updated (early_stopping_rounds moved to constructor)

---

## Current Training Status

### Minutes Models (Phase 2) - IN PROGRESS

**Status**: Debugging and fixing training pipeline

**Fixes Applied**:
1. ✅ Feature engineering now builds categorical mappings during training
2. ✅ Position column preserved for filtering
3. ✅ Validation set feature alignment (missing columns filled with 0)
4. ✅ XGBoost early_stopping_rounds parameter moved to constructor
5. ✅ Scripts updated for proper column handling

**Next**: Resume training execution

**Expected Output**:
- 8 model files (4 positions × 2 models)
- Model C: `minutes_pstart_{pos}_v1.pkl`
- Model D: `minutes_eminutes_{pos}_v1.pkl`
- Mappings files
- Metrics JSON

**Training Time**: ~30-45 minutes

---

## Remaining Training Tasks

### Priority 1: Complete Minutes Models Training
```bash
python ml/train_minutes_models.py
```

**Status**: Ready to resume after bug fixes

### Priority 2: Train Quantile Models (Phase 7)
```bash
python ml/quantile_models.py
```

**Status**: Code complete, ready to run
**Duration**: ~45-60 minutes
**Output**: 12 models (4 positions × 3 quantiles)

### Priority 3: Championship Data Collection (Phase 1)
```bash
python ml/data/championship_pipeline.py
```

**Status**: Code complete, requires execution
**Duration**: 2-3 hours (FBref rate limiting)
**Output**: 
- `championship_player_stats.csv` (~3,600 rows)
- `championship_team_stats.csv` (~144 rows)

### Priority 4: Build Archetype Database (Phase 3)
```bash
python ml/data/build_archetype_database_v2.py
```

**Status**: Code complete, requires Championship data
**Duration**: 5-10 minutes
**Output**: `promoted_archetypes_v2.pkl` (80-100 players)

### Priority 5: Train Translation Models (Phases 4-5)
```bash
# Model A: Team translation
python ml/translation/model_a_team.py

# Model B: Player translation  
python ml/translation/model_b_player.py
```

**Status**: Code complete, requires archetype database
**Duration**: 5-15 minutes total
**Output**:
- `translation_model_a_v1.pkl`
- `translation_model_b_v1.pkl`

### Priority 6: Walk-Forward Validation (Phase 10)
```bash
python ml/validation/walk_forward_promoted.py
```

**Status**: Code complete
**Duration**: 20-30 minutes
**Output**: Historical validation metrics, plots

---

## Deployment Configuration

### Target Season: 2026-27

**Promoted Teams**:
- Bristol City (1st)
- Ipswich Town (2nd)
- Hull City (3rd)

**Configuration File**: `ml/config/promoted_teams_2026.json`

**Deployment Script**: Ready
```bash
python scripts/deploy_promoted_teams_2026.py
```

---

## API Integration Status

### ✅ Auto-Upgrade Logic Implemented

The API will automatically detect and load v2 models:

```python
from ml.api_integration import load_predictor

# Auto-upgrades to v2 handler if models available
predictor = load_predictor('ml/models/xgboost_fwd_v1.pkl')
```

**Integration Points**:
- `ml/api_integration.py` - Modified with v2 handler detection
- `ml/predictor.py` - Integrated with minutes models
- `ml/promoted_teams_handler_v2.py` - Ready for activation

---

## Bug Fixes Applied During Training

### Issue 1: Categorical Mappings Not Initialized
**Problem**: FeatureEngineer tried to access mappings before building them
**Fix**: Modified `_encode_categorical` to build mappings during training
**File**: `ml/feature_engineering.py`
**Status**: ✅ Fixed

### Issue 2: Position Column Dropped
**Problem**: Position one-hot encoded and dropped, but needed for filtering
**Fix**: Don't encode position, keep it as-is
**File**: `ml/feature_engineering.py`
**Status**: ✅ Fixed

### Issue 3: Validation Feature Mismatch
**Problem**: Validation set missing team columns from training
**Fix**: Iterate through feature_cols and fill missing with 0
**Files**: `ml/minutes_model.py` (_train_model_c, _train_model_d)
**Status**: ✅ Fixed

### Issue 4: XGBoost API Change
**Problem**: `early_stopping_rounds` no longer accepted in fit()
**Fix**: Move to constructor parameters
**File**: `ml/minutes_model.py`
**Status**: ✅ Fixed

### Issue 5: EPL Team Stats Aggregation
**Problem**: Script looking for non-existent 'season' column
**Fix**: Process each file separately with season labels
**File**: `scripts/aggregate_epl_team_stats.py`
**Status**: ✅ Fixed and executed successfully

---

## Testing Status

### ✅ Data Leakage Tests
```bash
python ml/test_minutes_leakage.py
```
**Status**: All tests passing

### ✅ Integration Tests
```bash
python tests/test_promoted_teams_integration.py
```
**Status**: 11/11 tests passing

### Pending: Model Training Verification
After training completes, verify:
- Model C: AUC ≥ 0.80
- Model D: RMSE ≤ 12 minutes
- All 8 models saved successfully

---

## Production Checklist

### Before Season Start (August 2026)

- [ ] Complete all model training
- [ ] Scrape 2025-26 Championship data for promoted teams
- [ ] Test predictions on sample players
- [ ] Deploy to production API
- [ ] Configure monitoring/logging

### Season Timeline

**GW 1-6: Cold Start**
- Archetype matching + Championship scalars
- High uncertainty expected
- Log all predictions

**GW 7-12: Transition**  
- Blending phase
- Decreasing uncertainty
- Begin incremental updates

**GW 13+: Standard**
- Sufficient EPL data
- Handler auto-disables
- Full model accuracy

---

## Known Limitations

1. **Minutes Integration**: Context only (not direct multiplication)
   - Future: Retrain points models with points-per-90 targets

2. **Small Training Sets**:
   - Model A: ~30 teams (use Ridge + LOOCV)
   - Model B: ~20-30 per position
   - Mitigation: Report confidence intervals

3. **Championship Data**: FBref 2019-20 onwards only
   - Earlier seasons require manual collection

4. **Phase 8 Not Implemented**: Neo4j integration (optional)

---

## Next Immediate Actions

1. **Resume Minutes Models Training**
   - All bugs fixed
   - Ready to complete
   - Expected duration: 30-45 minutes

2. **Train Quantile Models**
   - Code complete
   - Ready to run after minutes models
   - Expected duration: 45-60 minutes

3. **Championship Data Collection** (Can run in parallel)
   - Long-running (2-3 hours)
   - Can be started while other models train
   - Required for translation models

4. **Complete Translation Models**
   - After Championship data + archetypes ready
   - Quick execution (10-15 minutes total)

---

## Documentation

- `PROMOTED_TEAMS_COMPLETE_SUMMARY.md` - Comprehensive guide
- `EXECUTION_SUMMARY.md` - Implementation summary
- `README_PROMOTED_TEAMS_IMPLEMENTATION.md` - Setup guide
- `DEPLOYMENT_STATUS.md` - This file
- `ml/MINUTES_MODELS_README.md` - Technical docs

---

## Contact

For issues or questions, refer to documentation or check integration test examples.

---

*Last Updated: June 22, 2026, 21:05*
*Training Status: Minutes models debugging complete, ready to resume*
*Deployment Target: Bristol City, Ipswich Town, Hull City (2026-27)*
