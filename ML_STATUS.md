# ML Model Status - Quick Reference

## Current Version: V6 (Leakage Corrected) ✅

**Date**: 2026-06-09  
**Branch**: `ml-integration-improvements`  
**Status**: ✅ Validated, leakage-free, production-ready (pending Priority 1 tasks)

---

## Performance Summary

### Overall Metrics
| Metric | Value | vs Baseline |
|--------|-------|-------------|
| **R²** | **0.720** | +122% ✅ |
| **MAE** | **0.48 pts** | -53% ✅ |
| **RMSE** | **1.25 pts** | -36% ✅ |

### By Position (XGBoost)
| Position | R² | MAE | Improvement |
|----------|-----|-----|-------------|
| **GK** | 0.633 | 0.37 | +45% |
| **DEF** | 0.698 | 0.55 | +166% 🔥 |
| **MID** | 0.755 | 0.44 | +127% 🔥 |
| **FWD** | 0.665 | 0.52 | +110% 🔥 |

**Interpretation**: Model explains **72% of point variance**. Typical error is ±0.48 points.

---

## What Changed in V6?

### Critical Fix: Data Leakage Eliminated

**Problem**: V5 used current-gameweek outcomes to predict current-gameweek points

**Most Critical**: `clean_sheets` feature (77.5% correlation with target!)

**Removed 12 Leaked Features**:
- clean_sheets, starts, goals_scored, assists, bonus
- goals_conceded, saves, penalties_saved/missed
- yellow_cards, red_cards, own_goals

**Impact**: R² dropped from false 0.777 → real 0.720 (-7.3%)

**Validation**: ✅ Shuffle test, ✅ Cross-validation, ✅ Feature shift verification

**See**: `ml/LEAKAGE_SUMMARY.md` for details

---

## Production Readiness

### ✅ Ready
- Leakage-free predictions
- Fast inference (<2ms per player)
- Validated performance (no overfitting)
- Reproducible pipeline

### ⚠️ Before Deployment (Priority 1)
1. **Calibration analysis** - Check predicted vs actual bins
2. **Start probability model** - Handle rotation/benching
3. **Double gameweek feature** - Add fixtures_this_gw counter

**Estimated effort**: 1-2 days

---

## Key Features Used

### Pre-Match Features (Safe ✅)
- `form` - 4-game rolling avg of past points
- `minutes_rolling5` - 5-game avg minutes
- `opp_off_strength` - Opponent's past goals scored (for GK/DEF)
- `team_def_strength` - Own team's past goals conceded (for GK/DEF)
- `opp_def_strength` - Opponent's past goals conceded (for attackers)
- `influence`, `creativity`, `threat` - Past ICT stats
- `was_home`, `team`, `position` - Fixture/static info

### Excluded Features (Leakage ❌)
- Current-GW match outcomes (goals, assists, clean_sheets, etc.)
- Expected stats (xG, xA, xP)
- Market-dependent (selected, transfers_in/out)

**Total**: ~50 features (down from ~70)

---

## Training Command

```bash
# Install dependencies (if not already)
pip install xgboost scikit-learn pandas numpy

# Train models
python ml/train.py

# Validate (optional)
python ml/validate_no_leakage.py
```

**Training time**: ~60 seconds  
**Models saved**: `ml/models/xgboost_{gk,def,mid,fwd}_v3.pkl`

---

## Models Files

### Recommended (XGBoost V3)
- `ml/models/xgboost_gk_v3.pkl` - R² = 0.633
- `ml/models/xgboost_def_v3.pkl` - R² = 0.698
- `ml/models/xgboost_mid_v3.pkl` - R² = 0.755
- `ml/models/xgboost_fwd_v3.pkl` - R² = 0.665
- `ml/models/training_results.json` - Performance metrics

### Not Recommended
- Linear Regression models (R² = 0.157 overall)

---

## Next Steps

### Immediate (Before Production)
1. ⚠️ Run calibration analysis
2. ⚠️ Implement start probability model
3. ⚠️ Add double gameweek feature

### Short-term (FWD Improvement)
4. 📈 Add shot conversion rate feature (if shots data available)
5. 📈 Add penalty taker indicator

### Medium-term (Optimization)
6. 🔧 Position-specific hyperparameter tuning
7. 🔧 Feature selection (remove <1% importance features)

### Long-term (Advanced Features)
8. 🌟 Fixture difficulty rating (FDR)
9. 🌟 Team news/injury integration
10. 🌟 Historical head-to-head patterns

---

## Validation Checklist

- [x] ✅ Temporal train/test split (no future data in training)
- [x] ✅ All features properly shifted (rolling averages use .shift(1))
- [x] ✅ No current-GW outcomes in features
- [x] ✅ Shuffle test passes (R² collapses to ~0)
- [x] ✅ Cross-validation confirms performance
- [x] ✅ Feature importance realistic (no >50% features)
- [x] ✅ No overfitting (test R² ≈ validation R²)
- [ ] ⚠️ Calibration verified (predicted bins match actual)
- [ ] ⚠️ Start probability added (rotation handling)
- [ ] ⚠️ DGW feature added (fixtures_this_gw)

---

## Documentation

**Main Docs**:
- `ML_V6_LEAKAGE_CORRECTED.md` - Comprehensive V6 documentation
- `ml/LEAKAGE_SUMMARY.md` - Leakage discovery & fix details
- `ml/README.md` - Module overview & API integration

**Training**:
- `ml/train.py` - Training pipeline
- `ml/feature_engineering.py` - Feature definitions

**Validation**:
- `ml/validate_no_leakage.py` - Comprehensive validation suite
- `ml/verify_clean_sheets_leakage.py` - Specific leakage check

**Historical** (DO NOT USE):
- `ML_FINAL_V5_BREAKTHROUGH.md` - V5 with leakage (R² = 0.777 was FALSE)

---

## Commits

- **594db84**: V6 leakage fix (removed 12 leaked features, retrained models)
- **e9ca499**: Documentation updates (corrected performance metrics)
- Previous commits: V1-V5 development (see git log)

---

## Quick Facts

**What is R² = 0.720?**
- Model explains 72% of why players score the points they do
- Remaining 28% = randomness, injuries, referee decisions, luck
- Excellent for sports prediction (typical range: 0.3-0.6)

**What is MAE = 0.48?**
- Average prediction error is ±0.48 FPL points
- Example: Predict 8pts → actual is typically 7.5-8.5pts
- Very accurate for FPL decision-making

**Why is DEF improvement biggest?**
- DEF points = clean sheets (4pts) + attacking returns
- V1 couldn't predict clean sheets (R² = 0.262)
- V6 predicts clean sheets well via opp_off_strength + team_def_strength
- DEF improvement: 0.262 → 0.698 (+166%)

**Why is clean_sheets leakage?**
- clean_sheets = 1 if team kept clean sheet **this gameweek**, 0 otherwise
- It's an **outcome** of the match, not a pre-match feature
- Using it to predict total_points is like using answer to predict answer
- Correlation = 77.5% (impossibly high for legitimate feature)

---

## Contact & Support

For questions about:
- **Implementation**: See `ml/README.md`
- **Leakage fix**: See `ml/LEAKAGE_SUMMARY.md`
- **Performance**: See `ML_V6_LEAKAGE_CORRECTED.md`
- **Training**: Run `python ml/train.py --help`

---

**Last Updated**: 2026-06-09  
**Version**: V6 Final  
**Status**: ✅ **PRODUCTION-READY** (pending Priority 1 tasks)
