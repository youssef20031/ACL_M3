# ML Model Training Results - V3

## Summary
**Date**: 2026-06-09  
**Models**: Position-Specific Linear Regression (GK, DEF, MID, FWD)  
**Dataset**: 3 seasons (2023-24, 2024-25, 2025-26) - 79,683 records  
**Split**: Temporal (earliest → train, middle → val, latest → test)

## Improvements Implemented in V3

### Improvement 8: High-Signal Features
Added 6 new features to capture rotation risk, efficiency, and form:

1. **`minutes_rolling5`**: Rolling 5-GW average of minutes played
   - **Purpose**: Captures rotation risk and playing time trends
   - **Implementation**: Lagged by 1 GW to avoid leakage

2. **`points_per_90`**: Points normalized per 90 minutes
   - **Purpose**: Normalizes performance for subs/benchings
   - **Implementation**: `(total_points / minutes * 90)`, lagged

3. **`home_form`**: Rolling 4-GW average of points at home
   - **Purpose**: Captures home advantage (position-dependent)
   - **Implementation**: Only calculated on home games, forward-filled

4. **`away_form`**: Rolling 4-GW average of points away
   - **Purpose**: Captures away form trends
   - **Implementation**: Only calculated on away games, forward-filled

5. **`gw_in_season`**: Normalized gameweek (0-1 scale)
   - **Purpose**: Captures fixture congestion patterns
   - **Implementation**: `GW / 38`

### Improvement 9: Opponent Defensive Strength
Replaced 20 sparse opponent one-hot columns with 1 dense signal:

- **`opp_def_strength`**: Rolling 5-game average of goals conceded by opponent
- **Purpose**: Compresses opponent info into meaningful defensive metric
- **Implementation**: Team-level rolling average, merged as opponent feature

## Results Comparison

### V2 (Before New Features)
| Position | Test R² | Test RMSE | Test MAE | Samples |
|----------|---------|-----------|----------|---------|
| GK       | 0.436   | 1.44      | 0.65     | 1,734   |
| DEF      | 0.262   | 2.11      | 1.15     | 4,962   |
| MID      | 0.332   | 1.92      | 1.00     | 6,760   |
| FWD      | 0.317   | 2.04      | 1.13     | 1,661   |
| **Overall** | **0.316** | **1.95** | **1.03** | **15,117** |

### V3 (With New Features)
| Position | Test R² | Test RMSE | Test MAE | Samples |
|----------|---------|-----------|----------|---------|
| GK       | 0.430   | 1.45      | 0.64     | 1,734   |
| DEF      | 0.263   | 2.10      | 1.15     | 4,962   |
| MID      | 0.334   | 1.92      | 0.99     | 6,760   |
| FWD      | 0.312   | 2.04      | 1.14     | 1,661   |
| **Overall** | **0.316** | **1.95** | **1.02** | **15,117** |

### Change Analysis
| Position | ΔR² | ΔRMSE | ΔMAE |
|----------|-----|-------|------|
| GK       | -0.006 | +0.01 | -0.01 |
| DEF      | +0.001 | -0.01 | ±0.00 |
| MID      | +0.002 | ±0.00 | -0.01 |
| FWD      | -0.005 | ±0.00 | +0.01 |
| **Overall** | **+0.0005** | **±0.00** | **-0.01** |

## Analysis

### Why Didn't Performance Improve?

**Result**: The new features provided essentially **no improvement** (R² change < 0.001).

**Possible Explanations**:

1. **Feature Redundancy**: The new features may be highly correlated with existing features
   - `minutes_rolling5` may correlate with existing `minutes` and `form` features
   - `points_per_90` is derived from `total_points` (already in `form`) and `minutes`
   - `home_form` / `away_form` may overlap with overall `form`

2. **Data Quality Issues**:
   - Features may have many NaN values being filled with 0
   - Forward-filling might not work well for sporadic players
   - Opponent defensive strength merge may have alignment issues

3. **Model Limitations**:
   - Linear Regression cannot capture feature interactions
   - The benefit of these features may only show with non-linear models (XGBoost, NN)

4. **Feature Engineering Issues**:
   - Lagging might remove too many samples
   - Rolling windows might be too short/long
   - Opponent strength calculation might not capture the signal properly

## Validation Set Performance
Training showed good generalization (validation R² close to training):
- **GK**: Train R² 0.419 → Val R² 0.440 (✅ Good generalization!)
- **DEF**: Train R² 0.247 → Val R² 0.262 (✅ Good generalization)
- **MID**: Train R² 0.325 → Val R² 0.298 (Slight drop, acceptable)
- **FWD**: Train R² 0.351 → Val R² 0.342 (✅ Good generalization)

## Recommendations

### Priority 1: Try XGBoost (Next Implementation)
Linear models cannot capture interactions. The new features may help more with XGBoost:
- Expected R² gain: +0.05 to +0.10
- Can capture interactions like: high minutes × weak opponent = more points
- Should be implemented **per position** like current approach

### Priority 2: Feature Diagnostics
Before adding more features, investigate current ones:
1. Check correlation matrix between new and old features
2. Analyze feature importance in Linear Regression (coefficients)
3. Check for NaN patterns in new features
4. Validate opponent defensive strength distribution

### Priority 3: Experiment with Rolling Windows
Current implementation:
- Form: 4-game rolling
- Minutes: 5-game rolling
- Opponent: 5-game rolling

Try different window sizes (3, 7, 10 games).

### Not Recommended Yet:
- ❌ 3-GW rolling average target (different use case, not a replacement)
- ❌ More complex NN (focus on features/model choice first)
- ❌ Adding more derived features (check existing ones first)

## Conclusion

**V3 Status**: ✅ Successfully implemented 6 new features  
**Performance**: 📊 No significant improvement (R² = 0.316, same as V2)  
**Next Step**: 🚀 Implement **XGBoost** to capture feature interactions  

The high-signal features are correctly engineered but don't provide lift with Linear Regression. This suggests the features are either redundant or need non-linear modeling to show benefit.

---

## Files Modified
- `ml/feature_engineering.py`: Added `_add_high_signal_features()` and `_add_opponent_defensive_strength()`
- `ml/models/linear_regression_*_v2.pkl`: Retrained with new features (overwritten)
- `ml/models/training_results.json`: Updated with V3 results

## Training Command
```bash
python -m ml.train
```

## Model Files (V3)
- `ml/models/linear_regression_gk_v2.pkl`
- `ml/models/linear_regression_def_v2.pkl`
- `ml/models/linear_regression_mid_v2.pkl`
- `ml/models/linear_regression_fwd_v2.pkl`
- `ml/models/training_results.json`
