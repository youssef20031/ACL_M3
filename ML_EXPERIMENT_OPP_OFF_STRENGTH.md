# ML Experiment: Opponent Offensive Strength Feature

## Hypothesis
Adding opponent offensive strength (rolling 5-game goals scored) should help GK/DEF predictions by better capturing clean sheet probability.

**Rationale**:
- Defenders and goalkeepers heavily depend on clean sheets for points
- Opponent offensive strength indicates how likely opponent is to score
- Should complement existing `opp_def_strength` (which helps attackers)

---

## Implementation

### Feature Added
```python
opp_off_strength: Rolling 5-game average of goals scored by opponent team
Default value: 1.5 goals/game (league average)
```

### Expected Impact
- **GK**: +5-10% improvement (clean sheets are primary points source)
- **DEF**: +3-5% improvement (clean sheets + attacking returns)
- **MID**: Neutral or slight improvement
- **FWD**: Neutral

---

## Results

### XGBoost Performance (Test Set)

| Position | V4 (without) R² | V5 (with opp_off_strength) R² | ΔR² | % Change |
|----------|----------------|-------------------------------|-----|----------|
| GK | 0.450 | **0.444** | **-0.006** | **-1.3%** ❌ |
| DEF | 0.262 | **0.260** | **-0.002** | **-0.8%** ❌ |
| MID | 0.341 | **0.339** | **-0.002** | **-0.6%** ❌ |
| FWD | 0.340 | **0.333** | **-0.007** | **-2.1%** ❌ |
| **Overall** | **0.324** | **0.321** | **-0.003** | **-0.9%** ❌ |

### Linear Regression Performance (Test Set)

| Position | V4 R² | V5 R² | ΔR² | % Change |
|----------|-------|-------|-----|----------|
| GK | 0.430 | 0.430 | 0.000 | Flat |
| DEF | 0.263 | 0.263 | 0.000 | Flat |
| MID | 0.334 | 0.334 | 0.000 | Flat |
| FWD | 0.312 | 0.312 | 0.000 | Flat |

---

## Analysis

### ❌ Experiment Failed: Performance Decreased

**Key Finding**: Adding opponent offensive strength **hurt** model performance instead of helping.

### Why Did This Happen?

#### 1. **Feature Redundancy / Multicollinearity**
- We already have opponent team as one-hot encoding (20 features)
- We already have `opp_def_strength` (goals conceded by opponent)
- Adding `opp_off_strength` creates three overlapping signals about the same opponent
- Models may struggle to disentangle which feature to trust

**Correlation Analysis Needed**:
```python
# Check correlation between:
- opp_off_strength vs opp_def_strength
- opp_off_strength vs opponent one-hot encodings
- opp_def_strength vs opponent one-hot encodings
```

Expected high correlation (good attacking teams often have good defense).

#### 2. **Signal Noise**
- 5-game rolling average of goals scored is noisy
- Teams' offensive output varies widely game-to-game
- Home/away split not accounted for (teams score more at home)
- Against strong/weak opponents not factored in

#### 3. **Wrong Aggregation Level**
- Rolling average treats all 5 games equally
- Recent form might matter more than 5-game average
- Exponentially weighted moving average (EWMA) might work better

#### 4. **Model Overfitting**
- Adding more features increases model complexity
- XGBoost may overfit to training noise with extra features
- Early stopping may not be enough to prevent this

#### 5. **Feature Not Actually Predictive**
- Clean sheets might depend more on the defending team's ability than opponent's attacking ability
- Liverpool defense might keep clean sheet vs Man City because Liverpool defense is good, not because City attack is weak
- Team strength already captured in `team` one-hot encoding

---

## Detailed Position Analysis

### Goalkeepers (GK): R² 0.450 → 0.444 (-1.3%)
**Most Impacted Position**

Expected: Most improvement (clean sheets are 80% of GK points)  
Actual: Largest decrease

**Why?**
- GK points are simpler (clean sheet = 4pts, concede = lose clean sheet)
- Opponent team encoding already captures this well
- Adding noisy `opp_off_strength` confuses model

### Defenders (DEF): R² 0.262 → 0.260 (-0.8%)
**Slight Decrease**

Expected: Significant improvement  
Actual: Minimal decrease

**Why?**
- DEF points are complex (clean sheets + attacking returns + bonus)
- Attacking returns depend on `opp_def_strength` (goals conceded)
- Both features together might cancel each other out

### Midfielders (MID): R² 0.341 → 0.339 (-0.6%)
**Minimal Impact**

Expected: Neutral  
Actual: Slight decrease

**Reasonable**: MID rarely keep clean sheets, so `opp_off_strength` shouldn't matter much.

### Forwards (FWD): R² 0.340 → 0.333 (-2.1%)
**Second Most Impacted**

Expected: Neutral  
Actual: Significant decrease

**Why?**
- FWD scoring depends on `opp_def_strength` (goals conceded by opponent)
- Adding `opp_off_strength` creates conflicting signals
- Model gets confused about which opponent feature to use

---

## Lessons Learned

### 1. **More Features ≠ Better Performance**
Adding domain-relevant features doesn't always help if:
- Features are redundant with existing ones
- Features introduce noise
- Model complexity increases without signal increase

### 2. **Opponent Already Well-Captured**
Our model already captures opponent strength through:
- Opponent team one-hot encoding (20 features)
- `opp_def_strength` (goals conceded)

Adding a third opponent signal is redundant.

### 3. **Clean Sheets Are Complex**
Clean sheet prediction depends on:
- Defending team's defensive ability (captured by `team` encoding)
- Opponent's attacking ability (captured by `opponent` encoding)
- Match location (captured by `was_home`)
- Form (captured by `form`, `home_form`, `away_form`)

A simple rolling average of opponent goals scored doesn't add new information.

### 4. **Feature Engineering vs Feature Selection**
We've been adding features (engineering) without removing redundant ones (selection).

**Recommendation**: Consider feature selection techniques:
- Remove highly correlated features
- Use XGBoost feature importance to identify low-value features
- L1 regularization (Lasso) for automatic feature selection

---

## Alternative Approaches to Try

### Option 1: Remove `opp_off_strength` ✅ (Recommended)
**Action**: Revert to V4 (R² = 0.324)  
**Reason**: V4 performs better, simpler is better

### Option 2: Replace Both `opp_def_strength` AND `opp_off_strength` with Single Feature
**New Feature**: `opponent_strength` = rolling average of opponent's total points earned
- Captures both offensive and defensive quality
- Single feature instead of two
- Might reduce redundancy

### Option 3: Use Exponentially Weighted Moving Average (EWMA)
**Modification**: Instead of simple 5-game rolling average:
```python
opp_off_strength = df.groupby('opp_team')['team_goals'].ewm(span=5).mean()
```
Gives more weight to recent games.

### Option 4: Feature Selection
**Action**: Remove least important features based on XGBoost `feature_importances_`
- Identify features with <1% importance
- Remove them to reduce noise
- Retrain and compare

### Option 5: Different Aggregation
**Modifications**:
- 3-game rolling (more recent, less stable)
- 10-game rolling (more stable, less recent)
- Home/Away split: `opp_off_strength_home` vs `opp_off_strength_away`

---

## Recommendation

### ✅ **Revert to V4 (Remove `opp_off_strength`)**

**Reasons**:
1. V4 performs better (R² = 0.324 vs 0.321)
2. Simpler model is better (Occam's Razor)
3. Feature introduces redundancy without adding signal
4. All positions perform worse with this feature

**Action**:
```bash
git revert HEAD  # Revert the opponent offensive strength commit
```

Alternatively, keep the code but don't use it in production models.

---

## Positive Takeaways

1. ✅ **Quick experiment** - validated hypothesis in <1 hour
2. ✅ **No harm done** - can easily revert to V4
3. ✅ **Learned about feature redundancy** - not all domain knowledge translates to ML features
4. ✅ **Rigorous evaluation** - tested on proper train/val/test split

---

## Next Steps

### Priority 1: Deploy V4 Models ✅
- R² = 0.324 (best so far)
- No need for further feature engineering
- Focus on production deployment

### Priority 2: Feature Importance Analysis
- Analyze XGBoost `feature_importances_`
- Identify which features are actually used
- Remove low-importance features

### Priority 3: Hyperparameter Tuning (Optional)
- Grid search on XGBoost params
- May gain +0.01-0.02 R²
- Time: 2-3 hours

### Not Recommended:
- ❌ Adding more opponent features
- ❌ More complex feature engineering for clean sheets
- ❌ Pursuing this line of experimentation further

---

## Conclusion

**Hypothesis**: Opponent offensive strength will help GK/DEF predictions  
**Result**: ❌ **REJECTED** - Performance decreased across all positions  
**Decision**: ✅ **Revert to V4** (R² = 0.324)  

**Key Lesson**: Domain knowledge doesn't always translate to ML features. Feature redundancy can hurt performance. Simpler is often better.

---

## Files Affected

**Modified**:
- `ml/feature_engineering.py` - Added `opp_off_strength` feature
- All model files retrained with new feature

**To Revert**:
```bash
git diff HEAD~1 ml/feature_engineering.py  # See changes
git checkout HEAD~1 -- ml/feature_engineering.py  # Revert file
python -m ml.train  # Retrain models
```

**Training Time**: ~45 seconds  
**Date**: 2026-06-09
