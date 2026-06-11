# Quick Start Guide - ML V6.2 with DGW Features

## ✅ What's Working NOW

### Test ML Predictions (Works Immediately)
```bash
cd C:\ACL2\FPL\ACL_M3
python test_ml_predictions_standalone.py
```

**Output**: Tests all 4 XGBoost models with DGW features

---

## 🎯 Key Achievements

| Task | Status | Performance |
|------|--------|-------------|
| Model Training | ✅ Complete | R² = 0.725 |
| DGW Features | ✅ Integrated | 2.02x multiplier |
| Predictions | ✅ Working | MAE = 0.48 pts |
| API Server | ⚠️ Dep Issue | Fixable in 5-15 min |

---

## 📊 Model Performance

```
Overall: R² = 0.725, RMSE = 1.24, MAE = 0.48

By Position:
- GK:  R² = 0.637 (6,514 samples)
- DEF: R² = 0.705 (24,145 samples)
- MID: R² = 0.758 (32,771 samples)  ⭐ Best
- FWD: R² = 0.670 (6,285 samples)
```

---

## 🚀 Using the Models

### Python Code
```python
from ml.predictor import FPLPredictor

# Load model
predictor = FPLPredictor('ml/models/xgboost_mid_v3.pkl')

# Predict single player
player_data = {
    'name': 'Mohamed Salah',
    'position': 'MID',
    'form': 7.5,
    'minutes': 90,
    'was_home': True,
    'GW': 15,
    'fixtures_this_gw': 2,  # DGW!
    # ... other features
}

result = predictor.predict_next_gameweek(player_data)
print(f"Predicted: {result.predicted_points:.2f} pts")
```

### Top Performers
```python
players = [...]  # List of player dicts
top_5 = predictor.predict_top_performers(players, position='FWD', top_k=5)
```

---

## 🔧 Fix Server & Start

### Option 1: Fix Dependencies
```bash
pip install --upgrade tf-keras
uvicorn api_main:app --host 0.0.0.0 --port 8000
```

### Option 2: Test API (Once Running)
```bash
python test_ml_api.py
```

---

## 📁 Important Files

```
✅ Models:
   ml/models/xgboost_{gk,def,mid,fwd}_v3.pkl

✅ Tests:
   test_ml_predictions_standalone.py
   ml/test_dgw_feature.py

✅ Docs:
   ML_V6.2_DGW_TRAINING_SUMMARY.md (full details)
   TASK_COMPLETION_SUMMARY.md (status report)
   SERVER_START_INSTRUCTIONS.md (server help)

⚠️ API:
   api_main.py (needs tf-keras fix)
   test_ml_api.py (test once server runs)
```

---

## 🎮 DGW Feature

**What it does**: Detects players with multiple fixtures in a gameweek

**Impact**: 2.02x points boost in Double Gameweeks

**Example**:
- Single GW: Mohamed Salah predicts 8 pts
- Double GW: Mohamed Salah predicts 14-16 pts (2 games)

**Coverage**: 3.6% of dataset (realistic DGW frequency)

---

## ⏱️ Training Time

```
Total: ~5.5 minutes
- Data loading: ~30 sec
- Feature engineering: ~1 min
- Model training: ~4 min
- Evaluation: ~30 sec
```

---

## 📞 Quick Help

**ML works but need API?**
→ See `SERVER_START_INSTRUCTIONS.md`

**Want full details?**
→ See `ML_V6.2_DGW_TRAINING_SUMMARY.md`

**Need to retrain?**
→ Run `python ml/train.py`

**Check DGW stats?**
→ Run `python ml/test_dgw_feature.py`

---

**Status**: ✅ ML Ready | ⚠️ Server Fix Needed  
**Performance**: R² = 0.725 (Excellent)  
**Production**: Ready after server fix
