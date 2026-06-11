# API Server Start Instructions

## Current Status

✅ **ML Models**: Successfully trained with DGW features  
✅ **Predictions**: Working (validated via standalone tests)  
⚠️ **API Server**: Dependency conflict (tf-keras/transformers)

---

## Quick Start (Recommended)

The API server has a dependency conflict between Keras 3 and transformers library. However, **the ML models work perfectly** via standalone scripts.

### Option 1: Use Standalone ML Predictions (WORKS NOW)

```bash
# Test all ML functionality
python test_ml_predictions_standalone.py

# This validates:
# - Model loading (4 position-specific XGBoost models)
# - Single player predictions
# - Batch predictions
# - DGW feature integration
```

### Option 2: Fix Dependencies and Start Server

```bash
# Install correct tf-keras for your Python version
pip install --upgrade tf-keras

# OR downgrade Keras if needed
pip install keras==2.15.0

# Then start server
$env:PYTHONPATH="C:\ACL2\FPL\ACL_M3"
uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Manual Server Start (Current Working Directory)

```powershell
# Open PowerShell in C:\ACL2\FPL\ACL_M3

# Set environment
$env:PYTHONPATH="C:\ACL2\FPL\ACL_M3"

# Start server
uvicorn api_main:app --host 0.0.0.0 --port 8000
```

---

## Testing ML Endpoints (Once Server Starts)

After server starts successfully:

```bash
# Test ML endpoints
python test_ml_api.py

# Or manually test:
curl http://localhost:8000/api/ml/status
```

---

## Dependency Issue Details

### Problem
The `embeddings.embedding_manager` module imports `sentence_transformers`, which imports `transformers`, which has a Keras 3 incompatibility.

### Installed Versions
- Python: 3.12 (using Python 3.14 packages - version mismatch)
- Keras: 3.x (incompatible)
- tf-keras: 2.15.0 (installed in user directory, not detected)

### Solution Options

**A. Install tf-keras system-wide:**
```bash
python -m pip install --upgrade tf-keras --user
```

**B. Downgrade transformers:**
```bash
pip install transformers==4.30.0
```

**C. Lazy-load embeddings (code change required):**
Modify `api_main.py` to import `EmbeddingManager` only when needed

**D. Skip embeddings for ML endpoints:**
Create ML-only API server without embeddings

---

## What Works RIGHT NOW

✅ **Training Pipeline**
```bash
python ml/train.py
```

✅ **DGW Feature Validation**
```bash
python ml/test_dgw_feature.py
```

✅ **Standalone Predictions**
```bash
python test_ml_predictions_standalone.py
```

✅ **Model Loading & Inference**
```python
from ml.predictor import FPLPredictor

# Load model
predictor = FPLPredictor('ml/models/xgboost_mid_v3.pkl')

# Predict
result = predictor.predict_next_gameweek(player_data)
print(f"Predicted: {result.predicted_points:.2f} pts")
```

---

## Summary

**Tasks Completed**:
1. ✅ Retrained V6 models with DGW features
2. ✅ Validated DGW feature integration (2.02x multiplier)
3. ✅ Tested end-to-end predictions (standalone)
4. ✅ Generated comprehensive documentation

**Outstanding**:
- ⚠️ Fix transformer/Keras dependency (5-15 min fix)
- ⚠️ Start API server on port 8000

**Current Workaround**:
- Use standalone ML predictions
- ML models fully functional
- API integration pending dependency resolution

---

## Contact / Notes

- Models trained: June 11, 2026
- Performance: R² = 0.725 (XGBoost)
- DGW Impact: 2.02x points multiplier
- Documentation: ML_V6.2_DGW_TRAINING_SUMMARY.md

**Next Session**: Start with Option 2 (fix dependencies) to get full API server running.
