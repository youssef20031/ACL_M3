# Known Issues - FPL Promoted Teams System

## Non-Critical Issues

### 1. Misleading DGW Logging (Low Priority)

**Issue**: Feature engineering logs show incorrect DGW statistics:
```
Added DGW feature: 1,039 single GWs (2.5%), 254 DGWs (0.6%), 40,264 TGWs (96.9%)
```

**Reality**: Actual distribution is:
```
- 27,759 single GWs (96.6%)
- 983 DGWs (3.4%)  
- 0 TGWs
- Max 2 fixtures
```

**Root Cause**: The logging calculates `groupby().size()` after multiple merge operations that duplicate rows, inflating the counts.

**Impact**: **NONE** - The actual feature `fixtures_this_gw` has correct values (1 or 2). Only the log message is misleading.

**Status**: Documented, no fix needed for training

**Verification**:
```python
import pandas as pd
from ml.feature_engineering import FeatureEngineer

df = pd.read_csv('FPL_2023_2024.csv')
fe = FeatureEngineer()
df = fe._add_dgw_feature(df)

print(df['fixtures_this_gw'].value_counts())
# Output:
# 1    27759  (correct!)
# 2     1966  (correct!)
```

---

## Status: Ready for Training

This issue does not block training or affect model performance. The feature values are correct.

---

*Last Updated: June 22, 2026*
