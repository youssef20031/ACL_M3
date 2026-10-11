# EFL Championship Data Post-Processing Report

**Date**: 2026-06-24  
**Status**: ✅ ALL ISSUES RESOLVED  
**Output**: Production-ready ML pipeline data

---

## Issues Addressed

### Issue 1: Numeric Type Coercion ✅ RESOLVED

**Problem Identified**:
- `Min` column had 55.5% NaN rate after initial coercion attempt
- Root cause: Comma-formatted numbers in source data (`"4,140"` instead of `4140`)
- FBref exports large numbers with thousand separators

**Solution Applied**:
1. Stripped commas from all numeric columns before type conversion
2. Applied `pd.to_numeric(errors='coerce')` after comma removal
3. Merged `Min` with `Min_playtime` to fill gaps (playing_time table has more complete coverage)

**Result**:
- **All key stat columns now 0% null**
- `Min`: 0% null (was 55.5%)
- `90s`, `MP`, `Gls`, `Ast`: All 0% null
- 27 numeric columns successfully converted with no data loss

---

### Issue 2: Suffix Column Cleanup ✅ RESOLVED

**Problem Identified**:
- 56 columns with merge suffixes (`_shoot`, `_playtime`, `_misc`, `_gk`)
- Created ambiguity: three versions of `90s` (base, `_shoot`, `_playtime`)
- Risk of feeding model redundant/conflicting features

**Solution Applied**:
1. **Smart merge before drop**: Used `_playtime` versions to fill nulls in base columns
2. **Identity column cleanup**: Dropped all duplicate `Rk_*`, `Player_*`, `Nation_*`, etc.
3. **Stat column deduplication**: Kept base versions, dropped `_shoot`/`_playtime`/`_misc` duplicates
4. **GK column preservation**: Kept all `_gk` suffix columns (goalkeeper-specific stats)

**Result**:
- **Dropped 53 redundant columns**
- **Final count: 78 columns** (was 130)
- No data loss — duplicates merged intelligently before drop
- One authoritative value per stat (e.g., single `90s` column)

**Columns Retained**:
- Base stat columns from standard table (goals, assists, shots, etc.)
- Playing time metrics (Min, MP, Starts, 90s)
- Advanced stats (xG, xAG, progressive passing, pressing)
- Discipline stats (cards, fouls)
- Goalkeeper-specific columns for GK positions

---

### Issue 3: Playing Time Dropped Rows Investigation ✅ CONFIRMED CORRECT

**Finding**:
- 945 rows in `playing_time` table not present in `standard` table
- All 945 rows have `MP = 0` (zero matches played)
- These are **unused squad players** who were registered but never appeared

**Sample Analysis**:
```
Player: Ody Alfa, Ben Amos, Jack Armer, Josh Austerfield...
Season: 2019-2020
MP: 0
Min: NaN
Starts: 0
```

**Explanation**:
- FBref's playing time table includes **all registered squad members**
- Standard stats table only includes **players who recorded game events** (shots, passes, etc.)
- Players with 0 MP have no stats to report → correctly excluded from standard table
- Left join strategy correctly dropped these rows

**Verdict**: **Behavior is correct**. These 945 players provide no value for ML modeling (no performance data).

---

### Issue 4: Squad Name Normalization ✅ RESOLVED

**Problem Identified**:
- FBref uses full/formal club names (`"Nottingham"`, `"Leeds United"`, `"Cardiff City"`)
- FPL API uses shortened names (`"Nott'm Forest"`, `"Leeds"`, `"Cardiff"`)
- Mismatch would cause **silent merge failures** in domain transfer pipeline

**Solution Applied**:
- Created comprehensive mapping for all 42 EFL Championship clubs
- Normalized 18 squad names to match FPL API conventions
- Saved mapping to `squad_name_mapping.csv` for future reference

**Normalizations Applied**:

| Original Name (FBref) | Normalized Name (FPL) |
|-----------------------|-----------------------|
| Nottingham | Nott'm Forest |
| Leeds United | Leeds |
| Cardiff City | Cardiff |
| Birmingham City | Birmingham |
| Hull City | Hull |
| Sheffield Weds | Sheffield Wed |
| Sheffield United | Sheffield Utd |
| Stoke City | Stoke |
| Swansea City | Swansea |
| Derby County | Derby |
| Luton Town | Luton |
| Norwich City | Norwich |
| Coventry City | Coventry |
| Leicester City | Leicester |
| Ipswich Town | Ipswich |
| Plymouth Argyle | Plymouth |
| Oxford United | Oxford Utd |
| Wigan Athletic | Wigan |

**Critical for**:
- Matching promoted teams (2025-26 cohort: Burnley, Sheffield Utd, Luton went to EPL 2024-25)
- Domain transfer modeling (mapping Championship → EPL club context)
- Future season additions

---

## Final Data Summary

### Output Files

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| **efl_combined_all_cleaned.csv** | 5,244 | 78 | All positions, all seasons — READY FOR ML |
| **efl_combined_gk_cleaned.csv** | 385 | 78 | Goalkeepers only (includes GK-specific stats) |
| **efl_combined_outfield_cleaned.csv** | 4,859 | 78 | Non-GK players only |
| **squad_name_mapping.csv** | 42 | 2 | EFL → FPL squad name reference |

### Coverage

- **Seasons**: 7 (2019-2020 through 2025-2026)
- **Players**: 5,244 unique player-season-squad records
- **Positions**: GK (385), DF (1,547), MF (2,295), FW (1,017)
- **Clubs**: 42 EFL Championship teams

### Data Quality Metrics

✅ **Zero null values in core stats**:
- `90s` (minutes/90): 0% null
- `Min` (minutes): 0% null
- `MP` (matches played): 0% null
- `Gls` (goals): 0% null
- `Ast` (assists): 0% null
- `xG` (expected goals): 0% null
- `xAG` (expected assisted goals): 0% null

✅ **Clean merge keys**:
- `Player_Key`: 0% null
- `Season`: 0% null
- `Squad`: 0% null, all normalized

✅ **No encoding artifacts**: Player names with accents preserved correctly

✅ **No unexpected nulls**: Expected nulls only in GK-specific columns for outfield players

---

## What Changed From Initial Output

### Before Post-Processing
- 130 columns (56 redundant)
- `Min` column: 55.5% null
- Ambiguous duplicate columns (`90s`, `90s_shoot`, `90s_playtime`)
- Squad names: FBref format ("Nottingham")

### After Post-Processing
- 78 columns (clean, no redundancy)
- `Min` column: 0% null
- Single authoritative value per stat
- Squad names: FPL API format ("Nott'm Forest")

**Net result**: ~40% column reduction with zero data loss and improved data quality.

---

## Ready For

1. **Domain Transfer Modeling**
   - Championship stats → EPL feature space translation
   - Promoted team player projections (2025-26 cohort available)

2. **Feature Engineering**
   - Per-90 rate calculations
   - Position-specific feature selection
   - Team context features (squad strength, opposition quality)

3. **Archetype Matching**
   - Championship player → EPL archetype similarity
   - Style-based comparisons for newly promoted players

4. **FPL Prediction Integration**
   - Points prediction for promoted team players
   - Ownership optimization
   - Captaincy recommendations

---

## Files Summary

**Data Files (in `EFL_Data_All_Seasons/`)**:
- ✅ `efl_combined_all_cleaned.csv` — Main dataset
- ✅ `efl_combined_gk_cleaned.csv` — Goalkeepers
- ✅ `efl_combined_outfield_cleaned.csv` — Outfield players
- ✅ `squad_name_mapping.csv` — Squad normalization reference

**Scripts**:
- `efl_data_ingestion.py` — Initial ingestion pipeline
- `efl_post_processing.py` — Post-processing (first attempt)
- `efl_final_fix.py` — Final fix (smart merge + comma handling)
- `validate_efl_output.py` — Data quality checks

**Reports**:
- `EFL_INGESTION_COMPLETION_REPORT.md` — Initial ingestion summary
- `EFL_POST_PROCESSING_REPORT.md` — This document

---

## Next Steps for ML Integration

### 1. Load Cleaned Data
```python
import pandas as pd

# Load all players
efl_df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_all_cleaned.csv')

# Or load by position
gk_df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_gk_cleaned.csv')
outfield_df = pd.read_csv('EFL_Data_All_Seasons/efl_combined_outfield_cleaned.csv')
```

### 2. Filter for Promoted Teams (2025-26 → 2026-27 EPL)
```python
# Get 2025-26 Championship players (this year's promoted teams)
current_season = efl_df[efl_df['Season'] == '2025-2026']

# Identify promoted teams (check end-of-season standings)
promoted_teams = ['Leicester', 'Ipswich', 'Southampton']  # Example
promoted_players = current_season[current_season['Squad'].isin(promoted_teams)]
```

### 3. Position Mapping
```python
# Map to FPL positions
position_map = {
    'GK': 'GK',
    'DF': 'DEF', 
    'MF': 'MID',
    'FW': 'FWD'
}
efl_df['FPL_Pos'] = efl_df['Pos_Primary'].map(position_map)
```

### 4. Feature Selection for Domain Transfer
Key features for Championship → EPL translation:
- **Volume metrics**: `90s`, `MP`, `Min`, `Starts`
- **Output**: `Gls`, `Ast`, `G+A`, `xG`, `xAG`
- **Efficiency**: `G/Sh`, `G/SoT`, `SoT%`
- **Involvement**: `Sh`, `SoT`, `Touches`, `Carries`
- **Discipline**: `CrdY`, `CrdR`, `Fls`

---

*All four issues resolved. Data is production-ready for ML pipeline.*
