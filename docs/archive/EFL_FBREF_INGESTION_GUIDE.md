# EFL Championship FBref Dataset — Ingestion & Processing Guide

> **Purpose**: This guide instructs an agentic LLM on how to correctly load, clean, and prepare
> FBref-scraped EFL Championship tables for use in an FPL ML pipeline. Follow every step in order.
> Do not skip sections — each addresses a known failure mode in this specific dataset.

---

## 1. What You Are Working With

The dataset contains multiple FBref statistical tables per EFL Championship season, covering
categories such as: Standard Stats, Shooting, Goalkeeping, Passing, Misc (discipline/miscellaneous),
and Playing Time.

There are typically **5–7 tables per season**, and the pipeline targets **5+ seasons** of data,
meaning 25–40 individual files total. All files were copy-pasted from FBref and saved as `.txt`
or `.csv`.

**Important**: FBref data is tab-separated (`\t`), not comma-separated. Even if files have a
`.csv` extension, always parse with `sep="\t"`. Treating them as true CSVs will silently corrupt
the data.

---

## 2. Known Structural Issues — Read Before Processing Any File

### Issue 1: Wrong Delimiter
FBref copy-paste produces tab-separated text. If loaded with a comma delimiter, every row
collapses into a single column.

**Fix — always use:**
```python
df = pd.read_csv("file.txt", sep="\t")
```

### Issue 2: Repeated Header Rows
FBref inserts a full header row every ~25 data rows for UI readability. These must be removed
or every 25th row will be a string header masquerading as a data row, breaking dtype inference.

**Fix:**
```python
df = df[df["Player"] != "Player"]   # primary filter
df = df[df["Rk"] != "Rk"]          # fallback if Rk column exists
df = df.reset_index(drop=True)
```

### Issue 3: Encoding Corruption (accented names)
Names with accented characters (é, ó, á, ě, etc.) are frequently misread as UTF-8 bytes
decoded under Latin-1, producing garbled output: `André` becomes `AndrÃ©`.

**Fix at load time (preferred):**
```python
df = pd.read_csv("file.txt", sep="\t", encoding="utf-8")
```

**Fix after the fact (if already loaded incorrectly):**
```python
df["Player"] = df["Player"].str.encode("latin1").str.decode("utf-8")
```

### Issue 4: Multi-Position Fields Breaking Column Alignment
Players listed at multiple positions have a comma in the `Pos` field (e.g., `MF,FW`).
If the file is ever parsed as CSV, this comma splits the row incorrectly and all columns
after `Pos` shift left and lose their values. The data is not missing — it is misaligned.
Using tab-separation (Issue 1 fix) fully prevents this.

Once loaded correctly, normalize the `Pos` field for safe processing:
```python
# Strip whitespace, keep raw multi-position string for now
df["Pos"] = df["Pos"].str.strip().str.replace(" ", "")

# Extract primary position (first listed)
df["Pos_Primary"] = df["Pos"].str.split(",").str[0]

# Create a boolean flag for multi-position players
df["Is_Multi_Pos"] = df["Pos"].str.contains(",")
```

---

## 3. Canonical Ingestion Pipeline

Run this function for **every file** before doing anything else. Do not process any file
that has not passed through this pipeline.

```python
import pandas as pd
import unicodedata
import os

def load_fbref_table(filepath: str, season: str, table_type: str) -> pd.DataFrame:
    """
    Load a single FBref EFL Championship table.

    Args:
        filepath:   path to the .txt or .csv file
        season:     season label, e.g. "2021-22"
        table_type: category label, e.g. "standard", "shooting", "goalkeeping"

    Returns:
        Clean DataFrame with metadata columns appended.
    """
    # Step 1: Load with tab delimiter and UTF-8 encoding
    df = pd.read_csv(filepath, sep="\t", encoding="utf-8", low_memory=False)

    # Step 2: Remove repeated FBref header rows
    if "Player" in df.columns:
        df = df[df["Player"] != "Player"]
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"]
    df = df.reset_index(drop=True)

    # Step 3: Normalize position field
    if "Pos" in df.columns:
        df["Pos"] = df["Pos"].str.strip().str.replace(" ", "")
        df["Pos_Primary"] = df["Pos"].str.split(",").str[0]
        df["Is_Multi_Pos"] = df["Pos"].str.contains(",", na=False)

    # Step 4: Normalize player name for cross-table merging
    if "Player" in df.columns:
        df["Player_Key"] = df["Player"].apply(normalize_name)

    # Step 5: Attach metadata
    df["Season"] = season
    df["Table_Type"] = table_type

    return df


def normalize_name(name: str) -> str:
    """
    Strip accents and lowercase for safe cross-dataset player matching.
    André Ayew → andre ayew
    """
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    return name.encode("ascii", "ignore").decode("ascii").lower().strip()
```

---

## 4. Processing Multiple Seasons and Tables

Organize your files with a consistent naming convention before processing:

```
data/
  efl/
    2019-20_standard.txt
    2019-20_shooting.txt
    2019-20_goalkeeping.txt
    2020-21_standard.txt
    ...
```

Then batch-load:

```python
import glob

all_frames = []

for filepath in glob.glob("data/efl/*.txt"):
    filename = os.path.basename(filepath).replace(".txt", "")
    parts = filename.split("_", 1)        # ["2021-22", "standard"]
    season = parts[0]
    table_type = parts[1] if len(parts) > 1 else "unknown"

    df = load_fbref_table(filepath, season=season, table_type=table_type)
    all_frames.append(df)

# Separate by table type for clean merging
standard_df   = pd.concat([f for f in all_frames if f["Table_Type"].iloc[0] == "standard"],   ignore_index=True)
shooting_df   = pd.concat([f for f in all_frames if f["Table_Type"].iloc[0] == "shooting"],   ignore_index=True)
goalkeeping_df = pd.concat([f for f in all_frames if f["Table_Type"].iloc[0] == "goalkeeping"], ignore_index=True)
```

---

## 5. Cross-Table and Cross-Season Merging

**Never merge on `Player` (raw name) alone.** Accents, formatting, and FBref inconsistencies
make raw name matching unreliable. Always merge on the composite key:

```python
merge_key = ["Player_Key", "Season", "Squad"]
```

`Squad` is included because a player can appear at two clubs in one season (mid-season transfer).
In that case, FBref lists two rows — one per squad — and both are valid records.

```python
combined = standard_df.merge(shooting_df, on=merge_key, how="left", suffixes=("", "_shoot"))
combined = combined.merge(goalkeeping_df, on=merge_key, how="left", suffixes=("", "_gk"))
```

---

## 6. Validation Checks — Run After Each Load

After processing each file, verify the following. If any check fails, stop and diagnose
before continuing.

```python
def validate_table(df: pd.DataFrame, filepath: str):
    issues = []

    # 1. Check for unsplit rows (entire row in column 1)
    if df.shape[1] < 5:
        issues.append(f"CRITICAL: Only {df.shape[1]} columns — likely parsed with wrong delimiter")

    # 2. Check for surviving header rows
    if "Player" in df.columns and (df["Player"] == "Player").any():
        issues.append("WARNING: Repeated header rows not fully removed")

    # 3. Check for encoding artifacts
    if "Player" in df.columns:
        garbled = df["Player"].str.contains("Ã|Â|â€", na=False, regex=True)
        if garbled.any():
            issues.append(f"WARNING: {garbled.sum()} player names with encoding artifacts")

    # 4. Check Player_Key populated
    if "Player_Key" not in df.columns or df["Player_Key"].isna().all():
        issues.append("WARNING: Player_Key column missing or empty")

    if issues:
        for issue in issues:
            print(f"[{filepath}] {issue}")
    else:
        print(f"[{filepath}] OK — {len(df)} rows, {df.shape[1]} columns")
```

---

## 7. Numeric Type Coercion

FBref columns arrive as strings due to the repeated header rows. After cleaning, convert
numeric columns explicitly:

```python
# Columns that should be numeric — adapt to whichever table you are processing
numeric_cols = [
    "90s", "Gls", "Ast", "G+A", "G-PK", "xG", "xAG", "xG+xAG",
    "Sh", "SoT", "SoT%", "G/Sh", "G/SoT",
    "Min", "Starts", "MP",
    "GA", "GA90", "Save%", "CS", "CS%",
    "Fls", "Fld", "Int", "TklW", "CrdY", "CrdR", "OG"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
```

`errors="coerce"` turns any non-numeric values (e.g. surviving junk rows) into `NaN`
rather than crashing.

---

## 8. Glossary

Only terms that have non-obvious meanings or use FBref-specific definitions are listed here.
Common terms (Goals, Assists, Minutes, Yellow Cards) are omitted.

### Structural / Identity Columns

| Column | Meaning |
|--------|---------|
| `Rk` | Row rank — recalculated on every sort. Not a stable player ID. Do not use as a key. |
| `Nation` | Player nationality — resolved from international records first, then youth, then Wikipedia citizenship, then birthplace. May differ from what you expect for dual-nationals. |
| `Pos` | Primary position. FBref uses a non-standard extended set. Multi-position players show comma-separated values (e.g. `MF,FW`). See position codes below. |
| `Age` | Age as of August 1 (winter leagues). Not current age — recalculate if needed. |
| `Squad` | Club name. Spelling is consistent within FBref but may not match FPL API team names — normalize before merging. |

### Position Codes

FBref uses more granular codes than FPL's GK/DEF/MID/FWD. Map as follows for model compatibility:

| FBref Code | Description | FPL Equivalent |
|------------|-------------|---------------|
| `GK` | Goalkeeper | GK |
| `DF`, `CB`, `FB`, `LB`, `RB` | Defenders | DEF |
| `MF`, `CM`, `DM`, `LM`, `RM`, `WM`, `AM` | Midfielders | MID |
| `FW`, `LW`, `RW` | Forwards | FWD |

When a player has `Pos_Primary` not in the above list, default to the broadest containing category.

### Playing Time

| Column | Meaning |
|--------|---------|
| `90s` | Total minutes played divided by 90. Not a count of full games. A player with 270 minutes has `90s = 3.0`. All per-90 rate stats use this as the denominator. |
| `Min%` | Percentage of the team's total minutes the player was on the pitch. Useful for identifying starters vs. squad players. |
| `Mn/MP` | Average minutes per match played — distinct from `Min/Start`. |
| `unSub` | Unused substitute appearances. The player was named in the squad but never came on. |

### Team Success (Contextual Features)

These columns describe team-level outcomes **while this specific player was on the pitch** — they are not season-level team stats.

| Column | Meaning |
|--------|---------|
| `PPM` | Points per match earned by the team in games the player appeared in. Reflects quality of opposition and team form during that player's appearances. |
| `onG` | Goals scored by the team while this player was on the pitch. |
| `onGA` | Goals conceded by the team while this player was on the pitch. |
| `+/-` | `onG - onGA`. Team goal difference while this player was on the pitch. |
| `+/-90` | `+/-` per 90 minutes. Normalized for playing time. |
| `On-Off` | Net goals per 90 while the player is on the pitch, minus net goals per 90 while the player is off the pitch. Measures marginal team impact. |

### Shooting

| Column | Meaning |
|--------|---------|
| `SoT%` | Shots on target percentage. Shots on target divided by total shots. Quality-of-contact indicator. |
| `G/Sh` | Goals per shot (all shots). Overall shooting efficiency. |
| `G/SoT` | Goals per shot on target. Finishing quality, independent of whether shots hit the target. |

### Goalkeeping

| Column | Meaning |
|--------|---------|
| `GA90` | Goals conceded per 90 minutes. Preferred over raw `GA` for comparing GKs with different playing time. |
| `SoTA` | Shots on target faced. The denominator for `Save%`. |
| `Save%` | `(SoTA - GA) / SoTA`. Note: shots blocked by outfield defenders before reaching the GK are **not** counted in `SoTA`, so this reflects only shots the GK actually faced. |
| `CS%` | Percentage of appearances resulting in a clean sheet. |
| `PKsv` | Penalty kicks saved. Not included in standard `Save%` — tracked separately. |

### Discipline / Misc

| Column | Meaning |
|--------|---------|
| `2CrdY` | Second yellow card (results in a red). Counted separately from `CrdR` (straight red). A player dismissed by two yellows appears in both `CrdY` (twice) and `2CrdY` (once). |
| `Fls` | Fouls committed by the player. |
| `Fld` | Fouls drawn (fouled by an opponent). |
| `TklW` | Tackles won — only tackles where the player's team regained possession. Total tackle attempts are in a separate column (`Tkl`) not always present in misc tables. |
| `PKwon` | Penalty kicks won (player was fouled in the box or handled triggered the penalty). |
| `PKcon` | Penalty kicks conceded (player committed the foul). |
| `OG` | Own goals. Rare but present — do not confuse with `GA` (goalkeeper goals against). |

---

## 9. What to Do If a File Fails Validation

Work through this checklist in order:

1. **Only 1–2 columns after load** → wrong delimiter. Re-load with `sep="\t"`.
2. **Header rows still present** → filter `df[df["Player"] != "Player"]` was not applied, or the column name is different. Print `df.columns` and adapt.
3. **Encoding artifacts in names** → apply the `latin1 → utf-8` decode fix from Section 2.
4. **Columns misaligned for one player** → this is the `MF,FW` issue and should not occur if you used `sep="\t"`. If it does, check whether the file was re-saved as CSV at any point.
5. **Unexpected NaN flood after numeric coercion** → a column contains non-numeric strings beyond header rows. Print `df[col].unique()` to inspect.

---

*This guide covers all known structural issues in FBref EFL Championship copy-paste exports.
Follow it for every new season added to the pipeline.*
