"""
merge_seasons.py
----------------
Preprocesses and merges new FPL season CSVs (2023-24, 2024-25, 2025-26) into
the format expected by FPLDataLoader, then appends them to the existing master
CSV (cleaned_merged_seasons_cleaned.csv).

Column mapping differences between old and new data
-----------------------------------------------------
Old master CSV columns (selected):
    season_x, name, position, team_x, opp_team_name, was_home, GW, fixture, ...

New season CSV columns (selected):
    name, position, team, was_home, GW, fixture, ...
    (no season column — injected from filename/argument)
    (no opp_team_name — must be reconstructed from fixture/team pairings)

Usage:
    python preprocessing/merge_seasons.py
    python preprocessing/merge_seasons.py --output my_combined.csv
    python preprocessing/merge_seasons.py --dry-run   # prints stats only, no write
"""

import pandas as pd
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (all relative to project root)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MASTER_CSV = os.path.join(ROOT, "cleaned_merged_seasons_cleaned.csv")
OUTPUT_CSV = os.path.join(ROOT, "cleaned_merged_seasons_cleaned.csv")  # overwrite in-place

# New season files with their season labels
NEW_SEASON_FILES = {
    "2023-24": os.path.join(ROOT, "FPL_2023_2024.csv"),
    "2024-25": os.path.join(ROOT, "FPL_2024_2025.csv"),
    "2025-26": os.path.join(ROOT, "FPL_2025_2026.csv"),
}

# ---------------------------------------------------------------------------
# Columns that must exist in the final merged output (mirrors old master CSV)
# Extra columns from newer seasons are preserved but NaN-filled for old rows.
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "season_x", "name", "position", "team_x",
    "assists", "bonus", "bps", "clean_sheets", "creativity",
    "element", "fixture", "goals_conceded", "goals_scored",
    "ict_index", "influence", "kickoff_time", "minutes",
    "opponent_team", "opp_team_name", "own_goals",
    "penalties_missed", "penalties_saved", "red_cards",
    "round", "saves", "selected", "team_a_score", "team_h_score",
    "threat", "total_points", "transfers_balance",
    "transfers_in", "transfers_out", "value", "was_home",
    "yellow_cards", "GW",
]

# Columns from newer seasons to carry forward (optional enrichment)
OPTIONAL_NEW_COLUMNS = [
    "xP", "expected_assists", "expected_goal_involvements",
    "expected_goals", "expected_goals_conceded", "starts",
    # 2025-26 defensive stats
    "tackles", "recoveries", "clearances_blocks_interceptions",
    "defensive_contribution",
]

# Manager-specific columns introduced in 2024-25 — drop them; they apply to
# manager entries, not regular players.
MANAGER_COLUMNS = [
    "mng_clean_sheets", "mng_draw", "mng_goals_scored",
    "mng_loss", "mng_underdog_draw", "mng_underdog_win", "mng_win",
    "modified",
]


def build_opp_team_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct opp_team_name for new-format CSVs.

    New CSVs have fixture IDs and team names but no explicit opponent column.
    For each fixture we identify the two teams present, then assign the
    opponent as the other team.
    """
    if "opp_team_name" in df.columns:
        return df  # already present

    logger.info("Reconstructing opp_team_name from fixture pairings...")

    # Build a fixture → {team_a, team_b} mapping
    fixture_teams = (
        df.groupby("fixture")["team"]
        .apply(lambda s: s.unique().tolist())
        .to_dict()
    )

    def get_opponent(row):
        teams = fixture_teams.get(row["fixture"], [])
        others = [t for t in teams if t != row["team"]]
        return others[0] if others else None

    df["opp_team_name"] = df.apply(get_opponent, axis=1)

    # Warn about fixtures with only one team observed (e.g., incomplete data)
    missing = df["opp_team_name"].isna().sum()
    if missing:
        logger.warning(
            f"{missing} rows could not have opp_team_name resolved "
            "(fixture may have only one team in dataset)."
        )

    return df


def normalise_new_season(df: pd.DataFrame, season_label: str) -> pd.DataFrame:
    """
    Normalise a new-format season DataFrame to match the master CSV schema.

    Steps:
    1. Drop manager-specific columns
    2. Inject season_x
    3. Rename team → team_x
    4. Rebuild opp_team_name
    5. Add opponent_team as numeric alias of opp_team_name (kept as-is in old data)
    6. Ensure all required columns exist (fill with NaN if absent)
    """
    logger.info(f"Normalising season {season_label} ({len(df)} rows)...")

    # 1. Drop manager/irrelevant columns
    df = df.drop(columns=[c for c in MANAGER_COLUMNS if c in df.columns])

    # 2. Inject season
    df["season_x"] = season_label

    # 3. Rename team → team_x
    if "team" in df.columns and "team_x" not in df.columns:
        df = df.rename(columns={"team": "team_x"})

    # 4. Rebuild opp_team_name (requires original 'team' values, now 'team_x')
    # Temporarily re-alias for the helper function
    df = df.rename(columns={"team_x": "team"})
    df = build_opp_team_name(df)
    df = df.rename(columns={"team": "team_x"})

    # 5. opponent_team — in old data this is a numeric ID; we'll leave it as NaN
    #    for new seasons since we don't have the numeric mapping.
    if "opponent_team" not in df.columns:
        df["opponent_team"] = None

    # 6. round — same as GW in new data
    if "round" not in df.columns and "GW" in df.columns:
        df["round"] = df["GW"]

    # Ensure all required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
            logger.debug(f"  Added missing required column: {col}")

    return df


def load_master(path: str) -> pd.DataFrame:
    logger.info(f"Loading master CSV: {path}")
    df = pd.read_csv(path)
    logger.info(f"  {len(df)} rows, seasons: {sorted(df['season_x'].unique())}")
    return df


def check_duplicates(master: pd.DataFrame, new: pd.DataFrame, season_label: str) -> bool:
    """Return True if season already present in master."""
    if "season_x" in master.columns and season_label in master["season_x"].values:
        logger.warning(f"Season {season_label} already exists in master — skipping.")
        return True
    return False


def merge_all(output_path: str, dry_run: bool = False) -> pd.DataFrame:
    """
    Main merge routine.

    Loads master, then appends each new season that isn't already present.
    Returns the combined DataFrame.
    """
    master = load_master(MASTER_CSV)

    frames = [master]

    for season_label, filepath in NEW_SEASON_FILES.items():
        if not os.path.exists(filepath):
            logger.warning(f"File not found, skipping: {filepath}")
            continue

        if check_duplicates(master, master, season_label):
            continue

        raw = pd.read_csv(filepath)
        logger.info(f"Loaded {filepath}: {len(raw)} rows, {len(raw.columns)} cols")

        normalised = normalise_new_season(raw, season_label)
        frames.append(normalised)
        logger.info(f"  → {len(normalised)} rows added for {season_label}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.info(
        f"\nCombined dataset: {len(combined)} rows, "
        f"seasons: {sorted(combined['season_x'].dropna().unique())}"
    )

    if not dry_run:
        combined.to_csv(output_path, index=False)
        logger.info(f"Written to: {output_path}")
    else:
        logger.info("Dry-run mode — no file written.")

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge new FPL season CSVs into master dataset.")
    parser.add_argument(
        "--output", default=OUTPUT_CSV,
        help="Output CSV path (default: overwrites cleaned_merged_seasons_cleaned.csv)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print statistics without writing any file"
    )
    args = parser.parse_args()

    merge_all(output_path=args.output, dry_run=args.dry_run)
