"""
build_embeddings_from_csv.py
-----------------------------
Builds prebuilt embeddings directly from the master CSV file.
No Neo4j connection required — designed for GitHub Actions CI.

Replicates the same aggregation as CypherQueries.get_player_embeddings_data()
so the resulting .pkl files are 100% compatible with the runtime embedding manager.

Usage:
    python build_embeddings_from_csv.py minilm
    python build_embeddings_from_csv.py mpnet
    python build_embeddings_from_csv.py both        # build both models
"""

import os
import sys
import gc
import pickle
import logging
import pandas as pd
import numpy as np
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "cleaned_merged_seasons_cleaned.csv")
OUTPUT_DIR = os.path.join(ROOT, "embeddings", "prebuilt")


# ---------------------------------------------------------------------------
# CSV → player records  (mirrors get_player_embeddings_data Cypher query)
# ---------------------------------------------------------------------------

def load_player_records(csv_path: str) -> list[dict]:
    """
    Aggregate the master CSV into per-player-per-season records identical
    to what get_player_embeddings_data() returns from Neo4j.
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  {len(df)} rows loaded")

    # Normalise season column name
    if "season_x" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"season_x": "season"})

    # Normalise team column
    if "team_x" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"team_x": "team"})

    # Ensure numeric columns exist (fill 0 if absent — older seasons)
    numeric_cols = [
        "total_points", "goals_scored", "assists", "clean_sheets",
        "bonus", "bps", "minutes", "ict_index", "influence",
        "creativity", "threat", "value", "selected",
        "saves", "goals_conceded", "yellow_cards", "red_cards",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

    logger.info("Aggregating per player per season...")

    agg = (
        df.groupby(["name", "season", "position"], as_index=False)
        .agg(
            total_points=("total_points", "sum"),
            goals_scored=("goals_scored", "sum"),
            assists=("assists", "sum"),
            clean_sheets=("clean_sheets", "sum"),
            bonus=("bonus", "sum"),
            bps=("bps", "sum"),
            minutes=("minutes", "sum"),
            ict_index=("ict_index", "mean"),
            influence=("influence", "mean"),
            creativity=("creativity", "mean"),
            threat=("threat", "mean"),
            value=("value", "mean"),
            selected=("selected", "max"),
            saves=("saves", "sum"),
            goals_conceded=("goals_conceded", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            games=("GW", "count"),
        )
    )

    # Build the "teams" list (home + away teams seen in player's fixtures)
    # mirror: COLLECT(DISTINCT ht.name) + COLLECT(DISTINCT at.name)
    if "team" in df.columns:
        player_teams = (
            df.groupby(["name", "season"])["team"]
            .apply(lambda s: list(s.dropna().unique()))
            .reset_index()
            .rename(columns={"team": "teams"})
        )
    else:
        player_teams = (
            agg[["name", "season"]]
            .copy()
            .assign(teams=lambda d: [[] for _ in range(len(d))])
        )

    agg = agg.merge(player_teams, on=["name", "season"], how="left")

    # Round float columns for cleanliness
    for col in ["ict_index", "influence", "creativity", "threat", "value"]:
        agg[col] = agg[col].round(2)

    records = agg.to_dict("records")
    logger.info(f"  {len(records)} player-season records prepared")
    return records


# ---------------------------------------------------------------------------
# Build + save one model
# ---------------------------------------------------------------------------

def build_one_model(model_key: str, records: list[dict]) -> str:
    """Build embeddings for one model key and save to disk. Returns output path."""
    from embeddings.embedding_manager import EmbeddingManager

    logger.info(f"\n{'='*60}")
    logger.info(f"Building embeddings: {model_key}")
    logger.info(f"{'='*60}")

    manager = EmbeddingManager(model_key=model_key)

    # Use larger batches in CI (no memory constraints like Railway)
    manager.build_player_embeddings(records, batch_size=32)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{model_key}_embeddings.pkl")
    manager.save_embeddings(output_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    logger.info(f"Saved {len(manager.player_embeddings)} embeddings → {output_path} ({size_mb:.1f} MB)")

    # Free memory before next model
    del manager
    gc.collect()

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "both"

    valid = {"minilm", "mpnet", "both"}
    if model_arg not in valid:
        print(f"Usage: python build_embeddings_from_csv.py [minilm|mpnet|both]")
        sys.exit(1)

    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found: {DATA_PATH}")
        sys.exit(1)

    records = load_player_records(DATA_PATH)

    models_to_build = ["minilm", "mpnet"] if model_arg == "both" else [model_arg]

    for model_key in models_to_build:
        build_one_model(model_key, records)

    logger.info("\n✅ All embeddings built successfully.")


if __name__ == "__main__":
    main()
