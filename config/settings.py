"""
Configuration settings for FPL FantasyTrivia Graph-RAG System
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j Configuration (Local)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# HuggingFace Configuration
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

# LLM Models Configuration
LLM_MODELS = {
    "gemma-2-2b": {
        "name": "google/gemma-2-2b-it",
        "description": "Gemma 2 2B Instruct - Google's lightweight model",
        "max_tokens": 1024,
        "temperature": 0.7
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Mistral 7B Instruct - High quality open model",
        "max_tokens": 1024,
        "temperature": 0.7
    },
    "llama-3-8b": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "description": "Llama 3 8B Instruct - Meta's latest model",
        "max_tokens": 1024,
        "temperature": 0.7
    }
}

# Embedding Models Configuration
EMBEDDING_MODELS = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "MiniLM - Fast and efficient",
        "dimensions": 384
    },
    "mpnet": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "description": "MPNet - Higher quality embeddings",
        "dimensions": 768
    }
}

# FPL Domain Configuration
POSITIONS = ["GK", "DEF", "MID", "FWD"]
POSITION_NAMES = {
    "GK": "Goalkeeper",
    "DEF": "Defender", 
    "MID": "Midfielder",
    "FWD": "Forward"
}

SEASONS = ["2020-21", "2021-22", "2022-23"]

# Stats columns for embeddings
STAT_COLUMNS = [
    "goals_scored", "assists", "total_points", "bonus", "bps",
    "clean_sheets", "minutes", "ict_index", "influence", 
    "creativity", "threat", "value", "form", "selected"
]

# Data path
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cleaned_merged_seasons_cleaned.csv")

# UI Configuration
APP_TITLE = "FPL FantasyTrivia"
APP_ICON = "⚽"
APP_DESCRIPTION = "A Graph-RAG powered Fantasy Premier League Trivia & Q&A Assistant"
