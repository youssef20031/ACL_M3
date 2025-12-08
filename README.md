# FPL FantasyTrivia - Graph-RAG System

A **Graph-RAG** (Retrieval-Augmented Generation) powered Fantasy Premier League trivia and Q&A assistant using Neo4j, Streamlit, and multiple LLMs via HuggingFace.

## 🎯 Features

- **Q&A Assistant**: Ask questions about FPL players, teams, and statistics
- **FantasyTrivia**: Test your FPL knowledge with dynamically generated questions
- **Player Search**: Search and analyze player statistics
- **Player Comparison**: Compare two players head-to-head
- **Model Comparison**: Compare responses from different LLMs
- **Hybrid Retrieval**: Combines Cypher queries and embedding-based semantic search

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │   Q&A   │ │ Trivia  │ │ Search  │ │ Compare │            │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘            │
└───────┼──────────┼──────────┼──────────┼────────────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Preprocessing Layer                          │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ Intent Classifier │  │ Entity Extractor │                 │
│  └──────────────────┘  └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Neo4j   │    │Embeddings│    │   LLM    │
    │  Cypher  │    │  Search  │    │ Manager  │
    │ Queries  │    │ (2 models)│   │(3 models)│
    └──────────┘    └──────────┘    └──────────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
              ┌─────────────────────┐
              │   Final Response    │
              └─────────────────────┘
```

## 📊 Knowledge Graph Schema

```
(Player)-[:PLAYS_POSITION]->(Position)
(Player)-[:PLAYS_FOR]->(Team)
(Player)-[:PLAYED_IN {stats}]->(Fixture)
(Fixture)-[:HOME_TEAM]->(Team)
(Fixture)-[:AWAY_TEAM]->(Team)
(Fixture)-[:PART_OF]->(Gameweek)
(Gameweek)-[:IN_SEASON]->(Season)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Neo4j Desktop or Neo4j AuraDB
- HuggingFace account (for API token)

### Installation

1. **Clone/Download the project**:
   ```bash
   cd d:\ACL\M3
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   copy .env.example .env
   # Edit .env with your Neo4j and HuggingFace credentials
   ```

5. **Start Neo4j** and create a database

6. **Load FPL data**:
   ```bash
   python load_data.py
   ```

7. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
M3/
├── app.py                    # Main Streamlit application
├── load_data.py              # Data loading script
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── cleaned_merged_seasons_cleaned.csv  # FPL data (2020-21, 2021-22, 2022-23)
├── README.md                 # This file
│
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration settings
│
├── graph/
│   ├── __init__.py
│   ├── connection.py         # Neo4j connection handler
│   ├── schema.py             # Graph schema definition
│   ├── queries.py            # Cypher query library (20+ queries)
│   └── data_loader.py        # CSV to Neo4j loader
│
├── preprocessing/
│   ├── __init__.py
│   ├── intent_classifier.py  # Query intent classification
│   └── entity_extractor.py   # Named entity extraction
│
├── embeddings/
│   ├── __init__.py
│   └── embedding_manager.py  # Embedding models & search
│
├── llm/
│   ├── __init__.py
│   ├── llm_manager.py        # LLM integration
│   └── prompts.py            # Prompt templates
│
├── trivia/
│   ├── __init__.py
│   └── trivia_generator.py   # Trivia question generation
│
└── utils/
    ├── __init__.py
    ├── helpers.py            # Utility functions
    └── visualization.py      # Graph visualization
```

## 🔧 Configuration

### Neo4j Setup

1. Download [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new project and database
3. Start the database
4. Note the connection URI (default: `bolt://localhost:7687`)

### HuggingFace API

1. Create account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create a new token with read permissions
4. Add token to `.env` file

## 📚 Cypher Query Library

The system includes 20+ parameterized Cypher queries:

| Query | Description |
|-------|-------------|
| `get_top_scorers_by_season` | Top goal scorers in a season |
| `get_top_assisters_by_season` | Top assist providers |
| `get_top_points_by_position` | Top FPL points by position |
| `get_player_season_stats` | Comprehensive player stats |
| `get_best_value_players` | Points per million analysis |
| `get_bonus_point_leaders` | Bonus point magnets |
| `get_clean_sheet_leaders` | Defenders/GKs with most CS |
| `get_ict_index_leaders` | Highest ICT index players |
| `compare_players` | Head-to-head comparison |
| `get_head_to_head` | Team vs team results |
| ... and more |

## 🤖 LLM Models

| Model | Description |
|-------|-------------|
| `gemma-2-2b` | Google's lightweight model (default) |
| `mistral-7b` | High-quality open model |
| `llama-3-8b` | Meta's latest instruction model |
| `phi-3-mini` | Microsoft's compact model |
| `zephyr-7b` | Fine-tuned Mistral |

## 🎮 Trivia Categories

- **Top Scorers**: Questions about goal/point leaders
- **Player Stats**: Individual player statistics
- **Records**: FPL records and achievements
- **True/False**: Fact verification questions
- **Multiple Choice**: Classic quiz format
- **Comparisons**: Player vs player questions
- **Team Facts**: Fixture and team questions

## 📈 Embedding Models

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| MiniLM | 384 | ⚡ Fast | Good |
| MPNet | 768 | Normal | ⭐ Better |

## 🔍 Retrieval Methods

1. **Baseline (Cypher)**: Direct graph queries
2. **Embeddings**: Semantic similarity search
3. **Hybrid**: Combines both methods

## 📄 License

MIT License - Feel free to use for educational purposes.

## 🙏 Acknowledgments

- Fantasy Premier League for the data
- Neo4j for the graph database
- HuggingFace for model hosting
- Streamlit for the UI framework
