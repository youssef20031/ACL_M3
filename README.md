# FPL FantasyTrivia - Graph-RAG System

A **Graph-RAG** (Retrieval-Augmented Generation) powered Fantasy Premier League trivia and Q&A assistant using Neo4j, React, FastAPI, and multiple LLMs via HuggingFace.

## 🎯 Features

- **Q&A Assistant**: Ask questions about FPL players, teams, and statistics
- **FantasyTrivia**: Test your FPL knowledge with dynamically generated questions
- **Player Search**: Search and analyze player statistics (accent-insensitive)
- **Player Comparison**: Compare two players head-to-head with radar and bar charts
- **Hybrid Retrieval**: Combines Cypher queries and embedding-based semantic search

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │   Q&A   │ │ Trivia  │ │ Search  │ │ Compare │            │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘            │
└───────┼──────────┼──────────┼──────────┼────────────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (:8000)                     │
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
    │ Queries  │    │ (2 models)│   │(HF API)  │
    └──────────┘    └──────────┘    └──────────┘
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

---

## 🚀 Running the Project

The app has three components that must all be running simultaneously: **Neo4j**, **FastAPI backend**, and the **React frontend**. Open a separate terminal for each.

### Prerequisites

- Python 3.9+ with a `.venv` virtual environment
- Node.js 18+
- Neo4j (Desktop or running via Docker)
- HuggingFace API token in `.env`

---

### Step 1 — Start Neo4j

**Option A: Neo4j Desktop**
Open Neo4j Desktop and start your database. It runs on `bolt://localhost:7687` by default.

**Option B: Docker**
First make sure Docker Desktop is open and running, then:
```powershell
docker compose up neo4j
```

Verify Neo4j is up before proceeding:
```powershell
netstat -ano | findstr ":7687"
# Should show a LISTENING entry
```

---

### Step 2 — Start the FastAPI Backend

Open a new terminal and run using the project's virtual environment directly:

```powershell
C:\ACL2\FPL\ACL_M3\.venv\Scripts\uvicorn.exe api_main:app --reload --port 8000 --app-dir C:\ACL2\FPL\ACL_M3
```

Or activate the venv first, then use uvicorn normally:

```powershell
# Activate (run once per terminal session)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
C:\ACL2\FPL\ACL_M3\.venv\Scripts\Activate.ps1

# Then start the server
uvicorn api_main:app --reload --port 8000
```

You should see:
```
✅ API Ready!
INFO: Application startup complete.
```

The Neo4j connection warning at startup is expected if Neo4j isn't up yet — you can connect manually from the Settings page.

---

### Step 3 — Start the React Frontend

Open another new terminal:

```powershell
cd C:\ACL2\FPL\ACL_M3
npm install        # first time only
npm run dev
```

The app will be available at **http://localhost:3000**

---

### Step 4 — Load the data (first time only)

```powershell
.venv\Scripts\python.exe load_data.py
```

This wipes Neo4j and loads all six seasons plus the 2026-27 fixtures (about 30 seconds locally, about 3 minutes against a remote database). Then open **http://localhost:3000**.

---

### Summary — Three terminals at once

| Terminal | Command |
|----------|---------|
| Neo4j | `docker compose up neo4j` or Neo4j Desktop |
| FastAPI | `C:\ACL2\FPL\ACL_M3\.venv\Scripts\uvicorn.exe api_main:app --reload --port 8000 --app-dir C:\ACL2\FPL\ACL_M3` |
| React | `cd C:\ACL2\FPL\ACL_M3 && npm run dev` |

---

## 📁 Project Structure

```
ACL_M3/
├── api_main.py            # FastAPI backend (all routes)
├── load_data.py           # Load or reload the Neo4j graph
├── Dockerfile, docker-compose.yml, requirements*.txt
├── package.json, vite.config.ts, tailwind.config.js, tsconfig*.json, index.html
│
├── src/                   # React frontend ("Pitch Me"): pages/, components/ui (design system),
│                          # components/shell (header, footer, search), services/api.ts, store/
├── public/                # Static assets (club crests in public/logos/)
│
├── config/                # settings.py: env vars, seasons, data paths
├── graph/                 # Neo4j connection, schema, loader, Cypher query library
├── preprocessing/         # Intent classifier and entity extractor (Q&A pipeline)
├── embeddings/            # Sentence-transformer embeddings (+ prebuilt pickles)
├── llm/                   # LLM manager (Groq) and prompt templates
├── services/              # Player index, avatars, predictions, fixtures, season dashboard
├── pipeline/              # Daily current-season job (FPL API -> data/live -> graph -> predictions)
├── ml/                    # Points model (v3), training, evaluation, legacy v2 stack
├── trivia/, utils/, migrations/
├── tests/                 # pytest suite
│
├── data/                  # Committed datasets: historical per-match CSVs, fixture lists
│   └── live/              # Pipeline output (gitignored; a Docker volume on the VM)
├── EFL_Data_All_Seasons/  # Championship data for promoted teams
├── scripts/               # One-off tools; run from the repo root: python -m scripts.<group>.<name>
│   ├── checks/            #   database/API checks and ad-hoc test scripts
│   ├── data/              #   fixture and season data preparation
│   ├── efl/               #   EFL (Championship) data ingestion
│   ├── embeddings/        #   rebuild embeddings from the CSV (used by CI)
│   └── analysis/          #   LLM evaluation, prediction plots
├── deploy/                # VM deployment: compose file, Caddy, deploy/setup scripts
├── docs/                  # Design brief, ML evaluation, reports, notes; docs/archive = history
└── legacy/                # Original Streamlit app and old API (not used)
```

---

## ☁️ Deployment

Production runs on a single VM with Docker Compose: Caddy (HTTPS and the built frontend), the FastAPI backend, and Neo4j. Setup steps for Oracle Cloud's free tier, plus day-to-day commands, are in **[deploy/README.md](deploy/README.md)**.

---

## 🔧 Environment Variables

Create or edit `.env` in the project root (for the VM, see `deploy/.env.example`):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
GROQ_API_KEY=gsk_your_key_here
# Optional
HUGGINGFACE_API_TOKEN=
GOOGLE_CUSTOM_SEARCH_API_KEY=
GOOGLE_CUSTOM_SEARCH_CX=
```

---

## 📚 Cypher Query Library

20+ parameterized queries covering:

| Query | Description |
|-------|-------------|
| `get_top_scorers_by_season` | Top goal scorers in a season |
| `get_top_assisters_by_season` | Top assist providers |
| `get_top_points_by_position` | Top FPL points by position |
| `get_player_season_stats` | Comprehensive player stats |
| `get_best_value_players` | Points per million analysis |
| `get_bonus_point_leaders` | Bonus point magnets |
| `get_clean_sheet_leaders` | Defenders/GKs with most clean sheets |
| `get_ict_index_leaders` | Highest ICT index players |
| `compare_players` | Head-to-head comparison |
| `get_head_to_head` | Team vs team results |

---

## 🤖 LLM Models

All served by Groq:

| Key | Model |
|-----|-------|
| `gpt-oss-120b` | OpenAI GPT-OSS 120B (default) |
| `gpt-oss-20b` | OpenAI GPT-OSS 20B (fast) |
| `qwen3-27b` | Qwen 3.8 27B |

---

## 📈 Embedding Models

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| MiniLM | 384 | ⚡ Fast | Good |
| MPNet | 768 | Normal | ⭐ Better |

---

## 🔍 Retrieval Methods

| Method | Description |
|--------|-------------|
| Baseline (Cypher) | Direct graph queries only |
| Embeddings | Semantic similarity search |
| Hybrid | Cypher + embeddings (recommended) |

---

## 🎮 Trivia Categories

- **Top Scorers** — goal and point leaders
- **Player Stats** — individual statistics
- **Records** — FPL records and achievements
- **True/False** — fact verification
- **Multiple Choice** — classic quiz format
- **Comparisons** — player vs player
- **Team Facts** — fixture and team questions

---

## 📄 License

MIT License — free to use for educational purposes.

## 🙏 Acknowledgments

- Fantasy Premier League for the data
- Neo4j for the graph database
- HuggingFace for model hosting
- React + Vite + Tailwind for the frontend
- FastAPI for the backend
