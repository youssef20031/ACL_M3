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

### Step 4 — First-Time Setup (in the browser)

1. Open **http://localhost:3000**
2. Go to **Settings**
3. Under **Neo4j Connection**, enter:
   - URI: `bolt://localhost:7687`
   - Username: `neo4j`
   - Password: `password`
   - Click **Connect**
4. Under **Data Management**, click **📥 Load FPL Data** — this imports the CSV into Neo4j (takes 1–3 minutes)
5. Optionally click **🔮 Build Embeddings** to enable Hybrid/Embeddings retrieval mode

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
├── api_main.py               # FastAPI backend (replaces app.py)
├── app.py                    # Original Streamlit app (legacy)
├── load_data.py              # Standalone data loading script
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (HF token, Neo4j creds)
├── cleaned_merged_seasons_cleaned.csv  # FPL data (2020-21, 2021-22, 2022-23)
│
├── src/                      # React frontend (Vite + TypeScript)
│   ├── main.tsx              # Entry point
│   ├── App.tsx               # Router setup
│   ├── App.css / index.css   # Global styles
│   ├── components/
│   │   ├── Layout.tsx        # Sidebar + navigation
│   │   └── GraphVisualization.tsx  # Canvas-based graph renderer
│   ├── pages/
│   │   ├── QAAssistant.tsx   # Chat interface
│   │   ├── Trivia.tsx        # Trivia game
│   │   ├── PlayerSearch.tsx  # Player search + detail view
│   │   ├── PlayerComparison.tsx  # Head-to-head comparison
│   │   └── Settings.tsx      # Connection + config panel
│   ├── services/
│   │   └── api.ts            # Axios API client
│   ├── store/
│   │   └── appStore.ts       # Zustand global state
│   └── utils/
│       └── cn.ts             # Tailwind class helper
│
├── config/
│   └── settings.py           # App configuration
├── graph/
│   ├── connection.py         # Neo4j connection handler
│   ├── queries.py            # Cypher query library (20+ queries)
│   ├── schema.py             # Graph schema definition
│   └── data_loader.py        # CSV → Neo4j loader
├── preprocessing/
│   ├── intent_classifier.py  # Query intent classification
│   └── entity_extractor.py   # Named entity extraction
├── embeddings/
│   └── embedding_manager.py  # Sentence-transformer embeddings
├── llm/
│   ├── llm_manager.py        # HuggingFace LLM integration
│   └── prompts.py            # Prompt templates
├── trivia/
│   └── trivia_generator.py   # Trivia question generation
│
├── vite.config.ts            # Vite config (proxies /api → :8000)
├── tailwind.config.js        # Tailwind CSS config
├── tsconfig.json             # TypeScript config
└── package.json              # Node dependencies
```

---

## ☁️ Deployment

The app deploys as three separate services:

| Component | Platform | Cost |
|-----------|----------|------|
| React frontend | Vercel | Free |
| FastAPI backend | Render | Free (sleeps after 15min inactivity) |
| Neo4j database | AuraDB | Free tier |

---

### 1 — Neo4j AuraDB (database)

1. Go to [console.neo4j.io](https://console.neo4j.io) and create a free account
2. Create a new **AuraDB Free** instance
3. Download the credentials file when prompted — you need:
   - **Connection URI** (looks like `neo4j+s://xxxxxxxx.databases.neo4j.io`)
   - **Username** (usually `neo4j`)
   - **Password** (auto-generated)
4. Keep these — you'll add them as environment variables in Railway

---

### 2 — Render (FastAPI backend)

1. Go to [render.com](https://render.com) and sign in with GitHub
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Render will auto-detect Python using `render.yaml`. Confirm these settings:
   - **Runtime**: Python
   - **Build Command**: *(auto-filled from render.yaml)*
   - **Start Command**: `uvicorn api_main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. Scroll to **Environment Variables** and add:

```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_aura_password
HUGGINGFACE_API_TOKEN=hf_your_token_here
```

6. Click **Create Web Service**. Render will build and deploy (takes ~5 minutes first time)
7. Copy the public URL it gives you (e.g. `https://fpl-fantasytrivia-api.onrender.com`)

> **Note**: On the free tier, the service spins down after 15 minutes of inactivity. The first request after that takes ~30 seconds to wake up. This is normal.

---

### 3 — Vercel (React frontend)

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click **Add New Project** and import this repository
3. Vercel will detect Vite automatically. Leave all build settings as-is
4. Before deploying, add this **Environment Variable**:
   ```
   VITE_API_URL=https://fpl-fantasytrivia-api.onrender.com
   ```
   (replace with your actual Render URL)
5. Also open `vercel.json` and replace both `REPLACE_WITH_YOUR_RAILWAY_URL` with your Render URL:
   ```json
   "destination": "https://fpl-fantasytrivia-api.onrender.com/api/:path*"
   ```
6. Click **Deploy**

Your app will be live at `https://your-project.vercel.app`

---

### 4 — Load data after deployment

Once both services are running:
1. Open your Vercel URL
2. Go to **Settings**
3. The Neo4j connection will auto-connect using the Railway env vars
4. Click **📥 Load FPL Data** to populate AuraDB
5. Click **🔮 Build Embeddings** if you want semantic search

---

### Environment variables reference

| Variable | Where | Description |
|----------|-------|-------------|
| `NEO4J_URI` | Render | AuraDB connection URI |
| `NEO4J_USER` | Render | AuraDB username |
| `NEO4J_PASSWORD` | Render | AuraDB password |
| `HUGGINGFACE_API_TOKEN` | Render | HuggingFace API token |
| `VITE_API_URL` | Vercel | Render backend public URL |

---

## 🔧 Environment Variables

Create or edit `.env` in the project root:

```env
HUGGINGFACE_API_TOKEN=hf_your_token_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
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

| Key | Model |
|-----|-------|
| `qwen-2.5-coder` | Qwen 2.5 Coder (default) |
| `llama-3.2-3b` | Llama 3.2 3B Instruct |
| `qwen-2.5-7b` | Qwen 2.5 7B |

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
