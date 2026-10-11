# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FPL FantasyTrivia: a Graph-RAG Fantasy Premier League assistant. React/Vite frontend (`src/`) → FastAPI backend (`api_main.py`) → Neo4j knowledge graph, plus prebuilt sentence-transformer embeddings, LLMs (Groq / HuggingFace Inference), and XGBoost point-prediction models (`ml/`).

## Commands

Backend (run from repo root; modules are imported relative to it):
```bash
.venv/Scripts/uvicorn.exe api_main:app --reload --port 8000
```
Frontend (dev server on :3000, proxies `/api`, `/health`, `/ws` to :8000 via `vite.config.ts`):
```bash
npm run dev
npm run build      # tsc && vite build → dist/
```
Load or reload the graph. This wipes the DB, takes ~3 min against Aura, and also loads 2026-27 fixtures from `FPL_2026_2027_fixtures_fixed.csv`. The UI has no load button; `POST /api/data/load` exists but skips fixtures.
```bash
.venv/Scripts/python.exe load_data.py
```
Local Neo4j alternative: `docker compose up neo4j` (bolt://localhost:7687, neo4j/password).

Dependencies: `requirements.txt` is runtime-only (what `api_main.py` imports); `requirements-dev.txt` adds pytest/hypothesis and training/analysis libs. In Linux containers install CPU torch first (see the Dockerfile).

Tests live in `tests/` (mix of `unittest` and pytest + hypothesis property tests; most mock `Neo4jConnection`). `tests/test_pipeline.py` asserts on substrings of the generated Cypher, so update it when rewriting a query in `graph/queries.py`. Run one test with:
```bash
.venv/Scripts/python.exe -m pytest tests/test_fixture_service.py::TestFixtureService::test_fixture_context_creation
```
`ml/test_*.py` are ad-hoc scripts, not a suite. `npm run lint` is defined but there is no ESLint config file, so it does not currently work.

Environment caveat: `.venv` is Python 3.14, while Docker/Railway/CI use 3.11.

## Configuration

`config/settings.py` loads `.env`: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `HUGGINGFACE_API_TOKEN`, `GROQ_API_KEY` (read directly in `llm/llm_manager.py`), optional `GOOGLE_CUSTOM_SEARCH_API_KEY`/`_CX` (avatar fallback), `REDIS_URL` (trivia cache) and `AVATAR_CACHE_PATH` (default `.cache/avatars.json`). `SEASONS` in settings lists the seasons with data; season extraction and defaults derive from it. The frontend uses `VITE_API_URL` in dev only; in production `src/services/api.ts` forces relative URLs.

## Backend architecture

`api_main.py` (~2000 lines) holds nearly all routes and glue. Shared state lives in the module-level `app_state` dict, populated in `lifespan()`: Neo4j connection, LLM manager, trivia cache, player index, avatar service/image proxy, embedding manager, ML integration. Routes get the connection through the `get_neo4j_conn` dependency; the connection can also be swapped at runtime via `POST /api/connection/connect` (Settings page), which calls `reset_graph_caches()`.

The Neo4j driver and the LLM clients are synchronous. Inside `async def` routes, wrap them (`await run_db(conn, q, p)`, `await asyncio.to_thread(...)`) or make the route a plain `def` so FastAPI runs it in its threadpool; a bare call blocks every other request for its full duration.

**Q&A pipeline (`POST /api/query`)** — the core flow, spread across several modules:
1. `preprocessing/intent_classifier.py` → `Intent` (rule/keyword based), and `get_query_type_for_intent()` maps an intent to a method name on `CypherQueries`.
2. `preprocessing/entity_extractor.py` → players/teams/seasons/stats/positions; `get_query_parameters()` builds kwargs.
3. The handler looks the method up with `getattr(CypherQueries, name)`, filters params by `inspect.signature`, and falls back to `get_top_players_all_positions` if required params are missing. A missing season defaults to `get_latest_season()` (latest season with `SEASON_STATS`). Adding a new query type means: add a static method to `graph/queries.py` returning `(cypher, params)`, then map an intent to it.
4. PREDICTION intent reads `ml/predictions/next_gw_predictions.csv` via `services/predictions.get_predictions_df()` (parsed once, re-read when the file's mtime changes) and falls back to live model inference.
5. Embeddings/Hybrid retrieval uses `embeddings/embedding_manager.py` with pickles in `embeddings/prebuilt/` (minilm, mpnet). Managers are constructed via `EmbeddingManager.__new__` so the transformer model is not loaded until a text query needs it. Pickles are rebuilt by `.github/workflows/build-embeddings.yml` (from the CSV, no Neo4j) and committed back.
6. `llm/prompts.py` `PromptTemplates.qa_template` builds the prompt; `llm/llm_manager.py` `LLMManager.generate` dispatches by the model's `provider` and walks `FALLBACK_ORDER` on quota or model-not-found errors. All live models are on Groq (`gpt-oss-120b` default, `gpt-oss-20b`, `qwen3-27b`); `RETIRED_MODELS` maps keys that browsers may still have persisted (`llama-3.3-70b`, the old HuggingFace models) to live ones. Keep it in sync with the model list in `src/pages/Settings.tsx` and the store migration in `src/store/appStore.ts`. `enforce_tied_top_scorers_in_answer` post-processes the answer.

**Graph schema** (`graph/schema.py`, `graph/data_loader.py`, `migrations/`):
- `(Player {name, name_normalized, last_team, last_season})-[:PLAYS_POSITION]->(Position)`: exactly one position, from the latest appearance.
- `(Player)-[:PLAYS_FOR]->(Team)`: current team, only for players in the latest season.
- `(Player)-[:PLAYED_IN {per-match stats, season, gameweek, team, was_home}]->(Fixture {season_id, gameweek})-[:PART_OF]->(Gameweek)-[:IN_SEASON]->(Season)`, plus `(Fixture)-[:HOME_TEAM|AWAY_TEAM]->(Team)`.
- `(Player)-[:SEASON_STATS]->(Season)`: precomputed per player-season **sums** (`goals`, `total_points`, `ict_sum`, `value_sum`, `games`, `appearances`, …) plus `team`/`position`. Season-level queries read these instead of aggregating `PLAYED_IN`. Averages are `sum / games`, so they stay correct across several seasons. Gameweek-level queries anchor on `Fixture {season_id, gameweek}` and expand `PLAYED_IN`; on Aura this beat relationship-property indexes, which were tried and dropped.
- 2026-27 fixtures: `(Fixture {season_id: '2026-27', kickoff_time: DateTime})-[:PLAYS_HOME|PLAYS_AWAY]->(Team)`, consumed by `services/fixture_service.py` (spec in `.kiro/specs/date-aware-match-predictions/`). Neo4j returns its own `DateTime` type; call `.to_native()` before JSON serialization.
- The loader drops `AM` (assistant-manager chip) rows. Player identity is the name string, so a few players appear under two spellings across seasons.

**Player search & names** (`services/player_index.py`): `PlayerIndex` is built once from Neo4j at startup (`get_player_index()`) and rebuilt after a reload. It powers search-as-you-type, `resolve()` (typed name → exact DB name, used by stats/compare/images) and `mentions()` (player names in a chat question, used by the entity extractor). Matching uses `utils/text.normalize_name` (handles `Ø`, `ł`, …), ranks by tier then popularity (current players, career points), and corrects typos word by word with rapidfuzz.

**Avatars** (`services/avatars.py`): sources are tried in this order:
1. FPL photo URL built from the player's `code` in the FPL bootstrap feed, which is fetched once in the background. The CDN allows hotlinking at `premierleague25/.../110x140/{code}.png`; the old `250x250/p{code}.jpg` format returns 403.
2. A single Wikipedia search.
3. Google CSE, if configured.

Results, including misses (which expire), persist to `.cache/avatars.json`. `/api/images/proxy` only fetches allow-listed hosts, checks every redirect hop, and keeps a small in-memory LRU.

**ML** (`ml/`): `ml/api_integration.py` defines `MLAPIIntegration` and `register_ml_routes` (`/api/ml/predict/*`, `/api/ml/status`). Routes are registered at import time against `_LazyMLIntegration`, a proxy that resolves the real instance from `app_state` after startup. Startup loads `ml/models/xgboost_{gk,def,mid,fwd}_v2.pkl`, falling back to linear regression. `/api/ml/top-performers` and `/api/ml/best-value` in `api_main.py` serve the precomputed CSV instead. Feature engineering and leakage checks are in `ml/feature_engineering.py` and `ml/validate_no_leakage.py`; promoted-team handling uses EFL data in `EFL_Data_All_Seasons/`.

## Frontend architecture

Routes in `src/App.tsx` (`/qa`, `/predictions`, `/trivia`, `/search`, `/compare`, `/settings`) under `components/Layout.tsx`. All HTTP goes through `src/services/api.ts` (axios, 60s timeout); global state in `src/store/appStore.ts` (zustand); TanStack Query for fetching. Pages are lazy-loaded per route; player photos render through `components/AvatarImage.tsx`. Styling is Tailwind with no theme tokens — colors are hardcoded utility classes per page, and dark mode is done by overriding gray/white classes in `src/index.css`.

## Deployment

Production is one VM (Oracle Cloud Ampere, **arm64**) running `deploy/docker-compose.yml`: `web` (Caddy: HTTPS, serves the built `dist/`, proxies `/api`, `/health`, `/ws`, `/docs`), `backend` (root `Dockerfile`), and `neo4j` (not public). See `deploy/README.md`.
- `deploy/setup-vm.sh` does one-time VM setup.
- `deploy/deploy.sh [--load-data]` pulls, rebuilds and restarts.
- `.github/workflows/deploy.yml` runs it over SSH once the `VM_*` secrets exist.
- Because the site and API share an origin there is no CORS config; extra origins come from `CORS_ORIGINS`.
- New runtime deps must ship linux/aarch64 wheels.
- The backend image is built from the repo root, so the root `.dockerignore` must keep excluding `deploy/` (which holds `deploy/.env`).

The root `docker-compose.yml` is for local dev only (Neo4j + backend). Railway, Vercel and Aura configs were removed.

## Repo layout notes

Past working notes and reports (`*_SUMMARY.md`, ML experiment write-ups) live in `docs/archive/`; they describe history, not current behavior. The repo root still has artifacts that are not part of the running app: one-off `check_*.py`/`test_*.py`/`verify_*.py`/`efl_*.py` scripts, `api_main_old.py`, and `app.py` (legacy Streamlit UI). The root-level `App.tsx`, `Layout.tsx`, `Settings.tsx`, `Trivia.tsx`, `QAAssistant.tsx`, `GraphVisualization.tsx`, `api.ts`, `appStore.ts` are stale copies — Vite builds only from `src/` (`index.html` → `/src/main.tsx`). Large data CSVs (`cleaned_merged_seasons_cleaned.csv`, `FPL_20xx_20xx.csv`) are committed and used by the loader, ML training, and embeddings build.
