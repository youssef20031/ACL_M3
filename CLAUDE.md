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
npm run dev:live   # same, but proxies the API to the deployed site (.env.live) — no backend needed
npm run build      # tsc && vite build → dist/
```
Load or reload the graph. A full load wipes the DB (~3 min against Aura), then adds promoted teams and the current season from `data/live/` (falling back to `data/FPL_2026_2027_fixtures_fixed.csv` for the schedule when no live data exists). The UI has no load button.
```bash
.venv/Scripts/python.exe load_data.py                 # full reload
.venv/Scripts/python.exe load_data.py --current-only  # just apply data/live to the graph
```

Current-season pipeline (`pipeline/`, data in `data/live/`, gitignored; a Docker volume on the VM):
```bash
.venv/Scripts/python.exe -m pipeline.current_season   # fetch from the FPL API -> data/live/*.csv
.venv/Scripts/python.exe -m pipeline.load_current     # apply it to the graph (idempotent, ~10 s)
.venv/Scripts/python.exe -m pipeline.predict_next_gw  # next-gameweek predictions -> data/live/
.venv/Scripts/python.exe -m pipeline.daily            # all of the above + API reload (what the VM timer runs)
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

`config/settings.py` loads `.env`: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `HUGGINGFACE_API_TOKEN`, `GROQ_API_KEY` (read directly in `llm/llm_manager.py`), optional `GOOGLE_CUSTOM_SEARCH_API_KEY`/`_CX` (avatar fallback), `REDIS_URL` (trivia cache) and `AVATAR_CACHE_PATH` (default `.cache/avatars.json`). `SEASONS` in settings lists the seasons with data, including the in-progress `CURRENT_SEASON` (2026-27); season extraction ("this season" = current, "last season" = previous), defaults and the season wording in `llm/prompts.py` derive from it. `LIVE_DATA_DIR` (default `data/live`) is where the pipeline reads and writes; `ADMIN_TOKEN` is required (as the `X-Admin-Token` header, via `services/security.require_admin`) by every state-changing endpoint: `/api/admin/reload`, `/api/data/load`, `/api/fixtures/load`, `/api/embeddings/build`, `/api/connection/*`; without it they return 404. `POST /api/query` is rate-limited per client (`QUERY_RATE_LIMITS`, default `10/60,120/3600`; the client is the right-most `X-Forwarded-For` entry set by Caddy). `QueryRequest` caps the question at 1000 chars and keeps at most 20 user/assistant history turns of 2000 chars (other roles are dropped); list endpoints cap `limit`/`top_k` at 50; `HEALTHCHECK_URL` gets pings from the daily job. The frontend uses `VITE_API_URL` in dev only; in production `src/services/api.ts` forces relative URLs.

## Backend architecture

`api_main.py` (~2000 lines) holds nearly all routes and glue. Shared state lives in the module-level `app_state` dict, populated in `lifespan()`: Neo4j connection, LLM manager, trivia cache, player index, avatar service/image proxy, embedding manager, ML integration. Routes get the connection through the `get_neo4j_conn` dependency; the connection can also be swapped at runtime via `POST /api/connection/connect` (Settings page). Both paths go through `_attach_neo4j()`, which resets graph caches, repoints the ML integration and rebuilds the player index. If Neo4j isn't reachable at startup, `lifespan()` keeps retrying in a background task (2 s doubling to 30 s) instead of giving up. After a VM reboot Docker restarts every container at once, ignoring `depends_on: service_healthy`, so the backend often starts before Neo4j.

The Neo4j driver and the LLM clients are synchronous. Inside `async def` routes, wrap them (`await run_db(conn, q, p)`, `await asyncio.to_thread(...)`) or make the route a plain `def` so FastAPI runs it in its threadpool; a bare call blocks every other request for its full duration.

**Q&A pipeline (`POST /api/query`)** — the core flow, spread across several modules:
1. `preprocessing/intent_classifier.py` → `Intent` (rule/keyword based), and `get_query_type_for_intent()` maps an intent to a method name on `CypherQueries`. Each matching pattern adds to an intent's score, and a tie goes to the intent listed first, except that a team intent with a club in its slot beats a player intent ("How did City do?", "Newcastle United vs Manchester United"). Team patterns take clubs from `EntityExtractor.TEAMS`.
2. `preprocessing/entity_extractor.py` → players/teams/seasons/stats/positions; `get_query_parameters()` builds kwargs (a mentioned player is `player_name` or `player1`/`player2`, plus `name_pattern` for the player search). Teams and stats match as whole words, longest key first (`EntityExtractor.find_teams`), so "Coventry City" isn't Man City and "goalkeepers" isn't goals. `TEAMS` maps every club in the graph to its graph name; add each season's promoted clubs there.
3. The handler looks the method up with `getattr(CypherQueries, name)`, filters params by `inspect.signature`, and falls back to `get_top_players_all_positions` if required params are missing. Before that, `IntentClassifier.refine_with_entities` turns a player intent with clubs but no extracted player into TEAM_ANALYSIS (one club) or HEAD_TO_HEAD (two). A missing season defaults to `get_latest_season()` (latest season with `SEASON_STATS`). Adding a new query type means: add a static method to `graph/queries.py` returning `(cypher, params)`, then map an intent to it.
4. PREDICTION intent reads `data/live/next_gw_predictions.csv` (written by `pipeline.predict_next_gw`; falls back to the bundled `ml/predictions/` copy) via `services/predictions.get_predictions_df()`, which picks the path per call and re-parses when the mtime changes. Rows include FPL `status`/`news`/`chance_of_playing_next_round`; `_availability_note` passes them to the LLM. Unavailable players stay in the file with 0 points on purpose, because a missing row would fall back to live inference and show non-zero points. TOP_SCORERS questions default to `sort_by="goals"`.
5. Embeddings/Hybrid retrieval uses `embeddings/embedding_manager.py` with pickles in `embeddings/prebuilt/` (minilm, mpnet). Managers are constructed via `EmbeddingManager.__new__` so the transformer model is not loaded until a text query needs it. Pickles are rebuilt by `.github/workflows/build-embeddings.yml` (from the CSV, no Neo4j) and committed back.
6. `llm/prompts.py` `PromptTemplates.qa_template` builds the prompt; `llm/llm_manager.py` `LLMManager.generate` dispatches by the model's `provider` and walks `FALLBACK_ORDER` on quota or model-not-found errors. All live models are on Groq (`gpt-oss-120b` default, `gpt-oss-20b`, `qwen3-27b`); `RETIRED_MODELS` maps keys that browsers may still have persisted (`llama-3.3-70b`, the old HuggingFace models) to live ones. Keep it in sync with the model list in `src/pages/Settings.tsx` and the store migration in `src/store/appStore.ts`. `enforce_tied_top_scorers_in_answer` post-processes the answer.

**Graph schema** (`graph/schema.py`, `graph/data_loader.py`, `migrations/`):
- `(Player {name, name_normalized, last_team, last_season, element_id, fpl_code, status, news, chance_of_playing_next_round, …})-[:PLAYS_POSITION]->(Position)`: exactly one position, the current one. `status`/`news` are FPL's official availability (a/d/i/s/u) and team news, refreshed by the pipeline.
- `(Player)-[:PLAYS_FOR]->(Team)`: the current 2026-27 squad, from the FPL status file. Players marked `u` whose news says they left (joined/loaned elsewhere) lose it but keep `last_team`.
- `(Player)-[:PLAYED_IN {per-match stats, season, gameweek, team, was_home}]->(Fixture {season_id, gameweek})-[:PART_OF]->(Gameweek)-[:IN_SEASON]->(Season)`, plus `(Fixture)-[:HOME_TEAM|AWAY_TEAM]->(Team)`.
- `(Player)-[:SEASON_STATS]->(Season)`: precomputed per player-season **sums** (`goals`, `total_points`, `ict_sum`, `value_sum`, `games`, `appearances`, …) plus `team`/`position`. Season-level queries read these instead of aggregating `PLAYED_IN`. Averages are `sum / games`, so they stay correct across several seasons. Gameweek-level queries anchor on `Fixture {season_id, gameweek}` and expand `PLAYED_IN`; on Aura this beat relationship-property indexes, which were tried and dropped.
- `Fixture.kickoff_time` is a Neo4j DateTime on every fixture (mixed string/DateTime values sort strings first and broke "latest matches" ordering across seasons).
- 2026-27 fixtures come from the FPL API, one node per match: `id = "{fpl_fixture_id}-2026-27"`, `status` (`scheduled`/`in_progress`/`finished`), scores only once finished. Each has both relationship styles: `PLAYS_HOME|PLAYS_AWAY` (fixture service, ML) and `HOME_TEAM|AWAY_TEAM` + `PART_OF` Gameweek (historical-style queries). Queries over results must filter `home_score IS NOT NULL`, or unplayed fixtures leak in (nulls sort first under DESC). The CSV schedule (`fixture-2026-27-…` ids) is only an offline fallback and is skipped once API fixtures exist. `load_data.py` first creates Team nodes (`promoted_season`, `efl_name`) for fixture-list clubs with no Premier League history, matched against the previous Championship table in `EFL_Data_All_Seasons/` (`--fixtures-only` reloads just this part). Fixture team names follow the graph's FPL short names (`Ipswich`, not `Ipswich Town`). Consumed by `services/fixture_service.py` (spec in `.kiro/specs/date-aware-match-predictions/`). Neo4j returns its own `DateTime` type; call `.to_native()` before JSON serialization.
- The loader drops `AM` (assistant-manager chip) rows. Player identity is the name string, so a few players appear under two spellings across seasons.

**Player search & names** (`services/player_index.py`): `PlayerIndex` is built once from Neo4j at startup (`get_player_index()`) and rebuilt after a reload. It powers search-as-you-type, `resolve()` (typed name → exact DB name, used by stats/compare/images) and `mentions()` (player names in a chat question, used by the entity extractor). Matching uses `utils/text.normalize_name` (handles `Ø`, `ł`, …), ranks by tier then popularity (current players, career points), and corrects typos word by word with rapidfuzz.

**Avatars** (`services/avatars.py`): sources are tried in this order:
1. FPL photo URL built from the player's `code` in the FPL bootstrap feed, which is fetched once in the background. The CDN allows hotlinking at `premierleague25/.../110x140/{code}.png`; the old `250x250/p{code}.jpg` format returns 403. About 1 in 6 codes has no photo (the CDN answers 403, mostly new signings), and the feed doesn't say which. So after loading the feed, every code gets one HEAD check in the background, and `peek()`/`get()` skip codes without a photo, including FPL URLs cached before the check.
2. A single Wikipedia search (top 3 hits), kept only when the page title matches the player's name.
3. Google CSE, if configured.

Results, including misses (which expire), persist to `.cache/avatars.json`. When no source has a photo, `PlayerAvatar` shows a blank silhouette. `/api/images/proxy` only fetches allow-listed hosts, checks every redirect hop, and keeps a small in-memory LRU.

**Season dashboard endpoints** (read-only GETs, all plain `def`): `/api/season/current`, `/api/fixtures?gameweek=`, `/api/table`, `/api/injuries?team=`, `/api/team-of-the-week?mode=predicted|actual&gameweek=`, `/api/players/leaders?season=&position=&sort=&limit=` (the Players page's default table; `games` = appearances, the `points_per_game` sort needs 270+ minutes). Their logic is in `services/season_service.py`: pure functions over plain lists, plus one cached query per view (2026-27 fixtures via `PLAYS_HOME|PLAYS_AWAY`, current-squad injuries, one gameweek's `PLAYED_IN` points). They also read `meta.json`, the status file (`web_name`, photo `code`) and the predictions file from `LIVE_DATA_DIR`. The in-process cache is keyed by inputs and file mtimes, has a 5-minute TTL, and `reset_graph_caches()` clears it, which covers `/api/admin/reload`. A missing gameweek returns 404 and Neo4j driver errors return 503. Photos come from `AvatarService.peek`, falling back to the graph's `fpl_code`, so these routes never make network calls. The table is computed from fixtures with stored scores. `Fixture.home_difficulty`/`away_difficulty` are FPL's FDR, set by `pipeline/load_current.py`. Team of the week uses `services/team_optimizer.py` (pure, scipy `milp`). Predicted mode picks a 15-man squad: £100.0m budget, at most 3 per club, `i/s/u/n` and players with no predicted points excluded. It maximises the XI first, then minimises the bench's cost, and the captain's points count double. Actual mode is the Dream Team: the best XI by gameweek points, with no budget, no club limit, no bench and no captain doubling.

**Page-view analytics** (`services/analytics.py`): cookie-free and aggregate-only, with no IP, user agent or identifier stored. `POST /api/analytics/pageview {path, referrer}` returns 204 and is sent by the frontend only after cookie consent. The path is reduced to its lowercase route (no query/hash or trailing slash, `^/[a-z0-9\-/]*$`, at most 64 chars, anything else becomes `/other`); the referrer becomes its host and is dropped when it matches the request's Host. The endpoint is rate-limited by `PAGEVIEW_RATE_LIMITS` (default `60/60,600/3600`). `PageViewStore` keeps per-UTC-day counts in memory and writes `$LIVE_DATA_DIR/analytics.json` atomically, at most every 10 s and on shutdown. It keeps 400 days and up to 200 paths/referrers per day (the rest pool into `/other`/`other`); a corrupt file is renamed to `.corrupt` and counting starts over. `GET /api/admin/analytics?days=30` (1..400, `require_admin`) returns `total_views`, zero-filled `by_day`, and the top 20 paths and referrers.

**ML** — the production model is **v3** (`ml/points_model.py`, `ml/models/points_v3.json` + `_meta.json`, XGBoost native JSON):
- **Data:** one row per player per fixture; the target is that fixture's points.
- **Features** (`build_features`) use only earlier gameweeks: rolling player form over 3 and 6 matches, the previous match on its own (`minutes_r1`, `started_r1`, `total_points_r1`), season to date, and team/opponent goals and xG over 6 and 20 matches (`team_gf`, `team_gf20`, ...) from a per-team-per-match table. Both fixtures of a double gameweek share pre-deadline values. The predicted fixture adds home/away, rest, fixtures in the gameweek, price and ownership rank.
- **Inference** appends "upcoming" rows to the same function, so training and serving can't drift apart.
- **Excluded:** FPL's historical `xP` is deliberately not a feature. It was recorded during gameweeks (it knows lineups) and leaked badly.
- **Retrain:** `python -m ml.train_points_v3` (~40 s, `train_and_report()`). It trains on 2020-21..2023-24, early-stops on 2024-25, tests on 2025-26 plus the current season against baselines, then refits on everything and writes `ml/models/points_v3_report.json`. Don't train or evaluate on the user's PC (heavy jobs have crashed it): the **ML train** workflow (Actions tab, `workflow_dispatch`) fetches the current season, runs `ml.train_points_v3` and `ml.evaluate_points_v3`, and uploads the bundled model files plus `docs/ml_evaluation/` as an artifact to commit.
- **Weekly retrain on the VM:** `pipeline/retrain.py`, run by the Monday-morning daily job in a child process (training peaks ~0.8 GB). A candidate goes to `$LIVE_DATA_DIR/models/` and replaces the active model only if its held-out top-20 beats every baseline and is within 10% of the active model's. Every attempt is logged in `retrain_history.json`. `pm.active_model_paths()` / `pm.load_model()` serve the live copy when present, else the bundled `ml/models/points_v3.json`. Force a run with `python -m pipeline.retrain --force` or `python -m pipeline.daily --retrain`.
- **Evaluation:** judge changes by test top-20 pick points and captain points, not training R². Before changing features or training settings, run the **ML backtest** workflow (`scripts/analysis/ml_backtest.py`): it scores variants on four held-out seasons (2023-24 .. current) under the refit, early-stopping and weekly-retrain protocols with 3 seeds and paired bootstrap CIs, plus subsets (promoted clubs' first 6 GWs, players new from the Championship, teams within 60 days of a manager change). Tried in 2026-10 and not adopted (no reliable gain over the current features): per-90 rates, a 10-match window, recency weighting, training on the last 3 seasons only, DC-points features, depth 6, `min_child_weight` 200, dropping `career_matches_before`, days since a manager change (`data/manager_changes.csv`), starting-XI rotation, and previous-Championship-season team/player stats (`EFL_Data_All_Seasons/`). The last-match and 20-match team features already capture most of what the manager and Championship data add.
- **Leakage tests:** `tests/test_points_model.py` checks no row duplication, gameweek freezing, and team strength excluding the match itself.

`pipeline/predict_next_gw.py` serves v3 for the next FPL gameweek:
- It reads the last 3 historical seasons plus the current one. Measured against all seasons: correlation 0.9998, same top 20, half the memory.
- Double gameweeks are summed; blank gameweeks get 0.
- Points are scaled by chance of playing, or set to 0 when the player is unavailable.

All user-facing predictions come from that file: chat, search, `/api/ml/top-performers`, `/api/ml/best-value`, and `POST /api/ml/predict/player` via `_LazyMLIntegration`. A player missing from it isn't in the current FPL game, and gets a 404 or "no prediction", not a guess.

The legacy v2 stack is still loaded at startup by `ml/api_integration.py` (`MLAPIIntegration`, `xgboost_*_v2.pkl`, `ml/feature_engineering.py`), but only as a fallback when no predictions file exists. Its training pipeline duplicated rows about 20x through `FeatureEngineer`'s opponent merges, so don't build on it. Promoted-team handling there uses EFL data in `EFL_Data_All_Seasons/`.

## Frontend architecture

The site is branded **Pitch Me** (FPL-inspired, never using Premier League/FPL marks). Routes in `src/App.tsx`, all under `components/shell/AppShell.tsx` (sticky `SiteHeader` with a "More" menu, `DeadlineBar` from `/api/season/current`, `SiteFooter` with the not-affiliated disclaimer, `PlayerSearchDialog` opened by the header or `/`):
- `/` Home dashboard, `/fixtures?gw=`, `/table`, `/team-of-the-week?mode=actual&gw=`, `/players`, `/predictions`, `/assistant` (full-height, no footer), `/compare`, `/trivia`, `/settings`, and a 404.
- `/qa` and `/search` redirect to `/assistant` and `/players`, keeping router state. Pages pass state between each other: `/players` takes `{ player, season, returnTo, compare }`, and `/assistant` takes `{ question }` and sends it.
- `/design` (outside the shell, not linked) shows every design-system component with sample data.

All HTTP goes through `src/services/api.ts` (axios, 60s timeout); global state in `src/store/appStore.ts` (zustand, persist version 4); TanStack Query for fetching. Pages are lazy-loaded per route; player photos render through `components/AvatarImage.tsx`. Search results carry an `avatar` when the backend knows it without a network call (`AvatarService.peek`); rows without one are filled by `hooks/usePlayerAvatars.ts`, which caches per name under `['player-avatar', name]`, shared across pages. `hooks/useHealthSync.ts` polls `['health']` and mirrors it into the store.

**Site essentials.** Every page calls `hooks/usePageMeta({ title, description })` (document title, meta description, OG/Twitter title+description, canonical); `index.html` holds the site-level OG/Twitter tags, icon set and manifest, with `%SITE_URL%` filled at build by the `pitchme-site-essentials` plugin in `vite.config.ts`, which also emits `robots.txt` and `sitemap.xml` (route list `SITEMAP_ROUTES`). `SITE_URL` defaults to the sslip address; Docker passes `SITE_ADDRESS` as the build arg and `toSiteUrl()` takes the first address of a Caddy list (adding `https://`), falling back to the default instead of failing the build. `node scripts/site/render.mjs` regenerates `public/og-image.png` and the favicons. The consent banner (`shell/ConsentBanner`, state in `lib/consent.ts`, key `pm-analytics-consent`, GPC = declined) gates `apiService.trackPageview`, called on route changes in `AppShell`. `/privacy`, `/terms` and `/contact` describe what the site actually does; update `/privacy` if data handling changes. `deploy/Caddyfile` lists the SPA routes: listed routes get 200, unknown paths get the app's 404 page with a real 404 status, so add new routes there and to `SITEMAP_ROUTES` too.

**Design system.** Colours are CSS variables in `src/styles/tokens.css`, mapped to Tailwind names in `tailwind.config.js` (`bg`, `surface`, `brand`, `line`, `fg`, `accent` mint, `accent-2` cyan, `pos-*`, `fdr-*`; font sizes `display/h1/h2/h3/body/data/micro`; Outfit + Inter from `@fontsource`). Use the tokens only, never raw palette classes or hex. Charts and SVG use `rgb(var(--accent))`. Shared components live in `src/components/ui` (barrel `index.ts`: Button/ButtonLink, Card, PageHeader, Tabs, DataTable, badges, ClubCrest, PlayerAvatar, Pitch, FixtureRow, states); page-specific pieces go in `src/components/<area>/`. Always merge classes with `cn()` (`utils/cn.ts`), which registers the custom font sizes with tailwind-merge; without that, `text-h1` is dropped when a colour class follows. The UI is dark-only: the store migration forces `theme: 'dark'`. Light mode is a future `[data-theme="light"]` token block. Club crests are 128px PNGs in `public/logos/`. `docs/FRONTEND_REDESIGN_BRIEF.md` lists the constraints (API contract, query keys, store shape, acceptance checklist).

## Deployment

Production is one Ubuntu VM with 4 GB+ RAM (Azure for Students x86_64 now; Oracle Ampere arm64 also supported, so keep both architectures building) running `deploy/docker-compose.yml`. Neo4j memory defaults (`NEO4J_HEAP=1G`, `NEO4J_PAGECACHE=512M`) are sized for 4 GB; the backend peaks around 0.9 GB. Services: `web` (Caddy: HTTPS, serves the built `dist/`, proxies `/api`, `/health`, `/ws`, `/docs`), `backend` (root `Dockerfile`), and `neo4j` (not public). See `deploy/README.md`.
- `deploy/setup-vm.sh` does one-time VM setup.
- `deploy/deploy.sh [--load-data]` pulls, rebuilds and restarts.
- `.github/workflows/deploy.yml` runs it over SSH once the `VM_*` secrets exist.
- Because the site and API share an origin there is no CORS config; extra origins come from `CORS_ORIGINS`.
- New runtime deps must ship linux/aarch64 wheels.
- The backend image is built from the repo root, so the root `.dockerignore` must keep excluding `deploy/` (which holds `deploy/.env`).

The root `docker-compose.yml` is for local dev only (Neo4j + backend). Railway, Vercel and Aura configs were removed.

## Repo layout notes

The root holds only entry points and config (`api_main.py`, `load_data.py`, Docker, package and build configs). Committed datasets are in `data/` (`cleaned_merged_seasons_cleaned.csv`, `FPL_20xx_20xx.csv`, the 2026-27 fixture CSVs, and `manager_changes.csv` from `python -m scripts.data.fetch_manager_changes`, used only by the backtest); their paths come from `config/settings.py` (`DATA_DIR`, `DATA_PATH`, `FIXTURES_FALLBACK_CSV`) and `ml/points_model.HISTORY_CSV`, while the many older `ml/` scripts use `data/...` relative to the repo root. One-off tools are in `scripts/` (`checks/`, `data/`, `efl/`, `embeddings/`, `analysis/`; see `scripts/README.md`), run from the root as `python -m scripts.<group>.<name>`. `legacy/` holds the original Streamlit `app.py` and `api_main_old.py`, not used. Past working notes live in `docs/archive/`, describing history, not current behavior; course reports and the LLM evaluation charts are in `docs/reports/`.
