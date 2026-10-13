# Frontend redesign brief

A brief for redesigning the UI/UX of the FPL Assistant frontend (`src/`). Read it fully
before changing anything. It says what the app does, what must keep working, what may
change, and how the work is checked.

## 1. The product

**FPL Assistant** is a Fantasy Premier League companion. It's a portfolio piece for an AI/ML
engineer, so it should look like a polished, credible data product.

- **Ask (chat):** natural-language questions about players, teams and seasons, answered by an
  LLM over a Neo4j knowledge graph (Graph-RAG). Answers can show how they were retrieved
  (intent, Cypher query, graph view).
- **Predictions:** next-gameweek expected points for every player from an XGBoost model,
  with official FPL availability (injured, doubtful, suspended).
- **Player search and profiles:** stats per season (2020-21 to the current 2026-27), charts,
  photos, club badges.
- **Compare:** two players head to head.
- **Trivia:** a quiz generated from the graph.
- **Settings:** LLM model, retrieval method, embedding model, theme, system status.

Users are FPL managers who want quick answers before each gameweek deadline. They use the
site on phones as much as on desktop.

## 2. What's wrong with the current UI

- It looks generic "AI-generated": purple/violet gradients and accents everywhere
  (`purple-500/600/700`, `violet-300/400/500`, ~100 occurrences), emoji in headings
  ("💬 FPL Q&A Assistant", "⚙️ Settings").
- No design system: colors are hard-coded Tailwind classes in each page; `tailwind.config.js`
  has no theme tokens.
- Dark mode is a hack: `src/index.css` overrides `.bg-white`, `.text-gray-*` etc. under
  `html.dark`, and pages pick classes with `isDark ? ... : ...` ternaries.
- Inconsistent spacing, type scale, cards, buttons and states between pages.
- Player Search makes you press a button even though the API answers in ~10 ms.
- Availability (injuries, team news) isn't visible in the UI.

## 3. Design direction

> **Decided (October 2026): "Pitch Me".** The owner chose a recreation inspired by the
> official FPL website rather than the editorial direction first suggested here. The ban
> was on generic violet/indigo AI gradients, not on FPL's own palette.

- **Brand:** "Pitch Me", with an original pitch-outline wordmark. Never use Premier League or
  FPL logos, marks or fonts; the footer says the site is not affiliated.
- **Palette:** deep purple surfaces (`#1E0123` page, `#37003C` brand), mint `#00FF85` for
  actions and active states, cyan `#04F5FF` as secondary and focus colour. All colours are CSS
  variables in `src/styles/tokens.css`, wired into Tailwind (`bg`, `surface`, `brand`, `fg`,
  `accent`, …); contrast ratios are documented there.
- **Themes:** dark only for now. Light mode is a later `[data-theme="light"]` token block; no
  component changes should be needed.
- **Type:** Outfit (display) and Inter (UI), self-hosted via `@fontsource`; tabular numerals for
  stats.
- **Components:** `src/components/ui` (live reference at the hidden `/design` route); app shell in
  `src/components/shell`.
- **Pages:** Home dashboard, Fixtures, Table, Team of the Week (pitch view), Players, Predictions,
  Assistant, plus Compare, Trivia and Settings under "More". Only real data, no placeholder news.
- Data density, semantic colours (positions, availability, fixture difficulty), club badges and
  player photos as imagery, and functional-only motion still apply.

## 4. Hard constraints (don't break these)

**Stack:** React 18, TypeScript (strict), Vite, Tailwind 3, TanStack Query 5, zustand 4,
react-router 6, recharts, react-markdown, lucide-react, framer-motion.
- Prefer what's installed.
- New dependencies need a reason and must be small. Headless primitives like Radix are OK if
  justified. No full UI kits that impose their own look.

**API contract:**
- `src/services/api.ts` is the only place that talks to the backend.
- Don't change endpoints, request bodies or how responses are interpreted.
- You may add new methods that call existing endpoints, and add optional fields that the
  backend already returns (see section 6).
- No backend changes.

**Data-fetching behavior:** keep these working (moving code is fine, losing the behavior
isn't):
- TanStack Query keys and caching: `['health']` (polled every 30 s by the layout, shared with
  Settings); `['players','search',term,limit]`; `['players','stats',name,season]`;
  `['player-avatar',name]` via `src/hooks/usePlayerAvatars.ts`; the 15-minute cache on
  predictions.
- Debounced search with stale-response protection (see `PlayerComparison.tsx`).

**State:**
- Keep the `src/store/appStore.ts` state shape, the persisted keys (`fpl-app-storage`) and the
  `version: 3` migration that maps retired model keys.
- Model keys sent to the API: `gpt-oss-120b`, `gpt-oss-20b`, `qwen3-27b`.
- Retrieval methods: `Baseline`, `Embeddings`, `Hybrid`.
- Embedding models: `minilm`, `mpnet`.

**Routes:** `/qa`, `/predictions`, `/trivia`, `/search`, `/compare`, `/settings`
(`/` → `/qa`). Keep route-level lazy loading in `src/App.tsx`. You may rename labels in the
UI, but keep the paths (or add redirects).

**Images:**
- Keep `src/components/AvatarImage.tsx` semantics: lazy loading, fallback on error, the
  fallback is the caller's.
- Club badges come from `src/utils/clubLogos.ts` + `public/logos/`.

**Build and quality:**
- `npm run build` (tsc + vite) must pass with no type errors.
- No `any` creep in new code.
- Production builds use relative API URLs (see `api.ts`); keep that.

**Accessibility and responsiveness:**
- WCAG AA contrast in both themes.
- Visible focus states, keyboard navigable, semantic HTML, labelled controls.
- Layouts work from 360 px phones to 1440 px+ desktops. No horizontal page scroll.
- Navigation must work on mobile (e.g. bottom tab bar or drawer).

**Theming:**
- Implement design tokens as CSS variables (light + dark), wired into
  `tailwind.config.js` `theme.extend`.
- Delete the `html.dark .bg-white`-style overrides in `src/index.css` and the per-component
  `isDark ? … : …` class ternaries.
- Theme state stays in the store (`theme`, `toggleTheme`) and is applied as the `dark` class
  on `<html>`.

## 5. Page-by-page: keep, and improve

**Layout (`components/Layout.tsx`)**
- Keep: navigation to all pages; the backend/ML status from `/health`; the theme toggle.
- Improve: a real app shell (desktop sidebar or top bar, mobile tab bar); status shown
  quietly, not as a big banner.

**Ask (`pages/QAAssistant.tsx`)**
- Keep:
  - chat history (store), markdown answers, sending the current settings (model, retrieval
    method, embedding model)
  - the first-message flag and chat history sent with each query
  - the retrieval details: intent, extracted entities, Cypher query, KG context, embedding
    context, ML context, and the graph visualization (`components/GraphVisualization.tsx`)
- Improve:
  - suggested starter questions (e.g. "Top scorers this season", "Who should I captain
    next gameweek?", "Compare Saka and Palmer")
  - retrieval details as a collapsible "How I found this" panel
  - better loading states (answers take 0.5–2 s)
  - copyable Cypher

**Predictions (`pages/Predictions.tsx`)**
- Keep: top performers and best value, the position filter, fixture context (opponent,
  home/away, kickoff, gameweek).
- Improve:
  - show availability: status chip, chance of playing, news on hover/tap
  - make the gameweek and deadline prominent
  - a scannable ranked table with badges
  - the best-value view shows price and value score

**Player search (`pages/PlayerSearch.tsx`)**
- Keep:
  - results with photo, club badge, position, predicted points
  - the profile view with season selector (seasons from 2020-21 to 2026-27, plus
    "All seasons"), stats and charts
  - navigation in from other pages via route state (`location.state.player`)
- Improve:
  - search-as-you-type (debounce ~200 ms; the API is fast, typo tolerant and
    accent-insensitive)
  - a real profile page layout: header with photo, club, position, price and availability;
    season tabs; per-90 and per-game stats; charts
  - a "Compare with…" action

**Compare (`pages/PlayerComparison.tsx`)**
- Keep: two pickers with autocomplete and photos, the season selector, comparison charts,
  the summary.
- Improve: a clear side-by-side layout with the better value highlighted per stat; works on
  mobile (stacked).

**Trivia (`pages/Trivia.tsx`)**
- Keep: difficulty selection, question and options, answer checking, explanation, score
  (store).
- Improve: a game-like but on-brand feel, clear correct/incorrect feedback, a streak or score
  summary.

**Settings (`pages/Settings.tsx`)**
- Keep: model, retrieval method and embedding model choices (exact keys above); health
  (Neo4j, LLM, embeddings, ML); database stats.
- Improve: grouped, explained options; status as a tidy diagnostics panel.

Global needs on every page:
- loading skeletons
- empty states
- error states with a retry
- a consistent page header pattern

## 6. API reference (what the backend returns)

The authoritative types are in `src/services/api.ts`. Useful facts:
- `GET /health`: `neo4j` status, `neo4j_stats` (node/relationship counts),
  `llm_available`, `embeddings_built`, `ml_available`, `ml_model_type`.
- `POST /api/query` `{question, model, retrieval_method, embedding_model, is_first_message,
  chat_history}` returns:
  - `answer` (markdown), `intent`, `entities`, `cypher_query`
  - `kg_context`, `embedding_context`, `ml_context`
  - `results[]`, `graph_data {nodes[], edges[]}`
- `POST /api/players/search` `{query, limit, include_avatars}` → `{players: [{player_name,
  position, element_id, team_name, avatar}]}`.
  - `avatar` is set when the backend knows the photo instantly; it may be null, and
    `usePlayerAvatars` fetches the rest.
  - With `include_avatars: true`, results also include `predicted_points`.
- `POST /api/players/stats` `{player_name, season?}` → `{stats: {...}}`, season totals and
  averages plus `avatar`.
- `POST /api/players/compare` `{player1, player2, season?}` → `{comparison: [...]}`.
- `GET /api/ml/top-performers?position=&limit=` and `GET /api/ml/best-value?position=&limit=`
  return `predictions[]`. Each prediction has:
  - `player_name`, `team_name`, `position`, `predicted_points`
  - `fixture_context {opponent_team, home_away, match_date, matchday}`
  - availability fields: `status` (`a`/`d`/`i`/`s`/`u`/`n`), `chance_of_playing_next_round`
    (0–100 or null), `news`
  - best value also has `value` (£m) and `value_score`
- `GET /api/images/search?query=` → `{image_url}`; player photos are direct CDN or Wikimedia
  URLs.
- `GET /api/trivia/new?difficulty=` and `POST /api/trivia/answer`.

## 7. Running it

```bash
npm install
npm run dev:live   # frontend on http://localhost:3000, API proxied to the deployed backend
npm run build      # must pass before every commit
```

`dev:live` uses the live backend (chat answers cost real LLM calls; please keep testing
reasonable). No Python, database or keys are needed.

## 8. Deliverables and process

1. Work on a branch `redesign/ui-overhaul`, in small commits:
   - design tokens and theme
   - app shell
   - one commit per page
   - cleanup
2. Before coding pages, write a short `docs/design-system.md`:
   - palette with tokens (light + dark) and contrast ratios
   - type scale and fonts
   - spacing and radius
   - component inventory: button, input, select, card, table, chip/badge, tabs, skeleton,
     toast, page header
   - semantic colors for positions and availability
3. Open a pull request that includes:
   - before/after screenshots of every page in light, dark and mobile (360 px)
   - a summary of the UX changes
   - confirmation of the checklist below

## 9. Acceptance checklist

- [ ] `npm run build` passes; no new TypeScript errors or `any` in new code.
- [ ] No raw Tailwind palette classes (`purple-*`, `gray-*`, `slate-*`, …) or hex values in pages; colours come from tokens. No emoji in headings.
- [ ] All colors come from design tokens. No `isDark ?` class ternaries. No `html.dark` overrides of utility classes.
- [ ] All routes work (old `/qa` and `/search` redirect), plus navigation from search to profile to compare.
- [ ] Chat sends the same request body as before; retrieval details and the graph view still render.
- [ ] Predictions show availability (status, chance, news) and fixture context.
- [ ] Player search updates as you type, with stale-response protection.
- [ ] Settings still writes the same store keys and values.
- [ ] The dark theme meets WCAG AA, with keyboard focus visible (light theme: later).
- [ ] Works at 360, 768, 1024 and 1440 px widths with no horizontal scroll.
- [ ] Loading, empty and error states exist on every data view.
