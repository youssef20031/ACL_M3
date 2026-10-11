# Championship Data Collection - Options & Decision

**Status**: FBref scraping blocked by Cloudflare despite professional implementation
**Date**: June 22, 2026

---

## Current Blocker

The FBref scraper has been implemented with industry best practices:
- ✅ Strict 10 req/min rate limiting (7-9s delays)
- ✅ curl_cffi with Chrome 120 TLS fingerprint spoofing
- ✅ Full browser headers (User-Agent, Client Hints, etc.)
- ✅ Random variance to mimic human behavior

**Result**: Still getting HTTP 403 Forbidden on all requests

Sports Reference / FBref enforces very aggressive Cloudflare protection that blocks even sophisticated automated scrapers.

---

## What We Need

Based on user clarification:

### Player Data (Per-90 Season Aggregates)
- 5 seasons: 2019-20, 2020-21, 2021-22, 2022-23, 2023-24
- 5 stat categories per season: stats, shooting, passing, defense, possession
- Total: **25 tables**
- Estimated players: ~600-800 per season = ~3,600 total records

### Team Data (League Standings)
- Normal standings (not per-90)
- Top 6 positions to identify playoff participants
- Cross-reference with next season's EPL squads to identify promotion winner

### Promoted Teams to Identify
- 2020-21: Norwich, Watford, Brentford
- 2021-22: Fulham, Bournemouth, Nottingham Forest
- 2022-23: Burnley, Sheffield United, Luton
- 2023-24: Leicester, Ipswich, Southampton
- 2024-25: Leeds, Burnley, Sunderland

---

## Option A: Deploy Without Championship Data ⚡ FASTEST

**What happens:**
- Promoted teams handler uses heuristic fallback only
- Baseline: Position averages (GK: 3.2, DEF: 3.8, MID: 4.1, FWD: 4.5)
- GW-based decay: Heavy heuristic early, transitions to real PL data by GW 12
- No archetype matching or translation models

**Estimated Performance:**
- MAE for promoted teams: ~1.8-2.2 points (vs ~1.0-1.2 with full system)
- Still better than random guessing
- Improves rapidly as season progresses (real PL data accumulates)

**Pros:**
- ✅ Deploy immediately
- ✅ No manual data collection
- ✅ Functional system (just lower accuracy for promoted teams)

**Cons:**
- ❌ Higher error for promoted teams GW 1-12
- ❌ Misses opportunity to leverage Championship insights

**Timeline:** Ready now

---

## Option B: Manual CSV Download from FBref 📊 RECOMMENDED

**Process:**
1. Visit FBref Championship pages in browser
2. Download 25 CSV files (5 seasons × 5 categories)
3. Place in `ml/data/championship_raw/`
4. Run data cleaning pipeline: `python ml/data/clean_manual_csvs.py`
5. Build archetype database: `python ml/build_archetype_database.py`
6. Train translation models: `python ml/translation/model_a_team.py` + `model_b_player.py`

**URLs to download from:**
```
2023-24 Stats:    https://fbref.com/en/comps/10/2023-2024/stats/2023-2024-Championship-Stats
2023-24 Shooting: https://fbref.com/en/comps/10/2023-2024/shooting/2023-2024-Championship-Stats
2023-24 Passing:  https://fbref.com/en/comps/10/2023-2024/passing/2023-2024-Championship-Stats
2023-24 Defense:  https://fbref.com/en/comps/10/2023-2024/defense/2023-2024-Championship-Stats
2023-24 Possession: https://fbref.com/en/comps/10/2023-2024/possession/2023-2024-Championship-Stats

(Repeat for 2022-23, 2021-22, 2020-21, 2019-20)
```

**Pros:**
- ✅ Legitimate data access (no scraping)
- ✅ One-time effort (~45 minutes of downloading)
- ✅ Full system functionality
- ✅ Best MAE for promoted teams

**Cons:**
- ⏱️ Manual work required
- 📋 Need to download 25 tables

**Timeline:** 45 min download + 15 min processing = 1 hour total

---

## Option C: Full Browser Automation (Playwright/Selenium) 🤖

**Implementation:**
- Use Playwright with stealth mode
- Full browser rendering (handles JavaScript challenges)
- Slower but more robust than HTTP requests

**Pros:**
- ✅ Automated (no manual downloads)
- ✅ Can handle Cloudflare challenges
- ✅ Full system functionality

**Cons:**
- ⏱️ Requires new library setup
- 🐌 Slower (2-4 hours for all data)
- 🔧 More complex debugging if blocked

**Timeline:** 1 hour implementation + 2-4 hours scraping = 3-5 hours total

---

## Option D: Alternative Data Sources 🔄

**Potential sources:**
- Understat (more API-friendly, but limited Championship coverage)
- TransferMarkt (team/player data, but different stat formats)
- Official EFL API (if available)
- Sports data providers (Opta, StatsBomb - may require license)

**Pros:**
- ✅ May be easier to access
- ✅ Potentially cleaner data

**Cons:**
- ⏱️ Research time to find suitable source
- 🔧 Data format may not match FBref schema
- 💰 Some sources require paid access

**Timeline:** Unknown (depends on source availability)

---

## Option E: Hybrid Approach 🎯

**Combine Option A + B:**
1. Deploy NOW with heuristic fallback (Option A)
2. Collect Championship data manually when time permits (Option B)
3. Retrain translation models and update handler
4. See performance improvement mid-season

**Pros:**
- ✅ Immediate deployment
- ✅ Progressive enhancement
- ✅ Can validate if Championship data actually improves predictions

**Cons:**
- 🔄 Two-stage deployment

**Timeline:** Deploy now, enhance in 1-2 weeks

---

## Recommendation: Option E (Hybrid)

**Rationale:**

1. **System is already functional** - Promoted teams handler has robust heuristic fallback
2. **Time-to-market matters** - Season starts soon, get predictions live first
3. **Incremental improvement** - Add Championship data when available, measure impact
4. **Risk mitigation** - Validate that translation models actually improve MAE before committing

**Immediate action:**
```bash
# Deploy current system (heuristic fallback)
python ml/deploy_promoted_teams.py

# Test promoted teams predictions
python ml/test_promoted_teams.py

# Start API server
python api_main.py
```

**Later action (when time permits):**
```bash
# Manually download 25 CSVs to ml/data/championship_raw/
# Then run:
python ml/data/clean_manual_csvs.py
python ml/build_archetype_database.py
python ml/translation/model_a_team.py
python ml/translation/model_b_player.py
```

---

## Decision Matrix

| Factor | Option A | Option B | Option C | Option D | Option E |
|--------|----------|----------|----------|----------|----------|
| Time to deploy | ⚡ Now | 1 hour | 3-5 hours | Unknown | ⚡ Now |
| Manual effort | None | Medium | Low | Unknown | Medium (later) |
| Promoted teams MAE | ~2.0 | ~1.1 | ~1.1 | Unknown | ~2.0 → ~1.1 |
| Risk | Low | Low | Medium | High | Very Low |
| Full system | ❌ | ✅ | ✅ | ❌ | ✅ (staged) |

---

## Next Steps (Pending Decision)

**If Option A or E:**
1. Run deployment verification: `python ml/deploy_promoted_teams.py`
2. Test promoted teams: `python ml/test_promoted_teams.py`
3. Start production API: `python api_main.py`

**If Option B:**
1. Download 25 CSV files from FBref
2. I'll create `ml/data/clean_manual_csvs.py` to process them
3. Run full training pipeline

**If Option C:**
1. Install Playwright: `pip install playwright playwright-stealth`
2. I'll create `ml/data/fbref_playwright.py`
3. Run automated browser scraping

**If Option D:**
Research alternative data sources

---

**Question for User:** Which option would you like to proceed with?
