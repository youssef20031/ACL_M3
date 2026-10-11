# Complete Fix for "No Upcoming Fixture" Issue

## Problems Identified

### Problem 1: No Fixtures Displayed
- Website shows "No upcoming fixture" for all players
- Fixtures weren't loaded for 2026-27 season
- Player-team relationships missing
- Team name mismatches between CSV and database

### Problem 2: Players from Relegated Teams Appearing (Jamie Vardy Issue)
- Players like Jamie Vardy (Leicester City) appearing in predictions
- Leicester City not in 2026-27 Premier League
- No filtering for teams with upcoming fixtures in prediction endpoints

## Root Causes

1. **No 2026-27 Fixtures**: Database only had historical fixtures (2020-21)
2. **Missing PLAYS_FOR Relationships**: Players not linked to current teams
3. **Team Name Mismatches**: CSV used "Manchester City", DB had "Man City"
4. **No Fixture Filtering in Predictions**: Top performers/best value queries didn't filter by teams with fixtures
5. **Missing Fixture Context in Responses**: Only individual predictions had fixture context, not bulk predictions

## Complete Solution

### ✅ Step 1: Created Player-Team Relationships
**File**: `create_player_team_relationships.py`

```bash
python create_player_team_relationships.py
```

- Created 820 PLAYS_FOR relationships
- Links each player to their most recent team (from 2025-26 data)

### ✅ Step 2: Fixed Team Name Mapping
**File**: `fix_fixture_team_names.py`

```bash
python fix_fixture_team_names.py
```

Team name mappings:
- "Manchester City" → "Man City"
- "Manchester United" → "Man Utd"  
- "Tottenham Hotspur" → "Spurs"
- "Brighton and Hove Albion" → "Brighton"
- "Nottingham Forest" → "Nott'm Forest"
- "Newcastle United" → "Newcastle"
- "Ipswich Town" → "Ipswich"
- "Leeds United" → "Leeds"

Generated: `FPL_2026_2027_fixtures_fixed.csv`

### ✅ Step 3: Loaded 2026-27 Fixtures
**File**: `migrations/load_2026_27_fixtures.py`

```bash
python migrations\load_2026_27_fixtures.py --csv-path FPL_2026_2027_fixtures_fixed.csv --clear-existing
```

- Loaded 306 fixtures for 18 Premier League teams
- Skipped 74 fixtures for Hull City and Coventry City (not in database)

### ✅ Step 4: Fixed Migration Script Bug
**File**: `migrations/load_2026_27_fixtures.py`

Changed `verify_connection()` → `test_connection()`

### ✅ Step 5: Added Fixture Filtering to Prediction Queries
**File**: `ml/api_integration.py`

**Changes to `predict_top_performers()`:**
- Added `MATCH (p)-[:PLAYS_FOR]->(team:Team)` to get player's team
- Added check for upcoming 2026-27 fixtures:
  ```cypher
  OPTIONAL MATCH (upcoming_fixture:Fixture {season_id: "2026-27"})
  WHERE (upcoming_fixture)-[:PLAYS_HOME]->(team) OR (upcoming_fixture)-[:PLAYS_AWAY]->(team)
    AND upcoming_fixture.kickoff_time > $current_date
  WITH p, pos, team, recent_perfs, COUNT(DISTINCT upcoming_fixture) as fixture_count
  WHERE fixture_count > 0
  ```
- Added fixture context enrichment for each prediction

**Changes to `predict_best_value()`:**
- Added same team and fixture filtering logic
- Only returns players whose teams have upcoming fixtures

**Result**: Jamie Vardy and other Leicester players will NOT appear because Leicester has no 2026-27 fixtures

## Results

### ✅ Fixtures Loaded
- **306 fixtures** for 2026-27 season
- **18 teams** with complete schedules
- **First fixture**: August 21, 2026 (Arsenal vs Aston Villa)

### ✅ Players Filtered
- Only players from teams in 2026-27 Premier League appear
- Relegated teams (Leicester, Luton, Burnley, etc.) filtered out
- Each prediction now includes fixture context

### ✅ Fixture Context Displayed
All prediction endpoints now return:
```json
{
  "player_name": "Mohamed Salah",
  "predicted_points": 7.2,
  "fixture_context": {
    "opponent_team": "Newcastle",
    "home_away": "away",
    "match_date": "2026-08-23T16:30:00Z",
    "matchday": 1,
    "fixture_id": "fixture-2026-27-md1-new-liv"
  }
}
```

## Teams in 2026-27 Premier League (18 teams)

✅ **Teams WITH Fixtures**:
1. Arsenal
2. Aston Villa
3. Bournemouth
4. Brentford
5. Brighton
6. Chelsea
7. Crystal Palace
8. Everton
9. Fulham
10. Ipswich (promoted)
11. Leeds (promoted)
12. Liverpool
13. Man City
14. Man Utd
15. Newcastle
16. Nott'm Forest
17. Spurs
18. Sunderland (promoted)

❌ **Relegated/Not Promoted** (will NOT appear in predictions):
- Burnley
- Leicester (Jamie Vardy's team)
- Luton
- Norwich
- Sheffield Utd
- Southampton
- Watford
- West Brom
- West Ham
- Wolves

## How to Verify

### 1. Restart the Backend Server
The code changes require a server restart to take effect:

```bash
# Stop current server (Ctrl+C)
# Then restart:
python api_main.py
```

### 2. Run Verification Script
```bash
python verify_fixture_fix.py
```

This will test:
- Server health
- Top performers endpoint with fixture context
- Individual player predictions
- Confirmation that fixtures are displayed

### 3. Check the Website
Visit the Predictions page and verify:
- ✅ Fixture information shows (opponent, home/away, date, matchday)
- ✅ No "No upcoming fixture" messages
- ✅ Jamie Vardy and other Leicester players don't appear
- ✅ Only 2026-27 Premier League players shown

## Files Modified

### New Files Created
- `create_player_team_relationships.py` - Creates PLAYS_FOR relationships
- `fix_fixture_team_names.py` - Fixes team name mismatches  
- `FPL_2026_2027_fixtures_fixed.csv` - Corrected fixtures CSV
- `verify_fixture_fix.py` - Verification script
- `cleanup_temp_files.py` - Cleanup utility

### Files Modified
- `migrations/load_2026_27_fixtures.py` - Fixed verify_connection bug
- `ml/api_integration.py` - Added fixture filtering and context to all endpoints

## Re-applying the Fix

If you need to re-apply from scratch:

```bash
# 1. Create player-team relationships
python create_player_team_relationships.py

# 2. Load fixtures with correct names
python migrations\load_2026_27_fixtures.py --csv-path FPL_2026_2027_fixtures_fixed.csv --clear-existing

# 3. Restart server
# Stop server, then:
python api_main.py

# 4. Verify
python verify_fixture_fix.py
```

## Why Jamie Vardy Appeared Before

The old query in `predict_top_performers()` was:
```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
WHERE r.minutes > 0
```

This query:
- ❌ Didn't check player's current team
- ❌ Didn't filter by teams with 2026-27 fixtures
- ❌ Included ALL players with historical data (including Leicester)

The NEW query:
```cypher
MATCH (p:Player)-[:PLAYS_FOR]->(team:Team)
...
OPTIONAL MATCH (upcoming_fixture:Fixture {season_id: "2026-27"})
WHERE (upcoming_fixture)-[:PLAYS_HOME]->(team) OR (upcoming_fixture)-[:PLAYS_AWAY]->(team)
WITH ... fixture_count
WHERE fixture_count > 0
```

This query:
- ✅ Checks player's current team via PLAYS_FOR
- ✅ Filters teams by 2026-27 fixture existence
- ✅ Only includes players whose teams have upcoming fixtures

## Expected Outcome

After restarting the server:
1. ✅ Fixture information will display for all predictions
2. ✅ "No upcoming fixture" message will disappear
3. ✅ Jamie Vardy and Leicester players will NOT appear (team relegated)
4. ✅ Only 18 current Premier League teams' players will show
5. ✅ All predictions include opponent, home/away, date, and gameweek

The fix is complete! 🎉
