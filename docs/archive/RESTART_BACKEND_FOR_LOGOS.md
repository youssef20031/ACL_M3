# How to See Club Logos in ML Predictions and Compare Players

## The Issue
Club logos are showing in Player Search but not in ML Predictions and Compare Players because the backend needs to be restarted to apply the Python code changes.

## Changes Made to Backend

1. **ML API Integration** (`ml/api_integration.py`):
   - Added `team_name` field to `PredictionResponse` model
   - Updated queries to fetch `team.name` for all predictions
   - Modified response construction to include team_name

2. **Player Comparison** (`graph/queries.py`):
   - Updated `compare_players` query to include `team_name`
   - Added `OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team)` 
   - Returns `t.name AS team_name` in results

## How to Restart the Backend

### Option 1: If running locally with Python
```bash
# Stop the current backend process (Ctrl+C)
# Then restart it:
python api_main.py
# or
python start_api_server.py
```

### Option 2: If deployed on Railway/Vercel
The backend will need to be redeployed to see the changes:

**Railway:**
1. Commit and push the changes to git
2. Railway will automatically redeploy
3. Wait for deployment to complete (check Railway dashboard)

**Manual Railway Deploy:**
```bash
git add .
git commit -m "Add team_name to ML predictions and player comparison"
git push
```

### Option 3: Hot Reload (if configured)
If your backend has hot reload enabled:
1. Just save the files (already done)
2. Wait a few seconds for auto-restart

## Verify Changes

Once the backend restarts, check these pages:

1. **ML Predictions** (`/predictions`):
   - Each player card should show their club logo
   - Top Performers section
   - Best Value section

2. **Compare Players** (`/compare`):
   - Player hero cards should show club logos
   - Club abbreviation appears next to player names

3. **Player Search** (`/search`):
   - Already working! ✓

## Debug Steps

If logos still don't appear after restart:

1. **Check Browser Console** (F12):
   ```javascript
   // Uncomment the debug line in Predictions.tsx:
   console.log('Player:', player.player_name, 'Team:', player.team_name);
   ```

2. **Check API Response**:
   - Open DevTools → Network tab
   - Refresh ML Predictions page
   - Look for `/api/ml/predict/top-performers` request
   - Check if `team_name` is in the response

3. **Verify Team Relationships** in Neo4j:
   ```cypher
   MATCH (p:Player)-[:PLAYS_FOR]->(t:Team)
   RETURN p.name, t.name
   LIMIT 10
   ```

## Expected API Response Format

After restart, the ML predictions API should return:
```json
{
  "predictions": [
    {
      "player_name": "Mohamed Salah",
      "predicted_points": 8.5,
      "team_name": "Liverpool",
      "fixture_context": { ... }
    }
  ]
}
```

And compare players should return:
```json
{
  "comparison": [
    {
      "player_name": "Mohamed Salah",
      "position": "MID",
      "team_name": "Liverpool",
      "total_points": 200,
      ...
    }
  ]
}
```

## Still Not Working?

If team_name is null/undefined after restart, it means:
1. Players don't have PLAYS_FOR relationships in Neo4j
2. Run the relationship creation script:
   ```bash
   python create_player_team_relationships.py
   ```

## Frontend Only Changes
The frontend has been updated and should work immediately once the backend returns `team_name`:
- ✓ `PlayerWithClub` component created
- ✓ Club mapping utility created
- ✓ Logos accessible at `/logos/*.png`
- ✓ All pages updated to use the component
