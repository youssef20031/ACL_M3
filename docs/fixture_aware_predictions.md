# Fixture-Aware Match Predictions API

This document provides usage examples for the fixture-aware prediction API endpoints.

## Overview

The date-aware match predictions feature integrates Premier League fixture schedules with ML predictions, enabling:
- Predictions specific to a player's next upcoming match
- Fixture context including opponent, home/away status, and match date
- Automatic cache management based on fixture timing
- Dynamic prediction updates as fixtures change

## Table of Contents

- [Getting Player Predictions with Fixture Context](#getting-player-predictions-with-fixture-context)
- [Loading Fixture Data](#loading-fixture-data)
- [Querying Upcoming Fixtures](#querying-upcoming-fixtures)
- [Error Handling](#error-handling)
- [Caching Behavior](#caching-behavior)

---

## Getting Player Predictions with Fixture Context

### Endpoint

```
POST /api/ml/predict/player
```

### Request

```json
{
  "player_name": "Mohamed Salah",
  "player_data": {}
}
```

The `player_data` field is optional. If empty, the system fetches player statistics from Neo4j automatically.

### Response

```json
{
  "player_name": "Mohamed Salah",
  "predicted_points": 8.5,
  "confidence_interval": null,
  "features_used": {
    "minutes": 90.0,
    "goals_scored": 1.2,
    "assists": 0.8,
    "form": 7.5
  },
  "model_version": "xgboost_fwd_v3",
  "fixture_context": {
    "opponent_team": "Manchester United",
    "home_away": "away",
    "match_date": "2026-08-22T15:00:00+00:00",
    "matchday": 2,
    "fixture_id": "fixture-2026-27-md2-liv-mun"
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `player_name` | string | Player's full name |
| `predicted_points` | float | Predicted FPL points for next gameweek |
| `confidence_interval` | array | Optional confidence range [lower, upper] |
| `features_used` | object | Player statistics used in prediction |
| `model_version` | string | ML model version identifier |
| `fixture_context` | object \| null | Next match details (null if no upcoming fixtures) |

### Fixture Context Fields

| Field | Type | Description |
|-------|------|-------------|
| `opponent_team` | string | Name of opposing team |
| `home_away` | string | "home" or "away" |
| `match_date` | string | ISO 8601 datetime in UTC |
| `matchday` | integer | Premier League matchday number (1-38) |
| `fixture_id` | string | Unique fixture identifier |

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/ml/predict/player" \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Mohamed Salah"
  }'
```

### Python Example

```python
import requests

url = "http://localhost:8000/api/ml/predict/player"
payload = {
    "player_name": "Mohamed Salah"
}

response = requests.post(url, json=payload)
prediction = response.json()

print(f"Player: {prediction['player_name']}")
print(f"Predicted Points: {prediction['predicted_points']:.1f}")

if prediction['fixture_context']:
    fixture = prediction['fixture_context']
    print(f"Next Match: vs {fixture['opponent_team']} ({fixture['home_away']})")
    print(f"Date: {fixture['match_date']}")
    print(f"Matchday: {fixture['matchday']}")
else:
    print("No upcoming fixture")
```

### JavaScript/TypeScript Example

```typescript
const getPrediction = async (playerName: string) => {
  const response = await fetch('http://localhost:8000/api/ml/predict/player', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      player_name: playerName,
    }),
  });

  const prediction = await response.json();
  
  console.log(`Player: ${prediction.player_name}`);
  console.log(`Predicted Points: ${prediction.predicted_points.toFixed(1)}`);
  
  if (prediction.fixture_context) {
    const fixture = prediction.fixture_context;
    console.log(`Next Match: vs ${fixture.opponent_team} (${fixture.home_away})`);
    console.log(`Date: ${new Date(fixture.match_date).toLocaleDateString()}`);
  }
};

getPrediction('Mohamed Salah');
```

---

## Loading Fixture Data

### Endpoint

```
POST /api/fixtures/load
```

### Request

```json
{
  "csv_path": "/path/to/FPL_2026_2027_fixtures.csv",
  "season_id": "2026-27",
  "clear_existing": false
}
```

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `csv_path` | string | Yes | Path to CSV file with fixture data |
| `season_id` | string | No | Season identifier (default: "2026-27") |
| `clear_existing` | boolean | No | Clear existing fixtures before loading (default: false) |

### CSV Format

The CSV file must include these columns:

| Column | Required | Description |
|--------|----------|-------------|
| Matchday | Yes | Matchday number (1-38) |
| Home Team | Yes | Home team name (must match database) |
| Away Team | Yes | Away team name (must match database) |
| Date | Yes | Match date (DD/MM/YYYY or YYYY-MM-DD) |
| Time | No | Match time in HH:MM format (default: 15:00) |

### Example CSV

```csv
Matchday,Home Team,Away Team,Date,Time
1,Liverpool,Arsenal,15/08/2026,15:00
1,Manchester City,Chelsea,15/08/2026,17:30
2,Arsenal,Manchester United,22/08/2026,15:00
```

### Response

```json
{
  "success": true,
  "fixtures_loaded": 380,
  "message": "Successfully loaded 380 fixtures for season 2026-27"
}
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/fixtures/load" \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "FPL_2026_2027_fixtures.csv",
    "season_id": "2026-27",
    "clear_existing": false
  }'
```

### Python Example

```python
import requests

url = "http://localhost:8000/api/fixtures/load"
payload = {
    "csv_path": "FPL_2026_2027_fixtures.csv",
    "season_id": "2026-27",
    "clear_existing": False
}

response = requests.post(url, json=payload)
result = response.json()

if result['success']:
    print(f"✅ Loaded {result['fixtures_loaded']} fixtures")
else:
    print(f"❌ Failed to load fixtures")
```

---

## Querying Upcoming Fixtures

### Endpoint

```
GET /api/fixtures/next?team={team_name}
```

### Query Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `team` | No | Team name to filter (omit for all teams) |

### Response

```json
{
  "fixtures": [
    {
      "home_team": "Liverpool",
      "away_team": "Arsenal",
      "match_date": "2026-08-15T15:00:00+00:00",
      "matchday": 1
    },
    {
      "home_team": "Manchester City",
      "away_team": "Liverpool",
      "match_date": "2026-08-22T17:30:00+00:00",
      "matchday": 2
    }
  ]
}
```

### cURL Example - Specific Team

```bash
curl "http://localhost:8000/api/fixtures/next?team=Liverpool"
```

### cURL Example - All Teams

```bash
curl "http://localhost:8000/api/fixtures/next"
```

### Python Example

```python
import requests

# Get next fixture for Liverpool
url = "http://localhost:8000/api/fixtures/next"
params = {"team": "Liverpool"}

response = requests.get(url, params=params)
data = response.json()

for fixture in data['fixtures']:
    print(f"Matchday {fixture['matchday']}: {fixture['home_team']} vs {fixture['away_team']}")
    print(f"  Date: {fixture['match_date']}")
```

---

## Error Handling

### Common Error Responses

#### 404 - Player Not Found

```json
{
  "detail": "Player 'Unknown Player' not found or no recent data available"
}
```

#### 400 - Invalid CSV Data

```json
{
  "detail": "Missing required columns: {'Date', 'Away Team'}"
}
```

#### 503 - Service Unavailable

```json
{
  "detail": "ML predictor not loaded. Train a model first."
}
```

### Error Handling Example

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/api/ml/predict/player",
        json={"player_name": "Mohamed Salah"}
    )
    response.raise_for_status()
    
    prediction = response.json()
    print(f"Predicted points: {prediction['predicted_points']}")
    
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print("Player not found")
    elif e.response.status_code == 503:
        print("Prediction service unavailable")
    else:
        print(f"Error: {e.response.json()['detail']}")
```

---

## Caching Behavior

### Cache TTL Strategy

Predictions are cached with dynamic TTL based on fixture timing:

| Time Until Fixture | Cache TTL |
|-------------------|-----------|
| > 7 days | 24 hours |
| 1-7 days | 6 hours |
| < 1 day | 1 hour |
| Past kickoff | No caching |

### Cache Invalidation

Caches are automatically invalidated when:
- A fixture is updated (date change, status change)
- A fixture is deleted

### Cache Key Format

```
prediction:{player_name}:{fixture_id}:{model_version}
```

Example:
```
prediction:Mohamed Salah:fixture-2026-27-md2-liv-mun:xgboost_fwd_v3
```

---

## Best Practices

1. **Fixture Data Loading**
   - Load fixtures before requesting predictions
   - Use `clear_existing=true` when refreshing full season data
   - Validate CSV format before uploading

2. **Prediction Requests**
   - Handle null `fixture_context` gracefully (no upcoming fixtures)
   - Cache predictions on the client side when appropriate
   - Display fixture details to users for context

3. **Error Handling**
   - Always check response status codes
   - Provide user-friendly error messages
   - Fall back to generic predictions when fixture lookup fails

4. **Performance**
   - Leverage server-side caching (automatic)
   - Batch prediction requests when possible
   - Filter fixture queries by team to reduce response size

---

## Additional Resources

- [OpenAPI/Swagger Documentation](http://localhost:8000/docs)
- [Requirements Document](../.kiro/specs/date-aware-match-predictions/requirements.md)
- [Design Document](../.kiro/specs/date-aware-match-predictions/design.md)
- [Neo4j Schema Migration](../migrations/add_fixture_schema.cypher)

---

## Support

For questions or issues, please refer to the project repository or contact the development team.
