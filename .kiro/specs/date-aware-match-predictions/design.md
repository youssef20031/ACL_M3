# Design Document: Date-Aware Match Predictions

## Overview

This design document describes the implementation of date-aware match predictions for the FPL FantasyTrivia application. The feature extends the existing ML prediction system to incorporate specific fixture schedules, enabling predictions that are contextualized with opponent information, home/away status, and match timing.

### Current System State

**Existing Components:**
- **ML Predictor** (`ml/predictor.py`): XGBoost position-specific models (GK/DEF/MID/FWD) trained on historical player performance
- **Feature Engineering** (`ml/feature_engineering.py`): Generates 50+ features including form, minutes, opponent strength, DGW detection
- **Neo4j Graph Database**: Stores Player, Team, Position, Season, Gameweek, Fixture nodes with relationships
- **FastAPI Backend** (`api_main.py`): Exposes prediction endpoints for player forecasts
- **Graph Queries** (`graph/queries.py`): Cypher query library for retrieving player/team/fixture data

**Current Prediction Flow:**
1. Frontend requests prediction for player
2. Backend retrieves player historical stats from Neo4j
3. Feature engineering creates input vector
4. XGBoost model predicts next gameweek points
5. Response includes predicted_points but NO opponent or fixture context

### Design Goal

**Transform predictions from generic gameweek forecasts to fixture-specific predictions** by:
1. Adding **Fixture Service** to determine each player's next match based on current date
2. Enhing **Neo4j Schema** with 2026-27 season fixture data
3. Extending **ML Predictor** to accept fixture context as input features
4. Updating **API responses** to include opponent, match date, home/away status
5. Implementing **caching strategy** keyed by (player_id, fixture_id) for performance


---

## Architecture

### System Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
│  Request: GET /api/predict/player?name=Salah&season=2026-27    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (api_main.py)                │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  MLAPIIntegration                                        │  │
│  │  - predict_player_next_gameweek()                       │  │
│  │  - Orchestrates FixtureService + MLPredictor            │  │
│  └────────┬────────────────────────────────┬────────────────┘  │
│           │                                │                   │
│           ▼                                ▼                   │
│  ┌────────────────────┐          ┌────────────────────────┐   │
│  │  FixtureService    │          │  FPLPredictor          │   │
│  │  NEW COMPONENT     │          │  ENHANCED              │   │
│  │                    │          │                        │   │
│  │  - get_next_match()│          │  - predict_next_gw()   │   │
│  │  - is_fixture_past()│         │  - Uses fixture ctx    │   │
│  │  - load_fixtures() │          │  - XGBoost models      │   │
│  └────────┬───────────┘          └────────┬───────────────┘   │
│           │                                │                   │
└───────────┼────────────────────────────────┼───────────────────┘
            │                                │
            ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Neo4j Graph Database                       │
│                                                                 │
│  (:Player)-[:PLAYS_FOR]->(:Team)<-[:PLAYS_HOME|AWAY]-(:Fixture)│
│                                                                 │
│  Query: Get next fixture for player after current date         │
│  Returns: opponent, home/away, match_date, matchday            │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Sequence

**Fixture-Aware Prediction Request:**


```
1. API receives prediction request for player "Mohamed Salah"
   └─> MLAPIIntegration.predict_player_next_gameweek(player_name="Mohamed Salah")

2. Determine player's next fixture
   └─> FixtureService.get_next_match(player_name="Mohamed Salah", current_date=datetime.now())
   └─> Cypher Query: MATCH (p:Player {name: $name})-[:PLAYS_FOR]->(t:Team)
                      MATCH (f:Fixture)-[:PLAYS_HOME|PLAYS_AWAY]->(t)
                      WHERE f.kickoff_time > $current_date
                      RETURN f, opponent ORDER BY f.kickoff_time LIMIT 1
   └─> Returns: {opponent: "Manchester United", home_away: "away", match_date: "2026-08-22", matchday: 2}

3. Retrieve player historical stats + fixture context
   └─> Neo4j query for last 5 gameweeks of player stats
   └─> Enriched with opponent_team="Manchester United", was_home=False

4. Feature engineering
   └─> FeatureEngineer.engineer_features(player_data)
   └─> Generates 50+ features including:
       - form (4-game rolling avg)
       - opp_def_strength (rolling 5-game goals conceded by Man Utd)
       - home_form, away_form
       - minutes_rolling5
   └─> Returns: numpy array [1, 50]

5. ML Prediction
   └─> FPLPredictor.predict_next_gameweek(features)
   └─> XGBoost FWD model predicts: 7.2 points

6. API Response
   └─> {
         "player_name": "Mohamed Salah",
         "predicted_points": 7.2,
         "fixture_context": {
           "opponent_team": "Manchester United",
           "home_away": "away",
           "match_date": "2026-08-22T15:00:00Z",
           "matchday": 2
         },
         "model_version": "xgboost_fwd_v3"
       }
```


---

## Components and Interfaces

### 1. FixtureService (NEW)

**File:** `services/fixture_service.py`

**Responsibility:** Manage fixture data and determine next matches based on current date.

**Class Definition:**

```python
from typing import Optional, Dict, Any, List
from datetime import datetime
from graph.connection import Neo4jConnection


class FixtureService:
    """Service for managing fixture data and determining next matches."""
    
    def __init__(self, neo4j_conn: Neo4jConnection):
        self.conn = neo4j_conn
    
    def get_next_match(
        self, 
        player_name: str, 
        current_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the next upcoming fixture for a player.
        
        Args:
            player_name: Player's full name
            current_date: Reference date (defaults to datetime.now(UTC))
        
        Returns:
            Dictionary with fixture details or None if no upcoming matches
            {
                "opponent_team": str,
                "home_away": "home" | "away",
                "match_date": datetime,
                "matchday": int,
                "fixture_id": str
            }
        """

    
    def get_team_next_match(
        self,
        team_name: str,
        current_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the next fixture for a specific team."""
    
    def is_fixture_past(
        self,
        fixture_id: str,
        current_date: Optional[datetime] = None
    ) -> bool:
        """Check if a fixture has already occurred."""
    
    def load_fixtures_from_csv(
        self,
        csv_path: str,
        season_id: str = "2026-27"
    ) -> int:
        """
        Load fixture data from CSV into Neo4j.
        
        Returns:
            Number of fixtures loaded
        """
    
    def update_fixture(
        self,
        fixture_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an existing fixture's details."""
```

**Key Cypher Queries:**

```cypher
# Get next match for player
MATCH (p:Player {name: $player_name})-[:PLAYS_FOR]->(t:Team)
MATCH (f:Fixture)
WHERE (f)-[:PLAYS_HOME]->(t) OR (f)-[:PLAYS_AWAY]->(t)
  AND f.kickoff_time > $current_date
WITH f, t,
  CASE 
    WHEN (f)-[:PLAYS_HOME]->(t) THEN 'home'
    ELSE 'away'
  END AS home_away,
  CASE
    WHEN (f)-[:PLAYS_HOME]->(t) 
    THEN [(f)-[:PLAYS_AWAY]->(opp) | opp.name][0]
    ELSE [(f)-[:PLAYS_HOME]->(opp) | opp.name][0]
  END AS opponent
ORDER BY f.kickoff_time ASC
LIMIT 1
RETURN f.id AS fixture_id,
       f.kickoff_time AS match_date,
       f.matchday AS matchday,
       opponent AS opponent_team,
       home_away
```


### 2. MLAPIIntegration (ENHANCED)

**File:** `ml/api_integration.py` (existing file)

**Changes Required:**

```python
# BEFORE (existing)
async def predict_player_next_gameweek(self, request: PlayerPredictionRequest):
    # Get player historical stats
    player_data = self._get_player_stats(request.player_name)
    # Predict without fixture context
    prediction = self.predictor.predict_next_gameweek(player_data)
    return prediction

# AFTER (enhanced)
async def predict_player_next_gameweek(self, request: PlayerPredictionRequest):
    # NEW: Get next fixture for player
    fixture_context = self.fixture_service.get_next_match(
        request.player_name,
        current_date=datetime.now(timezone.utc)
    )
    
    # Get player historical stats + fixture context
    player_data = self._get_player_stats_with_fixture(
        request.player_name,
        fixture_context
    )
    
    # Predict with fixture-enriched features
    prediction = self.predictor.predict_next_gameweek(player_data)
    
    # NEW: Include fixture context in response
    prediction.fixture_context = fixture_context
    return prediction
```

**New Constructor Dependency:**

```python
class MLAPIIntegration:
    def __init__(
        self,
        neo4j_conn: Neo4jConnection,
        query_executor: Optional[QueryExecutor] = None,
        fixture_service: Optional[FixtureService] = None  # NEW
    ):
        self.fixture_service = fixture_service or FixtureService(neo4j_conn)
```


### 3. FPLPredictor (MINIMAL CHANGES)

**File:** `ml/predictor.py`

**No structural changes needed.** The predictor already accepts `player_data` dictionary which will now include opponent context. Feature engineering handles the transformation.

**Existing Interface (unchanged):**

```python
def predict_next_gameweek(
    self, 
    player_data: Dict[str, Any]
) -> PlayerPrediction:
    """
    Predict points for a single player's next gameweek.
    
    Args:
        player_data: Dictionary with player's recent stats and fixture info
                     NOW INCLUDES: opponent_team, was_home from FixtureService
    """
```

The `player_data` dictionary will be enriched with fixture context before being passed to the predictor. The feature engineering pipeline (`FeatureEngineer`) already supports opponent team features (see `opp_def_strength`, `opp_off_strength` in feature_engineering.py).

### 4. API Endpoints (ENHANCED)

**File:** `api_main.py`

**Endpoint:** `GET /api/predict/player`

**Request Parameters:**
- `name` (required): Player name (e.g., "Mohamed Salah")
- `season` (optional): Season ID (default: "2026-27")

**Response Schema (ENHANCED):**

```python
class PlayerPredictionResponse(BaseModel):
    player_name: str
    predicted_points: float
    confidence_interval: Optional[Tuple[float, float]] = None
    model_version: str
    
    # NEW: Fixture context
    fixture_context: Optional[Dict[str, Any]] = None
    # {
    #   "opponent_team": "Manchester United",
    #   "home_away": "away",
    #   "match_date": "2026-08-22T15:00:00Z",
    #   "matchday": 2,
    #   "fixture_id": "fixture-2026-27-md2-liv-mun"
    # }
```


**New Endpoint:** `POST /api/fixtures/load`

```python
@app.post("/api/fixtures/load")
async def load_fixtures(request: DataLoadRequest):
    """
    Load fixture data from CSV file.
    
    Request Body:
    {
        "csv_path": "data/FPL_2026_2027_fixtures.csv",
        "season_id": "2026-27",
        "clear_existing": true
    }
    
    Response:
    {
        "success": true,
        "fixtures_loaded": 380,
        "message": "Loaded 380 fixtures for season 2026-27"
    }
    """
```

**New Endpoint:** `GET /api/fixtures/next`

```python
@app.get("/api/fixtures/next")
async def get_next_fixtures(team: Optional[str] = None):
    """
    Get upcoming fixtures for all teams or a specific team.
    
    Query Parameters:
    - team (optional): Team name
    
    Response:
    {
        "fixtures": [
            {
                "home_team": "Liverpool",
                "away_team": "Manchester United",
                "match_date": "2026-08-22T15:00:00Z",
                "matchday": 2
            },
            ...
        ]
    }
    """
```


---

## Data Models

### Neo4j Graph Schema Enhancements

**New Node Type: Fixture**

```cypher
CREATE CONSTRAINT fixture_id IF NOT EXISTS 
FOR (f:Fixture) REQUIRE f.id IS UNIQUE;

CREATE INDEX fixture_kickoff IF NOT EXISTS 
FOR (f:Fixture) ON (f.kickoff_time);

CREATE INDEX fixture_matchday IF NOT EXISTS
FOR (f:Fixture) ON (f.matchday);
```

**Fixture Node Properties:**

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `id` | String | Unique fixture identifier | "fixture-2026-27-md1-ars-wol" |
| `kickoff_time` | DateTime | Match start time (UTC) | "2026-08-15T12:30:00Z" |
| `matchday` | Integer | Round number (1-38) | 1 |
| `season_id` | String | Season identifier | "2026-27" |
| `home_score` | Integer | Home team goals (null before match) | null → 2 |
| `away_score` | Integer | Away team goals (null before match) | null → 1 |
| `status` | String | Match status | "scheduled", "live", "finished" |

**New Relationships:**

```cypher
# Fixture → Home Team
(:Fixture)-[:PLAYS_HOME]->(:Team)

# Fixture → Away Team
(:Fixture)-[:PLAYS_AWAY]->(:Team)

# Fixture → Matchday → Season
(:Fixture)-[:IN_MATCHDAY]->(:Matchday {number: 1})-[:IN_SEASON]->(:Season {id: "2026-27"})
```


**Graph Schema Diagram:**

```
┌─────────────┐
│   Player    │
│  name: str  │
└──────┬──────┘
       │ :PLAYS_FOR
       ▼
┌─────────────┐       ┌──────────────┐
│    Team     │◄──────│   Fixture    │
│  name: str  │ :HOME │  id: str     │
└─────────────┘       │  kickoff: dt │
       ▲              │  matchday: int│
       │              │  status: str │
       └──────────────┤              │
            :AWAY     └──────┬───────┘
                             │ :IN_MATCHDAY
                             ▼
                      ┌──────────────┐
                      │  Matchday    │
                      │  number: int │
                      └──────┬───────┘
                             │ :IN_SEASON
                             ▼
                      ┌──────────────┐
                      │   Season     │
                      │  id: "26-27" │
                      └──────────────┘
```

### Python Data Classes

**File:** `services/fixture_models.py` (new file)

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class FixtureContext:
    """Fixture context for ML predictions."""
    opponent_team: str
    home_away: str  # "home" or "away"
    match_date: datetime
    matchday: int
    fixture_id: str


@dataclass
class FixtureData:
    """Raw fixture data from CSV or API."""
    id: str
    home_team: str
    away_team: str
    kickoff_time: datetime
    matchday: int
    season_id: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    status: str = "scheduled"
```


### CSV Format for Fixture Data

**File:** `FPL_2026_2027.csv` (provided by user)

**Expected Columns:**
- `Matchday` (int): 1-38
- `Home Team` (str): Team name matching Neo4j Team.name
- `Away Team` (str): Team name matching Neo4j Team.name
- `Date` (str): Match date in format "DD/MM/YYYY" or "YYYY-MM-DD"
- `Time` (str): Kickoff time in format "HH:MM" (optional, defaults to "15:00")

**Example:**

```csv
Matchday,Home Team,Away Team,Date,Time
1,Arsenal,Wolverhampton Wanderers,15/08/2026,15:00
1,Brighton and Hove Albion,Everton,15/08/2026,15:00
1,Chelsea,Manchester City,15/08/2026,17:30
2,Liverpool,Manchester United,22/08/2026,12:30
```

### API Response Models

**Enhanced PlayerPrediction:**

```python
@dataclass
class PlayerPrediction:
    player_name: str
    predicted_points: float
    confidence_interval: Optional[Tuple[float, float]] = None
    features_used: Optional[Dict[str, Any]] = None
    model_version: str = "xgboost_fwd_v3"
    
    # NEW FIELD
    fixture_context: Optional[FixtureContext] = None
```


---

## Algorithms

### Next Match Determination Algorithm

**Function:** `FixtureService.get_next_match()`

**Input:**
- `player_name`: String
- `current_date`: DateTime (UTC)

**Output:**
- `FixtureContext` object or `None`

**Algorithm:**

```python
def get_next_match(player_name: str, current_date: datetime) -> Optional[FixtureContext]:
    """
    Determine the next upcoming fixture for a player.
    
    Logic:
    1. Get player's current team
    2. Find all fixtures for that team
    3. Filter fixtures where kickoff_time > current_date
    4. Sort by kickoff_time ascending
    5. Return first fixture (earliest upcoming match)
    6. If multiple fixtures on same day, return earliest by time
    """
    
    # Step 1: Get player's team
    team = neo4j.query(
        "MATCH (p:Player {name: $name})-[:PLAYS_FOR]->(t:Team) RETURN t.name",
        {"name": player_name}
    )
    
    if not team:
        return None
    
    # Step 2-5: Find next fixture
    fixture = neo4j.query(
        """
        MATCH (t:Team {name: $team_name})
        MATCH (f:Fixture)
        WHERE ((f)-[:PLAYS_HOME]->(t) OR (f)-[:PLAYS_AWAY]->(t))
          AND f.kickoff_time > $current_date
        WITH f, t,
          CASE WHEN (f)-[:PLAYS_HOME]->(t) THEN 'home' ELSE 'away' END AS venue,
          CASE 
            WHEN (f)-[:PLAYS_HOME]->(t) 
            THEN [(f)-[:PLAYS_AWAY]->(opp) | opp.name][0]
            ELSE [(f)-[:PLAYS_HOME]->(opp) | opp.name][0]
          END AS opponent
        ORDER BY f.kickoff_time ASC
        LIMIT 1
        RETURN f.id, f.kickoff_time, f.matchday, opponent, venue
        """,
        {"team_name": team, "current_date": current_date}
    )
    
    if not fixture:
        return None
    
    return FixtureContext(
        opponent_team=fixture["opponent"],
        home_away=fixture["venue"],
        match_date=fixture["f.kickoff_time"],
        matchday=fixture["f.matchday"],
        fixture_id=fixture["f.id"]
    )
```


**Edge Cases:**

1. **Player has no team:** Return `None`
2. **Team has no upcoming fixtures:** Return `None` (end of season)
3. **Multiple fixtures on same day (Double Gameweek):** Return earliest by kickoff time
4. **Fixture exactly at current_date:** Excluded (use strict `>` comparison)
5. **Timezone handling:** All dates stored and compared in UTC

**Time Complexity:** O(log n) due to Neo4j index on `kickoff_time`

### Fixture Data Loading Algorithm

**Function:** `FixtureService.load_fixtures_from_csv()`

**Algorithm:**

```python
def load_fixtures_from_csv(csv_path: str, season_id: str) -> int:
    """
    Load fixture data from CSV into Neo4j.
    
    Steps:
    1. Parse CSV file
    2. Validate team names against existing Team nodes
    3. Convert dates to UTC timestamps
    4. Create Fixture nodes with MERGE (idempotent)
    5. Create PLAYS_HOME and PLAYS_AWAY relationships
    6. Create IN_MATCHDAY relationships
    
    Returns: Number of fixtures created/updated
    """
    
    fixtures_loaded = 0
    
    # Step 1: Parse CSV
    df = pd.read_csv(csv_path)
    required_columns = {"Matchday", "Home Team", "Away Team", "Date"}
    
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_columns - set(df.columns)}")
    
    # Step 2: Get valid team names from Neo4j
    valid_teams = set(neo4j.query("MATCH (t:Team) RETURN t.name"))
    
    for _, row in df.iterrows():
        home_team = row["Home Team"]
        away_team = row["Away Team"]
        
        # Validate teams exist
        if home_team not in valid_teams or away_team not in valid_teams:
            logger.warning(f"Skipping fixture: Unknown teams {home_team} vs {away_team}")
            continue
        
        # Step 3: Convert date to UTC
        date_str = row["Date"]
        time_str = row.get("Time", "15:00")  # Default 3pm
        kickoff = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
        kickoff_utc = kickoff.replace(tzinfo=timezone.utc)
        
        # Step 4-6: Create fixture and relationships
        fixture_id = f"fixture-{season_id}-md{row['Matchday']}-{home_team[:3].lower()}-{away_team[:3].lower()}"
        
        neo4j.query(
            """
            MERGE (f:Fixture {id: $fixture_id})
            SET f.kickoff_time = $kickoff,
                f.matchday = $matchday,
                f.season_id = $season_id,
                f.status = 'scheduled'
            
            MERGE (ht:Team {name: $home_team})
            MERGE (at:Team {name: $away_team})
            MERGE (md:Matchday {number: $matchday, season_id: $season_id})
            MERGE (s:Season {id: $season_id})
            
            MERGE (f)-[:PLAYS_HOME]->(ht)
            MERGE (f)-[:PLAYS_AWAY]->(at)
            MERGE (f)-[:IN_MATCHDAY]->(md)
            MERGE (md)-[:IN_SEASON]->(s)
            """,
            {
                "fixture_id": fixture_id,
                "kickoff": kickoff_utc,
                "matchday": row["Matchday"],
                "season_id": season_id,
                "home_team": home_team,
                "away_team": away_team
            }
        )
        
        fixtures_loaded += 1
    
    return fixtures_loaded
```


### Prediction Caching Strategy

**Cache Key Structure:**

```python
cache_key = f"prediction:{player_id}:{fixture_id}:{model_version}"
# Example: "prediction:123:fixture-2026-27-md2-liv-mun:xgboost_fwd_v3"
```

**Cache TTL (Time-To-Live):**

```python
def calculate_cache_ttl(fixture_kickoff: datetime) -> int:
    """
    Calculate cache TTL in seconds.
    
    Logic:
    - If fixture > 7 days away: TTL = 24 hours
    - If fixture 1-7 days away: TTL = 6 hours
    - If fixture < 1 day away: TTL = 1 hour
    - After kickoff: TTL = 0 (no caching)
    """
    time_until_kickoff = fixture_kickoff - datetime.now(timezone.utc)
    days = time_until_kickoff.days
    
    if days > 7:
        return 86400  # 24 hours
    elif days >= 1:
        return 21600  # 6 hours
    elif time_until_kickoff.total_seconds() > 0:
        return 3600   # 1 hour
    else:
        return 0      # No caching for past fixtures
```

**Cache Invalidation Triggers:**

1. **Fixture date changes:** Invalidate all predictions for that fixture
2. **Player transfers:** Invalidate all predictions for that player
3. **Manual refresh:** Admin endpoint to clear specific caches
4. **Automatic expiry:** TTL-based expiration

**Implementation (Redis or In-Memory):**

```python
class PredictionCache:
    def __init__(self, redis_client: Optional[Redis] = None):
        self.redis = redis_client
        self.memory_cache: Dict[str, Tuple[Any, float]] = {}
    
    def get(self, key: str) -> Optional[PlayerPrediction]:
        if self.redis:
            data = self.redis.get(key)
            if data:
                return pickle.loads(data)
        else:
            if key in self.memory_cache:
                value, expires_at = self.memory_cache[key]
                if time.time() < expires_at:
                    return value
        return None
    
    def set(self, key: str, value: PlayerPrediction, ttl: int):
        if self.redis:
            self.redis.setex(key, ttl, pickle.dumps(value))
        else:
            self.memory_cache[key] = (value, time.time() + ttl)
```


---

## Error Handling

### Error Scenarios and Responses

| Scenario | HTTP Status | Error Message | Recovery Action |
|----------|-------------|---------------|-----------------|
| Player not found | 404 | "Player '{name}' not found in database" | Check spelling, list available players |
| No upcoming fixtures | 200 | Return prediction with `fixture_context: null` | Indicate end of season |
| Invalid fixture date | 400 | "Invalid date format: {date}" | Provide date format example |
| Team name mismatch in CSV | 400 | "Team '{name}' not found. Available teams: [...]" | Display valid team names |
| Neo4j connection failure | 503 | "Database temporarily unavailable" | Retry after 5 seconds |
| ML model not loaded | 503 | "Prediction model not available" | Check model file exists |
| Cache failure (Redis down) | 200 | (Fallback to no caching) | Log warning, proceed without cache |

**Graceful Degradation:**

```python
def predict_with_fallback(player_name: str) -> PlayerPrediction:
    """
    Attempt prediction with fixture context, fall back to generic prediction.
    """
    try:
        # Try fixture-aware prediction
        fixture_context = fixture_service.get_next_match(player_name)
        player_data = get_player_stats_with_fixture(player_name, fixture_context)
        prediction = predictor.predict_next_gameweek(player_data)
        prediction.fixture_context = fixture_context
        return prediction
    
    except FixtureServiceError as e:
        logger.warning(f"Fixture service failed, using generic prediction: {e}")
        # Fallback to generic prediction without fixture context
        player_data = get_player_stats(player_name)
        prediction = predictor.predict_next_gameweek(player_data)
        prediction.fixture_context = None
        return prediction
```


---

## Testing Strategy

### Unit Tests

**Test Coverage Areas:**

1. **FixtureService Tests** (`tests/test_fixture_service.py`)
   - `test_get_next_match_returns_earliest_fixture()`
   - `test_get_next_match_returns_none_when_no_fixtures()`
   - `test_get_next_match_handles_double_gameweek()`
   - `test_is_fixture_past_with_past_date()`
   - `test_is_fixture_past_with_future_date()`
   - `test_load_fixtures_from_csv_creates_nodes()`
   - `test_load_fixtures_validates_team_names()`

2. **Date/Time Logic Tests**
   - `test_kickoff_comparison_with_current_date()`
   - `test_timezone_normalization_to_utc()`
   - `test_fixture_exactly_at_current_time_excluded()`

3. **API Integration Tests**
   - `test_predict_player_includes_fixture_context()`
   - `test_predict_player_without_fixtures_returns_null_context()`
   - `test_load_fixtures_endpoint_returns_count()`

4. **Cache Tests**
   - `test_cache_hit_for_same_player_fixture()`
   - `test_cache_miss_after_ttl_expires()`
   - `test_cache_invalidation_on_fixture_update()`

### Integration Tests

1. **End-to-End Prediction Flow**
   ```python
   def test_e2e_fixture_aware_prediction():
       # Load fixtures
       response = client.post("/api/fixtures/load", json={
           "csv_path": "test_fixtures.csv",
           "season_id": "2026-27"
       })
       assert response.status_code == 200
       
       # Request prediction
       response = client.get("/api/predict/player", params={"name": "Mohamed Salah"})
       assert response.status_code == 200
       
       data = response.json()
       assert data["fixture_context"] is not None
       assert "opponent_team" in data["fixture_context"]
       assert "match_date" in data["fixture_context"]
   ```


---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After analyzing all acceptance criteria, several redundancies were identified:

- **2.4** (determine next match by comparing dates) is subsumed by **2.1** and **2.2**
- **3.1, 3.2, 3.3** (fixture context fields) can be combined into one comprehensive property
- **4.1, 4.2** (API response fields) are redundant with **3.1-3.3**
- **4.3** (null fixture handling) is redundant with **3.5**
- **7.2** (UTC comparison) is redundant with **2.1**
- **7.4** (timezone normalization) is redundant with **5.3**
- **8.3** (features used) is redundant with **3.4**
- **9.3** (graph traversal) is redundant with **2.1**
- **10.2, 10.4** (cache invalidation) are redundant with **6.4**

The following properties provide unique validation value:


### Property 1: Fixture Storage Completeness

*For any* valid fixture data with all required fields (date, time, home_team, away_team, matchday), storing the fixture SHALL result in a Neo4j Fixture node with all fields present and retrievable.

**Validates: Requirements 1.1**

### Property 2: Team Fixture Query Ordering

*For any* team with multiple fixtures, querying fixtures for that team SHALL return results ordered by kickoff date ascending.

**Validates: Requirements 1.2**

### Property 3: Fixture Validation Rejection

*For any* fixture data missing one or more required fields (date, time, home_team, away_team, matchday), the System SHALL reject the fixture and return an error message.

**Validates: Requirements 1.4**

### Property 4: Duplicate Fixture Rejection

*For any* fixture that already exists in the database, attempting to insert the same fixture again SHALL be rejected with an error message.

**Validates: Requirements 1.5**

### Property 5: Next Match Before Kickoff

*For any* player with an upcoming fixture and any current_date before that fixture's kickoff_time, the Fixture_Service SHALL return that fixture as the Next_Match.

**Validates: Requirements 2.1**

### Property 6: Next Match Past Fixture Exclusion

*For any* player with multiple fixtures and a current_date after the first fixture's kickoff_time, the Fixture_Service SHALL exclude past fixtures and return the next subsequent fixture.

**Validates: Requirements 2.2**


### Property 7: Double Gameweek Earliest Selection

*For any* team with multiple fixtures on the same day at different times, the Fixture_Service SHALL return the earliest fixture (by time) that has not yet been completed.

**Validates: Requirements 2.5**

### Property 8: Prediction Response Fixture Context

*For any* player with a next match, requesting a prediction SHALL return a response containing fixture_context with opponent_team, home_away status, match_date, and matchday fields.

**Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.2**

### Property 9: Prediction Features Include Fixture Context

*For any* player prediction request, the feature vector passed to the ML model SHALL include opponent strength (opp_def_strength, opp_off_strength) and home/away status (was_home) features.

**Validates: Requirements 3.4, 8.3**

### Property 10: Date Parsing to UTC

*For any* valid date and time string in the fixture data, parsing SHALL produce a datetime object with UTC timezone.

**Validates: Requirements 5.3, 7.4**

### Property 11: Team Name Validation

*For any* fixture with a team name that does not match an existing Team node in the database, the System SHALL reject the fixture with a descriptive error message.

**Validates: Requirements 5.4**

### Property 12: Fixture Update Idempotency

*For any* existing fixture, refreshing with identical data SHALL not create duplicate nodes and SHALL preserve the original fixture unchanged.

**Validates: Requirements 6.2**


### Property 13: Historical Fixture Preservation

*For any* fixture that has already occurred (kickoff_time < current_date), updating fixture attributes SHALL preserve the home_score and away_score fields.

**Validates: Requirements 6.3**

### Property 14: Cache Invalidation on Fixture Update

*For any* cached prediction for a fixture, updating that fixture's date or teams SHALL invalidate all cached predictions associated with that fixture_id.

**Validates: Requirements 6.4, 10.2, 10.4**

### Property 15: UTC Storage Invariant

*For all* fixture nodes in the database, the kickoff_time property SHALL have UTC timezone information.

**Validates: Requirements 7.1, 7.2**

### Property 16: Timezone Conversion Preserves Time

*For any* UTC datetime, converting to a local timezone and back to UTC SHALL produce the original datetime.

**Validates: Requirements 7.3**

### Property 17: Opponent Defensive Strength Calculation

*For any* team with match history, the opponent_defensive_strength SHALL equal the rolling 5-game average of goals conceded by that team.

**Validates: Requirements 8.1**

### Property 18: Opponent Offensive Strength Calculation

*For any* team with match history, the opponent_offensive_strength SHALL equal the rolling 5-game average of goals scored by that team.

**Validates: Requirements 8.2**


### Property 19: Strength Metric Update After Matchday

*For any* matchday that has completed, updating opponent strength metrics SHALL reflect the most recent match results in the rolling averages.

**Validates: Requirements 8.4**

### Property 20: Fixture Graph Relationships

*For any* created fixture, there SHALL exist exactly one PLAYS_HOME relationship to the home team and exactly one PLAYS_AWAY relationship to the away team.

**Validates: Requirements 9.2**

### Property 21: Fixture Deletion Cascade

*For any* fixture node that is deleted, all associated relationships (PLAYS_HOME, PLAYS_AWAY, IN_MATCHDAY) SHALL also be removed from the graph.

**Validates: Requirements 9.5**

### Property 22: Prediction Cache Key Structure

*For any* cached prediction, the cache key SHALL follow the format `prediction:{player_id}:{fixture_id}:{model_version}`.

**Validates: Requirements 10.1**

### Property 23: Cache TTL Based on Fixture Timing

*For any* prediction cached for a future fixture, the cache TTL SHALL decrease as the fixture approaches (24h if >7 days away, 6h if 1-7 days away, 1h if <1 day away).

**Validates: Requirements 10.3**

---


## Performance Considerations

### Query Optimization

**Indexed Queries:**
- Fixture lookups by `kickoff_time` leverage Neo4j B-tree index → O(log n)
- Team lookups by name use unique constraint → O(1)
- Player lookups by name use unique constraint → O(1)

**Query Complexity:**
- `get_next_match()`: Single Cypher query with 2 pattern matches + 1 filter + sort → ~5-10ms
- `load_fixtures_from_csv()`: Batch MERGE operations (380 fixtures) → ~2-3 seconds

**Caching Impact:**
- Cache hit rate target: >80% for fixture-aware predictions
- Cache hit: ~1ms (Redis) or <0.1ms (in-memory)
- Cache miss: ~50-100ms (Neo4j query + ML inference + feature engineering)
- **Speedup: 50-100x for cached predictions**

### Scalability

**Data Volume:**
- Fixtures per season: 380 (20 teams × 38 gameweeks / 2)
- Fixture nodes growth: 380 per year
- 10-year projection: 3,800 fixture nodes (negligible for Neo4j)

**Prediction Throughput:**
- Uncached: ~10-20 predictions/second (bottleneck: feature engineering)
- Cached: ~1,000 predictions/second (limited by Redis network latency)

### Memory Usage

**Cache Size Estimates:**
- PlayerPrediction object: ~500 bytes
- Predictions per gameweek: ~500 players × 20 teams = ~10,000 predictions
- Cache size per gameweek: 10,000 × 500 bytes = ~5 MB
- Total season cache: 5 MB × 38 gameweeks = ~190 MB (acceptable)


---

## Security Considerations

### Input Validation

**CSV Upload:**
- Validate file size limit: Max 10 MB
- Validate CSV structure: Required columns present
- Validate date formats: Reject malformed dates
- Sanitize team names: Prevent injection attacks in Cypher queries

**API Endpoints:**
- Rate limiting: 100 requests/minute per IP for prediction endpoints
- Authentication: Admin-only access for `/api/fixtures/load` endpoint
- Parameter validation: Reject overly long player names (max 100 chars)

### Data Integrity

**Transaction Safety:**
- Use Neo4j transactions for fixture loading (rollback on error)
- Atomic fixture updates (all-or-nothing for related nodes/relationships)

**Audit Logging:**
- Log all fixture data modifications with timestamp and user
- Log prediction cache invalidations
- Retain logs for 90 days

### Error Information Disclosure

**Safe Error Messages:**
- External API: "Player not found" (don't expose internal IDs)
- Internal logs: Include stack traces and detailed context
- Never expose Neo4j Cypher query syntax in error responses


---

## Deployment and Rollout

### Database Migration

**Step 1: Schema Updates** (no downtime)
```cypher
# Create new node type and indexes
CREATE CONSTRAINT fixture_id IF NOT EXISTS FOR (f:Fixture) REQUIRE f.id IS UNIQUE;
CREATE INDEX fixture_kickoff IF NOT EXISTS FOR (f:Fixture) ON (f.kickoff_time);
CREATE INDEX fixture_matchday IF NOT EXISTS FOR (f:Fixture) ON (f.matchday);
```

**Step 2: Fixture Data Load** (scheduled maintenance window)
```bash
# Load 2026-27 season fixtures
curl -X POST http://localhost:8000/api/fixtures/load \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "FPL_2026_2027.csv", "season_id": "2026-27"}'
```

**Step 3: Deploy New Code** (blue-green deployment)
1. Deploy new backend code to staging
2. Run integration tests
3. Switch production traffic to new version
4. Monitor error rates for 1 hour

### Feature Flag Strategy

**Phase 1: Shadow Mode** (Week 1)
- FixtureService deployed but not used by prediction API
- Log fixture context alongside predictions for analysis
- Verify fixture queries return expected results

**Phase 2: Partial Rollout** (Week 2)
- Enable fixture-aware predictions for 10% of requests
- A/B test: Compare prediction accuracy with/without fixture context
- Monitor cache hit rates and query performance

**Phase 3: Full Rollout** (Week 3)
- Enable fixture-aware predictions for 100% of requests
- Remove feature flag
- Publish updated API documentation


### Monitoring and Observability

**Key Metrics:**

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Prediction API latency (p95) | <100ms | >200ms |
| Cache hit rate | >80% | <70% |
| Fixture query latency (p95) | <10ms | >50ms |
| Next match query success rate | >99% | <95% |
| CSV load time (380 fixtures) | <5s | >10s |

**Dashboards:**
- Prediction request volume by endpoint
- Cache hit/miss rates over time
- Fixture query performance by team
- Error rate by error type

**Logging:**
```python
logger.info("Fixture query", extra={
    "player_name": "Mohamed Salah",
    "current_date": "2026-08-20T10:00:00Z",
    "next_fixture_id": "fixture-2026-27-md2-liv-mun",
    "query_time_ms": 8.5
})
```

### Rollback Plan

**Trigger Conditions:**
- Error rate >5% for prediction API
- Cache hit rate drops below 50%
- Neo4j connection failures >10% of requests

**Rollback Steps:**
1. Disable fixture-aware predictions via feature flag
2. Revert to previous backend version
3. Investigate root cause in staging environment
4. Fix issues and redeploy

**Data Rollback:**
- Fixture nodes can be safely deleted without affecting historical player data
- Cache can be flushed without data loss


---

## Future Enhancements

### Potential Extensions

**1. Live Match Updates**
- Integrate with Premier League live scores API
- Update fixture status to "live" during matches
- Invalidate predictions in real-time when matches start

**2. Historical Fixture Analysis**
- Store completed fixture results (home_score, away_score)
- Enable "how did this prediction perform?" retrospective analysis
- Build fixture difficulty rating (FDR) based on historical team performance

**3. Multi-Gameweek Predictions**
- Predict points for next 3-5 gameweeks instead of just next match
- Factor in fixture congestion (3 matches in 7 days)
- Rotation risk modeling based on fixture schedule

**4. Injury and Suspension Integration**
- Check injury news before returning prediction
- Return `null` prediction for suspended players
- Include injury probability in prediction confidence interval

**5. Transfer Market Integration**
- Track predicted player transfers (e.g., player moving teams)
- Update PLAYS_FOR relationships when transfers occur
- Invalidate predictions for transferred players

### Design Decisions and Trade-offs

**Decision 1: Neo4j vs. PostgreSQL for Fixtures**
- **Chosen:** Neo4j
- **Rationale:** Fixtures already fit naturally into existing graph schema (Team nodes), traversal queries are more elegant
- **Trade-off:** Fixture queries slightly slower than SQL (5-10ms vs 1-2ms), but acceptable for user-facing API

**Decision 2: Cache Key Includes Model Version**
- **Chosen:** `prediction:{player_id}:{fixture_id}:{model_version}`
- **Rationale:** Allows deploying new ML models without manually invalidating entire cache
- **Trade-off:** Cache misses when model version changes, but ensures predictions use correct model

**Decision 3: Single Next Match vs. Multi-Gameweek**
- **Chosen:** Single next match only
- **Rationale:** Simpler implementation, reduces feature complexity, sufficient for MVP
- **Trade-off:** Less useful for planning transfers multiple gameweeks ahead (future enhancement)

**Decision 4: UTC-Only Storage**
- **Chosen:** Store all times in UTC, convert to local for display only
- **Rationale:** Eliminates timezone ambiguity, simplifies date comparisons, standard best practice
- **Trade-off:** Requires timezone conversion in frontend, but this is a one-liner

