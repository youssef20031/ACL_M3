# Implementation Plan: Date-Aware Match Predictions

## Overview

This implementation plan breaks down the date-aware match predictions feature into discrete, actionable coding tasks. The feature integrates Premier League 2026-27 season fixture data into the existing FPL prediction system, enabling fixture-specific predictions that update automatically based on the current date.

**Implementation Language:** Python (FastAPI backend with Neo4j)

**Key Components:**
- FixtureService for managing fixture data and determining next matches
- Neo4j schema enhancements (Fixture nodes, relationships, indexes)
- API endpoint enhancements to include fixture context
- Fixture data loading from CSV
- Prediction caching with dynamic TTL
- Property-based tests for correctness validation

## Tasks

- [x] 1. Setup Neo4j schema for fixture storage
  - Create Fixture node type with unique constraint on `id`
  - Create indexes on `kickoff_time` and `matchday` properties
  - Create PLAYS_HOME and PLAYS_AWAY relationship types
  - Create IN_MATCHDAY relationship type
  - Create Matchday node type if not exists
  - _Requirements: 9.1, 9.2, 9.4_

- [x] 2. Implement FixtureService core functionality
  - [x] 2.1 Create `services/fixture_service.py` with FixtureService class
    - Implement `__init__` with Neo4j connection dependency
    - Define data models for FixtureContext and FixtureData
    - _Requirements: 1.1, 9.1_
  
  - [x] 2.2 Implement `get_next_match()` method
    - Write Cypher query to find player's next upcoming fixture
    - Determine home/away status from relationship type
    - Handle case where player has no team (return None)
    - Handle case where team has no upcoming fixtures (return None)
    - Handle Double Gameweek scenario (return earliest by kickoff time)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 9.3_
  
  - [ ]* 2.3 Write property test for next match determination
    - **Property 5: Next Match Before Kickoff**
    - **Validates: Requirements 2.1**
  
  - [ ]* 2.4 Write property test for past fixture exclusion
    - **Property 6: Next Match Past Fixture Exclusion**
    - **Validates: Requirements 2.2**
  
  - [ ]* 2.5 Write property test for Double Gameweek earliest selection
    - **Property 7: Double Gameweek Earliest Selection**
    - **Validates: Requirements 2.5**
  
  - [x] 2.6 Implement `get_team_next_match()` method
    - Write Cypher query for team-based fixture lookup
    - Reuse logic from `get_next_match()`
    - _Requirements: 1.2_
  
  - [x] 2.7 Implement `is_fixture_past()` method
    - Compare fixture kickoff_time against current_date
    - Handle timezone conversions to UTC
    - _Requirements: 2.2, 7.1, 7.2_

- [x] 3. Implement fixture data loading from CSV
  - [x] 3.1 Implement `load_fixtures_from_csv()` method in FixtureService
    - Parse CSV file with pandas
    - Validate required columns (Matchday, Home Team, Away Team, Date)
    - Convert date/time strings to UTC datetime objects
    - Validate team names against existing Team nodes in Neo4j
    - Generate unique fixture IDs
    - Use MERGE operations for idempotent loading
    - Create fixture nodes with all properties
    - Create PLAYS_HOME and PLAYS_AWAY relationships
    - Create IN_MATCHDAY relationships
    - Return count of fixtures loaded
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 9.1, 9.2_
  
  - [ ]* 3.2 Write property test for fixture storage completeness
    - **Property 1: Fixture Storage Completeness**
    - **Validates: Requirements 1.1**
  
  - [ ]* 3.3 Write property test for fixture validation rejection
    - **Property 3: Fixture Validation Rejection**
    - **Validates: Requirements 1.4**
  
  - [ ]* 3.4 Write property test for team name validation
    - **Property 11: Team Name Validation**
    - **Validates: Requirements 5.4**
  
  - [ ]* 3.5 Write property test for duplicate fixture rejection
    - **Property 4: Duplicate Fixture Rejection**
    - **Validates: Requirements 1.5**
  
  - [ ]* 3.6 Write unit tests for CSV parsing edge cases
    - Test missing Time column (defaults to 15:00)
    - Test malformed date formats
    - Test empty CSV file
    - _Requirements: 5.1, 5.5_

- [x] 4. Implement fixture update and refresh mechanism
  - [x] 4.1 Implement `update_fixture()` method in FixtureService
    - Accept fixture_id and updates dictionary
    - Query existing fixture node
    - Apply updates via Cypher SET operations
    - Preserve home_score and away_score for past fixtures
    - Return success boolean
    - _Requirements: 6.1, 6.3_
  
  - [ ]* 4.2 Write property test for fixture update idempotency
    - **Property 12: Fixture Update Idempotency**
    - **Validates: Requirements 6.2**
  
  - [ ]* 4.3 Write property test for historical fixture preservation
    - **Property 13: Historical Fixture Preservation**
    - **Validates: Requirements 6.3**
  
  - [x] 4.4 Add audit logging to update operations
    - Log fixture_id, timestamp, user, and changed fields
    - Store logs in application log file
    - _Requirements: 6.5_

- [ ] 5. Enhance ML prediction integration with fixture context
  - [ ] 5.1 Update MLAPIIntegration class in `ml/api_integration.py`
    - Add FixtureService as constructor dependency
    - Modify `predict_player_next_gameweek()` to call `fixture_service.get_next_match()`
    - Create `_get_player_stats_with_fixture()` helper method
    - Enrich player_data dictionary with fixture context (opponent_team, was_home)
    - Include fixture_context in PlayerPrediction response
    - Handle case where fixture_context is None (no upcoming fixtures)
    - _Requirements: 3.1, 3.2, 3.3, 3.5_
  
  - [ ]* 5.2 Write property test for prediction response fixture context
    - **Property 8: Prediction Response Fixture Context**
    - **Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.2**
  
  - [ ]* 5.3 Write property test for prediction features include fixture context
    - **Property 9: Prediction Features Include Fixture Context**
    - **Validates: Requirements 3.4, 8.3**
  
  - [ ]* 5.4 Write unit tests for null fixture handling
    - Test prediction when player has no team
    - Test prediction when team has no upcoming fixtures
    - Verify response includes null fixture_context field
    - _Requirements: 3.5, 4.3_

- [x] 6. Implement opponent strength calculation
  - [x] 6.1 Create `calculate_opponent_strength()` method in FixtureService or new util module
    - Query team's last 5 completed fixtures
    - Calculate rolling 5-game average of goals conceded (defensive strength)
    - Calculate rolling 5-game average of goals scored (offensive strength)
    - Handle teams with fewer than 5 matches (use available matches)
    - Return default league average when no data available
    - _Requirements: 8.1, 8.2, 8.5_
  
  - [ ]* 6.2 Write property test for opponent defensive strength calculation
    - **Property 17: Opponent Defensive Strength Calculation**
    - **Validates: Requirements 8.1**
  
  - [ ]* 6.3 Write property test for opponent offensive strength calculation
    - **Property 18: Opponent Offensive Strength Calculation**
    - **Validates: Requirements 8.2**
  
  - [ ] 6.4 Implement strength metric update trigger
    - Create method to update all team strength metrics after matchday completion
    - Integrate with existing matchday processing workflow (if exists)
    - _Requirements: 8.4_
  
  - [ ]* 6.5 Write property test for strength metric update after matchday
    - **Property 19: Strength Metric Update After Matchday**
    - **Validates: Requirements 8.4**

- [x] 7. Implement prediction caching strategy
  - [x] 7.1 Create PredictionCache class in new `services/cache.py` module
    - Support Redis and in-memory fallback
    - Implement `get(key)` method
    - Implement `set(key, value, ttl)` method
    - Implement `invalidate(key)` method
    - Implement `calculate_cache_ttl(fixture_kickoff)` method (dynamic TTL logic)
    - _Requirements: 10.1, 10.2, 10.3_
  
  - [ ]* 7.2 Write property test for cache key structure
    - **Property 22: Prediction Cache Key Structure**
    - **Validates: Requirements 10.1**
  
  - [ ]* 7.3 Write property test for cache TTL based on fixture timing
    - **Property 23: Cache TTL Based on Fixture Timing**
    - **Validates: Requirements 10.3**
  
  - [x] 7.4 Integrate caching into MLAPIIntegration
    - Check cache before generating prediction
    - Store prediction in cache after generation
    - Use composite key: `prediction:{player_id}:{fixture_id}:{model_version}`
    - _Requirements: 10.1_
  
  - [x] 7.5 Implement cache invalidation on fixture updates
    - Call cache invalidation in `update_fixture()` method
    - Invalidate all predictions for affected fixture_id
    - _Requirements: 6.4, 10.4_
  
  - [ ]* 7.6 Write property test for cache invalidation on fixture update
    - **Property 14: Cache Invalidation on Fixture Update**
    - **Validates: Requirements 6.4, 10.2, 10.4**

- [x] 8. Checkpoint - Ensure all core services are working
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Add API endpoints for fixture management
  - [x] 9.1 Add `POST /api/fixtures/load` endpoint in `api_main.py`
    - Accept DataLoadRequest with csv_path, season_id, clear_existing
    - Call `fixture_service.load_fixtures_from_csv()`
    - Return success status and count of fixtures loaded
    - Add authentication check (admin-only)
    - _Requirements: 1.3, 5.1_
  
  - [x] 9.2 Add `GET /api/fixtures/next` endpoint in `api_main.py`
    - Accept optional team query parameter
    - Return list of upcoming fixtures for team or all teams
    - Include home_team, away_team, match_date, matchday fields
    - _Requirements: 1.2_
  
  - [x] 9.3 Update `GET /api/predict/player` endpoint response schema
    - Add optional fixture_context field to PlayerPredictionResponse
    - Include opponent_team, home_away, match_date, matchday, fixture_id
    - Ensure backward compatibility (fixture_context can be null)
    - _Requirements: 4.1, 4.2, 4.4_
  
  - [x]* 9.4 Write integration tests for fixture endpoints
    - Test POST /api/fixtures/load with valid CSV
    - Test GET /api/fixtures/next with team filter
    - Test GET /api/fixtures/next without filter
    - Test error handling for invalid CSV
    - _Requirements: 1.3, 1.2, 4.5, 5.5_

- [x] 10. Implement date/time utility functions
  - [x] 10.1 Create `utils/datetime_utils.py` module
    - Implement `parse_to_utc(date_str, time_str)` function
    - Implement `is_past(datetime_utc)` function
    - Implement `to_local_timezone(datetime_utc, tz_name)` function
    - Handle daylight saving time transitions
    - _Requirements: 5.3, 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [x]* 10.2 Write property test for date parsing to UTC
    - **Property 10: Date Parsing to UTC**
    - **Validates: Requirements 5.3, 7.4**
  
  - [x]* 10.3 Write property test for UTC storage invariant
    - **Property 15: UTC Storage Invariant**
    - **Validates: Requirements 7.1, 7.2**
  
  - [x]* 10.4 Write property test for timezone conversion preserves time
    - **Property 16: Timezone Conversion Preserves Time**
    - **Validates: Requirements 7.3**
  
  - [x]* 10.5 Write unit tests for DST edge cases
    - Test spring forward (2am becomes 3am)
    - Test fall back (2am happens twice)
    - _Requirements: 7.5_

- [x] 11. Add Neo4j relationship validation and cleanup
  - [x] 11.1 Implement `validate_fixture_relationships()` method
    - Query fixture and count PLAYS_HOME relationships (must be exactly 1)
    - Query fixture and count PLAYS_AWAY relationships (must be exactly 1)
    - Return validation errors if constraints violated
    - _Requirements: 9.2_
  
  - [ ]* 11.2 Write property test for fixture graph relationships
    - **Property 20: Fixture Graph Relationships**
    - **Validates: Requirements 9.2**
  
  - [x] 11.3 Implement cascade delete for fixture nodes
    - Modify `delete_fixture()` method to remove all relationships first
    - Delete PLAYS_HOME, PLAYS_AWAY, IN_MATCHDAY relationships
    - Then delete fixture node
    - _Requirements: 9.5_
  
  - [ ]* 11.4 Write property test for fixture deletion cascade
    - **Property 21: Fixture Deletion Cascade**
    - **Validates: Requirements 9.5**

- [x] 12. Add error handling and validation
  - [x] 12.1 Create custom exception classes in `services/exceptions.py`
    - Define FixtureServiceError base class
    - Define FixtureNotFoundError
    - Define InvalidFixtureDataError
    - Define DuplicateFixtureError
    - _Requirements: 1.4, 1.5, 4.5, 5.5_
  
  - [x] 12.2 Add input validation to all FixtureService methods
    - Validate player_name is not empty
    - Validate fixture_id format
    - Validate date formats
    - Raise descriptive errors with helpful messages
    - _Requirements: 1.4, 4.5_
  
  - [x] 12.3 Add graceful degradation to prediction endpoint
    - Wrap fixture service calls in try-except
    - Fall back to generic prediction if fixture lookup fails
    - Log warnings but don't fail entire request
    - Return prediction with null fixture_context
    - _Requirements: 3.5, 4.3_
  
  - [ ]* 12.4 Write integration tests for error scenarios
    - Test 404 response for invalid player name
    - Test 400 response for malformed fixture data
    - Test 503 response for Neo4j connection failure (mocked)
    - _Requirements: 4.5_

- [x] 13. Checkpoint - Ensure all error handling is robust
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Add logging and monitoring
  - [x] 14.1 Add structured logging to FixtureService methods
    - Log fixture queries with execution time
    - Log fixture data load operations with counts
    - Log cache hits/misses
    - Include player_name, fixture_id, query_time_ms in log context
    - _Requirements: 6.5_
  
  - [ ] 14.2 Add monitoring metrics collection
    - Track prediction API latency (p95)
    - Track cache hit rate percentage
    - Track fixture query latency (p95)
    - Track next match query success rate
    - Use existing monitoring infrastructure (Prometheus/StatsD if available)

- [x] 15. Create database migration script
  - [x] 15.1 Create `migrations/add_fixture_schema.cypher` file
    - Include constraint creation statements
    - Include index creation statements
    - Add comments explaining each statement
    - _Requirements: 9.1, 9.4_
  
  - [x] 15.2 Create `migrations/load_2026_27_fixtures.py` script
    - Accept command-line arguments for CSV path
    - Call FixtureService.load_fixtures_from_csv()
    - Print progress and summary
    - Handle errors gracefully
    - _Requirements: 5.1_

- [x] 16. Update API documentation
  - [x] 16.1 Update OpenAPI/Swagger schemas in api_main.py
    - Document new fixture_context field in PlayerPredictionResponse
    - Document POST /api/fixtures/load endpoint
    - Document GET /api/fixtures/next endpoint
    - Include request/response examples
  
  - [x] 16.2 Create API usage examples in `docs/fixture_aware_predictions.md`
    - Show how to request player prediction with fixture context
    - Show how to load fixture data
    - Show how to query upcoming fixtures
    - Include curl examples

- [x] 17. Add frontend integration (if applicable)
  - [ ] 17.1 Update prediction display component to show fixture context
    - Display opponent team name
    - Display home/away indicator (H/A badge)
    - Display match date in local timezone
    - Display matchday number
    - Show "No upcoming fixture" message when context is null
  
  - [ ] 17.2 Add fixture loading UI for admin users (optional)
    - File upload component for CSV
    - Progress indicator during load
    - Success/error message display


- [x] 18. Write end-to-end integration tests
  - [x]* 18.1 Write E2E test for complete fixture-aware prediction flow
    - Load fixtures from test CSV
    - Request prediction for player with upcoming fixture
    - Verify response includes fixture_context
    - Verify fixture_context fields are correct
    - _Requirements: 1.1, 2.1, 3.1, 3.2, 3.3, 4.1_
  
  - [x]* 18.2 Write E2E test for prediction caching behavior
    - Request same prediction twice
    - Verify second request is faster (cache hit)
    - Update fixture
    - Verify cache is invalidated
    - _Requirements: 10.1, 10.2, 10.4_
  
  - [x]* 18.3 Write E2E test for fixture refresh workflow
    - Load initial fixtures
    - Modify fixture data
    - Reload fixtures
    - Verify updated data is reflected
    - Verify historical fixtures preserved
    - _Requirements: 6.2, 6.3_

- [ ] 19. Final checkpoint - Complete system verification
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based tests and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows
- Checkpoints ensure incremental validation at key milestones
- All fixture data handling uses UTC timezone internally
- Caching strategy uses dynamic TTL based on fixture timing for optimal performance
- Error handling includes graceful degradation to maintain service availability
