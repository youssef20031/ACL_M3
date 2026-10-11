# Requirements Document

## Introduction

This feature introduces date-aware match predictions to the FPL (Fantasy Premier League) assistant application. Currently, the ML prediction system generates player point predictions but lacks awareness of specific fixture schedules and match timing. This enhancement will integrate Premier League 2026-27 season fixture data to provide predictions for each player's next specific match, automatically updating as matches complete. The system will determine the "next match" based on the current date, ensuring predictions remain relevant throughout the season.

## Glossary

- **System**: The FPL FantasyTrivia application backend (FastAPI Python)
- **Fixture**: A scheduled Premier League match with date, time (Time is TBD but date is available), home team, and away team
- **Matchday**: A round of fixtures in the Premier League season (e.g., Matchday 1, Matchday 2)
- **Next_Match**: The upcoming fixture for a player or team based on the current date
- **ML_Predictor**: The existing machine learning prediction engine (ml/predictor.py)
- **Fixture_Service**: New service component for managing fixture data and queries
- **Current_Date**: The system date/time used to determine next matches
- **Fixture_Context**: Opponent team, home/away status, and fixture difficulty for a specific match
- **Prediction_Endpoint**: REST API endpoint that returns ML-predicted points

## Requirements

### Requirement 1: Fixture Data Storage and Management

**User Story:** As a system administrator, I want fixture schedule data stored and accessible, so that the system can provide date-aware predictions.

#### Acceptance Criteria

1. THE System SHALL store Premier League 2026-27 season fixture data including match date, match time, home team, away team, and matchday number
2. WHEN fixture data is requested for a specific team, THE Fixture_Service SHALL return all fixtures for that team ordered by date
3. THE System SHALL provide an API endpoint to update fixture data
4. THE System SHALL validate fixture data to ensure required fields (date, time, home_team, away_team, matchday) are present
5. WHEN duplicate fixture entries are detected, THE System SHALL reject the duplicate and return an error message

### Requirement 2: Next Match Determination Logic

**User Story:** As a developer, I want the system to determine each player's next match based on current date, so that predictions reflect upcoming fixtures.

#### Acceptance Criteria

1. WHEN the Current_Date is before a player's next fixture date, THE Fixture_Service SHALL return that fixture as the Next_Match
2. WHEN the Current_Date is after a fixture's completion time, THE Fixture_Service SHALL exclude that fixture and select the subsequent fixture
3. WHEN a player has no upcoming fixtures, THE Fixture_Service SHALL return null for Next_Match
4. THE Fixture_Service SHALL determine Next_Match by comparing Current_Date against fixture kickoff_time
5. WHEN multiple fixtures occur on the same day for a team, THE Fixture_Service SHALL return the earliest fixture not yet completed

### Requirement 3: Fixture-Aware ML Prediction Integration

**User Story:** As an API consumer, I want player predictions to include opponent and fixture context, so that predictions are more informative and accurate.

#### Acceptance Criteria

1. WHEN a prediction is requested for a player, THE ML_Predictor SHALL include the Next_Match opponent team in the prediction response
2. WHEN a prediction is requested for a player, THE System SHALL include home/away status in the prediction response
3. WHEN a prediction is requested for a player, THE System SHALL include fixture date in the prediction response
4. THE ML_Predictor SHALL use opponent strength and home/away status as input features when generating predictions
5. WHEN no Next_Match exists for a player, THE System SHALL return predictions without fixture context

### Requirement 4: Prediction API Enhancements

**User Story:** As a frontend developer, I want prediction endpoints to return fixture details, so that I can display next match information to users.

#### Acceptance Criteria

1. WHEN the `/api/predict/player` endpoint is called, THE Prediction_Endpoint SHALL return fixture details in the response payload
2. THE Prediction_Endpoint SHALL include fields for opponent_team, match_date, home_away_status, and matchday_number
3. WHEN fixture data is unavailable for a player, THE Prediction_Endpoint SHALL return null fixture fields without failing
4. THE Prediction_Endpoint SHALL return HTTP 200 status for successful predictions with fixture context
5. WHEN an invalid player identifier is provided, THE Prediction_Endpoint SHALL return HTTP 404 status with an error message

### Requirement 5: Fixture Data Source Integration

**User Story:** As a system operator, I want fixture data loaded from a reliable source, so that the system has accurate schedule information.

#### Acceptance Criteria

1. THE System SHALL support loading fixture data from CSV files
2. THE System SHALL support loading fixture data from API endpoints
3. WHEN fixture data is loaded, THE System SHALL parse date and time fields into UTC timestamps
4. THE System SHALL validate that team names in fixture data match existing team records in the database
5. WHEN fixture data loading fails, THE System SHALL log the error and return a descriptive error message

### Requirement 6: Fixture Update and Refresh Mechanism

**User Story:** As a system operator, I want to update fixture data periodically, so that schedule changes are reflected in predictions.

#### Acceptance Criteria

1. THE System SHALL provide an endpoint to refresh fixture data from the configured source
2. WHEN fixture data is refreshed, THE System SHALL compare new data against existing data and update only changed records
3. THE System SHALL preserve historical fixture data for matches that have already occurred
4. WHEN a fixture date is modified, THE System SHALL update the fixture record and invalidate cached predictions
5. THE System SHALL log all fixture update operations with timestamps for audit purposes

### Requirement 7: Date and Timezone Handling

**User Story:** As a developer, I want consistent date/time handling across the system, so that match timing logic is reliable.

#### Acceptance Criteria

1. THE System SHALL store all fixture dates and times in UTC format
2. WHEN comparing Current_Date to fixture dates, THE System SHALL use UTC timestamps
3. THE System SHALL provide utility functions to convert fixture times to local timezones for display
4. WHEN processing fixture data with timezone information, THE System SHALL normalize to UTC before storage
5. THE System SHALL handle daylight saving time transitions correctly when determining next matches

### Requirement 8: Fixture Difficulty and Opponent Strength

**User Story:** As a data analyst, I want opponent strength metrics included in predictions, so that fixture difficulty influences point forecasts.

#### Acceptance Criteria

1. THE System SHALL calculate opponent defensive strength based on historical data
2. THE System SHALL calculate opponent offensive strength based on historical data
3. WHEN generating predictions, THE ML_Predictor SHALL use opponent_defensive_strength and opponent_offensive_strength as features
4. THE System SHALL update opponent strength metrics after each matchday
5. WHEN opponent strength data is unavailable, THE ML_Predictor SHALL use league average strength values

### Requirement 9: Neo4j Graph Database Integration

**User Story:** As a backend developer, I want fixture data stored in Neo4j, so that it integrates with existing player and team data.

#### Acceptance Criteria

1. THE System SHALL create a Fixture node type in Neo4j with properties: date, time, matchday, home_score, away_score
2. THE System SHALL create PLAYS_HOME and PLAYS_AWAY relationships between Team nodes and Fixture nodes
3. WHEN querying for a player's next fixture, THE System SHALL traverse Player → PLAYS_FOR → Team → PLAYS_HOME/PLAYS_AWAY → Fixture relationships
4. THE System SHALL index Fixture nodes by date and matchday for query performance
5. WHEN deleting fixture data, THE System SHALL remove Fixture nodes and associated relationships

### Requirement 10: Prediction Caching Strategy

**User Story:** As a system operator, I want predictions cached by fixture, so that repeated requests are performant.

#### Acceptance Criteria

1. THE System SHALL cache predictions using a composite key of player_id and fixture_id
2. WHEN a fixture date passes, THE System SHALL invalidate cached predictions for that fixture
3. THE System SHALL set prediction cache TTL to match the fixture date
4. WHEN fixture data is updated, THE System SHALL invalidate all cached predictions for affected fixtures
5. THE System SHALL support both Redis and in-memory caching for prediction storage

