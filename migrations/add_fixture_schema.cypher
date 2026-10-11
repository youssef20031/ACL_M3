// ============================================================================
// Neo4j Schema Migration: Add Fixture Support
// ============================================================================
// This migration adds the schema for fixture-aware match predictions.
// It creates constraints, indexes, and node types for fixture management.
//
// Prerequisites:
// - Neo4j database must be running
// - User must have admin privileges
// - Team nodes must already exist in the database
//
// Usage:
// Run this file using Neo4j Browser, cypher-shell, or Neo4j Desktop:
//   cypher-shell -u neo4j -p password -f add_fixture_schema.cypher
//
// ============================================================================

// ----------------------------------------------------------------------------
// 1. Create Unique Constraint on Fixture ID
// ----------------------------------------------------------------------------
// Ensures each fixture has a unique identifier
// This prevents duplicate fixture entries and enables fast lookups
CREATE CONSTRAINT fixture_id_unique IF NOT EXISTS
FOR (f:Fixture)
REQUIRE f.id IS UNIQUE;

// ----------------------------------------------------------------------------
// 2. Create Index on Fixture Kickoff Time
// ----------------------------------------------------------------------------
// Optimizes queries that filter fixtures by date/time
// Critical for "next match" queries that find upcoming fixtures
CREATE INDEX fixture_kickoff_time IF NOT EXISTS
FOR (f:Fixture)
ON (f.kickoff_time);

// ----------------------------------------------------------------------------
// 3. Create Index on Fixture Matchday
// ----------------------------------------------------------------------------
// Optimizes queries that filter by matchday number
// Useful for matchday-specific analytics and reporting
CREATE INDEX fixture_matchday IF NOT EXISTS
FOR (f:Fixture)
ON (f.matchday);

// ----------------------------------------------------------------------------
// 4. Create Index on Fixture Season
// ----------------------------------------------------------------------------
// Optimizes queries that filter fixtures by season
// Enables efficient season-specific fixture queries
CREATE INDEX fixture_season IF NOT EXISTS
FOR (f:Fixture)
ON (f.season_id);

// ----------------------------------------------------------------------------
// 5. Create Index on Fixture Status
// ----------------------------------------------------------------------------
// Optimizes queries that filter by fixture status (scheduled, completed, etc.)
// Useful for filtering completed vs upcoming fixtures
CREATE INDEX fixture_status IF NOT EXISTS
FOR (f:Fixture)
ON (f.status);

// ----------------------------------------------------------------------------
// 6. Create Unique Constraint on Season ID
// ----------------------------------------------------------------------------
// Ensures each season has a unique identifier
CREATE CONSTRAINT season_id_unique IF NOT EXISTS
FOR (s:Season)
REQUIRE s.id IS UNIQUE;

// ----------------------------------------------------------------------------
// 7. Verify Schema
// ----------------------------------------------------------------------------
// Display all constraints and indexes to confirm successful creation
SHOW CONSTRAINTS;
SHOW INDEXES;

// ============================================================================
// Notes:
// ============================================================================
// - Fixture nodes have properties: id, kickoff_time, matchday, season_id, status
// - Optional properties: home_score, away_score (added after match completion)
// - Relationships: (:Fixture)-[:PLAYS_HOME]->(:Team)
//                 (:Fixture)-[:PLAYS_AWAY]->(:Team)
//                 (:Fixture)-[:IN_SEASON]->(:Season)
// - All timestamps are stored in UTC
// - Fixture IDs follow format: fixture-{season_id}-md{matchday}-{home_abbr}-{away_abbr}
// ============================================================================
