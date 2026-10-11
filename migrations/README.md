# Database Migrations

This directory contains Neo4j database schema migrations for the FPL FantasyTrivia application.

## Available Migrations

### 1. Add Fixture Schema (`add_fixture_schema.cypher`)

**Purpose**: Sets up the Neo4j schema for storing fixture data to enable date-aware match predictions.

**What it does**:
- Creates unique constraint on `Fixture.id`
- Creates indexes on `Fixture.kickoff_time` and `Fixture.matchday` for efficient queries
- Creates unique constraint on `Matchday.id`
- Prepares the schema for PLAYS_HOME, PLAYS_AWAY, and IN_MATCHDAY relationships

**Requirements**: Implements requirements 9.1, 9.2, 9.4 from the date-aware-match-predictions spec.

## Running Migrations

### Method 1: Using Python Script (Recommended)

```bash
# Make sure you're in the project root
cd c:\ACL2\FPL\ACL_M3

# Run the migration
python migrations/run_fixture_schema_migration.py
```

The script will:
1. Connect to your Neo4j database using credentials from `.env`
2. Execute all statements in the migration file
3. Verify the schema was created correctly
4. Show a summary of constraints and indexes

### Method 2: Manual Execution via Neo4j Browser

1. Open Neo4j Browser (usually at http://localhost:7474)
2. Copy the contents of `add_fixture_schema.cypher`
3. Paste into the query editor
4. Execute each statement (separated by semicolons)

### Method 3: Using Cypher Shell

```bash
cat migrations/add_fixture_schema.cypher | cypher-shell -u neo4j -p your_password
```

## Environment Variables

Make sure your `.env` file contains:

```env
NEO4J_URI=bolt://localhost:7687  # or your Neo4j instance URL
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Verifying Migration Success

After running the migration, verify it with these queries:

```cypher
// Check constraints
SHOW CONSTRAINTS;

// Check indexes
SHOW INDEXES;

// Check if any Fixture nodes exist (will be 0 until data is loaded)
MATCH (f:Fixture) RETURN count(f) as fixture_count;

// Check if any Matchday nodes exist
MATCH (m:Matchday) RETURN count(m) as matchday_count;
```

## Next Steps After Migration

1. **Load Fixture Data**: Use the `FixtureService.load_fixtures_from_csv()` method to import fixture data
2. **Verify Data**: Check that fixtures are correctly linked to teams
3. **Test Queries**: Run the next match determination queries from the design doc

## Troubleshooting

### "Already exists" errors
These are safe to ignore - the migration uses `IF NOT EXISTS` to make it idempotent (can be run multiple times safely).

### Connection errors
Check that:
- Neo4j is running
- Your `.env` file has the correct credentials
- The URI in `.env` matches your Neo4j instance

### Permission errors
Ensure your Neo4j user has admin privileges to create constraints and indexes.

## Schema Overview

After this migration, your graph will support these patterns:

```
(:Player)-[:PLAYS_FOR]->(:Team)<-[:PLAYS_HOME]-(:Fixture)
(:Player)-[:PLAYS_FOR]->(:Team)<-[:PLAYS_AWAY]-(:Fixture)
(:Fixture)-[:IN_MATCHDAY]->(:Matchday)-[:IN_SEASON]->(:Season)
```

This enables queries like:
- "Find the next fixture for a player after a specific date"
- "Get all fixtures in matchday X"
- "List upcoming home games for a team"
