"""Check fixtures in database"""
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from datetime import datetime, timezone

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Count total fixtures
results = conn.execute_query('MATCH (f:Fixture) RETURN count(f) as count')
print(f'Total fixtures in DB: {results[0]["count"]}')

# Count fixtures for 2026-27 season
results = conn.execute_query('MATCH (f:Fixture {season_id: "2026-27"}) RETURN count(f) as count')
print(f'Fixtures for 2026-27 season: {results[0]["count"]}')

# Get upcoming fixtures from current date
current_date = datetime.now(timezone.utc)
print(f'\nCurrent date: {current_date}')

results = conn.execute_query(
    'MATCH (f:Fixture) WHERE f.kickoff_time > $current_date RETURN count(f) as count',
    {"current_date": current_date}
)
print(f'Upcoming fixtures (after {current_date}): {results[0]["count"]}')

# Get first few fixtures
results = conn.execute_query(
    'MATCH (f:Fixture) RETURN f.kickoff_time as kickoff, f.id as id ORDER BY f.kickoff_time LIMIT 5'
)
print(f'\nFirst 5 fixtures:')
for r in results:
    print(f'  {r["id"]}: {r["kickoff"]}')

conn.close()
