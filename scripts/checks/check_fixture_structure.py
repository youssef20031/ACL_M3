"""Check fixture structure"""
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Check a sample fixture
results = conn.execute_query('''
    MATCH (f:Fixture {season_id: "2026-27"})
    OPTIONAL MATCH (f)-[:PLAYS_HOME]->(ht:Team)
    OPTIONAL MATCH (f)-[:PLAYS_AWAY]->(at:Team)
    RETURN f.id as id, f.kickoff_time as kickoff, f.matchday as matchday,
           ht.name as home_team, at.name as away_team
    LIMIT 5
''')

print('Sample 2026-27 fixtures:')
print('=' * 80)
for r in results:
    print(f'{r["id"]}:')
    print(f'  Home: {r["home_team"]}')
    print(f'  Away: {r["away_team"]}')
    print(f'  Date: {r["kickoff"]}')
    print(f'  Matchday: {r["matchday"]}')
    print()

conn.close()
