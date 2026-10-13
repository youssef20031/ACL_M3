"""Check what teams and players are in the database"""
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Get all teams
results = conn.execute_query('MATCH (t:Team) RETURN t.name as name ORDER BY name')
print(f'Teams in database ({len(results)}):')
for r in results:
    print(f'  - {r["name"]}')

print('\n' + '=' * 60)

# Get sample players with their teams
results = conn.execute_query('''
    MATCH (p:Player)-[:PLAYS_FOR]->(t:Team)
    RETURN p.name as player, t.name as team
    ORDER BY p.name
    LIMIT 20
''')
print(f'\nSample players with teams:')
for r in results:
    print(f'  {r["player"]} ({r["team"]})')

conn.close()
