"""Check relationship types"""
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

results = conn.execute_query('CALL db.relationshipTypes()')
print('Relationship types in database:')
for r in results:
    print(f"  - {r['relationshipType']}")

# Check if there are players
results = conn.execute_query('MATCH (p:Player) RETURN count(p) as count')
print(f'\nTotal players: {results[0]["count"]}')

# Get sample players
results = conn.execute_query('MATCH (p:Player) RETURN p.name as name LIMIT 10')
print('\nSample players:')
for r in results:
    print(f'  - {r["name"]}')

conn.close()
