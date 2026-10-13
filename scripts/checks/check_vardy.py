from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Search for Vardy
results = conn.execute_query('MATCH (p:Player) WHERE p.name CONTAINS "Vardy" RETURN p.name as name')
print('Players with "Vardy" in name:')
for r in results:
    print(f'  - {r["name"]}')

# Check if any Leicester players exist with team
results = conn.execute_query('''
    MATCH (p:Player)-[:PLAYS_FOR]->(t:Team {name: "Leicester"})
    RETURN p.name as name
    LIMIT 10
''')
print('\nLeicester players with PLAYS_FOR:')
for r in results:
    print(f'  - {r["name"]}')

conn.close()
