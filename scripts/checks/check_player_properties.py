"""Check player properties"""
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Get a sample player with properties
res = conn.execute_query('''
    MATCH (p:Player)
    WHERE p.name CONTAINS "Salah" OR p.name CONTAINS "Saka"
    RETURN p.name as name, properties(p) as props
    LIMIT 5
''')

print('Sample players and their properties:')
for r in res:
    print(f'\nPlayer: {r["name"]}')
    print(f'Properties: {list(r["props"].keys())}')
    if 'team' in r["props"]:
        print(f'Team property: {r["props"]["team"]}')

conn.close()
