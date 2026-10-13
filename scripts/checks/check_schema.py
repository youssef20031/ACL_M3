"""Check database schema"""
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Get all node labels
res = conn.execute_query('CALL db.labels()')
print('Node labels:')
for r in res:
    print(f'  - {r["label"]}')

print('\n' + '=' * 60)

# Check GameweekPerformance nodes
res = conn.execute_query('''
    MATCH (p:Player {name: "Mohamed Salah"})-[:PLAYED_IN]->(gw:GameweekPerformance)
    RETURN gw.team as team, gw.gameweek as gw, gw.season as season
    ORDER BY gw.season DESC, gw.gameweek DESC
    LIMIT 5
''')

print('\nMohamed Salah recent gameweeks:')
for r in res:
    print(f'  GW{r["gw"]} ({r["season"]}): Team = {r["team"]}')

conn.close()
