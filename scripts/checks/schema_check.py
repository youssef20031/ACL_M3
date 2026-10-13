import sys
sys.path.insert(0, 'C:/ACL2/FPL/ACL_M3')
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Get PLAYED_IN relationship properties
r1 = conn.execute_query("MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture) RETURN properties(r) as props LIMIT 1")
if r1:
    print("PLAYED_IN props:", list(r1[0]["props"].keys()))

# Check fixture properties  
r2 = conn.execute_query("MATCH (f:Fixture) RETURN properties(f) as props LIMIT 1")
if r2:
    print("Fixture props:", list(r2[0]["props"].keys()))

# Check player properties
r3 = conn.execute_query("MATCH (p:Player) RETURN properties(p) as props LIMIT 1")
if r3:
    print("Player props:", list(r3[0]["props"].keys()))

# Check for Salah
r4 = conn.execute_query("MATCH (p:Player) WHERE p.name CONTAINS 'Salah' RETURN p.name LIMIT 5")
print("Salah players:", r4)

# Get sample query with position
r5 = conn.execute_query("""
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
RETURN p.name as name, pos.code as position, properties(r) as stats, s.id as season
LIMIT 2
""")
for row in r5:
    print("Sample row keys:", list(row.keys()))
    print("Stats keys:", list(row["stats"].keys()) if row.get("stats") else "no stats")
    break
