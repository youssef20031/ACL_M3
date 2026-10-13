"""Find how players are mapped to teams"""
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Check if there's game data with team info
res = conn.execute_query('''
    MATCH (p:Player {name: "Mohamed Salah"})-[:PLAYED_IN]->(g:Game)
    RETURN g.team as team, g.season as season, g.date as date
    ORDER BY g.date DESC
    LIMIT 5
''')

print('Mohamed Salah recent games:')
for r in res:
    print(f'  Team: {r["team"]}, Season: {r["season"]}, Date: {r["date"]}')

print('\n' + '=' * 60)

# Get his most recent team
res = conn.execute_query('''
    MATCH (p:Player {name: "Mohamed Salah"})-[:PLAYED_IN]->(g:Game)
    RETURN g.team as team
    ORDER BY g.date DESC
    LIMIT 1
''')

if res:
    most_recent_team = res[0]["team"]
    print(f'\nMohamed Salah most recent team: {most_recent_team}')
    
    # Check if this team has fixtures
    res2 = conn.execute_query('''
        MATCH (f:Fixture)
        WHERE (f)-[:PLAYS_HOME]->(:Team {name: $team}) OR (f)-[:PLAYS_AWAY]->(:Team {name: $team})
        RETURN count(f) as fixture_count
    ''', {"team": most_recent_team})
    
    print(f'Fixtures for {most_recent_team}: {res2[0]["fixture_count"]}')

conn.close()
