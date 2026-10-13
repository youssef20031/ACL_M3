"""Check which teams have fixtures"""
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Get teams with fixtures
results = conn.execute_query('''
    MATCH (t:Team)
    OPTIONAL MATCH (f:Fixture)-[:PLAYS_HOME|PLAYS_AWAY]->(t)
    WHERE f.season_id = "2026-27"
    WITH t.name as team, count(DISTINCT f) as fixture_count
    ORDER BY team
    RETURN team, fixture_count
''')

print('Teams and their 2026-27 fixture counts:')
print('=' * 60)

teams_with_fixtures = []
teams_without_fixtures = []

for r in results:
    team = r['team']
    count = r['fixture_count']
    
    if count > 0:
        teams_with_fixtures.append((team, count))
        print(f'✅ {team}: {count} fixtures')
    else:
        teams_without_fixtures.append(team)

print('\n' + '=' * 60)
print(f'\nTeams WITH fixtures: {len(teams_with_fixtures)}')
print(f'Teams WITHOUT fixtures: {len(teams_without_fixtures)}')

if teams_without_fixtures:
    print(f'\nTeams missing fixtures:')
    for team in teams_without_fixtures:
        print(f'  ❌ {team}')

conn.close()
