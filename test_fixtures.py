from graph.queries import CypherQueries
from graph.connection import Neo4jConnection

# Connect (initialization automatically connects)
conn = Neo4jConnection('bolt://localhost:7687', 'neo4j', 'pass_1234')

# Query Arsenal fixtures
q = CypherQueries()
query_str, params = q.get_fixture_results('Arsenal')
results = conn.execute_query(query_str, params)

print(f'Total fixtures: {len(results)}')

# Group by season
seasons = {}
for r in results:
    season = r.get('season')
    if season not in seasons:
        seasons[season] = []
    seasons[season].append(r)

print(f'\nSeasons: {sorted(seasons.keys())}')
print(f'\nSample fixture data:')
print(results[0])
for s in sorted(seasons.keys()):
    fixtures = seasons[s]
    wins = sum(1 for f in fixtures if f.get('result') == 'W')
    draws = sum(1 for f in fixtures if f.get('result') == 'D')
    losses = sum(1 for f in fixtures if f.get('result') == 'L')
    print(f'{s}: {len(fixtures)} fixtures (W:{wins} D:{draws} L:{losses})')

conn.close()
