# Query 6: Get Team Top Performers

## Description

Retrieves top performing players from a specific team in a season.

## Method Signature

`CypherQueries.get_team_top_performers(team_name: str, season: str = None, limit: int = 5)`

## Parameters

- `team_name`: Name of the team (e.g., "Arsenal").
- `season` (optional): Season ID.
- `limit`: Number of results (default: 5).

## Example Natural Language Prompts

- "Arsenal best players 2022-23"
- "Who scored the most points for Liverpool?"
- "Spurs team analysis"

## Cypher Query

```cypher
MATCH (t:Team {name: $team_name})
MATCH (f:Fixture)-[:HOME_TEAM|AWAY_TEAM]->(t)
MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
MATCH (p:Player)-[r:PLAYED_IN]->(f)
WITH p, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals, SUM(r.assists) AS assists
RETURN p.name AS player_name, total_points, goals, assists
ORDER BY total_points DESC
LIMIT $limit
```
