# Query 9: Get Best Value Players

## Description

Identifies players offering the best value for money (points per million cost).

## Method Signature

`CypherQueries.get_best_value_players(season: str, position: Optional[str] = None, limit: int = 10)`

## Parameters

- `season`: Season ID.
- `position` (optional): Position filter code.
- `limit`: Number of results (default: 10).

## Example Natural Language Prompts

- "Best value midfielders in 2022-23"
- "Who are the best budget picks?"

## Cypher Query

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: $position})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, pos, SUM(r.total_points) AS total_points, AVG(r.value) AS avg_value
WHERE avg_value > 0
WITH p, pos, total_points, avg_value, (total_points * 10.0 / avg_value) AS points_per_million
RETURN p.name AS player_name, pos.code AS position, total_points,
       avg_value / 10.0 AS value_millions, round(points_per_million, 2) AS points_per_million
ORDER BY points_per_million DESC
LIMIT $limit
```
