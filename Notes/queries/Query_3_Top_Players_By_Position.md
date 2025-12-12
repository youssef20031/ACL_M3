# Query 3: Get Top Players by Position

## Description

Retrieves top performing players filtered by position and optionally sorted by specific metrics like goals, assists, or bonus points.

## Method Signature

`CypherQueries.get_top_points_by_position(position: str = None, season: str = None, sort_by: str = "total_points", limit: int = 10)`

## Parameters

- `position` (optional): Position code ('GK', 'DEF', 'MID', 'FWD'). If None, returns top 10 overall.
- `season` (optional): Season ID (e.g., '2022-23').
- `sort_by` (default: 'total_points'): Metric to sort by. Options: 'total_points', 'goals', 'assists', 'bonus'.
- `limit`: Number of results (default: 10).

## Example Natural Language Prompts

- "Top defenders by points in 2022-23"
- "Who are the top goal scoring defenders?"
- "Best midfielders"

## Cypher Query

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: $position})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, pos, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals,
     SUM(r.assists) AS assists, SUM(r.bonus) AS bonus
RETURN p.name AS player_name, pos.code AS position, total_points, goals, assists, bonus
ORDER BY {sort_field} DESC
LIMIT $limit
```
