# Query 1: Get Top Goal Scorers

## Description

Retrieves the top goal scorers for a specific season or across all seasons.

## Method Signature

`CypherQueries.get_top_scorers_by_season(season: str = None, limit: int = 10)`

## Parameters

- `season` (optional): Season ID (e.g., '2022-23'). If None, returns aggregation from all seasons.
- `limit`: Number of results to return (default: 10).

## Example Natural Language Prompts

- "Who are the top goal scorers in 2022-23?"
- "Top scorers all time"

## Cypher Query (Season)

```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, SUM(r.goals_scored) AS total_goals, SUM(r.total_points) AS total_points
WHERE total_goals > 0
RETURN p.name AS player_name, total_goals, total_points
ORDER BY total_goals DESC
LIMIT $limit
```
