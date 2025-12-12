# Query 2: Get Top Assist Providers

## Description

Retrieves the top assist providers for a specific season or across all seasons.

## Method Signature

`CypherQueries.get_top_assisters_by_season(season: str = None, limit: int = 10)`

## Parameters

- `season` (optional): Season ID (e.g., '2022-23'). If None, returns aggregation from all seasons.
- `limit`: Number of results to return (default: 10).

## Example Natural Language Prompts

- "Who has the most assists in 2021-22?"
- "Top assisters"

## Cypher Query (Season)

```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, SUM(r.assists) AS total_assists, SUM(r.total_points) AS total_points
WHERE total_assists > 0
RETURN p.name AS player_name, total_assists, total_points
ORDER BY total_assists DESC
LIMIT $limit
```
