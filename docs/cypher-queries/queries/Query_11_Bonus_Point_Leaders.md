# Query 11: Get Bonus Point Leaders

## Description

Retrieves players who have accumulated the most bonus points in a season.

## Method Signature

`CypherQueries.get_bonus_point_leaders(season: str, limit: int = 10)`

## Parameters

- `season`: Season ID.
- `limit`: Number of results.

## Example Natural Language Prompts

- "Who got the most bonus points?"
- "Bonus point leaders 2022-23"

## Cypher Query

```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, SUM(r.bonus) AS total_bonus, SUM(r.bps) AS total_bps, COUNT(f) AS games
WHERE total_bonus > 0
RETURN p.name AS player_name, total_bonus, total_bps, games,
       round(total_bonus * 1.0 / games, 2) AS bonus_per_game
ORDER BY total_bonus DESC
LIMIT $limit
```
