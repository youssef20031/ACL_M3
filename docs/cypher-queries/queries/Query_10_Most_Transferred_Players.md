# Query 10: Get Most Transferred Players

## Description

Finds players who were transferred in or out the most during a specific gameweek.

## Method Signature

`CypherQueries.get_most_transferred_players(season: str, gameweek: int, direction: str = "in", limit: int = 10)`

## Parameters

- `season`: Season ID.
- `gameweek`: Gameweek number.
- `direction`: "in" or "out" (default: "in").
- `limit`: Number of results.

## Example Natural Language Prompts

- "Most transferred in gameweek 5 2022-23"
- "Who are people selling in GW10?"

## Cypher Query

```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek {number: $gameweek})
MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
RETURN p.name AS player_name, r.{transfer_field} AS transfers, r.total_points AS points
ORDER BY r.{transfer_field} DESC
LIMIT $limit
```
