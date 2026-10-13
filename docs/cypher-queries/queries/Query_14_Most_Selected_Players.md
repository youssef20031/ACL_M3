# Query 14: Get Most Selected Players

## Description

Retrieves players with the highest ownership percentage in a specific gameweek.

## Method Signature

`CypherQueries.get_most_selected_players(season: str, gameweek: int, limit: int = 10)`

## Parameters

- `season`: Season ID.
- `gameweek`: Gameweek number.
- `limit`: Number of results.

## Example Natural Language Prompts

- "Most selected players gameweek 1 2022-23"
- "Who is in everyone's team?"

## Cypher Query

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek {number: $gameweek})
MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
RETURN p.name AS player_name, pos.code AS position, r.selected AS selected,
       r.value / 10.0 AS value_millions, r.total_points AS points
ORDER BY r.selected DESC
LIMIT $limit
```
