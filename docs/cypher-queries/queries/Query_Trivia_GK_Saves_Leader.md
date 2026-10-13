# Trivia Query: Goalkeeper Saves Leader

## Description

Retrieves the goalkeeper with the most saves in a season.

## Method Signature

`CypherQueries.get_goalkeeper_saves_leader(season: str)`

## Parameters

- `season`: Season ID.

## Example Natural Language Prompts

- "Who made the most saves?"
- "Top goalkeeper for saves 2022-23"

## Cypher Query

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: 'GK'})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, SUM(r.saves) AS total_saves
WHERE total_saves > 0
RETURN p.name AS player_name, total_saves
ORDER BY total_saves DESC
LIMIT 5
```
