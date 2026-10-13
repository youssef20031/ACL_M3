# Query 12: Get Clean Sheet Leaders

## Description

Retrieves Defenders and Goalkeepers with the most clean sheets.

## Method Signature

`CypherQueries.get_clean_sheet_leaders(season: str, limit: int = 10)`

## Parameters

- `season`: Season ID.
- `limit`: Number of results.

## Example Natural Language Prompts

- "Most clean sheets 2022-23"
- "Which defender has the most clean sheets?"

## Cypher Query

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
WHERE pos.code IN ['GK', 'DEF']
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, pos, SUM(r.clean_sheets) AS total_clean_sheets, SUM(r.total_points) AS total_points
WHERE total_clean_sheets > 0
RETURN p.name AS player_name, pos.code AS position, total_clean_sheets, total_points
ORDER BY total_clean_sheets DESC
LIMIT $limit
```
