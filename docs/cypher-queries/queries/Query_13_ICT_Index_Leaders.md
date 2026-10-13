# Query 13: Get ICT Index Leaders

## Description

Retrieves players with the highest average ICT (Influence, Creativity, Threat) Index.

## Method Signature

`CypherQueries.get_ict_index_leaders(season: str, limit: int = 10)`

## Parameters

- `season`: Season ID.
- `limit`: Number of results.

## Example Natural Language Prompts

- "Highest ICT index 2022-23"
- "Best underlying stats"

## Cypher Query

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WHERE r.minutes > 0
WITH p, pos, AVG(r.ict_index) AS avg_ict, AVG(r.influence) AS avg_influence,
     AVG(r.creativity) AS avg_creativity, AVG(r.threat) AS avg_threat,
     SUM(r.total_points) AS total_points, COUNT(f) AS games
WHERE games >= 10
RETURN p.name AS player_name, pos.code AS position,
       round(avg_ict, 2) AS avg_ict_index,
       round(avg_influence, 2) AS avg_influence,
       round(avg_creativity, 2) AS avg_creativity,
       round(avg_threat, 2) AS avg_threat,
       total_points, games
ORDER BY avg_ict DESC
LIMIT $limit
```
