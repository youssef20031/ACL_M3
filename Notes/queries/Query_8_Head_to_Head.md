# Query 8: Get Head-to-Head Results

## Description

Retrieves historical match results between two specific teams.

## Method Signature

`CypherQueries.get_head_to_head(team1: str, team2: str)`

## Parameters

- `team1`: Name of the first team.
- `team2`: Name of the second team.

## Example Natural Language Prompts

- "Arsenal vs Spurs head to head"
- "History of Liverpool vs Man City"

## Cypher Query

```cypher
MATCH (f:Fixture)-[:HOME_TEAM]->(ht:Team)
MATCH (f)-[:AWAY_TEAM]->(at:Team)
WHERE (ht.name = $team1 AND at.name = $team2) OR (ht.name = $team2 AND at.name = $team1)
MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
RETURN s.id AS season, gw.number AS gameweek,
       ht.name AS home_team, at.name AS away_team,
       f.home_score AS home_score, f.away_score AS away_score
ORDER BY s.id, gw.number
```
