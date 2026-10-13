# Trivia Query: Highest Single Gameweek Score

## Description

Finds the single highest score achieved by any player in a single gameweek for a given season.

## Method Signature

`CypherQueries.get_highest_single_gameweek_score(season: str)`

## Parameters

- `season`: Season ID.

## Example Natural Language Prompts

- "Who had the highest gameweek score in 2022-23?"
- "Best single game performance"

## Cypher Query

```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WHERE r.total_points > 0
RETURN p.name AS player_name, gw.number AS gameweek, r.total_points AS points,
       r.goals_scored AS goals, r.assists AS assists, r.bonus AS bonus
ORDER BY r.total_points DESC
LIMIT 1
```
