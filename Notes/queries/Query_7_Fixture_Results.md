# Query 7: Get Fixture Results

## Description

Retrieves all match results for a specific team in a season.

## Method Signature

`CypherQueries.get_fixture_results(team_name: str, season: str = None)`

## Parameters

- `team_name`: Name of the team.
- `season` (optional): Season ID.

## Example Natural Language Prompts

- "Liverpool fixture results 2022-23"
- "Chelsea games last season"

## Cypher Query

```cypher
MATCH (t:Team {name: $team_name})
MATCH (f:Fixture)-[:HOME_TEAM]->(ht:Team)
MATCH (f)-[:AWAY_TEAM]->(at:Team)
WHERE ht.name = $team_name OR at.name = $team_name
MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
RETURN s.id AS season, gw.number AS gameweek, ht.name AS home_team, at.name AS away_team,
       f.home_score AS home_score, f.away_score AS away_score,
       f.kickoff_time AS kickoff_time
ORDER BY gw.number
```
