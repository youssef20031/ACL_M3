# Query 19: Get All Teams

## Description

Retrieves a list of all football teams in the database.

## Method Signature

`CypherQueries.get_all_teams()`

## Parameters

None

## Example Natural Language Prompts

- "List all teams"
- "Which teams are in the league?"

## Cypher Query

```cypher
MATCH (t:Team)
RETURN t.name AS team_name
ORDER BY t.name
```
