# Query 16: Get Player Form History

## Description

Retrieves a player's form and points progression throughout the gameweeks of a season.

## Method Signature

`CypherQueries.get_player_form_history(player_name: str, season: str = None)`

## Parameters

- `player_name`: Full name of the player.
- `season` (optional): Season ID.

## Example Natural Language Prompts

- "Show me Salah's form history"
- "How has Haaland been playing recently?"

## Cypher Query

```cypher
MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
RETURN gw.number AS gameweek, r.total_points AS points, r.form AS form,
       r.goals_scored AS goals, r.assists AS assists, r.minutes AS minutes, s.id AS season
ORDER BY gw.number
```
