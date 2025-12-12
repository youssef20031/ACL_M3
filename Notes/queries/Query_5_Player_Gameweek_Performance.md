# Query 5: Get Player Gameweek Performance

## Description

Retrieves a player's performance details for a specific gameweek (points, opposition, minutes, etc.).

## Method Signature

`CypherQueries.get_player_gameweek_performance(player_name: str, season: str, gameweek: int)`

## Parameters

- `player_name`: Full name of the player.
- `season`: Season ID.
- `gameweek`: Gameweek number (1-38).

## Example Natural Language Prompts

- "How did Salah do in gameweek 1?"
- "Haaland points gameweek 5 2022-23"

## Cypher Query

```cypher
MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek {number: $gameweek})
MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
MATCH (f)-[:HOME_TEAM]->(ht:Team)
MATCH (f)-[:AWAY_TEAM]->(at:Team)
RETURN p.name AS player_name, gw.number AS gameweek,
       r.total_points AS points, r.goals_scored AS goals, r.assists AS assists,
       r.minutes AS minutes, r.bonus AS bonus, r.bps AS bps,
       r.ict_index AS ict_index, r.clean_sheets AS clean_sheets,
       ht.name AS home_team, at.name AS away_team,
       f.home_score AS home_score, f.away_score AS away_score
```
