# Query 4: Get Player Season Stats

## Description

Retrieves comprehensive statistics for a specific player in a specific season, including points, goals, assists, ICT index, and more.

## Method Signature

`CypherQueries.get_player_season_stats(player_name: str, season: str)`

## Parameters

- `player_name`: Full name of the player.
- `season`: Season ID (e.g., '2022-23').

## Example Natural Language Prompts

- "Mohamed Salah 2022-23 stats"
- "How did Haaland perform in 22/23?"

## Cypher Query

```cypher
MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
WITH p, pos,
     SUM(r.total_points) AS total_points,
     SUM(r.goals_scored) AS goals,
     SUM(r.assists) AS assists,
     SUM(r.clean_sheets) AS clean_sheets,
     SUM(r.bonus) AS bonus,
     SUM(r.minutes) AS minutes,
     AVG(r.ict_index) AS avg_ict,
     AVG(r.influence) AS avg_influence,
     AVG(r.creativity) AS avg_creativity,
     AVG(r.threat) AS avg_threat,
     MAX(r.value) AS max_value,
     MAX(r.selected) AS max_selected,
     COUNT(f) AS games
RETURN p.name AS player_name, pos.code AS position,
       total_points, goals, assists, clean_sheets, bonus,
       minutes, avg_ict, avg_influence, avg_creativity, avg_threat,
       max_value, max_selected, games
```
