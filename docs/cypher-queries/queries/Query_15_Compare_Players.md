# Query 15: Compare Players

## Description

Side-by-side comparison of two players stats' for a season.

## Method Signature

`CypherQueries.compare_players(player1: str, player2: str, season: str = None)`

## Parameters

- `player1`: First player's name.
- `player2`: Second player's name.
- `season` (optional): Season ID.

## Example Natural Language Prompts

- "Compare Mohamed Salah and Harry Kane"
- "Stats for Saka vs Odegaard"

## Cypher Query

```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WHERE p.name IN [$player1, $player2]
MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
WITH p, pos,
     SUM(r.total_points) AS total_points,
     SUM(r.goals_scored) AS goals,
     SUM(r.assists) AS assists,
# ... (aggregated metrics)
RETURN p.name AS player_name, pos.code AS position, total_points, goals, assists...
ORDER BY total_points DESC
```
