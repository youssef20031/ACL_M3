# Query 20: Season Summary

## Description

Provides aggregate statistics for a given season including total goals, total FPL points, and number of fixtures.

## Method Signature

`CypherQueries.get_season_summary(season: str)`

## Parameters

- `season`: Season ID (e.g., '2022-23').

## Example Natural Language Prompts

- "2022-23 season summary"
- "Overview of last season"

## Cypher Query

```cypher
MATCH (s:Season {id: $season})
MATCH (gw:Gameweek)-[:IN_SEASON]->(s)
WITH s, COUNT(DISTINCT gw) AS total_gameweeks
MATCH (f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s)
WITH s, total_gameweeks, COUNT(DISTINCT f) AS total_fixtures,
     SUM(f.home_score + f.away_score) AS total_goals
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(:Gameweek)-[:IN_SEASON]->(s)
WITH s, total_gameweeks, total_fixtures, total_goals,
     SUM(r.total_points) AS total_fpl_points, COUNT(DISTINCT p) AS total_players
RETURN s.id AS season, total_gameweeks, total_fixtures, total_goals,
       total_fpl_points, total_players,
       round(total_goals * 1.0 / total_fixtures, 2) AS goals_per_game
```
