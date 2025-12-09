# FPL Knowledge Graph - Cypher Query Reference

This document contains all Cypher queries used in the FPL Graph-RAG system. The queries are organized by category and include parameters and descriptions.

---

## Table of Contents

1. [Player Statistics Queries](#player-statistics-queries)
2. [Team Analysis Queries](#team-analysis-queries)
3. [Value & Transfer Analysis Queries](#value--transfer-analysis-queries)
4. [Performance Metrics Queries](#performance-metrics-queries)
5. [Comparison Queries](#comparison-queries)
6. [Search Queries](#search-queries)
7. [Trivia-Specific Queries](#trivia-specific-queries)

---

## Player Statistics Queries

### Query 1: Get Top Goal Scorers

**Method:** `get_top_scorers_by_season(season, limit)`

**Parameters:**
- `season` (optional): Season ID (e.g., '2022-23') - if None, returns from all seasons
- `limit`: Number of results to return (default: 10)

**With Season:**
```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, SUM(r.goals_scored) AS total_goals, SUM(r.total_points) AS total_points
WHERE total_goals > 0
RETURN p.name AS player_name, total_goals, total_points
ORDER BY total_goals DESC
LIMIT $limit
```

**All Seasons:**
```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
WITH p, SUM(r.goals_scored) AS total_goals, SUM(r.total_points) AS total_points
WHERE total_goals > 0
RETURN p.name AS player_name, total_goals, total_points
ORDER BY total_goals DESC
LIMIT $limit
```

---

### Query 2: Get Top Assist Providers

**Method:** `get_top_assisters_by_season(season, limit)`

**Parameters:**
- `season` (optional): Season ID - if None, returns from all seasons
- `limit`: Number of results (default: 10)

**With Season:**
```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, SUM(r.assists) AS total_assists, SUM(r.total_points) AS total_points
WHERE total_assists > 0
RETURN p.name AS player_name, total_assists, total_points
ORDER BY total_assists DESC
LIMIT $limit
```

**All Seasons:**
```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
WITH p, SUM(r.assists) AS total_assists, SUM(r.total_points) AS total_points
WHERE total_assists > 0
RETURN p.name AS player_name, total_assists, total_points
ORDER BY total_assists DESC
LIMIT $limit
```

---

### Query 3: Get Top Players by Position

**Method:** `get_top_points_by_position(position, season, limit)`

**Parameters:**
- `position` (optional): Position code (GK, DEF, MID, FWD) - if None, returns top overall
- `season` (optional): Season ID - if None, returns from all seasons
- `limit`: Number of results (default: 10)

**With Position:**
```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: $position})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, pos, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals, 
     SUM(r.assists) AS assists, SUM(r.bonus) AS bonus
RETURN p.name AS player_name, pos.code AS position, total_points, goals, assists, bonus
ORDER BY total_points DESC
LIMIT $limit
```

**All Positions:**
```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, pos, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals, 
     SUM(r.assists) AS assists, SUM(r.bonus) AS bonus
RETURN pos.code AS position, p.name AS player_name, total_points, goals, assists, bonus
ORDER BY total_points DESC
LIMIT $limit
```

---

### Query 3b: Get Top Players for ALL Positions (UNION)

**Method:** `get_top_players_all_positions(season, limit_per_position)`

**Parameters:**
- `season` (optional): Season ID - if None, returns from all seasons
- `limit_per_position`: Number of players per position (default: 5)

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: 'GK'})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
MATCH (gw)-[:IN_SEASON]->(s:Season)
WITH 'GK' AS position, p.name AS player_name, 
     SUM(r.total_points) AS total_points, 
     SUM(r.goals_scored) AS goals, 
     SUM(r.assists) AS assists, 
     SUM(r.bonus) AS bonus
ORDER BY total_points DESC
LIMIT $limit_per_position
RETURN position, player_name, total_points, goals, assists, bonus

UNION ALL

MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: 'DEF'})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
MATCH (gw)-[:IN_SEASON]->(s:Season)
WITH 'DEF' AS position, p.name AS player_name, 
     SUM(r.total_points) AS total_points, 
     SUM(r.goals_scored) AS goals, 
     SUM(r.assists) AS assists, 
     SUM(r.bonus) AS bonus
ORDER BY total_points DESC
LIMIT $limit_per_position
RETURN position, player_name, total_points, goals, assists, bonus

UNION ALL

MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: 'MID'})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
MATCH (gw)-[:IN_SEASON]->(s:Season)
WITH 'MID' AS position, p.name AS player_name, 
     SUM(r.total_points) AS total_points, 
     SUM(r.goals_scored) AS goals, 
     SUM(r.assists) AS assists, 
     SUM(r.bonus) AS bonus
ORDER BY total_points DESC
LIMIT $limit_per_position
RETURN position, player_name, total_points, goals, assists, bonus

UNION ALL

MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: 'FWD'})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
MATCH (gw)-[:IN_SEASON]->(s:Season)
WITH 'FWD' AS position, p.name AS player_name, 
     SUM(r.total_points) AS total_points, 
     SUM(r.goals_scored) AS goals, 
     SUM(r.assists) AS assists, 
     SUM(r.bonus) AS bonus
ORDER BY total_points DESC
LIMIT $limit_per_position
RETURN position, player_name, total_points, goals, assists, bonus
```

---

### Query 4: Get Player Season Stats

**Method:** `get_player_season_stats(player_name, season)`

**Parameters:**
- `player_name`: Full player name
- `season`: Season ID (e.g., '2022-23')

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

---

### Query 4b: Get Player All Seasons Stats

**Method:** `get_player_all_seasons_stats(player_name)`

**Parameters:**
- `player_name`: Full player name

```cypher
MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
WITH p, pos, s,
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
RETURN p.name AS player_name, pos.code AS position, s.id AS season,
       total_points, goals, assists, clean_sheets, bonus,
       minutes, round(avg_ict, 2) AS avg_ict, round(avg_influence, 2) AS avg_influence, 
       round(avg_creativity, 2) AS avg_creativity, round(avg_threat, 2) AS avg_threat,
       max_value, max_selected, games
ORDER BY s.id
```

---

### Query 5: Get Player Gameweek Performance

**Method:** `get_player_gameweek_performance(player_name, season, gameweek)`

**Parameters:**
- `player_name`: Full player name
- `season`: Season ID
- `gameweek`: Gameweek number (1-38)

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

---

## Team Analysis Queries

### Query 6: Get Team Top Performers

**Method:** `get_team_top_performers(team_name, season, limit)`

**Parameters:**
- `team_name`: Team name (e.g., 'Arsenal')
- `season` (optional): Season ID - if None, returns from all seasons
- `limit`: Number of results (default: 5)

**With Season:**
```cypher
MATCH (t:Team {name: $team_name})
MATCH (f:Fixture)-[:HOME_TEAM|AWAY_TEAM]->(t)
MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
MATCH (p:Player)-[r:PLAYED_IN]->(f)
WITH p, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals, SUM(r.assists) AS assists
RETURN p.name AS player_name, total_points, goals, assists
ORDER BY total_points DESC
LIMIT $limit
```

**All Seasons:**
```cypher
MATCH (t:Team {name: $team_name})
MATCH (f:Fixture)-[:HOME_TEAM|AWAY_TEAM]->(t)
MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
MATCH (p:Player)-[r:PLAYED_IN]->(f)
WITH p, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals, SUM(r.assists) AS assists
RETURN p.name AS player_name, total_points, goals, assists
ORDER BY total_points DESC
LIMIT $limit
```

---

### Query 7: Get Fixture Results

**Method:** `get_fixture_results(team_name, season)`

**Parameters:**
- `team_name`: Team name
- `season` (optional): Season ID - if None, returns from all seasons

**With Season:**
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

**All Seasons:**
```cypher
MATCH (t:Team {name: $team_name})
MATCH (f:Fixture)-[:HOME_TEAM]->(ht:Team)
MATCH (f)-[:AWAY_TEAM]->(at:Team)
WHERE ht.name = $team_name OR at.name = $team_name
MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
RETURN s.id AS season, gw.number AS gameweek, ht.name AS home_team, at.name AS away_team,
       f.home_score AS home_score, f.away_score AS away_score,
       f.kickoff_time AS kickoff_time
ORDER BY s.id, gw.number
```

---

### Query 8: Get Head-to-Head Results

**Method:** `get_head_to_head(team1, team2)`

**Parameters:**
- `team1`: First team name
- `team2`: Second team name

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

---

## Value & Transfer Analysis Queries

### Query 9: Get Best Value Players

**Method:** `get_best_value_players(season, position, limit)`

**Parameters:**
- `season`: Season ID (required)
- `position` (optional): Position code (GK, DEF, MID, FWD)
- `limit`: Number of results (default: 10)

**With Position:**
```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: $position})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, pos, SUM(r.total_points) AS total_points, AVG(r.value) AS avg_value
WHERE avg_value > 0
WITH p, pos, total_points, avg_value, (total_points * 10.0 / avg_value) AS points_per_million
RETURN p.name AS player_name, pos.code AS position, total_points, 
       avg_value / 10.0 AS value_millions, round(points_per_million, 2) AS points_per_million
ORDER BY points_per_million DESC
LIMIT $limit
```

**All Positions:**
```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, pos, SUM(r.total_points) AS total_points, AVG(r.value) AS avg_value
WHERE avg_value > 0
WITH p, pos, total_points, avg_value, (total_points * 10.0 / avg_value) AS points_per_million
RETURN p.name AS player_name, pos.code AS position, total_points,
       avg_value / 10.0 AS value_millions, round(points_per_million, 2) AS points_per_million
ORDER BY points_per_million DESC
LIMIT $limit
```

---

### Query 10: Get Most Transferred Players

**Method:** `get_most_transferred_players(season, gameweek, direction, limit)`

**Parameters:**
- `season`: Season ID (required)
- `gameweek`: Gameweek number (required)
- `direction`: 'in' for transfers in, 'out' for transfers out (default: 'in')
- `limit`: Number of results (default: 10)

**Transfers In:**
```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek {number: $gameweek})
MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
RETURN p.name AS player_name, r.transfers_in AS transfers, r.total_points AS points
ORDER BY r.transfers_in DESC
LIMIT $limit
```

**Transfers Out:**
```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek {number: $gameweek})
MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
RETURN p.name AS player_name, r.transfers_out AS transfers, r.total_points AS points
ORDER BY r.transfers_out DESC
LIMIT $limit
```

---

## Performance Metrics Queries

### Query 11: Get Bonus Point Leaders

**Method:** `get_bonus_point_leaders(season, limit)`

**Parameters:**
- `season`: Season ID (required)
- `limit`: Number of results (default: 10)

```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, SUM(r.bonus) AS total_bonus, SUM(r.bps) AS total_bps, COUNT(f) AS games
WHERE total_bonus > 0
RETURN p.name AS player_name, total_bonus, total_bps, games,
       round(total_bonus * 1.0 / games, 2) AS bonus_per_game
ORDER BY total_bonus DESC
LIMIT $limit
```

---

### Query 12: Get Clean Sheet Leaders

**Method:** `get_clean_sheet_leaders(season, limit)`

**Parameters:**
- `season`: Season ID (required)
- `limit`: Number of results (default: 10)

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
WHERE pos.code IN ['GK', 'DEF']
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, pos, SUM(r.clean_sheets) AS total_clean_sheets, SUM(r.total_points) AS total_points
WHERE total_clean_sheets > 0
RETURN p.name AS player_name, pos.code AS position, total_clean_sheets, total_points
ORDER BY total_clean_sheets DESC
LIMIT $limit
```

---

### Query 13: Get ICT Index Leaders

**Method:** `get_ict_index_leaders(season, limit)`

**Parameters:**
- `season`: Season ID (required)
- `limit`: Number of results (default: 10)

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

---

### Query 14: Get Most Selected Players

**Method:** `get_most_selected_players(season, gameweek, limit)`

**Parameters:**
- `season`: Season ID (required)
- `gameweek`: Gameweek number (required)
- `limit`: Number of results (default: 10)

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek {number: $gameweek})
MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
RETURN p.name AS player_name, pos.code AS position, r.selected AS selected,
       r.value / 10.0 AS value_millions, r.total_points AS points
ORDER BY r.selected DESC
LIMIT $limit
```

---

## Comparison Queries

### Query 15: Compare Players

**Method:** `compare_players(player1, player2, season)`

**Parameters:**
- `player1`: First player name
- `player2`: Second player name
- `season` (optional): Season ID - if None, compares across all seasons

**With Season:**
```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WHERE p.name IN [$player1, $player2]
MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
WITH p, pos,
     SUM(r.total_points) AS total_points,
     SUM(r.goals_scored) AS goals,
     SUM(r.assists) AS assists,
     SUM(r.clean_sheets) AS clean_sheets,
     SUM(r.bonus) AS bonus,
     SUM(r.minutes) AS minutes,
     AVG(r.ict_index) AS avg_ict,
     AVG(r.value) AS avg_value,
     COUNT(f) AS games
RETURN p.name AS player_name, pos.code AS position,
       total_points, goals, assists, clean_sheets, bonus, minutes,
       round(avg_ict, 2) AS avg_ict_index, 
       round(avg_value / 10.0, 2) AS avg_value_millions,
       games
ORDER BY total_points DESC
```

**All Seasons:**
```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
WHERE p.name IN [$player1, $player2]
MATCH (p)-[:PLAYS_POSITION]->(pos:Position)
WITH p, pos,
     SUM(r.total_points) AS total_points,
     SUM(r.goals_scored) AS goals,
     SUM(r.assists) AS assists,
     SUM(r.clean_sheets) AS clean_sheets,
     SUM(r.bonus) AS bonus,
     SUM(r.minutes) AS minutes,
     AVG(r.ict_index) AS avg_ict,
     AVG(r.value) AS avg_value,
     COUNT(f) AS games
RETURN p.name AS player_name, pos.code AS position,
       total_points, goals, assists, clean_sheets, bonus, minutes,
       round(avg_ict, 2) AS avg_ict_index, 
       round(avg_value / 10.0, 2) AS avg_value_millions,
       games
ORDER BY total_points DESC
```

---

### Query 16: Get Player Form History

**Method:** `get_player_form_history(player_name, season)`

**Parameters:**
- `player_name`: Full player name
- `season` (optional): Season ID - if None, returns from all seasons

**With Season:**
```cypher
MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
MATCH (gw)-[:IN_SEASON]->(s:Season {id: $season})
RETURN gw.number AS gameweek, r.total_points AS points, r.form AS form,
       r.goals_scored AS goals, r.assists AS assists, r.minutes AS minutes, s.id AS season
ORDER BY gw.number
```

**All Seasons:**
```cypher
MATCH (p:Player {name: $player_name})-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
MATCH (gw)-[:IN_SEASON]->(s:Season)
RETURN s.id AS season, gw.number AS gameweek, r.total_points AS points, r.form AS form,
       r.goals_scored AS goals, r.assists AS assists, r.minutes AS minutes
ORDER BY s.id, gw.number
```

---

## Search Queries

### Query 17: Search Players by Name

**Method:** `search_players_by_name(name_pattern, limit)`

**Parameters:**
- `name_pattern`: Search string (case-insensitive partial match)
- `limit`: Number of results (default: 20)

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
WHERE toLower(p.name) CONTAINS toLower($name_pattern)
RETURN p.name AS player_name, pos.code AS position, p.element_id AS element_id
ORDER BY p.name
LIMIT $limit
```

---

### Query 18: Get All Players by Position

**Method:** `get_all_players_by_position(position)`

**Parameters:**
- `position`: Position code (GK, DEF, MID, FWD)

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: $position})
RETURN p.name AS player_name, p.element_id AS element_id
ORDER BY p.name
```

---

### Query 19: Get All Teams

**Method:** `get_all_teams()`

**Parameters:** None

```cypher
MATCH (t:Team)
RETURN t.name AS team_name
ORDER BY t.name
```

---

### Query 20: Get Season Summary

**Method:** `get_season_summary(season)`

**Parameters:**
- `season`: Season ID (required)

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

---

### Query 20b: Get All Seasons Summary

**Method:** `get_all_seasons_summary()`

**Parameters:** None

```cypher
MATCH (s:Season)
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
ORDER BY s.id
```

---

## Trivia-Specific Queries

### Highest Single Gameweek Score

**Method:** `get_highest_single_gameweek_score(season)`

**Parameters:**
- `season`: Season ID (required)

```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WHERE r.total_points > 0
RETURN p.name AS player_name, gw.number AS gameweek, r.total_points AS points,
       r.goals_scored AS goals, r.assists AS assists, r.bonus AS bonus
ORDER BY r.total_points DESC
LIMIT 1
```

---

### Player with Most Cards

**Method:** `get_player_with_most_cards(season, card_type)`

**Parameters:**
- `season`: Season ID (required)
- `card_type`: 'yellow' or 'red' (default: 'yellow')

```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, SUM(r.yellow_cards) AS total_cards
WHERE total_cards > 0
RETURN p.name AS player_name, total_cards
ORDER BY total_cards DESC
LIMIT 5
```

---

### Goalkeeper Saves Leader

**Method:** `get_goalkeeper_saves_leader(season)`

**Parameters:**
- `season`: Season ID (required)

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: 'GK'})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, SUM(r.saves) AS total_saves, SUM(r.clean_sheets) AS clean_sheets
WHERE total_saves > 0
RETURN p.name AS player_name, total_saves, clean_sheets
ORDER BY total_saves DESC
LIMIT 5
```

---

### Highest Scoring Fixture

**Method:** `get_highest_scoring_fixture(season)`

**Parameters:**
- `season`: Season ID (required)

```cypher
MATCH (f:Fixture)-[:HOME_TEAM]->(ht:Team)
MATCH (f)-[:AWAY_TEAM]->(at:Team)
MATCH (f)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH f, ht, at, gw, (f.home_score + f.away_score) AS total_goals
RETURN ht.name AS home_team, at.name AS away_team, 
       f.home_score AS home_score, f.away_score AS away_score,
       total_goals, gw.number AS gameweek
ORDER BY total_goals DESC
LIMIT 5
```

---

## Graph Schema Reference

### Node Types
- **Player**: `name`, `element_id`
- **Team**: `name`
- **Position**: `code` (GK, DEF, MID, FWD)
- **Season**: `id` (e.g., '2022-23')
- **Gameweek**: `number` (1-38)
- **Fixture**: `home_score`, `away_score`, `kickoff_time`

### Relationship Types
- `PLAYS_POSITION`: Player → Position
- `PLAYED_IN`: Player → Fixture (with performance stats)
- `HOME_TEAM`: Fixture → Team
- `AWAY_TEAM`: Fixture → Team
- `PART_OF`: Fixture → Gameweek
- `IN_SEASON`: Gameweek → Season

### PLAYED_IN Relationship Properties
- `total_points`, `goals_scored`, `assists`
- `clean_sheets`, `bonus`, `bps`
- `minutes`, `saves`
- `ict_index`, `influence`, `creativity`, `threat`
- `value`, `selected`
- `transfers_in`, `transfers_out`
- `yellow_cards`, `red_cards`
- `form`

---

*Generated from `graph/queries.py`*
