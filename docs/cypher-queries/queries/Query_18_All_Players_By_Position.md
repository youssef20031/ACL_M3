# Query 18: Get All Players by Position

## Description

Retrieves a list of all players registered for a specific position.

## Method Signature

`CypherQueries.get_all_players_by_position(position: str)`

## Parameters

- `position`: Position code ('GK', 'DEF', 'MID', 'FWD').

## Example Natural Language Prompts

- "List all defenders"
- "Show me every goalkeeper"

## Cypher Query

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: $position})
RETURN p.name AS player_name, p.element_id AS element_id
ORDER BY p.name
```
