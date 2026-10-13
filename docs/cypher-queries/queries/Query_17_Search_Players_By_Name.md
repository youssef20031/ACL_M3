# Query 17: Search Players by Name

## Description

Performs a case-insensitive fuzzy search for players matching a name pattern.

## Method Signature

`CypherQueries.search_players_by_name(name_pattern: str, limit: int = 20)`

## Parameters

- `name_pattern`: String to search for (partial match).
- `limit`: Number of results (default: 20).

## Example Natural Language Prompts

- "Search for player Raya"
- "Find players named John"

## Cypher Query

```cypher
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
WHERE toLower(p.name) CONTAINS toLower($name_pattern)
RETURN p.name AS player_name, pos.code AS position, p.element_id AS element_id
ORDER BY p.name
LIMIT $limit
```
