# Trivia Query: Player with Most Cards

## Description

Identifies the player with the most yellow or red cards in a season.

## Method Signature

`CypherQueries.get_player_with_most_cards(season: str, card_type: str = "yellow")`

## Parameters

- `season`: Season ID.
- `card_type`: 'yellow' or 'red' (default 'yellow').

## Example Natural Language Prompts

- "Who got the most yellow cards?"
- "Most red cards in 2022-23"

## Cypher Query

```cypher
MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season {id: $season})
WITH p, SUM(r.{card_field}) AS total_cards
WHERE total_cards > 0
RETURN p.name AS player_name, total_cards
ORDER BY total_cards DESC
LIMIT 5
```
