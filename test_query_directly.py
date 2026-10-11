"""Test the Cypher query directly to find the error"""
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from datetime import datetime, timezone

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

print("Testing the new Cypher query...")
print("=" * 80)

current_date = datetime.now(timezone.utc)
position_clause = ""
season_clause = ""

query = f"""
MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position)
MATCH (p)-[:PLAYS_FOR]->(team:Team)
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)-[:IN_SEASON]->(s:Season)
WHERE r.minutes > 0
  {position_clause}
  {season_clause}
WITH p, pos, team, r, f
ORDER BY f.kickoff_time DESC
WITH p, pos, team, COLLECT(r)[..4] AS recent_perfs
WHERE SIZE(recent_perfs) >= 2

// NEW: Check if team has upcoming fixtures in 2026-27
OPTIONAL MATCH (upcoming_fixture:Fixture {{season_id: "2026-27"}})
WHERE ((upcoming_fixture)-[:PLAYS_HOME]->(team) OR (upcoming_fixture)-[:PLAYS_AWAY]->(team))
  AND upcoming_fixture.kickoff_time > $current_date
WITH p, pos, team, recent_perfs, COUNT(DISTINCT upcoming_fixture) as fixture_count
WHERE fixture_count > 0

UNWIND recent_perfs AS perf
RETURN
    p.name                                                   AS name,
    pos.code                                                 AS position,
    AVG(perf.minutes)                                        AS minutes,
    AVG(perf.goals_scored)                                   AS goals_scored,
    AVG(perf.assists)                                        AS assists,
    AVG(perf.total_points)                                   AS form,
    AVG(perf.bps)                                            AS bps,
    AVG(perf.ict_index)                                      AS ict_index,
    AVG(perf.influence)                                      AS influence,
    AVG(perf.creativity)                                     AS creativity,
    AVG(perf.threat)                                         AS threat,
    SUM(perf.clean_sheets)                                   AS clean_sheets,
    SUM(perf.bonus)                                          AS bonus,
    AVG(perf.value)                                          AS value
LIMIT 10
"""

params = {
    "position": None,
    "season": None,
    "current_date": current_date,
}

print(f"Current date: {current_date}\n")
print("Executing query...")

try:
    results = conn.execute_query(query, params)
    print(f"\n✅ Query succeeded! Got {len(results)} results\n")
    
    if results:
        print("Sample results:")
        for i, r in enumerate(results[:3], 1):
            print(f"\n{i}. {r['name']}")
            print(f"   Position: {r['position']}")
            print(f"   Form: {r['form']:.1f}")
            print(f"   Minutes: {r['minutes']:.1f}")
    else:
        print("⚠️  Query returned no results")
        print("\nThis means either:")
        print("1. No players have PLAYS_FOR relationships")
        print("2. No teams have 2026-27 fixtures")
        print("3. Query logic issue")
        
except Exception as e:
    print(f"\n❌ Query failed with error:")
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

conn.close()
