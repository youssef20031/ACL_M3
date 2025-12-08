"""Quick script to check Neo4j data status"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.connection import Neo4jConnection

try:
    conn = Neo4jConnection("bolt://localhost:7687", "neo4j", "pass_1234")
    
    # Run the exact UNION query
    query = """
        MATCH (p:Player)-[:PLAYS_POSITION]->(pos:Position {code: 'GK'})
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF]->(gw:Gameweek)
        MATCH (gw)-[:IN_SEASON]->(s:Season)
        WITH 'GK' AS position, p.name AS player_name, 
             SUM(r.total_points) AS total_points, 
             SUM(r.goals_scored) AS goals, 
             SUM(r.assists) AS assists, 
             SUM(r.bonus) AS bonus
        ORDER BY total_points DESC
        LIMIT 3
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
        LIMIT 3
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
        LIMIT 3
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
        LIMIT 3
        RETURN position, player_name, total_points, goals, assists, bonus
    """
    
    result = conn.execute_query(query)
    print(f"Total results from UNION query: {len(result)}")
    print("\nResults:")
    for r in result:
        print(f"  {r['position']}: {r['player_name']} - {r['total_points']} pts")
    
    conn.close()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
