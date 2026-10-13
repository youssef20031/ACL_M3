"""
Test that the database is in the correct state for the fix to work.
Run this BEFORE restarting the server to verify everything is ready.
"""
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from datetime import datetime, timezone

def test_database_state():
    """Test database state."""
    conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    print("=" * 80)
    print("DATABASE STATE VERIFICATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}\n")
    
    # Test 1: Check PLAYS_FOR relationships
    results = conn.execute_query('MATCH ()-[r:PLAYS_FOR]->() RETURN count(r) as count')
    plays_for_count = results[0]['count']
    print(f"1. PLAYS_FOR relationships: {plays_for_count}")
    if plays_for_count > 0:
        print("   ✅ PASS - Players linked to teams")
    else:
        print("   ❌ FAIL - Need to run: python -m scripts.data.create_player_team_relationships")
    
    # Test 2: Check 2026-27 fixtures
    results = conn.execute_query('MATCH (f:Fixture {season_id: "2026-27"}) RETURN count(f) as count')
    fixture_count = results[0]['count']
    print(f"\n2. 2026-27 Fixtures: {fixture_count}")
    if fixture_count >= 300:
        print("   ✅ PASS - Sufficient fixtures loaded")
    else:
        print("   ❌ FAIL - Need to run: python migrations\\load_2026_27_fixtures.py --csv-path data/FPL_2026_2027_fixtures_fixed.csv --clear-existing")
    
    # Test 3: Check teams with fixtures
    results = conn.execute_query('''
        MATCH (t:Team)
        OPTIONAL MATCH (f:Fixture {season_id: "2026-27"})-[:PLAYS_HOME|PLAYS_AWAY]->(t)
        WITH t.name as team, count(DISTINCT f) as fixtures
        WHERE fixtures > 0
        RETURN count(team) as teams_with_fixtures
    ''')
    teams_with_fixtures = results[0]['teams_with_fixtures']
    print(f"\n3. Teams with 2026-27 fixtures: {teams_with_fixtures}")
    if teams_with_fixtures >= 18:
        print("   ✅ PASS - Sufficient teams have fixtures")
    else:
        print("   ❌ FAIL - Not enough teams with fixtures")
    
    # Test 4: Check sample player has team and fixture
    test_players = ["Mohamed Salah", "Bukayo Saka", "Cole Palmer"]
    print(f"\n4. Sample player checks:")
    
    players_with_fixtures = 0
    for player_name in test_players:
        # Check if player has team
        results = conn.execute_query('''
            MATCH (p:Player {name: $name})-[:PLAYS_FOR]->(t:Team)
            RETURN t.name as team
        ''', {"name": player_name})
        
        if results:
            team_name = results[0]['team']
            
            # Check if team has fixtures
            current_date = datetime.now(timezone.utc)
            results2 = conn.execute_query('''
                MATCH (t:Team {name: $team})
                MATCH (f:Fixture {season_id: "2026-27"})
                WHERE (f)-[:PLAYS_HOME]->(t) OR (f)-[:PLAYS_AWAY]->(t)
                  AND f.kickoff_time > $current_date
                RETURN count(f) as fixtures
            ''', {"team": team_name, "current_date": current_date})
            
            fixture_count = results2[0]['fixtures'] if results2 else 0
            
            if fixture_count > 0:
                print(f"   ✅ {player_name}: Team={team_name}, Fixtures={fixture_count}")
                players_with_fixtures += 1
            else:
                print(f"   ⚠️  {player_name}: Team={team_name}, No fixtures")
        else:
            print(f"   ❌ {player_name}: No team assigned")
    
    if players_with_fixtures == len(test_players):
        print(f"   ✅ PASS - All test players have teams and fixtures")
    else:
        print(f"   ⚠️  WARNING - Only {players_with_fixtures}/{len(test_players)} players ready")
    
    # Test 5: Check if Jamie Vardy would be filtered out
    print(f"\n5. Relegated player check (Jamie Vardy):")
    results = conn.execute_query('''
        MATCH (p:Player {name: "Jamie Vardy"})
        OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team)
        OPTIONAL MATCH (f:Fixture {season_id: "2026-27"})
        WHERE t IS NOT NULL AND ((f)-[:PLAYS_HOME]->(t) OR (f)-[:PLAYS_AWAY]->(t))
        RETURN p.name as player, t.name as team, count(DISTINCT f) as fixtures
    ''')
    
    if results and results[0]['player']:
        team = results[0]['team']
        fixtures = results[0]['fixtures']
        if team is None:
            print("   ✅ PASS - Jamie Vardy has no team (will be filtered out)")
        else:
            print(f"   Team: {team}")
            print(f"   2026-27 Fixtures: {fixtures}")
            if fixtures == 0:
                print("   ✅ PASS - Jamie Vardy's team has no fixtures (will be filtered out)")
            else:
                print(f"   ❌ FAIL - Jamie Vardy's team has {fixtures} fixtures (should be 0)")
    else:
        print("   ✅ PASS - Jamie Vardy not in database or has no team")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nDebug: plays_for_count={plays_for_count}, fixture_count={fixture_count}, teams_with_fixtures={teams_with_fixtures}, players_with_fixtures={players_with_fixtures}")
    
    all_passed = (
        plays_for_count > 0 and
        fixture_count >= 300 and
        teams_with_fixtures >= 18 and
        players_with_fixtures == len(test_players)
    )
    
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nThe database is ready! You can now:")
        print("1. Restart the backend server (if running)")
        print("2. Check the website - fixtures should display")
        print("3. Run: python verify_fixture_fix.py (after server starts)")
        print("\nImportant notes:")
        print("- Only players from 2026-27 Premier League teams will appear")
        print("- Jamie Vardy and other relegated team players will NOT appear")
        print("- All predictions will include fixture context")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease run the required setup scripts:")
        if plays_for_count == 0:
            print("  - python -m scripts.data.create_player_team_relationships")
        if fixture_count < 300:
            print("  - python migrations\\load_2026_27_fixtures.py --csv-path data/FPL_2026_2027_fixtures_fixed.csv --clear-existing")
        print("\nThen run this test again.")
    
    print("=" * 80)
    
    conn.close()
    return all_passed

if __name__ == "__main__":
    test_database_state()
