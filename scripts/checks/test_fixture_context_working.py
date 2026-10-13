"""Test that predictions now include fixture context"""
from graph.connection import Neo4jConnection
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from services.fixture_service import FixtureService
from datetime import datetime, timezone

conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
fixture_service = FixtureService(conn)

# Test with a few players
test_players = ["Mohamed Salah", "Bukayo Saka", "Erling Haaland", "Kevin De Bruyne", "Cole Palmer"]

print("Testing fixture context for players:")
print("=" * 80)

current_date = datetime.now(timezone.utc)
print(f"Current date: {current_date}\n")

for player_name in test_players:
    try:
        fixture_context = fixture_service.get_next_match(player_name, current_date)
        
        if fixture_context:
            print(f"✅ {player_name}:")
            print(f"   Opponent: {fixture_context.opponent_team}")
            print(f"   Home/Away: {fixture_context.home_away}")
            print(f"   Match Date: {fixture_context.match_date}")
            print(f"   Matchday: {fixture_context.matchday}")
            print()
        else:
            print(f"⚠️  {player_name}: No upcoming fixture")
            print()
    except Exception as e:
        print(f"❌ {player_name}: Error - {e}")
        print()

conn.close()
