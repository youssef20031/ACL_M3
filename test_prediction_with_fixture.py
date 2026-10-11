"""Test prediction with fixture context"""
import requests
import json

# Test the prediction endpoint
base_url = "http://127.0.0.1:8000"

# Get a player prediction
print("Testing player prediction with fixture context...")
print("=" * 60)

# Try to get predictions for a well-known player (should have a team in the fixture list)
players_to_test = ["Mohamed Salah", "Erling Haaland", "Kevin De Bruyne", "Bukayo Saka"]

for player_name in players_to_test:
    try:
        response = requests.post(
            f"{base_url}/api/ml/predict",
            json={"player_name": player_name},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Player: {player_name}")
            print(f"  Predicted Points: {data.get('predicted_points', 'N/A')}")
            
            fixture_context = data.get('fixture_context')
            if fixture_context:
                print(f"  Upcoming Fixture:")
                print(f"    Opponent: {fixture_context.get('opponent_team', 'N/A')}")
                print(f"    Home/Away: {fixture_context.get('home_away', 'N/A')}")
                print(f"    Match Date: {fixture_context.get('match_date', 'N/A')}")
                print(f"    Matchday: {fixture_context.get('matchday', 'N/A')}")
            else:
                print(f"  No upcoming fixture (fixture_context is null)")
            
            break  # Found a working player
        elif response.status_code == 404:
            continue  # Try next player
        else:
            print(f"✗ Error for {player_name}: {response.status_code}")
    except Exception as e:
        print(f"✗ Exception for {player_name}: {e}")

print("\n" + "=" * 60)
