"""
Verify that the fixture fix is working end-to-end.
Run this after starting the backend server.
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not reachable: {e}")
        return False

def test_ml_predictions():
    """Test ML prediction endpoints with fixture context."""
    print("\n" + "=" * 80)
    print("Testing ML Predictions with Fixture Context")
    print("=" * 80)
    
    # Test top performers
    try:
        response = requests.post(
            f"{BASE_URL}/api/ml/predict/top-performers",
            json={"position": None, "limit": 5},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            
            print(f"\n✅ Top Performers API working ({len(predictions)} predictions)")
            
            fixtures_found = 0
            no_fixtures = 0
            
            for pred in predictions[:5]:
                player_name = pred.get('player_name', 'Unknown')
                predicted_points = pred.get('predicted_points', 0)
                fixture_context = pred.get('fixture_context')
                
                if fixture_context:
                    fixtures_found += 1
                    opponent = fixture_context.get('opponent_team', 'N/A')
                    home_away = fixture_context.get('home_away', 'N/A')
                    matchday = fixture_context.get('matchday', 'N/A')
                    print(f"\n   {player_name}: {predicted_points:.1f} pts")
                    print(f"   ├─ Opponent: {opponent} ({home_away})")
                    print(f"   └─ Matchday: {matchday}")
                else:
                    no_fixtures += 1
                    print(f"\n   {player_name}: {predicted_points:.1f} pts")
                    print(f"   └─ ⚠️  No upcoming fixture")
            
            print(f"\n   Summary: {fixtures_found} with fixtures, {no_fixtures} without")
            
            if fixtures_found > 0:
                print("\n   ✅ SUCCESS: Fixtures are being displayed!")
            else:
                print("\n   ❌ FAIL: No fixtures found in predictions")
            
        else:
            print(f"❌ Top performers API returned {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    
    except Exception as e:
        print(f"❌ Error testing top performers: {e}")

def test_individual_prediction():
    """Test individual player prediction."""
    print("\n" + "=" * 80)
    print("Testing Individual Player Prediction")
    print("=" * 80)
    
    test_players = ["Mohamed Salah", "Erling Haaland", "Bukayo Saka"]
    
    for player_name in test_players:
        try:
            response = requests.post(
                f"{BASE_URL}/api/ml/predict",
                json={"player_name": player_name},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                fixture_context = data.get('fixture_context')
                predicted_points = data.get('predicted_points', 0)
                
                print(f"\n✅ {player_name}: {predicted_points:.1f} pts")
                
                if fixture_context:
                    print(f"   ├─ Opponent: {fixture_context.get('opponent_team', 'N/A')}")
                    print(f"   ├─ Home/Away: {fixture_context.get('home_away', 'N/A')}")
                    print(f"   ├─ Match Date: {fixture_context.get('match_date', 'N/A')}")
                    print(f"   └─ Matchday: {fixture_context.get('matchday', 'N/A')}")
                else:
                    print(f"   └─ ⚠️  No upcoming fixture")
                    
            elif response.status_code == 404:
                print(f"⚠️  {player_name}: Player not found")
            else:
                print(f"❌ {player_name}: Error {response.status_code}")
                
        except Exception as e:
            print(f"❌ {player_name}: {e}")

def main():
    print("=" * 80)
    print("FIXTURE FIX VERIFICATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    if not test_health():
        print("\n⚠️  Server is not running. Please start it first:")
        print("   python api_main.py")
        return
    
    test_ml_predictions()
    test_individual_prediction()
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print("\nIf you see fixtures displayed above, the fix is working! ✅")
    print("The website should now show upcoming fixtures instead of 'No upcoming fixture'")

if __name__ == "__main__":
    main()
