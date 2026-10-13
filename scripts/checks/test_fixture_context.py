"""Test that API returns fixture context properly"""
import requests
import json

def test_fixture_context():
    print("=" * 80)
    print("TESTING FIXTURE CONTEXT IN API RESPONSES")
    print("=" * 80)
    
    try:
        # Test top performers
        print("\n1. Testing fixture_context in top performers...")
        response = requests.get(
            'http://localhost:8000/api/ml/top-performers',
            params={'limit': 3},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {response.status_code}")
            
            for i, pred in enumerate(data.get('predictions', [])[:3], 1):
                print(f"\n   {i}. {pred['player_name']} ({pred['team_name']})")
                print(f"      Predicted Points: {pred['predicted_points']:.2f}")
                
                if pred.get('fixture_context'):
                    fc = pred['fixture_context']
                    print(f"      Fixture Context: ✅")
                    print(f"        - Opponent: {fc.get('opponent_team')}")
                    print(f"        - Home/Away: {fc.get('home_away')}")
                    print(f"        - Match Date: {fc.get('match_date')}")
                    print(f"        - Gameweek: {fc.get('matchday')}")
                else:
                    print(f"      Fixture Context: ❌ MISSING")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   {response.text[:200]}")
        
        # Test best value
        print("\n2. Testing fixture_context in best value...")
        response = requests.get(
            'http://localhost:8000/api/ml/best-value',
            params={'limit': 2},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {response.status_code}")
            
            for i, pred in enumerate(data.get('predictions', [])[:2], 1):
                print(f"\n   {i}. {pred['player_name']} ({pred['team_name']})")
                print(f"      Value Score: {pred.get('value_score', 0):.2f}")
                
                if pred.get('fixture_context'):
                    fc = pred['fixture_context']
                    print(f"      Fixture Context: ✅")
                    print(f"        - Opponent: {fc.get('opponent_team')}")
                    print(f"        - Home/Away: {fc.get('home_away')}")
                else:
                    print(f"      Fixture Context: ❌ MISSING")
        else:
            print(f"   ❌ Error: {response.status_code}")
        
        print("\n" + "=" * 80)
        print("✅ FIXTURE CONTEXT TEST COMPLETE")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is it running on port 8000?")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_fixture_context()
