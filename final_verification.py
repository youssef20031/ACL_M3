"""Final verification that everything is working correctly"""
import requests

def final_check():
    print("=" * 80)
    print("FINAL VERIFICATION - ACTIVE PLAYERS & FIXTURE CONTEXT")
    print("=" * 80)
    
    try:
        # Test 1: Forwards
        print("\n📊 Test 1: Top 3 Forwards with Fixture Context")
        print("-" * 80)
        
        r = requests.get('http://localhost:8000/api/ml/top-performers', 
                        params={'position': 'FWD', 'limit': 3},
                        timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Status: {r.status_code}")
            print(f"✅ Model Version: {data['model_version']}")
            print(f"✅ Total Predictions: {data['total_predictions']}\n")
            
            for i, p in enumerate(data['predictions'][:3], 1):
                fc = p.get('fixture_context', {})
                print(f"{i}. {p['player_name']:30s} {p['predicted_points']:5.2f} pts")
                print(f"   Team: {p['team_name']:20s} Position: {p['position']}")
                print(f"   Opponent: {fc.get('opponent_team', 'N/A'):20s} {fc.get('home_away', 'N/A').upper()}")
                print(f"   Date: {fc.get('match_date', 'N/A')} | GW{fc.get('matchday', '?')}")
                print()
        else:
            print(f"❌ Error: {r.status_code}")
            return False
        
        # Test 2: Best Value
        print("\n💰 Test 2: Best Value Players")
        print("-" * 80)
        
        r = requests.get('http://localhost:8000/api/ml/best-value',
                        params={'limit': 3},
                        timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Status: {r.status_code}\n")
            
            for i, p in enumerate(data['predictions'][:3], 1):
                fc = p.get('fixture_context', {})
                print(f"{i}. {p['player_name']:30s} £{p['value']:.1f}m | Score: {p['value_score']:.2f}")
                print(f"   {p['predicted_points']:5.2f} pts vs {fc.get('opponent_team', 'N/A')} ({fc.get('home_away', 'N/A').upper()})")
                print()
        else:
            print(f"❌ Error: {r.status_code}")
            return False
        
        # Test 3: Check for retired players
        print("\n🔍 Test 3: Checking for Retired Players")
        print("-" * 80)
        
        r = requests.get('http://localhost:8000/api/ml/top-performers',
                        params={'limit': 1000},
                        timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            all_names = [p['player_name'] for p in data['predictions']]
            
            retired_players = ['Sergio Agüero', 'Sergio Aguero', 'Mesut Özil', 'David Silva']
            found_retired = [name for name in retired_players if name in all_names]
            
            if found_retired:
                print(f"❌ FAILED: Found retired players: {found_retired}")
                return False
            else:
                print(f"✅ PASSED: No retired players found")
                print(f"✅ Total Active Players: {len(data['predictions'])}")
        
        # Test 4: Verify fixture context exists for all
        print("\n🎯 Test 4: Verify All Predictions Have Fixture Context")
        print("-" * 80)
        
        missing_fc = [p['player_name'] for p in data['predictions'] if not p.get('fixture_context')]
        
        if missing_fc:
            print(f"❌ FAILED: {len(missing_fc)} predictions missing fixture context")
            print(f"   Examples: {missing_fc[:5]}")
            return False
        else:
            print(f"✅ PASSED: All {len(data['predictions'])} predictions have fixture context")
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - SYSTEM WORKING CORRECTLY")
        print("=" * 80)
        print("\nSummary:")
        print(f"  • Active Players: {len(data['predictions'])}")
        print(f"  • Retired Players: 0")
        print(f"  • Fixture Context: Complete for all players")
        print(f"  • Model Version: {data['model_version']}")
        print(f"  • API Status: Healthy")
        print("\n✅ Ready for production use!")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is it running on port 8000?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = final_check()
    exit(0 if success else 1)
