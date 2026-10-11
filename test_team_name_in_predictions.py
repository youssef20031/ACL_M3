"""
Quick test to verify team_name is included in ML predictions
Run this AFTER restarting the backend
"""
import requests
import json

API_BASE_URL = "http://localhost:8000"  # Change if your backend is on a different port

def test_top_performers():
    """Test if top performers endpoint returns team_name"""
    print("=" * 80)
    print("Testing ML Top Performers Endpoint")
    print("=" * 80)
    
    url = f"{API_BASE_URL}/api/ml/predict/top-performers"
    payload = {
        "top_k": 5,
        "position": None
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        predictions = data.get("predictions", [])
        
        if not predictions:
            print("❌ No predictions returned")
            return False
        
        print(f"\n✓ Got {len(predictions)} predictions\n")
        
        success = True
        for i, pred in enumerate(predictions[:3], 1):
            player_name = pred.get("player_name", "Unknown")
            team_name = pred.get("team_name")
            predicted_points = pred.get("predicted_points", 0)
            
            print(f"{i}. {player_name}")
            print(f"   Team: {team_name if team_name else '❌ MISSING'}")
            print(f"   Predicted Points: {predicted_points:.1f}")
            print()
            
            if not team_name:
                success = False
        
        if success:
            print("✓ All players have team_name!")
        else:
            print("❌ Some players missing team_name - backend needs restart")
        
        return success
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running?")
        print(f"   Tried: {API_BASE_URL}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_best_value():
    """Test if best value endpoint returns team_name"""
    print("\n" + "=" * 80)
    print("Testing ML Best Value Endpoint")
    print("=" * 80)
    
    url = f"{API_BASE_URL}/api/ml/predict/best-value"
    payload = {
        "top_k": 3,
        "position": None,
        "max_price": None
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        predictions = data.get("predictions", [])
        
        if not predictions:
            print("❌ No predictions returned")
            return False
        
        print(f"\n✓ Got {len(predictions)} predictions\n")
        
        success = True
        for i, pred in enumerate(predictions, 1):
            player_name = pred.get("player_name", "Unknown")
            team_name = pred.get("team_name")
            value = pred.get("value", 0)
            
            print(f"{i}. {player_name}")
            print(f"   Team: {team_name if team_name else '❌ MISSING'}")
            print(f"   Value Score: {value:.2f}")
            print()
            
            if not team_name:
                success = False
        
        if success:
            print("✓ All players have team_name!")
        else:
            print("❌ Some players missing team_name - backend needs restart")
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_compare_players():
    """Test if compare players endpoint returns team_name"""
    print("\n" + "=" * 80)
    print("Testing Compare Players Endpoint")
    print("=" * 80)
    
    url = f"{API_BASE_URL}/api/players/compare"
    payload = {
        "player1": "Mohamed Salah",
        "player2": "Erling Haaland",
        "season": None
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        comparison = data.get("comparison", [])
        
        if not comparison:
            print("❌ No comparison data returned")
            return False
        
        print(f"\n✓ Got comparison for {len(comparison)} players\n")
        
        success = True
        for player in comparison:
            player_name = player.get("player_name", "Unknown")
            team_name = player.get("team_name")
            total_points = player.get("total_points", 0)
            
            print(f"• {player_name}")
            print(f"  Team: {team_name if team_name else '❌ MISSING'}")
            print(f"  Total Points: {total_points}")
            print()
            
            if not team_name:
                success = False
        
        if success:
            print("✓ All players have team_name!")
        else:
            print("❌ Some players missing team_name - backend needs restart")
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n🔍 Testing Backend Endpoints for team_name Field\n")
    
    results = []
    
    # Test all endpoints
    results.append(("ML Top Performers", test_top_performers()))
    results.append(("ML Best Value", test_best_value()))
    results.append(("Compare Players", test_compare_players()))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_pass = all(success for _, success in results)
    
    if all_pass:
        print("\n🎉 All tests passed! Team names are being returned correctly.")
        print("   The frontend should now display club logos.")
    else:
        print("\n⚠️  Some tests failed. Please:")
        print("   1. Restart the backend server")
        print("   2. Run this test again")
        print("   3. If still failing, check Neo4j for PLAYS_FOR relationships")
