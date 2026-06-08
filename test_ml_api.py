"""
Python Script to Test ML API Endpoints
Usage: python test_ml_api.py
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_ml_status():
    """Test ML status endpoint."""
    print("\n" + "="*60)
    print("Test 1: ML Status Endpoint")
    print("="*60)
    print(f"GET {BASE_URL}/api/ml/status\n")
    
    try:
        response = requests.get(f"{BASE_URL}/api/ml/status")
        response.raise_for_status()
        
        print("✅ ML Status:")
        print(json.dumps(response.json(), indent=2))
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: API server not running")
        print("   Start server: python api_main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_predict_player(player_name="Mohamed Salah"):
    """Test single player prediction."""
    print("\n" + "="*60)
    print(f"Test 2: Predict Single Player ({player_name})")
    print("="*60)
    print(f"POST {BASE_URL}/api/ml/predict/player\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/ml/predict/player",
            json={"player_name": player_name}
        )
        response.raise_for_status()
        
        print(f"✅ Prediction for {player_name}:")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        # Extract key metrics
        if 'predicted_points' in result:
            print(f"\n📊 Predicted Points: {result['predicted_points']:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_top_performers(position="FWD", top_k=5):
    """Test top performers endpoint."""
    print("\n" + "="*60)
    print(f"Test 3: Top {top_k} {position}s")
    print("="*60)
    print(f"POST {BASE_URL}/api/ml/predict/top-performers\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/ml/predict/top-performers",
            json={"position": position, "top_k": top_k}
        )
        response.raise_for_status()
        
        print(f"✅ Top {top_k} {position}s:")
        result = response.json()
        
        if 'predictions' in result:
            for i, pred in enumerate(result['predictions'][:top_k], 1):
                name = pred.get('player_name', pred.get('name', 'Unknown'))
                points = pred.get('predicted_points', 0)
                print(f"  {i}. {name}: {points:.2f} pts")
        else:
            print(json.dumps(result, indent=2))
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_best_value(position="MID", max_price=8.0, top_k=5):
    """Test best value endpoint."""
    print("\n" + "="*60)
    print(f"Test 4: Best Value {position}s (£{max_price}m max)")
    print("="*60)
    print(f"POST {BASE_URL}/api/ml/predict/best-value\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/ml/predict/best-value",
            json={
                "position": position,
                "max_price": max_price,
                "top_k": top_k
            }
        )
        response.raise_for_status()
        
        print(f"✅ Best Value {position}s:")
        result = response.json()
        
        if isinstance(result, list):
            for i, pred in enumerate(result[:top_k], 1):
                name = pred.get('name', pred.get('player_name', 'Unknown'))
                points = pred.get('predicted_points', 0)
                value = pred.get('value', 0)
                ppm = pred.get('points_per_million', 0)
                print(f"  {i}. {name}: {points:.2f} pts (£{value}m, {ppm:.2f} pts/£m)")
        else:
            print(json.dumps(result, indent=2))
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("FPL ML API TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: Status
    results.append(("Status", test_ml_status()))
    
    if not results[0][1]:
        print("\n❌ Cannot proceed - API server not running")
        print("   Start server: python api_main.py")
        return
    
    # Test 2: Single player
    results.append(("Single Player", test_predict_player("Mohamed Salah")))
    
    # Test 3: Top performers
    results.append(("Top Performers", test_top_performers("FWD", 5)))
    
    # Test 4: Best value
    results.append(("Best Value", test_best_value("MID", 8.0, 5)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
