"""Test the ML endpoints directly to debug CORS/server issues"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("Testing ML Prediction Endpoints")
print("=" * 80)

# Test 1: Health check
print("\n1. Testing /health endpoint...")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   ✅ Server is running")
    else:
        print(f"   ❌ Unexpected status: {response.text}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    print("\n   Server might not be running. Start it with: python api_main.py")
    exit(1)

# Test 2: ML Status
print("\n2. Testing /api/ml/status endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/ml/status", timeout=5)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Predictor loaded: {data.get('predictor_loaded')}")
        print(f"   Model type: {data.get('model_type')}")
    else:
        print(f"   ❌ Unexpected response: {response.text}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test 3: Top Performers
print("\n3. Testing /api/ml/predict/top-performers endpoint...")
try:
    payload = {
        "position": None,
        "top_k": 5
    }
    response = requests.post(
        f"{BASE_URL}/api/ml/predict/top-performers",
        json=payload,
        timeout=30
    )
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        predictions = data.get('predictions', [])
        print(f"   ✅ Got {len(predictions)} predictions")
        
        if predictions:
            print(f"\n   Sample predictions:")
            for i, pred in enumerate(predictions[:3], 1):
                player = pred.get('player_name', 'Unknown')
                points = pred.get('predicted_points', 0)
                fixture = pred.get('fixture_context')
                
                print(f"\n   {i}. {player}: {points:.1f} pts")
                if fixture:
                    print(f"      Opponent: {fixture.get('opponent_team')} ({fixture.get('home_away')})")
                    print(f"      Matchday: {fixture.get('matchday')}")
                else:
                    print(f"      ⚠️  No fixture context")
    else:
        print(f"   ❌ ERROR {response.status_code}")
        print(f"   Response: {response.text[:500]}")
        
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 4: CORS headers
print("\n4. Checking CORS headers...")
try:
    response = requests.options(
        f"{BASE_URL}/api/ml/predict/top-performers",
        headers={"Origin": "http://localhost:3000"},
        timeout=5
    )
    print(f"   Status: {response.status_code}")
    print(f"   Access-Control-Allow-Origin: {response.headers.get('Access-Control-Allow-Origin', 'NOT SET')}")
    print(f"   Access-Control-Allow-Methods: {response.headers.get('Access-Control-Allow-Methods', 'NOT SET')}")
    
    if response.headers.get('Access-Control-Allow-Origin'):
        print("   ✅ CORS is configured")
    else:
        print("   ❌ CORS headers missing")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print("\n" + "=" * 80)
print("DONE")
