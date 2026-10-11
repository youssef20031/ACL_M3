"""Test API predictions to ensure v2 models are loaded and predictions are correct"""
import requests
import json

def test_api():
    base_url = 'http://localhost:8000'
    
    print("=" * 80)
    print("TESTING API PREDICTIONS (CSV-based GET endpoints)")
    print("=" * 80)
    
    try:
        # Test 1: Top performers
        print("\n1. Testing Top Performers (GET /api/ml/top-performers)...")
        print("   a) All positions, top 10:")
        response = requests.get(f'{base_url}/api/ml/top-performers', 
                               params={'limit': 10},
                               timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"      ✅ Status: {response.status_code}")
            print(f"      Model Version: {data.get('model_version', 'N/A')}")
            print(f"      Total Predictions: {data.get('total_predictions', 'N/A')}")
            print(f"\n      Top 10 Players (All Positions):")
            
            for i, pred in enumerate(data.get('predictions', [])[:10], 1):
                home_marker = 'H' if pred.get('is_home') else 'A'
                print(f"      {i:2d}. {pred['predicted_points']:5.2f} pts | {pred['position']:3s} | {pred['player_name']:30s} | vs {pred.get('opponent', 'TBD'):20s} ({home_marker})")
            
            # Check for retired players
            player_names = [p['player_name'] for p in data.get('predictions', [])]
            retired = ['Sergio Agüero', 'Sergio Aguero', 'Mesut Özil', 'David Silva']
            found_retired = [name for name in retired if name in player_names]
            
            if found_retired:
                print(f"\n      ⚠️ WARNING: Found retired players: {found_retired}")
            else:
                print(f"\n      ✅ No retired players detected")
        else:
            print(f"      ❌ Error: {response.status_code}")
            print(f"      {response.text[:200]}")
        
        # Test with position filter
        print("\n   b) Midfielders only, top 5:")
        response = requests.get(f'{base_url}/api/ml/top-performers',
                               params={'position': 'MID', 'limit': 5},
                               timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"      ✅ Status: {response.status_code}")
            print(f"\n      Top 5 Midfielders:")
            
            for i, pred in enumerate(data.get('predictions', [])[:5], 1):
                home_marker = 'H' if pred.get('is_home') else 'A'
                print(f"      {i}. {pred['predicted_points']:5.2f} pts | {pred['player_name']:30s} | vs {pred.get('opponent', 'TBD'):20s} ({home_marker})")
        else:
            print(f"      ❌ Error: {response.status_code}")
        
        # Test 2: Best value
        print("\n2. Testing Best Value (GET /api/ml/best-value)...")
        response = requests.get(f'{base_url}/api/ml/best-value',
                               params={'limit': 10},
                               timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {response.status_code}")
            print(f"\n   Top 10 Value Players:")
            
            for i, pred in enumerate(data.get('predictions', [])[:10], 1):
                value_score = pred.get('value_score', 0)
                cost = pred.get('value', 0)
                home_marker = 'H' if pred.get('is_home') else 'A'
                print(f"   {i:2d}. Score: {value_score:5.2f} | {pred['position']:3s} | {pred['player_name']:30s} | £{cost:.1f}m | {pred['predicted_points']:4.2f} pts")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   {response.text[:200]}")
        
        # Test 3: Check total predictions count
        print("\n3. Checking Total Predictions Count...")
        response = requests.get(f'{base_url}/api/ml/top-performers',
                               params={'limit': 1000},
                               timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            total = len(data.get('predictions', []))
            print(f"   Total predictions available: {total}")
            
            if 600 <= total <= 900:
                print(f"   ✅ Good! Within expected range for active players (~692-819)")
            else:
                print(f"   ⚠️ Note: {total} predictions (expected ~692-819 active players)")
        
        print("\n" + "=" * 80)
        print("✅ API TEST COMPLETE")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("   ❌ Could not connect to API. Is it running on port 8000?")
    except requests.exceptions.Timeout:
        print("   ❌ API request timed out")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_api()
