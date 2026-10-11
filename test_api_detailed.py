"""Test API with detailed error capture"""
import requests
import json

url = "http://localhost:8000/api/ml/predict/top-performers"
payload = {"position": None, "top_k": 5}

print("Testing top-performers endpoint...")
print("=" * 80)

try:
    response = requests.post(url, json=payload, timeout=30)
    
    print(f"Status Code: {response.status_code}")
    print(f"\nHeaders:")
    for key, value in response.headers.items():
        print(f"  {key}: {value}")
    
    print(f"\nResponse Body:")
    print(response.text)
    
    if response.status_code == 200:
        data = response.json()
        print("\n✅ SUCCESS!")
        print(f"Got {len(data.get('predictions', []))} predictions")
    else:
        print(f"\n❌ ERROR {response.status_code}")
        
        # Try to parse as JSON
        try:
            error_data = response.json()
            print(f"Error detail: {json.dumps(error_data, indent=2)}")
        except:
            print(f"Raw error: {response.text}")
            
except Exception as e:
    print(f"\n❌ EXCEPTION: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("\nNOTE: Check the server terminal logs for the full Python traceback!")
