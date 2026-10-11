"""Test RAG model with prediction queries and context awareness"""
import requests
import json

def test_rag_predictions():
    print("=" * 80)
    print("TESTING RAG MODEL WITH ML PREDICTIONS - UPDATED TESTS")
    print("=" * 80)
    
    base_url = "http://localhost:8000"
    
    # Test with conversation context (simulating a chat history)
    test_scenarios = [
        {
            "name": "Test 1: Who should I pick for upcoming GW? (fresh query)",
            "messages": [
                {"question": "Which player should I pick in the upcoming GW?", "chat_history": []}
            ],
            "expected_keywords": ["Haaland", "8.0", "predict", "Bournemouth", "Palmer"],
            "should_not_contain": ["2022-23", "Xavi Simons", "6.7"]
        },
        {
            "name": "Test 2: Who scored most goals + Pronoun follow-up",
            "messages": [
                {"question": "Who scored the most goals in 2022-23?", "chat_history": []},
                {"question": "How many points do you think he's gonna earn?", "chat_history": None}  # Will be populated
            ],
            "expected_keywords": ["Haaland", "8.0", "predict", "forecast"],
            "should_not_contain": ["36 goals", "2022-23 season"]
        },
        {
            "name": "Test 3: Forecasted points (direct query)",
            "messages": [
                {"question": "forecasted points for Haaland", "chat_history": []}
            ],
            "expected_keywords": ["Haaland", "8.0", "predict"],
            "should_not_contain": ["not explicitly listed", "Xavi Simons"]
        },
        {
            "name": "Test 4: Next gameweek predictions (general)",
            "messages": [
                {"question": "Who will perform well next gameweek?", "chat_history": []}
            ],
            "expected_keywords": ["predict", "next", "gameweek"],
            "should_not_contain": []
        },
        {
            "name": "Test 5: Context-aware follow-up (Haaland reference)",
            "messages": [
                {"question": "Tell me about Erling Haaland", "chat_history": []},
                {"question": "Will he score next week?", "chat_history": None}  # Will be populated
            ],
            "expected_keywords": ["predict", "Haaland", "8.0"],
            "should_not_contain": []
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*80}")
        print(f"{scenario['name']}")
        print(f"{'='*80}")
        
        chat_history = []
        last_answer = None
        
        for idx, msg in enumerate(scenario['messages']):
            # Use accumulated chat history for follow-up questions
            if msg['chat_history'] is None:
                msg['chat_history'] = chat_history
            
            print(f"\n  Message {idx + 1}: {msg['question']}")
            print(f"  Chat history: {len(msg['chat_history'])} messages")
            
            try:
                response = requests.post(
                    f"{base_url}/api/query",
                    json={
                        "question": msg['question'],
                        "model": "qwen-2.5-coder",
                        "retrieval_method": "Hybrid",
                        "embedding_model": "minilm",
                        "chat_history": msg['chat_history']
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get('answer', '')
                    intent = data.get('intent', '')
                    
                    print(f"  Intent: {intent}")
                    print(f"  Answer: {answer[:200]}...")
                    
                    # Update chat history for next message
                    chat_history.append({"role": "user", "content": msg['question']})
                    chat_history.append({"role": "assistant", "content": answer})
                    last_answer = answer
                else:
                    print(f"  ❌ API Error: {response.status_code}")
                    print(f"  {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                print(f"  ❌ Request timed out")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Check final answer against expectations
        if last_answer:
            print(f"\n  Final answer validation:")
            passed = True
            for keyword in scenario['expected_keywords']:
                if keyword.lower() in last_answer.lower():
                    print(f"    ✅ Found: '{keyword}'")
                else:
                    print(f"    ❌ Missing expected keyword: '{keyword}'")
                    passed = False
            
            for bad_keyword in scenario['should_not_contain']:
                if bad_keyword.lower() in last_answer.lower():
                    print(f"    ❌ Should NOT contain: '{bad_keyword}'")
                    passed = False
                else:
                    print(f"    ✅ Correctly absent: '{bad_keyword}'")
            
            if passed:
                print(f"  ✅ SCENARIO PASSED")
            else:
                print(f"  ❌ SCENARIO FAILED")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    test_rag_predictions()
