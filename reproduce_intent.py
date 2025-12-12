
from preprocessing.intent_classifier import IntentClassifier, Intent

def test_intent():
    classifier = IntentClassifier()
    query = "Get the top goal scoring defenders in 2022-2023"
    result = classifier.classify(query)
    print(f"Query: '{query}'")
    print(f"Intent: {result.intent}")
    print(f"Confidence: {result.confidence}")
    print(f"Matched Patterns: {result.matched_patterns}")

    query2 = "Get top scoring players by position in a season or all seasons"
    result2 = classifier.classify(query2)
    print(f"\nQuery: '{query2}'")
    print(f"Intent: {result2.intent}")
    print(f"Confidence: {result2.confidence}")

if __name__ == "__main__":
    test_intent()
