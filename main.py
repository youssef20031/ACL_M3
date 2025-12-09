from input_processing.intent_classification import IntentClassifier

def process_input(input_text):
    classifier = IntentClassifier()
    intent = classifier.classify(input_text)
    return intent

if __name__ == "__main__":
    user_input = ""
    result = process_input(user_input)
    print(f"Classified Intent: {result}")