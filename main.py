from pprint import pprint
from input_processing import IntentClassifier, EntityExtractor
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Example usage
if __name__ == "__main__":
    # Test with sample text
    text = "Get most transferred in players in gameweek 10 2021-2022"
    
    # Find Intent
    classifier = IntentClassifier()
    intent = classifier.classify(text)
    print(f"Intent: {intent}")  # Output should be "
    
    # Extract entities
    extractor = EntityExtractor()
    entities = extractor.extract(text)
    
    # Print only the entities (not the extra text)
    pprint(entities.to_dict())
