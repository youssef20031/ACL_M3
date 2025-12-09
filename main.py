from input_processing import IntentClassifier, EntityExtractor
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Example usage
if __name__ == "__main__":
    # Test with sample text
    text = "Get most transferred in players in gameweek 10 2021-22"
    
    # Find Intent
    classifier = IntentClassifier()
    intent = classifier.classify(text)
    print(f"Intent: {intent}")  # Output should be "
    
    # Extract entities
    extractor = EntityExtractor(text)
    entities = extractor.extract_entities()
    
    # Print only the entities (not the extra text)
    print(entities)
