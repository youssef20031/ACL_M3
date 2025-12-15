"""Quick script to check Neo4j data status"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from input_preprocessing.entity_extractor import EntityExtractor

# Test the entity extraction
extractor = EntityExtractor()
query = "Get comprehensive stats for Andrew Robertson across all seasons"

entities = extractor.extract(query)
params = extractor.get_query_parameters(entities)

print(f"Query: {query}")
print(f"\nExtracted entities:")
print(f"  Players: {entities.players}")
print(f"  Seasons: {entities.seasons}")
print(f"\nQuery parameters:")
for k, v in params.items():
    print(f"  {k}: {v}")

print(f"\n'season' in params: {'season' in params}")
