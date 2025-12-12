
import unittest
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from embeddings.embedding_manager import EmbeddingManager
from graph.queries import CypherQueries

class TestEmbeddingContent(unittest.TestCase):
    
    def setUp(self):
        # Initialize manager with mock model to avoid loading heavy weights
        # We only test string generation, so model doesn't matter much
        # But EmbeddingManager loads model in __init__. We can mock it or just let it load minilm
        self.manager = EmbeddingManager(model_key="minilm")
        # Mock the model to speed up if needed, but minilm is fast.
        
    def test_description_high_ict(self):
        data = {
            "name": "Erling Haaland",
            "position": "FWD",
            "season": "2022-23",
            "total_points": 272,
            "goals": 36,
            "assists": 8,
            "ict_index": 10.2,
            "influence": 40.0,
            "creativity": 10.0,
            "threat": 50.0,
            "value": 120,
            "games": 35,
            "selected": 8000000,
            "teams": ["Manchester City", "Manchester City", "Manchester City"]
        }
        desc = self.manager.create_player_description(data)
        print(f"\nDesc High ICT: {desc}")
        
        self.assertIn("playing for Manchester City", desc)
        self.assertIn("showing elite underlying stats and high ICT index", desc)
        self.assertIn("influence: 40.0", desc)
        self.assertIn("threat: 50.0", desc)
        self.assertIn("delivering elite returns", desc)

    def test_description_goalkeeper(self):
        data = {
            "name": "David Raya",
            "position": "GK",
            "season": "2022-23",
            "total_points": 150,
            "clean_sheets": 12,
            "saves": 154,
            "bonus": 20,
            "goals_conceded": 40,
            "games": 38
        }
        desc = self.manager.create_player_description(data)
        print(f"\nDesc GK: {desc}")
        
        self.assertIn("154 saves", desc)
        self.assertIn("conceding 40 goals", desc)

    def test_description_discipline(self):
        data = {
            "name": "Bad Boy",
            "position": "MID",
            "season": "2022-23",
            "total_points": 50,
            "yellow_cards": 10,
            "red_cards": 1,
            "games": 30
        }
        desc = self.manager.create_player_description(data)
        print(f"\nDesc Discipline: {desc}")
        
        self.assertIn("receiving 10 yellow cards (high disciplinary risk)", desc)
        self.assertIn("receiving 1 red cards", desc)
        
    def test_query_fields_exist(self):
        # Verify queries.py has the new fields in the query string
        query, _ = CypherQueries.get_player_embeddings_data()
        self.assertIn("saves", query)
        self.assertIn("goals_conceded", query)
        self.assertIn("yellow_cards", query)
        self.assertIn("bps", query)

if __name__ == '__main__':
    unittest.main()
