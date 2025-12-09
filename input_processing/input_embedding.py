from sentence_transformers import SentenceTransformer
import numpy as np

class AllMiniEmbedder:
    def __init__(self):
        # Load the Nomic Embed Text model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def encode(self, text):
        """Encode text into embedding"""
        return self.model.encode(text)