"""
Embedding Manager for FPL Graph-RAG System
Handles text-constructed embeddings and semantic similarity search
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Container for embedding search results."""
    player_name: str
    similarity_score: float
    metadata: Dict[str, Any]


class EmbeddingManager:
    """
    Manages embeddings for FPL players using text-constructed representations.
    Supports comparison between different embedding models.
    """
    
    # Available embedding models
    MODELS = {
        "minilm": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "description": "Fast, lightweight model - good for quick searches"
        },
        "mpnet": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "dimension": 768,
            "description": "Higher quality embeddings - better semantic understanding"
        }
    }
    
    def __init__(self, model_key: str = "minilm"):
        """
        Initialize embedding manager with specified model.
        
        Args:
            model_key: Key for model selection ('minilm' or 'mpnet')
        """
        if model_key not in self.MODELS:
            raise ValueError(f"Unknown model: {model_key}. Choose from {list(self.MODELS.keys())}")
        
        self.model_key = model_key
        self.model_info = self.MODELS[model_key]
        self.model = None
        self.player_embeddings: Dict[str, np.ndarray] = {}
        self.player_metadata: Dict[str, Dict] = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        logger.info(f"Loading embedding model: {self.model_info['name']}")
        try:
            self.model = SentenceTransformer(self.model_info['name'])
            logger.info(f"Model loaded successfully. Dimension: {self.model_info['dimension']}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def switch_model(self, model_key: str):
        """
        Switch to a different embedding model.
        Note: This clears existing embeddings.
        
        Args:
            model_key: Key for new model
        """
        if model_key == self.model_key:
            return
        
        logger.info(f"Switching model from {self.model_key} to {model_key}")
        self.model_key = model_key
        self.model_info = self.MODELS[model_key]
        self.player_embeddings = {}  # Clear cached embeddings
        self._load_model()
    
    def create_player_description(self, player_data: Dict[str, Any]) -> str:
        """
        Create a text description of a player for embedding.
        Constructs natural language from numerical stats.
        
        Args:
            player_data: Dictionary with player statistics
            
        Returns:
            Text description suitable for embedding
        """
        name = player_data.get("name", "Unknown Player")
        position = player_data.get("position", "")
        season = player_data.get("season", "")
        
        # Get stats with defaults
        total_points = player_data.get("total_points", 0)
        goals = player_data.get("goals_scored", player_data.get("goals", 0))
        assists = player_data.get("assists", 0)
        clean_sheets = player_data.get("clean_sheets", 0)
        bonus = player_data.get("bonus", 0)
        minutes = player_data.get("minutes", 0)
        ict_index = player_data.get("ict_index", player_data.get("avg_ict", 0))
        influence = player_data.get("influence", player_data.get("avg_influence", 0))
        creativity = player_data.get("creativity", player_data.get("avg_creativity", 0))
        threat = player_data.get("threat", player_data.get("avg_threat", 0))
        value = player_data.get("value", player_data.get("avg_value", 0))
        selected = player_data.get("selected", player_data.get("max_selected", 0))
        games = player_data.get("games", player_data.get("games_played", 0))
        
        # Map position codes to names
        position_names = {
            "GK": "goalkeeper",
            "DEF": "defender",
            "MID": "midfielder",
            "FWD": "forward"
        }
        pos_name = position_names.get(position, "player")
        
        # Build description
        parts = [f"{name} is a Premier League {pos_name}"]
        
        if season:
            parts.append(f"playing in the {season} season")
        
        parts.append(f"who scored {total_points} FPL points")
        
        if games > 0:
            ppg = total_points / games
            parts.append(f"averaging {ppg:.1f} points per game over {games} appearances")
        
        # Position-specific stats
        if position in ["FWD", "MID"]:
            parts.append(f"with {goals} goals and {assists} assists")
            if threat > 0:
                parts.append(f"showing high threat ({threat:.1f})")
        elif position == "DEF":
            parts.append(f"keeping {clean_sheets} clean sheets and providing {assists} assists")
        elif position == "GK":
            saves = player_data.get("saves", 0)
            parts.append(f"with {clean_sheets} clean sheets and {saves} saves")
        
        # General performance metrics
        if bonus > 0:
            parts.append(f"earning {bonus} bonus points")
        
        if ict_index > 0:
            parts.append(f"with an ICT index of {ict_index:.1f}")
            if creativity > 0:
                parts.append(f"(creativity: {creativity:.1f}, influence: {influence:.1f})")
        
        # Value information
        if value > 0:
            value_mil = value / 10 if value > 10 else value
            parts.append(f"valued at £{value_mil:.1f}m")
        
        # Popularity
        if selected > 100000:
            parts.append(f"selected by {selected:,} managers")
        
        description = ". ".join(parts) + "."
        return description
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into embedding vector.
        
        Args:
            text: Text to encode
            
        Returns:
            Numpy array with embedding
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts into embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Numpy array with embeddings (shape: num_texts x dimension)
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def build_player_embeddings(self, players_data: List[Dict[str, Any]]):
        """
        Build embeddings for all players.
        
        Args:
            players_data: List of player data dictionaries
        """
        logger.info(f"Building embeddings for {len(players_data)} players...")
        
        descriptions = []
        player_keys = []
        
        for player in players_data:
            name = player.get("name", "Unknown")
            season = player.get("season", "")
            key = f"{name}_{season}" if season else name
            
            description = self.create_player_description(player)
            descriptions.append(description)
            player_keys.append(key)
            
            # Store metadata
            self.player_metadata[key] = {
                "name": name,
                "season": season,
                "description": description,
                **player
            }
        
        # Batch encode all descriptions
        embeddings = self.encode_batch(descriptions)
        
        # Store embeddings
        for key, embedding in zip(player_keys, embeddings):
            self.player_embeddings[key] = embedding
        
        logger.info(f"Built {len(self.player_embeddings)} player embeddings")
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def find_similar_players(
        self, 
        query: str, 
        top_k: int = 5,
        season_filter: Optional[str] = None,
        position_filter: Optional[str] = None
    ) -> List[EmbeddingResult]:
        """
        Find players similar to a text query.
        
        Args:
            query: Search query (player name or description)
            top_k: Number of results to return
            season_filter: Optional season to filter by
            position_filter: Optional position to filter by
            
        Returns:
            List of similar players with scores
        """
        if not self.player_embeddings:
            logger.warning("No player embeddings available. Build embeddings first.")
            return []
        
        # Encode query
        query_embedding = self.encode_text(query)
        
        # Calculate similarities
        similarities = []
        
        for key, embedding in self.player_embeddings.items():
            metadata = self.player_metadata[key]
            
            # Apply filters
            if season_filter and metadata.get("season") != season_filter:
                continue
            if position_filter and metadata.get("position") != position_filter:
                continue
            
            similarity = self.compute_similarity(query_embedding, embedding)
            similarities.append(EmbeddingResult(
                player_name=metadata["name"],
                similarity_score=similarity,
                metadata=metadata
            ))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_k]
    
    def find_similar_to_player(
        self,
        player_name: str,
        season: str = "2022-23",
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[EmbeddingResult]:
        """
        Find players similar to a given player.
        
        Args:
            player_name: Name of the reference player
            season: Season to use
            top_k: Number of results
            exclude_self: Whether to exclude the query player
            
        Returns:
            List of similar players
        """
        key = f"{player_name}_{season}"
        
        if key not in self.player_embeddings:
            # Try without season
            matching_keys = [k for k in self.player_embeddings.keys() if k.startswith(player_name)]
            if matching_keys:
                key = matching_keys[0]
            else:
                logger.warning(f"Player not found: {player_name}")
                return []
        
        player_embedding = self.player_embeddings[key]
        player_position = self.player_metadata[key].get("position")
        
        # Find similar players (optionally same position)
        similarities = []
        
        for other_key, embedding in self.player_embeddings.items():
            if exclude_self and other_key == key:
                continue
            
            metadata = self.player_metadata[other_key]
            
            similarity = self.compute_similarity(player_embedding, embedding)
            similarities.append(EmbeddingResult(
                player_name=metadata["name"],
                similarity_score=similarity,
                metadata=metadata
            ))
        
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:top_k]
    
    def get_embedding_for_neo4j(self, player_name: str, season: str = "") -> Optional[List[float]]:
        """
        Get embedding as a list for Neo4j storage.
        
        Args:
            player_name: Player name
            season: Season
            
        Returns:
            Embedding as list of floats, or None if not found
        """
        key = f"{player_name}_{season}" if season else player_name
        
        if key in self.player_embeddings:
            return self.player_embeddings[key].tolist()
        
        # Try without season
        for k, v in self.player_embeddings.items():
            if k.startswith(player_name):
                return v.tolist()
        
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_key": self.model_key,
            "model_name": self.model_info["name"],
            "dimension": self.model_info["dimension"],
            "description": self.model_info["description"],
            "num_embeddings": len(self.player_embeddings)
        }
    
    def compare_models(
        self, 
        query: str, 
        players_data: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> Dict[str, List[EmbeddingResult]]:
        """
        Compare search results between different embedding models.
        
        Args:
            query: Search query
            players_data: Player data to embed
            top_k: Number of results per model
            
        Returns:
            Dictionary mapping model name to results
        """
        results = {}
        original_model = self.model_key
        
        for model_key in self.MODELS.keys():
            logger.info(f"Testing model: {model_key}")
            self.switch_model(model_key)
            self.build_player_embeddings(players_data)
            model_results = self.find_similar_players(query, top_k=top_k)
            results[model_key] = model_results
        
        # Restore original model
        self.switch_model(original_model)
        
        return results


def create_numerical_embedding(player_stats: Dict[str, Any]) -> np.ndarray:
    """
    Create a numerical embedding from player statistics.
    Alternative to text-based embeddings.
    
    Args:
        player_stats: Dictionary with player statistics
        
    Returns:
        Normalized numerical embedding
    """
    # Define features and their normalization ranges
    features = [
        ("total_points", 0, 300),
        ("goals_scored", 0, 30),
        ("assists", 0, 20),
        ("clean_sheets", 0, 20),
        ("bonus", 0, 50),
        ("bps", 0, 1000),
        ("minutes", 0, 3500),
        ("ict_index", 0, 15),
        ("influence", 0, 100),
        ("creativity", 0, 100),
        ("threat", 0, 100),
        ("value", 40, 150),  # £4m to £15m
        ("form", 0, 10),
        ("selected", 0, 5000000),
    ]
    
    embedding = []
    for feature, min_val, max_val in features:
        value = player_stats.get(feature, 0)
        # Normalize to 0-1 range
        if max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
            normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
        else:
            normalized = 0
        embedding.append(normalized)
    
    return np.array(embedding)
