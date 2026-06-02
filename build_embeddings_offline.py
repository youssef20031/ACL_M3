"""
Offline Embedding Builder
Builds embeddings locally with memory management and saves them for Railway deployment
"""
import os
import sys
import gc
import psutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from graph.connection import Neo4jConnection
from graph.queries import CypherQueries
from embeddings.embedding_manager import EmbeddingManager

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def build_embeddings_with_monitoring(model_key="minilm", output_path="embeddings/prebuilt"):
    """
    Build embeddings with memory monitoring.
    
    Args:
        model_key: "minilm" or "mpnet"
        output_path: Directory to save embeddings
    """
    print(f"🚀 Starting offline embedding build for {model_key}")
    print(f"💾 Initial memory: {get_memory_usage():.1f} MB\n")
    
    # Connect to Neo4j
    print("📡 Connecting to Neo4j...")
    conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    if not conn.test_connection():
        print("❌ Failed to connect to Neo4j")
        return False
    print("✅ Connected to Neo4j\n")
    
    # Get player data
    print("📊 Fetching player data...")
    query, query_params = CypherQueries.get_player_embeddings_data()
    players_data = conn.execute_query(query, query_params)
    print(f"✅ Retrieved {len(players_data)} players")
    print(f"💾 Memory after data fetch: {get_memory_usage():.1f} MB\n")
    
    if not players_data:
        print("❌ No player data found")
        return False
    
    # Initialize embedding manager
    print(f"🤖 Loading {model_key} model...")
    try:
        manager = EmbeddingManager(model_key=model_key)
        print(f"✅ Model loaded")
        print(f"💾 Memory after model load: {get_memory_usage():.1f} MB\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Build embeddings with small batch size to prevent crashes
    print("⚙️  Building embeddings...")
    print("   This may take 5-10 minutes depending on your system")
    print("   Batch size: 8 (reduced for memory safety)\n")
    
    try:
        # Use very small batch size to avoid memory spikes
        manager.build_player_embeddings(players_data, batch_size=8)
        print(f"\n✅ Built {len(manager.player_embeddings)} embeddings")
        print(f"💾 Memory after building: {get_memory_usage():.1f} MB\n")
    except Exception as e:
        print(f"❌ Failed to build embeddings: {e}")
        return False
    
    # Save embeddings
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, f"{model_key}_embeddings.pkl")
    
    print(f"💾 Saving embeddings to {filepath}...")
    try:
        manager.save_embeddings(filepath)
        file_size = os.path.getsize(filepath) / 1024 / 1024
        print(f"✅ Saved! File size: {file_size:.1f} MB\n")
    except Exception as e:
        print(f"❌ Failed to save embeddings: {e}")
        return False
    
    # Cleanup
    conn.close()
    del manager
    gc.collect()
    
    print(f"🎉 Success! Embeddings ready for deployment")
    print(f"💾 Final memory: {get_memory_usage():.1f} MB")
    print(f"\n📦 Next steps:")
    print(f"   1. Commit and push {filepath} to Git")
    print(f"   2. Railway will load these prebuilt embeddings on startup")
    print(f"   3. No need to build embeddings on Railway anymore!")
    
    return True


if __name__ == "__main__":
    # Check if psutil is installed
    try:
        import psutil
    except ImportError:
        print("⚠️  psutil not installed. Install it for memory monitoring:")
        print("   pip install psutil")
        print("\nContinuing without memory monitoring...\n")
    
    # Default to minilm (faster, less memory)
    model = sys.argv[1] if len(sys.argv) > 1 else "minilm"
    
    if model not in ["minilm", "mpnet"]:
        print(f"❌ Unknown model: {model}")
        print("   Usage: python build_embeddings_offline.py [minilm|mpnet]")
        sys.exit(1)
    
    success = build_embeddings_with_monitoring(model)
    sys.exit(0 if success else 1)
