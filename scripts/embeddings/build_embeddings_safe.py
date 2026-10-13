"""
ULTRA-SAFE Embedding Builder
Builds embeddings with extreme caution to prevent PC crashes
- Processes only 50 players at a time
- Saves checkpoints
- Can resume from crashes
"""
import os
import sys
import gc
import time
import pickle
from dotenv import load_dotenv

load_dotenv()

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from graph.connection import Neo4jConnection
from graph.queries import CypherQueries
from embeddings.embedding_manager import EmbeddingManager

CHECKPOINT_DIR = "embeddings/checkpoints"
OUTPUT_DIR = "embeddings/prebuilt"

def save_checkpoint(data, checkpoint_num):
    """Save checkpoint in case of crash."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_num}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ Checkpoint {checkpoint_num} saved")

def load_checkpoint(checkpoint_num):
    """Load checkpoint if it exists."""
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_num}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def build_embeddings_ultra_safe(model_key="minilm", chunk_size=50, start_from=0):
    """
    Build embeddings in ultra-safe mode.
    
    Args:
        model_key: "minilm" or "mpnet"
        chunk_size: Number of players per chunk (50 is safe)
        start_from: Chunk number to resume from
    """
    print("=" * 60)
    print("🛡️  ULTRA-SAFE EMBEDDING BUILDER")
    print("=" * 60)
    print(f"Model: {model_key}")
    print(f"Chunk size: {chunk_size} players at a time")
    print(f"Starting from chunk: {start_from}")
    print()
    
    # Connect to Neo4j
    print("📡 Connecting to Neo4j...")
    try:
        conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        if not conn.test_connection():
            print("❌ Failed to connect to Neo4j")
            return False
        print("✅ Connected\n")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Get player data
    print("📊 Fetching player data...")
    try:
        query, query_params = CypherQueries.get_player_embeddings_data()
        all_players = conn.execute_query(query, query_params)
        print(f"✅ Retrieved {len(all_players)} players\n")
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        conn.close()
        return False
    
    if not all_players:
        print("❌ No player data found")
        conn.close()
        return False
    
    # Calculate chunks
    total_chunks = (len(all_players) + chunk_size - 1) // chunk_size
    print(f"📦 Split into {total_chunks} chunks of {chunk_size} players each\n")
    
    # Initialize or load previous progress
    all_embeddings = {}
    all_metadata = {}
    
    # Check for resume
    if start_from > 0:
        print(f"🔄 Attempting to resume from chunk {start_from}...")
        prev_checkpoint = load_checkpoint(start_from - 1)
        if prev_checkpoint:
            all_embeddings = prev_checkpoint['embeddings']
            all_metadata = prev_checkpoint['metadata']
            print(f"✅ Resumed with {len(all_embeddings)} existing embeddings\n")
        else:
            print(f"⚠️  No checkpoint found, starting fresh\n")
            start_from = 0
    
    # Initialize model once
    print(f"🤖 Loading {model_key} model (this may take 30 seconds)...")
    try:
        manager = EmbeddingManager(model_key=model_key)
        print("✅ Model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        conn.close()
        return False
    
    # Process chunks
    for chunk_num in range(start_from, total_chunks):
        start_idx = chunk_num * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_players))
        chunk_players = all_players[start_idx:end_idx]
        
        print(f"⚙️  Chunk {chunk_num + 1}/{total_chunks} ({len(chunk_players)} players)")
        print(f"   Processing players {start_idx + 1} to {end_idx}...")
        
        try:
            # Build embeddings for this chunk ONLY
            # Use batch_size=4 for extreme safety
            manager.build_player_embeddings(chunk_players, batch_size=4)
            
            # Merge into main collection
            all_embeddings.update(manager.player_embeddings)
            all_metadata.update(manager.player_metadata)
            
            print(f"   ✅ Chunk complete! Total embeddings: {len(all_embeddings)}")
            
            # Save checkpoint every chunk
            save_checkpoint({
                'embeddings': all_embeddings,
                'metadata': all_metadata,
                'model_key': model_key
            }, chunk_num)
            
            # Clear chunk from memory
            manager.player_embeddings = {}
            manager.player_metadata = {}
            gc.collect()
            
            # Pause to let system recover
            time.sleep(2)
            print()
            
        except Exception as e:
            print(f"   ❌ Chunk failed: {e}")
            print(f"   📍 Resume with: python build_embeddings_safe.py {model_key} {chunk_num}")
            conn.close()
            return False
    
    # Save final embeddings
    print("💾 Saving final embeddings...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{model_key}_embeddings.pkl")
    
    try:
        final_data = {
            'model_key': model_key,
            'embeddings': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in all_embeddings.items()},
            'metadata': all_metadata,
        }
        with open(output_path, 'wb') as f:
            pickle.dump(final_data, f)
        
        file_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"✅ Saved {len(all_embeddings)} embeddings to {output_path}")
        print(f"   File size: {file_size:.1f} MB\n")
        
    except Exception as e:
        print(f"❌ Failed to save: {e}")
        conn.close()
        return False
    
    # Cleanup
    conn.close()
    
    # Clean up checkpoints
    print("🧹 Cleaning up checkpoints...")
    for i in range(total_chunks):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{i}.pkl")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
    
    print("=" * 60)
    print("🎉 SUCCESS! Embeddings ready for deployment")
    print("=" * 60)
    print(f"\n📦 Next steps:")
    print(f"   1. git add {output_path}")
    print(f"   2. git commit -m 'Add prebuilt {model_key} embeddings'")
    print(f"   3. git push origin main")
    print(f"   4. Railway will load these automatically!")
    
    return True


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "minilm"
    start_chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    if model not in ["minilm", "mpnet"]:
        print(f"❌ Unknown model: {model}")
        print("Usage: python build_embeddings_safe.py [minilm|mpnet] [start_chunk]")
        sys.exit(1)
    
    try:
        success = build_embeddings_ultra_safe(model, chunk_size=50, start_from=start_chunk)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print(f"Resume with: python build_embeddings_safe.py {model} [chunk_number]")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
