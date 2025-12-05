"""
Standalone script to load FPL data into Neo4j
Run this script to populate the database before using the app
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_PATH
from graph.connection import Neo4jConnection
from graph.data_loader import FPLDataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Load FPL data into Neo4j database."""
    print("=" * 60)
    print("FPL FantasyTrivia - Data Loader")
    print("=" * 60)
    
    # Get connection parameters
    uri = input(f"Neo4j URI [{NEO4J_URI}]: ").strip() or NEO4J_URI
    user = input(f"Username [{NEO4J_USER}]: ").strip() or NEO4J_USER
    password = input(f"Password: ").strip() or NEO4J_PASSWORD
    
    print("\nConnecting to Neo4j...")
    
    try:
        # Connect to Neo4j
        conn = Neo4jConnection(uri, user, password)
        
        if conn.test_connection():
            print("✅ Connected successfully!")
        else:
            print("❌ Connection failed!")
            return
        
        # Confirm data loading
        print(f"\nData file: {DATA_PATH}")
        
        if os.path.exists(DATA_PATH):
            print("✅ Data file found!")
        else:
            print(f"❌ Data file not found at {DATA_PATH}")
            print("Please ensure fpl_two_seasons.csv is in the project directory.")
            return
        
        confirm = input("\nThis will clear existing data and reload. Continue? (y/n): ").strip().lower()
        
        if confirm != 'y':
            print("Cancelled.")
            return
        
        # Load data
        print("\nLoading data into Neo4j...")
        loader = FPLDataLoader(conn)
        stats = loader.load_all(DATA_PATH, clear_existing=True)
        
        print("\n" + "=" * 60)
        print("✅ Data loaded successfully!")
        print("=" * 60)
        print(f"Total nodes: {stats['total_nodes']:,}")
        print(f"Total relationships: {stats['total_relationships']:,}")
        print("\nNode counts by label:")
        for label, count in stats['node_labels'].items():
            print(f"  - {label}: {count:,}")
        
        # Close connection
        conn.close()
        print("\n✅ Done! You can now run the Streamlit app.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Data loading failed")


if __name__ == "__main__":
    main()
