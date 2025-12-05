"""
Neo4j Database Connection Handler
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Manages Neo4j database connections and query execution."""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j database URI (e.g., bolt://localhost:7687)
            user: Database username
            password: Database password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # Verify connection
            self.driver.verify_connectivity()
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters dictionary
            
        Returns:
            List of result records as dictionaries
        """
        if parameters is None:
            parameters = {}
            
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                records = [record.data() for record in result]
                return records
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise
    
    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a write transaction.
        
        Args:
            query: Cypher query string
            parameters: Query parameters dictionary
            
        Returns:
            Query result summary
        """
        if parameters is None:
            parameters = {}
            
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                summary = result.consume()
                return summary
        except Exception as e:
            logger.error(f"Write execution failed: {e}")
            raise
    
    def execute_batch(self, query: str, batch_data: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """
        Execute batch operations for bulk data loading.
        
        Args:
            query: Cypher query string with $batch parameter
            batch_data: List of data dictionaries
            batch_size: Number of records per batch
            
        Returns:
            Total number of records processed
        """
        total_processed = 0
        
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            try:
                with self.driver.session() as session:
                    session.run(query, {"batch": batch})
                    total_processed += len(batch)
                    logger.info(f"Processed {total_processed}/{len(batch_data)} records")
            except Exception as e:
                logger.error(f"Batch execution failed at index {i}: {e}")
                raise
        
        return total_processed
    
    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        query = "MATCH (n) DETACH DELETE n"
        self.execute_write(query)
        logger.info("Database cleared")
    
    def get_node_count(self, label: Optional[str] = None) -> int:
        """
        Get count of nodes, optionally filtered by label.
        
        Args:
            label: Node label to filter by
            
        Returns:
            Count of nodes
        """
        if label:
            query = f"MATCH (n:{label}) RETURN count(n) as count"
        else:
            query = "MATCH (n) RETURN count(n) as count"
        
        result = self.execute_query(query)
        return result[0]["count"] if result else 0
    
    def get_relationship_count(self, rel_type: Optional[str] = None) -> int:
        """
        Get count of relationships, optionally filtered by type.
        
        Args:
            rel_type: Relationship type to filter by
            
        Returns:
            Count of relationships
        """
        if rel_type:
            query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
        else:
            query = "MATCH ()-[r]->() RETURN count(r) as count"
        
        result = self.execute_query(query)
        return result[0]["count"] if result else 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics including node and relationship counts.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {
            "total_nodes": self.get_node_count(),
            "total_relationships": self.get_relationship_count(),
            "node_labels": {},
            "relationship_types": {}
        }
        
        # Get counts by label
        labels_query = "CALL db.labels() YIELD label RETURN label"
        labels = self.execute_query(labels_query)
        for label_record in labels:
            label = label_record["label"]
            stats["node_labels"][label] = self.get_node_count(label)
        
        # Get counts by relationship type
        rels_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        rel_types = self.execute_query(rels_query)
        for rel_record in rel_types:
            rel_type = rel_record["relationshipType"]
            stats["relationship_types"][rel_type] = self.get_relationship_count(rel_type)
        
        return stats
    
    def test_connection(self) -> bool:
        """
        Test if the database connection is working.
        
        Returns:
            True if connection is successful
        """
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
