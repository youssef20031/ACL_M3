"""
Prediction Cache Service
Provides caching functionality for ML predictions with dynamic TTL.
"""
from typing import Optional, Any, Dict, Tuple
from datetime import datetime, timezone, timedelta
import logging
import pickle
import time

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Using in-memory cache fallback.")


class PredictionCache:
    """
    Cache for ML predictions with dynamic TTL based on fixture timing.
    
    Supports both Redis and in-memory caching.
    """
    
    def __init__(self, redis_client: Optional['redis.Redis'] = None):
        """
        Initialize prediction cache.
        
        Args:
            redis_client: Optional Redis client. If None, uses in-memory cache.
        """
        self.redis = redis_client if REDIS_AVAILABLE else None
        self.memory_cache: Dict[str, Tuple[Any, float]] = {}
        
        if self.redis:
            logger.info("PredictionCache initialized with Redis backend")
        else:
            logger.info("PredictionCache initialized with in-memory backend")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found or expired
        """
        if self.redis:
            try:
                data = self.redis.get(key)
                if data:
                    logger.debug(f"Cache HIT: {key}")
                    return pickle.loads(data)
                else:
                    logger.debug(f"Cache MISS: {key}")
                    return None
            except Exception as e:
                logger.error(f"Redis error on get: {e}. Falling back to in-memory cache.")
                # Fall through to in-memory cache
        
        # In-memory cache
        if key in self.memory_cache:
            value, expires_at = self.memory_cache[key]
            if time.time() < expires_at:
                logger.debug(f"Cache HIT (memory): {key}")
                return value
            else:
                # Expired - remove it
                del self.memory_cache[key]
                logger.debug(f"Cache MISS (expired): {key}")
        else:
            logger.debug(f"Cache MISS (memory): {key}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: int):
        """
        Store value in cache with TTL.
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds
            value: Value to cache
        """
        if ttl <= 0:
            logger.debug(f"Skipping cache (TTL={ttl}): {key}")
            return
        
        if self.redis:
            try:
                self.redis.setex(key, ttl, pickle.dumps(value))
                logger.debug(f"Cached in Redis: {key} (TTL={ttl}s)")
                return
            except Exception as e:
                logger.error(f"Redis error on set: {e}. Falling back to in-memory cache.")
                # Fall through to in-memory cache
        
        # In-memory cache
        expires_at = time.time() + ttl
        self.memory_cache[key] = (value, expires_at)
        logger.debug(f"Cached in memory: {key} (TTL={ttl}s)")
    
    def invalidate(self, key: str):
        """
        Invalidate (delete) a cached value.
        
        Args:
            key: Cache key to invalidate
        """
        if self.redis:
            try:
                self.redis.delete(key)
                logger.debug(f"Invalidated from Redis: {key}")
            except Exception as e:
                logger.error(f"Redis error on delete: {e}")
        
        # Also remove from in-memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            logger.debug(f"Invalidated from memory: {key}")
    
    def invalidate_pattern(self, pattern: str):
        """
        Invalidate all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., "prediction:*:fixture-123")
        """
        if self.redis:
            try:
                # Use SCAN to find keys matching pattern
                cursor = 0
                while True:
                    cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                    if keys:
                        self.redis.delete(*keys)
                        logger.info(f"Invalidated {len(keys)} keys matching pattern: {pattern}")
                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Redis error on pattern delete: {e}")
        
        # In-memory cache: delete matching keys
        keys_to_delete = [k for k in self.memory_cache.keys() if self._matches_pattern(k, pattern)]
        for key in keys_to_delete:
            del self.memory_cache[key]
        
        if keys_to_delete:
            logger.info(f"Invalidated {len(keys_to_delete)} keys from memory matching pattern: {pattern}")
    
    @staticmethod
    def _matches_pattern(key: str, pattern: str) -> bool:
        """
        Simple pattern matching for in-memory cache.
        
        Supports basic wildcards: * matches any sequence.
        """
        if '*' not in pattern:
            return key == pattern
        
        # Convert pattern to regex-like matching
        import re
        regex_pattern = pattern.replace('*', '.*')
        return bool(re.match(f"^{regex_pattern}$", key))
    
    def clear_all(self):
        """Clear all cached values."""
        if self.redis:
            try:
                self.redis.flushdb()
                logger.warning("Cleared all Redis cache")
            except Exception as e:
                logger.error(f"Redis error on flush: {e}")
        
        self.memory_cache.clear()
        logger.warning("Cleared all in-memory cache")
    
    @staticmethod
    def calculate_cache_ttl(fixture_kickoff: datetime) -> int:
        """
        Calculate cache TTL in seconds based on fixture timing.
        
        Logic:
        - If fixture > 7 days away: TTL = 24 hours
        - If fixture 1-7 days away: TTL = 6 hours
        - If fixture < 1 day away: TTL = 1 hour
        - After kickoff: TTL = 0 (no caching)
        
        Args:
            fixture_kickoff: Fixture kickoff datetime (UTC)
        
        Returns:
            TTL in seconds
        """
        # Ensure fixture_kickoff is timezone-aware
        if fixture_kickoff.tzinfo is None:
            fixture_kickoff = fixture_kickoff.replace(tzinfo=timezone.utc)
        
        time_until_kickoff = fixture_kickoff - datetime.now(timezone.utc)
        days = time_until_kickoff.days
        
        if time_until_kickoff.total_seconds() <= 0:
            # Fixture has already kicked off - no caching
            return 0
        elif days > 7:
            # More than 7 days away: cache for 24 hours
            return 86400  # 24 * 60 * 60
        elif days >= 1:
            # 1-7 days away: cache for 6 hours
            return 21600  # 6 * 60 * 60
        else:
            # Less than 1 day away: cache for 1 hour
            return 3600   # 1 * 60 * 60
    
    @staticmethod
    def generate_prediction_key(player_id: str, fixture_id: str, model_version: str) -> str:
        """
        Generate cache key for player prediction.
        
        Key format: prediction:{player_id}:{fixture_id}:{model_version}
        
        Args:
            player_id: Player identifier (can be name or ID)
            fixture_id: Fixture identifier
            model_version: Model version string
        
        Returns:
            Cache key string
        """
        return f"prediction:{player_id}:{fixture_id}:{model_version}"


def create_prediction_cache(redis_url: Optional[str] = None) -> PredictionCache:
    """
    Factory function to create PredictionCache with optional Redis.
    
    Args:
        redis_url: Optional Redis connection URL (e.g., "redis://localhost:6379/0")
    
    Returns:
        PredictionCache instance
    """
    redis_client = None
    
    if redis_url and REDIS_AVAILABLE:
        try:
            import redis
            redis_client = redis.from_url(redis_url)
            # Test connection
            redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache.")
            redis_client = None
    
    return PredictionCache(redis_client)
