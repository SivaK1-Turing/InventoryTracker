"""
Cache implementation for InventoryTracker analytics.
Provides caching for analytics results with a 10-minute expiration,
and mechanisms to invalidate cache entries based on transaction events.
"""

import functools
import time
import threading
import logging
from typing import Dict, Any, Callable, Optional, Tuple, List, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analytics_cache")

class AnalyticsCache:
    """
    Cache for analytics results with time-based expiration and transaction-based invalidation.
    """
    
    def __init__(self, expiration_seconds: int = 600):  # Default 10 minutes
        """
        Initialize the analytics cache.
        
        Args:
            expiration_seconds: Time in seconds before cache entries expire
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._expiration_seconds = expiration_seconds
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Track which product_ids are in which cache keys for efficient invalidation
        self._product_key_mapping: Dict[str, Set[str]] = {}
        
        logger.info(f"Initialized AnalyticsCache with {expiration_seconds}s expiration")
        
    def _generate_key(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """
        Generate a cache key from function name and arguments.
        
        Args:
            func_name: Name of the cached function
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            A string key representing the function call
        """
        # Convert args and kwargs to strings
        args_str = ",".join(str(arg) for arg in args)
        kwargs_str = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        # Combine into a single key
        return f"{func_name}({args_str},{kwargs_str})"
        
    def _extract_product_id(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Extract product_id from function arguments.
        
        Args:
            func_name: Name of the function being called
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            product_id if found, None otherwise
        """
        # Most analytics functions have product_id as their first argument
        if args and args[0]:
            return str(args[0])
            
        # Check if product_id is in kwargs
        if "product_id" in kwargs:
            return str(kwargs["product_id"])
            
        return None
        
    def cache(self, func: Callable) -> Callable:
        """
        Decorator to cache function results.
        
        Args:
            func: The function to cache
            
        Returns:
            Wrapped function that uses the cache
        """
        func_name = func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = self._generate_key(func_name, args, kwargs)
            product_id = self._extract_product_id(func_name, args, kwargs)
            
            with self._lock:
                # Check if we have a valid cached result
                if cache_key in self._cache:
                    entry = self._cache[cache_key]
                    current_time = time.time()
                    
                    # Check if entry is still valid
                    if current_time - entry["timestamp"] < self._expiration_seconds:
                        logger.debug(f"Cache hit for {cache_key}")
                        return entry["result"]
                    else:
                        # Remove expired entry
                        logger.debug(f"Cache expired for {cache_key}")
                        self._remove_cache_entry(cache_key)
                        
            # No valid cache entry, call the function
            result = func(*args, **kwargs)
            
            # Cache the result
            with self._lock:
                self._cache[cache_key] = {
                    "result": result,
                    "timestamp": time.time(),
                    "product_id": product_id
                }
                
                # Update product_id to cache key mapping for invalidation
                if product_id:
                    if product_id not in self._product_key_mapping:
                        self._product_key_mapping[product_id] = set()
                    self._product_key_mapping[product_id].add(cache_key)
                    
                logger.debug(f"Cached result for {cache_key}")
            
            return result
            
        return wrapper
        
    def _remove_cache_entry(self, cache_key: str) -> None:
        """
        Remove a cache entry and update mappings.
        
        Args:
            cache_key: The cache key to remove
        """
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            product_id = entry.get("product_id")
            
            # Remove the entry
            del self._cache[cache_key]
            
            # Update the product_id mapping
            if product_id and product_id in self._product_key_mapping:
                self._product_key_mapping[product_id].discard(cache_key)
                # Clean up empty sets
                if not self._product_key_mapping[product_id]:
                    del self._product_key_mapping[product_id]
    
    def invalidate_for_product(self, product_id: str) -> int:
        """
        Invalidate all cache entries for a specific product_id.
        
        Args:
            product_id: The product_id to invalidate cache for
            
        Returns:
            Number of cache entries invalidated
        """
        invalidated = 0
        product_id = str(product_id)  # Ensure string type
        
        with self._lock:
            if product_id in self._product_key_mapping:
                # Get all cache keys for this product_id
                cache_keys = list(self._product_key_mapping[product_id])
                
                # Remove each cache entry
                for cache_key in cache_keys:
                    self._remove_cache_entry(cache_key)
                    invalidated += 1
                    
                logger.info(f"Invalidated {invalidated} cache entries for product_id {product_id}")
                
        return invalidated
        
    def invalidate_all(self) -> int:
        """
        Invalidate all cache entries.
        
        Returns:
            Number of cache entries invalidated
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._product_key_mapping.clear()
            logger.info(f"Invalidated all {count} cache entries")
            return count
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                "entries": len(self._cache),
                "products": len(self._product_key_mapping),
                "expiration_seconds": self._expiration_seconds,
                "memory_usage_estimate_bytes": len(self._cache) * 256  # Rough estimate
            }


# Create a singleton instance of the analytics cache
analytics_cache = AnalyticsCache(expiration_seconds=600)  # 10 minutes