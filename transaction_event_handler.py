"""
Transaction event handler implementation for InventoryTracker.
Handles transaction events and invalidates analytics cache entries when needed.
"""

import logging
from typing import Dict, Any, Optional, Set

from .analytics_cache import analytics_cache

logger = logging.getLogger("transaction_event_handler")

class TransactionEventHandler:
    """
    Handles transaction events to invalidate relevant cache entries.
    """
    
    def __init__(self, cache=analytics_cache):
        """
        Initialize the transaction event handler.
        
        Args:
            cache: The analytics cache instance to manage
        """
        self.cache = cache
        self.logger = logging.getLogger("transaction_event_handler")
        self.logger.info("Initialized TransactionEventHandler")
        
    def handle_transaction_event(self, event_type: str, transaction_data: Dict[str, Any]) -> None:
        """
        Process a transaction event and invalidate relevant caches.
        
        Args:
            event_type: Type of transaction event (create, update, delete, etc.)
            transaction_data: Data associated with the transaction
        """
        if not transaction_data:
            self.logger.warning(f"Received empty transaction data for event: {event_type}")
            return
            
        # Extract product_id from transaction data
        product_ids = self._extract_product_ids(transaction_data)
        
        if not product_ids:
            self.logger.warning(f"No product_id found in transaction data: {transaction_data}")
            return
            
        # Invalidate cache for each affected product
        total_invalidated = 0
        for product_id in product_ids:
            invalidated = self.cache.invalidate_for_product(product_id)
            total_invalidated += invalidated
            
        self.logger.info(f"Transaction event '{event_type}' invalidated {total_invalidated} cache entries for {len(product_ids)} products")
        
    def _extract_product_ids(self, transaction_data: Dict[str, Any]) -> Set[str]:
        """
        Extract all product_ids from transaction data.
        
        Args:
            transaction_data: Dictionary containing transaction information
            
        Returns:
            Set of product_id strings
        """
        product_ids = set()
        
        # Direct product_id field
        if "product_id" in transaction_data:
            product_ids.add(str(transaction_data["product_id"]))
            
        # Check for products list/array
        if "products" in transaction_data and isinstance(transaction_data["products"], list):
            for product in transaction_data["products"]:
                if isinstance(product, dict) and "id" in product:
                    product_ids.add(str(product["id"]))
        
        # Handle StockTransaction specific format
        if "data" in transaction_data and isinstance(transaction_data["data"], dict):
            if "product_id" in transaction_data["data"]:
                product_ids.add(str(transaction_data["data"]["product_id"]))
                
            # Check for items array with product IDs
            if "items" in transaction_data["data"] and isinstance(transaction_data["data"]["items"], list):
                for item in transaction_data["data"]["items"]:
                    if isinstance(item, dict) and "product_id" in item:
                        product_ids.add(str(item["product_id"]))
        
        # Handle transfer transactions
        if "source_product_id" in transaction_data:
            product_ids.add(str(transaction_data["source_product_id"]))
            
        if "destination_product_id" in transaction_data:
            product_ids.add(str(transaction_data["destination_product_id"]))
            
        # Handle direct object properties for ORM objects
        if hasattr(transaction_data, "product_id"):
            product_ids.add(str(transaction_data.product_id))
            
        if hasattr(transaction_data, "items") and hasattr(transaction_data.items, "__iter__"):
            for item in transaction_data.items:
                if hasattr(item, "product_id"):
                    product_ids.add(str(item.product_id))
                    
        return product_ids
        
    def on_transaction_created(self, transaction_data: Dict[str, Any]) -> None:
        """
        Handle transaction creation event.
        
        Args:
            transaction_data: Data for the created transaction
        """
        # Usually we don't invalidate cache on creation, just log
        self.logger.debug(f"Transaction created: {transaction_data.get('id', 'unknown')}")
        
    def on_transaction_updated(self, transaction_data: Dict[str, Any]) -> None:
        """
        Handle transaction update event.
        
        Args:
            transaction_data: Data for the updated transaction
        """
        # Usually we don't invalidate cache on updates, just log
        self.logger.debug(f"Transaction updated: {transaction_data.get('id', 'unknown')}")
        
    def on_transaction_committed(self, transaction_data: Dict[str, Any]) -> None:
        """
        Handle transaction commit event.
        
        Args:
            transaction_data: Data for the committed transaction
        """
        # This is when we invalidate the cache, as the data has changed
        self.logger.info(f"Transaction committed: {transaction_data.get('id', 'unknown')}")
        self.handle_transaction_event("committed", transaction_data)
        
    def on_transaction_cancelled(self, transaction_data: Dict[str, Any]) -> None:
        """
        Handle transaction cancellation event.
        
        Args:
            transaction_data: Data for the cancelled transaction
        """
        # No need to invalidate cache for cancelled transactions
        self.logger.debug(f"Transaction cancelled: {transaction_data.get('id', 'unknown')}")


# Create a singleton transaction event handler