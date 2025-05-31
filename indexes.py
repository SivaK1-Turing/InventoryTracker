"""
indexes.py - Inverted index management for efficient searching

This module provides an IndexManager that maintains inverted indexes for fast lookups
of products by tags and product names. The indexes are updated incrementally on write
operations to avoid expensive full rebuilds.
"""
from typing import Dict, Set, List, Optional, Union, Callable, Any, Tuple
import uuid
import re
import threading
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class InvertedIndex:
    """
    Maintains an inverted index mapping attributes (e.g., tags, words in product names)
    to sets of product UUIDs that contain those attributes.
    
    The index is optimized for incremental updates and fast lookups.
    """
    
    def __init__(self, name: str, extractor_func: Callable[[Any], List[str]]):
        """
        Initialize an inverted index.
        
        Args:
            name: Name of this index (for logging purposes)
            extractor_func: Function that extracts indexable terms from an object
        """
        self.name = name
        self.extractor_func = extractor_func
        self.index: Dict[str, Set[uuid.UUID]] = defaultdict(set)
        self.reverse_index: Dict[uuid.UUID, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
    def add(self, obj_id: uuid.UUID, obj: Any) -> None:
        """
        Add or update an object in the index.
        
        Args:
            obj_id: UUID of the object
            obj: The object to index
        """
        with self._lock:
            # Extract terms from the object
            terms = self.extractor_func(obj)
            
            # Get previously indexed terms for this object
            old_terms = self.reverse_index.get(obj_id, set())
            
            # Find terms to add and remove
            new_terms = set(terms)
            terms_to_add = new_terms - old_terms
            terms_to_remove = old_terms - new_terms
            
            # Update the forward index (term -> object IDs)
            for term in terms_to_add:
                self.index[term].add(obj_id)
                
            for term in terms_to_remove:
                if obj_id in self.index[term]:
                    self.index[term].remove(obj_id)
                    # Clean up empty entries
                    if not self.index[term]:
                        del self.index[term]
            
            # Update the reverse index (object ID -> terms)
            if not new_terms:
                # If the object has no terms now, remove it from the reverse index
                if obj_id in self.reverse_index:
                    del self.reverse_index[obj_id]
            else:
                # Otherwise, update the terms for this object
                self.reverse_index[obj_id] = new_terms
                
    def remove(self, obj_id: uuid.UUID) -> None:
        """
        Remove an object from the index.
        
        Args:
            obj_id: UUID of the object to remove
        """
        with self._lock:
            # Get terms for this object
            if obj_id not in self.reverse_index:
                return
                
            terms = self.reverse_index[obj_id]
            
            # Remove object ID from each term's set
            for term in terms:
                if obj_id in self.index[term]:
                    self.index[term].remove(obj_id)
                    # Clean up empty entries
                    if not self.index[term]:
                        del self.index[term]
            
            # Remove from reverse index
            del self.reverse_index[obj_id]
            
    def search(self, term: str) -> Set[uuid.UUID]:
        """
        Search for objects matching the given term.
        
        Args:
            term: The term to search for
            
        Returns:
            Set of UUIDs for objects matching the term
        """
        with self._lock:
            return self.index.get(term, set()).copy()
    
    def search_prefix(self, prefix: str) -> Set[uuid.UUID]:
        """
        Search for objects matching the given prefix.
        
        Args:
            prefix: The prefix to search for
            
        Returns:
            Set of UUIDs for objects matching the prefix
        """
        with self._lock:
            result = set()
            for term, obj_ids in self.index.items():
                if term.startswith(prefix):
                    result.update(obj_ids)
            return result
    
    def search_regex(self, pattern: str) -> Set[uuid.UUID]:
        """
        Search for objects matching the given regex pattern.
        
        Args:
            pattern: The regex pattern to search for
            
        Returns:
            Set of UUIDs for objects matching the pattern
        """
        with self._lock:
            regex = re.compile(pattern, re.IGNORECASE)
            result = set()
            for term, obj_ids in self.index.items():
                if regex.search(term):
                    result.update(obj_ids)
            return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dict with statistics about the index
        """
        with self._lock:
            return {
                "index_name": self.name,
                "terms_count": len(self.index),
                "objects_count": len(self.reverse_index),
                "avg_objects_per_term": (
                    sum(len(ids) for ids in self.index.values()) / len(self.index) 
                    if self.index else 0
                ),
                "memory_usage_estimate_bytes": (
                    # Rough estimate of memory usage
                    sum(len(term) * 2 + len(ids) * 16 for term, ids in self.index.items()) +
                    sum(16 + len(terms) * 16 for _, terms in self.reverse_index.items())
                ),
            }


class IndexManager:
    """
    Manages multiple inverted indexes for product lookups.
    
    This class coordinates updates across different indexes to ensure consistency
    and provides a unified search interface.
    """
    
    def __init__(self):
        """Initialize the index manager with indexes for tags and product names."""
        # Index for tags
        self.tag_index = InvertedIndex(
            name="tags",
            extractor_func=lambda obj: [tag.lower().strip() for tag in getattr(obj, "tags", [])]
        )
        
        # Index for product names (tokenized)
        self.name_index = InvertedIndex(
            name="product_names",
            extractor_func=lambda obj: self._tokenize_name(getattr(obj, "name", ""))
        )
        
        # Index for SKUs
        self.sku_index = InvertedIndex(
            name="skus",
            extractor_func=lambda obj: [getattr(obj, "sku", "").upper()]
        )
        
        # Transaction tracking for optimistic concurrency control
        self.transaction_log: List[Tuple[datetime, str, uuid.UUID]] = []
        self.transaction_lock = threading.Lock()
        self.last_transaction_id = 0
        
    def _tokenize_name(self, name: str) -> List[str]:
        """
        Tokenize a product name into searchable terms.
        
        Args:
            name: Product name to tokenize
            
        Returns:
            List of terms extracted from the product name
        """
        if not name:
            return []
            
        # Convert to lowercase
        name = name.lower()
        
        # Split into words and filter out empty strings
        words = [w.strip() for w in re.split(r'\W+', name) if w.strip()]
        
        # Add original name too for exact matching
        words.append(name)
        
        # For multi-word names, include combinations of consecutive words
        if len(words) > 1:
            for i in range(len(words) - 1):
                words.append(f"{words[i]} {words[i+1]}")
        
        return words
        
    def update_product(self, product) -> None:
        """
        Update product in all indexes.
        
        Args:
            product: The product object to update
        """
        product_id = product.id
        
        # Acquire transaction ID for consistency
        with self.transaction_lock:
            self.last_transaction_id += 1
            transaction_id = self.last_transaction_id
            timestamp = datetime.now()
            self.transaction_log.append((timestamp, "update", product_id))
            
            # Trim transaction log if it's getting too large
            if len(self.transaction_log) > 1000:
                self.transaction_log = self.transaction_log[-1000:]
        
        # Update all indexes atomically
        logger.debug(f"Updating product {product_id} in indexes (transaction {transaction_id})")
        
        try:
            self.tag_index.add(product_id, product)
            self.name_index.add(product_id, product)
            self.sku_index.add(product_id, product)
            logger.debug(f"Successfully updated product {product_id} in all indexes")
        except Exception as e:
            logger.error(f"Error updating product {product_id} in indexes: {e}")
            # In a more robust implementation, we'd roll back the changes
            # that were already made to maintain consistency
            raise
    
    def remove_product(self, product_id: uuid.UUID) -> None:
        """
        Remove product from all indexes.
        
        Args:
            product_id: UUID of the product to remove
        """
        # Acquire transaction ID for consistency
        with self.transaction_lock:
            self.last_transaction_id += 1
            transaction_id = self.last_transaction_id
            timestamp = datetime.now()
            self.transaction_log.append((timestamp, "delete", product_id))
        
        # Remove from all indexes atomically
        logger.debug(f"Removing product {product_id} from indexes (transaction {transaction_id})")
        
        try:
            self.tag_index.remove(product_id)
            self.name_index.remove(product_id)
            self.sku_index.remove(product_id)
            logger.debug(f"Successfully removed product {product_id} from all indexes")
        except Exception as e:
            logger.error(f"Error removing product {product_id} from indexes: {e}")
            raise
    
    def search_by_tag(self, tag: str) -> Set[uuid.UUID]:
        """
        Search for products with the given tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            Set of product UUIDs matching the tag
        """
        return self.tag_index.search(tag.lower().strip())
    
    def search_by_name(self, name: str) -> Set[uuid.UUID]:
        """
        Search for products with the given name or name part.
        
        Args:
            name: Name or part of name to search for
            
        Returns:
            Set of product UUIDs matching the name
        """
        tokens = self._tokenize_name(name)
        result = set()
        for token in tokens:
            result.update(self.name_index.search(token))
        return result
        
    def search_by_sku(self, sku: str) -> Set[uuid.UUID]:
        """
        Search for products with the given SKU.
        
        Args:
            sku: SKU to search for
            
        Returns:
            Set of product UUIDs matching the SKU
        """
        return self.sku_index.search(sku.upper())
    
    def rebuild_index(self, products) -> None:
        """
        Rebuild all indexes from scratch.
        
        This should be used rarely, typically only during startup or recovery.
        
        Args:
            products: List of all products to index
        """
        # Create new indexes to replace the old ones
        new_tag_index = InvertedIndex(
            name="tags",
            extractor_func=lambda obj: [tag.lower().strip() for tag in getattr(obj, "tags", [])]
        )
        
        new_name_index = InvertedIndex(
            name="product_names",
            extractor_func=lambda obj: self._tokenize_name(getattr(obj, "name", ""))
        )
        
        new_sku_index = InvertedIndex(
            name="skus",
            extractor_func=lambda obj: [getattr(obj, "sku", "").upper()]
        )
        
        # Fill the new indexes
        for product in products:
            product_id = product.id
            new_tag_index.add(product_id, product)
            new_name_index.add(product_id, product)
            new_sku_index.add(product_id, product)
        
        # Replace old indexes with new ones (atomically)
        with self.transaction_lock:
            self.tag_index = new_tag_index
            self.name_index = new_name_index
            self.sku_index = new_sku_index
            
            # Reset transaction tracking
            self.last_transaction_id = 0
            self.transaction_log = []
            
            logger.info(f"Indexes rebuilt from scratch with {len(products)} products")
    
    def check_consistency(self, products) -> Dict[str, Any]:
        """
        Check index consistency against the actual product data.
        
        This is a diagnostic function that helps identify inconsistencies without
        performing a full rebuild.
        
        Args:
            products: List of all products to check against
            
        Returns:
            Dict with consistency check results
        """
        # Create sets of all product IDs
        product_ids = {product.id for product in products}
        tag_index_ids = set(self.tag_index.reverse_index.keys())
        name_index_ids = set(self.name_index.reverse_index.keys())
        sku_index_ids = set(self.sku_index.reverse_index.keys())
        
        # Check for products in data but not in indexes
        missing_from_tag_index = product_ids - tag_index_ids
        missing_from_name_index = product_ids - name_index_ids
        missing_from_sku_index = product_ids - sku_index_ids
        
        # Check for products in indexes but not in data
        extra_in_tag_index = tag_index_ids - product_ids
        extra_in_name_index = name_index_ids - product_ids
        extra_in_sku_index = sku_index_ids - product_ids
        
        return {
            "total_products": len(products),
            "tag_index_count": len(tag_index_ids),
            "name_index_count": len(name_index_ids),
            "sku_index_count": len(sku_index_ids),
            "missing_from_tag_index": list(missing_from_tag_index),
            "missing_from_name_index": list(missing_from_name_index),
            "missing_from_sku_index": list(missing_from_sku_index),
            "extra_in_tag_index": list(extra_in_tag_index),
            "extra_in_name_index": list(extra_in_name_index),
            "extra_in_sku_index": list(extra_in_sku_index),
            "is_consistent": (
                not missing_from_tag_index and 
                not missing_from_name_index and
                not missing_from_sku_index and
                not extra_in_tag_index and
                not extra_in_name_index and
                not extra_in_sku_index
            )
        }
    
    def repair_inconsistencies(self, products) -> Dict[str, int]:
        """
        Repair inconsistencies between indexes and actual product data.
        
        This performs targeted updates without a full rebuild.
        
        Args:
            products: List of all products to check against
            
        Returns:
            Dict with details of repairs made
        """
        # Get consistency check results
        check_results = self.check_consistency(products)
        
        # Create a lookup table for products by ID
        product_lookup = {product.id: product for product in products}
        
        # Track repair counts
        repairs = {
            "added_to_tag_index": 0,
            "added_to_name_index": 0,
            "added_to_sku_index": 0,
            "removed_from_tag_index": 0,
            "removed_from_name_index": 0,
            "removed_from_sku_index": 0,
        }
        
        # Add missing products to indexes
        for product_id in check_results["missing_from_tag_index"]:
            if product_id in product_lookup:
                self.tag_index.add(product_id, product_lookup[product_id])
                repairs["added_to_tag_index"] += 1
                
        for product_id in check_results["missing_from_name_index"]:
            if product_id in product_lookup:
                self.name_index.add(product_id, product_lookup[product_id])
                repairs["added_to_name_index"] += 1
                
        for product_id in check_results["missing_from_sku_index"]:
            if product_id in product_lookup:
                self.sku_index.add(product_id, product_lookup[product_id])
                repairs["added_to_sku_index"] += 1
        
        # Remove extra products from indexes
        for product_id in check_results["extra_in_tag_index"]:
            self.tag_index.remove(product_id)
            repairs["removed_from_tag_index"] += 1
            
        for product_id in check_results["extra_in_name_index"]:
            self.name_index.remove(product_id)
            repairs["removed_from_name_index"] += 1
            
        for product_id in check_results["extra_in_sku_index"]:
            self.sku_index.remove(product_id)
            repairs["removed_from_sku_index"] += 1
        
        logger.info(f"Repaired index inconsistencies: {repairs}")
        return repairs

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all indexes.
        
        Returns:
            Dict with statistics about each index
        """
        return {
            "tag_index": self.tag_index.get_statistics(),
            "name_index": self.name_index.get_statistics(),
            "sku_index": self.sku_index.get_statistics(),
            "transaction_log_size": len(self.transaction_log),
            "last_transaction_id": self.last_transaction_id,
        }


# Example integration with the store
class IndexedStore:
    """
    Extends the base store with indexing capabilities.
    
    This class wraps an existing store implementation and adds index management
    to ensure fast lookups while maintaining index consistency.
    """
    
    def __init__(self, base_store):
        """
        Initialize indexed store.
        
        Args:
            base_store: The underlying storage implementation
        """
        self.base_store = base_store
        self.index_manager = IndexManager()
        
        # Initialize indexes from existing data
        self._initialize_indexes()
        
    def _initialize_indexes(self):
        """Initialize indexes from existing data in the store."""
        try:
            products = self.base_store.get_all_products()
            logger.info(f"Initializing indexes with {len(products)} products")
            self.index_manager.rebuild_index(products)
        except Exception as e:
            logger.error(f"Error initializing indexes: {e}")
            raise
        
    def save_product(self, product):
        """
        Save a product to the store with index updates.
        
        Args:
            product: The product to save
        
        Returns:
            The saved product from the base store
        """
        # Save to underlying store first
        saved_product = self.base_store.save_product(product)
        
        # Update indexes
        self.index_manager.update_product(saved_product)
        
        return saved_product
    
    def delete_product(self, product_id):
        """
        Delete a product from the store and indexes.
        
        Args:
            product_id: UUID of the product to delete
        """
        # Delete from underlying store first
        self.base_store.delete_product(product_id)
        
        # Remove from indexes
        self.index_manager.remove_product(product_id)
    
    def get_product(self, product_id):
        """
        Get a product by ID.
        
        Args:
            product_id: UUID of the product to retrieve
        
        Returns:
            The product from the base store
        """
        return self.base_store.get_product(product_id)
    
    def get_all_products(self):
        """
        Get all products.
        
        Returns:
            List of all products from the base store
        """
        return self.base_store.get_all_products()
    
    def search_products(self, query=None, tag=None, name=None, sku=None):
        """
        Search for products using the indexes for efficiency.
        
        Args:
            query: General search term (searches across tags, names, and SKUs)
            tag: Specific tag to search for
            name: Specific name or part of name to search for
            sku: Specific SKU to search for
        
        Returns:
            List of products matching the search criteria
        """
        matching_ids = None
        
        # First, use indexes for fast narrowing down of results
        if tag:
            tag_matches = self.index_manager.search_by_tag(tag)
            matching_ids = tag_matches if matching_ids is None else matching_ids.intersection(tag_matches)
        
        if name:
            name_matches = self.index_manager.search_by_name(name)
            matching_ids = name_matches if matching_ids is None else matching_ids.intersection(name_matches)
            
        if sku:
            sku_matches = self.index_manager.search_by_sku(sku)
            matching_ids = sku_matches if matching_ids is None else matching_ids.intersection(sku_matches)
            
        if query:
            # General query searches across all indexes
            tag_matches = self.index_manager.search_by_tag(query)
            name_matches = self.index_manager.search_by_name(query)
            sku_matches = self.index_manager.search_by_sku(query)
            
            query_matches = tag_matches.union(name_matches, sku_matches)
            matching_ids = query_matches if matching_ids is None else matching_ids.intersection(query_matches)
        
        # If no search criteria or no matches, return empty list
        if matching_ids is None:
            return []
            
        # Retrieve full product objects for matching IDs
        matching_products = []
        for product_id in matching_ids:
            try:
                product = self.get_product(product_id)
                if product:
                    matching_products.append(product)
            except Exception as e:
                logger.warning(f"Error retrieving product {product_id}: {e}")
        
        return matching_products
    
    def check_index_consistency(self):
        """
        Check consistency of indexes with data store.
        
        Returns:
            Consistency check results
        """
        products = self.get_all_products()
        return self.index_manager.check_consistency(products)
    
    def repair_indexes(self):
        """
        Repair indexes without a full rebuild.
        
        Returns:
            Repair results
        """
        products = self.get_all_products()
        return self.index_manager.repair_inconsistencies(products)
    
    def rebuild_indexes(self):
        """Rebuild all indexes from scratch."""
        products = self.get_all_products()
        self.index_manager.rebuild_index(products)
        
    def get_index_statistics(self):
        """
        Get statistics about the indexes.
        
        Returns:
            Index statistics
        """
        return self.index_manager.get_statistics()


# Example usage
def example_usage():
    from inventorytracker.store import get_store
    from inventorytracker.models.product import Product
    import uuid
    
    # Get base store
    base_store = get_store()
    
    # Create indexed store wrapper
    indexed_store = IndexedStore(base_store)
    
    # Create sample product
    product = Product(
        id=uuid.uuid4(),
        name="Widget Pro",
        sku="WDG-PRO-123",
        price=29.99,
        reorder_level=10,
        tags=["hardware", "tool", "professional"]
    )
    
    # Save product (indexes are automatically updated)
    indexed_store.save_product(product)
    
    # Search using indexes
    hardware_products = indexed_store.search_products(tag="hardware")
    widget_products = indexed_store.search_products(name="widget")
    sku_products = indexed_store.search_products(sku="WDG")
    
    print(f"Found {len(hardware_products)} hardware products")
    print(f"Found {len(widget_products)} widget products")
    print(f"Found {len(sku_products)} products matching SKU 'WDG'")
    
    # Change product and update
    product.tags.append("bestseller")
    product.name = "Widget Pro Plus"
    indexed_store.save_product(product)
    
    # Check that indexes are updated
    bestseller_products = indexed_store.search_products(tag="bestseller")
    print(f"Found {len(bestseller_products)} bestseller products")
    
    # Check index consistency
    consistency = indexed_store.check_index_consistency()
    print(f"Indexes consistent: {consistency['is_consistent']}")
    
    if not consistency['is_consistent']:
        # Repair any inconsistencies
        repairs = indexed_store.repair_indexes()
        print(f"Repairs made: {repairs}")
    
    # Get index statistics
    stats = indexed_store.get_index_statistics()
    print(f"Index statistics: {stats}")
    
    
if __name__ == "__main__":
    example_usage()