#!/usr/bin/env python3
"""
Thread-safe in-memory store for Inventory Tracker.

This module provides a memory-based implementation of the storage interface
with thread safety guarantees using threading.RLock to prevent race conditions.
"""
import threading
import uuid
from collections import deque
from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Generic, cast

from inventorytracker.logging import logger
from inventorytracker.models.item import Item
from inventorytracker.models.category import Category
from inventorytracker.store import InventoryStore, StoreError, ItemNotFoundError


# Type variables for generics
T = TypeVar('T')
ID = str  # Type alias for IDs (UUIDs as strings)


class TransactionType(str, Enum):
    """Types of inventory transactions."""
    ADD = "add"          # Initial product addition or stock increase
    REMOVE = "remove"    # Stock decrease (sales, usage)
    ADJUST = "adjust"    # Inventory count adjustment
    UPDATE = "update"    # Metadata update (price, description)


class Transaction:
    """Record of an inventory transaction."""
    
    def __init__(
        self,
        transaction_id: Optional[ID] = None,
        item_id: Optional[ID] = None,
        transaction_type: TransactionType = TransactionType.ADD,
        quantity: int = 0,
        timestamp: Optional[datetime] = None,
        user: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize a Transaction record."""
        self.transaction_id = transaction_id or str(uuid.uuid4())
        self.item_id = item_id
        self.transaction_type = transaction_type
        self.quantity = quantity
        self.timestamp = timestamp or datetime.now()
        self.user = user
        self.notes = notes
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "transaction_id": self.transaction_id,
            "item_id": self.item_id,
            "transaction_type": self.transaction_type.value,
            "quantity": self.quantity,
            "timestamp": self.timestamp.isoformat(),
            "user": self.user,
            "notes": self.notes,
            "metadata": self.metadata.copy() if self.metadata else {},
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create a Transaction from dictionary data."""
        # Handle timestamp conversion from string
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            
        # Handle transaction type from string
        if isinstance(data.get("transaction_type"), str):
            data["transaction_type"] = TransactionType(data["transaction_type"])
            
        return cls(**data)


class ThreadSafeCollection(Generic[T]):
    """
    Thread-safe collection for storing objects with ID-based access.
    
    Uses RLock for thread safety and provides atomic operations.
    """
    
    def __init__(self) -> None:
        """Initialize the thread-safe collection."""
        self._items: Dict[ID, T] = {}
        self._lock = threading.RLock()  # Reentrant Lock for thread safety
        self._changelog = deque(maxlen=1000)  # Limited history for debugging
        
    def add(self, item_id: ID, item: T) -> None:
        """
        Add an item to the collection.
        
        Args:
            item_id: Unique identifier for the item
            item: The item to store
            
        Raises:
            ValueError: If an item with the given ID already exists
        """
        with self._lock:
            if item_id in self._items:
                raise ValueError(f"Item with ID {item_id} already exists")
            
            self._items[item_id] = item
            self._changelog.append((datetime.now(), "add", item_id))
            
    def get(self, item_id: ID) -> T:
        """
        Get an item by ID.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            The requested item
            
        Raises:
            KeyError: If no item with the given ID exists
        """
        with self._lock:
            if item_id not in self._items:
                raise KeyError(f"No item with ID {item_id}")
            
            # Return a deep copy to prevent external mutation
            return deepcopy(self._items[item_id])
            
    def update(self, item_id: ID, item: T) -> None:
        """
        Update an existing item.
        
        Args:
            item_id: ID of the item to update
            item: New version of the item
            
        Raises:
            KeyError: If no item with the given ID exists
        """
        with self._lock:
            if item_id not in self._items:
                raise KeyError(f"No item with ID {item_id}")
            
            self._items[item_id] = item
            self._changelog.append((datetime.now(), "update", item_id))
            
    def delete(self, item_id: ID) -> T:
        """
        Delete an item by ID.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            The deleted item
            
        Raises:
            KeyError: If no item with the given ID exists
        """
        with self._lock:
            if item_id not in self._items:
                raise KeyError(f"No item with ID {item_id}")
            
            item = self._items.pop(item_id)
            self._changelog.append((datetime.now(), "delete", item_id))
            
            # Return a deep copy
            return deepcopy(item)
            
    def list_all(self) -> List[T]:
        """
        Get a list of all items in the collection.
        
        Returns:
            List of all items (deep copies)
        """
        with self._lock:
            # Return deep copies to prevent external mutation
            return [deepcopy(item) for item in self._items.values()]
            
    def get_ids(self) -> Set[ID]:
        """
        Get all item IDs in the collection.
        
        Returns:
            Set of all item IDs
        """
        with self._lock:
            return set(self._items.keys())
            
    def exists(self, item_id: ID) -> bool:
        """
        Check if an item with the given ID exists.
        
        Args:
            item_id: ID to check
            
        Returns:
            True if item exists, False otherwise
        """
        with self._lock:
            return item_id in self._items
            
    def count(self) -> int:
        """
        Get the number of items in the collection.
        
        Returns:
            Count of items
        """
        with self._lock:
            return len(self._items)
            
    def clear(self) -> None:
        """Clear all items from the collection."""
        with self._lock:
            self._items.clear()
            self._changelog.append((datetime.now(), "clear", None))
            
    def atomic_operation(self, operation):
        """
        Execute an operation atomically within a lock.
        
        Args:
            operation: Callable that takes the items dict as argument
            
        Returns:
            Result of the operation
        """
        with self._lock:
            return operation(self._items)
            
    def bulk_add(self, items: Dict[ID, T]) -> None:
        """
        Add multiple items at once atomically.
        
        Args:
            items: Dictionary mapping IDs to items
            
        Raises:
            ValueError: If any item ID already exists
        """
        with self._lock:
            # Check all IDs first to avoid partial updates
            for item_id in items:
                if item_id in self._items:
                    raise ValueError(f"Item with ID {item_id} already exists")
                    
            # Add all items
            self._items.update(items)
            self._changelog.append((datetime.now(), "bulk_add", len(items)))
            
    def get_changelog(self) -> List[Tuple[datetime, str, Any]]:
        """
        Get the changelog for debugging purposes.
        
        Returns:
            List of (timestamp, operation, item_id) tuples
        """
        with self._lock:
            return list(self._changelog)


class MemoryStore(InventoryStore):
    """
    Thread-safe in-memory implementation of inventory storage.
    
    Uses RLocks to prevent race conditions when accessing or modifying data.
    """
    
    def __init__(self) -> None:
        """Initialize the memory store with thread-safe collections."""
        self._items = ThreadSafeCollection[Item]()
        self._categories = ThreadSafeCollection[Category]()
        self._transactions = ThreadSafeCollection[Transaction]()
        self._master_lock = threading.RLock()  # For operations across collections
        logger.debug("Initialized in-memory store with thread safety")
        
    def add_item(self, item: Item) -> Item:
        """
        Add a new inventory item.
        
        Args:
            item: The item to add
            
        Returns:
            The added item with assigned ID
            
        Raises:
            StoreError: If the item could not be added
        """
        try:
            # Ensure item has ID
            if not item.item_id:
                item.item_id = str(uuid.uuid4())
                
            # Insert the item
            self._items.add(item.item_id, item)
            
            # Create an ADD transaction
            self._add_transaction(Transaction(
                item_id=item.item_id,
                transaction_type=TransactionType.ADD,
                quantity=item.quantity,
                notes=f"Initial addition of {item.name}"
            ))
            
            logger.debug(f"Added item {item.item_id}: {item.name}")
            return self._items.get(item.item_id)
            
        except Exception as e:
            logger.error(f"Failed to add item: {str(e)}")
            raise StoreError(f"Could not add item: {str(e)}") from e
            
    def get_item(self, item_id: ID) -> Item:
        """
        Get an inventory item by ID.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            The requested item
            
        Raises:
            ItemNotFoundError: If the item does not exist
        """
        try:
            return self._items.get(item_id)
        except KeyError:
            raise ItemNotFoundError(f"Item {item_id} not found")
            
    def update_item(self, item: Item) -> Item:
        """
        Update an existing inventory item.
        
        Args:
            item: The item with updated values
            
        Returns:
            The updated item
            
        Raises:
            ItemNotFoundError: If the item does not exist
            StoreError: If the item could not be updated
        """
        try:
            # Get the existing item for comparison
            with self._master_lock:
                old_item = self.get_item(item.item_id)
                
                # Update the item
                self._items.update(item.item_id, item)
                
                # Create a transaction if the quantity changed
                if old_item.quantity != item.quantity:
                    quantity_diff = item.quantity - old_item.quantity
                    transaction_type = TransactionType.ADD if quantity_diff > 0 else TransactionType.REMOVE
                    
                    self._add_transaction(Transaction(
                        item_id=item.item_id,
                        transaction_type=TransactionType.ADJUST,
                        quantity=abs(quantity_diff),
                        notes=f"Adjusted {item.name} quantity by {quantity_diff}"
                    ))
                elif old_item.name != item.name or old_item.description != item.description or old_item.price != item.price:
                    # Record metadata updates
                    self._add_transaction(Transaction(
                        item_id=item.item_id,
                        transaction_type=TransactionType.UPDATE,
                        quantity=0,
                        notes=f"Updated {item.name} metadata"
                    ))
                
                logger.debug(f"Updated item {item.item_id}: {item.name}")
                return self._items.get(item.item_id)
                
        except ItemNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update item {item.item_id}: {str(e)}")
            raise StoreError(f"Could not update item: {str(e)}") from e
            
    def delete_item(self, item_id: ID) -> None:
        """
        Delete an inventory item.
        
        Args:
            item_id: ID of the item to delete
            
        Raises:
            ItemNotFoundError: If the item does not exist
            StoreError: If the item could not be deleted
        """
        try:
            # Atomic operation to get item then delete it
            with self._master_lock:
                # Check if item exists first
                item = self.get_item(item_id)
                
                # Record a REMOVE transaction
                self._add_transaction(Transaction(
                    item_id=item_id,
                    transaction_type=TransactionType.REMOVE,
                    quantity=item.quantity,
                    notes=f"Removed item {item.name} from inventory"
                ))
                
                # Delete the item
                self._items.delete(item_id)
                
                logger.debug(f"Deleted item {item_id}")
                
        except ItemNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete item {item_id}: {str(e)}")
            raise StoreError(f"Could not delete item: {str(e)}") from e
            
    def list_items(self) -> List[Item]:
        """
        Get all inventory items.
        
        Returns:
            List of all items
        """
        return self._items.list_all()
        
    def get_item_count(self) -> int:
        """
        Get the total number of items in inventory.
        
        Returns:
            Count of items
        """
        return self._items.count()
        
    def add_category(self, category: Category) -> Category:
        """
        Add a new category.
        
        Args:
            category: The category to add
            
        Returns:
            The added category with assigned ID
            
        Raises:
            StoreError: If the category could not be added
        """
        try:
            # Ensure category has ID
            if not category.category_id:
                category.category_id = str(uuid.uuid4())
                
            # Add the category
            self._categories.add(category.category_id, category)
            
            logger.debug(f"Added category {category.category_id}: {category.name}")
            return self._categories.get(category.category_id)
            
        except Exception as e:
            logger.error(f"Failed to add category: {str(e)}")
            raise StoreError(f"Could not add category: {str(e)}") from e
            
    def get_category(self, category_id: ID) -> Category:
        """
        Get a category by ID.
        
        Args:
            category_id: ID of the category to retrieve
            
        Returns:
            The requested category
            
        Raises:
            ItemNotFoundError: If the category does not exist
        """
        try:
            return self._categories.get(category_id)
        except KeyError:
            raise ItemNotFoundError(f"Category {category_id} not found")
            
    def list_categories(self) -> List[Category]:
        """
        Get all categories.
        
        Returns:
            List of all categories
        """
        return self._categories.list_all()
        
    def _add_transaction(self, transaction: Transaction) -> Transaction:
        """
        Add a transaction record.
        
        Args:
            transaction: The transaction to record
            
        Returns:
            The added transaction
            
        Raises:
            StoreError: If the transaction could not be added
        """
        try:
            # Ensure transaction has ID
            if not transaction.transaction_id:
                transaction.transaction_id = str(uuid.uuid4())
                
            # Add the transaction
            self._transactions.add(transaction.transaction_id, transaction)
            
            logger.debug(f"Recorded transaction {transaction.transaction_id} "
                        f"of type {transaction.transaction_type.value} "
                        f"for item {transaction.item_id}")
                        
            return self._transactions.get(transaction.transaction_id)
            
        except Exception as e:
            logger.error(f"Failed to record transaction: {str(e)}")
            raise StoreError(f"Could not record transaction: {str(e)}") from e
            
    def get_transaction(self, transaction_id: ID) -> Transaction:
        """
        Get a transaction by ID.
        
        Args:
            transaction_id: ID of the transaction to retrieve
            
        Returns:
            The requested transaction
            
        Raises:
            ItemNotFoundError: If the transaction does not exist
        """
        try:
            return self._transactions.get(transaction_id)
        except KeyError:
            raise ItemNotFoundError(f"Transaction {transaction_id} not found")
            
    def get_item_transactions(self, item_id: ID) -> List[Transaction]:
        """
        Get all transactions for a specific item.
        
        Args:
            item_id: ID of the item
            
        Returns:
            List of transactions for the item
            
        Raises:
            ItemNotFoundError: If the item does not exist
        """
        # First check if item exists
        if not self._items.exists(item_id):
            raise ItemNotFoundError(f"Item {item_id} not found")
            
        # Filter transactions for this item
        def filter_transactions(transactions_dict):
            return [
                deepcopy(tx) for tx in transactions_dict.values()
                if tx.item_id == item_id
            ]
            
        return self._transactions.atomic_operation(filter_transactions)
        
    def list_transactions(self) -> List[Transaction]:
        """
        Get all transaction records.
        
        Returns:
            List of all transactions
        """
        return self._transactions.list_all()
        
    def adjust_item_quantity(self, item_id: ID, quantity_change: int, notes: Optional[str] = None) -> Item:
        """
        Adjust the quantity of an item and record a transaction.
        
        Args:
            item_id: ID of the item to adjust
            quantity_change: Amount to change (positive or negative)
            notes: Optional notes for the transaction
            
        Returns:
            The updated item
            
        Raises:
            ItemNotFoundError: If the item does not exist
            ValueError: If the adjustment would make quantity negative and that's not allowed
            StoreError: If the adjustment could not be made
        """
        try:
            # Atomic operation for the entire adjustment
            with self._master_lock:
                # Get current item
                item = self.get_item(item_id)
                
                # Calculate new quantity
                new_quantity = item.quantity + quantity_change
                
                # Prevent negative quantities unless allowed
                if new_quantity < 0:
                    raise ValueError(f"Cannot adjust quantity to negative value: {new_quantity}")
                    
                # Update item with new quantity
                item.quantity = new_quantity
                self._items.update(item_id, item)
                
                # Determine transaction type
                transaction_type = (
                    TransactionType.ADD if quantity_change > 0 
                    else TransactionType.REMOVE
                )
                
                # Record transaction
                self._add_transaction(Transaction(
                    item_id=item_id,
                    transaction_type=transaction_type,
                    quantity=abs(quantity_change),
                    notes=notes or f"Adjusted quantity by {quantity_change}"
                ))
                
                logger.debug(f"Adjusted item {item_id} quantity by {quantity_change}")
                return self._items.get(item_id)
                
        except ItemNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to adjust item {item_id} quantity: {str(e)}")
            raise StoreError(f"Could not adjust item quantity: {str(e)}") from e
            
    def search_items(self, query: str) -> List[Item]:
        """
        Search for items by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching items
        """
        # Convert query to lowercase for case-insensitive search
        query = query.lower()
        
        def search_operation(items_dict):
            return [
                deepcopy(item) for item in items_dict.values()
                if query in item.name.lower() or (
                    item.description and query in item.description.lower()
                )
            ]
            
        return self._items.atomic_operation(search_operation)
        
    def clear_all(self) -> None:
        """
        Clear all data from the store.
        
        This is primarily for testing purposes.
        """
        with self._master_lock:
            self._items.clear()
            self._categories.clear()
            self._transactions.clear()
            logger.warning("Cleared all data from memory store")


# Factory function
def create_memory_store() -> MemoryStore:
    """
    Create a new instance of the memory store.
    
    Returns:
        MemoryStore: A new thread-safe memory store
    """
    return MemoryStore()


if __name__ == "__main__":
    # Simple demonstration
    store = create_memory_store()
    
    # Add some items
    item1 = Item(name="Test Item 1", quantity=10, price=19.99)
    item2 = Item(name="Test Item 2", quantity=5, price=29.99)
    
    store.add_item(item1)
    store.add_item(item2)
    
    # List items
    items = store.list_items()
    for item in items:
        print(f"{item.item_id}: {item.name} - Quantity: {item.quantity}, Price: ${item.price}")
        
    # Adjust quantity
    updated_item = store.adjust_item_quantity(item1.item_id, -3, "Sold 3 units")
    print(f"After adjustment: {updated_item.name} - Quantity: {updated_item.quantity}")
    
    # List transactions
    transactions = store.list_transactions()
    for tx in transactions:
        print(f"Transaction {tx.transaction_id}: {tx.transaction_type.value} - {tx.notes}")