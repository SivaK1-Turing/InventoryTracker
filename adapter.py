"""
store/adapter.py - Storage Adapter Interface for InventoryTracker

This module defines the abstract StorageAdapter interface that various storage implementations
must implement to provide persistence for the InventoryTracker application.
"""

import abc
from typing import Dict, List, Any, Optional, AsyncIterator, Union, Type, TypeVar
from contextlib import asynccontextmanager

from ..models.product import Product
from ..models.inventory import StockTransaction

T = TypeVar('T')


class StorageError(Exception):
    """Base exception for all storage-related errors"""
    pass


class StorageAdapter(abc.ABC):
    """
    Abstract base class for all storage adapters in the InventoryTracker system.
    
    Storage adapters are responsible for persisting and retrieving data in various formats
    such as JSON files, SQLite databases, or remote APIs.
    """
    
    @abc.abstractmethod
    async def save_product(self, product: Product) -> None:
        """
        Save a product to the storage.
        
        Args:
            product: The product to save
        
        Raises:
            StorageError: If the product cannot be saved
        """
        pass
    
    @abc.abstractmethod
    async def save_transaction(self, transaction: StockTransaction) -> None:
        """
        Save a stock transaction to the storage.
        
        Args:
            transaction: The transaction to save
        
        Raises:
            StorageError: If the transaction cannot be saved
        """
        pass
    
    @abc.abstractmethod
    async def load_all(self) -> Dict[str, List[Any]]:
        """
        Load all products and transactions from storage.
        
        Returns:
            A dictionary with 'products' and 'transactions' lists
        
        Raises:
            StorageError: If data cannot be loaded from storage
        """
        pass
    
    @abc.abstractmethod
    async def get_product(self, product_id: str) -> Optional[Product]:
        """
        Get a single product by ID.
        
        Args:
            product_id: The ID of the product to retrieve
            
        Returns:
            The Product if found, None otherwise
        """
        pass
    
    @abc.abstractmethod
    async def get_transactions(self, product_id: Optional[str] = None, 
                             limit: Optional[int] = None) -> List[StockTransaction]:
        """
        Get transactions, optionally filtered by product ID.
        
        Args:
            product_id: Optional product ID to filter transactions
            limit: Optional maximum number of transactions to return
            
        Returns:
            List of StockTransaction objects
        """
        pass
    
    # Optional streaming methods for memory-efficient data access
    
    @asynccontextmanager
    async def stream_products(self) -> AsyncIterator[Product]:
        """
        Stream products from storage one at a time to minimize memory usage.
        
        This is an optional method that storage adapters can implement to provide
        more efficient data access for large datasets. The default implementation
        falls back to load_all().
        
        Yields:
            AsyncIterator of Product objects
            
        Raises:
            StorageError: If products cannot be streamed from storage
        """
        data = await self.load_all()
        for product in data.get('products', []):
            yield product
    
    @asynccontextmanager
    async def stream_transactions(self) -> AsyncIterator[StockTransaction]:
        """
        Stream transactions from storage one at a time to minimize memory usage.
        
        This is an optional method that storage adapters can implement to provide
        more efficient data access for large datasets. The default implementation
        falls back to load_all().
        
        Yields:
            AsyncIterator of StockTransaction objects
            
        Raises:
            StorageError: If transactions cannot be streamed from storage
        """
        data = await self.load_all()
        for transaction in data.get('transactions', []):
            yield transaction
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for handling transactions (if supported by the adapter).
        
        This provides atomic operations where multiple changes are committed together
        or rolled back if there's an error. Default implementation is a no-op, but
        database adapters should override this to provide proper transaction handling.
        
        Example usage:
        ```python
        async with adapter.transaction():
            await adapter.save_product(product)
            await adapter.save_transaction(transaction)
        ```
        """
        try:
            yield
        except Exception:
            # Default implementation has no rollback capability
            raise