"""
store/adapter.py - Abstract Storage Adapter for Persistence & Import/Export feature

This module defines the interface for storage adapters that handle saving and
loading inventory data to/from various persistent storage formats.
"""

import abc
from typing import Dict, List, Any, Optional, Union
from uuid import UUID
import asyncio

from ..models.product import Product
from ..models.inventory import StockTransaction


class StorageAdapter(abc.ABC):
    """
    Abstract base class defining the interface for storage adapters.
    
    Storage adapters handle the persistence of inventory data to various
    storage backends (files, databases, cloud storage, etc).
    """
    
    @abc.abstractmethod
    async def save_product(self, product: Product) -> None:
        """
        Save a product to storage.
        
        Args:
            product: The Product object to save
        
        Raises:
            StorageError: If the product cannot be saved
        """
        pass
    
    @abc.abstractmethod
    async def save_transaction(self, transaction: StockTransaction) -> None:
        """
        Save a stock transaction to storage.
        
        Args:
            transaction: The StockTransaction object to save
        
        Raises:
            StorageError: If the transaction cannot be saved
        """
        pass
    
    @abc.abstractmethod
    async def load_all(self) -> Dict[str, Union[List[Product], List[StockTransaction]]]:
        """
        Load all products and transactions from storage.
        
        Returns:
            A dictionary with 'products' and 'transactions' keys mapping to
            lists of Product and StockTransaction objects respectively.
        
        Raises:
            StorageError: If the data cannot be loaded from storage
        """
        pass
    
    @abc.abstractmethod
    async def export_data(self, format: str, path: str) -> None:
        """
        Export all inventory data to the specified format and path.
        
        Args:
            format: The export format (e.g., 'json', 'csv', 'excel')
            path: Path where the export file should be saved
        
        Raises:
            StorageError: If the data cannot be exported
            ValueError: If the format is not supported
        """
        pass
    
    @abc.abstractmethod
    async def import_data(self, format: str, path: str) -> Dict[str, Any]:
        """
        Import inventory data from the specified format and path.
        
        Args:
            format: The import format (e.g., 'json', 'csv', 'excel')
            path: Path to the file to import
        
        Returns:
            A dictionary containing import statistics (products added, updated, etc.)
        
        Raises:
            StorageError: If the data cannot be imported
            ValueError: If the format is not supported or data is invalid
        """
        pass
    
    # Sync versions of the async methods for CLI contexts that can't use async
    
    def save_product_sync(self, product: Product) -> None:
        """Synchronous version of save_product for CLI contexts"""
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.save_product(product))
        except RuntimeError:
            # If there's no event loop in the current thread (like in a CLI),
            # create a new one just for this operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.save_product(product))
            loop.close()
    
    def save_transaction_sync(self, transaction: StockTransaction) -> None:
        """Synchronous version of save_transaction for CLI contexts"""
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.save_transaction(transaction))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.save_transaction(transaction))
            loop.close()
    
    def load_all_sync(self) -> Dict[str, Union[List[Product], List[StockTransaction]]]:
        """Synchronous version of load_all for CLI contexts"""
        loop = asyncio.get_event_loop()
        try:
            return loop.run_until_complete(self.load_all())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.load_all())
            loop.close()
            return result
    
    def export_data_sync(self, format: str, path: str) -> None:
        """Synchronous version of export_data for CLI contexts"""
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.export_data(format, path))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.export_data(format, path))
            loop.close()
    
    def import_data_sync(self, format: str, path: str) -> Dict[str, Any]:
        """Synchronous version of import_data for CLI contexts"""
        loop = asyncio.get_event_loop()
        try:
            return loop.run_until_complete(self.import_data(format, path))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.import_data(format, path))
            loop.close()
            return result


class StorageError(Exception):
    """Base exception for storage-related errors"""
    pass


class ValidationError(StorageError):
    """Exception raised when imported data fails validation"""
    pass