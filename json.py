"""
store/json.py - JSON File Storage Adapter for InventoryTracker

This module provides a JSONAdapter implementation of the StorageAdapter interface
that stores data in JSON files. It ensures atomic writes and includes corruption
detection for reliable persistence during development and testing.
"""

import os
import json
import uuid
import asyncio
import hashlib
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncIterator, Union, Tuple, Set

from ..models.product import Product
from ..models.inventory import StockTransaction
from .adapter import StorageAdapter, StorageError

# Set up logger
logger = logging.getLogger(__name__)


class JSONAdapter(StorageAdapter):
    """
    A storage adapter that persists data in JSON files with atomic write operations
    and data integrity verification.
    
    This adapter is primarily designed for development and testing, providing
    a simple and reliable way to store and retrieve inventory data.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the JSON file storage adapter.
        
        Args:
            data_dir: Directory where data files will be stored
        """
        self.data_dir = Path(data_dir)
        self.products_file = self.data_dir / "products.json"
        self.transactions_file = self.data_dir / "transactions.json"
        
        # In-memory cache
        self.products: Dict[str, Product] = {}
        self.transactions: List[StockTransaction] = []
        
        # Lock to prevent concurrent file access
        self.file_lock = asyncio.Lock()
        
        # Checksum file paths
        self.products_checksum_file = self.data_dir / "products.checksum"
        self.transactions_checksum_file = self.data_dir / "transactions.checksum"
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Flag to track if data has been loaded
        self.loaded = False
    
    async def _ensure_loaded(self) -> None:
        """Ensure that data has been loaded from disk."""
        if not self.loaded:
            await self._load_data()
    
    async def _load_data(self) -> None:
        """Load data from JSON files, with integrity verification."""
        async with self.file_lock:
            # Load products
            if self.products_file.exists():
                try:
                    # Verify integrity first
                    if not await self._verify_file_integrity(self.products_file, self.products_checksum_file):
                        logger.warning(f"Products file integrity check failed, using empty data")
                        self.products = {}
                    else:
                        with open(self.products_file, 'r') as f:
                            products_data = json.load(f)
                            
                        for product_dict in products_data:
                            # Convert to Product model
                            product = Product(
                                id=uuid.UUID(product_dict['id']),
                                name=product_dict['name'],
                                sku=product_dict['sku'],
                                price=float(product_dict['price']),
                                quantity=int(product_dict['quantity']),
                                description=product_dict.get('description', ''),
                                reorder_point=int(product_dict.get('reorder_point', 0)),
                                reorder_quantity=int(product_dict.get('reorder_quantity', 0)),
                                category=product_dict.get('category', ''),
                                supplier=product_dict.get('supplier', '')
                            )
                            
                            self.products[str(product.id)] = product
                except Exception as e:
                    logger.exception(f"Error loading products file: {e}")
                    # Start with empty data if file is corrupted
                    self.products = {}
            
            # Load transactions
            if self.transactions_file.exists():
                try:
                    # Verify integrity first
                    if not await self._verify_file_integrity(self.transactions_file, self.transactions_checksum_file):
                        logger.warning(f"Transactions file integrity check failed, using empty data")
                        self.transactions = []
                    else:
                        with open(self.transactions_file, 'r') as f:
                            transactions_data = json.load(f)
                            
                        for tx_dict in transactions_data:
                            # Convert to StockTransaction model
                            tx = StockTransaction(
                                id=uuid.UUID(tx_dict['id']),
                                product_id=uuid.UUID(tx_dict['product_id']),
                                delta=int(tx_dict['delta']),
                                timestamp=datetime.fromisoformat(tx_dict['timestamp']),
                                transaction_type=tx_dict.get('transaction_type', ''),
                                user=tx_dict.get('user', ''),
                                note=tx_dict.get('note', '')
                            )
                            
                            self.transactions.append(tx)
                except Exception as e:
                    logger.exception(f"Error loading transactions file: {e}")
                    # Start with empty data if file is corrupted
                    self.transactions = []
            
            logger.info(f"Loaded {len(self.products)} products and {len(self.transactions)} transactions from disk")
            self.loaded = True
    
    async def _compute_checksum(self, file_path: Path) -> str:
        """
        Compute SHA-256 checksum of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal string representation of the checksum
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
                
        return sha256.hexdigest()
    
    async def _save_checksum(self, file_path: Path, checksum_file: Path) -> None:
        """
        Save the checksum of a file.
        
        Args:
            file_path: Path to the file
            checksum_file: Path to save the checksum
        """
        try:
            checksum = await self._compute_checksum(file_path)
            
            with open(checksum_file, 'w') as f:
                f.write(checksum)
                
            logger.debug(f"Saved checksum for {file_path.name}")
        except Exception as e:
            logger.exception(f"Error saving checksum: {e}")
            # Continue despite error - this is non-critical
    
    async def _verify_file_integrity(self, file_path: Path, checksum_file: Path) -> bool:
        """
        Verify the integrity of a file using its saved checksum.
        
        Args:
            file_path: Path to the file
            checksum_file: Path to the checksum file
            
        Returns:
            True if integrity check passes, False otherwise
        """
        # If file doesn't exist, there's nothing to verify
        if not file_path.exists():
            return True
        
        # If checksum file doesn't exist, we can't verify
        if not checksum_file.exists():
            logger.warning(f"No checksum file for {file_path.name}")
            return True  # Assume valid to avoid data loss
        
        try:
            # Compute current checksum
            current_checksum = await self._compute_checksum(file_path)
            
            # Read saved checksum
            with open(checksum_file, 'r') as f:
                saved_checksum = f.read().strip()
            
            # Compare checksums
            if current_checksum != saved_checksum:
                logger.warning(f"Integrity check failed for {file_path.name}: "
                              f"saved={saved_checksum}, current={current_checksum}")
                return False
                
            return True
            
        except Exception as e:
            logger.exception(f"Error verifying file integrity: {e}")
            return False  # Fail closed for safety
    
    async def _save_products(self) -> None:
        """Save products to JSON file with atomic write and integrity verification."""
        async with self.file_lock:
            try:
                # Convert products to serializable format
                products_data = []
                for product in self.products.values():
                    product_dict = {
                        'id': str(product.id),
                        'name': product.name,
                        'description': product.description,
                        'sku': product.sku,
                        'price': product.price,
                        'quantity': product.quantity,
                        'reorder_point': product.reorder_point,
                        'reorder_quantity': product.reorder_quantity,
                        'category': product.category,
                        'supplier': product.supplier
                    }
                    products_data.append(product_dict)
                
                # Create a temporary file for atomic write
                with tempfile.NamedTemporaryFile(mode='w', dir=str(self.data_dir), 
                                                delete=False, suffix='.tmp') as tmp:
                    # Write JSON data to temporary file
                    json.dump(products_data, tmp, indent=2, default=str)
                    tmp.flush()  # Ensure all data is written
                    os.fsync(tmp.fileno())  # Force flush to disk
                    tmp_path = Path(tmp.name)
                
                # Create a temporary file for the checksum
                checksum = await self._compute_checksum(tmp_path)
                with tempfile.NamedTemporaryFile(mode='w', dir=str(self.data_dir),
                                              delete=False, suffix='.checksum.tmp') as cksum_tmp:
                    cksum_tmp.write(checksum)
                    cksum_tmp.flush()
                    os.fsync(cksum_tmp.fileno())
                    cksum_tmp_path = Path(cksum_tmp.name)
                
                # Atomically replace the files
                # On Unix, rename is atomic if target exists
                shutil.move(tmp_path, self.products_file)
                shutil.move(cksum_tmp_path, self.products_checksum_file)
                
                logger.debug(f"Saved {len(products_data)} products to {self.products_file}")
                
            except Exception as e:
                logger.exception(f"Error saving products: {e}")
                # Clean up temporary files if they exist
                for path in [tmp_path, cksum_tmp_path]:
                    if 'path' in locals() and path.exists():
                        path.unlink()
                raise StorageError(f"Failed to save products: {e}") from e
    
    async def _save_transactions(self) -> None:
        """Save transactions to JSON file with atomic write and integrity verification."""
        async with self.file_lock:
            try:
                # Convert transactions to serializable format
                transactions_data = []
                for tx in self.transactions:
                    tx_dict = {
                        'id': str(tx.id),
                        'product_id': str(tx.product_id),
                        'delta': tx.delta,
                        'timestamp': tx.timestamp.isoformat(),
                    }
                    
                    # Add optional fields
                    if hasattr(tx, 'transaction_type') and tx.transaction_type:
                        tx_dict['transaction_type'] = tx.transaction_type
                    if hasattr(tx, 'user') and tx.user:
                        tx_dict['user'] = tx.user
                    if hasattr(tx, 'note') and tx.note:
                        tx_dict['note'] = tx.note
                        
                    transactions_data.append(tx_dict)
                
                # Create a temporary file for atomic write
                with tempfile.NamedTemporaryFile(mode='w', dir=str(self.data_dir), 
                                              delete=False, suffix='.tmp') as tmp:
                    # Write JSON data to temporary file
                    json.dump(transactions_data, tmp, indent=2, default=str)
                    tmp.flush()  # Ensure all data is written
                    os.fsync(tmp.fileno())  # Force flush to disk
                    tmp_path = Path(tmp.name)
                
                # Create a temporary file for the checksum
                checksum = await self._compute_checksum(tmp_path)
                with tempfile.NamedTemporaryFile(mode='w', dir=str(self.data_dir),
                                              delete=False, suffix='.checksum.tmp') as cksum_tmp:
                    cksum_tmp.write(checksum)
                    cksum_tmp.flush()
                    os.fsync(cksum_tmp.fileno())
                    cksum_tmp_path = Path(cksum_tmp.name)
                
                # Atomically replace the files
                shutil.move(tmp_path, self.transactions_file)
                shutil.move(cksum_tmp_path, self.transactions_checksum_file)
                
                logger.debug(f"Saved {len(transactions_data)} transactions to {self.transactions_file}")
                
            except Exception as e:
                logger.exception(f"Error saving transactions: {e}")
                # Clean up temporary files if they exist
                for path in [tmp_path, cksum_tmp_path]:
                    if 'path' in locals() and path.exists():
                        path.unlink()
                raise StorageError(f"Failed to save transactions: {e}") from e
    
    # Recovery methods
    
    async def repair(self) -> Tuple[int, int]:
        """
        Attempt to repair corrupted files by recreating them from the in-memory cache.
        
        Returns:
            Tuple of (products_repaired, transactions_repaired)
            
        Raises:
            StorageError: If repair fails
        """
        async with self.file_lock:
            products_repaired = 0
            transactions_repaired = 0
            
            try:
                # Ensure data is loaded before repair
                await self._ensure_loaded()
                
                # Check products file
                if (self.products_file.exists() and 
                    not await self._verify_file_integrity(self.products_file, self.products_checksum_file)):
                    # Backup corrupted file
                    backup_path = self.products_file.with_suffix('.corrupt')
                    shutil.copy2(self.products_file, backup_path)
                    logger.info(f"Backed up corrupted products file to {backup_path}")
                    
                    # Force save products to recreate file
                    await self._save_products()
                    products_repaired = len(self.products)
                    logger.info(f"Repaired products file with {products_repaired} products")
                
                # Check transactions file
                if (self.transactions_file.exists() and 
                    not await self._verify_file_integrity(self.transactions_file, self.transactions_checksum_file)):
                    # Backup corrupted file
                    backup_path = self.transactions_file.with_suffix('.corrupt')
                    shutil.copy2(self.transactions_file, backup_path)
                    logger.info(f"Backed up corrupted transactions file to {backup_path}")
                    
                    # Force save transactions to recreate file
                    await self._save_transactions()
                    transactions_repaired = len(self.transactions)
                    logger.info(f"Repaired transactions file with {transactions_repaired} transactions")
                
                return (products_repaired, transactions_repaired)
                
            except Exception as e:
                logger.exception(f"Error during repair: {e}")
                raise StorageError(f"Failed to repair corrupted files: {e}") from e
    
    async def backup(self, backup_dir: Union[str, Path]) -> Tuple[Path, Path]:
        """
        Create a backup of all data files.
        
        Args:
            backup_dir: Directory where backups will be saved
            
        Returns:
            Tuple of (products_backup_path, transactions_backup_path)
            
        Raises:
            StorageError: If backup fails
        """
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        async with self.file_lock:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Backup products
                products_backup_path = backup_dir / f"products_{timestamp}.json"
                if self.products_file.exists():
                    shutil.copy2(self.products_file, products_backup_path)
                    # Also backup checksum
                    if self.products_checksum_file.exists():
                        shutil.copy2(self.products_checksum_file, 
                                     products_backup_path.with_suffix('.checksum'))
                else:
                    # Create empty file
                    products_backup_path.write_text("[]")
                
                # Backup transactions
                transactions_backup_path = backup_dir / f"transactions_{timestamp}.json"
                if self.transactions_file.exists():
                    shutil.copy2(self.transactions_file, transactions_backup_path)
                    # Also backup checksum
                    if self.transactions_checksum_file.exists():
                        shutil.copy2(self.transactions_checksum_file,
                                    transactions_backup_path.with_suffix('.checksum'))
                else:
                    # Create empty file
                    transactions_backup_path.write_text("[]")
                
                logger.info(f"Created backup at {backup_dir}")
                return (products_backup_path, transactions_backup_path)
                
            except Exception as e:
                logger.exception(f"Error during backup: {e}")
                raise StorageError(f"Failed to create backup: {e}") from e
    
    async def restore(self, products_path: Union[str, Path], transactions_path: Union[str, Path]) -> None:
        """
        Restore data from backup files.
        
        Args:
            products_path: Path to products backup file
            transactions_path: Path to transactions backup file
            
        Raises:
            StorageError: If restore fails
        """
        products_path = Path(products_path)
        transactions_path = Path(transactions_path)
        
        if not products_path.exists() or not transactions_path.exists():
            raise StorageError("Backup files not found")
            
        async with self.file_lock:
            try:
                # Backup current files before restore
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pre_restore_dir = self.data_dir / f"pre_restore_{timestamp}"
                pre_restore_dir.mkdir(exist_ok=True)
                
                await self.backup(pre_restore_dir)
                logger.info(f"Created pre-restore backup in {pre_restore_dir}")
                
                # Copy backup files to data directory
                shutil.copy2(products_path, self.products_file)
                shutil.copy2(transactions_path, self.transactions_file)
                
                # Compute and save checksums
                await self._save_checksum(self.products_file, self.products_checksum_file)
                await self._save_checksum(self.transactions_file, self.transactions_checksum_file)
                
                # Reset in-memory cache and load from restored files
                self.products = {}
                self.transactions = []
                self.loaded = False
                await self._load_data()
                
                logger.info(f"Restored data from backup: {len(self.products)} products and " 
                           f"{len(self.transactions)} transactions")
                
            except Exception as e:
                logger.exception(f"Error during restore: {e}")
                raise StorageError(f"Failed to restore data: {e}") from e
    
    # StorageAdapter interface implementation
    
    async def save_product(self, product: Product) -> None:
        """
        Save a product to storage.
        
        Args:
            product: The product to save
            
        Raises:
            StorageError: If the product cannot be saved
        """
        await self._ensure_loaded()
        
        # Store in memory
        self.products[str(product.id)] = product
        
        # Save to disk
        await self._save_products()
        
    async def save_transaction(self, transaction: StockTransaction) -> None:
        """
        Save a transaction to storage.
        
        Args:
            transaction: The transaction to save
            
        Raises:
            StorageError: If the transaction cannot be saved
        """
        await self._ensure_loaded()
        
        # Check if transaction already exists
        for i, tx in enumerate(self.transactions):
            if str(tx.id) == str(transaction.id):
                # Replace existing transaction
                self.transactions[i] = transaction
                break
        else:
            # Add new transaction
            self.transactions.append(transaction)
        
        # Save to disk
        await self._save_transactions()
        
    async def load_all(self) -> Dict[str, List[Any]]:
        """
        Load all products and transactions from storage.
        
        Returns:
            A dictionary with 'products' and 'transactions' lists
            
        Raises:
            StorageError: If data cannot be loaded from storage
        """
        await self._ensure_loaded()
        
        return {
            'products': list(self.products.values()),
            'transactions': self.transactions
        }
        
    async def get_product(self, product_id: str) -> Optional[Product]:
        """
        Get a product by ID.
        
        Args:
            product_id: The ID of the product to retrieve
            
        Returns:
            The product if found, None otherwise
            
        Raises:
            StorageError: If the product cannot be loaded
        """
        await self._ensure_loaded()
        return self.products.get(product_id)
        
    async def get_transactions(self, 
                           product_id: Optional[str] = None,
                           limit: Optional[int] = None) -> List[StockTransaction]:
        """
        Get transactions, optionally filtered by product ID.
        
        Args:
            product_id: If provided, only transactions for this product will be returned
            limit: Maximum number of transactions to return
            
        Returns:
            List of transactions matching the criteria
            
        Raises:
            StorageError: If transactions cannot be loaded
        """
        await self._ensure_loaded()
        
        # Filter by product_id if specified
        if product_id:
            filtered = [t for t in self.transactions if str(t.product_id) == product_id]
        else:
            filtered = self.transactions.copy()
        
        # Sort by timestamp (descending)
        filtered.sort(key=lambda t: t.timestamp, reverse=True)
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            filtered = filtered[:limit]
            
        return filtered
    
    @asynccontextmanager
    async def stream_products(self) -> AsyncIterator[Product]:
        """
        Stream products from storage one at a time.
        
        Yields:
            AsyncIterator of Product objects
            
        Raises:
            StorageError: If products cannot be streamed from storage
        """
        await self._ensure_loaded()
        for product in self.products.values():
            yield product
    
    @asynccontextmanager
    async def stream_transactions(self) -> AsyncIterator[StockTransaction]:
        """
        Stream transactions from storage one at a time.
        
        Yields:
            AsyncIterator of StockTransaction objects
            
        Raises:
            StorageError: If transactions cannot be streamed from storage
        """
        await self._ensure_loaded()
        for transaction in self.transactions:
            yield transaction
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for handling transactions.
        
        This implementation takes a snapshot of the current state and rolls back
        to that state if an exception occurs.
        
        Yields:
            None
            
        Raises:
            StorageError: If the transaction fails
        """
        await self._ensure_loaded()
        
        # Take snapshots
        products_snapshot = {k: v.copy() for k, v in self.products.items()}
        transactions_snapshot = self.transactions.copy()
        
        try:
            # Execute transaction body
            yield
            
            # Save changes to disk on successful completion
            await self._save_products()
            await self._save_transactions()
            
        except Exception as e:
            # Rollback to snapshots
            self.products = products_snapshot
            self.transactions = transactions_snapshot
            
            logger.error(f"Transaction rolled back due to error: {e}")
            raise StorageError(f"Transaction failed: {e}") from e


# Utility functions for testing data corruption

async def corrupt_file(file_path: Path, corrupt_bytes: int = 10) -> bool:
    """
    Intentionally corrupt a file for testing recovery mechanisms.
    
    Args:
        file_path: Path to the file to corrupt
        corrupt_bytes: Number of bytes to corrupt
        
    Returns:
        True if file was corrupted, False otherwise
    """
    if not file_path.exists():
        logger.error(f"Cannot corrupt non-existent file: {file_path}")
        return False
        
    try:
        # Read file
        with open(file_path, 'rb') as f:
            data = bytearray(f.read())
        
        # File too small to corrupt
        if len(data) <= corrupt_bytes:
            logger.error(f"File too small to corrupt: {file_path}")
            return False
        
        # Corrupt some bytes in the middle
        middle = len(data) // 2
        for i in range(corrupt_bytes):
            if middle + i < len(data):
                data[middle + i] = (data[middle + i] + 1) % 256
        
        # Write corrupted data back
        with open(file_path, 'wb') as f:
            f.write(data)
            
        logger.warning(f"Corrupted {corrupt_bytes} bytes in {file_path}")
        return True
        
    except Exception as e:
        logger.exception(f"Error corrupting file: {e}")
        return False


# Example usage
async def example_json_adapter():
    """Example showing how to use the JSONAdapter with corruption detection and atomic writes."""
    import uuid
    from datetime import datetime
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create adapter
    data_dir = Path("./data/dev")
    adapter = JSONAdapter(data_dir)
    
    try:
        # Create and save a sample product
        product_id = uuid.uuid4()
        product = Product(
            id=product_id,
            name="Test Product",
            description="A test product for the JSON adapter",
            sku="TEST-001",
            price=29.99,
            quantity=100,
            reorder_point=20,
            reorder_quantity=50,
            category="Test",
            supplier="Test Supplier"
        )
        
        print(f"Saving product: {product.name}")
        await adapter.save_product(product)
        
        # Create and save a transaction
        transaction = StockTransaction(
            id=uuid.uuid4(),
            product_id=product_id,
            delta=-10,
            timestamp=datetime.now(),
            transaction_type="adjustment",
            user="test_user",
            note="Test transaction"
        )
        
        print(f"Saving transaction: {transaction.id}")
        await adapter.save_transaction(transaction)
        
        # Load and print data
        data = await adapter.load_all()
        print(f"Loaded {len(data['products'])} products and {len(data['transactions'])} transactions")
        
        # Demonstrate transaction support
        print("\nDemonstrating transaction with rollback:")
        try:
            async with adapter.transaction():
                # Update product
                product = await adapter.get_product(str(product_id))
                product.quantity -= 5
                await adapter.save_product(product)
                
                # Save another transaction
                tx2 = StockTransaction(
                    id=uuid.uuid4(),
                    product_id=product_id,
                    delta=-5,
                    timestamp=datetime.now(),
                    transaction_type="adjustment",
                    user="test_user",
                    note="Transaction that will be rolled back"
                )
                await adapter.save_transaction(tx2)
                
                # Force an error
                raise ValueError("Simulated error to trigger rollback")
                
        except StorageError:
            print("Transaction rolled back as expected")
        
        # Verify rollback
        product = await adapter.get_product(str(product_id))
        print(f"Product quantity after rollback: {product.quantity}")
        
        # Demonstrate corruption detection and repair
        print("\nDemonstrating file corruption and repair:")
        
        # Create backup before corruption
        backup_dir = data_dir / "backups"
        backup_paths = await adapter.backup(backup_dir)
        print(f"Created backup at {backup_dir}")
        
        # Corrupt the products file
        if await corrupt_file(adapter.products_file):
            print(f"Corrupted products file")
            
            # Try to load data (should detect corruption)
            try:
                # Reset loaded flag to force reload
                adapter.loaded = False
                await adapter.load_all()
            except StorageError as e:
                print(f"Error detected during load: {e}")
            
            # Repair the corrupted file
            repaired = await adapter.repair()
            print(f"Repaired {repaired[0]} products and {repaired[1]} transactions")
            
            # Verify repair worked
            data = await adapter.load_all()
            print(f"Successfully loaded {len(data['products'])} products after repair")
            
        # Demonstrate restore from backup
        print("\nDemonstrating restore from backup:")
        await adapter.restore(backup_paths[0], backup_paths[1])
        print("Data restored from backup")
        
        # Verify restore
        data = await adapter.load_all()
        print(f"Loaded {len(data['products'])} products and {len(data['transactions'])} transactions after restore")
        
    except Exception as e:
        print(f"Error during example: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_json_adapter())
