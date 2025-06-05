"""
inventory_tracker/commands/sync.py - Data synchronization command

This module provides functionality to synchronize (migrate) data between
different storage adapters (e.g., from JSON files to SQLite database).
It includes schema validation to detect mismatches early and fail fast.
"""

import asyncio
import logging
import sys
import time
from typing import Dict, List, Tuple, Type, Optional, Any, Set, AsyncIterator
import argparse
from enum import Enum
from pydantic import BaseModel, ValidationError
import json

from ..store.adapter import StorageAdapter
from ..store.sql_adapter import SQLAdapter
from ..store.file_adapter import FileAdapter
# Import any other adapters as needed

from ..models.product import Product
from ..models.inventory import StockTransaction
from ..models.exceptions import NotFoundError, DatabaseError

# Set up logger
logger = logging.getLogger(__name__)

class SyncError(Exception):
    """Exception raised for synchronization errors"""
    pass

class SchemaError(SyncError):
    """Exception raised for schema validation errors"""
    pass

class SyncDirection(str, Enum):
    """Enum for synchronization direction options"""
    BIDIRECTIONAL = "bidirectional"  # Sync both ways, resolving conflicts
    SOURCE_TO_TARGET = "source-to-target"  # Source data overwrites target
    TARGET_TO_SOURCE = "target-to-source"  # Target data overwrites source

class SyncStats:
    """Statistics tracking for sync operations"""
    def __init__(self):
        self.products_synced = 0
        self.transactions_synced = 0
        self.products_failed = 0
        self.transactions_failed = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        self.start_time = time.time()
        self.end_time: Optional[float] = None
    
    @property
    def elapsed_time(self) -> float:
        """Return elapsed time in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def complete(self):
        """Mark the sync operation as complete"""
        self.end_time = time.time()
    
    def __str__(self) -> str:
        """Return a string representation of the stats"""
        return f"""
Sync Statistics:
----------------
Products synced: {self.products_synced}
Transactions synced: {self.transactions_synced}
Products failed: {self.products_failed}
Transactions failed: {self.transactions_failed}
Conflicts detected: {self.conflicts_detected}
Conflicts resolved: {self.conflicts_resolved}
Elapsed time: {self.elapsed_time:.2f} seconds
"""

async def validate_schema_compatibility(
    source: StorageAdapter, 
    target: StorageAdapter
) -> Tuple[bool, List[str]]:
    """
    Validate that source and target adapters have compatible schemas.
    
    Args:
        source: Source adapter
        target: Target adapter
        
    Returns:
        Tuple of (is_compatible, issues) where issues is a list of problems found
    """
    issues = []
    
    # Check if both adapters implement all required methods
    adapter_methods = [
        "get_product", "save_product", "delete_product", "list_products",
        "get_transaction", "save_transaction", "delete_transaction", "list_transactions",
    ]
    
    for method in adapter_methods:
        if not (hasattr(source, method) and callable(getattr(source, method))):
            issues.append(f"Source adapter missing required method: {method}")
        if not (hasattr(target, method) and callable(getattr(target, method))):
            issues.append(f"Target adapter missing required method: {method}")
    
    if issues:
        return False, issues
    
    # Test schema compatibility by sampling data
    try:
        # Check product model compatibility
        source_products = await source.list_products(limit=1)
        if source_products:
            sample_product = source_products[0]
            try:
                # Test serialization/deserialization between adapters
                product_data = sample_product.model_dump()
                
                # Try to save to target - this will validate if target can handle all fields
                await target.save_product(sample_product)
                await target.delete_product(sample_product.id)
            except ValidationError as e:
                issues.append(f"Product schema mismatch: {e}")
            except Exception as e:
                issues.append(f"Error testing product compatibility: {e}")
        
        # Do the same for transactions
        source_transactions = await source.list_transactions(limit=1)
        if source_transactions:
            sample_transaction = source_transactions[0]
            try:
                # First ensure the product exists
                try:
                    await target.get_product(sample_transaction.product_id)
                except:
                    sample_product = await source.get_product(sample_transaction.product_id)
                    if sample_product:
                        await target.save_product(sample_product)
                
                # Test serialization/deserialization
                transaction_data = sample_transaction.model_dump()
                
                # Try to save to target
                await target.save_transaction(sample_transaction)
                await target.delete_transaction(sample_transaction.id)
            except ValidationError as e:
                issues.append(f"Transaction schema mismatch: {e}")
            except Exception as e:
                issues.append(f"Error testing transaction compatibility: {e}")
    
    except Exception as e:
        issues.append(f"Error validating schema compatibility: {e}")
    
    return len(issues) == 0, issues

async def sync_products(
    source: StorageAdapter,
    target: StorageAdapter,
    direction: SyncDirection = SyncDirection.SOURCE_TO_TARGET,
    stats: Optional[SyncStats] = None
) -> SyncStats:
    """
    Synchronize products between source and target adapters.
    
    Args:
        source: Source adapter
        target: Target adapter
        direction: Direction of synchronization
        stats: Optional stats object to track progress
        
    Returns:
        SyncStats object with sync statistics
    """
    if stats is None:
        stats = SyncStats()
    
    logger.info("Starting product synchronization...")
    
    # Handle source to target sync
    if direction in [SyncDirection.SOURCE_TO_TARGET, SyncDirection.BIDIRECTIONAL]:
        try:
            # Use streaming to efficiently process large datasets
            async with source.stream_products() as products:
                async for product in products:
                    try:
                        # Check if product exists in target
                        existing_product = await target.get_product(product.id)
                        
                        if existing_product is not None and direction == SyncDirection.BIDIRECTIONAL:
                            # In bidirectional sync, handle conflicts (here we use source as source of truth)
                            stats.conflicts_detected += 1
                            stats.conflicts_resolved += 1
                        
                        # Save the product to the target
                        await target.save_product(product)
                        stats.products_synced += 1
                        
                        # Log progress periodically
                        if stats.products_synced % 100 == 0:
                            logger.info(f"Synced {stats.products_synced} products so far...")
                    
                    except ValidationError as e:
                        logger.error(f"Schema validation error for product {product.id}: {e}")
                        stats.products_failed += 1
                    except Exception as e:
                        logger.error(f"Error syncing product {product.id}: {e}")
                        stats.products_failed += 1
        
        except Exception as e:
            logger.error(f"Error during product synchronization: {e}")
            raise SyncError(f"Product synchronization failed: {e}")
    
    # Handle target to source sync for bidirectional mode
    if direction == SyncDirection.TARGET_TO_SOURCE:
        try:
            # Use streaming to efficiently process large datasets
            async with target.stream_products() as products:
                async for product in products:
                    try:
                        # Check if product exists in source
                        existing_product = await source.get_product(product.id)
                        
                        if existing_product is None:
                            # Only sync products that don't exist in the source
                            await source.save_product(product)
                            stats.products_synced += 1
                        
                        # Log progress periodically
                        if stats.products_synced % 100 == 0:
                            logger.info(f"Synced {stats.products_synced} products so far...")
                    
                    except ValidationError as e:
                        logger.error(f"Schema validation error for product {product.id}: {e}")
                        stats.products_failed += 1
                    except Exception as e:
                        logger.error(f"Error syncing product {product.id}: {e}")
                        stats.products_failed += 1
        
        except Exception as e:
            logger.error(f"Error during product synchronization: {e}")
            raise SyncError(f"Product synchronization failed: {e}")
    
    logger.info(f"Product synchronization complete. Synced {stats.products_synced} products.")
    return stats

async def sync_transactions(
    source: StorageAdapter,
    target: StorageAdapter,
    direction: SyncDirection = SyncDirection.SOURCE_TO_TARGET,
    stats: Optional[SyncStats] = None
) -> SyncStats:
    """
    Synchronize transactions between source and target adapters.
    
    Args:
        source: Source adapter
        target: Target adapter
        direction: Direction of synchronization
        stats: Optional stats object to track progress
        
    Returns:
        SyncStats object with sync statistics
    """
    if stats is None:
        stats = SyncStats()
    
    logger.info("Starting transaction synchronization...")
    
    # Handle source to target sync
    if direction in [SyncDirection.SOURCE_TO_TARGET, SyncDirection.BIDIRECTIONAL]:
        try:
            # Use streaming to efficiently process large datasets
            async with source.stream_transactions() as transactions:
                async for transaction in transactions:
                    try:
                        # Check if product exists in target
                        try:
                            await target.get_product(transaction.product_id)
                        except NotFoundError:
                            # Product doesn't exist in target, we need to sync it first
                            logger.info(f"Product {transaction.product_id} not found in target, syncing it now")
                            product = await source.get_product(transaction.product_id)
                            if product:
                                await target.save_product(product)
                            else:
                                raise SyncError(f"Cannot find product {transaction.product_id} for transaction {transaction.id}")
                        
                        # Check if transaction exists in target
                        existing_transaction = await target.get_transaction(transaction.id)
                        
                        if existing_transaction is not None and direction == SyncDirection.BIDIRECTIONAL:
                            # Handle conflicts (here we use source as source of truth)
                            stats.conflicts_detected += 1
                            stats.conflicts_resolved += 1
                        
                        # Save the transaction to the target
                        await target.save_transaction(transaction)
                        stats.transactions_synced += 1
                        
                        # Log progress periodically
                        if stats.transactions_synced % 500 == 0:
                            logger.info(f"Synced {stats.transactions_synced} transactions so far...")
                    
                    except ValidationError as e:
                        logger.error(f"Schema validation error for transaction {transaction.id}: {e}")
                        stats.transactions_failed += 1
                    except NotFoundError as e:
                        logger.error(f"Error syncing transaction {transaction.id}: {e}")
                        stats.transactions_failed += 1
                    except Exception as e:
                        logger.error(f"Error syncing transaction {transaction.id}: {e}")
                        stats.transactions_failed += 1
        
        except Exception as e:
            logger.error(f"Error during transaction synchronization: {e}")
            raise SyncError(f"Transaction synchronization failed: {e}")
    
    # Handle target to source sync for bidirectional mode
    if direction == SyncDirection.TARGET_TO_SOURCE:
        try:
            # Use streaming to efficiently process large datasets
            async with target.stream_transactions() as transactions:
                async for transaction in transactions:
                    try:
                        # Check if product exists in source
                        try:
                            await source.get_product(transaction.product_id)
                        except NotFoundError:
                            # Product doesn't exist in source, we need to sync it first
                            product = await target.get_product(transaction.product_id)
                            if product:
                                await source.save_product(product)
                            else:
                                raise SyncError(f"Cannot find product {transaction.product_id} for transaction {transaction.id}")
                        
                        # Check if transaction exists in source
                        existing_transaction = await source.get_transaction(transaction.id)
                        
                        if existing_transaction is None:
                            # Only sync transactions that don't exist in the source
                            await source.save_transaction(transaction)
                            stats.transactions_synced += 1
                        
                        # Log progress periodically
                        if stats.transactions_synced % 500 == 0:
                            logger.info(f"Synced {stats.transactions_synced} transactions so far...")
                    
                    except ValidationError as e:
                        logger.error(f"Schema validation error for transaction {transaction.id}: {e}")
                        stats.transactions_failed += 1
                    except NotFoundError as e:
                        logger.error(f"Error syncing transaction {transaction.id}: {e}")
                        stats.transactions_failed += 1
                    except Exception as e:
                        logger.error(f"Error syncing transaction {transaction.id}: {e}")
                        stats.transactions_failed += 1
        
        except Exception as e:
            logger.error(f"Error during transaction synchronization: {e}")
            raise SyncError(f"Transaction synchronization failed: {e}")
    
    logger.info(f"Transaction synchronization complete. Synced {stats.transactions_synced} transactions.")
    return stats

async def sync_data(
    source: StorageAdapter,
    target: StorageAdapter,
    validate_schema: bool = True,
    sync_direction: SyncDirection = SyncDirection.SOURCE_TO_TARGET,
    data_types: List[str] = ["products", "transactions"]
) -> SyncStats:
    """
    Synchronize data between source and target adapters.
    
    Args:
        source: Source adapter
        target: Target adapter
        validate_schema: Whether to validate schema compatibility before syncing
        sync_direction: Direction of synchronization
        data_types: List of data types to synchronize (products, transactions, or both)
        
    Returns:
        SyncStats object with sync statistics
        
    Raises:
        SyncError: If synchronization fails
        SchemaError: If schema validation fails
    """
    logger.info(f"Starting data synchronization: {sync_direction.value}")
    
    # Validate adapter schema compatibility
    if validate_schema:
        logger.info("Validating schema compatibility...")
        compatible, issues = await validate_schema_compatibility(source, target)
        if not compatible:
            error_message = "Schema validation failed:\n" + "\n".join(issues)
            logger.error(error_message)
            raise SchemaError(error_message)
        logger.info("Schema validation successful")
    
    # Initialize stats
    stats = SyncStats()
    
    try:
        # Sync products if requested
        if "products" in data_types:
            await sync_products(source, target, sync_direction, stats)
        
        # Sync transactions if requested
        if "transactions" in data_types:
            await sync_transactions(source, target, sync_direction, stats)
    
    except Exception as e:
        logger.error(f"Synchronization failed: {e}")
        raise SyncError(f"Synchronization failed: {e}")
    
    # Mark sync as complete
    stats.complete()
    logger.info(f"Data synchronization complete in {stats.elapsed_time:.2f} seconds")
    return stats

def create_adapter(adapter_type: str, connection_string: str) -> StorageAdapter:
    """
    Create an adapter instance based on type and connection string.
    
    Args:
        adapter_type: Type of adapter (sql, file, etc.)
        connection_string: Connection string or path for the adapter
        
    Returns:
        StorageAdapter instance
        
    Raises:
        ValueError: If adapter type is invalid
    """
    if adapter_type.lower() == "sql":
        return SQLAdapter(connection_string=connection_string)
    elif adapter_type.lower() == "file":
        from pathlib import Path
        return FileAdapter(data_dir=Path(connection_string))
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")

def setup_parser(subparsers):
    """
    Set up the command line parser for the sync command.
    
    Args:
        subparsers: Subparsers object from the main parser
    """
    sync_parser = subparsers.add_parser(
        'sync',
        help='Synchronize data between different storage adapters'
    )
    
    # Source adapter configuration
    sync_parser.add_argument(
        '--source-type',
        choices=['sql', 'file'],
        required=True,
        help='Type of source adapter'
    )
    
    sync_parser.add_argument(
        '--source-connection',
        type=str,
        required=True,
        help='Connection string or path for source adapter'
    )
    
    # Target adapter configuration
    sync_parser.add_argument(
        '--target-type',
        choices=['sql', 'file'],
        required=True,
        help='Type of target adapter'
    )
    
    sync_parser.add_argument(
        '--target-connection',
        type=str,
        required=True,
        help='Connection string or path for target adapter'
    )
    
    # Sync options
    sync_parser.add_argument(
        '--direction',
        choices=[d.value for d in SyncDirection],
        default=SyncDirection.SOURCE_TO_TARGET.value,
        help='Direction of synchronization (default: source-to-target)'
    )
    
    sync_parser.add_argument(
        '--data-types',
        choices=['products', 'transactions', 'all'],
        default='all',
        help='Data types to synchronize (default: all)'
    )
    
    sync_parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip schema validation (not recommended)'
    )
    
    sync_parser.set_defaults(func=handle_sync)

async def handle_sync(args, _):
    """
    Handle the sync command.
    
    Args:
        args: Parsed command line arguments
    """
    try:
        logger.info(f"Setting up sync from {args.source_type} to {args.target_type}")
        
        # Create source adapter
        source = create_adapter(args.source_type, args.source_connection)
        await source.connect()
        
        # Create target adapter
        target = create_adapter(args.target_type, args.target_connection)
        await target.connect()
        
        try:
            # Determine data types to sync
            data_types = ["products", "transactions"] if args.data_types == "all" else [args.data_types]
            
            # Run the sync operation
            stats = await sync_data(
                source=source,
                target=target,
                validate_schema=not args.skip_validation,
                sync_direction=SyncDirection(args.direction),
                data_types=data_types
            )
            
            # Print statistics
            print(stats)
            
        finally:
            # Ensure adapters are disconnected
            await source.disconnect()
            await target.disconnect()
            
    except SchemaError as e:
        logger.error(f"Schema validation failed: {e}")
        print(f"Error: Schema validation failed. Use --skip-validation to bypass (not recommended).", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)
    except SyncError as e:
        logger.error(f"Synchronization failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during sync: {e}")
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    print("This module should be imported and used as part of the InventoryTracker application.")