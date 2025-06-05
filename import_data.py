#!/usr/bin/env python3
"""
commands/import_data.py - Data Import Command for InventoryTracker

This command ingests data from JSON or CSV files, validates it via Pydantic,
and commits it to the database in transactions. It provides detailed row-level
error reporting without stopping the entire import process.
"""

import os
import sys
import json
import csv
import logging
import argparse
import asyncio
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Iterator
import uuid
import traceback
from pydantic import BaseModel, ValidationError, Field

# Add parent directory to path so we can import our packages
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from models.product import Product
from models.inventory import StockTransaction
from store import get_adapter, StorageAdapter, StorageError

# Set up logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImportType(str, Enum):
    """Type of data being imported"""
    PRODUCTS = "products"
    TRANSACTIONS = "transactions"
    
    
class FileFormat(str, Enum):
    """Format of the import file"""
    JSON = "json"
    CSV = "csv"
    AUTO = "auto"  # Auto-detect from file extension


class ImportMode(str, Enum):
    """Mode of import operation"""
    INSERT = "insert"  # Only insert new records
    UPDATE = "update"  # Only update existing records
    UPSERT = "upsert"  # Insert new and update existing (default)


class ImportStats(BaseModel):
    """Statistics for an import operation"""
    total_rows: int = 0
    successful: int = 0
    failed: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0
    warnings: int = 0
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0
    
    def complete(self):
        """Mark the import as complete and calculate duration"""
        self.completed_at = datetime.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def __str__(self) -> str:
        """Return a string representation of the statistics"""
        lines = [
            f"Import completed in {self.duration_seconds:.2f} seconds",
            f"Total rows: {self.total_rows}",
            f"Successful: {self.successful} ({self.successful/self.total_rows*100:.1f}% of total)" if self.total_rows else "Successful: 0",
            f"  - Created: {self.created}",
            f"  - Updated: {self.updated}",
            f"  - Skipped: {self.skipped}",
            f"Failed: {self.failed} ({self.failed/self.total_rows*100:.1f}% of total)" if self.total_rows else "Failed: 0",
            f"Warnings: {self.warnings}"
        ]
        return "\n".join(lines)


class RowError(BaseModel):
    """Information about an error in a row during import"""
    row_number: int
    row_data: Dict[str, Any]
    error_type: str
    error_message: str
    field: Optional[str] = None
    is_critical: bool = False
    
    def __str__(self) -> str:
        """Return a string representation of the error"""
        if self.field:
            return f"Row {self.row_number}: Error in field '{self.field}' - {self.error_message}"
        else:
            return f"Row {self.row_number}: {self.error_message}"


class ImportReport(BaseModel):
    """Report of an import operation with detailed error information"""
    import_type: ImportType
    file_path: str
    file_format: FileFormat
    import_mode: ImportMode
    stats: ImportStats = Field(default_factory=ImportStats)
    errors: List[RowError] = []
    
    def add_error(self, row_number: int, row_data: Dict[str, Any], 
                 error_type: str, error_message: str, 
                 field: Optional[str] = None, is_critical: bool = False) -> None:
        """Add an error to the report"""
        error = RowError(
            row_number=row_number,
            row_data=row_data,
            error_type=error_type,
            error_message=error_message,
            field=field,
            is_critical=is_critical
        )
        self.errors.append(error)
        self.stats.failed += 1
        if is_critical:
            logger.error(str(error))
        else:
            logger.warning(str(error))
    
    def save_to_file(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save the report to a file.
        
        Args:
            output_path: Optional specific path for the report. If not provided,
                         a default name will be generated.
                         
        Returns:
            Path to the saved report file
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.import_type}_import_report_{timestamp}.json"
            output_path = Path(f"./reports/{filename}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary and save as JSON
        with open(output_path, 'w') as f:
            json.dump(self.dict(exclude_none=True), f, indent=2, default=str)
            
        return output_path
    
    def print_summary(self, include_errors: bool = True, max_errors: int = 10) -> None:
        """
        Print a summary of the import operation.
        
        Args:
            include_errors: Whether to include error details in the summary
            max_errors: Maximum number of errors to include in the summary
        """
        print("\n" + "="*60)
        print(f"IMPORT SUMMARY: {self.import_type.upper()}")
        print("="*60)
        print(f"File: {self.file_path}")
        print(f"Format: {self.file_format}")
        print(f"Mode: {self.import_mode}")
        print("-"*60)
        print(str(self.stats))
        
        if include_errors and self.errors:
            print("-"*60)
            print(f"ERRORS ({len(self.errors)} total):")
            
            # Display the first max_errors errors
            for i, error in enumerate(self.errors):
                if i >= max_errors:
                    remaining = len(self.errors) - max_errors
                    print(f"...and {remaining} more errors (see report file for details)")
                    break
                print(f"{i+1}. {error}")
                
        print("="*60)
        if self.errors:
            print(f"Detailed error report saved to: {self.save_to_file().absolute()}")
        print()


class DataImporter:
    """
    Handles importing data from files into the inventory system.
    
    The importer supports both JSON and CSV formats and provides detailed
    validation and error reporting while allowing the import to continue
    despite row-level errors.
    """
    
    def __init__(self, adapter: StorageAdapter):
        """
        Initialize the data importer.
        
        Args:
            adapter: The storage adapter to use for persisting data
        """
        self.adapter = adapter
    
    async def import_file(self, 
                         file_path: Union[str, Path],
                         import_type: ImportType,
                         file_format: FileFormat = FileFormat.AUTO,
                         import_mode: ImportMode = ImportMode.UPSERT,
                         batch_size: int = 100) -> ImportReport:
        """
        Import data from a file.
        
        Args:
            file_path: Path to the file to import
            import_type: Type of data to import (products or transactions)
            file_format: Format of the file (auto-detect by default)
            import_mode: Import mode (upsert, insert, or update)
            batch_size: Number of rows to process in each transaction
            
        Returns:
            ImportReport with statistics and error information
        """
        file_path = Path(file_path)
        
        # Detect format from extension if set to auto
        if file_format == FileFormat.AUTO:
            extension = file_path.suffix.lower()
            if extension == '.json':
                file_format = FileFormat.JSON
            elif extension == '.csv':
                file_format = FileFormat.CSV
            else:
                raise ValueError(f"Cannot auto-detect format for file with extension '{extension}'")
        
        # Create import report
        report = ImportReport(
            import_type=import_type,
            file_path=str(file_path),
            file_format=file_format,
            import_mode=import_mode
        )
        
        # Check if file exists
        if not file_path.exists():
            report.add_error(0, {}, "FileNotFound", 
                           f"File not found: {file_path}", 
                           is_critical=True)
            report.stats.complete()
            return report
        
        # Call appropriate import method based on format and type
        try:
            if file_format == FileFormat.JSON:
                if import_type == ImportType.PRODUCTS:
                    return await self._import_products_json(file_path, import_mode, batch_size)
                else:  # TRANSACTIONS
                    return await self._import_transactions_json(file_path, import_mode, batch_size)
            else:  # CSV
                if import_type == ImportType.PRODUCTS:
                    return await self._import_products_csv(file_path, import_mode, batch_size)
                else:  # TRANSACTIONS
                    return await self._import_transactions_csv(file_path, import_mode, batch_size)
        except Exception as e:
            logger.exception(f"Critical error during import: {e}")
            report.add_error(0, {}, "ImportError", 
                           f"Critical error during import: {str(e)}", 
                           is_critical=True)
            report.stats.complete()
            return report
    
    async def _import_products_json(self, 
                                   file_path: Path,
                                   import_mode: ImportMode,
                                   batch_size: int) -> ImportReport:
        """
        Import products from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            import_mode: Import mode to use
            batch_size: Number of products to import in each transaction
            
        Returns:
            ImportReport with statistics and error information
        """
        report = ImportReport(
            import_type=ImportType.PRODUCTS,
            file_path=str(file_path),
            file_format=FileFormat.JSON,
            import_mode=import_mode
        )
        
        try:
            # Load JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle both array of products or object with products array
            if isinstance(data, dict) and 'products' in data:
                products_data = data['products']
            elif isinstance(data, list):
                products_data = data
            else:
                report.add_error(0, {}, "InvalidFormat", 
                               "JSON must contain either an array of products or an object with a 'products' array", 
                               is_critical=True)
                report.stats.complete()
                return report
            
            report.stats.total_rows = len(products_data)
            
            # Process products in batches
            for i in range(0, len(products_data), batch_size):
                batch = products_data[i:i+batch_size]
                await self._process_product_batch(batch, import_mode, report)
                
            report.stats.successful = report.stats.created + report.stats.updated + report.stats.skipped
            report.stats.complete()
            return report
            
        except json.JSONDecodeError as e:
            report.add_error(0, {}, "JSONDecodeError", 
                           f"Invalid JSON format: {str(e)}", 
                           is_critical=True)
        except Exception as e:
            report.add_error(0, {}, "ImportError", 
                           f"Error importing JSON file: {str(e)}", 
                           is_critical=True)
            
        report.stats.complete()
        return report

    async def _import_products_csv(self, 
                                  file_path: Path,
                                  import_mode: ImportMode,
                                  batch_size: int) -> ImportReport:
        """
        Import products from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            import_mode: Import mode to use
            batch_size: Number of products to import in each transaction
            
        Returns:
            ImportReport with statistics and error information
        """
        report = ImportReport(
            import_type=ImportType.PRODUCTS,
            file_path=str(file_path),
            file_format=FileFormat.CSV,
            import_mode=import_mode
        )
        
        try:
            # Count total rows for statistics (excluding header)
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                report.stats.total_rows = sum(1 for _ in reader)
            
            # Process the CSV file
            with open(file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                
                # Check required columns
                required_columns = ['name', 'sku', 'price', 'quantity']
                missing_columns = [col for col in required_columns if col not in reader.fieldnames]
                
                if missing_columns:
                    report.add_error(0, {}, "MissingColumns", 
                                   f"CSV is missing required columns: {', '.join(missing_columns)}", 
                                   is_critical=True)
                    report.stats.complete()
                    return report
                
                # Process in batches
                batch = []
                row_num = 1  # Start at 1 because we've already skipped the header
                
                for row in reader:
                    row_num += 1
                    batch.append((row_num, row))
                    
                    if len(batch) >= batch_size:
                        await self._process_product_batch([row for _, row in batch], import_mode, report, row_numbers=[num for num, _ in batch])
                        batch = []
                
                # Process any remaining items
                if batch:
                    await self._process_product_batch([row for _, row in batch], import_mode, report, row_numbers=[num for num, _ in batch])
                    
            report.stats.successful = report.stats.created + report.stats.updated + report.stats.skipped
            report.stats.complete()
            return report
            
        except Exception as e:
            logger.exception(f"Error in _import_products_csv: {e}")
            report.add_error(0, {}, "ImportError", 
                           f"Error importing CSV file: {str(e)}", 
                           is_critical=True)
            
        report.stats.complete()
        return report
            
    async def _process_product_batch(self, 
                                    product_batch: List[Dict[str, Any]], 
                                    import_mode: ImportMode,
                                    report: ImportReport,
                                    row_numbers: Optional[List[int]] = None) -> None:
        """
        Process a batch of products, validating and saving them.
        
        Args:
            product_batch: List of product data dictionaries
            import_mode: Import mode to use
            report: ImportReport to update with results
            row_numbers: Optional list of row numbers for error reporting
        """
        if row_numbers is None:
            # If not provided, use 1-based index in the batch
            row_numbers = list(range(1, len(product_batch) + 1))
        
        # Prepare collections for batch processing
        to_create: List[Product] = []
        to_update: List[Product] = []
        existing_products: Dict[str, Product] = {}
        
        # First pass: validate all products in the batch
        for i, product_data in enumerate(product_batch):
            row_num = row_numbers[i]
            
            try:
                # Convert types for numeric fields
                try:
                    if 'price' in product_data:
                        product_data['price'] = float(product_data['price'])
                    if 'quantity' in product_data:
                        product_data['quantity'] = int(product_data['quantity'])
                    if 'reorder_point' in product_data and product_data['reorder_point']:
                        product_data['reorder_point'] = int(product_data['reorder_point'])
                    if 'reorder_quantity' in product_data and product_data['reorder_quantity']:
                        product_data['reorder_quantity'] = int(product_data['reorder_quantity'])
                except (ValueError, TypeError) as e:
                    report.add_error(row_num, product_data, "TypeError", 
                                   f"Type conversion error: {str(e)}")
                    continue
                
                # Handle ID field
                has_id = 'id' in product_data and product_data['id']
                
                # Check if product exists
                if has_id:
                    try:
                        product_id = str(product_data['id'])
                        existing_product = await self.adapter.get_product(product_id)
                        
                        if existing_product:
                            existing_products[product_id] = existing_product
                    except Exception as e:
                        report.add_error(row_num, product_data, "LookupError", 
                                       f"Error looking up product: {str(e)}")
                        continue
                
                # Create Pydantic model for validation
                try:
                    # For new products
                    if not has_id or product_data['id'] not in existing_products:
                        if import_mode == ImportMode.UPDATE:
                            # Skip if update-only mode and product doesn't exist
                            report.stats.skipped += 1
                            continue
                            
                        # Assign new ID if not provided
                        if not has_id:
                            product_data['id'] = str(uuid.uuid4())
                            
                        # Validate with Pydantic
                        product = Product(**product_data)
                        to_create.append(product)
                    else:  # Existing product
                        if import_mode == ImportMode.INSERT:
                            # Skip if insert-only mode and product exists
                            report.stats.skipped += 1
                            continue
                            
                        # Get existing product
                        existing = existing_products[product_data['id']]
                        
                        # Update fields from import data
                        updated_data = existing.dict()
                        for key, value in product_data.items():
                            if value is not None and key != 'id':  # Don't change ID
                                updated_data[key] = value
                                
                        # Validate updated product
                        product = Product(**updated_data)
                        to_update.append(product)
                        
                except ValidationError as e:
                    # Extract field and error details from ValidationError
                    for error in e.errors():
                        field = '.'.join(map(str, error['loc']))
                        message = error['msg']
                        report.add_error(row_num, product_data, "ValidationError", 
                                       message, field=field)
                    continue
                    
                except Exception as e:
                    report.add_error(row_num, product_data, "ProcessingError", 
                                   f"Error processing product: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.exception(f"Unexpected error in row {row_num}")
                report.add_error(row_num, product_data, "UnexpectedError", 
                               f"Unexpected error: {str(e)}")
        
        # Second pass: save valid products within a transaction
        if to_create or to_update:
            try:
                async with self.adapter.transaction():
                    # Save new products
                    for product in to_create:
                        await self.adapter.save_product(product)
                        report.stats.created += 1
                    
                    # Update existing products
                    for product in to_update:
                        await self.adapter.save_product(product)
                        report.stats.updated += 1
                        
                logger.info(f"Saved batch: {len(to_create)} created, {len(to_update)} updated")
                
            except Exception as e:
                logger.error(f"Error saving batch: {e}")
                # Mark all products in batch as failed
                for i in range(len(product_batch)):
                    if i < len(row_numbers):
                        report.add_error(
                            row_numbers[i],
                            product_batch[i], 
                            "TransactionError",
                            f"Failed to save in transaction: {str(e)}"
                        )

    async def _import_transactions_json(self, 
                                      file_path: Path,
                                      import_mode: ImportMode,
                                      batch_size: int) -> ImportReport:
        """
        Import transactions from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            import_mode: Import mode to use
            batch_size: Number of transactions to import in each transaction
            
        Returns:
            ImportReport with statistics and error information
        """
        report = ImportReport(
            import_type=ImportType.TRANSACTIONS,
            file_path=str(file_path),
            file_format=FileFormat.JSON,
            import_mode=import_mode
        )
        
        try:
            # Load JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle both array of transactions or object with transactions array
            if isinstance(data, dict) and 'transactions' in data:
                transactions_data = data['transactions']
            elif isinstance(data, list):
                transactions_data = data
            else:
                report.add_error(0, {}, "InvalidFormat", 
                               "JSON must contain either an array of transactions or an object with a 'transactions' array", 
                               is_critical=True)
                report.stats.complete()
                return report
            
            report.stats.total_rows = len(transactions_data)
            
            # Process transactions in batches
            for i in range(0, len(transactions_data), batch_size):
                batch = transactions_data[i:i+batch_size]
                await self._process_transaction_batch(batch, import_mode, report)
                
            report.stats.successful = report.stats.created + report.stats.updated + report.stats.skipped
            report.stats.complete()
            return report
            
        except json.JSONDecodeError as e:
            report.add_error(0, {}, "JSONDecodeError", 
                           f"Invalid JSON format: {str(e)}", 
                           is_critical=True)
        except Exception as e:
            report.add_error(0, {}, "ImportError", 
                           f"Error importing JSON file: {str(e)}", 
                           is_critical=True)
            
        report.stats.complete()
        return report

    async def _import_transactions_csv(self, 
                                     file_path: Path,
                                     import_mode: ImportMode,
                                     batch_size: int) -> ImportReport:
        """
        Import transactions from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            import_mode: Import mode to use
            batch_size: Number of transactions to import in each transaction
            
        Returns:
            ImportReport with statistics and error information
        """
        report = ImportReport(
            import_type=ImportType.TRANSACTIONS,
            file_path=str(file_path),
            file_format=FileFormat.CSV,
            import_mode=import_mode
        )
        
        try:
            # Count total rows for statistics (excluding header)
            with open(file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                report.stats.total_rows = sum(1 for _ in reader)
            
            # Process the CSV file
            with open(file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                
                # Check required columns
                required_columns = ['product_id', 'delta']
                missing_columns = [col for col in required_columns if col not in reader.fieldnames]
                
                if missing_columns:
                    report.add_error(0, {}, "MissingColumns", 
                                   f"CSV is missing required columns: {', '.join(missing_columns)}", 
                                   is_critical=True)
                    report.stats.complete()
                    return report
                
                # Process in batches
                batch = []
                row_num = 1  # Start at 1 because we've already skipped the header
                
                for row in reader:
                    row_num += 1
                    batch.append((row_num, row))
                    
                    if len(batch) >= batch_size:
                        await self._process_transaction_batch([row for _, row in batch], import_mode, report, row_numbers=[num for num, _ in batch])
                        batch = []
                
                # Process any remaining items
                if batch:
                    await self._process_transaction_batch([row for _, row in batch], import_mode, report, row_numbers=[num for num, _ in batch])
                    
            report.stats.successful = report.stats.created + report.stats.updated + report.stats.skipped
            report.stats.complete()
            return report
            
        except Exception as e:
            logger.exception(f"Error in _import_transactions_csv: {e}")
            report.add_error(0, {}, "ImportError", 
                           f"Error importing CSV file: {str(e)}", 
                           is_critical=True)
            
        report.stats.complete()
        return report
    
    async def _process_transaction_batch(self, 
                                       transaction_batch: List[Dict[str, Any]], 
                                       import_mode: ImportMode,
                                       report: ImportReport,
                                       row_numbers: Optional[List[int]] = None) -> None:
        """
        Process a batch of transactions, validating and saving them.
        
        Args:
            transaction_batch: List of transaction data dictionaries
            import_mode: Import mode to use
            report: ImportReport to update with results
            row_numbers: Optional list of row numbers for error reporting
        """
        if row_numbers is None:
            # If not provided, use 1-based index in the batch
            row_numbers = list(range(1, len(transaction_batch) + 1))
        
        # Prepare collections for batch processing
        to_create: List[Tuple[StockTransaction, bool]] = []  # (transaction, update_inventory)
        to_update: List[Tuple[StockTransaction, bool]] = []  # (transaction, update_inventory)
        existing_transactions: Dict[str, StockTransaction] = {}
        affected_products: Dict[str, Product] = {}
        
        # First pass: validate all transactions in the batch
        for i, tx_data in enumerate(transaction_batch):
            row_num = row_numbers[i]
            
            try:
                # Convert types for numeric fields
                try:
                    if 'delta' in tx_data:
                        tx_data['delta'] = int(tx_data['delta'])
                except (ValueError, TypeError) as e:
                    report.add_error(row_num, tx_data, "TypeError", 
                                   f"Type conversion error: {str(e)}")
                    continue
                
                # Handle ID field
                has_id = 'id' in tx_data and tx_data['id']
                
                # Convert quantity_delta to delta if present (for CSV compatibility)
                if 'quantity_delta' in tx_data and 'delta' not in tx_data:
                    tx_data['delta'] = int(tx_data['quantity_delta'])
                
                # Check if product exists
                product_id = tx_data.get('product_id')
                if not product_id:
                    report.add_error(row_num, tx_data, "MissingField", 
                                   "product_id is required")
                    continue
                    
                product = await self.adapter.get_product(str(product_id))
                if not product:
                    report.add_error(row_num, tx_data, "InvalidProduct", 
                                   f"Product with ID {product_id} does not exist")
                    continue
                
                # Remember product for later inventory updates
                affected_products[str(product_id)] = product
                
                # Handle timestamp field
                if 'timestamp' not in tx_data or not tx_data['timestamp']:
                    tx_data['timestamp'] = datetime.now().isoformat()
                elif not isinstance(tx_data['timestamp'], datetime):
                    # Try to parse string timestamp
                    try:
                        tx_data['timestamp'] = datetime.fromisoformat(tx_data['timestamp'])
                    except ValueError:
                        report.add_error(row_num, tx_data, "InvalidTimestamp", 
                                       "timestamp must be in ISO format (YYYY-MM-DDTHH:MM:SS)")
                        continue
                
                # Check if transaction exists
                if has_id:
                    try:
                        tx_id = str(tx_data['id'])
                        # For transactions, we can't easily get a single transaction by ID
                        # so we'll check if it's in our existing transactions from earlier in the batch
                        if tx_id in existing_transactions:
                            existing_transactions[tx_id] = existing_transactions[tx_id]
                    except Exception as e:
                        report.add_error(row_num, tx_data, "LookupError", 
                                       f"Error looking up transaction: {str(e)}")
                        continue
                
                # Determine if inventory should be updated
                update_inventory = tx_data.get('update_inventory', 'true').lower() in ('true', 'yes', '1', '')
                
                # Create Pydantic model for validation
                try:
                    # For new transactions, we always create (transactions aren't typically updated)
                    if not has_id or tx_data['id'] not in existing_transactions:
                        if import_mode == ImportMode.UPDATE:
                            # Skip if update-only mode and transaction doesn't exist
                            report.stats.skipped += 1
                            continue
                            
                        # Assign new ID if not provided
                        if not has_id:
                            tx_data['id'] = str(uuid.uuid4())
                            
                        # Validate with Pydantic
                        transaction = StockTransaction(**tx_data)
                        to_create.append((transaction, update_inventory))
                    else:  # Existing transaction
                        if import_mode == ImportMode.INSERT:
                            # Skip if insert-only mode and transaction exists
                            report.stats.skipped += 1
                            continue
                            
                        # For simplicity, we'll just create a new transaction with the updated data
                        # In a real system, you might want more sophisticated handling here
                        transaction = StockTransaction(**tx_data)
                        to_update.append((transaction, update_inventory))
                        
                except ValidationError as e:
                    # Extract field and error details from ValidationError
                    for error in e.errors():
                        field = '.'.join(map(str, error['loc']))
                        message = error['msg']
                        report.add_error(row_num, tx_data, "ValidationError", 
                                       message, field=field)
                    continue
                    
                except Exception as e:
                    report.add_error(row_num, tx_data, "ProcessingError", 
                                   f"Error processing transaction: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.exception(f"Unexpected error in row {row_num}")
                report.add_error(row_num, tx_data, "UnexpectedError", 
                               f"Unexpected error: {str(e)}")
        
        # Second pass: save valid transactions within a transaction
        if to_create or to_update:
            try:
                async with self.adapter.transaction():
                    # Save new transactions
                    for transaction, update_inventory in to_create:
                        await self.adapter.save_transaction(transaction)
                        
                        # Update product inventory if requested
                        if update_inventory:
                            product = affected_products.get(str(transaction.product_id))
                            if product:
                                product.quantity += transaction.delta
                                await self.adapter.save_product(product)
                        
                        report.stats.created += 1
                    
                    # Update existing transactions (rare)
                    for transaction, update_inventory in to_update:
                        await self.adapter.save_transaction(transaction)
                        report.stats.updated += 1
                        
                logger.info(f"Saved batch: {len(to_create)} created, {len(to_update)} updated")
                
            except Exception as e:
                logger.error(f"Error saving batch: {e}")
                # Mark all transactions in batch as failed
                for i in range(len(transaction_batch)):
                    if i < len(row_numbers):
                        report.add_error(
                            row_numbers[i],
                            transaction_batch[i], 
                            "TransactionError",
                            f"Failed to save in transaction: {str(e)}"
                        )


async def main():
    """Process command line arguments and run import"""
    parser = argparse.ArgumentParser(description='Import data into the inventory system')
    parser.add_argument('file', help='Path to file to import')
    parser.add_argument('--type', '-t', choices=['products', 'transactions'], required=True,
                      help='Type of data to import')
    parser.add_argument('--format', '-f', choices=['auto', 'json', 'csv'], default='auto',
                      help='Format of import file (default: auto-detect from extension)')
    parser.add_argument('--mode', '-m', choices=['insert', 'update', 'upsert'], default='upsert',
                      help='Import mode (insert, update, or upsert)')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                      help='Number of rows to process in each transaction (default: 100)')
    parser.add_argument('--report', '-r', help='Path to save detailed report')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get adapter from configuration
    try:
        adapter = get_adapter(config_path=args.config)
    except Exception as e:
        logger.error(f"Failed to initialize storage adapter: {e}")
        return 1
    
    # Create importer
    importer = DataImporter(adapter)
    
    # Run import
    logger.info(f"Starting import of {args.type} from {args.file}")
    
    report = await importer.import_file(
        file_path=args.file,
        import_type=ImportType(args.type),
        file_format=FileFormat(args.format),
        import_mode=ImportMode(args.mode),
        batch_size=args.batch_size
    )
    
    # Print summary
    report.print_summary()
    
    # Save detailed report if requested
    if args.report:
        report_path = report.save_to_file(args.report)
        print(f"Detailed report saved to: {report_path}")
    
    # Return exit code based on success
    return 0 if report.stats.failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nImport cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)