"""
commands/export_data.py - Export data command for InventoryTracker

This module provides functionality to export product and transaction data
from the InventoryTracker system to JSON or CSV formats, utilizing streaming
techniques to handle potentially large datasets efficiently.
"""

import csv
import json
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, TextIO, BinaryIO, AsyncIterator, Literal
import argparse
import sys
import aiofiles
import aiofiles.os
from datetime import datetime

from ..store.adapter import StorageAdapter
from ..models.product import Product
from ..models.inventory import StockTransaction

# Set up logger
logger = logging.getLogger(__name__)

# Chunk size for streaming
CHUNK_SIZE = 1000  # Larger chunk size for better performance

class ExportError(Exception):
    """Exception raised for errors during data export"""
    pass

class ExportFormat:
    """Constants for export formats"""
    JSON = 'json'
    CSV = 'csv'


async def export_products_to_json(
    storage: StorageAdapter, 
    file_path: Union[str, Path], 
    pretty: bool = False
) -> int:
    """
    Export products to a JSON file using streaming to handle large datasets.
    
    Args:
        storage: The storage adapter to read data from
        file_path: Path to the output JSON file
        pretty: Whether to format the JSON with indentation
    
    Returns:
        Number of products exported
    
    Raises:
        ExportError: If export fails
    """
    try:
        path = Path(file_path).resolve()
        count = 0
        
        # Create parent directory if it doesn't exist
        parent_dir = path.parent
        if not parent_dir.exists():
            await aiofiles.os.makedirs(parent_dir, exist_ok=True)
            
        # Check if we have write permission to the target file/directory
        if path.exists() and not os.access(path, os.W_OK):
            raise PermissionError(f"No write permission for {path}")
        if not path.exists() and not os.access(parent_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory {parent_dir}")
        
        # Use list-style JSON array streaming
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            # Start JSON array with metadata
            await f.write('{\n')
            await f.write('  "metadata": {\n')
            await f.write(f'    "exported_at": "{datetime.now().isoformat()}",\n')
            await f.write('    "type": "products"\n')
            await f.write('  },\n')
            await f.write('  "data": [\n')
            
            item_added = False
            try:
                async with storage.stream_products() as products:
                    async for product in products:
                        try:
                            # Add comma for all but the first item
                            if item_added:
                                await f.write(',\n')
                            
                            # Convert product to JSON - handle potential serialization errors
                            try:
                                product_json = product.model_dump_json(indent=2 if pretty else None)
                                await f.write('    ' + product_json if pretty else product_json)
                            except Exception as e:
                                logger.warning(f"Failed to serialize product {getattr(product, 'id', 'unknown')}: {e}")
                                # Write a placeholder with error info instead of failing the entire export
                                error_json = json.dumps({
                                    "id": str(getattr(product, 'id', 'unknown')),
                                    "error": f"Failed to serialize: {str(e)}"
                                }, indent=2 if pretty else None)
                                await f.write('    ' + error_json if pretty else error_json)
                            
                            item_added = True
                            count += 1
                            
                            # Periodically flush to disk and report progress
                            if count % CHUNK_SIZE == 0:
                                await f.flush()
                                logger.info(f"Exported {count} products so far...")
                                
                        except Exception as item_err:
                            # Log the error but continue with next item
                            logger.warning(f"Error processing product: {item_err}")
            except Exception as stream_err:
                # Log streaming error but try to complete the file
                logger.error(f"Error while streaming products: {stream_err}")
                
            # End JSON structure
            await f.write('\n  ]\n}')
            
        logger.info(f"Exported {count} products to {path}")
        return count
        
    except PermissionError as pe:
        logger.error(f"Permission error: {pe}")
        raise ExportError(f"Permission denied: {pe}") from pe
    except Exception as e:
        logger.exception(f"Error exporting products to JSON: {e}")
        raise ExportError(f"Failed to export products to JSON: {e}") from e


async def export_transactions_to_json(
    storage: StorageAdapter, 
    file_path: Union[str, Path],
    pretty: bool = False
) -> int:
    """
    Export transactions to a JSON file using streaming to handle large datasets.
    
    Args:
        storage: The storage adapter to read data from
        file_path: Path to the output JSON file
        pretty: Whether to format the JSON with indentation
    
    Returns:
        Number of transactions exported
    
    Raises:
        ExportError: If export fails
    """
    try:
        path = Path(file_path).resolve()
        count = 0
        
        # Create parent directory if it doesn't exist
        parent_dir = path.parent
        if not parent_dir.exists():
            await aiofiles.os.makedirs(parent_dir, exist_ok=True)
            
        # Check if we have write permission to the target file/directory
        if path.exists() and not os.access(path, os.W_OK):
            raise PermissionError(f"No write permission for {path}")
        if not path.exists() and not os.access(parent_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory {parent_dir}")
        
        # Use list-style JSON array streaming
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            # Start JSON array with metadata
            await f.write('{\n')
            await f.write('  "metadata": {\n')
            await f.write(f'    "exported_at": "{datetime.now().isoformat()}",\n')
            await f.write('    "type": "transactions"\n')
            await f.write('  },\n')
            await f.write('  "data": [\n')
            
            item_added = False
            try:
                async with storage.stream_transactions() as transactions:
                    async for transaction in transactions:
                        try:
                            # Add comma for all but the first item
                            if item_added:
                                await f.write(',\n')
                            
                            # Convert transaction to JSON - handle potential serialization errors
                            try:
                                transaction_json = transaction.model_dump_json(indent=2 if pretty else None)
                                await f.write('    ' + transaction_json if pretty else transaction_json)
                            except Exception as e:
                                logger.warning(f"Failed to serialize transaction {getattr(transaction, 'id', 'unknown')}: {e}")
                                # Write a placeholder with error info instead of failing the entire export
                                error_json = json.dumps({
                                    "id": str(getattr(transaction, 'id', 'unknown')),
                                    "error": f"Failed to serialize: {str(e)}"
                                }, indent=2 if pretty else None)
                                await f.write('    ' + error_json if pretty else error_json)
                            
                            item_added = True
                            count += 1
                            
                            # Periodically flush to disk and report progress
                            if count % CHUNK_SIZE == 0:
                                await f.flush()
                                logger.info(f"Exported {count} transactions so far...")
                                
                        except Exception as item_err:
                            # Log the error but continue with next item
                            logger.warning(f"Error processing transaction: {item_err}")
            except Exception as stream_err:
                # Log streaming error but try to complete the file
                logger.error(f"Error while streaming transactions: {stream_err}")
                
            # End JSON structure
            await f.write('\n  ]\n}')
            
        logger.info(f"Exported {count} transactions to {path}")
        return count
        
    except PermissionError as pe:
        logger.error(f"Permission error: {pe}")
        raise ExportError(f"Permission denied: {pe}") from pe
    except Exception as e:
        logger.exception(f"Error exporting transactions to JSON: {e}")
        raise ExportError(f"Failed to export transactions to JSON: {e}") from e


async def export_products_to_csv(
    storage: StorageAdapter,
    file_path: Union[str, Path]
) -> int:
    """
    Export products to a CSV file using streaming to handle large datasets.
    
    Args:
        storage: The storage adapter to read data from
        file_path: Path to the output CSV file
    
    Returns:
        Number of products exported
    
    Raises:
        ExportError: If export fails
    """
    try:
        path = Path(file_path).resolve()
        count = 0
        
        # Create parent directory if it doesn't exist
        parent_dir = path.parent
        if not parent_dir.exists():
            await aiofiles.os.makedirs(parent_dir, exist_ok=True)
            
        # Check if we have write permission to the target file/directory
        if path.exists() and not os.access(path, os.W_OK):
            raise PermissionError(f"No write permission for {path}")
        if not path.exists() and not os.access(parent_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory {parent_dir}")
        
        # CSV field names for products (should match Product model attributes)
        fieldnames = ['id', 'name', 'sku', 'description', 'category', 'price', 
                     'reorder_point', 'reorder_quantity', 'created_at', 'updated_at']
        
        # Create file and write header
        async with aiofiles.open(path, 'w', encoding='utf-8', newline='') as f:
            # We need to use sync CSV writer with a wrapper because aiocsv is not widely used
            # and may not handle all edge cases and escaping properly
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write the header manually as a string
            header_row = ','.join(fieldnames) + '\n'
            await f.write(header_row)
            
            # Stream products
            try:
                async with storage.stream_products() as products:
                    async for product in products:
                        try:
                            # Convert product to dict for CSV
                            try:
                                # Convert the model to a dict, handling dates and other special types
                                product_dict = product.model_dump()
                                
                                # Convert complex types to strings for CSV compatibility
                                for key, value in product_dict.items():
                                    if isinstance(value, (datetime, dict, list)):
                                        product_dict[key] = str(value)
                                
                                # Create CSV row as a string with proper escaping
                                csv_row = []
                                for field in fieldnames:
                                    value = str(product_dict.get(field, ""))
                                    # Escape quotes and wrap in quotes if needed
                                    if '"' in value or ',' in value or '\n' in value:
                                        value = '"' + value.replace('"', '""') + '"'
                                    csv_row.append(value)
                                
                                # Write the row
                                await f.write(','.join(csv_row) + '\n')
                                
                            except Exception as e:
                                logger.warning(f"Failed to serialize product {getattr(product, 'id', 'unknown')}: {e}")
                                # Write a row with error information
                                error_row = [str(getattr(product, 'id', 'unknown'))]
                                error_row.extend(['ERROR'] * (len(fieldnames) - 1))
                                await f.write(','.join(error_row) + '\n')
                            
                            count += 1
                            
                            # Periodically flush to disk and report progress
                            if count % CHUNK_SIZE == 0:
                                await f.flush()
                                logger.info(f"Exported {count} products so far...")
                                
                        except Exception as item_err:
                            logger.warning(f"Error processing product: {item_err}")
            except Exception as stream_err:
                logger.error(f"Error while streaming products: {stream_err}")
            
        logger.info(f"Exported {count} products to {path}")
        return count
        
    except PermissionError as pe:
        logger.error(f"Permission error: {pe}")
        raise ExportError(f"Permission denied: {pe}") from pe
    except Exception as e:
        logger.exception(f"Error exporting products to CSV: {e}")
        raise ExportError(f"Failed to export products to CSV: {e}") from e


async def export_transactions_to_csv(
    storage: StorageAdapter,
    file_path: Union[str, Path]
) -> int:
    """
    Export transactions to a CSV file using streaming to handle large datasets.
    
    Args:
        storage: The storage adapter to read data from
        file_path: Path to the output CSV file
    
    Returns:
        Number of transactions exported
    
    Raises:
        ExportError: If export fails
    """
    try:
        path = Path(file_path).resolve()
        count = 0
        
        # Create parent directory if it doesn't exist
        parent_dir = path.parent
        if not parent_dir.exists():
            await aiofiles.os.makedirs(parent_dir, exist_ok=True)
            
        # Check if we have write permission to the target file/directory
        if path.exists() and not os.access(path, os.W_OK):
            raise PermissionError(f"No write permission for {path}")
        if not path.exists() and not os.access(parent_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory {parent_dir}")
        
        # CSV field names for transactions (should match StockTransaction model attributes)
        fieldnames = ['id', 'product_id', 'quantity', 'transaction_type', 
                     'timestamp', 'reference', 'notes']
        
        # Create file and write header
        async with aiofiles.open(path, 'w', encoding='utf-8', newline='') as f:
            # Write the header manually as a string
            header_row = ','.join(fieldnames) + '\n'
            await f.write(header_row)
            
            # Stream transactions
            try:
                async with storage.stream_transactions() as transactions:
                    async for transaction in transactions:
                        try:
                            # Convert transaction to dict for CSV
                            try:
                                # Convert the model to a dict, handling dates and other special types
                                transaction_dict = transaction.model_dump()
                                
                                # Convert complex types to strings for CSV compatibility
                                for key, value in transaction_dict.items():
                                    if isinstance(value, (datetime, dict, list)):
                                        transaction_dict[key] = str(value)
                                
                                # Create CSV row as a string with proper escaping
                                csv_row = []
                                for field in fieldnames:
                                    value = str(transaction_dict.get(field, ""))
                                    # Escape quotes and wrap in quotes if needed
                                    if '"' in value or ',' in value or '\n' in value:
                                        value = '"' + value.replace('"', '""') + '"'
                                    csv_row.append(value)
                                
                                # Write the row
                                await f.write(','.join(csv_row) + '\n')
                                
                            except Exception as e:
                                logger.warning(f"Failed to serialize transaction {getattr(transaction, 'id', 'unknown')}: {e}")
                                # Write a row with error information
                                error_row = [str(getattr(transaction, 'id', 'unknown'))]
                                error_row.extend(['ERROR'] * (len(fieldnames) - 1))
                                await f.write(','.join(error_row) + '\n')
                            
                            count += 1
                            
                            # Periodically flush to disk and report progress
                            if count % CHUNK_SIZE == 0:
                                await f.flush()
                                logger.info(f"Exported {count} transactions so far...")
                                
                        except Exception as item_err:
                            logger.warning(f"Error processing transaction: {item_err}")
            except Exception as stream_err:
                logger.error(f"Error while streaming transactions: {stream_err}")
            
        logger.info(f"Exported {count} transactions to {path}")
        return count
        
    except PermissionError as pe:
        logger.error(f"Permission error: {pe}")
        raise ExportError(f"Permission denied: {pe}") from pe
    except Exception as e:
        logger.exception(f"Error exporting transactions to CSV: {e}")
        raise ExportError(f"Failed to export transactions to CSV: {e}") from e


async def export_data(
    storage: StorageAdapter,
    data_type: Literal['products', 'transactions', 'all'],
    output_format: Literal['json', 'csv'],
    output_dir: Union[str, Path],
    pretty: bool = False
) -> Dict[str, int]:
    """
    Export data from the inventory system to the specified format.
    
    Args:
        storage: Storage adapter to retrieve data from
        data_type: Type of data to export ('products', 'transactions', or 'all')
        output_format: Format to export data in ('json' or 'csv')
        output_dir: Directory to save exported files
        pretty: Whether to format JSON with indentation (ignored for CSV)
        
    Returns:
        Dictionary with count of exported items by type
    
    Raises:
        ExportError: If export fails
    """
    output_path = Path(output_dir).resolve()
    if not output_path.exists():
        await aiofiles.os.makedirs(output_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {}
    
    try:
        # Export products
        if data_type in ('products', 'all'):
            product_file = output_path / f"products_{timestamp}.{output_format}"
            
            if output_format == ExportFormat.JSON:
                count = await export_products_to_json(storage, product_file, pretty)
            else:  # CSV
                count = await export_products_to_csv(storage, product_file)
                
            result['products'] = count
            
        # Export transactions
        if data_type in ('transactions', 'all'):
            transaction_file = output_path / f"transactions_{timestamp}.{output_format}"
            
            if output_format == ExportFormat.JSON:
                count = await export_transactions_to_json(storage, transaction_file, pretty)
            else:  # CSV
                count = await export_transactions_to_csv(storage, transaction_file)
                
            result['transactions'] = count
            
        return result
        
    except Exception as e:
        logger.exception(f"Error during data export: {e}")
        raise ExportError(f"Export failed: {e}") from e


def setup_parser(subparsers):
    """
    Set up the command line parser for the export command.
    
    Args:
        subparsers: Subparsers object from the main parser
    """
    export_parser = subparsers.add_parser(
        'export',
        help='Export inventory data to JSON or CSV format'
    )
    
    export_parser.add_argument(
        '--type',
        choices=['products', 'transactions', 'all'],
        default='all',
        help='Type of data to export (default: all)'
    )
    
    export_parser.add_argument(
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='Output format (default: json)'
    )
    
    export_parser.add_argument(
        '--output-dir',
        type=str,
        default='./exports',
        help='Directory to save exported files (default: ./exports)'
    )
    
    export_parser.add_argument(
        '--pretty',
        action='store_true',
        help='Format JSON with indentation (ignored for CSV)'
    )
    
    export_parser.set_defaults(func=handle_export)


async def handle_export(args, storage: StorageAdapter):
    """
    Handle the export command.
    
    Args:
        args: Parsed command line arguments
        storage: Storage adapter instance
    """
    try:
        logger.info(f"Starting export: type={args.type}, format={args.format}, dir={args.output_dir}")
        
        result = await export_data(
            storage=storage,
            data_type=args.type,
            output_format=args.format,
            output_dir=args.output_dir,
            pretty=args.pretty
        )
        
        # Print summary of export
        print("\nExport completed successfully:")
        for data_type, count in result.items():
            print(f"  - {data_type.capitalize()}: {count} records")
        print(f"Output directory: {args.output_dir}")
        
    except ExportError as e:
        logger.error(f"Export failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during export: {e}")
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")