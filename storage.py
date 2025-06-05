"""
benchmarks/storage.py - Storage adapter performance benchmarks

This module provides benchmarking tools to compare performance of different
storage adapters for InventoryTracker, focusing on write and read speeds
for large datasets (10,000 products and 50,000 transactions).
"""

import asyncio
import time
import uuid
import random
import string
import logging
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Callable, AsyncGenerator
import argparse
import json
import csv
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inventory_tracker.store.adapter import StorageAdapter
from inventory_tracker.store.sql_adapter import SQLAdapter
from inventory_tracker.store.file_adapter import FileAdapter

from inventory_tracker.models.product import Product
from inventory_tracker.models.inventory import StockTransaction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmarks")

# Constants for benchmark dataset sizes
PRODUCT_COUNT = 10000
TRANSACTION_COUNT = 50000

class BenchmarkResults:
    """Class to store and report benchmark results"""
    
    def __init__(self, name: str):
        self.name = name
        self.operations: Dict[str, Dict[str, float]] = {}
    
    def record_operation(self, operation: str, dataset: str, duration: float):
        """Record the duration of an operation on a dataset"""
        if operation not in self.operations:
            self.operations[operation] = {}
        
        self.operations[operation][dataset] = duration
    
    def get_operation_result(self, operation: str, dataset: str) -> Optional[float]:
        """Get the duration of a specific operation on a dataset"""
        return self.operations.get(operation, {}).get(dataset)
    
    def print_results(self):
        """Print benchmark results to console"""
        print(f"\nBenchmark Results for {self.name}")
        print("=" * 60)
        
        for operation, datasets in self.operations.items():
            print(f"\n{operation}:")
            print("-" * 30)
            
            for dataset, duration in datasets.items():
                operations_per_sec = None
                if "product" in dataset.lower():
                    count = PRODUCT_COUNT
                elif "transaction" in dataset.lower():
                    count = TRANSACTION_COUNT
                else:
                    count = None
                
                if count:
                    operations_per_sec = count / duration if duration > 0 else 0
                    print(f"  {dataset}: {duration:.4f} seconds ({operations_per_sec:.2f} ops/sec)")
                else:
                    print(f"  {dataset}: {duration:.4f} seconds")
    
    def to_markdown(self) -> str:
        """Convert results to markdown format for documentation"""
        lines = [f"# Benchmark Results for {self.name}", ""]
        
        # Create a table for each operation
        for operation, datasets in self.operations.items():
            lines.append(f"## {operation}")
            lines.append("")
            lines.append("| Dataset | Duration (seconds) | Operations/second |")
            lines.append("|---------|-------------------|-------------------|")
            
            for dataset, duration in datasets.items():
                if "product" in dataset.lower():
                    count = PRODUCT_COUNT
                elif "transaction" in dataset.lower():
                    count = TRANSACTION_COUNT
                else:
                    count = "-"
                    
                if count != "-":
                    operations_per_sec = count / duration if duration > 0 else 0
                    lines.append(f"| {dataset} | {duration:.4f} | {operations_per_sec:.2f} |")
                else:
                    lines.append(f"| {dataset} | {duration:.4f} | - |")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def save_markdown(self, filename: str):
        """Save results to a markdown file"""
        markdown = self.to_markdown()
        with open(filename, 'w') as f:
            f.write(markdown)
        
        logger.info(f"Benchmark results saved to {filename}")

class TestDataGenerator:
    """Generator for test data used in benchmarks"""
    
    @staticmethod
    def generate_products(count: int) -> List[Product]:
        """Generate a list of test products"""
        products = []
        categories = ["Electronics", "Clothing", "Home", "Food", "Sports", "Books", "Toys"]
        
        for i in range(count):
            name = f"Test Product {i+1}"
            sku = f"TP-{i:06d}"
            category = random.choice(categories)
            price = round(random.uniform(1.0, 1000.0), 2)
            
            product = Product(
                id=uuid.uuid4(),
                name=name,
                sku=sku,
                description=f"Description for {name}",
                category=category,
                price=price,
                reorder_point=random.randint(5, 50),
                reorder_quantity=random.randint(10, 100),
                created_at=datetime.now() - timedelta(days=random.randint(0, 365)),
                updated_at=datetime.now()
            )
            
            products.append(product)
        
        return products
    
    @staticmethod
    def generate_transactions(count: int, products: List[Product]) -> List[StockTransaction]:
        """Generate a list of test transactions using the given products"""
        transactions = []
        transaction_types = ["purchase", "sale", "adjustment", "return", "transfer"]
        
        for i in range(count):
            product = random.choice(products)
            tx_type = random.choice(transaction_types)
            
            # Generate a reasonable quantity based on transaction type
            if tx_type == "purchase":
                quantity = random.randint(1, 100)
            elif tx_type == "sale":
                quantity = -random.randint(1, 10)
            elif tx_type == "adjustment":
                quantity = random.randint(-20, 20)
            elif tx_type == "return":
                quantity = random.randint(1, 5)
            else:  # transfer
                quantity = -random.randint(1, 50) if random.random() < 0.5 else random.randint(1, 50)
            
            # Create reference based on type
            if tx_type == "purchase":
                reference = f"PO-{random.randint(1000, 9999)}"
            elif tx_type == "sale":
                reference = f"SO-{random.randint(1000, 9999)}"
            else:
                reference = f"REF-{i:06d}"
                
            transaction = StockTransaction(
                id=uuid.uuid4(),
                product_id=product.id,
                quantity=quantity,
                transaction_type=tx_type,
                timestamp=datetime.now() - timedelta(days=random.randint(0, 180)),
                reference=reference,
                notes=f"Test transaction {i+1}"
            )
            
            transactions.append(transaction)
        
        return transactions

async def benchmark_adapter(adapter_factory: Callable[[], StorageAdapter], name: str) -> BenchmarkResults:
    """
    Run benchmarks on a storage adapter.
    
    Args:
        adapter_factory: Factory function to create the adapter
        name: Name of the adapter for reporting
        
    Returns:
        BenchmarkResults object with benchmark results
    """
    logger.info(f"Starting benchmark for {name}")
    results = BenchmarkResults(name)
    
    # Create and connect adapter
    adapter = adapter_factory()
    await adapter.connect()
    
    try:
        # Clear any existing data
        await adapter.clear_all()
        
        # Generate test data
        logger.info(f"Generating {PRODUCT_COUNT} test products...")
        products = TestDataGenerator.generate_products(PRODUCT_COUNT)
        
        logger.info(f"Generating {TRANSACTION_COUNT} test transactions...")
        transactions = TestDataGenerator.generate_transactions(TRANSACTION_COUNT, products)
        
        # Benchmark 1: Bulk product insertion
        logger.info(f"Benchmarking bulk product insertion...")
        start_time = time.time()
        
        for product in products:
            await adapter.save_product(product)
        
        duration = time.time() - start_time
        results.record_operation("Bulk Insert", "Products", duration)
        logger.info(f"Inserted {PRODUCT_COUNT} products in {duration:.4f} seconds ({PRODUCT_COUNT/duration:.2f} products/sec)")
        
        # Benchmark 2: Bulk transaction insertion
        logger.info(f"Benchmarking bulk transaction insertion...")
        start_time = time.time()
        
        for transaction in transactions:
            await adapter.save_transaction(transaction)
        
        duration = time.time() - start_time
        results.record_operation("Bulk Insert", "Transactions", duration)
        logger.info(f"Inserted {TRANSACTION_COUNT} transactions in {duration:.4f} seconds ({TRANSACTION_COUNT/duration:.2f} transactions/sec)")
        
        # Benchmark 3: Product retrieval by ID
        logger.info(f"Benchmarking product retrieval by ID...")
        sample_size = min(1000, PRODUCT_COUNT)
        product_samples = random.sample(products, sample_size)
        
        start_time = time.time()
        
        for product in product_samples:
            loaded_product = await adapter.get_product(product.id)
            assert loaded_product is not None, f"Failed to retrieve product {product.id}"
        
        duration = time.time() - start_time
        results.record_operation("Retrieval by ID", "Products", duration)
        logger.info(f"Retrieved {sample_size} products by ID in {duration:.4f} seconds ({sample_size/duration:.2f} retrievals/sec)")
        
        # Benchmark 4: Transaction retrieval by ID
        logger.info(f"Benchmarking transaction retrieval by ID...")
        sample_size = min(2000, TRANSACTION_COUNT)
        transaction_samples = random.sample(transactions, sample_size)
        
        start_time = time.time()
        
        for transaction in transaction_samples:
            loaded_transaction = await adapter.get_transaction(transaction.id)
            assert loaded_transaction is not None, f"Failed to retrieve transaction {transaction.id}"
        
        duration = time.time() - start_time
        results.record_operation("Retrieval by ID", "Transactions", duration)
        logger.info(f"Retrieved {sample_size} transactions by ID in {duration:.4f} seconds ({sample_size/duration:.2f} retrievals/sec)")
        
        # Benchmark 5: Streaming all products
        logger.info(f"Benchmarking product streaming...")
        start_time = time.time()
        
        count = 0
        async with adapter.stream_products() as stream:
            async for _ in stream:
                count += 1
        
        duration = time.time() - start_time
        results.record_operation("Full Streaming", "Products", duration)
        logger.info(f"Streamed {count} products in {duration:.4f} seconds ({count/duration:.2f} products/sec)")
        
        # Benchmark 6: Streaming all transactions
        logger.info(f"Benchmarking transaction streaming...")
        start_time = time.time()
        
        count = 0
        async with adapter.stream_transactions() as stream:
            async for _ in stream:
                count += 1
        
        duration = time.time() - start_time
        results.record_operation("Full Streaming", "Transactions", duration)
        logger.info(f"Streamed {count} transactions in {duration:.4f} seconds ({count/duration:.2f} transactions/sec)")
        
        # Benchmark 7: Product updates
        logger.info(f"Benchmarking product updates...")
        sample_size = min(1000, PRODUCT_COUNT)
        product_samples = random.sample(products, sample_size)
        
        start_time = time.time()
        
        for product in product_samples:
            updated_product = product.copy()
            updated_product.price = round(updated_product.price * 1.1, 2)  # 10% price increase
            updated_product.updated_at = datetime.now()
            await adapter.save_product(updated_product)
        
        duration = time.time() - start_time
        results.record_operation("Updates", "Products", duration)
        logger.info(f"Updated {sample_size} products in {duration:.4f} seconds ({sample_size/duration:.2f} updates/sec)")
        
        # Benchmark 8: Transaction with rollback stress test
        logger.info(f"Benchmarking transaction with rollback...")
        sample_size = 100
        product_samples = random.sample(products, sample_size)
        success_count = 0
        rollback_count = 0
        
        start_time = time.time()
        
        for i, product in enumerate(product_samples):
            try:
                async with adapter.transaction():
                    # Update product
                    updated_product = product.copy()
                    updated_product.price = round(updated_product.price * 1.2, 2)  # 20% price increase
                    updated_product.updated_at = datetime.now()
                    await adapter.save_product(updated_product)
                    
                    # Force an error on every other product to test rollback
                    if i % 2 == 0:
                        raise ValueError("Simulated error to trigger rollback")
                    
                    success_count += 1
            except ValueError:
                rollback_count += 1
                
                # Verify rollback worked by checking price wasn't updated
                reloaded_product = await adapter.get_product(product.id)
                assert reloaded_product.price != round(product.price * 1.2, 2), "Rollback failed, price was updated"
        
        duration = time.time() - start_time
        results.record_operation("Transaction with Rollback", "Mixed", duration)
        logger.info(f"Processed {sample_size} transactions (committed {success_count}, rolled back {rollback_count}) in {duration:.4f} seconds")
        
        return results
        
    finally:
        # Clean up
        logger.info(f"Cleaning up adapter {name}")
        try:
            await adapter.clear_all()
        except:
            logger.warning(f"Failed to clear all data from {name}")
            
        await adapter.disconnect()

async def run_benchmarks():
    """Run benchmarks on all adapter types and save results"""
    results = []
    
    # Create temp directory for file adapter
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Benchmark SQLite adapter
        sqlite_results = await benchmark_adapter(
            lambda: SQLAdapter(connection_string="sqlite:///:memory:"),
            "SQLite In-Memory Adapter"
        )
        results.append(sqlite_results)
        
        # Benchmark SQLite file adapter
        sqlite_file_path = os.path.join(temp_dir, "benchmark.db")
        sqlite_file_results = await benchmark_adapter(
            lambda: SQLAdapter(connection_string=f"sqlite:///{sqlite_file_path}"),
            "SQLite File Adapter"
        )
        results.append(sqlite_file_results)
        
        # Benchmark File adapter
        file_adapter_dir = os.path.join(temp_dir, "file_adapter")
        os.makedirs(file_adapter_dir, exist_ok=True)
        file_results = await benchmark_adapter(
            lambda: FileAdapter(data_dir=Path(file_adapter_dir)),
            "JSON File Adapter"
        )
        results.append(file_results)
        
        # Print and save results
        for result in results:
            result.print_results()
        
        # Create combined markdown file
        combined_markdown = "# InventoryTracker Storage Adapter Benchmarks\n\n"
        combined_markdown += f"Benchmark run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        combined_markdown += f"Test dataset: {PRODUCT_COUNT} products, {TRANSACTION_COUNT} transactions\n\n"
        combined_markdown += "## System Information\n\n"
        # Add system info if available
        try:
            import platform
            combined_markdown += f"- OS: {platform.system()} {platform.release()}\n"
            combined_markdown += f"- Python: {platform.python_version()}\n"
        except ImportError:
            combined_markdown += "- System information not available\n"
        combined_markdown += "\n"
            
        # Add each result
        for result in results:
            combined_markdown += result.to_markdown() + "\n\n"
            
        # Add summary and comparison
        combined_markdown += "## Summary and Comparison\n\n"
        combined_markdown += "### Product Insertion Speed (products/second)\n\n"
        combined_markdown += "| Adapter | Speed |\n"
        combined_markdown += "|---------|-------|\n"
        
        for result in results:
            duration = result.get_operation_result("Bulk Insert", "Products")
            if duration:
                speed = PRODUCT_COUNT / duration
                combined_markdown += f"| {result.name} | {speed:.2f} |\n"
        
        combined_markdown += "\n### Transaction Insertion Speed (transactions/second)\n\n"
        combined_markdown += "| Adapter | Speed |\n"
        combined_markdown += "|---------|-------|\n"
        
        for result in results:
            duration = result.get_operation_result("Bulk Insert", "Transactions")
            if duration:
                speed = TRANSACTION_COUNT / duration
                combined_markdown += f"| {result.name} | {speed:.2f} |\n"
        
        # Write final output
        os.makedirs("docs", exist_ok=True)
        with open("docs/perf.md", "w") as f:
            f.write(combined_markdown)
            
        print("\nBenchmark results saved to docs/perf.md")
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def main():
    """Main function to run benchmarks from command line"""
    parser = argparse.ArgumentParser(description="Run storage adapter benchmarks")
    parser.add_argument("--product-count", type=int, default=PRODUCT_COUNT, help="Number of products to test")
    parser.add_argument("--transaction-count", type=int, default=TRANSACTION_COUNT, help="Number of transactions to test")
    
    args = parser.parse_args()
    
    # Update global constants if provided in args
    global PRODUCT_COUNT, TRANSACTION_COUNT
    PRODUCT_COUNT = args.product_count
    TRANSACTION_COUNT = args.transaction_count
    
    print(f"Running benchmarks with {PRODUCT_COUNT} products and {TRANSACTION_COUNT} transactions")
    
    asyncio.run(run_benchmarks())

if __name__ == "__main__":
    main()