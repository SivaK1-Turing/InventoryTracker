"""
conftest.py - Test fixtures for list command testing

This module provides common fixtures and utilities for testing the list
command's filtering, sorting, and search capabilities.
"""
import pytest
import uuid
import random
import json
import shutil
from typing import List, Dict, Any, Callable, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
import time
import subprocess
from io import StringIO

from inventorytracker.models.product import Product
from inventorytracker.store import get_store
import inventorytracker.store as store_module


@pytest.fixture(scope="session")
def test_data_dir():
    """Create and return a temporary directory for test data."""
    test_dir = Path("./test_data")
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Clear existing test database
    for item in test_dir.glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    
    yield test_dir
    
    # Cleanup after tests
    # Comment out the next lines to preserve test data for debugging
    # for item in test_dir.glob("*"):
    #     if item.is_file():
    #         item.unlink()
    #     elif item.is_dir():
    #         shutil.rmtree(item)


@pytest.fixture(scope="session")
def test_db_path(test_data_dir):
    """Return a path to the test SQLite database."""
    return test_data_dir / "test.db"


@pytest.fixture(scope="session")
def setup_test_store(test_db_path):
    """Set up a test store and configure environment."""
    # Configure environment for test store
    import os
    original_db_path = os.environ.get("INVENTORY_TRACKER_DB_PATH")
    original_store_type = os.environ.get("INVENTORY_TRACKER_STORE_TYPE")
    
    os.environ["INVENTORY_TRACKER_DB_PATH"] = str(test_db_path)
    os.environ["INVENTORY_TRACKER_STORE_TYPE"] = "sqlite"
    
    # Get the store
    store = get_store()
    
    yield store
    
    # Restore original environment
    if original_db_path:
        os.environ["INVENTORY_TRACKER_DB_PATH"] = original_db_path
    else:
        os.environ.pop("INVENTORY_TRACKER_DB_PATH", None)
        
    if original_store_type:
        os.environ["INVENTORY_TRACKER_STORE_TYPE"] = original_store_type
    else:
        os.environ.pop("INVENTORY_TRACKER_STORE_TYPE", None)


@pytest.fixture(scope="session")
def product_generator():
    """Factory fixture to generate test products."""
    
    def create_product(
        name: Optional[str] = None,
        sku: Optional[str] = None,
        price: Optional[float] = None,
        stock: Optional[int] = None,
        reorder_level: Optional[int] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        archived: bool = False,
    ) -> Product:
        """Generate a test product with the given attributes."""
        # Product name components for auto-generation
        categories = ["Widget", "Gadget", "Tool", "Device", "Component"]
        adjectives = ["Smart", "Pro", "Advanced", "Basic", "Premium"]
        models = ["X", "S", "Pro", "Plus", "Lite"]
        
        # Generate random name if not provided
        if name is None:
            category = random.choice(categories)
            adjective = random.choice(adjectives)
            model = random.choice(models)
            name = f"{adjective} {category} {model}"
            
            # Sometimes add a number
            if random.random() < 0.5:
                name += f" {random.randint(1, 999)}"
        
        # Generate SKU if not provided
        if sku is None:
            if name:
                # Use initials from name
                initials = ''.join(word[0] for word in name.split())
                sku = f"{initials}-{random.randint(1000, 9999)}".upper()
            else:
                sku = f"SKU-{random.randint(1000, 9999)}"
                
        # Generate other attributes if not provided
        if price is None:
            price = round(random.uniform(9.99, 499.99), 2)
            
        if stock is None:
            stock = random.randint(0, 100)
            
        if reorder_level is None:
            reorder_level = random.randint(5, 25)
            
        if tags is None:
            all_tags = ["electronics", "hardware", "tools", "industrial", 
                      "home", "office", "outdoor", "professional"]
            tags = random.sample(all_tags, random.randint(1, 3))
            
        if notes is None:
            notes_options = [
                f"Popular {random.choice(['summer', 'winter', 'spring', 'fall'])} item",
                f"Made in {random.choice(['USA', 'China', 'Germany', 'Japan'])}",
                f"New for {random.randint(2023, 2025)}",
                f"Warranty: {random.choice(['1 year', '2 years', '5 years'])}",
                f"Requires {random.choice(['batteries', 'power adapter'])}"
            ]
            notes = random.choice(notes_options)
            
        # Create timestamps
        now = datetime.now()
        days_ago = random.randint(1, 365)
        created_at = now - timedelta(days=days_ago)
        updated_at = created_at + timedelta(days=random.randint(0, days_ago))
        
        # Set archived info if needed
        archived_at = None
        if archived:
            days_since_creation = (now - created_at).days
            if days_since_creation > 0:
                days_to_archive = random.randint(1, days_since_creation)
                archived_at = created_at + timedelta(days=days_to_archive)
        
        # Create the product
        return Product(
            id=uuid.uuid4(),
            name=name,
            sku=sku,
            price=price,
            current_stock=stock,
            reorder_level=reorder_level,
            tags=tags,
            notes=notes,
            created_at=created_at,
            updated_at=updated_at,
            archived=archived,
            archived_at=archived_at,
        )
    
    return create_product


@pytest.fixture(scope="session")
def test_products(setup_test_store, product_generator):
    """Generate and store a set of test products with known characteristics."""
    store = setup_test_store
    
    # Get existing products
    existing_count = len(list(store.get_all_products()))
    
    # If we already have products, don't recreate them
    if existing_count >= 200:
        return list(store.get_all_products())
    
    products = []
    
    # Create products with specific tags for testing
    tag_combinations = [
        ["electronics"],
        ["hardware"],
        ["tools"],
        ["electronics", "hardware"],
        ["electronics", "tools"],
        ["hardware", "tools"],
        ["electronics", "hardware", "tools"],
        ["professional"],
        ["home"],
        ["office"],
    ]
    
    # Generate a mix of archived and non-archived products
    archived_ratio = 0.1  # 10% archived
    
    # Generate products with diverse characteristics
    for i in range(200):  # 200 products should be enough for testing
        # Select tags
        tags = random.choice(tag_combinations)
        
        # Determine if product should be archived
        archived = random.random() < archived_ratio
        
        # Create product 
        product = product_generator(
            tags=tags,
            archived=archived,
        )
        
        # Ensure some products have distinctive names for search testing
        if i % 20 == 0:
            product.name = f"TestWidget {i//20}"
            
        # Ensure some products have distinctive SKUs for search testing
        if i % 20 == 1:
            product.sku = f"TEST-{i//20:03d}"
            
        # Store the product
        store.save_product(product)
        products.append(product)
        
    return products


@pytest.fixture
def run_list_command():
    """
    Fixture to run the list-products command and capture its output.
    
    Returns a function that can be called with command-line arguments.
    The function returns a tuple of (stdout, stderr, execution_time, return_code).
    """
    def run(args: List[str]) -> Tuple[str, str, float, int]:
        """
        Run the list-products command with the given arguments.
        
        Args:
            args: List of command-line arguments to pass to list-products
            
        Returns:
            Tuple of (stdout, stderr, execution_time, return_code)
        """
        # Construct the command
        cmd = ["python", "-m", "inventorytracker", "list"] + args
        
        # Measure execution time
        start_time = time.time()
        
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        
        # Get output
        stdout, stderr = process.communicate()
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        return stdout, stderr, execution_time, process.returncode
    
    return run


@pytest.fixture
def parse_table_output():
    """
    Fixture to parse table output from list-products command.
    
    Returns a function that parses the output string and returns a list of dictionaries.
    """
    def parse(output: str) -> List[Dict[str, str]]:
        """
        Parse table output into a list of dictionaries.
        
        Args:
            output: String output from list-products command
            
        Returns:
            List of dictionaries, each representing a row with column headers as keys
        """
        lines = output.strip().split("\n")
        
        # Find the header line and separator line
        header_line = None
        separator_line = None
        
        for i, line in enumerate(lines):
            if "Name" in line and "SKU" in line and "Price" in line:
                header_line = i
                separator_line = i + 1
                break
                
        if header_line is None or separator_line is None:
            return []
            
        # Parse header columns
        headers = []
        header = lines[header_line]
        
        # Simple column detection based on spaces
        # This is a basic approach - a more robust parser might be needed for complex outputs
        column_positions = [0]
        current_pos = 0
        
        while current_pos < len(header):
            # Find next double space, which likely indicates column boundary
            next_pos = header.find("  ", current_pos + 1)
            if next_pos == -1:
                break
                
            # Skip consecutive spaces
            while next_pos + 2 < len(header) and header[next_pos + 2] == " ":
                next_pos += 1
                
            column_positions.append(next_pos + 2)
            current_pos = next_pos + 2
            
        # Extract header names
        for i in range(len(column_positions)):
            start = column_positions[i]
            end = column_positions[i + 1] if i + 1 < len(column_positions) else len(header)
            header_name = header[start:end].strip()
            headers.append(header_name)
            
        # Parse data rows
        results = []
        
        for line_index in range(separator_line + 1, len(lines)):
            line = lines[line_index]
            
            # Skip empty lines and summary lines
            if not line.strip() or "Showing " in line:
                continue
                
            # Extract column values
            row = {}
            for i in range(len(column_positions)):
                start = column_positions[i]
                end = column_positions[i + 1] if i + 1 < len(column_positions) else len(line)
                
                # Ensure we don't go out of bounds
                if start >= len(line):
                    value = ""
                else:
                    value = line[start:end].strip()
                
                row[headers[i]] = value
                
            results.append(row)
            
        return results
    
    return parse


@pytest.fixture
def assert_performance():
    """
    Fixture to assert performance thresholds.
    
    Returns a function that can be called to verify execution time is below a threshold.
    """
    def check(execution_time: float, threshold_ms: float, operation_desc: str):
        """
        Assert that execution time is below the threshold.
        
        Args:
            execution_time: Measured execution time in seconds
            threshold_ms: Maximum acceptable time in milliseconds
            operation_desc: Description of the operation for error messages
        """
        execution_ms = execution_time * 1000
        assert execution_ms <= threshold_ms, (
            f"Performance threshold exceeded for {operation_desc}. "
            f"Execution took {execution_ms:.2f} ms, threshold is {threshold_ms:.2f} ms"
        )
        
        # Also print performance data for information
        print(f"Performance for {operation_desc}: {execution_ms:.2f} ms (threshold: {threshold_ms:.2f} ms)")
        
    return check


@contextmanager
def time_operation(operation_name: str = "Operation"):
    """
    Context manager to time an operation and print the result.
    
    Args:
        operation_name: Name of the operation for the output message
        
    Yields:
        Nothing
    """
    start_time = time.time()
    yield
    execution_time = time.time() - start_time
    print(f"{operation_name} took {execution_time * 1000:.2f} ms")