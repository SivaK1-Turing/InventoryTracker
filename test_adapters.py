"""
tests/feature6/test_adapters.py - Adapter implementation tests

This module provides parametrized tests for all storage adapters, verifying:
1. Save/load data fidelity
2. Transaction support with proper rollback behavior
"""

import pytest
import tempfile
import os
import uuid
from datetime import datetime
from pathlib import Path
import shutil
import asyncio
from typing import Dict, Any, List, Optional

from inventory_tracker.store.adapter import StorageAdapter
from inventory_tracker.store.sql_adapter import SQLAdapter
from inventory_tracker.store.file_adapter import FileAdapter
# Import any other adapters here
# from inventory_tracker.store.memory_adapter import MemoryAdapter

from inventory_tracker.models.product import Product
from inventory_tracker.models.inventory import StockTransaction
from inventory_tracker.models.exceptions import NotFoundError, DatabaseError

# Mark all tests as asyncio tests
pytestmark = pytest.mark.asyncio

# Test data generator function
def create_test_data():
    """Create fresh test data for each test."""
    products = [
        Product(
            id=uuid.uuid4(),
            name="Test Product 1",
            sku="TP001",
            description="A test product",
            category="Test Category",
            price=10.99,
            reorder_point=5,
            reorder_quantity=10,
            created_at=datetime.now(),
            updated_at=datetime.now()
        ),
        Product(
            id=uuid.uuid4(),
            name="Test Product 2",
            sku="TP002",
            description="Another test product",
            category="Test Category",
            price=20.99,
            reorder_point=10,
            reorder_quantity=20,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    ]
    
    transactions = [
        StockTransaction(
            id=uuid.uuid4(),
            product_id=products[0].id,
            quantity=5,
            transaction_type="purchase",
            timestamp=datetime.now(),
            reference="PO12345",
            notes="Initial stock"
        ),
        StockTransaction(
            id=uuid.uuid4(),
            product_id=products[1].id,
            quantity=-2,
            transaction_type="sale",
            timestamp=datetime.now(),
            reference="SO12345",
            notes="Customer order"
        )
    ]
    
    return products, transactions

# Adapter-specific fixtures
@pytest.fixture
async def sql_adapter():
    """Create a SQL adapter with SQLite in-memory database."""
    adapter = SQLAdapter(connection_string="sqlite:///:memory:")
    await adapter.connect()
    await adapter.clear_all()
    
    yield adapter
    
    # Cleanup
    await adapter.disconnect()

@pytest.fixture
async def file_adapter():
    """Create a file adapter with a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    adapter = FileAdapter(data_dir=Path(temp_dir))
    await adapter.connect()
    await adapter.clear_all()
    
    yield adapter
    
    # Cleanup
    await adapter.disconnect()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

# Parametrize adapter fixture to test all adapter types
@pytest.fixture(params=['sql_adapter', 'file_adapter'])
async def adapter(request):
    """Parametrized fixture that provides each type of adapter."""
    adapter_fixture = request.getfixturevalue(request.param)
    return adapter_fixture

# Test Functions
async def test_product_save_load_fidelity(adapter):
    """Test that products can be saved and loaded with perfect fidelity."""
    products, _ = create_test_data()
    
    # Save all sample products
    for product in products:
        await adapter.save_product(product)
    
    # Retrieve each product and verify fidelity
    for original_product in products:
        loaded_product = await adapter.get_product(original_product.id)
        
        assert loaded_product is not None
        assert loaded_product.id == original_product.id
        assert loaded_product.name == original_product.name
        assert loaded_product.sku == original_product.sku
        assert loaded_product.description == original_product.description
        assert loaded_product.category == original_product.category
        assert loaded_product.price == original_product.price
        assert loaded_product.reorder_point == original_product.reorder_point
        assert loaded_product.reorder_quantity == original_product.reorder_quantity
        # Date comparison may need just the date portion depending on how adapters store dates
        assert loaded_product.created_at.date() == original_product.created_at.date()
        assert loaded_product.updated_at.date() == original_product.updated_at.date()

async def test_transaction_save_load_fidelity(adapter):
    """Test that transactions can be saved and loaded with perfect fidelity."""
    products, transactions = create_test_data()
    
    # First, save the products referenced by transactions
    for product in products:
        await adapter.save_product(product)
    
    # Save all sample transactions
    for transaction in transactions:
        await adapter.save_transaction(transaction)
    
    # Retrieve each transaction and verify fidelity
    for original_tx in transactions:
        loaded_tx = await adapter.get_transaction(original_tx.id)
        
        assert loaded_tx is not None
        assert loaded_tx.id == original_tx.id
        assert loaded_tx.product_id == original_tx.product_id
        assert loaded_tx.quantity == original_tx.quantity
        assert loaded_tx.transaction_type == original_tx.transaction_type
        assert loaded_tx.reference == original_tx.reference
        assert loaded_tx.notes == original_tx.notes
        # Date comparison may need just the date portion depending on how adapters store dates
        assert loaded_tx.timestamp.date() == original_tx.timestamp.date()

async def test_transaction_rollback_on_error(adapter):
    """Test that transactions are properly rolled back when errors occur."""
    products, _ = create_test_data()
    
    # Save a valid product
    await adapter.save_product(products[0])
    
    # Get the initial price for comparison
    initial_price = products[0].price
    
    # Start a transaction that will fail
    try:
        async with adapter.transaction():
            # Update the product (valid operation)
            product = products[0].copy()
            product.price = 15.99
            await adapter.save_product(product)
            
            # Verify the price changed within the transaction
            current_product = await adapter.get_product(products[0].id)
            assert current_product.price == 15.99
            
            # Now trigger an error by trying to get a non-existent product
            nonexistent_id = uuid.uuid4()
            await adapter.get_product(nonexistent_id, strict=True)  # Should raise NotFoundError
            
            pytest.fail("Expected an exception but none was raised")
            
    except NotFoundError:
        # Exception is expected, transaction should be rolled back
        pass
    
    # Verify the product price was rolled back
    reloaded_product = await adapter.get_product(products[0].id)
    assert reloaded_product.price == initial_price, "Transaction was not rolled back properly"

async def test_successful_transaction_commit(adapter):
    """Test that transactions successfully commit when no errors occur."""
    products, _ = create_test_data()
    
    # Save a valid product
    await adapter.save_product(products[0])
    
    # Start a transaction that should succeed
    async with adapter.transaction():
        # Update the product
        product = products[0].copy()
        product.price = 15.99
        await adapter.save_product(product)
    
    # Verify the product price was committed
    reloaded_product = await adapter.get_product(products[0].id)
    assert reloaded_product.price == 15.99, "Transaction changes were not committed"

async def test_nested_transactions(adapter):
    """Test nested transaction behavior."""
    products, _ = create_test_data()
    
    # Save a valid product
    await adapter.save_product(products[0])
    
    # Start an outer transaction
    async with adapter.transaction():
        # Update the product
        product = products[0].copy()
        product.price = 15.99
        await adapter.save_product(product)
        
        try:
            # Start a nested transaction that will fail
            async with adapter.transaction():
                # Update the product again
                product.price = 25.99
                await adapter.save_product(product)
                
                # Force an error
                raise ValueError("Simulating an error in nested transaction")
        except ValueError:
            # Expected error, nested transaction should roll back
            pass
        
        # Outer transaction should still be active
        # Price should be reset to the value from the outer transaction
        current_product = await adapter.get_product(products[0].id)
        assert current_product.price == 15.99
    
    # After outer transaction commits, changes should be saved
    final_product = await adapter.get_product(products[0].id)
    assert final_product.price == 15.99

async def test_list_products(adapter):
    """Test that we can list all products."""
    products, _ = create_test_data()
    
    # Save all sample products
    for product in products:
        await adapter.save_product(product)
    
    # Get all products
    all_products = await adapter.list_products()
    
    # Verify the count and contents
    assert len(all_products) == len(products)
    
    # Check that each original product exists in the list
    product_ids = {p.id for p in all_products}
    for original_product in products:
        assert original_product.id in product_ids

async def test_list_transactions(adapter):
    """Test that we can list all transactions."""
    products, transactions = create_test_data()
    
    # First, we need to save the products referenced by transactions
    for product in products:
        await adapter.save_product(product)
    
    # Save all sample transactions
    for transaction in transactions:
        await adapter.save_transaction(transaction)
    
    # Get all transactions
    all_transactions = await adapter.list_transactions()
    
    # Verify the count and contents
    assert len(all_transactions) == len(transactions)
    
    # Check that each original transaction exists in the list
    transaction_ids = {t.id for t in all_transactions}
    for original_tx in transactions:
        assert original_tx.id in transaction_ids

async def test_update_product(adapter):
    """Test that products can be updated."""
    products, _ = create_test_data()
    
    # Save a product
    await adapter.save_product(products[0])
    
    # Update the product
    updated_product = products[0].copy()
    updated_product.price = 25.99
    updated_product.name = "Updated Product Name"
    updated_product.description = "Updated description"
    updated_product.updated_at = datetime.now()
    
    await adapter.save_product(updated_product)
    
    # Retrieve the product and verify the updates
    loaded_product = await adapter.get_product(products[0].id)
    assert loaded_product is not None
    assert loaded_product.price == 25.99
    assert loaded_product.name == "Updated Product Name"
    assert loaded_product.description == "Updated description"

async def test_delete_product(adapter):
    """Test that products can be deleted."""
    products, _ = create_test_data()
    
    # Save all sample products
    for product in products:
        await adapter.save_product(product)
    
    # Delete one product
    await adapter.delete_product(products[0].id)
    
    # Verify it's gone
    deleted_product = await adapter.get_product(products[0].id)
    assert deleted_product is None
    
    # Other products should still exist
    remaining_product = await adapter.get_product(products[1].id)
    assert remaining_product is not None

async def test_delete_transaction(adapter):
    """Test that transactions can be deleted."""
    products, transactions = create_test_data()
    
    # First, we need to save the products referenced by transactions
    for product in products:
        await adapter.save_product(product)
    
    # Save all sample transactions
    for transaction in transactions:
        await adapter.save_transaction(transaction)
    
    # Delete one transaction
    await adapter.delete_transaction(transactions[0].id)
    
    # Verify it's gone
    deleted_tx = await adapter.get_transaction(transactions[0].id)
    assert deleted_tx is None
    
    # Other transactions should still exist
    remaining_tx = await adapter.get_transaction(transactions[1].id)
    assert remaining_tx is not None

async def test_streaming_products(adapter):
    """Test that products can be streamed efficiently."""
    products, _ = create_test_data()
    
    # Save a larger number of products to test streaming
    for i in range(10):
        for product in products:
            # Create variations of each product
            new_product = product.copy()
            new_product.id = uuid.uuid4()
            new_product.name = f"{product.name} - Variation {i}"
            new_product.sku = f"{product.sku}-{i}"
            await adapter.save_product(new_product)
    
    # Stream products
    count = 0
    async with adapter.stream_products() as stream:
        async for product in stream:
            # Verify product has all required fields
            assert product.id is not None
            assert product.name is not None
            assert product.sku is not None
            assert product.price is not None
            count += 1
    
    # We should have 20 products (10 variations of 2 products)
    assert count == 20

async def test_streaming_transactions(adapter):
    """Test that transactions can be streamed efficiently."""
    products, transactions = create_test_data()
    
    # Save the products
    for product in products:
        await adapter.save_product(product)
    
    # Save a larger number of transactions
    for i in range(10):
        for tx in transactions:
            # Create variations of each transaction
            new_tx = tx.copy()
            new_tx.id = uuid.uuid4()
            new_tx.quantity = tx.quantity * (i + 1)
            new_tx.reference = f"{tx.reference}-{i}"
            await adapter.save_transaction(new_tx)
    
    # Stream transactions
    count = 0
    async with adapter.stream_transactions() as stream:
        async for tx in stream:
            # Verify transaction has all required fields
            assert tx.id is not None
            assert tx.product_id is not None
            assert tx.quantity is not None
            assert tx.transaction_type is not None
            count += 1
    
    # We should have 20 transactions (10 variations of 2 transactions)
    assert count == 20

async def test_error_handling(adapter):
    """Test adapter's error handling for various scenarios."""
    # Test get with invalid ID
    nonexistent_id = uuid.uuid4()
    
    # Non-strict mode should return None
    product = await adapter.get_product(nonexistent_id)
    assert product is None
    
    # Strict mode should raise NotFoundError
    with pytest.raises(NotFoundError):
        await adapter.get_product(nonexistent_id, strict=True)
    
    # Test invalid operations in transaction
    products, _ = create_test_data()
    
    # Save a valid product
    await adapter.save_product(products[0])
    
    try:
        async with adapter.transaction():
            # Delete a product that doesn't exist (should fail)
            await adapter.delete_product(uuid.uuid4())
            # This might fail differently depending on adapter
            # Some might just do nothing for non-existent IDs
            
            # Try to save a product with the same unique key (e.g., SKU)
            duplicate = products[0].copy()
            duplicate.id = uuid.uuid4()  # New ID but same SKU
            await adapter.save_product(duplicate)
            await adapter.save_product(duplicate)  # Second save should fail on unique constraint
            
            pytest.fail("Expected a database error but none was raised")
    except Exception:
        # Some exception should be raised (type may vary by adapter)
        pass
    
    # The database should be in a clean state after rollback
    recovered_product = await adapter.get_product(products[0].id)
    assert recovered_product is not None, "Transaction rollback failed to preserve existing data"

if __name__ == "__main__":
    pytest.main()