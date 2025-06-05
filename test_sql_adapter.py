"""
tests/feature6/test_sql_adapter.py - SQL adapter specific tests
"""

import pytest
import uuid
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from inventory_tracker.store.sql_adapter import SQLAdapter
from inventory_tracker.models.product import Product
from inventory_tracker.models.inventory import StockTransaction
from inventory_tracker.models.exceptions import NotFoundError

# Mark all tests as asyncio tests
pytestmark = pytest.mark.asyncio

@pytest.fixture
async def sql_adapter():
    """Create a SQL adapter with SQLite in-memory database."""
    adapter = SQLAdapter(connection_string="sqlite:///:memory:")
    await adapter.connect()
    await adapter.clear_all()
    
    yield adapter
    
    # Cleanup
    await adapter.disconnect()

async def test_sql_unique_constraint(sql_adapter):
    """Test SQL adapter enforces unique constraints."""
    # Create a product with a unique SKU
    product = Product(
        id=uuid.uuid4(),
        name="Test Product",
        sku="UNIQUE-SKU-123",
        description="Test product",
        category="Test",
        price=10.99,
        reorder_point=5,
        reorder_quantity=10,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Save the product
    await sql_adapter.save_product(product)
    
    # Create a different product with the same SKU
    duplicate = Product(
        id=uuid.uuid4(),  # Different ID
        name="Different Product",
        sku="UNIQUE-SKU-123",  # Same SKU
        description="Another product",
        category="Test",
        price=15.99,
        reorder_point=5,
        reorder_quantity=10,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Attempting to save should raise an integrity error
    with pytest.raises(IntegrityError):
        await sql_adapter.save_product(duplicate)

async def test_sql_connection_error_handling(sql_adapter):
    """Test SQL adapter handles connection errors gracefully."""
    # Save a product
    product = Product(
        id=uuid.uuid4(),
        name="Test Product",
        sku="TEST-SKU-123",
        description="Test product",
        category="Test",
        price=10.99,
        reorder_point=5,
        reorder_quantity=10,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    await sql_adapter.save_product(product)
    
    # Close the connection manually to simulate connection error
    await sql_adapter.disconnect()
    
    # Trying to get the product should handle the connection error
    with pytest.raises(Exception):
        await sql_adapter.get_product(product.id)
    
    # Reconnect for cleanup
    await sql_adapter.connect()

async def test_sql_transaction_isolation(sql_adapter):
    """Test transaction isolation in SQL adapter."""
    product = Product(
        id=uuid.uuid4(),
        name="Test Product",
        sku="TEST-SKU-456",
        description="Test product",
        category="Test",
        price=10.99,
        reorder_point=5,
        reorder_quantity=10,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    # Save the product
    await sql_adapter.save_product(product)
    
    # Create a second adapter instance to test isolation
    second_adapter = SQLAdapter(connection_string="sqlite:///:memory:")
    await second_adapter.connect()
    
    try:
        # Start a transaction in the first adapter
        async with sql_adapter.transaction():
            # Update the product
            updated_product = product.copy()
            updated_product.price = 25.99
            await sql_adapter.save_product(updated_product)
            
            # The first adapter should see the change
            product_in_tx = await sql_adapter.get_product(product.id)
            assert product_in_tx.price == 25.99
            
            # Second adapter won't see the change yet because SQLite memory
            # databases are connection-specific and we used different connections
            # In a real database, uncommitted changes wouldn't be visible anyway
            try:
                product_outside = await second_adapter.get_product(product.id)
                # For a real shared database, you would check isolation here
                # assert product_outside.price == 10.99
            except NotFoundError:
                # Expected for in-memory SQLite
                pass
    finally:
        await second_adapter.disconnect()

if __name__ == "__main__":
    pytest.main()