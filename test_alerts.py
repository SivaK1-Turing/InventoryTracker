# tests/test_alerts.py
import pytest
import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import uuid
from copy import deepcopy

from inventorytracker.alerts import (
    detect_low_stock, StockAlert, clear_alert_cache,
    get_critical_stock_alerts, _process_product_batch
)
from inventorytracker.models.product import Product

# Sample product data for testing
@pytest.fixture
def sample_products():
    """
    Fixture providing a variety of product scenarios for testing low stock detection.
    """
    return [
        # Standard case - below reorder level
        Product(
            id=uuid.uuid4(),
            name="Product A",
            sku="SKU001",
            price=10.99,
            current_stock=5,
            reorder_level=10
        ),
        # Edge case - exactly at reorder level (should not trigger)
        Product(
            id=uuid.uuid4(),
            name="Product B",
            sku="SKU002",
            price=20.50,
            current_stock=15,
            reorder_level=15
        ),
        # Edge case - out of stock (priority 1)
        Product(
            id=uuid.uuid4(),
            name="Product C",
            sku="SKU003",
            price=5.99,
            current_stock=0,
            reorder_level=8
        ),
        # Edge case - negative stock (data error but should be handled)
        Product(
            id=uuid.uuid4(),
            name="Product D",
            sku="SKU004",
            price=15.75,
            current_stock=-2,  # Indicates a data error
            reorder_level=10
        ),
        # Edge case - zero reorder level (special case product)
        Product(
            id=uuid.uuid4(),
            name="Product E",
            sku="SKU005",
            price=25.00,
            current_stock=5,
            reorder_level=0
        ),
        # Standard case - well above reorder level
        Product(
            id=uuid.uuid4(),
            name="Product F",
            sku="SKU006",
            price=30.00,
            current_stock=100,
            reorder_level=10
        )
    ]

@pytest.fixture
def mock_store(sample_products):
    """
    Fixture providing a mock store with sample products.
    """
    mock = Mock()
    mock.get_all_products.return_value = sample_products
    # Make supports_query return False to use default processing path
    mock.supports_query = False
    return mock

@pytest.fixture
def mock_store_with_query(sample_products):
    """
    Fixture providing a mock store with QueryManager support
    """
    mock = Mock()
    # Use the same products but return them via the query path
    mock.supports_query = True
    mock.get_all_products.return_value = sample_products
    return mock

@pytest.fixture
def expected_alerts(sample_products):
    """
    Fixture providing the expected alerts for the sample products.
    """
    # Only products A, C, D should generate alerts
    # Product B is exactly at reorder level, not below
    # Product E has reorder level 0, so not below
    # Product F is above reorder level
    
    return [
        StockAlert(
            product_id=sample_products[0].id,
            product_name=sample_products[0].name,
            product_sku=sample_products[0].sku,
            current_stock=sample_products[0].current_stock,
            reorder_level=sample_products[0].reorder_level,
            original_reorder_level=sample_products[0].reorder_level,
            deficit=sample_products[0].reorder_level - sample_products[0].current_stock,
            timestamp=datetime.datetime.now(),  # This will be different, we'll compare without timestamp
        ),
        StockAlert(
            product_id=sample_products[2].id,
            product_name=sample_products[2].name,
            product_sku=sample_products[2].sku,
            current_stock=sample_products[2].current_stock,
            reorder_level=sample_products[2].reorder_level,
            original_reorder_level=sample_products[2].reorder_level,
            deficit=sample_products[2].reorder_level - sample_products[2].current_stock,
            timestamp=datetime.datetime.now(),
        ),
        StockAlert(
            product_id=sample_products[3].id,
            product_name=sample_products[3].name,
            product_sku=sample_products[3].sku,
            current_stock=sample_products[3].current_stock,
            reorder_level=sample_products[3].reorder_level,
            original_reorder_level=sample_products[3].reorder_level,
            deficit=sample_products[3].reorder_level - sample_products[3].current_stock,
            timestamp=datetime.datetime.now(),
        ),
    ]

def compare_alerts_ignoring_timestamp(actual, expected):
    """Helper to compare alerts ignoring timestamp differences."""
    if len(actual) != len(expected):
        return False
    
    for a, e in zip(actual, expected):
        if (a.product_id != e.product_id or
            a.product_name != e.product_name or
            a.product_sku != e.product_sku or
            a.current_stock != e.current_stock or
            a.reorder_level != e.reorder_level or
            a.original_reorder_level != e.original_reorder_level or
            a.deficit != e.deficit):
            return False
    
    return True

# Test the basic functionality
def test_detect_low_stock_basic(mock_store, expected_alerts):
    """Test the basic functionality of detect_low_stock."""
    # Run the detection
    actual_alerts = detect_low_stock(store=mock_store)
    
    # Verify correct number of alerts
    assert len(actual_alerts) == len(expected_alerts)
    
    # Compare alerts ignoring timestamps
    assert compare_alerts_ignoring_timestamp(actual_alerts, expected_alerts)
    
    # Verify they're sorted by priority
    priorities = [alert.priority for alert in actual_alerts]
    assert priorities == sorted(priorities)

# Test with the query-based store path
def test_detect_low_stock_with_query_support(mock_store_with_query, expected_alerts):
    """Test detect_low_stock with a store that supports queries."""
    # Run the detection
    actual_alerts = detect_low_stock(store=mock_store_with_query)
    
    # Verify correct number of alerts
    assert len(actual_alerts) == len(expected_alerts)
    
    # Compare alerts ignoring timestamps
    assert compare_alerts_ignoring_timestamp(actual_alerts, expected_alerts)

# Test the overrides functionality
def test_detect_low_stock_with_global_override(mock_store, sample_products):
    """Test that global overrides are correctly applied."""
    # Set a global override that should catch more products
    global_override_level = 20
    
    # Run the detection with global override
    actual_alerts = detect_low_stock(
        store=mock_store,
        global_override=global_override_level,
    )
    
    # Should now include Product B and F (which were above original reorder level)
    # along with the original 3 alerts
    assert len(actual_alerts) == 5
    
    # Verify the override was applied correctly
    for alert in actual_alerts:
        assert alert.reorder_level == global_override_level
        assert alert.original_reorder_level != global_override_level
        assert alert.was_overridden
        assert alert.override_source == "global_override"

def test_detect_low_stock_with_sku_override(mock_store, sample_products):
    """Test that SKU-specific overrides are correctly applied."""
    # Create overrides for specific SKUs
    sku_overrides = {
        "SKU002": 30,  # Override Product B to trigger an alert
        "SKU006": 150  # Override Product F to trigger an alert
    }
    
    # Run the detection with SKU overrides
    actual_alerts = detect_low_stock(
        store=mock_store,
        sku_overrides=sku_overrides
    )
    
    # Should now include Product B and F along with the original 3
    assert len(actual_alerts) == 5
    
    # Verify the overrides were applied correctly
    sku_to_alert = {alert.product_sku: alert for alert in actual_alerts}
    
    # Check Product B override
    assert "SKU002" in sku_to_alert
    assert sku_to_alert["SKU002"].reorder_level == 30
    assert sku_to_alert["SKU002"].original_reorder_level == 15
    assert sku_to_alert["SKU002"].was_overridden
    assert sku_to_alert["SKU002"].override_source == "sku_override"
    
    # Check Product F override
    assert "SKU006" in sku_to_alert
    assert sku_to_alert["SKU006"].reorder_level == 150
    assert sku_to_alert["SKU006"].original_reorder_level == 10
    assert sku_to_alert["SKU006"].was_overridden
    assert sku_to_alert["SKU006"].override_source == "sku_override"

def test_sku_override_precedence_over_global(mock_store, sample_products):
    """Test that SKU overrides take precedence over global overrides."""
    # Set both global and SKU-specific overrides
    global_override_level = 20
    sku_overrides = {
        "SKU002": 30  # This should take precedence
    }
    
    # Run the detection with both types of overrides
    actual_alerts = detect_low_stock(
        store=mock_store,
        global_override=global_override_level,
        sku_overrides=sku_overrides
    )
    
    # Verify precedence
    sku_to_alert = {alert.product_sku: alert for alert in actual_alerts}
    
    # Check Product B got the SKU override, not global
    assert "SKU002" in sku_to_alert
    assert sku_to_alert["SKU002"].reorder_level == 30
    assert sku_to_alert["SKU002"].original_reorder_level == 15
    assert sku_to_alert["SKU002"].override_source == "sku_override"
    
    # Check a product with only global override
    assert "SKU006" in sku_to_alert  # Product F should have global override
    assert sku_to_alert["SKU006"].reorder_level == global_override_level
    assert sku_to_alert["SKU006"].original_reorder_level == 10
    assert sku_to_alert["SKU006"].override_source == "global_override"

@patch('os.environ', {
    'REORDER_LEVEL_SKU002': '25',  # Environment override for Product B
    'REORDER_LEVEL_SKU006': '120'  # Environment override for Product F
})
def test_detect_low_stock_with_env_overrides(mock_store, sample_products):
    """Test that environment variable overrides are correctly applied."""
    # Run the detection with env overrides enabled
    actual_alerts = detect_low_stock(
        store=mock_store,
        include_env_overrides=True
    )
    
    # Should include Products B and F along with the original 3
    assert len(actual_alerts) == 5
    
    # Verify the environment overrides were applied
    sku_to_alert = {alert.product_sku: alert for alert in actual_alerts}
    
    # Check Product B override from environment
    assert "SKU002" in sku_to_alert
    assert sku_to_alert["SKU002"].reorder_level == 25
    assert sku_to_alert["SKU002"].original_reorder_level == 15
    
    # Check Product F override from environment
    assert "SKU006" in sku_to_alert
    assert sku_to_alert["SKU006"].reorder_level == 120
    assert sku_to_alert["SKU006"].original_reorder_level == 10

@patch('os.environ', {'REORDER_LEVEL_SKU002': '25'})
def test_env_overrides_disabled(mock_store, expected_alerts):
    """Test that environment overrides can be disabled."""
    # Run the detection with env overrides disabled
    actual_alerts = detect_low_stock(
        store=mock_store,
        include_env_overrides=False
    )
    
    # Should be the original 3 alerts only (no env overrides)
    assert len(actual_alerts) == len(expected_alerts)
    assert compare_alerts_ignoring_timestamp(actual_alerts, expected_alerts)

def test_override_precedence_all_types(mock_store):
    """
    Test the precedence order of overrides: 
    CLI-provided sku_overrides > config > environment variables
    """
    # Patch environment variables
    env_patch = patch('os.environ', {
        'REORDER_LEVEL_SKU002': '25',  # Env override for SKU002
    })
    
    # Patch config overrides
    config_overrides = {
        "SKU002": 30,  # Config override should take precedence over env
    }
    config_patch = patch(
        'inventorytracker.alerts.get_overrides_from_config',
        return_value=config_overrides
    )
    
    # CLI overrides (highest precedence)
    cli_overrides = {
        "SKU002": 35,  # CLI override should take precedence over all
    }
    
    # Apply all patches
    with env_patch, config_patch:
        # Run detection with all override types enabled
        actual_alerts = detect_low_stock(
            store=mock_store,
            sku_overrides=cli_overrides,
            include_env_overrides=True,
            include_config_overrides=True
        )
        
        # Check that CLI override took precedence
        sku_to_alert = {alert.product_sku: alert for alert in actual_alerts}
        assert "SKU002" in sku_to_alert
        assert sku_to_alert["SKU002"].reorder_level == 35  # CLI value


# Edge case tests

def test_edge_case_exact_threshold(mock_store, sample_products):
    """Test the edge case where stock is exactly at reorder level (no alert expected)."""
    # Get alerts
    alerts = detect_low_stock(store=mock_store)
    
    # Product B is exactly at threshold and should not trigger an alert
    assert "SKU002" not in {a.product_sku for a in alerts}
    
    # Now modify Product B to be just below threshold
    modified_products = deepcopy(sample_products)
    modified_products[1].current_stock = 14  # One below reorder level
    
    # Update mock store
    mock_store.get_all_products.return_value = modified_products
    
    # Get alerts again
    alerts = detect_low_stock(store=mock_store)
    
    # Now Product B should trigger an alert
    assert "SKU002" in {a.product_sku for a in alerts}

def test_edge_case_negative_stock(mock_store, sample_products):
    """Test handling of negative stock values."""
    # Product D has negative stock, ensure it's handled correctly
    alerts = detect_low_stock(store=mock_store)

    # Find alert for Product D
    product_d_alerts = [a for a in alerts if a.product_sku == "SKU004"]
    assert len(product_d_alerts) == 1
    
    # Verify deficit calculation is correct for negative stock
    product_d = product_d_alerts[0]
    assert product_d.current_stock == -2
    assert product_d.reorder_level == 10
    assert product_d.deficit == 12  # 10 - (-2) = 12

def test_edge_case_zero_reorder_level(mock_store, sample_products):
    """Test handling of products with zero reorder level."""
    # Product E has reorder_level=0
    alerts = detect_low_stock(store=mock_store)
    
    # Product E should not trigger an alert since its current_stock (5) is not below reorder_level (0)
    assert "SKU005" not in {a.product_sku for a in alerts}
    
    # Now modify Product E to have -1 stock (below reorder level of 0)
    modified_products = deepcopy(sample_products)
    modified_products[4].current_stock = -1
    
    # Update mock store
    mock_store.get_all_products.return_value = modified_products
    
    # Get alerts again
    alerts = detect_low_stock(store=mock_store)
    
    # Now Product E should trigger an alert
    assert "SKU005" in {a.product_sku for a in alerts}

def test_empty_product_list(mock_store):
    """Test handling of an empty product list."""
    # Set empty product list
    mock_store.get_all_products.return_value = []
    
    # Run detection
    alerts = detect_low_stock(store=mock_store)
    
    # Should be empty but not fail
    assert len(alerts) == 0

# Performance tests

def test_batch_processing():
    """Test that _process_product_batch correctly identifies low stock products."""
    # Create test products
    products = [
        Product(
            id=uuid.uuid4(),
            name=f"Product {i}",
            sku=f"SKU{i:03d}",
            price=float(i),
            current_stock=5 if i % 2 == 0 else 15,  # Half are low stock
            reorder_level=10
        )
        for i in range(10)
    ]
    
    # Process batch
    alerts = _process_product_batch(products)
    
    # Half should trigger alerts (those with stock=5, reorder=10)
    assert len(alerts) == 5
    
    # All alerts should be for products with current_stock=5
    for alert in alerts:
        assert alert.current_stock == 5

def generate_large_dataset(size: int, low_stock_ratio: float = 0.2) -> List[Product]:
    """
    Generate a large dataset of test products.
    
    Args:
        size: Number of products to generate
        low_stock_ratio: Ratio of products that should be below reorder level
        
    Returns:
        List of Product objects
    """
    products = []
    low_stock_count = int(size * low_stock_ratio)
    
    for i in range(size):
        # Make some products below reorder level
        if i < low_stock_count:
            current_stock = 5
            reorder_level = 10
        else:
            current_stock = 20
            reorder_level = 10
        
        product = Product(
            id=uuid.uuid4(),
            name=f"Product {i}",
            sku=f"SKU{i:06d}",
            price=float(i % 100) + 0.99,
            current_stock=current_stock,
            reorder_level=reorder_level
        )
        products.append(product)
    
    return products

@pytest.mark.slow
def test_large_dataset_performance():
    """Test performance with a large dataset (50,000 products)."""
    # Skip in normal test runs - use pytest -m slow to run
    import time
    
    # Generate 50,000 products (with 20% below reorder level)
    products = generate_large_dataset(50000, 0.2)
    
    # Create a mock store
    mock_store = Mock()
    mock_store.get_all_products.return_value = products
    mock_store.supports_query = False
    
    # Measure execution time
    start_time = time.time()
    alerts = detect_low_stock(store=mock_store)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # Verify expected alerts count (20% of 50,000 = 10,000)
    assert len(alerts) == 10000
    
    # Log execution time for information
    print(f"\nLarge dataset (50,000 products) processed in {execution_time:.2f} seconds")
    
    # This is an approximate benchmark - adjust as needed
    # On modern hardware, this should be well under 2 seconds
    assert execution_time < 5.0, f"Performance too slow: {execution_time:.2f}s"

@pytest.mark.parametrize("chunk_size,max_workers", [
    (1000, 1),    # Single worker, large chunks
    (100, 4),     # More workers, smaller chunks
    (5000, 2),    # Few workers, very large chunks
    (50, 8)       # Many workers, tiny chunks
])
@pytest.mark.slow
def test_chunking_and_parallelism(chunk_size, max_workers):
    """
    Test different combinations of chunk sizes and worker counts.
    This helps find the optimal configuration for different dataset sizes.
    """
    import time
    from inventorytracker.alerts import CHUNK_SIZE, MAX_WORKERS
    
    # Patch the constants
    with patch('inventorytracker.alerts.CHUNK_SIZE', chunk_size), \
         patch('inventorytracker.alerts.MAX_WORKERS', max_workers):
        
        # Generate 10,000 products
        products = generate_large_dataset(10000, 0.2)
        
        # Create a mock store
        mock_store = Mock()
        mock_store.get_all_products.return_value = products
        mock_store.supports_query = False
        
        # Measure execution time
        start_time = time.time()
        alerts = detect_low_stock(store=mock_store)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Verify expected alerts count
        assert len(alerts) == 2000  # 20% of 10,000
        
        # Log performance data
        print(f"\nConfiguration: chunk_size={chunk_size}, max_workers={max_workers}")
        print(f"Execution time: {execution_time:.4f} seconds")

def test_caching_with_critical_alerts():
    """Test that get_critical_stock_alerts caches results correctly."""
    # Generate test data
    products = generate_large_dataset(100, 0.5)
    
    # Create mock store
    mock_store = Mock()
    mock_store.get_all_products.return_value = products
    mock_store.supports_query = False
    
    # Clear cache to start clean
    clear_alert_cache()
    
    # Call get_critical_stock_alerts first time
    with patch('inventorytracker.alerts.detect_low_stock') as mock_detect:
        # Make mock_detect pass through to real implementation
        mock_detect.side_effect = lambda *args, **kwargs: detect_low_stock(
            mock_store, *args, **kwargs
        )
        
        # First call should hit detect_low_stock
        alerts1 = get_critical_stock_alerts()
        
        # Second call should use cache
        alerts2 = get_critical_stock_alerts()
        
        # Verify detect_low_stock was called once
        assert mock_detect.call_count == 1
        
        # Verify cache hit produces same results
        assert len(alerts1) == len(alerts2)

def test_different_override_combinations_bust_cache():
    """Test that different override combinations don't share the same cache entry."""
    # Generate test data
    products = generate_large_dataset(100, 0.5)
    
    # Create mock store
    mock_store = Mock()
    mock_store.get_all_products.return_value = products
    mock_store.supports_query = False
    
    # Clear cache to start clean
    clear_alert_cache()
    
    with patch('inventorytracker.alerts.detect_low_stock') as mock_detect:
        # Make mock_detect pass through to real implementation
        mock_detect.side_effect = lambda *args, **kwargs: detect_low_stock(
            mock_store, *args, **kwargs
        )
        
        # Call with different override combinations
        get_critical_stock_alerts()  # No overrides
        get_critical_stock_alerts(global_override=15)  # Global override
        get_critical_stock_alerts(sku_overrides={"SKU001": 20})  # SKU override
        get_critical_stock_alerts(include_env_overrides=False)  # Env disabled
        
        # Each different combination should call detect_low_stock
        assert mock_detect.call_count == 4

# Additional tests for real-world scenarios

def test_reorder_level_higher_than_possible_stock():
    """
    Test a realistic scenario where reorder level is higher than 
    the maximum possible stock (e.g., due to supplier constraints).
    """
    # Create a product with reorder_level higher than its capacity
    product = Product(
        id=uuid.uuid4(),
        name="Limited Supply Product",
        sku="LSP001",
        price=99.99,
        current_stock=5,  # Current stock
        reorder_level=100  # Unrealistically high reorder level
    )
    
    # Create mock store with this product
    mock_store = Mock()
    mock_store.get_all_products.return_value = [product]
    mock_store.supports_query = False
    
    # Get alerts
    alerts = detect_low_stock(store=mock_store)
    
    # Should trigger an alert since current_stock < reorder_level
    assert len(alerts) == 1
    assert alerts[0].deficit == 95  # 100 - 5
    assert alerts[0].priority == 2  # High priority due to large deficit

def test_alerts_sorting_by_priority():
    """Test that alerts are properly sorted by priority."""
    # Create products with different priorities
    products = [
        # Priority 1 (out of stock)
        Product(
            id=uuid.uuid4(),
            name="Out of Stock",
            sku="P1",
            price=10.00,
            current_stock=0,
            reorder_level=10
        ),
        # Priority 3 (moderate deficit)
        Product(
            id=uuid.uuid4(),
            name="Moderate Deficit",
            sku="P3",
            price=10.00,
            current_stock=6,
            reorder_level=10
        ),
        # Priority 2 (large deficit)
        Product(
            id=uuid.uuid4(),
            name="Large Deficit",
            sku="P2",
            price=10.00,
            current_stock=2,
            reorder_level=10
        )
    ]
    
    # Create mock store with these products
    mock_store = Mock()
    mock_store.get_all_products.return_value = products
    mock_store.supports_query = False
    
    # Get alerts
    alerts = detect_low_stock(store=mock_store)
    
    # Verify sorting by priority
    assert len(alerts) == 3
    assert alerts[0].product_sku == "P1"  # Priority 1 (out of stock)
    assert alerts[1].product_sku == "P2"  # Priority 2 (large deficit)
    assert alerts[2].product_sku == "P3"  # Priority 3 (moderate deficit)