# tests/test_entry_points.py
import pytest
from unittest.mock import patch, MagicMock
import importlib

from inventorytracker.factories import (
    HookRegistry, create_product, ENTRY_POINT_GROUP
)

# Mock an entry point
class MockEntryPoint:
    def __init__(self, name, module, object_name, hook_result):
        self.name = name
        self.module = module
        self.object_name = object_name
        self._hook_result = hook_result

    def load(self):
        return lambda: self._hook_result


@pytest.fixture
def reset_registry():
    """Reset the hook registry before and after tests."""
    HookRegistry.clear_hooks()
    yield
    HookRegistry.clear_hooks()


@pytest.fixture
def mock_entry_points():
    """Set up mock entry points for testing."""
    # Create a sample pre-save hook
    def unique_name_validator(data):
        if data.get('name', '').lower() == 'duplicate':
            raise ValueError("Product name must be unique")
        return data
        
    # Create a sample post-save hook
    def log_product_created(product):
        # This would log the product creation, but we'll just pass for testing
        pass
        
    # Return a list of mock entry points
    return [
        MockEntryPoint(
            "unique_name_validator", 
            "example_plugin.unique_name_validator",
            "get_hooks",
            [('pre_save', 'Product', unique_name_validator)]
        ),
        MockEntryPoint(
            "product_logger", 
            "another_plugin.hooks",
            "get_hooks",
            [('post_save', 'Product', log_product_created)]
        )
    ]


def test_entry_point_hook_loading(reset_registry, mock_entry_points):
    """Test that hooks are loaded from entry points."""
    # Mock the entry point discovery
    with patch('importlib.metadata.entry_points') as mock_entry_points_func:
        # Return our mock entry points when called with the correct group
        mock_entry_points_func.return_value = mock_entry_points
        
        # Force reloading of entry point hooks
        HookRegistry._hooks_loaded = False
        HookRegistry.load_entry_point_hooks()
        
        # Check hooks were registered
        pre_hooks = HookRegistry.get_pre_save_hooks('Product')
        post_hooks = HookRegistry.get_post_save_hooks('Product')
        
        # Should have our default hook plus the one from the entry point
        assert len(pre_hooks) == 2 
        assert len(post_hooks) == 2


def test_entry_point_hook_execution(reset_registry, mock_entry_points):
    """Test that hooks from entry points are executed."""
    # Mock the entry point discovery
    with patch('importlib.metadata.entry_points') as mock_entry_points_func:
        # Return our mock entry points when called with the correct group
        mock_entry_points_func.return_value = mock_entry_points
        
        # Force reloading of entry point hooks
        HookRegistry._hooks_loaded = False
        
        # Try to create a product with a name that will be rejected by the validator
        with pytest.raises(ValueError) as excinfo:
            create_product(
                name="duplicate",  # This will trigger the validator
                sku="TEST123",
                price=10.00,
                reorder_level=5
            )
            
        assert "unique" in str(excinfo.value).lower()
        
        # Try with a valid name
        product = create_product(
            name="valid name",
            sku="TEST123",
            price=10.00,
            reorder_level=5
        )
        
        # Should succeed and have the title-cased name
        assert product.name == "Valid Name"


def test_entry_point_error_handling(reset_registry):
    """Test that errors in entry points are handled gracefully."""
    # Create a mock entry point that raises an exception
    def failing_loader():
        raise RuntimeError("Failed to load hook")
        
    mock_ep = MockEntryPoint(
        "failing_plugin", 
        "failing.plugin",
        "get_hooks",
        None  # Will be ignored because loader raises exception
    )
    mock_ep.load = failing_loader
    
    # Mock the entry point discovery
    with patch('importlib.metadata.entry_points') as mock_entry_points_func, \
         patch('inventorytracker.factories.logger') as mock_logger:
        # Return our failing mock entry point
        mock_entry_points_func.return_value = [mock_ep]
        
        # Force reloading of entry point hooks
        HookRegistry._hooks_loaded = False
        HookRegistry.load_entry_point_hooks()
        
        # Should log the error but not crash
        mock_logger.error.assert_called()
        
        # Should still work with default hooks
        product = create_product(
            name="test product",
            sku="TEST123",
            price=10.00,
            reorder_level=5
        )
        
        # Default hook should still run
        assert product.name == "Test Product"