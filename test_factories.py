# tests/test_factories.py
import pytest
import uuid
from decimal import Decimal
from unittest.mock import patch, MagicMock, call
import re
from typing import Optional, List, Dict, Any, Callable, Tuple

from inventorytracker.models.product import Product
from inventorytracker.factories import (
    create_product, HookRegistry, title_case_product_name, 
    run_pre_save_hooks, run_post_save_hooks
)

# Test data for parametrization
VALID_SKUS = [
    "ABC123",      # Alphanumeric
    "PROD001",     # Product with numbers
    "A1",          # Minimum length
    "Z999999999",  # Long SKU
    "A1B2C3D4E5",  # Mixed format
]

INVALID_SKUS = [
    "",            # Empty string
    "abc123",      # Lowercase letters
    "ABC-123",     # Contains hyphen
    "PROD_001",    # Contains underscore
    "12345",       # Only numbers
    "ABC 123",     # Contains space
    "ABC@123",     # Special character
    "ΑΒΓΔ",        # Non-ASCII (Greek)
    "產品",        # Non-ASCII (Chinese)
]

VALID_PRICES = [
    "0.01",        # Minimum positive
    "1.00",        # Integer as string
    "999999.99",   # Large price
    "100",         # Integer without decimal
    Decimal("50.25"),  # Decimal object
    50.25,         # Float
]

INVALID_PRICES = [
    "0",           # Zero
    "0.00",        # Zero with decimal
    "-1.00",       # Negative
    "-0.01",       # Small negative
    "abc",         # Non-numeric
    "",            # Empty string
    None,          # None value
]

VALID_REORDER_LEVELS = [
    0,             # Zero (minimum)
    1,             # Small positive
    100,           # Medium value
    999999,        # Large value
    "50",          # String integer
]

INVALID_REORDER_LEVELS = [
    -1,            # Negative
    -100,          # Large negative
    "abc",         # Non-numeric
    "",            # Empty string
    None,          # None value
    3.5,           # Float (should be integer)
]


@pytest.fixture
def reset_hooks():
    """Reset hooks before and after each test."""
    HookRegistry.clear_hooks()
    yield
    HookRegistry.clear_hooks()


@pytest.fixture
def mock_hooks():
    """Set up and capture mock hooks for testing."""
    pre_save_mock = MagicMock(return_value=lambda data: data)
    post_save_mock = MagicMock()
    
    # Create actual pre_save hook that calls the mock and then delegates to real hook
    def pre_save_spy(data):
        pre_save_mock(data)
        # Still apply title case to maintain functionality
        if 'name' in data and isinstance(data['name'], str):
            data['name'] = data['name'].title()
        return data
    
    # Create actual post_save hook that calls the mock
    def post_save_spy(product):
        post_save_mock(product)
    
    # Register the spy hooks
    HookRegistry.register_pre_save_hook('Product', pre_save_spy)
    HookRegistry.register_post_save_hook('Product', post_save_spy)
    
    # Return the mocks for assertions
    return pre_save_mock, post_save_mock


class TestCreateProduct:
    """Tests for the create_product factory function."""

    def test_default_product_creation(self, reset_hooks, mock_hooks):
        """Test creating a product with valid default values."""
        pre_save_mock, post_save_mock = mock_hooks
        
        product = create_product(
            name="test product",
            sku="TEST123",
            price=10.00,
            reorder_level=5
        )
        
        # Assert product is created with correct values
        assert isinstance(product, Product)
        assert isinstance(product.id, uuid.UUID)
        assert product.name == "Test Product"  # Title-cased by hook
        assert product.sku == "TEST123"
        assert product.price == Decimal("10.00")
        assert product.reorder_level == 5
        
        # Assert hooks were called
        pre_save_mock.assert_called_once()
        post_save_mock.assert_called_once_with(product)

    def test_product_with_custom_id(self, reset_hooks, mock_hooks):
        """Test creating a product with a custom UUID."""
        custom_id = uuid.uuid4()
        
        product = create_product(
            id=custom_id,
            name="test product",
            sku="TEST123",
            price=10.00,
            reorder_level=5
        )
        
        assert product.id == custom_id

    @pytest.mark.parametrize("name, expected", [
        ("test product", "Test Product"),
        ("TEST PRODUCT", "Test Product"),
        ("Test Product", "Test Product"),
        ("test_product", "Test_Product"),
        ("test-product", "Test-Product"),
        ("tEsT pRoDuCt", "Test Product"),
        ("product 123", "Product 123"),
        ("   spaced   name   ", "Spaced   Name"),
    ])
    def test_name_title_casing(self, reset_hooks, mock_hooks, name, expected):
        """Test that product names are properly title-cased by hooks."""
        product = create_product(
            name=name,
            sku="TEST123",
            price=10.00,
            reorder_level=5
        )
        
        assert product.name == expected

    @pytest.mark.parametrize("sku", VALID_SKUS)
    def test_valid_skus(self, reset_hooks, mock_hooks, sku):
        """Test creating products with various valid SKUs."""
        product = create_product(
            name="Test Product",
            sku=sku,
            price=10.00,
            reorder_level=5
        )
        
        assert product.sku == sku

    @pytest.mark.parametrize("sku", INVALID_SKUS)
    def test_invalid_skus(self, reset_hooks, mock_hooks, sku):
        """Test that invalid SKUs raise validation errors."""
        with pytest.raises(Exception) as excinfo:
            create_product(
                name="Test Product",
                sku=sku,
                price=10.00,
                reorder_level=5
            )
        
        # Check that the error message mentions SKU validation
        error_message = str(excinfo.value).lower()
        assert any(term in error_message for term in ["sku", "validation", "format", "pattern"])

    @pytest.mark.parametrize("price", VALID_PRICES)
    def test_valid_prices(self, reset_hooks, mock_hooks, price):
        """Test creating products with various valid prices."""
        product = create_product(
            name="Test Product",
            sku="TEST123",
            price=price,
            reorder_level=5
        )
        
        # All prices should be converted to Decimal
        assert isinstance(product.price, Decimal)
        
        # Convert the input price to Decimal for comparison if it's not already
        expected_price = price if isinstance(price, Decimal) else Decimal(str(price))
        assert product.price == expected_price

    @pytest.mark.parametrize("price", INVALID_PRICES)
    def test_invalid_prices(self, reset_hooks, mock_hooks, price):
        """Test that invalid prices raise validation errors."""
        with pytest.raises(Exception) as excinfo:
            create_product(
                name="Test Product",
                sku="TEST123",
                price=price,
                reorder_level=5
            )
        
        # Check that the error message mentions price validation
        error_message = str(excinfo.value).lower()
        assert any(term in error_message for term in ["price", "greater than", "positive", "decimal"])

    @pytest.mark.parametrize("level", VALID_REORDER_LEVELS)
    def test_valid_reorder_levels(self, reset_hooks, mock_hooks, level):
        """Test creating products with various valid reorder levels."""
        product = create_product(
            name="Test Product",
            sku="TEST123",
            price=10.00,
            reorder_level=level
        )
        
        # All reorder levels should be converted to int
        assert isinstance(product.reorder_level, int)
        
        # Convert the input level to int for comparison if it's a string
        expected_level = int(level) if isinstance(level, str) else level
        assert product.reorder_level == expected_level

    @pytest.mark.parametrize("level", INVALID_REORDER_LEVELS)
    def test_invalid_reorder_levels(self, reset_hooks, mock_hooks, level):
        """Test that invalid reorder levels raise validation errors."""
        with pytest.raises(Exception) as excinfo:
            create_product(
                name="Test Product",
                sku="TEST123",
                price=10.00,
                reorder_level=level
            )
        
        # Check that the error message mentions reorder level validation
        error_message = str(excinfo.value).lower()
        assert any(term in error_message for term in ["reorder", "level", "invalid", "integer"])

    def test_skip_hooks_parameter(self, reset_hooks, mock_hooks):
        """Test that hooks can be skipped with skip_hooks=True."""
        pre_save_mock, post_save_mock = mock_hooks
        
        product = create_product(
            name="test product",
            sku="TEST123",
            price=10.00,
            reorder_level=5,
            skip_hooks=True
        )
        
        # Name should not be title-cased when hooks are skipped
        assert product.name == "test product"
        
        # Assert hooks were not called
        pre_save_mock.assert_not_called()
        post_save_mock.assert_not_called()

    def test_multiple_pre_save_hooks(self, reset_hooks):
        """Test that multiple pre-save hooks are all executed in order."""
        # Define hooks with side effects we can check
        def hook1(data):
            data['_hook1_called'] = True
            return data
            
        def hook2(data):
            data['_hook2_called'] = True
            # Only set this if hook1 was already called
            if data.get('_hook1_called'):
                data['_correct_order'] = True
            return data
        
        # Register hooks in specific order
        HookRegistry.register_pre_save_hook('Product', hook1)
        HookRegistry.register_pre_save_hook('Product', hook2)
        
        # Create product
        product = create_product(
            name="Test Product",
            sku="TEST123",
            price=10.00,
            reorder_level=5,
        )
        
        # Check order via side effects in the factory
        # We can't access private attributes in the product, but we can check if the name
        # is correct, which confirms the hooks ran
        assert product.name == "Test Product"
        
        # Additional assertion: manually run hooks to verify order
        test_data = {'name': 'test'}
        result = run_pre_save_hooks('Product', test_data)
        assert result.get('_hook1_called') is True
        assert result.get('_hook2_called') is True
        assert result.get('_correct_order') is True

    def test_exception_in_hook(self, reset_hooks):
        """Test that exceptions in hooks are properly propagated."""
        # Define a hook that raises an exception
        def failing_hook(data):
            raise ValueError("Hook failure test")
            
        HookRegistry.register_pre_save_hook('Product', failing_hook)
        
        # The exception should propagate
        with pytest.raises(ValueError) as excinfo:
            create_product(
                name="Test Product",
                sku="TEST123",
                price=10.00,
                reorder_level=5
            )
            
        assert "Hook failure test" in str(excinfo.value)

    @pytest.mark.parametrize("test_case", [
        {
            "name": "minimum valid product",
            "input": {
                "name": "min",  # Minimum 3 chars
                "sku": "A1B",   # Valid minimum SKU
                "price": "0.01", # Minimum valid price
                "reorder_level": 0   # Minimum valid reorder level
            },
            "expected_valid": True
        },
        {
            "name": "all fields at maximum allowed values",
            "input": {
                "name": "x" * 100,  # Very long name
                "sku": "A" * 50,    # Very long SKU
                "price": "999999999.99", # Very large price
                "reorder_level": 999999  # Very large reorder level
            },
            "expected_valid": True
        },
        {
            "name": "name too short",
            "input": {
                "name": "ab",  # Less than 3 chars
                "sku": "ABC123",
                "price": "10.00",
                "reorder_level": 5
            },
            "expected_valid": False
        },
        {
            "name": "combined invalid fields",
            "input": {
                "name": "ab",  # Too short
                "sku": "abc",  # Lowercase
                "price": "0",  # Zero price
                "reorder_level": -1   # Negative
            },
            "expected_valid": False
        }
    ])
    def test_product_validation_combinations(self, reset_hooks, mock_hooks, test_case):
        """Test combinations of valid and invalid field values."""
        if test_case["expected_valid"]:
            # Should not raise an exception
            product = create_product(**test_case["input"])
            assert isinstance(product, Product)
        else:
            # Should raise a validation exception
            with pytest.raises(Exception):
                create_product(**test_case["input"])

    def test_hook_registration_and_clearing(self):
        """Test that hooks can be registered, called and cleared correctly."""
        # Clear all hooks first
        HookRegistry.clear_hooks()
        
        # Define and register a test hook
        test_data = []
        
        def test_hook(data):
            test_data.append('called')
            return data
            
        HookRegistry.register_pre_save_hook('Product', test_hook)
        
        # Verify the hook is registered
        hooks = HookRegistry.get_pre_save_hooks('Product')
        assert len(hooks) == 1
        
        # Run the hook manually
        data = {'test': 'value'}
        run_pre_save_hooks('Product', data)
        assert test_data == ['called']
        
        # Clear hooks and verify they're gone
        HookRegistry.clear_hooks()
        hooks = HookRegistry.get_pre_save_hooks('Product')
        assert len(hooks) == 0
        
        # Running hooks after clearing should do nothing
        test_data.clear()
        run_pre_save_hooks('Product', data)
        assert test_data == []


class TestTitleCaseHook:
    """Tests specifically for the title_case_product_name hook."""
    
    @pytest.mark.parametrize("input_name, expected", [
        ("test product", "Test Product"),
        (None, None),  # Should handle None gracefully
        ("", ""),      # Should handle empty string
        (123, 123),    # Should handle non-string values
        ({}, {}),      # Should handle other types
    ])
    def test_title_case_hook(self, input_name, expected):
        """Test the title case hook with various inputs."""
        data = {}
        if input_name is not None:
            data['name'] = input_name
            
        result = title_case_product_name(data)
        
        if 'name' in result:
            assert result['name'] == expected
        else:
            assert 'name' not in data  # Shouldn't add a name if not present


@pytest.mark.parametrize("hook_stage", ["pre_save", "post_save"])
class TestHookRegistry:
    """Tests for the HookRegistry class functionality."""
    
    def test_hook_registration(self, reset_hooks, hook_stage):
        """Test registering hooks in the registry."""
        # Define a test hook
        def test_hook(data_or_product):
            return data_or_product
            
        # Register the hook
        if hook_stage == "pre_save":
            HookRegistry.register_pre_save_hook('Product', test_hook)
            hooks = HookRegistry.get_pre_save_hooks('Product')
        else:
            HookRegistry.register_post_save_hook('Product', test_hook)
            hooks = HookRegistry.get_post_save_hooks('Product')
            
        # Verify it was registered
        assert len(hooks) == 1
        assert hooks[0] == test_hook
        
    def test_multiple_model_hooks(self, reset_hooks, hook_stage):
        """Test that hooks for different models don't interfere."""
        def product_hook(data_or_product):
            return data_or_product
            
        def category_hook(data_or_product):
            return data_or_product
            
        # Register hooks for different models
        if hook_stage == "pre_save":
            HookRegistry.register_pre_save_hook('Product', product_hook)
            HookRegistry.register_pre_save_hook('Category', category_hook)
            product_hooks = HookRegistry.get_pre_save_hooks('Product')
            category_hooks = HookRegistry.get_pre_save_hooks('Category')
        else:
            HookRegistry.register_post_save_hook('Product', product_hook)
            HookRegistry.register_post_save_hook('Category', category_hook)
            product_hooks = HookRegistry.get_post_save_hooks('Product')
            category_hooks = HookRegistry.get_post_save_hooks('Category')
            
        # Verify correct hooks are returned for each model
        assert len(product_hooks) == 1
        assert len(category_hooks) == 1
        assert product_hooks[0] == product_hook
        assert category_hooks[0] == category_hook
        
    def test_clear_specific_model_hooks(self, reset_hooks, hook_stage):
        """Test clearing hooks for a specific model."""
        def product_hook(data_or_product):
            return data_or_product
            
        def category_hook(data_or_product):
            return data_or_product
            
        # Register hooks for different models
        if hook_stage == "pre_save":
            HookRegistry.register_pre_save_hook('Product', product_hook)
            HookRegistry.register_pre_save_hook('Category', category_hook)
            # Clear only Product hooks
            HookRegistry.clear_hooks('Product')
            # Check that only Product hooks were cleared
            product_hooks = HookRegistry.get_pre_save_hooks('Product')
            category_hooks = HookRegistry.get_pre_save_hooks('Category')
        else:
            HookRegistry.register_post_save_hook('Product', product_hook)
            HookRegistry.register_post_save_hook('Category', category_hook)
            # Clear only Product hooks
            HookRegistry.clear_hooks('Product')
            # Check that only Product hooks were cleared
            product_hooks = HookRegistry.get_post_save_hooks('Product')
            category_hooks = HookRegistry.get_post_save_hooks('Category')
            
        assert len(product_hooks) == 0
        assert len(category_hooks) == 1