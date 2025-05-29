# tests/feature2/test_product_commands.py
import pytest
import os
import json
import re
import uuid
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner
import jsonschema

from inventorytracker.main import app
from inventorytracker.store import get_store
from inventorytracker.models.product import Product

# Setup a CliRunner for testing Typer apps
runner = CliRunner()

# Product JSON Schema for validation
PRODUCT_SCHEMA = {
    "type": "object",
    "required": ["id", "name", "sku", "price", "reorder_level"],
    "properties": {
        "id": {"type": "string", "format": "uuid"},
        "name": {"type": "string", "minLength": 3},
        "sku": {
            "type": "string", 
            "pattern": "^[A-Z0-9]+$"
        },
        "price": {"type": "number", "exclusiveMinimum": 0},
        "reorder_level": {"type": "integer", "minimum": 0}
    }
}

# Products list schema
PRODUCTS_LIST_SCHEMA = {
    "type": "array",
    "items": PRODUCT_SCHEMA
}


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for tests."""
    db_file = tmp_path / "test_inventory.db"
    return str(db_file)


@pytest.fixture
def mock_store(temp_db_path):
    """Set up a test store with a temporary database."""
    # Mock the get_store function to return a test store
    with patch('inventorytracker.commands.products.get_store') as mock_get_store, \
         patch('inventorytracker.store.get_store') as global_mock_get_store:
        
        # Dynamically import the store implementation to avoid circular imports
        from inventorytracker.store.sqlite_store import SQLiteStore
        
        # Create a test store with the temporary database
        test_store = SQLiteStore(db_path=temp_db_path)
        
        # Configure mocks to return our test store
        mock_get_store.return_value = test_store
        global_mock_get_store.return_value = test_store
        
        # Provide the store to the test
        yield test_store
        
        # Cleanup
        test_store.close()
        if os.path.exists(temp_db_path):
            try:
                os .remove(temp_db_path)
            except PermissionError:
                # If Windows locks the file, just leave it for cleanup later
                pass


def extract_uuid_from_output(output: str) -> str:
    """Extract UUID from command output."""
    matches = re.search(r'ID:\s+([0-9a-f-]+)', output)
    if matches:
        return matches.group(1)
    return None


def parse_json_output(output: str) -> Dict[str, Any]:
    """Extract and parse JSON from command output."""
    # Find the start of JSON (to handle any leading text)
    json_start = output.find('[')
    if json_start == -1:
        json_start = output.find('{')
    
    if json_start == -1:
        raise ValueError(f"No JSON found in output: {output}")
        
    # Extract JSON part
    json_text = output[json_start:]
    
    # Parse JSON
    return json.loads(json_text)


@pytest.mark.integration
class TestProductCommands:
    """Integration tests for product-related CLI commands."""
    
    def test_add_and_list_products(self, mock_store):
        """
        Test adding a product with interactive prompts and then listing it.
        
        This test:
        1. Runs add-product with simulated user input
        2. Runs list-products --json to get machine-readable output
        3. Validates the new product appears with correct schema
        """
        # Test data for the product
        product_name = "Test Water Bottle"
        product_sku = "BOTTLE001"
        product_price = "24.99"
        product_reorder = "15"
        
        # Run the add-product command with simulated input
        result = runner.invoke(
            app, 
            ["product", "add"],
            input=f"{product_name}\n{product_sku}\n{product_price}\n{product_reorder}\n"
        )
        
        # Check that the command completed successfully
        assert result.exit_code == 0, f"Command failed with output: {result.stdout}"
        
        # Check for successful product creation message
        assert "successfully" in result.stdout
        assert product_name in result.stdout
        assert product_sku in result.stdout
        assert product_price in result.stdout
        assert product_reorder in result.stdout
        
        # Extract the product ID for verification later
        product_id = extract_uuid_from_output(result.stdout)
        assert product_id is not None, "Could not extract product ID from output"
        
        # Now run list-products with JSON output
        list_result = runner.invoke(
            app,
            ["product", "list", "--json"]
        )
        
        # Check that the command completed successfully
        assert list_result.exit_code == 0, f"List command failed with output: {list_result.stdout}"
        
        # Parse the JSON output
        products = parse_json_output(list_result.stdout)
        
        # Validate against the schema
        jsonschema.validate(products, PRODUCTS_LIST_SCHEMA)
        
        # Check that our product is in the list
        found_product = next((p for p in products if p.get("id") == product_id), None)
        assert found_product is not None, f"Added product with ID {product_id} not found in list output"
        
        # Verify the product details
        assert found_product["name"] == product_name
        assert found_product["sku"] == product_sku
        assert float(found_product["price"]) == float(product_price)
        assert int(found_product["reorder_level"]) == int(product_reorder)
    
    def test_add_product_with_invalid_input_retry(self, mock_store):
        """
        Test that add-product properly handles invalid input and allows retry.
        
        This test:
        1. Provides invalid input first (too short name, invalid SKU)
        2. Then provides valid input on the retry
        3. Confirms the product was created with the corrected input
        """
        # First round: Invalid inputs
        invalid_name = "ab"  # Too short (min 3 chars)
        invalid_sku = "sku-123"  # Invalid format (has dash)
        
        # Second round: Valid corrections
        valid_name = "Valid Product"
        valid_sku = "SKU123"
        valid_price = "19.99"
        valid_reorder = "10"
        
        # Combined input stream - invalid values followed by valid values
        input_stream = (
            f"{invalid_name}\n"  # Invalid name
            f"{valid_name}\n"    # Valid name on retry
            f"{invalid_sku}\n"   # Invalid SKU
            f"{valid_sku}\n"     # Valid SKU on retry
            f"{valid_price}\n"   # Price
            f"{valid_reorder}\n" # Reorder level
        )
        
        # Run command with the input sequence
        result = runner.invoke(
            app,
            ["product", "add"],
            input=input_stream
        )
        
        # Check command succeeded despite the initial invalid inputs
        assert result.exit_code == 0, f"Command failed with output: {result.stdout}"
        
        # Verify error messages for invalid inputs
        assert "Error" in result.stdout
        assert "3 characters" in result.stdout  # Name too short error
        assert "only uppercase letters and numbers" in result.stdout  # SKU format error
        
        # Verify the product was created with the valid inputs
        assert "successfully" in result.stdout
        assert valid_name in result.stdout
        assert valid_sku in result.stdout
        
        # Extract product ID and verify in list
        product_id = extract_uuid_from_output(result.stdout)
        assert product_id is not None
        
        # Verify with list command
        list_result = runner.invoke(app, ["product", "list", "--json"])
        products = parse_json_output(list_result.stdout)
        
        found_product = next((p for p in products if p.get("id") == product_id), None)
        assert found_product is not None
        assert found_product["name"] == valid_name
        assert found_product["sku"] == valid_sku
    
    def test_add_duplicate_product_with_overwrite(self, mock_store):
        """
        Test adding a product with an existing SKU and confirming overwrite.
        
        This test:
        1. Adds an initial product
        2. Tries to add another with the same SKU
        3. Confirms the overwrite prompt
        4. Verifies the product was updated with new values
        """
        # Original product details
        original_name = "Original Product"
        original_sku = "TESTSKU"
        original_price = "10.99"
        original_reorder = "5"
        
        # Create the original product
        result1 = runner.invoke(
            app,
            ["product", "add"],
            input=f"{original_name}\n{original_sku}\n{original_price}\n{original_reorder}\n"
        )
        assert result1.exit_code == 0
        
        # Updated product details
        updated_name = "Updated Product"
        updated_price = "15.99"
        updated_reorder = "10"
        
        # Try to add a product with the same SKU but different details
        result2 = runner.invoke(
            app,
            ["product", "add"],
            input=(
                f"{updated_name}\n"   # New name
                f"{original_sku}\n"   # Same SKU
                f"{updated_price}\n"  # New price
                f"{updated_reorder}\n"# New reorder level
                "y\n"                 # Confirm overwrite
            )
        )
        
        # Check command succeeded
        assert result2.exit_code == 0, f"Command failed with output: {result2.stdout}"
        
        # Verify overwrite warning appeared
        assert "already exists" in result2.stdout
        assert "Overwrite" in result2.stdout
        
        # Verify update success message
        assert "updated successfully" in result2.stdout
        
        # Check that the product was updated by listing products
        list_result = runner.invoke(app, ["product", "list", "--json"])
        products = parse_json_output(list_result.stdout)
        
        # There should be only one product with the sku
        matching_products = [p for p in products if p.get("sku") == original_sku]
        assert len(matching_products) == 1, f"Expected 1 product with SKU {original_sku}, found {len(matching_products)}"
        
        # Verify it has the updated values
        updated_product = matching_products[0]
        assert updated_product["name"] == updated_name
        assert float(updated_product["price"]) == float(updated_price)
        assert int(updated_product["reorder_level"]) == int(updated_reorder)
    
    def test_add_product_non_interactive(self, mock_store):
        """
        Test adding a product in non-interactive mode.
        
        This test:
        1. Uses the non-interactive flag with all required parameters
        2. Verifies the product is created correctly
        """
        name = "CLI Product"
        sku = "CLIPROD1"
        price = "29.99"
        reorder_level = "20"
        
        # Run command with all arguments and non-interactive flag
        result = runner.invoke(
            app,
            [
                "product", "add",
                "--non-interactive",
                "--name", name,
                "--sku", sku,
                "--price", price,
                "--reorder-level", reorder_level
            ]
        )
        
        # Check command succeeded
        assert result.exit_code == 0, f"Command failed with output: {result.stdout}"
        
        # Verify success message
        assert "successfully" in result.stdout
        
        # Extract ID
        product_id = extract_uuid_from_output(result.stdout)
        
        # Verify with list command
        list_result = runner.invoke(app, ["product", "list", "--json"])
        products = parse_json_output(list_result.stdout)
        
        found_product = next((p for p in products if p.get("id") == product_id), None)
        assert found_product is not None
        assert found_product["name"] == name
        assert found_product["sku"] == sku
        assert float(found_product["price"]) == float(price)
        assert int(found_product["reorder_level"]) == int(reorder_level)
    
    def test_add_product_cancel_overwrite(self, mock_store):
        """
        Test cancelling an overwrite operation when a duplicate SKU is detected.
        
        This test:
        1. Adds an initial product
        2. Tries to add another with the same SKU
        3. Declines the overwrite prompt
        4. Verifies the original product remains unchanged
        """
        # Original product details
        original_name = "Product to Keep"
        original_sku = "KEEPME"
        original_price = "10.99"
        original_reorder = "5"
        
        # Create the original product
        result1 = runner.invoke(
            app,
            ["product", "add"],
            input=f"{original_name}\n{original_sku}\n{original_price}\n{original_reorder}\n"
        )
        assert result1.exit_code == 0
        original_id = extract_uuid_from_output(result1.stdout)
        
        # Try to add a product with the same SKU but different details
        result2 = runner.invoke(
            app,
            ["product", "add"],
            input=(
                f"Replacement Product\n"  # New name
                f"{original_sku}\n"       # Same SKU
                f"15.99\n"                # New price
                f"10\n"                   # New reorder level
                "n\n"                     # Decline overwrite
            )
        )
        
        # Check that the command exited cleanly (but didn't overwrite)
        assert result2.exit_code == 0
        assert "canceled" in result2.stdout.lower()
        
        # Verify the original product is still there and unchanged
        list_result = runner.invoke(app, ["product", "list", "--json"])
        products = parse_json_output(list_result.stdout)
        
        # Find the product with our SKU
        matching_products = [p for p in products if p.get("sku") == original_sku]
        assert len(matching_products) == 1
        
        # Verify it still has the original values
        product = matching_products[0]
        assert product["id"] == original_id
        assert product["name"] == original_name
        assert float(product["price"]) == float(original_price)
        assert int(product["reorder_level"]) == int(original_reorder)
        
    @pytest.mark.parametrize("flag,expected_format", [
        ("--json", "json"),
        ("--csv", "csv"),
        ("--table", "table"),
    ])
    def test_list_products_output_formats(self, mock_store, flag, expected_format):
        """Test different output formats for the list-products command."""
        # Add a product to list
        runner.invoke(
            app,
            ["product", "add"],
            input="Test Product\nTEST123\n9.99\n5\n"
        )
        
        # List products with the specified format
        result = runner.invoke(app, ["product", "list", flag])
        
        # Check command succeeded
        assert result.exit_code == 0
        
        # Verify correct format based on output characteristics
        if expected_format == "json":
            # JSON should have square brackets for array
            assert result.stdout.strip().startswith("[")
            assert result.stdout.strip().endswith("]")
            # Should be parseable as JSON
            products = json.loads(result.stdout)
            assert isinstance(products, list)
            
        elif expected_format == "csv":
            # CSV should have header row and data rows
            lines = result.stdout.strip().split("\n")
            assert len(lines) >= 2  # Header + at least one data row
            assert "id,name,sku,price,reorder_level" in lines[0].lower().replace(" ", "")
            
        elif expected_format == "table":
            # Table should have separators and formatted data
            assert "│" in result.stdout or "|" in result.stdout  # Table borders
            assert "ID" in result.stdout
            assert "Name" in result.stdout
            assert "SKU" in result.stdout


# Additional helper test for schema validation
def test_product_schema_validation():
    """Test that the JSON schema correctly validates product data."""
    # Valid product
    valid_product = {
        "id": str(uuid.uuid4()),
        "name": "Valid Product",
        "sku": "ABC123",
        "price": 19.99,
        "reorder_level": 10
    }
    
    # Validate this product against our schema
    jsonschema.validate(valid_product, PRODUCT_SCHEMA)
    
    # Invalid product (too short name)
    invalid_product = valid_product.copy()
    invalid_product["name"] = "Ab"  # Too short
    
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(invalid_product, PRODUCT_SCHEMA)
        
    # Invalid SKU
    invalid_product = valid_product.copy()
    invalid_product["sku"] = "abc-123"  # Invalid format
    
    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(invalid_product, PRODUCT_SCHEMA)