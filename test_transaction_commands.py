# tests/feature3/test_transaction_commands.py

import json
import os
import subprocess
import tempfile
import uuid
from decimal import Decimal
from pathlib import Path

import pytest
from typing import Dict, Any, List


class TestTransactionCommands:
    """Integration tests for transaction CLI commands."""

    @pytest.fixture
    def isolated_cli_environment(self):
        """Fixture to provide an isolated environment for CLI testing."""
        # Create a temporary directory for test data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up environment variables to use this directory for storage
            env = os.environ.copy()
            env["INVTRACK_DATA_DIR"] = temp_dir
            
            # Return the environment and temp dir path for use in the tests
            yield {"env": env, "data_dir": Path(temp_dir)}

    @pytest.fixture
    def sample_product(self, isolated_cli_environment) -> Dict[str, Any]:
        """Create a sample product for testing."""
        env = isolated_cli_environment["env"]
        
        # Generate a unique SKU to avoid conflicts
        sku = f"TEST{uuid.uuid4().hex[:6].upper()}"
        
        # Add a product using the CLI
        subprocess.run([
            "python", "-m", "inventorytracker", "add-product",
            "--non-interactive",
            "--name", "Test Transaction Product",
            "--sku", sku,
            "--price", "29.99",
            "--reorder-level", "5"
        ], env=env, check=True)
        
        # Get the product details including the ID
        result = subprocess.run([
            "python", "-m", "inventorytracker", "list-products",
            "--json"
        ], env=env, capture_output=True, text=True, check=True)
        
        products = json.loads(result.stdout)
        product = next(p for p in products if p["sku"] == sku)
        return product

    def run_command(self, command: List[str], env: Dict[str, str]) -> subprocess.CompletedProcess:
        """Helper method to run a command with the given environment."""
        return subprocess.run(
            ["python", "-m", "inventorytracker"] + command,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )

    def get_product_json(self, product_id: str, env: Dict[str, str]) -> Dict[str, Any]:
        """Get product details in JSON format using show-product command."""
        result = self.run_command(["show-product", "--json", product_id], env)
        return json.loads(result.stdout)

    def test_add_stock_transaction(self, isolated_cli_environment, sample_product):
        """Test adding stock increases the stock level correctly."""
        env = isolated_cli_environment["env"]
        product_id = sample_product["id"]
        
        # Verify initial stock level should be 0
        product_before = self.get_product_json(product_id, env)
        assert product_before["stock_level"] == 0
        
        # Add stock with a positive transaction
        add_qty = 10
        self.run_command([
            "add-transaction",
            "--product-id", product_id,
            "--quantity", str(add_qty),
            "--note", "Initial stock"
        ], env)
        
        # Verify stock level increased
        product_after = self.get_product_json(product_id, env)
        assert product_after["stock_level"] == add_qty
        
        # Verify transaction appears in transaction history
        result = self.run_command(["list-transactions", "--json", "--product-id", product_id], env)
        transactions = json.loads(result.stdout)
        
        # Find our transaction
        assert len(transactions) == 1
        assert transactions[0]["delta"] == add_qty
        assert transactions[0]["note"] == "Initial stock"

    def test_remove_stock_transaction(self, isolated_cli_environment, sample_product):
        """Test removing stock decreases the stock level correctly."""
        env = isolated_cli_environment["env"]
        product_id = sample_product["id"]
        
        # First add some stock to work with
        initial_qty = 20
        self.run_command([
            "add-transaction",
            "--product-id", product_id,
            "--quantity", str(initial_qty),
            "--note", "Initial stock"
        ], env)
        
        # Remove some stock
        remove_qty = 8
        self.run_command([
            "remove-transaction",
            "--product-id", product_id,
            "--quantity", str(remove_qty),
            "--note", "Sold items"
        ], env)
        
        # Verify stock level decreased correctly
        product_after = self.get_product_json(product_id, env)
        assert product_after["stock_level"] == initial_qty - remove_qty
        
        # Verify both transactions appear in history
        result = self.run_command(["list-transactions", "--json", "--product-id", product_id], env)
        transactions = json.loads(result.stdout)
        
        assert len(transactions) == 2
        # Transactions are typically returned in chronological order, newest first
        assert transactions[1]["delta"] == initial_qty
        assert transactions[0]["delta"] == -remove_qty
        assert transactions[0]["note"] == "Sold items"

    def test_insufficient_stock_error(self, isolated_cli_environment, sample_product):
        """Test that removing more stock than available raises an error."""
        env = isolated_cli_environment["env"]
        product_id = sample_product["id"]
        
        # Add some initial stock
        initial_qty = 5
        self.run_command([
            "add-transaction",
            "--product-id", product_id,
            "--quantity", str(initial_qty),
            "--note", "Initial stock"
        ], env)
        
        # Try to remove more stock than available
        remove_qty = initial_qty + 1
        
        # This should fail with a non-zero exit code
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            self.run_command([
                "remove-transaction",
                "--product-id", product_id,
                "--quantity", str(remove_qty),
                "--note", "This should fail"
            ], env)
        
        # Verify error message mentions insufficient stock
        assert "Insufficient stock" in exc_info.value.stderr
        
        # Verify stock level remains unchanged
        product_after = self.get_product_json(product_id, env)
        assert product_after["stock_level"] == initial_qty

    def test_multiple_transactions(self, isolated_cli_environment, sample_product):
        """Test multiple transactions with various quantities."""
        env = isolated_cli_environment["env"]
        product_id = sample_product["id"]
        
        # Series of transactions to be applied
        transactions = [
            {"type": "add", "qty": 10, "note": "Initial stock"},
            {"type": "remove", "qty": 3, "note": "First sale"},
            {"type": "add", "qty": 5, "note": "Restocking"},
            {"type": "remove", "qty": 7, "note": "Big order"}
        ]
        
        expected_stock = 0
        
        # Apply all transactions
        for tx in transactions:
            cmd = "add-transaction" if tx["type"] == "add" else "remove-transaction"
            qty = tx["qty"]
            
            self.run_command([
                cmd,
                "--product-id", product_id,
                "--quantity", str(qty),
                "--note", tx["note"]
            ], env)
            
            # Update expected stock level
            expected_stock += qty if tx["type"] == "add" else -qty
            
            # Verify current stock level
            product_after = self.get_product_json(product_id, env)
            assert product_after["stock_level"] == expected_stock
        
        # Verify all transactions appear in history
        result = self.run_command(["list-transactions", "--json", "--product-id", product_id], env)
        tx_history = json.loads(result.stdout)
        
        assert len(tx_history) == len(transactions)
        
        # Verify transaction details (ordered by most recent first)
        for i, expected_tx in enumerate(reversed(transactions)):
            tx = tx_history[i]
            expected_delta = expected_tx["qty"] if expected_tx["type"] == "add" else -expected_tx["qty"]
            assert tx["delta"] == expected_delta
            assert tx["note"] == expected_tx["note"]

    def test_transaction_csv_export(self, isolated_cli_environment, sample_product):
        """Test exporting transactions to CSV and verifying the results."""
        env = isolated_cli_environment["env"]
        product_id = sample_product["id"]
        
        # Add some transactions
        self.run_command([
            "add-transaction", 
            "--product-id", product_id,
            "--quantity", "15",
            "--note", "Initial stock"
        ], env)
        
        self.run_command([
            "remove-transaction",
            "--product-id", product_id,
            "--quantity", "5",
            "--note", "First sale"
        ], env)
        
        # Export transactions to CSV
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            csv_path = tmp_file.name
        
        try:
            self.run_command([
                "export-transactions",
                "--format", "csv",
                "--output", csv_path,
                "--product-id", product_id
            ], env)
            
            # Read the CSV file and verify its contents
            with open(csv_path, 'r') as f:
                csv_lines = f.readlines()
            
            # Check header line
            assert "id,product_id,delta,timestamp,note" in csv_lines[0]
            
            # Check transaction data (skip header)
            data_lines = csv_lines[1:]
            assert len(data_lines) == 2
            
            # Verify second transaction (first line after header)
            assert "-5" in data_lines[0]
            assert "First sale" in data_lines[0]
            
            # Verify first transaction
            assert "15" in data_lines[1]
            assert "Initial stock" in data_lines[1]
        finally:
            # Clean up temp file
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_low_stock_warning(self, isolated_cli_environment, sample_product):
        """Test that low stock warnings are generated correctly."""
        env = isolated_cli_environment["env"]
        product_id = sample_product["id"]
        reorder_level = 5  # Set in sample_product fixture
        
        # Add stock above the reorder level
        self.run_command([
            "add-transaction",
            "--product-id", product_id,
            "--quantity", "10",
            "--note", "Initial stock"
        ], env)
        
        # Verify no warnings initially
        result = self.run_command(["show-product", product_id], env)
        assert "Low stock warning" not in result.stdout
        
        # Remove stock to go below reorder level
        self.run_command([
            "remove-transaction",
            "--product-id", product_id,
            "--quantity", "6",  # 10 - 6 = 4, which is below reorder level of 5
            "--note", "Reducing below reorder level"
        ], env)
        
        # Check for low stock warning
        result = self.run_command(["show-product", product_id], env)
        assert "Low stock warning" in result.stdout
        
        # Verify stock level
        product_json = self.get_product_json(product_id, env)
        assert product_json["stock_level"] == 4
        assert product_json["stock_level"] < reorder_level
        assert product_json.get("low_stock", False) is True

    def test_adjust_inventory_command(self, isolated_cli_environment, sample_product):
        """Test the adjust-inventory command that synchronizes stock levels."""
        env = isolated_cli_environment["env"]
        product_id = sample_product["id"]
        
        # First add some stock
        self.run_command([
            "add-transaction",
            "--product-id", product_id,
            "--quantity", "8",
            "--note", "Initial stock"
        ], env)
        
        # Now adjust inventory to a specific value
        new_quantity = 12
        result = self.run_command([
            "adjust-inventory",
            "--product-id", product_id,
            "--actual-quantity", str(new_quantity),
            "--note", "Inventory count adjustment"
        ], env)
        
        # Verify success message
        assert "Adjusted inventory" in result.stdout
        assert str(new_quantity) in result.stdout
        
        # Verify stock level updated
        product_json = self.get_product_json(product_id, env)
        assert product_json["stock_level"] == new_quantity
        
        # Verify an adjustment transaction was created
        result = self.run_command(["list-transactions", "--json", "--product-id", product_id], env)
        transactions = json.loads(result.stdout)
        
        # Should have 2 transactions - original addition and adjustment
        assert len(transactions) == 2
        
        # Newest transaction should be the adjustment
        latest_tx = transactions[0]
        assert latest_tx["delta"] == 4  # Adjust from 8 to 12
        assert "Inventory count adjustment" in latest_tx["note"]

    def test_batch_transaction_processing(self, isolated_cli_environment, sample_product):
        """Test processing multiple transactions from a file."""
        env = isolated_cli_environment["env"]
        product_id = sample_product["id"]
        
        # Create a batch file with multiple transactions
        batch_data = [
            {"product_id": product_id, "delta": 20, "note": "Initial batch stock"},
            {"product_id": product_id, "delta": -5, "note": "First batch removal"},
            {"product_id": product_id, "delta": 7, "note": "Batch restocking"},
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as batch_file:
            json.dump(batch_data, batch_file)
            batch_path = batch_file.name
        
        try:
            # Process the batch file
            result = self.run_command([
                "process-transaction-batch",
                "--file", batch_path
            ], env)
            
            # Verify success messages
            assert "Processed 3 transactions" in result.stdout
            
            # Verify final stock level (20 - 5 + 7 = 22)
            product_json = self.get_product_json(product_id, env)
            assert product_json["stock_level"] == 22
            
            # Check transaction history
            result = self.run_command(["list-transactions", "--json", "--product-id", product_id], env)
            transactions = json.loads(result.stdout)
            
            assert len(transactions) == 3
            
            # Check all transactions were recorded properly
            for i, expected_tx in enumerate(reversed(batch_data)):
                assert transactions[i]["product_id"] == expected_tx["product_id"]
                assert transactions[i]["delta"] == expected_tx["delta"]
                assert transactions[i]["note"] == expected_tx["note"]
                
        finally:
            # Clean up temp file
            if os.path.exists(batch_path):
                os.unlink(batch_path)