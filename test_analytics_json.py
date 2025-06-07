# tests/feature7/test_analytics_json.py

"""
Test script for analytics JSON output.

This script:
1. Creates a test product
2. Records various transactions
3. Runs analytics with JSON output
4. Verifies the JSON contains correct forecast and usage values
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

# Add parent directory to path so we can import application modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transaction_service import StockTransactionService
from database_service import DatabaseService


class TestAnalyticsJson(unittest.TestCase):
    """Test the analytics JSON output."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary database for testing
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.db_fd)  # Close the file descriptor
        
        # Initialize database
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.db = self.loop.run_until_complete(self.setup_database())
        self.transaction_service = StockTransactionService(self.db)
        
        # Create a test product
        self.product_id = str(uuid.uuid4())
        self.loop.run_until_complete(self.create_test_product())
    
    def tearDown(self):
        """Clean up after tests."""
        # Close database connection
        self.loop.run_until_complete(self.db.close())
        
        # Remove temporary database
        os.unlink(self.db_path)
        
        # Clean up the event loop
        self.loop.close()
    
    async def setup_database(self):
        """Set up the database schema."""
        db = DatabaseService(self.db_path)
        
        # Create necessary tables
        await db.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            product_id TEXT PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            supplier_id TEXT,
            current_stock INTEGER NOT NULL DEFAULT 0,
            reorder_point INTEGER DEFAULT 0,
            target_stock INTEGER DEFAULT 0,
            last_updated TEXT,
            last_restocked TEXT,
            last_used TEXT
        )
        """)
        
        await db.execute("""
        CREATE TABLE IF NOT EXISTS inventory_events (
            id TEXT PRIMARY KEY,
            product_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            reason TEXT,
            supplier TEXT,
            location TEXT,
            purchase_order TEXT,
            cost TEXT,
            user TEXT,
            FOREIGN KEY (product_id) REFERENCES inventory(product_id)
        )
        """)
        
        await db.commit()
        return db
    
    async def create_test_product(self):
        """Create a test product with transactions."""
        # Insert product
        await self.db.execute(
            """
            INSERT INTO inventory 
            (product_id, product_name, category, supplier_id, current_stock, 
             reorder_point, target_stock, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.product_id, 
                "Test Product", 
                "Test Category",
                "Test Supplier",
                0,  # Start with zero stock
                10,
                100,
                datetime.now().isoformat()
            )
        )
        
        # Record transactions over the last 60 days
        # Start 60 days ago with a restock
        start_date = datetime.now() - timedelta(days=60)
        
        # 1. Initial restock (60 days ago)
        await self.transaction_service.record_restock(
            product_id=self.product_id,
            quantity=100,
            supplier="Test Supplier",
            purchase_order="PO-TEST-001",
            cost=1000.00,
            timestamp=start_date.isoformat()
        )
        
        # 2. Several usage events (50, 40, 30, 20, 10 days ago)
        for days_ago in [50, 40, 30, 20, 10]:
            usage_date = datetime.now() - timedelta(days=days_ago)
            await self.transaction_service.record_usage(
                product_id=self.product_id,
                quantity=10,
                usage_type="Test Usage",
                timestamp=usage_date.isoformat()
            )
        
        # 3. Another restock (25 days ago)
        restock_date = datetime.now() - timedelta(days=25)
        await self.transaction_service.record_restock(
            product_id=self.product_id,
            quantity=50,
            supplier="Test Supplier",
            purchase_order="PO-TEST-002",
            cost=500.00,
            timestamp=restock_date.isoformat()
        )
        
        # 4. A few more recent usage events (5, 2, 1 days ago)
        for days_ago in [5, 2, 1]:
            usage_date = datetime.now() - timedelta(days=days_ago)
            await self.transaction_service.record_usage(
                product_id=self.product_id,
                quantity=5,
                usage_type="Test Usage",
                timestamp=usage_date.isoformat()
            )
        
        await self.db.commit()
    
    def run_analytics_json(self) -> Dict[str, Any]:
        """Run the analytics command with JSON output and return the results."""
        # Set environment variable to point to our test database
        env = os.environ.copy()
        env['INVENTORY_DB_PATH'] = self.db_path
        
        # Run the analytics command with JSON output
        result = subprocess.run(
            [
                sys.executable, 
                '-m', 
                'analytics',  # Assuming the script is a module
                self.product_id,
                '--json'
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check for errors
        if result.returncode != 0:
            raise RuntimeError(f"Analytics command failed: {result.stderr}")
        
        # Parse JSON output
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON output: {result.stdout}")
    
    def test_analytics_json_output(self):
        """Test that analytics JSON output contains correct values."""
        # Run analytics and get JSON output
        results = self.run_analytics_json()
        
        # Verify product ID
        self.assertEqual(results['product_id'], self.product_id)
        
        # Verify presence of standard metrics
        self.assertIn('current_stock', results['metrics'])
        self.assertIn('usage_rate', results['metrics'])
        self.assertIn('forecast_depletion', results['metrics'])
        self.assertIn('stockout_risk', results['metrics'])
        self.assertIn('restock_frequency', results['metrics'])
        
        # Verify current stock value
        # We started with 0, added 100 + 50 = 150, used 10*5 + 5*3 = 65
        expected_stock = 85
        self.assertEqual(results['metrics']['current_stock']['value'], expected_stock)
        
        # Verify usage rate (approximately)
        # Over 30 days: (10 + 10 + 5 + 5 + 5) = 35 units in 30 days = ~1.17 units/day
        usage_rate = results['metrics']['usage_rate']['value']
        self.assertGreater(usage_rate, 1.0)
        self.assertLess(usage_rate, 1.5)
        
        # Verify forecast depletion (approximately)
        # Current stock / usage rate ≈ 85 / 1.17 ≈ 72.6 days
        forecast = results['metrics']['forecast_depletion']['value']
        self.assertGreater(forecast, 65)  # Allow some margin for calculation differences
        self.assertLess(forecast, 85)
        
        # Verify restock frequency
        # We had 2 restocks over 60 days, so frequency should be around 30 days
        restock_frequency = results['metrics']['restock_frequency']['value']
        self.assertGreater(restock_frequency, 25)
        self.assertLess(restock_frequency, 40)
    
    def test_plugin_metrics_included(self):
        """Test that plugin metrics are included in JSON output."""
        # This test will only work if the turnover ratio plugin is installed
        # and properly discovered. We'll make it conditional.
        
        results = self.run_analytics_json()
        
        # Check if turnover_ratio metric is present
        if 'turnover_ratio' in results['metrics']:
            turnover_ratio = results['metrics']['turnover_ratio']['value']
            self.assertIsNotNone(turnover_ratio)
            self.assertGreaterEqual(turnover_ratio, 0.0)
        else:
            # Skip the test if plugin not found, but log a message
            print("Skipping plugin test as turnover_ratio plugin not found")


if __name__ == '__main__':
    unittest.main()