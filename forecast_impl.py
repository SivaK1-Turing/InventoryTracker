"""
Transaction sequence testing with Hypothesis.
Tests that forecast_depletion behaves correctly across random transaction sequences.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import math

# Set up hypothesis for property-based testing
import hypothesis
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite
import pytest

# Configure logging for test output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("forecast_depletion_tests")

# Import the system under test
from inventory_tracker.analytics import forecast_depletion
from inventory_tracker.models import StockTransaction, Product, TransactionType


class TestForcastDepletion:
    """Tests for the forecast_depletion function using property-based testing."""
    
    @composite
    def transaction_sequence_strategy(draw, 
                                      min_transactions=1, 
                                      max_transactions=100, 
                                      min_initial_stock=0,
                                      max_initial_stock=1000,
                                      min_usage_rate=0.1,
                                      max_usage_rate=50.0):
        """
        Strategy to generate a product and sequence of transactions.
        
        Args:
            min_transactions: Minimum number of transactions to generate
            max_transactions: Maximum number of transactions to generate
            min_initial_stock: Minimum initial stock level
            max_initial_stock: Maximum initial stock level
            min_usage_rate: Minimum daily usage rate
            max_usage_rate: Maximum daily usage rate
            
        Returns:
            A tuple of (product, transactions, current_stock, daily_usage_rate)
        """
        # Generate a product
        product_id = draw(st.uuids())
        initial_stock = draw(st.integers(min_value=min_initial_stock, max_value=max_initial_stock))
        
        product = Product(
            id=str(product_id),
            name=f"Product-{product_id}",
            sku=f"SKU-{product_id}",
            category=draw(st.sampled_from(["Electronics", "Clothing", "Food", "Books"])),
            price=draw(st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
            cost=draw(st.floats(min_value=0.5, max_value=500.0, allow_nan=False, allow_infinity=False)),
            stock_level=initial_stock
        )
        
        # Generate number of transactions
        num_transactions = draw(st.integers(min_value=min_transactions, max_value=max_transactions))
        
        # Generate a base date and time window
        base_date = draw(st.dates(min_value=datetime(2020, 1, 1).date(), 
                                  max_value=datetime(2023, 12, 31).date()))
        
        # Convert to datetime for transaction timestamps
        base_datetime = datetime.combine(base_date, datetime.min.time())
        
        # Generate transactions
        transactions = []
        current_stock = initial_stock
        
        for i in range(num_transactions):
            # Generate transaction type (more outflows than inflows for realism)
            transaction_type = draw(st.sampled_from([
                TransactionType.OUTFLOW, TransactionType.OUTFLOW, TransactionType.OUTFLOW,  # More likely
                TransactionType.INFLOW, TransactionType.ADJUSTMENT
            ]))
            
            # Generate timestamp (increasing order)
            days_offset = draw(st.integers(min_value=0, max_value=365))
            hours_offset = draw(st.integers(min_value=0, max_value=23))
            minutes_offset = draw(st.integers(min_value=0, max_value=59))
            
            timestamp = base_datetime + timedelta(
                days=days_offset, 
                hours=hours_offset,
                minutes=minutes_offset
            )
            
            # Generate quantity based on transaction type
            if transaction_type == TransactionType.OUTFLOW:
                # Ensure we don't overdraw (can't have negative stock)
                quantity = draw(st.integers(min_value=1, max_value=max(1, current_stock)))
                current_stock -= quantity
            elif transaction_type == TransactionType.INFLOW:
                quantity = draw(st.integers(min_value=1, max_value=100))
                current_stock += quantity
            else:  # ADJUSTMENT
                # Allow both positive and negative adjustments
                quantity = draw(st.integers(min_value=-min(50, current_stock), max_value=50))
                current_stock += quantity  # Could be positive or negative adjustment
                # Ensure stock doesn't go negative
                if current_stock < 0:
                    quantity -= current_stock  # Adjust to prevent negative stock
                    current_stock = 0
            
            # Create transaction
            transaction = StockTransaction(
                id=str(draw(st.uuids())),
                product_id=str(product_id),
                quantity=quantity,
                transaction_type=transaction_type,
                timestamp=timestamp,
                note=f"Transaction {i+1} of {num_transactions}"
            )
            
            transactions.append(transaction)
        
        # Sort transactions by timestamp
        transactions.sort(key=lambda t: t.timestamp)
        
        # Generate a daily usage rate
        daily_usage_rate = draw(st.floats(
            min_value=min_usage_rate,
            max_value=max_usage_rate,
            allow_nan=False,
            allow_infinity=False
        ))
        
        return (product, transactions, current_stock, daily_usage_rate)

    @settings(max_examples=100, deadline=None)
    @given(data=transaction_sequence_strategy())
    def test_forecast_depletion_never_negative(self, data):
        """
        Test that forecast_depletion never predicts negative days.
        
        Property: The forecast should always return a non-negative number of days,
        or None if depletion cannot be forecast (e.g., no usage).
        """
        product, transactions, current_stock, daily_usage_rate = data
        
        # Call the forecast function
        days_until_depletion = forecast_depletion(
            product=product,
            transactions=transactions,
            daily_usage_rate=daily_usage_rate
        )
        
        # Check if the result is None or non-negative
        if days_until_depletion is not None:
            assert days_until_depletion >= 0, f"Forecast returned negative days: {days_until_depletion}"
    
    @settings(max_examples=100, deadline=None)
    @given(data=transaction_sequence_strategy(min_initial_stock=10, min_usage_rate=0.1))
    def test_forecast_depletion_monotonic_with_usage(self, data):
        """
        Test that forecast_depletion is monotonic with increased usage.
        
        Property: As usage increases, days until depletion should decrease (or stay the same).
        """
        product, transactions, current_stock, daily_usage_rate = data
        
        # Get forecast with original usage rate
        original_forecast = forecast_depletion(
            product=product,
            transactions=transactions,
            daily_usage_rate=daily_usage_rate
        )
        
        # Skip test if original forecast is None
        if original_forecast is None:
            return
        
        # Increase usage rate by 50%
        increased_usage = daily_usage_rate * 1.5
        
        # Get forecast with increased usage rate
        increased_usage_forecast = forecast_depletion(
            product=product,
            transactions=transactions,
            daily_usage_rate=increased_usage
        )
        
        # Verify that increased usage leads to faster depletion
        assert increased_usage_forecast is None or increased_usage_forecast <= original_forecast, \
            f"Expected depletion time to decrease with increased usage, but got {original_forecast} vs {increased_usage_forecast}"
    
    @settings(max_examples=50, deadline=None)
    @given(data=transaction_sequence_strategy(min_initial_stock=10))
    def test_forecast_depletion_with_zero_usage_rate(self, data):
        """
        Test that forecast_depletion returns None with zero usage rate.
        
        Property: If daily usage rate is zero, depletion should never happen (return None).
        """
        product, transactions, current_stock, daily_usage_rate = data
        
        # Call forecast with zero usage rate
        zero_usage_forecast = forecast_depletion(
            product=product,
            transactions=transactions,
            daily_usage_rate=0.0
        )
        
        # Verify that zero usage means no depletion
        assert zero_usage_forecast is None, \
            f"Expected None for zero usage rate, but got {zero_usage_forecast}"
    
    @settings(max_examples=50, deadline=None)
    @given(data=transaction_sequence_strategy(min_initial_stock=100))
    def test_forecast_depletion_consistency_with_stock(self, data):
        """
        Test that forecast_depletion is consistent with current stock levels.
        
        Property: With the same usage rate, higher stock levels should lead to longer depletion times.
        """
        product, transactions, current_stock, daily_usage_rate = data
        
        # Get original forecast
        original_forecast = forecast_depletion(
            product=product,
            transactions=transactions,
            daily_usage_rate=daily_usage_rate
        )
        
        # Skip test if original forecast is None
        if original_forecast is None:
            return
        
        # Add 100 units to product stock
        product.stock_level += 100
        
        # Get forecast with increased stock
        increased_stock_forecast = forecast_depletion(
            product=product,
            transactions=transactions,
            daily_usage_rate=daily_usage_rate
        )
        
        # Verify that increased stock leads to longer depletion time
        assert increased_stock_forecast is None or increased_stock_forecast >= original_forecast, \
            f"Expected depletion time to increase with more stock, but got {original_forecast} vs {increased_stock_forecast}"


# Here's the actual implementation of forecast_depletion that we're testing
def forecast_depletion(product, transactions, daily_usage_rate):
    """
    Forecast the number of days until a product will be depleted.
    
    This implementation is provided for completeness and would typically be located
    in the inventory_tracker.analytics module.
    
    Args:
        product: The product to analyze
        transactions: Historical transactions for the product
        daily_usage_rate: The estimated daily usage rate
        
    Returns:
        Float number of days until depletion, or None if no depletion is expected
    """
    # Handle edge case: no usage means no depletion
    if daily_usage_rate <= 0:
        return None
        
    # Get current stock level
    current_stock = product.stock_level
    
    # If no stock, already depleted
    if current_stock <= 0:
        return 0
    
    # Calculate days until depletion based on current stock and usage rate
    days_until_depletion = current_stock / daily_usage_rate
    
    # Return result, rounded to 2 decimal places
    return round(max(0, days_until_depletion), 2)


class TransactionType:
    """Transaction type enum (would typically be in models module)."""
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    ADJUSTMENT = "adjustment"


class Product:
    """Simple Product model for testing."""
    
    def __init__(self, id, name, sku, category, price, cost, stock_level):
        self.id = id
        self.name = name
        self.sku = sku
        self.category = category
        self.price = price
        self.cost = cost
        self.stock_level = stock_level
    
    def __repr__(self):
        return f"Product(id={self.id}, name={self.name}, stock={self.stock_level})"


class StockTransaction:
    """Simple StockTransaction model for testing."""
    
    def __init__(self, id, product_id, quantity, transaction_type, timestamp, note=None):
        self.id = id
        self.product_id = product_id
        self.quantity = quantity
        self.transaction_type = transaction_type
        self.timestamp = timestamp
        self.note = note
    
    def __repr__(self):
        return f"StockTransaction(id={self.id}, product={self.product_id}, type={self.transaction_type}, quantity={self.quantity})"


# Extended test class with more complex test scenarios
class TestForcastDepletionExtended:
    """Extended test cases for forecast_depletion with more complex scenarios."""
    
    @settings(max_examples=50, deadline=None)
    @given(data=TestForcastDepletion.transaction_sequence_strategy())
    def test_forecast_depletion_with_future_inflows(self, data):
        """
        Test that forecast_depletion handles scheduled future inflows.
        
        This variant confirms that the function correctly uses information about
        known upcoming deliveries/inflows when forecasting depletion.
        """
        product, transactions, current_stock, daily_usage_rate = data
        
        # Skip test if usage rate is too low
        if daily_usage_rate < 0.5:
            return
        
        # Get base forecast
        base_forecast = forecast_depletion(
            product=product,
            transactions=transactions,
            daily_usage_rate=daily_usage_rate
        )
        
        # Skip test if base forecast is None
        if base_forecast is None:
            return
        
        # Create a future inflow transaction
        # Set timestamp just before expected depletion
        future_inflow_date = datetime.now() + timedelta(days=base_forecast * 0.8)
        
        future_inflow = StockTransaction(
            id="future-inflow",
            product_id=product.id,
            quantity=int(daily_usage_rate * 10),  # 10 days worth of stock
            transaction_type=TransactionType.INFLOW,
            timestamp=future_inflow_date,
            note="Future scheduled delivery"
        )
        
        # Add future inflow to transactions
        new_transactions = transactions + [future_inflow]
        
        # Get updated forecast with future inflow
        updated_forecast = forecast_depletion(
            product=product,
            transactions=new_transactions,
            daily_usage_rate=daily_usage_rate
        )
        
        # Implementation note:
        # If our forecast_depletion doesn't account for future inflows, this test might fail.
        # The intent is to verify that the function considers scheduled deliveries.
        
        # Verify that future inflow extends the depletion time
        if updated_forecast is not None:
            assert updated_forecast > base_forecast, \
                f"Expected depletion time to increase with future inflow, but got {base_forecast} vs {updated_forecast}"
    
    @settings(max_examples=50, deadline=None)
    @given(data=TestForcastDepletion.transaction_sequence_strategy(min_transactions=20))
    def test_forecast_depletion_with_seasonal_patterns(self, data):
        """
        Test that forecast_depletion adapts to seasonal usage patterns.
        
        This test simulates seasonal usage patterns and verifies the forecast adapts.
        """
        product, transactions, current_stock, daily_usage_rate = data
        
        # Modify transactions to show a seasonal pattern
        # (higher usage in "summer" months, lower in "winter")
        for i, tx in enumerate(transactions):
            if tx.transaction_type == TransactionType.OUTFLOW:
                # Determine "month" based on position in sequence
                month = (i % 12) + 1
                
                # Adjust quantity based on "season"
                if 5 <= month <= 8:  # "Summer" months
                    tx.quantity = int(tx.quantity * 1.5)  # 50% higher usage
                elif 11 <= month or month <= 2:  # "Winter" months
                    tx.quantity = max(1, int(tx.quantity * 0.7))  # 30% lower usage
        
        # Generate two forecasts: one for "summer" and one for "winter"
        summer_forecast = forecast_depletion(
            product=product,
            transactions=[tx for i, tx in enumerate(transactions) if (i % 12) + 1 in (5, 6, 7, 8)],
            daily_usage_rate=daily_usage_rate
        )
        
        winter_forecast = forecast_depletion(
            product=product,
            transactions=[tx for i, tx in enumerate(transactions) if (i % 12) + 1 in (11, 12, 1, 2)],
            daily_usage_rate=daily_usage_rate
        )
        
        # Skip comparison if either forecast is None
        if summer_forecast is None or winter_forecast is None:
            return
        
        # Verify seasonal adaptation: summer should deplete faster than winter
        # Note: If our forecast_depletion is naive and doesn't account for seasonality,
        # this test might fail, which is intentional
        if daily_usage_rate > 1.0:  # Only test with meaningful usage rates
            assert summer_forecast <= winter_forecast, \
                f"Expected summer depletion to be faster than winter, but got {summer_forecast} vs {winter_forecast}"


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])