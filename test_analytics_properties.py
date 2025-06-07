# test_analytics_properties.py

import hypothesis
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
from datetime import datetime, timedelta
import uuid
import math
from typing import List, Dict, Optional

# Import our application code
from cache_implementation import analytics_cache
import analytics
from analytics import forecast_depletion

# Mock database and transaction manager for testing
class MockDB:
    def __init__(self):
        self.inventory = {}
        self.events = []
    
    def get_connection(self):
        return self
    
    def cursor(self):
        return self
    
    def execute(self, query, params=None):
        self.last_query = query
        self.last_params = params
        
        # For inventory query
        if "FROM inventory" in query and "product_id = ?" in query:
            product_id = params[0] if params else None
            if product_id in self.inventory:
                self.result = [(self.inventory[product_id]['current_stock'],)]
            else:
                self.result = []
        
        # For inventory_events query (usage)
        elif "FROM inventory_events" in query and "event_type = 'USAGE'" in query:
            product_id = params[0] if params else None
            start_date = params[1] if len(params) > 1 else None
            end_date = params[2] if len(params) > 2 else None
            
            # Filter events by product, type and date range
            filtered_events = []
            for event in self.events:
                if (event['product_id'] == product_id and 
                    event['event_type'] == 'USAGE' and
                    (start_date is None or event['timestamp'].split('T')[0] >= start_date) and
                    (end_date is None or event['timestamp'].split('T')[0] <= end_date)):
                    filtered_events.append(event)
            
            # Group by date
            date_changes = {}
            for event in filtered_events:
                date_str = event['timestamp'].split('T')[0]
                if date_str not in date_changes:
                    date_changes[date_str] = 0
                date_changes[date_str] += event['quantity_change']
            
            self.result = [(date, change) for date, change in date_changes.items()]
        
        # For product queries
        elif "FROM products" in query:
            self.result = [(k,) for k in self.inventory.keys()]
        
        else:
            self.result = []
    
    def fetchall(self):
        return self.result if hasattr(self, 'result') else []
    
    def fetchone(self):
        result = self.fetchall()
        return result[0] if result else None
    
    def close(self):
        pass
    
    # Methods to populate test data
    def set_inventory(self, product_id: str, current_stock: int):
        self.inventory[product_id] = {
            'product_id': product_id,
            'current_stock': current_stock
        }
    
    def add_event(self, product_id: str, event_type: str, quantity_change: int, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.events.append({
            'product_id': product_id,
            'event_type': event_type,
            'quantity_change': quantity_change,
            'timestamp': timestamp
        })


class MockTransaction:
    def __init__(self, id: str, product_id: str, data=None):
        self.id = id
        self.product_id = product_id
        self.data = data or {}
        self.timestamp = datetime.now().isoformat()


class MockTransactionManager:
    def __init__(self):
        self.callbacks = {}
    
    def on(self, event_name):
        def decorator(func):
            if event_name not in self.callbacks:
                self.callbacks[event_name] = []
            self.callbacks[event_name].append(func)
            return func
        return decorator
    
    def emit(self, event_name, transaction):
        if event_name in self.callbacks:
            for callback in self.callbacks[event_name]:
                callback(transaction)


# Initialize test environment
db = MockDB()
tx_manager = MockTransactionManager()

# Monkey patch the database module
analytics.database = db

# Register transaction hooks
analytics.register_transaction_hooks(tx_manager)


# Define Hypothesis strategies for generating test data

# Strategy for generating random dates within the past 60 days
@st.composite
def past_dates(draw):
    days_ago = draw(st.integers(min_value=0, max_value=60))
    return (datetime.now() - timedelta(days=days_ago)).date()

# Strategy for generating valid product IDs
product_id_strategy = st.uuids().map(str)

# Strategy for generating valid stock quantities
stock_quantity_strategy = st.integers(min_value=0, max_value=1000)

# Strategy for generating usage events (always negative quantities)
usage_change_strategy = st.integers(min_value=-50, max_value=-1)

# Strategy for generating a sequence of usage events
@st.composite
def usage_events_sequence(draw, min_events=1, max_events=30):
    product_id = draw(product_id_strategy)
    num_events = draw(st.integers(min_value=min_events, max_value=max_events))
    
    events = []
    for _ in range(num_events):
        quantity = draw(usage_change_strategy)
        date = draw(past_dates())
        events.append({
            "product_id": product_id,
            "quantity": quantity,
            "date": date
        })
    
    return {
        "product_id": product_id,
        "events": events
    }

# Tests for forecast_depletion function

@given(
    product_id=product_id_strategy,
    initial_stock=stock_quantity_strategy,
    events_data=usage_events_sequence(min_events=5, max_events=20)
)
@settings(max_examples=50, deadline=None)  # Allow longer test times
def test_forecast_depletion_is_positive_or_none(product_id, initial_stock, events_data):
    """
    Test that forecast_depletion either returns None or a positive number of days.
    """
    # Clear cache between test runs
    analytics_cache.clear()
    
    # Set up initial inventory
    db.inventory = {}  # Clear previous inventory
    db.events = []     # Clear previous events
    db.set_inventory(product_id, initial_stock)
    
    # Add usage events
    for event in events_data["events"]:
        db.add_event(
            product_id=product_id,
            event_type="USAGE",
            quantity_change=event["quantity"],
            timestamp=event["date"].isoformat()
        )
    
    # Call the forecast function
    forecast = analytics.forecast_depletion(product_id)
    
    # Check the result is either None or positive
    if forecast['days_until_depletion'] is not None:
        assert forecast['days_until_depletion'] >= 0, "Days until depletion should be positive or None"
    
    # Check that current_stock matches what we set
    assert forecast['current_stock'] == initial_stock


@given(
    product_id=product_id_strategy, 
    initial_stock=stock_quantity_strategy,
    base_events=usage_events_sequence(min_events=5, max_events=10),
    additional_usage=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=50, deadline=None)
def test_increased_usage_decreases_depletion_time(
    product_id, initial_stock, base_events, additional_usage
):
    """
    Test that increasing usage decreases the days until depletion (or keeps it None).
    """
    # Ensure we have enough stock to detect changes
    assume(initial_stock >= 50)
    
    # Clear cache between test runs
    analytics_cache.clear()
    
    # Set up environment for first forecast
    db.inventory = {}  # Clear previous inventory
    db.events = []     # Clear previous events
    db.set_inventory(product_id, initial_stock)
    
    # Add base usage events
    for event in base_events["events"]:
        db.add_event(
            product_id=product_id,
            event_type="USAGE",
            quantity_change=event["quantity"],
            timestamp=event["date"].isoformat()
        )
    
    # Get first forecast
    forecast1 = analytics.forecast_depletion(product_id)
    
    # Now add additional usage events with higher usage rates
    for event in base_events["events"]:
        # Create a similar event but with higher usage (more negative)
        db.add_event(
            product_id=product_id,
            event_type="USAGE",
            quantity_change=event["quantity"] - additional_usage,  # More negative
            timestamp=event["date"].isoformat()
        )
    
    # Clear cache to force recalculation
    analytics_cache.clear()
    
    # Get second forecast after increased usage
    forecast2 = analytics.forecast_depletion(product_id)
    
    # Check the monotonicity property:
    if forecast1['days_until_depletion'] is not None and forecast2['days_until_depletion'] is not None:
        # More usage should deplete stock faster
        assert forecast2['days_until_depletion'] <= forecast1['days_until_depletion'], (
            f"Expected days until depletion to decrease or stay the same with "
            f"increased usage, but got {forecast1['days_until_depletion']} -> "
            f"{forecast2['days_until_depletion']}"
        )
    
    # If first forecast was None, second shouldn't suddenly have a value
    if forecast1['days_until_depletion'] is None:
        assert forecast2['days_until_depletion'] is None, (
            "If initial forecast returned None, increased usage shouldn't give a value"
        )


# Define a state machine to test sequences of inventory transactions
class InventoryTransactionMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.db = MockDB()
        self.tx_manager = MockTransactionManager()
        analytics.database = self.db
        analytics.register_transaction_hooks(self.tx_manager)
        
        # Generate a consistent product ID for this test run
        self.product_id = str(uuid.uuid4())
        self.current_stock = 100
        self.db.set_inventory(self.product_id, self.current_stock)
        
        # Track usage events for validation
        self.usage_events = []
        
        # Track forecasts for comparison
        self.last_forecast = None
    
    @rule(quantity=st.integers(min_value=1, max_value=20))
    def add_usage_event(self, quantity):
        """Add usage event (negative quantity change)."""
        timestamp = datetime.now().isoformat()
        
        self.db.add_event(
            product_id=self.product_id,
            event_type="USAGE",
            quantity_change=-quantity,  # Negative for usage
            timestamp=timestamp
        )
        
        # Update current stock
        self.current_stock -= quantity
        if self.current_stock < 0:
            self.current_stock = 0
        self.db.set_inventory(self.product_id, self.current_stock)
        
        # Track for validation
        self.usage_events.append({
            'quantity': quantity,
            'timestamp': timestamp
        })
        
        # Create and commit transaction to trigger cache invalidation
        tx = MockTransaction(
            id=str(uuid.uuid4()),
            product_id=self.product_id,
            data={'operation': 'usage', 'quantity': -quantity}
        )
        self.tx_manager.emit('commit', tx)
    
    @rule(quantity=st.integers(min_value=1, max_value=50))
    def add_restock_event(self, quantity):
        """Add restock event (positive quantity change)."""
        timestamp = datetime.now().isoformat()
        
        self.db.add_event(
            product_id=self.product_id,
            event_type="RESTOCK",
            quantity_change=quantity,  # Positive for restock
            timestamp=timestamp
        )
        
        # Update current stock
        self.current_stock += quantity
        self.db.set_inventory(self.product_id, self.current_stock)
        
        # Create and commit transaction to trigger cache invalidation
        tx = MockTransaction(
            id=str(uuid.uuid4()),
            product_id=self.product_id,
            data={'operation': 'restock', 'quantity': quantity}
        )
        self.tx_manager.emit('commit', tx)
    
    @rule()
    def check_forecast_depletion(self):
        """Check forecast_depletion after transactions."""
        forecast = analytics.forecast_depletion(self.product_id)
        
        # Validate forecast
        assert forecast['current_stock'] == self.current_stock
        
        if forecast['days_until_depletion'] is not None:
            assert forecast['days_until_depletion'] >= 0
        
        # Store for comparison in invariants
        self.last_forecast = forecast
    
    @invariant()
    def validate_forecast(self):
        """Validate that forecast makes sense given the current state."""
        if not self.usage_events:
            return  # No events yet, nothing to validate
            
        if self.last_forecast is None:
            return  # No forecast has been made yet
            
        # Check if we have any usage rate
        if self.last_forecast['daily_usage_rate'] == 0:
            # If no usage rate, days_until_depletion should be None
            assert self.last_forecast['days_until_depletion'] is None
        else:
            # If we have stock and usage rate, days_until_depletion should be calculated
            if self.current_stock > 0 and self.last_forecast['daily_usage_rate'] > 0:
                expected_days = self.current_stock / self.last_forecast['daily_usage_rate']
                # Allow some floating-point tolerance
                assert math.isclose(
                    self.last_forecast['days_until_depletion'], 
                    expected_days,
                    rel_tol=0.01  # 1% tolerance for floating point calculations
                )


# Run the state machine with Hypothesis
TestInventoryTransactions = InventoryTransactionMachine.TestCase


if __name__ == "__main__":
    # Run individual tests
    test_forecast_depletion_is_positive_or_none()
    test_increased_usage_decreases_depletion_time()
    
    # Run the state machine
    unittest.main()