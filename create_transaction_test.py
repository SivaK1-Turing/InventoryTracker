import pytest
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from inventorytracker.models.product import Product
from inventorytracker.models.stock_transaction import StockTransaction, LowStockEvent
from inventorytracker.factories import create_transaction, TransactionError
from inventorytracker.store.memory import MemoryStore


class EventCapture:
    """Helper class to capture emitted events"""
    def __init__(self):
        self.events = []
        
    def handler(self, event):
        self.events.append(event)
        
    def clear(self):
        self.events.clear()


@pytest.fixture
def product():
    return Product(
        id=uuid4(),
        name="Test Product",
        sku="TST001",
        price=Decimal("19.99"),
        reorder_level=5
    )


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def event_capture():
    return EventCapture()


@pytest.fixture
def setup_inventory(store, product):
    """Setup store with initial inventory for product"""
    # Initialize product with 10 units in inventory
    initial_transaction = StockTransaction(
        id=uuid4(),
        product_id=product.id,
        delta=10,
        timestamp=datetime.now(),
        note="Initial inventory"
    )
    store.save_transaction(initial_transaction)
    return initial_transaction


def test_create_transaction_positive_delta(product, store):
    """Test creating a transaction with positive delta"""
    # Arrange
    delta = 5
    note = "New stock arrived"
    
    # Act
    transaction = create_transaction(
        product_id=product.id,
        delta=delta,
        note=note,
        store=store
    )
    
    # Assert
    assert isinstance(transaction, StockTransaction)
    assert isinstance(transaction.id, UUID)
    assert transaction.product_id == product.id
    assert transaction.delta == delta
    assert transaction.note == note
    
    # Verify transaction was saved to store
    saved_transaction = store.get_transaction(transaction.id)
    assert saved_transaction is not None
    assert saved_transaction.id == transaction.id


def test_create_transaction_negative_delta(product, store, setup_inventory):
    """Test creating a transaction with negative delta"""
    # Arrange
    delta = -3
    note = "Stock sold"
    
    # Act
    transaction = create_transaction(
        product_id=product.id,
        delta=delta,
        note=note,
        store=store
    )
    
    # Assert
    assert transaction.delta == delta
    
    # Verify current stock level
    stock_level = store.get_stock_level(product.id)
    assert stock_level == 7  # 10 initial - 3 sold


def test_create_transaction_zero_delta(product, store):
    """Test creating a transaction with zero delta (should fail)"""
    # Arrange
    delta = 0
    
    # Act & Assert
    with pytest.raises(ValueError, match="Transaction delta cannot be zero"):
        create_transaction(
            product_id=product.id,
            delta=delta,
            note="Invalid transaction",
            store=store
        )


def test_create_transaction_negative_stock_error(product, store, setup_inventory):
    """Test error when transaction would cause negative stock"""
    # Arrange
    initial_stock = store.get_stock_level(product.id)
    delta = -(initial_stock + 1)  # Try to remove more than available
    
    # Act & Assert
    with pytest.raises(TransactionError, match="Insufficient stock"):
        create_transaction(
            product_id=product.id,
            delta=delta,
            note="This would cause negative stock",
            store=store
        )
    
    # Verify stock level hasn't changed
    assert store.get_stock_level(product.id) == initial_stock


def test_create_transaction_exact_zero_stock(product, store, setup_inventory):
    """Test transaction that reduces stock to exactly zero"""
    # Arrange
    initial_stock = store.get_stock_level(product.id)
    delta = -initial_stock  # Remove all stock
    
    # Act
    transaction = create_transaction(
        product_id=product.id,
        delta=delta,
        note="Removing all stock",
        store=store
    )
    
    # Assert
    assert transaction.delta == delta
    assert store.get_stock_level(product.id) == 0


def test_create_transaction_emits_low_stock_event(product, store, setup_inventory, event_capture):
    """Test that low stock event is emitted when stock goes below reorder level"""
    # Register event handler
    from inventorytracker.events import event_bus
    event_bus.subscribe(LowStockEvent, event_capture.handler)
    
    try:
        # Arrange - product has 10 in stock, reorder_level is 5
        # Create transaction that brings stock to 4 (below reorder_level)
        delta = -6
        
        # Act
        transaction = create_transaction(
            product_id=product.id,
            delta=delta,
            note="Reducing below reorder level",
            store=store
        )
        
        # Assert
        assert len(event_capture.events) == 1
        event = event_capture.events[0]
        assert isinstance(event, LowStockEvent)
        assert event.product_id == product.id
        assert event.current_level == 4
        assert event.reorder_level == 5
        
    finally:
        # Clean up
        event_bus.unsubscribe(LowStockEvent, event_capture.handler)


def test_create_transaction_no_event_above_reorder(product, store, setup_inventory, event_capture):
    """Test that no event is emitted when stock stays above reorder level"""
    # Register event handler
    from inventorytracker.events import event_bus
    event_bus.subscribe(LowStockEvent, event_capture.handler)
    
    try:
        # Arrange - product has 10 in stock, reorder_level is 5
        # Create transaction that brings stock to 6 (still above reorder_level)
        delta = -4
        
        # Act
        transaction = create_transaction(
            product_id=product.id,
            delta=delta,
            note="Still above reorder level",
            store=store
        )
        
        # Assert
        assert len(event_capture.events) == 0
        
    finally:
        # Clean up
        event_bus.unsubscribe(LowStockEvent, event_capture.handler)


def test_create_transaction_no_duplicate_events(product, store, setup_inventory, event_capture):
    """Test that low stock events aren't emitted repeatedly for the same condition"""
    # Register event handler
    from inventorytracker.events import event_bus
    event_bus.subscribe(LowStockEvent, event_capture.handler)
    
    try:
        # First transaction brings stock below reorder level
        create_transaction(
            product_id=product.id,
            delta=-6,  # 10 -> 4 (below reorder level of 5)
            note="First transaction",
            store=store
        )
        
        # Clear captured events
        event_capture.clear()
        
        # Second transaction keeps it below reorder level but doesn't cross the boundary again
        create_transaction(
            product_id=product.id,
            delta=-1,  # 4 -> 3 (still below reorder level)
            note="Second transaction",
            store=store
        )
        
        # Assert no new events
        assert len(event_capture.events) == 0
        
    finally:
        # Clean up
        event_bus.unsubscribe(LowStockEvent, event_capture.handler)


def test_create_transaction_repeated_events_after_restock(product, store, setup_inventory, event_capture):
    """Test that low stock events are emitted again after restocking and falling below again"""
    # Register event handler
    from inventorytracker.events import event_bus
    event_bus.subscribe(LowStockEvent, event_capture.handler)
    
    try:
        # First transaction brings stock below reorder level
        create_transaction(
            product_id=product.id,
            delta=-6,  # 10 -> 4 (below reorder level of 5)
            note="Going below first time",
            store=store
        )
        
        # Clear captured events
        event_capture.clear()
        
        # Restock above reorder level
        create_transaction(
            product_id=product.id,
            delta=6,  # 4 -> 10 (above reorder level)
            note="Restocking",
            store=store
        )
        
        # Create another transaction that brings it below again
        create_transaction(
            product_id=product.id,
            delta=-6,  # 10 -> 4 (below reorder level again)
            note="Going below second time",
            store=store
        )
        
        # Assert new event was emitted
        assert len(event_capture.events) == 1
        
    finally:
        # Clean up
        event_bus.unsubscribe(LowStockEvent, event_capture.handler)


def test_create_transaction_non_existent_product(store):
    """Test creating a transaction for a non-existent product"""
    # Act & Assert
    with pytest.raises(ValueError, match="Product .* not found"):
        create_transaction(
            product_id=uuid4(),
            delta=5,
            note="Non-existent product",
            store=store
        )


# Parametrized tests for various edge cases
@pytest.mark.parametrize("initial_stock,delta,expected_stock,should_emit", [
    (10, -5, 5, False),    # At reorder level exactly
    (10, -6, 4, True),     # Just below reorder level
    (5, -1, 4, True),      # From reorder level to below
    (4, -4, 0, True),      # From below reorder to zero
    (4, 2, 6, False),      # From below to above reorder
    (0, 5, 5, False),      # From zero to reorder level exactly
])
def test_create_transaction_parametrized(product, store, event_capture, 
                                         initial_stock, delta, expected_stock, should_emit):
    """Parametrized test for various stock level scenarios"""
    # Setup
    from inventorytracker.events import event_bus
    event_bus.subscribe(LowStockEvent, event_capture.handler)
    
    try:
        # Set initial stock
        if initial_stock > 0:
            initial_tx = StockTransaction(
                id=uuid4(),
                product_id=product.id,
                delta=initial_stock,
                timestamp=datetime.now(),
                note="Setting initial stock"
            )
            store.save_transaction(initial_tx)
            
        # Create the transaction
        transaction = create_transaction(
            product_id=product.id,
            delta=delta,
            note=f"Test with delta {delta}",
            store=store
        )
        
        # Verify stock level
        assert store.get_stock_level(product.id) == expected_stock
        
        # Check event emission
        if should_emit:
            assert len(event_capture.events) == 1
            assert event_capture.events[0].current_level == expected_stock
        else:
            assert len(event_capture.events) == 0
            
    finally:
        # Clean up
        event_bus.unsubscribe(LowStockEvent, event_capture.handler)