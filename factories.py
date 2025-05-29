def create_transaction(product_id: UUID, delta: int, note: Optional[str] = None, tx=None) -> StockTransaction:
    """
    Create a stock transaction with validation.
    
    Args:
        product_id: The UUID of the product
        delta: The quantity change (positive for additions, negative for removals)
        note: Optional note about the transaction
        tx: Optional transaction context for atomic operations
        
    Returns:
        StockTransaction: The created transaction
        
    Raises:
        ValueError: If the transaction would make inventory negative
        ConcurrencyError: If concurrent modification is detected
    """
    from .models.stock_transaction import StockTransaction
    from .store import get_store
    
    store = get_store()
    
    # If we're inside a transaction context, use it, otherwise create a new one
    in_existing_tx = tx is not None
    tx = tx or store.atomic_transaction()
    
    try:
        if not in_existing_tx:
            tx.__enter__()
            
        # Get current product and stock level with lock
        product = store.get_product(product_id, for_update=True)
        if not product:
            raise ValueError(f"Product with id {product_id} does not exist")
        
        current_inventory = store.get_inventory_level(product_id, for_update=True)
        
        # Validate inventory won't go negative - do this check again inside the transaction
        if delta < 0 and (current_inventory + delta < 0):
            raise ValueError(f"Cannot remove {abs(delta)} units; only {current_inventory} in stock")
        
        # Create and save the transaction
        transaction = StockTransaction(
            product_id=product_id,
            delta=delta,
            note=note
        )
        
        # Save the transaction record
        store.save_transaction(transaction)
        
        # Update inventory level
        new_level = current_inventory + delta
        store.update_inventory_level(product_id, new_level)
        
        # Check if inventory is below reorder level and emit event if needed
        if new_level <= product.reorder_level:
            event_bus.emit(LowStockEvent(product_id=product_id, current_level=new_level, reorder_level=product.reorder_level))
        
        # If we created our own transaction, commit it
        if not in_existing_tx:
            tx.__exit__(None, None, None)
            
        return transaction
    
    except Exception as e:
        # If we created our own transaction, roll it back on error
        if not in_existing_tx:
            tx.__exit__(type(e), e, e.__traceback__)
        # Re-raise the exception
        raise