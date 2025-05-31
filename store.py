# Concurrency error class
class ConcurrencyError(Exception):
    """Raised when a concurrent modification is detected"""
    pass

class Store:
    # Add versioning to our store
    def __init__(self):
        # ... existing initialization ...
        self._versions = {}  # Track version numbers for optimistic concurrency
        self._locks = {}     # Resource locks for pessimistic concurrency
    
    def get_product(self, product_id: UUID, for_update=False):
        """
        Get product by ID with optional locking for updates.
        
        Args:
            product_id: The product UUID to retrieve
            for_update: If True, locks the resource for update until transaction completes
            
        Returns:
            The product or None if not found
            
        Raises:
            ConcurrencyError: If the resource is already locked by another transaction
        """
        key = f"product:{product_id}"
        
        # Check if we're requesting a lock and it's already locked by another transaction
        if for_update and key in self._locks and self._locks[key] != id(self._current_transaction):
            raise ConcurrencyError(f"Product {product_id} is locked by another transaction")
            
        # If we're in a transaction and for_update is True, lock this resource
        if for_update and self._in_transaction:
            self._locks[key] = id(self._current_transaction)
            
            # Store the current version for version checking on commit
            if key not in self._versions:
                self._versions[key] = 1
            self._transaction_versions[key] = self._versions[key]
            
        return self._data.get(key)
    
    def get_inventory_level(self, product_id: UUID, for_update=False) -> int:
        """
        Get current inventory level with optional locking for updates.
        
        Args:
            product_id: The product UUID to check
            for_update: If True, locks the resource for update until transaction completes
            
        Returns:
            Current inventory level (0 if no transactions exist)
            
        Raises:
            ConcurrencyError: If the resource is already locked by another transaction
        """
        key = f"inventory:{product_id}"
        
        # Same locking logic as get_product
        if for_update and key in self._locks and self._locks[key] != id(self._current_transaction):
            raise ConcurrencyError(f"Inventory for product {product_id} is locked by another transaction")
            
        if for_update and self._in_transaction:
            self._locks[key] = id(self._current_transaction)
            
            # Store the current version for version checking on commit
            if key not in self._versions:
                self._versions[key] = 1
            self._transaction_versions[key] = self._versions[key]
        
        return self._data.get(key, 0)
    
    def update_inventory_level(self, product_id: UUID, new_level: int):
        """Update inventory level for a product"""
        key = f"inventory:{product_id}"
        
        # If we're in a transaction and haven't locked this resource, raise an error
        if self._in_transaction and key not in self._locks:
            raise ValueError("Must lock resource with for_update=True before updating")
        
        # If in transaction, save original state for potential rollback
        if self._in_transaction and key not in self._transaction_original_states:
            self._transaction_original_states[key] = self._data.get(key, 0)
        
        # Update the data
        self._data[key] = new_level
        
        # If we're in a transaction, record the operation
        if self._in_transaction:
            self._transaction_operations.append({
                'action': 'update',
                'key': key,
                'value': new_level
            })
        else:
            # If not in transaction, immediately queue for persistence
            if hasattr(self, '_persistence_queue'):
                self._persistence_queue.put({
                    'type': 'update',
                    'key': key,
                    'value': new_level
                })
    
    def atomic_transaction(self):
        """Context manager to ensure atomicity of operations."""
        return AtomicTransaction(self)

class AtomicTransaction:
    def __init__(self, store):
        self.store = store
        self.operations = []
        self.original_states = {}
        self.locked_resources = set()
        self.transaction_versions = {}
    
    def __enter__(self):
        # Start transaction tracking
        self.store._in_transaction = True
        self.store._current_transaction = self  # Store reference to current transaction
        self.store._transaction_operations = self.operations
        self.store._transaction_original_states = self.original_states
        self.store._transaction_versions = self.transaction_versions
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is not None:
                # Exception occurred, rollback changes in memory
                for key, value in self.original_states.items():
                    self.store._data[key] = value
                
                # Don't suppress the exception
                return False
            
            # Verify versions haven't changed since we locked the resources
            for key, version in self.transaction_versions.items():
                if self.store._versions[key] != version:
                    # Version mismatch! Someone modified this while we were working
                    # Rollback and raise an error
                    for k, v in self.original_states.items():
                        self.store._data[k] = v
                    raise ConcurrencyError(f"Resource {key} was modified concurrently")
            
            # No exceptions and versions match, commit all operations as a batch
            if hasattr(self.store, '_persistence_queue'):
                # Create a version update for each modified resource
                version_updates = []
                for key in self.transaction_versions:
                    self.store._versions[key] += 1
                    version_updates.append({
                        'action': 'update_version',
                        'key': key,
                        'value': self.store._versions[key]
                    })
                
                # Queue operations and version updates atomically
                self.store._persistence_queue.put({
                    'type': 'transaction_batch',
                    'operations': self.operations + version_updates
                })
            
            return False
            
        finally:
            # Clear transaction state and release locks
            self.store._in_transaction = False
            self.store._current_transaction = None
            self.store._transaction_operations = []
            self.store._transaction_original_states = {}
            self.store._transaction_versions = {}
            
            # Release all locks held by this transaction
            for key in list(self.store._locks.keys()):
                if self.store._locks[key] == id(self):
                    del self.store._locks[key]