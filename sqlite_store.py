# inventorytracker/store/sqlite_store.py
import os
import sqlite3
import logging
from typing import Optional, List, Dict, Any, Callable, Union
from uuid import UUID
import asyncio
from datetime import datetime
import threading
import time

from pydantic import BaseModel

from ..models.product import Product
from .persistence import PersistenceQueue, OperationType, QueuedOperation

logger = logging.getLogger(__name__)

class SQLiteStore:
    """SQLite-based storage implementation with async persistence queue."""
    
    def __init__(self, db_path: str = "~/.invtrack/store.db", **queue_settings):
        self.db_path = os.path.expanduser(db_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database if it doesn't exist
        self._initialize_db()
        
        # Create persistence queue
        self.queue = PersistenceQueue(
            db_path=self.db_path,
            **queue_settings
        )
        
        # For synchronous operations that need to wait for async results
        self._sync_event_loop = None
        self._event_loop_thread = None
        
        # Cache for frequently accessed data
        self._cache = {
            "products_by_sku": {},
            "products_by_id": {},
        }
        self._cache_lock = threading.RLock()
        
        # Start background tasks
        self._ensure_event_loop()
    
    def _initialize_db(self):
        """Initialize the database schema if needed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sku TEXT NOT NULL UNIQUE,
            price REAL NOT NULL,
            reorder_level INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Add indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku)')
        
        conn.commit()
        conn.close()
    
    def _ensure_event_loop(self):
        """Ensure an event loop is running in a background thread."""
        if self._sync_event_loop is not None:
            return
            
        # Create and start event loop in a background thread
        self._event_loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="PersistenceEventLoop"
        )
        self._event_loop_thread.start()
        
        # Wait for event loop to be ready
        while self._sync_event_loop is None:
            time.sleep(0.01)
    
    def _run_event_loop(self):
        """Run the event loop in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._sync_event_loop = loop
        
        # Start the persistence queue
        loop.run_until_complete(self.queue.start())
        
        # Run the event loop
        try:
            loop.run_forever()
        finally:
            # Shutdown gracefully
            pending_tasks = asyncio.all_tasks(loop)
            for task in pending_tasks:
                task.cancel()
            
            loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
            loop.run_until_complete(self.queue.stop())
            loop.close()
            logger.info("Event loop closed")
    
    def _run_coroutine_sync(self, coro):
        """Run a coroutine synchronously from a different thread."""
        if threading.current_thread() == self._event_loop_thread:
            # We're already in the event loop thread, just run the coroutine
            return asyncio.run_coroutine_threadsafe(coro, self._sync_event_loop).result()
        else:
            # We're in a different thread, use run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(coro, self._sync_event_loop)
            return future.result()
    
    def _update_cache(self, model: str, item: BaseModel):
        """Update the in-memory cache with a model instance."""
        with self._cache_lock:
            if model == "Product":
                product = item
                self._cache["products_by_id"][product.id] = product
                self._cache["products_by_sku"][product.sku] = product
                
    def _remove_from_cache(self, model: str, item_id: UUID, item_data: Dict[str, Any] = None):
        """Remove an item from the in-memory cache."""
        with self._cache_lock:
            if model == "Product":
                if item_id in self._cache["products_by_id"]:
                    product = self._cache["products_by_id"].pop(item_id)
                    # Also remove from SKU cache
                    sku = item_data.get('sku') if item_data else getattr(product, 'sku', None)
                    if sku and sku in self._cache["products_by_sku"]:
                        del self._cache["products_by_sku"][sku]
    
    def close(self):
        """Close the store and perform cleanup."""
        if self._sync_event_loop:
            # Signal the event loop to stop
            self._sync_event_loop.call_soon_threadsafe(self._sync_event_loop.stop)
            
            # Wait for the thread to finish
            if self._event_loop_thread and self._event_loop_thread.is_alive():
                self._event_loop_thread.join(timeout=5.0)
                
        logger.info("SQLite store closed")
    
    # ======== Product Methods ========
    
    def save_product(self, product: Product, sync: bool = False) -> bool:
        """
        Save a product to the store.
        
        Args:
            product: The product to save
            sync: Whether to wait for the operation to complete
            
        Returns:
            True if the operation was queued successfully, False otherwise
        """
        # Check if this is a create or update
        existing = self.get_product_by_id(product.id)
        op_type = OperationType.UPDATE if existing else OperationType.CREATE
        
        # Update cache optimistically
        self._update_cache("Product", product)
        
        # Create the operation
        operation = QueuedOperation(
            op_type=op_type,
            model_type="Product",
            id=product.id,
            data=product.dict()
        )
        
        if sync:
            # Use a threading.Event for synchronization
            done_event = threading.Event()
            
            def callback():
                done_event.set()
            
            operation.callback = callback
            
            # Enqueue and wait
            result = self._run_coroutine_sync(self.queue.enqueue(operation))
            
            if result:
                # Wait for operation to complete with timeout
                done_event.wait(timeout=10.0)
                return True
            else:
                # Enqueue failed (backpressure)
                logger.warning("Failed to save product: queue backpressure active")
                return False
        else:
            # Async operation
            return self._run_coroutine_sync(self.queue.enqueue(operation))
    
    def get_product_by_id(self, product_id: UUID) -> Optional[Product]:
        """Get a product by ID."""
        # Check cache first
        with self._cache_lock:
            if product_id in self._cache["products_by_id"]:
                return self._cache["products_by_id"][product_id]
        
        # Not in cache, need to query
        result = self._get_product_sync(product_id=product_id)
        
        # Update cache if found
        if result:
            self._update_cache("Product", result)
            
        return result
    
    def _get_product_sync(self, product_id: Optional[UUID] = None, sku: Optional[str] = None) -> Optional[Product]:
        """
        Synchronously get a product by ID or SKU.
        
        This uses a direct database connection rather than the queue for
        immediate reads without waiting for async processing.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if product_id is not None:
                cursor.execute("SELECT * FROM products WHERE id = ?", (str(product_id),))
            elif sku is not None:
                cursor.execute("SELECT * FROM products WHERE sku = ?", (sku,))
            else:
                raise ValueError("Either product_id or sku must be provided")
                
            row = cursor.fetchone()
            
            if row:
                return Product(
                    id=UUID(row['id']),
                    name=row['name'],
                    sku=row['sku'],
                    price=row['price'],
                    reorder_level=row['reorder_level']
                )
            return None
        except Exception as e:
            logger.error(f"Error getting product: {e}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_product_by_sku(self, sku: str) -> Optional[Product]:
        """Get a product by SKU."""
        # Check cache first
        with self._cache_lock:
            if sku in self._cache["products_by_sku"]:
                return self._cache["products_by_sku"][sku]
        
        # Not in cache, need to query
        result = self._get_product_sync(sku=sku)
        
        # Update cache if found
        if result:
            self._update_cache("Product", result)
            
        return result
    
    def product_exists_by_sku(self, sku: str) -> bool:
        """Check if a product exists by SKU."""
        return self.get_product_by_sku(sku) is not None
    
    def delete_product(self, product_id: UUID, sync: bool = False) -> bool:
        """Delete a product by ID."""
        # Get product data for cache management
        product = self.get_product_by_id(product_id)
        
        # Remove from cache
        if product:
            self._remove_from_cache("Product", product_id, product.dict())
        
        # Create the operation
        operation = QueuedOperation(
            op_type=OperationType.DELETE,
            model_type="Product",
            id=product_id,
            data={}
        )
        
        if sync:
            # Use a threading.Event for synchronization
            done_event = threading.Event()
            
            def callback():
                done_event.set()
            
            operation.callback = callback
            
            # Enqueue and wait
            result = self._run_coroutine_sync(self.queue.enqueue(operation))
            
            if result:
                # Wait for operation to complete with timeout
                done_event.wait(timeout=10.0)
                return True
            else:
                return False
        else:
            # Async operation
            return self._run_coroutine_sync(self.queue.enqueue(operation))
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the queue."""
        return self.queue.get_stats()
    
    def handle_backpressure(self) -> bool:
        """
        Check if backpressure is active and attempt to handle it.
        
        Returns True if backpressure is active, False otherwise.
        
        This method can be used to implement client-side backpressure strategies.
        """
        stats = self.get_queue_stats()
        
        if stats["backpressure_active"]:
            # Implement application-specific backpressure strategies
            # For example:
            # 1. Temporary disable UI elements for adding new items
            # 2. Show a "system busy" indicator
            # 3. Delay new operations with increasing backoff
            
            # Calculate time under backpressure
            backpressure_duration = 0
            if stats["last_backpressure_time"]:
                backpressure_duration = time.time() - stats["last_backpressure_time"]
                
            # Log warning if backpressure has been active for a while
            if backpressure_duration > 30:  # 30 seconds
                logger.warning(
                    f"Persistent backpressure for {backpressure_duration:.1f}s. "
                    f"Queue: {stats['current_queue_size']}/{self.queue.max_queue_size}"
                )
                
            return True
            
        return False