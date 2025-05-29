# inventorytracker/store/persistence.py
import asyncio
import threading
import time
import logging
import queue
from typing import Dict, Any, Optional, Union, List, Callable, TypeVar, Generic, Tuple
from enum import Enum
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import signal
import os
import json
from uuid import UUID
from datetime import datetime

from ..models.product import Product

# Configure logger
logger = logging.getLogger(__name__)

# Define types for clarity
T = TypeVar('T')

class OperationType(str, Enum):
    """Types of operations that can be queued."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    QUERY = "query"

@dataclass
class QueuedOperation:
    """Represents an operation queued for processing."""
    op_type: OperationType
    model_type: str  # e.g., "Product", "Category"
    data: Dict[str, Any]
    id: Optional[UUID] = None  # The ID of the entity for update/delete
    callback: Optional[Callable] = None  # Optional callback for when operation completes
    timestamp: float = time.time()  # When the operation was queued
    attempt: int = 0  # Number of attempts to process this operation
    
    def increment_attempt(self):
        """Increment the attempt counter and return self."""
        self.attempt += 1
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        result = {
            "op_type": self.op_type,
            "model_type": self.model_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "attempt": self.attempt
        }
        if self.id is not None:
            result["id"] = str(self.id)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueuedOperation':
        """Create from a dictionary (for deserialization from persistent queue)."""
        id_value = data.get("id")
        if id_value is not None:
            id_value = UUID(id_value)
            
        return cls(
            op_type=data["op_type"],
            model_type=data["model_type"],
            data=data["data"],
            id=id_value,
            timestamp=data.get("timestamp", time.time()),
            attempt=data.get("attempt", 0)
        )

class PersistenceQueue:
    """
    Queue system for asynchronous persistence operations with backpressure handling.
    
    Features:
    - Async queue for non-blocking operations
    - Background worker for processing
    - Backpressure handling
    - Retry mechanism for failed operations
    - Persistence of the queue itself to handle program restarts
    """
    
    # Default settings
    DEFAULT_MAX_QUEUE_SIZE = 1000
    DEFAULT_BATCH_SIZE = 50
    DEFAULT_FLUSH_INTERVAL = 1.0  # seconds
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0  # seconds
    DEFAULT_QUEUE_FILE = "~/.invtrack/queue_backup.json"
    
    def __init__(
        self, 
        db_path: str, 
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        queue_file: str = DEFAULT_QUEUE_FILE,
        thread_pool_size: int = 4
    ):
        """
        Initialize the persistence queue.
        
        Args:
            db_path: Path to the SQLite database
            max_queue_size: Maximum queue size before applying backpressure
            batch_size: Number of operations to process in a batch
            flush_interval: Time in seconds between flush attempts
            max_retries: Maximum number of retries for failed operations
            retry_delay: Time in seconds between retries
            queue_file: Path to save queue state for recovery
            thread_pool_size: Number of threads in the executor pool
        """
        self.db_path = db_path
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.queue_file = os.path.expanduser(queue_file)
        
        # Create async queue
        self._queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Thread pool for running sqlite operations (which are blocking)
        self._executor = ThreadPoolExecutor(max_workers=thread_pool_size)
        
        # For storing callbacks and managing state
        self._callbacks: Dict[UUID, Callable] = {}
        
        # For backpressure signaling
        self._backpressure_event = asyncio.Event()
        self._backpressure_event.set()  # Initially no backpressure
        
        # For tracking high water mark
        self._high_water_mark = int(max_queue_size * 0.8)  # 80% of max
        self._low_water_mark = int(max_queue_size * 0.5)   # 50% of max
        
        # State tracking
        self._running = False
        self._worker_task = None
        self._flush_task = None
        
        # Statistics
        self._stats = {
            "operations_queued": 0,
            "operations_processed": 0,
            "operations_failed": 0,
            "batches_processed": 0,
            "backpressure_events": 0,
            "last_backpressure_time": None,
            "queue_high_water_mark": 0,
        }
        
        # Protection for shared resources
        self._stats_lock = threading.RLock()
        
        # For queue recovery if we previously crashed
        self._recover_queue()
        
    def _recover_queue(self):
        """Recover the queue from disk if a previous backup exists."""
        if not os.path.exists(self.queue_file):
            logger.info(f"No queue backup found at {self.queue_file}")
            return
            
        try:
            with open(self.queue_file, 'r') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                logger.error(f"Invalid queue backup format at {self.queue_file}")
                return
                
            # Re-add each operation to the queue
            recovered_count = 0
            for op_data in data:
                try:
                    operation = QueuedOperation.from_dict(op_data)
                    # Use the sync version to avoid event loop issues during initialization
                    self._queue.put_nowait(operation)
                    recovered_count += 1
                except Exception as e:
                    logger.error(f"Failed to recover operation: {e}")
                    
            logger.info(f"Recovered {recovered_count} operations from backup")
            
            # Update stats
            with self._stats_lock:
                self._stats["operations_queued"] += recovered_count
                
            # Delete the backup after successful recovery
            os.remove(self.queue_file)
            
        except Exception as e:
            logger.error(f"Failed to recover queue from {self.queue_file}: {e}")
            
    def _backup_queue(self):
        """Save the current queue to disk for recovery."""
        try:
            # Get all items without removing them
            queue_items = list(self._queue._queue)  # Access internal deque directly
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.queue_file), exist_ok=True)
            
            # Serialize and save
            with open(self.queue_file, 'w') as f:
                json.dump([item.to_dict() for item in queue_items], f)
                
            logger.info(f"Backed up {len(queue_items)} operations to {self.queue_file}")
            
        except Exception as e:
            logger.error(f"Failed to backup queue to {self.queue_file}: {e}")
    
    async def start(self):
        """Start the background worker and periodic flush tasks."""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._flush_task = asyncio.create_task(self._periodic_flush())
        
        logger.info("PersistenceQueue worker started")
        
    async def stop(self):
        """
        Stop the background worker and flush any remaining items.
        This performs a graceful shutdown.
        """
        if not self._running:
            return
            
        self._running = False
        
        logger.info(f"Stopping PersistenceQueue worker ({self._queue.qsize()} items remaining)...")
        
        # Wait for worker to complete current work
        if self._worker_task:
            await asyncio.wait_for(self._worker_task, timeout=5.0)
            
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining items
        if not self._queue.empty():
            logger.info(f"Flushing {self._queue.qsize()} remaining items...")
            await self._flush_queue()
            
        # Backup any items that couldn't be processed
        if not self._queue.empty():
            self._backup_queue()
            
        # Shutdown the thread pool
        self._executor.shutdown(wait=True)
        
        logger.info("PersistenceQueue worker stopped")
    
    async def enqueue(self, operation: QueuedOperation) -> bool:
        """
        Enqueue an operation for processing.
        
        Returns True if the operation was enqueued, False if backpressure is active.
        """
        # Check for backpressure
        if not self._backpressure_event.is_set():
            # Under backpressure
            with self._stats_lock:
                self._stats["backpressure_events"] += 1
                self._stats["last_backpressure_time"] = time.time()
            
            # Decide whether to block or fail
            if operation.op_type in (OperationType.CREATE, OperationType.UPDATE):
                # For creates/updates, wait for backpressure to clear with a timeout
                try:
                    await asyncio.wait_for(self._backpressure_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Backpressure timeout, rejecting {operation.op_type} operation")
                    return False
            else:
                # For queries/deletes, fail immediately
                logger.warning(f"Backpressure active, rejecting {operation.op_type} operation")
                return False
                
        # Store callback if provided
        if operation.callback:
            self._callbacks[operation.id] = operation.callback
            # Remove callback from operation to avoid serialization issues
            operation.callback = None
        
        # Try to enqueue
        try:
            await self._queue.put(operation)
            
            with self._stats_lock:
                self._stats["operations_queued"] += 1
                
                # Track high-water mark
                current_size = self._queue.qsize()
                if current_size > self._stats["queue_high_water_mark"]:
                    self._stats["queue_high_water_mark"] = current_size
                
                # Apply backpressure if needed
                if current_size >= self._high_water_mark and self._backpressure_event.is_set():
                    logger.warning(f"Queue size {current_size} reached high-water mark, applying backpressure")
                    self._backpressure_event.clear()
                    
            return True
        except Exception as e:
            logger.error(f"Failed to enqueue operation: {e}")
            return False

    async def _worker_loop(self):
        """Background task that processes the queue."""
        while self._running or not self._queue.empty():
            try:
                # Try to process a batch of operations
                if not self._queue.empty():
                    await self._flush_queue()
                
                # Release backpressure if queue has drained enough
                if (not self._backpressure_event.is_set() and 
                    self._queue.qsize() <= self._low_water_mark):
                    logger.info(f"Queue size dropped below low-water mark, releasing backpressure")
                    self._backpressure_event.set()
                
                # Wait a bit before next flush
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in persistence worker: {e}")
                await asyncio.sleep(1.0)  # Avoid tight loop on persistent errors
    
    async def _periodic_flush(self):
        """Task that periodically flushes the queue."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                if not self._queue.empty():
                    await self._flush_queue()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")
    
    async def _flush_queue(self):
        """Flush pending operations to the database."""
        # Process in batches
        batch_size = min(self.batch_size, self._queue.qsize())
        if batch_size == 0:
            return
            
        batch = []
        try:
            # Collect a batch of operations
            for _ in range(batch_size):
                if self._queue.empty():
                    break
                batch.append(await self._queue.get())
                
            # Process the batch in a thread pool to avoid blocking the event loop
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._process_batch,
                batch
            )
            
            # Update stats
            with self._stats_lock:
                self._stats["batches_processed"] += 1
                
        except Exception as e:
            logger.error(f"Error flushing queue: {e}")
            
            # Put failed operations back in the queue
            for op in batch:
                await self._handle_failed_operation(op)
    
    def _process_batch(self, batch: List[QueuedOperation]):
        """
        Process a batch of operations in a separate thread.
        
        This runs in a thread from the thread pool and performs the actual SQLite operations.
        """
        if not batch:
            return
            
        # Group operations by type for more efficient processing
        creates = []
        updates = []
        deletes = []
        queries = []
        
        for op in batch:
            if op.op_type == OperationType.CREATE:
                creates.append(op)
            elif op.op_type == OperationType.UPDATE:
                updates.append(op)
            elif op.op_type == OperationType.DELETE:
                deletes.append(op)
            elif op.op_type == OperationType.QUERY:
                queries.append(op)
        
        # Connect to the database
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Process creates
            if creates:
                self._process_creates(creates, conn, cursor)
                
            # Process updates
            if updates:
                self._process_updates(updates, conn, cursor)
                
            # Process deletes
            if deletes:
                self._process_deletes(deletes, conn, cursor)
                
            # Process queries (these aren't batched since each might be different)
            for op in queries:
                self._process_query(op, conn, cursor)
                
            # Commit all changes
            conn.commit()
            
            # Run callbacks (in the thread pool, not the main thread)
            for op in batch:
                if op.id and op.id in self._callbacks:
                    try:
                        callback = self._callbacks.pop(op.id)
                        if callback:
                            callback()
                    except Exception as e:
                        logger.error(f"Error executing callback for operation {op.id}: {e}")
            
            # Update stats
            with self._stats_lock:
                self._stats["operations_processed"] += len(batch)
            
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            # Put failed operations back in the queue for retry
            for op in batch:
                asyncio.run_coroutine_threadsafe(
                    self._handle_failed_operation(op), 
                    asyncio.get_event_loop()
                )
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _process_creates(self, operations: List[QueuedOperation], conn, cursor):
        """Process a batch of create operations."""
        # Group by model type
        by_model = {}
        for op in operations:
            if op.model_type not in by_model:
                by_model[op.model_type] = []
            by_model[op.model_type].append(op)
            
        # Process each model type
        for model_type, ops in by_model.items():
            if model_type == "Product":
                self._create_products(ops, conn, cursor)
            # Add other model types as needed
            else:
                logger.warning(f"Unknown model type for batch create: {model_type}")
    
    def _create_products(self, operations: List[QueuedOperation], conn, cursor):
        """Create multiple products in a single batch."""
        # Set up parameters for batch insert
        values = []
        for op in operations:
            data = op.data
            values.append((
                str(op.data.get('id', op.id)),
                data.get('name', ''),
                data.get('sku', ''),
                float(data.get('price', 0)),
                data.get('reorder_level', 0),
                datetime.now().isoformat()
            ))
            
        # Batch insert
        cursor.executemany(
            """
            INSERT OR REPLACE INTO products 
            (id, name, sku, price, reorder_level, updated_at) 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            values
        )
    
    def _process_updates(self, operations: List[QueuedOperation], conn, cursor):
        """Process a batch of update operations."""
        # Similar to _process_creates but for updates
        # Implementation depends on your database schema
        by_model = {}
        for op in operations:
            if op.model_type not in by_model:
                by_model[op.model_type] = []
            by_model[op.model_type].append(op)
            
        for model_type, ops in by_model.items():
            if model_type == "Product":
                self._update_products(ops, conn, cursor)
            else:
                logger.warning(f"Unknown model type for batch update: {model_type}")
    
    def _update_products(self, operations: List[QueuedOperation], conn, cursor):
        """Update multiple products in a batch."""
        for op in operations:
            data = op.data
            cursor.execute(
                """
                UPDATE products 
                SET name = ?, sku = ?, price = ?, reorder_level = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    data.get('name', ''),
                    data.get('sku', ''),
                    float(data.get('price', 0)),
                    data.get('reorder_level', 0),
                    datetime.now().isoformat(),
                    str(op.id)
                )
            )
    
    def _process_deletes(self, operations: List[QueuedOperation], conn, cursor):
        """Process a batch of delete operations."""
        by_model = {}
        for op in operations:
            if op.model_type not in by_model:
                by_model[op.model_type] = []
            by_model[op.model_type].append(op)
            
        for model_type, ops in by_model.items():
            if model_type == "Product":
                ids = [str(op.id) for op in ops if op.id]
                if ids:
                    placeholders = ', '.join('?' for _ in ids)
                    cursor.execute(
                        f"DELETE FROM products WHERE id IN ({placeholders})",
                        ids
                    )
            else:
                logger.warning(f"Unknown model type for batch delete: {model_type}")
    
    def _process_query(self, operation: QueuedOperation, conn, cursor):
        """Process a single query operation."""
        # Queries are likely too varied to batch effectively
        if operation.model_type == "Product":
            if operation.op_type == OperationType.QUERY:
                query_type = operation.data.get("query_type")
                
                if query_type == "get_by_id" and operation.id:
                    cursor.execute(
                        "SELECT * FROM products WHERE id = ?",
                        (str(operation.id),)
                    )
                    row = cursor.fetchone()
                    if row and operation.callback:
                        # For queries, we pass the result to the callback
                        product = self._row_to_product(row)
                        asyncio.run_coroutine_threadsafe(
                            self._call_query_callback(operation.id, product),
                            asyncio.get_event_loop()
                        )
                        
                elif query_type == "get_by_sku":
                    sku = operation.data.get("sku")
                    cursor.execute(
                        "SELECT * FROM products WHERE sku = ?",
                        (sku,)
                    )
                    row = cursor.fetchone()
                    if operation.callback:
                        product = self._row_to_product(row) if row else None
                        asyncio.run_coroutine_threadsafe(
                            self._call_query_callback(operation.id, product),
                            asyncio.get_event_loop()
                        )
        else:
            logger.warning(f"Unknown model type for query: {operation.model_type}")
    
    def _row_to_product(self, row) -> Product:
        """Convert a database row to a Product model."""
        return Product(
            id=UUID(row['id']),
            name=row['name'],
            sku=row['sku'],
            price=row['price'],
            reorder_level=row['reorder_level']
        )
    
    async def _call_query_callback(self, op_id, result):
        """Call a query callback with its result."""
        if op_id in self._callbacks:
            callback = self._callbacks.pop(op_id)
            if callback:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in query callback: {e}")
    
    async def _handle_failed_operation(self, operation: QueuedOperation):
        """Handle a failed operation with retries or dead-letter queue."""
        operation = operation.increment_attempt()
        
        if operation.attempt <= self.max_retries:
            # Schedule retry with exponential backoff
            delay = self.retry_delay * (2 ** (operation.attempt - 1))  # Exponential backoff
            logger.info(f"Scheduling retry {operation.attempt}/{self.max_retries} in {delay:.2f} seconds")
            
            # Wait before retrying
            await asyncio.sleep(delay)
            
            # Re-queue the operation
            await self._queue.put(operation)
        else:
            # Too many retries, move to dead-letter queue
            logger.error(f"Operation failed after {operation.attempt} attempts, moving to dead-letter queue")
            
            # Store in a dead-letter file for manual recovery
            dead_letter_file = os.path.expanduser("~/.invtrack/dead_letter_queue.json")
            os.makedirs(os.path.dirname(dead_letter_file), exist_ok=True)
            
            entries = []
            if os.path.exists(dead_letter_file):
                try:
                    with open(dead_letter_file, 'r') as f:
                        entries = json.load(f)
                except:
                    entries = []
            
            entries.append(operation.to_dict())
            
            with open(dead_letter_file, 'w') as f:
                json.dump(entries, f, indent=2)
                
            with self._stats_lock:
                self._stats["operations_failed"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        with self._stats_lock:
            stats = self._stats.copy()
            # Add current queue size
            stats["current_queue_size"] = self._queue.qsize()
            stats["backpressure_active"] = not self._backpressure_event.is_set()
            return stats