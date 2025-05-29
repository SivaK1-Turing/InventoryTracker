import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TypeVar, Generic, Callable, Awaitable

logger = logging.getLogger(__name__)

T = TypeVar('T')

class TransactionState(Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"

@dataclass
class Transaction:
    id: str
    data: Dict[str, Any]
    timestamp: float
    state: TransactionState = TransactionState.PENDING
    retries: int = 0

@dataclass
class BatchConfig:
    max_batch_size: int = 100  # Maximum number of transactions per batch
    flush_interval_seconds: float = 5.0  # Time between flushes
    max_pending_transactions: int = 1000  # High water mark for backpressure
    min_pending_transactions: int = 100  # Low water mark for backpressure
    max_retries: int = 3  # Maximum retry attempts for failed transactions
    checkpoint_frequency: int = 1000  # Number of transactions between checkpoints

class AsyncPersistenceManager(Generic[T]):
    """
    Manages asynchronous persistence with batched writes for improved performance
    while maintaining durability guarantees.
    """
    
    def __init__(
        self,
        data_dir: Path,
        serializer: Callable[[T], Dict[str, Any]],
        deserializer: Callable[[Dict[str, Any]], T],
        config: Optional[BatchConfig] = None,
    ):
        self.data_dir = data_dir
        self.data_file = data_dir / "data.json"
        self.wal_dir = data_dir / "wal"
        self.serializer = serializer
        self.deserializer = deserializer
        self.config = config or BatchConfig()
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.wal_dir.mkdir(exist_ok=True, parents=True)
        
        # Transaction handling
        self.transaction_queue: asyncio.Queue[Transaction] = asyncio.Queue()
        self.pending_transactions: Dict[str, Transaction] = {}
        self.backpressure = asyncio.Event()
        self.backpressure.set()  # No backpressure initially
        
        # Data state
        self.data: Dict[str, T] = {}
        self.dirty_keys: Set[str] = set()
        
        # Task management
        self.batch_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self.checkpoint_counter = 0
        
    async def start(self):
        """Initialize and start the background batch processing task."""
        await self._recover_from_wal()
        await self._load_data()
        self.batch_task = asyncio.create_task(self._batch_processor())
        logger.info("Persistence manager started")
        
    async def stop(self):
        """Gracefully stop the batch processor and flush pending transactions."""
        if self.batch_task:
            self.shutdown_event.set()
            try:
                # Wait for the batch task to complete (should flush remaining transactions)
                await asyncio.wait_for(self.batch_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Batch processor did not complete in time during shutdown")
                if self.batch_task:
                    self.batch_task.cancel()
            
            # Final flush to ensure all pending transactions are written
            await self._flush_all_pending()
            logger.info("Persistence manager stopped")
            
    async def save(self, key: str, item: T) -> None:
        """
        Queue an item for saving with backpressure control.
        
        When too many items are queued, this will pause to prevent memory issues.
        """
        # Wait if backpressure is active
        await self.backpressure.wait()
        
        # Create transaction
        transaction_id = str(uuid.uuid4())
        transaction = Transaction(
            id=transaction_id,
            data={
                "key": key,
                "value": self.serializer(item),
                "operation": "save"
            },
            timestamp=time.time()
        )
        
        # Log to WAL first for durability
        await self._append_to_wal(transaction)
        
        # Update in-memory state
        self.data[key] = item
        self.dirty_keys.add(key)
        
        # Queue for batch processing
        self.pending_transactions[transaction_id] = transaction
        await self.transaction_queue.put(transaction)
        
        # Apply backpressure if queue is too large
        if len(self.pending_transactions) >= self.config.max_pending_transactions:
            logger.warning(f"Applying backpressure: {len(self.pending_transactions)} pending transactions")
            self.backpressure.clear()
            
    async def delete(self, key: str) -> None:
        """Queue an item for deletion with backpressure control."""
        # Wait if backpressure is active
        await self.backpressure.wait()
        
        # Create transaction
        transaction_id = str(uuid.uuid4())
        transaction = Transaction(
            id=transaction_id,
            data={
                "key": key,
                "operation": "delete"
            },
            timestamp=time.time()
        )
        
        # Log to WAL first for durability
        await self._append_to_wal(transaction)
        
        # Update in-memory state
        if key in self.data:
            del self.data[key]
            self.dirty_keys.add(key)
        
        # Queue for batch processing
        self.pending_transactions[transaction_id] = transaction
        await self.transaction_queue.put(transaction)
        
        # Apply backpressure if queue is too large
        if len(self.pending_transactions) >= self.config.max_pending_transactions:
            logger.warning(f"Applying backpressure: {len(self.pending_transactions)} pending transactions")
            self.backpressure.clear()
            
    async def get(self, key: str) -> Optional[T]:
        """Get an item by key from the in-memory store."""
        return self.data.get(key)
    
    async def get_all(self) -> Dict[str, T]:
        """Get all items from the in-memory store."""
        return dict(self.data)
    
    async def _batch_processor(self) -> None:
        """Background task that processes batches of transactions."""
        batch: List[Transaction] = []
        last_flush_time = time.time()
        
        while not self.shutdown_event.is_set():
            try:
                # Try to get a transaction with timeout
                try:
                    transaction = await asyncio.wait_for(
                        self.transaction_queue.get(), 
                        timeout=self.config.flush_interval_seconds
                    )
                    batch.append(transaction)
                    self.transaction_queue.task_done()
                except asyncio.TimeoutError:
                    # Flush on timeout if we have transactions
                    if batch:
                        last_flush_time = time.time()
                        await self._flush_batch(batch)
                        batch = []
                    continue
                
                # Flush if we've reached max batch size
                current_time = time.time()
                if (len(batch) >= self.config.max_batch_size or 
                    current_time - last_flush_time >= self.config.flush_interval_seconds):
                    last_flush_time = current_time
                    await self._flush_batch(batch)
                    batch = []
                
                # Release backpressure if we're below low water mark
                if (not self.backpressure.is_set() and 
                    len(self.pending_transactions) <= self.config.min_pending_transactions):
                    logger.info(f"Releasing backpressure: {len(self.pending_transactions)} pending transactions")
                    self.backpressure.set()
                
            except Exception as e:
                logger.exception(f"Error in batch processor: {e}")
                await asyncio.sleep(1.0)  # Avoid tight error loops
        
        # Flush any remaining transactions on shutdown
        if batch:
            await self._flush_batch(batch)
    
    async def _flush_batch(self, batch: List[Transaction]) -> None:
        """Flush a batch of transactions to permanent storage."""
        if not batch:
            return
            
        try:
            # First phase: mark transactions as committing in WAL
            for transaction in batch:
                transaction.state = TransactionState.COMMITTED
                await self._update_wal_entry(transaction)
            
            # Second phase: persist data file with changes
            await self._save_data_file()
            
            # Remove completed transactions from pending
            for transaction in batch:
                if transaction.id in self.pending_transactions:
                    del self.pending_transactions[transaction.id]
                
            # Clean WAL files for committed transactions
            for transaction in batch:
                await self._remove_wal_entry(transaction)
                
            # Perform checkpoint if needed
            self.checkpoint_counter += len(batch)
            if self.checkpoint_counter >= self.config.checkpoint_frequency:
                await self._checkpoint()
                self.checkpoint_counter = 0
                
            logger.debug(f"Flushed batch of {len(batch)} transactions")
            
        except Exception as e:
            logger.exception(f"Error flushing batch: {e}")
            
            # Retry logic for failed transactions
            for transaction in batch:
                transaction.retries += 1
                if transaction.retries < self.config.max_retries:
                    # Re-queue for later processing
                    await self.transaction_queue.put(transaction)
                else:
                    logger.error(f"Transaction {transaction.id} exceeded max retries")
                    # Move to dead letter queue or error log in a real implementation
    
    async def _flush_all_pending(self) -> None:
        """Flush all pending transactions during shutdown."""
        batch = []
        try:
            while not self.transaction_queue.empty():
                batch.append(await self.transaction_queue.get())
                self.transaction_queue.task_done()
                
                if len(batch) >= self.config.max_batch_size:
                    await self._flush_batch(batch)
                    batch = []
            
            # Flush final batch
            if batch:
                await self._flush_batch(batch)
                
        except Exception as e:
            logger.exception(f"Error flushing pending transactions: {e}")
    
    async def _load_data(self) -> None:
        """Load data from storage into memory."""
        if not self.data_file.exists():
            self.data = {}
            return
            
        try:
            async with asyncio.to_thread(open, self.data_file, 'r') as f:
                data_dict = await asyncio.to_thread(json.load, f)
            
            # Convert raw data to objects
            self.data = {
                key: self.deserializer(value) 
                for key, value in data_dict.items()
            }
            logger.info(f"Loaded {len(self.data)} items from {self.data_file}")
        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            self.data = {}
    
    async def _save_data_file(self) -> None:
        """Save data to storage with atomic write pattern."""
        if not self.dirty_keys:
            return
            
        try:
            # Create data dictionary from objects
            data_dict = {
                key: self.serializer(value)
                for key, value in self.data.items()
            }
            
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(mode='w', dir=self.data_dir, delete=False) as temp_file:
                json.dump(data_dict, temp_file, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Ensure data is on disk
                
            # Atomic rename for durability
            temp_path = Path(temp_file.name)
            temp_path.rename(self.data_file)
            
            # Clear dirty keys
            self.dirty_keys.clear()
            
        except Exception as e:
            logger.exception(f"Error saving data file: {e}")
            raise
    
    async def _append_to_wal(self, transaction: Transaction) -> None:
        """Append a transaction to the write-ahead log."""
        wal_path = self.wal_dir / f"{transaction.id}.json"
        
        try:
            transaction_data = {
                "id": transaction.id,
                "data": transaction.data,
                "timestamp": transaction.timestamp,
                "state": transaction.state.value
            }
            
            with open(wal_path, 'w') as f:
                json.dump(transaction_data, f)
                f.flush()
                os.fsync(f.fileno())  # Ensure WAL entry is durable
                
        except Exception as e:
            logger.exception(f"Error appending to WAL: {e}")
            raise
    
    async def _update_wal_entry(self, transaction: Transaction) -> None:
        """Update an existing WAL entry with new state."""
        wal_path = self.wal_dir / f"{transaction.id}.json"
        
        if not wal_path.exists():
            return
            
        try:
            transaction_data = {
                "id": transaction.id,
                "data": transaction.data,
                "timestamp": transaction.timestamp,
                "state": transaction.state.value
            }
            
            with open(wal_path, 'w') as f:
                json.dump(transaction_data, f)
                f.flush()
                os.fsync(f.fileno())
                
        except Exception as e:
            logger.exception(f"Error updating WAL entry: {e}")
            raise
    
    async def _remove_wal_entry(self, transaction: Transaction) -> None:
        """Remove a WAL entry after successful commit."""
        wal_path = self.wal_dir / f"{transaction.id}.json"
        
        if wal_path.exists():
            try:
                wal_path.unlink()
            except Exception as e:
                logger.exception(f"Error removing WAL entry: {e}")
    
    async def _recover_from_wal(self) -> None:
        """Recover any pending transactions from the WAL."""
        try:
            wal_files = list(self.wal_dir.glob("*.json"))
            if not wal_files:
                return
                
            logger.info(f"Found {len(wal_files)} WAL entries for recovery")
            
            for wal_file in wal_files:
                try:
                    with open(wal_file, 'r') as f:
                        transaction_data = json.load(f)
                        
                    # Recreate transaction
                    transaction = Transaction(
                        id=transaction_data["id"],
                        data=transaction_data["data"],
                        timestamp=transaction_data["timestamp"],
                        state=TransactionState(transaction_data["state"])
                    )
                    
                    # Handle based on state
                    if transaction.state == TransactionState.COMMITTED:
                        # Apply the committed transaction
                        key = transaction.data["key"]
                        operation = transaction.data.get("operation")
                        
                        if operation == "save":
                            self.data[key] = self.deserializer(transaction.data["value"])
                            self.dirty_keys.add(key)
                        elif operation == "delete":
                            if key in self.data:
                                del self.data[key]
                                self.dirty_keys.add(key)
                                
                        # Remove entry as it's been applied
                        await self._remove_wal_entry(transaction)
                        
                    elif transaction.state == TransactionState.PENDING:
                        # Re-queue pending transaction
                        self.pending_transactions[transaction.id] = transaction
                        await self.transaction_queue.put(transaction)
                    
                except Exception as e:
                    logger.exception(f"Error recovering WAL entry {wal_file}: {e}")
                    
            logger.info(f"Recovered {len(self.pending_transactions)} pending transactions")
            
        except Exception as e:
            logger.exception(f"Error during WAL recovery: {e}")
    
    async def _checkpoint(self) -> None:
        """
        Create a checkpoint by consolidating the current state.
        This saves the current data and cleans up the WAL.
        """
        try:
            logger.info("Starting checkpoint")
            
            # Save current state
            await self._save_data_file()
            
            # Clean up WAL for committed transactions
            cleaned = 0
            for wal_file in self.wal_dir.glob("*.json"):
                try:
                    with open(wal_file, 'r') as f:
                        transaction_data = json.load(f)
                    
                    if transaction_data["state"] == TransactionState.COMMITTED.value:
                        wal_file.unlink()
                        cleaned += 1
                        
                except Exception as e:
                    logger.exception(f"Error cleaning WAL file {wal_file}: {e}")
            
            logger.info(f"Checkpoint complete: cleaned {cleaned} WAL entries")
            
        except Exception as e:
            logger.exception(f"Error during checkpoint: {e}")