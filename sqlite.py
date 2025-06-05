"""
store/sqlite.py - SQLite Storage Adapter Implementation

This module provides a concrete implementation of the StorageAdapter interface
using SQLite as the database backend with SQLAlchemy Core for database operations.
It includes schema migration support through a schema_version table and idempotent DDL.
"""

import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, AsyncIterator, Union, Tuple
from contextlib import asynccontextmanager
import json
import uuid

import sqlalchemy as sa
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.sql.expression import select, insert, update, delete
from sqlalchemy import MetaData, Table, Column, Integer, String, Float, DateTime, Text, ForeignKey, Index
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..models.product import Product
from ..models.inventory import StockTransaction
from .adapter import StorageAdapter, StorageError

# Set up logger
logger = logging.getLogger(__name__)


class SQLiteMigrationError(StorageError):
    """Exception raised for errors during migration operations"""
    pass


class SQLiteAdapter(StorageAdapter):
    """
    A storage adapter implementation that persists data in a SQLite database.
    
    This adapter uses SQLAlchemy Core for database operations and includes
    schema migration support for evolving the database schema over time.
    """
    
    # Current schema version - increment this when changing the schema
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: Union[str, Path], echo: bool = False):
        """
        Initialize the SQLite storage adapter.
        
        Args:
            db_path: Path to the SQLite database file
            echo: If True, log all SQL statements
        """
        self.db_path = Path(db_path)
        self.echo = echo
        self.engine = None
        self.metadata = MetaData()
        self._define_tables()
        self._transaction_conn = None
        
    def _define_tables(self) -> None:
        """Define database tables using SQLAlchemy Core."""
        
        # Schema version table for migrations
        self.schema_version = Table(
            'schema_version', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('version', Integer, nullable=False),
            Column('applied_at', DateTime, nullable=False, default=datetime.utcnow),
            Column('description', String, nullable=False),
        )
        
        # Products table
        self.products = Table(
            'products', self.metadata,
            Column('id', String(36), primary_key=True),
            Column('name', String(100), nullable=False),
            Column('description', Text),
            Column('sku', String(50), nullable=False, unique=True),
            Column('price', Float, nullable=False, default=0.0),
            Column('quantity', Integer, nullable=False, default=0),
            Column('reorder_point', Integer, default=0),
            Column('reorder_quantity', Integer, default=0),
            Column('category', String(50)),
            Column('supplier', String(100)),
            Column('created_at', DateTime, nullable=False, default=datetime.utcnow),
            Column('updated_at', DateTime, nullable=False, default=datetime.utcnow, 
                 onupdate=datetime.utcnow),
            
            # Indexes
            Index('idx_products_sku', 'sku', unique=True),
            Index('idx_products_name', 'name'),
            Index('idx_products_category', 'category'),
        )
        
        # Stock transactions table
        self.transactions = Table(
            'transactions', self.metadata,
            Column('id', String(36), primary_key=True),
            Column('product_id', String(36), ForeignKey('products.id'), nullable=False),
            Column('delta', Integer, nullable=False),
            Column('timestamp', DateTime, nullable=False, default=datetime.utcnow),
            Column('transaction_type', String(50)),
            Column('user', String(100)),
            Column('note', Text),
            
            # Indexes
            Index('idx_transactions_product', 'product_id'),
            Index('idx_transactions_timestamp', 'timestamp'),
        )
    
    async def connect(self) -> None:
        """
        Connect to the SQLite database and ensure schema is up to date.
        
        This method should be called before using the adapter.
        """
        if self.engine is not None:
            return
            
        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database URL
        db_url = f"sqlite:///{self.db_path}"
        
        # Create engine
        self.engine = sa.create_engine(db_url, echo=self.echo)
        
        # Apply migrations
        await self._run_migrations()
        
        logger.info(f"Connected to SQLite database at {self.db_path}")
    
    async def _run_migrations(self) -> None:
        """Apply any pending database migrations."""
        # Get a connection
        conn = self.engine.connect()
        
        try:
            # Create schema_version table if it doesn't exist
            if not self.engine.dialect.has_table(conn, 'schema_version'):
                self.schema_version.create(self.engine)
                logger.info("Created schema_version table")
                
                # Insert initial version
                conn.execute(insert(self.schema_version).values(
                    version=0,
                    applied_at=datetime.utcnow(),
                    description="Initial schema version"
                ))
                conn.commit()
            
            # Get current schema version
            result = conn.execute(select(self.schema_version.c.version).
                                order_by(self.schema_version.c.version.desc()).
                                limit(1))
            current_version = result.scalar() or 0
            
            # Apply migrations if needed
            if current_version < self.SCHEMA_VERSION:
                logger.info(f"Database schema needs upgrade from v{current_version} to v{self.SCHEMA_VERSION}")
                await self._apply_migrations(conn, current_version)
            else:
                logger.info(f"Database schema is up-to-date at version {current_version}")
                
        except Exception as e:
            conn.rollback()
            logger.exception(f"Error during database migration: {e}")
            raise SQLiteMigrationError(f"Failed to apply migrations: {e}") from e
            
        finally:
            conn.close()
    
    async def _apply_migrations(self, conn: Connection, current_version: int) -> None:
        """
        Apply necessary migrations to update from current_version to SCHEMA_VERSION.
        
        Args:
            conn: Database connection
            current_version: Current schema version
        """
        try:
            # Schema v0 to v1: Create initial tables
            if current_version < 1:
                logger.info("Applying migration v0 to v1: Creating initial tables")
                
                # Create tables using sqlalchemy's idempotent CREATE TABLE IF NOT EXISTS
                # First check if tables exist to make migration truly idempotent
                if not self.engine.dialect.has_table(conn, 'products'):
                    self.products.create(self.engine)
                    logger.info("Created products table")
                
                if not self.engine.dialect.has_table(conn, 'transactions'):
                    self.transactions.create(self.engine)
                    logger.info("Created transactions table")
                
                # Update schema version
                conn.execute(insert(self.schema_version).values(
                    version=1,
                    applied_at=datetime.utcnow(),
                    description="Initial schema with products and transactions tables"
                ))
                current_version = 1
                logger.info("Applied migration to v1")
            
            # Add more migrations here as needed
            # Example: Schema v1 to v2
            # if current_version < 2:
            #     logger.info("Applying migration v1 to v2: Adding new column")
            #     
            #     # Check if column exists first to make it idempotent
            #     columns = [c.name for c in self.products.columns]
            #     if 'new_column' not in columns:
            #         conn.execute('ALTER TABLE products ADD COLUMN new_column TEXT')
            #         logger.info("Added new_column to products table")
            #     
            #     # Update schema version
            #     conn.execute(insert(self.schema_version).values(
            #         version=2,
            #         applied_at=datetime.utcnow(),
            #         description="Added new_column to products table"
            #     ))
            #     current_version = 2
            #     logger.info("Applied migration to v2")
            
            conn.commit()
            logger.info(f"Schema migrated successfully to version {current_version}")
            
        except Exception as e:
            conn.rollback()
            logger.exception(f"Error applying migrations: {e}")
            raise SQLiteMigrationError(f"Failed to apply migrations: {e}") from e
    
    async def close(self) -> None:
        """Close the database connection."""
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
            logger.info("Closed SQLite database connection")
    
    async def _product_from_row(self, row: Dict[str, Any]) -> Product:
        """Convert a database row to a Product object."""
        product_data = {
            'id': uuid.UUID(row['id']),
            'name': row['name'],
            'sku': row['sku'],
            'price': float(row['price']),
            'quantity': int(row['quantity']),
        }
        
        # Add optional fields if present
        if row.get('description') is not None:
            product_data['description'] = row['description']
        if row.get('reorder_point') is not None:
            product_data['reorder_point'] = row['reorder_point']
        if row.get('reorder_quantity') is not None:
            product_data['reorder_quantity'] = row['reorder_quantity']
        if row.get('category') is not None:
            product_data['category'] = row['category']
        if row.get('supplier') is not None:
            product_data['supplier'] = row['supplier']
        
        return Product(**product_data)
    
    async def _transaction_from_row(self, row: Dict[str, Any]) -> StockTransaction:
        """Convert a database row to a StockTransaction object."""
        transaction_data = {
            'id': uuid.UUID(row['id']),
            'product_id': uuid.UUID(row['product_id']),
            'delta': int(row['delta']),
            'timestamp': row['timestamp'],
        }
        
        # Add optional fields if present
        if row.get('transaction_type') is not None:
            transaction_data['transaction_type'] = row['transaction_type']
        if row.get('user') is not None:
            transaction_data['user'] = row['user']
        if row.get('note') is not None:
            transaction_data['note'] = row['note']
        
        return StockTransaction(**transaction_data)
    
    # StorageAdapter interface implementation
    
    async def save_product(self, product: Product) -> None:
        """
        Save a product to storage.
        
        Args:
            product: The product to save
            
        Raises:
            StorageError: If the product cannot be saved
        """
        await self.connect()
        
        try:
            # Convert Product to dict for database
            product_data = {
                'id': str(product.id),
                'name': product.name,
                'description': product.description,
                'sku': product.sku,
                'price': product.price,
                'quantity': product.quantity,
                'reorder_point': product.reorder_point,
                'reorder_quantity': product.reorder_quantity,
                'category': product.category,
                'supplier': product.supplier,
                'updated_at': datetime.utcnow(),
            }
            
            # Use SQLAlchemy connection
            if self._transaction_conn is not None:
                # We're in a transaction, use the transaction's connection
                conn = self._transaction_conn
                close_conn = False
            else:
                # Create a new connection
                conn = self.engine.connect()
                close_conn = True
            
            try:
                # Use SQLite-specific "upsert"
                upsert_stmt = sqlite_insert(self.products).values(product_data)
                
                # "ON CONFLICT" clause for SQLite (adds 'created_at' only for new rows)
                on_conflict_stmt = upsert_stmt.on_conflict_do_update(
                    index_elements=['id'],
                    set_={
                        'name': product_data['name'],
                        'description': product_data['description'],
                        'sku': product_data['sku'],
                        'price': product_data['price'],
                        'quantity': product_data['quantity'],
                        'reorder_point': product_data['reorder_point'],
                        'reorder_quantity': product_data['reorder_quantity'],
                        'category': product_data['category'],
                        'supplier': product_data['supplier'],
                        'updated_at': product_data['updated_at'],
                    }
                )
                
                conn.execute(on_conflict_stmt)
                
                # Commit if we're not in a transaction
                if not self._transaction_conn:
                    conn.commit()
                
                logger.debug(f"Saved product {product.id}")
                
            except Exception as e:
                # Rollback if we're not in a transaction
                if not self._transaction_conn:
                    conn.rollback()
                logger.exception(f"Error saving product: {e}")
                raise StorageError(f"Failed to save product: {e}") from e
                
            finally:
                if close_conn:
                    conn.close()
                
        except StorageError:
            # Re-raise storage errors
            raise
        except Exception as e:
            logger.exception(f"Error in save_product: {e}")
            raise StorageError(f"Failed to save product: {e}") from e
    
    async def save_transaction(self, transaction: StockTransaction) -> None:
        """
        Save a transaction to storage.
        
        Args:
            transaction: The transaction to save
            
        Raises:
            StorageError: If the transaction cannot be saved
        """
        await self.connect()
        
        try:
            # Convert StockTransaction to dict for database
            transaction_data = {
                'id': str(transaction.id),
                'product_id': str(transaction.product_id),
                'delta': transaction.delta,
                'timestamp': transaction.timestamp,
                'transaction_type': getattr(transaction, 'transaction_type', None),
                'user': getattr(transaction, 'user', None),
                'note': getattr(transaction, 'note', None),
            }
            
            # Use SQLAlchemy connection
            if self._transaction_conn is not None:
                # We're in a transaction, use the transaction's connection
                conn = self._transaction_conn
                close_conn = False
            else:
                # Create a new connection
                conn = self.engine.connect()
                close_conn = True
            
            try:
                # Use SQLite-specific "upsert"
                upsert_stmt = sqlite_insert(self.transactions).values(transaction_data)
                
                # "ON CONFLICT" clause for SQLite
                on_conflict_stmt = upsert_stmt.on_conflict_do_update(
                    index_elements=['id'],
                    set_={
                        'product_id': transaction_data['product_id'],
                        'delta': transaction_data['delta'],
                        'timestamp': transaction_data['timestamp'],
                        'transaction_type': transaction_data['transaction_type'],
                        'user': transaction_data['user'],
                        'note': transaction_data['note'],
                    }
                )
                
                conn.execute(on_conflict_stmt)
                
                # Commit if we're not in a transaction
                if not self._transaction_conn:
                    conn.commit()
                
                logger.debug(f"Saved transaction {transaction.id}")
                
            except Exception as e:
                # Rollback if we're not in a transaction
                if not self._transaction_conn:
                    conn.rollback()
                logger.exception(f"Error saving transaction: {e}")
                raise StorageError(f"Failed to save transaction: {e}") from e
                
            finally:
                if close_conn:
                    conn.close()
                
        except StorageError:
            # Re-raise storage errors
            raise
        except Exception as e:
            logger.exception(f"Error in save_transaction: {e}")
            raise StorageError(f"Failed to save transaction: {e}") from e
    
    async def load_all(self) -> Dict[str, List[Any]]:
        """
        Load all products and transactions from storage.
        
        Returns:
            A dictionary with 'products' and 'transactions' lists
            
        Raises:
            StorageError: If data cannot be loaded from storage
        """
        await self.connect()
        
        try:
            products = []
            transactions = []
            
            # Use SQLAlchemy connection
            if self._transaction_conn is not None:
                # We're in a transaction, use the transaction's connection
                conn = self._transaction_conn
                close_conn = False
            else:
                # Create a new connection
                conn = self.engine.connect()
                close_conn = True
            
            try:
                # Load all products
                result = conn.execute(select(self.products))
                for row in result:
                    products.append(await self._product_from_row(row._asdict()))
                
                # Load all transactions
                result = conn.execute(select(self.transactions))
                for row in result:
                    transactions.append(await self._transaction_from_row(row._asdict()))
                
                logger.debug(f"Loaded {len(products)} products and {len(transactions)} transactions")
                
                return {
                    'products': products,
                    'transactions': transactions,
                }
                
            except Exception as e:
                logger.exception(f"Error loading data: {e}")
                raise StorageError(f"Failed to load data: {e}") from e
                
            finally:
                if close_conn:
                    conn.close()
                
        except StorageError:
            # Re-raise storage errors
            raise
        except Exception as e:
            logger.exception(f"Error in load_all: {e}")
            raise StorageError(f"Failed to load all data: {e}") from e
    
    async def get_product(self, product_id: str) -> Optional[Product]:
        """
        Get a product by ID.
        
        Args:
            product_id: The ID of the product to retrieve
            
        Returns:
            The product if found, None otherwise
            
        Raises:
            StorageError: If the product cannot be loaded
        """
        await self.connect()
        
        try:
            # Use SQLAlchemy connection
            if self._transaction_conn is not None:
                # We're in a transaction, use the transaction's connection
                conn = self._transaction_conn
                close_conn = False
            else:
                # Create a new connection
                conn = self.engine.connect()
                close_conn = True
            
            try:
                # Query for the product
                result = conn.execute(
                    select(self.products).where(self.products.c.id == product_id)
                )
                row = result.fetchone()
                
                if row:
                    return await self._product_from_row(row._asdict())
                else:
                    return None
                
            except Exception as e:
                logger.exception(f"Error getting product: {e}")
                raise StorageError(f"Failed to get product: {e}") from e
                
            finally:
                if close_conn:
                    conn.close()
                
        except StorageError:
            # Re-raise storage errors
            raise
        except Exception as e:
            logger.exception(f"Error in get_product: {e}")
            raise StorageError(f"Failed to get product: {e}") from e
    
    async def get_transactions(self, 
                            product_id: Optional[str] = None,
                            limit: Optional[int] = None) -> List[StockTransaction]:
        """
        Get transactions, optionally filtered by product ID.
        
        Args:
            product_id: If provided, only transactions for this product will be returned
            limit: Maximum number of transactions to return
            
        Returns:
            List of transactions matching the criteria
            
        Raises:
            StorageError: If transactions cannot be loaded
        """
        await self.connect()
        
        try:
            transactions = []
            
            # Use SQLAlchemy connection
            if self._transaction_conn is not None:
                # We're in a transaction, use the transaction's connection
                conn = self._transaction_conn
                close_conn = False
            else:
                # Create a new connection
                conn = self.engine.connect()
                close_conn = True
            
            try:
                # Build the query
                query = select(self.transactions)
                
                # Add product_id filter if specified
                if product_id:
                    query = query.where(self.transactions.c.product_id == product_id)
                
                # Add ordering
                query = query.order_by(self.transactions.c.timestamp.desc())
                
                # Add limit if specified
                if limit is not None and limit > 0:
                    query = query.limit(limit)
                
                # Execute query
                result = conn.execute(query)
                
                # Process results
                for row in result:
                    transactions.append(await self._transaction_from_row(row._asdict()))
                
                return transactions
                
            except Exception as e:
                logger.exception(f"Error getting transactions: {e}")
                raise StorageError(f"Failed to get transactions: {e}") from e
                
            finally:
                if close_conn:
                    conn.close()
                
        except StorageError:
            # Re-raise storage errors
            raise
        except Exception as e:
            logger.exception(f"Error in get_transactions: {e}")
            raise StorageError(f"Failed to get transactions: {e}") from e
    
    # Stream implementation for memory-efficient access
    
    @asynccontextmanager
    async def stream_products(self) -> AsyncIterator[Product]:
        """
        Stream products from storage one at a time to minimize memory usage.
        
        Yields:
            AsyncIterator of Product objects
            
        Raises:
            StorageError: If products cannot be streamed from storage
        """
        await self.connect()
        
        conn = self.engine.connect()
        try:
            # Build the query
            query = select(self.products)
            
            # Execute query
            result = conn.execute(query)
            
            # Yield results one at a time
            for row in result:
                yield await self._product_from_row(row._asdict())
                
        except Exception as e:
            logger.exception(f"Error streaming products: {e}")
            raise StorageError(f"Failed to stream products: {e}") from e
            
        finally:
            conn.close()
    
    @asynccontextmanager
    async def stream_transactions(self) -> AsyncIterator[StockTransaction]:
        """
        Stream transactions from storage one at a time to minimize memory usage.
        
        Yields:
            AsyncIterator of StockTransaction objects
            
        Raises:
            StorageError: If transactions cannot be streamed from storage
        """
        await self.connect()
        
        conn = self.engine.connect()
        try:
            # Build the query
            query = select(self.transactions)
            
            # Execute query
            result = conn.execute(query)
            
            # Yield results one at a time
            for row in result:
                yield await self._transaction_from_row(row._asdict())
                
        except Exception as e:
            logger.exception(f"Error streaming transactions: {e}")
            raise StorageError(f"Failed to stream transactions: {e}") from e
            
        finally:
            conn.close()
    
    # Transaction support
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for handling transactions.
        
        This implementation uses SQLite transactions to ensure atomicity.
        
        Yields:
            None
            
        Raises:
            StorageError: If the transaction fails
        """
        await self.connect()
        
        # We can only have one active transaction at a time
        if self._transaction_conn is not None:
            raise StorageError("Nested transactions are not supported")
        
        conn = self.engine.connect()
        
        try:
            # Start a transaction
            trans = conn.begin()
            self._transaction_conn = conn
            
            try:
                # Execute transaction body
                yield
                
                # Commit transaction
                trans.commit()
                logger.debug("Transaction committed successfully")
                
            except Exception as e:
                # Rollback transaction on error
                trans.rollback()
                logger.error(f"Transaction rolled back: {e}")
                raise StorageError(f"Transaction failed: {e}") from e
                
        finally:
            # Clean up
            if self._transaction_conn is not None:
                self._transaction_conn = None
                conn.close()
    
    # Additional SQLite-specific methods
    
    async def get_schema_version(self) -> int:
        """Get the current database schema version."""
        await self.connect()
        
        conn = self.engine.connect()
        try:
            result = conn.execute(select(self.schema_version.c.version).
                                order_by(self.schema_version.c.version.desc()).
                                limit(1))
            return result.scalar() or 0
                
        except Exception as e:
            logger.exception(f"Error getting schema version: {e}")
            raise StorageError(f"Failed to get schema version: {e}") from e
            
        finally:
            conn.close()
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query and return the results.
        
        This method is provided for advanced usage and diagnostics.
        
        Args:
            query: SQL query to execute
            params: Optional parameters for the query
            
        Returns:
            List of dictionaries representing the query results
            
        Raises:
            StorageError: If the query fails
        """
        await self.connect()
        
        conn = self.engine.connect()
        try:
            result = conn.execute(sa.text(query), params or {})
            
            # Convert rows to dictionaries
            rows = []
            for row in result:
                rows.append({key: value for key, value in zip(result.keys(), row)})
                
            return rows
                
        except Exception as e:
            logger.exception(f"Error executing query: {e}")
            raise StorageError(f"Failed to execute query: {e}") from e
            
        finally:
            conn.close()


# Example usage
async def example_sqlite_adapter():
    """Example showing how to use the SQLite storage adapter."""
    import uuid
    from datetime import datetime
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create adapter
    db_path = Path("./data/inventory.db")
    adapter = SQLiteAdapter(db_path, echo=True)
    
    try:
        # Connect and apply migrations
        await adapter.connect()
        
        # Print schema version
        version = await adapter.get_schema_version()
        print(f"Database schema version: {version}")
        
        # Create a sample product
        product = Product(
            id=uuid.uuid4(),
            name="Sample Product",
            description="A sample product for testing",
            sku="SAMPLE-001",
            price=19.99,
            quantity=100,
            reorder_point=20,
            reorder_quantity=50,
            category="Test",
            supplier="Test Supplier"
        )
        
        # Save the product
        await adapter.save_product(product)
        print(f"Saved product: {product.id}")
        
        # Create a transaction
        transaction = StockTransaction(
            id=uuid.uuid4(),
            product_id=product.id,
            delta=-5,
            timestamp=datetime.utcnow(),
            transaction_type="adjustment",
            user="system",
            note="Test transaction"
        )
        
        # Use a transaction to update the product quantity and save the transaction
        async with adapter.transaction():
            # Update product quantity
            product.quantity += transaction.delta
            
            # Save updated product
            await adapter.save_product(product)
            
            # Save transaction
            await adapter.save_transaction(transaction)
            
            print(f"Updated product quantity to {product.quantity} and saved transaction in a single transaction")
        
        # Load and print all data
        data = await adapter.load_all()
        print(f"Loaded {len(data['products'])} products and {len(data['transactions'])} transactions")
        
        # Get specific product
        retrieved_product = await adapter.get_product(str(product.id))
        print(f"Retrieved product: {retrieved_product.name}, quantity: {retrieved_product.quantity}")
        
        # Get transactions for the product
        transactions = await adapter.get_transactions(str(product.id))
        print(f"Retrieved {len(transactions)} transactions for product {product.id}")
        
    finally:
        # Close connection
        await adapter.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_sqlite_adapter())