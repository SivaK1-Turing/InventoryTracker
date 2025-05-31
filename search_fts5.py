"""
search_fts5.py - SQLite FTS5 implementation for full-text product search

This module integrates SQLite's FTS5 extension for efficient full-text searching
of product names and notes.
"""
import sqlite3
import os
import time
import uuid
import random
from typing import List, Dict, Any, Tuple, Optional, Union
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SQLiteFTS5Search:
    """
    Implements full-text search for products using SQLite's FTS5 extension.
    
    FTS5 is an advanced full-text search engine built into SQLite that provides
    efficient text search with features like stemming, tokenization, and ranking.
    """
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the FTS5 search engine.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_db_dir()
        self.conn = self._get_connection()
        self.setup_complete = False
        
        # Verify FTS5 availability
        try:
            self.conn.execute("SELECT sqlite_version()")
            self.conn.execute("SELECT fts5(?)", ('dummy',))
            logger.info("SQLite FTS5 is available")
        except sqlite3.OperationalError:
            logger.error("SQLite FTS5 is not available - this implementation will not work")
            raise ValueError("SQLite FTS5 extension is not available in your SQLite installation")
            
        # Set up the database schema
        self._setup_schema()
        
    def _ensure_db_dir(self):
        """Ensure the directory for the database file exists."""
        db_dir = self.db_path.parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
            
    def _get_connection(self):
        """Get SQLite connection with proper settings for FTS5."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _setup_schema(self):
        """Set up the database schema including the FTS5 virtual table."""
        with self.conn:
            # Create products table if it doesn't exist
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    name TEXT NOT NULL,
                    notes TEXT
                )
            """)
            
            # Create FTS5 virtual table
            try:
                self.conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
                        name, notes,
                        content='products',
                        content_rowid='rowid',
                        tokenize='porter unicode61'
                    )
                """)
                
                # Create triggers to keep the FTS index in sync with the products table
                self.conn.executescript("""
                    -- Trigger for inserts
                    CREATE TRIGGER IF NOT EXISTS products_ai AFTER INSERT ON products BEGIN
                        INSERT INTO products_fts(rowid, name, notes)
                        VALUES (new.rowid, new.name, new.notes);
                    END;
                    
                    -- Trigger for updates
                    CREATE TRIGGER IF NOT EXISTS products_au AFTER UPDATE ON products BEGIN
                        INSERT INTO products_fts(products_fts, rowid, name, notes)
                        VALUES ('delete', old.rowid, old.name, old.notes);
                        INSERT INTO products_fts(rowid, name, notes)
                        VALUES (new.rowid, new.name, new.notes);
                    END;
                    
                    -- Trigger for deletes
                    CREATE TRIGGER IF NOT EXISTS products_ad AFTER DELETE ON products BEGIN
                        INSERT INTO products_fts(products_fts, rowid, name, notes)
                        VALUES ('delete', old.rowid, old.name, old.notes);
                    END;
                """)
                
                self.setup_complete = True
                logger.info("FTS5 schema setup complete")
            except sqlite3.OperationalError as e:
                logger.error(f"Error setting up FTS5: {e}")
                raise
            
    def rebuild_index(self):
        """
        Rebuild the FTS5 index from scratch.
        
        This is useful if the index gets out of sync with the main table.
        """
        with self.conn:
            # Delete everything from the FTS table
            self.conn.execute("DELETE FROM products_fts")
            
            # Repopulate the FTS table from the products table
            self.conn.execute("""
                INSERT INTO products_fts(rowid, name, notes)
                SELECT rowid, name, notes FROM products
            """)
            
        logger.info("FTS5 index rebuilt")
            
    def add_product(self, product: Dict[str, Any]):
        """
        Add a product to the search index.
        
        Args:
            product: Product data dictionary
        """
        product_id = str(product.get('id', uuid.uuid4()))
        name = product.get('name', '')
        notes = product.get('notes', '')
        
        # Store as JSON for flexible schema
        product_data = json.dumps(product)
        
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO products (id, data, name, notes) VALUES (?, ?, ?, ?)",
                (product_id, product_data, name, notes)
            )
            
    def add_products(self, products: List[Dict[str, Any]]):
        """
        Add multiple products to the search index in a single transaction.
        
        Args:
            products: List of product data dictionaries
        """
        with self.conn:
            for product in products:
                product_id = str(product.get('id', uuid.uuid4()))
                name = product.get('name', '')
                notes = product.get('notes', '')
                
                # Store as JSON for flexible schema
                product_data = json.dumps(product)
                
                self.conn.execute(
                    "INSERT OR REPLACE INTO products (id, data, name, notes) VALUES (?, ?, ?, ?)",
                    (product_id, product_data, name, notes)
                )
                
    def delete_product(self, product_id: str):
        """
        Remove a product from the search index.
        
        Args:
            product_id: ID of the product to remove
        """
        with self.conn:
            self.conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
            
    def search(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for products matching the query text.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            
        Returns:
            List of matching product dictionaries
        """
        # Prepare the FTS5 query
        fts_query = self._format_fts5_query(query)
        
        # Execute the search with ranking
        cursor = self.conn.execute("""
            SELECT p.id, p.data, p.name, p.notes,
                   rank FROM products_fts
            JOIN products p ON products_fts.rowid = p.rowid
            WHERE products_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (fts_query, limit))
        
        # Process results
        results = []
        for row in cursor:
            try:
                product_data = json.loads(row['data'])
                results.append(product_data)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode product data for ID {row['id']}")
                
        return results
    
    def _format_fts5_query(self, query: str) -> str:
        """
        Format a user query string for FTS5.
        
        Args:
            query: User input query
            
        Returns:
            Formatted FTS5 query string
        """
        # Basic sanitization
        query = query.strip()
        
        # Handle empty queries
        if not query:
            return ''
            
        # For simple queries, search for the terms in both columns
        # The {name} and {notes} syntax is FTS5's column filter syntax
        terms = query.split()
        formatted_terms = []
        
        for term in terms:
            # Add wildcards for partial matching
            if len(term) >= 3:  # Only add wildcards for terms of sufficient length
                formatted_terms.append(f"{term}*")
            else:
                formatted_terms.append(term)
                
        formatted_query = ' '.join(formatted_terms)
        
        # Search in both name and notes columns with higher weight for name
        return f"name: {formatted_query} OR notes: {formatted_query}"
        
    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a product by ID.
        
        Args:
            product_id: ID of the product to retrieve
            
        Returns:
            Product data dictionary or None if not found
        """
        cursor = self.conn.execute(
            "SELECT data FROM products WHERE id = ?",
            (product_id,)
        )
        row = cursor.fetchone()
        
        if row:
            try:
                return json.loads(row['data'])
            except json.JSONDecodeError:
                logger.warning(f"Could not decode product data for ID {product_id}")
                
        return None
    
    def get_product_count(self) -> int:
        """
        Get the total number of products in the index.
        
        Returns:
            Count of products
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM products")
        return cursor.fetchone()[0]
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def generate_sample_data(count: int = 5000) -> List[Dict[str, Any]]:
    """
    Generate sample product data for testing.
    
    Args:
        count: Number of products to generate
        
    Returns:
        List of product dictionaries
    """
    products = []
    
    # Product name components for realistic test data
    categories = ["Widget", "Gadget", "Tool", "Device", "Component", "Module", "System", "Unit"]
    adjectives = ["Smart", "Pro", "Advanced", "Basic", "Premium", "Ultra", "Compact", "Portable", 
                  "Digital", "Analog", "Wireless", "Solar", "Eco", "Heavy-Duty", "Industrial"]
    models = ["X", "S", "Pro", "Plus", "Lite", "Mini", "Max", "Elite", "Prime", "Core"]
    materials = ["Steel", "Aluminum", "Titanium", "Carbon", "Plastic", "Composite", "Wood", "Metal"]
    
    # Generate notes components
    features = ["Durable", "Lightweight", "Energy-efficient", "Bluetooth-enabled", "Waterproof", 
                "Shockproof", "Customizable", "Programmable", "Foldable", "Rechargeable",
                "Multi-functional", "Modular", "Low-maintenance", "High-performance"]
    
    usages = ["Perfect for home use", "Ideal for industrial applications", 
              "Great for outdoor activities", "Suitable for professional use",
              "Designed for beginners", "Engineered for experts", 
              "Built for extreme conditions", "Made for everyday tasks"]
    
    for i in range(count):
        # Generate a unique product name
        category = random.choice(categories)
        adjective = random.choice(adjectives)
        model = random.choice(models)
        
        # Some products include material in name
        if random.random() < 0.3:
            material = random.choice(materials)
            name = f"{adjective} {material} {category} {model}"
        else:
            name = f"{adjective} {category} {model}"
        
        # Sometimes add a number to the model
        if random.random() < 0.5:
            name += f" {random.randint(1, 9)}{random.randint(0, 9)}{random.randint(0, 9)}"
        
        # Generate notes with varying length and content
        notes_parts = []
        # Add 1-3 features
        feature_count = random.randint(1, 3)
        selected_features = random.sample(features, feature_count)
        notes_parts.extend(selected_features)
        
        # Add a usage suggestion
        notes_parts.append(random.choice(usages))
        
        # Add manufacturing info to some products
        if random.random() < 0.4:
            notes_parts.append(f"Manufactured in {random.choice(['USA', 'China', 'Germany', 'Japan', 'Korea', 'Taiwan'])}")
        
        # Add warranty info to some products
        if random.random() < 0.3:
            warranty = random.choice(["1 year", "2 years", "3 years", "5 years", "limited lifetime"])
            notes_parts.append(f"Comes with {warranty} warranty")
        
        # Join notes parts with punctuation
        notes = ". ".join(notes_parts) + "."
        
        # Create the product dictionary
        product = {
            "id": str(uuid.uuid4()),
            "name": name,
            "sku": f"{category[:3]}-{model}-{random.randint(100, 999)}".upper(),
            "price": round(random.uniform(9.99, 499.99), 2),
            "current_stock": random.randint(0, 100),
            "reorder_level": random.randint(5, 25),
            "notes": notes,
            "tags": random.sample(["electronics", "hardware", "tools", "industrial", 
                                  "home", "office", "outdoor", "professional"], 
                                 random.randint(1, 3))
        }
        
        products.append(product)
        
    return products


def benchmark_fts5(db_path: str = "test_fts5.db", item_count: int = 5000):
    """
    Benchmark SQLite FTS5 search performance.
    
    Args:
        db_path: Path to the database file
        item_count: Number of test items to generate
    """
    print(f"--- SQLite FTS5 Benchmark ({item_count} items) ---")
    
    # Remove existing test database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Setup timing
    setup_start = time.time()
    
    # Initialize FTS5 search
    fts = SQLiteFTS5Search(db_path)
    
    # Generate and add test data
    print(f"Generating {item_count} test products...")
    products = generate_sample_data(item_count)
    
    # Add products
    print(f"Adding {item_count} products to FTS5 index...")
    fts.add_products(products)
    
    setup_time = time.time() - setup_start
    print(f"Setup time: {setup_time:.3f} seconds")
    
    # Size of database
    db_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
    print(f"Database size: {db_size:.2f} MB")
    
    # Run search benchmarks
    print("\nRunning search benchmarks:")
    
    test_queries = [
        "smart widget",        # Common words
        "titanium device",     # Material + category
        "bluetooth wireless",  # Features
        "pro portable",        # Modifiers
        "waterproof outdoor",  # Specific feature + usage
        "japan warranty",      # Manufacturing + warranty
        "x 100",               # Model + number
    ]
    
    for query in test_queries:
        # Warm-up run (not timed)
        fts.search(query, limit=100)
        
        # Timed runs
        times = []
        for _ in range(5):  # Run 5 times for average
            start_time = time.time()
            results = fts.search(query, limit=100)
            query_time = time.time() - start_time
            times.append(query_time)
            
        avg_time = sum(times) / len(times)
        print(f"Query: '{query}' - Found {len(results)} results - Avg time: {avg_time*1000:.2f} ms")
        
    # Close the database
    fts.close()


if __name__ == "__main__":
    # Run the FTS5 benchmark
    benchmark_fts5()