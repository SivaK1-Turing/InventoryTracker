"""
commands/list_products.py - Command for listing and filtering products with streaming export

This module provides CLI commands for listing products with various sorting and filtering
options, including efficient streaming export for large datasets.
"""
import typer
from typing import List, Dict, Any, Optional, Tuple, Union, AsyncGenerator, BinaryIO
import re
import asyncio
import csv
import json
import sys
from enum import Enum
from pathlib import Path
import io
from datetime import datetime
from contextlib import asynccontextmanager
from functools import partial
import signal
from inventorytracker.search import SearchQuery

# Import appropriate search implementation
try:
    from inventorytracker.search_fts5 import SQLiteFTS5Search as SearchEngine
    SEARCH_IMPLEMENTATION = "fts5"
except ImportError:
    try:
        from inventorytracker.search_trigram import TrigramSearch as SearchEngine
        SEARCH_IMPLEMENTATION = "trigram"
    except ImportError:
        from inventorytracker.search import SearchQuery
        SearchEngine = None
        SEARCH_IMPLEMENTATION = "basic"

# For pretty printing in interactive mode
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

app = typer.Typer(help="List and filter products in inventory")

if RICH_AVAILABLE:
    console = Console()
else:
    # Simple console fallback
    class SimpleConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = SimpleConsole()


class OutputFormat(str, Enum):
    TABLE = "table"
    JSON = "json"
    CSV = "csv"


class ExportFormat(str, Enum):
    """Export formats for streaming large datasets."""
    CSV = "csv"
    JSONL = "jsonl"  # JSON Lines format (one JSON object per line)


class SortOrderParser:
    """Parser for sort order specifications like 'name:asc,stock:desc'."""
    
    SORT_PATTERN = re.compile(r'([a-zA-Z_]+)(?::(asc|desc))?')
    
    # Map of CLI field names to model field names
    FIELD_MAPPING = {
        # CLI name -> Model field name
        "name": "name",
        "sku": "sku",
        "price": "price", 
        "stock": "current_stock",
        "reorder": "reorder_level",
        "created": "created_at",
        "updated": "updated_at",
        "id": "id"
    }
    
    @classmethod
    def parse(cls, sort_spec: str) -> List[Tuple[str, bool]]:
        """Parse a sort specification string into a list of (field, descending) tuples."""
        if not sort_spec:
            return []
            
        order_by_clauses = []
        
        # Split by commas
        for part in sort_spec.split(','):
            part = part.strip()
            if not part:
                continue
                
            # Parse field and direction
            match = cls.SORT_PATTERN.fullmatch(part)
            if not match:
                raise ValueError(f"Invalid sort specification: '{part}'. "
                                 f"Expected format: 'field:direction' or just 'field'")
            
            field, direction = match.groups()
            
            # Validate field
            if field not in cls.FIELD_MAPPING:
                valid_fields = ", ".join(sorted(cls.FIELD_MAPPING.keys()))
                raise ValueError(f"Unknown sort field: '{field}'. "
                                 f"Valid fields are: {valid_fields}")
            
            # Map CLI field to model field
            model_field = cls.FIELD_MAPPING[field]
            
            # Set direction (default to ascending if not specified)
            is_descending = direction == "desc" if direction else False
            
            order_by_clauses.append((model_field, is_descending))
            
        return order_by_clauses


class SearchWrapper:
    """
    Wrapper class that provides a unified interface to different search implementations.
    """
    
    def __init__(self):
        """Initialize the appropriate search engine."""
        self.engine_type = SEARCH_IMPLEMENTATION
        
        if self.engine_type == "fts5":
            # SQLite FTS5 implementation
            db_path = Path("./data/fts5_index.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = SearchEngine(db_path)
            if RICH_AVAILABLE:
                console.print(f"[green]Using SQLite FTS5 search engine[/green]")
            else:
                console.print("Using SQLite FTS5 search engine")
            
        elif self.engine_type == "trigram":
            # Trigram implementation
            index_path = Path("./data/trigram_index.json")
            index_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = SearchEngine(index_path)
            if RICH_AVAILABLE:
                console.print(f"[green]Using Trigram search engine[/green]")
            else:
                console.print("Using Trigram search engine")
            
        else:
            # Fall back to basic search
            self.engine = None
            if RICH_AVAILABLE:
                console.print(f"[yellow]Using basic search (no advanced text search capabilities)[/yellow]")
            else:
                console.print("Using basic search (no advanced text search capabilities)")
        
    def search(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search for products matching the text query."""
        if not query:
            return []
            
        if self.engine:
            # Use advanced search implementation
            return self.engine.search(query, limit=limit)
        else:
            # Use basic search
            from inventorytracker.store import get_store
            store = get_store()
            search = SearchQuery()
            search.product(query).limit(limit)
            results = search.execute()
            return results.get('products', [])
    
    async def stream_all_products(self, chunk_size: int = 100) -> AsyncGenerator[List[Any], None]:
        """
        Stream all products in chunks to avoid loading everything into memory at once.
        
        Args:
            chunk_size: Number of products to retrieve in each chunk
            
        Yields:
            Chunks of products
        """
        # Get total count for progress reporting
        total_count = await self.get_product_count()
        offset = 0
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task(f"Exporting products", total=total_count)
                
                while True:
                    chunk = await self._get_product_chunk(offset, chunk_size)
                    if not chunk:
                        break
                        
                    progress.update(task, advance=len(chunk))
                    yield chunk
                    offset += chunk_size
                    
                    # Give other tasks a chance to run
                    await asyncio.sleep(0)
        else:
            # Simple progress indication without rich
            while True:
                chunk = await self._get_product_chunk(offset, chunk_size)
                if not chunk:
                    break
                    
                # Simple progress indication
                if offset % 1000 == 0:
                    print(f"Exported {offset}/{total_count} products", file=sys.stderr)
                    
                yield chunk
                offset += chunk_size
                
                # Give other tasks a chance to run
                await asyncio.sleep(0)
    
    async def _get_product_chunk(self, offset: int, limit: int) -> List[Any]:
        """
        Get a chunk of products.
        
        Args:
            offset: Starting offset
            limit: Maximum number of products to return
            
        Returns:
            List of products
        """
        # Implementation varies based on engine type
        if self.engine_type == "fts5":
            # Use SQLite pagination
            # This needs to be implemented with proper async support for SQLite
            # For now, we'll use an executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._get_products_from_fts5,
                offset,
                limit
            )
        elif self.engine_type == "trigram":
            # Use in-memory pagination from trigram index
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._get_products_from_trigram,
                offset,
                limit
            )
        else:
            # Use basic search with pagination
            from inventorytracker.store import get_store
            store = get_store()
            search = SearchQuery()
            search.offset(offset).limit(limit)
            
            # Execute in an executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: search.execute()
            )
            return results.get('products', [])
    
    def _get_products_from_fts5(self, offset: int, limit: int) -> List[Any]:
        """Get a chunk of products from FTS5 index."""
        # This is a simplified implementation - adapt to your actual FTS5 interface
        if hasattr(self.engine, 'conn'):
            cursor = self.engine.conn.execute(
                "SELECT id, data FROM products LIMIT ? OFFSET ?",
                (limit, offset)
            )
            results = []
            for row in cursor:
                try:
                    product_data = json.loads(row[1])
                    results.append(product_data)
                except (json.JSONDecodeError, KeyError):
                    pass
            return results
        return []
    
    def _get_products_from_trigram(self, offset: int, limit: int) -> List[Any]:
        """Get a chunk of products from trigram index."""
        # This is a simplified implementation - adapt to your actual Trigram interface
        if hasattr(self.engine, 'products'):
            product_ids = list(self.engine.products.keys())[offset:offset+limit]
            return [self.engine.products[pid] for pid in product_ids]
        return []
    
    async def get_product_count(self) -> int:
        """Get the total number of products."""
        if self.engine_type == "fts5" and hasattr(self.engine, 'conn'):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.engine.conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            )
        elif self.engine_type == "trigram" and hasattr(self.engine, 'products'):
            return len(self.engine.products)
        else:
            # Use basic search count
            from inventorytracker.store import get_store
            store = get_store()
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: len(store.get_all_products())
            )


# Create a global search engine instance
search_engine = SearchWrapper()


@asynccontextmanager
async def handle_export_cancellation():
    """Context manager to handle graceful cancellation of exports."""
    # Set up cancellation handling
    loop = asyncio.get_event_loop()
    signal_handler = lambda signum, frame: loop.stop()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        yield
    finally:
        # Clean-up: restore default signal handlers
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        
        if RICH_AVAILABLE:
            console.print("\n[yellow]Export process completed or interrupted.[/yellow]")
        else:
            print("\nExport process completed or interrupted.", file=sys.stderr)


class CSVWriter:
    """Async wrapper for CSV writing with chunked output."""
    
    def __init__(self, output: BinaryIO = sys.stdout.buffer):
        """
        Initialize CSV writer.
        
        Args:
            output: Output binary stream (default: stdout)
        """
        self.output = output
        self.writer = None
        self.text_buffer = io.StringIO()
        self.csv_writer = None
        
    async def write_header(self, fields: List[str]):
        """Write CSV header row."""
        # Create CSV writer if needed
        if self.csv_writer is None:
            self.csv_writer = csv.writer(self.text_buffer)
            
        # Write header
        self.csv_writer.writerow(fields)
        
        # Flush to output
        await self._flush()
        
    async def write_products(self, products: List[Any]):
        """
        Write products as CSV rows.
        
        Args:
            products: List of product objects to write
        """
        if not products:
            return
            
        if self.csv_writer is None:
            self.csv_writer = csv.writer(self.text_buffer)
            
        # Convert products to rows and write
        for product in products:
            row = [
                str(getattr(product, "id", "")),
                getattr(product, "name", ""),
                getattr(product, "sku", ""),
                f"{getattr(product, 'price', 0):.2f}",
                getattr(product, "current_stock", 0),
                getattr(product, "reorder_level", 0),
                ",".join(getattr(product, "tags", []) or []),
                getattr(product, "notes", "")
            ]
            self.csv_writer.writerow(row)
            
            # Flush periodically to avoid building up too much in memory
            if self.text_buffer.tell() > 8192:  # 8KB buffer
                await self._flush()
                
        # Ensure any remaining data is flushed
        await self._flush()
        
    async def _flush(self):
        """Flush buffered CSV data to output."""
        data = self.text_buffer.getvalue()
        if data:
            # Reset buffer
            self.text_buffer = io.StringIO()
            self.csv_writer = csv.writer(self.text_buffer)
            
            # Write data to output
            self.output.write(data.encode('utf-8'))
            self.output.flush()
            
            # Let other tasks run
            await asyncio.sleep(0)


class JSONLWriter:
    """Async wrapper for JSON Lines writing with chunked output."""
    
    def __init__(self, output: BinaryIO = sys.stdout.buffer):
        """
        Initialize JSONL writer.
        
        Args:
            output: Output binary stream (default: stdout)
        """
        self.output = output
        self.buffer = bytearray()
        
    async def write_products(self, products: List[Any]):
        """
        Write products as JSON Lines.
        
        Args:
            products: List of product objects to write
        """
        if not products:
            return
            
        # Convert each product to JSON and write as a line
        for product in products:
            # Convert to dict if needed
            if hasattr(product, 'dict'):
                product_dict = product.dict()
            else:
                # Simple conversion
                product_dict = {
                    'id': str(getattr(product, 'id', '')),
                    'name': getattr(product, 'name', ''),
                    'sku': getattr(product, 'sku', ''),
                    'price': float(getattr(product, 'price', 0)),
                    'current_stock': getattr(product, 'current_stock', 0),
                    'reorder_level': getattr(product, 'reorder_level', 0),
                    'tags': getattr(product, 'tags', []),
                    'notes': getattr(product, 'notes', '')
                }
                
            # Serialize to JSON line
            json_line = json.dumps(product_dict, ensure_ascii=False)
            self.buffer.extend(f"{json_line}\n".encode('utf-8'))
            
            # Flush if buffer is getting large
            if len(self.buffer) > 8192:  # 8KB buffer
                await self._flush()
                
        # Ensure any remaining data is flushed
        await self._flush()
        
    async def _flush(self):
        """Flush buffered JSON Lines data to output."""
        if self.buffer:
            # Write and clear buffer
            self.output.write(self.buffer)
            self.output.flush()
            self.buffer = bytearray()
            
            # Let other tasks run
            await asyncio.sleep(0)


async def export_products(
    format: ExportFormat,
    chunk_size: int = 100,
    output: BinaryIO = sys.stdout.buffer
):
    """
    Export products in streaming fashion to avoid high memory usage.
    
    Args:
        format: Export format (CSV or JSONL)
        chunk_size: Number of products to process in each chunk
        output: Output binary stream
    """
    async with handle_export_cancellation():
        if format == ExportFormat.CSV:
            writer = CSVWriter(output)
            
            # Write header
            await writer.write_header([
                "ID", "Name", "SKU", "Price", "Stock", "Reorder Level", "Tags", "Notes"
            ])
            
            # Stream products in chunks
            async for chunk in search_engine.stream_all_products(chunk_size):
                await writer.write_products(chunk)
                
        elif format == ExportFormat.JSONL:
            writer = JSONLWriter(output)
            
            # Stream products in chunks
            async for chunk in search_engine.stream_all_products(chunk_size):
                await writer.write_products(chunk)


@app.command("list")
def list_products(
    # Search/filter parameters
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Text search across name and SKU"),
    text_search: Optional[str] = typer.Option(None, "--text", "-t", help="Advanced text search (uses FTS5/trigrams)"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", help="Filter by tag (can be used multiple times)"),
    min_stock: Optional[int] = typer.Option(None, "--min-stock", help="Minimum stock level"),
    max_stock: Optional[int] = typer.Option(None, "--max-stock", help="Maximum stock level"),
    below_reorder: Optional[bool] = typer.Option(False, "--below-reorder", "-r", help="Show only products below reorder level"),
    
    # Sorting parameters
    sort_by: Optional[str] = typer.Option(
        "name:asc", "--sort-by", "-s", 
        help="Sort order (comma-separated list of field:direction, e.g., 'name:asc,stock:desc')"
    ),
    
    # Pagination parameters
    page_size: int = typer.Option(10, "--page-size", "-p", help="Number of results per page"),
    
    # Output format parameters
    format: OutputFormat = typer.Option(
        OutputFormat.TABLE, "--format", "-f", 
        help="Output format (table, json, or csv)"
    ),
    
    # Export parameters
    export: Optional[ExportFormat] = typer.Option(
        None, "--export", "-e",
        help="Export all products in streaming mode (csv or jsonl)"
    ),
    export_chunk_size: int = typer.Option(
        100, "--export-chunk-size",
        help="Number of products to process in each export chunk"
    ),
):
    """
    List and filter products in the inventory with efficient export capability.
    
    Examples:
      list-products --query widget                    # Basic search
      list-products --below-reorder --format json     # Format output as JSON
      list-products --sort-by price:desc,name:asc     # Sort by multiple fields
      list-products --export csv > products.csv       # Export all products as CSV
      list-products --export jsonl > products.jsonl   # Export as JSON Lines
    """
    # Handle export mode (async streaming of all products)
    if export:
        # Run the async export
        asyncio.run(export_products(
            format=export,
            chunk_size=export_chunk_size
        ))
        return
    
    # Regular (non-export) mode
    # Use advanced text search if available
    if text_search:
        products = search_engine.search(text_search, limit=page_size)
        
        # Apply additional filters if needed
        filtered_products = _apply_filters(
            products, 
            tag=tag, 
            min_stock=min_stock, 
            max_stock=max_stock, 
            below_reorder=below_reorder
        )
            
        # Manual sorting
        try:
            order_by_clauses = SortOrderParser.parse(sort_by)
            products = _sort_products(filtered_products, order_by_clauses)
        except ValueError as e:
            if RICH_AVAILABLE:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
            else:
                console.print(f"Error: {str(e)}")
            raise typer.Exit(1)
            
    else:
        # Use standard search with SearchQuery
        search = SearchQuery()
        
        # Apply filters
        if query:
            search.product(query)
            
        if tag:
            for t in tag:
                search.tag(t)
                
        if min_stock is not None:
            search.min_stock(min_stock)
            
        if max_stock is not None:
            search.max_stock(max_stock)
            
        if below_reorder:
            search.below_reorder()
        
        # Apply sorting
        try:
            order_by_clauses = SortOrderParser.parse(sort_by)
            for field, descending in order_by_clauses:
                search.order_by(field, descending)
        except ValueError as e:
            if RICH_AVAILABLE:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
            else:
                console.print(f"Error: {str(e)}")
            raise typer.Exit(1)
        
        # Apply pagination
        search.limit(page_size)
        
        # Execute search
        results = search.execute()
        products = results['products']
    
    # Format and output results (non-export mode)
    if format == OutputFormat.JSON:
        _output_json(products)
    elif format == OutputFormat.CSV:
        _output_csv(products)
    else:  # TABLE
        _output_table(products)


def _apply_filters(
    products: List[Any],
    tag: Optional[List[str]] = None,
    min_stock: Optional[int] = None,
    max_stock: Optional[int] = None,
    below_reorder: bool = False
) -> List[Any]:
    """Apply additional filters to products."""
    if not (tag or min_stock is not None or max_stock is not None or below_reorder):
        return products
        
    filtered = []
    for product in products:
        # Check tags
        if tag and not any(t in getattr(product, 'tags', []) for t in tag):
            continue
            
        # Check stock levels
        stock = getattr(product, 'current_stock', 0)
        if min_stock is not None and stock < min_stock:
            continue
        if max_stock is not None and stock > max_stock:
            continue
        
        # Check reorder level
        if below_reorder:
            reorder = getattr(product, 'reorder_level', 0)
            if stock > reorder:
                continue
                
        # Product passed all filters
        filtered.append(product)
        
    return filtered


def _sort_products(
    products: List[Any],
    order_by_clauses: List[Tuple[str, bool]]
) -> List[Any]:
    """Sort products by specified fields."""
    # Create a copy to avoid modifying the original list
    result = list(products)
    
    # Sort by each field in reverse order
    for field, descending in reversed(order_by_clauses):
        result.sort(
            key=lambda x: getattr(x, field, 0) if hasattr(x, field) else 0,
            reverse=descending
        )
        
    return result


def _output_table(products: List[Any]):
    """Output products as a rich table."""
    if not products:
        console.print("No products found matching your criteria.")
        return
    
    if RICH_AVAILABLE:
        table = Table(show_header=True, header_style="bold")
        
        # Add columns
        table.add_column("Name")
        table.add_column("SKU")
        table.add_column("Price", justify="right")
        table.add_column("Stock", justify="right")
        table.add_column("Reorder Level", justify="right")
        table.add_column("Tags")
        
        # Add rows
        for product in products:
            # Format tags
            tags = ", ".join(getattr(product, "tags", []) or [])
            
            # Format price
            price = f"${getattr(product, 'price', 0):.2f}"
            
            # Highlight stock if below reorder level
            stock = str(getattr(product, "current_stock", 0))
            reorder_level = str(getattr(product, "reorder_level", 0))
            
            if getattr(product, "current_stock", 0) <= getattr(product, "reorder_level", 0):
                stock = f"[bold red]{stock}[/bold red]"
            
            table.add_row(
                getattr(product, "name", ""),
                getattr(product, "sku", ""),
                price,
                stock,
                reorder_level,
                tags
            )
        
        # Show count as footer
        console.print(table)
        console.print(f"[blue]Showing {len(products)} products[/blue]")
    else:
        # Simple table output without rich
        print(f"{'Name':<30} {'SKU':<15} {'Price':>10} {'Stock':>10} {'Reorder':>10} {'Tags':<30}")
        print("-" * 110)
        
        for product in products:
            name = getattr(product, "name", "")
            sku = getattr(product, "sku", "")
            price = f"${getattr(product, 'price', 0):.2f}"
            stock = str(getattr(product, "current_stock", 0))
            reorder = str(getattr(product, "reorder_level", 0))
            tags = ", ".join(getattr(product, "tags", []) or [])
            
            print(f"{name:<30} {sku:<15} {price:>10} {stock:>10} {reorder:>10} {tags:<30}")
            
        print(f"\nShowing {len(products)} products")


def _output_json(products: List[Any]):
    """Output products as JSON."""
    # Convert products to dictionaries for JSON serialization
    product_dicts = []
    for product in products:
        # Use product.dict() for Pydantic models or convert manually
        if hasattr(product, "dict"):
            product_dict = product.dict()
        else:
            # Manual conversion for non-Pydantic objects
            product_dict = {
                "id": str(getattr(product, "id", "")),
                "name": getattr(product, "name", ""),
                "sku": getattr(product, "sku", ""),
                "price": float(getattr(product, "price", 0)),
                "current_stock": getattr(product, "current_stock", 0),
                "reorder_level": getattr(product, "reorder_level", 0),
                "tags": getattr(product, "tags", [])
            }
        product_dicts.append(product_dict)
    
    # Build response 
    response = {
        "products": product_dicts,
        "count": len(product_dicts),
        "search_engine": search_engine.engine_type
    }
    
    # Output JSON
    json_str = json.dumps(response, indent=2)
    console.print(json_str)


def _output_csv(products: List[Any]):
    """Output products as CSV."""
    if not products:
        console.print("No products found matching your criteria.")
        return
    
    # Create CSV writer
    writer = csv.writer(sys.stdout)
    
    # Write header
    writer.writerow([
        "ID", "Name", "SKU", "Price", "Stock", "Reorder Level", "Tags"
    ])
    
    # Write product rows
    for product in products:
        writer.writerow([
            str(getattr(product, "id", "")),
            getattr(product, "name", ""),
            getattr(product, "sku", ""),
            f"{getattr(product, 'price', 0):.2f}",
            getattr(product, "current_stock", 0),
            getattr(product, "reorder_level", 0),
            ",".join(getattr(product, "tags", []) or [])
        ])


if __name__ == "__main__":
    app()