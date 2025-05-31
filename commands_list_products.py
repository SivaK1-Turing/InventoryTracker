"""
commands/list_products.py - Command for listing and filtering products with keyset pagination

This module provides the CLI commands for listing products with various
sorting and filtering options, including efficient keyset pagination.
"""
import typer
from typing import List, Dict, Any, Optional, Tuple, Union
import re
import base64
import json
import hmac
import hashlib
import time
from enum import Enum
from inventorytracker.search import SearchQuery
from inventorytracker.config import get_config
import csv
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="List and filter products in inventory")
console = Console()


class OutputFormat(str, Enum):
    TABLE = "table"
    JSON = "json"
    CSV = "csv"


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
        """
        Parse a sort specification string into a list of (field, descending) tuples.
        
        Args:
            sort_spec: Comma-separated list of field:direction pairs (e.g., "name:asc,stock:desc")
                       Direction is optional and defaults to "asc"
        
        Returns:
            List of (field_name, is_descending) tuples suitable for order_by() calls
            
        Raises:
            ValueError: If the sort specification has invalid syntax
        """
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


class CursorTokenManager:
    """
    Manages pagination cursor tokens with secure encoding and validation.
    
    This class handles the creation, validation, and parsing of cursor tokens
    that securely encode pagination state for keyset pagination.
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize the cursor token manager.
        
        Args:
            secret_key: Secret key for HMAC signature. If None, will use a key from config.
        """
        config = get_config()
        # Use provided secret or get from config (which should default to a secure random value)
        self.secret_key = secret_key or config.get('cursor_token_secret', 'default-not-for-production')
        # Token expiry time in seconds (24 hours by default)
        self.token_expiry = config.get('cursor_token_expiry', 24 * 60 * 60)
        
    def create_token(self, 
                     last_values: Dict[str, Any], 
                     sort_fields: List[Tuple[str, bool]]) -> str:
        """
        Create a secure cursor token for pagination.
        
        Args:
            last_values: Dict mapping sort field names to their values from the last row
            sort_fields: List of (field_name, is_descending) tuples that define the sort order
            
        Returns:
            Encoded cursor token string
        """
        # Construct payload
        payload = {
            "v": 1,  # Version 
            "ts": int(time.time()),  # Timestamp for expiry
            "sort": [(field, desc) for field, desc in sort_fields],
            "last": last_values
        }
        
        # Serialize and encode payload
        payload_json = json.dumps(payload, separators=(',', ':'), default=str)
        encoded_payload = base64.urlsafe_b64encode(payload_json.encode()).decode()
        
        # Create HMAC signature
        signature = hmac.new(
            self.secret_key.encode(),
            encoded_payload.encode(),
            hashlib.sha256
        ).digest()
        encoded_signature = base64.urlsafe_b64encode(signature).decode()
        
        # Combine as token
        return f"{encoded_payload}.{encoded_signature}"
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and verify a cursor token.
        
        Args:
            token: Cursor token string to decode
            
        Returns:
            Dict with decoded payload
            
        Raises:
            ValueError: If token is invalid, tampered with, or expired
        """
        try:
            # Split token
            if '.' not in token:
                raise ValueError("Invalid token format")
                
            encoded_payload, encoded_signature = token.split('.')
            
            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                encoded_payload.encode(),
                hashlib.sha256
            ).digest()
            actual_signature = base64.urlsafe_b64decode(encoded_signature)
            
            if not hmac.compare_digest(expected_signature, actual_signature):
                raise ValueError("Invalid token signature")
            
            # Decode payload
            payload_json = base64.urlsafe_b64decode(encoded_payload).decode()
            payload = json.loads(payload_json)
            
            # Verify version
            if payload.get('v') != 1:
                raise ValueError("Unsupported token version")
                
            # Check expiry
            if int(time.time()) - payload.get('ts', 0) > self.token_expiry:
                raise ValueError("Token has expired")
                
            return payload
            
        except (ValueError, json.JSONDecodeError, TypeError, base64.binascii.Error) as e:
            raise ValueError(f"Invalid pagination token: {str(e)}")
    
    
class KeysetPaginator:
    """
    Implements keyset pagination for efficient, consistent pagination through large datasets.
    
    Instead of using offset/limit which can be inconsistent with changing data,
    keyset pagination uses the values of the last seen row to determine the next page.
    """
    
    def __init__(self, search_query: SearchQuery, sort_fields: List[Tuple[str, bool]], 
                 page_size: int = 10):
        """
        Initialize the keyset paginator.
        
        Args:
            search_query: Base SearchQuery object with filters applied
            sort_fields: List of (field_name, is_descending) sort fields
            page_size: Number of items per page
        """
        self.search_query = search_query
        self.sort_fields = sort_fields
        self.page_size = page_size
        self.token_manager = CursorTokenManager()
        
        # Ensure we have at least one sort field for deterministic ordering
        if not self.sort_fields:
            self.sort_fields = [('id', False)]  # Default to sorting by ID ascending
            
    def get_next_page(self, cursor_token: Optional[str] = None) -> Tuple[List[Any], Optional[str]]:
        """
        Get the next page of results.
        
        Args:
            cursor_token: Optional cursor token from the previous page
            
        Returns:
            Tuple of (items, next_cursor_token)
            next_cursor_token will be None if there are no more pages
        """
        # Start with a copy of the base query
        query = self.search_query
        
        # Apply sorting
        for field, descending in self.sort_fields:
            query = query.order_by(field, descending)
            
        # Apply keyset condition if we have a cursor token
        if cursor_token:
            try:
                token_data = self.token_manager.decode_token(cursor_token)
                last_values = token_data['last']
                
                # Enhance the query with keyset conditions
                query = self._apply_keyset_condition(query, last_values)
            except ValueError as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
                cursor_token = None  # Reset and start from beginning
        
        # Apply page size
        query = query.limit(self.page_size + 1)  # +1 to check if there are more pages
        
        # Execute query
        results = query.execute()
        items = results['products']
        
        # Determine if there are more pages
        has_more = len(items) > self.page_size
        if has_more:
            items = items[:self.page_size]  # Remove the extra item
        
        # Create next cursor token if needed
        next_cursor_token = None
        if has_more and items:
            # Extract values from the last item for the next keyset
            last_item = items[-1]
            last_values = {field: getattr(last_item, field, None) for field, _ in self.sort_fields}
            next_cursor_token = self.token_manager.create_token(last_values, self.sort_fields)
            
        return items, next_cursor_token
    
    def _apply_keyset_condition(self, query: SearchQuery, last_values: Dict[str, Any]) -> SearchQuery:
        """
        Apply keyset condition to query based on last seen values.
        
        This generates SQL that implements a proper keyset pagination condition.
        For example, if sorting by (name ASC, id ASC), it generates conditions like:
        (name > last_name) OR (name = last_name AND id > last_id)
        
        Args:
            query: Current search query
            last_values: Dict of values from the last item of the previous page
            
        Returns:
            Updated query with keyset conditions
        """
        # Implementation for keyset condition will vary depending on database vs in-memory
        # For simplicity, we'll use the query.filter_custom() method (assumed to exist)
        # In a real implementation, this would generate proper SQL or in-memory filters
        
        # This is a simplified example - a real implementation would need to handle
        # different data types, null values, etc.
        def keyset_filter(item):
            """In-memory filter implementing keyset pagination logic."""
            for i, (field, desc) in enumerate(self.sort_fields):
                # Get values for comparison
                item_value = getattr(item, field, None)
                last_value = last_values.get(field)
                
                # Skip if either value is None
                if item_value is None or last_value is None:
                    continue
                
                # Compare based on sort direction
                if desc:  # Descending order
                    if item_value < last_value:
                        return True
                    if item_value > last_value:
                        return False
                else:  # Ascending order
                    if item_value > last_value:
                        return True
                    if item_value < last_value:
                        return False
                    
                # If values are equal, continue to the next field
                # If all fields match exactly, this item is excluded
                
            # If we get here, all fields matched exactly - exclude this item
            return False
        
        # Add custom filter to query
        # Note: The real implementation needs SearchQuery.filter_custom() or similar
        # For demonstration purposes, we're just returning the query for now
        try:
            return query.filter_custom(keyset_filter)
        except AttributeError:
            # If filter_custom doesn't exist, we need to implement the keyset logic
            # differently based on what SearchQuery supports
            console.print("[yellow]Warning: Keyset pagination is not fully implemented for this storage backend.[/yellow]")
            return query


@app.command("list")
def list_products(
    # Search/filter parameters
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Text search across name and SKU"),
    tag: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Filter by tag (can be used multiple times)"),
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
    cursor: Optional[str] = typer.Option(None, "--cursor", "-c", help="Pagination cursor for next page"),
    
    # Output format parameters
    format: OutputFormat = typer.Option(
        OutputFormat.TABLE, "--format", "-f", 
        help="Output format (table, json, or csv)"
    ),
):
    """
    List and filter products in the inventory using efficient keyset pagination.
    
    Examples:
      list-products --query widget --tag hardware --sort-by name:asc,stock:desc
      list-products --below-reorder --format json
      list-products --cursor "eyJ2IjoxLC..."  # Get next page using cursor
    """
    # Build search query
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
    
    # Parse sort criteria
    try:
        order_by_clauses = SortOrderParser.parse(sort_by)
        # Ensure we have a stable sort by adding ID as the last sort field if not present
        if not any(field == 'id' for field, _ in order_by_clauses):
            order_by_clauses.append(('id', False))  # Default to ID ascending
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)
    
    # Create paginator
    paginator = KeysetPaginator(search, order_by_clauses, page_size)
    
    # Get page
    products, next_cursor = paginator.get_next_page(cursor)
    
    # Show pagination info
    if next_cursor:
        console.print(Panel(
            f"[bold green]Next page available![/bold green]\n"
            f"Use the following cursor to get the next page:\n\n"
            f"[blue]--cursor \"{next_cursor}\"[/blue]",
            title="Pagination",
            expand=False
        ))
    
    # Format and output results
    if format == OutputFormat.JSON:
        _output_json(products, next_cursor)
    elif format == OutputFormat.CSV:
        _output_csv(products)
    else:  # TABLE
        _output_table(products, order_by_clauses, next_cursor is not None)
        
        
def _output_table(products: List[Any], sort_by: List[Tuple[str, bool]] = None, has_next_page: bool = False):
    """Output products as a rich table."""
    if not products:
        console.print("[yellow]No products found matching your criteria.[/yellow]")
        return
    
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
    
    # Add a caption showing the sort order
    if sort_by:
        sort_desc = ", ".join(
            f"{field} ({'descending' if desc else 'ascending'})" 
            for field, desc in sort_by
        )
        table.caption = f"Sorted by: {sort_desc}"
    
    # Show count as footer
    console.print(table)
    page_info = f"Showing {len(products)} products"
    if has_next_page:
        page_info += " (more available)"
    console.print(f"[blue]{page_info}[/blue]")


def _output_json(products: List[Any], next_cursor: Optional[str] = None):
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
    
    # Build response with pagination info
    response = {
        "products": product_dicts,
        "pagination": {
            "count": len(product_dicts),
            "has_next_page": next_cursor is not None
        }
    }
    
    # Include next_cursor if available
    if next_cursor:
        response["pagination"]["next_cursor"] = next_cursor
    
    # Output JSON
    json_str = json.dumps(response, indent=2)
    console.print(json_str)


def _output_csv(products: List[Any]):
    """Output products as CSV."""
    if not products:
        console.print("[yellow]No products found matching your criteria.[/yellow]")
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