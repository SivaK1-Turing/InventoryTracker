"""
commands/list_products.py - Command for listing and filtering products

This module provides the CLI commands for listing products with various
sorting and filtering options.
"""
import typer
from typing import List, Dict, Any, Optional, Tuple
import re
from enum import Enum
from inventorytracker.search import SearchQuery
import csv
import sys
import json
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="List and filter products in inventory")
console = Console()


class OutputFormat(str, Enum):
    TABLE = "table"
    JSON = "json"
    CSV = "csv"


class SortOrderParser:
    """Parser for sort order specifications like 'name:asc,stock:desc'."""
    
    SORT_PATTERN = re.compile(r'([a-zA-Z_]+)(?::(asc|desc))?')
    
    # Map of CLI field names to model field names (add more as needed)
    FIELD_MAPPING = {
        # CLI name -> Model field name
        "name": "name",
        "sku": "sku",
        "price": "price", 
        "stock": "current_stock",
        "reorder": "reorder_level",
        "created": "created_at",
        "updated": "updated_at",
        # Add more mappings as needed
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
    
    # Output format parameters
    format: OutputFormat = typer.Option(
        OutputFormat.TABLE, "--format", "-f", 
        help="Output format (table, json, or csv)"
    ),
    
    # Pagination parameters
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Maximum number of results to show"),
    offset: Optional[int] = typer.Option(0, "--offset", "-o", help="Number of results to skip"),
):
    """
    List and filter products in the inventory.
    
    Examples:
      list-products --query widget --tag hardware --sort-by name:asc,stock:desc
      list-products --below-reorder --format json
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
    
    # Apply sorting
    try:
        order_by_clauses = SortOrderParser.parse(sort_by)
        for field, descending in order_by_clauses:
            search.order_by(field, descending)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(1)
    
    # Apply pagination
    if offset is not None:
        search.offset(offset)
        
    if limit is not None:
        search.limit(limit)
    
    # Execute search
    results = search.execute()
    products = results['products']
    
    # Format and output results
    if format == OutputFormat.JSON:
        _output_json(products)
    elif format == OutputFormat.CSV:
        _output_csv(products)
    else:  # TABLE
        _output_table(products, order_by_clauses)
        
        
def _output_table(products: List[Any], sort_by: List[Tuple[str, bool]] = None):
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
    console.print(f"[blue]Showing {len(products)} products[/blue]")


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
    
    # Output JSON
    json_str = json.dumps({"products": product_dicts}, indent=2)
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