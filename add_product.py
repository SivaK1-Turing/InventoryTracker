# inventorytracker/commands/add_product.py
import re
import typer
import sys
from typing import Optional, Dict, Any, List, Set, Tuple, Union
from uuid import uuid4
from decimal import Decimal, InvalidOperation
from enum import Enum
from difflib import SequenceMatcher
import importlib.util

from ..factories import create_product
from ..store import get_store
from ..models.product import Product
from ..utils.validation import is_valid_sku

# Check if rich is available and create graceful fallbacks
RICH_AVAILABLE = importlib.util.find_spec("rich") is not None

if RICH_AVAILABLE:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    from rich.text import Text
    
    # Create console for rich output
    console = Console()
else:
    # Create fallback classes and functions when rich is not available
    class FallbackConsole:
        def print(self, *args, **kwargs):
            # Strip rich formatting markers like [bold], [green], etc.
            text = str(args[0]) if args else ""
            # Simple regex to remove rich formatting tags
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\[/.*?\]', '', text)
            print(text)
            
    class FallbackPanel:
        @staticmethod
        def fit(renderable, **kwargs):
            if isinstance(renderable, str):
                return renderable
            return str(renderable)
            
    class FallbackTable:
        def __init__(self, **kwargs):
            self.title = kwargs.get("title", "")
            self.rows = []
            self.columns = []
            
        def add_column(self, header, **kwargs):
            self.columns.append(header)
            
        def add_row(self, *args):
            self.rows.append(args)
            
        def __str__(self):
            result = [f"--- {self.title} ---"] if self.title else []
            if self.columns:
                result.append(" | ".join(self.columns))
                result.append("-" * (len(" | ".join(self.columns))))
            
            for row in self.rows:
                result.append(" | ".join(str(cell) for cell in row))
                
            return "\n".join(result)
    
    # Create placeholders
    console = FallbackConsole()
    Panel = FallbackPanel
    Table = FallbackTable
    
    def rprint(*args, **kwargs):
        print(*args, **kwargs)
    
    class Text:
        def __init__(self, text, **kwargs):
            self.text = text
            
        def __str__(self):
            return self.text

# Create the command group
app = typer.Typer(help="Product management commands")

# Enum for comparison result
class DiffType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"

def validate_name(value: str) -> str:
    """Validate product name."""
    if not value or len(value.strip()) < 3:
        raise typer.BadParameter("Product name must be at least 3 characters long")
    return value.strip()

def validate_sku(value: str, check_exists: bool = True) -> str:
    """
    Validate SKU format.
    
    Args:
        value: The SKU to validate
        check_exists: Whether to check if the SKU exists in the store
        
    Returns:
        The validated SKU
    """
    value = value.strip().upper()
    if not re.match(r'^[A-Z0-9]+$', value):
        raise typer.BadParameter("SKU must contain only uppercase letters and numbers")
    
    # The existence check is separated to provide a different flow, not an error
    return value

def validate_price(value: str) -> Decimal:
    """Validate and convert price to Decimal."""
    try:
        price = Decimal(value.strip())
        if price <= 0:
            raise typer.BadParameter("Price must be greater than zero")
        return price
    except InvalidOperation:
        raise typer.BadParameter("Please enter a valid number for price")

def validate_reorder_level(value: str) -> int:
    """Validate and convert reorder level to int."""
    try:
        level = int(value.strip())
        if level < 0:
            raise typer.BadParameter("Reorder level cannot be negative")
        return level
    except ValueError:
        raise typer.BadParameter("Please enter a valid integer for reorder level")

def prompt_with_validation(prompt_text: str, validation_func, default=None, hide_input=False, **validation_kwargs):
    """
    Prompt for input with validation and graceful re-prompting.
    
    Args:
        prompt_text: Text to display in the prompt
        validation_func: Function to validate input
        default: Default value if empty input
        hide_input: Whether to hide input (for passwords, etc.)
        **validation_kwargs: Additional keyword arguments for the validation function
        
    Returns:
        Validated input value
    """
    while True:
        try:
            value = typer.prompt(
                prompt_text,
                default=default,
                hide_input=hide_input,
                show_default=default is not None
            )
            return validation_func(value, **validation_kwargs)
        except typer.BadParameter as e:
            error_msg = f"Error: {str(e)}"
            if RICH_AVAILABLE:
                console.print(f"[bold red]{error_msg}[/]")
            else:
                console.print(error_msg)
            # Loop continues for re-prompting

def diff_pydantic_models(model1: Product, model2: Product) -> Dict[str, Tuple[Any, Any, DiffType]]:
    """
    Compare two Pydantic models and return the differences.
    
    Args:
        model1: First model (existing)
        model2: Second model (new)
        
    Returns:
        Dictionary of field_name: (model1_value, model2_value, diff_type)
    """
    # Get fields to compare (excluding id which should remain the same)
    fields_to_compare = {field for field in model1.dict() if field != "id"}
    
    # Extract dictionaries
    dict1 = model1.dict()
    dict2 = model2.dict()
    
    # Compare fields
    diff = {}
    for field in fields_to_compare:
        value1 = dict1.get(field)
        value2 = dict2.get(field)
        
        if field not in dict2:
            diff[field] = (value1, None, DiffType.REMOVED)
        elif field not in dict1:
            diff[field] = (None, value2, DiffType.ADDED)
        elif value1 != value2:
            diff[field] = (value1, value2, DiffType.CHANGED)
        else:
            diff[field] = (value1, value2, DiffType.UNCHANGED)
    
    return diff

def format_value_for_display(value: Any) -> str:
    """Format a value for display in the comparison table."""
    if isinstance(value, Decimal):
        return f"${float(value):.2f}"
    if value is None:
        return "N/A"
    return str(value)

def display_product_comparison(existing_product: Product, new_data: Dict[str, Any]):
    """
    Display a side-by-side comparison of existing and new product data.
    
    Args:
        existing_product: The existing product in the store
        new_data: New data for the product
    """
    # Create a temporary new product model for comparison
    new_data_with_id = new_data.copy()
    new_data_with_id['id'] = existing_product.id  # Keep the same ID for comparison
    
    # Create new product model
    new_product = Product(**new_data_with_id)
    
    # Get differences
    diffs = diff_pydantic_models(existing_product, new_product)
    
    # Create comparison table
    table = Table(title=f"Product Comparison for SKU: {existing_product.sku}")
    
    table.add_column("Field", style="cyan" if RICH_AVAILABLE else None)
    table.add_column("Existing Value", style="green" if RICH_AVAILABLE else None)
    table.add_column("New Value", style="yellow" if RICH_AVAILABLE else None)
    table.add_column("Status", style="magenta" if RICH_AVAILABLE else None)
    
    # Add rows for each field, highlighting differences
    for field, (old_value, new_value, diff_type) in diffs.items():
        old_display = format_value_for_display(old_value)
        new_display = format_value_for_display(new_value)
        
        # Set status text based on diff type
        if diff_type == DiffType.UNCHANGED:
            status = "[green]Unchanged[/]" if RICH_AVAILABLE else "Unchanged"
        elif diff_type == DiffType.CHANGED:
            status = "[yellow]Changed[/]" if RICH_AVAILABLE else "Changed"
        elif diff_type == DiffType.ADDED:
            status = "[green]Added[/]" if RICH_AVAILABLE else "Added"
        else:  # REMOVED
            status = "[red]Removed[/]" if RICH_AVAILABLE else "Removed"
            
        table.add_row(
            field.capitalize(),
            old_display,
            new_display, 
            status
        )
    
    console.print(table)

def create_product_summary_panel(product: Product, action: str = "added") -> Union[Panel, str]:
    """
    Create a styled panel summarizing product details.
    
    Args:
        product: The product to summarize
        action: The action that was performed (added, updated)
        
    Returns:
        A rich Panel or formatted string if rich is not available
    """
    # Create the content
    if RICH_AVAILABLE:
        content = Text.from_markup(
            f"[bold]ID:[/] {product.id}\n"
            f"[bold]Name:[/] {product.name}\n"
            f"[bold]SKU:[/] {product.sku}\n"
            f"[bold]Price:[/] ${float(product.price):.2f}\n"
            f"[bold]Reorder Level:[/] {product.reorder_level}"
        )
        
        # Create panel with styling
        return Panel(
            content,
            title=f"Product {action} successfully",
            title_align="center",
            border_style="green",
            padding=(1, 2)
        )
    else:
        # Fallback to plain text
        return (
            f"Product {action} successfully:\n"
            f"  ID: {product.id}\n"
            f"  Name: {product.name}\n"
            f"  SKU: {product.sku}\n"
            f"  Price: ${float(product.price):.2f}\n"
            f"  Reorder Level: {product.reorder_level}"
        )

@app.command("add")
def add_product(
    interactive: bool = typer.Option(
        True, "--interactive/--non-interactive", "-i/-n", 
        help="Use interactive prompts for input"
    ),
    force_overwrite: bool = typer.Option(
        False, "--force", "-f", 
        help="Overwrite existing product without confirmation"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Product name (min 3 characters)"
    ),
    sku: Optional[str] = typer.Option(
        None, "--sku", "-s", help="Stock keeping unit (uppercase alphanumeric)"
    ),
    price: Optional[float] = typer.Option(
        None, "--price", "-p", help="Product price (must be greater than 0)"
    ),
    reorder_level: Optional[int] = typer.Option(
        None, "--reorder-level", "-r", help="Inventory level that triggers reordering"
    )
):
    """
    Add a new product to inventory.
    
    Interactive mode will prompt for each field with validation.
    Non-interactive mode requires all parameters to be provided via options.
    """
    try:
        store = get_store()
        
        if interactive:
            header_message = "Adding a new product"
            if RICH_AVAILABLE:
                console.print(f"[bold blue]{header_message}[/]")
            else:
                console.print(header_message)
            console.print("Please enter the following information:")
            
            # Multi-step input flow with validation
            name_value = prompt_with_validation(
                "Product name (min 3 characters)",
                validate_name
            )
            
            # First get a valid SKU format without checking existence
            sku_value = prompt_with_validation(
                "SKU (uppercase alphanumeric)",
                validate_sku,
                check_exists=False
            )
            
            # Then check if it exists
            existing_product = store.get_product_by_sku(sku_value)
            if existing_product and not force_overwrite:
                # Collect all data before showing comparison
                price_value = prompt_with_validation(
                    "Price",
                    validate_price
                )
                
                reorder_value = prompt_with_validation(
                    "Reorder level",
                    validate_reorder_level,
                    default="10"
                )
                
                # Prepare new data
                new_data = {
                    'name': name_value,
                    'sku': sku_value,
                    'price': price_value,
                    'reorder_level': reorder_value
                }
                
                # Show warning about existing product
                console.print()
                if RICH_AVAILABLE:
                    console.print("[bold yellow]A product with this SKU already exists![/]")
                else:
                    console.print("Warning: A product with this SKU already exists!")
                
                display_product_comparison(existing_product, new_data)
                
                if not typer.confirm("\nDo you want to overwrite the existing product?", default=False):
                    if RICH_AVAILABLE:
                        console.print("[yellow]Operation canceled by user[/]")
                    else:
                        console.print("Operation canceled by user")
                    raise typer.Exit()
                
                # If confirmed, continue with the existing ID
                product = create_product(
                    id=existing_product.id,
                    **new_data
                )
                action = "updated"
            else:
                # New product or force overwrite, continue collecting data if needed
                if not existing_product:
                    price_value = prompt_with_validation(
                        "Price",
                        validate_price
                    )
                    
                    reorder_value = prompt_with_validation(
                        "Reorder level",
                        validate_reorder_level,
                        default="10"
                    )
                    action = "added"
                else:
                    # When force_overwrite is True and we have an existing product
                    if RICH_AVAILABLE:
                        console.print("[bold yellow]Overwriting existing product (--force flag used)[/]")
                    else:
                        console.print("Warning: Overwriting existing product (--force flag used)")
                        
                    price_value = prompt_with_validation(
                        "Price",
                        validate_price,
                        default=str(existing_product.price)
                    )
                    
                    reorder_value = prompt_with_validation(
                        "Reorder level",
                        validate_reorder_level,
                        default=str(existing_product.reorder_level)
                    )
                    action = "updated"
                
                # Create product (using existing ID if overwriting)
                product_id = existing_product.id if existing_product else None
                product = create_product(
                    name=name_value,
                    sku=sku_value,
                    price=price_value,
                    reorder_level=reorder_value,
                    id=product_id
                )
        else:
            # Non-interactive mode validation
            if not all([name, sku, price is not None, reorder_level is not None]):
                error_msg = "All parameters are required in non-interactive mode"
                if RICH_AVAILABLE:
                    console.print(f"[bold red]Error:[/] {error_msg}")
                else:
                    console.print(f"Error: {error_msg}")
                raise typer.Exit(code=1)
                
            name_value = validate_name(name)
            sku_value = validate_sku(sku, check_exists=False)
            price_value = validate_price(str(price))
            reorder_value = validate_reorder_level(str(reorder_level))
            
            # Check for existing product
            existing_product = store.get_product_by_sku(sku_value)
            if existing_product and not force_overwrite:
                error_msg = "Product with this SKU already exists. Use --force to overwrite."
                if RICH_AVAILABLE:
                    console.print(f"[bold red]Error:[/] {error_msg}")
                else:
                    console.print(f"Error: {error_msg}")
                raise typer.Exit(code=1)
            
            # Create product (using existing ID if overwriting)
            product_id = existing_product.id if existing_product else None
            action = "updated" if existing_product else "added"
            
            product = create_product(
                name=name_value,
                sku=sku_value,
                price=price_value,
                reorder_level=reorder_value,
                id=product_id
            )

        # Save to store
        store.save_product(product)
        
        # Display the product summary
        console.print()
        summary = create_product_summary_panel(product, action)
        console.print(summary)
        
    except typer.Exit:
        # Re-raise exits to maintain typer's exit code handling
        raise
    except Exception as e:
        error_msg = str(e)
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error:[/] {error_msg}")
        else:
            console.print(f"Error: {error_msg}")
        raise typer.Exit(code=1)