import typer
from typing import List, Optional, Tuple
from uuid import UUID
import re
from difflib import get_close_matches
from rich.console import Console
from rich.table import Table

from ..models.product import Product
from ..factories import create_transaction
from ..store import get_store
from ..utils.validation import prompt_with_validation

console = Console()
app = typer.Typer(help="Manage inventory transactions")

# In commands/transact.py - modify the add_transaction command

@app.command("add")
def add_transaction(
    product: Optional[str] = typer.Option(None, "--product", "-p", help="Product SKU or name"),
    quantity: Optional[int] = typer.Option(None, "--quantity", "-q", help="Quantity (positive for in, negative for out)"),
    note: Optional[str] = typer.Option(None, "--note", "-n", help="Transaction note")
):
    """
    Add a stock transaction to track inventory changes.
    
    Positive quantity adds to inventory, negative quantity removes from inventory.
    """
    # ... existing code for prompting and transaction processing ...
    
    store = get_store()
    try:
        # ... existing transaction code ...
        
        # After successful transaction, show result
        table = create_transaction_table(product_obj, quantity, new_stock_level, note, transaction.id)        
        console.print(table)
        
        # Display transaction history
        show_transaction_history(product_obj)
        
    except ConcurrencyError as e:
        console.print(f"[red]Transaction conflict: {str(e)}. Please try again.[/red]")
    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")


def create_transaction_table(product, quantity, new_stock_level, note, transaction_id):
    """Create a rich table for transaction details with fallback to plain text"""
    try:
        from rich.table import Table
        
        table = Table(title=f"Transaction Complete")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Product", f"{product.name} ({product.sku})")
        table.add_row("Quantity Change", f"{quantity:+d}")  # Shows + or - sign
        table.add_row("New Stock Level", f"{new_stock_level}")
        if note:
            table.add_row("Note", note)
        table.add_row("Transaction ID", str(transaction_id))
        
        return table
    
    except ImportError:
        # Fallback to plain text if rich is not available
        result = [
            "Transaction Complete",
            "-" * 40,
            f"Product: {product.name} ({product.sku})",
            f"Quantity Change: {quantity:+d}",
            f"New Stock Level: {new_stock_level}",
        ]
        if note:
            result.append(f"Note: {note}")
        result.append(f"Transaction ID: {transaction_id}")
        result.append("-" * 40)
        
        return "\n".join(result)


def show_transaction_history(product):
    """
    Display the last 5 transactions for a product in a rich colored table
    with fallback to plain ASCII if rich is not available.
    """
    store = get_store()
    transactions = store.get_product_transactions(product.id, limit=5)
    
    if not transactions:
        try:
            from rich.panel import Panel
            console.print(Panel(f"No previous transactions for this product", 
                               title="Transaction History", 
                               border_style="yellow"))
        except ImportError:
            print("\nTransaction History")
            print("-" * 40)
            print("No previous transactions for this product")
            print("-" * 40)
        return
    
    try:
        # Try to use rich formatting
        from rich.table import Table
        from rich.text import Text
        
        table = Table(title=f"Recent Transactions for {product.name}")
        table.add_column("Date/Time", style="cyan")
        table.add_column("Change", style="bold")
        table.add_column("Note", style="italic")
        
        for tx in transactions:
            # Format timestamp
            timestamp = tx.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Format change with color based on positive/negative
            if tx.delta > 0:
                change = Text(f"+{tx.delta}", style="green")
            else:
                change = Text(f"{tx.delta}", style="red")
                
            # Add note or placeholder
            note = tx.note if tx.note else "-"
            
            table.add_row(timestamp, change, note)
            
        console.print(table)
        
    except ImportError:
        # Fallback to plain ASCII if rich is not available
        print("\nRecent Transactions for", product.name)
        print("-" * 60)
        print(f"{'Date/Time':<20} | {'Change':^8} | {'Note':<30}")
        print("-" * 60)
        
        for tx in transactions:
            timestamp = tx.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            change = f"+{tx.delta}" if tx.delta > 0 else f"{tx.delta}"
            note = tx.note if tx.note else "-"
            
            print(f"{timestamp:<20} | {change:^8} | {note:<30}")
        
        print("-" * 60)

def prompt_for_product(products: List[Product]) -> UUID:
    """
    Interactive prompt with fuzzy matching to select a product.
    
    Returns the product ID for the selected product.
    """
    # Show available products
    table = Table(title="Available Products")
    table.add_column("#", style="dim")
    table.add_column("SKU", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Current Stock", style="yellow")
    
    # Sort products by name for easier lookup
    sorted_products = sorted(products, key=lambda p: p.name)
    
    for idx, product in enumerate(sorted_products, 1):
        current_stock = get_store().get_inventory_level(product.id)
        table.add_row(
            str(idx), 
            product.sku, 
            product.name,
            str(current_stock)
        )
    
    console.print(table)
    
    # Prompt user to select a product
    while True:
        search = typer.prompt("Enter product number, SKU, or name (partial match works)")
        
        # Check if input is a number (index)
        if search.isdigit() and 1 <= int(search) <= len(sorted_products):
            return sorted_products[int(search) - 1].id
        
        # Try to match against SKU or name
        matches = []
        for product in sorted_products:
            if search.upper() == product.sku:  # Exact SKU match
                return product.id
            if search.lower() in product.name.lower():  # Partial name match
                matches.append(product)
        
        # If we found exactly one match, use it
        if len(matches) == 1:
            return matches[0].id
        # If we found multiple matches, show them and ask again
        elif len(matches) > 1:
            console.print("[yellow]Multiple matches found:[/yellow]")
            for idx, product in enumerate(matches, 1):
                console.print(f"{idx}. {product.name} ({product.sku})")
            
            selection = typer.prompt(
                "Enter number of desired product", 
                type=int,
                show_choices=False
            )
            
            if 1 <= selection <= len(matches):
                return matches[selection - 1].id
            else:
                console.print("[red]Invalid selection[/red]")
        else:
            console.print("[red]No matches found[/red]")

def prompt_for_quantity() -> int:
    """
    Prompt for quantity with validation.
    
    Returns positive for stock in, negative for stock out.
    """
    transaction_type = typer.prompt(
        "Transaction type",
        type=typer.Choice(["in", "out"]),
        show_choices=True,
        show_default=True,
    )
    
    # Validate quantity is positive
    def validate_quantity(value: str) -> int:
        try:
            qty = int(value)
            if qty <= 0:
                raise ValueError("Quantity must be a positive number")
            return qty
        except ValueError:
            raise ValueError("Please enter a valid positive number")
    
    quantity = prompt_with_validation("Quantity", validate_quantity)
    
    # Make negative if it's a stock out transaction
    if transaction_type == "out":
        quantity = -quantity
    
    return quantity

def find_product(search: str, products: List[Product]) -> Optional[Product]:
    """Find a product by SKU or name (fuzzy match)"""
    # Exact SKU match
    for product in products:
        if product.sku.upper() == search.upper():
            return product
    
    # Partial name match
    matches = []
    for product in products:
        if search.lower() in product.name.lower():
            matches.append(product)
    
    if matches:
        return matches[0]  # Return first match
        
    return None