#!/usr/bin/env python3
"""
commands/analytics.py - Command line interface for inventory analytics

This module provides CLI commands for displaying inventory analytics,
including current stock levels, usage rates, and depletion forecasts.
"""

import sys
import argparse
from datetime import datetime, timedelta
import textwrap

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Import from parent directory
sys.path.append('..')
import analytics
import database

def format_date(days):
    """
    Convert days to a future date string.
    
    Args:
        days (float or None): Number of days until depletion
        
    Returns:
        str: Formatted date or status message
    """
    if days is None:
        return "Unknown"
    
    if days <= 0:
        return "Out of stock"
    
    # Calculate the future date
    future_date = datetime.now().date() + timedelta(days=int(days))
    return future_date.strftime("%Y-%m-%d")

def get_depletion_color(days):
    """
    Get an appropriate color based on days until depletion.
    
    Args:
        days (float or None): Number of days until depletion
        
    Returns:
        str: Color name for rich formatting
    """
    if days is None:
        return "dim"
    if days <= 3:
        return "red"
    if days <= 7:
        return "yellow"
    if days <= 14:
        return "green"
    return "blue"

def get_stock_color(stock):
    """
    Get an appropriate color based on stock level.
    
    Args:
        stock (int): Current stock level
        
    Returns:
        str: Color name for rich formatting
    """
    if stock <= 0:
        return "red"
    if stock <= 10:
        return "yellow"
    if stock <= 50:
        return "green"
    return "blue"

def display_inventory_report_rich(products, forecasts, sort_by="depletion"):
    """
    Display a rich formatted inventory report with aligned columns.
    
    Args:
        products (list): List of product dictionaries
        forecasts (dict): Dictionary mapping product_id to forecast data
        sort_by (str): Column to sort by ("depletion", "stock", "usage", or "name")
    """
    console = Console()
    
    # Create a table
    table = Table(
        show_header=True, 
        header_style="bold cyan",
        box=box.ROUNDED,
        title="Inventory Depletion Forecast"
    )
    
    # Add columns with specific formatting
    table.add_column("Product ID", style="dim")
    table.add_column("Product Name", no_wrap=True)
    table.add_column("Current Stock", justify="right")
    table.add_column("Daily Usage", justify="right")
    table.add_column("Days Until\nDepletion", justify="right")
    table.add_column("Depletion Date", justify="center")
    table.add_column("Confidence", justify="center")
    
    # Prepare data for sorting
    table_data = []
    for product in products:
        product_id = product["id"]
        name = product["name"]
        
        # Get forecast data or default values
        if product_id in forecasts:
            forecast = forecasts[product_id]
            stock = forecast["current_stock"]
            usage = forecast["daily_usage_rate"]
            days = forecast["days_until_depletion"]
            confidence = forecast["confidence"]
        else:
            stock = 0
            usage = 0
            days = None
            confidence = "unavailable"
        
        table_data.append({
            "id": product_id,
            "name": name,
            "stock": stock,
            "usage": usage, 
            "days": days,
            "confidence": confidence
        })
    
    # Sort the data
    if sort_by == "depletion":
        # Sort by days until depletion (None values at the end)
        table_data.sort(key=lambda x: (x["days"] is None, x["days"] if x["days"] is not None else float('inf')))
    elif sort_by == "stock":
        table_data.sort(key=lambda x: x["stock"])
    elif sort_by == "usage":
        table_data.sort(key=lambda x: x["usage"], reverse=True)
    elif sort_by == "name":
        table_data.sort(key=lambda x: x["name"].lower())
    
    # Add rows to the table
    for item in table_data:
        product_id = item["id"]
        name = item["name"]
        stock = item["stock"]
        usage = item["usage"]
        days = item["days"]
        confidence = item["confidence"]
        
        # Truncate long product names
        display_name = (name[:27] + "...") if len(name) > 30 else name
        
        # Format values and add colors
        stock_display = f"[{get_stock_color(stock)}]{stock}[/]"
        usage_display = f"{usage:.2f}" if usage is not None else "Unknown"
        
        if days is not None:
            days_display = f"[{get_depletion_color(days)}]{days:.1f}[/]"
            date_display = f"[{get_depletion_color(days)}]{format_date(days)}[/]"
        else:
            days_display = "[dim]N/A[/]"
            date_display = "[dim]Unknown[/]"
        
        # Set confidence display
        if confidence == "high":
            conf_display = "[green]High[/]"
        elif confidence == "medium":
            conf_display = "[yellow]Medium[/]"
        elif confidence == "low":
            conf_display = "[red]Low[/]"
        else:
            conf_display = "[dim]Unknown[/]"
        
        table.add_row(
            product_id, 
            display_name, 
            stock_display, 
            usage_display, 
            days_display,
            date_display,
            conf_display
        )
    
    console.print(table)

def display_inventory_report_ascii(products, forecasts, sort_by="depletion"):
    """
    Display an ASCII formatted inventory report with aligned columns for systems without rich.
    
    Args:
        products (list): List of product dictionaries
        forecasts (dict): Dictionary mapping product_id to forecast data
        sort_by (str): Column to sort by ("depletion", "stock", "usage", or "name")
    """
    # Prepare data for display and sorting
    table_data = []
    for product in products:
        product_id = product["id"]
        name = product["name"]
        
        # Get forecast data or default values
        if product_id in forecasts:
            forecast = forecasts[product_id]
            stock = forecast["current_stock"]
            usage = forecast["daily_usage_rate"]
            days = forecast["days_until_depletion"]
            confidence = forecast["confidence"]
        else:
            stock = 0
            usage = 0
            days = None
            confidence = "unavailable"
        
        table_data.append({
            "id": product_id,
            "name": name,
            "stock": stock,
            "usage": usage, 
            "days": days,
            "confidence": confidence
        })
    
    # Sort the data
    if sort_by == "depletion":
        # Sort by days until depletion (None values at the end)
        table_data.sort(key=lambda x: (x["days"] is None, x["days"] if x["days"] is not None else float('inf')))
    elif sort_by == "stock":
        table_data.sort(key=lambda x: x["stock"])
    elif sort_by == "usage":
        table_data.sort(key=lambda x: x["usage"], reverse=True)
    elif sort_by == "name":
        table_data.sort(key=lambda x: x["name"].lower())
    
    # Define column widths
    columns = {
        "id": 12,
        "name": 30,
        "stock": 12,
        "usage": 12,
        "days": 12,
        "date": 12,
        "confidence": 12
    }
    
    # Print header row
    header = (
        f"{'ID':<{columns['id']}} "
        f"{'Product Name':<{columns['name']}} "
        f"{'Stock':>{columns['stock']}} "
        f"{'Daily Usage':>{columns['usage']}} "
        f"{'Days Until':>{columns['days']}} "
        f"{'Depletion Date':<{columns['date']}} "
        f"{'Confidence':<{columns['confidence']}}"
    )
    print(header)
    print('-' * (sum(columns.values()) + len(columns)))
    
    # Print each row
    for item in table_data:
        product_id = item["id"]
        name = item["name"]
        stock = item["stock"]
        usage = item["usage"]
        days = item["days"]
        confidence = item["confidence"]
        
        # Truncate long product names
        display_name = (name[:27] + "...") if len(name) > 30 else name
        
        # Format values
        usage_display = f"{usage:.2f}" if usage is not None else "Unknown"
        
        if days is not None:
            days_display = f"{days:.1f}"
            date_display = format_date(days)
        else:
            days_display = "N/A"
            date_display = "Unknown"
        
        row = (
            f"{product_id:<{columns['id']}} "
            f"{display_name:<{columns['name']}} "
            f"{stock:>{columns['stock']}} "
            f"{usage_display:>{columns['usage']}} "
            f"{days_display:>{columns['days']}} "
            f"{date_display:<{columns['date']}} "
            f"{confidence:<{columns['confidence']}}"
        )
        print(row)
    
    print('-' * (sum(columns.values()) + len(columns)))

def main():
    """
    Main function to parse arguments and display inventory analytics.
    """
    parser = argparse.ArgumentParser(
        description="Display inventory analytics and depletion forecasts"
    )
    
    # Add command line arguments
    parser.add_argument(
        "--sort-by", 
        choices=["depletion", "stock", "usage", "name"],
        default="depletion",
        help="Field to sort the products by (default: depletion)"
    )
    parser.add_argument(
        "--critical-only", 
        action="store_true",
        help="Only show products with less than 14 days until depletion"
    )
    parser.add_argument(
        "--ascii", 
        action="store_true",
        help="Use ASCII output instead of rich formatting"
    )
    parser.add_argument(
        "--days", 
        type=int,
        default=30,
        help="Number of days to analyze for usage patterns (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Force ASCII mode if rich is not available
    use_ascii = args.ascii or not HAS_RICH
    
    try:
        # Get all active products
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM products WHERE active = 1")
        all_products = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]
        conn.close()
        
        # Calculate forecasts for all products
        forecasts = {}
        for product in all_products:
            forecast = analytics.forecast_depletion(product["id"], history_days=args.days)
            if forecast["days_until_depletion"] is not None:
                forecasts[product["id"]] = forecast
        
        # Filter to critical products if requested
        if args.critical_only:
            filtered_products = []
            for product in all_products:
                if (product["id"] in forecasts and 
                    forecasts[product["id"]]["days_until_depletion"] is not None and
                    forecasts[product["id"]]["days_until_depletion"] <= 14):
                    filtered_products.append(product)
            products = filtered_products
        else:
            products = all_products
        
        if not products:
            print("No products match the criteria.")
            return
        
        # Display the report using the appropriate formatter
        if use_ascii:
            display_inventory_report_ascii(products, forecasts, args.sort_by)
        else:
            display_inventory_report_rich(products, forecasts, args.sort_by)
        
    except Exception as e:
        print(f"Error generating report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()