# Update the rich display function to include sparklines
def display_inventory_report_rich(products, forecasts, sort_by="depletion", show_sparklines=True):
    """Display a rich formatted inventory report with aligned columns and sparklines."""
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
    if show_sparklines:
        table.add_column("Usage (14d)", justify="left")
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
        
        # Get usage history for sparklines
        if show_sparklines:
            usage_history = get_usage_history(product_id)
            sparkline = generate_sparkline(usage_history)
        else:
            usage_history = []
            sparkline = ""
        
        table_data.append({
            "id": product_id,
            "name": name,
            "stock": stock,
            "usage": usage, 
            "days": days,
            "usage_history": usage_history,
            "sparkline": sparkline,
            "confidence": confidence
        })
    
    # Sort the data (same as before)
    if sort_by == "depletion":
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
        sparkline = item["sparkline"]
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
        
        # Create row with or without sparkline
        row_data = [
            product_id, 
            display_name, 
            stock_display, 
            usage_display
        ]
        
        if show_sparklines:
            row_data.append(sparkline)
            
        row_data.extend([
            days_display,
            date_display,
            conf_display
        ])
        
        table.add_row(*row_data)
    
    console.print(table)

# Update the ASCII display function to include sparklines
def display_inventory_report_ascii(products, forecasts, sort_by="depletion", show_sparklines=True):
    """Display an ASCII formatted inventory report with aligned columns and sparklines."""
    # Prepare data for display and sorting (same as before, with sparklines)
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
        
        # Get usage history for sparklines
        if show_sparklines:
            usage_history = get_usage_history(product_id)
            # Use a simpler version for ASCII that works in all terminals
            sparkline = generate_sparkline(usage_history, filled_chars="_▁▂▃▄▅▆▇█")
        else:
            usage_history = []
            sparkline = ""
            
        table_data.append({
            "id": product_id,
            "name": name,
            "stock": stock,
            "usage": usage, 
            "days": days,
            "usage_history": usage_history,
            "sparkline": sparkline,
            "confidence": confidence
        })
    
    # Sort the data (same as before)
    if sort_by == "depletion":
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
        "name": 25,
        "stock": 12,
        "usage": 12,
        "sparkline": 16,
        "days": 10,
        "date": 12,
        "confidence": 10
    }
    
    # Print header row
    header_parts = [
        f"{'ID':<{columns['id']}}",
        f"{'Product Name':<{columns['name']}}",
        f"{'Stock':>{columns['stock']}}",
        f"{'Daily Usage':>{columns['usage']}}"
    ]
    
    if show_sparklines:
        header_parts.append(f"{'Usage (14d)':<{columns['sparkline']}}")
        
    header_parts.extend([
        f"{'Days Until':>{columns['days']}}",
        f"{'Depl. Date':<{columns['date']}}",
        f"{'Confidence':<{columns['confidence']}}"
    ])
    
    header = " ".join(header_parts)
    print(header)
    print('-' * len(header))
    
    # Print each row
    for item in table_data:
        product_id = item["id"]
        name = item["name"]
        stock = item["stock"]
        usage = item["usage"]
        days = item["days"]
        sparkline = item["sparkline"]
        confidence = item["confidence"]
        
        # Truncate long product names (shorter for ASCII to fit sparkline)
        display_name = (name[:22] + "...") if len(name) > 25 else name
        
        # Format values
        usage_display = f"{usage:.2f}" if usage is not None else "Unknown"
        
        if days is not None:
            days_display = f"{days:.1f}"
            date_display = format_date(days)
        else:
            days_display = "N/A"
            date_display = "Unknown"
        
        # Create row parts
        row_parts = [
            f"{product_id:<{columns['id']}}",
            f"{display_name:<{columns['name']}}",
            f"{stock:>{columns['stock']}}",
            f"{usage_display:>{columns['usage']}}"
        ]
        
        if show_sparklines:
            row_parts.append(f"{sparkline:<{columns['sparkline']}}")
            
        row_parts.extend([
            f"{days_display:>{columns['days']}}",
            f"{date_display:<{columns['date']}}",
            f"{confidence:<{columns['confidence']}}"
        ])
        
        row = " ".join(row_parts)
        print(row)
    
    print('-' * len(header))