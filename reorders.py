# inventorytracker/commands/reorders.py (updated with override support)
"""
Command-line interface for reordering and low-stock alerts.

This module provides commands for viewing products that need to be reordered,
generating reordering reports, and managing stock alerts.
"""

import datetime
from typing import List, Optional, Tuple, Dict, Any
from uuid import UUID
import math
import typer
import json
from enum import Enum
from pathlib import Path

# Rich library components for enhanced terminal UI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.text import Text
from rich.box import Box, ROUNDED
from rich.style import Style
from rich.measure import Measurement
from rich.console import Group

from ..alerts import (
    detect_low_stock, get_critical_stock_alerts, generate_reorder_report, 
    StockAlert, clear_alert_cache
)
from ..utils.formatting import format_currency, format_quantity
from ..config import get_config
from ..store import get_store

# Initialize Typer app/command group
reorder_app = typer.Typer(help="Commands for managing reorders and low stock alerts")

# Initialize Rich console for output
console = Console()

# Adjust these values based on your business logic
URGENCY_THRESHOLDS = {
    "critical": 80,   # 80-100% urgency
    "high": 60,       # 60-79% urgency
    "medium": 40,     # 40-59% urgency
    "low": 20,        # 20-39% urgency
    "minimal": 0      # 0-19% urgency
}

class SortOrder(str, Enum):
    """Sorting options for the reorder list."""
    URGENCY = "urgency"
    DEFICIT = "deficit"
    SKU = "sku"
    NAME = "name"
    STOCK = "stock"

class OutputFormat(str, Enum):
    """Output format options."""
    TABLE = "table"
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"  # Excel format

class OverrideSource(str, Enum):
    """Sources for reorder level overrides."""
    ALL = "all"
    CLI = "cli"
    ENV = "env"
    CONFIG = "config"
    NONE = "none"

def calculate_urgency_score(alert: StockAlert) -> float:
    """
    Calculate an urgency score from 0-100 for reordering.
    
    The urgency score is calculated based on:
    1. How far below reorder level (percentage of deficit vs reorder level)
    2. Whether the item is completely out of stock
    3. Time since first detected (based on timestamp)
    
    Args:
        alert: The StockAlert to calculate urgency for
        
    Returns:
        A score from 0-100 where 100 is the most urgent
    """
    # Base score from the percentage below reorder level
    if alert.reorder_level == 0:
        base_percentage = 100 if alert.current_stock <= 0 else 0
    else:
        base_percentage = min(100, (alert.deficit / alert.reorder_level) * 100)
    
    # Additional score if completely out of stock (multiply by 1.25)
    out_of_stock_factor = 1.25 if alert.is_out_of_stock else 1.0
    
    # Additional score based on time since alert was created
    # Newer alerts get less urgency, older alerts become more urgent
    now = datetime.datetime.now()
    hours_since_alert = (now - alert.timestamp).total_seconds() / 3600
    time_factor = min(1.2, 1.0 + (hours_since_alert / 168))  # Max 20% increase after 1 week
    
    # Combine all factors
    urgency = base_percentage * out_of_stock_factor * time_factor
    
    # Cap at 100
    return min(100, urgency)

def get_urgency_category(score: float) -> str:
    """Get the category name for an urgency score."""
    if score >= URGENCY_THRESHOLDS["critical"]:
        return "critical"
    elif score >= URGENCY_THRESHOLDS["high"]:
        return "high"
    elif score >= URGENCY_THRESHOLDS["medium"]:
        return "medium"
    elif score >= URGENCY_THRESHOLDS["low"]:
        return "low"
    else:
        return "minimal"

def get_urgency_color(category: str) -> str:
    """Get the display color for an urgency category."""
    colors = {
        "critical": "bright_red",
        "high": "red",
        "medium": "yellow",
        "low": "green",
        "minimal": "blue"
    }
    return colors.get(category, "white")

def get_excel_color(category: str) -> str:
    """Get Excel-compatible color for an urgency category."""
    colors = {
        "critical": "#FF0000",  # Bright Red
        "high": "#CC0000",      # Red
        "medium": "#FFCC00",    # Yellow
        "low": "#00CC00",       # Green
        "minimal": "#0066CC"    # Blue
    }
    return colors.get(category, "#FFFFFF")

class UrgencyBar(BarColumn):
    """
    Custom implementation of BarColumn that changes color based on urgency level.
    """
    def __init__(
        self,
        *args,
        complete_style: Optional[Style] = None,
        table_column: Optional[str] = None,
        **kwargs,
    ):
        self.table_column = table_column
        super().__init__(*args, **kwargs)
        
    def render(self, task: "Task") -> Text:
        """Render a bar chart."""
        
        completed = task.completed
        total = task.total
        
        if completed < 0 or total <= 0:
            return Text(" " * self.width, style="bar.back")
        
        # Calculate percentage and urgency category
        percentage = min(100, max(0, completed / total * 100))
        category = get_urgency_category(percentage)
        color = get_urgency_color(category)
        
        # Calculate bar width
        width = min(self.width, self.width * completed / total)
        bar = Text("━" * math.ceil(width), style=f"bar.complete {color}")
        
        # If we didn't complete the bar, add the pulse block
        if width < self.width:
            pulse_width = max(1, width % 1 * self.width)
            bar.append("━" * math.ceil(pulse_width), style=f"bar.pulse {color}")
        
        # Add the rest of the bar
        bar.pad_right(self.width)
        return bar

def format_alert_for_display(
    alert: StockAlert, 
    urgency_score: float,
    max_width: int = 100,
    show_overrides: bool = True
) -> Dict[str, Any]:
    """Format a stock alert for display in a table or JSON output."""
    
    urgency_category = get_urgency_category(urgency_score)
    urgency_color = get_urgency_color(urgency_category)
    
    result = {
        "sku": alert.product_sku,
        "name": alert.product_name[:max_width],
        "current_stock": alert.current_stock,
        "reorder_level": alert.reorder_level,
        "deficit": alert.deficit,
        "urgency_score": urgency_score,
        "urgency_category": urgency_category,
        "urgency_color": urgency_color,
        "is_out_of_stock": alert.is_out_of_stock,
    }
    
    # Add override information if requested
    if show_overrides and alert.was_overridden:
        result.update({
            "has_override": True,
            "original_reorder_level": alert.original_reorder_level,
            "override_source": alert.override_source or "unknown"
        })
    else:
        result.update({
            "has_override": False,
            "original_reorder_level": alert.original_reorder_level,
            "override_source": None
        })
    
    return result

def render_alert_table(
    alerts: List[Tuple[StockAlert, float]],
    show_bars: bool = True,
    show_overrides: bool = True
) -> Table:
    """
    Render a rich table with alert information.
    
    Args:
        alerts: List of tuples containing (alert, urgency_score)
        show_bars: Whether to show urgency bars or just numerical scores
        show_overrides: Whether to show reorder level override information
    
    Returns:
        A rich Table object ready for display
    """
    table = Table(
        title="Products Requiring Reorder",
        box=ROUNDED,
        highlight=True,
        header_style="bold cyan"
    )
    
    # Add table columns
    table.add_column("SKU", style="dim")
    table.add_column("Product Name")
    table.add_column("Stock", justify="right")
    table.add_column("Reorder Level", justify="right")
    
    # Add override column if showing overrides
    if show_overrides:
        table.add_column("Original Level", justify="right", style="dim")
    
    table.add_column("Deficit", justify="right", style="bold")
    
    # Add urgency column - either a bar or a numerical score
    if show_bars:
        table.add_column("Urgency", width=30)
    else:
        table.add_column("Urgency", justify="right")
    
    # Add rows to the table
    for alert, urgency_score in alerts:
        formatted = format_alert_for_display(alert, urgency_score, show_overrides=show_overrides)
        
        # Create status column with appropriate color
        urgency_category = formatted["urgency_category"]
        color = formatted["urgency_color"]
        
        # Create stock status with color formatting
        stock_text = str(formatted["current_stock"])
        if formatted["is_out_of_stock"]:
            stock_text = f"[bright_red]{stock_text}[/bright_red]"
            
        # Format reorder level to show if it was overridden
        if formatted["has_override"]:
            reorder_text = f"[cyan]{formatted['reorder_level']}*[/cyan]"
        else:
            reorder_text = str(formatted["reorder_level"])
            
        # Display the urgency
        if show_bars:
            # Create a progress bar for the urgency
            progress = Progress(
                TextColumn("{task.percentage:.0f}%", style=color),
                UrgencyBar(complete_style=color, table_column="Urgency"),
                expand=True,
                padding=(0, 1)
            )
            # Add the task and update it immediately to the correct percentage
            task_id = progress.add_task("", total=100, completed=urgency_score)
            
            # Create a renderable group with the progress bar
            urgency_display = Group(progress)
        else:
            # Just show the numerical score with color
            urgency_display = f"[{color}]{urgency_score:.1f}%[/{color}] ({urgency_category})"
        
        # Build row data
        row_data = [
            formatted["sku"],
            formatted["name"],
            stock_text,
            reorder_text,
        ]
        
        # Add original level if showing overrides
        if show_overrides:
            row_data.append(str(formatted["original_reorder_level"]))
        
        # Add deficit and urgency
        row_data.extend([
            f"[bold]{formatted['deficit']}[/bold]",
            urgency_display
        ])
        
        # Add the row to the table
        table.add_row(*row_data)
        
    # Add a legend if we have overrides
    if show_overrides and any(alert.was_overridden for alert, _ in alerts):
        table.caption = "[cyan]*[/cyan] Indicates reorder level has been overridden from original value"
        
    return table

def parse_sku_overrides(overrides: List[str]) -> Dict[str, int]:
    """
    Parse SKU overrides from command line arguments.
    
    Args:
        overrides: List of strings in format "SKU:level", e.g. ["ABC123:50", "XYZ789:100"]
        
    Returns:
        Dictionary mapping SKUs to override values
    """
    result = {}
    
    for override in overrides:
        parts = override.split(":", 1)
        if len(parts) != 2:
            raise typer.BadParameter(f"Invalid override format: {override}. Use SKU:level format.")
            
        sku, level_str = parts
        
        try:
            level = int(level_str)
            if level < 0:
                raise typer.BadParameter(f"Reorder level must be non-negative: {override}")
                
            result[sku] = level
        except ValueError:
            raise typer.BadParameter(f"Reorder level must be an integer: {override}")
            
    return result

def export_to_excel(
    alerts_with_urgency: List[Tuple[StockAlert, float]],
    filepath: Path,
    include_report: bool = False,
    show_overrides: bool = True
) -> None:
    """
    Export alerts to an Excel file with formatting.
    
    Args:
        alerts_with_urgency: List of tuples containing (alert, urgency_score)
        filepath: Path where the Excel file should be saved
        include_report: Whether to include the summary report
        show_overrides: Whether to show reorder level override information
    """
    try:
        import xlsxwriter
    except ImportError:
        console.print("[bold red]Error:[/bold red] xlsxwriter package is required for Excel export.")
        console.print("Install it with: pip install xlsxwriter")
        raise typer.Exit(1)
    
    # Create a workbook and add a worksheet
    workbook = xlsxwriter.Workbook(filepath)
    worksheet = workbook.add_worksheet("Low Stock Items")
    
    # Add formats for different urgency levels and headers
    header_format = workbook.add_format({
        'bold': True, 
        'bg_color': '#0078D7', 
        'font_color': 'white',
        'border': 1
    })
    
    # Format for overridden values
    override_format = workbook.add_format({
        'font_color': '#0078D7',
        'bold': True
    })
    
    # Formats for urgency categories
    urgency_formats = {
        category: workbook.add_format({
            'bg_color': get_excel_color(category),
            'font_color': 'black' if category in ['low', 'medium', 'minimal'] else 'white',
        })
        for category in ['critical', 'high', 'medium', 'low', 'minimal']
    }
    
    # Format for out of stock items
    out_of_stock_format = workbook.add_format({
        'font_color': 'red',
        'bold': True
    })
    
    # Standard format for numbers
    number_format = workbook.add_format({'num_format': '0'})
    
    # Format for percentage
    percent_format = workbook.add_format({'num_format': '0.0%'})
    
    # Build headers
    headers = [
        "SKU", "Product Name", "Current Stock", "Reorder Level"
    ]
    
    # Add original level if showing overrides
    if show_overrides:
        headers.append("Original Level")
    
    # Add remaining headers
    headers.extend(["Deficit", "Urgency Score", "Urgency Level"])
    
    # Add override source if showing overrides
    if show_overrides:
        headers.append("Override Source")
    
    # Write headers
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)
    
    # Auto-filter for headers
    worksheet.autofilter(0, 0, 0, len(headers) - 1)
    
    # Write data rows
    for row, (alert, urgency_score) in enumerate(alerts_with_urgency, start=1):
        urgency_category = get_urgency_category(urgency_score)
        
        # Format stock numbers in red if out of stock
        stock_format = out_of_stock_format if alert.is_out_of_stock else number_format
        
        # Format reorder level to indicate if overridden
        reorder_format = override_format if alert.was_overridden else number_format
        
        # Current column index
        col = 0
        
        # Write each column
        worksheet.write(row, col, alert.product_sku); col += 1
        worksheet.write(row, col, alert.product_name); col += 1
        worksheet.write_number(row, col, alert.current_stock, stock_format); col += 1
        worksheet.write_number(row, col, alert.reorder_level, reorder_format); col += 1
        
        # Add original level if showing overrides
        if show_overrides:
            worksheet.write_number(row, col, alert.original_reorder_level, number_format); col += 1
            
        worksheet.write_number(row, col, alert.deficit, number_format); col += 1
        worksheet.write_number(row, col, urgency_score / 100, percent_format); col += 1  # Convert to Excel percentage
        worksheet.write(row, col, urgency_category.title(), urgency_formats[urgency_category]); col += 1
        
        # Add override source if showing overrides
        if show_overrides:
            source = alert.override_source or "N/A" if alert.was_overridden else "N/A"
            worksheet.write(row, col, source); col += 1
    
    # Set column widths
    worksheet.set_column(0, 0, 15)  # SKU
    worksheet.set_column(1, 1, 40)  # Product Name
    worksheet.set_column(2, worksheet.dim_colmax, 15)  # All other columns
    
    # If we need to include a summary report, add another sheet
    if include_report:
        report = generate_reorder_report([alert for alert, _ in alerts_with_urgency])
        report_sheet = workbook.add_worksheet("Summary Report")
        
        # Headers for summary data
        report_sheet.write(0, 0, "Metric", header_format)
        report_sheet.write(0, 1, "Value", header_format)
        
        # Summary data
        summary_data = [
            ("Report Date", report["report_date"]),
            ("Total Alerts", report["total_alerts"]),
            ("Out of Stock Items", report["out_of_stock_count"]),
            ("Critical Items", report["critical_count"]),
            ("Total Units Needed", report["total_items_needed"]),
            ("Items with Overrides", report.get("override_count", 0)),
        ]
        
        for row, (metric, value) in enumerate(summary_data, start=1):
            report_sheet.write(row, 0, metric)
            report_sheet.write(row, 1, value)
        
        # Add override source breakdown if present
        if "override_sources" in report and report["override_sources"]:
            report_sheet.write(8, 0, "Override Sources", header_format)
            report_sheet.write(8, 1, "Count", header_format)
            
            row = 9
            sources = report["override_sources"]
            if "global_override" in sources:
                report_sheet.write(row, 0, "Global Override")
                report_sheet.write(row, 1, sources["global_override"])
                row += 1
                
            if "sku_override" in sources:
                report_sheet.write(row, 0, "SKU-Specific Override")
                report_sheet.write(row, 1, sources["sku_override"])
                row += 1
        
        # Add breakdown by priority
        report_sheet.write(11, 0, "Priority Level", header_format)
        report_sheet.write(11, 1, "Count", header_format)
        
        row = 12
        for priority in sorted(report["priority_breakdown"].keys()):
            priority_label = f"Priority {priority}"
            if priority == 1:
                priority_label += " (Out of Stock)"
            elif priority == 2:
                priority_label += " (Very Low)"
            elif priority == 3:
                priority_label += " (Low)"
            elif priority == 4:
                priority_label += " (Moderate)"
            else:
                priority_label += " (Approaching)"
                
            report_sheet.write(row, 0, priority_label)
            report_sheet.write(row, 1, report["priority_breakdown"][priority])
            row += 1
            
        # Set column widths for report sheet
        report_sheet.set_column(0, 0, 30)
        report_sheet.set_column(1, 1, 15)
            
    # Close the workbook
    workbook.close()

@reorder_app.command("list")
def reorder_list(
    sort_by: SortOrder = typer.Option(
        SortOrder.URGENCY,
        "--sort", "-s",
        help="Field to sort by"
    ),
    limit: int = typer.Option(
        50, 
        "--limit", "-l",
        help="Maximum number of items to display"
    ),
    all_items: bool = typer.Option(
        False, 
        "--all", "-a",
        help="Show all items below reorder level, not just critical ones"
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--output", "-o",
        help="Output format (table, json, csv, xlsx)"
    ),
    show_bars: bool = typer.Option(
        True,
        "--bars/--no-bars",
        help="Show graphical urgency bars in table output"
    ),
    show_overrides: bool = typer.Option(
        True,
        "--show-overrides/--hide-overrides",
        help="Show reorder level override information"
    ),
    global_level: Optional[int] = typer.Option(
        None,
        "--global-level", "-g",
        help="Global reorder level to override all products"
    ),
    sku_overrides: List[str] = typer.Option(
        [],
        "--sku-override", "-k",
        help="SKU-specific overrides in format SKU:level, can be specified multiple times"
    ),
    override_sources: OverrideSource = typer.Option(
        OverrideSource.ALL,
        "--override-source",
        help="Which override sources to use"
    ),
    refresh_cache: bool = typer.Option(
        False,
        "--refresh",
        help="Force refresh of alert cache"
    ),
    export_path: Optional[Path] = typer.Option(
        None,
        "--export", "-e",
        help="Export results to a file"
    )
):
    """
    List products that need to be reordered, sorted by urgency.
    
    Reorder levels can be overridden globally or per-SKU using command options,
    environment variables (REORDER_LEVEL_SKU123=50), or configuration file.
    """
    # Validate output format and export path
    if export_path:
        # If the output format doesn't match the file extension, adjust the format
        if output == OutputFormat.TABLE and export_path.suffix.lower() not in [".txt", ".ansi"]:
            # Try to infer format from file extension
            if export_path.suffix.lower() == ".json":
                output = OutputFormat.JSON
            elif export_path.suffix.lower() == ".csv":
                output = OutputFormat.CSV
            elif export_path.suffix.lower() in [".xlsx", ".xls"]:
                output = OutputFormat.XLSX
                
        # Make sure the extension matches the format for Excel files
        if output == OutputFormat.XLSX and export_path.suffix.lower() not in [".xlsx", ".xls"]:
            export_path = Path(f"{export_path}.xlsx")

    # Clear cache if requested
    if refresh_cache:
        clear_alert_cache()

    # Parse SKU overrides 
    parsed_sku_overrides = parse_sku_overrides(sku_overrides) if sku_overrides else None
    
    # Determine which override sources to use
    include_env_overrides = override_sources in [OverrideSource.ALL, OverrideSource.ENV]
    include_config_overrides = override_sources in [OverrideSource.ALL, OverrideSource.CONFIG]
    
    # If NONE, don't use any overrides
    if override_sources == OverrideSource.NONE:
        global_level = None
        parsed_sku_overrides = None
        include_env_overrides = False
        include_config_overrides = False
    
    # If CLI only, just use CLI overrides
    if override_sources == OverrideSource.CLI and not (global_level or parsed_sku_overrides):
        include_env_overrides = False
        include_config_overrides = False

    # Set up Rich text for better progress display
    with console.status(f"[bold blue]Scanning inventory for low stock items...") as status:
        # Get alerts from the alert system
        if all_items:
            alerts = detect_low_stock(
                global_override=global_level,
                sku_overrides=parsed_sku_overrides,
                include_env_overrides=include_env_overrides,
                include_config_overrides=include_config_overrides
            )
            status.update(f"[bold blue]Found {len(alerts)} products below reorder level")
        else:
            alerts = get_critical_stock_alerts(
                global_override=global_level,
                sku_overrides=parsed_sku_overrides,
                include_env_overrides=include_env_overrides,
                include_config_overrides=include_config_overrides
            )
            status.update(f"[bold blue]Found {len(alerts)} critical products below reorder level")
        
        # Calculate urgency scores for all alerts
        status.update("[bold blue]Calculating urgency scores...")
        alerts_with_urgency = [(alert, calculate_urgency_score(alert)) for alert in alerts]
        
        # Sort the alerts based on the sort field
        status.update("[bold blue]Sorting results...")
        if sort_by == SortOrder.URGENCY:
            alerts_with_urgency.sort(key=lambda item: item[1], reverse=True)  # Higher urgency first
        elif sort_by == SortOrder.DEFICIT:
            alerts_with_urgency.sort(key=lambda item: item[0].deficit, reverse=True)  # Higher deficit first
        elif sort_by == SortOrder.SKU:
            alerts_with_urgency.sort(key=lambda item: item[0].product_sku)  # Alphabetical by SKU
        elif sort_by == SortOrder.NAME:
            alerts_with_urgency.sort(key=lambda item: item[0].product_name)  # Alphabetical by name
        elif sort_by == SortOrder.STOCK:
            alerts_with_urgency.sort(key=lambda item: item[0].current_stock)  # Lowest stock first
    
        # Limit the results
        alerts_with_urgency = alerts_with_urgency[:limit]
    
    # Count overrides
    override_count = sum(1 for alert, _ in alerts_with_urgency if alert.was_overridden)
        
    if override_count > 0:
        console.print(f"[cyan]Note:[/cyan] {override_count} products have overridden reorder levels.")
    
    # Prepare output based on format
    if output == OutputFormat.TABLE:
        # Create a rich table
        table = render_alert_table(alerts_with_urgency, show_bars=show_bars, show_overrides=show_overrides)
        
        # Display to console or export
        if export_path:
            with open(export_path, "w") as f:
                console = Console(file=f, width=120)
                console.print(table)
            typer.echo(f"Exported reorder list to {export_path}")
        else:
            # Display the table
            console.print()
            console.print(table)
            console.print()
    
    elif output == OutputFormat.JSON:
        # Format as JSON
        result = {
            "generated_at": datetime.datetime.now().isoformat(),
            "items": [
                {
                    "sku": alert.product_sku,
                    "name": alert.product_name,
                    "current_stock": alert.current_stock,
                    "reorder_level": alert.reorder_level,
                    "original_reorder_level": alert.original_reorder_level if show_overrides else None,
                    "deficit": alert.deficit,
                    "urgency": urgency_score,
                    "urgency_category": get_urgency_category(urgency_score),
                    "out_of_stock": alert.is_out_of_stock,
                    "was_overridden": alert.was_overridden if show_overrides else None,
                    "override_source": alert.override_source if show_overrides and alert.was_overridden else None
                }
                for alert, urgency_score in alerts_with_urgency
            ],
            "override_count": override_count if show_overrides else None 
        }
        
        # Output JSON
        json_data = json.dumps(result, indent=2)
        if export_path:
            with open(export_path, "w") as f:
                f.write(json_data)
            typer.echo(f"Exported JSON to {export_path}")
        else:
            typer.echo(json_data)
    
    elif output == OutputFormat.CSV:
        # Format as CSV
        import csv
        
        # Basic headers
        headers = [
            "SKU", "Name", "Current Stock", "Reorder Level"
        ]
        
        # Add override info if needed
        if show_overrides:
            headers.extend(["Original Level", "Was Overridden", "Override Source"])
            
        # Add remaining columns
        headers.extend(["Deficit", "Urgency Score", "Urgency Category", "Out of Stock"])
        
        # Each row in the CSV
        rows = []
        for alert, urgency_score in alerts_with_urgency:
            # Start with basic fields
            row = [
                alert.product_sku,
                alert.product_name,
                alert.current_stock,
                alert.reorder_level,
            ]
            
            # Add override info if showing
            if show_overrides:
                row.extend([
                    alert.original_reorder_level,
                    "Yes" if alert.was_overridden else "No",
                    alert.override_source if alert.was_overridden else "N/A"
                ])
                
            # Add remaining fields
            row.extend([
                alert.deficit,
                f"{urgency_score:.1f}",
                get_urgency_category(urgency_score),
                "Yes" if alert.is_out_of_stock else "No"
            ])
            
            rows.append(row)
        
        if export_path:
            with open(export_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            typer.echo(f"Exported CSV to {export_path}")
        else:
            # Print to console in CSV format
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(headers)
            writer.writerows(rows)
            typer.echo(output.getvalue())
    
    elif output == OutputFormat.XLSX:
        # Make sure we have an export path for Excel
        if not export_path:
            export_path = Path("reorder_list.xlsx")
            
        # Export to Excel with XlsxWriter
        with console.status(f"[bold blue]Generating Excel file..."):
            export_to_excel(
                alerts_with_urgency,
                export_path,
                include_report=True,  # Include summary report for Excel exports
                show_overrides=show_overrides
            )
            
        typer.echo(f"Exported Excel file to {export_path}")

@reorder_app.command("report")
def reorder_report(
    output: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--output", "-o",
        help="Output format (table, json, csv, xlsx)"
    ),
    show_overrides: bool = typer.Option(
        True,
        "--show-overrides/--hide-overrides",
        help="Show reorder level override information"
    ),
    global_level: Optional[int] = typer.Option(
        None,
        "--global-level", "-g",
        help="Global reorder level to override all products"
    ),
    sku_overrides: List[str] = typer.Option(
        [],
        "--sku-override", "-k",
        help="SKU-specific overrides in format SKU:level, can be specified multiple times"
    ),
    override_sources: OverrideSource = typer.Option(
        OverrideSource.ALL,
        "--override-source",
        help="Which override sources to use"
    ),
    refresh_cache: bool = typer.Option(
        False,
        "--refresh",
        help="Force refresh of alert cache"
    ),
    export_path: Optional[Path] = typer.Option(
        None,
        "--export", "-e",
        help="Export report to a file"
    )
):
    """
    Generate a comprehensive report of reordering needs.
    
    Reorder levels can be overridden globally or per-SKU using command options,
    environment variables (REORDER_LEVEL_SKU123=50), or configuration file.
    """
    # Validate output format and export path
    if export_path:
        # Adjust file extension if needed for Excel
        if output == OutputFormat.XLSX and export_path.suffix.lower() not in [".xlsx", ".xls"]:
            export_path = Path(f"{export_path}.xlsx")

    # Clear cache if requested
    if refresh_cache:
        clear_alert_cache()
        
    # Parse SKU overrides
    parsed_sku_overrides = parse_sku_overrides(sku_overrides) if sku_overrides else None
    
    # Determine which override sources to use
    include_env_overrides = override_sources in [OverrideSource.ALL, OverrideSource.ENV]
    include_config_overrides = override_sources in [OverrideSource.ALL, OverrideSource.CONFIG]
    
    # If NONE, don't use any overrides
    if override_sources == OverrideSource.NONE:
        global_level = None
        parsed_sku_overrides = None
        include_env_overrides = False
        include_config_overrides = False
    
    # If CLI only, just use CLI overrides
    if override_sources == OverrideSource.CLI and not (global_level or parsed_sku_overrides):
        include_env_overrides = False
        include_config_overrides = False

    with console.status("[bold blue]Generating reorder report...") as status:
        # Generate the report
        report = generate_reorder_report(
            global_override=global_level,
            sku_overrides=parsed_sku_overrides,
            include_env_overrides=include_env_overrides,
            include_config_overrides=include_config_overrides
        )
        
        # If exporting to Excel, we'll need the original alerts with urgency scores
        if output == OutputFormat.XLSX:
            alerts = detect_low_stock(
                global_override=global_level,
                sku_overrides=parsed_sku_overrides,
                include_env_overrides=include_env_overrides,
                include_config_overrides=include_config_overrides
            )
            alerts_with_urgency = [(alert, calculate_urgency_score(alert)) for alert in alerts]
        
        # Create a summary table
        summary_table = Table(title="Reorder Summary", box=ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Alerts", str(report["total_alerts"]))
        summary_table.add_row("Out of Stock Items", str(report["out_of_stock_count"]))
        summary_table.add_row("Critical Items", str(report["critical_count"]))
        summary_table.add_row("Total Units Needed", str(report["total_items_needed"]))
        
        # Add override info if available
        if show_overrides and "override_count" in report:
            summary_table.add_row("Items with Overrides", str(report["override_count"]))
            
            # Show breakdown by override source if any overrides were used
            if report["override_count"] > 0 and "override_sources" in report:
                summary_table.add_row("", "")  # Empty row as separator
                sources = report["override_sources"]
                
                if "global_override" in sources and sources["global_override"] > 0:
                    summary_table.add_row("  Global Overrides", str(sources["global_override"]))
                    
                if "sku_override" in sources and sources["sku_override"] > 0:
                    summary_table.add_row("  SKU Overrides", str(sources["sku_override"]))
        
        # Create a priority breakdown table
        priority_table = Table(title="Alerts by Priority", box=ROUNDED)
        priority_table.add_column("Priority Level", style="cyan")
        priority_table.add_column("Count", style="green")
        
        for priority, count in sorted(report["priority_breakdown"].items()):
            # Priority 1 is highest
            if priority == 1:
                priority_level = "1 - Out of Stock"
                style = "bright_red"
            elif priority == 2:
                priority_level = "2 - Very Low"
                style = "red"
            elif priority == 3:
                priority_level = "3 - Low" 
                style = "yellow"
            elif priority == 4:
                priority_level = "4 - Moderate"
                style = "green"
            else:
                priority_level = "5 - Approaching"
                style = "blue"
                
            priority_table.add_row(
                f"[{style}]{priority_level}[/{style}]", 
                str(count)
            )
    
    # Display the report based on output format
    if output == OutputFormat.TABLE:
        # Create a panel to contain both tables
        if export_path:
            with open(export_path, "w") as f:
                console = Console(file=f, width=120)
                console.print(Panel(Group(summary_table, priority_table), 
                             title="Reorder Report", 
                             subtitle=f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"))
            typer.echo(f"Exported reorder report to {export_path}")
        else:
            console.print()
            console.print(Panel(Group(summary_table, priority_table), 
                         title="Reorder Report", 
                         subtitle=f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"))
            console.print()
            
    elif output == OutputFormat.JSON:
        # Format as JSON (already in good structure)
        json_data = json.dumps(report, indent=2, default=str)
        if export_path:
            with open(export_path, "w") as f:
                f.write(json_data)
            typer.echo(f"Exported JSON report to {export_path}")
        else:
            typer.echo(json_data)
            
    elif output == OutputFormat.CSV:
        # For CSV, we'll flatten the structure a bit
        import csv
        
        # Prepare main summary rows
        rows = [
            ["Report Date", report["report_date"]],
            ["Total Alerts", report["total_alerts"]],
            ["Out of Stock Items", report["out_of_stock_count"]],
            ["Critical Items", report["critical_count"]],
            ["Total Units Needed", report["total_items_needed"]],
        ]
        
        # Add override info if showing
        if show_overrides and "override_count" in report:
            rows.append(["Items with Overrides", report["override_count"]])
            
            if report["override_count"] > 0 and "override_sources" in report:
                sources = report["override_sources"]
                
                if "global_override" in sources:
                    rows.append(["Global Overrides", sources["global_override"]])
                
                if "sku_override" in sources:
                    rows.append(["SKU Overrides", sources["sku_override"]])
        
        rows.append([""])  # Empty row as separator
        rows.append(["Priority Level", "Count"])  # Header for priority breakdown
        
        # Add priority breakdown
        for priority in sorted(report["priority_breakdown"].keys()):
            rows.append([f"Priority {priority}", report["priority_breakdown"][priority]])
            
        if export_path:
            with open(export_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            typer.echo(f"Exported CSV report to {export_path}")
        else:
            # Print to console in CSV format
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerows(rows)
            typer.echo(output.getvalue())
            
    elif output == OutputFormat.XLSX:
        # Make sure we have an export path for Excel
        if not export_path:
            export_path = Path("reorder_report.xlsx")
            
        # For a complete report, we'll create a workbook with multiple sheets
        try:
            import xlsxwriter
        except ImportError:
            console.print("[bold red]Error:[/bold red] xlsxwriter package is required for Excel export.")
            console.print("Install it with: pip install xlsxwriter")
            raise typer.Exit(1)
            
        # Create the Excel file
        with console.status(f"[bold blue]Generating Excel report..."):
            # Export alerts with urgency to one sheet
            export_to_excel(
                alerts_with_urgency,
                export_path,
                include_report=True,
                show_overrides=show_overrides
            )
            
        typer.echo(f"Exported Excel report to {export_path}")

@reorder_app.command("clear-cache")
def clear_caches():
    """
    Clear the alert cache to force a fresh scan on next command.
    """
    clear_alert_cache()
    console.print("[green]Alert cache cleared. Next scan will refresh all data.[/green]")

if __name__ == "__main__":
    reorder_app()