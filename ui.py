# inventorytracker/utils/ui.py
import importlib.util
import re

# Check if rich is available
RICH_AVAILABLE = importlib.util.find_spec("rich") is not None

if RICH_AVAILABLE:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    from rich.text import Text
    
    console = Console()
else:
    # Fallback implementations
    # ... fallback classes as shown earlier ...

def format_success(message):
    """Format a success message."""
    return f"[bold green]{message}[/]" if RICH_AVAILABLE else message

def format_error(message):
    """Format an error message."""
    return f"[bold red]Error:[/] {message}" if RICH_AVAILABLE else f"Error: {message}"

def format_warning(message):
    """Format a warning message."""
    return f"[bold yellow]{message}[/]" if RICH_AVAILABLE else f"Warning: {message}"

def create_summary_panel(title, content_dict):
    """
    Create a summary panel from a dictionary of values.
    
    Args:
        title: Panel title
        content_dict: Dictionary of {label: value} pairs
        
    Returns:
        A Panel or formatted string
    """
    if RICH_AVAILABLE:
        content = "\n".join(f"[bold]{k}:[/] {v}" for k, v in content_dict.items())
        return Panel(
            Text.from_markup(content),
            title=title,
            title_align="center",
            border_style="green",
            padding=(1, 2)
        )
    else:
        result = [f"{title}:"]
        for k, v in content_dict.items():
            result.append(f"  {k}: {v}")
        return "\n".join(result)