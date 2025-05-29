#!/usr/bin/env python3
"""
Inventory Tracker - Main entry point with automatic command discovery.

This module provides the main CLI interface for the Inventory Tracker application
and automatically discovers and registers all command modules.
"""
import importlib
import inspect
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any, Tuple

import typer

from inventorytracker.config import load_cli_config, InventoryTrackerConfig
from inventorytracker.logging import add_logging_options, configure_logging, logger

# Create the Typer app
app = typer.Typer(help="Inventory Tracker - Manage your inventory from the command line")

# Add logging options (--debug, --log-file)
app = add_logging_options(app)

# Global configuration
config: Optional[InventoryTrackerConfig] = None


@app.callback()
def main(
    ctx: typer.Context,
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
    ignore_unknown: bool = typer.Option(
        False, "--ignore-unknown-config", help="Ignore unknown config keys"
    ),
    measure_startup: bool = typer.Option(
        False, "--measure-startup", help="Measure and report startup time"
    ),
) -> None:
    """
    Inventory Tracker - Manage your inventory from the command line.
    
    Run a command with --help to see its specific options.
    """
    # Record startup time if requested
    start_time = time.time() if measure_startup else None
    
    # Initialize context object if not already present
    if ctx.obj is None:
        ctx.obj = {}
    
    # Store raw CLI args in context for potential use elsewhere
    ctx.obj["cli_args"] = sys.argv
    
    global config
    
    # Load configuration
    config = load_cli_config(config_file=config_file, ignore_unknown=ignore_unknown)
    
    # Configure logging based on loaded config
    # Note: Basic logging is already set up by add_logging_options,
    # but we can refine it based on the full config
    configure_logging(config=config, cli_args=ctx.obj.get("cli_args"))
    
    # Log application startup
    logger.info(f"Inventory Tracker v{config.version} starting")
    logger.debug(f"Using {config.storage_type.value} storage at {config.storage_path}")
    
    # Report startup time if requested
    if measure_startup and start_time:
        elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
        logger.info(f"Startup time: {elapsed:.2f} ms")
        if ctx.invoked_subcommand is None:
            typer.echo(f"Startup complete in {elapsed:.2f} ms")


def discover_and_register_commands() -> Dict[str, List[str]]:
    """
    Discover and register all command modules.
    
    This function scans the commands directory, imports each module,
    and registers any functions decorated with @app.command().
    
    Returns:
        Dict[str, List[str]]: Map of module names to registered command names
    """
    start_time = time.time()
    
    # Get the directory containing command modules
    commands_dir = Path(__file__).parent / "commands"
    
    # Track registered commands
    registered_commands: Dict[str, List[str]] = {}
    
    # Ensure the commands directory exists
    if not commands_dir.exists() or not commands_dir.is_dir():
        logger.warning(f"Commands directory not found: {commands_dir}")
        return registered_commands
    
    # Get all .py files in the commands directory
    py_files = [f for f in commands_dir.glob("*.py") if f.name != "__init__.py"]
    
    # Package prefix for imports
    package_prefix = "inventorytracker.commands."
    
    # Import each module
    for py_file in py_files:
        module_name = py_file.stem  # Get filename without extension
        module_path = f"{package_prefix}{module_name}"
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Track commands registered by this module
            registered_commands[module_name] = []
            
            # Log discovery
            logger.debug(f"Discovered command module: {module_name}")
            
            # No need to do anything else - Typer registers commands when decorators run
            # during module import. We just need to track which commands were registered
            # by inspecting the module.
            
            # Find functions with app.command decorator by checking for _click_params
            for name, obj in inspect.getmembers(module):
                if (inspect.isfunction(obj) and 
                    hasattr(obj, "_click_params") and  # Has been processed by Click/Typer
                    not name.startswith("_")):  # Not a private function
                    registered_commands[module_name].append(name)
                    logger.debug(f"  - Registered command: {name}")
            
        except Exception as e:
            logger.error(f"Error loading command module {module_name}: {str(e)}")
    
    elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
    logger.debug(f"Command discovery completed in {elapsed:.2f} ms")
    
    return registered_commands


# Discover and register commands during import
discovered_commands = discover_and_register_commands()


# Add a command to list available commands with their modules
@app.command("commands")
def list_commands():
    """List all available commands and their modules."""
    typer.echo("Available commands by module:")
    
    for module_name, commands in discovered_commands.items():
        typer.echo(f"\n[{module_name}]")
        for cmd in commands:
            typer.echo(f"  • {cmd}")


if __name__ == "__main__":
    app()