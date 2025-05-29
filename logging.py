#!/usr/bin/env python3
"""
Logging configuration for the Inventory Tracker application.

Features:
- Rich console formatting for beautiful development logs
- Rotating file handlers for production logging
- Support for quick debug mode activation via --debug or DEBUG=1
- Integration with the application's configuration system
"""
import logging
import os
import sys
from enum import Enum
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

# Import our application's config module
from inventorytracker.config import InventoryTrackerConfig, LogLevel, load_config


# Install Rich traceback handler for beautiful exception display
install_rich_traceback(show_locals=True)


# Constants
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
RICH_LOG_FORMAT = "%(message)s"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5
APP_NAME = "inventorytracker"


# Map between LogLevel enum and logging module levels
LOG_LEVEL_MAP = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}


class LoggingMode(str, Enum):
    """Mode for logging configuration."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"


def get_environment_log_level() -> Optional[int]:
    """
    Check for debug flags in environment variables.
    
    Supports:
    - DEBUG=1
    - INVTRACK_DEBUG=1
    - LOGLEVEL=DEBUG (or other level names)
    - INVTRACK_LOGLEVEL=DEBUG (or other level names)
    
    Returns:
        Optional[int]: Logging level or None if not specified in environment
    """
    # Check explicit debug flags
    if os.environ.get("DEBUG") == "1" or os.environ.get("INVTRACK_DEBUG") == "1":
        return logging.DEBUG

    # Check log level from environment
    log_level_str = os.environ.get("LOGLEVEL") or os.environ.get("INVTRACK_LOGLEVEL")
    if log_level_str:
        # Convert to upper case for case-insensitive matching
        log_level_str = log_level_str.upper()
        
        # Check if it's a valid level name (DEBUG, INFO, etc.)
        numeric_level = getattr(logging, log_level_str, None)
        if isinstance(numeric_level, int):
            return numeric_level
            
    return None


def check_cli_debug_flag(args: List[str] = None) -> bool:
    """
    Check if --debug flag is present in CLI arguments.
    
    Args:
        args: List of command-line arguments, defaults to sys.argv
        
    Returns:
        bool: True if --debug flag is present
    """
    if args is None:
        args = sys.argv[1:]
        
    return "--debug" in args


def get_console_handler(rich: bool = True) -> logging.Handler:
    """
    Get a console handler for logging.
    
    Args:
        rich: Whether to use Rich formatting
        
    Returns:
        logging.Handler: Configured console handler
    """
    if rich:
        # Create a Rich console with appropriate settings
        console = Console(color_system="auto", width=None, highlight=True)
        
        # Create a Rich handler that outputs to the console
        handler = RichHandler(
            console=console,
            show_time=False,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_extra_lines=3,
            tracebacks_show_locals=True,
        )
        # Rich handler includes a lot of formatting already
        handler.setFormatter(logging.Formatter(RICH_LOG_FORMAT))
    else:
        # Standard stream handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        
    return handler


def get_file_handler(log_file: Path, max_bytes: int = MAX_LOG_SIZE, backup_count: int = LOG_BACKUP_COUNT) -> logging.Handler:
    """
    Get a rotating file handler for logging.
    
    Args:
        log_file: Path to the log file
        max_bytes: Maximum size of log file before rotating
        backup_count: Number of backup files to keep
        
    Returns:
        logging.Handler: Configured file handler
    """
    # Ensure directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a rotating file handler
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    
    return handler


@lru_cache(maxsize=1)
def get_effective_log_level(
    config: InventoryTrackerConfig = None,
    cli_args: List[str] = None,
) -> int:
    """
    Determine the effective log level based on multiple sources with priority:
    1. CLI --debug flag
    2. Environment variables (DEBUG, LOGLEVEL)
    3. Configuration setting
    
    Args:
        config: Application configuration
        cli_args: Command-line arguments
        
    Returns:
        int: Logging module log level (logging.DEBUG, logging.INFO, etc.)
    """
    # Check for --debug flag (highest priority)
    if cli_args is not None and check_cli_debug_flag(cli_args):
        return logging.DEBUG
    
    # Check environment variables (second priority)
    env_level = get_environment_log_level()
    if env_level is not None:
        return env_level
    
    # Fall back to configuration (lowest priority)
    if config is not None:
        return LOG_LEVEL_MAP.get(config.log_level, logging.INFO)
    
    # Default if nothing else is specified
    return logging.INFO


def configure_logging(
    config: Optional[InventoryTrackerConfig] = None,
    mode: LoggingMode = LoggingMode.DEVELOPMENT,
    log_level: Optional[Union[int, str, LogLevel]] = None,
    log_file: Optional[Path] = None,
    cli_args: Optional[List[str]] = None,
) -> None:
    """
    Configure application logging.
    
    Args:
        config: Application configuration (optional)
        mode: Logging mode (development or production)
        log_level: Override log level
        log_file: Override log file path
        cli_args: Command-line arguments to check for --debug
    """
    # Load configuration if not provided
    if config is None:
        config = load_config(cli_args=cli_args)
    
    # Determine effective log level considering all sources
    effective_level = logging.INFO
    
    if log_level is not None:
        # Use the explicitly provided level if one was given
        if isinstance(log_level, int):
            effective_level = log_level
        elif isinstance(log_level, str):
            # Convert string level to int
            numeric_level = getattr(logging, log_level.upper(), None)
            if isinstance(numeric_level, int):
                effective_level = numeric_level
        elif isinstance(log_level, LogLevel):
            # Convert enum to int
            effective_level = LOG_LEVEL_MAP[log_level]
    else:
        # Use the prioritized level from CLI, env vars, or config
        effective_level = get_effective_log_level(config, cli_args)
    
    # Determine log file path
    effective_log_file = None
    if log_file:
        effective_log_file = log_file
    elif config.log_file:
        effective_log_file = config.log_file
    elif mode == LoggingMode.PRODUCTION:
        # Default log file in production mode
        effective_log_file = Path.home() / ".invtrack" / "logs" / "app.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)
    
    # Remove existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # Always add console handler in development mode
    use_rich = mode == LoggingMode.DEVELOPMENT and config.color_output
    root_logger.addHandler(get_console_handler(rich=use_rich))
    
    # Add file handler if needed
    if effective_log_file and (mode == LoggingMode.PRODUCTION or config.log_file is not None):
        root_logger.addHandler(get_file_handler(
            effective_log_file, 
            max_bytes=MAX_LOG_SIZE,
            backup_count=config.backup_count
        ))
        
    # Create our app logger as a convenience for imports
    app_logger = logging.getLogger(APP_NAME)
    
    # Log startup info
    app_logger.debug(f"Logging configured in {mode} mode at level {logging.getLevelName(effective_level)}")
    if effective_log_file:
        app_logger.debug(f"Logging to file: {effective_log_file}")


# Helper for CLI apps
def add_logging_options(app: typer.Typer) -> typer.Typer:
    """
    Add logging-related options to a Typer app.
    
    This decorates the app's callback with debug and log_file options.
    
    Args:
        app: Typer app to modify
        
    Returns:
        typer.Typer: Modified Typer app
    """
    # Get the existing callback
    original_callback = app.callback()
    
    # Define a new callback that wraps the original one
    def new_callback(
        ctx: typer.Context,
        debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
        log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log to specified file"),
        *args, **kwargs
    ):
        # Configure logging
        mode = (
            LoggingMode.DEVELOPMENT 
            if debug or os.environ.get("DEBUG") == "1" 
            else LoggingMode.PRODUCTION
        )
        configure_logging(
            mode=mode,
            log_level=logging.DEBUG if debug else None,
            log_file=log_file,
            cli_args=sys.argv,
        )
        
        # Call the original callback
        if original_callback:
            return original_callback(ctx, *args, **kwargs)
    
    # Replace the app's callback
    app.callback()(new_callback)
    
    return app


# Module-level logger for convenience
logger = logging.getLogger(APP_NAME)


# Configure a basic default logger to avoid errors if used before proper configuration
def _setup_default_logger():
    """Set up a basic default logger to prevent errors before proper configuration."""
    configure_logging(
        mode=LoggingMode.DEVELOPMENT if os.environ.get("DEBUG") == "1" else LoggingMode.PRODUCTION,
        cli_args=sys.argv,
    )


# Initialize a basic logger when module is imported
_setup_default_logger()


# Example usage in "__main__" block
if __name__ == "__main__":
    # Simple demonstration
    configure_logging(mode=LoggingMode.DEVELOPMENT)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    try:
        # Demonstrate rich traceback
        x = 1 / 0
    except Exception as e:
        logger.exception("An exception occurred")