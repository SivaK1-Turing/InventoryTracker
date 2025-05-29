#!/usr/bin/env python3
"""
Error handling for the Inventory Tracker application.

This module provides:
1. Global exception handling for the Typer app
2. Custom exception classes for specific error scenarios
3. Error logging with sanitized command context 
4. User-friendly error messages
"""
import functools
import inspect
import os
import re
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

import typer

from inventorytracker.logging import logger


# Custom exceptions for the application
class InventoryTrackerError(Exception):
    """Base class for all Inventory Tracker exceptions."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code or "INVTRK-GEN-ERR"
        super().__init__(message)


class ConfigError(InventoryTrackerError):
    """Error related to configuration issues."""
    def __init__(self, message: str):
        super().__init__(message, "INVTRK-CFG-ERR")


class ValidationError(InventoryTrackerError):
    """Error related to input validation."""
    def __init__(self, message: str):
        super().__init__(message, "INVTRK-VAL-ERR")


class StorageError(InventoryTrackerError):
    """Error related to data storage operations."""
    def __init__(self, message: str):
        super().__init__(message, "INVTRK-STR-ERR")


class ItemNotFoundError(InventoryTrackerError):
    """Error when an item is not found."""
    def __init__(self, message: str):
        super().__init__(message, "INVTRK-NF-ERR")


class PermissionError(InventoryTrackerError):
    """Error related to permission issues."""
    def __init__(self, message: str):
        super().__init__(message, "INVTRK-PERM-ERR")


# Patterns for sensitive information that should be redacted
SENSITIVE_PATTERNS = [
    # Passwords in CLI args or environment vars
    re.compile(r'(?i)(password|passwd|secret|token|key|auth)=([^&\s]+)'),
    re.compile(r'(?i)(api[_\-]?key)=([^&\s]+)'),
    # Credit card numbers
    re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
    # Social security numbers (US)
    re.compile(r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'),
    # Various token formats
    re.compile(r'eyJ[a-zA-Z0-9_-]{5,}\.eyJ[a-zA-Z0-9_-]{5,}'),  # JWT
    re.compile(r'gh[ps]_[A-Za-z0-9_]{36,255}'),  # GitHub tokens
    re.compile(r'sk_live_[A-Za-z0-9]{24,}'),  # Stripe keys
    re.compile(r'sk-[A-Za-z0-9]{32,}'),  # OpenAI keys
]

# Additional CLI args that should be fully redacted by name
SENSITIVE_ARG_NAMES = {
    'password', 'secret', 'key', 'token', 'credential', 'apikey', 'api-key', 
    'api_key', 'auth', 'passphrase'
}


def sanitize_string(text: str) -> str:
    """
    Remove sensitive information from a string.
    
    Args:
        text: The string to sanitize
        
    Returns:
        Sanitized string with sensitive information redacted
    """
    if not text:
        return text
        
    sanitized = text
    
    # Apply all redaction patterns
    for pattern in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(r'\1=[REDACTED]', sanitized)
        
    return sanitized


def sanitize_command_args(args: List[str]) -> List[str]:
    """
    Sanitize command-line arguments to remove sensitive information.
    
    Args:
        args: List of command-line arguments
        
    Returns:
        Sanitized list with sensitive values redacted
    """
    if not args:
        return args
        
    sanitized_args = []
    
    # Process each argument
    i = 0
    while i < len(args):
        arg = args[i]
        
        # Skip the script name
        if i == 0 and not arg.startswith("-"):
            sanitized_args.append(arg)
            i += 1
            continue
            
        # Handle --key=value format
        if "=" in arg and arg.startswith("--"):
            key, value = arg.split("=", 1)
            key_name = key[2:].lower()  # Remove -- prefix
            
            if key_name in SENSITIVE_ARG_NAMES:
                sanitized_args.append(f"{key}=[REDACTED]")
            else:
                sanitized_args.append(f"{key}={sanitize_string(value)}")
                
        # Handle --key value format
        elif arg.startswith("--") or arg.startswith("-"):
            key = arg
            key_name = key[2:].lower() if key.startswith("--") else key[1:].lower()
            sanitized_args.append(key)
            
            # Check if this is a flag or key-value pair
            if (i + 1 < len(args) and not args[i + 1].startswith("-") and 
                key_name in SENSITIVE_ARG_NAMES):
                # This is a sensitive key-value pair
                sanitized_args.append("[REDACTED]")
                i += 1
            elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                # This is a non-sensitive key-value pair
                sanitized_args.append(sanitize_string(args[i + 1]))
                i += 1
        else:
            # Regular argument
            sanitized_args.append(sanitize_string(arg))
            
        i += 1
        
    return sanitized_args


def sanitize_dict(data: Dict[str, Any], depth: int = 3) -> Dict[str, Any]:
    """
    Recursively sanitize a dictionary to remove sensitive information.
    
    Args:
        data: Dictionary to sanitize
        depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        Sanitized dictionary
    """
    if depth <= 0 or not isinstance(data, dict):
        return {"[MAX_DEPTH]": "..."} if depth <= 0 else data
        
    result = {}
    
    for key, value in data.items():
        # Check if this is a sensitive key
        key_str = str(key).lower()
        is_sensitive = any(pattern in key_str for pattern in SENSITIVE_ARG_NAMES)
        
        if is_sensitive:
            # Redact sensitive values
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            result[key] = sanitize_dict(value, depth - 1)
        elif isinstance(value, (list, tuple)):
            # Sanitize lists/tuples
            result[key] = [
                sanitize_dict(item, depth - 1) if isinstance(item, dict)
                else sanitize_string(str(item)) if isinstance(item, str)
                else item
                for item in value
            ]
        elif isinstance(value, str):
            # Sanitize strings
            result[key] = sanitize_string(value)
        else:
            # Keep other values unchanged
            result[key] = value
            
    return result


def generate_error_id() -> str:
    """
    Generate a unique error ID for tracking purposes.
    
    Returns:
        String error ID (truncated UUID)
    """
    return str(uuid.uuid4())[:8]


def get_command_context(app: typer.Typer, ctx: typer.Context) -> Dict[str, Any]:
    """
    Get sanitized context information about the command being executed.
    
    Args:
        app: Typer app instance
        ctx: Typer context
        
    Returns:
        Dictionary with sanitized context information
    """
    context_info = {
        "command": ctx.command_path,
        "command_name": ctx.info_name,
        "invoked_subcommand": ctx.invoked_subcommand,
        "args": sanitize_command_args(sys.argv),
        "parent_info": None,
    }
    
    # Add parent context if available, but avoid sensitive info
    if ctx.parent:
        context_info["parent_info"] = {
            "command": ctx.parent.command_path,
            "command_name": ctx.parent.info_name,
            "invoked_subcommand": ctx.parent.invoked_subcommand,
        }
    
    # Add sanitized parameters (avoiding sensitive values)
    if hasattr(ctx, "params"):
        # Make a copy to avoid modifying the original
        params = ctx.params.copy() if ctx.params else {}
        context_info["params"] = sanitize_dict(params)
    
    # Add sanitized environment information
    env_info = {}
    for key, value in os.environ.items():
        if key.startswith("INVTRACK_"):
            # Only include our app's environment variables
            # and sanitize their values
            env_info[key] = (
                "[REDACTED]" if any(pattern in key.lower() for pattern in SENSITIVE_ARG_NAMES)
                else sanitize_string(value)
            )
    context_info["env"] = env_info
    
    return context_info


def setup_exception_handler(app: typer.Typer) -> None:
    """
    Set up the global exception handler for the Typer app.
    
    This function must be called before the app is run to ensure
    all exceptions are properly handled.
    
    Args:
        app: The Typer app instance to configure
    """
    
    @app.callback(invoke_without_command=True)
    def global_exception_handler(ctx: typer.Context):
        # Store the context for potential exception handling
        ctx.ensure_object(dict)
        ctx.obj["_start_time"] = datetime.now()
    
    # Create a decorator for exception handling
    def exception_handler(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Generate an error ID for tracking
                error_id = generate_error_id()
                
                # Get context if available (from Typer command)
                ctx = None
                command_context = {}
                
                # Try to find Typer context in args or kwargs
                for arg in args:
                    if isinstance(arg, typer.Context):
                        ctx = arg
                        break
                
                if ctx is None and "ctx" in kwargs and isinstance(kwargs["ctx"], typer.Context):
                    ctx = kwargs["ctx"]
                
                if ctx:
                    command_context = get_command_context(app, ctx)
                else:
                    # Fallback to basic command info if no context available
                    command_context = {
                        "args": sanitize_command_args(sys.argv),
                        "command": sys.argv[0] if sys.argv else "unknown",
                    }
                
                # Determine if this is one of our custom exceptions
                is_known_error = isinstance(e, InventoryTrackerError)
                error_code = getattr(e, "error_code", "INVTRK-UNK-ERR") if is_known_error else "INVTRK-UNK-ERR"
                
                # Get the full traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                tb_text = "".join(tb_lines)
                
                # Log the detailed error with context
                logger.error(
                    f"Exception occurred [ID: {error_id}] [Code: {error_code}]\n"
                    f"Error: {str(e)}\n"
                    f"Command context: {command_context}\n\n"
                    f"Traceback:\n{tb_text}"
                )
                
                # Provide a user-friendly message
                typer.secho("Error: ", fg=typer.colors.RED, bold=True, nl=False)
                
                if is_known_error:
                    # For known errors, display the actual error message
                    typer.secho(f"{str(e)}", fg=typer.colors.RED)
                else:
                    # For unknown errors, display a generic message with reference ID
                    typer.secho(
                        f"An unexpected error occurred [ID: {error_id}].\n"
                        f"Please check the log file for details.",
                        fg=typer.colors.RED
                    )
                
                # Log file location hint
                typer.secho(
                    f"See log for details: {get_log_file_location()}",
                    fg=typer.colors.YELLOW
                )
                
                # Exit with error code
                sys.exit(1)
        
        return wrapper
    
    # Patch all command functions with the exception handler
    patch_typer_commands(app, exception_handler)


def get_log_file_location() -> str:
    """
    Get the location of the log file.
    
    Returns:
        String path to log file or message about logs
    """
    # Try to access log file from logging configuration
    # This is a simplification - in a real app, you'd access this from your logging config
    default_log_location = Path.home() / ".invtrack" / "logs" / "app.log"
    
    if default_log_location.exists():
        return str(default_log_location)
    
    # Fallback
    return "application logs"


def patch_typer_commands(app: typer.Typer, decorator: Callable) -> None:
    """
    Patch all Typer commands with a decorator.
    
    This recursively processes the app and all subcommands.
    
    Args:
        app: Typer app to patch
        decorator: Decorator to apply to all commands
    """
    if not hasattr(app, "registered_commands"):
        return
    
    # Process all registered commands
    for command in app.registered_commands:
        if hasattr(command, "callback") and callable(command.callback):
            # Save original callback
            original_callback = command.callback
            
            # Replace with decorated version
            command.callback = decorator(original_callback)
    
    # Process all registered groups (subcommands)
    if hasattr(app, "registered_groups"):
        for group in app.registered_groups:
            if hasattr(group, "typer_instance"):
                # Recursively process subcommands
                patch_typer_commands(group.typer_instance, decorator)


def handle_keyboard_interrupt(func):
    """
    Decorator to handle keyboard interrupts gracefully.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            typer.echo("\nOperation cancelled by user.")
            sys.exit(130)  # Standard exit code for SIGINT
    
    return wrapper


def apply_error_handling(app: typer.Typer) -> typer.Typer:
    """
    Apply error handling to a Typer app.
    
    This function sets up exception handling and keyboard interrupt handling.
    
    Args:
        app: Typer app to configure
        
    Returns:
        Configured Typer app
    """
    # Set up the exception handler
    setup_exception_handler(app)
    
    # Wrap the app's main function with interrupt handling
    original_main = app.__call__
    
    @handle_keyboard_interrupt
    def wrapped_main(*args, **kwargs):
        return original_main(*args, **kwargs)
    
    app.__call__ = wrapped_main
    
    return app