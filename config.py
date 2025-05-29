#!/usr/bin/env python3
"""
Configuration management for Inventory Tracker.

This module handles loading, merging, and validating configuration from:
1. Default values
2. User config file (~/.invtrack/config.toml)
3. Environment variables (prefixed with INVTRACK_)
4. Command-line arguments
"""
from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, cast

import tomli
import typer
from pydantic import BaseModel, BaseSettings, Field, root_validator, validator
from pydantic.error_wrappers import ErrorWrapper, ValidationError
from pydantic.fields import ModelField


# Define configuration schema with Pydantic
class StorageType(str, Enum):
    """Supported storage backend types."""
    JSON = "json"
    SQLITE = "sqlite"
    CSV = "csv"


class LogLevel(str, Enum):
    """Log levels for application logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ServerConfig(BaseModel):
    """Server-related configuration settings."""
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    enable_api: bool = False

    @validator("port")
    def validate_port(cls, v: int) -> int:
        """Ensure port is within valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Port must be between 1-65535, got {v}")
        return v

    @validator("workers")
    def validate_workers(cls, v: int) -> int:
        """Ensure workers is a positive integer."""
        if v < 1:
            raise ValueError(f"Workers must be a positive integer, got {v}")
        return v


class InventoryTrackerConfig(BaseSettings):
    """Main configuration model for Inventory Tracker application."""
    # Application settings
    app_name: str = "Inventory Tracker"
    version: str = "0.1.0"
    
    # Storage settings
    storage_type: StorageType = StorageType.JSON
    storage_path: Path = Path.home() / ".invtrack" / "data"
    backup_enabled: bool = True
    backup_count: int = 5
    
    # Display settings
    color_output: bool = True
    table_format: str = "simple"
    items_per_page: int = 20
    
    # Server settings (if API enabled)
    server: ServerConfig = ServerConfig()
    
    # Logging settings
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[Path] = None
    
    # Validation settings
    enforce_unique_names: bool = True
    allow_negative_stock: bool = False
    
    class Config:
        """Pydantic config for settings."""
        env_prefix = "INVTRACK_"
        env_nested_delimiter = "__"
        case_sensitive = False
        extra = "forbid"  # Raise error on unknown fields

    @root_validator(pre=True)
    def check_unknown_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Identify and report unknown configuration fields."""
        model_fields = {field.alias for field in cls.__fields__.values()}
        server_fields = {f"server__{field.alias}" for field in ServerConfig.__fields__.values()}
        allowed_fields = model_fields.union(server_fields)
        
        unknown_fields = set(values.keys()) - allowed_fields
        if unknown_fields:
            raise ValueError(f"Unknown configuration fields: {', '.join(unknown_fields)}")
        
        return values


# Deep merge utility for configurations
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with values from override taking precedence.
    
    If both values are dictionaries, they are deep-merged recursively.
    Otherwise, the value from override is used.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result


# Load configuration from TOML file
def load_toml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a TOML file.
    
    Returns an empty dict if file doesn't exist or cannot be parsed.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {}
            
        with open(path, "rb") as f:
            return tomli.load(f)
    except (tomli.TOMLDecodeError, PermissionError, IsADirectoryError) as e:
        typer.echo(f"Error loading config file {file_path}: {str(e)}", err=True)
        return {}


# Convert environment variables to config dict
def env_to_config_dict() -> Dict[str, Any]:
    """
    Convert environment variables with INVTRACK_ prefix to a nested config dictionary.
    
    Example: INVTRACK_SERVER__PORT=8080 becomes {'server': {'port': 8080}}
    """
    config_dict: Dict[str, Any] = {}
    prefix = InventoryTrackerConfig.Config.env_prefix
    delimiter = InventoryTrackerConfig.Config.env_nested_delimiter
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and split by delimiter
            config_key = key[len(prefix):]
            
            if delimiter in config_key:
                # Handle nested keys
                parts = config_key.lower().split(delimiter)
                
                # Navigate to the right nested dictionary
                current = config_dict
                for part in parts[:-1]:
                    current.setdefault(part, {})
                    current = current[part]
                    
                # Set the value at the leaf
                current[parts[-1]] = value
            else:
                # Top-level key
                config_dict[config_key.lower()] = value
                
    return config_dict


# Type conversion for CLI arguments
def convert_cli_value(value: str, field_type: Any) -> Any:
    """Convert string CLI value to appropriate type based on field type."""
    if field_type == bool:
        return value.lower() in ("true", "yes", "y", "1")
    elif field_type == int:
        return int(value)
    elif field_type == float:
        return float(value)
    elif field_type == Path:
        return Path(value)
    elif hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
        # Handle Optional types
        for arg in field_type.__args__:
            if arg is not type(None):  # noqa
                try:
                    return convert_cli_value(value, arg)
                except (ValueError, TypeError):
                    continue
        raise ValueError(f"Could not convert {value} to any of {field_type.__args__}")
    elif issubclass(field_type, Enum):
        # Handle Enum types
        return field_type(value)
    
    # Default to returning the string value
    return value


# CLI argument parser
def cli_to_config_dict(cli_args: List[str]) -> Dict[str, Any]:
    """
    Parse CLI arguments in the format --key=value or --nested.key=value.
    
    Example: --server.port=8080 becomes {'server': {'port': 8080}}
    """
    config_dict: Dict[str, Any] = {}
    
    for arg in cli_args:
        if arg.startswith("--") and "=" in arg:
            # Remove -- prefix and split by =
            key, value = arg[2:].split("=", 1)
            
            if "." in key:
                # Handle nested keys
                parts = key.lower().split(".")
                
                # Navigate to the right nested dictionary
                current = config_dict
                for part in parts[:-1]:
                    current.setdefault(part, {})
                    current = current[part]
                    
                # Set the value at the leaf
                field_path = ".".join(parts)
                field_type = get_field_type(field_path)
                if field_type:
                    try:
                        current[parts[-1]] = convert_cli_value(value, field_type)
                    except (ValueError, TypeError):
                        typer.echo(f"Invalid value for {key}: {value}", err=True)
                        continue
                else:
                    current[parts[-1]] = value
            else:
                # Top-level key
                field_type = get_field_type(key)
                if field_type:
                    try:
                        config_dict[key.lower()] = convert_cli_value(value, field_type)
                    except (ValueError, TypeError):
                        typer.echo(f"Invalid value for {key}: {value}", err=True)
                        continue
                else:
                    config_dict[key.lower()] = value
                
    return config_dict


# Helper to get field type from model
def get_field_type(field_path: str) -> Any:
    """
    Get the type of a field in the config model based on its path.
    
    Returns None if field not found.
    """
    parts = field_path.lower().split(".")
    if len(parts) == 1:
        field = InventoryTrackerConfig.__fields__.get(parts[0])
        return field.type_ if field else None
    
    if parts[0] == "server" and len(parts) == 2:
        field = ServerConfig.__fields__.get(parts[1])
        return field.type_ if field else None
    
    return None


# Main configuration loader
def load_config(
    config_file: Optional[Path] = None, 
    cli_args: Optional[List[str]] = None,
    raise_unknown: bool = True
) -> InventoryTrackerConfig:
    """
    Load and merge configuration from all sources.
    
    Order of precedence (highest to lowest):
    1. CLI arguments
    2. Environment variables
    3. User config file
    4. Default values from InventoryTrackerConfig
    
    Args:
        config_file: Optional path to config file. If None, uses default location.
        cli_args: Optional list of CLI args to parse. If None, uses sys.argv.
        raise_unknown: Whether to raise error on unknown config keys.
        
    Returns:
        InventoryTrackerConfig: The loaded and validated configuration.
        
    Raises:
        ValidationError: If configuration is invalid or contains unknown fields.
    """
    # Load config file (use default location if not specified)
    if config_file is None:
        config_file = Path.home() / ".invtrack" / "config.toml"
    
    file_config = load_toml_config(config_file)
    
    # Load environment variables
    env_config = env_to_config_dict()
    
    # Load CLI arguments
    if cli_args is None:
        # Skip the first argument (script name)
        cli_args = sys.argv[1:] if len(sys.argv) > 1 else []
    cli_config = cli_to_config_dict(cli_args)
    
    # Merge configurations with precedence
    merged_config = {}
    merged_config = deep_merge(merged_config, file_config)
    merged_config = deep_merge(merged_config, env_config)
    merged_config = deep_merge(merged_config, cli_config)
    
    # Validate and create config object
    try:
        return InventoryTrackerConfig.parse_obj(merged_config)
    except ValidationError as e:
        # If we don't want to raise on unknown fields, filter those errors
        if not raise_unknown:
            # Keep only errors that aren't about extra/unknown fields
            filtered_errors = [
                err for err in e.raw_errors 
                if not (isinstance(err, ErrorWrapper) and "extra fields" in str(err.exc))
            ]
            
            if not filtered_errors:
                # If there are no errors after filtering, create a config ignoring extra fields
                class TempConfig(BaseSettings):
                    class Config:
                        extra = "ignore"
                        
                # Dynamically create a subclass that ignores extra fields
                temp_config = type(
                    "TempInventoryTrackerConfig", 
                    (InventoryTrackerConfig,), 
                    {"Config": TempConfig.Config}
                )
                return temp_config.parse_obj(merged_config)
            
            # Create a new validation error with the filtered errors
            raise ValidationError(filtered_errors, InventoryTrackerConfig)
        
        # Re-raise the original error
        raise


# Default configuration instance for easy importing
config = load_config()


# CLI helper for loading config with Typer app parameters
def load_cli_config(
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", 
        help="Path to config file (default: ~/.invtrack/config.toml)"
    ),
    ignore_unknown: bool = typer.Option(
        False, "--ignore-unknown-config", 
        help="Ignore unknown configuration keys instead of raising an error"
    ),
) -> InventoryTrackerConfig:
    """
    Helper for Typer apps to load configuration with CLI params.
    
    Use in your main Typer app callback.
    """
    return load_config(
        config_file=config_file,
        cli_args=sys.argv[1:],
        raise_unknown=not ignore_unknown
    )


if __name__ == "__main__":
    # Simple test/demo
    try:
        cfg = load_config()
        print(f"Loaded configuration:")
        print(f"Storage type: {cfg.storage_type}")
        print(f"Storage path: {cfg.storage_path}")
        print(f"Server port: {cfg.server.port}")
    except ValidationError as e:
        print(f"Configuration error: {e}")