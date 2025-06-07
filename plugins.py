# inventory_tracker/analytics/plugins.py

import os
import sys
import importlib
import inspect
import pkgutil
import logging
from typing import Dict, Any, List, Callable, Optional, Set, Type

logger = logging.getLogger("analytics.plugins")

class AnalyticsPlugin:
    """Base class for analytics plugins"""
    
    # Plugin metadata
    name: str = "base_plugin"
    description: str = "Base analytics plugin"
    version: str = "1.0.0"
    author: str = "Unknown"
    
    # Plugin priority (lower numbers run first)
    priority: int = 100
    
    @classmethod
    def get_metrics(cls, product: Any, transactions: List[Any], **kwargs) -> Dict[str, Any]:
        """
        Calculate custom metrics for a product.
        
        Args:
            product: The product to calculate metrics for
            transactions: List of transactions for the product
            kwargs: Additional parameters for the calculation
            
        Returns:
            Dictionary of metric names and values
        """
        # Base implementation returns empty dict
        return {}


class PluginManager:
    """
    Manager for analytics plugins.
    Discovers and loads plugins, then integrates them into analytics calculations.
    """
    
    def __init__(self):
        """Initialize the plugin manager"""
        self.plugins: Dict[str, Type[AnalyticsPlugin]] = {}
        self.loaded_plugins: List[Type[AnalyticsPlugin]] = []
        self.plugin_dirs: List[str] = []
    
    def register_plugin(self, plugin_class: Type[AnalyticsPlugin]) -> bool:
        """
        Register a plugin with the manager.
        
        Args:
            plugin_class: Class derived from AnalyticsPlugin
            
        Returns:
            True if registration succeeded, False otherwise
        """
        if not inspect.isclass(plugin_class) or not issubclass(plugin_class, AnalyticsPlugin):
            logger.warning(f"Attempted to register invalid plugin: {plugin_class}")
            return False
        
        plugin_name = plugin_class.name
        
        if plugin_name in self.plugins:
            logger.warning(f"Plugin '{plugin_name}' is already registered")
            return False
        
        self.plugins[plugin_name] = plugin_class
        self.loaded_plugins.append(plugin_class)
        self.loaded_plugins.sort(key=lambda p: p.priority)
        
        logger.info(f"Registered analytics plugin: {plugin_name} v{plugin_class.version}")
        return True
    
    def add_plugin_directory(self, directory: str) -> None:
        """
        Add a directory to search for plugins.
        
        Args:
            directory: Path to directory containing plugin modules
        """
        if not os.path.isdir(directory):
            logger.warning(f"Plugin directory not found: {directory}")
            return
            
        if directory not in self.plugin_dirs:
            self.plugin_dirs.append(directory)
            
        # Add to Python path to make imports work
        if directory not in sys.path:
            sys.path.append(directory)
    
    def discover_plugins(self) -> int:
        """
        Discover plugins in registered directories.
        
        Returns:
            Number of new plugins discovered
        """
        # Find plugins in the built-in plugins directory
        builtin_plugins_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "plugins", 
            "analytics"
        )
        
        # Add the built-in directory if it exists
        if os.path.isdir(builtin_plugins_dir):
            self.add_plugin_directory(builtin_plugins_dir)
        
        count = 0
        
        # Look through all plugin directories
        for directory in self.plugin_dirs:
            logger.debug(f"Searching for plugins in: {directory}")
            
            # Find all Python modules in the directory
            for _, module_name, is_pkg in pkgutil.iter_modules([directory]):
                try:
                    # Import the module
                    module = importlib.import_module(module_name)
                    
                    # Look for plugin classes
                    for item_name in dir(module):
                        item = getattr(module, item_name)
                        
                        # If it's a plugin class, register it
                        if (
                            inspect.isclass(item) and 
                            issubclass(item, AnalyticsPlugin) and 
                            item is not AnalyticsPlugin
                        ):
                            if self.register_plugin(item):
                                count += 1
                                
                except ImportError as e:
                    logger.error(f"Error importing plugin module {module_name}: {e}")
                except Exception as e:
                    logger.error(f"Error loading plugins from {module_name}: {e}")
        
        logger.info(f"Discovered {count} new analytics plugins")
        return count
    
    def get_plugin_metrics(self, product: Any, transactions: List[Any], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics from all registered plugins for a product.
        
        Args:
            product: The product to get metrics for
            transactions: List of transactions for the product
            kwargs: Additional parameters
            
        Returns:
            Dictionary of plugin names to metric dictionaries
        """
        all_metrics = {}
        
        for plugin_class in self.loaded_plugins:
            try:
                plugin_name = plugin_class.name
                metrics = plugin_class.get_metrics(product, transactions, **kwargs)
                
                if metrics:
                    all_metrics[plugin_name] = metrics
            except Exception as e:
                logger.error(f"Error getting metrics from plugin {plugin_class.name}: {e}")
        
        return all_metrics


# Create a singleton plugin manager
plugin_manager = PluginManager()


# Example usage of the plugin system
def integrate_plugin_metrics(analytics_result: Dict[str, Any], product: Any, transactions: List[Any], **kwargs) -> Dict[str, Any]:
    """
    Integrate plugin metrics into analytics results.
    
    Args:
        analytics_result: Existing analytics results
        product: The product being analyzed
        transactions: List of transactions for the product
        kwargs: Additional parameters
        
    Returns:
        Updated analytics results with plugin metrics
    """
    # Get metrics from plugins
    plugin_metrics = plugin_manager.get_plugin_metrics(product, transactions, **kwargs)
    
    # Add plugins section to analytics if we have plugin metrics
    if plugin_metrics:
        analytics_result["plugin_metrics"] = plugin_metrics
    
    return analytics_result


# Function to initialize plugins
def initialize_plugins(additional_dirs: Optional[List[str]] = None) -> None:
    """
    Initialize the plugin system and discover plugins.
    
    Args:
        additional_dirs: Optional list of additional plugin directories
    """
    if additional_dirs:
        for directory in additional_dirs:
            plugin_manager.add_plugin_directory(directory)
    
    # Discover plugins in registered directories
    plugin_manager.discover_plugins()