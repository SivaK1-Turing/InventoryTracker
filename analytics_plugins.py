# analytics_plugins.py

"""
Plugin system for custom analytics metrics in InventoryTracker.

This module provides the infrastructure for discovering, registering,
and executing custom analytics metric plugins.
"""

import importlib
import inspect
import logging
import os
import pkgutil
import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Type, Callable, Optional, Set, Tuple

# Configure logging
logger = logging.getLogger("analytics_plugins")

class MetricValueType(Enum):
    """Types of values that can be returned by metric calculators."""
    NUMERIC = 'numeric'  # Float or integer
    PERCENTAGE = 'percentage'  # 0-100 value
    RATIO = 'ratio'  # 0-1 value
    DURATION = 'duration'  # Time duration (days, hours)
    CURRENCY = 'currency'  # Monetary value
    CATEGORICAL = 'categorical'  # Fixed set of values
    STRING = 'string'  # General text output


class MetricCalculator(ABC):
    """Base class for custom metric calculators."""
    
    @property
    @abstractmethod
    def metric_id(self) -> str:
        """Unique identifier for this metric."""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for this metric."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this metric represents."""
        pass
    
    @property
    @abstractmethod
    def value_type(self) -> MetricValueType:
        """Type of value returned by this metric."""
        pass
    
    @abstractmethod
    async def calculate(self, product_id: str, **kwargs) -> Any:
        """
        Calculate the metric value for the given product.
        
        Args:
            product_id: ID of the product to calculate metric for
            **kwargs: Additional parameters that may be required
            
        Returns:
            The calculated metric value
        """
        pass
    
    @property
    def parameters(self) -> Dict[str, Tuple[type, Any]]:
        """
        Parameters that this calculator accepts besides product_id.
        
        Returns a dictionary mapping parameter names to (type, default_value) tuples.
        """
        return {}
    
    @property
    def category(self) -> str:
        """Category for this metric, used for grouping in UIs."""
        return "Custom"
    
    @property
    def priority(self) -> int:
        """
        Display priority (lower values = higher priority).
        
        Used when displaying multiple metrics in a limited space.
        """
        return 100  # Default priority for custom metrics


class PluginRegistry:
    """Registry for analytics metric plugins."""
    
    def __init__(self):
        self._calculators: Dict[str, MetricCalculator] = {}
        self._loaded_modules: Set[str] = set()
    
    def register(self, calculator: MetricCalculator) -> None:
        """
        Register a metric calculator.
        
        Args:
            calculator: The calculator instance to register
        """
        metric_id = calculator.metric_id
        
        if metric_id in self._calculators:
            logger.warning(f"Overriding existing calculator for metric '{metric_id}'")
        
        self._calculators[metric_id] = calculator
        logger.info(f"Registered calculator for metric '{metric_id}': {calculator.display_name}")
    
    def unregister(self, metric_id: str) -> bool:
        """
        Unregister a metric calculator.
        
        Args:
            metric_id: ID of the calculator to unregister
            
        Returns:
            True if successfully unregistered, False if not found
        """
        if metric_id in self._calculators:
            del self._calculators[metric_id]
            logger.info(f"Unregistered calculator for metric '{metric_id}'")
            return True
        
        logger.warning(f"Attempted to unregister non-existent metric '{metric_id}'")
        return False
    
    def get_calculator(self, metric_id: str) -> Optional[MetricCalculator]:
        """
        Get a calculator by its metric ID.
        
        Args:
            metric_id: ID of the calculator to retrieve
            
        Returns:
            Calculator instance or None if not found
        """
        return self._calculators.get(metric_id)
    
    def list_metrics(self) -> List[Dict[str, Any]]:
        """
        List all registered metrics with their metadata.
        
        Returns:
            List of dictionaries with metric metadata
        """
        return [
            {
                "id": calc.metric_id,
                "name": calc.display_name,
                "description": calc.description,
                "value_type": calc.value_type.value,
                "category": calc.category,
                "parameters": {
                    name: {
                        "type": param_type.__name__,
                        "default": default_value
                    }
                    for name, (param_type, default_value) in calc.parameters.items()
                }
            }
            for calc in self._calculators.values()
        ]
    
    def discover_plugins(self, plugin_dirs: Optional[List[str]] = None) -> None:
        """
        Discover and load plugins from specified directories.
        
        Args:
            plugin_dirs: List of directories to search for plugins.
                        If None, will search in default locations.
        """
        if plugin_dirs is None:
            # Default plugin locations
            plugin_dirs = [
                os.path.join(os.path.dirname(__file__), "plugins"),
                os.path.join(os.path.expanduser("~"), ".inventorytracker", "plugins"),
                os.path.join(sys.prefix, "share", "inventorytracker", "plugins"),
            ]
            
            # Add any directories from INVENTORY_TRACKER_PLUGIN_PATH
            if "INVENTORY_TRACKER_PLUGIN_PATH" in os.environ:
                for path in os.environ["INVENTORY_TRACKER_PLUGIN_PATH"].split(os.pathsep):
                    if path and path not in plugin_dirs:
                        plugin_dirs.append(path)
        
        for plugin_dir in plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
                
            if not os.path.isdir(plugin_dir):
                logger.warning(f"Skipping non-directory plugin path: {plugin_dir}")
                continue
                
            # Add to Python path if not already there
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            # Look for Python modules in each plugin directory
            for _, name, is_pkg in pkgutil.iter_modules([plugin_dir]):
                if name.startswith('_'):
                    continue  # Skip private modules
                
                if is_pkg:
                    module_name = name
                else:
                    # For non-packages, we need to check if they're plugins
                    if not name.endswith('_plugin'):
                        continue
                    module_name = name
                
                if module_name in self._loaded_modules:
                    continue  # Skip already loaded modules

                try:
                    module = importlib.import_module(module_name)
                    self._loaded_modules.add(module_name)
                    self._load_calculators_from_module(module)
                    
                except Exception as e:
                    logger.error(f"Error loading plugin module {module_name}: {e}")
    
    def _load_calculators_from_module(self, module) -> None:
        """
        Load calculator classes from a module.
        
        Args:
            module: Module to search for calculator classes
        """
        # Find calculator classes in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if it's a subclass of MetricCalculator but not MetricCalculator itself
            if (issubclass(obj, MetricCalculator) and 
                obj is not MetricCalculator and
                getattr(obj, '__module__', '') == module.__name__):
                
                try:
                    # Instantiate and register
                    calculator = obj()
                    self.register(calculator)
                except Exception as e:
                    logger.error(f"Error instantiating calculator class {name}: {e}")
    
    async def calculate_metric(self, metric_id: str, product_id: str, 
                         **kwargs) -> Optional[Dict[str, Any]]:
        """
        Calculate a metric for a product.
        
        Args:
            metric_id: ID of the metric to calculate
            product_id: ID of the product to calculate for
            **kwargs: Additional parameters for the calculator
            
        Returns:
            Dictionary with metric value and metadata, or None if calculator not found
        """
        calculator = self.get_calculator(metric_id)
        if not calculator:
            logger.warning(f"No calculator found for metric '{metric_id}'")
            return None
        
        try:
            # Filter kwargs to only include parameters accepted by the calculator
            valid_params = calculator.parameters.keys()
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            
            # Calculate the metric
            value = await calculator.calculate(product_id, **filtered_kwargs)
            
            return {
                "id": metric_id,
                "name": calculator.display_name,
                "value": value,
                "value_type": calculator.value_type.value,
                "description": calculator.description,
                "category": calculator.category
            }
            
        except Exception as e:
            logger.error(f"Error calculating metric '{metric_id}' for product '{product_id}': {e}")
            return {
                "id": metric_id,
                "name": calculator.display_name,
                "error": str(e),
                "value": None,
                "value_type": calculator.value_type.value
            }
    
    async def calculate_all_metrics(self, product_id: str, 
                              **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Calculate all registered metrics for a product.
        
        Args:
            product_id: ID of the product to calculate for
            **kwargs: Additional parameters for calculators
            
        Returns:
            Dictionary mapping metric IDs to their results
        """
        results = {}
        
        for metric_id in self._calculators:
            result = await self.calculate_metric(metric_id, product_id, **kwargs)
            if result:
                results[metric_id] = result
        
        return results


# Create singleton instance
registry = PluginRegistry()

# Helper function for plugin authors
def register_metric(calculator_class: Type[MetricCalculator]) -> Type[MetricCalculator]:
    """
    Decorator to register a metric calculator.
    
    Example:
        @register_metric
        class TurnoverRatioCalculator(MetricCalculator):
            ...
    
    Args:
        calculator_class: Calculator class to register
    
    Returns:
        The calculator class (unchanged)
    """
    registry.register(calculator_class())
    return calculator_class