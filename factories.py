# inventorytracker/factories.py
import importlib.metadata
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic, Type, Union
from uuid import UUID, uuid4
from pydantic import BaseModel

from .models.product import Product
from .utils.metrics import record_metric

# Configure logger
logger = logging.getLogger(__name__)

# Define types for our hooks
T = TypeVar('T', bound=BaseModel)
PreSaveHook = Callable[[Dict[str, Any]], Dict[str, Any]]  # Takes data dict, returns modified data
PostSaveHook = Callable[[T], None]  # Takes created object, returns nothing

# Entry point group for plugin discovery
ENTRY_POINT_GROUP = "codesnip_product_hooks"

class HookRegistry:
    _pre_save_hooks: Dict[str, List[PreSaveHook]] = {}
    _post_save_hooks: Dict[str, List[PostSaveHook]] = {}
    _hooks_loaded = False
    
    @classmethod
    def register_pre_save_hook(cls, model_name: str, hook: PreSaveHook) -> None:
        """Register a hook to be called before creating a model instance."""
        if model_name not in cls._pre_save_hooks:
            cls._pre_save_hooks[model_name] = []
        cls._pre_save_hooks[model_name].append(hook)
        logger.debug(f"Registered pre-save hook {hook.__name__} for {model_name}")
    
    @classmethod
    def register_post_save_hook(cls, model_name: str, hook: PostSaveHook) -> None:
        """Register a hook to be called after creating a model instance."""
        if model_name not in cls._post_save_hooks:
            cls._post_save_hooks[model_name] = []
        cls._post_save_hooks[model_name].append(hook)
        logger.debug(f"Registered post-save hook {hook.__name__} for {model_name}")
    
    @classmethod
    def get_pre_save_hooks(cls, model_name: str) -> List[PreSaveHook]:
        """Get all pre-save hooks for a model."""
        # Ensure entry point hooks are loaded
        cls.load_entry_point_hooks()
        return cls._pre_save_hooks.get(model_name, [])
    
    @classmethod
    def get_post_save_hooks(cls, model_name: str) -> List[PostSaveHook]:
        """Get all post-save hooks for a model."""
        # Ensure entry point hooks are loaded
        cls.load_entry_point_hooks()
        return cls._post_save_hooks.get(model_name, [])
    
    @classmethod
    def clear_hooks(cls, model_name: Optional[str] = None) -> None:
        """
        Clear hooks for a specific model or all models.
        Mainly used for testing purposes.
        """
        if model_name:
            cls._pre_save_hooks.pop(model_name, None)
            cls._post_save_hooks.pop(model_name, None)
        else:
            cls._pre_save_hooks.clear()
            cls._post_save_hooks.clear()
        # Reset loaded state so hooks will be reloaded on next use
        cls._hooks_loaded = False
    
    @classmethod
    def load_entry_point_hooks(cls) -> None:
        """
        Discover and load hooks from entry points.
        
        This only runs once per application lifecycle unless hooks are cleared.
        """
        if cls._hooks_loaded:
            return
        
        logger.info("Loading entry point hooks...")
        # Use entrypoints interface to discover plugins
        try:
            # Using importlib.metadata for Python 3.8+
            entry_points = []
            try:
                # Python 3.10+ style
                entry_points = list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))
            except TypeError:
                # Python 3.8, 3.9 style
                entry_points = [
                    ep for ep in importlib.metadata.entry_points()
                    if ep.group == ENTRY_POINT_GROUP
                ]
            
            if not entry_points:
                logger.debug(f"No entry points found for {ENTRY_POINT_GROUP}")
            
            # Load each entry point
            for entry_point in entry_points:
                try:
                    hook_provider = entry_point.load()
                    
                    # Check if this is a hook provider function that returns hooks
                    if callable(hook_provider):
                        hooks = hook_provider()
                        if not isinstance(hooks, list):
                            hooks = [hooks]
                            
                        for hook in hooks:
                            # The hook provider should return a tuple of (hook_type, model_name, hook_function)
                            if not isinstance(hook, tuple) or len(hook) != 3:
                                logger.warning(
                                    f"Invalid hook format from {entry_point.name}. "
                                    f"Expected (hook_type, model_name, hook_function), got {hook}"
                                )
                                continue
                                
                            hook_type, model_name, hook_function = hook
                            
                            if hook_type == 'pre_save':
                                cls.register_pre_save_hook(model_name, hook_function)
                            elif hook_type == 'post_save':
                                cls.register_post_save_hook(model_name, hook_function)
                            else:
                                logger.warning(f"Unknown hook type {hook_type} from {entry_point.name}")
                    else:
                        logger.warning(f"Entry point {entry_point.name} is not a callable hook provider")
                        
                except Exception as e:
                    logger.error(f"Error loading hooks from {entry_point.name}: {e}")
            
            # Mark hooks as loaded
            cls._hooks_loaded = True
            logger.info(f"Loaded hooks from {len(entry_points)} entry points")
            
        except Exception as e:
            logger.error(f"Error discovering entry points: {e}")
            # Don't mark hooks as loaded so we'll try again next time
            cls._hooks_loaded = False


# Default hooks for Product
def title_case_product_name(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert product name to title case."""
    if 'name' in data and isinstance(data['name'], str):
        data['name'] = data['name'].title()
    return data

def record_product_metrics(product: Product) -> None:
    """Record metrics for product creation."""
    record_metric('product.created', 1)
    record_metric('product.price', float(product.price))

# Generic model factory
def create_model(model_cls: Type[T], data: Dict[str, Any], skip_hooks: bool = False) -> T:
    """
    Generic factory function for creating model instances.
    
    Args:
        model_cls: The Pydantic model class to instantiate
        data: Dictionary of data for the model
        skip_hooks: Skip running hooks (useful for migrations/testing)
        
    Returns:
        An instance of the model
    """
    model_name = model_cls.__name__
    
    # Apply pre-save hooks if not skipped
    if not skip_hooks:
        for hook in HookRegistry.get_pre_save_hooks(model_name):
            data = hook(data)
    
    # Create the instance
    instance = model_cls(**data)
    
    # Apply post-save hooks if not skipped
    if not skip_hooks:
        for hook in HookRegistry.get_post_save_hooks(model_name):
            hook(instance)
    
    return instance

# Factory function for creating Product instances
def create_product(
    name: str, 
    sku: str, 
    price: Union[float, str], 
    reorder_level: int, 
    id: Optional[UUID] = None,
    skip_hooks: bool = False
) -> Product:
    """
    Create a new Product instance with pre-save transformations and post-save hooks.
    
    Args:
        name: Product name (will be title-cased)
        sku: Stock keeping unit (uppercase alphanumeric)
        price: Product price (must be > 0)
        reorder_level: Level at which reordering is triggered
        id: Optional UUID (generated if not provided)
        skip_hooks: Skip running hooks (useful for migrations/testing)
        
    Returns:
        A new Product instance
    """
    data = {
        'id': id or uuid4(),
        'name': name,
        'sku': sku,
        'price': price,
        'reorder_level': reorder_level
    }
    
    return create_model(Product, data, skip_hooks)

# Register default hooks
HookRegistry.register_pre_save_hook('Product', title_case_product_name)
HookRegistry.register_post_save_hook('Product', record_product_metrics)

# Helper to run all hooks manually (useful for testing or migration)
def run_pre_save_hooks(model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Run all pre-save hooks for a model manually."""
    for hook in HookRegistry.get_pre_save_hooks(model_name):
        data = hook(data)
    return data

def run_post_save_hooks(model_name: str, instance: Any) -> None:
    """Run all post_save hooks for a model manually."""
    for hook in HookRegistry.get_post_save_hooks(model_name):
        hook(instance)