# in a plugin file (e.g., metrics_plugin.py)
from inventorytracker.factories import HookRegistry
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Define a post-save hook for metrics
def track_product_creation(product):
    logger.info(f"Product created: {product.name} (SKU: {product.sku})")
    # Could also send metrics to a monitoring system
    # metrics.increment('product.created')

# Define a pre-save hook for additional data transformation
def standardize_sku(data):
    if 'sku' in data and isinstance(data['sku'], str):
        # Ensure SKU is always uppercase
        data['sku'] = data['sku'].upper()
    return data

# Register hooks with the system
HookRegistry.register_post_save_hook('Product', track_product_creation)
HookRegistry.register_pre_save_hook('Product', standardize_sku)