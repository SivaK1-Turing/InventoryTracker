"""
commands/notify.py - CLI command to send notifications for low-stock items

This implements Feature 5 - Low-Stock Detector with Notifications
"""

import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set

# Import analytics functions for low-stock detection
from inventory_tracker.analytics import forecast_depletion
from inventory_tracker.database import get_product_list, get_product_details
from inventory_tracker.notifications.registry import get_registered_notifiers
from inventory_tracker.notifications.base import NotificationPriority, DeliveryStatus, Notification
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.status import Status

# Set up logging
logger = logging.getLogger(__name__)

# Configure rich console
console = Console()

async def detect_low_stock(
    days_threshold: int = 14, 
    history_days: int = 30,
    reorder_point_multiplier: float = 1.2
) -> List[Dict[str, Any]]:
    """
    Detect products with low stock, defined as either:
    1. Estimated to run out within the specified number of days based on usage patterns
    2. Current stock is below the product's reorder point (with optional multiplier)
    
    Args:
        days_threshold: Number of days threshold for considering a product as low stock
        history_days: Number of days of history to analyze for usage patterns
        reorder_point_multiplier: Multiplier to apply to the product's reorder point
        
    Returns:
        List of dictionaries containing information about low-stock products
    """
    # Get all products
    products = await get_product_list()
    
    # Track low-stock products
    low_stock_products = []
    
    # Check each product
    for product in products:
        product_id = str(product['id'])
        
        # Get detailed product info
        product_details = await get_product_details(product_id)
        
        # Get the reorder point (default to 0 if not set)
        reorder_point = product_details.get('reorder_point', 0)
        adjusted_reorder_point = reorder_point * reorder_point_multiplier
        
        # Get stock forecast data
        forecast = await forecast_depletion(product_id, history_days=history_days)
        
        is_low_stock = False
        low_stock_reason = None
        
        # Check if product is already at or near depletion
        if forecast['current_stock'] <= 0:
            is_low_stock = True
            low_stock_reason = "Out of stock"
        # Check if below reorder point
        elif forecast['current_stock'] <= adjusted_reorder_point:
            is_low_stock = True
            low_stock_reason = f"Below reorder point ({reorder_point})"
        # Check if it will be depleted soon based on forecast
        elif (forecast['days_until_depletion'] is not None and 
              forecast['days_until_depletion'] <= days_threshold):
            is_low_stock = True
            low_stock_reason = f"Estimated depletion in {forecast['days_until_depletion']:.1f} days"
            
        # Add to low stock list if criteria met
        if is_low_stock:
            # Get estimated depletion date if available
            depletion_date = None
            if forecast['days_until_depletion'] is not None:
                depletion_date = datetime.now().date() + \
                                 timedelta(days=forecast['days_until_depletion'])
            
            low_stock_products.append({
                'id': product_id,
                'name': product_details.get('name', f"Product {product_id}"),
                'sku': product_details.get('sku', 'N/A'),
                'current_stock': forecast['current_stock'],
                'reorder_point': reorder_point,
                'days_until_depletion': forecast['days_until_depletion'],
                'depletion_date': depletion_date,
                'daily_usage_rate': forecast['daily_usage_rate'],
                'reason': low_stock_reason,
                'confidence': forecast['confidence']
            })
    
    # Sort by days until depletion (with None values at the end)
    low_stock_products.sort(
        key=lambda x: (x['days_until_depletion'] is None, 
                      x['days_until_depletion'] if x['days_until_depletion'] is not None else float('inf'))
    )
    
    return low_stock_products

async def send_notifications(
    low_stock_products: List[Dict[str, Any]],
    notification_types: List[str],
    dry_run: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Send notifications for low stock products using all registered notifiers.
    
    Args:
        low_stock_products: List of low-stock products from detect_low_stock()
        notification_types: List of notification types to use (email, sms, webhook, slack, etc.)
        dry_run: If True, don't actually send notifications
        
    Returns:
        Dictionary with statistics by notification type: 
        {
            'email': {'success': 2, 'failure': 1, 'total': 3},
            'webhook': {'success': 3, 'failure': 0, 'total': 3}
        }
    """
    if not low_stock_products:
        return {}
        
    # Get registered notifiers
    registered_notifiers = get_registered_notifiers()
    
    # Filter by requested types
    if notification_types and notification_types != ['all']:
        notifiers = {k: v for k, v in registered_notifiers.items() if k in notification_types}
    else:
        notifiers = registered_notifiers
    
    if not notifiers:
        raise ValueError(f"No notifiers found for types: {notification_types}")
        
    # Prepare notification content
    subject = f"Inventory Alert: {len(low_stock_products)} products with low stock"
    
    # Build message content
    text_content = [
        f"Low Stock Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"{'-' * 50}",
        f"The following {len(low_stock_products)} products are low on stock and may require attention:",
        "\n"
    ]
    
    for product in low_stock_products:
        depletion_info = (f"Depleting in {product['days_until_depletion']:.1f} days" 
                          if product['days_until_depletion'] is not None 
                          else "Unknown depletion timeline")
        
        text_content.extend([
            f"* {product['name']} (SKU: {product['sku']})",
            f"  Current Stock: {product['current_stock']} units",
            f"  Status: {product['reason']}",
            f"  {depletion_info}",
            ""
        ])
    
    # Add a footer
    text_content.extend([
        "This is an automated notification from the Inventory Tracker system.",
        "Please take appropriate action to restock these items."
    ])
    
    # Join all lines
    body = "\n".join(text_content)