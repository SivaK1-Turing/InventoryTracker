# notifications/example_usage.py
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from .base import Notifier, NotificationPriority, send_notification
from .email_notifier import EmailNotifier
from .webhook_notifier import WebhookNotifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InventoryNotifier:
    """
    Example class that manages inventory notifications using both
    synchronous and asynchronous notifiers.
    """
    
    def __init__(self, email_config: Dict[str, Any], webhook_config: Dict[str, Any]):
        """Initialize with configurations for both notifier types."""
        self.email_notifier = EmailNotifier(email_config)
        self.webhook_notifier = WebhookNotifier(webhook_config)
    
    def notify_low_stock(self, product_id: str, product_name: str, 
                        current_level: int, threshold: int):
        """
        Send low stock notification using appropriate channels.
        
        Args:
            product_id: Product identifier
            product_name: Human-readable product name
            current_level: Current stock level
            threshold: Low stock threshold
        """
        subject = f"Low Stock Alert: {product_name}"
        body = (f"Product {product_name} (ID: {product_id}) is below threshold.\n"
                f"Current level: {current_level}, Threshold: {threshold}")
                
        # Determine priority based on how low the stock is
        ratio = current_level / threshold
        if ratio <= 0.25:
            priority = NotificationPriority.URGENT
        elif ratio <= 0.5:
            priority = NotificationPriority.HIGH
        else:
            priority = NotificationPriority.NORMAL
            
        # Send email to administrators (sync operation)
        email_result = self.email_notifier.send(
            subject=subject,
            body=body,
            recipients=["inventory@example.com", "manager@example.com"],
            priority=priority
        )
        
        # Additional data for webhook
        metadata = {
            'product_id': product_id,
            'current_level': current_level,
            'threshold': threshold,
            'event_type': 'low_stock_alert'
        }
        
        # Schedule webhook notification (async operation)
        # We don't wait for it to complete
        webhook_future = self.webhook_notifier.send(
            subject=subject,
            body=body,
            metadata=metadata,
            priority=priority,
            timestamp=datetime.now().isoformat()
        )
        
        # Log the email result immediately since it's synchronous
        if email_result:
            logger.info(f"Email notification sent for product {product_id}")
        else:
            logger.error(f"Failed to send email notification for product {product_id}")
            
        # Return the webhook future in case the caller wants to wait for it
        return webhook_future

async def example_main():
    """Example of using the InventoryNotifier."""
    
    # Configure email notifier
    email_config = {
        'smtp_server': 'smtp.example.com',
        'smtp_port': 587,
        'sender_email': 'alerts@inventorytracker.com',
        'username': 'alerts@inventorytracker.com',
        'password': 'password123',
        'use_tls': True,
        'retry_count': 3
    }
    
    # Configure webhook notifier
    webhook_config = {
        'urls': [
            'https://integration.example.com/webhook',
            'https://backup.example.com/alerts'
        ],
        'method': 'POST',
        'content_type': 'application/json',
        'headers': {
            'X-API-Key': 'your-api-key-here'
        },
        'retry_count': 3
    }
    
    # Create the notifier
    inventory_notifier = InventoryNotifier(email_config, webhook_config)
    
    # Send a notification and wait for the webhook to complete
    webhook_future = inventory_notifier.notify_low_stock(
        product_id="PROD-1234",
        product_name="Wireless Headphones",
        current_level=5,
        threshold=10
    )
    
    # Wait for webhook to complete if needed
    webhook_result = await webhook_future
    
    if webhook_result:
        logger.info("Webhook notification completed successfully")
    else:
        logger.error("Webhook notification failed")
    
    # Alternatively, use the unified helper function
    result = await send_notification(
        inventory_notifier.webhook_notifier,
        subject="Test Alert",
        body="This is a test notification",
        metadata={"test": True},
        wait=True  # Ensures we wait for the async result
    )
    
    logger.info(f"Unified notification result: {result}")

if __name__ == "__main__":
    asyncio.run(example_main())