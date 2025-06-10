# inventory_tracker/plugins/slack_notifier.py
import logging
import aiohttp
from typing import Dict, Any, List, Optional
from inventory_tracker.events import EventEmitter, EventType, Event
from inventory_tracker.config import Config

logger = logging.getLogger("inventory_tracker.plugins.slack")

class SlackNotifier:
    """
    Slack notification plugin that listens to notification events
    and forwards them to a Slack webhook.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the Slack notifier plugin.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.webhook_url = config.get('plugins.slack.webhook_url')
        self.channel = config.get('plugins.slack.channel', '#inventory-alerts')
        self.username = config.get('plugins.slack.username', 'InventoryTracker')
        self.icon_emoji = config.get('plugins.slack.icon_emoji', ':package:')
        self.enabled = config.get('plugins.slack.enabled', False)
        
        if self.enabled and not self.webhook_url:
            logger.error("Slack plugin is enabled but webhook_url is not configured")
            self.enabled = False
        
        self.event_emitter = EventEmitter.get_instance()
    
    def register(self):
        """Register event handlers if plugin is enabled."""
        if not self.enabled:
            logger.info("Slack plugin is disabled, not registering handlers")
            return
            
        logger.info(f"Registering Slack plugin handlers (channel: {self.channel})")
        
        # Register for notification events
        self.event_emitter.on(EventType.NOTIFICATION_SENT, self.handle_notification_sent)
        self.event_emitter.on(EventType.NOTIFICATION_SUPPRESSED, self.handle_notification_suppressed)
        self.event_emitter.on(EventType.NOTIFICATION_FAILED, self.handle_notification_failed)
    
    def unregister(self):
        """Unregister event handlers."""
        if not self.enabled:
            return
            
        logger.info("Unregistering Slack plugin handlers")
        
        # Unregister handlers
        self.event_emitter.off(EventType.NOTIFICATION_SENT, self.handle_notification_sent)
        self.event_emitter.off(EventType.NOTIFICATION_SUPPRESSED, self.handle_notification_suppressed)
        self.event_emitter.off(EventType.NOTIFICATION_FAILED, self.handle_notification_failed)
    
    async def handle_notification_sent(self, event: Event):
        """Handle notification sent events."""
        products = event.data.get('products', [])
        notification_types = event.data.get('types', [])
        results = event.data.get('results', {})
        
        # Build a simple message for Slack
        text = f"*Inventory Alert*: {len(products)} products with low stock\n"
        
        # Add product details
        text += "\n_Low Stock Items:_\n"
        for product in products[:5]:  # Limit to first 5 products
            text += f"• *{product['name']}* (SKU: {product['sku']}): {product['current_stock']} in stock\n"
            
        if len(products) > 5:
            text += f"_(and {len(products) - 5} more items...)_\n"
            
        # Add notification results
        text += "\n_Notification Results:_\n"
        for notifier_type, stats in results.items():
            text += f"• {notifier_type}: {stats['success']} sent, {stats['failure']} failed\n"
        
        await self._send_to_slack(text)
    
    async def handle_notification_suppressed(self, event: Event):
        """Handle notification suppressed events."""
        products = event.data.get('products', [])
        reason = event.data.get('reason', 'unknown')
        
        text = f"*Inventory Alert Suppressed* ({reason})\n"
        text += f"{len(products)} products have low stock but notifications were not sent.\n"
        
        if reason == "quiet hours":
            quiet_hours = event.data.get('quiet_hours', 'configured')
            text += f"_Notifications are currently in quiet hours ({quiet_hours})._"
        
        await self._send_to_slack(text)
    
    async def handle_notification_failed(self, event: Event):
        """Handle notification failed events."""
        notifier_type = event.data.get('type', 'unknown')
        failures = event.data.get('failures', 0)
        
        text = f"*⚠️ Notification Delivery Failure*\n"
        text += f"{failures} {notifier_type} notifications failed to send."
        
        await self._send_to_slack(text)
    
    async def _send_to_slack(self, text: str):
        """
        Send a message to the configured Slack webhook.
        
        Args:
            text: Markdown-formatted message text
        """
        if not self.enabled or not self.webhook_url:
            return
            
        payload = {
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "text": text
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status >= 400:
                        response_text = await response.text()
                        logger.error(f"Error sending to Slack: {response.status} - {response_text}")
                    elif response.status == 200:
                        logger.debug("Successfully sent message to Slack")
        except Exception as e:
            logger.error(f"Failed to send message to Slack: {e}")