# examples/notification_adapters.py
"""
Example notification adapters for inventory alerts.

This file contains examples of how to implement notification adapters
that can be registered with the alert hook system.
"""

import logging
import requests
import json
from typing import List, Dict, Any
from pathlib import Path

from inventorytracker.alerts import (
    StockAlert, register_alert_hook, HookPriority, HookFailurePolicy
)

logger = logging.getLogger(__name__)

# --- Slack Notification Adapter ---

def send_slack_alert(
    alerts: List[StockAlert], 
    report: Dict[str, Any]
) -> None:
    """
    Send alert notifications to Slack.
    
    Args:
        alerts: List of stock alerts
        report: Generated report information
    """
    from inventorytracker.config import get_config
    
    # Get Slack configuration
    config = get_config()
    slack_config = config.get('notifications', {}).get('slack', {})
    webhook_url = slack_config.get('webhook_url')
    
    if not webhook_url:
        logger.error("Slack webhook URL not configured")
        return
    
    # Count critical alerts
    out_of_stock = sum(1 for alert in alerts if alert.is_out_of_stock)
    critical = report.get("critical_count", 0)
    
    # Create a Slack message
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Inventory Alert Report",
                "emoji": True
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{report.get('total_alerts', 0)}* products need reordering\n"
                        f"*{out_of_stock}* products out of stock\n"
                        f"*{critical}* critical low stock items"
            }
        }
    ]
    
    # Add critical items section if any
    if critical > 0:
        critical_alerts = [a for a in alerts if a.priority <= 2]
        
        # Add a section for critical items
        critical_text = "*Critical Items:*\n"
        for alert in critical_alerts[:5]:  # Show up to 5 items
            emoji = "🚨" if alert.is_out_of_stock else "⚠️"
            critical_text += f"{emoji} *{alert.product_name}* (SKU: {alert.product_sku})\n"
            critical_text += f"   Current: {alert.current_stock}, Reorder: {alert.reorder_level}\n"
            
        if len(critical_alerts) > 5:
            critical_text += f"...and {len(critical_alerts) - 5} more critical items"
            
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": critical_text
                }
            }
        )
    
    # Create the payload
    payload = {
        "blocks": blocks,
        "text": f"Inventory Alert: {critical} critical items, {out_of_stock} out of stock"
    }
    
    # Send to Slack
    try:
        response = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            logger.error(
                f"Failed to send Slack notification. "
                f"Status: {response.status_code}, Response: {response.text}"
            )
        else:
            logger.info("Successfully sent Slack notification")
            
    except Exception as e:
        logger.exception(f"Error sending Slack notification: {e}")
        raise

# --- SMS Notification Adapter ---

def send_sms_alert(
    alerts: List[StockAlert], 
    report: Dict[str, Any]
) -> None:
    """
    Send critical alerts via SMS.
    
    This example uses Twilio, but any SMS service could be used.
    
    Args:
        alerts: List of stock alerts
        report: Generated report information
    """
    try:
        # Only import if SMS is actually being used
        from twilio.rest import Client
    except ImportError:
        logger.error("Twilio is required for SMS notifications but not installed")
        return
    
    from inventorytracker.config import get_config
    
    # Get SMS configuration
    config = get_config()
    sms_config = config.get('notifications', {}).get('sms', {})
    
    account_sid = sms_config.get('account_sid')
    auth_token = sms_config.get('auth_token')
    from_number = sms_config.get('from_number')
    to_numbers = sms_config.get('to_numbers', [])
    
    if not (account_sid and auth_token and from_number and to_numbers):
        logger.error("SMS configuration incomplete")
        return
    
    # Filter to critical alerts only
    critical_alerts = [a for a in alerts if a.is_out_of_stock or a.priority <= 2]
    
    # Don't send if no critical alerts
    if not critical_alerts:
        logger.debug("No critical alerts, not sending SMS")
        return
    
    # Create the message
    message_body = f"INVENTORY ALERT: {len(critical_alerts)} critical items\n"
    
    # Add out of stock items first (limit to 3 for SMS)
    out_of_stock = [a for a in critical_alerts if a.is_out_of_stock]
    if out_of_stock:
        message_body += f"OUT OF STOCK: {len(out_of_stock)} items\n"
        for alert in out_of_stock[:3]:
            message_body += f"- {alert.product_name} (SKU: {alert.product_sku})\n"
    
    # Add overflow message if needed
    if len(critical_alerts) > 3:
        message_body += f"...and {len(critical_alerts) - 3} more items need attention.\n"
    
    message_body += "Check your email for details."
    
    # Send SMS via Twilio
    try:
        client = Client(account_sid, auth_token)
        
        for to_number in to_numbers:
            message = client.messages.create(
                body=message_body,
                from_=from_number,
                to=to_number
            )
            logger.info(f"SMS sent to {to_number}, SID: {message.sid}")
            
    except Exception as e:
        logger.exception(f"Error sending SMS notification: {e}")
        raise

# --- Register the hooks ---

def register_notification_adapters():
    """Register all notification adapters with the hook system."""
    # Register Slack notifications with high priority
    register_alert_hook(
        "slack_notifications", 
        send_slack_alert,
        priority=HookPriority.HIGH,
        failure_policy=HookFailurePolicy.CONTINUE
    )
    
    # Register emergency SMS alerts with critical priority
    register_alert_hook(
        "sms_notifications", 
        send_sms_alert,
        priority=HookPriority.CRITICAL,
        failure_policy=HookFailurePolicy.RETRY,
        max_retries=2
    )
    
    # Email notifications are registered separately in notifications/email.py
    
    logger.info("Notification adapters registered")