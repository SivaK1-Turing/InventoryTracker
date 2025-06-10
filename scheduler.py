# scheduler.py

import asyncio
import logging
import time
from datetime import datetime, time as time_of_day
from typing import Dict, Any, List, Optional

from inventory_tracker.config import Config
from inventory_tracker.notifications import send_notifications
from inventory_tracker.inventory_analysis import detect_low_stock

logger = logging.getLogger("inventory_tracker.scheduler")

class NotificationScheduler:
    """
    Scheduler for running periodic inventory checks and sending notifications.
    Supports quiet hours and notification suppression features.
    """
    
    def __init__(self, config: Config):
        """Initialize the scheduler with configuration settings."""
        self.config = config
        self.running = False
        self.check_interval_seconds = config.get('scheduler.check_interval_seconds', 3600)  # Default: hourly
        
        # Parse quiet hours configuration (format: "22-6" for 10 PM to 6 AM)
        self.quiet_hours_enabled = False
        quiet_hours_config = config.get('notifications.quiet_hours', None)
        if quiet_hours_config:
            try:
                start_hour, end_hour = map(int, quiet_hours_config.split('-'))
                self.quiet_hours_start = time_of_day(hour=start_hour, minute=0)
                self.quiet_hours_end = time_of_day(hour=end_hour, minute=0)
                self.quiet_hours_enabled = True
                logger.info(f"Quiet hours configured: {start_hour}:00 - {end_hour}:00")
            except Exception as e:
                logger.error(f"Invalid quiet_hours configuration '{quiet_hours_config}': {e}")
        
        # Check if notifications are globally disabled
        self.notifications_enabled = not config.get('notifications.no_notify', False)
        if not self.notifications_enabled:
            logger.info("Notifications are globally disabled (--no-notify flag set)")
    
    def is_within_quiet_hours(self) -> bool:
        """
        Check if the current time is within configured quiet hours.
        
        Returns:
            True if current time is within quiet hours, False otherwise
        """
        if not self.quiet_hours_enabled:
            return False
            
        current_time = datetime.now().time()
        
        # Handle case where quiet hours span midnight
        if self.quiet_hours_start > self.quiet_hours_end:
            return current_time >= self.quiet_hours_start or current_time < self.quiet_hours_end
        else:
            return self.quiet_hours_start <= current_time < self.quiet_hours_end
    
    async def start(self):
        """Start the scheduler loop."""
        if self.running:
            return
            
        self.running = True
        logger.info("Starting notification scheduler")
        
        while self.running:
            try:
                # Run the inventory check
                low_stock_products = await detect_low_stock(
                    days_threshold=self.config.get('inventory.days_threshold', 14),
                    history_days=self.config.get('inventory.history_days', 30),
                    reorder_point_multiplier=self.config.get('inventory.reorder_point_multiplier', 1.2)
                )
                
                if low_stock_products:
                    logger.info(f"Detected {len(low_stock_products)} products with low stock")
                    
                    # Determine if we should send notifications now
                    should_notify = self.notifications_enabled and not self.is_within_quiet_hours()
                    
                    if should_notify:
                        # Get notification types from config
                        notification_types = self.config.get('notifications.types', ['email'])
                        
                        # Send notifications
                        notification_results = await send_notifications(
                            low_stock_products=low_stock_products,
                            notification_types=notification_types,
                            dry_run=False
                        )
                        
                        # Log results
                        for notifier_type, stats in notification_results.items():
                            logger.info(f"Sent {stats['success']} {notifier_type} notifications "
                                      f"({stats['failure']} failures)")
                    else:
                        reason = "notifications disabled" if not self.notifications_enabled else "quiet hours"
                        logger.info(f"Suppressing notifications due to {reason}")
                        
                        # If we're in quiet hours, we might want to queue these for later
                        if self.quiet_hours_enabled and not self.notifications_enabled:
                            # TODO: Implement queuing of notifications for sending after quiet hours
                            pass
                
                # Sleep until next check
                await asyncio.sleep(self.check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)  # Sleep for a minute before retrying on error
    
    async def stop(self):
        """Stop the scheduler loop."""
        if not self.running:
            return
            
        self.running = False
        logger.info("Stopping notification scheduler")


# Configuration module to support the new options

# config.py
import argparse
import os
import yaml
from typing import Dict, Any, Optional, Union

class Config:
    """
    Configuration manager for InventoryTracker, supporting both file-based
    configuration and command-line arguments.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration from file and environment variables.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.config = {}
        self._load_defaults()
        
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
            
        self._load_from_env()
    
    def _load_defaults(self):
        """Load default configuration values."""
        self.config = {
            'scheduler': {
                'check_interval_seconds': 3600,  # 1 hour
            },
            'notifications': {
                'enabled': True,
                'no_notify': False,
                'types': ['email'],
                'quiet_hours': None
            },
            'inventory': {
                'days_threshold': 14,
                'history_days': 30,
                'reorder_point_multiplier': 1.2
            }
        }
    
    def _load_from_file(self, config_file: str):
        """
        Load configuration from a YAML file.
        
        Args:
            config_file: Path to the configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config and isinstance(file_config, dict):
                    # Recursively update the configuration
                    self._deep_update(self.config, file_config)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Handle common settings through environment variables
        if 'INVENTORY_CHECK_INTERVAL' in os.environ:
            try:
                self.config['scheduler']['check_interval_seconds'] = int(os.environ['INVENTORY_CHECK_INTERVAL'])
            except ValueError:
                pass
                
        if 'INVENTORY_NO_NOTIFY' in os.environ:
            self.config['notifications']['no_notify'] = os.environ['INVENTORY_NO_NOTIFY'].lower() in ('true', '1', 'yes')
            
        if 'INVENTORY_QUIET_HOURS' in os.environ:
            self.config['notifications']['quiet_hours'] = os.environ['INVENTORY_QUIET_HOURS']
    
    def update_from_args(self, args: argparse.Namespace):
        """
        Update configuration from command-line arguments.
        
        Args:
            args: Command-line arguments parsed by argparse
        """
        # Update config with command line arguments
        if hasattr(args, 'no_notify') and args.no_notify:
            self.config['notifications']['no_notify'] = True
            
        if hasattr(args, 'quiet_hours') and args.quiet_hours:
            self.config['notifications']['quiet_hours'] = args.quiet_hours
            
        if hasattr(args, 'check_interval') and args.check_interval:
            self.config['scheduler']['check_interval_seconds'] = args.check_interval * 60  # Convert to seconds
            
        if hasattr(args, 'notification_types') and args.notification_types:
            self.config['notifications']['types'] = args.notification_types
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """
        Recursively update a nested dictionary.
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates to apply 
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a dot-separated path.
        
        Args:
            path: Dot-separated path (e.g., 'notifications.quiet_hours')
            default: Default value if path doesn't exist
            
        Returns:
            Configuration value or default if not found
        """
        # Split the path and navigate through the config
        parts = path.split('.')
        current = self.config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current


# Example usage in main application

# main.py
import asyncio
import argparse
import logging
from inventory_tracker.scheduler import NotificationScheduler
from inventory_tracker.config import Config

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="InventoryTracker")
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--no-notify', action='store_true', help='Disable all notifications')
    parser.add_argument('--quiet-hours', help='Set quiet hours (format: "22-6" for 10 PM to 6 AM)')
    parser.add_argument('--check-interval', type=int, help='Check interval in minutes')
    parser.add_argument('--notification-types', nargs='+', help='Notification types to enable')
    
    args = parser.parse_args()
    
    # Set up configuration
    config = Config(args.config)
    config.update_from_args(args)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start scheduler
    scheduler = NotificationScheduler(config)
    await scheduler.start()

if __name__ == "__main__":
    asyncio.run(main())