# inventory_tracker/main.py (updated)

import asyncio
import argparse
import logging
from inventory_tracker.scheduler import NotificationScheduler
from inventory_tracker.config import Config
from inventory_tracker.events import EventEmitter
from inventory_tracker.plugins import load_plugins

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="InventoryTracker")
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--no-notify', action='store_true', help='Disable all notifications')
    parser.add_argument('--quiet-hours', help='Set quiet hours (format: "22-6" for 10 PM to 6 AM)')
    parser.add_argument('--check-interval', type=int, help='Check interval in minutes')
    parser.add_argument('--notification-types', nargs='+', help='Notification types to enable')
    parser.add_argument('--enable-plugin', action='append', help='Enable specific plugins')
    parser.add_argument('--disable-plugin', action='append', help='Disable specific plugins')
    
    args = parser.parse_args()
    
    # Set up configuration
    config = Config(args.config)
    config.update_from_args(args)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize the event emitter (singleton)
    event_emitter = EventEmitter.get_instance()
    
    # Load and register plugins
    plugins = load_plugins(config, args.enable_plugin, args.disable_plugin)
    
    try:
        # Create and start scheduler
        scheduler = NotificationScheduler(config)
        await scheduler.start()
    finally:
        # Ensure plugins are unregistered properly
        for plugin in plugins:
            if hasattr(plugin, 'unregister'):
                plugin.unregister()