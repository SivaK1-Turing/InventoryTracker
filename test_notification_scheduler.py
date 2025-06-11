# tests/test_notification_scheduler.py

import pytest
import asyncio
from datetime import datetime, time, timedelta
from unittest.mock import Mock, patch, AsyncMock
from freezegun import freeze_time

from inventory_tracker.scheduler import NotificationScheduler
from inventory_tracker.config import Config


@pytest.fixture
def mock_config():
    """Create a mock config with test settings."""
    config = Config()
    # Override with test configuration
    config.config = {
        'scheduler': {
            'check_interval_seconds': 300,  # 5 minutes for testing
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
    return config


@pytest.fixture
def mock_low_stock_products():
    """Sample low stock products for testing."""
    return [
        {
            'id': 'product1',
            'name': 'Test Product 1',
            'sku': 'TP-001',
            'current_stock': 5,
            'reorder_point': 10,
            'days_until_depletion': 7,
            'reason': 'Below reorder point (10)'
        },
        {
            'id': 'product2',
            'name': 'Test Product 2',
            'sku': 'TP-002',
            'current_stock': 3,
            'reorder_point': 8,
            'days_until_depletion': 4,
            'reason': 'Estimated depletion in 4.0 days'
        }
    ]


class AsyncIterator:
    """Helper class to make our scheduler run once and stop."""
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.has_run = False
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if not self.has_run:
            self.has_run = True
            return None
        else:
            self.scheduler.running = False  # Stop the scheduler after one iteration
            raise StopAsyncIteration


class TestNotificationScheduler:
    
    @pytest.mark.asyncio
    async def test_scheduler_normal_hours(self, mock_config, mock_low_stock_products):
        """Test scheduler sends notifications during normal hours."""
        # Set up mocks
        with patch('inventory_tracker.scheduler.detect_low_stock', new_callable=AsyncMock) as mock_detect, \
             patch('inventory_tracker.scheduler.send_notifications', new_callable=AsyncMock) as mock_send, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            # Configure mocks
            mock_detect.return_value = mock_low_stock_products
            mock_send.return_value = {'email': {'success': 2, 'failure': 0, 'total': 2}}
            mock_sleep.side_effect = AsyncIterator
            
            # Create scheduler and run during normal hours (12 noon)
            with freeze_time("2023-06-01 12:00:00"):
                scheduler = NotificationScheduler(mock_config)
                await scheduler.start()
                
            # Assert detect_low_stock was called
            mock_detect.assert_called_once()
            
            # Assert send_notifications was called with correct parameters
            mock_send.assert_called_once_with(
                low_stock_products=mock_low_stock_products,
                notification_types=['email'],
                dry_run=False
            )
            
            # Assert sleep was called with correct interval
            mock_sleep.assert_called_once_with(300)  # 5 minutes
    
    @pytest.mark.asyncio
    async def test_scheduler_quiet_hours(self, mock_config, mock_low_stock_products):
        """Test scheduler doesn't send notifications during quiet hours."""
        # Configure quiet hours
        mock_config.config['notifications']['quiet_hours'] = "22-6"
        
        # Set up mocks
        with patch('inventory_tracker.scheduler.detect_low_stock', new_callable=AsyncMock) as mock_detect, \
             patch('inventory_tracker.scheduler.send_notifications', new_callable=AsyncMock) as mock_send, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            # Configure mocks
            mock_detect.return_value = mock_low_stock_products
            mock_sleep.side_effect = AsyncIterator
            
            # Create scheduler and run during quiet hours (11 PM)
            with freeze_time("2023-06-01 23:00:00"):
                scheduler = NotificationScheduler(mock_config)
                await scheduler.start()
                
            # Assert detect_low_stock was called
            mock_detect.assert_called_once()
            
            # Assert send_notifications was NOT called
            mock_send.assert_not_called()
            
            # Assert sleep was called with correct interval
            mock_sleep.assert_called_once_with(300)  # 5 minutes
    
    @pytest.mark.asyncio
    async def test_scheduler_midnight_quiet_hours(self, mock_config, mock_low_stock_products):
        """Test scheduler doesn't send notifications during quiet hours across midnight."""
        # Configure quiet hours
        mock_config.config['notifications']['quiet_hours'] = "22-6"
        
        # Set up mocks
        with patch('inventory_tracker.scheduler.detect_low_stock', new_callable=AsyncMock) as mock_detect, \
             patch('inventory_tracker.scheduler.send_notifications', new_callable=AsyncMock) as mock_send, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            # Configure mocks
            mock_detect.return_value = mock_low_stock_products
            mock_sleep.side_effect = AsyncIterator
            
            # Create scheduler and run near midnight during quiet hours
            with freeze_time("2023-06-02 01:30:00"):
                scheduler = NotificationScheduler(mock_config)
                await scheduler.start()
                
            # Assert detect_low_stock was called
            mock_detect.assert_called_once()
            
            # Assert send_notifications was NOT called
            mock_send.assert_not_called()
            
            # Assert sleep was called
            mock_sleep.assert_called_once_with(300)
    
    @pytest.mark.asyncio
    async def test_scheduler_no_notify(self, mock_config, mock_low_stock_products):
        """Test scheduler doesn't send notifications when no_notify is set."""
        # Configure no_notify
        mock_config.config['notifications']['no_notify'] = True
        
        # Set up mocks
        with patch('inventory_tracker.scheduler.detect_low_stock', new_callable=AsyncMock) as mock_detect, \
             patch('inventory_tracker.scheduler.send_notifications', new_callable=AsyncMock) as mock_send, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            # Configure mocks
            mock_detect.return_value = mock_low_stock_products
            mock_sleep.side_effect = AsyncIterator
            
            # Create scheduler and run during normal hours (with no_notify)
            with freeze_time("2023-06-01 12:00:00"):
                scheduler = NotificationScheduler(mock_config)
                await scheduler.start()
                
            # Assert detect_low_stock was called
            mock_detect.assert_called_once()
            
            # Assert send_notifications was NOT called (due to no_notify)
            mock_send.assert_not_called()
            
            # Assert sleep was called
            mock_sleep.assert_called_once_with(300)
    
    @pytest.mark.asyncio
    async def test_scheduler_no_low_stock(self, mock_config):
        """Test scheduler behavior when no low stock products are found."""
        # Set up mocks
        with patch('inventory_tracker.scheduler.detect_low_stock', new_callable=AsyncMock) as mock_detect, \
             patch('inventory_tracker.scheduler.send_notifications', new_callable=AsyncMock) as mock_send, \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            # Configure mocks - no low stock products
            mock_detect.return_value = []
            mock_sleep.side_effect = AsyncIterator
            
            # Create scheduler and run
            scheduler = NotificationScheduler(mock_config)
            await scheduler.start()
                
            # Assert detect_low_stock was called
            mock_detect.assert_called_once()
            
            # Assert send_notifications was NOT called (since no low stock products)
            mock_send.assert_not_called()
            
            # Assert sleep was called
            mock_sleep.assert_called_once_with(300)
            
    @pytest.mark.parametrize("current_time,quiet_hours,expected", [
        # Test cases for normal hours
        ("2023-06-01 12:00:00", "22-6", False),  # Noon, outside quiet hours
        ("2023-06-01 21:59:59", "22-6", False),  # Just before quiet hours
        ("2023-06-01 06:00:00", "22-6", False),  # Right at end of quiet hours
        
        # Test cases for quiet hours
        ("2023-06-01 22:00:00", "22-6", True),   # Start of quiet hours
        ("2023-06-01 23:30:00", "22-6", True),   # During evening quiet hours
        ("2023-06-01 00:30:00", "22-6", True),   # During morning quiet hours
        ("2023-06-01 05:59:59", "22-6", True),   # Just before end of quiet hours
        
        # Test other quiet hour ranges
        ("2023-06-01 13:30:00", "12-14", True),  # Middle of afternoon quiet hours
        ("2023-06-01 07:30:00", "7-9", True),    # Morning quiet hours
        ("2023-06-01 19:30:00", "18-8", True),   # Evening to morning quiet hours
    ])
    def test_is_within_quiet_hours(self, mock_config, current_time, quiet_hours, expected):
        """Test the quiet hours detection logic with various time ranges."""
        mock_config.config['notifications']['quiet_hours'] = quiet_hours
        
        with freeze_time(current_time):
            scheduler = NotificationScheduler(mock_config)
            result = scheduler.is_within_quiet_hours()
            assert result == expected, f"At {current_time}, quiet_hours={quiet_hours} should return {expected}"