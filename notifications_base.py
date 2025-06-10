from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, Protocol, runtime_checkable
import asyncio
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class NotificationPriority(Enum):
    LOW = "low"
    NORMAL = "normal"  
    HIGH = "high"
    URGENT = "urgent"

@runtime_checkable
class NotifierProtocol(Protocol):
    """Protocol defining the interface for notification implementations."""
    def send(self, subject: str, body: str, **kwargs) -> Any:
        """Send a notification with the given subject and body."""
        ...

class Notifier(ABC):
    """
    Abstract base class for notification services.
    
    This class defines a common interface for both synchronous and asynchronous
    notification implementations, allowing them to be used interchangeably.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notifier with configuration parameters.
        
        Args:
            config: A dictionary containing configuration parameters
        """
        self.config = config
        self._validate_config()
        
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        pass
    
    @abstractmethod
    def send(self, subject: str, body: str, **kwargs) -> Union[bool, asyncio.Future]:
        """
        Send a notification with the given subject and body.
        
        Args:
            subject: The notification subject
            body: The notification body/content
            **kwargs: Additional parameters specific to the notification type
            
        Returns:
            For synchronous implementations: bool indicating success
            For asynchronous implementations: Future that resolves to bool
        """
        pass
    
    @classmethod
    def create(cls, notifier_type: str, config: Dict[str, Any]) -> 'Notifier':
        """
        Factory method to create appropriate notifier instance.
        
        Args:
            notifier_type: The type of notifier to create ('email', 'webhook', etc.)
            config: Configuration dictionary for the notifier
            
        Returns:
            An instance of the appropriate Notifier subclass
            
        Raises:
            ValueError: If notifier_type is not supported
        """
        # Import implementations here to avoid circular imports
        from .email_notifier import EmailNotifier
        from .webhook_notifier import WebhookNotifier
        
        notifier_map = {
            'email': EmailNotifier,
            'webhook': WebhookNotifier,
        }
        
        if notifier_type not in notifier_map:
            raise ValueError(f"Unsupported notifier type: {notifier_type}")
        
        return notifier_map[notifier_type](config)

class SyncNotifier(Notifier):
    """Base class for synchronous notification implementations."""
    
    def send(self, subject: str, body: str, **kwargs) -> bool:
        """
        Send a notification synchronously.
        
        Args:
            subject: The notification subject
            body: The notification body
            **kwargs: Additional parameters specific to the notification type
            
        Returns:
            bool: True if sending was successful, False otherwise
        """
        try:
            return self._send_impl(subject, body, **kwargs)
        except Exception as e:
            logger.exception(f"Error sending notification: {e}")
            return False
    
    @abstractmethod
    def _send_impl(self, subject: str, body: str, **kwargs) -> bool:
        """Implementation-specific sending logic."""
        pass

class AsyncNotifier(Notifier):
    """Base class for asynchronous notification implementations."""
    
    def send(self, subject: str, body: str, **kwargs) -> asyncio.Future:
        """
        Send a notification asynchronously.
        
        Args:
            subject: The notification subject
            body: The notification body
            **kwargs: Additional parameters specific to the notification type
            
        Returns:
            asyncio.Future: Future that resolves to bool indicating success
        """
        loop = asyncio.get_event_loop()
        return asyncio.ensure_future(self._send_impl(subject, body, **kwargs))
    
    @abstractmethod
    async def _send_impl(self, subject: str, body: str, **kwargs) -> bool:
        """Implementation-specific sending logic."""
        pass

def send_notification(
    notifier: NotifierProtocol, 
    subject: str, 
    body: str, 
    wait: bool = True,
    **kwargs
) -> Union[bool, asyncio.Future]:
    """
    Helper function to send a notification using any notifier implementation.
    
    This function handles both synchronous and asynchronous notifiers
    transparently and optionally waits for async operations to complete.
    
    Args:
        notifier: Any object implementing the NotifierProtocol
        subject: The notification subject
        body: The notification body
        wait: For async notifiers, whether to await the result
        **kwargs: Additional parameters to pass to the notifier
        
    Returns:
        Either a boolean result or a Future depending on the notifier
        and the wait parameter
    """
    result = notifier.send(subject, body, **kwargs)
    
    # If it's a coroutine or future and wait is True, await it
    if asyncio.isfuture(result) and wait:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context
            return result
        else:
            # We're in a sync context, create a new event loop if needed
            return loop.run_until_complete(result)
    
    return result