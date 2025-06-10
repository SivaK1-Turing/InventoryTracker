# inventory_tracker/events.py
import asyncio
import logging
from typing import Dict, Any, List, Callable, Awaitable, Optional, Tuple
from enum import Enum, auto

logger = logging.getLogger("inventory_tracker.events")

class EventType(Enum):
    """Types of events that can be emitted by the system."""
    NOTIFICATION_SENT = auto()
    NOTIFICATION_SUPPRESSED = auto()
    NOTIFICATION_FAILED = auto()
    LOW_STOCK_DETECTED = auto()
    INVENTORY_UPDATED = auto()
    SCHEDULER_STARTED = auto()
    SCHEDULER_STOPPED = auto()

class Event:
    """Event object containing event data."""
    
    def __init__(self, event_type: EventType, data: Dict[str, Any], source: str = "system"):
        """
        Initialize a new event.
        
        Args:
            event_type: Type of event
            data: Event payload data
            source: Component that generated the event
        """
        self.event_type = event_type
        self.data = data
        self.source = source
    
    def __str__(self) -> str:
        return f"Event({self.event_type.name}) from {self.source}"

# Type definition for event handlers
EventHandler = Callable[[Event], Awaitable[None]]

class EventEmitter:
    """
    Event system that allows components to publish events and register handlers.
    Ensures that handler failures don't affect the publisher or other handlers.
    """
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls) -> 'EventEmitter':
        """Get the singleton EventEmitter instance."""
        if cls._instance is None:
            cls._instance = EventEmitter()
        return cls._instance
    
    def __init__(self):
        """Initialize the event emitter with empty handler registry."""
        self.handlers: Dict[EventType, List[Tuple[EventHandler, bool]]] = {
            event_type: [] for event_type in EventType
        }
        self.logger = logging.getLogger("inventory_tracker.events")
    
    def on(self, event_type: EventType, handler: EventHandler, critical: bool = False) -> None:
        """
        Register a handler for a specific event type.
        
        Args:
            event_type: Type of event to listen for
            handler: Async function to call when event occurs
            critical: If True, event emission will raise exceptions from this handler
                      If False, exceptions will be logged but not propagated
        """
        if not asyncio.iscoroutinefunction(handler):
            raise ValueError(f"Event handler must be an async function: {handler}")
        
        self.handlers[event_type].append((handler, critical))
        self.logger.debug(f"Registered {'critical' if critical else 'non-critical'} "
                          f"handler for {event_type.name}")
    
    def off(self, event_type: EventType, handler: EventHandler) -> bool:
        """
        Remove a handler for a specific event type.
        
        Args:
            event_type: Type of event
            handler: Handler function to remove
            
        Returns:
            True if handler was removed, False if not found
        """
        for idx, (h, critical) in enumerate(self.handlers[event_type]):
            if h == handler:
                self.handlers[event_type].pop(idx)
                self.logger.debug(f"Removed handler for {event_type.name}")
                return True
        
        return False
    
    async def emit(self, event: Event) -> None:
        """
        Emit an event to all registered handlers.
        Non-critical handler failures are logged but don't stop other handlers.
        
        Args:
            event: Event object to emit
        """
        self.logger.debug(f"Emitting {event}")
        
        handler_tasks = []
        critical_handlers = []
        
        # Process handlers for this event type
        for handler, is_critical in self.handlers[event.event_type]:
            if is_critical:
                critical_handlers.append(handler)
            else:
                handler_tasks.append(self._safe_execute_handler(handler, event))
        
        # Execute non-critical handlers concurrently
        if handler_tasks:
            await asyncio.gather(*handler_tasks)
        
        # Execute critical handlers sequentially to ensure errors are properly handled
        for handler in critical_handlers:
            await handler(event)
    
    async def _safe_execute_handler(self, handler: EventHandler, event: Event) -> None:
        """
        Execute a handler safely, catching and logging exceptions.
        
        Args:
            handler: Event handler to execute
            event: Event data to pass to handler
        """
        try:
            await handler(event)
        except Exception as e:
            self.logger.error(f"Error in event handler for {event.event_type.name}: {e}")
            self.logger.exception(e)