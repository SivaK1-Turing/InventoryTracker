# inventorytracker/events.py
from typing import Dict, List, Type, Callable, TypeVar, Generic

T = TypeVar('T')

class Event:
    """Base class for all events in the system"""
    pass

class EventBus:
    """Simple event bus implementation for pub/sub pattern"""
    
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[Callable]] = {}
        
    def subscribe(self, event_type: Type[T], handler: Callable[[T], None]) -> None:
        """Subscribe to events of a specific type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            
    def unsubscribe(self, event_type: Type[Event], handler: Callable) -> None:
        """Unsubscribe from events of a specific type"""
        if event_type in self._subscribers and handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers"""
        event_type = type(event)
        
        # Exact type match
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                handler(event)
                
        # Also notify subscribers of parent event types
        for subscribed_type in self._subscribers:
            if issubclass(event_type, subscribed_type) and subscribed_type != event_type:
                for handler in self._subscribers[subscribed_type]:
                    handler(event)

# Create a global instance of the event bus
event_bus = EventBus()