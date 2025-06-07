"""
Transaction hooks implementation for InventoryTracker.
Provides mechanisms to hook into transaction events for cache invalidation.
"""

import logging
from typing import Dict, Any, Callable, Optional, Union

logger = logging.getLogger("transaction_hooks")

class TransactionHookManager:
    """
    Manages hooks for transaction events to trigger cache invalidation.
    """
    
    def __init__(self, transaction_handler):
        """
        Initialize the transaction hook manager.
        
        Args:
            transaction_handler: The handler that will process transaction events
        """
        self.transaction_handler = transaction_handler
        self.logger = logging.getLogger("transaction_hooks.manager")
        
    def register_hooks(self, transaction_manager) -> bool:
        """
        Register hooks with the transaction manager using the best available method.
        
        Args:
            transaction_manager: The transaction manager to hook into
            
        Returns:
            True if hooks were registered successfully, False otherwise
        """
        # Try each method in order of preference
        if self._try_event_subscription(transaction_manager):
            self.logger.info("Registered hooks using event subscription")
            return True
            
        if self._try_method_wrapping(transaction_manager):
            self.logger.info("Registered hooks using method wrapping")
            return True
            
        if self._try_signal_connection(transaction_manager):
            self.logger.info("Registered hooks using signal connection")
            return True
            
        self.logger.warning("Failed to register transaction hooks - no compatible methods found")
        return False
        
    def _try_event_subscription(self, transaction_manager) -> bool:
        """
        Try to register hooks using event subscription pattern.
        
        Args:
            transaction_manager: The transaction manager to hook into
            
        Returns:
            True if successful, False otherwise
        """
        # Check for EventEmitter-like interface
        if hasattr(transaction_manager, 'on'):
            try:
                transaction_manager.on('transaction.created', self.transaction_handler.on_transaction_created)
                transaction_manager.on('transaction.updated', self.transaction_handler.on_transaction_updated)
                transaction_manager.on('transaction.committed', self.transaction_handler.on_transaction_committed)
                transaction_manager.on('transaction.cancelled', self.transaction_handler.on_transaction_cancelled)
                return True
            except Exception as e:
                self.logger.error(f"Error registering event handlers via 'on' method: {e}")
                
        # Check for explicit subscribe method
        if hasattr(transaction_manager, 'subscribe'):
            try:
                transaction_manager.subscribe('transaction.created', self.transaction_handler.on_transaction_created)
                transaction_manager.subscribe('transaction.updated', self.transaction_handler.on_transaction_updated)
                transaction_manager.subscribe('transaction.committed', self.transaction_handler.on_transaction_committed)
                transaction_manager.subscribe('transaction.cancelled', self.transaction_handler.on_transaction_cancelled)
                return True
            except Exception as e:
                self.logger.error(f"Error registering event handlers via 'subscribe' method: {e}")
                
        # Check for addEventListener pattern
        if hasattr(transaction_manager, 'addEventListener'):
            try:
                transaction_manager.addEventListener('transaction.created', self.transaction_handler.on_transaction_created)
                transaction_manager.addEventListener('transaction.updated', self.transaction_handler.on_transaction_updated)
                transaction_manager.addEventListener('transaction.committed', self.transaction_handler.on_transaction_committed)
                transaction_manager.addEventListener('transaction.cancelled', self.transaction_handler.on_transaction_cancelled)
                return True
            except Exception as e:
                self.logger.error(f"Error registering event handlers via 'addEventListener' method: {e}")
                
        return False
        
    def _try_method_wrapping(self, transaction_manager) -> bool:
        """
        Try to register hooks by wrapping transaction manager methods.
        
        Args:
            transaction_manager: The transaction manager to hook into
            
        Returns:
            True if successful, False otherwise
        """
        # Check if transaction manager has expected methods
        methods_wrapped = 0
        
        # Wrap create_transaction method
        if hasattr(transaction_manager, 'create_transaction') and callable(transaction_manager.create_transaction):
            original_create = transaction_manager.create_transaction
            
            def create_transaction_wrapper(*args, **kwargs):
                result = original_create(*args, **kwargs)
                try:
                    # Convert to dict if it's an object
                    data = result if isinstance(result, dict) else result.__dict__
                    self.transaction_handler.on_transaction_created(data)
                except Exception as e:
                    self.logger.error(f"Error in create_transaction hook: {e}")
                return result
                
            transaction_manager.create_transaction = create_transaction_wrapper
            methods_wrapped += 1
            
        # Wrap update_transaction method
        if hasattr(transaction_manager, 'update_transaction') and callable(transaction_manager.update_transaction):
            original_update = transaction_manager.update_transaction
            
            def update_transaction_wrapper(*args, **kwargs):
                result = original_update(*args, **kwargs)
                try:
                    data = result if isinstance(result, dict) else result.__dict__
                    self.transaction_handler.on_transaction_updated(data)
                except Exception as e:
                    self.logger.error(f"Error in update_transaction hook: {e}")
                return result
                
            transaction_manager.update_transaction = update_transaction_wrapper
            methods_wrapped += 1
            
        # Wrap commit_transaction method
        if hasattr(transaction_manager, 'commit_transaction') and callable(transaction_manager.commit_transaction):
            original_commit = transaction_manager.commit_transaction
            
            def commit_transaction_wrapper(*args, **kwargs):
                result = original_commit(*args, **kwargs)
                try:
                    data = result if isinstance(result, dict) else result.__dict__
                    self.transaction_handler.on_transaction_committed(data)
                except Exception as e:
                    self.logger.error(f"Error in commit_transaction hook: {e}")
                return result
                
            transaction_manager.commit_transaction = commit_transaction_wrapper
            methods_wrapped += 1
            
        # Wrap cancel_transaction method
        if hasattr(transaction_manager, 'cancel_transaction') and callable(transaction_manager.cancel_transaction):
            original_cancel = transaction_manager.cancel_transaction
            
            def cancel_transaction_wrapper(*args, **kwargs):
                result = original_cancel(*args, **kwargs)
                try:
                    data = result if isinstance(result, dict) else result.__dict__
                    self.transaction_handler.on_transaction_cancelled(data)
                except Exception as e:
                    self.logger.error(f"Error in cancel_transaction hook: {e}")
                return result
                
            transaction_manager.cancel_transaction = cancel_transaction_wrapper
            methods_wrapped += 1
            
        return methods_wrapped > 0
        
    def _try_signal_connection(self, transaction_manager) -> bool:
        """
        Try to register hooks using signal pattern (Django/Qt style).
        
        Args:
            transaction_manager: The transaction manager to hook into
            
        Returns:
            True if successful, False otherwise
        """
        signals_connected = 0
        
        # Check for Django-like signals
        if (hasattr(transaction_manager, 'transaction_created') and 
            hasattr(transaction_manager.transaction_created, 'connect')):
            try:
                transaction_manager.transaction_created.connect(self.transaction_handler.on_transaction_created)
                signals_connected += 1
            except Exception as e:
                self.logger.error(f"Error connecting to transaction_created signal: {e}")
                
        if (hasattr(transaction_manager, 'transaction_updated') and 
            hasattr(transaction_manager.transaction_updated, 'connect')):
            try:
                transaction_manager.transaction_updated.connect(self.transaction_handler.on_transaction_updated)
                signals_connected += 1
            except Exception as e:
                self.logger.error(f"Error connecting to transaction_updated signal: {e}")
                
        if (hasattr(transaction_manager, 'transaction_committed') and 
            hasattr(transaction_manager.transaction_committed, 'connect')):
            try:
                transaction_manager.transaction_committed.connect(self.transaction_handler.on_transaction_committed)
                signals_connected += 1
            except Exception as e:
                self.logger.error(f"Error connecting to transaction_committed signal: {e}")
                
        if (hasattr(transaction_manager, 'transaction_cancelled') and 
            hasattr(transaction_manager.transaction_cancelled, 'connect')):
            try:
                transaction_manager.transaction_cancelled.connect(self.transaction_handler.on_transaction_cancelled)
                signals_connected += 1
            except Exception as e:
                self.logger.error(f"Error connecting to transaction_cancelled signal: {e}")
                
        return signals_connected > 0


def register_cache_invalidation_hooks(transaction_manager, transaction_handler) -> bool:
    """
    Register cache invalidation hooks with the transaction manager.
    
    Args:
        transaction_manager: The transaction manager to hook into
        transaction_handler: The handler that will process transaction events
        
    Returns:
        True if hooks were registered successfully, False otherwise
    """
    hook_manager = TransactionHookManager(transaction_handler)
    return hook_manager.register_hooks(transaction_manager)