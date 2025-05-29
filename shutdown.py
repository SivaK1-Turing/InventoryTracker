#!/usr/bin/env python3
"""
Graceful shutdown handling for Inventory Tracker.

This module provides mechanisms to ensure clean shutdown when signals
like SIGINT and SIGTERM are received. It ensures:

1. Pending tasks are properly cancelled
2. Data is flushed to storage
3. Resources are released in a controlled manner
4. The application exits with appropriate status codes

Using contextlib.ExitStack provides clear ordering of shutdown operations
and explicit resource management.
"""
import atexit
import contextlib
import functools
import logging
import os
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from types import FrameType
from typing import Callable, Dict, List, Optional, Set, Any, Union, TypeVar, cast

import typer

from inventorytracker.logging import logger


# Type for shutdown hooks
ShutdownHook = Callable[[], None]

# Singleton exit stack for the application
_exit_stack = contextlib.ExitStack()

# Track whether shutdown has been initiated to prevent duplicates
_shutdown_initiated = False
_shutdown_lock = threading.RLock()

# Track the original signal handlers
_original_handlers: Dict[signal.Signals, Any] = {}

# Track registered metrics flush operations
_metrics_operations: List[Callable[[], None]] = []

# Track registered store flush operations
_store_operations: List[Callable[[], None]] = []

# Exit code to use when shutting down
_exit_code = 0


class ShutdownStage(Enum):
    """Stages of the shutdown process, executed in order."""
    CANCEL_TASKS = 1      # Cancel running tasks
    FLUSH_METRICS = 2     # Flush metrics data
    FLUSH_STORES = 3      # Flush data stores
    CLOSE_RESOURCES = 4   # Close files, connections, etc.
    FINAL_LOGGING = 5     # Final logging messages


# Track operations registered for each stage
_stage_operations: Dict[ShutdownStage, List[Callable[[], None]]] = {
    stage: [] for stage in ShutdownStage
}


def _signal_handler(sig: int, frame: Optional[FrameType]) -> None:
    """
    Handle signals for graceful shutdown.
    
    This is called when SIGINT (Ctrl+C) or SIGTERM is received.
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    global _exit_code
    
    signal_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else f"signal {sig}"
    logger.info(f"Received {signal_name}, initiating graceful shutdown...")
    
    # Set exit code - SIGTERM (15) or SIGINT (2) result in clean exit
    if sig == signal.SIGTERM:
        _exit_code = 0
    elif sig == signal.SIGINT:
        _exit_code = 130  # Standard Unix exit code for SIGINT
    else:
        _exit_code = 1
    
    # Initiate graceful shutdown  
    initiate_shutdown()


def _handle_unraisable_exception(hook_info):
    """
    Handle uncaught exceptions during shutdown.
    
    Args:
        hook_info: Information about the exception
    """
    exc_type, exc_value, exc_tb = hook_info.exc_type, hook_info.exc_value, hook_info.exc_traceback
    logger.error(f"Unhandled exception during shutdown: {exc_value}")
    if exc_tb:
        logger.debug(
            "Traceback: \n" + 
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        )


def install_signal_handlers() -> None:
    """
    Install signal handlers for graceful shutdown.
    
    This configures SIGINT and SIGTERM handlers to initiate graceful shutdown.
    """
    global _original_handlers
    
    # Store original handlers for possible restoration
    _original_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
    _original_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
    
    # Install our handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    # Set up atexit handler as a backup for when signals aren't triggered
    atexit.register(initiate_shutdown)
    
    # Handle uncaught exceptions during shutdown
    sys.unraisablehook = _handle_unraisable_exception
    
    logger.debug("Installed signal handlers for graceful shutdown")


def restore_signal_handlers() -> None:
    """Restore original signal handlers if they were saved."""
    for sig, handler in _original_handlers.items():
        try:
            signal.signal(sig, handler)
        except Exception:
            pass  # Ignore failures when restoring handlers


def register_metrics_operation(operation: Callable[[], None]) -> None:
    """
    Register a metrics flush operation to run during shutdown.
    
    Args:
        operation: Callable to run during metrics flush stage
    """
    _metrics_operations.append(operation)
    logger.debug(f"Registered metrics operation: {operation.__name__}")
    
    # Also register at the correct stage
    register_shutdown_operation(ShutdownStage.FLUSH_METRICS, operation)


def register_store_operation(operation: Callable[[], None]) -> None:
    """
    Register a data store flush operation to run during shutdown.
    
    Args:
        operation: Callable to run during store flush stage
    """
    _store_operations.append(operation)
    logger.debug(f"Registered store operation: {operation.__name__}")
    
    # Also register at the correct stage
    register_shutdown_operation(ShutdownStage.FLUSH_STORES, operation)


def register_shutdown_operation(stage: ShutdownStage, operation: Callable[[], None]) -> None:
    """
    Register a shutdown operation at a specific stage.
    
    This allows precise control over the order of shutdown operations.
    
    Args:
        stage: The shutdown stage at which to run this operation
        operation: Callable to run during this stage
    """
    _stage_operations[stage].append(operation)
    logger.debug(f"Registered {operation.__name__} for {stage.name}")


def register_context_manager(cm: contextlib.AbstractContextManager) -> None:
    """
    Register a context manager with the application exit stack.
    
    This ensures the context manager's __exit__ method will be called
    during shutdown in the reverse order of registration.
    
    Args:
        cm: Context manager to register
    """
    # This is thread-safe since _exit_stack.enter_context is atomic
    _exit_stack.enter_context(cm)
    logger.debug(f"Registered context manager: {cm.__class__.__name__}")


def _execute_stage_operations(stage: ShutdownStage) -> None:
    """
    Execute all operations registered for a specific shutdown stage.
    
    Args:
        stage: The shutdown stage to execute
    """
    logger.debug(f"Executing shutdown stage: {stage.name}")
    
    operations = _stage_operations[stage]
    if not operations:
        logger.debug(f"No operations registered for {stage.name}")
        return
    
    for operation in operations:
        try:
            logger.debug(f"Running {operation.__name__}")
            operation()
        except Exception as e:
            logger.error(f"Error in {operation.__name__} during {stage.name}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")


def initiate_shutdown() -> None:
    """
    Initiate the graceful shutdown process.
    
    This function is idempotent - calling it multiple times will
    only execute the shutdown sequence once.
    """
    global _shutdown_initiated
    
    # Use a lock to ensure the shutdown sequence runs only once
    with _shutdown_lock:
        if _shutdown_initiated:
            return
        _shutdown_initiated = True
    
    logger.info("Initiating graceful shutdown sequence...")
    
    try:
        # Disable signal handlers during shutdown to prevent re-entry
        restore_signal_handlers()
        
        # Execute each shutdown stage in order
        for stage in ShutdownStage:
            _execute_stage_operations(stage)
        
        # Close all registered context managers
        logger.debug("Closing registered context managers")
        _exit_stack.close()
        
        logger.info("Graceful shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # Exit if this isn't being called during Python's normal exit
    if not sys.is_finalizing():
        sys.exit(_exit_code)


class ShutdownManager:
    """
    Context manager for managing application lifecycle and shutdown.
    
    This provides a convenient way to ensure resources are properly
    cleaned up, even if an exception occurs.
    
    Example:
        with ShutdownManager():
            # Application code
            app.run()
    """
    
    def __init__(self, prevent_sigint_exit: bool = False):
        """
        Initialize the shutdown manager.
        
        Args:
            prevent_sigint_exit: If True, SIGINT (Ctrl+C) will not immediately exit
        """
        self.prevent_sigint_exit = prevent_sigint_exit
        self.original_sigint = None
    
    def __enter__(self):
        """Set up signal handlers and prepare for clean shutdown."""
        # Install signal handlers
        install_signal_handlers()
        
        # If requested, modify behavior for SIGINT
        if self.prevent_sigint_exit:
            # This allows Ctrl+C to trigger our handler but not exit immediately
            def _custom_sigint_handler(sig, frame):
                logger.info("Received SIGINT (Ctrl+C), but continuing execution...")
                # Here you could set a flag to stop processing
            
            self.original_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _custom_sigint_handler)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure clean shutdown when the context is exited."""
        if exc_type:
            logger.error(f"Error during application execution: {exc_val}")
            if exc_tb:
                logger.debug(
                    "Traceback: \n" + 
                    "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
                )
        
        # Restore original SIGINT handler if we modified it
        if self.prevent_sigint_exit and self.original_sigint:
            signal.signal(signal.SIGINT, self.original_sigint)
        
        # Initiate shutdown
        initiate_shutdown()
        
        # Don't suppress exceptions
        return False


class GracefulTask:
    """
    Wrapper for cancellable tasks.
    
    This helps track running tasks and ensures they can be cancelled
    during shutdown.
    
    Example:
        task = GracefulTask(name="background_process")
        with task:
            # This code can be interrupted during shutdown
            process_data()
    """
    
    # Track all active tasks
    _active_tasks: Set["GracefulTask"] = set()
    _tasks_lock = threading.RLock()
    
    @classmethod
    def cancel_all_tasks(cls) -> None:
        """Cancel all currently running tasks."""
        with cls._tasks_lock:
            active_tasks = list(cls._active_tasks)
        
        if active_tasks:
            logger.info(f"Cancelling {len(active_tasks)} active tasks...")
            for task in active_tasks:
                try:
                    task.cancel()
                except Exception as e:
                    logger.error(f"Error cancelling task {task.name}: {e}")
    
    def __init__(self, name: str = None, timeout: float = 5.0):
        """
        Initialize a new cancellable task.
        
        Args:
            name: Name of the task (for logging)
            timeout: Timeout in seconds for task cancellation
        """
        self.name = name or f"task-{id(self)}"
        self.timeout = timeout
        self.is_cancelled = False
        self.cancel_event = threading.Event()
    
    def __enter__(self):
        """Register this task as active."""
        with GracefulTask._tasks_lock:
            GracefulTask._active_tasks.add(self)
        logger.debug(f"Task started: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Unregister this task when it completes."""
        with GracefulTask._tasks_lock:
            if self in GracefulTask._active_tasks:
                GracefulTask._active_tasks.remove(self)
        
        if self.is_cancelled:
            logger.debug(f"Cancelled task completed: {self.name}")
        elif exc_type:
            logger.warning(f"Task {self.name} exited with error: {exc_val}")
        else:
            logger.debug(f"Task completed: {self.name}")
        
        # Don't suppress exceptions
        return False
    
    def cancel(self) -> None:
        """Cancel this task."""
        self.is_cancelled = True
        self.cancel_event.set()
        logger.debug(f"Requested cancellation of task: {self.name}")
    
    def check_cancelled(self) -> bool:
        """
        Check if this task has been cancelled.
        
        Returns:
            True if the task has been cancelled
        """
        return self.is_cancelled or self.cancel_event.is_set()
    
    def wait_for_cancellation(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for cancellation to be requested.
        
        Args:
            timeout: Maximum time to wait in seconds, or None to use the task's default timeout
            
        Returns:
            True if cancellation was requested, False if timeout occurred
        """
        if timeout is None:
            timeout = self.timeout
        
        return self.cancel_event.wait(timeout)


# Register task cancellation as the first shutdown operation
register_shutdown_operation(ShutdownStage.CANCEL_TASKS, GracefulTask.cancel_all_tasks)


# Metrics flush handler example
def flush_metrics() -> None:
    """
    Example function to flush metrics during shutdown.
    
    In a real application, this would send queued metrics to a monitoring system.
    """
    logger.info("Flushing application metrics...")
    # Simulate metrics flush
    time.sleep(0.1)
    logger.info("Metrics flushed successfully")


# Store flush handler example
def flush_store(store_name: str) -> None:
    """
    Example function to flush a data store during shutdown.
    
    Args:
        store_name: Name of the store to flush
    """
    logger.info(f"Flushing data store: {store_name}")
    # Simulate store flush
    time.sleep(0.2)
    logger.info(f"Store {store_name} flushed successfully")


# Final logging example
def log_shutdown_complete() -> None:
    """Log final shutdown message."""
    logger.info("Application shutdown complete")


# Register example handlers for testing
register_metrics_operation(flush_metrics)
register_store_operation(lambda: flush_store("main_store"))
register_shutdown_operation(ShutdownStage.FINAL_LOGGING, log_shutdown_complete)


# Create and provide a default shutdown manager
default_manager = ShutdownManager()


def setup_app_lifecycle(app: typer.Typer) -> None:
    """
    Set up application lifecycle management for a Typer app.
    
    This integrates shutdown handling with the Typer app.
    
    Args:
        app: The Typer app to configure
    """
    # Get the existing callback
    original_callback = app.callback()
    
    # Define a new callback that wraps the original one
    def lifecycle_callback(
        ctx: typer.Context,
        *args, **kwargs
    ):
        # Set up the shutdown manager
        with ShutdownManager():
            # Call the original callback
            if original_callback:
                return original_callback(ctx, *args, **kwargs)
    
    # Replace the app's callback
    app.callback()(lifecycle_callback)
    
    return app


if __name__ == "__main__":
    # Example usage
    logger.setLevel(logging.DEBUG)
    
    print("Press Ctrl+C to test graceful shutdown...")
    
    # Register a dummy context manager for testing
    class DummyContextManager:
        def __enter__(self):
            print("DummyContextManager: Enter")
            return self
            
        def __exit__(self, *exc_info):
            print("DummyContextManager: Exit - cleaning up resources")
            return False
    
    # Register with the exit stack
    register_context_manager(DummyContextManager())
    
    # Simulate a long-running application
    with ShutdownManager():
        try:
            # Create a task that runs for a while
            with GracefulTask(name="example_task"):
                while True:
                    time.sleep(1)
                    print("Application still running...")
        except KeyboardInterrupt:
            # This shouldn't normally be reached due to our signal handler
            print("Interrupted!")