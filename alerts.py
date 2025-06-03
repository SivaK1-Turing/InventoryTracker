# inventorytracker/alerts.py (additional code for hooks)

from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple, Union
import inspect
import logging
from functools import wraps
import traceback

logger = logging.getLogger(__name__)

# Type definition for alert hooks
AlertHookCallback = Callable[[List[StockAlert], Dict[str, Any]], None]

class HookPriority(Enum):
    """Priority levels for notification hooks."""
    CRITICAL = 100  # Highest priority, executed first (e.g., emergency SMS)
    HIGH = 75       # High priority (e.g., Slack for on-call team)
    NORMAL = 50     # Normal priority (e.g., standard email)
    LOW = 25        # Low priority (e.g., dashboard updates)
    ARCHIVE = 0     # Lowest priority, for archival purposes
    
class HookFailurePolicy(Enum):
    """Policy for handling hook failures."""
    CONTINUE = "continue"    # Continue to next hook if one fails
    ABORT = "abort"          # Abort further processing if a hook fails
    RETRY = "retry"          # Retry the hook before continuing

class AlertHook:
    """Represents a registered hook for alert notifications."""
    
    def __init__(
        self,
        callback: AlertHookCallback,
        name: str,
        priority: HookPriority = HookPriority.NORMAL,
        failure_policy: HookFailurePolicy = HookFailurePolicy.CONTINUE,
        max_retries: int = 1,
        enabled: bool = True
    ):
        """
        Initialize a new alert hook.
        
        Args:
            callback: Function to call when alerts are generated
            name: Unique name for this hook
            priority: Priority level that determines execution order
            failure_policy: How to handle failures in this hook
            max_retries: Maximum retry attempts if failure_policy is RETRY
            enabled: Whether this hook is currently active
        """
        self.callback = callback
        self.name = name
        self.priority = priority
        self.failure_policy = failure_policy
        self.max_retries = max_retries
        self.enabled = enabled
        self.error_count = 0
        self.last_error = None

class AlertHookManager:
    """
    Manages notification hooks for stock alerts.
    
    This class provides a centralized registry for alert notification hooks
    and handles their execution with proper error handling.
    """
    
    def __init__(self):
        """Initialize the hook manager."""
        self.hooks: Dict[str, AlertHook] = {}
        
    def register(
        self,
        name_or_func: Union[str, AlertHookCallback],
        callback: Optional[AlertHookCallback] = None,
        priority: HookPriority = HookPriority.NORMAL,
        failure_policy: HookFailurePolicy = HookFailurePolicy.CONTINUE,
        max_retries: int = 1,
        enabled: bool = True
    ) -> Optional[AlertHookCallback]:
        """
        Register a new alert hook.
        
        Can be used as a decorator or as a direct function call:
        
        @hook_manager.register("email_notifier", priority=HookPriority.HIGH)
        def send_email_alerts(alerts, report):
            ...
        
        OR
        
        hook_manager.register("email_notifier", send_email_alerts, priority=HookPriority.HIGH)
        
        Args:
            name_or_func: Hook name (str) or the callback function when used as decorator
            callback: Function to call when alerts are generated (not used with decorator form)
            priority: Priority level that determines execution order
            failure_policy: How to handle failures in this hook
            max_retries: Maximum retry attempts if failure_policy is RETRY
            enabled: Whether this hook is currently active
            
        Returns:
            The callback function (only when used as a decorator)
            
        Raises:
            ValueError: If a hook with the same name is already registered
        """
        # Handle decorator case
        if callable(name_or_func) and callback is None:
            # When used as @register decorator without parameters
            func = name_or_func
            name = func.__name__
            return self.register(name, func, priority, failure_policy, max_retries, enabled)
        
        # Handle regular case
        name = name_or_func if isinstance(name_or_func, str) else name_or_func.__name__
        
        if name in self.hooks:
            raise ValueError(f"Hook '{name}' is already registered")
        
        hook = AlertHook(
            callback=callback,
            name=name,
            priority=priority,
            failure_policy=failure_policy,
            max_retries=max_retries,
            enabled=enabled
        )
        
        self.hooks[name] = hook
        logger.info(f"Registered alert hook '{name}' with priority {priority.name}")
        
        # Return the function for decorator usage
        return callback if callback else None
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a hook.
        
        Args:
            name: Name of the hook to unregister
            
        Returns:
            True if the hook was removed, False if it wasn't found
        """
        if name in self.hooks:
            del self.hooks[name]
            logger.info(f"Unregistered alert hook '{name}'")
            return True
        return False
    
    def enable(self, name: str) -> bool:
        """
        Enable a hook.
        
        Args:
            name: Name of the hook to enable
            
        Returns:
            True if the hook was enabled, False if it wasn't found
        """
        if name in self.hooks:
            self.hooks[name].enabled = True
            logger.debug(f"Enabled alert hook '{name}'")
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """
        Disable a hook temporarily without unregistering it.
        
        Args:
            name: Name of the hook to disable
            
        Returns:
            True if the hook was disabled, False if it wasn't found
        """
        if name in self.hooks:
            self.hooks[name].enabled = False
            logger.debug(f"Disabled alert hook '{name}'")
            return True
        return False
    
    def execute_all(
        self, 
        alerts: List[StockAlert], 
        report: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Execute all registered hooks in priority order.
        
        Args:
            alerts: List of StockAlert objects to send to hooks
            report: Additional report data to provide to hooks
            
        Returns:
            Tuple of (success, error_list) where:
                - success is True if all hooks executed successfully
                - error_list contains error messages from any failed hooks
        """
        # Sort hooks by priority (highest first)
        sorted_hooks = sorted(
            [hook for hook in self.hooks.values() if hook.enabled],
            key=lambda h: h.priority.value,
            reverse=True
        )
        
        success = True
        error_messages = []
        
        for hook in sorted_hooks:
            try:
                # Execute hook with retry logic if configured
                retries = 0
                while True:
                    try:
                        hook.callback(alerts, report)
                        # Success, reset error count
                        hook.error_count = 0
                        hook.last_error = None
                        break
                    except Exception as e:
                        retries += 1
                        hook.error_count += 1
                        hook.last_error = str(e)
                        
                        error_msg = f"Error in alert hook '{hook.name}': {str(e)}"
                        logger.error(error_msg)
                        
                        # If we've hit max retries or not using retry policy, break out
                        if retries > hook.max_retries or hook.failure_policy != HookFailurePolicy.RETRY:
                            error_messages.append(error_msg)
                            success = False
                            break
                        
                        logger.info(f"Retrying hook '{hook.name}' (attempt {retries}/{hook.max_retries})")
                
                # Handle abort policy
                if not success and hook.failure_policy == HookFailurePolicy.ABORT:
                    logger.warning(f"Hook '{hook.name}' failed with ABORT policy, skipping remaining hooks")
                    break
                    
            except Exception as e:
                # Handle unexpected errors at the hook execution level
                error_msg = f"Unexpected error executing hook '{hook.name}': {str(e)}"
                logger.exception(error_msg)
                error_messages.append(error_msg)
                success = False
                
                # Handle abort policy
                if hook.failure_policy == HookFailurePolicy.ABORT:
                    logger.warning(f"Hook '{hook.name}' failed with ABORT policy, skipping remaining hooks")
                    break
        
        return success, error_messages

# Create the global hook manager instance
alert_hook_manager = AlertHookManager()

# For backward compatibility, expose hook_manager functions at module level
register_alert_hook = alert_hook_manager.register
unregister_alert_hook = alert_hook_manager.unregister
enable_alert_hook = alert_hook_manager.enable
disable_alert_hook = alert_hook_manager.disable

# Update the scheduled_low_stock_detection function to use hooks
def scheduled_low_stock_detection(
    output_dir: Optional[Path] = None,
    notify: bool = True
) -> List[StockAlert]:
    """
    Run low stock detection as a scheduled task and optionally save results.
    
    Args:
        output_dir: Directory to save report files. If None, reports are not saved.
        notify: Whether to trigger notifications for critical items.
        
    Returns:
        List of stock alerts.
    """
    logger.info("Running scheduled low stock detection")
    
    # Clear the cache to ensure we get fresh data
    clear_alert_cache()
    
    # Get configuration settings
    config = get_config()
    use_config_overrides = config.get("alerts.scheduled.use_config_overrides", True)
    use_env_overrides = config.get("alerts.scheduled.use_env_overrides", True)
    
    # Run the detection with configured overrides
    alerts = detect_low_stock(
        include_config_overrides=use_config_overrides,
        include_env_overrides=use_env_overrides
    )
    
    # Generate a report
    report = generate_reorder_report(alerts)
    
    saved_files = []
    
    # Save reports if a directory is specified
    if output_dir:
        # Ensure the directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save a JSON report
        json_path = output_dir / f"reorder_report_{timestamp}.json"
        try:
            import json
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Saved JSON report to {json_path}")
            saved_files.append(json_path)
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
        
        # Save an Excel report if xlsxwriter is available
        try:
            from .commands.reorders import export_to_excel, calculate_urgency_score
            
            xlsx_path = output_dir / f"reorder_report_{timestamp}.xlsx"
            alerts_with_urgency = [(alert, calculate_urgency_score(alert)) for alert in alerts]
            export_to_excel(
                alerts_with_urgency,
                xlsx_path,
                include_report=True,
                show_overrides=True
            )
            logger.info(f"Saved Excel report to {xlsx_path}")
            saved_files.append(xlsx_path)
        except ImportError:
            logger.warning("xlsxwriter not available, skipping Excel report generation")
        except Exception as e:
            logger.error(f"Failed to save Excel report: {e}")
    
    # Process notifications if enabled
    if notify and alerts:
        # Set report metadata
        report['saved_files'] = [str(path) for path in saved_files]
        report['timestamp'] = datetime.datetime.now().isoformat()
        
        # Run notification hooks
        success, errors = alert_hook_manager.execute_all(alerts, report)
        
        if not success:
            logger.warning(f"Some notification hooks failed: {'; '.join(errors)}")
        else:
            logger.info("All notification hooks executed successfully")
    
    return alerts