"""
scheduler.py - Non-blocking task scheduler for InventoryTracker

This module provides a background scheduler that can run tasks at specified intervals
without blocking CLI operations or other application functionality.
"""

import time
import schedule
import threading
import logging
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Any, Optional, List

# Configure logging
logger = logging.getLogger(__name__)

class NonBlockingScheduler:
    """
    A non-blocking scheduler that runs tasks in the background.
    Designed to work with the schedule library without blocking CLI operations.
    """
    
    def __init__(self):
        """Initialize the scheduler."""
        self.stop_event = threading.Event()
        self.scheduler_thread = None
        self.is_running = False
        self.jobs = {}  # Track scheduled jobs by name
    
    def start(self):
        """Start the scheduler in a separate thread."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
            
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        self.is_running = True
        logger.info("Background scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
            
        self.stop_event.set()
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        self.is_running = False
        logger.info("Background scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop in a background thread."""
        logger.info("Scheduler thread started")
        
        while not self.stop_event.is_set():
            # Run any pending scheduled jobs
            schedule.run_pending()
            
            # Sleep briefly, checking for stop event at intervals
            self.stop_event.wait(1.0)  # Check every second
    
    def schedule_job(self, name: str, schedule_time: str, job_func: Callable, *args, **kwargs):
        """
        Schedule a job to run at specified time.
        
        Args:
            name: Unique name for the job
            schedule_time: Time in 'HH:MM' format to run the job
            job_func: Function to run
            *args, **kwargs: Arguments to pass to the job function
        """
        # Create a wrapper function that logs execution and handles exceptions
        def job_wrapper():
            try:
                logger.info(f"Running scheduled job: {name}")
                start_time = time.time()
                result = job_func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"Job '{name}' completed in {elapsed:.2f} seconds")
                return result
            except Exception as e:
                logger.error(f"Error executing scheduled job '{name}': {str(e)}", exc_info=True)
        
        # Cancel existing job with the same name if it exists
        if name in self.jobs:
            schedule.cancel_job(self.jobs[name])
            logger.info(f"Replaced existing scheduled job: {name}")
        
        # Schedule the job
        job = schedule.every().day.at(schedule_time).do(job_wrapper)
        self.jobs[name] = job
        
        logger.info(f"Scheduled job '{name}' to run daily at {schedule_time}")
        
        # Calculate time until next run
        next_run = job.next_run
        if next_run:
            time_until = next_run - datetime.now()
            hours, remainder = divmod(time_until.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info(f"Next run of '{name}': {next_run} ({int(hours)}h {int(minutes)}m from now)")
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all scheduled jobs.
        
        Returns:
            List of dictionaries with job information
        """
        jobs_info = []
        
        for name, job in self.jobs.items():
            next_run = job.next_run
            time_until = None
            if next_run:
                time_until = next_run - datetime.now()
            
            jobs_info.append({
                'name': name,
                'interval': str(job.interval),
                'next_run': next_run,
                'time_until': time_until
            })
        
        return jobs_info
    
    def remove_job(self, name: str) -> bool:
        """
        Remove a scheduled job by name.
        
        Args:
            name: Name of job to remove
            
        Returns:
            True if removed, False if job not found
        """
        if name in self.jobs:
            schedule.cancel_job(self.jobs[name])
            del self.jobs[name]
            logger.info(f"Removed scheduled job: {name}")
            return True
        else:
            logger.warning(f"No job found with name: {name}")
            return False

# Create the scheduler instance
scheduler = NonBlockingScheduler()

def run_low_stock_notification():
    """Run the low stock notification command."""
    # Get the path to the notify command
    cmd_dir = Path(__file__).parent.parent / "commands"
    notify_script = cmd_dir / "notify.py"
    
    if not notify_script.exists():
        logger.error(f"Notify script not found at {notify_script}")
        return
    
    # Prepare the command
    cmd = [
        sys.executable,  # Python executable
        str(notify_script),
        "--days", "14",  # Default threshold for low stock
        "--history", "30",  # Days of history to analyze
        "--type", "email",  # Default notification type
    ]
    
    # Log the command we're about to run
    logger.info(f"Running low-stock notification: {' '.join(cmd)}")
    
    # Run the command as a subprocess
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # Log the results
        if result.returncode == 0:
            logger.info("Low-stock notification completed successfully")
            
            # Extract summary from stderr (if available)
            if result.stderr:
                logger.info(f"Notification summary: {result.stderr.strip()}")
        else:
            logger.error(f"Low-stock notification failed with exit code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
        
        # Log detailed output at debug level
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.debug(f"Notification output: {line}")
                
    except Exception as e:
        logger.exception(f"Error running low-stock notification: {e}")

def setup_low_stock_notification_job():
    """Schedule the low stock notification to run daily at 07:00."""
    scheduler.schedule_job(
        name="low_stock_notification",
        schedule_time="07:00",
        job_func=run_low_stock_notification
    )

def start():
    """
    Start the scheduler and set up default jobs.
    This function should be called when the application starts.
    """
    # Start the scheduler thread
    scheduler.start()
    
    # Schedule the low-stock notification job
    setup_low_stock_notification_job()
    
    logger.info("Scheduler initialized with default jobs")

def stop():
    """
    Stop the scheduler.
    This function should be called when the application stops.
    """
    scheduler.stop()
    logger.info("Scheduler stopped")

def list_jobs() -> List[Dict[str, Any]]:
    """List all scheduled jobs."""
    return scheduler.list_jobs()

# Convenience function to add or modify a scheduled job
def schedule_task(name: str, time_str: str, task_func: Callable, *args, **kwargs):
    """
    Schedule a task to run at a specific time daily.
    
    Args:
        name: Name for the scheduled task
        time_str: Time string in format "HH:MM"
        task_func: Function to execute
        *args, **kwargs: Arguments to pass to the task function
    """
    # Validate time format
    try:
        datetime.strptime(time_str, "%H:%M")
    except ValueError:
        raise ValueError("Time must be in format HH:MM (24-hour format)")
    
    # Schedule the job
    scheduler.schedule_job(name, time_str, task_func, *args, **kwargs)
    return True

if __name__ == "__main__":
    # Example usage when run directly
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Start the scheduler
    start()
    
    # Keep the main thread alive for testing
    try:
        print("Scheduler is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping scheduler...")
        stop()
        print("Scheduler stopped.")