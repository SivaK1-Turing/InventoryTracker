# inventorytracker/commands/scheduler.py
"""
Commands for managing scheduled tasks including low stock alerts.

This module provides CLI commands to start, stop, and monitor the scheduler
that runs automated tasks like low stock detection.
"""

import datetime
import time
import signal
import sys
import os
from typing import Optional, List
from pathlib import Path
import threading
import asyncio

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.status import Status
from rich.live import Live
from rich.box import ROUNDED

from ..alerts import (
    start_scheduler, stop_scheduler, is_scheduler_running, 
    get_next_scheduled_run, setup_scheduled_tasks, scheduled_low_stock_detection
)
from ..config import get_config

# Initialize Typer command group
scheduler_app = typer.Typer(help="Manage scheduled tasks like low stock detection")

# Initialize Rich console
console = Console()

@scheduler_app.command("start")
def start_scheduler_command(
    foreground: bool = typer.Option(
        False,
        "--foreground",
        "-f",
        help="Run in foreground instead of as a daemon"
    ),
    now: bool = typer.Option(
        False,
        "--now",
        "-n",
        help="Run the scheduled detection immediately after starting"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save reports (defaults to config setting)"
    )
):
    """
    Start the scheduled task runner for automated low stock detection.
    
    This command starts the scheduler which will run low stock detection
    according to the configured schedule.
    """
    # Check if scheduler is already running
    if is_scheduler_running():
        console.print("[bold yellow]Scheduler is already running[/bold yellow]")
        return
    
    # If output_dir is not specified, get from config
    if output_dir is None:
        config = get_config()
        config_dir = config.get("alerts.scheduled.output_dir")
        if config_dir:
            output_dir = Path(config_dir)

    # Start the scheduler
    daemon = not foreground
    success = start_scheduler(daemon=daemon)
    
    if not success:
        console.print("[bold red]Failed to start scheduler - no jobs configured[/bold red]")
        console.print("Check your configuration to ensure scheduled alerts are enabled.")
        raise typer.Exit(code=1)
    
    # Get the next scheduled run
    next_run = get_next_scheduled_run()
    
    console.print("[bold green]Scheduler started successfully[/bold green]")
    if next_run:
        console.print(f"Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run immediately if requested
    if now:
        console.print("[yellow]Running low stock detection now...[/yellow]")
        alerts = scheduled_low_stock_detection(output_dir=output_dir)
        console.print(f"[green]Detected {len(alerts)} products below reorder level[/green]")
    
    # If running in foreground, keep the process alive with status display
    if foreground:
        try:
            with console.status("[bold green]Scheduler running. Press Ctrl+C to stop...[/bold green]") as status:
                while is_scheduler_running():
                    next_run = get_next_scheduled_run()
                    if next_run:
                        time_until = next_run - datetime.datetime.now()
                        hours, remainder = divmod(time_until.total_seconds(), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        
                        if time_until.total_seconds() < 0:
                            status.update("[bold yellow]Executing scheduled task now...[/bold yellow]")
                        else:
                            status.update(
                                f"[bold green]Scheduler running. "
                                f"Next run in {int(hours):02}:{int(minutes):02}:{int(seconds):02}. "
                                f"Press Ctrl+C to stop...[/bold green]"
                            )
                    
                    # Sleep briefly to avoid high CPU usage
                    time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping scheduler...[/yellow]")
            stop_scheduler()
            console.print("[green]Scheduler stopped[/green]")

@scheduler_app.command("stop")
def stop_scheduler_command():
    """
    Stop the scheduler for automated low stock detection.
    """
    if not is_scheduler_running():
        console.print("[bold yellow]Scheduler is not running[/bold yellow]")
        return
    
    console.print("[yellow]Stopping scheduler...[/yellow]")
    stop_scheduler()
    console.print("[green]Scheduler stopped successfully[/green]")

@scheduler_app.command("status")
def scheduler_status():
    """
    Show the current status of the scheduler and next scheduled runs.
    """
    is_running = is_scheduler_running()
    
    # Create a display table
    table = Table(title="Scheduler Status", box=ROUNDED)
    table.add_column("Status", style="cyan")
    table.add_column("Value", style="green")
    
    # Add status row
    if is_running:
        table.add_row("Scheduler", "[green]Running[/green]")
    else:
        table.add_row("Scheduler", "[red]Stopped[/red]")
    
    # Add next run information if scheduler is running
    if is_running:
        next_run = get_next_scheduled_run()
        if next_run:
            time_until = next_run - datetime.datetime.now()
            if time_until.total_seconds() < 0:
                table.add_row("Next Run", "[yellow]Executing now...[/yellow]")
            else:
                hours, remainder = divmod(time_until.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                table.add_row(
                    "Next Run", 
                    f"{next_run.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"({int(hours)}h {int(minutes)}m {int(seconds)}s from now)"
                )
        else:
            table.add_row("Next Run", "[yellow]No scheduled runs[/yellow]")
    
    # Add configuration information
    config = get_config()
    if config.get("alerts.scheduled.enabled", False):
        time_str = config.get("alerts.scheduled.time", "08:00")
        days = config.get("alerts.scheduled.days", ["monday", "tuesday", "wednesday", "thursday", "friday"])
        days_str = ", ".join(days)
        
        table.add_row("Scheduled Time", time_str)
        table.add_row("Scheduled Days", days_str)
        
        output_dir = config.get("alerts.scheduled.output_dir")
        if output_dir:
            table.add_row("Report Directory", output_dir)
        else:
            table.add_row("Report Directory", "[yellow]Not configured[/yellow]")
    else:
        table.add_row("Configuration", "[red]Scheduled alerts disabled in config[/red]")
    
    # Display the table
    console.print()
    console.print(table)
    console.print()

@scheduler_app.command("run")
def run_now(
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save reports (defaults to config setting)"
    ),
    notify: bool = typer.Option(
        True,
        "--notify/--no-notify",
        help="Send notifications for critical items"
    )
):
    """
    Run the low stock detection immediately without waiting for the schedule.
    """
    # If output_dir is not specified, get from config
    if output_dir is None:
        config = get_config()
        config_dir = config.get("alerts.scheduled.output_dir")
        if config_dir:
            output_dir = Path(config_dir)
    
    with console.status("[bold blue]Running low stock detection...") as status:
        alerts = scheduled_low_stock_detection(output_dir=output_dir, notify=notify)
    
    # Display summary
    critical_count = sum(1 for alert in alerts if alert.priority <= 2)
    out_of_stock = sum(1 for alert in alerts if alert.is_out_of_stock)
    
    console.print()
    console.print(f"[green]Detection completed:[/green]")
    console.print(f"  • Total alerts: [bold]{len(alerts)}[/bold]")
    
    if out_of_stock > 0:
        console.print(f"  • Out of stock: [bold red]{out_of_stock}[/bold red]")
    else:
        console.print(f"  • Out of stock: 0")
    
    if critical_count > 0:
        console.print(f"  • Critical items: [bold yellow]{critical_count}[/bold yellow]")
    else:
        console.print(f"  • Critical items: 0")
    
    if output_dir:
        console.print(f"  • Reports saved to: [bold cyan]{output_dir}[/bold cyan]")
    
    if notify and critical_count > 0:
        console.print(f"  • Notifications sent for {critical_count} critical items")
    
    console.print()

@scheduler_app.callback()
def callback():
    """
    Manage scheduled tasks like low stock detection.
    """
    pass