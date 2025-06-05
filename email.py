# inventorytracker/notifications/email.py
"""
Email notification system for inventory alerts and reports.

This module provides functionality to send email notifications about
inventory status, including low stock alerts and reorder reports.
"""

import os
import time
import smtplib
import socket
import logging
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.utils import formatdate, make_msgid
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import random
from datetime import datetime

import jinja2

from ..config import get_config
from ..alerts import StockAlert

logger = logging.getLogger(__name__)

# Define retry parameters
MAX_RETRIES = 5
BASE_DELAY = 1.0  # Base delay in seconds before retrying
MAX_DELAY = 60.0  # Maximum delay in seconds
JITTER_FACTOR = 0.1  # Random jitter factor to avoid thundering herd

# Define SMTP error categories
TRANSIENT_ERRORS = (
    smtplib.SMTPServerDisconnected,
    smtplib.SMTPConnectError,
    smtplib.SMTPHeloError,
    smtplib.SMTPNotSupportedError,
    smtplib.SMTPSenderRefused,
    smtplib.SMTPDataError,
    socket.gaierror,
    socket.timeout,
    ConnectionError,
    ConnectionResetError,
    ConnectionRefusedError,
    TimeoutError,
)

PERMANENT_ERRORS = (
    smtplib.SMTPRecipientsRefused,  # All recipients were rejected
    smtplib.SMTPResponseException,  # Specific SMTP error codes
    smtplib.SMTPAuthenticationError,  # Authentication failed
)

class EmailNotificationError(Exception):
    """Base exception for email notification errors."""
    pass

class TemporaryEmailError(EmailNotificationError):
    """Temporary error that might be resolved by retrying."""
    pass

class PermanentEmailError(EmailNotificationError):
    """Permanent error that won't be resolved by retrying."""
    pass

class EmailConfig:
    """Configuration for email notifications."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize email configuration from config dictionary.
        
        Args:
            config_dict: Dictionary with configuration values. If None, loads from global config.
        """
        if config_dict is None:
            config = get_config()
            config_dict = config.get("notifications", {}).get("email", {})
            
        # SMTP server settings
        self.host = config_dict.get("host", "localhost")
        self.port = int(config_dict.get("port", 25))
        self.use_tls = bool(config_dict.get("use_tls", False))
        self.use_ssl = bool(config_dict.get("use_ssl", False))
        self.username = config_dict.get("username")
        self.password = config_dict.get("password")
        
        # Email settings
        self.sender = config_dict.get("sender")
        self.sender_name = config_dict.get("sender_name", "Inventory Tracker")
        self.reply_to = config_dict.get("reply_to")
        self.default_recipients = config_dict.get("recipients", [])
        
        # Content settings
        template_dir = config_dict.get("template_dir")
        self.template_dir = Path(template_dir) if template_dir else None
        
        # Retry settings
        self.max_retries = int(config_dict.get("max_retries", MAX_RETRIES))
        self.base_delay = float(config_dict.get("base_delay", BASE_DELAY))
        self.max_delay = float(config_dict.get("max_delay", MAX_DELAY))
        self.enable_retries = bool(config_dict.get("enable_retries", True))
        
    @property
    def is_valid(self) -> bool:
        """Check if the configuration has the minimum required values."""
        return bool(self.host and self.sender) 
        
    def validate(self) -> List[str]:
        """
        Validate the configuration and return a list of error messages.
        
        Returns:
            List of error messages, empty if configuration is valid.
        """
        errors = []
        
        if not self.host:
            errors.append("SMTP host is not configured")
        
        if not self.sender:
            errors.append("Sender email is not configured")
        
        if self.use_ssl and self.use_tls:
            errors.append("Both TLS and SSL are enabled, choose one")
        
        if (self.username or self.password) and not (self.username and self.password):
            errors.append("Both username and password must be provided for authentication")
            
        if not self.default_recipients:
            errors.append("No default recipients configured")
            
        return errors

class EmailNotifier:
    """
    Handles sending email notifications for inventory events.
    
    This class provides methods to send notifications about low stock levels,
    reorder reports, and other inventory-related events.
    """
    
    def __init__(self, config: Optional[EmailConfig] = None):
        """
        Initialize the email notifier with the provided configuration.
        
        Args:
            config: Email configuration. If None, loads from global config.
        """
        self.config = config or EmailConfig()
        
        # Set up template environment
        if self.config.template_dir and self.config.template_dir.exists():
            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(self.config.template_dir)),
                autoescape=True
            )
        else:
            # Use package templates as fallback
            package_dir = Path(__file__).parent.parent
            template_dir = package_dir / "templates" / "email"
            if template_dir.exists():
                self.jinja_env = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(str(template_dir)),
                    autoescape=True
                )
            else:
                self.jinja_env = None
                logger.warning(f"No email templates found at {template_dir}")
    
    def send_reorder_report(
        self, 
        report: Dict[str, Any],
        alerts: List[StockAlert],
        recipients: Optional[List[str]] = None,
        attachments: Optional[List[Path]] = None,
        subject: Optional[str] = None,
    ) -> bool:
        """
        Send a reorder report email with information about low stock items.
        
        Args:
            report: Reorder report dictionary
            alerts: List of stock alerts
            recipients: Email recipients. If None, uses default recipients from config.
            attachments: Optional list of files to attach
            subject: Email subject. If None, uses a default subject.
            
        Returns:
            True if email was sent successfully, False otherwise
            
        Raises:
            PermanentEmailError: For permanent errors that won't be resolved by retrying
            TemporaryEmailError: For temporary errors that might be resolved by retrying
        """
        if not self.config.is_valid:
            validation_errors = self.config.validate()
            error_msg = f"Invalid email configuration: {', '.join(validation_errors)}"
            logger.error(error_msg)
            raise PermanentEmailError(error_msg)
            
        # Use default recipients if none provided
        recipients = recipients or self.config.default_recipients
        
        if not recipients:
            logger.error("No recipients specified for reorder report email")
            raise PermanentEmailError("No recipients specified")
        
        # Create the email
        msg = MIMEMultipart()
        msg['From'] = f"{self.config.sender_name} <{self.config.sender}>"
        msg['To'] = ", ".join(recipients)
        msg['Date'] = formatdate(localtime=True)
        msg['Message-ID'] = make_msgid(domain=self.config.sender.split('@')[-1])
        
        # Set reply-to if configured
        if self.config.reply_to:
            msg['Reply-To'] = self.config.reply_to
            
        # Set subject
        if subject:
            msg['Subject'] = subject
        else:
            # Create default subject based on report severity
            critical_count = report.get("critical_count", 0)
            out_of_stock_count = report.get("out_of_stock_count", 0)
            
            if out_of_stock_count > 0:
                msg['Subject'] = f"URGENT: {out_of_stock_count} products out of stock"
            elif critical_count > 0:
                msg['Subject'] = f"Critical: {critical_count} products need immediate reordering"
            else:
                total_alerts = report.get("total_alerts", 0)
                msg['Subject'] = f"Reorder Report: {total_alerts} products below reorder level"
        
        # Generate the email body
        body_html = self._render_email_template(
            "reorder_report.html",
            report=report,
            alerts=alerts,
            current_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        
        # Fallback to plain text if HTML template fails
        if body_html is None:
            body_text = self._generate_text_report(report, alerts)
            msg.attach(MIMEText(body_text, 'plain'))
        else:
            # Attach both HTML and plain text versions
            text_version = self._render_email_template(
                "reorder_report.txt",
                report=report,
                alerts=alerts,
                current_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            ) or self._generate_text_report(report, alerts)
            
            msg.attach(MIMEText(text_version, 'plain'))
            msg.attach(MIMEText(body_html, 'html'))
        
        # Add attachments if any
        if attachments:
            for attachment_path in attachments:
                if not attachment_path.exists():
                    logger.warning(f"Attachment not found: {attachment_path}")
                    continue
                    
                try:
                    with open(attachment_path, "rb") as attachment_file:
                        part = MIMEApplication(attachment_file.read())
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename="{attachment_path.name}"'
                        )
                        msg.attach(part)
                except Exception as e:
                    logger.error(f"Failed to attach file {attachment_path}: {e}")
        
        # Send the email with retries
        if self.config.enable_retries:
            return self._send_with_retry(msg)
        else:
            return self._send_email(msg)
    
    def send_stock_alert(
        self,
        alert: StockAlert,
        recipients: Optional[List[str]] = None,
        subject: Optional[str] = None,
    ) -> bool:
        """
        Send an email notification about a specific stock alert.
        
        Args:
            alert: Stock alert to notify about
            recipients: Email recipients. If None, uses default recipients from config.
            subject: Email subject. If None, uses a default subject.
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        if not self.config.is_valid:
            validation_errors = self.config.validate()
            error_msg = f"Invalid email configuration: {', '.join(validation_errors)}"
            logger.error(error_msg)
            raise PermanentEmailError(error_msg)
            
        # Use default recipients if none provided
        recipients = recipients or self.config.default_recipients
        
        if not recipients:
            logger.error("No recipients specified for stock alert email")
            raise PermanentEmailError("No recipients specified")
        
        # Create the default subject if not provided
        if subject is None:
            if alert.is_out_of_stock:
                subject = f"URGENT: {alert.product_name} (SKU: {alert.product_sku}) is OUT OF STOCK"
            else:
                subject = f"Low Stock Alert: {alert.product_name} (SKU: {alert.product_sku})"
        
        # Generate the email body
        body_html = self._render_email_template(
            "stock_alert.html",
            alert=alert,
            current_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        
        # Create the email
        msg = MIMEMultipart()
        msg['From'] = f"{self.config.sender_name} <{self.config.sender}>"
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = subject
        msg['Date'] = formatdate(localtime=True)
        msg['Message-ID'] = make_msgid(domain=self.config.sender.split('@')[-1])
        
        # Set reply-to if configured
        if self.config.reply_to:
            msg['Reply-To'] = self.config.reply_to
        
        # Fallback to plain text if HTML template fails
        if body_html is None:
            # Simple text version
            text_body = f"""
Stock Alert: {alert.product_name}
SKU: {alert.product_sku}
Current Stock: {alert.current_stock}
Reorder Level: {alert.reorder_level}
Deficit: {alert.deficit}
{'OUT OF STOCK!' if alert.is_out_of_stock else 'Low stock'}

This is an automated notification from Inventory Tracker.
            """
            msg.attach(MIMEText(text_body, 'plain'))
        else:
            # Attach both HTML and plain text versions
            text_version = self._render_email_template(
                "stock_alert.txt",
                alert=alert,
                current_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
            
            if text_version:
                msg.attach(MIMEText(text_version, 'plain'))
            msg.attach(MIMEText(body_html, 'html'))
            
        # Send the email with retries
        if self.config.enable_retries:
            return self._send_with_retry(msg)
        else:
            return self._send_email(msg)
    
    def _send_email(self, msg: MIMEMultipart) -> bool:
        """
        Send an email message via SMTP.
        
        Args:
            msg: The email message to send
            
        Returns:
            True if sent successfully
            
        Raises:
            TemporaryEmailError: For temporary failures that may succeed with retry
            PermanentEmailError: For permanent failures
        """
        sender = self.config.sender
        recipients = msg['To'].split(', ')
        
        # Choose the appropriate SMTP connection method
        smtp_class = smtplib.SMTP_SSL if self.config.use_ssl else smtplib.SMTP
        
        try:
            # Connect to the SMTP server
            with smtp_class(self.config.host, self.config.port, timeout=30) as server:
                if self.config.use_tls and not self.config.use_ssl:
                    # Upgrade connection to TLS if configured
                    server.starttls()
                
                # Authenticate if credentials are provided
                if self.config.username and self.config.password:
                    server.login(self.config.username, self.config.password)
                
                # Send the email
                server.send_message(msg)
                
            logger.info(f"Email sent to {len(recipients)} recipients")
            return True
            
        except PERMANENT_ERRORS as e:
            error_message = f"Permanent email error: {str(e)}"
            logger.error(error_message)
            
            # Handle authentication errors specifically
            if isinstance(e, smtplib.SMTPAuthenticationError):
                logger.error("SMTP authentication failed - check username and password")
                
            raise PermanentEmailError(error_message) from e
            
        except TRANSIENT_ERRORS as e:
            error_message = f"Temporary email error: {str(e)}"
            logger.warning(error_message)
            raise TemporaryEmailError(error_message) from e
            
        except smtplib.SMTPException as e:
            # Generic SMTP exceptions - might be temporary or permanent
            error_message = f"SMTP error: {str(e)}"
            logger.error(error_message)
            
            # Try to classify based on error code if available
            if hasattr(e, 'smtp_code') and 400 <= e.smtp_code < 500:
                raise TemporaryEmailError(error_message) from e
            elif hasattr(e, 'smtp_code') and e.smtp_code >= 500:
                raise PermanentEmailError(error_message) from e
            else:
                # Default to temporary error if we can't determine
                raise TemporaryEmailError(error_message) from e
                
        except Exception as e:
            # Unexpected errors
            error_message = f"Unexpected error sending email: {str(e)}"
            logger.exception(error_message)
            raise TemporaryEmailError(error_message) from e
    
    def _send_with_retry(self, msg: MIMEMultipart) -> bool:
        """
        Send an email with exponential backoff retry logic.
        
        Args:
            msg: The email message to send
            
        Returns:
            True if sent successfully, False after all retries failed
        """
        retries = 0
        
        while retries <= self.config.max_retries:
            try:
                return self._send_email(msg)
            except PermanentEmailError:
                # Don't retry permanent errors
                logger.error("Encountered permanent error, not retrying")
                return False
            except TemporaryEmailError as e:
                # Log the error and retry with backoff
                retries += 1
                
                if retries > self.config.max_retries:
                    logger.error(f"Failed to send email after {self.config.max_retries} retries")
                    return False
                
                # Calculate delay using exponential backoff with jitter
                delay = min(
                    self.config.base_delay * (2 ** (retries - 1)),
                    self.config.max_delay
                )
                
                # Add jitter to avoid thundering herd problem
                jitter = delay * JITTER_FACTOR * random.uniform(-1, 1)
                delay += jitter
                
                logger.warning(
                    f"Temporary error sending email, retry {retries}/{self.config.max_retries} "
                    f"after {delay:.2f}s: {str(e)}"
                )
                
                # Sleep before retry
                time.sleep(delay)
        
        # If we get here, all retries failed
        return False
    
    def _render_email_template(self, template_name: str, **context) -> Optional[str]:
        """
        Render an email template with the provided context.
        
        Args:
            template_name: Name of the template file
            **context: Template context variables
            
        Returns:
            The rendered template string, or None if template rendering failed
        """
        if not self.jinja_env:
            return None
            
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**context)
        except jinja2.exceptions.TemplateNotFound:
            logger.warning(f"Email template not found: {template_name}")
            return None
        except Exception as e:
            logger.error(f"Error rendering email template {template_name}: {e}")
            return None
    
    def _generate_text_report(self, report: Dict[str, Any], alerts: List[StockAlert]) -> str:
        """
        Generate a plain text report when templates are not available.
        
        Args:
            report: Reorder report dictionary
            alerts: List of stock alerts
            
        Returns:
            Plain text report
        """
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        total_alerts = report.get("total_alerts", 0)
        out_of_stock = report.get("out_of_stock_count", 0)
        critical_count = report.get("critical_count", 0)
        
        text = f"""
INVENTORY REORDER REPORT
Generated: {report_date}

SUMMARY:
- Total items below reorder level: {total_alerts}
- Out of stock items: {out_of_stock}
- Critical items: {critical_count}
- Total units needed: {report.get("total_items_needed", 0)}

"""
        
        if alerts:
            # Group alerts by priority
            priority_groups = {1: [], 2: [], 3: [], 4: [], 5: []}
            for alert in alerts:
                priority_groups[alert.priority].append(alert)
                
            # Add out-of-stock items first
            if priority_groups[1]:
                text += "\nOUT OF STOCK ITEMS:\n"
                for alert in priority_groups[1]:
                    text += f"- {alert.product_name} (SKU: {alert.product_sku})\n"
                    text += f"  Current: {alert.current_stock}, Reorder: {alert.reorder_level}, Deficit: {alert.deficit}\n"
                    
            # Add critical items next
            if priority_groups[2]:
                text += "\nCRITICAL LOW STOCK ITEMS:\n"
                for alert in priority_groups[2]:
                    text += f"- {alert.product_name} (SKU: {alert.product_sku})\n"
                    text += f"  Current: {alert.current_stock}, Reorder: {alert.reorder_level}, Deficit: {alert.deficit}\n"
                    
            # Add other items if any
            other_items = sum(len(items) for priority, items in priority_groups.items() if priority > 2)
            if other_items > 0:
                text += f"\nADDITIONAL LOW STOCK ITEMS: {other_items}\n"
                text += "See attached report or dashboard for details.\n"
        
        text += "\n\nThis is an automated notification from Inventory Tracker."
        return text


def send_reorder_report_email(
    report: Dict[str, Any],
    alerts: List[StockAlert],
    recipients: Optional[List[str]] = None,
    attachments: Optional[List[Path]] = None,
) -> bool:
    """
    Convenience function to send a reorder report email.
    
    Args:
        report: Reorder report data
        alerts: List of stock alerts
        recipients: List of email recipients
        attachments: Optional list of file attachments
        
    Returns:
        True if email was sent successfully, False otherwise
    """
    try:
        notifier = EmailNotifier()
        return notifier.send_reorder_report(
            report=report,
            alerts=alerts,
            recipients=recipients,
            attachments=attachments
        )
    except (TemporaryEmailError, PermanentEmailError) as e:
        logger.error(f"Failed to send reorder report email: {e}")
        return False