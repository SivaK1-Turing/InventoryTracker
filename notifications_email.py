# notifications/email.py
import os
import ssl
import time
import logging
import tomli
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import smtplib

# Import optional aiosmtplib for async support
try:
    import aiosmtplib
    ASYNC_SUPPORT = True
except ImportError:
    ASYNC_SUPPORT = False

from .base import Notifier, SyncNotifier, AsyncNotifier, NotificationPriority

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Utility class to load email configuration from config.toml"""
    
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load email configuration from config.toml file.
        
        Args:
            config_path: Path to config.toml file. If None, will check standard locations.
            
        Returns:
            Dictionary with email configuration
            
        Raises:
            FileNotFoundError: If config file cannot be found
            ValueError: If config is invalid or missing required sections
        """
        # Search paths for config file
        search_paths = [
            config_path,  # User-specified path
            os.environ.get('INVENTORY_CONFIG'),  # Environment variable
            './config.toml',  # Current directory
            './config/config.toml',  # Config subdirectory
            '/etc/inventory_tracker/config.toml',  # System config
            str(Path.home() / '.inventory_tracker' / 'config.toml')  # User home
        ]
        
        # Filter out None values
        search_paths = [path for path in search_paths if path]
        
        # Try each path until we find a config file
        for path in search_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        config_data = tomli.load(f)
                        
                    # Email config should be under 'email' section
                    if 'email' not in config_data:
                        logger.warning(f"No 'email' section in config file: {path}")
                        continue
                        
                    email_config = config_data['email']
                    
                    # Validate required fields
                    if not all(k in email_config for k in ['smtp_server', 'smtp_port', 'sender_email']):
                        raise ValueError("Missing required email configuration fields")
                        
                    # Add environment variables if credentials are not in config
                    # This allows for more secure credential handling
                    if 'username' not in email_config:
                        email_config['username'] = os.environ.get('INVENTORY_SMTP_USERNAME')
                        
                    if 'password' not in email_config:
                        email_config['password'] = os.environ.get('INVENTORY_SMTP_PASSWORD')
                    
                    # Set default values for optional fields
                    email_config.setdefault('use_tls', True)
                    email_config.setdefault('retry_count', 3)
                    email_config.setdefault('retry_delay', 2)  # seconds
                    email_config.setdefault('timeout', 10)  # seconds
                    email_config.setdefault('async_mode', ASYNC_SUPPORT)
                    
                    logger.info(f"Loaded email configuration from {path}")
                    return email_config
                    
                except Exception as e:
                    logger.error(f"Error loading config from {path}: {e}")
                    continue
        
        # If we get here, no valid config file was found
        raise FileNotFoundError("No valid config.toml file found for email configuration")


class EmailNotifier(Notifier):
    """
    Email notifier implementation that can operate in both synchronous and asynchronous modes.
    
    This implementation automatically chooses between smtplib and aiosmtplib based on:
    1. Configuration setting 'async_mode'
    2. Availability of aiosmtplib package
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize EmailNotifier with either explicit config or from config file.
        
        Args:
            config: Dictionary with email configuration. If None, will load from config.toml
            config_path: Path to config.toml file, used only if config is None
        """
        # Load config from file if not provided explicitly
        if config is None:
            config = ConfigLoader.load_config(config_path)
            
        # Determine mode based on config and available packages
        self.async_mode = config.get('async_mode', False) and ASYNC_SUPPORT
        
        super().__init__(config)
    
    def _validate_config(self) -> None:
        """Validate email notifier configuration."""
        required_fields = ['smtp_server', 'smtp_port', 'sender_email']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
    
    def send(self, subject: str, body: str, **kwargs) -> Union[bool, asyncio.Future]:
        """
        Send an email notification.
        
        Args:
            subject: Email subject
            body: Email body content
            **kwargs: Additional parameters including:
                - recipients: List of email recipients (required)
                - cc: List of CC recipients (optional)
                - bcc: List of BCC recipients (optional)
                - html: Whether body is HTML (default: False)
                - priority: NotificationPriority (default: NORMAL)
                
        Returns:
            If async_mode is False: bool indicating success
            If async_mode is True: Future that resolves to bool
        """
        # Validate recipients
        recipients = kwargs.get('recipients')
        if not recipients:
            raise ValueError("Recipients list is required")
            
        try:
            # Create the message
            message = self._create_message(subject, body, **kwargs)
            
            # Choose sync or async sending
            if self.async_mode:
                return asyncio.ensure_future(self._send_async(message, recipients, **kwargs))
            else:
                return self._send_sync(message, recipients, **kwargs)
                
        except Exception as e:
            logger.exception(f"Error preparing email: {e}")
            # Return appropriate type based on mode
            if self.async_mode:
                future = asyncio.Future()
                future.set_result(False)
                return future
            else:
                return False
    
    def _create_message(self, subject: str, body: str, **kwargs) -> MIMEMultipart:
        """
        Create email message with proper formatting and headers.
        
        Args:
            subject: Email subject
            body: Email body
            **kwargs: Additional parameters
            
        Returns:
            Prepared MIMEMultipart message object
        """
        recipients = kwargs.get('recipients', [])
        cc = kwargs.get('cc', [])
        bcc = kwargs.get('bcc', [])
        is_html = kwargs.get('html', False)
        priority = kwargs.get('priority', NotificationPriority.NORMAL)
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.config['sender_email']
        msg['To'] = ', '.join(recipients)
        
        if cc:
            msg['Cc'] = ', '.join(cc)

        # Set standard email priority headers
        if priority == NotificationPriority.HIGH or priority == NotificationPriority.URGENT:
            msg['X-Priority'] = '1'  # High
            msg['X-MSMail-Priority'] = 'High'
            msg['Importance'] = 'High'
        elif priority == NotificationPriority.LOW:
            msg['X-Priority'] = '5'  # Low
            msg['X-MSMail-Priority'] = 'Low'
            msg['Importance'] = 'Low'
        else:
            msg['X-Priority'] = '3'  # Normal
            msg['X-MSMail-Priority'] = 'Normal'
            msg['Importance'] = 'Normal'
            
        # Attach body
        content_type = 'html' if is_html else 'plain'
        msg.attach(MIMEText(body, content_type))
        
        return msg
    
    def _send_sync(self, message: MIMEMultipart, recipients: List[str], **kwargs) -> bool:
        """
        Send email synchronously using smtplib.
        
        Args:
            message: Prepared email message
            recipients: List of recipients
            **kwargs: Additional parameters
            
        Returns:
            bool: True if sending was successful, False otherwise
        """
        cc = kwargs.get('cc', [])
        bcc = kwargs.get('bcc', [])
        all_recipients = recipients + cc + bcc
        
        retry_count = self.config['retry_count']
        retry_delay = self.config['retry_delay']
        timeout = self.config['timeout']
        
        for attempt in range(retry_count):
            try:
                if self.config['use_tls']:
                    context = ssl.create_default_context()
                    with smtplib.SMTP(
                        self.config['smtp_server'], 
                        self.config['smtp_port'],
                        timeout=timeout
                    ) as server:
                        server.starttls(context=context)
                        
                        # Handle authentication if credentials provided
                        if self._has_credentials():
                            server.login(
                                self.config['username'],
                                self.config['password']
                            )
                        
                        server.send_message(message, self.config['sender_email'], all_recipients)
                else:
                    with smtplib.SMTP(
                        self.config['smtp_server'], 
                        self.config['smtp_port'],
                        timeout=timeout
                    ) as server:
                        # Handle authentication if credentials provided
                        if self._has_credentials():
                            server.login(
                                self.config['username'],
                                self.config['password']
                            )
                        
                        server.send_message(message, self.config['sender_email'], all_recipients)
                
                logger.info(f"Email sent successfully to {len(all_recipients)} recipients")
                return True
                
            except (smtplib.SMTPException, ssl.SSLError, TimeoutError, ConnectionError) as e:
                logger.warning(f"Failed to send email (attempt {attempt+1}/{retry_count}): {e}")
                
                if attempt < retry_count - 1:
                    # Wait before retrying, with exponential backoff
                    backoff_time = self._calculate_backoff(attempt, retry_delay)
                    logger.info(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                else:
                    logger.error(f"Failed to send email after {retry_count} attempts: {e}")
                    return False
        
        return False
    
    async def _send_async(self, message: MIMEMultipart, recipients: List[str], **kwargs) -> bool:
        """
        Send email asynchronously using aiosmtplib.
        
        Args:
            message: Prepared email message
            recipients: List of recipients
            **kwargs: Additional parameters
            
        Returns:
            bool: True if sending was successful, False otherwise
        """
        if not ASYNC_SUPPORT:
            logger.error("Async mode requested but aiosmtplib is not available")
            return False
            
        cc = kwargs.get('cc', [])
        bcc = kwargs.get('bcc', [])
        all_recipients = recipients + cc + bcc
        
        retry_count = self.config['retry_count']
        retry_delay = self.config['retry_delay']
        timeout = self.config['timeout']
        
        for attempt in range(retry_count):
            try:
                smtp_client = aiosmtplib.SMTP(
                    hostname=self.config['smtp_server'],
                    port=self.config['smtp_port'],
                    timeout=timeout
                )
                
                await smtp_client.connect()
                
                if self.config['use_tls']:
                    context = ssl.create_default_context()
                    await smtp_client.starttls(validate_certs=True, ssl_context=context)
                
                # Handle authentication if credentials provided
                if self._has_credentials():
                    await smtp_client.login(
                        self.config['username'],
                        self.config['password']
                    )
                
                # Convert message to flat string for aiosmtplib
                message_str = message.as_string()
                await smtp_client.sendmail(
                    self.config['sender_email'], 
                    all_recipients, 
                    message_str
                )
                
                await smtp_client.quit()
                logger.info(f"Email sent asynchronously to {len(all_recipients)} recipients")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to send async email (attempt {attempt+1}/{retry_count}): {e}")
                
                if attempt < retry_count - 1:
                    # Wait before retrying, with exponential backoff
                    backoff_time = self._calculate_backoff(attempt, retry_delay)
                    logger.info(f"Retrying in {backoff_time} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Failed to send async email after {retry_count} attempts: {e}")
                    return False
        
        return False
    
    def _has_credentials(self) -> bool:
        """Check if SMTP credentials are available."""
        return (
            'username' in self.config and 
            'password' in self.config and 
            self.config['username'] is not None and 
            self.config['password'] is not None
        )
    
    def _calculate_backoff(self, attempt: int, base_delay: float) -> float:
        """Calculate exponential backoff time with jitter."""
        # Exponential backoff with a small random jitter
        backoff = base_delay * (2 ** attempt)
        jitter = 0.1 * backoff * (2 * (0.5 - (time.time() % 1)))  # +/- 10% randomness
        return max(0, backoff + jitter)


# Factory function for easy creation
def create_email_notifier(config_path: Optional[str] = None) -> EmailNotifier:
    """
    Create and configure an EmailNotifier from config file.
    
    Args:
        config_path: Optional path to config.toml file
        
    Returns:
        Configured EmailNotifier instance
    """
    config = ConfigLoader.load_config(config_path)
    return EmailNotifier(config)


# Example config.toml structure:
"""
[email]
smtp_server = "smtp.gmail.com"
smtp_port = 587
sender_email = "alerts@inventory-tracker.com"
username = "alerts@inventory-tracker.com"  # Optional: can use env var instead
password = "app-specific-password"  # Optional: can use env var instead
use_tls = true
retry_count = 3
retry_delay = 2
timeout = 10
async_mode = true  # Set to false to use synchronous mode
"""