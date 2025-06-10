# notifications/email_notifier.py
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
import time
import logging

from .base import SyncNotifier, NotificationPriority

logger = logging.getLogger(__name__)

class EmailNotifier(SyncNotifier):
    """
    Synchronous email notifier implementation using SMTP.
    """
    
    def _validate_config(self) -> None:
        """Validate email notifier configuration."""
        required_fields = ['smtp_server', 'smtp_port', 'sender_email']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Set defaults for optional fields
        self.config.setdefault('use_tls', True)
        self.config.setdefault('retry_count', 3)
        self.config.setdefault('retry_delay', 2)  # seconds
        self.config.setdefault('timeout', 10)  # seconds
    
    def _send_impl(self, subject: str, body: str, **kwargs) -> bool:
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
            bool: True if sending was successful, False otherwise
        """
        recipients = kwargs.get('recipients')
        if not recipients:
            raise ValueError("Recipients list is required")
            
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
        
        # Set priority headers
        if priority == NotificationPriority.HIGH or priority == NotificationPriority.URGENT:
            msg['X-Priority'] = '1'  # High priority
        elif priority == NotificationPriority.LOW:
            msg['X-Priority'] = '5'  # Low priority
        else:
            msg['X-Priority'] = '3'  # Normal priority
            
        # Attach body
        content_type = 'html' if is_html else 'plain'
        msg.attach(MIMEText(body, content_type))
        
        # Prepare list of all recipients
        all_recipients = recipients + cc + bcc
        
        # Send email with retry logic
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
                        if 'username' in self.config and 'password' in self.config:
                            server.login(
                                self.config['username'],
                                self.config['password']
                            )
                        
                        server.send_message(msg, self.config['sender_email'], all_recipients)
                else:
                    with smtplib.SMTP(
                        self.config['smtp_server'], 
                        self.config['smtp_port'],
                        timeout=timeout
                    ) as server:
                        # Handle authentication if credentials provided
                        if 'username' in self.config and 'password' in self.config:
                            server.login(
                                self.config['username'],
                                self.config['password']
                            )
                        
                        server.send_message(msg, self.config['sender_email'], all_recipients)
                
                logger.info(f"Email sent successfully to {len(all_recipients)} recipients")
                return True
                
            except (smtplib.SMTPException, ssl.SSLError, TimeoutError, ConnectionError) as e:
                logger.warning(f"Failed to send email (attempt {attempt+1}/{retry_count}): {e}")
                
                if attempt < retry_count - 1:
                    # Wait before retrying, with exponential backoff
                    backoff_time = retry_delay * (2 ** attempt)
                    time.sleep(backoff_time)
                else:
                    logger.error(f"Failed to send email after {retry_count} attempts: {e}")
                    return False
        
        # Should never reach here, but just in case
        return False