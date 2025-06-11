# notifications/webhook_notifier.py
import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, Union

from .base import AsyncNotifier, NotificationPriority

logger = logging.getLogger(__name__)

class WebhookNotifier(AsyncNotifier):
    """
    Asynchronous webhook notifier implementation using HTTP requests.
    """
    
    def _validate_config(self) -> None:
        """Validate webhook notifier configuration."""
        # Webhook URLs are required
        if 'urls' not in self.config:
            raise ValueError("Missing required 'urls' configuration")
            
        if not isinstance(self.config['urls'], list) or not self.config['urls']:
            raise ValueError("'urls' configuration must be a non-empty list")
            
        # Set defaults for optional fields
        self.config.setdefault('method', 'POST')
        self.config.setdefault('content_type', 'application/json')
        self.config.setdefault('timeout', 30)  # seconds
        self.config.setdefault('retry_count', 3)
        self.config.setdefault('retry_delay', 2)  # seconds
        self.config.setdefault('headers', {})
    
    async def _send_impl(self, subject: str, body: str, **kwargs) -> bool:
        """
        Send a webhook notification.
        
        Args:
            subject: Notification subject
            body: Notification body content
            **kwargs: Additional parameters including:
                - metadata: Dict of additional data to include (optional)
                - priority: NotificationPriority (default: NORMAL)
                
        Returns:
            bool: True if sending was successful to at least one endpoint
        """
        metadata = kwargs.get('metadata', {})
        priority = kwargs.get('priority', NotificationPriority.NORMAL)
        
        # Prepare payload
        payload = {
            'subject': subject,
            'body': body,
            'priority': priority.value,
            'timestamp': kwargs.get('timestamp'),
            **metadata
        }
        
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        # Send to all configured webhook endpoints
        urls = self.config['urls']
        results = await asyncio.gather(*[
            self._send_to_endpoint(url, payload) 
            for url in urls
        ], return_exceptions=True)
        
        # If at least one succeeded, consider it a success
        successful = any(result is True for result in results)
        if not successful:
            logger.error(f"Failed to send to any webhook endpoint. Results: {results}")
            
        return successful
    
    async def _send_to_endpoint(self, url: str, payload: Dict[str, Any]) -> bool:
        """
        Send notification to a specific webhook endpoint with retry logic.
        
        Args:
            url: The webhook URL
            payload: The data payload to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        method = self.config['method']
        content_type = self.config['content_type']
        timeout = self.config['timeout']
        retry_count = self.config['retry_count']
        retry_delay = self.config['retry_delay']
        custom_headers = self.config['headers']
        
        headers = {
            'Content-Type': content_type,
            **custom_headers
        }
        
        # Prepare request kwargs based on content type
        request_kwargs = {
            'headers': headers,
            'timeout': aiohttp.ClientTimeout(total=timeout)
        }
        
        if content_type == 'application/json':
            request_kwargs['json'] = payload
        elif content_type == 'application/x-www-form-urlencoded':
            request_kwargs['data'] = payload
        else:
            # Default to sending as JSON in the body
            request_kwargs['data'] = json.dumps(payload)
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(retry_count):
                try:
                    async with getattr(session, method.lower())(url, **request_kwargs) as response:
                        if 200 <= response.status < 300:
                            logger.info(f"Successfully sent webhook to {url}")
                            return True
                        else:
                            response_text = await response.text()
                            logger.warning(
                                f"Webhook request failed with status {response.status}: {response_text}"
                            )
                            
                            # Don't retry for certain status codes
                            if response.status in (400, 401, 403, 404, 422):
                                logger.error(f"Non-retriable HTTP error {response.status} for {url}")
                                return False
                                
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Webhook request failed (attempt {attempt+1}/{retry_count}): {e}")
                
                if attempt < retry_count - 1:
                    # Wait before retrying, with exponential backoff
                    backoff_time = retry_delay * (2 ** attempt)
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Failed to send webhook to {url} after {retry_count} attempts")
                    return False
            
        # Should never reach here, but just in case
        return False