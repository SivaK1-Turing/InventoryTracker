"""
Analytics service for InventoryTracker with caching support.
Provides product analytics with automatic caching and transaction-based invalidation.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .analytics_cache import analytics_cache
from .transaction_hooks import register_cache_invalidation_hooks
from .transaction_event_handler import transaction_handler

logger = logging.getLogger("analytics_service")

class AnalyticsService:
    """
    Service for product analytics with built-in caching.
    """
    
    def __init__(self, transaction_manager):
        """
        Initialize the analytics service.
        
        Args:
            transaction_manager: The transaction manager to hook into for cache invalidation
        """
        self.logger = logging.getLogger("analytics_service")
        self.logger.info("Initializing AnalyticsService with caching")
        
        # Set up cache invalidation hooks
        self._setup_cache_hooks(transaction_manager)
        
    def _setup_cache_hooks(self, transaction_manager) -> None:
        """
        Set up cache invalidation hooks with the transaction manager.
        
        Args:
            transaction_manager: The transaction manager to connect to
        """
        self.logger.info("Setting up analytics cache invalidation hooks")
        success = register_cache_invalidation_hooks(transaction_manager, transaction_handler)
        
        if success:
            self.logger.info("Successfully registered cache invalidation hooks")
        else:
            self.logger.warning("Failed to register cache invalidation hooks - cache consistency may be affected")
    
    @analytics_cache.cache
    def get_product_sales_analysis(self, product_id: str, start_date: Optional[datetime] = None, 
                                  end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get sales analysis for a product with caching.
        Results will be cached for 10 minutes per product_id.
        
        Args:
            product_id: ID of the product to analyze
            start_date: Optional start date for the analysis period
            end_date: Optional end date for the analysis period
            
        Returns:
            Dictionary with sales analysis results
        """
        self.logger.info(f"Computing sales analysis for product {product_id}")
        
        # In a real implementation, you would query the database and do calculations here
        # For this example, we'll just simulate a time-consuming operation
        time.sleep(0.5)
        
        # Example analysis result
        result = {
            "product_id": product_id,
            "period": {
                "start": start_date.isoformat() if start_date else "all_time",
                "end": end_date.isoformat() if end_date else "now",
            },
            "metrics": {
                "total_sales": 1250,
                "total_units": 125,
                "average_price": 10.00,
                "revenue_trend": [120, 130, 125, 140, 150, 160, 180, 200, 190, 205],
            },
            "computed_at": datetime.now().isoformat(),
            "cached": False  # Will be overwritten on retrieval from cache
        }
        
        return result
    
    @analytics_cache.cache
    def get_product_inventory_history(self, product_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get inventory history for a product with caching.
        Results will be cached for 10 minutes per product_id.
        
        Args:
            product_id: ID of the product to analyze
            limit: Maximum number of history records to return
            
        Returns:
            List of inventory history records
        """
        self.logger.info(f"Computing inventory history for product {product_id}")
        
        # Simulate complex database query
        time.sleep(0.3)
        
        # Example history data (would come from database in real implementation)
        result = [
            {
                "timestamp": (datetime.now().timestamp() - (i * 3600)),
                "quantity": 100 - i,
                "type": "adjustment" if i % 3 == 0 else "sale",
                "reference": f"TX{1000 + i}"
            }
            for i in range(min(limit, 20))
        ]
        
        return result
    
    @analytics_cache.cache
    def get_product_performance_metrics(self, product_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a product with caching.
        Results will be cached for 10 minutes per product_id.
        
        Args:
            product_id: ID of the product to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        self.logger.info(f"Computing performance metrics for product {product_id}")
        
        # Simulate complex calculation
        time.sleep(0.7)
        
        # Example performance metrics (would be calculated from data in real implementation)
        result = {
            "product_id": product_id,
            "turnover_rate": 4.2,
            "days_in_stock": 86,
            "reorder_frequency": 45,
            "stockout_risk": "low",
            "optimal_stock_level": 150,
            "current_stock": 125,
            "computed_at": datetime.now().isoformat(),
            "cached": False  # Will be overwritten on retrieval from cache
        }
        
        return result

    def clear_product_cache(self, product_id: str) -> int:
        """
        Manually clear the cache for a specific product.
        
        Args:
            product_id: ID of the product to clear cache for
            
        Returns:
            Number of cache entries cleared
        """
        return analytics_cache.invalidate_for_product(product_id)
        
    def clear_all_cache(self) -> int:
        """
        Manually clear all cached analytics data.
        
        Returns:
            Number of cache entries cleared
        """
        return analytics_cache.invalidate_all()