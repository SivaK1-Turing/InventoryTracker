# plugins/turnover_ratio_plugin.py

"""
Turnover Ratio Plugin for InventoryTracker.

This plugin adds a custom metric that calculates the turnover ratio for products.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

from analytics_plugins import MetricCalculator, MetricValueType, register_metric
from database_service import get_db_service

logger = logging.getLogger("turnover_ratio_plugin")

@register_metric
class TurnoverRatioCalculator(MetricCalculator):
    """Calculates inventory turnover ratio for a product."""

    @property
    def metric_id(self) -> str:
        return "turnover_ratio"
    
    @property
    def display_name(self) -> str:
        return "Inventory Turnover Ratio"
    
    @property
    def description(self) -> str:
        return "Measures how many times inventory is sold and replaced in a given period"
    
    @property
    def value_type(self) -> MetricValueType:
        return MetricValueType.RATIO
    
    @property
    def category(self) -> str:
        return "Financial"
    
    @property
    def parameters(self) -> Dict[str, Tuple[type, Any]]:
        return {
            "months": (int, 6),  # Default to 6 months
        }
    
    @property
    def priority(self) -> int:
        return 50  # Higher priority than default custom metrics
    
    async def calculate(self, product_id: str, **kwargs) -> float:
        """
        Calculate inventory turnover ratio.
        
        Formula: Cost of Goods Sold / Average Inventory Value
        
        Args:
            product_id: Product ID to calculate for
            months: Number of months to analyze (default: 6)
            
        Returns:
            Turnover ratio as a float
        """
        months = kwargs.get('months', 6)
        
        # Get database connection
        db = get_db_service()
        
        # Calculate time range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months*30)
        
        # Get total quantity sold in period
        result = await db.execute(
            """
            SELECT SUM(quantity) 
            FROM inventory_events 
            WHERE product_id = ? 
            AND event_type = 'USAGE'
            AND timestamp BETWEEN ? AND ?
            """,
            (product_id, start_date.isoformat(), end_date.isoformat())
        )
        row = await result.fetchone()
        quantity_sold = row[0] if row and row[0] else 0
        
        # Get current stock
        result = await db.execute(
            "SELECT current_stock FROM inventory WHERE product_id = ?",
            (product_id,)
        )
        row = await result.fetchone()
        current_stock = row[0] if row else 0
        
        # Get average unit cost (from restock events)
        result = await db.execute(
            """
            SELECT AVG(CAST(cost AS REAL) / quantity) 
            FROM inventory_events 
            WHERE product_id = ? 
            AND event_type = 'RESTOCK' 
            AND cost IS NOT NULL AND quantity > 0
            AND timestamp BETWEEN ? AND ?
            """,
            (product_id, start_date.isoformat(), end_date.isoformat())
        )
        row = await result.fetchone()
        avg_unit_cost = row[0] if row and row[0] else 0
        
        # Calculate cost of goods sold
        cogs = quantity_sold * avg_unit_cost
        
        # Get beginning inventory (estimate from current + usage - restocks)
        result = await db.execute(
            """
            SELECT SUM(CASE WHEN event_type = 'RESTOCK' THEN quantity 
                           WHEN event_type = 'USAGE' THEN -quantity 
                           ELSE 0 END)
            FROM inventory_events 
            WHERE product_id = ? 
            AND timestamp BETWEEN ? AND ?
            """,
            (product_id, start_date.isoformat(), end_date.isoformat())
        )
        row = await result.fetchone()
        net_change = row[0] if row and row[0] else 0
        
        beginning_inventory = current_stock - net_change
        if beginning_inventory < 0:
            beginning_inventory = 0  # Avoid negative inventory
            
        # Calculate average inventory value
        avg_inventory = (beginning_inventory + current_stock) / 2
        avg_inventory_value = avg_inventory * avg_unit_cost
        
        # Avoid division by zero
        if avg_inventory_value <= 0:
            return 0.0
            
        # Calculate turnover ratio
        turnover_ratio = cogs / avg_inventory_value
        
        return turnover_ratio