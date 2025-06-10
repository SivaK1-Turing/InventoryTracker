# inventory_tracker/plugins/analytics/turnover_ratio_plugin.py

from datetime import datetime, timedelta
from typing import Dict, Any, List

from inventory_tracker.analytics.plugins import AnalyticsPlugin

class TurnoverRatioPlugin(AnalyticsPlugin):
    """Plugin to calculate inventory turnover ratio metrics"""
    
    name = "turnover_ratio"
    description = "Calculates inventory turnover ratio and related metrics"
    version = "1.0.0"
    author = "InventoryTracker Team"
    priority = 50
    
    @classmethod
    def get_metrics(cls, product, transactions, **kwargs) -> Dict[str, Any]:
        """
        Calculate turnover ratio metrics.
        
        Args:
            product: The product to analyze
            transactions: List of transactions for the product
            
        Returns:
            Dictionary of turnover metrics
        """
        # Extract timeframe parameter with default of last 365 days
        days = kwargs.get('days', 365)
        
        # Filter transactions to the specified timeframe
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_transactions = [tx for tx in transactions if tx.timestamp >= cutoff_date]
        
        # Calculate metrics
        cogs = cls._calculate_cogs(product, recent_transactions)
        avg_inventory = cls._calculate_avg_inventory(product, recent_transactions)
        
        # Calculate turnover ratio
        turnover_ratio = cogs / avg_inventory if avg_inventory else 0
        
        # Calculate days in inventory (DOI)
        days_in_inventory = days / turnover_ratio if turnover_ratio else float('inf')
        
        return {
            "turnover_ratio": round(turnover_ratio, 2),
            "days_in_inventory": round(days_in_inventory, 1) if days_in_inventory != float('inf') else None,
            "cost_of_goods_sold": round(cogs, 2),
            "average_inventory_value": round(avg_inventory, 2),
            "timeframe_days": days,
            "is_healthy": turnover_ratio >= 4.0,  # Industry benchmark
            "benchmark_comparison": round(turnover_ratio / 4.0, 2) if turnover_ratio else 0
        }
    
    @classmethod
    def _calculate_cogs(cls, product, transactions) -> float:
        """Calculate Cost of Goods Sold (COGS)"""
        # Sum of the cost of all units sold
        cogs = 0
        
        for tx in transactions:
            if tx.transaction_type == 'outflow':
                # Cost per unit * quantity
                cogs += product.cost * tx.quantity
        
        return cogs
    
    @classmethod
    def _calculate_avg_inventory(cls, product, transactions) -> float:
        """Calculate average inventory value"""
        # Simplified method: current inventory value
        return product.stock_level * product.cost