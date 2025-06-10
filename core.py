# inventory_tracker/analytics/core.py

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from inventory_tracker.models import Product, StockTransaction
from inventory_tracker.analytics.plugins import plugin_manager, integrate_plugin_metrics

logger = logging.getLogger("analytics.core")

def calculate_product_analytics(
    product: Product, 
    transactions: List[StockTransaction], 
    include_plugins: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Calculate comprehensive analytics for a product.
    
    Args:
        product: The product to analyze
        transactions: List of transactions for the product
        include_plugins: Whether to include metrics from plugins
        kwargs: Additional parameters for the calculation
        
    Returns:
        Dictionary of analytics results
    """
    # Calculate core metrics
    result = {
        "product_id": product.id,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "sales": calculate_sales_metrics(product, transactions, **kwargs),
            "inventory": calculate_inventory_metrics(product, transactions, **kwargs),
            "forecast": forecast_depletion_metrics(product, transactions, **kwargs),
            "trends": calculate_trend_metrics(product, transactions, **kwargs)
        }
    }
    
    # Add plugin metrics if requested
    if include_plugins:
        result = integrate_plugin_metrics(result, product, transactions, **kwargs)
    
    return result

def calculate_sales_metrics(product: Product, transactions: List[StockTransaction], **kwargs) -> Dict[str, Any]:
    """Calculate sales-related metrics"""
    # Extract outflow transactions
    outflow_txs = [tx for tx in transactions if tx.transaction_type == 'outflow']
    
    total_units_sold = sum(tx.quantity for tx in outflow_txs)
    total_revenue = sum(tx.quantity * tx.price_per_unit for tx in outflow_txs)
    avg_price = total_revenue / total_units_sold if total_units_sold else 0
    
    # Get recent sales (last 30 days)
    thirty_days_ago = datetime.now() - timedelta(days=30)
    recent_outflows = [tx for tx in outflow_txs if tx.timestamp >= thirty_days_ago]
    recent_units_sold = sum(tx.quantity for tx in recent_outflows)
    recent_revenue = sum(tx.quantity * tx.price_per_unit for tx in recent_outflows)
    
    return {
        "total_units_sold": total_units_sold,
        "total_revenue": round(total_revenue, 2),
        "average_price": round(avg_price, 2),
        "recent_units_sold": recent_units_sold,
        "recent_revenue": round(recent_revenue, 2),
        "profit_margin": round((avg_price - product.cost) / avg_price * 100, 1) if avg_price else 0
    }

def calculate_inventory_metrics(product: Product, transactions: List[StockTransaction], **kwargs) -> Dict[str, Any]:
    """Calculate inventory-related metrics"""
    current_stock = product.stock_level
    stock_value = current_stock * product.cost
    
    # Count inflows and outflows
    inflow_txs = [tx for tx in transactions if tx.transaction_type == 'inflow']
    outflow_txs = [tx for tx in transactions if tx.transaction_type == 'outflow']
    
    total_inflow = sum(tx.quantity for tx in inflow_txs)
    total_outflow = sum(tx.quantity for tx in outflow_txs)
    
    return {
        "current_stock": current_stock,
        "stock_value": round(stock_value, 2),
        "total_received": total_inflow,
        "total_sold": total_outflow,
        "net_change": total_inflow - total_outflow
    }

def forecast_depletion_metrics(product: Product, transactions: List[StockTransaction], **kwargs) -> Dict[str, Any]:
    """Calculate forecast and depletion metrics"""
    # Get daily usage rate (default or provided)
    daily_usage_rate = kwargs.get('daily_usage_rate')
    
    if daily_usage_rate is None:
        # Calculate from recent transactions (last 90 days)
        ninety_days_ago = datetime.now() - timedelta(days=90)
        recent_outflows = [
            tx for tx in transactions 
            if tx.transaction_type == 'outflow' and tx.timestamp >= ninety_days_ago
        ]
        
        total_recent_outflow = sum(tx.quantity for tx in recent_outflows)
        daily_usage_rate = total_recent_outflow / 90 if recent_outflows else 0
    
    # Calculate days until depletion
    if daily_usage_rate > 0:
        days_until_depletion = product.stock_level / daily_usage_rate
    else:
        days_until_depletion = None  # Can't forecast depletion if no usage
    
    # Calculate reorder point (using safety factor from kwargs or default)
    safety_factor = kwargs.get('safety_factor', 1.5)
    lead_time_days = kwargs.get('lead_time_days', 14)
    
    reorder_point = (daily_usage_rate * lead_time_days) + (safety_factor * daily_usage_rate * (lead_time_days ** 0.5))
    
    return {
        "daily_usage_rate": round(daily_usage_rate, 2),
        "days_until_depletion": round(days_until_depletion, 1) if days_until_depletion else None,
        "needs_reorder": product.stock_level <= reorder_point,
        "reorder_point": round(reorder_point, 0),
        "lead_time_days": lead_time_days,
        "depletion_date": (
            (datetime.now() + timedelta(days=days_until_depletion)).isoformat()
            if days_until_depletion 
            else None
        )
    }

def calculate_trend_metrics(product: Product, transactions: List[StockTransaction], **kwargs) -> Dict[str, Any]:
    """Calculate trend-related metrics"""
    # Group transactions by month
    monthly_data = {}
    
    for tx in transactions:
        month_key = tx.timestamp.strftime("%Y-%m")
        
        if month_key not in monthly_data:
            monthly_data[month_key] = {"inflow": 0, "outflow": 0, "revenue": 0}
        
        if tx.transaction_type == 'inflow':
            monthly_data[month_key]["inflow"] += tx.quantity
        elif tx.transaction_type == 'outflow':
            monthly_data[month_key]["outflow"] += tx.quantity
            monthly_data[month_key]["revenue"] += tx.quantity * tx.price_per_unit
    
    # Sort months and extract data
    months = sorted(monthly_data.keys())
    
    # Build trend data
    trend_data = {
        "periods": months,
        "inflow": [monthly_data[m]["inflow"] for m in months],
        "outflow": [monthly_data[m]["outflow"] for m in months],
        "revenue": [monthly_data[m]["revenue"] for m in months]
    }
    
    # Calculate growth metrics (if we have enough data)
    growth = {}
    
    if len(months) >= 2:
        last_month = months[-1]
        previous_month = months[-2]
        
        # Calculate month-over-month growth
        outflow_growth = (
            (monthly_data[last_month]["outflow"] - monthly_data[previous_month]["outflow"]) / 
            monthly_data[previous_month]["outflow"] * 100
            if monthly_data[previous_month]["outflow"] else 0
        )
        
        revenue_growth = (
            (monthly_data[last_month]["revenue"] - monthly_data[previous_month]["revenue"]) / 
            monthly_data[previous_month]["revenue"] * 100
            if monthly_data[previous_month]["revenue"] else 0
        )
        
        growth = {
            "outflow_mom_pct": round(outflow_growth, 1),
            "revenue_mom_pct": round(revenue_growth, 1)
        }
    
    return {
        "trends": trend_data,
        "growth": growth,
        "num_periods": len(months)
    }

# Initialize plugins when module is imported
from inventory_tracker.analytics.plugins import initialize_plugins
initialize_plugins()