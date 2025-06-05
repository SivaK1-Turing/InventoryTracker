def forecast_depletion(product_id, window_days=30, smoothing_factor=0.3):
    """
    Forecast the number of days until a product's inventory is depleted
    based on recent usage patterns.
    
    Uses exponential smoothing to handle erratic transactions and provides
    fallback mechanisms for products with zero or minimal transactions.
    
    Args:
        product_id (str): The ID of the product to analyze
        window_days (int, optional): The number of days of history to analyze. Defaults to 30.
        smoothing_factor (float, optional): Alpha value for exponential smoothing. Defaults to 0.3.
        
    Returns:
        float: Estimated number of days until inventory depletion
               -1 if the product has no usage data
               0 if the product is already out of stock
               float('inf') if usage rate is zero (no depletion expected)
    """
    # Get current stock level
    conn = database.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT current_stock FROM inventory
        WHERE product_id = ?
    """, (product_id,))
    
    result = cursor.fetchone()
    if not result:
        conn.close()
        raise ValueError(f"Product with ID {product_id} not found.")
        
    current_stock = result[0]
    
    # If already out of stock, return 0
    if current_stock <= 0:
        conn.close()
        return 0
        
    # Retrieve daily consumption for the past window_days
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=window_days-1)
    
    cursor.execute("""
        SELECT DATE(timestamp) as event_date, SUM(quantity) as total_quantity 
        FROM inventory_events 
        WHERE product_id = ? AND event_type = 'STOCK_OUT' 
        AND DATE(timestamp) BETWEEN ? AND ?
        GROUP BY DATE(timestamp)
        ORDER BY event_date ASC
    """, (product_id, start_date, end_date))
    
    daily_usage = cursor.fetchall()
    conn.close()
    
    # Handle case with no usage data
    if not daily_usage:
        # Try to use category average if available
        category_avg = _get_category_average_usage(product_id)
        if category_avg > 0:
            return current_stock / category_avg
        else:
            # No data to make prediction
            return -1
    
    # Extract quantities and handle erratic data
    quantities = [row[1] for row in daily_usage]
    
    # Apply statistical analysis to handle erratic data
    avg_usage = _calculate_smoothed_usage_rate(quantities, smoothing_factor)
    
    # If usage rate is zero (or negative due to returns), no depletion expected
    if avg_usage <= 0:
        return float('inf')
    
    # Calculate and return days until depletion
    return current_stock / avg_usage

def _calculate_smoothed_usage_rate(quantities, alpha=0.3, remove_outliers=True):
    """
    Calculate a smoothed average usage rate, handling erratic data.
    
    Args:
        quantities (list): List of daily usage quantities
        alpha (float): Smoothing factor for exponential smoothing
        remove_outliers (bool): Whether to remove statistical outliers
        
    Returns:
        float: Smoothed average daily usage rate
    """
    if not quantities:
        return 0
        
    # Remove outliers if requested and if we have enough data
    if remove_outliers and len(quantities) >= 5:
        quantities = _remove_statistical_outliers(quantities)
        
    if not quantities:  # If all values were outliers
        return 0
    
    # For very few data points, simple average may be better
    if len(quantities) < 3:
        return sum(quantities) / len(quantities)
    
    # Apply exponential smoothing to handle erratic patterns
    smoothed = quantities[0]
    for qty in quantities[1:]:
        smoothed = alpha * qty + (1 - alpha) * smoothed
    
    return max(0, smoothed)  # Ensure non-negative rate

def _remove_statistical_outliers(quantities, threshold=2.0):
    """
    Remove statistical outliers from the dataset.
    
    Args:
        quantities (list): List of values
        threshold (float): Z-score threshold for outlier detection
        
    Returns:
        list: Filtered list with outliers removed
    """
    if len(quantities) < 3:  # Need at least 3 values for meaningful statistics
        return quantities
        
    # Calculate mean and standard deviation
    mean = sum(quantities) / len(quantities)
    std_dev = (sum((x - mean) ** 2 for x in quantities) / len(quantities)) ** 0.5
    
    if std_dev < 0.0001:  # All values nearly identical
        return quantities
    
    # Filter out values more than threshold standard deviations from mean
    filtered = [x for x in quantities if abs(x - mean) <= threshold * std_dev]
    
    # Ensure we don't filter out all values
    return filtered if filtered else quantities

def _get_category_average_usage(product_id):
    """
    Get the average usage rate for products in the same category.
    
    Args:
        product_id (str): The product ID
        
    Returns:
        float: Average usage rate for the product category, or 0 if unavailable
    """
    conn = database.get_connection()
    cursor = conn.cursor()
    
    # First get the product's category
    cursor.execute("""
        SELECT category FROM products WHERE id = ?
    """, (product_id,))
    
    result = cursor.fetchone()
    if not result:
        conn.close()
        return 0
        
    category = result[0]
    
    # Get average usage for this category over the last 30 days
    cursor.execute("""
        SELECT AVG(daily_usage) FROM (
            SELECT p.id, SUM(e.quantity) / COUNT(DISTINCT DATE(e.timestamp)) as daily_usage
            FROM products p
            JOIN inventory_events e ON p.id = e.product_id
            WHERE p.category = ? 
              AND e.event_type = 'STOCK_OUT'
              AND e.timestamp >= datetime('now', '-30 days')
            GROUP BY p.id
        )
    """, (category,))
    
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result and result[0] else 0

def get_depletion_risk(product_id, reorder_days=5):
    """
    Calculate the risk of stock depletion for a product.
    
    Args:
        product_id (str): Product ID to evaluate
        reorder_days (int): Threshold number of days for high risk
        
    Returns:
        dict: Risk assessment including:
            - days_remaining: estimated days until depletion
            - risk_level: 'high', 'medium', 'low', or 'unknown'
            - confidence_score: 0.0-1.0 indicating prediction confidence
            - recommendation: Action suggestion based on risk level
    """
    days_remaining = forecast_depletion(product_id)
    
    # Get product details for recommendations
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, reorder_point, reorder_quantity 
        FROM products WHERE id = ?
    """, (product_id,))
    product = cursor.fetchone()
    conn.close()
    
    if not product:
        return {
            'days_remaining': days_remaining,
            'risk_level': 'unknown',
            'confidence_score': 0.0,
            'recommendation': 'Product not found in database'
        }
    
    name, reorder_point, reorder_quantity = product
    
    # Calculate confidence score based on transaction history
    confidence_score = _calculate_confidence_score(product_id)
    
    # Determine risk level and recommendation
    if days_remaining == -1:
        risk_level = 'unknown'
        recommendation = f"Insufficient data to forecast depletion for {name}"
    elif days_remaining == 0:
        risk_level = 'high'
        recommendation = f"URGENT: {name} is out of stock. Reorder {reorder_quantity} units immediately."
    elif days_remaining <= reorder_days:
        risk_level = 'high'
        recommendation = f"Order {reorder_quantity} units of {name} immediately. Depletion expected in {days_remaining:.1f} days."
    elif days_remaining <= reorder_days * 2:
        risk_level = 'medium'
        recommendation = f"Plan to reorder {name} soon. Estimated {days_remaining:.1f} days of stock remaining."
    elif days_remaining == float('inf'):
        risk_level = 'low'
        recommendation = f"No recent usage detected for {name}. Consider reviewing inventory strategy."
    else:
        risk_level = 'low'
        recommendation = f"Stock level for {name} is adequate. Estimated {days_remaining:.1f} days remaining."
    
    return {
        'days_remaining': days_remaining,
        'risk_level': risk_level,
        'confidence_score': confidence_score,
        'recommendation': recommendation
    }

def _calculate_confidence_score(product_id, days=90):
    """
    Calculate a confidence score (0-1) for the depletion forecast.
    Higher scores indicate more consistent usage patterns.
    
    Args:
        product_id (str): Product ID to evaluate
        days (int): Number of days of history to consider
        
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    conn = database.get_connection()
    cursor = conn.cursor()
    
    # Get daily transaction counts
    cursor.execute("""
        SELECT COUNT(DISTINCT DATE(timestamp)) as days_with_transactions
        FROM inventory_events
        WHERE product_id = ? 
          AND event_type = 'STOCK_OUT'
          AND timestamp >= datetime('now', ?||' days')
    """, (product_id, -days))
    
    days_with_transactions = cursor.fetchone()[0]
    
    # Calculate transaction consistency as % of days with activity
    transaction_consistency = days_with_transactions / days if days > 0 else 0
    
    # Get coefficient of variation to measure consistency in quantity
    cursor.execute("""
        SELECT AVG(daily_qty), STDEV(daily_qty)
        FROM (
            SELECT DATE(timestamp) as day, SUM(quantity) as daily_qty
            FROM inventory_events
            WHERE product_id = ?
              AND event_type = 'STOCK_OUT'
              AND timestamp >= datetime('now', ?||' days')
            GROUP BY DATE(timestamp)
        )
    """, (product_id, -days))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result or not result[0]:
        return 0.1  # Very low confidence with no data
    
    mean, std_dev = result
    
    # Coefficient of variation (lower is more consistent)
    cv = std_dev / mean if mean > 0 else float('inf')
    
    # Convert CV to a consistency score (1 when CV is 0, approaching 0 as CV grows)
    quantity_consistency = 1 / (1 + cv) if cv != float('inf') else 0
    
    # Combine metrics with weights
    confidence_score = (0.6 * transaction_consistency) + (0.4 * quantity_consistency)
    
    # Ensure within bounds
    return max(0.0, min(1.0, confidence_score))