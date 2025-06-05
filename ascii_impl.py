def generate_sparkline(data, width=14, empty_char="⋅", filled_chars="▁▂▃▄▅▆▇█"):
    """
    Generate an ASCII sparkline from a list of numeric data.
    
    Args:
        data (list): List of numeric values
        width (int): Desired width of the sparkline
        empty_char (str): Character to use for empty/zero values
        filled_chars (str): String of characters to use for increasing values
        
    Returns:
        str: ASCII sparkline representing the data trend
    """
    # Handle empty data
    if not data or all(x is None or x == 0 for x in data):
        return empty_char * width
    
    # Filter out None values and get min/max
    valid_data = [x for x in data if x is not None and x >= 0]
    if not valid_data:
        return empty_char * width
    
    # Scale the data to fit our character set
    min_val = min(valid_data)
    max_val = max(valid_data)
    
    # If all values are the same, use a mid-level character
    if min_val == max_val:
        mid_char = filled_chars[len(filled_chars) // 2]
        return mid_char * min(width, len(data))
    
    # Scale between 0 and len(filled_chars)-1
    char_range = len(filled_chars) - 1
    
    # Ensure we don't exceed width
    data_to_use = data[-width:] if len(data) > width else data
    
    # Build the sparkline
    result = []
    for value in data_to_use:
        if value is None or value < 0:
            result.append(empty_char)
        elif value == 0:
            result.append(empty_char)
        else:
            # Scale and find the right character
            normalized = (value - min_val) / (max_val - min_val)
            char_index = int(normalized * char_range)
            result.append(filled_chars[char_index])
    
    # Pad to width if needed
    padding = width - len(result)
    if padding > 0:
        result = [empty_char] * padding + result
    
    return ''.join(result)


def get_usage_history(product_id, days=14):
    """
    Get daily usage history for a product over the specified number of days.
    
    Args:
        product_id (str): The ID of the product
        days (int): Number of days of history to retrieve
        
    Returns:
        list: List of daily usage values (positive numbers)
    """
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        
        conn = database.get_connection()
        cursor = conn.cursor()
        
        # Query for all negative inventory changes (usage)
        cursor.execute("""
            SELECT DATE(timestamp) as event_date, SUM(ABS(quantity_change)) as usage 
            FROM inventory_transactions 
            WHERE product_id = ? 
            AND DATE(timestamp) BETWEEN ? AND ?
            AND quantity_change < 0  -- Only consider usage (negative changes)
            GROUP BY DATE(timestamp)
            ORDER BY event_date ASC
        """, (product_id, start_date, end_date))
        
        usage_data = cursor.fetchall()
        conn.close()
        
        # Create a dictionary of dates to usage values
        usage_by_date = {row[0]: row[1] for row in usage_data}
        
        # Fill in all dates in the range
        daily_usage = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            usage = usage_by_date.get(date_str, 0)
            daily_usage.append(usage)
            current_date += timedelta(days=1)
        
        return daily_usage
        
    except Exception as e:
        logger.error(f"Error retrieving usage history for product {product_id}: {e}")
        return [0] * days