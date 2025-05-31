"""
search.py - Fluent interface for searching data in the inventory tracker.

This module provides a flexible search API that can be used with both SQLAlchemy 
and in-memory collections. The SearchQuery class follows the builder pattern, allowing
chained method calls to construct complex queries.

Example usage:
    # Search for hardware widgets with transactions since Jan 1, 2025
    results = SearchQuery().product('widget').tag('hardware').from_date('2025-01-01').execute()
"""
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid
from dateutil.parser import parse as parse_date
import re

try:
    from sqlalchemy import and_, or_, Column, String, DateTime, cast, func, Integer
    from sqlalchemy.orm import Query as SAQuery
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Type for filter predicates that work on in-memory objects
PredicateType = Callable[[Any], bool]

# Type for query filters that work with SQLAlchemy
if SQLALCHEMY_AVAILABLE:
    FilterType = Union[PredicateType, Any]  # The Any is for SQLAlchemy filter expressions
else:
    FilterType = PredicateType

class SearchQuery:
    """
    A fluent interface for building search queries on inventory data.
    
    This class provides methods that can be chained together to build complex
    search criteria. The execute() method applies these criteria against either
    SQLAlchemy queries or in-memory collections.
    """
    
    def __init__(self, backend=None):
        """
        Initialize a new search query.
        
        Args:
            backend: Optional backend to use for the search. If None, will use the
                    default backend configured in the application.
        """
        from inventorytracker.store import get_store
        
        self.store = backend or get_store()
        self.filters: List[FilterType] = []
        self.product_filters: List[FilterType] = []
        self.transaction_filters: List[FilterType] = []
        self._order_by = []
        self._limit = None
        self._offset = 0
        self._include_products = True
        self._include_transactions = False
        
    def product(self, name_or_sku: str) -> 'SearchQuery':
        """
        Filter by product name or SKU.
        
        Args:
            name_or_sku: Product name or SKU to search for (partial matches supported)
            
        Returns:
            The SearchQuery instance for method chaining
        """
        pattern = f'.*{re.escape(name_or_sku)}.*'
        regex = re.compile(pattern, re.IGNORECASE)
        
        # In-memory predicate
        def product_name_or_sku_filter(item):
            if hasattr(item, 'name') and hasattr(item, 'sku'):  # Product object
                return (regex.match(item.name) is not None or 
                        regex.match(item.sku) is not None)
            elif hasattr(item, 'product'):  # Transaction with product reference
                product = item.product
                return (regex.match(product.name) is not None or 
                        regex.match(product.sku) is not None)
            return False
        
        # Add in-memory filter
        self.product_filters.append(product_name_or_sku_filter)
        
        # Add SQLAlchemy filter if available
        if SQLALCHEMY_AVAILABLE:
            from inventorytracker.models.product import Product
            
            # This will be applied differently depending on what we're querying
            sa_product_filter = or_(
                Product.name.ilike(f'%{name_or_sku}%'),
                Product.sku.ilike(f'%{name_or_sku}%')
            )
            self.product_filters.append(sa_product_filter)
            
        return self
    
    def tag(self, tag: str) -> 'SearchQuery':
        """
        Filter by product tag.
        
        Args:
            tag: Tag to filter products by
            
        Returns:
            The SearchQuery instance for method chaining
        """
        # In-memory predicate
        def tag_filter(item):
            if hasattr(item, 'tags'):
                return tag.lower() in [t.lower() for t in item.tags]
            elif hasattr(item, 'product') and hasattr(item.product, 'tags'):
                return tag.lower() in [t.lower() for t in item.product.tags]
            return False
        
        # Add in-memory filter
        self.product_filters.append(tag_filter)
        
        # Add SQLAlchemy filter if available
        if SQLALCHEMY_AVAILABLE:
            from inventorytracker.models.product import Product, ProductTag
            
            sa_tag_filter = Product.tags.any(ProductTag.name.ilike(tag))
            self.product_filters.append(sa_tag_filter)
            
        return self
    
    def from_date(self, date_str: str) -> 'SearchQuery':
        """
        Filter transactions by date, including only those after the given date.
        
        Args:
            date_str: Date string in ISO format or human-readable format
            
        Returns:
            The SearchQuery instance for method chaining
        """
        # Parse the date string to a datetime object
        start_date = parse_date(date_str)
        
        # In-memory predicate
        def date_filter(item):
            if hasattr(item, 'timestamp'):
                return item.timestamp >= start_date
            return True  # No timestamp means we can't filter, so include it
        
        # Add in-memory filter
        self.transaction_filters.append(date_filter)
        
        # Add SQLAlchemy filter if available
        if SQLALCHEMY_AVAILABLE:
            from inventorytracker.models.inventory import StockTransaction
            
            sa_date_filter = StockTransaction.timestamp >= start_date
            self.transaction_filters.append(sa_date_filter)
            
        # Since we're filtering by date, include transactions in results
        self._include_transactions = True
            
        return self
    
    def to_date(self, date_str: str) -> 'SearchQuery':
        """
        Filter transactions by date, including only those before the given date.
        
        Args:
            date_str: Date string in ISO format or human-readable format
            
        Returns:
            The SearchQuery instance for method chaining
        """
        # Parse the date string to a datetime object
        end_date = parse_date(date_str)
        
        # In-memory predicate
        def date_filter(item):
            if hasattr(item, 'timestamp'):
                return item.timestamp <= end_date
            return True  # No timestamp means we can't filter, so include it
        
        # Add in-memory filter
        self.transaction_filters.append(date_filter)
        
        # Add SQLAlchemy filter if available
        if SQLALCHEMY_AVAILABLE:
            from inventorytracker.models.inventory import StockTransaction
            
            sa_date_filter = StockTransaction.timestamp <= end_date
            self.transaction_filters.append(sa_date_filter)
            
        # Since we're filtering by date, include transactions in results
        self._include_transactions = True
            
        return self
    
    def min_stock(self, quantity: int) -> 'SearchQuery':
        """
        Filter products by current stock level, including only those with at least
        the given quantity in stock.
        
        Args:
            quantity: Minimum stock level
            
        Returns:
            The SearchQuery instance for method chaining
        """
        # In-memory predicate
        def min_stock_filter(item):
            if hasattr(item, 'current_stock'):
                return item.current_stock >= quantity
            return True  # No stock attribute means we can't filter, so include it
        
        # Add in-memory filter
        self.product_filters.append(min_stock_filter)
        
        # Add SQLAlchemy filter if available
        if SQLALCHEMY_AVAILABLE:
            from inventorytracker.models.product import Product
            
            sa_stock_filter = Product.current_stock >= quantity
            self.product_filters.append(sa_stock_filter)
            
        return self
    
    def max_stock(self, quantity: int) -> 'SearchQuery':
        """
        Filter products by current stock level, including only those with at most
        the given quantity in stock.
        
        Args:
            quantity: Maximum stock level
            
        Returns:
            The SearchQuery instance for method chaining
        """
        # In-memory predicate
        def max_stock_filter(item):
            if hasattr(item, 'current_stock'):
                return item.current_stock <= quantity
            return True  # No stock attribute means we can't filter, so include it
        
        # Add in-memory filter
        self.product_filters.append(max_stock_filter)
        
        # Add SQLAlchemy filter if available
        if SQLALCHEMY_AVAILABLE:
            from inventorytracker.models.product import Product
            
            sa_stock_filter = Product.current_stock <= quantity
            self.product_filters.append(sa_stock_filter)
            
        return self
    
    def below_reorder(self, margin: int = 0) -> 'SearchQuery':
        """
        Filter products that are at or below their reorder level.
        
        Args:
            margin: Optional margin to add to the reorder level. For example,
                  a margin of 5 would include products with stock <= reorder_level + 5
            
        Returns:
            The SearchQuery instance for method chaining
        """
        # In-memory predicate
        def below_reorder_filter(item):
            if hasattr(item, 'current_stock') and hasattr(item, 'reorder_level'):
                return item.current_stock <= (item.reorder_level + margin)
            return False
        
        # Add in-memory filter
        self.product_filters.append(below_reorder_filter)
        
        # Add SQLAlchemy filter if available
        if SQLALCHEMY_AVAILABLE:
            from inventorytracker.models.product import Product
            
            sa_reorder_filter = Product.current_stock <= (Product.reorder_level + margin)
            self.product_filters.append(sa_reorder_filter)
            
        return self
    
    def include_transactions(self, include: bool = True) -> 'SearchQuery':
        """
        Specify whether to include transactions in the search results.
        
        Args:
            include: Whether to include transactions (default True)
            
        Returns:
            The SearchQuery instance for method chaining
        """
        self._include_transactions = include
        return self
    
    def include_products(self, include: bool = True) -> 'SearchQuery':
        """
        Specify whether to include products in the search results.
        
        Args:
            include: Whether to include products (default True)
            
        Returns:
            The SearchQuery instance for method chaining
        """
        self._include_products = include
        return self
    
    def order_by(self, field: str, descending: bool = False) -> 'SearchQuery':
        """
        Specify ordering for the search results.
        
        Args:
            field: Field name to order by
            descending: Whether to sort in descending order
            
        Returns:
            The SearchQuery instance for method chaining
        """
        self._order_by.append((field, descending))
        return self
    
    def limit(self, limit: int) -> 'SearchQuery':
        """
        Limit the number of search results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            The SearchQuery instance for method chaining
        """
        self._limit = limit
        return self
    
    def offset(self, offset: int) -> 'SearchQuery':
        """
        Skip a number of search results.
        
        Args:
            offset: Number of results to skip
            
        Returns:
            The SearchQuery instance for method chaining
        """
        self._offset = offset
        return self
    
    def _apply_filters_sqlalchemy(self, base_query, model, filters):
        """Apply SQLAlchemy filters to a query."""
        if not filters:
            return base_query
            
        sa_filters = [f for f in filters if not callable(f)]
        if sa_filters:
            base_query = base_query.filter(and_(*sa_filters))
            
        return base_query
    
    def _apply_filters_memory(self, items, filters):
        """Apply in-memory filters to a list of items."""
        if not filters:
            return items
            
        mem_filters = [f for f in filters if callable(f)]
        result = items
        
        for filter_func in mem_filters:
            result = [item for item in result if filter_func(item)]
            
        return result
    
    def _execute_sqlalchemy(self):
        """Execute the search using SQLAlchemy."""
        results = {'products': [], 'transactions': []}
        
        if SQLALCHEMY_AVAILABLE:
            from inventorytracker.models.product import Product
            from inventorytracker.models.inventory import StockTransaction
            
            # Get database session from store
            session = self.store.get_session()
            
            # Query products if requested
            if self._include_products:
                product_query = session.query(Product)
                product_query = self._apply_filters_sqlalchemy(product_query, Product, self.product_filters)
                
                # Apply ordering
                for field_name, desc in self._order_by:
                    if hasattr(Product, field_name):
                        field = getattr(Product, field_name)
                        if desc:
                            field = field.desc()
                        product_query = product_query.order_by(field)
                
                # Apply pagination
                if self._offset:
                    product_query = product_query.offset(self._offset)
                if self._limit:
                    product_query = product_query.limit(self._limit)
                
                results['products'] = product_query.all()
            
            # Query transactions if requested
            if self._include_transactions:
                txn_query = session.query(StockTransaction)
                txn_query = self._apply_filters_sqlalchemy(txn_query, StockTransaction, self.transaction_filters)
                
                # Join with products and apply product filters if needed
                if self.product_filters:
                    txn_query = txn_query.join(Product)
                    txn_query = self._apply_filters_sqlalchemy(txn_query, Product, self.product_filters)
                    
                # Apply ordering
                for field_name, desc in self._order_by:
                    if hasattr(StockTransaction, field_name):
                        field = getattr(StockTransaction, field_name)
                        if desc:
                            field = field.desc()
                        txn_query = txn_query.order_by(field)
                
                # Apply pagination
                if self._offset:
                    txn_query = txn_query.offset(self._offset)
                if self._limit:
                    txn_query = txn_query.limit(self._limit)
                    
                results['transactions'] = txn_query.all()
                
        return results
    
    def _execute_memory(self):
        """Execute the search using in-memory filtering."""
        results = {'products': [], 'transactions': []}
        
        # Query products if requested
        if self._include_products:
            products = self.store.get_all_products()
            filtered_products = self._apply_filters_memory(products, self.product_filters)
            
            # Apply ordering
            for field_name, desc in reversed(self._order_by):
                filtered_products.sort(
                    key=lambda x: getattr(x, field_name, 0) if hasattr(x, field_name) else 0, 
                    reverse=desc
                )
            
            # Apply pagination
            start_idx = self._offset
            end_idx = start_idx + self._limit if self._limit else None
            results['products'] = filtered_products[start_idx:end_idx]
        
        # Query transactions if requested
        if self._include_transactions:
            transactions = self.store.get_all_transactions()
            
            # Apply transaction filters
            filtered_txns = self._apply_filters_memory(transactions, self.transaction_filters)
            
            # Apply product filters to transactions
            if self.product_filters:
                filtered_txns = self._apply_filters_memory(filtered_txns, self.product_filters)
                
            # Apply ordering
            for field_name, desc in reversed(self._order_by):
                filtered_txns.sort(
                    key=lambda x: getattr(x, field_name, 0) if hasattr(x, field_name) else 0, 
                    reverse=desc
                )
            
            # Apply pagination
            start_idx = self._offset
            end_idx = start_idx + self._limit if self._limit else None
            results['transactions'] = filtered_txns[start_idx:end_idx]
            
        return results
    
    def execute(self):
        """
        Execute the search and return the results.
        
        The search is performed against the configured backend (SQLAlchemy or in-memory).
        
        Returns:
            Dict with 'products' and 'transactions' keys, each containing a list of matching items.
        """
        if SQLALCHEMY_AVAILABLE and hasattr(self.store, 'get_session'):
            return self._execute_sqlalchemy()
        else:
            return self._execute_memory()
    
    def count(self):
        """
        Count the number of items that match the search criteria without fetching them.
        
        Returns:
            Dict with 'products' and 'transactions' keys, each containing the count.
        """
        if SQLALCHEMY_AVAILABLE and hasattr(self.store, 'get_session'):
            from inventorytracker.models.product import Product
            from inventorytracker.models.inventory import StockTransaction
            
            counts = {'products': 0, 'transactions': 0}
            session = self.store.get_session()
            
            if self._include_products:
                product_query = session.query(func.count(Product.id))
                product_query = self._apply_filters_sqlalchemy(product_query, Product, self.product_filters)
                counts['products'] = product_query.scalar() or 0
                
            if self._include_transactions:
                txn_query = session.query(func.count(StockTransaction.id))
                txn_query = self._apply_filters_sqlalchemy(txn_query, StockTransaction, self.transaction_filters)
                
                if self.product_filters:
                    txn_query = txn_query.join(Product)
                    txn_query = self._apply_filters_sqlalchemy(txn_query, Product, self.product_filters)
                    
                counts['transactions'] = txn_query.scalar() or 0
                
            return counts
        else:
            # For in-memory, we just execute the search and count the results
            results = self.execute()
            return {
                'products': len(results['products']), 
                'transactions': len(results['transactions'])
            }
            
def example_search():
    """Example demonstrating the search API usage."""
    # The exact query requested in the requirements:
    search = SearchQuery().product('widget').tag('hardware').from_date('2025-01-01')
    results = search.execute()
    
    # Print results
    print(f"Found {len(results['products'])} matching products")
    print(f"Found {len(results['transactions'])} matching transactions")
    
    # More complex example:
    complex_search = (
        SearchQuery()
        .product('widget')
        .tag('hardware')
        .from_date('2025-01-01')
        .to_date('2025-05-30')
        .min_stock(10)
        .order_by('name')
        .limit(50)
    )
    
    complex_results = complex_search.execute()
    
    # This would print details about the matching products
    print(f"\nComplex search found {len(complex_results['products'])} products "
          f"and {len(complex_results['transactions'])} transactions")
    
    return results
    
if __name__ == "__main__":
    example_search()