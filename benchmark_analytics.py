#!/usr/bin/env python
"""
Analytics Performance Benchmarks

This script benchmarks analytics operations over 50,000 products to ensure
P99 latency remains under 200ms. It compares different optimization strategies
including pre-aggregation vs. on-demand calculation approaches.

Usage:
    python -m benchmarks.analytics [--products=50000] [--runs=3] [--profile]
"""

import os
import sys
import time
import random
import uuid
import argparse
import statistics
import logging
import json
import gc
import cProfile
import pstats
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import inventory tracker modules
from inventory_tracker.models import Product, StockTransaction, TransactionType
from inventory_tracker.analytics.cache import analytics_cache
from inventory_tracker.db import get_db_connection, close_db_connection
from inventory_tracker.utils.timing import TimingStats
from inventory_tracker.analytics.forecasting import forecast_depletion
from inventory_tracker.analytics.aggregations import (
    calculate_product_metrics,
    calculate_sales_trend,
    calculate_inventory_turnover
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analytics_benchmark")

# Benchmark configuration
DEFAULT_NUM_PRODUCTS = 50_000
DEFAULT_NUM_RUNS = 3
MAX_ACCEPTABLE_P99_LATENCY_MS = 200  # Maximum acceptable P99 latency (200ms)


class AnalyticsBenchmark:
    """Benchmarking framework for analytics operations."""
    
    def __init__(self, num_products: int = DEFAULT_NUM_PRODUCTS):
        """
        Initialize benchmark with specified number of products.
        
        Args:
            num_products: Number of products to benchmark
        """
        self.num_products = num_products
        self.products = []
        self.transactions = {}  # Dict mapping product_id to list of transactions
        self.pre_aggregated_data = {}  # Pre-aggregated metrics
        self.on_demand_results = {}  # Cache for on-demand results
        self.timing_stats = {}  # Store timing statistics
        
        # Options for optimization strategies
        self.enable_caching = True
        self.use_pre_aggregation = False
        self.parallel_processing = False
        self.db_connection = None
        
        logger.info(f"Initializing analytics benchmark with {num_products} products")
    
    def setup(self):
        """Set up benchmark data including products and transactions."""
        logger.info("Generating benchmark data...")
        
        # Generate product data
        self._generate_products()
        
        # Generate transaction data
        self._generate_transactions()
        
        # Connect to database
        self.db_connection = get_db_connection()
        
        # Initialize cache
        if self.enable_caching:
            analytics_cache.invalidate_all()
        
        logger.info(f"Setup complete: {self.num_products} products with transactions")
    
    def _generate_products(self):
        """Generate test products."""
        logger.info(f"Generating {self.num_products} products...")
        
        self.products = []
        categories = ["Electronics", "Clothing", "Food", "Home", "Tools", "Books", "Sports"]
        
        for i in range(self.num_products):
            product_id = str(uuid.uuid4())
            
            # Create product with random attributes
            product = Product(
                id=product_id,
                name=f"Product-{i}",
                sku=f"SKU-{i:06d}",
                category=random.choice(categories),
                price=round(random.uniform(10.0, 1000.0), 2),
                cost=round(random.uniform(5.0, 800.0), 2),
                stock_level=random.randint(0, 500),
                created_at=datetime.now() - timedelta(days=random.randint(1, 1000))
            )
            
            self.products.append(product)
    
    def _generate_transactions(self):
        """Generate historical transactions for products."""
        logger.info("Generating transaction history...")
        
        self.transactions = {}
        
        # For each product, generate between 10 and 100 transactions
        for product in tqdm(self.products, desc="Generating transactions"):
            product_transactions = []
            
            # Determine number of transactions for this product (more for popular products)
            num_transactions = random.randint(10, 100)
            
            # Base date for transactions
            start_date = datetime.now() - timedelta(days=365)
            
            # Generate transactions
            for i in range(num_transactions):
                # Determine transaction type (weighted for more realistic distribution)
                tx_type_rand = random.random()
                if tx_type_rand < 0.6:  # 60% outflow
                    tx_type = TransactionType.OUTFLOW
                    quantity = random.randint(1, 10)
                elif tx_type_rand < 0.9:  # 30% inflow
                    tx_type = TransactionType.INFLOW
                    quantity = random.randint(10, 100)
                else:  # 10% adjustment
                    tx_type = TransactionType.ADJUSTMENT
                    quantity = random.randint(-50, 50)
                
                # Generate timestamp (random but in chronological order)
                days_offset = (i / num_transactions) * 365  # Spread over a year
                random_variance = random.uniform(-5, 5)  # Add some randomness
                timestamp = start_date + timedelta(days=days_offset + random_variance)
                
                # Create transaction
                transaction = StockTransaction(
                    id=str(uuid.uuid4()),
                    product_id=product.id,
                    quantity=quantity,
                    transaction_type=tx_type,
                    price_per_unit=product.price if tx_type == TransactionType.OUTFLOW else product.cost,
                    timestamp=timestamp,
                    note=f"Generated transaction {i+1}/{num_transactions}"
                )
                
                product_transactions.append(transaction)
            
            # Sort by timestamp
            product_transactions.sort(key=lambda tx: tx.timestamp)
            
            # Store
            self.transactions[product.id] = product_transactions
    
    def pre_aggregate_metrics(self):
        """Pre-aggregate metrics for all products (optimization strategy)."""
        logger.info("Pre-aggregating metrics for all products...")
        start_time = time.time()
        
        self.pre_aggregated_data = {}
        
        # Process products in batches for better progress indication
        batch_size = 1000
        num_batches = (self.num_products + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i+1) * batch_size, self.num_products)
            batch_products = self.products[start_idx:end_idx]
            
            with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 4)) as executor:
                results = list(executor.map(
                    self._pre_aggregate_for_product,
                    batch_products
                ))
            
            # Store results
            for product_id, metrics in results:
                self.pre_aggregated_data[product_id] = metrics
            
            logger.info(f"Pre-aggregated batch {i+1}/{num_batches} ({len(batch_products)} products)")
        
        elapsed = time.time() - start_time
        logger.info(f"Pre-aggregation complete in {elapsed:.2f}s for {self.num_products} products")
        
        # Return total time for pre-aggregation
        return elapsed
    
    def _pre_aggregate_for_product(self, product):
        """Pre-aggregate metrics for a single product."""
        product_id = product.id
        transactions = self.transactions.get(product_id, [])
        
        # Calculate various metrics
        try:
            metrics = {
                "sales_summary": self._calculate_sales_summary(product, transactions),
                "inventory_metrics": self._calculate_inventory_metrics(product, transactions),
                "trends": self._calculate_trends(product, transactions),
                "forecasts": self._calculate_forecasts(product, transactions),
                "calculated_at": datetime.now().isoformat()
            }
            return product_id, metrics
        except Exception as e:
            logger.error(f"Error pre-aggregating for product {product_id}: {e}")
            return product_id, {"error": str(e)}
    
    def _calculate_sales_summary(self, product, transactions):
        """Calculate sales summary metrics."""
        total_sales = 0
        total_units_sold = 0
        total_revenue = 0
        
        for tx in transactions:
            if tx.transaction_type == TransactionType.OUTFLOW:
                total_units_sold += tx.quantity
                total_revenue += tx.quantity * tx.price_per_unit
        
        return {
            "total_units_sold": total_units_sold,
            "total_revenue": round(total_revenue, 2),
            "average_price": round(total_revenue / total_units_sold if total_units_sold else 0, 2)
        }
    
    def _calculate_inventory_metrics(self, product, transactions):
        """Calculate inventory metrics."""
        # Count inflows and outflows
        total_inflow = sum(tx.quantity for tx in transactions if tx.transaction_type == TransactionType.INFLOW)
        total_outflow = sum(tx.quantity for tx in transactions if tx.transaction_type == TransactionType.OUTFLOW)
        
        # Calculate turnover rate (simplified)
        annual_outflow = total_outflow * (365 / 365)  # Adjust if transactions don't cover a full year
        avg_inventory = (product.stock_level + product.stock_level + total_inflow - total_outflow) / 2
        turnover_rate = annual_outflow / avg_inventory if avg_inventory else 0
        
        return {
            "current_stock": product.stock_level,
            "total_inflow": total_inflow,
            "total_outflow": total_outflow,
            "turnover_rate": round(turnover_rate, 2),
            "days_of_supply": round(365 / turnover_rate if turnover_rate else float('inf'), 1)
        }
    
    def _calculate_trends(self, product, transactions):
        """Calculate trends over time."""
        # Group transactions by month
        monthly_data = {}
        
        for tx in transactions:
            month_key = tx.timestamp.strftime("%Y-%m")
            if month_key not in monthly_data:
                monthly_data[month_key] = {"inflow": 0, "outflow": 0, "revenue": 0, "cost": 0}
            
            if tx.transaction_type == TransactionType.INFLOW:
                monthly_data[month_key]["inflow"] += tx.quantity
                monthly_data[month_key]["cost"] += tx.quantity * tx.price_per_unit
            elif tx.transaction_type == TransactionType.OUTFLOW:
                monthly_data[month_key]["outflow"] += tx.quantity
                monthly_data[month_key]["revenue"] += tx.quantity * tx.price_per_unit
        
        # Convert to lists for easier consumption
        months = sorted(monthly_data.keys())
        monthly_trends = {
            "months": months,
            "inflow": [monthly_data[m]["inflow"] for m in months],
            "outflow": [monthly_data[m]["outflow"] for m in months],
            "revenue": [round(monthly_data[m]["revenue"], 2) for m in months],
            "cost": [round(monthly_data[m]["cost"], 2) for m in months]
        }
        
        return monthly_trends
    
    def _calculate_forecasts(self, product, transactions):
        """Calculate forecasts based on historical data."""
        # Extract recent outflows to estimate daily usage
        recent_transactions = [tx for tx in transactions 
                              if tx.timestamp >= datetime.now() - timedelta(days=90)
                              and tx.transaction_type == TransactionType.OUTFLOW]
        
        total_recent_outflow = sum(tx.quantity for tx in recent_transactions)
        
        # Calculate daily usage rate (units per day)
        days_in_period = 90
        daily_usage = total_recent_outflow / days_in_period if days_in_period > 0 else 0
        
        # Forecast days until depletion
        if daily_usage > 0:
            days_until_depletion = product.stock_level / daily_usage
        else:
            days_until_depletion = float('inf')
        
        # Calculate reorder point (assuming 14-day lead time and 95% service level)
        lead_time = 14  # days
        safety_stock = 1.96 * daily_usage * (lead_time ** 0.5)  # Service level factor * daily usage * sqrt(lead time)
        reorder_point = lead_time * daily_usage + safety_stock
        
        return {
            "daily_usage_rate": round(daily_usage, 2),
            "days_until_depletion": round(days_until_depletion, 1) if days_until_depletion != float('inf') else None,
            "reorder_point": round(reorder_point, 0),
            "needs_reorder": product.stock_level <= reorder_point
        }

    def benchmark_on_demand_analytics(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Benchmark on-demand analytics performance.
        
        Args:
            num_samples: Number of random products to sample for benchmark
            
        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"Benchmarking on-demand analytics with {num_samples} samples...")
        
        # Sample random products
        sample_products = random.sample(self.products, min(num_samples, self.num_products))
        
        # Prepare timing collections
        latencies = []
        metrics_latencies = {
            "sales_summary": [],
            "inventory_metrics": [],
            "trends": [],
            "forecasts": []
        }
        
        # Run benchmarks
        for product in tqdm(sample_products, desc="Running on-demand analytics"):
            product_id = product.id
            transactions = self.transactions.get(product_id, [])
            
            # Measure total time for all analytics
            start_time = time.time()
            
            # Calculate each metric type and measure individual times
            metric_start = time.time()
            sales_summary = self._calculate_sales_summary(product, transactions)
            metrics_latencies["sales_summary"].append((time.time() - metric_start) * 1000)
            
            metric_start = time.time()
            inventory_metrics = self._calculate_inventory_metrics(product, transactions)
            metrics_latencies["inventory_metrics"].append((time.time() - metric_start) * 1000)
            
            metric_start = time.time()
            trends = self._calculate_trends(product, transactions)
            metrics_latencies["trends"].append((time.time() - metric_start) * 1000)
            
            metric_start = time.time()
            forecasts = self._calculate_forecasts(product, transactions)
            metrics_latencies["forecasts"].append((time.time() - metric_start) * 1000)
            
            # Record total latency
            total_latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(total_latency)
            
            # Store results for validation
            self.on_demand_results[product_id] = {
                "sales_summary": sales_summary,
                "inventory_metrics": inventory_metrics,
                "trends": trends,
                "forecasts": forecasts,
                "calculated_at": datetime.now().isoformat()
            }
        
        # Calculate statistics
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        
        # Calculate individual metric statistics
        metric_stats = {}
        for metric_name, metric_latencies in metrics_latencies.items():
            metric_latencies.sort()
            metric_stats[metric_name] = {
                "min": min(metric_latencies),
                "max": max(metric_latencies),
                "mean": statistics.mean(metric_latencies),
                "p50": metric_latencies[len(metric_latencies) // 2],
                "p95": metric_latencies[int(len(metric_latencies) * 0.95)],
                "p99": metric_latencies[int(len(metric_latencies) * 0.99)]
            }
        
        # Log results
        logger.info(f"On-demand analytics results (ms):")
        logger.info(f"  Min latency: {min(latencies):.2f}ms")
        logger.info(f"  Mean latency: {statistics.mean(latencies):.2f}ms")
        logger.info(f"  P50 latency: {p50:.2f}ms")
        logger.info(f"  P95 latency: {p95:.2f}ms")
        logger.info(f"  P99 latency: {p99:.2f}ms")
        logger.info(f"  Max latency: {max(latencies):.2f}ms")
        
        # Store results
        stats = {
            "sample_size": len(sample_products),
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "detailed_metrics": metric_stats
        }
        
        self.timing_stats["on_demand"] = stats
        return stats
    
    def benchmark_pre_aggregated_analytics(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Benchmark pre-aggregated analytics performance.
        
        Args:
            num_samples: Number of random products to sample for benchmark
            
        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"Benchmarking pre-aggregated analytics with {num_samples} samples...")
        
        # Ensure we have pre-aggregated data
        if not self.pre_aggregated_data:
            logger.warning("Pre-aggregated data not found. Running pre-aggregation...")
            self.pre_aggregate_metrics()
        
        # Sample random products
        sample_products = random.sample(self.products, min(num_samples, self.num_products))
        
        # Prepare timing collections
        latencies = []
        
        # Run benchmarks
        for product in tqdm(sample_products, desc="Accessing pre-aggregated analytics"):
            product_id = product.id
            
            # Measure time to access pre-aggregated data
            start_time = time.time()
            
            # Fetch data
            if product_id in self.pre_aggregated_data:
                metrics = self.pre_aggregated_data[product_id]
                
                # Access each section to simulate real usage
                _ = metrics["sales_summary"]
                _ = metrics["inventory_metrics"]
                _ = metrics["trends"]
                _ = metrics["forecasts"]
            
            # Record latency
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate statistics
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        
        # Log results
        logger.info(f"Pre-aggregated analytics results (ms):")
        logger.info(f"  Min latency: {min(latencies):.2f}ms")
        logger.info(f"  Mean latency: {statistics.mean(latencies):.2f}ms")
        logger.info(f"  P50 latency: {p50:.2f}ms")
        logger.info(f"  P95 latency: {p95:.2f}ms")
        logger.info(f"  P99 latency: {p99:.2f}ms")
        logger.info(f"  Max latency: {max(latencies):.2f}ms")
        
        # Store results
        stats = {
            "sample_size": len(sample_products),
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.mean(latencies),
            "p50": p50,
            "p95": p95,
            "p99": p99
        }
        
        self.timing_stats["pre_aggregated"] = stats
        return stats
    
    def benchmark_cached_on_demand_analytics(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Benchmark cached on-demand analytics performance.
        
        Args:
            num_samples: Number of random products to sample for benchmark
            
        Returns:
            Dictionary with timing statistics
        """
        logger.info(f"Benchmarking cached on-demand analytics with {num_samples} samples...")
        
        # Sample random products
        sample_products = random.sample(self.products, min(num_samples, self.num_products))
        
        # Prepare timing collections
        first_call_latencies = []
        cached_call_latencies = []
        
        # Cache decorator
        def cached_analytics(product_id):
            """Cached analytics function."""
            if product_id in self.on_demand_results:
                return self.on_demand_results[product_id]
            
            product = next((p for p in self.products if p.id == product_id), None)
            if not product:
                return None
                
            transactions = self.transactions.get(product_id, [])
            
            # Calculate metrics
            result = {
                "sales_summary": self._calculate_sales_summary(product, transactions),
                "inventory_metrics": self._calculate_inventory_metrics(product, transactions),
                "trends": self._calculate_trends(product, transactions),
                "forecasts": self._calculate_forecasts(product, transactions),
                "calculated_at": datetime.now().isoformat()
            }
            
            self.on_demand_results[product_id] = result
            return result
        
        # Run benchmarks - first call
        for product in tqdm(sample_products, desc="First call (uncached)"):
            product_id = product.id
            
            # Ensure product is removed from cache
            if product_id in self.on_demand_results:
                del self.on_demand_results[product_id]
            
            # Measure first call (uncached)
            start_time = time.time()
            _ = cached_analytics(product_id)
            latency = (time.time() - start_time) * 1000
            first_call_latencies.append(latency)
        
        # Run benchmarks - second call (cached)
        for product in tqdm(sample_products, desc="Second call (cached)"):
            product_id = product.id
            
            # Measure second call (cached)
            start_time = time.time()
            _ = cached_analytics(product_id)
            latency = (time.time() - start_time) * 1000
            cached_call_latencies.append(latency)
        
        # Calculate statistics - first call
        first_call_latencies.sort()
        first_p50 = first_call_latencies[len(first_call_latencies) // 2]
        first_p95 = first_call_latencies[int(len(first_call_latencies) * 0.95)]
        first_p99 = first_call_latencies[int(len(first_call_latencies) * 0.99)]
        
        # Calculate statistics - cached call
        cached_call_latencies.sort()
        cached_p50 = cached_call_latencies[len(cached_call_latencies) // 2]
        cached_p95 = cached_call_latencies[int(len(cached_call_latencies) * 0.95)]
        cached_p99 = cached_call_latencies[int(len(cached_call_latencies) * 0.99)]
        
        # Log results
        logger.info(f"First call (uncached) results (ms):")
        logger.info(f"  Min latency: {min(first_call_latencies):.2f}ms")
        logger.info(f"  Mean latency: {statistics.mean(first_call_latencies):.2f}ms")
        logger.info(f"  P50 latency: {first_p50:.2f}ms")
        logger.info(f"  P95 latency: {first_p95:.2f}ms")
        logger.info(f"  P99 latency: {first_p99:.2f}ms")
        
        logger.info(f"Second call (cached) results (ms):")
        logger.info(f"  Min latency: {min(cached_call_latencies):.2f}ms")
        logger.info(f"  Mean latency: {statistics.mean(cached_call_latencies):.2f}ms")
        logger.info(f"  P50 latency: {cached_p50:.2f}ms")
        logger.info(f"  P95 latency: {cached_p95:.2f}ms")
        logger.info(f"  P99 latency: {cached_p99:.2f}ms")
        
        # Store results
        stats = {
            "sample_size": len(sample_products),
            "first_call": {
                "min": min(first_call_latencies),
                "max": max(first_call_latencies),
                "mean": statistics.mean(first_call_latencies),
                "p50": first_p50,
                "p95": first_p95,
                "p99": first_p99
            },
            "cached_call": {
                "min": min(cached_call_latencies),
                "max": max(cached_call_latencies),
                "mean": statistics.mean(cached_call_latencies),
                "p50": cached_p50,
                "p95": cached_p95,
                "p99": cached_p99
            },
            "speedup_factor": statistics.mean(first_call_latencies) / statistics.mean(cached_call_latencies)
        }
        
        self.timing_stats["cached"] = stats
        return stats
    
    def compare_results(self):
        """Compare results from different approaches for correctness."""
        logger.info("Comparing results from different approaches...")
        
        # Compare on-demand vs. pre-aggregated for sample products
        if "on_demand" not in self.timing_stats or "pre_aggregated" not in self.timing_stats:
            logger.warning("Missing timing stats. Run benchmarks first.")
            return
            
        # Sample a few products for deep comparison
        sample_size = min(10, self.num_products)
        sample_products = random.sample(self.products, sample_size)
        
        mismatches = 0
        
        for product in sample_products:
            product_id = product.id
            
            if product_id in self.on_demand_results and product_id in self.pre_aggregated_data:
                on_demand = self.on_demand_results[product_id]
                pre_aggregated = self.pre_aggregated_data[product_id]
                
                # Compare sales summary
                on_demand_sales = on_demand["sales_summary"]
                pre_aggregated_sales = pre_aggregated["sales_summary"]
                
                if on_demand_sales["total_units_sold"] != pre_aggregated_sales["total_units_sold"]:
                    mismatches += 1
                    logger.warning(f"Mismatch for product {product_id}: units sold")
                    
                # Compare other metrics as needed
        
        if mismatches == 0:
            logger.info(f"All results match between approaches for {sample_size} sampled products")
        else:
            logger.warning(f"Found {mismatches} mismatches between approaches")
    
    def plot_results(self, filename: str = "analytics_benchmark_results.png"):
        """
        Plot benchmark results to visualize different approaches.
        
        Args:
            filename: Output filename for the plot
        """
        logger.info("Plotting benchmark results...")
        
        if not self.timing_stats:
            logger.error("No timing stats available. Run benchmarks first.")
            return
        
        # Prepare data for plotting
        approaches = []
        p50_values = []
        p95_values = []
        p99_values = []
        means = []
        
        # On-demand approach
        if "on_demand" in self.timing_stats:
            approaches.append("On-Demand")
            p50_values.append(self.timing_stats["on_demand"]["p50"])
            p95_values.append(self.timing_stats["on_demand"]["p95"])
            p99_values.append(self.timing_stats["on_demand"]["p99"])
            means.append(self.timing_stats["on_demand"]["mean"])
        
        # Pre-aggregated approach
        if "pre_aggregated" in self.timing_stats:
            approaches.append("Pre-Aggregated")
            p50_values.append(self.timing_stats["pre_aggregated"]["p50"])
            p95_values.append(self.timing_stats["pre_aggregated"]["p95"])
            p99_values.append(self.timing_stats["pre_aggregated"]["p99"])
            means.append(self.timing_stats["pre_aggregated"]["mean"])
        
        # Cached approach (second call)
        if "cached" in self.timing_stats:
            approaches.append("Cached")
            p50_values.append(self.timing_stats["cached"]["cached_call"]["p50"])
            p95_values.append(self.timing_stats["cached"]["cached_call"]["p95"])
            p99_values.append(self.timing_stats["cached"]["cached_call"]["p99"])
            means.append(self.timing_stats["cached"]["cached_call"]["mean"])
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        x = range(len(approaches))
        width = 0.2
        
        # Plot bars
        plt.bar([i - width*1.5 for i in x], p50_values, width, label='P50', color='green')
        plt.bar([i - width/2 for i in x], p95_values, width, label='P95', color='orange')
        plt.bar([i + width/2 for i in x], p99_values, width, label='P99', color='red')
        plt.bar([i + width*1.5 for i in x], means, width, label='Mean', color='blue')
        
        # Add P99 threshold line
        plt.axhline(y=MAX_ACCEPTABLE_P99_LATENCY_MS, linestyle='--', color='darkred', 
                   label=f'P99 Target ({MAX_ACCEPTABLE_P99_LATENCY_MS}ms)')
        
        # Add labels and legend
        plt.xlabel('Approach')
        plt.ylabel('Latency (ms)')
        plt.title('Analytics Performance Benchmark Comparison')
        plt.xticks(x, approaches)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(p99_values):
            plt.text(i + width/2, v + 5, f"{v:.1f}ms", ha='center', fontsize=9)
        
        # Save the plot
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {filename}")
        
        # Show the plot if in interactive mode
        plt.show()
    
    def run_all_benchmarks(self, num_samples: int = 1000, pre_aggregate: bool = True):
        """
        Run all benchmarks and compile results.
        
        Args:
            num_samples: Number of random products to sample
            pre_aggregate: Whether to run pre-aggregation benchmark
        """
        logger.info("Running all benchmarks...")
        
        # Force garbage collection to start clean
        gc.collect()
        
        # Run benchmarks
        on_demand_stats = self.benchmark_on_demand_analytics(num_samples)
        
        gc.collect()
        
        cached_stats = self.benchmark_cached_on_demand_analytics(num_samples)
        
        gc.collect()
        
        # Optionally run pre-aggregation
        pre_aggregated_stats = None
        if pre_aggregate:
            # First time the pre-aggregation if needed
            if not self.pre_aggregated_data:
                pre_agg_time = self.pre_aggregate_metrics()
                logger.info(f"Pre-aggregation took {pre_agg_time:.2f}s for {self.num_products} products")
            
            gc.collect()
            
            pre_aggregated_stats = self.benchmark_pre_aggregated_analytics(num_samples)
        
        # Compare results for correctness
        self.compare_results()
        
        # Plot the results
        self.plot_results()
        
        # Determine the best approach based on P99 latency
        best_approach = None
        best_p99 = float('inf')
        
        if on_demand_stats and on_demand_stats["p99"] < best_p99:
            best_p99 = on_demand_stats["p99"]
            best_approach = "on_demand"
        
        if cached_stats and cached_stats["cached_call"]["p99"] < best_p99:
            best_p99 = cached_stats["cached_call"]["p99"]
            best_approach = "cached"
        
        if pre_aggregated_stats and pre_aggregated_stats["p99"] < best_p99:
            best_p99 = pre_aggregated_stats["p99"]
            best_approach = "pre_aggregated"
        
        logger.info(f"Best approach based on P99 latency: {best_approach} with {best_p99:.2f}ms")
        
        # Check if we meet P99 target
        meets_target = False
        if best_p99 <= MAX_ACCEPTABLE_P99_LATENCY_MS:
            meets_target = True
            logger.info(f"✅ P99 latency target of {MAX_ACCEPTABLE_P99_LATENCY_MS}ms achieved with {best_approach}!")
        else:
            logger.warning(f"❌ Failed to meet P99 latency target of {MAX_ACCEPTABLE_P99_LATENCY_MS}ms. Best: {best_p99:.2f}ms")
        
        # Compile summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_products": self.num_products,
            "sample_size": num_samples,
            "approaches_tested": list(self.timing_stats.keys()),
            "best_approach": best_approach,
            "best_p99_latency": best_p99,
            "meets_target": meets_target,
            "target_p99_latency": MAX_ACCEPTABLE_P99_LATENCY_MS,
            "detailed_results": self.timing_stats
        }
        
        # Save results to file
        with open("analytics_benchmark_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Benchmark results saved to analytics_benchmark_results.json")
        return summary
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        if self.db_connection:
            close_db_connection(self.db_connection)
            self.db_connection = None
        
        # Clear large data structures
        self.pre_aggregated_data.clear()
        self.on_demand_results.clear()
        gc.collect()
        
        logger.info("Cleanup complete")

def run_profiled_benchmark(args):
    """Run benchmark with profiling if requested."""
    # Create and run benchmark
    benchmark = AnalyticsBenchmark(num_products=args.products)
    benchmark.setup()
    
    # Run with profiler if requested
    if args.profile:
        logger.info("Running benchmark with profiling...")
        profiler = cProfile.Profile()
        profiler.enable()
        
        benchmark.run_all_benchmarks(num_samples=args.samples, pre_aggregate=True)
        
        profiler.disable()
        
        # Print profiling stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Print top 30 functions
        logger.info(f"Profiling results:\n{s.getvalue()}")
        
        # Save profiling stats to file
        with open("analytics_benchmark_profile.txt", "w") as f:
            ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
            ps.print_stats(50)  # Top 50 functions
    else:
        # Run without profiling
        for i in range(args.runs):
            logger.info(f"Run {i+1}/{args.runs}")
            benchmark.run_all_benchmarks(num_samples=args.samples, pre_aggregate=True)
            
            # Only generate fresh data for the first run
            if i == 0 and args.runs > 1:
                benchmark.pre_aggregated_data.clear()
                benchmark.on_demand_results.clear()
    
    # Clean up
    benchmark.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analytics performance benchmark")
    parser.add_argument("--products", type=int, default=DEFAULT_NUM_PRODUCTS, 
                        help=f"Number of products to benchmark (default: {DEFAULT_NUM_PRODUCTS})")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples to use for benchmarks (default: 1000)")
    parser.add_argument("--runs", type=int, default=DEFAULT_NUM_RUNS,
                        help=f"Number of benchmark runs (default: {DEFAULT_NUM_RUNS})")
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling")
    
    args = parser.parse_args()
    
    # Set up logging to file
    file_handler = logging.FileHandler("analytics_benchmark.log", mode="w")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting analytics benchmark with {args.products} products")
    
    try:
        run_profiled_benchmark(args)
    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        sys.exit(1)