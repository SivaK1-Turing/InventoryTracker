#!/usr/bin/env python
"""
benchmarks/listing.py - Performance benchmarks for product listing operations

This utility measures performance metrics for product listing operations across
different query patterns, database backends, and filtering conditions. The focus
is on 95th percentile latency for realistic workloads.

Usage:
  python -m benchmarks.listing --backend sqlite --products 10000 --iterations 100
  python -m benchmarks.listing --help
"""

import argparse
import asyncio
import csv
import json
import numpy as np
import os
import random
import statistics
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, Tuple

# Add project root to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import project modules
from inventorytracker.search import SearchQuery
from inventorytracker.models.product import Product
import inventorytracker.store as store_module


# Configure console output
if RICH_AVAILABLE:
    console = Console()
else:
    import contextlib
    
    @contextlib.contextmanager
    def dummy_progress(*args, **kwargs):
        yield None
    
    class DummyConsole:
        def print(self, *args, **kwargs):
            print(*args)
            
        def rule(self, *args, **kwargs):
            print("-" * 80)
    
    console = DummyConsole()
    Progress = dummy_progress


class Benchmark:
    """Base class for listing performance benchmarks."""
    
    def __init__(
        self,
        backend_type: str = "sqlite",
        product_count: int = 10000,
        iterations: int = 100,
        warm_up_iterations: int = 10,
        reset_db: bool = True,
        output_file: Optional[str] = None,
        percentiles: List[int] = [50, 75, 90, 95, 99],
    ):
        """
        Initialize benchmark parameters.
        
        Args:
            backend_type: Storage backend to use ('sqlite', 'memory', 'json')
            product_count: Number of test products to generate
            iterations: Number of iterations for each test
            warm_up_iterations: Number of warm-up iterations to run (not measured)
            reset_db: Whether to reset the database before running
            output_file: Path to save benchmark results
            percentiles: List of percentiles to calculate (default: 50, 75, 90, 95, 99)
        """
        self.backend_type = backend_type
        self.product_count = product_count
        self.iterations = iterations
        self.warm_up_iterations = warm_up_iterations
        self.reset_db = reset_db
        self.output_file = output_file
        self.percentiles = percentiles
        
        # Set up test data storage
        self.db_path = Path(f"./benchmark_data/{backend_type}_benchmark.db")
        self.data_dir = Path("./benchmark_data")
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Store for benchmark results
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        
    def setup(self):
        """Set up benchmark environment and test data."""
        console.rule("[bold]Setting up benchmark environment")
        self._setup_store()
        
        if self.reset_db:
            self._reset_database()
        
        # Check if we need to generate test data
        product_count = len(list(self.store.get_all_products()))
        
        if product_count < self.product_count:
            console.print(f"Generating {self.product_count - product_count} additional test products...")
            self._generate_test_data(self.product_count - product_count)
            
        console.print(f"[green]Setup complete. Database has {len(list(self.store.get_all_products()))} products.[/green]")
        
    def _setup_store(self):
        """Initialize the appropriate store based on backend type."""
        if self.backend_type == "sqlite":
            store_args = {"db_path": str(self.db_path)}
            os.environ["INVENTORY_TRACKER_DB_PATH"] = str(self.db_path)
        elif self.backend_type == "json":
            json_path = self.data_dir / "json_store"
            json_path.mkdir(exist_ok=True)
            store_args = {"base_path": str(json_path)}
            os.environ["INVENTORY_TRACKER_STORE_PATH"] = str(json_path)
        else:  # memory
            store_args = {}
            
        os.environ["INVENTORY_TRACKER_STORE_TYPE"] = self.backend_type
        self.store = store_module.get_store()
        
    def _reset_database(self):
        """Reset the database, removing all existing data."""
        if self.backend_type == "sqlite" and self.db_path.exists():
            self.db_path.unlink()
            self._setup_store()
        elif self.backend_type == "json":
            json_path = self.data_dir / "json_store"
            if json_path.exists():
                for file in json_path.glob("*.json"):
                    file.unlink()
            self._setup_store()
        elif self.backend_type == "memory":
            self._setup_store()
            
    def _generate_test_data(self, count: int):
        """
        Generate test product data.
        
        Args:
            count: Number of products to generate
        """
        # Product name components
        categories = ["Widget", "Gadget", "Tool", "Device", "Component", "Module", "System", "Unit"]
        adjectives = ["Smart", "Pro", "Advanced", "Basic", "Premium", "Ultra", "Compact", "Portable", 
                     "Digital", "Analog", "Wireless", "Solar", "Eco", "Heavy-Duty", "Industrial"]
        models = ["X", "S", "Pro", "Plus", "Lite", "Mini", "Max", "Elite", "Prime", "Core"]
        materials = ["Steel", "Aluminum", "Titanium", "Carbon", "Plastic", "Composite", "Wood", "Metal"]
        
        # For faster bulk insertion
        products_batch = []
        batch_size = 100  # Adjust based on memory constraints
        
        if RICH_AVAILABLE:
            progress_context = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
            )
        else:
            progress_context = dummy_progress()
        
        with progress_context as progress:
            task = progress.add_task(f"Generating {count} products...", total=count)
            
            for i in range(count):
                # Generate a product name
                category = random.choice(categories)
                adjective = random.choice(adjectives)
                model = random.choice(models)
                
                # Some products include material in name
                if random.random() < 0.3:
                    material = random.choice(materials)
                    name = f"{adjective} {material} {category} {model}"
                else:
                    name = f"{adjective} {category} {model}"
                
                # Sometimes add a number to the model
                if random.random() < 0.5:
                    name += f" {random.randint(1, 9)}{random.randint(0, 9)}{random.randint(0, 9)}"
                
                # Generate SKU
                sku = f"{category[:3]}-{model}-{random.randint(100, 999)}".upper()
                
                # Generate price
                price = round(random.uniform(9.99, 499.99), 2)
                
                # Generate stock levels
                current_stock = random.randint(0, 100)
                reorder_level = random.randint(5, 25)
                
                # Generate tags
                all_tags = ["electronics", "hardware", "tools", "industrial", 
                          "home", "office", "outdoor", "professional",
                          "safety", "maintenance", "battery-powered", "wired",
                          "measurement", "lighting", "storage", "communication"]
                tags = random.sample(all_tags, random.randint(1, 5))
                
                # Generate notes
                notes_options = [
                    f"Popular in {random.choice(['summer', 'winter', 'spring', 'fall'])}",
                    f"Manufactured in {random.choice(['USA', 'China', 'Germany', 'Japan', 'Korea'])}",
                    f"Customer favorite",
                    f"New model for {random.randint(2023, 2025)}",
                    f"Warranty: {random.choice(['1 year', '2 years', '3 years', '5 years', 'lifetime'])}",
                    f"Requires {random.choice(['batteries', 'power adapter', 'special maintenance', 'calibration'])}",
                    f"Compatible with {adjective} {category} accessories",
                ]
                notes = random.choice(notes_options)
                
                # Randomize creation date within the last year
                days_ago = random.randint(1, 365)
                created_at = datetime.now() - timedelta(days=days_ago)
                updated_at = created_at + timedelta(days=random.randint(0, days_ago))
                
                # Randomize archived status (10% chance of being archived)
                archived = random.random() < 0.1
                archived_at = None
                if archived:
                    # Archive date is between creation and now
                    days_since_creation = (datetime.now() - created_at).days
                    if days_since_creation > 0:
                        days_to_archive = random.randint(1, days_since_creation)
                        archived_at = created_at + timedelta(days=days_to_archive)
                
                # Create product
                product = Product(
                    id=uuid.uuid4(),
                    name=name,
                    sku=sku,
                    price=price,
                    current_stock=current_stock,
                    reorder_level=reorder_level,
                    tags=tags,
                    notes=notes,
                    created_at=created_at,
                    updated_at=updated_at,
                    archived=archived,
                    archived_at=archived_at,
                )
                
                # Add to batch
                products_batch.append(product)
                
                # Save batch if it reaches the batch size
                if len(products_batch) >= batch_size:
                    for p in products_batch:
                        self.store.save_product(p)
                    products_batch.clear()
                    
                # Update progress
                progress.update(task, advance=1)
            
            # Save any remaining products
            for p in products_batch:
                self.store.save_product(p)
                
        console.print(f"[green]Successfully generated {count} test products[/green]")
        
    def run_query_benchmark(self, name: str, query_function: Callable[[], Any]):
        """
        Run benchmark for a specific query function.
        
        Args:
            name: Name of the benchmark
            query_function: Function that executes the query to benchmark
        
        Returns:
            Dict with benchmark results
        """
        timings = []
        
        # Warm up
        console.print(f"Warming up ({self.warm_up_iterations} iterations)...", end="")
        for _ in range(self.warm_up_iterations):
            query_function()
        console.print("[green]done[/green]")
        
        # Run benchmark iterations
        if RICH_AVAILABLE:
            progress_context = Progress(
                TextColumn(f"[bold blue]Benchmarking [cyan]{name}"),
                BarColumn(),
                TaskProgressColumn(),
            )
        else:
            progress_context = dummy_progress()
            print(f"Benchmarking {name}...")
            
        with progress_context as progress:
            task = progress.add_task(f"Running {self.iterations} iterations", total=self.iterations)
            
            for i in range(self.iterations):
                start_time = time.perf_counter()
                result = query_function()
                end_time = time.perf_counter()
                
                elapsed_ms = (end_time - start_time) * 1000
                timings.append(elapsed_ms)
                
                # Track result size for reporting
                if i == 0:
                    if isinstance(result, dict) and "products" in result:
                        result_size = len(result["products"])
                    elif isinstance(result, list):
                        result_size = len(result)
                    else:
                        result_size = 1
                
                progress.update(task, advance=1)
        
        # Calculate statistics
        mean = statistics.mean(timings)
        median = statistics.median(timings)
        p95 = np.percentile(timings, 95)
        min_time = min(timings)
        max_time = max(timings)
        
        # Calculate all requested percentiles
        percentile_results = {}
        for p in self.percentiles:
            percentile_results[f"p{p}"] = np.percentile(timings, p)
        
        results = {
            "name": name,
            "result_size": result_size,
            "iterations": self.iterations,
            "min_ms": min_time,
            "max_ms": max_time,
            "mean_ms": mean,
            "median_ms": median,
            "p95_ms": p95,
            "percentiles": percentile_results,
            "raw_timings": timings
        }
        
        return results
        
    def run_all_benchmarks(self):
        """Run all defined benchmark scenarios."""
        console.rule("[bold]Running listing benchmarks")
        
        benchmarks = {}
        
        # 1. Basic listing (no filters)
        benchmarks["basic_listing"] = self.run_query_benchmark(
            "Basic listing (no filters)",
            lambda: SearchQuery().execute()["products"]
        )
        
        # 2. Filter by SKU pattern (prefix search)
        benchmarks["sku_prefix_search"] = self.run_query_benchmark(
            "SKU prefix search",
            lambda: SearchQuery().product("WID").execute()["products"]
        )
        
        # 3. Filter by text in product name
        benchmarks["name_search"] = self.run_query_benchmark(
            "Product name search",
            lambda: SearchQuery().product("Smart").execute()["products"]
        )
        
        # 4. Filter by specific tag
        benchmarks["tag_filter"] = self.run_query_benchmark(
            "Filter by tag",
            lambda: SearchQuery().tag("electronics").execute()["products"]
        )
        
        # 5. Filter by multiple tags
        def multi_tag_search():
            search = SearchQuery()
            search.tag("electronics").tag("professional")
            return search.execute()["products"]
            
        benchmarks["multi_tag_filter"] = self.run_query_benchmark(
            "Filter by multiple tags",
            multi_tag_search
        )
        
        # 6. Filter by stock level
        benchmarks["stock_filter"] = self.run_query_benchmark(
            "Filter by stock level",
            lambda: SearchQuery().min_stock(50).execute()["products"]
        )
        
        # 7. Complex filter (multiple conditions)
        def complex_filter():
            search = SearchQuery()
            search.product("Pro").tag("electronics").min_stock(10).max_stock(50)
            return search.execute()["products"]
            
        benchmarks["complex_filter"] = self.run_query_benchmark(
            "Complex filter (multiple conditions)",
            complex_filter
        )
        
        # 8. Sort by name
        benchmarks["sort_by_name"] = self.run_query_benchmark(
            "Sort by name",
            lambda: SearchQuery().order_by("name", False).execute()["products"]
        )
        
        # 9. Sort by price (descending)
        benchmarks["sort_by_price_desc"] = self.run_query_benchmark(
            "Sort by price (descending)",
            lambda: SearchQuery().order_by("price", True).execute()["products"]
        )
        
        # 10. Filtering archived products
        benchmarks["archived_filter"] = self.run_query_benchmark(
            "Show only archived products",
            lambda: SearchQuery().include_archived(True).only_archived().execute()["products"]
        )
        
        # 11. Pagination (first page)
        benchmarks["pagination_first_page"] = self.run_query_benchmark(
            "Pagination - first page",
            lambda: SearchQuery().limit(20).offset(0).execute()["products"]
        )
        
        # 12. Pagination (middle page)
        benchmarks["pagination_middle_page"] = self.run_query_benchmark(
            "Pagination - middle page",
            lambda: SearchQuery().limit(20).offset(500).execute()["products"]
        )
        
        # 13. Combined search, filter, sort, and pagination
        def combined_operations():
            search = SearchQuery()
            search.product("Tool").tag("professional").min_stock(5)
            search.order_by("price", True).limit(10).offset(5)
            return search.execute()["products"]
            
        benchmarks["combined_operations"] = self.run_query_benchmark(
            "Combined search, filter, sort, and pagination",
            combined_operations
        )
        
        # Store results
        self.results = benchmarks
        self._print_benchmark_results()
        
        if self.output_file:
            self._save_benchmark_results()
            
        return benchmarks
        
    def _print_benchmark_results(self):
        """Print benchmark results in a formatted table."""
        console.rule("[bold]Benchmark Results")
        
        if RICH_AVAILABLE:
            # Create rich table
            table = Table(title=f"Product Listing Performance ({self.backend_type} backend, {self.product_count} products)")
            
            table.add_column("Benchmark", style="cyan")
            table.add_column("Results", style="cyan", justify="right")
            table.add_column("Min (ms)", justify="right")
            table.add_column("Mean (ms)", justify="right")
            table.add_column("Median (ms)", justify="right")
            table.add_column("95th (ms)", style="bold red", justify="right")
            table.add_column("Max (ms)", justify="right")
            
            for name, result in self.results.items():
                table.add_row(
                    result["name"],
                    str(result["result_size"]),
                    f"{result['min_ms']:.2f}",
                    f"{result['mean_ms']:.2f}",
                    f"{result['median_ms']:.2f}",
                    f"{result['p95_ms']:.2f}",
                    f"{result['max_ms']:.2f}",
                )
                
            console.print(table)
            
        else:
            # Print simple table for non-rich environment
            print(f"Product Listing Performance ({self.backend_type} backend, {self.product_count} products)")
            print("-" * 100)
            print(f"{'Benchmark':<40} {'Results':>8} {'Min (ms)':>10} {'Mean (ms)':>10} {'Median':>10} {'95th':>10} {'Max (ms)':>10}")
            print("-" * 100)
            
            for name, result in self.results.items():
                print(f"{result['name']:<40} {result['result_size']:>8} {result['min_ms']:>10.2f} {result['mean_ms']:>10.2f} {result['median_ms']:>10.2f} {result['p95_ms']:>10.2f} {result['max_ms']:>10.2f}")
                
    def _save_benchmark_results(self):
        """Save benchmark results to file."""
        file_path = self.output_file
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"benchmark_results_{self.backend_type}_{timestamp}.json"
            
        output_dir = Path("./benchmark_results")
        output_dir.mkdir(exist_ok=True, parents=True)
        full_path = output_dir / file_path
        
        # Prepare data for serialization (exclude raw timings to save space)
        serializable_results = {}
        for name, result in self.results.items():
            serializable_results[name] = {k: v for k, v in result.items() if k != "raw_timings"}
            
        # Add metadata
        benchmark_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "backend": self.backend_type,
                "product_count": self.product_count,
                "iterations": self.iterations
            },
            "results": serializable_results
        }
        
        # Save as JSON
        with open(full_path, "w") as f:
            json.dump(benchmark_data, f, indent=2)
            
        # Also save as CSV for easier analysis
        csv_path = full_path.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Benchmark", "Backend", "Products", "Result Size", 
                "Min (ms)", "Mean (ms)", "Median (ms)", "P95 (ms)", "Max (ms)",
                *[f"P{p} (ms)" for p in self.percentiles]
            ])
            
            # Write data rows
            for name, result in self.results.items():
                percentiles = result["percentiles"]
                writer.writerow([
                    result["name"], 
                    self.backend_type,
                    self.product_count,
                    result["result_size"],
                    f"{result['min_ms']:.2f}",
                    f"{result['mean_ms']:.2f}",
                    f"{result['median_ms']:.2f}",
                    f"{result['p95_ms']:.2f}",
                    f"{result['max_ms']:.2f}",
                    *[f"{percentiles[f'p{p}']:.2f}" for p in self.percentiles]
                ])
                
        console.print(f"[green]Benchmark results saved to {full_path} and {csv_path}[/green]")
        
    def generate_markdown_report(self):
        """Generate a Markdown performance report."""
        if not self.results:
            console.print("[yellow]No benchmark results available. Run benchmarks first.[/yellow]")
            return None
            
        # Prepare markdown content
        md = [
            f"# Product Listing Performance Benchmarks",
            "",
            f"## Configuration",
            "",
            f"- **Backend:** {self.backend_type}",
            f"- **Product Count:** {self.product_count:,}",
            f"- **Test Iterations:** {self.iterations}",
            f"- **Date Executed:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"## Performance Summary",
            "",
            f"| Scenario | Results | Min (ms) | Mean (ms) | Median (ms) | 95th (ms) | Max (ms) |",
            f"|----------|--------:|--------:|--------:|--------:|--------:|--------:|",
        ]
        
        # Add result rows
        for name, result in self.results.items():
            md.append(
                f"| {result['name']} | {result['result_size']} | "
                f"{result['min_ms']:.2f} | {result['mean_ms']:.2f} | "
                f"{result['median_ms']:.2f} | **{result['p95_ms']:.2f}** | "
                f"{result['max_ms']:.2f} |"
            )
            
        # Add percentile details section
        md.extend([
            "",
            "## Detailed Percentiles",
            "",
            f"| Scenario | " + " | ".join([f"P{p}" for p in self.percentiles]) + " |",
            f"|----------|" + "|".join(["---:" for _ in self.percentiles]) + "|",
        ])
        
        # Add percentile rows
        for name, result in self.results.items():
            percentiles = result["percentiles"]
            md.append(
                f"| {result['name']} | " + 
                " | ".join([f"{percentiles[f'p{p}']:.2f} ms" for p in self.percentiles]) + 
                " |"
            )
            
        # Add analysis section
        md.extend([
            "",
            "## Analysis",
            "",
            "### Key Findings",
            "",
            "- The 95th percentile latency for basic listing is " + 
            f"**{self.results['basic_listing']['p95_ms']:.2f} ms**",
            f"- Complex filters with multiple conditions show a " + 
            f"**{(self.results['complex_filter']['p95_ms'] / self.results['basic_listing']['p95_ms']):.2f}x** " +
            f"increase in latency compared to basic listing",
            f"- Sorting by product name adds " + 
            f"**{(self.results['sort_by_name']['p95_ms'] - self.results['basic_listing']['p95_ms']):.2f} ms** " +
            f"to the 95th percentile latency",
            "",
            "### Recommendations",
            "",
            "- Consider implementing caching for frequently accessed listing patterns",
            f"- Monitor performance as the product database grows beyond {self.product_count:,} items",
            "- Evaluate index optimization if filter-heavy queries show degraded performance",
        ])
        
        # Combine all lines
        return "\n".join(md)
        

def main():
    parser = argparse.ArgumentParser(description="Run product listing benchmarks")
    
    parser.add_argument("--backend", choices=["sqlite", "memory", "json"], default="sqlite",
                      help="Storage backend to use (default: sqlite)")
    parser.add_argument("--products", type=int, default=10000,
                      help="Number of test products to generate (default: 10,000)")
    parser.add_argument("--iterations", type=int, default=100,
                      help="Number of iterations for each benchmark (default: 100)")
    parser.add_argument("--warm-up", type=int, default=10,
                      help="Number of warm-up iterations (default: 10)")
    parser.add_argument("--no-reset", action="store_true",
                      help="Don't reset the database before running")
    parser.add_argument("--output", type=str, default=None,
                      help="Output file for results (default: auto-generated)")
    parser.add_argument("--report", action="store_true",
                      help="Generate a Markdown report in docs/perf.md")
    
    args = parser.parse_args()
    
    # Run benchmarks
    benchmark = Benchmark(
        backend_type=args.backend,
        product_count=args.products,
        iterations=args.iterations,
        warm_up_iterations=args.warm_up,
        reset_db=not args.no_reset,
        output_file=args.output
    )
    
    # Set up environment
    benchmark.setup()
    
    # Run all benchmarks
    benchmark.run_all_benchmarks()
    
    # Generate report if requested
    if args.report:
        report = benchmark.generate_markdown_report()
        
        if report:
            docs_dir = Path("./docs")
            docs_dir.mkdir(exist_ok=True, parents=True)
            
            with open(docs_dir / "perf.md", "w") as f:
                f.write(report)
                
            console.print("[green]Performance report generated at docs/perf.md[/green]")
    
if __name__ == "__main__":
    main()