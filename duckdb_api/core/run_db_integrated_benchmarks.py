#!/usr/bin/env python3
"""
Database-Integrated Benchmark Runner

This script runs benchmarks on models and hardware platforms, storing results
directly in the DuckDB database system. It uses the ORM layer for database access
and provides a streamlined interface for running benchmarks with database integration.

It demonstrates the integration of the test runner system with the database layer
developed for Phase 16.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add DuckDB database support
try:
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
    BENCHMARK_DB_AVAILABLE = True
except ImportError:
    BENCHMARK_DB_AVAILABLE = False
    logger.warning("benchmark_db_api not available. Using deprecated JSON fallback.")


# Always deprecate JSON output in favor of DuckDB
DEPRECATE_JSON_OUTPUT = os.environ.get("DEPRECATE_JSON_OUTPUT", "1").lower() in ("1", "true", "yes")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import database models
try:
    import benchmark_db_models
    from benchmark_db_models import (
        BenchmarkDB, Models, HardwarePlatforms, TestRuns, 
        PerformanceResults, HardwareCompatibility
    )
    HAS_DB_MODELS = True
except ImportError:
    logger.warning("Database models not found. Run will generate JSON output only.")
    HAS_DB_MODELS = False

# Import ModelBenchmarkRunner
try:
    from run_model_benchmarks import ModelBenchmarkRunner, KEY_MODEL_SET, SMALL_MODEL_SET, DEFAULT_BATCH_SIZES
except ImportError:
    logger.error("ModelBenchmarkRunner not found. Please ensure run_model_benchmarks.py is available.")
    sys.exit(1)

class DatabaseIntegratedRunner:
    """
    Runs benchmarks and stores results directly in the database.
    """
    
    def __init__(
        self, 
        db_path: str = "./benchmark_db.duckdb",
        models_set: str = "key",
        hardware_types: Optional[List[str]] = None,
        batch_sizes: Optional[List[int]] = None,
        output_dir: str = "./benchmark_results",
        output_json: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize the database-integrated runner.
        
        Args:
            db_path: Path to the DuckDB database
            models_set: Which model set to use ('key', 'small', or 'custom')
            hardware_types: Hardware platforms to test
            batch_sizes: Batch sizes to test
            output_dir: Directory for output files
            output_json: Path for JSON output (if desired)
            debug: Enable debug logging
        """
        self.db_path = db_path
        self.models_set = models_set
        self.hardware_types = hardware_types
        self.batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES
        self.output_dir = output_dir
        self.output_json = output_json
        self.debug = debug
        
        # Mark start time
        self.start_time = datetime.datetime.now()
        
        # Set up models
        if models_set == "key":
            self.models = KEY_MODEL_SET
        elif models_set == "small":
            self.models = SMALL_MODEL_SET
        else:
            logger.error(f"Invalid models_set: {models_set}")
            sys.exit(1)
        
        # Connect to database if available
        self.db_conn = None
        if HAS_DB_MODELS:
            try:
                self.db_conn = BenchmarkDB(db_path=db_path)
                logger.info(f"Connected to database: {db_path}")
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                logger.warning("Continuing without database connection")
        
        # Initialize ModelBenchmarkRunner
        self.runner = ModelBenchmarkRunner(
            output_dir=output_dir,
            models_set=models_set,
            hardware_types=hardware_types,
            batch_sizes=batch_sizes,
            verify_functionality=True,
            measure_performance=True,
            generate_plots=True,
            update_compatibility_matrix=True,
            use_resource_pool=True
        )
        
    def __del__(self):
        """Clean up database connection on exit"""
        if self.db_conn:
            self.db_conn.close()
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """
        Run benchmarks and store results in the database.
        
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting database-integrated benchmark run")
        
        # Run benchmarks with ModelBenchmarkRunner
        results = self.runner.run_benchmarks()
        
        # Store results in database if available
        if self.db_conn and results:
            logger.info("Storing benchmark results in database")
            self._store_results_in_database(results)
        
        # Save to JSON if requested
        if self.output_json and results:
            # JSON output deprecated in favor of database storage
            if not DEPRECATE_JSON_OUTPUT:
                with open(self.output_json, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {self.output_json}")
            else:
                logger.info("JSON output is deprecated. Results are stored directly in the database.")
        
        return results
    
    def _store_results_in_database(self, results: Dict[str, Any]) -> None:
        """
        Store benchmark results in the database.
        
        Args:
            results: Benchmark results dictionary
        """
        try:
            # Create a test run entry
            run_data = {
                'test_name': 'db_integrated_benchmark',
                'test_type': 'comprehensive',
                'started_at': self.start_time,
                'completed_at': datetime.datetime.now(),
                'execution_time_seconds': (datetime.datetime.now() - self.start_time).total_seconds(),
                'success': True,
                'command_line': ' '.join(sys.argv),
                'metadata': {
                    'models_set': self.models_set,
                    'hardware_types': self.hardware_types,
                    'batch_sizes': self.batch_sizes
                }
            }
            
            test_run = TestRuns(**run_data)
            run_id = self.db_conn.insert_test_runs(test_run)
            logger.info(f"Created test run entry with ID: {run_id}")
            
            # Store functionality verification results if available
            if results.get("functionality_verification"):
                self._store_functionality_results(results["functionality_verification"], run_id)
            
            # Store performance benchmark results if available
            if results.get("performance_benchmarks"):
                self._store_performance_results(results["performance_benchmarks"], run_id)
                
            logger.info("Successfully stored benchmark results in database")
            
        except Exception as e:
            logger.error(f"Error storing results in database: {e}")
    
    def _store_functionality_results(self, functionality_results: Dict[str, Any], run_id: int) -> None:
        """
        Store functionality verification results in the database.
        
        Args:
            functionality_results: Functionality verification results
            run_id: Test run ID
        """
        compatibility_count = 0
        
        for hw_type, hw_results in functionality_results.items():
            # Extract model results (handle different formats)
            model_results = {}
            
            if "models" in hw_results:
                model_results = hw_results["models"]
            elif "model_results" in hw_results:
                model_results = hw_results["model_results"]
            
            # Process each model result
            for model_key, result in model_results.items():
                if model_key not in self.models:
                    continue
                
                # Get or add model
                model_name = self.models[model_key]["name"]
                model_id = self.db_conn.get_or_add_model(model_name)
                
                # Get or add hardware platform
                hardware_id = self.db_conn.get_or_add_hardware(hw_type)
                
                # Extract success and error information
                success = False
                error_message = None
                
                if isinstance(result, dict):
                    success = result.get("success", False)
                    error_message = result.get("error")
                elif isinstance(result, bool):
                    success = result
                
                # Create compatibility record
                compatibility = HardwareCompatibility(
                    run_id=run_id,
                    model_id=model_id,
                    hardware_id=hardware_id,
                    is_compatible=success,
                    detection_success=True,
                    initialization_success=success,
                    error_message=error_message,
                    error_type="functionality_test",
                    compatibility_score=1.0 if success else 0.0
                )
                
                # Insert into database
                self.db_conn.insert_hardware_compatibility(compatibility)
                compatibility_count += 1
        
        logger.info(f"Stored {compatibility_count} compatibility records")
    
    def _store_performance_results(self, performance_results: Dict[str, Any], run_id: int) -> None:
        """
        Store performance benchmark results in the database.
        
        Args:
            performance_results: Performance benchmark results
            run_id: Test run ID
        """
        performance_count = 0
        
        for family, family_results in performance_results.items():
            if "benchmarks" not in family_results:
                continue
            
            # Process each model in family
            for model_name, hw_results in family_results["benchmarks"].items():
                # Get or add model
                model_id = self.db_conn.get_or_add_model(model_name, family)
                
                # Process each hardware platform
                for hw_type, hw_metrics in hw_results.items():
                    if "performance_summary" not in hw_metrics:
                        continue
                    
                    # Get performance summary
                    perf = hw_metrics["performance_summary"]
                    
                    # Get or add hardware platform
                    hardware_id = self.db_conn.get_or_add_hardware(hw_type)
                    
                    # Process each batch size if available
                    if "benchmark_results" in hw_metrics:
                        for config, config_results in hw_metrics["benchmark_results"].items():
                            if config_results.get("status") != "completed":
                                continue
                                
                            # Extract batch size from config
                            batch_size = 1
                            if "batch_" in config:
                                parts = config.split("_")
                                batch_idx = parts.index("batch") + 1
                                if batch_idx < len(parts):
                                    try:
                                        batch_size = int(parts[batch_idx])
                                    except ValueError:
                                        pass
                            
                            # Create performance result
                            perf_result = PerformanceResults(
                                run_id=run_id,
                                model_id=model_id,
                                hardware_id=hardware_id,
                                test_case=family,
                                batch_size=batch_size,
                                precision=config_results.get("precision", "fp32"),
                                total_time_seconds=config_results.get("total_time", 0.0),
                                average_latency_ms=config_results.get("avg_latency", 0.0) * 1000,  # Convert to ms
                                throughput_items_per_second=config_results.get("throughput", 0.0),
                                memory_peak_mb=config_results.get("memory_usage", {}).get("peak", 0.0),
                                iterations=config_results.get("iterations", 0),
                                warmup_iterations=config_results.get("warmup_iterations", 0),
                                metrics=config_results.get("metrics", {})
                            )
                            
                            # Insert into database
                            self.db_conn.insert_performance_results(perf_result)
                            performance_count += 1
                    else:
                        # No batch-specific results, create single entry from summary
                        latency_ms = 0.0
                        throughput = 0.0
                        memory_peak = 0.0
                        
                        if "latency" in perf and "mean" in perf["latency"]:
                            latency_ms = perf["latency"]["mean"] * 1000  # Convert to ms
                        
                        if "throughput" in perf and "mean" in perf["throughput"]:
                            throughput = perf["throughput"]["mean"]
                        
                        if "memory" in perf and "max_allocated" in perf["memory"]:
                            memory_peak = perf["memory"]["max_allocated"]
                        
                        # Create performance result
                        perf_result = PerformanceResults(
                            run_id=run_id,
                            model_id=model_id,
                            hardware_id=hardware_id,
                            test_case=family,
                            batch_size=1,  # Default if not specified
                            precision="fp32",  # Default if not specified
                            average_latency_ms=latency_ms,
                            throughput_items_per_second=throughput,
                            memory_peak_mb=memory_peak,
                            metrics={}
                        )
                        
                        # Insert into database
                        self.db_conn.insert_performance_results(perf_result)
                        performance_count += 1
        
        logger.info(f"Stored {performance_count} performance results")

def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description="Database-Integrated Benchmark Runner")
    
    # Database options
    parser.add_argument("--db-path", type=str, default="./benchmark_db.duckdb",
                      help="Path to the DuckDB database")
    
    # Benchmark options
    parser.add_argument("--models-set", choices=["key", "small"], default="small",
                      help="Which model set to use")
    parser.add_argument("--hardware", type=str, nargs="+",
                      help="Hardware platforms to test (defaults to all available)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES,
                      help="Batch sizes to test")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                      help="Directory for output files")
    parser.add_argument("--output-json", type=str,
                      help="Path for JSON output (if desired)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    
    parser.add_argument("--db-path", type=str, default=None,
                      help="Path to the benchmark database")
    parser.add_argument("--db-only", action="store_true",
                      help="Store results only in the database, not in JSON")
args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Create and run the integrated runner
    runner = DatabaseIntegratedRunner(
        db_path = args.db_path
    if db_path is None:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        logger.info(f"Using database path from environment: {db_path}"),
        models_set=args.models_set,
        hardware_types=args.hardware,
        batch_sizes=args.batch_sizes,
        output_dir=args.output_dir,
        output_json=args.output_json,
        debug=args.debug
    )
    
    # Run benchmarks
    results = runner.run_benchmarks()
    
    # Print summary
    print("\nDatabase-Integrated Benchmark Summary:")
    print(f"- Models set: {args.models_set}")
    print(f"- Hardware: {args.hardware or 'all available'}")
    print(f"- Batch sizes: {args.batch_sizes}")
    print(f"- Results stored in database: {args.db_path}")
    
    if args.output_dir:
        print(f"- Full results available in: {args.output_dir}")
    
    if args.output_json:
        print(f"- JSON results saved to: {args.output_json}")

if __name__ == "__main__":
    main()