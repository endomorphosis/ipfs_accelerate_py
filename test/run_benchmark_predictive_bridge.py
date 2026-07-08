#!/usr/bin/env python3
"""
Run Benchmark to Predictive Performance Bridge

This script provides a convenient way to run the Benchmark to Predictive Performance Bridge,
which synchronizes benchmark results with the predictive performance system.
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_benchmark_predictive_bridge")

# Try to import bridge components
try:
    from test.integration.benchmark_predictive_performance_bridge import BenchmarkPredictivePerformanceBridge
    from test.integration.bridge_service import BridgeService
    from test.integration.bridge_config import load_config, get_bridge_config, create_default_config
except ImportError as e:
    logger.error(f"Failed to import bridge components: {e}")
    print("\nError importing bridge components. Make sure you're running this script from the project root directory.")
    sys.exit(1)

def find_benchmark_db():
    """
    Find the benchmark database file.
    
    Returns:
        Path to the benchmark database file, or None if not found
    """
    # Check common locations
    locations = [
        "./benchmark_db.duckdb",
        "./test/benchmark_db.duckdb",
        "./benchmarks/benchmark_db.duckdb",
        "./test/refactored_benchmark_suite/benchmark_db.duckdb"
    ]
    
    for location in locations:
        if os.path.exists(location):
            return location
    
    return None

def check_servers(api_url, timeout=5):
    """
    Check if the Unified API Server and Predictive Performance API are running.
    
    Args:
        api_url: Base URL of the Unified API Server
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (server_running, predictive_api_running)
    """
    import requests
    
    server_running = False
    predictive_api_running = False
    
    # Check server
    try:
        response = requests.get(f"{api_url}/health", timeout=timeout)
        if response.status_code == 200:
            server_running = True
            logger.info(f"Unified API Server is running at {api_url}")
        else:
            logger.warning(f"Unified API Server returned status code {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not connect to Unified API Server: {e}")
    
    # Check Predictive Performance API
    try:
        response = requests.get(f"{api_url}/api/predictive-performance/health", timeout=timeout)
        if response.status_code == 200:
            predictive_api_running = True
            logger.info("Predictive Performance API is running")
        else:
            logger.warning(f"Predictive Performance API returned status code {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not connect to Predictive Performance API: {e}")
    
    return server_running, predictive_api_running

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Benchmark to Predictive Performance Bridge")
    
    # Server options
    parser.add_argument("--api-url", type=str, default="http://localhost:8080",
                      help="URL of the Unified API Server")
    parser.add_argument("--benchmark-db", type=str,
                      help="Path to the benchmark database file")
    parser.add_argument("--config", type=str,
                      help="Path to configuration file")
    
    # Operation modes
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--service", action="store_true",
                      help="Run as a continuous service")
    group.add_argument("--sync", action="store_true",
                      help="Run a single synchronization cycle")
    group.add_argument("--report", action="store_true",
                      help="Generate a synchronization report")
    group.add_argument("--create-config", action="store_true",
                      help="Create a default configuration file")
    
    # Options for specific modes
    parser.add_argument("--model", type=str,
                      help="Synchronize results for a specific model (--sync mode)")
    parser.add_argument("--days", type=int, default=30,
                      help="Number of days to look back (--sync mode)")
    parser.add_argument("--limit", type=int, default=100,
                      help="Maximum number of results to synchronize (--sync mode)")
    parser.add_argument("--output", type=str,
                      help="Path to write report JSON (--report mode)")
    parser.add_argument("--log-file", type=str,
                      help="Path to log file (--service mode)")
    
    args = parser.parse_args()
    
    # Create default configuration if requested
    if args.create_config:
        config_path = args.config or "bridge_config.json"
        if create_default_config(config_path):
            print(f"Created default configuration at {config_path}")
        else:
            print(f"Failed to create configuration at {config_path}")
        return 0
    
    # Find benchmark database if not specified
    benchmark_db = args.benchmark_db or find_benchmark_db()
    if not benchmark_db:
        print("ERROR: Could not find benchmark database file")
        print("Please specify the path with --benchmark-db")
        return 1
    
    if not os.path.exists(benchmark_db):
        print(f"ERROR: Benchmark database file not found at {benchmark_db}")
        return 1
    
    print(f"Using benchmark database: {benchmark_db}")
    
    # Check if servers are running
    server_running, predictive_api_running = check_servers(args.api_url)
    
    if not server_running:
        print("WARNING: Unified API Server does not appear to be running")
        print(f"  Expected at: {args.api_url}")
        print("  You may need to start it with: python test/unified_api_server.py")
        
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != "y":
            return 1
    
    if not predictive_api_running:
        print("WARNING: Predictive Performance API does not appear to be running")
        print("  You may need to start it with: python test/run_integrated_api_servers.py")
        
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != "y":
            return 1
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    bridge_config = get_bridge_config(config, "benchmark_predictive_performance")
    bridge_config["benchmark_db_path"] = benchmark_db
    bridge_config["predictive_api_url"] = args.api_url
    
    # Run as service
    if args.service:
        print("Starting bridge service in continuous mode...")
        service = BridgeService(
            config_path=args.config,
            log_file=args.log_file
        )
        service.start()
        return 0
    
    # Single synchronization cycle
    elif args.sync:
        print("Running benchmark to predictive performance synchronization...")
        
        # Create bridge
        bridge = BenchmarkPredictivePerformanceBridge(
            benchmark_db_path=benchmark_db,
            predictive_api_url=args.api_url
        )
        
        try:
            # Check connections
            status = bridge.check_connections()
            if not status["benchmark_db"]:
                print(f"ERROR: Could not connect to benchmark database at {benchmark_db}")
                return 1
            
            if not status["predictive_api"]:
                print(f"ERROR: Could not connect to Predictive Performance API at {args.api_url}")
                return 1
            
            # Synchronize results
            if args.model:
                print(f"Synchronizing results for model {args.model}...")
                result = bridge.sync_by_model(
                    model_name=args.model,
                    days=args.days,
                    limit=args.limit
                )
            else:
                print(f"Synchronizing recent benchmark results (limit: {args.limit})...")
                result = bridge.sync_recent_results(limit=args.limit)
            
            # Print summary
            print(f"Synchronization complete: {result['message']}")
            print(f"Total: {result['total']}, Synced: {result['synced']}, Failed: {result['failed']}")
            
            return 0
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return 1
        
        finally:
            # Close connections
            bridge.close()
    
    # Generate report
    elif args.report:
        print("Generating synchronization report...")
        
        # Create bridge
        bridge = BenchmarkPredictivePerformanceBridge(
            benchmark_db_path=benchmark_db,
            predictive_api_url=args.api_url
        )
        
        try:
            # Check connections
            status = bridge.check_connections()
            if not status["benchmark_db"]:
                print(f"ERROR: Could not connect to benchmark database at {benchmark_db}")
                return 1
            
            # Generate report
            report = bridge.generate_report()
            
            # Print report summary
            print(f"Report generated at {report['generated_at']}")
            print(f"Total results: {report['total_results']}")
            print(f"Synced results: {report['synced_results']} ({report['sync_percentage']}%)")
            
            # Print top models
            print("\nTop models by result count:")
            for model in report['model_coverage'][:5]:
                print(f"  {model['model_name']}: {model['total_results']} results, {model['synced_results']} synced ({model['sync_percentage']}%)")
            
            # Print hardware coverage
            print("\nHardware coverage:")
            for hw in report['hardware_coverage']:
                print(f"  {hw['hardware_type']}: {hw['total_results']} results, {hw['synced_results']} synced ({hw['sync_percentage']}%)")
            
            # Write report to file if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(report, f, indent=2)
                print(f"\nFull report written to {args.output}")
            
            return 0
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return 1
        
        finally:
            # Close connections
            bridge.close()
    
    # If no mode specified, show help
    else:
        parser.print_help()
        print("\nYou must specify one of: --service, --sync, --report, or --create-config")
        return 1

if __name__ == "__main__":
    sys.exit(main())