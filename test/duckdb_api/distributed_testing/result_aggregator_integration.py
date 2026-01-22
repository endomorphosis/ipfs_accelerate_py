#!/usr/bin/env python3
"""
Distributed Testing Framework - Result Aggregator Integration

This module integrates the ResultAggregatorService with the BenchmarkDBAPI
and provides easy-to-use functions for setting up and using the aggregator.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("result_aggregator_integration")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import core components
from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
from duckdb_api.core.aggregation_db_extensions import extend_benchmark_db_api
from duckdb_api.schema.aggregation_schema import create_aggregation_tables
from duckdb_api.distributed_testing.result_aggregator import ResultAggregatorService
from duckdb_api.distributed_testing.performance_trend_analyzer import PerformanceTrendAnalyzer

# Constants from result_aggregator for convenience
from duckdb_api.distributed_testing.result_aggregator import (
    RESULT_TYPE_PERFORMANCE,
    RESULT_TYPE_COMPATIBILITY,
    RESULT_TYPE_INTEGRATION,
    RESULT_TYPE_WEB_PLATFORM,
    AGGREGATION_LEVEL_TEST_RUN,
    AGGREGATION_LEVEL_MODEL,
    AGGREGATION_LEVEL_HARDWARE,
    AGGREGATION_LEVEL_MODEL_HARDWARE,
    AGGREGATION_LEVEL_TASK_TYPE,
    AGGREGATION_LEVEL_WORKER,
)


def setup_result_aggregator(db_path: str = None) -> Tuple[ResultAggregatorService, BenchmarkDBAPI]:
    """Set up and configure the ResultAggregatorService with the database.
    
    Args:
        db_path: Path to the DuckDB database file (None for in-memory)
        
    Returns:
        Tuple of (ResultAggregatorService, BenchmarkDBAPI)
    """
    # Extend BenchmarkDBAPI with aggregation methods
    ExtendedBenchmarkDBAPI = extend_benchmark_db_api(BenchmarkDBAPI)
    
    # Initialize extended database API
    db_api = ExtendedBenchmarkDBAPI(db_path=db_path)
    
    # Create aggregation tables in the database
    create_aggregation_tables(db_api.conn)
    
    # Initialize performance trend analyzer
    trend_analyzer = PerformanceTrendAnalyzer(db_manager=db_api)
    
    # Initialize result aggregator
    result_aggregator = ResultAggregatorService(
        db_manager=db_api,
        trend_analyzer=trend_analyzer
    )
    
    # Configure default settings
    result_aggregator.configure({
        "cache_ttl_seconds": 300,  # 5 minutes cache TTL
        "anomaly_threshold": 2.5,  # Z-score threshold for anomalies
        "min_data_points": 5,      # Minimum data points for analysis
        "comparative_lookback_days": 7,  # Week of historical data for comparison
    })
    
    # Start the trend analyzer
    trend_analyzer.start()
    
    logger.info(f"Result aggregator set up with database at {db_path or 'in-memory'}")
    return result_aggregator, db_api


def generate_performance_report(aggregator: ResultAggregatorService, 
                              output_dir: str = "reports",
                              format: str = "json") -> Dict[str, str]:
    """Generate a comprehensive performance report across all models and hardware.
    
    Args:
        aggregator: ResultAggregatorService instance
        output_dir: Directory to save report files
        format: Report format ('json' or 'csv')
        
    Returns:
        Dictionary mapping report names to file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Reports to generate
    reports = [
        # Model-level performance
        {
            "name": "model_performance",
            "result_type": RESULT_TYPE_PERFORMANCE,
            "aggregation_level": AGGREGATION_LEVEL_MODEL
        },
        # Hardware-level performance
        {
            "name": "hardware_performance",
            "result_type": RESULT_TYPE_PERFORMANCE,
            "aggregation_level": AGGREGATION_LEVEL_HARDWARE
        },
        # Model-hardware combinations
        {
            "name": "model_hardware_performance",
            "result_type": RESULT_TYPE_PERFORMANCE,
            "aggregation_level": AGGREGATION_LEVEL_MODEL_HARDWARE
        },
        # Hardware compatibility
        {
            "name": "hardware_compatibility",
            "result_type": RESULT_TYPE_COMPATIBILITY,
            "aggregation_level": AGGREGATION_LEVEL_HARDWARE
        },
        # Integration test results
        {
            "name": "integration_tests",
            "result_type": RESULT_TYPE_INTEGRATION,
            "aggregation_level": AGGREGATION_LEVEL_TASK_TYPE
        },
        # Web platform performance
        {
            "name": "web_platform_performance",
            "result_type": RESULT_TYPE_WEB_PLATFORM,
            "aggregation_level": AGGREGATION_LEVEL_TASK_TYPE
        }
    ]
    
    # Generate each report
    report_files = {}
    
    for report in reports:
        try:
            # Generate the report file path
            file_path = os.path.join(output_dir, f"{report['name']}.{format}")
            
            # Export the report
            aggregator.export_results(
                result_type=report["result_type"],
                aggregation_level=report["aggregation_level"],
                format=format,
                file_path=file_path
            )
            
            report_files[report["name"]] = file_path
            logger.info(f"Generated {report['name']} report at {file_path}")
            
        except Exception as e:
            logger.error(f"Error generating {report['name']} report: {e}")
            
    # Generate anomaly report
    try:
        # Get anomalies across all result types
        all_anomalies = {}
        
        for result_type in [RESULT_TYPE_PERFORMANCE, RESULT_TYPE_WEB_PLATFORM]:
            anomalies = aggregator.get_result_anomalies(
                result_type=result_type,
                aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE
            )
            
            if anomalies["anomaly_count"] > 0:
                all_anomalies[result_type] = anomalies
                
        # If we found anomalies, write to file
        if all_anomalies:
            import json
            anomaly_path = os.path.join(output_dir, f"anomalies.json")
            
            with open(anomaly_path, 'w') as f:
                json.dump(all_anomalies, f, indent=2)
                
            report_files["anomalies"] = anomaly_path
            logger.info(f"Generated anomalies report at {anomaly_path}")
            
    except Exception as e:
        logger.error(f"Error generating anomalies report: {e}")
        
    return report_files


def analyze_test_results(aggregator: ResultAggregatorService,
                       test_run_id: str) -> Dict[str, Any]:
    """Analyze the results of a specific test run.
    
    Args:
        aggregator: ResultAggregatorService instance
        test_run_id: ID of the test run to analyze
        
    Returns:
        Dictionary with analysis results
    """
    # Filter for specific test run
    filter_params = {"run_id": test_run_id}
    
    # Collect results for all result types
    results = {}
    
    # Performance results
    performance = aggregator.aggregate_results(
        result_type=RESULT_TYPE_PERFORMANCE,
        aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE,
        filter_params=filter_params
    )
    results["performance"] = performance
    
    # Compatibility results
    compatibility = aggregator.aggregate_results(
        result_type=RESULT_TYPE_COMPATIBILITY,
        aggregation_level=AGGREGATION_LEVEL_MODEL_HARDWARE,
        filter_params=filter_params
    )
    results["compatibility"] = compatibility
    
    # Integration test results
    integration = aggregator.aggregate_results(
        result_type=RESULT_TYPE_INTEGRATION,
        aggregation_level=AGGREGATION_LEVEL_TASK_TYPE,
        filter_params=filter_params
    )
    results["integration"] = integration
    
    # Web platform results
    web_platform = aggregator.aggregate_results(
        result_type=RESULT_TYPE_WEB_PLATFORM,
        aggregation_level=AGGREGATION_LEVEL_TASK_TYPE,
        filter_params=filter_params
    )
    results["web_platform"] = web_platform
    
    # Extract summary statistics
    summary = {
        "test_run_id": test_run_id,
        "anomalies_detected": 0,
        "performance_metrics": {},
        "compatibility_rate": 0.0,
        "integration_test_pass_rate": 0.0,
        "web_platform_success_rate": 0.0
    }
    
    # Count anomalies
    for result_type, result in results.items():
        if "anomalies" in result.get("results", {}):
            anomalies = result["results"]["anomalies"]
            summary["anomalies_detected"] += len(anomalies)
    
    # Extract performance metrics if available
    perf_stats = performance.get("results", {}).get("basic_statistics", {})
    if perf_stats:
        # Calculate average across all model-hardware combinations
        latencies = []
        throughputs = []
        
        for stats in perf_stats.values():
            if "average_latency_ms" in stats:
                latencies.append(stats["average_latency_ms"].get("mean", 0))
            if "throughput_items_per_second" in stats:
                throughputs.append(stats["throughput_items_per_second"].get("mean", 0))
                
        if latencies:
            summary["performance_metrics"]["avg_latency_ms"] = sum(latencies) / len(latencies)
        if throughputs:
            summary["performance_metrics"]["avg_throughput"] = sum(throughputs) / len(throughputs)
    
    # Extract compatibility rate if available
    compat_stats = compatibility.get("results", {}).get("distributions", {})
    for stats in compat_stats.values():
        if "is_compatible" in stats:
            dist = stats["is_compatible"].get("distribution", {})
            if "True" in dist:
                total = stats["is_compatible"].get("total_values", 0)
                if total > 0:
                    compatible_count = dist["True"].get("count", 0)
                    summary["compatibility_rate"] = (compatible_count / total) * 100
                    break
    
    # Extract integration test pass rate if available
    int_stats = integration.get("results", {}).get("distributions", {})
    for stats in int_stats.values():
        if "status" in stats:
            dist = stats["status"].get("distribution", {})
            if "pass" in dist:
                total = stats["status"].get("total_values", 0)
                if total > 0:
                    pass_count = dist["pass"].get("count", 0)
                    summary["integration_test_pass_rate"] = (pass_count / total) * 100
                    break
    
    # Extract web platform success rate if available
    web_stats = web_platform.get("results", {}).get("distributions", {})
    for stats in web_stats.values():
        if "success" in stats:
            dist = stats["success"].get("distribution", {})
            if "True" in dist:
                total = stats["success"].get("total_values", 0)
                if total > 0:
                    success_count = dist["True"].get("count", 0)
                    summary["web_platform_success_rate"] = (success_count / total) * 100
                    break
    
    # Add summary to results
    results["summary"] = summary
    
    return results


def main():
    """Main function for CLI usage."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Result Aggregator CLI")
    parser.add_argument("--db-path", default=None,
                      help="Path to DuckDB database file")
    parser.add_argument("--report-dir", default="reports",
                      help="Directory to save reports")
    parser.add_argument("--format", choices=["json", "csv"], default="json",
                      help="Report format")
    parser.add_argument("--test-run", default=None,
                      help="Test run ID to analyze")
    
    args = parser.parse_args()
    
    # Set up aggregator
    aggregator, db_api = setup_result_aggregator(args.db_path)
    
    try:
        if args.test_run:
            # Analyze specific test run
            results = analyze_test_results(aggregator, args.test_run)
            
            # Print summary
            print("\n===== Test Run Analysis =====")
            print(f"Test Run ID: {results['summary']['test_run_id']}")
            print(f"Anomalies Detected: {results['summary']['anomalies_detected']}")
            print(f"Compatibility Rate: {results['summary']['compatibility_rate']:.1f}%")
            print(f"Integration Test Pass Rate: {results['summary']['integration_test_pass_rate']:.1f}%")
            print(f"Web Platform Success Rate: {results['summary']['web_platform_success_rate']:.1f}%")
            
            # Print performance metrics
            if results['summary']['performance_metrics']:
                print("\nPerformance Metrics:")
                for metric, value in results['summary']['performance_metrics'].items():
                    print(f"  {metric}: {value:.2f}")
                    
        else:
            # Generate comprehensive reports
            report_files = generate_performance_report(
                aggregator, 
                output_dir=args.report_dir,
                format=args.format
            )
            
            # Print report files
            print("\n===== Generated Reports =====")
            for name, file_path in report_files.items():
                print(f"{name}: {file_path}")
                
    finally:
        # Clean up
        if hasattr(db_api, 'close'):
            db_api.close()
            
        # Stop the trend analyzer
        aggregator.trend_analyzer.stop()
    

if __name__ == "__main__":
    main()