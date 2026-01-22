#!/usr/bin/env python3
"""
Run Script for Result Aggregator Performance Analyzer

This script demonstrates the capabilities of the Result Aggregator Performance Analyzer
by generating sample test data and then analyzing it with the various analysis tools.

Usage:
    python run_test_performance_analyzer.py [--db-path DB_PATH] [--output-dir OUTPUT_DIR]
                                           [--generate-data] [--num-results NUM]
                                           [--format FORMAT] [--mode MODE]
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_test_performance_analyzer")

# Import required modules
try:
    from result_aggregator.service import ResultAggregatorService
    from result_aggregator.performance_analyzer import PerformanceAnalyzer
except ImportError:
    logger.error("Required modules not found. Make sure you have the Result Aggregator installed.")
    sys.exit(1)

def generate_sample_data(service, num_results=500, with_trend=True, with_regression=True, with_hardware_diversity=True):
    """Generate sample test results for demonstration."""
    logger.info(f"Generating {num_results} sample test results...")
    
    # Test types
    test_types = ["benchmark", "unit_test", "integration_test", "performance_test"]
    
    # Statuses
    statuses = ["completed", "failed"]
    status_weights = [0.9, 0.1]  # 90% completed, 10% failed
    
    # Worker IDs
    worker_ids = [f"worker_{i}" for i in range(1, 6)]
    
    # Hardware types
    hardware_types = ["cpu", "cuda", "rocm", "webgpu", "webnn"]
    
    # Models
    models = ["bert", "t5", "vit", "whisper", "llama"]
    
    # Batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    # Precision types
    precision_types = ["fp32", "fp16", "int8", "int4"]
    
    # Metrics
    metrics = {
        "throughput": {"min": 80, "max": 150, "unit": "items/s"},
        "latency_ms": {"min": 2, "max": 12, "unit": "ms"},
        "memory_usage_mb": {"min": 500, "max": 2000, "unit": "MB"},
        "execution_time": {"min": 1, "max": 30, "unit": "s"},
        "qps": {"min": 100, "max": 500, "unit": "queries/s"},
        "response_time_ms": {"min": 20, "max": 200, "unit": "ms"},
        "cpu_usage_percent": {"min": 10, "max": 90, "unit": "%"},
        "power_consumption": {"min": 50, "max": 300, "unit": "W"}
    }
    
    # Performance profiles by hardware
    hardware_profiles = {
        "cpu": {
            "throughput": {"multiplier": 1.0},
            "latency_ms": {"multiplier": 1.5},
            "memory_usage_mb": {"multiplier": 0.8},
            "qps": {"multiplier": 1.0},
            "power_consumption": {"multiplier": 0.7}
        },
        "cuda": {
            "throughput": {"multiplier": 3.0},
            "latency_ms": {"multiplier": 0.5},
            "memory_usage_mb": {"multiplier": 1.5},
            "qps": {"multiplier": 3.0},
            "power_consumption": {"multiplier": 2.0}
        },
        "rocm": {
            "throughput": {"multiplier": 2.5},
            "latency_ms": {"multiplier": 0.6},
            "memory_usage_mb": {"multiplier": 1.3},
            "qps": {"multiplier": 2.5},
            "power_consumption": {"multiplier": 1.8}
        },
        "webgpu": {
            "throughput": {"multiplier": 1.8},
            "latency_ms": {"multiplier": 0.7},
            "memory_usage_mb": {"multiplier": 1.0},
            "qps": {"multiplier": 1.8},
            "power_consumption": {"multiplier": 1.2}
        },
        "webnn": {
            "throughput": {"multiplier": 1.5},
            "latency_ms": {"multiplier": 0.8},
            "memory_usage_mb": {"multiplier": 0.9},
            "qps": {"multiplier": 1.5},
            "power_consumption": {"multiplier": 1.0}
        }
    }
    
    # Generate results with timestamps over a period of time
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Trend direction for each metric (for time-based trends)
    trend_directions = {}
    for metric in metrics:
        trend_directions[metric] = random.choice(["increasing", "decreasing", "stable"])
    
    # Generate test results
    result_ids = []
    
    for i in range(num_results):
        # Generate timestamp within the time range
        progress = i / num_results
        days_offset = int(progress * 90)
        timestamp = start_date + timedelta(days=days_offset)
        
        # Select random test parameters
        test_type = random.choice(test_types)
        status = random.choices(statuses, weights=status_weights)[0]
        worker_id = random.choice(worker_ids)
        
        # Generate hardware requirements
        if with_hardware_diversity:
            hardware = random.choice(hardware_types)
        else:
            hardware = "cpu"  # Use only CPU for simplicity
        
        # Select model and batch size
        model = random.choice(models)
        batch_size = random.choice(batch_sizes)
        precision = random.choice(precision_types)
        
        # Generate base metrics
        result_metrics = {}
        for metric_name, metric_config in metrics.items():
            base_value = random.uniform(metric_config["min"], metric_config["max"])
            
            # Apply hardware-specific multipliers
            if hardware in hardware_profiles and metric_name in hardware_profiles[hardware]:
                multiplier = hardware_profiles[hardware][metric_name]["multiplier"]
                value = base_value * multiplier
            else:
                value = base_value
            
            # Apply time-based trend if enabled
            if with_trend:
                trend_direction = trend_directions[metric_name]
                if trend_direction == "increasing":
                    trend_factor = 1.0 + (progress * 0.5)  # Up to 50% increase
                elif trend_direction == "decreasing":
                    trend_factor = 1.0 - (progress * 0.3)  # Up to 30% decrease
                else:  # stable
                    trend_factor = 1.0 + (random.uniform(-0.1, 0.1))  # +/- 10% noise
                
                value *= trend_factor
            
            # Apply regression at the end (last 10% of data) if enabled
            if with_regression and progress > 0.9:
                # Introduce regression for throughput and qps (lower is worse)
                if metric_name in ["throughput", "qps"]:
                    value *= 0.7  # 30% regression
                
                # Introduce regression for latency and response time (higher is worse)
                elif metric_name in ["latency_ms", "response_time_ms"]:
                    value *= 1.4  # 40% regression
            
            # Store metric
            result_metrics[metric_name] = value
        
        # Create result object
        result = {
            "task_id": f"task_{i}_{random.randint(1000, 9999)}",
            "worker_id": worker_id,
            "timestamp": timestamp.isoformat(),
            "type": test_type,
            "status": status,
            "duration": result_metrics.get("execution_time", 1.0),
            "details": {
                "requirements": {
                    "hardware": [hardware]
                },
                "metadata": {
                    "model": model,
                    "batch_size": batch_size,
                    "precision": precision
                }
            },
            "metrics": result_metrics
        }
        
        # Store result and get ID
        result_id = service.store_result(result)
        result_ids.append(result_id)
        
        if (i + 1) % 50 == 0:
            logger.info(f"Generated {i + 1}/{num_results} results")
    
    logger.info(f"Generated {num_results} sample test results")
    return result_ids

def run_performance_analysis(service, analyzer, args):
    """Run various performance analysis tools and generate reports."""
    output_dir = args.output_dir
    format = args.format
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run regression analysis
    logger.info("Running regression analysis...")
    regression_analysis = analyzer.detect_performance_regression(
        metric_name="throughput",
        baseline_period="30d",
        comparison_period="7d"
    )
    
    # Save regression analysis
    if output_dir:
        file_path = os.path.join(output_dir, f"regression_analysis.{format}")
        if format == "json":
            with open(file_path, "w") as f:
                json.dump(regression_analysis, f, indent=2)
        else:  # markdown or html
            report = analyzer.generate_performance_report(
                report_type="regression",
                format=format
            )
            with open(file_path, "w") as f:
                f.write(report)
        
        logger.info(f"Saved regression analysis to {file_path}")
    
    # Run hardware comparison
    logger.info("Running hardware performance comparison...")
    hardware_comparison = analyzer.compare_hardware_performance(
        metrics=["throughput", "latency_ms", "memory_usage_mb"],
        time_period="60d"
    )
    
    # Save hardware comparison
    if output_dir:
        file_path = os.path.join(output_dir, f"hardware_comparison.{format}")
        if format == "json":
            with open(file_path, "w") as f:
                json.dump(hardware_comparison, f, indent=2)
        else:  # markdown or html
            report = analyzer.generate_performance_report(
                report_type="hardware_comparison",
                format=format
            )
            with open(file_path, "w") as f:
                f.write(report)
        
        logger.info(f"Saved hardware comparison to {file_path}")
    
    # Run resource efficiency analysis
    logger.info("Running resource efficiency analysis...")
    efficiency_analysis = analyzer.analyze_resource_efficiency(
        time_period="60d"
    )
    
    # Save efficiency analysis
    if output_dir:
        file_path = os.path.join(output_dir, f"resource_efficiency.{format}")
        if format == "json":
            with open(file_path, "w") as f:
                json.dump(efficiency_analysis, f, indent=2)
        else:  # markdown or html
            report = analyzer.generate_performance_report(
                report_type="efficiency",
                format=format
            )
            with open(file_path, "w") as f:
                f.write(report)
        
        logger.info(f"Saved resource efficiency analysis to {file_path}")
    
    # Run time-based analysis for throughput
    logger.info("Running time-based performance analysis...")
    time_analysis = analyzer.analyze_performance_over_time(
        metric_name="throughput",
        time_period="90d"
    )
    
    # Save time analysis
    if output_dir:
        file_path = os.path.join(output_dir, f"time_analysis.{format}")
        if format == "json":
            with open(file_path, "w") as f:
                json.dump(time_analysis, f, indent=2)
        else:  # markdown or html
            report = analyzer.generate_performance_report(
                report_type="time_analysis",
                format=format
            )
            with open(file_path, "w") as f:
                f.write(report)
        
        logger.info(f"Saved time-based analysis to {file_path}")
    
    # Generate comprehensive report
    logger.info("Generating comprehensive performance report...")
    comprehensive_report = analyzer.generate_performance_report(
        report_type="comprehensive",
        format=format,
        time_period="90d"
    )
    
    # Save comprehensive report
    if output_dir:
        file_path = os.path.join(output_dir, f"comprehensive_report.{format}")
        with open(file_path, "w") as f:
            f.write(comprehensive_report)
        
        logger.info(f"Saved comprehensive report to {file_path}")
    
    logger.info("Performance analysis complete!")
    
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run Result Aggregator Performance Analyzer")
    parser.add_argument("--db-path", default="./test_performance_analyzer.duckdb", help="Path to DuckDB database")
    parser.add_argument("--output-dir", default="./performance_reports", help="Directory to store generated reports")
    parser.add_argument("--generate-data", action="store_true", help="Generate sample data")
    parser.add_argument("--num-results", type=int, default=500, help="Number of sample results to generate")
    parser.add_argument("--format", choices=["json", "markdown", "html"], default="markdown", help="Report format")
    parser.add_argument("--mode", choices=["all", "regression", "hardware", "efficiency", "time"], default="all", help="Analysis mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize the service
        service = ResultAggregatorService(
            db_path=args.db_path,
            enable_ml=True,
            enable_visualization=True
        )
        
        # Initialize the analyzer
        analyzer = PerformanceAnalyzer(service)
        
        # Generate sample data if requested
        if args.generate_data:
            generate_sample_data(
                service,
                num_results=args.num_results,
                with_trend=True,
                with_regression=True,
                with_hardware_diversity=True
            )
        
        # Run performance analysis
        run_performance_analysis(service, analyzer, args)
        
    except Exception as e:
        logger.error(f"Error running performance analyzer: {e}", exc_info=True)
    finally:
        if 'service' in locals():
            service.close()
            logger.info("Service closed")

if __name__ == "__main__":
    main()