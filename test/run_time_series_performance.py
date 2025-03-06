#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner script for Time-Series Performance Tracking

This script demonstrates the usage of time-series performance tracking 
functionality from the next steps implementation.

Author: IPFS Accelerate Python Framework Team
Date: March 2025
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the time-series performance module
from time_series_performance import TimeSeriesPerformance

def run_quick_test(db_path=None):
    """Run a quick test of the time-series performance tracker."""
    # Use temporary database if none provided
    if db_path is None:
        db_file = tempfile.mktemp(suffix='.duckdb')
        logger.info(f"Using temporary database at {db_file}")
        db_path = db_file
    
    # Initialize the time-series performance tracking system
    ts_perf = TimeSeriesPerformance(db_path=db_path)
    
    # Generate sample data
    logger.info("Generating sample performance data...")
    # Import the test module for sample data generation
    from test_time_series_performance import generate_sample_performance_data
    generate_sample_performance_data(ts_perf, days=30, samples_per_day=1)
    
    # Set baselines for all model-hardware combinations
    logger.info("Setting performance baselines...")
    baseline_results = ts_perf.set_all_baselines(days_lookback=7, min_samples=2)
    logger.info(f"Set {len([r for r in baseline_results if r['status'] == 'success'])} baselines")
    
    # Detect regressions
    logger.info("Detecting performance regressions...")
    regressions = ts_perf.detect_regressions(days_lookback=14)
    logger.info(f"Detected {len(regressions)} regressions")
    
    if regressions:
        logger.info("Regression details:")
        for reg in regressions:
            logger.info(f"  {reg['model_name']} on {reg['hardware_type']}: "
                      f"{reg['regression_type']} degraded by {reg['severity']:.2f}%")
    
    # Analyze trends
    logger.info("Analyzing performance trends...")
    trends = ts_perf.analyze_trends(metric='throughput', days_lookback=30, min_samples=3)
    logger.info(f"Analyzed {len(trends)} trends")
    
    if trends:
        logger.info("Trend details:")
        for trend in sorted(trends, key=lambda x: x['trend_magnitude'], reverse=True)[:3]:
            logger.info(f"  {trend['model_name']} on {trend['hardware_type']}: "
                       f"{trend['metric']} {trend['trend_direction']} by {trend['trend_magnitude']:.2f}% "
                       f"(confidence: {trend['trend_confidence']:.2f})")
    
    # Generate visualization
    logger.info("Generating performance visualizations...")
    for metric in ['throughput', 'latency', 'memory']:
        viz_path = ts_perf.generate_trend_visualization(
            metric=metric,
            days_lookback=30,
            output_path=f"performance_{metric}_trend.png"
        )
        logger.info(f"Generated {metric} visualization at {viz_path}")
    
    # Generate report
    logger.info("Generating performance report...")
    report_path = ts_perf.export_performance_report(
        days_lookback=30,
        format='markdown',
        output_path="performance_report.md"
    )
    logger.info(f"Generated report at {report_path}")
    
    return db_path

def run_full_test(db_path=None):
    """Run a full test of the time-series performance tracker."""
    logger.info("Running full test suite...")
    # Import and run the test module
    from test_time_series_performance import test_time_series_performance
    test_time_series_performance()
    logger.info("Full test suite completed")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Time Series Performance Runner')
    parser.add_argument('--db-path', help='Database path (optional, will use temp DB if not provided)')
    parser.add_argument('--full-test', action='store_true', help='Run full test suite')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with sample data')
    args = parser.parse_args()
    
    if not (args.full_test or args.quick_test):
        parser.print_help()
        sys.exit(1)
    
    if args.full_test:
        run_full_test(args.db_path)
    
    if args.quick_test:
        db_path = run_quick_test(args.db_path)
        logger.info(f"Quick test completed. Generated files are in the current directory.")
        if not args.db_path:
            logger.info(f"Temporary database was at: {db_path}")

if __name__ == "__main__":
    main()