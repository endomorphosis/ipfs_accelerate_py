#!/usr/bin/env python3
"""
End-to-End Test Runner for the Enhanced Visualization Dashboard with Regression Detection.

This script provides a comprehensive end-to-end test of the regression detection functionality:
1. Sets up a temporary test database with performance data containing known regressions
2. Launches the dashboard with regression detection enabled
3. Tests the regression detection workflow interactively
4. Exports regression reports and visualizations
5. Verifies all functionality works as expected

Usage:
    python run_regression_detection_e2e_test.py [--port PORT] [--no-browser] [--debug]
"""

import os
import sys
import argparse
import logging
import tempfile
import time
import webbrowser
import signal
import subprocess
import threading
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_regression_detection_e2e_test")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import necessary components
try:
    from duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard import EnhancedVisualizationDashboard
    from duckdb_api.distributed_testing.dashboard.regression_detection import RegressionDetector
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
    HAS_REQUIRED_COMPONENTS = True
except ImportError as e:
    logger.error(f"Error importing required components: {e}")
    HAS_REQUIRED_COMPONENTS = False


def generate_performance_data_with_regressions(output_path):
    """Generate performance data with known regressions and save to a temporary DuckDB file."""
    logger.info("Generating performance data with known regressions...")
    np.random.seed(42)  # For reproducibility
    
    # Create date range for 100 days
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    
    # Create model and hardware combinations
    models = ['bert-base', 'gpt2-medium', 'vit-base']
    hardware = ['cpu', 'cuda', 'webgpu']
    
    rows = []
    
    # Generate data with regressions
    logger.info("Creating data for multiple models and hardware combinations...")
    for model in models:
        for hw in hardware:
            # Base values for different metrics
            if hw == 'cpu':
                base_latency = 500  # ms
                base_throughput = 10  # items/sec
                base_memory = 2000  # MB
            elif hw == 'cuda':
                base_latency = 100  # ms
                base_throughput = 50  # items/sec
                base_memory = 3000  # MB
            else:  # webgpu
                base_latency = 200  # ms
                base_throughput = 30  # items/sec
                base_memory = 2500  # MB
                
            # Adjust for model size
            if model == 'bert-base':
                latency_factor = 1.0
                throughput_factor = 1.0
                memory_factor = 1.0
            elif model == 'gpt2-medium':
                latency_factor = 1.5
                throughput_factor = 0.8
                memory_factor = 1.3
            else:  # vit-base
                latency_factor = 0.9
                throughput_factor = 1.2
                memory_factor = 0.8
            
            # Generate metrics for each day
            for i, date in enumerate(dates):
                # Add sudden regression for specific model/hardware at day 30
                if i == 30 and model == 'bert-base' and hw == 'webgpu':
                    logger.info(f"Injecting sudden regression at day 30 for {model} on {hw}")
                    # 25% increase in latency
                    latency_factor *= 1.25
                    # 15% decrease in throughput
                    throughput_factor *= 0.85
                
                # Add gradual regression for specific model/hardware starting at day 60
                if i >= 60 and model == 'gpt2-medium' and hw == 'cuda':
                    if i == 60:
                        logger.info(f"Starting gradual regression at day 60 for {model} on {hw}")
                    # 0.5% increase in memory usage per day
                    memory_factor *= 1.005
                
                # Calculate values with some noise
                latency = base_latency * latency_factor * (1 + np.random.normal(0, 0.05))
                throughput = base_throughput * throughput_factor * (1 + np.random.normal(0, 0.05))
                memory = base_memory * memory_factor * (1 + np.random.normal(0, 0.03))
                
                # Ensure valid values
                latency = max(10, latency)
                throughput = max(1, throughput)
                memory = max(100, memory)
                
                # Add row to data
                rows.append({
                    'timestamp': date,
                    'model_name': model,
                    'hardware_type': hw,
                    'batch_size': 1,
                    'latency_ms': latency,
                    'throughput_items_per_second': throughput,
                    'memory_usage_mb': memory,
                    'test_id': f"{model}_{hw}_{i}",
                    'success': True
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Create a temporary DuckDB database
    logger.info(f"Creating DuckDB database at {output_path}...")
    db_api = BenchmarkDBAPI(db_path=output_path, create_if_missing=True)
    
    # Create benchmark_results table if it doesn't exist
    db_api.execute("""
    CREATE TABLE IF NOT EXISTS benchmark_results (
        timestamp TIMESTAMP,
        model_name VARCHAR,
        hardware_type VARCHAR,
        batch_size INTEGER,
        latency_ms DOUBLE,
        throughput_items_per_second DOUBLE,
        memory_usage_mb DOUBLE,
        test_id VARCHAR,
        success BOOLEAN
    )
    """)
    
    # Insert the generated data
    db_api.insert_dataframe(df, "benchmark_results")
    
    # Check that the data was inserted
    count = db_api.execute_to_df("SELECT COUNT(*) FROM benchmark_results").iloc[0, 0]
    logger.info(f"Inserted {count} rows of performance data with known regressions")
    
    return count


def run_dashboard(db_path, output_dir, port, debug=False, open_browser=True):
    """Run the Enhanced Visualization Dashboard with regression detection enabled."""
    logger.info(f"Starting Enhanced Visualization Dashboard on port {port}...")
    
    # Create a database connection
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Create dashboard with regression detection
    dashboard = EnhancedVisualizationDashboard(
        db_conn=db_api,
        output_dir=output_dir,
        enable_regression_detection=True,
        debug=debug
    )
    
    # Configure host and port
    host = '0.0.0.0' if debug else 'localhost'
    
    # Display information
    logger.info(f"Dashboard URL: http://{host}:{port}")
    logger.info("Known regressions in test data:")
    logger.info("1. Sudden regression at day 30 (2025-01-31) for bert-base on webgpu")
    logger.info("   - 25% increase in latency")
    logger.info("   - 15% decrease in throughput")
    logger.info("2. Gradual regression starting at day 60 (2025-03-01) for gpt2-medium on cuda")
    logger.info("   - 0.5% daily increase in memory usage")
    logger.info("\nRegression Detection Test Instructions:")
    logger.info("1. Navigate to the 'Regression Detection' tab")
    logger.info("2. Select 'latency_ms' from the dropdown and filter for 'bert-base' model and 'webgpu' hardware")
    logger.info("3. Run regression analysis and observe the detected regression around Jan 31, 2025")
    logger.info("4. Run correlation analysis to see relationships with other metrics")
    logger.info("5. Repeat with 'throughput_items_per_second' to see inverse correlation")
    logger.info("6. Test gradual regression detection by selecting 'memory_usage_mb' with 'gpt2-medium' model and 'cuda' hardware")
    logger.info("\nPress Ctrl+C to stop the dashboard when testing is complete")
    
    # Open browser if requested
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    
    # Run the dashboard
    dashboard.run_server(
        host=host,
        port=port,
        debug=debug
    )


def verify_regression_detection(db_path, output_dir):
    """Verify regression detection works correctly with the test data."""
    logger.info("Verifying regression detection functionality...")
    
    # Create a database connection
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Create a regression detector
    detector = RegressionDetector()
    
    # Test scenarios to verify
    scenarios = [
        {
            'description': 'Sudden latency regression for bert-base on webgpu',
            'query': """
                SELECT timestamp, latency_ms as value
                FROM benchmark_results
                WHERE model_name = 'bert-base'
                AND hardware_type = 'webgpu'
                ORDER BY timestamp
            """,
            'expected_change_point': 30,  # We injected a regression at day 30
            'output_file': 'bert_webgpu_latency_regression.html'
        },
        {
            'description': 'Sudden throughput regression for bert-base on webgpu',
            'query': """
                SELECT timestamp, throughput_items_per_second as value
                FROM benchmark_results
                WHERE model_name = 'bert-base'
                AND hardware_type = 'webgpu'
                ORDER BY timestamp
            """,
            'expected_change_point': 30,  # We injected a regression at day 30
            'output_file': 'bert_webgpu_throughput_regression.html'
        },
        {
            'description': 'Gradual memory regression for gpt2-medium on cuda',
            'query': """
                SELECT timestamp, memory_usage_mb as value
                FROM benchmark_results
                WHERE model_name = 'gpt2-medium'
                AND hardware_type = 'cuda'
                ORDER BY timestamp
            """,
            'expected_change_point': 60,  # We injected a regression at day 60
            'output_file': 'gpt2_cuda_memory_regression.html'
        }
    ]
    
    # Test each scenario
    results = {}
    for scenario in scenarios:
        logger.info(f"Testing: {scenario['description']}")
        
        # Query the data
        data = db_api.execute_to_df(scenario['query'])
        
        # Run regression detection
        regression_results = detector.detect_regressions(
            data,
            timestamp_col='timestamp',
            value_col='value'
        )
        
        # Check if the expected change point was detected
        expected_cp = scenario['expected_change_point']
        detected = False
        closest_cp = None
        min_distance = float('inf')
        
        for cp in regression_results['change_points']:
            distance = abs(cp - expected_cp)
            if distance < min_distance:
                min_distance = distance
                closest_cp = cp
                
            if distance <= 5:  # Within 5 days of expected change point
                detected = True
                break
        
        # Generate visualization
        fig = detector.create_regression_visualization(
            data,
            regression_results,
            timestamp_col='timestamp',
            value_col='value',
            title=f"Regression Detection: {scenario['description']}"
        )
        
        # Save visualization
        output_path = os.path.join(output_dir, scenario['output_file'])
        fig.write_html(output_path)
        
        # Store results
        results[scenario['description']] = {
            'expected_change_point': expected_cp,
            'closest_detected_change_point': closest_cp,
            'detection_successful': detected,
            'visualization_path': output_path,
            'num_regressions': len(regression_results['regressions']),
            'num_change_points': len(regression_results['change_points'])
        }
        
        if detected:
            logger.info(f"✅ Successfully detected regression near day {expected_cp}")
        else:
            logger.warning(f"❌ Failed to detect regression near day {expected_cp}. Closest point found at day {closest_cp}")
    
    # Summary
    logger.info("\nVerification Summary:")
    success_count = sum(1 for r in results.values() if r['detection_successful'])
    logger.info(f"Successfully detected {success_count} out of {len(scenarios)} injected regressions")
    
    for desc, result in results.items():
        status = "✅ Detected" if result['detection_successful'] else "❌ Not detected"
        logger.info(f"{status}: {desc}")
        logger.info(f"  - Expected change at day: {result['expected_change_point']}")
        logger.info(f"  - Closest detected change at day: {result['closest_detected_change_point']}")
        logger.info(f"  - Visualization saved to: {result['visualization_path']}")
        logger.info(f"  - Number of regressions detected: {result['num_regressions']}")
    
    return success_count == len(scenarios)


def main():
    """Main function to run the end-to-end test."""
    if not HAS_REQUIRED_COMPONENTS:
        logger.error("Required components not available. Please check your installation.")
        return 1
    
    parser = argparse.ArgumentParser(description="Run end-to-end test for regression detection")
    parser.add_argument("--port", type=int, default=8082, help="Port to run the dashboard on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verify-only", action="store_true", help="Verify regression detection without running dashboard")
    parser.add_argument("--output-dir", help="Output directory for visualizations (default: temporary directory)")
    parser.add_argument("--db-path", help="Path to DuckDB database (default: temporary file)")
    
    args = parser.parse_args()
    
    # Create a temporary directory if not specified
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        cleanup_temp_dir = False
    else:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name
        cleanup_temp_dir = True
    
    # Create a temporary database if not specified
    if args.db_path:
        db_path = args.db_path
        cleanup_temp_db = False
    else:
        db_path = os.path.join(output_dir, "test_benchmark.duckdb")
        cleanup_temp_db = True
    
    try:
        # Generate test data if database doesn't exist or is empty
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            generate_performance_data_with_regressions(db_path)
        
        # Verify regression detection
        if args.verify_only:
            success = verify_regression_detection(db_path, output_dir)
            return 0 if success else 1
        
        # Run the dashboard with regression detection enabled
        run_dashboard(
            db_path=db_path,
            output_dir=output_dir,
            port=args.port,
            debug=args.debug,
            open_browser=not args.no_browser
        )
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
        return 0
    
    except Exception as e:
        logger.error(f"Error in end-to-end test: {e}", exc_info=True)
        return 1
    
    finally:
        # Clean up temporary files if we created them
        if cleanup_temp_dir and 'temp_dir' in locals():
            temp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(main())