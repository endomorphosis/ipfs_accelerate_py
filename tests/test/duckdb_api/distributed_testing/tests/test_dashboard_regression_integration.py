#!/usr/bin/env python3
"""
Integration test for the Enhanced Visualization Dashboard with Regression Detection.

This script tests the end-to-end workflow of the dashboard's regression detection functionality:
- Running the dashboard with regression detection enabled
- Executing regression analysis on real performance data
- Visualizing regressions with statistical significance
- Performing correlation analysis between metrics
- Testing dashboard interactions and callbacks
"""

import os
import sys
import unittest
import tempfile
import logging
import numpy as np
import pandas as pd
import time
import threading
from pathlib import Path
from unittest import mock

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_dashboard_regression_integration")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import dependencies conditionally to handle missing dependencies
try:
    from duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard import EnhancedVisualizationDashboard
    from duckdb_api.distributed_testing.dashboard.regression_detection import RegressionDetector
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
    HAS_REQUIRED_COMPONENTS = True
except ImportError as e:
    logger.error(f"Error importing required components: {e}")
    HAS_REQUIRED_COMPONENTS = False

try:
    import dash
    from dash.testing.application_runners import ThreadedRunner
    from dash.testing import wait
    HAS_DASH_TESTING = True
except ImportError:
    HAS_DASH_TESTING = False
    logger.warning("Dash testing components not available, some integration tests will be skipped.")


def generate_performance_data_with_regressions(output_path):
    """Generate performance data with known regressions and save to a temporary DuckDB file."""
    np.random.seed(42)  # For reproducibility
    
    # Create date range for 100 days
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    
    # Create model and hardware combinations
    models = ['bert-base', 'gpt2-medium', 'vit-base']
    hardware = ['cpu', 'cuda', 'webgpu']
    
    rows = []
    
    # Generate data with regressions
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
                    # 25% increase in latency
                    latency_factor *= 1.25
                    # 15% decrease in throughput
                    throughput_factor *= 0.85
                
                # Add gradual regression for specific model/hardware starting at day 60
                if i >= 60 and model == 'gpt2-medium' and hw == 'cuda':
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
    logger.info(f"Inserted {count} rows of performance data with regressions")
    
    return count


class TestDashboardRegressionIntegration(unittest.TestCase):
    """Integration tests for the dashboard with regression detection."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        if not HAS_REQUIRED_COMPONENTS:
            raise unittest.SkipTest("Required components not available, skipping integration tests.")
        
        # Create a temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = os.path.join(cls.temp_dir.name, "dashboard_output")
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Create a temporary database with test data
        cls.db_path = os.path.join(cls.temp_dir.name, "test_benchmark.duckdb")
        cls.row_count = generate_performance_data_with_regressions(cls.db_path)
        
        # Create a database connection for tests
        cls.db_api = BenchmarkDBAPI(db_path=cls.db_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()
    
    def test_run_dashboard_with_regression_detection(self):
        """Test running the dashboard with regression detection enabled."""
        # Create dashboard with regression detection
        dashboard = EnhancedVisualizationDashboard(
            db_conn=self.db_api,
            output_dir=self.output_dir,
            enable_regression_detection=True
        )
        
        # Check that the dashboard was created with regression detection
        self.assertTrue(hasattr(dashboard, 'regression_detector'))
        self.assertIsInstance(dashboard.regression_detector, RegressionDetector)
        
        # Check that the dashboard has all required components
        self.assertIsNotNone(dashboard.app)
        
        # Verify the regression detection tab was added
        tabs = dashboard._create_tabs()
        tab_ids = [tab.id for tab in tabs]
        self.assertIn('tab-regression-detection', tab_ids)
    
    @unittest.skipIf(not HAS_DASH_TESTING, "Dash testing components not available")
    def test_dashboard_regression_workflow(self):
        """Test the end-to-end regression detection workflow using a running dashboard instance."""
        # Create dashboard with regression detection
        dashboard = EnhancedVisualizationDashboard(
            db_conn=self.db_api,
            output_dir=self.output_dir,
            enable_regression_detection=True
        )
        
        # Create a threaded server for testing
        app = dashboard.app
        
        # Configure server
        app.config.suppress_callback_exceptions = True
        
        # Use a mock server to test the workflow
        with mock.patch.object(dashboard, '_get_regression_data_for_metric') as mock_get_data:
            # Set up mock data for testing
            test_metric = 'latency_ms'
            model_filter = 'bert-base'
            hardware_filter = 'webgpu'
            
            # Query the real database for this subset of data
            query = f"""
            SELECT timestamp, {test_metric} as value
            FROM benchmark_results
            WHERE model_name = '{model_filter}'
            AND hardware_type = '{hardware_filter}'
            ORDER BY timestamp
            """
            test_data = self.db_api.execute_to_df(query)
            
            # Configure mock to return our test data
            mock_get_data.return_value = test_data
            
            # Test the regression analysis workflow
            results = dashboard._regression_analysis_callback(
                metric_name=test_metric,
                confidence_level=0.95,
                min_change_percent=5.0,
                model_filter=model_filter,
                hardware_filter=hardware_filter
            )
            
            # Verify results
            self.assertIsInstance(results, tuple)
            self.assertEqual(len(results), 5)  # Should return 5 components
            
            # Check that the figure was generated
            self.assertIsNotNone(results[0])
            
            # Check for regression results table
            self.assertIsNotNone(results[1])
            table_content = str(results[1])
            
            # The table should include regression information
            self.assertIn('Regression Details', table_content)
            
            # Verify that we found the regression we inserted at day 30
            self.assertIn('2025-01-31', table_content)  # Approximate date of the regression
            
            # Check that the dashboard cached the results
            self.assertIsNotNone(dashboard._last_regression_results)
            
            # Test the correlation analysis workflow
            corr_results = dashboard._correlation_analysis_callback(
                n_clicks=1,
                regression_detected=True
            )
            
            # Verify correlation results
            self.assertIsInstance(corr_results, tuple)
            self.assertEqual(len(corr_results), 2)  # Should return 2 components
            
            # Check correlation figure and insights
            self.assertIsNotNone(corr_results[0])  # Correlation figure
            self.assertIsNotNone(corr_results[1])  # Insights text
    
    def test_multiple_metric_regression_analysis(self):
        """Test regression analysis on multiple metrics for the same model/hardware."""
        # Create dashboard with regression detection
        dashboard = EnhancedVisualizationDashboard(
            db_conn=self.db_api,
            output_dir=self.output_dir,
            enable_regression_detection=True
        )
        
        # List of metrics to test
        metrics = ['latency_ms', 'throughput_items_per_second', 'memory_usage_mb']
        model_filter = 'bert-base'
        hardware_filter = 'webgpu'
        
        # Test each metric for regressions
        results_by_metric = {}
        
        for metric in metrics:
            # Query the real database for this subset of data
            query = f"""
            SELECT timestamp, {metric} as value
            FROM benchmark_results
            WHERE model_name = '{model_filter}'
            AND hardware_type = '{hardware_filter}'
            ORDER BY timestamp
            """
            test_data = self.db_api.execute_to_df(query)
            
            # Run regression detection directly
            results = dashboard.regression_detector.detect_regressions(
                test_data,
                timestamp_col='timestamp',
                value_col='value'
            )
            
            # Store results for this metric
            results_by_metric[metric] = results
            
            # Verify that results were returned
            self.assertIsNotNone(results)
            self.assertIn('regressions', results)
        
        # Verify that we found regressions in latency and throughput (known inserted regressions)
        self.assertTrue(len(results_by_metric['latency_ms']['regressions']) > 0)
        self.assertTrue(len(results_by_metric['throughput_items_per_second']['regressions']) > 0)
        
        # There should be at least one change point detected for each of these metrics
        self.assertTrue(len(results_by_metric['latency_ms']['change_points']) > 0)
        self.assertTrue(len(results_by_metric['throughput_items_per_second']['change_points']) > 0)
        
        # The change points should be around day 30 (index 30)
        if results_by_metric['latency_ms']['change_points']:
            change_point = results_by_metric['latency_ms']['change_points'][0]
            self.assertLess(abs(change_point - 30), 5)  # Within 5 days of day 30
    
    def test_regression_detector_integration_with_real_data(self):
        """Test the RegressionDetector with real data from the database."""
        # Create a regression detector
        detector = RegressionDetector()
        
        # Query data for the gradual memory regression we inserted
        query = """
        SELECT timestamp, memory_usage_mb as value
        FROM benchmark_results
        WHERE model_name = 'gpt2-medium'
        AND hardware_type = 'cuda'
        ORDER BY timestamp
        """
        test_data = self.db_api.execute_to_df(query)
        
        # Run regression detection
        results = detector.detect_regressions(
            test_data,
            timestamp_col='timestamp',
            value_col='value'
        )
        
        # Verify the results
        self.assertIsNotNone(results)
        self.assertIn('regressions', results)
        
        # We should find at least one change point (inserted at day 60)
        self.assertGreater(len(results['change_points']), 0)
        
        # Generate visualization
        fig = detector.create_regression_visualization(
            test_data,
            results,
            timestamp_col='timestamp',
            value_col='value',
            title='Memory Usage Regression - gpt2-medium on CUDA'
        )
        
        # Verify visualization was created
        self.assertIsNotNone(fig)
        
        # Save visualization to file
        output_path = os.path.join(self.output_dir, 'memory_regression_visualization.html')
        fig.write_html(output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Generate regression report
        report = detector.generate_regression_report(
            test_data,
            results,
            timestamp_col='timestamp',
            value_col='value',
            metric_name='Memory Usage (MB)'
        )
        
        # Verify report was generated
        self.assertIsNotNone(report)
        self.assertIn('metric_name', report)
        self.assertIn('num_regressions', report)
        self.assertIn('summary', report)
    
    def test_dashboard_config_persistence(self):
        """Test that dashboard configuration for regression detection persists."""
        # Create dashboard with specific regression detection config
        custom_config = {
            'min_samples': 10,
            'window_size': 15,
            'regression_threshold': 3.0,
            'confidence_level': 0.99,
            'severity_thresholds': {
                'critical': 20.0,
                'high': 15.0,
                'medium': 8.0,
                'low': 3.0
            }
        }
        
        dashboard = EnhancedVisualizationDashboard(
            db_conn=self.db_api,
            output_dir=self.output_dir,
            enable_regression_detection=True,
            regression_config=custom_config
        )
        
        # Verify that the configuration was applied
        for key, value in custom_config.items():
            if key != 'severity_thresholds':  # Special case for nested dict
                self.assertEqual(dashboard.regression_detector.config[key], value)
            else:
                for severity, threshold in value.items():
                    self.assertEqual(
                        dashboard.regression_detector.config['severity_thresholds'][severity], 
                        threshold
                    )
        
        # Test that the dashboard can save and load configuration
        config_path = os.path.join(self.output_dir, 'dashboard_config.json')
        
        # Save configuration
        dashboard.save_configuration(config_path)
        self.assertTrue(os.path.exists(config_path))
        
        # Create a new dashboard with the saved configuration
        new_dashboard = EnhancedVisualizationDashboard(
            db_conn=self.db_api,
            output_dir=self.output_dir,
            enable_regression_detection=True,
            config_path=config_path
        )
        
        # Verify that the configuration was loaded
        for key, value in custom_config.items():
            if key != 'severity_thresholds':  # Special case for nested dict
                self.assertEqual(new_dashboard.regression_detector.config[key], value)
            else:
                for severity, threshold in value.items():
                    self.assertEqual(
                        new_dashboard.regression_detector.config['severity_thresholds'][severity], 
                        threshold
                    )


if __name__ == "__main__":
    unittest.main()