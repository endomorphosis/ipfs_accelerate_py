#!/usr/bin/env python3
"""
Test script for the RegressionDetector class and its integration with the EnhancedVisualizationDashboard.

This script tests the statistical regression detection capabilities, including:
- Change point detection
- Statistical significance testing
- Severity classification
- Visualization generation
- Correlation analysis
"""

import os
import sys
import unittest
import tempfile
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from unittest import mock
import scipy.stats as stats

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_regression_detection")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import dependencies conditionally to handle missing dependencies
try:
    from duckdb_api.distributed_testing.dashboard.regression_detection import RegressionDetector
    HAS_REGRESSION_DETECTION = True
except ImportError as e:
    logger.error(f"Error importing RegressionDetector: {e}")
    HAS_REGRESSION_DETECTION = False

try:
    from duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard import EnhancedVisualizationDashboard
    HAS_DASHBOARD = True
except ImportError as e:
    logger.error(f"Error importing EnhancedVisualizationDashboard: {e}")
    HAS_DASHBOARD = False

try:
    import ruptures
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    logger.warning("Ruptures package not available, some tests will be skipped.")


def generate_test_time_series_with_regression(length=100, change_point=50, before_mean=100, after_mean=120, noise=5.0):
    """Generate a test time series with a regression at the specified change point."""
    np.random.seed(42)  # For reproducibility
    
    # Generate data before and after change point
    before = np.random.normal(loc=before_mean, scale=noise, size=change_point)
    after = np.random.normal(loc=after_mean, scale=noise, size=length-change_point)
    
    # Combine to single series
    data = np.concatenate([before, after])
    
    # Create timestamp index
    timestamps = pd.date_range(start='2025-01-01', periods=length, freq='D')
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': data
    })
    
    return df, change_point


def generate_random_metrics_dataframe(length=100, metrics=3, with_correlations=True):
    """Generate random metrics data with optional correlations between them."""
    np.random.seed(42)  # For reproducibility
    
    # Create timestamp index
    timestamps = pd.date_range(start='2025-01-01', periods=length, freq='D')
    
    # Create base signals
    base_signal_1 = np.random.normal(loc=100, scale=10, size=length)
    base_signal_2 = np.random.normal(loc=200, scale=20, size=length)
    
    # Create DataFrame with timestamps
    df = pd.DataFrame({
        'timestamp': timestamps
    })
    
    # Add metrics with correlations if requested
    metric_names = []
    for i in range(metrics):
        if with_correlations and i < 2:
            # First two metrics are correlated with base signals
            if i == 0:
                values = base_signal_1 + np.random.normal(loc=0, scale=5, size=length)
                name = 'latency_ms'
            else:
                # Negative correlation with first signal
                values = 300 - 0.9 * base_signal_1 + np.random.normal(loc=0, scale=5, size=length)
                name = 'throughput_items_per_second'
        else:
            # Other metrics are random
            values = np.random.normal(loc=100 * (i+1), scale=10 * (i+1), size=length)
            if i == 2:
                name = 'memory_usage_mb'
            elif i == 3:
                name = 'cpu_usage_percent'
            else:
                name = f'metric_{i}'
                
        df[name] = values
        metric_names.append(name)
    
    return df, metric_names


class TestRegressionDetector(unittest.TestCase):
    """Test the RegressionDetector class functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        if not HAS_REGRESSION_DETECTION:
            raise unittest.SkipTest("RegressionDetector not available, skipping tests.")
        
        # Create a RegressionDetector instance
        cls.detector = RegressionDetector()
        
        # Generate test data
        cls.test_data, cls.change_point = generate_test_time_series_with_regression()
        cls.metrics_data, cls.metric_names = generate_random_metrics_dataframe()
        
        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()
    
    def test_initialize_regression_detector(self):
        """Test that the RegressionDetector can be initialized properly."""
        detector = RegressionDetector()
        self.assertIsNotNone(detector)
        self.assertIsInstance(detector.config, dict)
        self.assertIn('min_samples', detector.config)
        self.assertIn('metrics_config', detector.config)
    
    def test_detect_regressions_simple(self):
        """Test basic regression detection on a simple time series."""
        # Set up test data
        df = self.test_data.copy()
        
        # Run regression detection
        results = self.detector.detect_regressions(
            df, 
            timestamp_col='timestamp',
            value_col='value'
        )
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIn('regressions', results)
        self.assertIn('change_points', results)
        
        # Check for the known change point
        if results['change_points']:
            # The detected change point should be near the actual one
            detected = results['change_points'][0]
            # Allow some tolerance since change point detection isn't always exact
            self.assertLess(abs(detected - self.change_point), 5)
    
    def test_statistical_significance(self):
        """Test that statistical significance is computed correctly."""
        # Set up test data with large difference for clear significance
        df, _ = generate_test_time_series_with_regression(
            before_mean=100, 
            after_mean=150,  # Large difference for clear significance
            noise=5.0
        )
        
        # Run regression detection
        results = self.detector.detect_regressions(
            df, 
            timestamp_col='timestamp',
            value_col='value'
        )
        
        # Verify results contain statistical significance data
        self.assertIn('regressions', results)
        if results['regressions']:
            regression = results['regressions'][0]
            self.assertIn('p_value', regression)
            self.assertIn('significant', regression)
            self.assertIn('confidence_interval', regression)
            
            # With these parameters, the regression should be significant
            self.assertTrue(regression['significant'])
            self.assertLess(regression['p_value'], 0.05)
    
    def test_severity_classification(self):
        """Test that severity classification works correctly."""
        # Set up test data with different magnitudes of change
        test_cases = [
            # (change percentage, expected severity)
            (5.0, 'low'),
            (15.0, 'medium'),
            (25.0, 'high'),
            (35.0, 'critical')
        ]
        
        for change_pct, expected_severity in test_cases:
            # Calculate before and after means to achieve the desired change percentage
            before_mean = 100
            after_mean = before_mean * (1 + change_pct/100)
            
            df, _ = generate_test_time_series_with_regression(
                before_mean=before_mean,
                after_mean=after_mean,
                noise=2.0  # Low noise for clearer detection
            )
            
            # Run regression detection
            results = self.detector.detect_regressions(
                df, 
                timestamp_col='timestamp',
                value_col='value'
            )
            
            # Verify severity classification
            if results['regressions']:
                regression = results['regressions'][0]
                self.assertIn('severity', regression)
                self.assertEqual(regression['severity'], expected_severity,
                                f"Expected severity {expected_severity} for {change_pct}% change, got {regression['severity']}")
    
    @unittest.skipIf(not HAS_RUPTURES, "Ruptures package not available")
    def test_change_point_detection_with_ruptures(self):
        """Test change point detection using ruptures package."""
        # Set up test data with clear change point
        df, change_point = generate_test_time_series_with_regression(
            length=200,
            change_point=100,
            before_mean=100,
            after_mean=150,
            noise=3.0
        )
        
        # Create detector with ruptures configuration
        detector = RegressionDetector()
        detector.config['change_point_model'] = 'l2'
        detector.config['change_point_penalty'] = 10
        
        # Run regression detection
        results = detector.detect_regressions(
            df, 
            timestamp_col='timestamp',
            value_col='value'
        )
        
        # Verify change point detection results
        self.assertIn('change_points', results)
        if results['change_points']:
            # The detected change point should be near the actual one
            detected = results['change_points'][0]
            # Allow some tolerance since change point detection isn't always exact
            self.assertLess(abs(detected - change_point), 10)
    
    def test_visualization_generation(self):
        """Test that regression visualizations can be generated."""
        # Set up test data
        df = self.test_data.copy()
        
        # Run regression detection
        results = self.detector.detect_regressions(
            df, 
            timestamp_col='timestamp',
            value_col='value'
        )
        
        # Generate visualization
        fig = self.detector.create_regression_visualization(
            df,
            results,
            timestamp_col='timestamp',
            value_col='value',
            title='Test Regression Visualization'
        )
        
        # Verify visualization was generated
        self.assertIsNotNone(fig)
        
        # Check that the figure can be saved
        output_path = os.path.join(self.temp_dir.name, 'regression_viz.html')
        fig.write_html(output_path)
        self.assertTrue(os.path.exists(output_path))
    
    def test_correlation_analysis(self):
        """Test correlation analysis between metrics."""
        # Set up test data with correlated metrics
        df, metric_names = self.metrics_data, self.metric_names
        
        # Run correlation analysis
        corr_results = self.detector.create_correlation_analysis(
            df,
            metric_cols=metric_names,
            method='pearson'
        )
        
        # Verify correlation analysis results
        self.assertIsNotNone(corr_results)
        self.assertIn('correlation_matrix', corr_results)
        self.assertIn('correlation_fig', corr_results)
        
        # Check correlation values - first two metrics should be correlated
        if len(metric_names) >= 2:
            corr_matrix = corr_results['correlation_matrix']
            # First two metrics should have negative correlation
            self.assertIsNotNone(corr_matrix)
            corr_value = corr_matrix.loc[metric_names[0], metric_names[1]]
            # Should be negative correlation
            self.assertLess(corr_value, 0)
    
    def test_generate_regression_report(self):
        """Test generating a comprehensive regression report."""
        # Set up test data
        df = self.test_data.copy()
        
        # Run regression detection
        results = self.detector.detect_regressions(
            df, 
            timestamp_col='timestamp',
            value_col='value'
        )
        
        # Generate report
        report = self.detector.generate_regression_report(
            df,
            results,
            timestamp_col='timestamp',
            value_col='value',
            metric_name='Test Metric'
        )
        
        # Verify report structure
        self.assertIsNotNone(report)
        self.assertIn('metric_name', report)
        self.assertIn('num_regressions', report)
        self.assertIn('regression_details', report)
        self.assertIn('summary', report)
        
        # Check report content
        self.assertEqual(report['metric_name'], 'Test Metric')
        self.assertGreaterEqual(report['num_regressions'], 0)
        
        # If regressions were detected, verify details
        if report['num_regressions'] > 0:
            regression = report['regression_details'][0]
            self.assertIn('timestamp', regression)
            self.assertIn('percent_change', regression)
            self.assertIn('before_value', regression)
            self.assertIn('after_value', regression)
            self.assertIn('severity', regression)
            self.assertIn('significant', regression)


class TestEnhancedVisualizationDashboardIntegration(unittest.TestCase):
    """Test the integration of RegressionDetector with EnhancedVisualizationDashboard."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        if not HAS_REGRESSION_DETECTION or not HAS_DASHBOARD:
            raise unittest.SkipTest("Required components not available, skipping integration tests.")
        
        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()
        
        # Generate test data for multiple metrics
        cls.metrics_data, cls.metric_names = generate_random_metrics_dataframe(
            length=100, 
            metrics=4, 
            with_correlations=True
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()
    
    def test_dashboard_initialization_with_regression_detector(self):
        """Test that EnhancedVisualizationDashboard initializes with RegressionDetector."""
        # Mock database connection
        mock_db_conn = mock.MagicMock()
        
        # Create dashboard with regression detector
        dashboard = EnhancedVisualizationDashboard(
            db_conn=mock_db_conn,
            output_dir=self.temp_dir.name,
            enable_regression_detection=True
        )
        
        # Verify regression detector was initialized
        self.assertTrue(hasattr(dashboard, 'regression_detector'))
        self.assertIsInstance(dashboard.regression_detector, RegressionDetector)
    
    def test_dashboard_regression_detection_callback(self):
        """Test the dashboard regression detection callback functionality."""
        # Mock the database connection
        mock_db_conn = mock.MagicMock()
        
        # Configure mock to return our test data when queried
        def mock_execute_to_df(query):
            return self.metrics_data
        
        mock_db_conn.execute_to_df.side_effect = mock_execute_to_df
        
        # Create dashboard with regression detector
        dashboard = EnhancedVisualizationDashboard(
            db_conn=mock_db_conn,
            output_dir=self.temp_dir.name,
            enable_regression_detection=True
        )
        
        # Mock the callbacks that would normally interact with the Dash app
        with mock.patch.object(dashboard, '_get_regression_data_for_metric') as mock_get_data:
            mock_get_data.return_value = self.metrics_data
            
            # Call the regression analysis callback directly
            result = dashboard._regression_analysis_callback(
                metric_name=self.metric_names[0],
                confidence_level=0.95,
                min_change_percent=5.0
            )
            
            # Verify callback returned the expected components
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 5)  # Should return 5 components for the UI
            
            # Check that the figure was generated (it's the first component)
            self.assertIsNotNone(result[0])
            
            # Check that the results table has data (it's the second component)
            self.assertIsNotNone(result[1])
            
            # Store results for the correlation test
            dashboard._last_regression_results = {
                'metric': self.metric_names[0],
                'results': {'regressions': [{'timestamp': '2025-01-15', 'percent_change': 15.0}]}
            }
            
            # Test the correlation analysis callback
            corr_result = dashboard._correlation_analysis_callback(
                n_clicks=1,
                regression_detected=True
            )
            
            # Verify correlation result
            self.assertIsInstance(corr_result, tuple)
            self.assertEqual(len(corr_result), 2)  # Should return 2 components
            self.assertIsNotNone(corr_result[0])  # Correlation figure
            self.assertIsNotNone(corr_result[1])  # Insights text
    
    def test_dashboard_data_caching(self):
        """Test that the dashboard properly caches regression analysis results."""
        # Mock database connection
        mock_db_conn = mock.MagicMock()
        
        # Configure mock to return our test data when queried
        def mock_execute_to_df(query):
            return self.metrics_data
        
        mock_db_conn.execute_to_df.side_effect = mock_execute_to_df
        
        # Create dashboard with regression detector
        dashboard = EnhancedVisualizationDashboard(
            db_conn=mock_db_conn,
            output_dir=self.temp_dir.name,
            enable_regression_detection=True
        )
        
        # Test that the cache starts empty
        self.assertIsNone(dashboard._last_regression_results)
        
        # Mock the dashboard's data retrieval function
        with mock.patch.object(dashboard, '_get_regression_data_for_metric') as mock_get_data:
            mock_get_data.return_value = self.metrics_data
            
            # Call the regression analysis callback
            dashboard._regression_analysis_callback(
                metric_name=self.metric_names[0],
                confidence_level=0.95,
                min_change_percent=5.0
            )
            
            # Verify results were cached
            self.assertIsNotNone(dashboard._last_regression_results)
            self.assertIn('metric', dashboard._last_regression_results)
            self.assertEqual(dashboard._last_regression_results['metric'], self.metric_names[0])
            self.assertIn('results', dashboard._last_regression_results)
    
    def test_dashboard_layout_generation(self):
        """Test that the dashboard generates the regression detection layout correctly."""
        # Mock database connection
        mock_db_conn = mock.MagicMock()
        
        # Create dashboard with regression detector
        dashboard = EnhancedVisualizationDashboard(
            db_conn=mock_db_conn,
            output_dir=self.temp_dir.name,
            enable_regression_detection=True
        )
        
        # Get the regression detection tab layout
        with mock.patch.object(dashboard, '_get_available_metrics') as mock_get_metrics:
            mock_get_metrics.return_value = self.metric_names
            
            layout = dashboard._create_regression_detection_tab()
            
            # Check that the layout contains essential components
            self.assertIsNotNone(layout)
            
            # Convert layout to string representation for easier checking
            layout_str = str(layout)
            
            # Check for key elements in the layout
            self.assertIn('regression-metric-dropdown', layout_str)
            self.assertIn('confidence-level-slider', layout_str)
            self.assertIn('min-change-percent-input', layout_str)
            self.assertIn('run-regression-analysis-button', layout_str)
            self.assertIn('regression-results-graph', layout_str)
            self.assertIn('regression-results-table', layout_str)
            self.assertIn('run-correlation-analysis-button', layout_str)
    
    def test_graceful_degradation_when_missing_dependencies(self):
        """Test that the dashboard gracefully degrades when regression detection dependencies are missing."""
        # Mock database connection
        mock_db_conn = mock.MagicMock()
        
        # Mock the import check to simulate missing dependencies
        with mock.patch.dict('sys.modules', {'ruptures': None}):
            with mock.patch('duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard.RUPTURES_AVAILABLE', False):
                # Create dashboard with regression detector enabled
                dashboard = EnhancedVisualizationDashboard(
                    db_conn=mock_db_conn,
                    output_dir=self.temp_dir.name,
                    enable_regression_detection=True
                )
                
                # Verify dashboard still initializes
                self.assertIsNotNone(dashboard)
                
                # It should still have a regression detector
                self.assertTrue(hasattr(dashboard, 'regression_detector'))
                
                # Get the regression detection tab layout
                with mock.patch.object(dashboard, '_get_available_metrics') as mock_get_metrics:
                    mock_get_metrics.return_value = self.metric_names
                    
                    layout = dashboard._create_regression_detection_tab()
                    
                    # Check that the layout contains warning about missing dependencies
                    layout_str = str(layout)
                    self.assertIn('Some advanced features may be limited', layout_str)


if __name__ == "__main__":
    unittest.main()