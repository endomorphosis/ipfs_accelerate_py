#!/usr/bin/env python3
"""
Test script for the RegressionVisualization class.

This script tests the enhanced regression visualization capabilities, including:
- Interactive time-series regression plots
- Comparative regression visualizations
- Regression heatmaps
- HTML report generation
- Export capabilities
"""

import os
import sys
import unittest
import tempfile
import logging
import numpy as np
import datetime
from pathlib import Path
from unittest import mock

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_regression_visualization")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import dependencies conditionally to handle missing dependencies
try:
    from duckdb_api.distributed_testing.dashboard.regression_visualization import RegressionVisualization
    HAS_REGRESSION_VISUALIZATION = True
except ImportError as e:
    logger.error(f"Error importing RegressionVisualization: {e}")
    HAS_REGRESSION_VISUALIZATION = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    logger.warning("Pandas not available. Some tests will be skipped.")
    HAS_PANDAS = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    logger.warning("Plotly not available. Some tests will be skipped.")
    HAS_PLOTLY = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    logger.warning("SciPy not available. Some tests will be skipped.")
    HAS_SCIPY = False


def generate_time_series_with_regression(length=100, change_point=50, before_mean=100, after_mean=120, noise=5.0):
    """Generate a test time series with a regression at the specified change point."""
    np.random.seed(42)  # For reproducibility
    
    # Generate data before and after change point
    before = np.random.normal(loc=before_mean, scale=noise, size=change_point)
    after = np.random.normal(loc=after_mean, scale=noise, size=length-change_point)
    
    # Combine to single series
    values = np.concatenate([before, after])
    
    # Create timestamp sequence
    start_date = datetime.datetime(2025, 1, 1)
    timestamps = [start_date + datetime.timedelta(days=i) for i in range(length)]
    
    # Convert to strings for consistency
    timestamp_strings = [ts.strftime("%Y-%m-%d") for ts in timestamps]
    
    # Create formatted data
    time_series_data = {
        "timestamps": timestamp_strings,
        "values": values.tolist()
    }
    
    # Create mock regression data
    regression = {
        "change_point_time": timestamp_strings[change_point],
        "before_mean": before_mean,
        "after_mean": after_mean,
        "percentage_change": ((after_mean - before_mean) / before_mean) * 100,
        "p_value": 0.001,
        "is_significant": True,
        "severity": "high",
        "is_regression": True,
        "direction": "increase" if after_mean > before_mean else "decrease"
    }
    
    return time_series_data, [regression]


def generate_multi_metric_data(num_metrics=3, length=100):
    """Generate test data for multiple metrics with some correlated regressions."""
    np.random.seed(42)  # For reproducibility
    
    # Create timestamp sequence
    start_date = datetime.datetime(2025, 1, 1)
    timestamps = [start_date + datetime.timedelta(days=i) for i in range(length)]
    timestamp_strings = [ts.strftime("%Y-%m-%d") for ts in timestamps]
    
    metrics_data = {}
    regressions_by_metric = {}
    
    # Generate metrics with different regression points
    for i in range(num_metrics):
        metric_name = f"metric_{i}"
        if i == 0:
            metric_name = "latency_ms"
            change_point = 30
            before_mean = 100
            after_mean = 120
        elif i == 1:
            metric_name = "throughput_items_per_second"
            change_point = 60
            before_mean = 50
            after_mean = 40
        else:
            metric_name = "memory_usage_mb"
            change_point = 80
            before_mean = 2000
            after_mean = 2200
        
        # Generate values with regression
        before = np.random.normal(loc=before_mean, scale=before_mean*0.05, size=change_point)
        after = np.random.normal(loc=after_mean, scale=after_mean*0.05, size=length-change_point)
        values = np.concatenate([before, after])
        
        # Create time series data
        metrics_data[metric_name] = {
            "timestamps": timestamp_strings,
            "values": values.tolist()
        }
        
        # Create regression
        regression = {
            "change_point_time": timestamp_strings[change_point],
            "before_mean": before_mean,
            "after_mean": after_mean,
            "percentage_change": ((after_mean - before_mean) / before_mean) * 100,
            "p_value": 0.001,
            "is_significant": True,
            "severity": "high" if abs(((after_mean - before_mean) / before_mean) * 100) > 20 else "medium",
            "is_regression": (after_mean > before_mean) if i in [0, 2] else (after_mean < before_mean),
            "direction": "increase" if after_mean > before_mean else "decrease"
        }
        
        regressions_by_metric[metric_name] = [regression]
    
    return metrics_data, regressions_by_metric


class TestRegressionVisualization(unittest.TestCase):
    """Test the RegressionVisualization class functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        if not HAS_REGRESSION_VISUALIZATION:
            raise unittest.SkipTest("RegressionVisualization not available, skipping tests.")
        
        # Create a temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()
        
        # Create visualization instance
        cls.visualizer = RegressionVisualization(output_dir=cls.temp_dir.name)
        
        # Generate test data
        cls.time_series_data, cls.regressions = generate_time_series_with_regression()
        cls.metrics_data, cls.regressions_by_metric = generate_multi_metric_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the RegressionVisualization can be initialized properly."""
        visualizer = RegressionVisualization(output_dir=self.temp_dir.name)
        self.assertIsNotNone(visualizer)
        self.assertEqual(visualizer.output_dir, self.temp_dir.name)
        self.assertEqual(visualizer.theme, "dark")
        self.assertIn("dark", visualizer.color_scheme)
        self.assertIn("light", visualizer.color_scheme)
    
    @unittest.skipIf(not HAS_PLOTLY, "Plotly not available")
    def test_create_interactive_regression_figure(self):
        """Test creating an interactive regression visualization."""
        # Create visualization
        figure_dict = self.visualizer.create_interactive_regression_figure(
            self.time_series_data,
            self.regressions,
            metric="test_metric",
            title="Test Regression Visualization"
        )
        
        # Check that a figure was created
        self.assertIsNotNone(figure_dict)
        self.assertIn("data", figure_dict)
        self.assertIn("layout", figure_dict)
        
        # Check that the title was set
        self.assertEqual(figure_dict["layout"]["title"]["text"], "Test Regression Visualization")
        
        # Check that we have at least one trace
        self.assertGreater(len(figure_dict["data"]), 0)
    
    @unittest.skipIf(not HAS_PLOTLY, "Plotly not available")
    def test_create_comparative_regression_visualization(self):
        """Test creating a comparative regression visualization."""
        # Create visualization
        figure_dict = self.visualizer.create_comparative_regression_visualization(
            self.metrics_data,
            self.regressions_by_metric,
            title="Comparative Regression Analysis"
        )
        
        # Check that a figure was created
        self.assertIsNotNone(figure_dict)
        self.assertIn("data", figure_dict)
        self.assertIn("layout", figure_dict)
        
        # Check that the title was set
        self.assertEqual(figure_dict["layout"]["title"]["text"], "Comparative Regression Analysis")
        
        # Check that we have at least one trace per metric
        expected_traces = len(self.metrics_data) * 2  # Each metric has a line trace and regression markers
        self.assertGreaterEqual(len(figure_dict["data"]), len(self.metrics_data))
    
    @unittest.skipIf(not HAS_PLOTLY, "Plotly not available")
    def test_create_regression_heatmap(self):
        """Test creating a regression heatmap visualization."""
        # Create sample data
        time_ranges = ["Last Week", "Last Month", "Last Quarter"]
        metrics = ["latency_ms", "throughput_items_per_second", "memory_usage_mb"]
        regression_data = {
            "Last Week": {
                "latency_ms": 15.5,
                "throughput_items_per_second": -10.2,
                "memory_usage_mb": 5.3,
                "latency_ms_p_value": 0.001,
                "throughput_items_per_second_p_value": 0.02,
                "memory_usage_mb_p_value": 0.08
            },
            "Last Month": {
                "latency_ms": 20.1,
                "throughput_items_per_second": -15.7,
                "memory_usage_mb": 10.9,
                "latency_ms_p_value": 0.005,
                "throughput_items_per_second_p_value": 0.01,
                "memory_usage_mb_p_value": 0.03
            },
            "Last Quarter": {
                "latency_ms": 25.3,
                "throughput_items_per_second": -20.8,
                "memory_usage_mb": 15.2,
                "latency_ms_p_value": 0.01,
                "throughput_items_per_second_p_value": 0.008,
                "memory_usage_mb_p_value": 0.02
            }
        }
        
        # Create visualization
        figure_dict = self.visualizer.create_regression_heatmap(
            time_ranges,
            metrics,
            regression_data,
            title="Regression Heatmap"
        )
        
        # Check that a figure was created
        self.assertIsNotNone(figure_dict)
        self.assertIn("data", figure_dict)
        self.assertIn("layout", figure_dict)
        
        # Check that the title was set
        self.assertEqual(figure_dict["layout"]["title"]["text"], "Regression Heatmap")
        
        # Check that we have a heatmap trace
        self.assertEqual(figure_dict["data"][0]["type"], "heatmap")
    
    @unittest.skipIf(not HAS_PLOTLY or not HAS_PANDAS, "Plotly or Pandas not available")
    def test_create_regression_summary_report(self):
        """Test generating a comprehensive regression summary report."""
        # Create report
        report_path = self.visualizer.create_regression_summary_report(
            self.metrics_data,
            self.regressions_by_metric,
            include_plots=True
        )
        
        # Check that a report was created
        self.assertIsNotNone(report_path)
        self.assertTrue(os.path.exists(report_path))
        
        # Verify file is HTML
        with open(report_path, 'r') as f:
            content = f.read()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("Regression Analysis Report", content)
            
            # Check that it includes each metric
            for metric in self.metrics_data.keys():
                self.assertIn(metric, content)
    
    @unittest.skipIf(not HAS_PLOTLY, "Plotly not available")
    def test_export_regression_visualization(self):
        """Test exporting a regression visualization to different formats."""
        # Create a visualization to export
        figure_dict = self.visualizer.create_interactive_regression_figure(
            self.time_series_data,
            self.regressions,
            metric="test_metric",
            title="Export Test Visualization"
        )
        
        # Try exporting to HTML
        html_path = self.visualizer.export_regression_visualization(
            figure_dict,
            format="html"
        )
        
        # Check that HTML file was created
        self.assertIsNotNone(html_path)
        self.assertTrue(os.path.exists(html_path))
        self.assertTrue(html_path.endswith(".html"))
        
        # Try exporting to JSON
        json_path = self.visualizer.export_regression_visualization(
            figure_dict,
            format="json"
        )
        
        # Check that JSON file was created
        self.assertIsNotNone(json_path)
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(json_path.endswith(".json"))
    
    def test_theme_switching(self):
        """Test switching between light and dark themes."""
        # Check default theme
        self.assertEqual(self.visualizer.theme, "dark")
        
        # Switch to light theme
        self.visualizer.set_theme("light")
        self.assertEqual(self.visualizer.theme, "light")
        
        # Switch back to dark theme
        self.visualizer.set_theme("dark")
        self.assertEqual(self.visualizer.theme, "dark")
        
        # Test invalid theme
        original_theme = self.visualizer.theme
        self.visualizer.set_theme("invalid_theme")
        self.assertEqual(self.visualizer.theme, original_theme)  # Should keep original theme


if __name__ == "__main__":
    unittest.main()