#!/usr/bin/env python3
"""
Integration test for the Enhanced Visualization UI features.

This script tests the enhanced UI components added to the visualization dashboard:
- Visualization options panel with controls for confidence intervals, trend lines, and annotations
- Export functionality integrated into the visualization options
- Callbacks for handling visualization options
- Theme integration between dashboard and visualizations
"""

import os
import sys
import unittest
import tempfile
import logging
import numpy as np
import pandas as pd
import time
from pathlib import Path
from unittest import mock

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_enhanced_visualization_ui")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import dependencies conditionally to handle missing dependencies
try:
    from duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard import EnhancedVisualizationDashboard
    from duckdb_api.distributed_testing.dashboard.regression_detection import RegressionDetector
    from duckdb_api.distributed_testing.dashboard.regression_visualization import RegressionVisualization
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
    HAS_REQUIRED_COMPONENTS = True
except ImportError as e:
    logger.error(f"Error importing required components: {e}")
    HAS_REQUIRED_COMPONENTS = False

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash.testing.application_runners import ThreadedRunner
    from dash.testing import wait
    HAS_DASH_TESTING = True
except ImportError:
    HAS_DASH_TESTING = False
    logger.warning("Dash testing components not available, some UI tests will be skipped.")


def generate_test_data():
    """Generate test data for visualization testing."""
    np.random.seed(42)  # For reproducibility
    
    # Create date range for 100 days
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    
    # Generate time series with a regression
    base_value = 100.0
    values = []
    
    for i in range(100):
        if i < 50:
            # Before regression
            value = base_value + np.random.normal(0, 5)
        else:
            # After regression (25% higher)
            value = base_value * 1.25 + np.random.normal(0, 5)
        values.append(value)
    
    # Create time series data in the format expected by the visualization
    time_series_data = {
        "timestamps": dates.tolist(),
        "values": values
    }
    
    # Create regression results
    regressions = [{
        "change_point_time": dates[50],
        "metric": "test_metric",
        "display_name": "Test Metric",
        "unit": "ms",
        "before_mean": base_value,
        "after_mean": base_value * 1.25,
        "percentage_change": 25.0,
        "absolute_change": base_value * 0.25,
        "is_regression": True,
        "is_improvement": False,
        "p_value": 0.001,
        "is_significant": True,
        "significance": 0.999,
        "severity": "high",
        "direction": "increase"
    }]
    
    return time_series_data, regressions


class TestEnhancedVisualizationUI(unittest.TestCase):
    """Integration tests for the enhanced visualization UI components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        if not HAS_REQUIRED_COMPONENTS:
            raise unittest.SkipTest("Required components not available, skipping UI tests.")
        
        # Create a temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = os.path.join(cls.temp_dir.name, "dashboard_output")
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Generate test data
        cls.time_series_data, cls.regressions = generate_test_data()
        
        # Initialize regression visualization
        cls.regression_visualization = RegressionVisualization(output_dir=cls.output_dir)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()
    
    def test_create_interactive_regression_figure_with_options(self):
        """Test creating interactive regression figure with different visualization options."""
        # Test with all options enabled
        fig_dict_all = self.regression_visualization.create_interactive_regression_figure(
            self.time_series_data,
            self.regressions,
            "test_metric",
            title="Test Visualization - All Options",
            include_annotations=True,
            include_confidence_intervals=True,
            include_trend_lines=True
        )
        
        self.assertIsNotNone(fig_dict_all)
        self.assertIn("data", fig_dict_all)
        self.assertIn("layout", fig_dict_all)
        
        # Check for annotations (they should be present)
        self.assertIn("annotations", fig_dict_all["layout"])
        self.assertTrue(len(fig_dict_all["layout"]["annotations"]) > 0)
        
        # Test with no options
        fig_dict_none = self.regression_visualization.create_interactive_regression_figure(
            self.time_series_data,
            self.regressions,
            "test_metric",
            title="Test Visualization - No Options",
            include_annotations=False,
            include_confidence_intervals=False,
            include_trend_lines=False
        )
        
        self.assertIsNotNone(fig_dict_none)
        
        # Check that annotations are not present or empty
        self.assertIn("annotations", fig_dict_none["layout"])
        self.assertEqual(len(fig_dict_none["layout"]["annotations"]), 0)
        
        # Count traces to verify confidence intervals and trend lines
        self.assertGreater(len(fig_dict_all["data"]), len(fig_dict_none["data"]))
    
    def test_theme_integration(self):
        """Test theme integration between dashboard and visualization."""
        # Test dark theme
        self.regression_visualization.set_theme("dark")
        fig_dark = self.regression_visualization.create_interactive_regression_figure(
            self.time_series_data,
            self.regressions,
            "test_metric",
            title="Dark Theme Test"
        )
        
        self.assertIsNotNone(fig_dark)
        self.assertEqual(fig_dark["layout"]["template"]["layout"]["template"]["data"], "plotly_dark")
        
        # Test light theme
        self.regression_visualization.set_theme("light")
        fig_light = self.regression_visualization.create_interactive_regression_figure(
            self.time_series_data,
            self.regressions,
            "test_metric",
            title="Light Theme Test"
        )
        
        self.assertIsNotNone(fig_light)
        self.assertEqual(fig_light["layout"]["template"]["layout"]["template"]["data"], "plotly_white")
    
    def test_export_format_support(self):
        """Test support for different export formats."""
        # Create a figure for testing
        fig_dict = self.regression_visualization.create_interactive_regression_figure(
            self.time_series_data,
            self.regressions,
            "test_metric",
            title="Export Format Test"
        )
        
        self.assertIsNotNone(fig_dict)
        
        # Convert back to a figure object for export testing
        fig = self.regression_visualization._dict_to_figure(fig_dict)
        
        # Test HTML export
        html_path = os.path.join(self.output_dir, "test_export.html")
        fig.write_html(html_path)
        self.assertTrue(os.path.exists(html_path))
        
        # Test PNG export
        png_path = os.path.join(self.output_dir, "test_export.png")
        fig.write_image(png_path)
        self.assertTrue(os.path.exists(png_path))
        
        # Test SVG export
        svg_path = os.path.join(self.output_dir, "test_export.svg")
        fig.write_image(svg_path)
        self.assertTrue(os.path.exists(svg_path))
        
        # Test JSON export
        json_path = os.path.join(self.output_dir, "test_export.json")
        fig.write_json(json_path)
        self.assertTrue(os.path.exists(json_path))
    
    @unittest.skipIf(not HAS_DASH_TESTING, "Dash testing components not available")
    def test_dashboard_visualization_options_panel(self):
        """Test the visualization options panel UI component in the dashboard."""
        with mock.patch('duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard.BenchmarkDBAPI'):
            # Create a dashboard instance with mocked database connection
            dashboard = EnhancedVisualizationDashboard(
                db_conn=mock.MagicMock(),
                output_dir=self.output_dir,
                enable_regression_detection=True,
                enhanced_visualization=True
            )
            
            # Get the layout of the regression detection tab
            with mock.patch.object(dashboard, '_get_available_metrics', return_value=["test_metric"]):
                regression_tab = dashboard._create_regression_detection_tab()
                
                # Convert layout to string to check for UI components
                layout_str = str(regression_tab)
                
                # Check for visualization options panel
                self.assertIn("visualization-options-panel", layout_str)
                
                # Check for visualization options controls
                self.assertIn("regression-viz-options", layout_str)
                
                # Check for export format dropdown
                self.assertIn("export-format-dropdown", layout_str)
                
                # Check for export buttons
                self.assertIn("export-regression-viz-btn", layout_str)
    
    @unittest.skipIf(not HAS_DASH_TESTING, "Dash testing components not available")
    def test_dashboard_visualization_callbacks(self):
        """Test the visualization-related callbacks in the dashboard."""
        with mock.patch('duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard.BenchmarkDBAPI'):
            # Create a dashboard instance with mocked database connection
            dashboard = EnhancedVisualizationDashboard(
                db_conn=mock.MagicMock(),
                output_dir=self.output_dir,
                enable_regression_detection=True,
                enhanced_visualization=True
            )
            
            # Mock the data cache with our test data
            dashboard.data_cache["regression_analysis"]["last_time_series_data"] = self.time_series_data
            dashboard.data_cache["regression_analysis"]["last_regression_results"] = {
                "metric": "test_metric",
                "results": {"regressions": self.regressions}
            }
            
            # Mock data_cache visualization options
            dashboard.data_cache["regression_analysis"]["visualization_options"] = {
                "include_confidence_intervals": True,
                "include_trend_lines": True,
                "include_annotations": True,
                "export_format": "html"
            }
            
            # Test the update_regression_visualization callback
            with mock.patch.object(dashboard.regression_visualization, 'create_interactive_regression_figure') as mock_create_fig:
                mock_create_fig.return_value = {"data": [], "layout": {}}
                
                # Call the callback with different visualization options
                result = dashboard._update_regression_visualization_callback(
                    ["ci", "trend", "annotations"],  # All options enabled
                    "html"
                )
                
                # Verify the callback result
                self.assertIsNotNone(result)
                
                # Verify that the visualization function was called with the right parameters
                mock_create_fig.assert_called_with(
                    self.time_series_data,
                    self.regressions,
                    "test_metric",
                    include_confidence_intervals=True,
                    include_trend_lines=True,
                    include_annotations=True
                )
                
                # Test with some options disabled
                result = dashboard._update_regression_visualization_callback(
                    ["ci"],  # Only confidence intervals
                    "png"
                )
                
                # Verify the updated data cache
                self.assertTrue(dashboard.data_cache["regression_analysis"]["visualization_options"]["include_confidence_intervals"])
                self.assertFalse(dashboard.data_cache["regression_analysis"]["visualization_options"]["include_trend_lines"])
                self.assertFalse(dashboard.data_cache["regression_analysis"]["visualization_options"]["include_annotations"])
                self.assertEqual(dashboard.data_cache["regression_analysis"]["visualization_options"]["export_format"], "png")
    
    @unittest.skipIf(not HAS_DASH_TESTING, "Dash testing components not available")
    def test_export_regression_visualization_callback(self):
        """Test the export regression visualization callback."""
        with mock.patch('duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard.BenchmarkDBAPI'):
            # Create a dashboard instance with mocked database connection
            dashboard = EnhancedVisualizationDashboard(
                db_conn=mock.MagicMock(),
                output_dir=self.output_dir,
                enable_regression_detection=True,
                enhanced_visualization=True
            )
            
            # Mock the data cache with our test data
            dashboard.data_cache["regression_analysis"]["last_time_series_data"] = self.time_series_data
            dashboard.data_cache["regression_analysis"]["last_regression_results"] = {
                "metric": "test_metric",
                "results": {"regressions": self.regressions}
            }
            
            # Set visualization options
            dashboard.data_cache["regression_analysis"]["visualization_options"] = {
                "include_confidence_intervals": True,
                "include_trend_lines": True,
                "include_annotations": True,
                "export_format": "html"
            }
            
            # Create a mock figure
            fig_dict = self.regression_visualization.create_interactive_regression_figure(
                self.time_series_data,
                self.regressions,
                "test_metric"
            )
            
            # Mock the visualization engine
            with mock.patch.object(dashboard.regression_visualization, 'create_interactive_regression_figure') as mock_create_fig:
                mock_create_fig.return_value = fig_dict
                
                # Mock the figure export methods
                with mock.patch('plotly.graph_objects.Figure.write_html') as mock_write_html:
                    with mock.patch('plotly.graph_objects.Figure.write_image') as mock_write_image:
                        with mock.patch('plotly.graph_objects.Figure.write_json') as mock_write_json:
                            # Set up the context object to simulate which button was clicked
                            with mock.patch('dash.callback_context') as mock_context:
                                # Simulate clicking the export button
                                mock_context.triggered = [{"prop_id": "export-regression-viz-btn.n_clicks"}]
                                
                                # Call the export callback
                                result = dashboard._export_regression_visualization_callback(1, 0, "html")
                                
                                # Verify callback result (success message)
                                self.assertIsNotNone(result)
                                self.assertIn("Success", result[0])
                                
                                # Verify HTML export was called
                                mock_write_html.assert_called_once()
                                mock_write_image.assert_not_called()
                                mock_write_json.assert_not_called()
                                
                                # Reset mocks
                                mock_write_html.reset_mock()
                                mock_write_image.reset_mock()
                                mock_write_json.reset_mock()
                                
                                # Test PNG export
                                dashboard.data_cache["regression_analysis"]["visualization_options"]["export_format"] = "png"
                                result = dashboard._export_regression_visualization_callback(2, 0, "png")
                                
                                # Verify PNG export was called
                                mock_write_html.assert_not_called()
                                mock_write_image.assert_called_once()
                                mock_write_json.assert_not_called()
                                
                                # Reset mocks
                                mock_write_html.reset_mock()
                                mock_write_image.reset_mock()
                                mock_write_json.reset_mock()
                                
                                # Test clicking the inline export button
                                mock_context.triggered = [{"prop_id": "export-regression-viz-btn-inline.n_clicks"}]
                                dashboard.data_cache["regression_analysis"]["visualization_options"]["export_format"] = "json"
                                result = dashboard._export_regression_visualization_callback(2, 1, "json")
                                
                                # Verify JSON export was called
                                mock_write_html.assert_not_called()
                                mock_write_image.assert_not_called()
                                mock_write_json.assert_called_once()

    def test_generate_regression_report_with_options(self):
        """Test generating regression reports with different visualization options."""
        # Create a dashboard with mock DB
        with mock.patch('duckdb_api.distributed_testing.dashboard.enhanced_visualization_dashboard.BenchmarkDBAPI'):
            dashboard = EnhancedVisualizationDashboard(
                db_conn=mock.MagicMock(),
                output_dir=self.output_dir,
                enable_regression_detection=True,
                enhanced_visualization=True
            )
            
            # Mock the dashboard's data cache
            dashboard.data_cache["regression_analysis"]["last_time_series_data"] = self.time_series_data
            dashboard.data_cache["regression_analysis"]["last_regression_results"] = {
                "metric": "test_metric",
                "results": {"regressions": self.regressions}
            }
            
            # Test with all visualization options
            dashboard.data_cache["regression_analysis"]["visualization_options"] = {
                "include_confidence_intervals": True,
                "include_trend_lines": True,
                "include_annotations": True,
                "export_format": "html"
            }
            
            # Mock the regression detector
            with mock.patch.object(dashboard.regression_detector, 'generate_regression_report') as mock_generate_report:
                mock_generate_report.return_value = {
                    "metric_name": "test_metric",
                    "num_regressions": 1,
                    "regression_details": self.regressions,
                    "summary": "Found 1 regression in test_metric"
                }
                
                # Call the generate report callback
                result = dashboard._generate_regression_report_callback(1)
                
                # Verify the report was generated
                self.assertIsNotNone(result)
                
                # Verify the visualization options were passed correctly
                mock_generate_report.assert_called_once()


if __name__ == "__main__":
    unittest.main()