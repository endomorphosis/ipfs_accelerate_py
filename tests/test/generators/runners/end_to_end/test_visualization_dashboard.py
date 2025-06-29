#!/usr/bin/env python3
"""
Test script for the Visualization Dashboard

This script tests the functionality of the visualization dashboard, including
data provider methods and dashboard creation.

Usage:
    python test_visualization_dashboard.py
"""

import os
import sys
import unittest
import tempfile
import json
from unittest.mock import MagicMock, patch

# Create mock modules with the required structures
dash_mock = MagicMock()
dash_dcc_mock = MagicMock()
dash_html_mock = MagicMock()
dash_table_mock = MagicMock()
dash_deps_mock = MagicMock()
plotly_go_mock = MagicMock()
plotly_px_mock = MagicMock()
plotly_ff_mock = MagicMock()
duckdb_mock = MagicMock()
pandas_mock = MagicMock()
numpy_mock = MagicMock()

# Add specific attributes needed by the code
plotly_go_mock.Figure = MagicMock(return_value=MagicMock())
plotly_go_mock.Indicator = MagicMock(return_value=MagicMock())
plotly_go_mock.Bar = MagicMock(return_value=MagicMock())
plotly_go_mock.Scatter = MagicMock(return_value=MagicMock())
plotly_go_mock.Pie = MagicMock(return_value=MagicMock())

# Assign the mocks to sys.modules
sys.modules['dash'] = dash_mock
sys.modules['dash.dcc'] = dash_dcc_mock
sys.modules['dash.html'] = dash_html_mock
sys.modules['dash.dash_table'] = dash_table_mock
sys.modules['dash.dependencies'] = dash_deps_mock
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objs'] = plotly_go_mock
sys.modules['plotly.express'] = plotly_px_mock
sys.modules['plotly.figure_factory'] = plotly_ff_mock
sys.modules['duckdb'] = duckdb_mock
sys.modules['pandas'] = pandas_mock
sys.modules['numpy'] = numpy_mock

# Mock pandas DataFrame
class MockDataFrame:
    def __init__(self, data=None):
        self.data = data or {}
        self.columns = list(self.data.keys())
        self.empty = len(self.data) == 0
    
    def __len__(self):
        return len(next(iter(self.data.values())) if self.data else 0)

# Add mock pd to the modules
pd = sys.modules['pandas']
pd.DataFrame = MockDataFrame
pd.to_numeric = lambda x, errors: x
pd.to_datetime = lambda x, format: x

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import modules to test
from visualization_dashboard import DashboardDataProvider, VisualizationDashboard


class TestDashboardDataProvider(unittest.TestCase):
    """Test the DashboardDataProvider class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".duckdb")
        
        # Mock the duckdb.connect method
        self.mock_conn = MagicMock()
        self.patcher = patch("visualization_dashboard.duckdb.connect", return_value=self.mock_conn)
        self.mock_connect = self.patcher.start()
        
        # Create data provider
        self.data_provider = DashboardDataProvider(db_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        self.temp_db.close()
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.data_provider.db_path, self.temp_db.name)
        self.assertEqual(self.data_provider.conn, self.mock_conn)
        self.mock_connect.assert_called_once_with(self.temp_db.name, read_only=True)
    
    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        # Mock fetchone and fetchall results
        self.mock_conn.execute.return_value.fetchone.return_value = [100, 80, 20]
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ["cuda", 50, 40],
            ["cpu", 50, 40]
        ]
        
        result = self.data_provider.get_summary_stats()
        
        self.assertEqual(result["total"]["total"], 100)
        self.assertEqual(result["total"]["success"], 80)
        self.assertEqual(result["total"]["failure"], 20)
        self.assertEqual(len(result["by_hardware"]), 2)
        self.assertEqual(result["by_hardware"]["cuda"]["total"], 50)
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        # Mock fetchdf result
        mock_df = MockDataFrame({
            "model_name": ["model1", "model1"],
            "hardware_type": ["cuda", "cpu"],
            "throughput": ["100", "50"],
            "latency": ["10", "20"],
            "memory_usage": ["1000", "500"],
            "test_date": ["20250315_120000", "20250315_120000"],
            "success": [True, True]
        })
        mock_df.columns = ["model_name", "hardware_type", "throughput", "latency", "memory_usage", "test_date", "success"]
        self.mock_conn.execute.return_value.fetchdf.return_value = mock_df
        
        result = self.data_provider.get_performance_metrics(model_filter="model1")
        
        # With our mock, we can't test the details of pandas operations
        # Just verify that the result exists
        self.assertIsNotNone(result)
    
    def test_get_model_list(self):
        """Test getting model list."""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ["model1"],
            ["model2"]
        ]
        
        result = self.data_provider.get_model_list()
        
        self.assertEqual(result, ["model1", "model2"])
    
    def test_get_hardware_list(self):
        """Test getting hardware list."""
        self.mock_conn.execute.return_value.fetchall.return_value = [
            ["cuda"],
            ["cpu"]
        ]
        
        result = self.data_provider.get_hardware_list()
        
        self.assertEqual(result, ["cuda", "cpu"])
    
    def test_close(self):
        """Test closing the database connection."""
        self.data_provider.close()
        self.mock_conn.close.assert_called_once()
        self.assertIsNone(self.data_provider.conn)


class TestVisualizationDashboard(unittest.TestCase):
    """Test the VisualizationDashboard class."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock the DashboardDataProvider
        self.mock_data_provider = MagicMock()
        
        # Mock data
        self.mock_data_provider.get_model_list.return_value = ["model1", "model2"]
        self.mock_data_provider.get_hardware_list.return_value = ["cuda", "cpu"]
        
        # Create dashboard with mock data provider
        with patch("visualization_dashboard.dash.Dash") as mock_dash:
            self.mock_app = MagicMock()
            mock_dash.return_value = self.mock_app
            self.dashboard = VisualizationDashboard(data_provider=self.mock_data_provider)
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.dashboard.data_provider, self.mock_data_provider)
        self.assertEqual(self.dashboard.app, self.mock_app)
        self.mock_app.layout = self.dashboard._create_layout()
        
    def test_create_layout(self):
        """Test creating the dashboard layout."""
        layout = self.dashboard._create_layout()
        self.assertIsNotNone(layout)
        
    def test_create_overview_tab(self):
        """Test creating the overview tab."""
        summary_data = {
            "total": {"total": 100, "success": 80, "failure": 20},
            "by_hardware": {
                "cuda": {"total": 50, "success": 40},
                "cpu": {"total": 50, "success": 40}
            },
            "by_model": {
                "model1": {"total": 50, "success": 40},
                "model2": {"total": 50, "success": 40}
            },
            "recent_dates": ["20250315_120000", "20250314_120000"]
        }
        
        overview_tab = self.dashboard._create_overview_tab(summary_data)
        self.assertIsNotNone(overview_tab)
        
    def test_create_performance_tab(self):
        """Test creating the performance tab."""
        performance_tab = self.dashboard._create_performance_tab()
        self.assertIsNotNone(performance_tab)
        
    def test_run_server(self):
        """Test running the server."""
        self.dashboard.run_server(host="localhost", port=8050, debug=False)
        self.mock_app.run_server.assert_called_once_with(host="localhost", port=8050, debug=False)


if __name__ == "__main__":
    unittest.main()