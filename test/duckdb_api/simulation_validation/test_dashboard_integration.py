#!/usr/bin/env python3
"""
Test module for the dashboard integration functionality of the Simulation Accuracy and Validation Framework.

This module tests the connection, authentication, and dashboard/panel creation capabilities
of the ValidationVisualizerDBConnector class when integrating with a monitoring dashboard.
"""

import os
import sys
import unittest
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_dashboard_integration")

# Import the components to test
from duckdb_api.simulation_validation.db_integration import SimulationValidationDBIntegration
from duckdb_api.simulation_validation.visualization.validation_visualizer import ValidationVisualizer
from duckdb_api.simulation_validation.visualization.validation_visualizer_db_connector import ValidationVisualizerDBConnector
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult,
    CalibrationRecord,
    DriftDetectionResult
)

class MockResponse:
    """Mock HTTP response for testing API calls."""
    
    def __init__(self, status_code, json_data, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        
    def json(self):
        return self._json_data


class TestDashboardIntegration(unittest.TestCase):
    """Test case for dashboard integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock database integration
        self.db_integration = MagicMock(spec=SimulationValidationDBIntegration)
        
        # Create mock visualizer
        self.visualizer = MagicMock(spec=ValidationVisualizer)
        
        # Dashboard integration parameters
        self.dashboard_url = "http://dashboard.example.com/api"
        self.dashboard_api_key = "test_api_key"
        
        # Create connector with dashboard integration enabled
        self.connector = ValidationVisualizerDBConnector(
            db_integration=self.db_integration,
            visualizer=self.visualizer,
            dashboard_integration=True,
            dashboard_url=self.dashboard_url,
            dashboard_api_key=self.dashboard_api_key
        )
        
        # Mock successful authentication response
        self.auth_success_response = MockResponse(
            200, 
            {
                "token": "test_session_token",
                "expires_in": 3600  # 1 hour
            }
        )

    @patch('requests.post')
    def test_connect_to_dashboard_success(self, mock_post):
        """Test successful connection to dashboard."""
        # Set up the mock to return a successful response
        mock_post.return_value = self.auth_success_response
        
        # Attempt to connect
        result = self.connector._connect_to_dashboard()
        
        # Verify the connection was successful
        self.assertTrue(result)
        self.assertTrue(self.connector.dashboard_connected)
        self.assertEqual(self.connector.dashboard_session_token, "test_session_token")
        
        # Verify the API call was made correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['headers']['X-API-Key'], self.dashboard_api_key)
        self.assertEqual(kwargs['json']['client'], "simulation_validation_framework")

    @patch('requests.post')
    def test_connect_to_dashboard_failure(self, mock_post):
        """Test failed connection to dashboard."""
        # Set up the mock to return a failed response
        mock_post.return_value = MockResponse(401, {}, "Unauthorized")
        
        # Attempt to connect
        result = self.connector._connect_to_dashboard()
        
        # Verify the connection failed
        self.assertFalse(result)
        self.assertFalse(self.connector.dashboard_connected)
        self.assertIsNone(self.connector.dashboard_session_token)

    @patch('requests.post')
    def test_ensure_dashboard_connection_reconnect_expired(self, mock_post):
        """Test reconnection when token is expired."""
        # Set up initial state with an expired token
        self.connector.dashboard_connected = True
        self.connector.dashboard_session_token = "expired_token"
        self.connector.dashboard_session_expires = datetime.now() - timedelta(hours=1)
        
        # Set up the mock to return a successful response
        mock_post.return_value = self.auth_success_response
        
        # Ensure connection
        result = self.connector._ensure_dashboard_connection()
        
        # Verify reconnection was successful
        self.assertTrue(result)
        self.assertTrue(self.connector.dashboard_connected)
        self.assertEqual(self.connector.dashboard_session_token, "test_session_token")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_create_dashboard_panel(self, mock_post):
        """Test creating a dashboard panel."""
        # Set up the connector with an active connection
        self.connector.dashboard_connected = True
        self.connector.dashboard_session_token = "active_token"
        self.connector.dashboard_session_expires = datetime.now() + timedelta(hours=1)
        
        # Set up the mock to return a successful response
        panel_response = MockResponse(
            201, 
            {
                "panel_id": "test_panel_id",
                "dashboard_id": "test_dashboard_id",
                "title": "Test Panel"
            }
        )
        mock_post.return_value = panel_response
        
        # Create a dashboard panel
        result = self.connector.create_dashboard_panel_from_db(
            panel_type="mape_comparison",
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            metric="throughput_items_per_second",
            dashboard_id="test_dashboard_id",
            panel_title="MAPE Comparison"
        )
        
        # Verify the panel creation was successful
        self.assertTrue(result["success"])
        self.assertEqual(result["panel_id"], "test_panel_id")
        
        # Verify the API call was made correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['headers']['Authorization'], "Bearer active_token")
        self.assertEqual(kwargs['json']['panel_type'], "mape_comparison")
        self.assertEqual(kwargs['json']['hardware_type'], "gpu_rtx3080")
        self.assertEqual(kwargs['json']['title'], "MAPE Comparison")

    @patch('requests.post')
    def test_create_comprehensive_monitoring_dashboard(self, mock_post):
        """Test creating a comprehensive monitoring dashboard."""
        # Set up the connector with an active connection
        self.connector.dashboard_connected = True
        self.connector.dashboard_session_token = "active_token"
        self.connector.dashboard_session_expires = datetime.now() + timedelta(hours=1)
        
        # Set up the mock to return a successful response for dashboard creation
        dashboard_response = MockResponse(
            201, 
            {
                "dashboard_id": "test_dashboard_id",
                "title": "Test Dashboard"
            }
        )
        
        # Set up the mock to return a successful response for panel creation
        panel_response = MockResponse(
            201, 
            {
                "panel_id": "test_panel_id",
                "dashboard_id": "test_dashboard_id",
                "title": "Test Panel"
            }
        )
        
        # Set up the side effect to return different responses for different calls
        mock_post.side_effect = [dashboard_response] + [panel_response] * 6  # Dashboard + 6 panels
        
        # Create a comprehensive dashboard
        result = self.connector.create_comprehensive_monitoring_dashboard(
            dashboard_title="Test Dashboard",
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            include_panels=["mape_comparison", "time_series"],
            metrics=["throughput_items_per_second", "average_latency_ms"]
        )
        
        # Verify the dashboard creation was successful
        self.assertTrue(result["success"])
        self.assertEqual(result["dashboard_id"], "test_dashboard_id")
        
        # Verify the API calls were made correctly
        self.assertEqual(mock_post.call_count, 5)  # 1 dashboard + 2 panels x 2 metrics

    @patch('requests.post')
    def test_set_up_real_time_monitoring(self, mock_post):
        """Test setting up real-time monitoring."""
        # Set up the connector with an active connection
        self.connector.dashboard_connected = True
        self.connector.dashboard_session_token = "active_token"
        self.connector.dashboard_session_expires = datetime.now() + timedelta(hours=1)
        
        # Set up the mock to return a successful response
        monitoring_response = MockResponse(
            201, 
            {
                "monitoring_id": "test_monitoring_id",
                "dashboard_id": "test_dashboard_id"
            }
        )
        mock_post.return_value = monitoring_response
        
        # Set up real-time monitoring
        result = self.connector.set_up_real_time_monitoring(
            hardware_type="gpu_rtx3080",
            model_type="bert-base-uncased",
            metrics=["throughput_items_per_second", "average_latency_ms"],
            dashboard_id="test_dashboard_id"
        )
        
        # Verify the monitoring setup was successful
        self.assertTrue(result["success"])
        self.assertEqual(result["monitoring_id"], "test_monitoring_id")
        self.assertEqual(result["dashboard_id"], "test_dashboard_id")
        
        # Verify the API call was made correctly
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['headers']['Authorization'], "Bearer active_token")
        self.assertEqual(kwargs['json']['hardware_type'], "gpu_rtx3080")
        self.assertEqual(kwargs['json']['model_type'], "bert-base-uncased")
        self.assertEqual(kwargs['json']['metrics'], ["throughput_items_per_second", "average_latency_ms"])

    @patch('requests.post')
    def test_visualization_methods_with_dashboard_panel_creation(self, mock_post):
        """Test visualization methods with dashboard panel creation."""
        # Set up the connector with an active connection
        self.connector.dashboard_connected = True
        self.connector.dashboard_session_token = "active_token"
        self.connector.dashboard_session_expires = datetime.now() + timedelta(hours=1)
        
        # Set up the mock to return a successful response
        panel_response = MockResponse(
            201, 
            {
                "panel_id": "test_panel_id",
                "dashboard_id": "test_dashboard_id",
                "title": "Test Panel"
            }
        )
        mock_post.return_value = panel_response
        
        # Test visualization methods with dashboard panel creation enabled
        methods_to_test = [
            ("create_mape_comparison_chart_from_db", {"hardware_ids": ["gpu_rtx3080"], "model_ids": ["bert-base-uncased"]}),
            ("create_hardware_comparison_heatmap_from_db", {"model_ids": ["bert-base-uncased"]}),
            ("create_time_series_chart_from_db", {"hardware_id": "gpu_rtx3080", "model_id": "bert-base-uncased", "metric_name": "throughput_items_per_second"}),
            ("create_drift_visualization_from_db", {"hardware_type": "gpu_rtx3080", "model_type": "bert-base-uncased"}),
            ("create_calibration_improvement_chart_from_db", {"hardware_type": "gpu_rtx3080", "model_type": "bert-base-uncased"})
        ]
        
        for method_name, params in methods_to_test:
            # Reset mock
            mock_post.reset_mock()
            
            # Call the method with dashboard panel creation enabled
            method = getattr(self.connector, method_name)
            result = method(dashboard_id="test_dashboard_id", create_dashboard_panel=True, **params)
            
            # Verify the panel creation was successful
            self.assertTrue(result["success"])
            self.assertEqual(result["panel_id"], "test_panel_id")
            
            # Verify the API call was made correctly
            mock_post.assert_called_once()


if __name__ == "__main__":
    unittest.main()