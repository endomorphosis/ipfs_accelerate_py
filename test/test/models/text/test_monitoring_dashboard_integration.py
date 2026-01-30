#!/usr/bin/env python3
"""
Test module for the MonitoringDashboardConnector in the Simulation Accuracy and Validation Framework.

This module tests the direct integration with the monitoring dashboard system,
ensuring proper connection, dashboard creation, panel management, and real-time monitoring.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, ANY
import json
import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_monitoring_dashboard_integration")

# Import the component to test
from data.duckdb.simulation_validation.visualization.monitoring_dashboard_connector import (
    MonitoringDashboardConnector,
    get_monitoring_dashboard_connector
)

# Import related components
from data.duckdb.simulation_validation.visualization.validation_visualizer_db_connector import ValidationVisualizerDBConnector
from data.duckdb.simulation_validation.db_performance_optimizer import DatabasePerformanceOptimizer

class MockResponse:
    """Mock HTTP response for testing API calls."""
    
    def __init__(self, status_code, json_data, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        
    def json(self):
        return self._json_data

class TestMonitoringDashboardConnector(unittest.TestCase):
    """Test case for the MonitoringDashboardConnector."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock dependencies
        self.db_connector = MagicMock(spec=ValidationVisualizerDBConnector)
        self.db_optimizer = MagicMock(spec=DatabasePerformanceOptimizer)
        
        # Dashboard integration parameters
        self.dashboard_url = "http://dashboard.example.com/api"
        self.dashboard_api_key = "test-api-key"
        
        # Mock metrics data
        self.db_optimizer.get_performance_metrics.return_value = {
            "query_time": {
                "value": 150,
                "previous_value": 180,
                "change_pct": -16.67,
                "unit": "ms",
                "status": "good",
                "history": [180, 175, 160, 150]
            },
            "storage_size": {
                "value": 500000000,
                "previous_value": 450000000,
                "change_pct": 11.11,
                "unit": "bytes",
                "status": "warning",
                "history": [400000000, 425000000, 450000000, 500000000]
            }
        }
        
        self.db_optimizer.get_overall_status.return_value = "good"
        
        # Create mock successful auth response
        self.auth_success_response = MockResponse(
            200,
            {
                "token": "mock-session-token",
                "expires_in": 3600  # 1 hour
            }
        )
        
        # Create connector with mocked dependencies using patch
        with patch('requests.post', return_value=self.auth_success_response):
            self.connector = MonitoringDashboardConnector(
                dashboard_url=self.dashboard_url,
                dashboard_api_key=self.dashboard_api_key,
                db_connector=self.db_connector,
                db_optimizer=self.db_optimizer
            )
    
    @patch('requests.post')
    def test_connect_to_dashboard_success(self, mock_post):
        """Test successful connection to dashboard."""
        # Set up the mock
        mock_post.return_value = self.auth_success_response
        
        # Attempt to connect
        result = self.connector._connect_to_dashboard()
        
        # Verify successful connection
        self.assertTrue(result)
        self.assertTrue(self.connector.connected)
        self.assertEqual(self.connector.session_token, "mock-session-token")
        
        # Verify the POST request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['headers']['X-API-Key'], self.dashboard_api_key)
        self.assertEqual(kwargs['json']['client'], "simulation_validation_framework")
        self.assertIn("application/json", kwargs['headers']['Content-Type'])
    
    @patch('requests.post')
    def test_connect_to_dashboard_failure(self, mock_post):
        """Test failed connection to dashboard."""
        # Set up the mock to return an error
        mock_post.return_value = MockResponse(401, {}, "Unauthorized")
        
        # Create a new connector to test the failure scenario
        connector = MonitoringDashboardConnector(
            dashboard_url=self.dashboard_url,
            dashboard_api_key="invalid-key",
            db_connector=self.db_connector,
            db_optimizer=self.db_optimizer
        )
        
        # Verify the connection failed
        self.assertFalse(connector.connected)
        self.assertIsNone(connector.session_token)
    
    @patch('requests.post')
    def test_ensure_connection_expired_token(self, mock_post):
        """Test reconnection when token is expired."""
        # Set up the mock
        mock_post.return_value = self.auth_success_response
        
        # Set up an expired token
        self.connector.session_expires = datetime.datetime.now() - datetime.timedelta(minutes=10)
        
        # Ensure connection
        result = self.connector._ensure_connection()
        
        # Verify successful reconnection
        self.assertTrue(result)
        self.assertTrue(self.connector.connected)
        self.assertEqual(self.connector.session_token, "mock-session-token")
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_create_dashboard(self, mock_post):
        """Test creating a dashboard."""
        # Set up the mock
        mock_post.return_value = MockResponse(
            201,
            {
                "dashboard_id": "mock-dashboard-id",
                "title": "Test Dashboard",
                "url": "http://dashboard.example.com/dashboards/mock-dashboard-id"
            }
        )
        
        # Create a dashboard
        result = self.connector.create_dashboard(
            title="Test Dashboard",
            description="Test dashboard description",
            tags=["test", "simulation"]
        )
        
        # Verify the dashboard creation
        self.assertTrue(result["success"])
        self.assertEqual(result["dashboard_id"], "mock-dashboard-id")
        self.assertEqual(result["title"], "Test Dashboard")
        self.assertEqual(result["url"], "http://dashboard.example.com/dashboards/mock-dashboard-id")
        
        # Verify the POST request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['title'], "Test Dashboard")
        self.assertEqual(kwargs['json']['description'], "Test dashboard description")
        self.assertEqual(kwargs['json']['tags'], ["test", "simulation"])
    
    @patch('requests.post')
    def test_create_panel(self, mock_post):
        """Test creating a panel in a dashboard."""
        # Set up the mock
        mock_post.return_value = MockResponse(
            201,
            {
                "panel_id": "mock-panel-id",
                "dashboard_id": "mock-dashboard-id",
                "title": "Test Panel"
            }
        )
        
        # Create a panel
        result = self.connector.create_panel(
            dashboard_id="mock-dashboard-id",
            panel_type="line-chart",
            title="Test Panel",
            data_source={
                "type": "simulation_validation",
                "query": {"metrics": ["throughput_items_per_second"]}
            },
            position={"x": 0, "y": 0},
            size={"width": 6, "height": 4},
            refresh_interval=300
        )
        
        # Verify the panel creation
        self.assertTrue(result["success"])
        self.assertEqual(result["panel_id"], "mock-panel-id")
        self.assertEqual(result["dashboard_id"], "mock-dashboard-id")
        self.assertEqual(result["title"], "Test Panel")
        
        # Verify the POST request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['title'], "Test Panel")
        self.assertEqual(kwargs['json']['type'], "line-chart")
        self.assertEqual(kwargs['json']['size'], {"width": 6, "height": 4})
        self.assertEqual(kwargs['json']['refresh_interval'], 300)
    
    @patch('requests.put')
    def test_update_panel_data(self, mock_put):
        """Test updating panel data."""
        # Set up the mock
        mock_put.return_value = MockResponse(
            200,
            {
                "panel_id": "mock-panel-id",
                "dashboard_id": "mock-dashboard-id",
                "updated": True
            }
        )
        
        # Update panel data
        result = self.connector.update_panel_data(
            dashboard_id="mock-dashboard-id",
            panel_id="mock-panel-id",
            data={"value": 123, "unit": "ms"}
        )
        
        # Verify the data update
        self.assertTrue(result["success"])
        self.assertEqual(result["panel_id"], "mock-panel-id")
        self.assertEqual(result["dashboard_id"], "mock-dashboard-id")
        
        # Verify the PUT request
        mock_put.assert_called_once()
        args, kwargs = mock_put.call_args
        self.assertEqual(kwargs['json']['data'], {"value": 123, "unit": "ms"})
    
    @patch('requests.post')
    def test_create_alert(self, mock_post):
        """Test creating an alert for a panel."""
        # Set up the mock
        mock_post.return_value = MockResponse(
            201,
            {
                "alert_id": "mock-alert-id",
                "dashboard_id": "mock-dashboard-id",
                "panel_id": "mock-panel-id",
                "name": "Test Alert"
            }
        )
        
        # Create an alert
        result = self.connector.create_alert(
            name="Test Alert",
            description="Test alert description",
            dashboard_id="mock-dashboard-id",
            panel_id="mock-panel-id",
            condition={"metric": "query_time", "comparison": "gt", "value": 1000},
            notification_channels=["email", "slack"],
            severity="warning",
            interval=300
        )
        
        # Verify the alert creation
        self.assertTrue(result["success"])
        self.assertEqual(result["alert_id"], "mock-alert-id")
        self.assertEqual(result["dashboard_id"], "mock-dashboard-id")
        self.assertEqual(result["panel_id"], "mock-panel-id")
        self.assertEqual(result["name"], "Test Alert")
        
        # Verify the POST request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['name'], "Test Alert")
        self.assertEqual(kwargs['json']['condition']['metric'], "query_time")
        self.assertEqual(kwargs['json']['notification_channels'], ["email", "slack"])
    
    @patch('requests.get')
    @patch('requests.put')
    def test_update_database_performance_metrics(self, mock_put, mock_get):
        """Test updating database performance metrics."""
        # Set up the mocks
        mock_get.side_effect = [
            # First call to get panels
            MockResponse(
                200,
                {
                    "panels": [
                        {"panel_id": "panel1", "type": "database-metric"},
                        {"panel_id": "panel2", "type": "database-summary"}
                    ]
                }
            ),
            # Second call to get panel1 details
            MockResponse(
                200,
                {
                    "panel_id": "panel1",
                    "type": "database-metric",
                    "data_source": {
                        "query": {"metric": "query_time"}
                    }
                }
            ),
            # Third call to get panel2 details
            MockResponse(
                200,
                {
                    "panel_id": "panel2",
                    "type": "database-summary"
                }
            )
        ]
        
        # Set up put mock responses
        mock_put.side_effect = [
            MockResponse(200, {"updated": True}),
            MockResponse(200, {"updated": True})
        ]
        
        # Update metrics
        result = self.connector.update_database_performance_metrics(
            dashboard_id="mock-dashboard-id"
        )
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["dashboard_id"], "mock-dashboard-id")
        self.assertEqual(result["updated_panels"], 2)
        
        # Verify API calls
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_put.call_count, 2)
    
    @patch('requests.post')
    def test_simulation_validation_dashboard(self, mock_post):
        """Test creating a simulation validation dashboard."""
        # Set up the mocks for multiple post calls
        mock_post.side_effect = [
            # Dashboard creation response
            MockResponse(
                201,
                {
                    "dashboard_id": "mock-dashboard-id",
                    "title": "Simulation Validation Dashboard",
                    "url": "http://dashboard.example.com/dashboards/mock-dashboard-id"
                }
            ),
            # Panel creation responses
            MockResponse(201, {"panel_id": "panel1", "dashboard_id": "mock-dashboard-id"}),
            MockResponse(201, {"panel_id": "panel2", "dashboard_id": "mock-dashboard-id"}),
            MockResponse(201, {"panel_id": "panel3", "dashboard_id": "mock-dashboard-id"}),
            MockResponse(201, {"panel_id": "panel4", "dashboard_id": "mock-dashboard-id"}),
            MockResponse(201, {"panel_id": "panel5", "dashboard_id": "mock-dashboard-id"})
        ]
        
        # Create dashboard
        result = self.connector.create_simulation_validation_dashboard(
            hardware_types=["gpu"],
            model_types=["bert"],
            metrics=["throughput_items_per_second", "average_latency_ms"],
            dashboard_title="Simulation Validation Dashboard"
        )
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["dashboard_id"], "mock-dashboard-id")
        self.assertEqual(result["dashboard_title"], "Simulation Validation Dashboard")
        self.assertEqual(result["panel_count"], 5)
        
        # Verify API calls
        self.assertEqual(mock_post.call_count, 6)
    
    @patch('requests.post')
    def test_setup_real_time_monitoring(self, mock_post):
        """Test setting up real-time monitoring."""
        # Set up the mocks for multiple post calls
        mock_post.side_effect = [
            # Dashboard creation response
            MockResponse(
                201,
                {
                    "dashboard_id": "mock-dashboard-id",
                    "title": "Real-Time Monitoring Dashboard",
                    "url": "http://dashboard.example.com/dashboards/mock-dashboard-id"
                }
            ),
            # Panel creation responses (simplified for the test)
            MockResponse(201, {"panel_id": "panel1", "dashboard_id": "mock-dashboard-id"}),
            # Monitoring configuration response
            MockResponse(
                201,
                {
                    "monitoring_id": "mock-monitoring-id",
                    "dashboard_id": "mock-dashboard-id",
                    "url": "http://dashboard.example.com/monitoring/mock-monitoring-id"
                }
            )
        ]
        
        # Setup monitoring
        result = self.connector.setup_real_time_monitoring(
            hardware_types=["gpu"],
            model_types=["bert"],
            metrics=["throughput_items_per_second"],
            interval=300
        )
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["monitoring_id"], "mock-monitoring-id")
        self.assertEqual(result["dashboard_id"], "mock-dashboard-id")
        self.assertEqual(result["interval"], 300)
        
        # Verify API calls
        self.assertEqual(mock_post.call_count, 3)
    
    def test_get_monitoring_dashboard_connector(self):
        """Test the factory function for getting a connector instance."""
        # Set up mocks
        with patch('requests.post', return_value=self.auth_success_response):
            # Create a connector using the factory function
            connector = get_monitoring_dashboard_connector(
                dashboard_url=self.dashboard_url,
                dashboard_api_key=self.dashboard_api_key,
                db_connector=self.db_connector,
                db_optimizer=self.db_optimizer
            )
            
            # Verify the connector was created correctly
            self.assertIsInstance(connector, MonitoringDashboardConnector)
            self.assertEqual(connector.dashboard_url, self.dashboard_url)
            self.assertEqual(connector.dashboard_api_key, self.dashboard_api_key)
            self.assertEqual(connector.db_connector, self.db_connector)
            self.assertEqual(connector.db_optimizer, self.db_optimizer)
            self.assertTrue(connector.connected)

    @patch('requests.post')
    def test_create_database_performance_dashboard(self, mock_post):
        """Test creating a dedicated database performance dashboard."""
        # Set up the mocks for multiple POST requests
        mock_post.side_effect = [
            # Dashboard creation response
            MockResponse(
                201, 
                {
                    "dashboard_id": "mock-db-dashboard-id",
                    "title": "Database Performance Dashboard",
                    "url": "http://dashboard.example.com/dashboards/mock-db-dashboard-id"
                }
            ),
            # Multiple panel creation responses
            MockResponse(201, {"panel_id": "panel1", "dashboard_id": "mock-db-dashboard-id"}),  # Summary panel
            MockResponse(201, {"panel_id": "panel2", "dashboard_id": "mock-db-dashboard-id"}),  # Trend panel
            MockResponse(201, {"panel_id": "panel3", "dashboard_id": "mock-db-dashboard-id"}),  # Metric panel 1
            MockResponse(201, {"panel_id": "panel4", "dashboard_id": "mock-db-dashboard-id"}),  # Metric panel 2
            # Alert creation responses
            MockResponse(201, {"alert_id": "alert1", "panel_id": "panel3"}),
            MockResponse(201, {"alert_id": "alert2", "panel_id": "panel4"}),
            # Scheduled update response
            MockResponse(
                201,
                {
                    "update_id": "mock-update-id",
                    "dashboard_id": "mock-db-dashboard-id",
                    "schedule": "3600s"
                }
            )
        ]
        
        # Mock available metrics
        self.db_optimizer.get_performance_metrics.return_value = {
            "query_time": {"value": 150, "status": "good", "unit": "ms"},
            "storage_size": {"value": 5000000, "status": "warning", "unit": "bytes"}
        }
        
        # Create dashboard
        result = self.connector.create_database_performance_dashboard(
            dashboard_title="Database Performance Dashboard",
            visualization_style="detailed",
            create_alerts=True,
            auto_update=True
        )
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["dashboard_id"], "mock-db-dashboard-id")
        self.assertEqual(result["dashboard_title"], "Database Performance Dashboard")
        self.assertEqual(result["visualization_style"], "detailed")
        self.assertTrue(result["auto_update"])
        self.assertEqual(result["update_interval"], 3600)
        
        # Verify the API calls
        self.assertEqual(mock_post.call_count, 8)  # 1 dashboard + 4 panels + 2 alerts + 1 scheduled update
    
    @patch('requests.post')
    @patch('requests.get')
    def test_create_complete_monitoring_solution(self, mock_get, mock_post):
        """Test creating a complete monitoring solution with both database and validation metrics."""
        # Set up the mocks for multiple POST and GET requests
        mock_post.side_effect = [
            # Dashboard creation response
            MockResponse(
                201, 
                {
                    "dashboard_id": "mock-complete-dashboard-id",
                    "title": "Complete Monitoring Solution",
                    "url": "http://dashboard.example.com/dashboards/mock-complete-dashboard-id"
                }
            ),
            # Database panels - db monitoring setup result
            MockResponse(
                201,
                {
                    "success": True,
                    "dashboard_id": "mock-complete-dashboard-id",
                    "panel_count": 5,
                    "alert_count": 2
                }
            ),
            # Simulation panels
            MockResponse(201, {"panel_id": "sim-panel1", "dashboard_id": "mock-complete-dashboard-id"}),
            MockResponse(201, {"panel_id": "sim-panel2", "dashboard_id": "mock-complete-dashboard-id"}),
            MockResponse(201, {"panel_id": "sim-panel3", "dashboard_id": "mock-complete-dashboard-id"}),
            # Scheduled update response
            MockResponse(
                201,
                {
                    "update_id": "mock-update-id",
                    "dashboard_id": "mock-complete-dashboard-id",
                    "schedule": "3000s"
                }
            )
        ]
        
        # Mock GET requests
        mock_get.return_value = MockResponse(
            200,
            {"panel": {"type": "mape-comparison", "data_source": {"query": {}}}}
        )
        
        # Mock available metrics
        self.db_optimizer.get_performance_metrics.return_value = {
            "query_time": {"value": 150, "status": "good", "unit": "ms"},
            "storage_size": {"value": 5000000, "status": "warning", "unit": "bytes"}
        }
        
        # Create the complete solution
        result = self.connector.create_complete_monitoring_solution(
            dashboard_title="Complete Monitoring Solution",
            include_database_performance=True,
            include_validation_metrics=True,
            hardware_types=["gpu"],
            model_types=["bert"],
            visualization_style="compact"
        )
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["dashboard_id"], "mock-complete-dashboard-id")
        self.assertEqual(result["dashboard_title"], "Complete Monitoring Solution")
        self.assertTrue(result["includes_database_performance"])
        self.assertTrue(result["includes_validation_metrics"])
        self.assertEqual(result["visualization_style"], "compact")
        
        # Verify API calls
        self.assertGreater(mock_post.call_count, 3)  # At least dashboard + monitoring setup + scheduled update

if __name__ == "__main__":
    unittest.main()