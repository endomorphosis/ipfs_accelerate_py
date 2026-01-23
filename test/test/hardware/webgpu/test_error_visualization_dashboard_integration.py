#!/usr/bin/env python3
"""
Test Error Visualization Dashboard Integration.

This script tests the integration between the Error Visualization system and the Monitoring Dashboard,
focusing on WebSocket communication, API endpoints, and UI interactions.
"""

import os
import sys
import time
import json
import anyio
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add parent directory to path to import the modules
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import aiohttp
    from aiohttp import web
    aiohttp_available = True
except ImportError:
    aiohttp_available = False

from duckdb_api.distributed_testing.dashboard.error_visualization_integration import ErrorVisualizationIntegration
from duckdb_api.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard

# Check if we should skip the tests that require aiohttp
# These tests are more comprehensive and test actual server functionality
SKIP_AIOHTTP_TESTS = not aiohttp_available


class TestDashboardRoutes(unittest.IsolatedAsyncioTestCase):
    """Test dashboard routes for error visualization."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create test database file
        self.db_path = os.path.join(self.output_dir, "test_error_viz.duckdb")
        
        # Set up monitoring dashboard mock
        self.dashboard = MagicMock()
        self.dashboard.error_viz = ErrorVisualizationIntegration(
            output_dir=self.output_dir,
            db_path=self.db_path
        )
        
        # Mock request
        self.request = MagicMock()
        self.request.app = {"dashboard": self.dashboard}
        
        # Generate a sample error
        self.sample_error = {
            "timestamp": datetime.now().isoformat(),
            "worker_id": "test-worker-1",
            "type": "ResourceError",
            "error_category": "RESOURCE_EXHAUSTED",
            "message": "Failed to allocate GPU memory",
            "system_context": {
                "hostname": "test-node-1",
                "metrics": {
                    "cpu": {"percent": 85},
                    "memory": {"used_percent": 70},
                    "disk": {"used_percent": 60}
                }
            },
            "hardware_context": {
                "hardware_type": "cuda",
                "hardware_status": {
                    "overheating": False,
                    "memory_pressure": True,
                    "throttling": False
                }
            }
        }
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    async def test_api_report_error(self):
        """Test the report-error API endpoint."""
        if SKIP_AIOHTTP_TESTS:
            self.skipTest("aiohttp not available")
        
        # Mock request with error data
        request = AsyncMock()
        request.app = {"dashboard": self.dashboard}
        request.json = AsyncMock(return_value=self.sample_error)
        
        # Mock dashboard.error_viz.report_error
        self.dashboard.error_viz.report_error = AsyncMock(return_value=True)
        
        # Import the route handler
        from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_routes import api_report_error
        
        # Call the handler
        response = await api_report_error(request)
        
        # Check response status and content
        self.assertEqual(response.status, 200)
        response_data = json.loads(response.text)
        self.assertEqual(response_data["status"], "success")
        
        # Verify that report_error was called with the correct data
        self.dashboard.error_viz.report_error.assert_called_once_with(self.sample_error)
    
    async def test_api_get_errors(self):
        """Test the get-errors API endpoint."""
        if SKIP_AIOHTTP_TESTS:
            self.skipTest("aiohttp not available")
        
        # Mock request for get errors
        request = MagicMock()
        request.app = {"dashboard": self.dashboard}
        request.query = {"time_range": "24"}
        
        # Generate mock error data
        mock_error_data = {
            "summary": {"total_errors": 10},
            "timestamp": datetime.now().isoformat(),
            "recent_errors": [{"id": 1, "message": "Test error"}]
        }
        
        # Mock dashboard.error_viz.get_error_data
        self.dashboard.error_viz.get_error_data = AsyncMock(return_value=mock_error_data)
        
        # Import the route handler
        from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_routes import api_get_errors
        
        # Call the handler
        response = await api_get_errors(request)
        
        # Check response status and content
        self.assertEqual(response.status, 200)
        response_data = json.loads(response.text)
        self.assertEqual(response_data["status"], "success")
        self.assertEqual(response_data["data"], mock_error_data)
        
        # Verify that get_error_data was called with the correct time range
        self.dashboard.error_viz.get_error_data.assert_called_once_with(time_range_hours=24)


@unittest.skipIf(SKIP_AIOHTTP_TESTS, "aiohttp not available")
class TestDashboardServer(unittest.IsolatedAsyncioTestCase):
    """Test the dashboard server with error visualization integration."""
    
    async def asyncSetUp(self):
        """Set up the test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create test database file
        self.db_path = os.path.join(self.output_dir, "test_dashboard.duckdb")
        
        # Create the dashboard
        self.dashboard = MonitoringDashboard(
            host="localhost",
            port=0,  # Use a random available port
            db_path=self.db_path,
            enable_error_visualization=True
        )
        
        # Patch the start method to avoid actually starting the server
        self.original_start = self.dashboard.start
        self.dashboard.start = AsyncMock()
        
        # Initialize the dashboard internals
        await self.dashboard._initialize()
        
        # Check that error visualization was initialized
        self.assertIsNotNone(self.dashboard.error_viz)
        
        # Generate a sample error
        self.sample_error = {
            "timestamp": datetime.now().isoformat(),
            "worker_id": "test-worker-1",
            "type": "ResourceError",
            "error_category": "RESOURCE_EXHAUSTED",
            "message": "Failed to allocate GPU memory",
            "system_context": {
                "hostname": "test-node-1",
                "metrics": {
                    "cpu": {"percent": 85},
                    "memory": {"used_percent": 70},
                    "disk": {"used_percent": 60}
                }
            },
            "hardware_context": {
                "hardware_type": "cuda",
                "hardware_status": {
                    "overheating": False,
                    "memory_pressure": True,
                    "throttling": False
                }
            }
        }
    
    async def asyncTearDown(self):
        """Clean up after tests."""
        # Restore original start method
        self.dashboard.start = self.original_start
        
        # Clean up
        self.temp_dir.cleanup()
    
    async def test_error_visualization_initialization(self):
        """Test that error visualization is properly initialized."""
        # Verify that error visualization is enabled
        self.assertTrue(self.dashboard.enable_error_visualization)
        
        # Verify that the error_viz object is created
        self.assertIsNotNone(self.dashboard.error_viz)
        
        # Verify that the error_viz db_path matches dashboard db_path
        self.assertEqual(self.dashboard.error_viz.db_path, self.db_path)
    
    async def test_websocket_handler(self):
        """Test the WebSocket handler for error visualization messages."""
        # Create mock WebSocket
        ws = AsyncMock()
        ws.receive_json = AsyncMock()
        ws.receive_json.side_effect = [
            {"type": "error_visualization_init", "time_range": 24},
            {"type": "subscribe", "topic": "error_visualization"},
            web.WSMsgType.CLOSE  # Simulate close message
        ]
        
        # Create mock request
        request = MagicMock()
        request.app = {"dashboard": self.dashboard}
        
        # Import the WebSocket handler
        from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_routes import websocket_handler
        
        # Patch the dashboard.websocket_manager.register method
        self.dashboard.websocket_manager.register = AsyncMock()
        
        # Call the handler
        with patch('aiohttp.web.WebSocketResponse', return_value=ws):
            await websocket_handler(request)
        
        # Verify that the WebSocket was registered
        self.dashboard.websocket_manager.register.assert_called()
    
    async def test_report_error_integration(self):
        """Test the report_error method integration."""
        # Patch the dashboard.error_viz.report_error method
        self.dashboard.error_viz.report_error = AsyncMock(return_value=True)
        
        # Report an error
        result = await self.dashboard.report_error(self.sample_error)
        
        # Verify result
        self.assertTrue(result)
        
        # Verify that report_error was called
        self.dashboard.error_viz.report_error.assert_called_once_with(self.sample_error)
    
    async def test_get_errors_integration(self):
        """Test the get_errors method integration."""
        # Generate mock error data
        mock_error_data = {
            "summary": {"total_errors": 10},
            "timestamp": datetime.now().isoformat(),
            "recent_errors": [{"id": 1, "message": "Test error"}]
        }
        
        # Patch the dashboard.error_viz.get_error_data method
        self.dashboard.error_viz.get_error_data = AsyncMock(return_value=mock_error_data)
        
        # Get errors
        result = await self.dashboard.get_errors(time_range_hours=24)
        
        # Verify result
        self.assertEqual(result, mock_error_data)
        
        # Verify that get_error_data was called
        self.dashboard.error_viz.get_error_data.assert_called_once_with(time_range_hours=24)


class TestErrorVisualizationHTML(unittest.TestCase):
    """Test error visualization HTML template."""
    
    def setUp(self):
        """Set up the test environment."""
        # Path to the error visualization HTML template
        self.template_path = os.path.join(
            parent_dir,
            "duckdb_api",
            "distributed_testing",
            "dashboard",
            "templates",
            "error_visualization.html"
        )
        
        # Check if template exists
        if not os.path.exists(self.template_path):
            self.skipTest(f"Template not found at {self.template_path}")
        
        # Read the template
        with open(self.template_path, "r") as f:
            self.template_content = f.read()
    
    def test_sound_notification_code(self):
        """Test that the template includes sound notification code."""
        # Check for the playErrorNotification function
        self.assertIn("function playErrorNotification", self.template_content)
        
        # Check for sound file references
        self.assertIn("error-critical.mp3", self.template_content)
        self.assertIn("error-warning.mp3", self.template_content)
        self.assertIn("error-info.mp3", self.template_content)
        self.assertIn("error-notification.mp3", self.template_content)
        
        # Check for volume control references
        self.assertIn("notificationVolume", self.template_content)
        self.assertIn("function changeNotificationVolume", self.template_content)
        self.assertIn("function toggleMute", self.template_content)
    
    def test_error_severity_code(self):
        """Test that the template includes error severity detection code."""
        # Check for severity determination logic
        self.assertIn("errorType === 'critical'", self.template_content)
        self.assertIn("errorType === 'warning'", self.template_content)
        self.assertIn("errorType === 'info'", self.template_content)
        
        # Check for error category checks
        self.assertIn("HARDWARE_NOT_AVAILABLE", self.template_content)
        self.assertIn("RESOURCE_EXHAUSTED", self.template_content)
        self.assertIn("WORKER_CRASH", self.template_content)
    
    def test_websocket_integration(self):
        """Test that the template includes WebSocket integration code."""
        # Check for WebSocket initialization
        self.assertIn("function initializeWebSocket", self.template_content)
        self.assertIn("new WebSocket", self.template_content)
        
        # Check for WebSocket event handlers
        self.assertIn("socket.onopen", self.template_content)
        self.assertIn("socket.onmessage", self.template_content)
        self.assertIn("socket.onclose", self.template_content)
        self.assertIn("socket.onerror", self.template_content)
        
        # Check for error message handling
        self.assertIn("handleErrorUpdate", self.template_content)
        self.assertIn("addErrorToList", self.template_content)
        self.assertIn("playErrorNotification", self.template_content)
    
    def test_accessibility_features(self):
        """Test that the template includes accessibility features."""
        # Check for ARIA attributes
        self.assertIn("aria-label", self.template_content)
        self.assertIn("aria-live", self.template_content)
        self.assertIn("aria-atomic", self.template_content)
        
        # Check for visually hidden text
        self.assertIn("visually-hidden", self.template_content)
        
        # Check for high contrast mode support
        self.assertIn("forced-colors", self.template_content)


if __name__ == "__main__":
    unittest.main()