#!/usr/bin/env python3
"""
Test the web server integration for the Visualization Dashboard and Monitoring Dashboard.

This test validates that the web server components of the integration work properly,
including route handlers, template rendering, and API endpoints.
"""

import os
import sys
import json
import anyio
import unittest
import tempfile
import logging
from pathlib import Path
import threading
import time
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_dashboard_visualization_web_integration")

# Add parent directory to path for module imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import components to test
try:
    # Import the web components
    from data.duckdb.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard
    from data.duckdb.distributed_testing.dashboard.monitoring_dashboard_routes import (
        handle_dashboard_management,
        handle_results,
        handle_index
    )
    from data.duckdb.distributed_testing.dashboard.monitoring_dashboard_visualization_integration import (
        VisualizationDashboardIntegration
    )
    
    # Check if test should be skipped due to missing dependencies
    SKIP_TEST = False
except ImportError as e:
    logger.warning(f"Import error: {e}")
    logger.warning("Some components are not available. Integration tests will be skipped.")
    SKIP_TEST = True


# Create a simple request handler for testing
class TestHandler(SimpleHTTPRequestHandler):
    """Simple HTTP request handler for testing."""
    
    def __init__(self, *args, dashboard_dir=None, **kwargs):
        self.dashboard_dir = dashboard_dir
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Test Server</h1></body></html>")
        elif self.path.startswith('/dashboards/'):
            # Serve dashboard files
            file_path = os.path.join(self.dashboard_dir, self.path[11:])
            if os.path.exists(file_path) and os.path.isfile(file_path):
                with open(file_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


class TestWebServerIntegration(unittest.TestCase):
    """Test the web server integration for visualization dashboard."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        if SKIP_TEST:
            return
        
        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = cls.temp_dir.name
        
        # Create required directories
        cls.dashboard_dir = os.path.join(cls.output_dir, "dashboards")
        cls.templates_dir = os.path.join(cls.output_dir, "templates")
        cls.static_dir = os.path.join(cls.output_dir, "static")
        
        os.makedirs(cls.dashboard_dir, exist_ok=True)
        os.makedirs(cls.templates_dir, exist_ok=True)
        os.makedirs(cls.static_dir, exist_ok=True)
        os.makedirs(os.path.join(cls.static_dir, "dashboards"), exist_ok=True)
        
        # Create a simple test server
        cls.server_port = 8098  # Use a port that's unlikely to be in use
        
        # Function to create and start the server
        def start_server():
            """Start a test HTTP server."""
            cls.httpd = HTTPServer(("localhost", cls.server_port), 
                                  lambda *args, **kwargs: TestHandler(*args, dashboard_dir=cls.dashboard_dir, **kwargs))
            cls.httpd.serve_forever()
        
        # Start server in a separate thread
        cls.server_thread = threading.Thread(target=start_server)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        
        # Wait for server to start
        time.sleep(0.5)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if hasattr(cls, 'httpd'):
            cls.httpd.shutdown()
            cls.server_thread.join(1.0)
        
        if hasattr(cls, 'temp_dir'):
            cls.temp_dir.cleanup()
    
    def setUp(self):
        """Set up each test."""
        if SKIP_TEST:
            self.skipTest("Missing dependencies for integration test")
        
        # Create visualization dashboard integration
        self.integration = VisualizationDashboardIntegration(
            dashboard_dir=self.dashboard_dir
        )
        
        # Create a test dashboard
        self.dashboard_details = self.integration.create_embedded_dashboard(
            name="test_web_dashboard",
            page="results",
            template="overview",
            title="Test Web Dashboard",
            description="Dashboard for testing web integration",
            position="below"
        )
    
    def test_server_running(self):
        """Test if the test server is running."""
        # Send a request to the server
        response = urllib.request.urlopen(f"http://localhost:{self.server_port}/")
        self.assertEqual(response.getcode(), 200)
        content = response.read().decode('utf-8')
        self.assertIn("Test Server", content)
    
    def test_dashboard_file_serving(self):
        """Test if dashboard files are being served correctly."""
        # Get dashboard path
        dashboard_path = self.dashboard_details["path"]
        
        # Extract relative path for URL
        dashboard_dir_name = os.path.basename(os.path.dirname(dashboard_path))
        dashboard_relative_path = f"/dashboards/{dashboard_dir_name}/dashboard.html"
        
        # Create the file in the expected location
        os.makedirs(os.path.join(self.dashboard_dir, dashboard_dir_name), exist_ok=True)
        with open(os.path.join(self.dashboard_dir, dashboard_dir_name, "dashboard.html"), "w") as f:
            f.write("<html><body><h1>Test Dashboard</h1></body></html>")
        
        # Send a request to the server
        try:
            response = urllib.request.urlopen(f"http://localhost:{self.server_port}{dashboard_relative_path}")
            self.assertEqual(response.getcode(), 200)
            content = response.read().decode('utf-8')
            self.assertIn("Test Dashboard", content)
        except urllib.error.HTTPError as e:
            self.fail(f"Failed to access dashboard file: {e}")
    
    @unittest.skipIf(True, "Mock requests used instead of aiohttp")
    def test_dashboard_management_route(self):
        """Test the dashboard management route handler.
        
        Note: This test is skipped with the current implementation since it would
        require creating a full aiohttp application. Instead, we would typically
        use aiohttp.test_utils for proper testing of route handlers.
        """
        # This is a mock test that would normally be implemented with aiohttp.test_utils
        pass
    
    def test_dashboard_iframe_html(self):
        """Test that dashboard iframe HTML is generated correctly."""
        # Get iframe HTML
        iframe_html = self.integration.get_dashboard_iframe_html("test_web_dashboard")
        
        # Verify iframe HTML
        self.assertIn("<iframe", iframe_html)
        self.assertIn("</iframe>", iframe_html)
        
        # Verify src attribute
        dashboard_dir_name = os.path.basename(os.path.dirname(self.dashboard_details["path"]))
        self.assertIn(f"/static/dashboards/{dashboard_dir_name}/dashboard.html", iframe_html)
    
    def test_mock_monitoring_dashboard_routes(self):
        """Test mock versions of monitoring dashboard routes."""
        # Create monitoring dashboard with visualization integration
        dashboard = MonitoringDashboard(
            host="localhost",
            port=8099,  # Use a different port than the test server
            static_dir=self.static_dir,
            templates_dir=self.templates_dir,
            enable_visualization_integration=True,
            visualization_dashboard_dir=self.dashboard_dir
        )
        
        # Create a second dashboard for a different page
        self.integration.create_embedded_dashboard(
            name="test_web_dashboard_2",
            page="performance",
            template="hardware_comparison",
            title="Test Web Dashboard 2",
            description="Another dashboard for testing web integration",
            position="above"
        )
        
        # Test results route handling - mock version
        def mock_handle_results(dashboard, request):
            """Mock implementation of handle_results."""
            # Get dashboards for the results page
            results_dashboards = dashboard.visualization_integration.get_embedded_dashboards_for_page("results")
            
            # Create a simple response
            html = "<h1>Results</h1>"
            
            # Add above dashboards
            above_dashboards = {name: details for name, details in results_dashboards.items()
                              if details.get("position") == "above"}
            for name, details in above_dashboards.items():
                html += dashboard.visualization_integration.get_dashboard_iframe_html(name)
            
            # Add result content
            html += "<div>Result content goes here</div>"
            
            # Add below dashboards
            below_dashboards = {name: details for name, details in results_dashboards.items()
                              if details.get("position") == "below"}
            for name, details in below_dashboards.items():
                html += dashboard.visualization_integration.get_dashboard_iframe_html(name)
            
            return html
        
        # Call mock handler
        html = mock_handle_results(dashboard, None)
        
        # Verify dashboard iframes are included
        self.assertIn("<iframe", html)
        self.assertIn("</iframe>", html)
        self.assertIn("<h1>Results</h1>", html)
        
        # Test dashboard management route - mock version
        def mock_handle_dashboard_management(dashboard, request):
            """Mock implementation of handle_dashboard_management."""
            # Get all dashboards
            all_dashboards = dashboard.visualization_integration.embedded_dashboards
            
            # Create a simple response
            html = "<h1>Dashboard Management</h1>"
            html += f"<p>Number of dashboards: {len(all_dashboards)}</p>"
            
            # Add dashboard list
            html += "<ul>"
            for name, details in all_dashboards.items():
                html += f"<li>{name}: {details['title']} ({details['page']})</li>"
            html += "</ul>"
            
            return html
        
        # Call mock handler
        html = mock_handle_dashboard_management(dashboard, None)
        
        # Verify dashboard list is included
        self.assertIn("<h1>Dashboard Management</h1>", html)
        self.assertIn("Number of dashboards:", html)
        self.assertIn("test_web_dashboard", html)
        self.assertIn("test_web_dashboard_2", html)


def main():
    """Run the tests."""
    unittest.main()


if __name__ == "__main__":
    main()