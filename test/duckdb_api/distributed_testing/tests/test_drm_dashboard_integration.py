"""
Test the integration between Dynamic Resource Management Visualization and Monitoring Dashboard.

This module tests the integration between the DRM visualization system and the monitoring dashboard,
ensuring that resource utilization, allocation, and scaling visualizations can be displayed in the 
dashboard UI.
"""

import os
import unittest
import tempfile
import asyncio
import shutil
import time
from unittest.mock import MagicMock, patch

from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from aiohttp import web

# Import the necessary modules
try:
    from duckdb_api.distributed_testing.dynamic_resource_management_visualization import DRMVisualization
    from duckdb_api.distributed_testing.dashboard.drm_visualization_integration import DRMVisualizationIntegration
    from duckdb_api.distributed_testing.dashboard.monitoring_dashboard import MonitoringDashboard
    from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_routes import handle_drm_dashboard
    DRM_VISUALIZATION_AVAILABLE = True
except ImportError:
    DRM_VISUALIZATION_AVAILABLE = False

@unittest.skipIf(not DRM_VISUALIZATION_AVAILABLE, "DRM Visualization not available")
class TestDRMDashboardIntegration(AioHTTPTestCase):
    """Test the integration between DRM Visualization and Monitoring Dashboard."""
    
    async def get_application(self):
        """Set up the test application."""
        # Create a temporary directory for test visualizations
        self.test_dir = tempfile.mkdtemp()
        self.dashboard_dir = os.path.join(self.test_dir, "dashboards")
        self.static_dir = os.path.join(self.test_dir, "static")
        os.makedirs(self.dashboard_dir, exist_ok=True)
        os.makedirs(self.static_dir, exist_ok=True)
        
        # Create a mock resource manager
        self.mock_resource_manager = MagicMock()
        self.mock_resource_manager.get_worker_resources.return_value = {
            "worker1": {"cpu": 4, "memory": 8192, "gpu": 1},
            "worker2": {"cpu": 8, "memory": 16384, "gpu": 2}
        }
        self.mock_resource_manager.get_resource_utilization.return_value = {
            "worker1": {"cpu": 0.75, "memory": 0.5, "gpu": 0.8},
            "worker2": {"cpu": 0.6, "memory": 0.4, "gpu": 0.7}
        }
        self.mock_resource_manager.get_scaling_history.return_value = [
            {"timestamp": time.time() - 3600, "event": "scale_up", "workers_added": 1},
            {"timestamp": time.time() - 1800, "event": "scale_down", "workers_removed": 1}
        ]
        
        # Create mock visualizations
        self.viz_dir = os.path.join(self.dashboard_dir, "drm_visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Create sample visualization files
        self.create_sample_visualizations()
        
        # Initialize the dashboard with a mock template loader
        self.dashboard = MonitoringDashboard(
            host='localhost',
            port=8080,
            static_dir=self.static_dir,
            template_dir=os.path.join(os.path.dirname(__file__), "..", "dashboard", "templates"),
            dashboard_dir=self.dashboard_dir
        )
        
        # Create the drm_visualization_integration instance
        self.drm_integration = DRMVisualizationIntegration(
            output_dir=self.viz_dir,
            update_interval=1,
            resource_manager=self.mock_resource_manager
        )
        
        # Patch the drm_visualization
        self.drm_integration.visualization_available = True
        self.drm_integration.drm_visualization = MagicMock()
        self.drm_integration.drm_visualization.create_resource_utilization_heatmap.return_value = \
            os.path.join(self.viz_dir, "resource_utilization_heatmap.png")
        self.drm_integration.drm_visualization.create_scaling_history_visualization.return_value = \
            os.path.join(self.viz_dir, "scaling_history.png")
        self.drm_integration.drm_visualization.create_resource_allocation_visualization.return_value = \
            os.path.join(self.viz_dir, "resource_allocation.png")
        self.drm_integration.drm_visualization.create_resource_efficiency_visualization.return_value = \
            os.path.join(self.viz_dir, "resource_efficiency.png")
        self.drm_integration.drm_visualization.create_cloud_resource_visualization.return_value = \
            os.path.join(self.viz_dir, "cloud_resource_usage.png")
        self.drm_integration.drm_visualization.create_resource_dashboard.return_value = \
            os.path.join(self.viz_dir, "dashboard.html")
        
        # Add the integration to the dashboard
        self.dashboard.drm_visualization_integration = self.drm_integration
        
        # Set up the app
        app = web.Application()
        app['dashboard'] = self.dashboard
        
        # Set up routes
        app.router.add_get('/drm-dashboard', handle_drm_dashboard)
        app.router.add_static('/static/', self.static_dir)
        
        return app
    
    def create_sample_visualizations(self):
        """Create sample visualization files for testing."""
        # Create dummy image files
        with open(os.path.join(self.viz_dir, "resource_utilization_heatmap.png"), "w") as f:
            f.write("dummy heatmap image")
        
        with open(os.path.join(self.viz_dir, "scaling_history.png"), "w") as f:
            f.write("dummy scaling history image")
        
        with open(os.path.join(self.viz_dir, "resource_allocation.png"), "w") as f:
            f.write("dummy resource allocation image")
        
        with open(os.path.join(self.viz_dir, "resource_efficiency.png"), "w") as f:
            f.write("dummy resource efficiency image")
        
        with open(os.path.join(self.viz_dir, "cloud_resource_usage.png"), "w") as f:
            f.write("dummy cloud resource usage image")
        
        # Create a dummy dashboard HTML file
        with open(os.path.join(self.viz_dir, "dashboard.html"), "w") as f:
            f.write("<html><body><h1>DRM Dashboard</h1></body></html>")
        
        # Create a visualization registry
        registry = {
            "resource_heatmap": {
                "type": "resource_heatmap",
                "title": "Resource Utilization Heatmap",
                "description": "Heatmap showing resource utilization across workers",
                "path": os.path.join(self.viz_dir, "resource_utilization_heatmap.png"),
                "updated_at": "2025-03-21T01:52:14"
            },
            "scaling_history": {
                "type": "scaling_history",
                "title": "Scaling History",
                "description": "Visualization of scaling decisions over time",
                "path": os.path.join(self.viz_dir, "scaling_history.png"),
                "updated_at": "2025-03-21T01:52:15"
            },
            "resource_allocation": {
                "type": "resource_allocation",
                "title": "Resource Allocation",
                "description": "Visualization of resource allocation across workers",
                "path": os.path.join(self.viz_dir, "resource_allocation.png"),
                "updated_at": "2025-03-21T01:52:16"
            },
            "resource_efficiency": {
                "type": "resource_efficiency",
                "title": "Resource Efficiency",
                "description": "Visualization of resource allocation efficiency",
                "path": os.path.join(self.viz_dir, "resource_efficiency.png"),
                "updated_at": "2025-03-21T01:52:17"
            },
            "cloud_resources": {
                "type": "cloud_resources",
                "title": "Cloud Resource Usage",
                "description": "Visualization of cloud provider resource usage",
                "path": os.path.join(self.viz_dir, "cloud_resource_usage.png"),
                "updated_at": "2025-03-21T01:52:18"
            },
            "dashboard": {
                "type": "dashboard",
                "title": "Resource Management Dashboard",
                "description": "Comprehensive dashboard with all visualizations",
                "path": os.path.join(self.viz_dir, "dashboard.html"),
                "updated_at": "2025-03-21T01:52:19"
            }
        }
        
        with open(os.path.join(self.viz_dir, "visualization_registry.json"), "w") as f:
            import json
            json.dump(registry, f, indent=4)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        super().tearDown()
    
    @unittest_run_loop
    async def test_drm_dashboard_route(self):
        """Test that the DRM dashboard route returns a valid response."""
        # Make a request to the dashboard
        response = await self.client.get('/drm-dashboard')
        
        # Check the response status
        self.assertEqual(response.status, 200)
        
        # Get the response content
        content = await response.text()
        
        # Check that the response contains expected elements
        self.assertIn('Dynamic Resource Management Dashboard', content)
        self.assertIn('Resource Utilization', content)
        self.assertIn('Scaling History', content)
        self.assertIn('Resource Allocation', content)
        self.assertIn('Resource Efficiency', content)
    
    @unittest_run_loop
    async def test_drm_dashboard_refresh(self):
        """Test that the DRM dashboard refresh functionality works."""
        # Make a request to refresh the dashboard
        response = await self.client.get('/drm-dashboard?refresh=true')
        
        # Check the response status
        self.assertEqual(response.status, 200)
        
        # Verify that update_visualizations was called with force=True
        self.drm_integration.update_visualizations.assert_called_once_with(force=True)
    
    @unittest_run_loop
    async def test_drm_dashboard_server_actions(self):
        """Test that the DRM dashboard server actions work."""
        # Test starting the server
        self.drm_integration.start_dashboard_server.return_value = "http://localhost:8050"
        
        response = await self.client.get('/drm-dashboard?action=start-server')
        self.assertEqual(response.status, 200)
        content = await response.text()
        
        # Verify that start_dashboard_server was called
        self.drm_integration.start_dashboard_server.assert_called_once()
        
        # Test stopping the server
        response = await self.client.get('/drm-dashboard?action=stop-server')
        self.assertEqual(response.status, 200)
        
        # Verify that stop_dashboard_server was called
        self.drm_integration.stop_dashboard_server.assert_called_once()
        
    @unittest_run_loop
    async def test_visualization_iframe_html(self):
        """Test that visualization iframe HTML is correctly generated."""
        # Make a request to the dashboard
        response = await self.client.get('/drm-dashboard')
        
        # Check the response status
        self.assertEqual(response.status, 200)
        
        # Get the response content
        content = await response.text()
        
        # Check that the response contains iframe HTML for visualizations
        self.assertIn('<img src="/static/drm_visualizations/', content)
        self.assertIn('resource_utilization_heatmap.png', content)
        self.assertIn('scaling_history.png', content)
        self.assertIn('resource_allocation.png', content)
        self.assertIn('resource_efficiency.png', content)
        
        # Check for dashboard iframe
        self.assertIn('<iframe src="/static/drm_visualizations/', content)
        self.assertIn('dashboard.html', content)

if __name__ == '__main__':
    unittest.main()