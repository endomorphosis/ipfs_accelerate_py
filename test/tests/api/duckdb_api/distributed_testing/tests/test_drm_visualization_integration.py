#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the DRM Visualization Integration component of the Distributed Testing Framework.
"""

import unittest
import sys
import os
import json
import time
import tempfile
import shutil
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the components we want to test
from dashboard.drm_visualization_integration import DRMVisualizationIntegration


class TestDRMVisualizationIntegration(unittest.TestCase):
    """Test suite for the DRM Visualization Integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock resource manager
        self.mock_resource_manager = MagicMock()
        self.mock_resource_manager.get_worker_statistics.return_value = {
            "total_workers": 2,
            "active_tasks": 3,
            "workers": {
                "worker-1": {"utilization": {"cpu": 0.7, "memory": 0.6, "gpu": 0.5, "overall": 0.6}},
                "worker-2": {"utilization": {"cpu": 0.3, "memory": 0.4, "overall": 0.35}}
            },
            "overall_utilization": {"cpu": 0.5, "memory": 0.5, "gpu": 0.25, "overall": 0.45}
        }
        
        # Mock visualization paths
        self.paths = {
            "resource_heatmap": os.path.join(self.temp_dir, "resource_heatmap.png"),
            "scaling_history": os.path.join(self.temp_dir, "scaling_history.png"),
            "resource_allocation": os.path.join(self.temp_dir, "resource_allocation.png"),
            "resource_efficiency": os.path.join(self.temp_dir, "resource_efficiency.png"),
            "dashboard": os.path.join(self.temp_dir, "dashboard.html")
        }
        
        # Create empty files at the mock paths
        for path in self.paths.values():
            with open(path, 'w') as f:
                f.write("Mock content")

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp directory
        shutil.rmtree(self.temp_dir)

    @patch('dashboard.drm_visualization_integration.DRMVisualization')
    def test_initialization(self, mock_drm_vis_class):
        """Test initialization of the integration module."""
        # Setup the mock
        mock_drm_vis_instance = MagicMock()
        mock_drm_vis_class.return_value = mock_drm_vis_instance
        
        # Create the integration
        integration = DRMVisualizationIntegration(
            output_dir=self.temp_dir,
            resource_manager=self.mock_resource_manager
        )
        
        # Check initialization
        self.assertEqual(integration.output_dir, self.temp_dir)
        self.assertEqual(integration.resource_manager, self.mock_resource_manager)
        self.assertTrue(integration.visualization_available)
        
        # Check that DRMVisualization was initialized
        mock_drm_vis_class.assert_called_once()
        self.assertEqual(integration.drm_visualization, mock_drm_vis_instance)

    @patch('dashboard.drm_visualization_integration.DRMVisualization')
    def test_update_visualizations(self, mock_drm_vis_class):
        """Test updating visualizations."""
        # Setup the mock
        mock_drm_vis_instance = MagicMock()
        mock_drm_vis_class.return_value = mock_drm_vis_instance
        
        # Set return values for visualization methods
        mock_drm_vis_instance.create_resource_utilization_heatmap.return_value = self.paths["resource_heatmap"]
        mock_drm_vis_instance.create_scaling_history_visualization.return_value = self.paths["scaling_history"]
        mock_drm_vis_instance.create_resource_allocation_visualization.return_value = self.paths["resource_allocation"]
        mock_drm_vis_instance.create_resource_efficiency_visualization.return_value = self.paths["resource_efficiency"]
        mock_drm_vis_instance.create_resource_dashboard.return_value = self.paths["dashboard"]
        
        # Create the integration
        integration = DRMVisualizationIntegration(
            output_dir=self.temp_dir,
            resource_manager=self.mock_resource_manager
        )
        
        # Update visualizations
        result = integration.update_visualizations(force=True)
        
        # Check that all visualizations were updated
        self.assertEqual(len(result), 5)
        self.assertEqual(result["resource_heatmap"], self.paths["resource_heatmap"])
        self.assertEqual(result["scaling_history"], self.paths["scaling_history"])
        self.assertEqual(result["resource_allocation"], self.paths["resource_allocation"])
        self.assertEqual(result["resource_efficiency"], self.paths["resource_efficiency"])
        self.assertEqual(result["dashboard"], self.paths["dashboard"])
        
        # Check that the registry was updated
        self.assertEqual(len(integration.visualization_registry), 5)
        
        # Verify the registry contains expected data
        for viz_type in ["resource_heatmap", "scaling_history", "resource_allocation", "resource_efficiency", "dashboard"]:
            self.assertIn(viz_type, integration.visualization_registry)
            self.assertEqual(integration.visualization_registry[viz_type]["path"], self.paths[viz_type])

    @patch('dashboard.drm_visualization_integration.DRMVisualization')
    def test_get_iframe_html(self, mock_drm_vis_class):
        """Test generating iframe HTML."""
        # Setup the mock
        mock_drm_vis_instance = MagicMock()
        mock_drm_vis_class.return_value = mock_drm_vis_instance
        
        # Create the integration
        integration = DRMVisualizationIntegration(
            output_dir=self.temp_dir,
            resource_manager=self.mock_resource_manager
        )
        
        # Add a registry entry manually
        integration.visualization_registry["dashboard"] = {
            "type": "dashboard",
            "title": "Test Dashboard",
            "description": "Dashboard for testing",
            "path": self.paths["dashboard"],
            "updated_at": datetime.now().isoformat()
        }
        
        # Generate iframe HTML
        html = integration.get_iframe_html("dashboard")
        
        # Check the HTML
        self.assertIn('<iframe', html)
        self.assertIn('Test Dashboard', html)
        self.assertIn('Dashboard for testing', html)
        self.assertIn('dashboard.html', html)

    @patch('dashboard.drm_visualization_integration.DRMVisualization')
    def test_start_stop_dashboard_server(self, mock_drm_vis_class):
        """Test starting and stopping the dashboard server."""
        # Setup the mock
        mock_drm_vis_instance = MagicMock()
        mock_drm_vis_class.return_value = mock_drm_vis_instance
        mock_drm_vis_instance.start_dashboard_server.return_value = "http://localhost:8889"
        
        # Create the integration
        integration = DRMVisualizationIntegration(
            output_dir=self.temp_dir,
            resource_manager=self.mock_resource_manager
        )
        
        # Start dashboard server
        url = integration.start_dashboard_server(port=8889)
        
        # Check result
        self.assertEqual(url, "http://localhost:8889")
        mock_drm_vis_instance.start_dashboard_server.assert_called_once_with(port=8889)
        
        # Stop dashboard server
        result = integration.stop_dashboard_server()
        
        # Check result
        self.assertTrue(result)
        mock_drm_vis_instance.stop_dashboard_server.assert_called_once()

    @patch('dashboard.drm_visualization_integration.DRMVisualization')
    def test_registry_persistence(self, mock_drm_vis_class):
        """Test that the registry is saved and loaded correctly."""
        # Setup the mock
        mock_drm_vis_instance = MagicMock()
        mock_drm_vis_class.return_value = mock_drm_vis_instance
        
        # Create the integration
        integration = DRMVisualizationIntegration(
            output_dir=self.temp_dir,
            resource_manager=self.mock_resource_manager
        )
        
        # Add registry entries
        integration.visualization_registry["test_viz"] = {
            "type": "test_viz",
            "title": "Test Visualization",
            "description": "Visualization for testing",
            "path": os.path.join(self.temp_dir, "test_viz.png"),
            "updated_at": datetime.now().isoformat()
        }
        
        # Save registry
        integration._save_registry()
        
        # Check that registry file exists
        registry_file = os.path.join(self.temp_dir, "visualization_registry.json")
        self.assertTrue(os.path.exists(registry_file))
        
        # Create a new integration to load the registry
        new_integration = DRMVisualizationIntegration(
            output_dir=self.temp_dir
        )
        
        # Check that registry was loaded
        self.assertIn("test_viz", new_integration.visualization_registry)
        self.assertEqual(
            new_integration.visualization_registry["test_viz"]["title"], 
            "Test Visualization"
        )

    @patch('dashboard.drm_visualization_integration.DRMVisualization')
    def test_cleanup(self, mock_drm_vis_class):
        """Test cleanup of resources."""
        # Setup the mock
        mock_drm_vis_instance = MagicMock()
        mock_drm_vis_class.return_value = mock_drm_vis_instance
        
        # Create the integration
        integration = DRMVisualizationIntegration(
            output_dir=self.temp_dir,
            resource_manager=self.mock_resource_manager
        )
        
        # Call cleanup
        result = integration.cleanup()
        
        # Check result
        self.assertTrue(result)
        mock_drm_vis_instance.stop_dashboard_server.assert_called_once()
        mock_drm_vis_instance.cleanup.assert_called_once()


if __name__ == '__main__':
    unittest.main()