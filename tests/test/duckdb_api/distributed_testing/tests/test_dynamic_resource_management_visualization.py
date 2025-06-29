#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration tests for the Dynamic Resource Management Visualization component.

This test suite verifies that the visualization capabilities of the Dynamic Resource Management
system are working correctly, including:
- Creating various types of visualizations
- Handling test data and simulations
- Generating dashboard components
- Integration with DynamicResourceManager
"""

import unittest
import sys
import os
import json
import time
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock

# Add parent directory to path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the components to test
from dynamic_resource_management_visualization import DRMVisualization
from dynamic_resource_manager import DynamicResourceManager, ScalingDecision


class TestDRMVisualization(unittest.TestCase):
    """Test suite for DRMVisualization class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a DynamicResourceManager instance with test data
        self.drm = DynamicResourceManager(
            target_utilization=0.7,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3
        )
        
        # Register test workers
        self.drm.register_worker("worker-1", {
            "cpu": {"cores": 8, "available_cores": 4},
            "memory": {"total_mb": 16384, "available_mb": 8192},
            "gpu": {"devices": 1, "memory_mb": 8192, "available_memory_mb": 4096}
        })
        
        self.drm.register_worker("worker-2", {
            "cpu": {"cores": 4, "available_cores": 3},
            "memory": {"total_mb": 8192, "available_mb": 6144}
        })
        
        # Create a visualization instance with non-interactive mode for testing
        self.visualization = DRMVisualization(
            dynamic_resource_manager=self.drm,
            output_dir=self.temp_dir,
            interactive=False,
            update_interval=1  # Fast updates for testing
        )
        
        # Mock the last_scaling_decision
        self.drm.last_scaling_decision = ScalingDecision(
            action="scale_up",
            reason="Test scaling decision",
            count=2,
            utilization=0.85
        )
        
        # Generate some test data
        self._generate_test_data()

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop data collection
        self.visualization.cleanup()
        
        # Clean up the DRM
        self.drm.cleanup()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def _generate_test_data(self):
        """Generate test data for visualizations."""
        # Reserve some resources to create utilization
        self.drm.reserve_resources(
            task_id="task-1",
            resource_requirements={
                "cpu_cores": 2,
                "memory_mb": 4096,
                "gpu_memory_mb": 2048
            }
        )
        
        self.drm.reserve_resources(
            task_id="task-2",
            resource_requirements={
                "cpu_cores": 1,
                "memory_mb": 2048
            }
        )
        
        # Add a scaling history entry
        scaling_history_entry = {
            "timestamp": datetime.now(),
            "decision": {
                "action": "scale_up",
                "reason": "Testing visualization",
                "count": 2
            }
        }
        
        # Add directly to the visualization's history
        self.visualization.scaling_history.append(scaling_history_entry)
        
        # Force a data collection cycle to populate resource history
        self.visualization._collect_resource_data()
        
        # Add cloud usage history
        self.visualization.cloud_usage_history = {
            "aws": {
                "workers": [
                    {
                        "timestamp": datetime.now() - timedelta(minutes=10),
                        "count": 2,
                        "workers": ["worker-1", "worker-2"]
                    },
                    {
                        "timestamp": datetime.now(),
                        "count": 3,
                        "workers": ["worker-1", "worker-2", "worker-3"]
                    }
                ],
                "cost": [
                    {
                        "timestamp": datetime.now() - timedelta(minutes=10),
                        "cost": 1.50
                    },
                    {
                        "timestamp": datetime.now(),
                        "cost": 2.25
                    }
                ]
            },
            "gcp": {
                "workers": [
                    {
                        "timestamp": datetime.now() - timedelta(minutes=10),
                        "count": 1,
                        "workers": ["worker-4"]
                    },
                    {
                        "timestamp": datetime.now(),
                        "count": 1,
                        "workers": ["worker-4"]
                    }
                ],
                "cost": [
                    {
                        "timestamp": datetime.now() - timedelta(minutes=10),
                        "cost": 0.75
                    },
                    {
                        "timestamp": datetime.now(),
                        "cost": 0.75
                    }
                ]
            }
        }

    def test_initialization(self):
        """Test initialization of the visualization module."""
        self.assertEqual(self.visualization.output_dir, self.temp_dir)
        self.assertEqual(self.visualization.drm, self.drm)
        self.assertEqual(self.visualization.interactive, False)
        self.assertGreater(len(self.visualization.resource_history), 0)
        self.assertGreater(len(self.visualization.scaling_history), 0)
        self.assertEqual(len(self.visualization.worker_history), 2)  # Two workers

    def test_resource_utilization_heatmap(self):
        """Test resource utilization heatmap visualization."""
        # Create the visualization
        output_path = self.visualization.create_resource_utilization_heatmap()
        
        # Check that the file was created
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)

    def test_scaling_history_visualization(self):
        """Test scaling history visualization."""
        # Create the visualization
        output_path = self.visualization.create_scaling_history_visualization()
        
        # Check that the file was created
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)

    def test_resource_allocation_visualization(self):
        """Test resource allocation visualization."""
        # Create the visualization
        output_path = self.visualization.create_resource_allocation_visualization()
        
        # Check that the file was created
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)

    def test_resource_efficiency_visualization(self):
        """Test resource efficiency visualization."""
        # Create the visualization
        output_path = self.visualization.create_resource_efficiency_visualization()
        
        # Check that the file was created
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)

    def test_cloud_resource_visualization(self):
        """Test cloud resource visualization."""
        # Create the visualization
        output_path = self.visualization.create_cloud_resource_visualization()
        
        # Check that the file was created
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)

    @patch('matplotlib.pyplot.savefig')
    def test_create_resource_dashboard(self, mock_savefig):
        """Test dashboard creation."""
        # Mock matplotlib to avoid file format issues
        mock_savefig.return_value = None
        
        # Create our own visualization object with mocked outputs for this test
        with patch('os.path.join', return_value='mock_path.html'):
            with patch('os.path.basename', return_value='mock_file.html'):
                # Create a visualization instance with non-interactive mode for testing
                test_viz = DRMVisualization(
                    dynamic_resource_manager=self.drm,
                    output_dir=self.temp_dir,
                    interactive=False
                )
                
                # Add some test data
                test_viz.resource_history = self.visualization.resource_history.copy()
                test_viz.scaling_history = self.visualization.scaling_history.copy()
                test_viz.worker_history = self.visualization.worker_history.copy()
                
                # Mock all visualization methods
                with patch.object(test_viz, 'create_resource_utilization_heatmap', return_value='heatmap.png'):
                    with patch.object(test_viz, 'create_scaling_history_visualization', return_value='scaling.png'):
                        with patch.object(test_viz, 'create_resource_allocation_visualization', return_value='allocation.png'):
                            with patch.object(test_viz, 'create_resource_efficiency_visualization', return_value='efficiency.png'):
                                with patch.object(test_viz, 'create_cloud_resource_visualization', return_value='cloud.png'):
                                    # Mock file writing
                                    with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
                                        # Create the dashboard
                                        dashboard_path = test_viz.create_resource_dashboard()
                                        
                                        # Check that the dashboard creation was completed
                                        self.assertIsNotNone(dashboard_path)
                                        mock_file.assert_called_once()
                                        
                                        # Clean up
                                        test_viz.cleanup()

    @patch('tornado.web.Application')
    @patch('tornado.ioloop.IOLoop')
    def test_start_dashboard_server(self, mock_ioloop, mock_application):
        """Test starting the dashboard server."""
        # Setup mocks
        mock_app_instance = Mock()
        mock_application.return_value = mock_app_instance
        mock_server = Mock()
        mock_app_instance.listen.return_value = mock_server
        
        mock_io_instance = Mock()
        mock_ioloop.current.return_value = mock_io_instance
        
        # Test starting the server in background mode
        url = self.visualization.start_dashboard_server(port=9999, background=True)
        
        # Check that the server was started
        self.assertIsNotNone(url)
        self.assertEqual(url, "http://localhost:9999")
        mock_application.assert_called_once()
        mock_app_instance.listen.assert_called_once_with(9999)
        self.assertTrue(self.visualization.dashboard_running)
        
        # Test stopping the server
        self.visualization.stop_dashboard_server()
        mock_server.stop.assert_called_once()
        self.assertFalse(self.visualization.dashboard_running)

    def test_data_collection_and_pruning(self):
        """Test data collection and pruning functionality."""
        # Set a very short retention period
        self.visualization.data_retention_days = 0.0001  # Less than a minute
        
        # Add some old test data
        old_time = datetime.now() - timedelta(days=1)
        self.visualization.resource_history.append({
            "timestamp": old_time,
            "worker_count": 3,
            "active_tasks": 5,
            "overall_utilization": {"cpu": 0.7, "memory": 0.6, "gpu": 0.5, "overall": 0.6},
            "workers": {}
        })
        
        self.visualization.scaling_history.append({
            "timestamp": old_time,
            "decision": {"action": "scale_down", "reason": "Old test data"}
        })
        
        # Run data pruning
        self.visualization._prune_old_data()
        
        # Check that old data was pruned
        for entry in self.visualization.resource_history:
            self.assertGreater(entry["timestamp"], old_time)
            
        for entry in self.visualization.scaling_history:
            self.assertGreater(entry["timestamp"], old_time)

    def test_interactive_mode(self):
        """Test visualization in interactive mode if Plotly is available."""
        # Only run this test if Plotly is available
        plotly_available = False
        try:
            import plotly
            plotly_available = True
        except ImportError:
            pass
            
        if plotly_available:
            interactive_viz = DRMVisualization(
                dynamic_resource_manager=self.drm,
                output_dir=self.temp_dir,
                interactive=True
            )
            
            # Create a visualization in interactive mode
            output_path = interactive_viz.create_resource_utilization_heatmap()
            
            # Check that the file was created as HTML
            self.assertIsNotNone(output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(output_path.endswith('.html'))
            
            # Clean up
            interactive_viz.cleanup()

    def test_no_drm_available(self):
        """Test visualization functionality when DRM is not available."""
        # Create visualization without DRM
        no_drm_viz = DRMVisualization(
            dynamic_resource_manager=None,
            output_dir=self.temp_dir,
            interactive=False
        )
        
        # Add some test data manually
        no_drm_viz.resource_history = [{
            "timestamp": datetime.now(),
            "worker_count": 2,
            "active_tasks": 3,
            "overall_utilization": {"cpu": 0.6, "memory": 0.5, "gpu": 0.4, "overall": 0.5},
            "workers": {"worker-1": {}, "worker-2": {}}
        }]
        
        no_drm_viz.worker_history = {
            "worker-1": [{
                "timestamp": datetime.now(),
                "utilization": {"cpu": 0.6, "memory": 0.5, "gpu": 0.4, "overall": 0.5},
                "tasks": 2,
                "resources": {}
            }],
            "worker-2": [{
                "timestamp": datetime.now(),
                "utilization": {"cpu": 0.7, "memory": 0.6, "gpu": 0.0, "overall": 0.65},
                "tasks": 1,
                "resources": {}
            }]
        }
        
        # Create visualizations
        output_path = no_drm_viz.create_resource_utilization_heatmap()
        
        # Check that the file was created
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Clean up
        no_drm_viz.cleanup()


if __name__ == '__main__':
    unittest.main()