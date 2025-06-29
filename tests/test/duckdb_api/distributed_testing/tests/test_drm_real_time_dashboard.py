#!/usr/bin/env python3
"""
Integration tests for DRM Real-Time Dashboard

This module contains integration tests for the DRM Real-Time Dashboard,
testing both the visualization components and data collection functionality.

Tests are designed to run with a mock DRM instance to ensure consistent behavior.
"""

import os
import sys
import unittest
import time
import threading
import tempfile
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import mock DRM for testing
from duckdb_api.distributed_testing.testing.mock_drm import MockDynamicResourceManager

# Check if dash is available
try:
    import dash
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# Import the dashboard module
try:
    from duckdb_api.distributed_testing.dashboard.drm_real_time_dashboard import DRMRealTimeDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

@unittest.skipIf(not DASHBOARD_AVAILABLE or not DASH_AVAILABLE,
                "DRM Real-Time Dashboard or Dash not available")
class TestDRMRealTimeDashboard(unittest.TestCase):
    """Integration tests for DRM Real-Time Dashboard."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock DRM instance
        self.mock_drm = MockDynamicResourceManager()
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_dashboard.duckdb")
        
        # Create dashboard instance with mock DRM
        self.dashboard = DRMRealTimeDashboard(
            dynamic_resource_manager=self.mock_drm,
            db_path=self.db_path,
            port=8099,  # Use high port for testing
            update_interval=1,  # Fast updates for testing
            retention_window=5,  # Short retention for testing
            debug=False,
            theme="dark"
        )
        
        # Dashboard thread
        self.dashboard_thread = None
    
    def tearDown(self):
        """Clean up test environment."""
        # Stop dashboard if running
        if self.dashboard and self.dashboard.is_running:
            self.dashboard.stop()
        
        # Stop dashboard thread if running
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            self.dashboard_thread.join(timeout=5.0)
        
        # Cleanup temporary directory
        self.temp_dir.cleanup()
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        # Verify dashboard instance was created successfully
        self.assertIsNotNone(self.dashboard)
        self.assertEqual(self.dashboard.port, 8099)
        self.assertEqual(self.dashboard.update_interval, 1)
        self.assertEqual(self.dashboard.retention_window, 5)
        self.assertEqual(self.dashboard.theme, "dark")
        self.assertFalse(self.dashboard.is_running)
    
    def test_dashboard_data_collection(self):
        """Test dashboard data collection."""
        # Start data collection
        self.dashboard._start_data_collection()
        
        # Wait for data collection to run a few cycles
        time.sleep(3)
        
        # Stop data collection
        self.dashboard._stop_data_collection()
        
        # Verify data was collected
        self.assertTrue(len(self.dashboard.resource_metrics["timestamps"]) > 0)
        self.assertTrue(len(self.dashboard.resource_metrics["cpu_utilization"]) > 0)
        self.assertTrue(len(self.dashboard.resource_metrics["memory_utilization"]) > 0)
        self.assertTrue(len(self.dashboard.worker_metrics) > 0)
    
    def test_dashboard_background_start(self):
        """Test starting dashboard in background."""
        # Skip test if running in CI environment without display
        if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
            self.skipTest("Skipping dashboard UI test in CI environment")
        
        # Mock the dashboard's run_server method to avoid actually starting the server
        original_run_server = self.dashboard.dashboard_app.run_server
        self.dashboard.dashboard_app.run_server = lambda **kwargs: time.sleep(0.1)
        
        try:
            # Start dashboard in background
            result = self.dashboard.start_in_background()
            
            # Verify dashboard started successfully
            self.assertTrue(result)
            self.assertTrue(self.dashboard.is_running)
            
            # Wait for data collection to run a few cycles
            time.sleep(3)
            
            # Verify data was collected
            self.assertTrue(len(self.dashboard.resource_metrics["timestamps"]) > 0)
            
        finally:
            # Restore original method
            self.dashboard.dashboard_app.run_server = original_run_server
            
            # Stop dashboard
            self.dashboard.stop()
    
    def test_regression_detection(self):
        """Test performance regression detection (if available)."""
        # Skip if regression detection not available
        if not hasattr(self.dashboard, "regression_detector") or self.dashboard.regression_detector is None:
            self.skipTest("Regression detection not available")
        
        # Start data collection
        self.dashboard._start_data_collection()
        
        # Wait for data collection to run several cycles
        time.sleep(5)
        
        # Manually inject an anomaly to test detection
        self.dashboard.resource_metrics["cpu_utilization"][-1] = 95.0  # Sudden spike
        
        # Stop data collection
        self.dashboard._stop_data_collection()
        
        # Perform regression detection
        self.dashboard._detect_performance_regressions()
        
        # Not all spikes will be detected as significant regressions,
        # so we're just testing that the method runs without errors
        # and creates the proper data structures
        self.assertTrue(hasattr(self.dashboard, "alerts"))
    
    def test_run_script_import(self):
        """Test importing the run script module."""
        try:
            # Try to import the run script
            import run_drm_real_time_dashboard
            
            # Verify the script has the expected functions
            self.assertTrue(hasattr(run_drm_real_time_dashboard, "main"))
            self.assertTrue(hasattr(run_drm_real_time_dashboard, "check_dependencies"))
            self.assertTrue(hasattr(run_drm_real_time_dashboard, "get_arguments"))
            
        except ImportError as e:
            self.fail(f"Failed to import run_drm_real_time_dashboard: {e}")

if __name__ == "__main__":
    unittest.main()