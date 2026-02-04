#!/usr/bin/env python3
"""
Test the health monitoring component of the distributed testing framework.

This script tests the HealthMonitor's ability to detect worker failures and 
trigger appropriate recovery actions.
"""

import os
import sys
import unittest
import tempfile
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import framework components
from data.duckdb.distributed_testing.coordinator import DatabaseManager
from data.duckdb.distributed_testing.health_monitor import HealthMonitor
from data.duckdb.distributed_testing.coordinator import WORKER_STATUS_ACTIVE, WORKER_STATUS_UNAVAILABLE


class HealthMonitorTest(unittest.TestCase):
    """
    Tests for the HealthMonitor component.
    
    Tests the detection of worker failures and recovery mechanisms.
    """
    
    def setUp(self):
        """Set up test environment with HealthMonitor and test data."""
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".duckdb")
        
        # Create database manager
        self.db_manager = DatabaseManager(self.db_path)
        
        # Add test workers to database
        for i in range(3):
            worker_id = f"test_worker_{i}"
            hostname = f"test_host_{i}"
            capabilities = {
                "hardware_types": ["cpu"],
                "memory_gb": 8
            }
            self.db_manager.add_worker(worker_id, hostname, capabilities)
            self.db_manager.update_worker_status(worker_id, WORKER_STATUS_ACTIVE)
        
        # Create dictionary to track recovery actions
        self.recovery_actions = {}
        
        # Create mock recovery function
        def mock_recovery_action(worker_id, alert_type, alert_data):
            self.recovery_actions[worker_id] = {
                "alert_type": alert_type,
                "alert_data": alert_data,
                "time": datetime.now()
            }
            return True
        
        # Create health monitor with short heartbeat timeout and check interval
        self.health_monitor = HealthMonitor(
            db_manager=self.db_manager,
            heartbeat_timeout=2,  # 2 seconds timeout for testing
            check_interval=1,     # Check every 1 second
            recovery_action=mock_recovery_action
        )
        
        # Start health monitor in a separate thread
        self.monitor_thread = threading.Thread(
            target=self.health_monitor.start_monitoring,
            daemon=True
        )
        self.monitor_thread.start()
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop health monitor
        self.health_monitor.stop_monitoring()
        self.monitor_thread.join(timeout=5.0)
        
        # Close the database connection
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close()
        
        # Remove temporary database
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def test_heartbeat_timeout_detection(self):
        """Test detection of worker heartbeat timeouts."""
        # Initially, all workers should be active
        for i in range(3):
            worker_id = f"test_worker_{i}"
            worker = self.db_manager.get_worker(worker_id)
            self.assertEqual(worker["status"], WORKER_STATUS_ACTIVE)
        
        # Update heartbeat for two workers
        self.db_manager.update_worker_heartbeat("test_worker_0")
        self.db_manager.update_worker_heartbeat("test_worker_1")
        
        # Let third worker timeout
        time.sleep(3)  # Wait for heartbeat timeout
        
        # Check that the third worker is marked as unavailable
        worker = self.db_manager.get_worker("test_worker_2")
        self.assertEqual(worker["status"], WORKER_STATUS_UNAVAILABLE)
        
        # Verify recovery action was triggered
        self.assertIn("test_worker_2", self.recovery_actions)
        self.assertEqual(self.recovery_actions["test_worker_2"]["alert_type"], "heartbeat_timeout")
    
    def test_recovery_from_unavailable(self):
        """Test recovery of workers from unavailable state."""
        # Mark a worker as unavailable
        self.db_manager.update_worker_status("test_worker_0", WORKER_STATUS_UNAVAILABLE)
        
        # Update heartbeat for the worker to simulate reconnection
        self.db_manager.update_worker_heartbeat("test_worker_0")
        
        # Update worker heartbeat for recently active workers too
        self.db_manager.update_worker_heartbeat("test_worker_1")
        self.db_manager.update_worker_heartbeat("test_worker_2")
        
        # Wait for health check to run
        time.sleep(2)
        
        # Check that the worker is active again
        worker = self.db_manager.get_worker("test_worker_0")
        self.assertEqual(worker["status"], WORKER_STATUS_ACTIVE)
    
    def test_multiple_worker_failures(self):
        """Test detection of multiple worker failures."""
        # Let all workers timeout
        time.sleep(3)  # Wait for heartbeat timeout
        
        # Check that all workers are marked as unavailable
        for i in range(3):
            worker_id = f"test_worker_{i}"
            worker = self.db_manager.get_worker(worker_id)
            self.assertEqual(worker["status"], WORKER_STATUS_UNAVAILABLE)
            
            # Verify recovery action was triggered for each worker
            self.assertIn(worker_id, self.recovery_actions)
            self.assertEqual(self.recovery_actions[worker_id]["alert_type"], "heartbeat_timeout")
    
    def test_monitoring_stopping(self):
        """Test stopping the health monitor."""
        # Stop the monitoring
        self.health_monitor.stop_monitoring()
        
        # Wait for thread to stop
        self.monitor_thread.join(timeout=5.0)
        
        # Verify monitoring has stopped
        self.assertFalse(self.health_monitor.is_monitoring())
        
        # Mark all workers as active again
        for i in range(3):
            worker_id = f"test_worker_{i}"
            self.db_manager.update_worker_status(worker_id, WORKER_STATUS_ACTIVE)
        
        # Update heartbeat for two workers, leave one to timeout
        self.db_manager.update_worker_heartbeat("test_worker_0")
        self.db_manager.update_worker_heartbeat("test_worker_1")
        
        # Wait for what would be a heartbeat timeout
        time.sleep(3)
        
        # The third worker should still be marked as active since monitoring is stopped
        worker = self.db_manager.get_worker("test_worker_2")
        self.assertEqual(worker["status"], WORKER_STATUS_ACTIVE)
    
    def test_get_health_status(self):
        """Test getting health status of the system."""
        # Get health status
        health_status = self.health_monitor.get_health_status()
        
        # Verify status contains worker data
        self.assertIn("workers", health_status)
        self.assertEqual(len(health_status["workers"]), 3)
        
        # Verify status shows all workers as healthy initially
        for worker in health_status["workers"]:
            self.assertTrue(worker["healthy"])
        
        # Let one worker timeout
        time.sleep(3)
        
        # Get updated health status
        health_status = self.health_monitor.get_health_status()
        
        # Count healthy and unhealthy workers
        healthy_count = sum(1 for w in health_status["workers"] if w["healthy"])
        unhealthy_count = sum(1 for w in health_status["workers"] if not w["healthy"])
        
        # Should have 0 healthy workers and 3 unhealthy
        self.assertEqual(healthy_count, 0)
        self.assertEqual(unhealthy_count, 3)
        
        # Update heartbeat for all workers
        for i in range(3):
            worker_id = f"test_worker_{i}"
            self.db_manager.update_worker_heartbeat(worker_id)
        
        # Wait for health check to run
        time.sleep(2)
        
        # Get updated health status
        health_status = self.health_monitor.get_health_status()
        
        # All workers should be healthy again
        for worker in health_status["workers"]:
            self.assertTrue(worker["healthy"])
    
    def test_get_active_alerts(self):
        """Test getting active alerts from health monitor."""
        # Initially there should be no alerts
        alerts = self.health_monitor.get_active_alerts()
        self.assertEqual(len(alerts), 0)
        
        # Let workers timeout to generate alerts
        time.sleep(3)
        
        # Get active alerts
        alerts = self.health_monitor.get_active_alerts()
        
        # Should have alerts for all workers
        self.assertEqual(len(alerts), 3)
        
        # Verify alert properties
        for alert in alerts:
            self.assertIn("worker_id", alert)
            self.assertIn("type", alert)
            self.assertIn("time", alert)
            self.assertEqual(alert["type"], "heartbeat_timeout")
            
            # Worker ID should be one of the test workers
            self.assertIn(alert["worker_id"], [f"test_worker_{i}" for i in range(3)])


if __name__ == "__main__":
    unittest.main()