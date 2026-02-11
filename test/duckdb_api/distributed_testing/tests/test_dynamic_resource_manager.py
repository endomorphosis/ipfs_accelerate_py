#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Dynamic Resource Manager component of the Distributed Testing Framework.
"""

import unittest
import anyio
import json
import time
from unittest.mock import MagicMock, patch
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dynamic_resource_manager import DynamicResourceManager
from constants import (
    DEFAULT_TARGET_UTILIZATION,
    DEFAULT_SCALE_UP_THRESHOLD,
    DEFAULT_SCALE_DOWN_THRESHOLD,
    DEFAULT_EVALUATION_WINDOW,
    DEFAULT_SCALE_UP_COOLDOWN,
    DEFAULT_SCALE_DOWN_COOLDOWN
)


class TestDynamicResourceManager(unittest.TestCase):
    """Test suite for DynamicResourceManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.resource_manager = DynamicResourceManager()
        
        # Sample worker resources
        self.worker1_resources = {
            "cpu": {
                "cores": 8,
                "physical_cores": 4,
                "available_cores": 6.5
            },
            "memory": {
                "total_mb": 16384,
                "available_mb": 8192
            },
            "gpu": {
                "devices": 1,
                "available_devices": 1,
                "total_memory_mb": 8192,
                "available_memory_mb": 6144
            }
        }
        
        self.worker2_resources = {
            "cpu": {
                "cores": 16,
                "physical_cores": 8,
                "available_cores": 12.0
            },
            "memory": {
                "total_mb": 32768,
                "available_mb": 24576
            },
            "gpu": {
                "devices": 2,
                "available_devices": 2,
                "total_memory_mb": 16384,
                "available_memory_mb": 14336
            }
        }
        
        # Sample task resource requirements
        self.small_task_resources = {
            "cpu_cores": 2,
            "memory_mb": 4096,
            "gpu_memory_mb": 2048
        }
        
        self.large_task_resources = {
            "cpu_cores": 8,
            "memory_mb": 16384,
            "gpu_memory_mb": 8192
        }

    def test_init_default_values(self):
        """Test initialization with default values."""
        self.assertEqual(self.resource_manager.target_utilization, DEFAULT_TARGET_UTILIZATION)
        self.assertEqual(self.resource_manager.scale_up_threshold, DEFAULT_SCALE_UP_THRESHOLD)
        self.assertEqual(self.resource_manager.scale_down_threshold, DEFAULT_SCALE_DOWN_THRESHOLD)
        self.assertEqual(self.resource_manager.evaluation_window, DEFAULT_EVALUATION_WINDOW)
        self.assertEqual(self.resource_manager.scale_up_cooldown, DEFAULT_SCALE_UP_COOLDOWN)
        self.assertEqual(self.resource_manager.scale_down_cooldown, DEFAULT_SCALE_DOWN_COOLDOWN)
        self.assertEqual(len(self.resource_manager.worker_resources), 0)
        self.assertEqual(len(self.resource_manager.resource_reservations), 0)

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        custom_manager = DynamicResourceManager(
            target_utilization=0.6,
            scale_up_threshold=0.75,
            scale_down_threshold=0.4,
            evaluation_window=600,
            scale_up_cooldown=450,
            scale_down_cooldown=900
        )
        
        self.assertEqual(custom_manager.target_utilization, 0.6)
        self.assertEqual(custom_manager.scale_up_threshold, 0.75)
        self.assertEqual(custom_manager.scale_down_threshold, 0.4)
        self.assertEqual(custom_manager.evaluation_window, 600)
        self.assertEqual(custom_manager.scale_up_cooldown, 450)
        self.assertEqual(custom_manager.scale_down_cooldown, 900)

    def test_register_worker(self):
        """Test worker registration."""
        # Register worker1
        result = self.resource_manager.register_worker("worker1", self.worker1_resources)
        self.assertTrue(result)
        self.assertIn("worker1", self.resource_manager.worker_resources)
        self.assertEqual(
            self.resource_manager.worker_resources["worker1"]["cpu"]["cores"],
            self.worker1_resources["cpu"]["cores"]
        )
        
        # Register worker2
        result = self.resource_manager.register_worker("worker2", self.worker2_resources)
        self.assertTrue(result)
        self.assertIn("worker2", self.resource_manager.worker_resources)
        
        # Register worker1 again (should update existing resources)
        updated_resources = self.worker1_resources.copy()
        updated_resources["cpu"]["available_cores"] = 7.0
        result = self.resource_manager.register_worker("worker1", updated_resources)
        self.assertTrue(result)
        self.assertEqual(
            self.resource_manager.worker_resources["worker1"]["cpu"]["available_cores"],
            7.0
        )

    def test_update_worker_resources(self):
        """Test updating worker resources."""
        # Register worker first
        self.resource_manager.register_worker("worker1", self.worker1_resources)
        
        # Update resources
        updated_resources = self.worker1_resources.copy()
        updated_resources["cpu"]["available_cores"] = 4.0
        updated_resources["memory"]["available_mb"] = 6144
        
        result = self.resource_manager.update_worker_resources("worker1", updated_resources)
        self.assertTrue(result)
        self.assertEqual(
            self.resource_manager.worker_resources["worker1"]["cpu"]["available_cores"],
            4.0
        )
        self.assertEqual(
            self.resource_manager.worker_resources["worker1"]["memory"]["available_mb"],
            6144
        )
        
        # Update non-existent worker
        result = self.resource_manager.update_worker_resources("nonexistent", updated_resources)
        self.assertFalse(result)

    def test_reserve_resources(self):
        """Test resource reservation."""
        # Register worker
        self.resource_manager.register_worker("worker1", self.worker1_resources)
        
        # Reserve resources for a task
        reservation_id = self.resource_manager.reserve_resources(
            worker_id="worker1", 
            task_id="task1", 
            resource_requirements=self.small_task_resources
        )
        
        self.assertIsNotNone(reservation_id)
        self.assertIn(reservation_id, self.resource_manager.resource_reservations)
        self.assertEqual(
            self.resource_manager.resource_reservations[reservation_id]["worker_id"],
            "worker1"
        )
        self.assertEqual(
            self.resource_manager.resource_reservations[reservation_id]["task_id"],
            "task1"
        )
        
        # Check that worker resources are updated
        worker_resources = self.resource_manager.worker_resources["worker1"]
        self.assertEqual(
            worker_resources["cpu"]["reserved_cores"],
            self.small_task_resources["cpu_cores"]
        )
        self.assertEqual(
            worker_resources["memory"]["reserved_mb"],
            self.small_task_resources["memory_mb"]
        )
        self.assertEqual(
            worker_resources["gpu"]["reserved_memory_mb"],
            self.small_task_resources["gpu_memory_mb"]
        )
        
        # Check task-reservation mapping
        self.assertEqual(
            self.resource_manager.task_reservation["task1"],
            reservation_id
        )
        
        # Check worker-tasks mapping
        self.assertIn("task1", self.resource_manager.worker_tasks["worker1"])

    def test_reserve_resources_insufficient(self):
        """Test resource reservation with insufficient resources."""
        # Register worker with limited resources
        limited_resources = {
            "cpu": {"cores": 4, "physical_cores": 2, "available_cores": 3.0},
            "memory": {"total_mb": 8192, "available_mb": 4096},
            "gpu": {"devices": 1, "available_devices": 1, "total_memory_mb": 4096, "available_memory_mb": 2048}
        }
        self.resource_manager.register_worker("limited_worker", limited_resources)
        
        # Try to reserve more resources than available
        reservation_id = self.resource_manager.reserve_resources(
            worker_id="limited_worker", 
            task_id="large_task", 
            resource_requirements=self.large_task_resources
        )
        
        self.assertIsNone(reservation_id)
        self.assertNotIn("large_task", self.resource_manager.task_reservation)

    def test_release_resources(self):
        """Test releasing reserved resources."""
        # Register worker and reserve resources
        self.resource_manager.register_worker("worker1", self.worker1_resources)
        reservation_id = self.resource_manager.reserve_resources(
            worker_id="worker1", 
            task_id="task1", 
            resource_requirements=self.small_task_resources
        )
        
        # Release resources
        result = self.resource_manager.release_resources(reservation_id)
        self.assertTrue(result)
        
        # Check that reservation is removed
        self.assertNotIn(reservation_id, self.resource_manager.resource_reservations)
        
        # Check that worker resources are updated
        worker_resources = self.resource_manager.worker_resources["worker1"]
        self.assertEqual(worker_resources["cpu"].get("reserved_cores", 0), 0)
        self.assertEqual(worker_resources["memory"].get("reserved_mb", 0), 0)
        self.assertEqual(worker_resources["gpu"].get("reserved_memory_mb", 0), 0)
        
        # Check task-reservation mapping is removed
        self.assertNotIn("task1", self.resource_manager.task_reservation)
        
        # Check worker-tasks mapping is updated
        self.assertNotIn("task1", self.resource_manager.worker_tasks["worker1"])
        
        # Try to release non-existent reservation
        result = self.resource_manager.release_resources("nonexistent")
        self.assertFalse(result)

    def test_check_resource_availability(self):
        """Test checking resource availability."""
        # Register worker
        self.resource_manager.register_worker("worker1", self.worker1_resources)
        
        # Check availability for small task (should be available)
        availability = self.resource_manager.check_resource_availability(
            worker_id="worker1",
            resource_requirements=self.small_task_resources
        )
        self.assertTrue(availability["available"])
        
        # Reserve some resources
        self.resource_manager.reserve_resources(
            worker_id="worker1", 
            task_id="task1", 
            resource_requirements=self.small_task_resources
        )
        
        # Check availability for another small task (should still be available)
        availability = self.resource_manager.check_resource_availability(
            worker_id="worker1",
            resource_requirements=self.small_task_resources
        )
        self.assertTrue(availability["available"])
        
        # Check availability for large task (should not be available)
        availability = self.resource_manager.check_resource_availability(
            worker_id="worker1",
            resource_requirements=self.large_task_resources
        )
        self.assertFalse(availability["available"])
        self.assertIn("cpu", availability["reasons"])
        
        # Check for non-existent worker
        availability = self.resource_manager.check_resource_availability(
            worker_id="nonexistent",
            resource_requirements=self.small_task_resources
        )
        self.assertFalse(availability["available"])
        self.assertIn("worker_not_found", availability["reasons"])

    def test_calculate_task_worker_fitness(self):
        """Test calculation of task-worker fitness."""
        # Register workers
        self.resource_manager.register_worker("worker1", self.worker1_resources)
        self.resource_manager.register_worker("worker2", self.worker2_resources)
        
        # Calculate fitness for small task on worker1
        fitness1 = self.resource_manager.calculate_task_worker_fitness(
            worker_id="worker1",
            resource_requirements=self.small_task_resources
        )
        
        # Calculate fitness for small task on worker2
        fitness2 = self.resource_manager.calculate_task_worker_fitness(
            worker_id="worker2",
            resource_requirements=self.small_task_resources
        )
        
        # Worker2 has more resources, so fitness should be higher
        self.assertGreater(fitness2, fitness1)
        
        # Calculate fitness for large task on worker1
        fitness3 = self.resource_manager.calculate_task_worker_fitness(
            worker_id="worker1",
            resource_requirements=self.large_task_resources
        )
        
        # Calculate fitness for large task on worker2
        fitness4 = self.resource_manager.calculate_task_worker_fitness(
            worker_id="worker2",
            resource_requirements=self.large_task_resources
        )
        
        # Worker2 has more resources, so fitness should be higher
        self.assertGreater(fitness4, fitness3)
        
        # For worker1, large task should have lower fitness than small task
        self.assertGreater(fitness1, fitness3)
        
        # For non-existent worker, fitness should be 0
        fitness5 = self.resource_manager.calculate_task_worker_fitness(
            worker_id="nonexistent",
            resource_requirements=self.small_task_resources
        )
        self.assertEqual(fitness5, 0.0)

    def test_evaluate_scaling_no_workers(self):
        """Test scaling evaluation with no workers."""
        # No workers registered yet, should recommend scale up
        scaling_decision = self.resource_manager.evaluate_scaling()
        self.assertEqual(scaling_decision.action, "scale_up")
        self.assertGreater(scaling_decision.count, 0)

    def test_evaluate_scaling_high_utilization(self):
        """Test scaling evaluation with high utilization."""
        # Register worker with high utilization
        high_util_resources = self.worker1_resources.copy()
        high_util_resources["cpu"]["available_cores"] = 1.0  # Only 1 out of 8 cores available
        high_util_resources["memory"]["available_mb"] = 2048  # Only 2GB out of 16GB available
        self.resource_manager.register_worker("high_util_worker", high_util_resources)
        
        # Mock utilization history
        self.resource_manager.utilization_history = [
            {"timestamp": datetime.now() - timedelta(minutes=4), "cpu": 0.85, "memory": 0.88, "gpu": 0.75},
            {"timestamp": datetime.now() - timedelta(minutes=3), "cpu": 0.87, "memory": 0.89, "gpu": 0.78},
            {"timestamp": datetime.now() - timedelta(minutes=2), "cpu": 0.89, "memory": 0.90, "gpu": 0.82},
            {"timestamp": datetime.now() - timedelta(minutes=1), "cpu": 0.91, "memory": 0.92, "gpu": 0.85}
        ]
        
        # Should recommend scale up
        scaling_decision = self.resource_manager.evaluate_scaling()
        self.assertEqual(scaling_decision.action, "scale_up")
        self.assertGreater(scaling_decision.count, 0)

    def test_evaluate_scaling_low_utilization(self):
        """Test scaling evaluation with low utilization."""
        # Register multiple workers with low utilization
        low_util_resources1 = self.worker1_resources.copy()
        low_util_resources1["cpu"]["available_cores"] = 7.5  # 7.5 out of 8 cores available
        low_util_resources1["memory"]["available_mb"] = 14336  # 14GB out of 16GB available
        self.resource_manager.register_worker("low_util_worker1", low_util_resources1)
        
        low_util_resources2 = self.worker2_resources.copy()
        low_util_resources2["cpu"]["available_cores"] = 15.0  # 15 out of 16 cores available
        low_util_resources2["memory"]["available_mb"] = 30720  # 30GB out of 32GB available
        self.resource_manager.register_worker("low_util_worker2", low_util_resources2)
        
        # Mock utilization history
        self.resource_manager.utilization_history = [
            {"timestamp": datetime.now() - timedelta(minutes=4), "cpu": 0.15, "memory": 0.12, "gpu": 0.08},
            {"timestamp": datetime.now() - timedelta(minutes=3), "cpu": 0.14, "memory": 0.11, "gpu": 0.07},
            {"timestamp": datetime.now() - timedelta(minutes=2), "cpu": 0.12, "memory": 0.10, "gpu": 0.06},
            {"timestamp": datetime.now() - timedelta(minutes=1), "cpu": 0.10, "memory": 0.09, "gpu": 0.05}
        ]
        
        # Set last scale down time to be outside cooldown period
        self.resource_manager.last_scale_down_time = time.time() - (self.resource_manager.scale_down_cooldown + 60)
        
        # Should recommend scale down
        scaling_decision = self.resource_manager.evaluate_scaling()
        self.assertEqual(scaling_decision.action, "scale_down")
        self.assertGreater(len(scaling_decision.worker_ids), 0)

    def test_evaluate_scaling_optimal_utilization(self):
        """Test scaling evaluation with optimal utilization."""
        # Register worker with moderate utilization
        moderate_util_resources = self.worker1_resources.copy()
        moderate_util_resources["cpu"]["available_cores"] = 4.0  # 4 out of 8 cores available
        moderate_util_resources["memory"]["available_mb"] = 8192  # 8GB out of 16GB available
        self.resource_manager.register_worker("moderate_util_worker", moderate_util_resources)
        
        # Mock utilization history
        self.resource_manager.utilization_history = [
            {"timestamp": datetime.now() - timedelta(minutes=4), "cpu": 0.55, "memory": 0.52, "gpu": 0.48},
            {"timestamp": datetime.now() - timedelta(minutes=3), "cpu": 0.54, "memory": 0.53, "gpu": 0.47},
            {"timestamp": datetime.now() - timedelta(minutes=2), "cpu": 0.56, "memory": 0.50, "gpu": 0.49},
            {"timestamp": datetime.now() - timedelta(minutes=1), "cpu": 0.55, "memory": 0.51, "gpu": 0.48}
        ]
        
        # Should recommend maintain (no scaling)
        scaling_decision = self.resource_manager.evaluate_scaling()
        self.assertEqual(scaling_decision.action, "maintain")


if __name__ == '__main__':
    unittest.main()