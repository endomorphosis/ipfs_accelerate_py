#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Multi-Device Orchestrator component.
"""

import unittest
import os
import sys
import json
import time
import anyio
import threading
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Add parent directory to path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from multi_device_orchestrator import (
    MultiDeviceOrchestrator,
    TaskStatus,
    SubtaskStatus,
    SplitStrategy
)


class TestMultiDeviceOrchestrator(unittest.TestCase):
    """Test suite for MultiDeviceOrchestrator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_coordinator = MagicMock()
        self.mock_task_manager = MagicMock()
        self.mock_worker_manager = MagicMock()
        self.mock_resource_manager = MagicMock()
        
        # Initialize orchestrator
        self.orchestrator = MultiDeviceOrchestrator(
            coordinator=self.mock_coordinator,
            task_manager=self.mock_task_manager,
            worker_manager=self.mock_worker_manager,
            resource_manager=self.mock_resource_manager
        )
        
        # Sample task data
        self.sample_task = {
            "task_id": "test_task_1",
            "type": "benchmark",
            "config": {
                "model": "bert-base-uncased",
                "batch_size": 4
            },
            "input_data": [f"input_{i}" for i in range(10)]
        }

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop orchestrator
        self.orchestrator.stop()

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.orchestrator.coordinator, self.mock_coordinator)
        self.assertEqual(self.orchestrator.task_manager, self.mock_task_manager)
        self.assertEqual(self.orchestrator.worker_manager, self.mock_worker_manager)
        self.assertEqual(self.orchestrator.resource_manager, self.mock_resource_manager)
        
        self.assertIsNotNone(self.orchestrator.split_strategies)
        self.assertIsNotNone(self.orchestrator.merge_strategies)
        self.assertIsNotNone(self.orchestrator.orchestrated_tasks)
        self.assertIsNotNone(self.orchestrator.subtasks)
        self.assertIsNotNone(self.orchestrator.task_subtasks)
        self.assertIsNotNone(self.orchestrator.subtask_results)

    def test_orchestrate_task_data_parallel(self):
        """Test orchestrating a task with data parallel strategy."""
        # Mock worker manager to have 3 workers
        self.mock_worker_manager.workers = {"worker1": {}, "worker2": {}, "worker3": {}}
        
        # Orchestrate task
        task_id = self.orchestrator.orchestrate_task(
            task_data=self.sample_task,
            strategy=SplitStrategy.DATA_PARALLEL
        )
        
        # Let the orchestration thread run
        time.sleep(0.5)
        
        # Verify task was recorded
        self.assertIn(task_id, self.orchestrator.orchestrated_tasks)
        self.assertEqual(self.orchestrator.orchestrated_tasks[task_id]["strategy"], SplitStrategy.DATA_PARALLEL)
        
        # Verify task status was updated
        self.assertEqual(self.orchestrator.orchestrated_tasks[task_id]["status"], TaskStatus.IN_PROGRESS)
        
        # Verify subtasks were created
        self.assertIn(task_id, self.orchestrator.task_subtasks)
        self.assertGreater(len(self.orchestrator.task_subtasks[task_id]), 0)
        
        # Verify task_manager.add_task was called for each subtask
        self.mock_task_manager.add_task.assert_called()
        self.assertEqual(self.mock_task_manager.add_task.call_count, len(self.orchestrator.task_subtasks[task_id]))

    def test_orchestrate_task_model_parallel(self):
        """Test orchestrating a task with model parallel strategy."""
        # Modify task to include model components
        task_data = self.sample_task.copy()
        task_data["model_components"] = ["embedding", "encoder", "decoder"]
        
        # Orchestrate task
        task_id = self.orchestrator.orchestrate_task(
            task_data=task_data,
            strategy=SplitStrategy.MODEL_PARALLEL
        )
        
        # Let the orchestration thread run
        time.sleep(0.5)
        
        # Verify task was recorded
        self.assertIn(task_id, self.orchestrator.orchestrated_tasks)
        self.assertEqual(self.orchestrator.orchestrated_tasks[task_id]["strategy"], SplitStrategy.MODEL_PARALLEL)
        
        # Verify task status was updated
        self.assertEqual(self.orchestrator.orchestrated_tasks[task_id]["status"], TaskStatus.IN_PROGRESS)
        
        # Verify subtasks were created
        self.assertIn(task_id, self.orchestrator.task_subtasks)
        self.assertEqual(len(self.orchestrator.task_subtasks[task_id]), 3)  # One for each component
        
        # Verify task_manager.add_task was called for each subtask
        self.mock_task_manager.add_task.assert_called()
        self.assertEqual(self.mock_task_manager.add_task.call_count, 3)

    def test_orchestrate_task_ensemble(self):
        """Test orchestrating a task with ensemble strategy."""
        # Orchestrate task
        task_id = self.orchestrator.orchestrate_task(
            task_data=self.sample_task,
            strategy=SplitStrategy.ENSEMBLE
        )
        
        # Let the orchestration thread run
        time.sleep(0.5)
        
        # Verify task was recorded
        self.assertIn(task_id, self.orchestrator.orchestrated_tasks)
        self.assertEqual(self.orchestrator.orchestrated_tasks[task_id]["strategy"], SplitStrategy.ENSEMBLE)
        
        # Verify task status was updated
        self.assertEqual(self.orchestrator.orchestrated_tasks[task_id]["status"], TaskStatus.IN_PROGRESS)
        
        # Verify subtasks were created
        self.assertIn(task_id, self.orchestrator.task_subtasks)
        self.assertGreater(len(self.orchestrator.task_subtasks[task_id]), 0)

    def test_get_task_status(self):
        """Test getting task status."""
        # Orchestrate task
        task_id = self.orchestrator.orchestrate_task(
            task_data=self.sample_task,
            strategy=SplitStrategy.DATA_PARALLEL
        )
        
        # Let the orchestration thread run
        time.sleep(0.5)
        
        # Get task status
        status = self.orchestrator.get_task_status(task_id)
        
        # Verify status
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], TaskStatus.IN_PROGRESS)
        self.assertIn("subtasks", status)
        self.assertIn("total_subtasks", status)
        self.assertIn("completion_percentage", status)
        
        # Test non-existent task
        status = self.orchestrator.get_task_status("non_existent_task")
        self.assertEqual(status["status"], "not_found")

    def test_cancel_task(self):
        """Test cancelling a task."""
        # Orchestrate task
        task_id = self.orchestrator.orchestrate_task(
            task_data=self.sample_task,
            strategy=SplitStrategy.DATA_PARALLEL
        )
        
        # Let the orchestration thread run
        time.sleep(0.5)
        
        # Cancel task
        result = self.orchestrator.cancel_task(task_id)
        
        # Verify result
        self.assertTrue(result)
        
        # Verify task status was updated
        self.assertEqual(self.orchestrator.orchestrated_tasks[task_id]["status"], TaskStatus.CANCELLED)
        
        # Verify subtasks were cancelled
        for subtask_id in self.orchestrator.task_subtasks[task_id]:
            if subtask_id in self.orchestrator.subtasks:
                self.assertEqual(self.orchestrator.subtasks[subtask_id]["status"], SubtaskStatus.CANCELLED)
        
        # Test cancelling non-existent task
        result = self.orchestrator.cancel_task("non_existent_task")
        self.assertFalse(result)

    def test_process_subtask_result(self):
        """Test processing subtask results."""
        # Orchestrate task
        task_id = self.orchestrator.orchestrate_task(
            task_data=self.sample_task,
            strategy=SplitStrategy.DATA_PARALLEL
        )
        
        # Let the orchestration thread run
        time.sleep(0.5)
        
        # Get a subtask ID
        subtask_ids = list(self.orchestrator.task_subtasks[task_id])
        self.assertGreater(len(subtask_ids), 0)
        subtask_id = subtask_ids[0]
        
        # Process result for the subtask
        self.orchestrator.process_subtask_result(
            subtask_id=subtask_id,
            result={"status": "completed", "output": "Test output"},
            success=True
        )
        
        # Verify subtask status was updated
        self.assertEqual(self.orchestrator.subtasks[subtask_id]["status"], SubtaskStatus.COMPLETED)
        
        # Verify result was stored
        self.assertIn(subtask_id, self.orchestrator.subtask_results)
        
        # Process results for all subtasks
        for subtask_id in subtask_ids[1:]:
            self.orchestrator.process_subtask_result(
                subtask_id=subtask_id,
                result={"status": "completed", "output": f"Output for {subtask_id}"},
                success=True
            )
        
        # Wait for task completion
        time.sleep(0.5)
        
        # Verify task status was updated to COMPLETED
        self.assertEqual(self.orchestrator.orchestrated_tasks[task_id]["status"], TaskStatus.COMPLETED)
        
        # Verify task result is available
        self.assertIn("result", self.orchestrator.orchestrated_tasks[task_id])

    def test_process_subtask_result_with_failure(self):
        """Test processing subtask results with failure."""
        # Orchestrate task
        task_id = self.orchestrator.orchestrate_task(
            task_data=self.sample_task,
            strategy=SplitStrategy.DATA_PARALLEL
        )
        
        # Let the orchestration thread run
        time.sleep(0.5)
        
        # Get a subtask ID
        subtask_ids = list(self.orchestrator.task_subtasks[task_id])
        self.assertGreater(len(subtask_ids), 0)
        subtask_id = subtask_ids[0]
        
        # Process result for the subtask with failure
        self.orchestrator.process_subtask_result(
            subtask_id=subtask_id,
            result={"status": "failed", "error": "Test error"},
            success=False
        )
        
        # Verify subtask status was updated
        self.assertEqual(self.orchestrator.subtasks[subtask_id]["status"], SubtaskStatus.FAILED)
        
        # Verify result was stored
        self.assertIn(subtask_id, self.orchestrator.subtask_results)
        
        # Wait for task completion
        time.sleep(0.5)
        
        # Verify task status was updated to FAILED
        self.assertEqual(self.orchestrator.orchestrated_tasks[task_id]["status"], TaskStatus.FAILED)

    def test_data_parallel_merge(self):
        """Test merging results from data parallel subtasks."""
        # Create a task
        task_id = "merge_test_1"
        self.orchestrator.orchestrated_tasks[task_id] = {
            "task_data": self.sample_task,
            "strategy": SplitStrategy.DATA_PARALLEL,
            "status": TaskStatus.IN_PROGRESS,
            "start_time": datetime.now()
        }
        
        # Create subtasks
        subtask_ids = [f"{task_id}_{i}" for i in range(3)]
        self.task_subtasks = {task_id: set(subtask_ids)}
        
        # Create subtask results
        for i, subtask_id in enumerate(subtask_ids):
            # Create subtask
            self.orchestrator.subtasks[subtask_id] = {
                "subtask_id": subtask_id,
                "task_id": task_id,
                "strategy": SplitStrategy.DATA_PARALLEL,
                "status": SubtaskStatus.COMPLETED,
                "subtask_data": {
                    "partition_index": i,
                    "num_partitions": 3
                }
            }
            
            # Create result
            self.orchestrator.subtask_results[subtask_id] = {
                "status": "completed",
                "results": [f"result_{i}_{j}" for j in range(3)],
                "metrics": {
                    "latency": 10 + i,
                    "throughput": 100 - i * 10
                }
            }
        
        # Manually call merge
        result = self.orchestrator._merge_subtask_results(task_id)
        
        # Verify merged result
        self.assertEqual(result["task_id"], task_id)
        self.assertEqual(result["strategy"], "data_parallel")
        self.assertEqual(result["num_partitions"], 3)
        self.assertEqual(len(result["results"]), 9)  # 3 results per subtask
        self.assertIn("metrics", result)
        self.assertIn("latency", result["metrics"])
        self.assertIn("throughput", result["metrics"])

    def test_model_parallel_merge(self):
        """Test merging results from model parallel subtasks."""
        # Create a task
        task_id = "merge_test_2"
        self.orchestrator.orchestrated_tasks[task_id] = {
            "task_data": self.sample_task,
            "strategy": SplitStrategy.MODEL_PARALLEL,
            "status": TaskStatus.IN_PROGRESS,
            "start_time": datetime.now()
        }
        
        # Create subtasks
        components = ["embedding", "encoder", "decoder"]
        subtask_ids = [f"{task_id}_{i}" for i in range(len(components))]
        self.task_subtasks = {task_id: set(subtask_ids)}
        
        # Create subtask results
        for i, subtask_id in enumerate(subtask_ids):
            # Create subtask
            self.orchestrator.subtasks[subtask_id] = {
                "subtask_id": subtask_id,
                "task_id": task_id,
                "strategy": SplitStrategy.MODEL_PARALLEL,
                "status": SubtaskStatus.COMPLETED,
                "subtask_data": {
                    "model_component": components[i],
                    "component_index": i,
                    "num_components": len(components)
                }
            }
            
            # Create result
            self.orchestrator.subtask_results[subtask_id] = {
                "status": "completed",
                "metrics": {
                    f"metric_{components[i]}": i * 10
                }
            }
            
            # Add output to last component
            if i == len(components) - 1:
                self.orchestrator.subtask_results[subtask_id]["output"] = "Final output"
        
        # Manually call merge
        result = self.orchestrator._merge_subtask_results(task_id)
        
        # Verify merged result
        self.assertEqual(result["task_id"], task_id)
        self.assertEqual(result["strategy"], "model_parallel")
        self.assertEqual(result["num_components"], 3)
        self.assertIn("component_results", result)
        for component in components:
            self.assertIn(component, result["component_results"])
        self.assertIn("output", result)
        self.assertEqual(result["output"], "Final output")

    def test_ensemble_merge(self):
        """Test merging results from ensemble subtasks."""
        # Create a task
        task_id = "merge_test_3"
        self.orchestrator.orchestrated_tasks[task_id] = {
            "task_data": self.sample_task,
            "strategy": SplitStrategy.ENSEMBLE,
            "status": TaskStatus.IN_PROGRESS,
            "start_time": datetime.now()
        }
        
        # Create subtasks
        variants = ["fp16", "fp32", "batch1", "batch4"]
        subtask_ids = [f"{task_id}_{i}" for i in range(len(variants))]
        self.task_subtasks = {task_id: set(subtask_ids)}
        
        # Create subtask results
        for i, subtask_id in enumerate(subtask_ids):
            # Create subtask
            self.orchestrator.subtasks[subtask_id] = {
                "subtask_id": subtask_id,
                "task_id": task_id,
                "strategy": SplitStrategy.ENSEMBLE,
                "status": SubtaskStatus.COMPLETED,
                "subtask_data": {
                    "ensemble_variant": variants[i],
                    "ensemble_index": i,
                    "num_ensembles": len(variants)
                }
            }
            
            # Create result
            self.orchestrator.subtask_results[subtask_id] = {
                "status": "completed",
                "predictions": {
                    "class_0": 0.1 + i * 0.1,
                    "class_1": 0.9 - i * 0.1
                },
                "metrics": {
                    "accuracy": 0.8 + i * 0.02,
                    "latency": 10 - i
                }
            }
        
        # Manually call merge
        result = self.orchestrator._merge_subtask_results(task_id)
        
        # Verify merged result
        self.assertEqual(result["task_id"], task_id)
        self.assertEqual(result["strategy"], "ensemble")
        self.assertEqual(result["num_ensembles"], 4)
        self.assertIn("variant_results", result)
        for variant in variants:
            self.assertIn(variant, result["variant_results"])
        self.assertIn("ensemble_predictions", result)
        self.assertIn("ensemble_metrics", result)
        self.assertIn("accuracy", result["ensemble_metrics"])
        self.assertIn("latency", result["ensemble_metrics"])


if __name__ == '__main__':
    unittest.main()