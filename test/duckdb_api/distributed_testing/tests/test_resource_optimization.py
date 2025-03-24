#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the ResourceOptimizer component of the Dynamic Resource Management system.

This test suite validates the ResourceOptimizer's functionality, including:
- Task resource requirement prediction
- Resource allocation for task batches
- Worker type recommendations based on task requirements
- Scaling recommendations integration with DRM
- Task result recording and workload pattern analysis
"""

import unittest
import os
import sys
import json
import time
import logging
import tempfile
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from dynamic_resource_manager import DynamicResourceManager, ScalingDecision
from resource_performance_predictor import ResourcePerformancePredictor
from cloud_provider_manager import CloudProviderManager
from resource_optimization import ResourceOptimizer, TaskRequirements, ResourceAllocation


class TestResourceOptimizer(unittest.TestCase):
    """Test suite for ResourceOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.resource_manager = MagicMock(spec=DynamicResourceManager)
        self.performance_predictor = MagicMock(spec=ResourcePerformancePredictor)
        self.cloud_manager = MagicMock(spec=CloudProviderManager)
        
        # Initialize optimizer
        self.optimizer = ResourceOptimizer(
            resource_manager=self.resource_manager,
            performance_predictor=self.performance_predictor,
            cloud_manager=self.cloud_manager
        )
        
        # Sample task data
        self.text_embedding_task = {
            "model_type": "text_embedding",
            "model_name": "bert-base-uncased",
            "batch_size": 16
        }
        
        self.text_generation_task = {
            "model_type": "text_generation",
            "model_name": "llama-7b",
            "batch_size": 1
        }
        
        self.vision_task = {
            "model_type": "vision",
            "model_name": "vit-base-patch16-224",
            "batch_size": 8
        }
        
        # Sample worker resources
        self.cpu_resources = {
            "cpu": {
                "cores": 8,
                "physical_cores": 4,
                "available_cores": 7.5
            },
            "memory": {
                "total_mb": 16384,
                "available_mb": 12288
            },
            "gpu": {
                "devices": 0,
                "available_devices": 0,
                "total_memory_mb": 0,
                "available_memory_mb": 0
            }
        }
        
        self.gpu_resources = {
            "cpu": {
                "cores": 16,
                "physical_cores": 8,
                "available_cores": 14.0
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
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up optimizer
        self.optimizer.cleanup()
    
    def test_predict_task_requirements(self):
        """Test prediction of task resource requirements."""
        # Configure performance predictor mock
        self.performance_predictor.predict_resource_requirements.return_value = {
            "cpu_cores": 4,
            "memory_mb": 8192,
            "gpu_memory_mb": 2048,
            "confidence": 0.85,
            "prediction_method": "historical",
            "execution_time_ms": 5000
        }
        
        # Test with text embedding task
        requirements = self.optimizer.predict_task_requirements(self.text_embedding_task)
        
        # Verify requirements
        self.assertEqual(requirements.cpu_cores, 4)
        self.assertEqual(requirements.memory_mb, 8192)
        self.assertEqual(requirements.gpu_memory_mb, 2048)
        self.assertEqual(requirements.model_type, "text_embedding")
        self.assertEqual(requirements.batch_size, 16)
        self.assertEqual(requirements.confidence, 0.85)
        self.assertEqual(requirements.prediction_method, "historical")
        
        # Verify predictor was called with correct arguments
        self.performance_predictor.predict_resource_requirements.assert_called_once_with(self.text_embedding_task)
    
    def test_predict_task_requirements_fallback(self):
        """Test prediction of task requirements with fallback to defaults."""
        # Configure performance predictor to raise exception
        self.performance_predictor.predict_resource_requirements.side_effect = Exception("Prediction error")
        
        # Test with text generation task (should use default requirements)
        requirements = self.optimizer.predict_task_requirements(self.text_generation_task)
        
        # Verify default requirements were used
        self.assertGreater(requirements.cpu_cores, 0)
        self.assertGreater(requirements.memory_mb, 0)
        self.assertEqual(requirements.model_type, "text_generation")
        self.assertEqual(requirements.batch_size, 1)
        
        # Verify predictor was called
        self.performance_predictor.predict_resource_requirements.assert_called_once()
    
    def test_allocate_resources(self):
        """Test allocation of resources for task batch."""
        # Configure resource manager mock
        self.resource_manager.worker_resources = {
            "worker-1": {
                "resources": self.cpu_resources
            },
            "worker-2": {
                "resources": self.gpu_resources
            }
        }
        
        # Configure reserve_resources to return reservation IDs
        self.resource_manager.reserve_resources.side_effect = lambda **kwargs: f"res-{kwargs['task_id']}"
        
        # Create task batch
        task_batch = [
            {
                "task_id": "task-1",
                "type": "benchmark",
                "priority": 1,
                "config": {
                    "model_type": "text_embedding",
                    "model": "bert-base-uncased",
                    "batch_size": 16
                }
            },
            {
                "task_id": "task-2",
                "type": "benchmark",
                "priority": 2,
                "config": {
                    "model_type": "text_generation",
                    "model": "llama-7b",
                    "batch_size": 1
                }
            }
        ]
        
        # Configure performance predictor to return different requirements for each task type
        def mock_predict_resources(task_data):
            if task_data.get("model_type") == "text_embedding":
                return {
                    "cpu_cores": 2,
                    "memory_mb": 4096,
                    "gpu_memory_mb": 0,
                    "confidence": 0.9
                }
            else:  # text_generation
                return {
                    "cpu_cores": 4,
                    "memory_mb": 8192,
                    "gpu_memory_mb": 4096,
                    "confidence": 0.9
                }
        
        self.performance_predictor.predict_resource_requirements.side_effect = mock_predict_resources
        
        # Test resource allocation
        allocations = self.optimizer.allocate_resources(task_batch, ["worker-1", "worker-2"])
        
        # Verify allocations
        self.assertEqual(len(allocations), 2)
        
        # First task (text embedding) should be allocated to worker-1 (CPU worker)
        self.assertEqual(allocations[0].task_id, "task-1")
        self.assertEqual(allocations[0].worker_id, "worker-1")
        self.assertTrue(allocations[0].allocated)
        self.assertEqual(allocations[0].reservation_id, "res-task-1")
        
        # Second task (text generation with GPU) should be allocated to worker-2 (GPU worker)
        self.assertEqual(allocations[1].task_id, "task-2")
        self.assertEqual(allocations[1].worker_id, "worker-2")
        self.assertTrue(allocations[1].allocated)
        self.assertEqual(allocations[1].reservation_id, "res-task-2")
        
        # Verify resource manager was called correctly
        self.assertEqual(self.resource_manager.reserve_resources.call_count, 2)
    
    def test_allocate_resources_insufficient(self):
        """Test allocation with insufficient resources."""
        # Configure resource manager mock
        self.resource_manager.worker_resources = {
            "worker-1": {
                "resources": {
                    "cpu": {
                        "cores": 2,
                        "physical_cores": 1,
                        "available_cores": 1.5
                    },
                    "memory": {
                        "total_mb": 4096,
                        "available_mb": 2048
                    }
                }
            }
        }
        
        # Create task batch with high requirements
        task_batch = [
            {
                "task_id": "high-cpu-task",
                "type": "benchmark",
                "priority": 1,
                "config": {
                    "model_type": "text_generation",
                    "model": "llama-65b",
                    "batch_size": 1
                }
            }
        ]
        
        # Configure performance predictor to return high requirements
        self.performance_predictor.predict_resource_requirements.return_value = {
            "cpu_cores": 8,
            "memory_mb": 32768,
            "gpu_memory_mb": 16384,
            "confidence": 0.9
        }
        
        # Test resource allocation
        allocations = self.optimizer.allocate_resources(task_batch, ["worker-1"])
        
        # Verify allocation failed due to insufficient resources
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].task_id, "high-cpu-task")
        self.assertFalse(allocations[0].allocated)
        self.assertEqual(allocations[0].worker_id, "")
        self.assertIn("No suitable worker found", allocations[0].reason)
    
    def test_recommend_worker_types(self):
        """Test worker type recommendation based on pending tasks."""
        # Create pending tasks
        pending_tasks = [
            {
                "task_id": "cpu-task-1",
                "type": "benchmark",
                "config": {
                    "model_type": "text_embedding",
                    "model": "bert-base-uncased",
                    "batch_size": 8
                }
            },
            {
                "task_id": "cpu-task-2",
                "type": "benchmark",
                "config": {
                    "model_type": "text_embedding",
                    "model": "bert-base-uncased",
                    "batch_size": 16
                }
            },
            {
                "task_id": "gpu-task-1",
                "type": "benchmark",
                "config": {
                    "model_type": "text_generation",
                    "model": "llama-7b",
                    "batch_size": 1
                }
            },
            {
                "task_id": "gpu-task-2",
                "type": "benchmark",
                "config": {
                    "model_type": "vision",
                    "model": "vit-large-patch16-224",
                    "batch_size": 4
                }
            }
        ]
        
        # Configure performance predictor to return different requirements for each task type
        def mock_predict_resources(task_data):
            if task_data.get("model_type") == "text_embedding":
                return {
                    "cpu_cores": 2,
                    "memory_mb": 4096,
                    "gpu_memory_mb": 0,
                    "confidence": 0.9
                }
            elif task_data.get("model_type") == "text_generation":
                return {
                    "cpu_cores": 4,
                    "memory_mb": 8192,
                    "gpu_memory_mb": 4096,
                    "confidence": 0.9
                }
            else:  # vision
                return {
                    "cpu_cores": 2,
                    "memory_mb": 4096,
                    "gpu_memory_mb": 2048,
                    "confidence": 0.9
                }
        
        self.performance_predictor.predict_resource_requirements.side_effect = mock_predict_resources
        
        # Configure cloud manager to return preferred providers
        self.cloud_manager.get_preferred_provider.return_value = "aws"
        
        # Test worker type recommendations
        recommendations = self.optimizer.recommend_worker_types(pending_tasks)
        
        # Verify recommendations
        self.assertEqual(len(recommendations), 2)  # One for CPU tasks, one for GPU tasks
        
        # Find CPU and GPU recommendations
        cpu_rec = next((r for r in recommendations if r.recommended_type == "cpu"), None)
        gpu_rec = next((r for r in recommendations if r.recommended_type == "gpu"), None)
        
        # Verify CPU recommendation
        self.assertIsNotNone(cpu_rec)
        self.assertEqual(cpu_rec.estimated_task_count, 2)  # 2 text embedding tasks
        self.assertIn("cpu_cores", cpu_rec.required_resources)
        self.assertIn("memory_mb", cpu_rec.required_resources)
        self.assertEqual(cpu_rec.provider, "aws")
        
        # Verify GPU recommendation
        self.assertIsNotNone(gpu_rec)
        self.assertEqual(gpu_rec.estimated_task_count, 2)  # 1 text generation + 1 vision task
        self.assertIn("cpu_cores", gpu_rec.required_resources)
        self.assertIn("memory_mb", gpu_rec.required_resources)
        self.assertIn("gpu_memory_mb", gpu_rec.required_resources)
        self.assertEqual(gpu_rec.provider, "aws")
        
        # Verify that GPU recommendation has higher priority
        self.assertGreater(gpu_rec.priority, cpu_rec.priority)
    
    def test_get_scaling_recommendations(self):
        """Test generation of scaling recommendations."""
        # Configure resource manager to return a scaling decision
        base_decision = ScalingDecision(
            action="scale_up",
            reason="High utilization across workers",
            utilization=0.85,
            count=2,
            worker_type="default"
        )
        self.resource_manager.evaluate_scaling.return_value = base_decision
        
        # Test scaling recommendations
        scaling = self.optimizer.get_scaling_recommendations()
        
        # Verify scaling decision was enhanced
        self.assertEqual(scaling.action, "scale_up")
        self.assertIn("High utilization across workers", scaling.reason)
        self.assertEqual(scaling.count, 2)
        
        # Verify resource manager was called
        self.resource_manager.evaluate_scaling.assert_called_once()
    
    def test_record_task_result(self):
        """Test recording of task execution results."""
        # Create task result
        task_result = {
            "task_data": {
                "model_type": "text_embedding",
                "model_name": "bert-base-uncased",
                "batch_size": 16
            },
            "metrics": {
                "cpu_cores_used": 3.6,
                "memory_mb_used": 5120,
                "gpu_memory_mb_used": 0,
                "execution_time_ms": 850
            },
            "success": True
        }
        
        # Test recording task result
        success = self.optimizer.record_task_result("task-1", "worker-1", task_result)
        
        # Verify success
        self.assertTrue(success)
        
        # Verify performance predictor was called
        self.performance_predictor.record_task_execution.assert_called_once()
        call_args = self.performance_predictor.record_task_execution.call_args[0]
        self.assertEqual(call_args[0], "task-1")  # task_id
        
        # Verify resource manager was called
        self.resource_manager.record_task_execution.assert_called_once()
        call_args = self.resource_manager.record_task_execution.call_args[0]
        self.assertEqual(call_args[0], "task-1")  # task_id
        
        # Verify workload history was updated
        self.assertEqual(len(self.optimizer.workload_history), 1)
        self.assertEqual(self.optimizer.workload_history[0]["task_id"], "task-1")
        self.assertEqual(self.optimizer.workload_history[0]["worker_id"], "worker-1")


class TestResourceOptimizerIntegration(unittest.TestCase):
    """Integration tests for ResourceOptimizer with actual components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create actual components
        self.resource_manager = DynamicResourceManager()
        self.performance_predictor = ResourcePerformancePredictor(
            db_path=os.path.join(self.temp_dir.name, "predictor.db")
        )
        
        # Initialize optimizer with actual components
        self.optimizer = ResourceOptimizer(
            resource_manager=self.resource_manager,
            performance_predictor=self.performance_predictor
        )
        
        # Register workers with resources
        self.resource_manager.register_worker(
            "cpu-worker", 
            {
                "cpu": {"cores": 8, "available_cores": 7.5},
                "memory": {"total_mb": 16384, "available_mb": 12288}
            }
        )
        
        self.resource_manager.register_worker(
            "gpu-worker", 
            {
                "cpu": {"cores": 16, "available_cores": 14.0},
                "memory": {"total_mb": 32768, "available_mb": 24576},
                "gpu": {"devices": 2, "available_devices": 2, "total_memory_mb": 16384, "available_memory_mb": 14336}
            }
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up components
        self.optimizer.cleanup()
        self.performance_predictor.cleanup()
        self.resource_manager.cleanup()
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_end_to_end_task_lifecycle(self):
        """Test end-to-end task lifecycle with resource optimization."""
        # Create task batch
        task_batch = [
            {
                "task_id": "cpu-task",
                "type": "benchmark",
                "priority": 1,
                "config": {
                    "model_type": "text_embedding",
                    "model": "bert-base-uncased",
                    "batch_size": 16
                }
            },
            {
                "task_id": "gpu-task",
                "type": "benchmark",
                "priority": 2,
                "config": {
                    "model_type": "text_generation",
                    "model": "llama-7b",
                    "batch_size": 1
                },
                "requirements": {
                    "cpu_cores": 4,
                    "memory_mb": 8192,
                    "gpu_memory_mb": 4096
                }
            }
        ]
        
        # Step 1: Allocate resources
        allocations = self.optimizer.allocate_resources(task_batch, ["cpu-worker", "gpu-worker"])
        
        # Verify allocations
        self.assertEqual(len(allocations), 2)
        
        # Find CPU and GPU task allocations
        cpu_allocation = next((a for a in allocations if a.task_id == "cpu-task"), None)
        gpu_allocation = next((a for a in allocations if a.task_id == "gpu-task"), None)
        
        # Verify CPU task was allocated to CPU worker
        self.assertIsNotNone(cpu_allocation)
        self.assertTrue(cpu_allocation.allocated)
        self.assertEqual(cpu_allocation.worker_id, "cpu-worker")
        
        # Verify GPU task was allocated to GPU worker
        self.assertIsNotNone(gpu_allocation)
        self.assertTrue(gpu_allocation.allocated)
        self.assertEqual(gpu_allocation.worker_id, "gpu-worker")
        
        # Step 2: Record task results
        cpu_result = {
            "task_data": {
                "model_type": "text_embedding",
                "model_name": "bert-base-uncased",
                "batch_size": 16
            },
            "metrics": {
                "cpu_cores_used": 3.6,
                "memory_mb_used": 5120,
                "gpu_memory_mb_used": 0,
                "execution_time_ms": 850
            },
            "success": True
        }
        
        gpu_result = {
            "task_data": {
                "model_type": "text_generation",
                "model_name": "llama-7b",
                "batch_size": 1
            },
            "metrics": {
                "cpu_cores_used": 3.8,
                "memory_mb_used": 7168,
                "gpu_memory_mb_used": 3584,
                "execution_time_ms": 2500
            },
            "success": True
        }
        
        # Record results
        self.optimizer.record_task_result("cpu-task", "cpu-worker", cpu_result)
        self.optimizer.record_task_result("gpu-task", "gpu-worker", gpu_result)
        
        # Step 3: Release resources
        self.resource_manager.release_reservation(cpu_allocation.reservation_id)
        self.resource_manager.release_reservation(gpu_allocation.reservation_id)
        
        # Step 4: Get worker type recommendations for similar future tasks
        future_tasks = [
            {
                "task_id": "future-cpu-task",
                "type": "benchmark",
                "config": {
                    "model_type": "text_embedding",
                    "model": "bert-base-uncased",
                    "batch_size": 32
                }
            },
            {
                "task_id": "future-gpu-task",
                "type": "benchmark",
                "config": {
                    "model_type": "text_generation",
                    "model": "llama-7b",
                    "batch_size": 2
                }
            }
        ]
        
        # Get recommendations
        recommendations = self.optimizer.recommend_worker_types(future_tasks)
        
        # Verify recommendations
        self.assertGreaterEqual(len(recommendations), 1)
        
        # Step 5: Get scaling recommendations
        scaling = self.optimizer.get_scaling_recommendations()
        
        # Verify scaling decision
        self.assertIn(scaling.action, ["scale_up", "scale_down", "maintain"])


class TestResourceOptimizerPerformance(unittest.TestCase):
    """Performance tests for ResourceOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create components
        self.resource_manager = DynamicResourceManager()
        self.performance_predictor = ResourcePerformancePredictor()
        
        # Initialize optimizer
        self.optimizer = ResourceOptimizer(
            resource_manager=self.resource_manager,
            performance_predictor=self.performance_predictor
        )
        
        # Register many workers for performance testing
        for i in range(20):
            worker_id = f"worker-{i}"
            if i < 15:  # 75% CPU workers
                resources = {
                    "cpu": {"cores": 8, "available_cores": 7.5},
                    "memory": {"total_mb": 16384, "available_mb": 12288}
                }
            else:  # 25% GPU workers
                resources = {
                    "cpu": {"cores": 16, "available_cores": 14.0},
                    "memory": {"total_mb": 32768, "available_mb": 24576},
                    "gpu": {"devices": 2, "available_devices": 2, "total_memory_mb": 16384, "available_memory_mb": 14336}
                }
            
            self.resource_manager.register_worker(worker_id, resources)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.optimizer.cleanup()
        self.performance_predictor.cleanup()
        self.resource_manager.cleanup()
    
    def test_allocation_performance(self):
        """Test performance of resource allocation for large task batches."""
        # Create large task batch (100 tasks)
        task_batch = []
        for i in range(100):
            if i % 2 == 0:  # Even tasks are CPU-bound
                task = {
                    "task_id": f"cpu-task-{i}",
                    "type": "benchmark",
                    "priority": i % 10,
                    "config": {
                        "model_type": "text_embedding",
                        "model": "bert-base-uncased",
                        "batch_size": 16
                    }
                }
            else:  # Odd tasks are GPU-bound
                task = {
                    "task_id": f"gpu-task-{i}",
                    "type": "benchmark",
                    "priority": i % 10,
                    "config": {
                        "model_type": "text_generation",
                        "model": "llama-7b",
                        "batch_size": 1
                    }
                }
            
            task_batch.append(task)
        
        # Measure allocation time
        start_time = time.time()
        allocations = self.optimizer.allocate_resources(task_batch, [f"worker-{i}" for i in range(20)])
        end_time = time.time()
        
        # Verify allocations
        self.assertEqual(len(allocations), 100)
        
        # Count successful allocations
        successful = sum(1 for a in allocations if a.allocated)
        
        # Log performance
        duration = end_time - start_time
        logger.info(f"Allocated {successful}/{len(allocations)} tasks in {duration:.4f} seconds")
        
        # Verify performance is reasonable (under 1 second for 100 tasks)
        self.assertLess(duration, 1.0)
    
    def test_scaling_recommendation_performance(self):
        """Test performance of scaling recommendations."""
        # Configure resource manager with utilization history
        self.resource_manager.utilization_history = [
            {"timestamp": datetime.now() - timedelta(minutes=i), "cpu": 0.85, "memory": 0.88, "gpu": 0.87}
            for i in range(5, 0, -1)
        ]
        
        # Measure scaling recommendation time
        start_time = time.time()
        for _ in range(10):  # Get recommendations 10 times
            scaling = self.optimizer.get_scaling_recommendations()
        end_time = time.time()
        
        # Log performance
        duration = (end_time - start_time) / 10
        logger.info(f"Generated scaling recommendation in {duration:.4f} seconds")
        
        # Verify performance is reasonable (under 10ms per recommendation)
        self.assertLess(duration, 0.01)


if __name__ == "__main__":
    unittest.main()