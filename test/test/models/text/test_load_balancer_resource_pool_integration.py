#!/usr/bin/env python3
"""
Integration test for Adaptive Load Balancer with Resource Pool Bridge

This script tests the integration between the Adaptive Load Balancer component
from the Distributed Testing Framework and the WebGPU/WebNN Resource Pool Bridge.

Usage:
    python test_load_balancer_resource_pool_integration.py
"""

import asyncio
import json
import logging
import os
import sys
import time
import unittest
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components
from distributed_testing.resource_pool_bridge import ResourcePoolBridgeIntegration
from distributed_testing.model_sharding import ShardedModelExecution
from duckdb_api.distributed_testing.load_balancer import LoadBalancerService, WorkerCapabilities, TestRequirements

class TestLoadBalancerResourcePoolIntegration(unittest.TestCase):
    """
    Integration tests for the Adaptive Load Balancer with Resource Pool Bridge.
    
    This test suite verifies that the Adaptive Load Balancer can effectively
    distribute browser resources and workloads across worker nodes.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Initialize load balancer
        self.load_balancer = LoadBalancerService()
        
        # Mock connection pool for resource pool bridge
        self.connection_pool = {
            'chrome-1': {
                'id': 'chrome-1',
                'type': 'chrome',
                'status': 'ready',
                'capabilities': {'webgpu': True, 'webnn': False},
                'active_models': set()
            },
            'firefox-1': {
                'id': 'firefox-1',
                'type': 'firefox',
                'status': 'ready',
                'capabilities': {'webgpu': True, 'webnn': False},
                'active_models': set()
            },
            'edge-1': {
                'id': 'edge-1',
                'type': 'edge',
                'status': 'ready',
                'capabilities': {'webgpu': True, 'webnn': True},
                'active_models': set()
            }
        }
        
        # Create mock for ResourcePoolBridgeIntegration
        self.mock_integration = MagicMock(spec=ResourcePoolBridgeIntegration)
        self.mock_integration.connection_pool = self.connection_pool
        self.mock_integration.initialize = MagicMock(return_value=None)
        self.mock_integration.get_model = MagicMock(return_value=MagicMock())
        
        # Patch the ResourcePoolBridgeIntegration class
        self.patcher = patch('distributed_testing.resource_pool_bridge.ResourcePoolBridgeIntegration', 
                             return_value=self.mock_integration)
        self.mock_resource_pool_bridge_class = self.patcher.start()
        
        # Initialize workers
        self.workers = {}
        for i in range(3):
            worker_id = f"worker-{i}"
            self.workers[worker_id] = {
                'id': worker_id,
                'capabilities': WorkerCapabilities(
                    worker_id=worker_id,
                    hardware_specs={
                        'cpu': {'cores': 8},
                        'memory': {'total_gb': 16},
                        'gpu': {'name': 'Test GPU', 'vram_gb': 8}
                    },
                    available_memory=14.0 - i * 2,  # Different memory availability
                    cpu_cores=8,
                    has_gpu=True,
                    has_webgpu=True if i < 2 else False,  # worker-2 doesn't have WebGPU
                    has_webnn=True if i == 0 else False,  # Only worker-0 has WebNN
                    browser_support=['chrome', 'firefox', 'edge']
                ),
                'bridge_integration': None,
                'active_tests': set()
            }
            
            # Register worker with load balancer
            self.load_balancer.register_worker(worker_id, self.workers[worker_id]['capabilities'])
    
    async def create_resource_pool_for_worker(self, worker_id):
        """Create and initialize resource pool bridge for a worker."""
        # Create integration instance for worker
        integration = ResourcePoolBridgeIntegration(
            max_connections=3,
            browser_preferences={
                'audio': 'firefox',
                'vision': 'chrome',
                'text_embedding': 'edge'
            },
            adaptive_scaling=True,
            enable_fault_tolerance=True
        )
        
        # Store in worker data
        self.workers[worker_id]['bridge_integration'] = integration
        
        # Mock initialization
        await integration.initialize()
        
        return integration
    
    def test_worker_registration(self):
        """Test worker registration with load balancer."""
        self.assertEqual(len(self.load_balancer.workers), 3)
        
        for worker_id in self.workers:
            self.assertIn(worker_id, self.load_balancer.workers)
    
    async def test_test_assignment(self):
        """Test test assignment to optimal workers."""
        # Create resource pools for all workers
        for worker_id in self.workers:
            await self.create_resource_pool_for_worker(worker_id)
        
        # Create test requirements for different model types
        tests = []
        
        # Vision model test (Chrome optimal)
        tests.append(TestRequirements(
            test_id="test-vision-1",
            model_id="vit-base-patch16-224",
            model_type="vision",
            minimum_memory=2.0,
            priority=2,
            browser_requirements={"preferred": "chrome"}
        ))
        
        # Audio model test (Firefox optimal)
        tests.append(TestRequirements(
            test_id="test-audio-1",
            model_id="whisper-tiny",
            model_type="audio",
            minimum_memory=3.0,
            priority=3,
            browser_requirements={"preferred": "firefox"}
        ))
        
        # Text embedding model test (Edge optimal due to WebNN)
        tests.append(TestRequirements(
            test_id="test-text-1",
            model_id="bert-base-uncased",
            model_type="text_embedding",
            minimum_memory=1.5,
            priority=1,
            browser_requirements={"preferred": "edge"}
        ))
        
        # Large model that requires sharding
        tests.append(TestRequirements(
            test_id="test-large-1",
            model_id="llama-13b",
            model_type="large_language_model",
            minimum_memory=10.0,  # Too large for a single worker
            priority=3,
            requires_sharding=True,
            browser_requirements={"preferred": "chrome"}
        ))
        
        # Submit tests to load balancer
        for test in tests:
            self.load_balancer.submit_test(test)
        
        # Get assignments
        assignments = await self.load_balancer.get_assignments()
        
        # Verify that all tests are assigned
        self.assertEqual(len(assignments), len(tests) - 1)  # -1 because the large model needs manual handling
        
        # Verify that vision test is assigned to a worker with Chrome support
        vision_assignment = next((a for a in assignments if a[1].test_id == "test-vision-1"), None)
        self.assertIsNotNone(vision_assignment)
        worker_id, test_req = vision_assignment
        self.assertIn('chrome', self.workers[worker_id]['capabilities'].browser_support)
        
        # Verify that audio test is assigned to a worker with Firefox support
        audio_assignment = next((a for a in assignments if a[1].test_id == "test-audio-1"), None)
        self.assertIsNotNone(audio_assignment)
        worker_id, test_req = audio_assignment
        self.assertIn('firefox', self.workers[worker_id]['capabilities'].browser_support)
        
        # Verify that text embedding test is assigned to worker with WebNN
        text_assignment = next((a for a in assignments if a[1].test_id == "test-text-1"), None)
        self.assertIsNotNone(text_assignment)
        worker_id, test_req = text_assignment
        self.assertTrue(self.workers[worker_id]['capabilities'].has_webnn)
    
    async def test_execute_models_across_workers(self):
        """Test executing models across different workers with load balancing."""
        # Create resource pools for all workers
        for worker_id in self.workers:
            await self.create_resource_pool_for_worker(worker_id)
        
        # Submit tests with different resource requirements
        tests = []
        
        # Add several tests with different requirements
        for i in range(5):
            test_id = f"test-{i}"
            model_type = "vision" if i % 2 == 0 else "audio" if i % 3 == 0 else "text_embedding"
            model_id = f"model-{model_type}-{i}"
            memory_req = 1.0 + (i * 0.5)  # Increasing memory requirements
            
            test = TestRequirements(
                test_id=test_id,
                model_id=model_id,
                model_type=model_type,
                minimum_memory=memory_req,
                priority=i % 3 + 1,  # Priority 1-3
                browser_requirements={"preferred": "chrome" if i % 3 == 0 else "firefox" if i % 3 == 1 else "edge"}
            )
            
            tests.append(test)
            self.load_balancer.submit_test(test)
        
        # Get first batch of assignments
        assignments = await self.load_balancer.get_assignments()
        
        # Verify assignments are balanced
        worker_assignments = {}
        for worker_id, test_req in assignments:
            if worker_id not in worker_assignments:
                worker_assignments[worker_id] = []
            
            worker_assignments[worker_id].append(test_req)
        
        # Verify each worker has assignments
        self.assertTrue(all(worker_id in worker_assignments for worker_id in self.workers))
        
        # Mark some tests as started to simulate execution
        for worker_id, test_reqs in worker_assignments.items():
            for test_req in test_reqs:
                self.load_balancer.mark_test_started(test_req.test_id, worker_id)
                self.workers[worker_id]['active_tests'].add(test_req.test_id)
        
        # Simulate resource usage change
        # Worker 0 is now heavily loaded
        high_load_worker = "worker-0"
        self.load_balancer.update_worker_utilization(
            high_load_worker,
            cpu_utilization=85.0,
            memory_utilization=90.0,
            gpu_utilization=95.0
        )
        
        # Add more tests and get assignments
        for i in range(5, 10):
            test_id = f"test-{i}"
            model_type = "vision" if i % 2 == 0 else "audio" if i % 3 == 0 else "text_embedding"
            model_id = f"model-{model_type}-{i}"
            memory_req = 1.0 + (i * 0.5) % 3  # Memory requirements that cycle
            
            test = TestRequirements(
                test_id=test_id,
                model_id=model_id,
                model_type=model_type,
                minimum_memory=memory_req,
                priority=i % 3 + 1,
                browser_requirements={"preferred": "chrome" if i % 3 == 0 else "firefox" if i % 3 == 1 else "edge"}
            )
            
            tests.append(test)
            self.load_balancer.submit_test(test)
        
        # Get next batch of assignments
        new_assignments = await self.load_balancer.get_assignments()
        
        # Verify high-load worker gets fewer assignments
        new_worker_assignments = {}
        for worker_id, test_req in new_assignments:
            if worker_id not in new_worker_assignments:
                new_worker_assignments[worker_id] = []
            
            new_worker_assignments[worker_id].append(test_req)
        
        # If high-load worker got assignments, it should be fewer than others
        if high_load_worker in new_worker_assignments:
            for other_worker, assignments in new_worker_assignments.items():
                if other_worker != high_load_worker:
                    self.assertLessEqual(
                        len(new_worker_assignments.get(high_load_worker, [])),
                        len(assignments),
                        f"High load worker {high_load_worker} got more assignments than {other_worker}"
                    )
        
        # Run detection of load imbalance
        imbalance = self.load_balancer.detect_load_imbalance()
        self.assertIsNotNone(imbalance)
        
        if imbalance:
            # Balance load if imbalance detected
            migrations = await self.load_balancer.balance_load()
            
            # Verify migrations include moving tests from high-load worker
            if migrations:
                sources = [source for source, _, _ in migrations]
                self.assertIn(high_load_worker, sources, 
                              f"High load worker {high_load_worker} should have tests migrated away")
    
    async def test_sharded_model_execution(self):
        """Test executing a sharded model across multiple workers."""
        # Create resource pools for all workers
        for worker_id in self.workers:
            await self.create_resource_pool_for_worker(worker_id)
        
        # Create sharded model test
        test = TestRequirements(
            test_id="test-sharded-1",
            model_id="llama-13b",
            model_type="large_language_model",
            minimum_memory=12.0,  # Too large for a single worker
            priority=3,
            requires_sharding=True,
            sharding_requirements={
                "strategy": "layer_balanced", 
                "num_shards": 3
            }
        )
        
        # Submit to load balancer
        self.load_balancer.submit_test(test)
        
        # Use the load balancer's own capability detection to find suitable workers
        # This would normally be handled by a coordinator component
        suitable_workers = []
        
        for worker_id, capabilities in self.load_balancer.workers.items():
            if capabilities.has_webgpu and capabilities.available_memory >= 4.0:  # Need at least 4GB per shard
                suitable_workers.append(worker_id)
        
        # Ensure we have enough workers
        self.assertGreaterEqual(len(suitable_workers), test.sharding_requirements["num_shards"],
                               "Not enough suitable workers for sharding")
        
        # In a real system, we would assign the test to all suitable workers
        # Here we'll just simulate creating a ShardedModelExecution across the workers
        sharded_executions = {}
        
        for i, worker_id in enumerate(suitable_workers[:test.sharding_requirements["num_shards"]]):
            # Mark the test as assigned to this worker
            self.load_balancer.mark_test_assigned(test.test_id, worker_id)
            self.workers[worker_id]['active_tests'].add(test.test_id)
            
            # Create a sharded model execution (mocked)
            sharded_execution = MagicMock(spec=ShardedModelExecution)
            sharded_execution.initialize = MagicMock(return_value=None)
            sharded_execution.run_inference = MagicMock(return_value={"output": f"Shard output from worker {worker_id}"})
            
            # Store in worker data
            self.workers[worker_id]['sharded_execution'] = sharded_execution
            sharded_executions[worker_id] = sharded_execution
        
        # Verify that test has been assigned across multiple workers
        test_workers = [worker_id for worker_id in self.workers if test.test_id in self.workers[worker_id]['active_tests']]
        self.assertEqual(len(test_workers), test.sharding_requirements["num_shards"])
        
        # In a real system, a coordinator would now collect results from all workers
        # and combine them. Here we'll just verify the assignments.
        for worker_id, execution in sharded_executions.items():
            result = execution.run_inference({"input": "Test input"})
            self.assertIn("output", result)
            self.assertIn(worker_id, result["output"])
        
        # Simulate load changes during execution
        # One worker becomes heavily loaded
        heavy_worker = test_workers[0]
        self.load_balancer.update_worker_utilization(
            heavy_worker,
            cpu_utilization=90.0,
            memory_utilization=95.0,
            gpu_utilization=95.0
        )
        
        # In a real system with workload-aware sharding, the adaptive load balancer
        # would detect the imbalance and trigger migrations
        imbalance = self.load_balancer.detect_load_imbalance()
        
        # If we detect an imbalance, verify that balanced load includes migration of shards
        if imbalance:
            migrations = await self.load_balancer.balance_load()
            self.assertGreaterEqual(len(migrations), 1, "Should migrate at least one task")
    
    def tearDown(self):
        """Clean up test environment."""
        self.patcher.stop()
        
        # Clean up resource pool bridges
        for worker_id, worker in self.workers.items():
            if worker['bridge_integration']:
                integration = worker['bridge_integration']
                if hasattr(integration, 'close'):
                    integration.close()


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests using unittest."""
    
    def run_async(self, test_method, *args, **kwargs):
        """Run an async test method."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(test_method(*args, **kwargs))


class TestLoadBalancerResourcePoolIntegrationAsync(AsyncTestCase, TestLoadBalancerResourcePoolIntegration):
    """Async version of tests for LoadBalancerResourcePoolIntegration."""
    
    def test_test_assignment(self):
        """Test assignment of tests to workers."""
        self.run_async(super().test_test_assignment)
    
    def test_execute_models_across_workers(self):
        """Test executing models across workers."""
        self.run_async(super().test_execute_models_across_workers)
    
    def test_sharded_model_execution(self):
        """Test sharded model execution."""
        self.run_async(super().test_sharded_model_execution)


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests()