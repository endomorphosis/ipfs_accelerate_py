#!/usr/bin/env python3
"""
CI/CD Integration Example for Distributed Testing Framework

This example demonstrates how to use the CI/CD Integration plugin to report
test results to CI/CD systems like GitHub Actions, GitLab CI, Jenkins, and Azure DevOps.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import plugin and plugin architecture
from distributed_testing.plugin_architecture import Plugin, PluginType, HookType
from distributed_testing.integration.ci_cd_integration_plugin import CICDIntegrationPlugin

# Create a simple mock coordinator for the example
class MockCoordinator:
    """Simple mock coordinator for testing the CI/CD Integration plugin."""
    
    def __init__(self):
        """Initialize the mock coordinator."""
        self.tasks = {}
        self.workers = {}
    
    async def create_task(self, task_id: str, task_data: Dict[str, Any]):
        """Create a task in the mock coordinator."""
        self.tasks[task_id] = {
            "id": task_id,
            "data": task_data,
            "status": "created",
            "created_at": datetime.now().isoformat()
        }
        
        # Invoke hooks for task creation
        await self.plugin_manager.invoke_hook(HookType.TASK_CREATED, task_id, task_data)
        
        return task_id
    
    async def complete_task(self, task_id: str, result: Any):
        """Complete a task in the mock coordinator."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["result"] = result
            
            # Invoke hooks for task completion
            await self.plugin_manager.invoke_hook(HookType.TASK_COMPLETED, task_id, result)
            
            return True
        return False
    
    async def fail_task(self, task_id: str, error: str):
        """Fail a task in the mock coordinator."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["failed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["error"] = error
            
            # Invoke hooks for task failure
            await self.plugin_manager.invoke_hook(HookType.TASK_FAILED, task_id, error)
            
            return True
        return False
    
    async def register_worker(self, worker_id: str, capabilities: Dict[str, Any]):
        """Register a worker in the mock coordinator."""
        self.workers[worker_id] = {
            "id": worker_id,
            "capabilities": capabilities,
            "status": "registered",
            "registered_at": datetime.now().isoformat()
        }
        
        # Invoke hooks for worker registration
        await self.plugin_manager.invoke_hook(HookType.WORKER_REGISTERED, worker_id, capabilities)
        
        return worker_id
    
    async def disconnect_worker(self, worker_id: str):
        """Disconnect a worker from the mock coordinator."""
        if worker_id in self.workers:
            self.workers[worker_id]["status"] = "disconnected"
            self.workers[worker_id]["disconnected_at"] = datetime.now().isoformat()
            
            # Invoke hooks for worker disconnection
            await self.plugin_manager.invoke_hook(HookType.WORKER_DISCONNECTED, worker_id)
            
            return True
        return False
    
    async def start(self):
        """Start the mock coordinator."""
        # Initialize plugin manager
        self.plugin_manager = MockPluginManager(self)
        
        # Invoke startup hook
        await self.plugin_manager.invoke_hook(HookType.COORDINATOR_STARTUP, self)
        
        logger.info("Mock coordinator started")
        
        return True
    
    async def shutdown(self):
        """Shutdown the mock coordinator."""
        # Invoke shutdown hook
        await self.plugin_manager.invoke_hook(HookType.COORDINATOR_SHUTDOWN, self)
        
        # Shutdown plugin manager
        await self.plugin_manager.shutdown()
        
        logger.info("Mock coordinator shutdown")
        
        return True

# Create a simple mock plugin manager for the example
class MockPluginManager:
    """Simple mock plugin manager for testing the CI/CD Integration plugin."""
    
    def __init__(self, coordinator):
        """Initialize the mock plugin manager."""
        self.coordinator = coordinator
        self.plugins = {}
        self.hooks = {}
        
        for hook_type in HookType:
            self.hooks[hook_type] = []
    
    async def load_plugin(self, plugin: Plugin):
        """Load a plugin in the mock plugin manager."""
        # Initialize plugin
        await plugin.initialize(self.coordinator)
        
        # Store plugin
        self.plugins[plugin.id] = plugin
        
        # Register hooks
        for hook_type, callbacks in plugin.hooks.items():
            for callback in callbacks:
                self.hooks[hook_type].append((plugin.id, callback))
        
        return plugin.id
    
    async def invoke_hook(self, hook_type: HookType, *args, **kwargs):
        """Invoke a hook in the mock plugin manager."""
        results = []
        
        for plugin_id, callback in self.hooks.get(hook_type, []):
            if plugin_id in self.plugins:
                plugin = self.plugins[plugin_id]
                
                if plugin.enabled:
                    try:
                        result = callback(*args, **kwargs)
                        
                        # Handle coroutines
                        if asyncio.iscoroutine(result):
                            result = await result
                            
                        results.append((plugin_id, result))
                    except Exception as e:
                        logger.error(f"Error invoking hook {hook_type.value} in plugin {plugin.name}: {str(e)}")
        
        return results
    
    async def shutdown(self):
        """Shutdown the mock plugin manager."""
        for plugin_id, plugin in list(self.plugins.items()):
            try:
                await plugin.shutdown()
                logger.info(f"Shutdown plugin {plugin.name}")
            except Exception as e:
                logger.error(f"Error shutting down plugin {plugin.name}: {str(e)}")
        
        # Clear registries
        self.plugins.clear()
        
        for hook_type in HookType:
            self.hooks[hook_type] = []
        
        logger.info("Mock plugin manager shutdown complete")

# Main example function
async def run_example():
    """Run the CI/CD Integration example."""
    logger.info("Starting CI/CD Integration example...")
    
    # Create mock coordinator
    coordinator = MockCoordinator()
    
    # Create CI/CD Integration plugin
    ci_plugin = CICDIntegrationPlugin()
    
    # Configure plugin for simulation mode
    ci_plugin.configure({
        "ci_system": "github",  # Simulate GitHub Actions
        "repository": "user/repo",
        "api_token": "mock_token",
        "update_interval": 5,  # More frequent updates for the example
        "detailed_logging": True,
        "artifact_dir": "ci_artifacts"
    })
    
    # Start coordinator
    await coordinator.start()
    
    # Load plugin
    await coordinator.plugin_manager.load_plugin(ci_plugin)
    
    # Register workers
    await coordinator.register_worker("worker-001", {
        "hardware_type": "gpu",
        "cpu_cores": 8,
        "memory_gb": 16,
        "gpu_memory_gb": 8,
        "supports_cuda": True,
        "supports_webgpu": True,
        "supports_webnn": True
    })
    
    await coordinator.register_worker("worker-002", {
        "hardware_type": "cpu",
        "cpu_cores": 16,
        "memory_gb": 32,
        "supports_cuda": False,
        "supports_webgpu": False,
        "supports_webnn": False
    })
    
    # Create and execute tasks
    task_ids = []
    
    # Create 10 tasks
    for i in range(1, 11):
        task_id = f"task-{i}"
        task_data = {
            "name": f"Test Task {i}",
            "type": "model_test",
            "model_name": f"model-{i}",
            "hardware_requirements": {
                "min_memory_gb": 4,
                "min_gpu_memory_gb": 4 if i % 2 == 0 else 0,
                "requires_cuda": i % 2 == 0,
                "requires_webgpu": i % 4 == 0,
                "requires_webnn": i % 4 == 0
            }
        }
        
        await coordinator.create_task(task_id, task_data)
        task_ids.append(task_id)
        
        logger.info(f"Created task {task_id}")
    
    # Process tasks with some successes and failures
    for i, task_id in enumerate(task_ids):
        # Wait briefly to simulate task execution
        await asyncio.sleep(0.5)
        
        # Complete or fail the task
        if i % 5 == 4:  # Fail every 5th task
            await coordinator.fail_task(task_id, f"Task {task_id} failed due to simulated error")
            logger.info(f"Failed task {task_id}")
        else:
            result = {
                "execution_time": i * 1.5,
                "memory_usage": i * 512,
                "accuracy": 0.9 - (i * 0.01),
                "status": "success"
            }
            
            await coordinator.complete_task(task_id, result)
            logger.info(f"Completed task {task_id}")
    
    # Disconnect workers
    await coordinator.disconnect_worker("worker-001")
    await coordinator.disconnect_worker("worker-002")
    
    # Wait for periodic updates to occur
    logger.info("Waiting for periodic updates...")
    await asyncio.sleep(10)
    
    # Shutdown coordinator
    await coordinator.shutdown()
    
    logger.info("CI/CD Integration example completed")

# Run the example
if __name__ == "__main__":
    asyncio.run(run_example())