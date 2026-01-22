#!/usr/bin/env python3
"""
Test Script for Resource Pool Integration Plugin

This script demonstrates the integration between the WebGPU/WebNN Resource Pool
and the Distributed Testing Framework via the plugin architecture.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime

# Add necessary parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import plugin architecture
from distributed_testing.plugin_architecture import Plugin, PluginType, HookType, PluginManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockCoordinator:
    """
    Mock coordinator for testing purposes.
    
    This class simulates the coordinator's functionality for plugin testing.
    """
    
    def __init__(self):
        """Initialize the mock coordinator."""
        self.tasks = {}
        self.workers = set()
        self.plugin_manager = None
        
        logger.info("MockCoordinator initialized")
    
    async def initialize(self, plugin_dirs=None):
        """
        Initialize the coordinator.
        
        Args:
            plugin_dirs: List of plugin directories
        """
        # Initialize plugin manager
        self.plugin_manager = PluginManager(self, plugin_dirs or ["plugins"])
        
        # Discover plugins
        discovered_plugins = await self.plugin_manager.discover_plugins()
        logger.info(f"Discovered {len(discovered_plugins)} plugins: {discovered_plugins}")
        
        # Load discovered plugins
        for plugin_name in discovered_plugins:
            logger.info(f"Loading plugin: {plugin_name}")
            plugin_id = await self.plugin_manager.load_plugin(plugin_name)
            
            if plugin_id:
                logger.info(f"Loaded plugin: {plugin_id}")
            else:
                logger.error(f"Failed to load plugin: {plugin_name}")
        
        # Invoke coordinator startup hook
        await self.plugin_manager.invoke_hook(HookType.COORDINATOR_STARTUP, self)
        
        logger.info("MockCoordinator initialization complete")
    
    async def shutdown(self):
        """Shutdown the coordinator."""
        # Invoke coordinator shutdown hook
        if self.plugin_manager:
            await self.plugin_manager.invoke_hook(HookType.COORDINATOR_SHUTDOWN, self)
            await self.plugin_manager.shutdown()
        
        logger.info("MockCoordinator shutdown complete")
    
    async def create_task(self, task_id, task_data):
        """
        Create a task.
        
        Args:
            task_id: Task ID
            task_data: Task data
        """
        # Store task
        self.tasks[task_id] = {
            "id": task_id,
            "data": task_data,
            "status": "created",
            "created_at": datetime.now().isoformat()
        }
        
        # Invoke task created hook
        if self.plugin_manager:
            await self.plugin_manager.invoke_hook(
                HookType.TASK_CREATED, task_id, task_data
            )
        
        logger.info(f"Created task {task_id}")
    
    async def complete_task(self, task_id, result):
        """
        Complete a task.
        
        Args:
            task_id: Task ID
            result: Task result
        """
        # Update task
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["result"] = result
            
            # Invoke task completed hook
            if self.plugin_manager:
                await self.plugin_manager.invoke_hook(
                    HookType.TASK_COMPLETED, task_id, result
                )
            
            logger.info(f"Completed task {task_id}")
    
    async def fail_task(self, task_id, error):
        """
        Fail a task.
        
        Args:
            task_id: Task ID
            error: Error message
        """
        # Update task
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["failed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["error"] = error
            
            # Invoke task failed hook
            if self.plugin_manager:
                await self.plugin_manager.invoke_hook(
                    HookType.TASK_FAILED, task_id, error
                )
            
            logger.info(f"Failed task {task_id}")
    
    async def register_worker(self, worker_id, worker_info):
        """
        Register a worker.
        
        Args:
            worker_id: Worker ID
            worker_info: Worker information
        """
        # Store worker
        self.workers.add(worker_id)
        
        # Invoke worker registered hook
        if self.plugin_manager:
            await self.plugin_manager.invoke_hook(
                HookType.WORKER_REGISTERED, worker_id, worker_info
            )
        
        logger.info(f"Registered worker {worker_id}")
    
    async def notify_worker_disconnected(self, worker_id):
        """
        Notify worker disconnected.
        
        Args:
            worker_id: Worker ID
        """
        # Remove worker
        if worker_id in self.workers:
            self.workers.remove(worker_id)
            
            # Invoke worker disconnected hook
            if self.plugin_manager:
                await self.plugin_manager.invoke_hook(
                    HookType.WORKER_DISCONNECTED, worker_id
                )
            
            logger.info(f"Worker {worker_id} disconnected")
    
    async def start_recovery(self, component_id, error):
        """
        Start recovery process.
        
        Args:
            component_id: Component ID
            error: Error message
        """
        # Invoke recovery started hook
        if self.plugin_manager:
            await self.plugin_manager.invoke_hook(
                HookType.RECOVERY_STARTED, component_id, error
            )
        
        logger.info(f"Started recovery for component {component_id}")
    
    async def complete_recovery(self, component_id, result):
        """
        Complete recovery process.
        
        Args:
            component_id: Component ID
            result: Recovery result
        """
        # Invoke recovery completed hook
        if self.plugin_manager:
            await self.plugin_manager.invoke_hook(
                HookType.RECOVERY_COMPLETED, component_id, result
            )
        
        logger.info(f"Completed recovery for component {component_id}")
    
    def update_task_data(self, task_id, additional_data):
        """
        Update task data.
        
        Args:
            task_id: Task ID
            additional_data: Additional data to add to task
        """
        if task_id in self.tasks:
            self.tasks[task_id]["data"].update(additional_data)
            logger.info(f"Updated data for task {task_id}")
    
    def get_plugin_status(self, plugin_type=None):
        """
        Get plugin status.
        
        Args:
            plugin_type: Filter by plugin type
            
        Returns:
            Dictionary of plugin status information
        """
        if not self.plugin_manager:
            return {}
        
        if plugin_type:
            plugins = self.plugin_manager.get_plugins_by_type(plugin_type)
        else:
            plugins = self.plugin_manager.get_all_plugins()
        
        status = {}
        
        for plugin_id, plugin in plugins.items():
            status[plugin_id] = plugin.get_info()
            
            # Add plugin-specific status if available
            if hasattr(plugin, "get_resource_pool_status") and callable(getattr(plugin, "get_resource_pool_status")):
                status[plugin_id]["resource_pool_status"] = plugin.get_resource_pool_status()
        
        return status

async def run_test_scenario(coordinator, resource_pool_test=False, simulate_tasks=0, 
                           simulate_recovery=False, test_duration=60):
    """
    Run a test scenario.
    
    Args:
        coordinator: Mock coordinator instance
        resource_pool_test: Whether to test resource pool features
        simulate_tasks: Number of tasks to simulate
        simulate_recovery: Whether to simulate recovery
        test_duration: Test duration in seconds
    """
    logger.info(f"Running test scenario with {simulate_tasks} tasks for {test_duration} seconds")
    
    # Generate task IDs
    task_ids = [f"task-{i+1}" for i in range(simulate_tasks)]
    
    # Create tasks
    for task_id in task_ids:
        # Create task with or without resource pool
        if resource_pool_test:
            # Create task with resource pool requirements
            task_data = {
                "name": f"Test task {task_id}",
                "resource_pool": True,
                "model_type": "text_embedding",
                "model_name": "bert-base-uncased",
                "hardware_preferences": {
                    "priority_list": ["webgpu", "cpu"]
                },
                "fault_tolerance": {
                    "recovery_timeout": 30,
                    "state_persistence": True,
                    "failover_strategy": "immediate"
                }
            }
        else:
            # Create task without resource pool
            task_data = {
                "name": f"Test task {task_id}",
                "description": "Test task without resource pool"
            }
        
        await coordinator.create_task(task_id, task_data)
    
    # Wait for task creation to be processed
    await asyncio.sleep(2)
    
    # Simulate recovery if requested
    if simulate_recovery:
        logger.info("Simulating recovery scenario")
        
        # Simulate browser failure
        await coordinator.start_recovery("browser-1", "Connection lost")
        
        # Wait for recovery to start
        await asyncio.sleep(2)
        
        # Simulate recovery completion
        await coordinator.complete_recovery("browser-1", {"status": "recovered"})
    
    # Wait for specified test duration
    logger.info(f"Running test for {test_duration} seconds")
    await asyncio.sleep(test_duration)
    
    # Complete tasks
    for task_id in task_ids:
        # Randomly complete or fail tasks
        if task_id.endswith("3") or task_id.endswith("7"):
            # Fail task
            await coordinator.fail_task(task_id, "Simulated failure")
        else:
            # Complete task
            await coordinator.complete_task(task_id, {"status": "success"})
    
    # Wait for task completion to be processed
    await asyncio.sleep(2)
    
    # Get and display plugin status
    status = coordinator.get_plugin_status(PluginType.INTEGRATION)
    logger.info(f"Plugin status: {json.dumps(status, indent=2)}")

async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Resource Pool Integration Plugin")
    
    parser.add_argument("--plugin-dirs", type=str, default="plugins",
                        help="Comma-separated list of plugin directories")
    parser.add_argument("--simulate-tasks", type=int, default=5,
                        help="Number of tasks to simulate")
    parser.add_argument("--resource-pool-test", action="store_true",
                        help="Test resource pool features")
    parser.add_argument("--simulate-recovery", action="store_true",
                        help="Simulate recovery scenario")
    parser.add_argument("--test-duration", type=int, default=60,
                        help="Test duration in seconds")
    
    args = parser.parse_args()
    
    # Create and initialize coordinator
    coordinator = MockCoordinator()
    
    plugin_dirs = args.plugin_dirs.split(",")
    await coordinator.initialize(plugin_dirs)
    
    try:
        # Run test scenario
        await run_test_scenario(
            coordinator,
            resource_pool_test=args.resource_pool_test,
            simulate_tasks=args.simulate_tasks,
            simulate_recovery=args.simulate_recovery,
            test_duration=args.test_duration
        )
    finally:
        # Shutdown coordinator
        await coordinator.shutdown()

if __name__ == "__main__":
    # Run main function
    asyncio.run(main())