#!/usr/bin/env python3
"""
Test Script for Plugin Architecture

This script demonstrates the functionality of the plugin architecture by running
the coordinator with various plugins and simulating task lifecycle events.

Usage:
    python run_test_plugins.py [--plugins PLUGIN_LIST] [--enable-external-reporting]
                              [--ci-system SYSTEM_NAME] [--simulate-tasks NUM_TASKS]
"""

import argparse
import anyio
import json
import logging
import os
import signal
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from .coordinator import DistributedTestingCoordinator
from plugin_architecture import PluginManager, HookType, PluginType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("plugin_test.log")
    ]
)
logger = logging.getLogger(__name__)

async def simulate_tasks(coordinator, num_tasks=5):
    """
    Simulate task lifecycle events for testing plugins.
    
    Args:
        coordinator: Coordinator instance
        num_tasks: Number of tasks to simulate
    """
    logger.info(f"Simulating {num_tasks} tasks")
    
    # Generate tasks
    tasks = []
    
    for i in range(num_tasks):
        task_id = f"task-{i+1}"
        task_data = {
            "id": task_id,
            "type": "benchmark",
            "model": f"model-{i % 3 + 1}",
            "batch_size": 2 ** (i % 4 + 1),
            "created_at": datetime.now().isoformat()
        }
        
        tasks.append((task_id, task_data))
        
        # Create task
        if coordinator.plugin_manager:
            await coordinator.plugin_manager.invoke_hook(
                HookType.TASK_CREATED, task_id, task_data
            )
        
        # Store task in coordinator
        coordinator.tasks[task_id] = task_data
        coordinator.pending_tasks.add(task_id)
        
        logger.info(f"Created task {task_id}")
    
    # Wait a bit
    await anyio.sleep(1)
    
    # Assign tasks to workers
    workers = [f"worker-{i+1}" for i in range(3)]
    
    for i, (task_id, _) in enumerate(tasks):
        worker_id = workers[i % len(workers)]
        
        # Assign task
        if coordinator.plugin_manager:
            await coordinator.plugin_manager.invoke_hook(
                HookType.TASK_ASSIGNED, task_id, worker_id
            )
        
        # Update task in coordinator
        coordinator.tasks[task_id]["worker_id"] = worker_id
        coordinator.tasks[task_id]["assigned_at"] = datetime.now().isoformat()
        coordinator.running_tasks[task_id] = worker_id
        coordinator.pending_tasks.remove(task_id)
        
        logger.info(f"Assigned task {task_id} to worker {worker_id}")
    
    # Wait a bit
    await anyio.sleep(1)
    
    # Start tasks
    for i, (task_id, _) in enumerate(tasks):
        worker_id = coordinator.running_tasks[task_id]
        
        # Start task
        if coordinator.plugin_manager:
            await coordinator.plugin_manager.invoke_hook(
                HookType.TASK_STARTED, task_id, worker_id
            )
        
        # Update task in coordinator
        coordinator.tasks[task_id]["started_at"] = datetime.now().isoformat()
        coordinator.tasks[task_id]["status"] = "running"
        
        logger.info(f"Started task {task_id} on worker {worker_id}")
    
    # Wait for tasks to "run"
    await anyio.sleep(2)
    
    # Complete or fail tasks
    for i, (task_id, _) in enumerate(tasks):
        # Fail every 3rd task
        if i % 3 == 2:
            # Fail task
            if coordinator.plugin_manager:
                await coordinator.plugin_manager.invoke_hook(
                    HookType.TASK_FAILED, task_id, "Simulated failure"
                )
            
            # Update task in coordinator
            coordinator.tasks[task_id]["status"] = "failed"
            coordinator.tasks[task_id]["failed_at"] = datetime.now().isoformat()
            coordinator.tasks[task_id]["error"] = "Simulated failure"
            coordinator.failed_tasks.add(task_id)
            del coordinator.running_tasks[task_id]
            
            logger.info(f"Failed task {task_id}")
        else:
            # Complete task
            result = {
                "task_id": task_id,
                "duration": 1.5 + (i * 0.2),
                "metrics": {
                    "throughput": 100 + (i * 10),
                    "latency": 10 - (i * 0.5),
                    "memory": 200 + (i * 20)
                }
            }
            
            if coordinator.plugin_manager:
                await coordinator.plugin_manager.invoke_hook(
                    HookType.TASK_COMPLETED, task_id, result
                )
            
            # Update task in coordinator
            coordinator.tasks[task_id]["status"] = "completed"
            coordinator.tasks[task_id]["completed_at"] = datetime.now().isoformat()
            coordinator.tasks[task_id]["result"] = result
            coordinator.completed_tasks.add(task_id)
            del coordinator.running_tasks[task_id]
            
            logger.info(f"Completed task {task_id}")
    
    # Wait a bit
    await anyio.sleep(1)
    
    # Cancel a task
    if len(tasks) > 3:
        task_id = tasks[3][0]
        
        # Check if task is still running
        if task_id in coordinator.running_tasks:
            worker_id = coordinator.running_tasks[task_id]
            
            # Cancel task
            if coordinator.plugin_manager:
                await coordinator.plugin_manager.invoke_hook(
                    HookType.TASK_CANCELLED, task_id, "User requested cancellation"
                )
            
            # Update task in coordinator
            coordinator.tasks[task_id]["status"] = "cancelled"
            coordinator.tasks[task_id]["cancelled_at"] = datetime.now().isoformat()
            coordinator.tasks[task_id]["cancel_reason"] = "User requested cancellation"
            del coordinator.running_tasks[task_id]
            
            logger.info(f"Cancelled task {task_id}")

async def configure_plugins(coordinator, args):
    """
    Configure plugins based on command-line arguments.
    
    Args:
        coordinator: Coordinator instance
        args: Command-line arguments
    """
    if not coordinator.plugin_manager:
        logger.warning("Plugin manager not available, skipping plugin configuration")
        return
    
    # Get loaded plugins
    plugins = coordinator.plugin_manager.get_all_plugins()
    
    # Configure reporter plugin
    reporter_plugin = None
    for plugin_id, plugin in plugins.items():
        if plugin.name == "TaskReporter":
            reporter_plugin = plugin
            break
    
    if reporter_plugin and args.enable_external_reporting:
        logger.info("Configuring TaskReporter plugin for external reporting")
        
        # Configure for external reporting
        config = {
            "enable_external_reporting": True,
            "external_system_url": "https://example.com/api/reports",
            "report_interval": 10,
            "log_level": "INFO"
        }
        
        reporter_plugin.configure(config)
    
    # Configure CI integration plugin
    ci_plugin = None
    for plugin_id, plugin in plugins.items():
        if plugin.name == "CIIntegration":
            ci_plugin = plugin
            break
    
    if ci_plugin and args.ci_system:
        logger.info(f"Configuring CIIntegration plugin for {args.ci_system}")
        
        # Configure for CI system
        config = {
            "ci_system": args.ci_system,
            "api_url": f"https://{args.ci_system.lower()}.com/api",
            "repository": "example/ipfs_accelerate_py",
            "update_interval": 15,
            "update_on_completion_only": False
        }
        
        ci_plugin.configure(config)

async def display_plugin_info(coordinator):
    """
    Display information about loaded plugins.
    
    Args:
        coordinator: Coordinator instance
    """
    if not coordinator.plugin_manager:
        logger.warning("Plugin manager not available")
        return
    
    plugins = coordinator.plugin_manager.get_all_plugins()
    
    if not plugins:
        logger.info("No plugins loaded")
        return
    
    logger.info(f"Loaded plugins ({len(plugins)}):")
    
    for plugin_id, plugin in plugins.items():
        info = plugin.get_info()
        
        logger.info(f"  - {info['name']} v{info['version']} ({info['type']})")
        logger.info(f"    ID: {plugin_id}")
        logger.info(f"    Enabled: {info['enabled']}")
        logger.info(f"    Hooks: {', '.join(info['hooks'])}")
    
    # Get special plugin insights
    for plugin_id, plugin in plugins.items():
        if plugin.name == "TaskReporter":
            # Get task summary
            summary = plugin.get_task_summary()
            
            logger.info("\nTask Reporter Summary:")
            logger.info(f"  Total tasks: {summary['total_tasks']}")
            logger.info(f"  Status counts: {json.dumps(summary['status_counts'])}")
            logger.info(f"  Average duration: {summary['avg_duration']:.2f} seconds")
        
        elif plugin.name == "CIIntegration":
            # Get CI status
            status = plugin.get_ci_status()
            
            logger.info("\nCI Integration Status:")
            logger.info(f"  CI system: {status['ci_system']}")
            logger.info(f"  Repository: {status['repository']}")
            logger.info(f"  Test run ID: {status['test_run_id']}")
            logger.info(f"  Test run status: {status['test_run_status']}")

async def main():
    """Main entry point for the test script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Plugin Architecture")
    
    parser.add_argument("--plugins", default="sample_reporter_plugin.py,ci_integration_plugin.py",
                       help="Comma-separated list of plugins to load")
    parser.add_argument("--enable-external-reporting", action="store_true",
                       help="Enable external reporting in the TaskReporter plugin")
    parser.add_argument("--ci-system", default="github",
                       help="CI system to use (github, jenkins, gitlab)")
    parser.add_argument("--simulate-tasks", type=int, default=5,
                       help="Number of tasks to simulate")
    
    args = parser.parse_args()
    
    # Parse plugin list
    plugin_dirs = ["plugins"]
    
    logger.info("Starting plugin architecture test")
    logger.info(f"Plugin directories: {plugin_dirs}")
    
    # Create coordinator
    coordinator = DistributedTestingCoordinator(
        db_path=":memory:",
        host="127.0.0.1",
        port=8080,
        security_config=None,
        enable_advanced_scheduler=False,
        enable_health_monitor=False,
        enable_load_balancer=False,
        enable_auto_recovery=False,
        enable_redundancy=False,
        enable_plugins=True,
        plugin_dirs=plugin_dirs
    )
    
    # Initialize plugins
    await coordinator.start()
    
    try:
        # Configure plugins
        await configure_plugins(coordinator, args)
        
        # Display loaded plugins
        await display_plugin_info(coordinator)
        
        # Simulate task lifecycle
        if args.simulate_tasks > 0:
            await simulate_tasks(coordinator, args.simulate_tasks)
            
            # Wait for any asynchronous plugin operations
            await anyio.sleep(2)
            
            # Display plugin info after simulation
            await display_plugin_info(coordinator)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
    finally:
        # Shutdown coordinator
        await coordinator.stop()
    
    logger.info("Plugin architecture test completed")

if __name__ == "__main__":
    anyio.run(main())