#!/usr/bin/env python3
"""
Run test for fault tolerance components of the distributed testing framework.

This script demonstrates how to use the enhanced fault tolerance features of the 
distributed testing framework, including coordinator redundancy, distributed state
management, and comprehensive error recovery.

Usage:
    python run_test_fault_tolerance.py [options]
"""

import argparse
import anyio
import json
import logging
import os
import random
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("fault_tolerance_test")

async def start_coordinator(args, coordinator_id, port):
    """
    Start a coordinator instance.
    
    Args:
        args: Command line arguments
        coordinator_id: Coordinator ID
        port: Port to bind to
    """
    from coordinator import DistributedTestingCoordinator
    
    db_path = f"test_db_{coordinator_id}.duckdb"
    security_config = f"security_config_{coordinator_id}.json"
    
    # Create cluster nodes list
    cluster_nodes = [f"http://localhost:{p}" for p in range(args.base_port, args.base_port + args.num_coordinators)]
    
    # Create coordinator
    coordinator = DistributedTestingCoordinator(
        db_path=db_path,
        host="0.0.0.0", 
        port=port,
        security_config=security_config,
        enable_redundancy=True,
        cluster_nodes=cluster_nodes,
        node_id=f"coordinator-{coordinator_id}"
    )
    
    # Start coordinator
    runner = coordinator.app.make_handler()
    server = await # TODO: Remove event loop management - asyncio.get_event_loop().create_server(runner, '0.0.0.0', port)
    
    logger.info(f"Coordinator {coordinator_id} started on port {port}")
    
    return coordinator, runner, server

async def start_worker(args, worker_id, coordinator_port):
    """
    Start a worker instance.
    
    Args:
        args: Command line arguments
        worker_id: Worker ID
        coordinator_port: Port of coordinator
    """
    from worker import DistributedTestingWorker
    
    # Create worker
    worker = DistributedTestingWorker(
        coordinator_url=f"http://localhost:{coordinator_port}",
        worker_id=f"worker-{worker_id}",
        hostname=f"worker-host-{worker_id}"
    )
    
    # Start worker
    await worker.start()
    
    logger.info(f"Worker {worker_id} started and connected to coordinator on port {coordinator_port}")
    
    return worker

async def submit_test_tasks(args, coordinator_port, num_tasks=5):
    """
    Submit test tasks to coordinator.
    
    Args:
        args: Command line arguments
        coordinator_port: Port of coordinator
        num_tasks: Number of tasks to submit
    """
    import aiohttp
    
    logger.info(f"Submitting {num_tasks} test tasks to coordinator on port {coordinator_port}")
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_tasks):
            # Create task
            task_data = {
                "type": "test",
                "priority": random.randint(1, 10),
                "config": {
                    "test_name": f"test_{i}",
                    "timeout_seconds": 30,
                    "retry_policy": {
                        "max_retries": 3
                    }
                },
                "requirements": {
                    "hardware": ["cpu"],
                    "min_memory_gb": 1
                }
            }
            
            # Submit task
            async with session.post(
                f"http://localhost:{coordinator_port}/api/tasks",
                json=task_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Task {i} submitted successfully: {result}")
                else:
                    logger.error(f"Error submitting task {i}: {response.status}")

async def simulate_coordinator_failure(args, coordinator_id, coordinator, server):
    """
    Simulate a coordinator failure.
    
    Args:
        args: Command line arguments
        coordinator_id: Coordinator ID
        coordinator: Coordinator instance
        server: Server instance
    """
    logger.info(f"Simulating failure of coordinator {coordinator_id}")
    
    # Stop the server
    server.close()
    await server.wait_closed()
    
    logger.info(f"Coordinator {coordinator_id} stopped (simulated failure)")
    
    # Wait for some time to simulate downtime
    await anyio.sleep(10)
    
    # Restart the server (recovery)
    port = args.base_port + coordinator_id
    server = await # TODO: Remove event loop management - asyncio.get_event_loop().create_server(
        coordinator.app.make_handler(), 
        '0.0.0.0', 
        port
    )
    
    logger.info(f"Coordinator {coordinator_id} restarted (recovered from failure)")
    
    return server

async def simulate_worker_failure(args, worker_id, worker):
    """
    Simulate a worker failure.
    
    Args:
        args: Command line arguments
        worker_id: Worker ID
        worker: Worker instance
    """
    logger.info(f"Simulating failure of worker {worker_id}")
    
    # Stop the worker
    await worker.stop()
    
    logger.info(f"Worker {worker_id} stopped (simulated failure)")
    
    # Wait for some time to simulate downtime
    await anyio.sleep(10)
    
    # Restart the worker (recovery)
    await worker.start()
    
    logger.info(f"Worker {worker_id} restarted (recovered from failure)")

async def simulate_network_partition(args, coordinator_id1, coordinator_id2):
    """
    Simulate a network partition between two coordinators.
    
    Args:
        args: Command line arguments
        coordinator_id1: First coordinator ID
        coordinator_id2: Second coordinator ID
    """
    logger.info(f"Simulating network partition between coordinators {coordinator_id1} and {coordinator_id2}")
    
    # This is a simplified simulation - in a real environment, you would use tools like
    # iptables or tc to create network partitions between the nodes
    
    # We'll inject a "partition" flag into the coordinators' sessions that will make them
    # reject connections from the partitioned coordinator
    
    # In a real implementation, you would modify the coordinator code to check for this flag,
    # but for this demonstration we'll just log the simulation
    
    logger.info(f"Network partition active between coordinators {coordinator_id1} and {coordinator_id2}")
    
    # Wait for some time to simulate partition duration
    await anyio.sleep(20)
    
    logger.info(f"Network partition resolved between coordinators {coordinator_id1} and {coordinator_id2}")

async def monitor_cluster_status(args, coordinator_ports):
    """
    Monitor the status of the cluster.
    
    Args:
        args: Command line arguments
        coordinator_ports: List of coordinator ports
    """
    import aiohttp
    
    logger.info("Starting cluster status monitoring")
    
    while True:
        try:
            # Check status of each coordinator
            async with aiohttp.ClientSession() as session:
                for port in coordinator_ports:
                    try:
                        async with session.get(f"http://localhost:{port}/status", timeout=2) as response:
                            if response.status == 200:
                                data = await response.json()
                                workers = data.get("workers", {})
                                tasks = data.get("tasks", {})
                                
                                logger.info(f"Coordinator on port {port}: {workers.get('active', 0)}/{workers.get('total', 0)} active workers, {tasks.get('running', 0)}/{tasks.get('total', 0)} running tasks")
                            else:
                                logger.warning(f"Coordinator on port {port} returned status {response.status}")
                    except Exception as e:
                        logger.error(f"Error connecting to coordinator on port {port}: {e}")
            
            # Wait before next check
            await anyio.sleep(5)
        except anyio.get_cancelled_exc_class():
            break
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            await anyio.sleep(5)

async def test_distributed_state_management(args):
    """
    Test the distributed state management functionality.
    
    Args:
        args: Command line arguments
    """
    from distributed_state_management import DistributedStateManager
    
    logger.info("Testing distributed state management")
    
    # Create cluster nodes list
    cluster_nodes = [f"http://localhost:{p}" for p in range(args.base_port, args.base_port + args.num_coordinators)]
    
    # Create state managers
    state_managers = []
    
    for i in range(args.num_coordinators):
        state_dir = f"state_test_{i}"
        os.makedirs(state_dir, exist_ok=True)
        
        state_manager = DistributedStateManager(
            coordinator=None,  # Not needed for this test
            cluster_nodes=cluster_nodes,
            node_id=f"node-{i}",
            state_dir=state_dir
        )
        
        await state_manager.start()
        state_managers.append(state_manager)
    
    logger.info(f"Created {len(state_managers)} state managers")
    
    # Update some state on the first manager
    state_managers[0].update("workers", "worker-1", {"status": "active", "last_seen": time.time()})
    state_managers[0].update("workers", "worker-2", {"status": "idle", "last_seen": time.time()})
    state_managers[0].update("tasks", "task-1", {"status": "running", "worker_id": "worker-1"})
    
    logger.info("Updated state on first manager, waiting for propagation")
    
    # Wait for state to propagate
    await anyio.sleep(10)
    
    # Check state on the second manager
    worker1 = state_managers[1].get("workers", "worker-1")
    worker2 = state_managers[1].get("workers", "worker-2")
    task1 = state_managers[1].get("tasks", "task-1")
    
    logger.info(f"State from second manager: worker1={worker1}, worker2={worker2}, task1={task1}")
    
    # Update state on the second manager
    state_managers[1].update("workers", "worker-3", {"status": "active", "last_seen": time.time()})
    state_managers[1].update("tasks", "task-2", {"status": "pending"})
    
    logger.info("Updated state on second manager, waiting for propagation")
    
    # Wait for state to propagate
    await anyio.sleep(10)
    
    # Check state on the first manager
    worker3 = state_managers[0].get("workers", "worker-3")
    task2 = state_managers[0].get("tasks", "task-2")
    
    logger.info(f"State from first manager: worker3={worker3}, task2={task2}")
    
    # Clean up
    for state_manager in state_managers:
        await state_manager.stop()
    
    logger.info("Distributed state management test completed")

async def test_error_recovery_strategies(args):
    """
    Test the error recovery strategies.
    
    Args:
        args: Command line arguments
    """
    from error_recovery_strategies import EnhancedErrorRecoveryManager, ErrorCategory
    
    logger.info("Testing error recovery strategies")
    
    # Create a mock coordinator
    class MockCoordinator:
        def __init__(self):
            self.workers = {}
            self.tasks = {}
            self.running_tasks = {}
            self.pending_tasks = set()
            self.failed_tasks = set()
            self.worker_connections = {}
            self.db = None
    
    # Create mock coordinator
    coordinator = MockCoordinator()
    
    # Add some mock data
    coordinator.workers = {
        "worker-1": {"worker_id": "worker-1", "status": "active", "hostname": "host1"},
        "worker-2": {"worker_id": "worker-2", "status": "idle", "hostname": "host2"}
    }
    coordinator.tasks = {
        "task-1": {"task_id": "task-1", "status": "running", "worker_id": "worker-1", "attempts": 1, "config": {"retry_policy": {"max_retries": 3}}},
        "task-2": {"task_id": "task-2", "status": "pending", "attempts": 0, "config": {}}
    }
    coordinator.running_tasks = {"task-1": "worker-1"}
    
    # Create recovery manager
    recovery_manager = EnhancedErrorRecoveryManager(coordinator)
    
    # Test worker offline recovery
    logger.info("Testing worker offline recovery")
    
    # Simulate worker-1 going offline
    success, info = await recovery_manager.recover(
        Exception("Worker disconnected"),
        {"component": "worker", "worker_id": "worker-1"}
    )
    
    logger.info(f"Worker offline recovery success: {success}, task-1 status: {coordinator.tasks['task-1']['status']}")
    
    # Test retry strategy
    logger.info("Testing retry strategy")
    
    # Define a function that will fail the first time but succeed the second time
    attempt_count = 0
    async def flaky_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise Exception("First attempt failed")
        return "Success"
    
    # Try to recover from a failure with retry
    success, info = await recovery_manager.recover(
        Exception("Operation failed"),
        {"component": "task", "operation": flaky_operation}
    )
    
    logger.info(f"Retry strategy success: {success}, attempts: {attempt_count}")
    
    # Get recovery stats
    stats = recovery_manager.get_strategy_stats()
    logger.info(f"Recovery strategy stats: {stats}")
    
    logger.info("Error recovery strategies test completed")

async def run_test(args):
    """
    Run the main test.
    
    Args:
        args: Command line arguments
    """
    # Create coordinators
    coordinators = []
    runners = []
    servers = []
    
    for i in range(args.num_coordinators):
        port = args.base_port + i
        coordinator, runner, server = await start_coordinator(args, i, port)
        coordinators.append(coordinator)
        runners.append(runner)
        servers.append(server)
    
    # Wait for coordinators to initialize
    await anyio.sleep(5)
    
    # Create workers
    workers = []
    for i in range(args.num_workers):
        # Connect to a random coordinator
        coordinator_port = args.base_port + random.randint(0, args.num_coordinators - 1)
        worker = await start_worker(args, i, coordinator_port)
        workers.append(worker)
    
    # Wait for workers to register
    await anyio.sleep(5)
    
    # Submit tasks to a coordinator
    coordinator_port = args.base_port  # Use the first coordinator
    await submit_test_tasks(args, coordinator_port, args.num_tasks)
    
    # Start monitoring
    monitor_task = # TODO: Replace with task group - asyncio.create_task(monitor_cluster_status(args, [args.base_port + i for i in range(args.num_coordinators)]))
    
    # Wait for tasks to start executing
    await anyio.sleep(10)
    
    # Perform test 1: Coordinator failure and recovery
    if args.test_coordinator_failure:
        await simulate_coordinator_failure(args, 0, coordinators[0], servers[0])
        
        # Wait for the cluster to stabilize
        await anyio.sleep(10)
    
    # Perform test 2: Worker failure and recovery
    if args.test_worker_failure:
        await simulate_worker_failure(args, 0, workers[0])
        
        # Wait for the cluster to stabilize
        await anyio.sleep(10)
    
    # Perform test 3: Network partition
    if args.test_network_partition and args.num_coordinators >= 2:
        await simulate_network_partition(args, 0, 1)
        
        # Wait for the cluster to stabilize
        await anyio.sleep(10)
    
    # Test distributed state management
    if args.test_state_management:
        await test_distributed_state_management(args)
    
    # Test error recovery strategies
    if args.test_error_recovery:
        await test_error_recovery_strategies(args)
    
    # Let the system run for a while
    if args.run_time > 0:
        logger.info(f"Running the system for {args.run_time} seconds")
        await anyio.sleep(args.run_time)
    
    # Stop monitoring
    monitor_task.cancel()
    try:
        await monitor_task
    except anyio.get_cancelled_exc_class():
        pass
    
    # Stop workers
    for worker in workers:
        await worker.stop()
    
    # Stop coordinators
    for server in servers:
        server.close()
        await server.wait_closed()
    
    for coordinator in coordinators:
        if hasattr(coordinator, 'stop'):
            await coordinator.stop()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the fault tolerance components of the distributed testing framework")
    
    parser.add_argument("--num-coordinators", type=int, default=2, help="Number of coordinator instances to start")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of worker instances to start")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks to submit")
    parser.add_argument("--base-port", type=int, default=8090, help="Base port for coordinators")
    parser.add_argument("--run-time", type=int, default=60, help="Time to run the test in seconds")
    
    parser.add_argument("--test-coordinator-failure", action="store_true", help="Test coordinator failure and recovery")
    parser.add_argument("--test-worker-failure", action="store_true", help="Test worker failure and recovery")
    parser.add_argument("--test-network-partition", action="store_true", help="Test network partition")
    parser.add_argument("--test-state-management", action="store_true", help="Test distributed state management")
    parser.add_argument("--test-error-recovery", action="store_true", help="Test error recovery strategies")
    parser.add_argument("--test-all", action="store_true", help="Test all fault tolerance components")
    
    args = parser.parse_args()
    
    # If test_all is specified, enable all tests
    if args.test_all:
        args.test_coordinator_failure = True
        args.test_worker_failure = True
        args.test_network_partition = True
        args.test_state_management = True
        args.test_error_recovery = True
    
    return args

def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Fault Tolerance Test - Starting")
    logger.info(f"Configuration: {args}")
    
    # Run the test
    anyio.run(run_test(args))
    
    logger.info("Fault Tolerance Test - Completed")

if __name__ == "__main__":
    main()