#!/usr/bin/env python3
"""
Test runner for the auto recovery module of the distributed testing framework.

This script starts a coordinator and a few worker nodes, simulates failures,
and demonstrates the auto recovery features.

Usage:
    python run_test_auto_recovery.py
"""

import asyncio
import argparse
import logging
import signal
import sys
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_coordinator(db_path='./test_auto_recovery.duckdb', port=8081):
    """Run the coordinator process."""
    import subprocess
    
    # Start coordinator with health monitoring and auto recovery
    cmd = [
        'python', 'coordinator.py',
        '--db-path', db_path,
        '--port', str(port),
        '--security-config', './test_security_config.json',
        '--generate-admin-key',
        '--generate-worker-key'
    ]
    
    # Start the coordinator process
    logger.info("Starting coordinator...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for startup
    await asyncio.sleep(2)
    
    # Read output to get API keys
    output, error = proc.communicate(timeout=1)
    
    # Check if started successfully
    if proc.returncode is not None:
        logger.error(f"Failed to start coordinator: {error.decode()}")
        return None, None
    
    # Parse output to get API keys
    admin_key = None
    worker_key = None
    
    output_str = output.decode()
    for line in output_str.splitlines():
        if "Admin API key:" in line:
            admin_key = line.split("Admin API key:")[1].strip()
        elif "Worker API key:" in line:
            worker_key = line.split("Worker API key:")[1].strip()
    
    logger.info(f"Coordinator started on port {port}")
    
    return proc, worker_key

async def run_worker(worker_id, coordinator_url, api_key):
    """Run a worker process."""
    import subprocess
    
    cmd = [
        'python', 'worker.py',
        '--coordinator', coordinator_url,
        '--worker-id', worker_id,
        '--api-key', api_key
    ]
    
    # Start the worker process
    logger.info(f"Starting worker {worker_id}...")
    proc = subprocess.Popen(cmd)
    
    # Wait for startup
    await asyncio.sleep(2)
    
    logger.info(f"Worker {worker_id} started")
    
    return proc

async def simulate_worker_failure(worker_proc):
    """Simulate a worker failure by terminating the process."""
    logger.info("Simulating worker failure...")
    
    # Terminate the worker process
    worker_proc.terminate()
    
    # Wait for termination
    await asyncio.sleep(1)
    
    logger.info("Worker terminated")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the auto recovery module")
    parser.add_argument("--db-path", default="./test_auto_recovery.duckdb", help="Path to DuckDB database")
    parser.add_argument("--port", type=int, default=8081, help="Port for coordinator")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of workers to start")
    parser.add_argument("--run-time", type=int, default=60, help="How long to run the test in seconds")
    
    args = parser.parse_args()
    
    try:
        # Start coordinator
        coordinator_proc, worker_key = await run_coordinator(args.db_path, args.port)
        if not coordinator_proc or not worker_key:
            logger.error("Failed to start coordinator or get worker key")
            return
        
        # Start workers
        worker_procs = []
        for i in range(args.num_workers):
            worker_id = f"worker-{i+1}"
            worker_proc = await run_worker(worker_id, f"http://localhost:{args.port}", worker_key)
            worker_procs.append((worker_id, worker_proc))
        
        # Wait for everything to stabilize
        logger.info("System started, waiting for stabilization...")
        await asyncio.sleep(10)
        
        # Simulate worker failure for the first worker
        if worker_procs:
            worker_id, worker_proc = worker_procs[0]
            logger.info(f"Simulating failure for worker {worker_id}")
            await simulate_worker_failure(worker_proc)
        
        # Restart the failed worker after a delay
        await asyncio.sleep(15)
        if worker_procs:
            worker_id, _ = worker_procs[0]
            logger.info(f"Restarting worker {worker_id}")
            new_worker_proc = await run_worker(worker_id, f"http://localhost:{args.port}", worker_key)
            worker_procs[0] = (worker_id, new_worker_proc)
        
        # Run for the specified time
        logger.info(f"Running test for {args.run_time} seconds...")
        await asyncio.sleep(args.run_time)
        
        # Clean up
        logger.info("Test completed, cleaning up...")
        
        # Terminate workers
        for _, proc in worker_procs:
            proc.terminate()
        
        # Terminate coordinator
        coordinator_proc.terminate()
        
        logger.info("Test completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())