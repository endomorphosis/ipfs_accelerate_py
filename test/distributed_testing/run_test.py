#!/usr/bin/env python3
"""
Distributed Testing Framework - Test Runner

This script helps run and test the distributed testing framework components.
It can start both coordinator and worker processes to test their interaction.

Usage:
    python run_test.py --mode=all --db-path=./test_db.duckdb
    python run_test.py --mode=coordinator --db-path=./test_db.duckdb
    python run_test.py --mode=worker --coordinator=http://localhost:8080
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run_test.log")
    ]
)
logger = logging.getLogger(__name__)

async def run_coordinator(db_path, host="localhost", port=8080):
    """Run coordinator process."""
    logger.info(f"Starting coordinator with database at {db_path}")
    
    # Ensure database directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    process = await asyncio.create_subprocess_exec(
        sys.executable, "./coordinator.py",
        f"--host={host}", f"--port={port}", f"--db-path={db_path}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    logger.info(f"Coordinator process started with PID {process.pid}")
    
    # Return process for later termination
    return process

async def run_worker(coordinator_url, db_path=None, worker_id=None):
    """Run worker process."""
    logger.info(f"Starting worker connecting to {coordinator_url}")
    
    cmd = [sys.executable, "./worker.py", f"--coordinator={coordinator_url}"]
    
    if db_path:
        # Ensure database directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        cmd.append(f"--db-path={db_path}")
    
    if worker_id:
        cmd.append(f"--worker-id={worker_id}")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    logger.info(f"Worker process started with PID {process.pid}")
    
    # Return process for later termination
    return process

async def log_process_output(process, name):
    """Log process output."""
    async def read_stream(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            logger.info(f"{prefix} {line.decode().strip()}")
    
    await asyncio.gather(
        read_stream(process.stdout, f"[{name} stdout]"),
        read_stream(process.stderr, f"[{name} stderr]")
    )

async def submit_test_tasks(coordinator_url, num_tasks=3):
    """Submit test tasks to the coordinator."""
    import aiohttp
    
    logger.info(f"Submitting {num_tasks} test tasks to coordinator")
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_tasks):
            # Create benchmark task
            task_data = {
                "type": "benchmark",
                "priority": 1,
                "config": {
                    "model": f"test-model-{i+1}",
                    "batch_sizes": [1, 2, 4, 8],
                    "precision": "fp16",
                    "iterations": 5
                },
                "requirements": {
                    "hardware": ["cpu"],
                    "min_memory_gb": 1
                }
            }
            
            try:
                async with session.post(f"{coordinator_url}/api/tasks", json=task_data) as resp:
                    result = await resp.json()
                    logger.info(f"Task {i+1} submission result: {result}")
                    
                    # Wait a bit between task submissions
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error submitting task {i+1}: {str(e)}")

async def run_all_tests(db_path, host="localhost", port=8080, num_workers=2, run_time=60):
    """Run all tests - coordinator and workers."""
    coordinator_url = f"http://{host}:{port}"
    
    try:
        # Start coordinator
        coordinator_process = await run_coordinator(db_path, host, port)
        
        # Wait for coordinator to start
        logger.info("Waiting for coordinator to start...")
        await asyncio.sleep(5)
        
        # Start multiple workers
        worker_processes = []
        for i in range(num_workers):
            worker_process = await run_worker(coordinator_url, db_path)
            worker_processes.append(worker_process)
            
            # Wait a bit between starting workers
            await asyncio.sleep(1)
        
        # Wait for workers to connect
        logger.info("Waiting for workers to connect...")
        await asyncio.sleep(5)
        
        # Submit test tasks
        await submit_test_tasks(coordinator_url, num_tasks=5)
        
        # Log output for all processes
        log_tasks = []
        log_tasks.append(asyncio.create_task(log_process_output(coordinator_process, "Coordinator")))
        for i, proc in enumerate(worker_processes):
            log_tasks.append(asyncio.create_task(log_process_output(proc, f"Worker-{i+1}")))
        
        # Run for specified time
        logger.info(f"Running test for {run_time} seconds...")
        await asyncio.sleep(run_time)
        
        # Terminate processes
        logger.info("Terminating processes...")
        coordinator_process.terminate()
        for proc in worker_processes:
            proc.terminate()
        
        # Wait for processes to terminate
        await coordinator_process.wait()
        for proc in worker_processes:
            await proc.wait()
        
        # Cancel log tasks
        for task in log_tasks:
            task.cancel()
            
        logger.info("Test completed")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        
    finally:
        # Clean up any remaining processes
        try:
            coordinator_process.terminate()
        except:
            pass
            
        for proc in worker_processes:
            try:
                proc.terminate()
            except:
                pass

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Distributed Testing Framework Test Runner")
    parser.add_argument("--mode", choices=["coordinator", "worker", "all"], default="all", 
                        help="Which component(s) to run")
    parser.add_argument("--db-path", default="./test_db.duckdb", help="Path to DuckDB database")
    parser.add_argument("--host", default="localhost", help="Host for coordinator")
    parser.add_argument("--port", type=int, default=8080, help="Port for coordinator")
    parser.add_argument("--coordinator", default=None, 
                        help="URL of coordinator (for worker mode)")
    parser.add_argument("--worker-id", default=None, help="Worker ID (for worker mode)")
    parser.add_argument("--num-workers", type=int, default=2, 
                        help="Number of workers to start (for all mode)")
    parser.add_argument("--run-time", type=int, default=60, 
                        help="How long to run the test in seconds (for all mode)")
    
    args = parser.parse_args()
    
    if args.mode == "coordinator":
        # Run coordinator only
        coordinator_process = await run_coordinator(args.db_path, args.host, args.port)
        await log_process_output(coordinator_process, "Coordinator")
        
    elif args.mode == "worker":
        # Run worker only
        if not args.coordinator:
            logger.error("Coordinator URL must be provided in worker mode")
            return
            
        worker_process = await run_worker(args.coordinator, args.db_path, args.worker_id)
        await log_process_output(worker_process, "Worker")
        
    elif args.mode == "all":
        # Run full test with coordinator and workers
        await run_all_tests(
            args.db_path, args.host, args.port, 
            args.num_workers, args.run_time
        )

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    
    asyncio.run(main())