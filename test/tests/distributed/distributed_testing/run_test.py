#!/usr/bin/env python3
"""
Distributed Testing Framework - Test Runner

This script helps run and test the distributed testing framework components.
It can start both coordinator and worker processes to test their interaction.

IMPORTANT NOTICE: As of March 16, 2025, security and authentication features have been 
marked as OUT OF SCOPE for the distributed testing framework. Please refer to
SECURITY_DEPRECATED.md for more information. Any security-related tests
will be skipped.

Usage:
    python run_test.py --mode=all --db-path=./test_db.duckdb
    python run_test.py --mode=coordinator --db-path=./test_db.duckdb
    python run_test.py --mode=worker --coordinator=http://localhost:8080
"""

from ipfs_accelerate_py.anyio_helpers import gather, wait_for
import anyio
    import argparse
    import anyio
    import json
    import logging
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

# Configure logging
    logging.basicConfig())))))))
    level=logging.INFO,
    format='%())))))))asctime)s - %())))))))name)s - %())))))))levelname)s - %())))))))message)s',
    handlers=[]],,
    logging.StreamHandler())))))))),
    logging.FileHandler())))))))"run_test.log")
    ]
    )
    logger = logging.getLogger())))))))__name__)

    async def run_coordinator())))))))db_path, host="localhost", port=8080, security_config="./security_config.json",
    generate_keys=True, disable_health_monitor=False, disable_advanced_scheduler=False,
                   disable_load_balancer=False, max_tasks_per_worker=1):
                       """
                       Run coordinator process.
                       
                       NOTE: Security features are OUT OF SCOPE. Security parameters are kept for 
                       backward compatibility but have no effect. See SECURITY_DEPRECATED.md.
                       """
                       logger.info())))))))f"Starting coordinator with database at {}}}db_path}")
                       logger.info())))))))f"NOTICE: Security features are OUT OF SCOPE.")
    
    # Ensure database directory exists
                       db_dir = os.path.dirname())))))))db_path)
    if db_dir and not os.path.exists())))))))db_dir):
        os.makedirs())))))))db_dir)
    
        cmd = []],,
        sys.executable, "./coordinator.py",
        f"--host={}}}host}", 
        f"--port={}}}port}", 
        f"--db-path={}}}db_path}",
        f"--security-config={}}}security_config}",
        f"--max-tasks-per-worker={}}}max_tasks_per_worker}"
        ]
    
    # Generate keys if requested:
    if generate_keys:
        cmd.append())))))))"--generate-admin-key")
        cmd.append())))))))"--generate-worker-key")
    
    # Add feature flags
    if disable_health_monitor:
        cmd.append())))))))"--disable-health-monitor")
    
    if disable_advanced_scheduler:
        cmd.append())))))))"--disable-advanced-scheduler")
        
    if disable_load_balancer:
        cmd.append())))))))"--disable-load-balancer")
    
        process = await anyio.open_process())))))))
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )
    
        logger.info())))))))f"Coordinator process started with PID {}}}process.pid}")
    
    # Wait for a moment to ensure keys are generated and displayed
    if generate_keys:
        await anyio.sleep())))))))1)
        
        # Extract API keys for workers
        lines = []],,]
        while len())))))))lines) < 50:  # Capture enough lines to find API keys
            try:
                line = await wait_for())))))))process.stdout.readline())))))))), 0.1)
                if not line:
                break
                    
                decoded_line = line.decode())))))))).strip()))))))))
                lines.append())))))))decoded_line)
                if "=== WORKER API KEY ===" in decoded_line:
                    # Extract worker API key
                    api_key_line_index = lines.index())))))))decoded_line)
                    if len())))))))lines) > api_key_line_index + 1:
                    return process, lines[]],,api_key_line_index + 1]
            except TimeoutError:
                    break
                
    # Return process and None for API key if not found
                return process, None
:
async def run_worker())))))))coordinator_url, db_path=None, worker_id=None, api_key=None):
    """
    Run worker process.
    
    NOTE: Security features are OUT OF SCOPE. API key parameter is kept for 
    backward compatibility but has no effect. See SECURITY_DEPRECATED.md.
    """
    logger.info())))))))f"Starting worker connecting to {}}}coordinator_url}")
    if api_key:
        logger.info())))))))f"NOTICE: Security features are OUT OF SCOPE. API key has no effect.")
    
    cmd = []],,sys.executable, "./worker.py", f"--coordinator={}}}coordinator_url}"]
    
    if db_path:
        # Ensure database directory exists
        db_dir = os.path.dirname())))))))db_path)
        if db_dir and not os.path.exists())))))))db_dir):
            os.makedirs())))))))db_dir)
            
            cmd.append())))))))f"--db-path={}}}db_path}")
    
    if worker_id:
        cmd.append())))))))f"--worker-id={}}}worker_id}")
        
    if api_key:
        cmd.append())))))))f"--api-key={}}}api_key}")
    
        process = await anyio.open_process())))))))
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )
    
        logger.info())))))))f"Worker process started with PID {}}}process.pid}")
    
    # Return process for later termination
        return process

async def log_process_output())))))))process, name):
    """Log process output."""
    async def read_stream())))))))stream, prefix):
        while True:
            line = await stream.readline()))))))))
            if not line:
            break
            logger.info())))))))f"{}}}prefix} {}}}line.decode())))))))).strip()))))))))}")
    
            await gather())))))))
            read_stream())))))))process.stdout, f"[]],,{}}}name} stdout]"),
            read_stream())))))))process.stderr, f"[]],,{}}}name} stderr]")
            )

async def submit_test_tasks())))))))coordinator_url, num_tasks=3, api_key=None):
    """
    Submit test tasks to the coordinator.
    
    NOTE: Security features are OUT OF SCOPE. API key parameter is kept for 
    backward compatibility but has no effect. See SECURITY_DEPRECATED.md.
    """
    import aiohttp
    
    logger.info())))))))f"Submitting {}}}num_tasks} test tasks to coordinator")
    logger.info())))))))f"NOTICE: Security features are OUT OF SCOPE. Authentication is not required.")
    
    # Security is out of scope, no need to load API keys
    if api_key:
        logger.info())))))))f"NOTICE: API key provided but security features are OUT OF SCOPE.")
    
    # No authentication headers - security is out of scope
    headers = {}}}}:)
    
    async with aiohttp.ClientSession())))))))headers=headers) as session:
        for i in range())))))))num_tasks):
            # Create benchmark task
            task_data = {}}}
            "type": "benchmark",
            "priority": 1,
            "config": {}}}
            "model": f"test-model-{}}}i+1}",
            "batch_sizes": []],,1, 2, 4, 8],
            "precision": "fp16",
            "iterations": 5
            },
            "requirements": {}}}
            "hardware": []],,"cpu"],
            "min_memory_gb": 1
            }
            }
            
            try:
                async with session.post())))))))f"{}}}coordinator_url}/api/tasks", json=task_data) as resp:
                    if resp.status == 401:
                        logger.error())))))))"Authentication failed when submitting task. Check API key.")
                    return False
                    
                    result = await resp.json()))))))))
                    logger.info())))))))f"Task {}}}i+1} submission result: {}}}result}")
                    
                    # Wait a bit between task submissions
                    await anyio.sleep())))))))1)
                    
            except Exception as e:
                logger.error())))))))f"Error submitting task {}}}i+1}: {}}}str())))))))e)}")
                
                    return True

                    async def run_all_tests())))))))db_path, host="localhost", port=8080, num_workers=2, run_time=60,
                    security_config="./test_security_config.json", disable_health_monitor=False,
                    disable_advanced_scheduler=False, disable_load_balancer=False,
                   max_tasks_per_worker=1, num_test_tasks=5):
                       """Run all tests - coordinator and workers."""
                       coordinator_url = f"http://{}}}host}:{}}}port}"
                       worker_processes = []],,]
    
    try:
        # Start coordinator (security features out of scope)
        logger.info())))))))"NOTICE: Security features are OUT OF SCOPE. Starting coordinator...")
        coordinator_process, worker_api_key = await run_coordinator())))))))
        db_path, host, port, security_config, generate_keys=False,  # No key generation needed
        disable_health_monitor=disable_health_monitor,
        disable_advanced_scheduler=disable_advanced_scheduler,
        disable_load_balancer=disable_load_balancer,
        max_tasks_per_worker=max_tasks_per_worker
        )
        
        # Wait for coordinator to start
        logger.info())))))))"Waiting for coordinator to start...")
        await anyio.sleep())))))))5)
        
        # SECURITY OUT OF SCOPE: No authentication needed
        logger.info())))))))f"NOTICE: Security features are OUT OF SCOPE. No authentication needed.")
        
        # Start multiple workers
        for i in range())))))))num_workers):
            worker_process = await run_worker())))))))coordinator_url, db_path, api_key=worker_api_key)
            worker_processes.append())))))))worker_process)
            
            # Wait a bit between starting workers
            await anyio.sleep())))))))1)
        
        # Wait for workers to connect
            logger.info())))))))"Waiting for workers to connect...")
            await anyio.sleep())))))))8)
        
        # SECURITY OUT OF SCOPE: No need for API keys
            logger.info())))))))f"NOTICE: Security features are OUT OF SCOPE. No API keys needed for task submission.")
            admin_api_key = None  # Kept for backward compatibility
        
        # Submit test tasks with admin API key
            await submit_test_tasks())))))))coordinator_url, num_tasks=num_test_tasks, api_key=admin_api_key)
        
        # Log output for all processes
            log_tasks = []],,]
            log_tasks.append())))))))# TODO: Replace with task group - anyio task group for coordinator logging
        for i, proc in enumerate())))))))worker_processes):
            log_tasks.append())))))))# TODO: Replace with task group - anyio task group for worker logging
        
        # Run for specified time
            logger.info())))))))f"Running test for {}}}run_time} seconds...")
            await anyio.sleep())))))))run_time)
        
        # Terminate processes
            logger.info())))))))"Terminating processes...")
            coordinator_process.terminate()))))))))
        for proc in worker_processes:
            proc.terminate()))))))))
        
        # Wait for processes to terminate
            await coordinator_process.wait()))))))))
        for proc in worker_processes:
            await proc.wait()))))))))
        
        # Cancel log tasks
        for task in log_tasks:
            task.cancel()))))))))
            
            logger.info())))))))"Test completed")
        
    except Exception as e:
        logger.error())))))))f"Error in test: {}}}str())))))))e)}")
        
    finally:
        # Clean up any remaining processes
        try:
            if 'coordinator_process' in locals())))))))):
                coordinator_process.terminate()))))))))
        except:
                pass
            
        for proc in worker_processes:
            try:
                proc.terminate()))))))))
            except:
                pass

async def main())))))))):
    """Main function."""
    parser = argparse.ArgumentParser())))))))description="Distributed Testing Framework Test Runner")
    parser.add_argument())))))))"--mode", choices=[]],,"coordinator", "worker", "all"], default="all", 
    help="Which component())))))))s) to run")
    parser.add_argument())))))))"--db-path", default="./test_db.duckdb", help="Path to DuckDB database")
    parser.add_argument())))))))"--host", default="localhost", help="Host for coordinator")
    parser.add_argument())))))))"--port", type=int, default=8080, help="Port for coordinator")
    parser.add_argument())))))))"--coordinator", default=None, 
    help="URL of coordinator ())))))))for worker mode)")
    parser.add_argument())))))))"--worker-id", default=None, help="Worker ID ())))))))for worker mode)")
    parser.add_argument())))))))"--api-key", default=None, help="API key ())))))))for worker mode)")
    parser.add_argument())))))))"--security-config", default="./test_security_config.json", 
    help="Path to security configuration file")
    parser.add_argument())))))))"--num-workers", type=int, default=2, 
    help="Number of workers to start ())))))))for all mode)")
    parser.add_argument())))))))"--run-time", type=int, default=60, 
    help="How long to run the test in seconds ())))))))for all mode)")
    parser.add_argument())))))))"--generate-keys", action="store_true", 
    help="Generate new API keys for testing")
    parser.add_argument())))))))"--disable-health-monitor", action="store_true",
    help="Disable the health monitoring system")
    parser.add_argument())))))))"--disable-advanced-scheduler", action="store_true",
    help="Disable the advanced task scheduler")
    parser.add_argument())))))))"--disable-load-balancer", action="store_true",
    help="Disable the adaptive load balancer")
    parser.add_argument())))))))"--max-tasks-per-worker", type=int, default=1,
    help="Maximum number of tasks per worker ())))))))for advanced scheduler)")
    parser.add_argument())))))))"--num-test-tasks", type=int, default=5,
    help="Number of test tasks to submit in test mode")
    
    # Add a new argument to skip security tests (default True because security is OUT OF SCOPE)
    parser.add_argument())))))))"--skip-security-tests", action="store_true", default=True,
    help="Skip security-related tests (default: True as security is OUT OF SCOPE)")
    
    args = parser.parse_args()))))))))
    
    # Print notice about security being out of scope
    logger.info())))))))f"NOTICE: Security and authentication features have been marked as OUT OF SCOPE.")
    logger.info())))))))f"Please refer to SECURITY_DEPRECATED.md for more information.")
    logger.info())))))))f"Security-related tests will be skipped."))
    
    if args.mode == "coordinator":
        # Run coordinator only (security features out of scope)
        logger.info())))))))f"NOTICE: Security features are OUT OF SCOPE. Running coordinator...")
        coordinator_process, worker_api_key = await run_coordinator())))))))
        args.db_path,
        args.host,
        args.port,
        args.security_config,
        False,  # No key generation needed (security is OUT OF SCOPE)
        args.disable_health_monitor,
        args.disable_advanced_scheduler,
        args.disable_load_balancer,
        args.max_tasks_per_worker
        )
            
            await log_process_output())))))))coordinator_process, "Coordinator")
        
    elif args.mode == "worker":
        # Run worker only (security features out of scope)
        if not args.coordinator:
            logger.error())))))))"Coordinator URL must be provided in worker mode")
        return
        
        logger.info())))))))f"NOTICE: Security features are OUT OF SCOPE. Running worker...")
        
        # No API key needed (security is OUT OF SCOPE)
        worker_process = await run_worker())))))))
        args.coordinator,
        args.db_path,
        args.worker_id,
        None  # No API key needed (security is OUT OF SCOPE)
        )
        
        await log_process_output())))))))worker_process, "Worker")
        
    elif args.mode == "all":
        # Run full test with coordinator and workers
        await run_all_tests())))))))
        args.db_path,
        args.host,
        args.port,
        args.num_workers,
        args.run_time,
        args.security_config,
        args.disable_health_monitor,
        args.disable_advanced_scheduler,
        args.disable_load_balancer,
        args.max_tasks_per_worker,
        args.num_test_tasks
        )

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal())))))))signal.SIGINT, lambda sig, frame: sys.exit())))))))0))
    
    anyio.run())))))))main())))))))))