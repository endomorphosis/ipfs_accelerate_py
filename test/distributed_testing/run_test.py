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
                       """Run coordinator process."""
                       logger.info())))))))f"Starting coordinator with database at {}}}db_path}")
    
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
    
        process = await asyncio.create_subprocess_exec())))))))
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        )
    
        logger.info())))))))f"Coordinator process started with PID {}}}process.pid}")
    
    # Wait for a moment to ensure keys are generated and displayed
    if generate_keys:
        await asyncio.sleep())))))))1)
        
        # Extract API keys for workers
        lines = []],,]
        while len())))))))lines) < 50:  # Capture enough lines to find API keys
            try:
                line = await asyncio.wait_for())))))))process.stdout.readline())))))))), 0.1)
                if not line:
                break
                    
                decoded_line = line.decode())))))))).strip()))))))))
                lines.append())))))))decoded_line)
                if "=== WORKER API KEY ===" in decoded_line:
                    # Extract worker API key
                    api_key_line_index = lines.index())))))))decoded_line)
                    if len())))))))lines) > api_key_line_index + 1:
                    return process, lines[]],,api_key_line_index + 1]
            except asyncio.TimeoutError:
                    break
                
    # Return process and None for API key if not found
                return process, None
:
async def run_worker())))))))coordinator_url, db_path=None, worker_id=None, api_key=None):
    """Run worker process."""
    logger.info())))))))f"Starting worker connecting to {}}}coordinator_url}")
    
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
    
        process = await asyncio.create_subprocess_exec())))))))
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
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
    
            await asyncio.gather())))))))
            read_stream())))))))process.stdout, f"[]],,{}}}name} stdout]"),
            read_stream())))))))process.stderr, f"[]],,{}}}name} stderr]")
            )

async def submit_test_tasks())))))))coordinator_url, num_tasks=3, api_key=None):
    """Submit test tasks to the coordinator."""
    import aiohttp
    
    logger.info())))))))f"Submitting {}}}num_tasks} test tasks to coordinator")
    
    # Try to get API key from security config if not provided:
    if not api_key:
        try:
            import json
            with open())))))))"./test_security_config.json", "r") as f:
                security_config = json.load())))))))f)
                # Get the first admin API key
                for key, details in security_config.get())))))))"api_keys", {}}}}).items())))))))):
                    if "admin" in details.get())))))))"roles", []],,]):
                        api_key = key
                        logger.info())))))))f"Found admin API key from config: {}}}api_key[]],,:8]}...")
                    break
        except Exception as e:
            logger.warning())))))))f"Could not load API key from security config: {}}}str())))))))e)}")
    
    # Prepare headers with authentication if we have an API key
    headers = {}}}}:
    if api_key:
        headers[]],,"X-API-Key"] = api_key
        logger.info())))))))"Using API key authentication for task submission")
    else:
        logger.warning())))))))"No API key available. Task submission may fail due to authentication.")
    
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
                    await asyncio.sleep())))))))1)
                    
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
        # Start coordinator with security
        logger.info())))))))"Starting coordinator with security...")
        coordinator_process, worker_api_key = await run_coordinator())))))))
        db_path, host, port, security_config, generate_keys=True,
        disable_health_monitor=disable_health_monitor,
        disable_advanced_scheduler=disable_advanced_scheduler,
        disable_load_balancer=disable_load_balancer,
        max_tasks_per_worker=max_tasks_per_worker
        )
        
        # Wait for coordinator to start
        logger.info())))))))"Waiting for coordinator to start...")
        await asyncio.sleep())))))))5)
        
        if not worker_api_key:
            logger.warning())))))))"No worker API key found. Workers might fail to authenticate.")
        else:
            logger.info())))))))f"Worker API key obtained for testing: {}}}worker_api_key[]],,:8]}...")
        
        # Start multiple workers
        for i in range())))))))num_workers):
            worker_process = await run_worker())))))))coordinator_url, db_path, api_key=worker_api_key)
            worker_processes.append())))))))worker_process)
            
            # Wait a bit between starting workers
            await asyncio.sleep())))))))1)
        
        # Wait for workers to connect and authenticate
            logger.info())))))))"Waiting for workers to connect and authenticate...")
            await asyncio.sleep())))))))8)
        
        # Try to extract admin API key from security config
            admin_api_key = None
        try:
            import json
            with open())))))))security_config, "r") as f:
                config = json.load())))))))f)
                # Find admin API key
                for key, details in config.get())))))))"api_keys", {}}}}).items())))))))):
                    if "admin" in details.get())))))))"roles", []],,]):
                        admin_api_key = key
                        logger.info())))))))f"Found admin API key for task submission: {}}}admin_api_key[]],,:8]}...")
                    break
        except Exception as e:
            logger.warning())))))))f"Failed to extract admin API key from config: {}}}str())))))))e)}")
        
        # Submit test tasks with admin API key
            await submit_test_tasks())))))))coordinator_url, num_tasks=num_test_tasks, api_key=admin_api_key)
        
        # Log output for all processes
            log_tasks = []],,]
            log_tasks.append())))))))asyncio.create_task())))))))log_process_output())))))))coordinator_process, "Coordinator")))
        for i, proc in enumerate())))))))worker_processes):
            log_tasks.append())))))))asyncio.create_task())))))))log_process_output())))))))proc, f"Worker-{}}}i+1}")))
        
        # Run for specified time
            logger.info())))))))f"Running test for {}}}run_time} seconds...")
            await asyncio.sleep())))))))run_time)
        
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
    
    args = parser.parse_args()))))))))
    
    if args.mode == "coordinator":
        # Run coordinator only
        coordinator_process, worker_api_key = await run_coordinator())))))))
        args.db_path,
        args.host,
        args.port,
        args.security_config,
        args.generate_keys,
        args.disable_health_monitor,
        args.disable_advanced_scheduler,
        args.disable_load_balancer,
        args.max_tasks_per_worker
        )
        
        if worker_api_key and args.generate_keys:
            logger.info())))))))f"Generated worker API key: {}}}worker_api_key}")
            
            await log_process_output())))))))coordinator_process, "Coordinator")
        
    elif args.mode == "worker":
        # Run worker only
        if not args.coordinator:
            logger.error())))))))"Coordinator URL must be provided in worker mode")
        return
            
        worker_process = await run_worker())))))))
        args.coordinator,
        args.db_path,
        args.worker_id,
        args.api_key
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
    
    asyncio.run())))))))main())))))))))