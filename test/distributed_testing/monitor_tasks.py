#!/usr/bin/env python3
"""
Distributed Testing Framework - Task Monitor

This script monitors the status of tasks in the distributed testing framework.
It provides a command-line interface for viewing tasks, workers, and system metrics.

Usage:
    python monitor_tasks.py --status all
    python monitor_tasks.py --status pending
    python monitor_tasks.py --status running
    python monitor_tasks.py --status completed
    python monitor_tasks.py --status failed
    python monitor_tasks.py --workers
    python monitor_tasks.py --metrics
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

import aiohttp
import numpy as np
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def get_request(url: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Make a GET request to the coordinator API.
    
    Args:
        url: URL to request
        api_key: API key for authentication
        
    Returns:
        JSON response as a dictionary
    """
    headers = {"X-API-Key": api_key} if api_key else {}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                error_text = await resp.text()
                raise ValueError(f"API request failed: {resp.status} - {error_text}")

async def load_api_key(security_config_path: str) -> Optional[str]:
    """
    Load API key from security configuration.
    
    Args:
        security_config_path: Path to security configuration file
        
    Returns:
        API key or None if not found
    """
    if not os.path.exists(security_config_path):
        logger.warning(f"Security configuration file not found: {security_config_path}")
        return None
    
    try:
        with open(security_config_path, 'r') as f:
            config = json.load(f)
            
        # Look for an admin API key
        for key, info in config.get("api_keys", {}).items():
            if "admin" in info.get("roles", []):
                logger.info(f"Found admin API key in security configuration")
                return key
            
        # If no admin key, look for any key with worker role
        for key, info in config.get("api_keys", {}).items():
            if "worker" in info.get("roles", []):
                logger.info(f"Found worker API key in security configuration")
                return key
                
        logger.warning("No suitable API key found in security configuration")
        return None
        
    except Exception as e:
        logger.error(f"Error loading security configuration: {str(e)}")
        return None

async def monitor_tasks(
    coordinator_url: str,
    api_key: Optional[str] = None,
    status: str = "all",
    limit: int = 100,
    format: str = "table"
) -> None:
    """
    Monitor tasks in the distributed testing framework.
    
    Args:
        coordinator_url: URL of the coordinator server
        api_key: API key for authentication
        status: Task status to filter by (all, pending, running, completed, failed)
        limit: Maximum number of tasks to display
        format: Output format (table, json)
    """
    # Add API URL if not present
    if not coordinator_url.endswith("/api/tasks"):
        if not coordinator_url.endswith("/"):
            coordinator_url += "/"
        coordinator_url += "api/tasks"
    
    # Add status filter if needed
    if status != "all":
        coordinator_url += f"?status={status}"
    
    # Add limit parameter
    if "?" in coordinator_url:
        coordinator_url += f"&limit={limit}"
    else:
        coordinator_url += f"?limit={limit}"
    
    try:
        # Get tasks from coordinator
        response = await get_request(coordinator_url, api_key)
        tasks = response.get("tasks", [])
        
        if not tasks:
            logger.info(f"No tasks found with status '{status}'")
            return
        
        if format == "json":
            # Print JSON format
            print(json.dumps(tasks, indent=2))
        else:
            # Print table format
            if HAS_TABULATE:
                # Prepare table data
                table_data = []
                for task in tasks:
                    # Format timestamps
                    created = task.get("created", "")
                    started = task.get("started", "")
                    ended = task.get("ended", "")
                    
                    if created:
                        try:
                            created_dt = datetime.fromisoformat(created)
                            created = created_dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    
                    if started:
                        try:
                            started_dt = datetime.fromisoformat(started)
                            started = started_dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    
                    if ended:
                        try:
                            ended_dt = datetime.fromisoformat(ended)
                            ended = ended_dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    
                    # Format requirements
                    requirements = task.get("requirements", {})
                    req_str = ""
                    if requirements:
                        if "hardware" in requirements:
                            req_str += f"hw:{','.join(requirements['hardware'])} "
                        if "min_memory_gb" in requirements:
                            req_str += f"mem:{requirements['min_memory_gb']}GB "
                    
                    # Add to table
                    table_data.append([
                        task.get("task_id", "")[:8] + "...",
                        task.get("type", ""),
                        task.get("status", ""),
                        task.get("priority", ""),
                        created,
                        started,
                        ended,
                        task.get("worker_id", ""),
                        req_str
                    ])
                
                # Print table
                headers = ["ID", "Type", "Status", "Priority", "Created", "Started", "Ended", "Worker", "Requirements"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
                print(f"\nTotal tasks: {len(tasks)}")
            else:
                # Print simple format (no tabulate)
                for task in tasks:
                    print(f"Task ID: {task.get('task_id')}")
                    print(f"  Type: {task.get('type')}")
                    print(f"  Status: {task.get('status')}")
                    print(f"  Priority: {task.get('priority')}")
                    print(f"  Created: {task.get('created')}")
                    print(f"  Started: {task.get('started')}")
                    print(f"  Ended: {task.get('ended')}")
                    print(f"  Worker: {task.get('worker_id')}")
                    print("  Requirements:")
                    for k, v in task.get("requirements", {}).items():
                        print(f"    {k}: {v}")
                    print()
                
                print(f"\nTotal tasks: {len(tasks)}")
        
    except ValueError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Error monitoring tasks: {str(e)}")

async def monitor_workers(
    coordinator_url: str,
    api_key: Optional[str] = None,
    format: str = "table"
) -> None:
    """
    Monitor workers in the distributed testing framework.
    
    Args:
        coordinator_url: URL of the coordinator server
        api_key: API key for authentication
        format: Output format (table, json)
    """
    # Add API URL if not present
    if not coordinator_url.endswith("/api/workers"):
        if not coordinator_url.endswith("/"):
            coordinator_url += "/"
        coordinator_url += "api/workers"
    
    try:
        # Get workers from coordinator
        response = await get_request(coordinator_url, api_key)
        workers = response.get("workers", [])
        
        if not workers:
            logger.info("No workers found")
            return
        
        if format == "json":
            # Print JSON format
            print(json.dumps(workers, indent=2))
        else:
            # Print table format
            if HAS_TABULATE:
                # Prepare table data
                table_data = []
                for worker in workers:
                    # Format last heartbeat
                    last_heartbeat = worker.get("last_heartbeat", "")
                    if last_heartbeat:
                        try:
                            heartbeat_dt = datetime.fromisoformat(last_heartbeat)
                            last_heartbeat = heartbeat_dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    
                    # Format capabilities summary
                    capabilities = worker.get("capabilities_summary", {})
                    cap_str = ""
                    if capabilities:
                        if "hardware" in capabilities:
                            cap_str += f"hw:{','.join(capabilities['hardware'])} "
                        if "gpu" in capabilities:
                            gpu = capabilities["gpu"]
                            if isinstance(gpu, dict):
                                cap_str += f"gpu:{gpu.get('name', '')} "
                        if "memory_gb" in capabilities:
                            cap_str += f"mem:{capabilities['memory_gb']}GB "
                    
                    # Add to table
                    table_data.append([
                        worker.get("worker_id", "")[:8] + "...",
                        worker.get("hostname", ""),
                        worker.get("status", ""),
                        last_heartbeat,
                        cap_str
                    ])
                
                # Print table
                headers = ["ID", "Hostname", "Status", "Last Heartbeat", "Capabilities"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
                print(f"\nTotal workers: {len(workers)}")
            else:
                # Print simple format (no tabulate)
                for worker in workers:
                    print(f"Worker ID: {worker.get('worker_id')}")
                    print(f"  Hostname: {worker.get('hostname')}")
                    print(f"  Status: {worker.get('status')}")
                    print(f"  Last Heartbeat: {worker.get('last_heartbeat')}")
                    print("  Capabilities:")
                    for k, v in worker.get("capabilities_summary", {}).items():
                        print(f"    {k}: {v}")
                    print()
                
                print(f"\nTotal workers: {len(workers)}")
        
    except ValueError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Error monitoring workers: {str(e)}")

async def monitor_system_metrics(
    coordinator_url: str,
    api_key: Optional[str] = None,
    format: str = "table"
) -> None:
    """
    Monitor system metrics in the distributed testing framework.
    
    Args:
        coordinator_url: URL of the coordinator server
        api_key: API key for authentication
        format: Output format (table, json)
    """
    # Add API URL if not present
    if not coordinator_url.endswith("/status"):
        if not coordinator_url.endswith("/"):
            coordinator_url += "/"
        coordinator_url += "status"
    
    try:
        # Get system metrics from coordinator
        metrics = await get_request(coordinator_url, api_key)
        
        if format == "json":
            # Print JSON format
            print(json.dumps(metrics, indent=2))
        else:
            # Print formatted metrics
            print("System Status:")
            print(f"  Version: {metrics.get('version', 'unknown')}")
            print(f"  Uptime: {metrics.get('uptime', 0)} seconds")
            
            workers = metrics.get("workers", {})
            tasks = metrics.get("tasks", {})
            
            print("\nWorkers:")
            print(f"  Total: {workers.get('total', 0)}")
            print(f"  Active: {workers.get('active', 0)}")
            print(f"  Idle: {workers.get('idle', 0)}")
            print(f"  Offline: {workers.get('offline', 0)}")
            
            print("\nTasks:")
            print(f"  Total: {tasks.get('total', 0)}")
            print(f"  Pending: {tasks.get('pending', 0)}")
            print(f"  Running: {tasks.get('running', 0)}")
            print(f"  Completed: {tasks.get('completed', 0)}")
            print(f"  Failed: {tasks.get('failed', 0)}")
            
            # Calculate task completion rate
            completion_rate = 0
            if tasks.get("total", 0) > 0:
                completion_rate = (tasks.get("completed", 0) / tasks.get("total", 0)) * 100
            
            print(f"\nCompletion Rate: {completion_rate:.2f}%")
        
    except ValueError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Error monitoring system metrics: {str(e)}")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Monitor tasks in the distributed testing framework")
    parser.add_argument("--coordinator", default="http://localhost:8080",
                       help="URL of coordinator server")
    parser.add_argument("--api-key",
                       help="API key for authentication")
    parser.add_argument("--security-config", default="./security_config.json",
                       help="Path to security configuration file")
    parser.add_argument("--format", choices=["table", "json"], default="table",
                       help="Output format")
    parser.add_argument("--limit", type=int, default=100,
                       help="Maximum number of tasks to display")
    
    # Add mutually exclusive group for monitoring options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", choices=["all", "pending", "running", "completed", "failed"], default="all",
                     help="Task status to filter by")
    group.add_argument("--workers", action="store_true",
                     help="Show workers instead of tasks")
    group.add_argument("--metrics", action="store_true",
                     help="Show system metrics")
    
    args = parser.parse_args()
    
    # If API key not provided, try to load from security config
    api_key = args.api_key
    if not api_key:
        api_key = await load_api_key(args.security_config)
        if not api_key:
            logger.warning("No API key provided or found in security config. Authentication may fail.")
    
    try:
        # Monitor based on option
        if args.workers:
            await monitor_workers(args.coordinator, api_key, args.format)
        elif args.metrics:
            await monitor_system_metrics(args.coordinator, api_key, args.format)
        else:
            await monitor_tasks(args.coordinator, api_key, args.status, args.limit, args.format)
        
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())