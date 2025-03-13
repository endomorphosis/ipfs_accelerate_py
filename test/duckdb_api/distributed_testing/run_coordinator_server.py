#!/usr/bin/env python3
"""
Run the Coordinator WebSocket Server as a standalone process.

This script starts the coordinator WebSocket server for manual testing
and development of worker reconnection functionality.
"""

import os
import sys
import asyncio
import argparse
import logging
import signal
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import coordinator server
from duckdb_api.distributed_testing.coordinator_websocket_server import (
    CoordinatorWebSocketServer, run_server
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("run_coordinator_server")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Coordinator WebSocket Server")
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="Hostname to bind to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--demo-tasks",
        type=int,
        default=0,
        help="Number of demo tasks to submit (0 to disable)"
    )
    
    return parser.parse_args()


async def submit_demo_tasks(server: CoordinatorWebSocketServer, num_tasks: int):
    """
    Submit demo tasks to the server.
    
    Args:
        server: The coordinator server instance
        num_tasks: Number of tasks to submit
    """
    logger.info(f"Submitting {num_tasks} demo tasks...")
    
    for i in range(num_tasks):
        task_config = {
            "type": "demo_task",
            "name": f"Task-{i+1}",
            "iterations": (i % 5) + 5,  # 5-9 iterations
            "sleep": 0.5
        }
        
        task_id = await server.submit_task(task_config)
        logger.info(f"Submitted task {task_id}: {task_config['name']}")
        
        # Short delay between submissions
        await asyncio.sleep(0.5)
    
    logger.info(f"Submitted {num_tasks} demo tasks")


async def run_coordinator_with_demo(host: str, port: int, num_demo_tasks: int):
    """
    Run the coordinator server with optional demo tasks.
    
    Args:
        host: Hostname to bind to
        port: Port to listen on
        num_demo_tasks: Number of demo tasks to submit (0 to disable)
    """
    # Create server instance
    server = CoordinatorWebSocketServer(host, port)
    
    # Set up signal handlers
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    
    def signal_handler():
        logger.info("Received signal, shutting down...")
        stop_event.set()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        # Start server
        start_task = asyncio.create_task(server.start())
        
        # Submit demo tasks if requested (after a short delay)
        if num_demo_tasks > 0:
            asyncio.create_task(
                submit_demo_tasks_after_delay(server, num_demo_tasks, delay=5.0)
            )
        
        # Wait for stop signal
        await stop_event.wait()
        
        # Stop server
        await server.stop()
        
        # Wait for server to stop
        await start_task
        
    except Exception as e:
        logger.error(f"Error running coordinator server: {e}")
        import traceback
        logger.debug(traceback.format_exc())


async def submit_demo_tasks_after_delay(
    server: CoordinatorWebSocketServer,
    num_tasks: int,
    delay: float = 5.0
):
    """
    Submit demo tasks after a delay.
    
    Args:
        server: The coordinator server instance
        num_tasks: Number of tasks to submit
        delay: Delay in seconds before submitting tasks
    """
    await asyncio.sleep(delay)
    await submit_demo_tasks(server, num_tasks)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run coordinator server
    try:
        asyncio.run(run_coordinator_with_demo(
            host=args.host,
            port=args.port,
            num_demo_tasks=args.demo_tasks
        ))
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")
    except Exception as e:
        logger.error(f"Error running coordinator server: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()