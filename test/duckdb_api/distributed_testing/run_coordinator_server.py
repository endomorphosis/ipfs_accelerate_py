#!/usr/bin/env python3
"""
Run the Coordinator WebSocket Server as a standalone process.

This script starts the coordinator WebSocket server for manual testing
and development of worker reconnection functionality.
"""

import os
import sys
import anyio
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

# Import circuit breaker integration
try:
    from duckdb_api.distributed_testing.coordinator_integration import (
        integrate_circuit_breaker_with_coordinator
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    logger = logging.getLogger("run_coordinator_server")
    logger.warning("Circuit Breaker integration not available. Advanced fault tolerance features disabled.")

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
    
    parser.add_argument(
        "--circuit-breaker",
        action="store_true",
        help="Enable circuit breaker pattern for enhanced fault tolerance"
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
        await anyio.sleep(0.5)
    
    logger.info(f"Submitted {num_tasks} demo tasks")


async def run_coordinator_with_demo(host: str, port: int, num_demo_tasks: int, enable_circuit_breaker: bool = False):
    """
    Run the coordinator server with optional demo tasks.
    
    Args:
        host: Hostname to bind to
        port: Port to listen on
        num_demo_tasks: Number of demo tasks to submit (0 to disable)
        enable_circuit_breaker: Whether to enable circuit breaker pattern
    """
    # Create server instance
    server = CoordinatorWebSocketServer(host, port)
    
    # Set up signal handlers
    stop_event = anyio.Event()
    
    async def _signal_listener():
        async with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
            async for _ in signals:
                logger.info("Received signal, shutting down...")
                stop_event.set()
                break
    
    try:
        # Integrate circuit breaker pattern if enabled
        if enable_circuit_breaker and CIRCUIT_BREAKER_AVAILABLE:
            logger.info("Integrating circuit breaker pattern with coordinator...")
            success = integrate_circuit_breaker_with_coordinator(server)
            if success:
                logger.info("Circuit breaker pattern successfully integrated with coordinator")
            else:
                logger.warning("Failed to integrate circuit breaker pattern with coordinator")
        elif enable_circuit_breaker and not CIRCUIT_BREAKER_AVAILABLE:
            logger.warning("Circuit breaker pattern requested but not available. Advanced fault tolerance features disabled.")
        
        async with anyio.create_task_group() as tg:
            tg.start_soon(_signal_listener)

            # Start server
            tg.start_soon(server.start)
            
            # Submit demo tasks if requested (after a short delay)
            if num_demo_tasks > 0:
                tg.start_soon(submit_demo_tasks_after_delay, server, num_demo_tasks, 5.0)
            
            # Wait for stop signal
            await stop_event.wait()
            
            # Stop server
            await server.stop()
            tg.cancel_scope.cancel()
        
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
    await anyio.sleep(delay)
    await submit_demo_tasks(server, num_tasks)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run coordinator server
    try:
        anyio.run(run_coordinator_with_demo(
            host=args.host,
            port=args.port,
            num_demo_tasks=args.demo_tasks,
            enable_circuit_breaker=args.circuit_breaker
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