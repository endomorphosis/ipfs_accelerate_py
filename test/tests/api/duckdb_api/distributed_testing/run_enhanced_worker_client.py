#!/usr/bin/env python3
"""
Run a Worker Client with the Enhanced Worker Reconnection System.

This script starts a worker client that connects to the coordinator server
and uses the enhanced worker reconnection system for reliable communication
with advanced features.
"""

import os
import sys
import time
import json
import uuid
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import enhanced worker reconnection module
from data.duckdb.distributed_testing.worker_reconnection_enhancements import (
    EnhancedWorkerReconnectionManager, EnhancedWorkerReconnectionPlugin,
    create_enhanced_worker_reconnection_plugin, MessagePriority
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("enhanced_worker_client")


class EnhancedWorkerClient:
    """Worker client that uses the enhanced worker reconnection system."""
    
    def __init__(self, worker_id: str, coordinator_url: str, api_key: Optional[str] = None,
                 capabilities: Dict[str, Any] = None, heartbeat_interval: float = 5.0,
                 reconnect_delay: float = 1.0, max_reconnect_delay: float = 60.0,
                 enable_compression: bool = True, compression_level: int = 6,
                 enable_priority_queue: bool = True, adaptive_parameters: bool = True):
        """
        Initialize the enhanced worker client.
        
        Args:
            worker_id: Unique identifier for this worker
            coordinator_url: URL of the coordinator server
            api_key: API key for authentication
            capabilities: Worker capabilities to report
            heartbeat_interval: Heartbeat interval in seconds
            reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay in seconds
            enable_compression: Whether to enable message compression
            compression_level: Compression level (0-9)
            enable_priority_queue: Whether to enable priority-based message queue
            adaptive_parameters: Whether to enable adaptive connection parameters
        """
        self.worker_id = worker_id
        self.coordinator_url = coordinator_url
        self.api_key = api_key
        self.capabilities = capabilities or {
            "cpu": 4,
            "memory": 8,
            "type": "enhanced_worker"
        }
        
        # Configure reconnection
        self.reconnection_config = {
            "heartbeat_interval": heartbeat_interval,
            "initial_reconnect_delay": reconnect_delay,
            "max_reconnect_delay": max_reconnect_delay,
            "reconnect_jitter": 0.1
        }
        
        # Enhancement options
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        self.enable_priority_queue = enable_priority_queue
        self.adaptive_parameters = adaptive_parameters
        
        # Create reconnection plugin
        self.reconnection_plugin = None
        
        # Active tasks
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_threads = {}
        
        # Control flags
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Metrics reporting
        self.metrics_thread = None
    
    def start(self):
        """Start the enhanced worker client."""
        if self.is_running:
            logger.warning("Enhanced worker client is already running")
            return
        
        logger.info(f"Starting enhanced worker client {self.worker_id}")
        
        # Create enhanced reconnection plugin
        self.reconnection_plugin = create_enhanced_worker_reconnection_plugin(
            worker=self,
            api_key=self.api_key,
            enable_compression=self.enable_compression,
            compression_level=self.compression_level,
            enable_priority_queue=self.enable_priority_queue,
            adaptive_parameters=self.adaptive_parameters
        )
        
        # Set running flag
        self.is_running = True
        
        # Start status reporting thread
        threading.Thread(
            target=self._status_reporting_loop,
            daemon=True
        ).start()
        
        # Start metrics reporting thread
        self.metrics_thread = threading.Thread(
            target=self._metrics_reporting_loop,
            daemon=True
        )
        self.metrics_thread.start()
    
    def stop(self):
        """Stop the enhanced worker client."""
        if not self.is_running:
            logger.warning("Enhanced worker client is not running")
            return
        
        logger.info(f"Stopping enhanced worker client {self.worker_id}")
        
        # Set stop event
        self.stop_event.set()
        
        # Stop reconnection plugin
        if self.reconnection_plugin:
            self.reconnection_plugin.stop()
        
        # Set running flag
        self.is_running = False
    
    def execute_task(self, task_id: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.
        
        Args:
            task_id: ID of the task
            task_config: Task configuration
            
        Returns:
            Task result
        """
        logger.info(f"Executing task {task_id}")
        
        # Add to active tasks
        self.active_tasks[task_id] = {
            "config": task_config,
            "status": "running",
            "started_at": datetime.now().isoformat()
        }
        
        # Start task thread
        thread = threading.Thread(
            target=self._task_thread,
            args=(task_id, task_config),
            daemon=True
        )
        self.task_threads[task_id] = thread
        thread.start()
        
        # Return placeholder result (actual result will be submitted by task thread)
        return {"status": "running"}
    
    def _task_thread(self, task_id: str, task_config: Dict[str, Any]):
        """
        Task execution thread.
        
        Args:
            task_id: ID of the task
            task_config: Task configuration
        """
        try:
            # Check if we have a checkpoint to resume from
            checkpoint_data = None
            if self.reconnection_plugin:
                checkpoint_data = self.reconnection_plugin.get_latest_checkpoint(task_id)
            
            # Resume from checkpoint if available
            start_iteration = 0
            if checkpoint_data and "iteration" in checkpoint_data:
                start_iteration = checkpoint_data["iteration"]
                logger.info(f"Resuming task {task_id} from checkpoint (iteration {start_iteration})")
            
            # Get task parameters
            task_type = task_config.get("type", "unknown")
            task_name = task_config.get("name", task_id)
            iterations = task_config.get("iterations", 10)
            sleep_time = task_config.get("sleep", 0.5)
            
            # Execute task
            for i in range(start_iteration, iterations):
                # Check if we should stop
                if self.stop_event.is_set():
                    logger.info(f"Task {task_id} interrupted due to worker shutdown")
                    return
                
                # Calculate progress
                progress = (i + 1) / iterations * 100
                
                # Update task state
                if self.reconnection_plugin:
                    self.reconnection_plugin.update_task_state(task_id, {
                        "progress": progress,
                        "current_iteration": i + 1,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Create checkpoint every 2 iterations
                if self.reconnection_plugin and (i + 1) % 2 == 0:
                    checkpoint_data = {
                        "iteration": i + 1,
                        "progress": progress,
                        "timestamp": datetime.now().isoformat()
                    }
                    checkpoint_id = self.reconnection_plugin.create_checkpoint(
                        task_id, checkpoint_data
                    )
                    logger.debug(f"Created checkpoint {checkpoint_id} for task {task_id} (iteration {i+1})")
                
                # Do simulated work
                logger.info(f"Task {task_id} ({task_name}): iteration {i+1}/{iterations} - {progress:.1f}%")
                time.sleep(sleep_time)
            
            # Task completed
            result = {
                "status": "completed",
                "iterations": iterations,
                "task_type": task_type,
                "task_name": task_name,
                "completed_at": datetime.now().isoformat()
            }
            
            # Update task state
            if self.reconnection_plugin:
                self.reconnection_plugin.update_task_state(task_id, {
                    "status": "completed",
                    "progress": 100.0,
                    "completed_at": datetime.now().isoformat()
                })
            
            # Submit result
            if self.reconnection_plugin:
                self.reconnection_plugin.submit_task_result(task_id, result)
            
            # Move to completed tasks
            with threading.RLock():
                if task_id in self.active_tasks:
                    self.completed_tasks[task_id] = {
                        **self.active_tasks[task_id],
                        "status": "completed",
                        "result": result,
                        "completed_at": datetime.now().isoformat()
                    }
                    del self.active_tasks[task_id]
            
            logger.info(f"Task {task_id} ({task_name}) completed successfully")
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Report error
            error_info = {
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            
            # Submit error result
            if self.reconnection_plugin:
                self.reconnection_plugin.submit_task_result(task_id, {}, error_info)
            
            # Move to completed tasks with error
            with threading.RLock():
                if task_id in self.active_tasks:
                    self.completed_tasks[task_id] = {
                        **self.active_tasks[task_id],
                        "status": "failed",
                        "error": error_info,
                        "completed_at": datetime.now().isoformat()
                    }
                    del self.active_tasks[task_id]
        
        finally:
            # Remove task thread
            with threading.RLock():
                if task_id in self.task_threads:
                    del self.task_threads[task_id]
    
    def _status_reporting_loop(self):
        """Status reporting thread function."""
        while not self.stop_event.is_set():
            try:
                # Report connection status
                connection_state = "DISCONNECTED"
                if self.reconnection_plugin:
                    if self.reconnection_plugin.is_connected():
                        connection_state = "CONNECTED"
                
                # Get metrics
                metrics = {}
                if self.reconnection_plugin:
                    try:
                        metrics = self.reconnection_plugin.get_performance_metrics()
                    except:
                        pass
                
                # Report status
                if metrics:
                    logger.info(
                        f"Worker {self.worker_id} status: {connection_state}, "
                        f"Active tasks: {len(self.active_tasks)}, "
                        f"Completed tasks: {len(self.completed_tasks)}, "
                        f"Latency: {metrics.get('average_message_latency', 0):.1f}ms, "
                        f"Messages: {metrics.get('message_count', 0)}, "
                        f"Reconnections: {metrics.get('reconnections', 0)}"
                    )
                else:
                    logger.info(
                        f"Worker {self.worker_id} status: {connection_state}, "
                        f"Active tasks: {len(self.active_tasks)}, "
                        f"Completed tasks: {len(self.completed_tasks)}"
                    )
                
            except Exception as e:
                logger.error(f"Error in status reporting: {e}")
            
            # Sleep until next report or stop
            self.stop_event.wait(10.0)
    
    def _metrics_reporting_loop(self):
        """Metrics reporting thread function."""
        # Wait a bit before first report
        time.sleep(30)
        
        while not self.stop_event.is_set():
            try:
                # Get metrics
                if self.reconnection_plugin:
                    metrics = self.reconnection_plugin.get_performance_metrics()
                    
                    # Print detailed metrics report
                    logger.info(f"====== Performance Metrics for {self.worker_id} ======")
                    logger.info(f"Message count: {metrics.get('message_count', 0)}")
                    logger.info(f"Average message size: {metrics.get('average_message_size', 0):.1f} bytes")
                    logger.info(f"Average message latency: {metrics.get('average_message_latency', 0):.1f} ms")
                    logger.info(f"Message errors: {metrics.get('message_errors', 0)}")
                    logger.info(f"Reconnections: {metrics.get('reconnections', 0)}")
                    logger.info(f"Average reconnection duration: {metrics.get('average_reconnection_duration', 0):.1f} s")
                    logger.info(f"Task execution count: {metrics.get('task_execution_count', 0)}")
                    logger.info(f"Task success rate: {metrics.get('task_success_rate', 0):.2f}")
                    logger.info(f"Average task duration: {metrics.get('average_task_duration', 0):.1f} s")
                    logger.info(f"Checkpoints created: {metrics.get('checkpoints_created', 0)}")
                    logger.info(f"Checkpoints resumed: {metrics.get('checkpoints_resumed', 0)}")
                    logger.info(f"Compression ratio: {metrics.get('compression_ratio', 1.0):.2f}")
                    logger.info(f"Uptime: {metrics.get('uptime', 0):.1f} s")
                    logger.info("=============================================")
            
            except Exception as e:
                logger.error(f"Error in metrics reporting: {e}")
            
            # Sleep until next report or stop
            self.stop_event.wait(60.0)  # Report once per minute


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Enhanced Worker Client")
    
    parser.add_argument(
        "--worker-id",
        default="",
        help="Worker ID (leave empty for auto-generated ID)"
    )
    
    parser.add_argument(
        "--coordinator-host",
        default="localhost",
        help="Coordinator hostname"
    )
    
    parser.add_argument(
        "--coordinator-port",
        type=int,
        default=8765,
        help="Coordinator port"
    )
    
    parser.add_argument(
        "--api-key",
        default="test-api-key",
        help="API key for authentication"
    )
    
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=5.0,
        help="Heartbeat interval in seconds"
    )
    
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=1.0,
        help="Initial reconnection delay in seconds"
    )
    
    parser.add_argument(
        "--max-reconnect-delay",
        type=float,
        default=60.0,
        help="Maximum reconnection delay in seconds"
    )
    
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable message compression"
    )
    
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(0, 10),
        default=6,
        help="Compression level (0-9, higher is more compression but slower)"
    )
    
    parser.add_argument(
        "--no-priority-queue",
        action="store_true",
        help="Disable priority-based message queue"
    )
    
    parser.add_argument(
        "--no-adaptive-parameters",
        action="store_true",
        help="Disable adaptive connection parameters"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--simulate-disconnect",
        type=float,
        default=0,
        help="Simulate disconnect after specified seconds (0 to disable)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Generate worker ID if not provided
    worker_id = args.worker_id
    if not worker_id:
        worker_id = f"enhanced-worker-{uuid.uuid4()}"
    
    # Build coordinator URL
    coordinator_url = f"ws://{args.coordinator_host}:{args.coordinator_port}/api/v1/worker/{{worker_id}}/ws"
    
    # Create enhanced worker client
    worker = EnhancedWorkerClient(
        worker_id=worker_id,
        coordinator_url=coordinator_url,
        api_key=args.api_key,
        heartbeat_interval=args.heartbeat_interval,
        reconnect_delay=args.reconnect_delay,
        max_reconnect_delay=args.max_reconnect_delay,
        enable_compression=not args.no_compression,
        compression_level=args.compression_level,
        enable_priority_queue=not args.no_priority_queue,
        adaptive_parameters=not args.no_adaptive_parameters
    )
    
    try:
        # Start worker
        worker.start()
        
        # Simulate disconnect if requested
        if args.simulate_disconnect > 0:
            def simulate_disconnect():
                logger.info(f"Simulating disconnect after {args.simulate_disconnect} seconds")
                time.sleep(args.simulate_disconnect)
                
                # Force reconnection
                if worker.reconnection_plugin:
                    logger.info("Forcing reconnection")
                    worker.reconnection_plugin.force_reconnect()
            
            threading.Thread(target=simulate_disconnect, daemon=True).start()
        
        # Wait for shutdown
        logger.info(f"Enhanced worker {worker_id} started, press Ctrl+C to stop")
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Interrupted by user, shutting down...")
                break
    
    finally:
        # Stop worker
        worker.stop()
        logger.info(f"Enhanced worker {worker_id} stopped")


if __name__ == "__main__":
    main()