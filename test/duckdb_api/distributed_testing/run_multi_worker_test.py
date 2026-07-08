#!/usr/bin/env python3
"""
Test script to run multiple worker clients in parallel.

This script starts multiple worker clients that connect to the coordinator server
and tests the worker reconnection system with concurrent workers.
"""

import os
import sys
import time
import uuid
import random
import logging
import argparse
import subprocess
import threading
import signal
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("multi_worker_test")


class WorkerProcess:
    """Class for managing a worker client process."""
    
    def __init__(self, worker_id: str, coordinator_host: str, coordinator_port: int,
                 heartbeat_interval: float = 5.0, reconnect_delay: float = 1.0,
                 max_reconnect_delay: float = 60.0, log_level: str = "INFO",
                 simulate_disconnect: float = 0.0):
        """
        Initialize a worker process.
        
        Args:
            worker_id: ID of the worker
            coordinator_host: Coordinator hostname
            coordinator_port: Coordinator port
            heartbeat_interval: Heartbeat interval in seconds
            reconnect_delay: Initial reconnection delay in seconds
            max_reconnect_delay: Maximum reconnection delay in seconds
            log_level: Logging level
            simulate_disconnect: Seconds after which to simulate a disconnect (0 to disable)
        """
        self.worker_id = worker_id
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.log_level = log_level
        self.simulate_disconnect = simulate_disconnect
        
        # Process state
        self.process = None
        self.log_file = None
        self.is_running = False
    
    def start(self):
        """Start the worker process."""
        if self.is_running:
            logger.warning(f"Worker {self.worker_id} is already running")
            return
        
        # Open log file
        log_filename = f"worker_{self.worker_id}.log"
        self.log_file = open(log_filename, "w")
        
        # Build command
        script_path = str(Path(__file__).parent / "run_worker_client.py")
        command = [
            sys.executable,
            script_path,
            "--worker-id", self.worker_id,
            "--coordinator-host", self.coordinator_host,
            "--coordinator-port", str(self.coordinator_port),
            "--heartbeat-interval", str(self.heartbeat_interval),
            "--reconnect-delay", str(self.reconnect_delay),
            "--max-reconnect-delay", str(self.max_reconnect_delay),
            "--log-level", self.log_level
        ]
        
        # Add simulate disconnect if enabled
        if self.simulate_disconnect > 0:
            command.extend(["--simulate-disconnect", str(self.simulate_disconnect)])
        
        # Start process
        self.process = subprocess.Popen(
            command,
            stdout=self.log_file,
            stderr=subprocess.STDOUT
        )
        
        self.is_running = True
        logger.info(f"Started worker {self.worker_id}")
    
    def stop(self):
        """Stop the worker process."""
        if not self.is_running:
            logger.warning(f"Worker {self.worker_id} is not running")
            return
        
        # Terminate process
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Worker {self.worker_id} did not terminate, killing")
                self.process.kill()
                self.process.wait()
        
        # Close log file
        if self.log_file:
            self.log_file.close()
        
        self.is_running = False
        logger.info(f"Stopped worker {self.worker_id}")
    
    def is_alive(self):
        """Check if the worker process is still alive."""
        if not self.is_running:
            return False
        
        if self.process:
            return self.process.poll() is None
        
        return False


class CoordinatorProcess:
    """Class for managing the coordinator server process."""
    
    def __init__(self, host: str = "localhost", port: int = 8765,
                 log_level: str = "INFO", demo_tasks: int = 0):
        """
        Initialize the coordinator process.
        
        Args:
            host: Hostname to bind to
            port: Port to listen on
            log_level: Logging level
            demo_tasks: Number of demo tasks to submit (0 to disable)
        """
        self.host = host
        self.port = port
        self.log_level = log_level
        self.demo_tasks = demo_tasks
        
        # Process state
        self.process = None
        self.log_file = None
        self.is_running = False
    
    def start(self):
        """Start the coordinator process."""
        if self.is_running:
            logger.warning("Coordinator is already running")
            return
        
        # Open log file
        log_filename = "coordinator.log"
        self.log_file = open(log_filename, "w")
        
        # Build command
        script_path = str(Path(__file__).parent / "run_coordinator_server.py")
        command = [
            sys.executable,
            script_path,
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", self.log_level
        ]
        
        # Add demo tasks if enabled
        if self.demo_tasks > 0:
            command.extend(["--demo-tasks", str(self.demo_tasks)])
        
        # Start process
        self.process = subprocess.Popen(
            command,
            stdout=self.log_file,
            stderr=subprocess.STDOUT
        )
        
        self.is_running = True
        logger.info(f"Started coordinator on {self.host}:{self.port}")
        
        # Wait for server to start
        time.sleep(2)
    
    def stop(self):
        """Stop the coordinator process."""
        if not self.is_running:
            logger.warning("Coordinator is not running")
            return
        
        # Terminate process
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Coordinator did not terminate, killing")
                self.process.kill()
                self.process.wait()
        
        # Close log file
        if self.log_file:
            self.log_file.close()
        
        self.is_running = False
        logger.info("Stopped coordinator")
    
    def is_alive(self):
        """Check if the coordinator process is still alive."""
        if not self.is_running:
            return False
        
        if self.process:
            return self.process.poll() is None
        
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Multi-Worker Test")
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Number of worker clients to run"
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
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--random-disconnects",
        action="store_true",
        help="Randomly disconnect workers"
    )
    
    parser.add_argument(
        "--demo-tasks",
        type=int,
        default=10,
        help="Number of demo tasks to submit (0 to disable)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration of the test in seconds"
    )
    
    return parser.parse_args()


def run_multi_worker_test(args):
    """
    Run the multi-worker test.
    
    Args:
        args: Command line arguments
    """
    # Start coordinator
    coordinator = CoordinatorProcess(
        host=args.coordinator_host,
        port=args.coordinator_port,
        log_level=args.log_level,
        demo_tasks=args.demo_tasks
    )
    coordinator.start()
    
    # Create workers
    workers = []
    for i in range(args.num_workers):
        worker_id = f"worker-{i+1:04d}"
        
        # Set simulate disconnect time if random disconnects enabled
        simulate_disconnect = 0.0
        if args.random_disconnects:
            # Random time between 10 and 40 seconds
            simulate_disconnect = random.uniform(10, 40)
        
        worker = WorkerProcess(
            worker_id=worker_id,
            coordinator_host=args.coordinator_host,
            coordinator_port=args.coordinator_port,
            heartbeat_interval=args.heartbeat_interval,
            reconnect_delay=args.reconnect_delay,
            max_reconnect_delay=args.max_reconnect_delay,
            log_level=args.log_level,
            simulate_disconnect=simulate_disconnect
        )
        workers.append(worker)
    
    # Start workers
    for worker in workers:
        worker.start()
        # Stagger worker starts
        time.sleep(0.5)
    
    try:
        # Wait for duration or until interrupted
        logger.info(f"Test running for {args.duration} seconds...")
        
        # Set up random reconnection thread if needed
        if args.random_disconnects:
            should_stop = threading.Event()
            reconnect_thread = threading.Thread(
                target=random_reconnect_thread,
                args=(workers, should_stop),
                daemon=True
            )
            reconnect_thread.start()
        
        # Wait for duration
        start_time = time.time()
        while time.time() - start_time < args.duration:
            try:
                time.sleep(1)
                
                # Check if processes are still alive
                if not coordinator.is_alive():
                    logger.error("Coordinator process died, restarting")
                    coordinator.stop()
                    coordinator.start()
                
                for worker in workers:
                    if not worker.is_alive():
                        logger.warning(f"Worker {worker.worker_id} died, restarting")
                        worker.stop()
                        worker.start()
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user, stopping test...")
                break
        
        # Stop random reconnect thread if running
        if args.random_disconnects:
            should_stop.set()
            reconnect_thread.join(timeout=2)
        
    finally:
        # Stop all workers
        for worker in workers:
            worker.stop()
        
        # Stop coordinator
        coordinator.stop()
        
        logger.info("Multi-worker test completed")


def random_reconnect_thread(workers: List[WorkerProcess], should_stop: threading.Event):
    """
    Thread that randomly forces workers to reconnect.
    
    Args:
        workers: List of worker processes
        should_stop: Event to signal thread to stop
    """
    while not should_stop.is_set():
        try:
            # Sleep for a random interval (5-15 seconds)
            sleep_time = random.uniform(5, 15)
            if should_stop.wait(sleep_time):
                break
            
            # Select a random worker
            worker = random.choice(workers)
            
            # Restart worker to simulate a more severe disconnection
            logger.info(f"Randomly restarting worker {worker.worker_id}")
            worker.stop()
            time.sleep(1)
            worker.start()
            
        except Exception as e:
            logger.error(f"Error in random reconnect thread: {e}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Set up signal handlers
    def signal_handler(sig, frame):
        logger.info("Received signal, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run test
    run_multi_worker_test(args)


if __name__ == "__main__":
    main()