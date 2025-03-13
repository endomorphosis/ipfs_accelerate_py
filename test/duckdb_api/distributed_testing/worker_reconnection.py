#!/usr/bin/env python3
"""
Worker Reconnection System for Distributed Testing Framework

This module provides a robust reconnection system for workers that get disconnected
from the coordinator. It implements automatic state synchronization, task recovery,
and progressive retry mechanisms to ensure reliable operation in unstable network
environments.

Key features:
1. Automatic worker reconnection with exponential backoff
2. State synchronization after reconnection
3. Task recovery and resumption from checkpoints
4. Connection quality monitoring and adaptation
5. Heartbeat-based monitoring with failure detection

This component complements the hardware-aware fault tolerance system by focusing
specifically on network-related failures and recovery.
"""

import os
import sys
import json
import time
import logging
import threading
import queue
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
import traceback
import websocket
import requests
from enum import Enum, auto
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("worker_reconnection")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from hardware-aware fault tolerance
from duckdb_api.distributed_testing.hardware_aware_fault_tolerance import (
    RecoveryAction, RecoveryStrategy, FailureType
)


class ConnectionState(Enum):
    """Connection states for worker-coordinator communication."""
    CONNECTED = auto()       # Successfully connected and authenticated
    CONNECTING = auto()      # Actively trying to connect
    DISCONNECTED = auto()    # Not connected, but trying to reconnect
    FAILED = auto()          # Connection failed, manual intervention required
    SHUTDOWN = auto()        # Gracefully shut down


@dataclass
class ConnectionStats:
    """Statistics about connection quality and reliability."""
    connected_time: float = 0.0              # Total time connected (seconds)
    disconnected_time: float = 0.0           # Total time disconnected (seconds)
    connection_attempts: int = 0             # Number of connection attempts
    successful_connections: int = 0          # Number of successful connections
    failed_connections: int = 0              # Number of failed connection attempts
    last_connected: Optional[datetime] = None  # Last time successfully connected
    last_disconnected: Optional[datetime] = None  # Last time disconnected
    latency_samples: List[float] = field(default_factory=list)  # Heartbeat latencies (ms)
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency from samples."""
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)
    
    @property
    def connection_stability(self) -> float:
        """Calculate connection stability (0.0-1.0)."""
        total_time = self.connected_time + self.disconnected_time
        if total_time == 0:
            return 1.0
        return self.connected_time / total_time
    
    @property
    def connection_success_rate(self) -> float:
        """Calculate connection success rate (0.0-1.0)."""
        total_attempts = self.connection_attempts
        if total_attempts == 0:
            return 1.0
        return self.successful_connections / total_attempts
    
    def add_latency_sample(self, latency: float):
        """Add a latency sample and maintain a reasonable history."""
        self.latency_samples.append(latency)
        # Keep at most 100 samples
        if len(self.latency_samples) > 100:
            self.latency_samples = self.latency_samples[-100:]


class WorkerReconnectionManager:
    """
    Manager for worker reconnection and state synchronization.
    
    This class handles automatic reconnection of workers to the coordinator
    when network interruptions occur, with proper state synchronization and
    task recovery.
    """
    
    def __init__(self, worker_id: str, coordinator_url: str, 
                 api_key: Optional[str] = None, capabilities: Optional[Dict] = None,
                 task_executor: Optional[Callable] = None):
        """
        Initialize the worker reconnection manager.
        
        Args:
            worker_id: Unique identifier for this worker
            coordinator_url: URL of the coordinator server
            api_key: API key for authentication
            capabilities: Worker capabilities to report
            task_executor: Callback for executing tasks
        """
        self.worker_id = worker_id
        self.coordinator_url = coordinator_url
        self.api_key = api_key
        self.capabilities = capabilities or {}
        self.task_executor = task_executor
        
        # Connection state
        self.connection_state = ConnectionState.DISCONNECTED
        self.ws_conn = None
        self.http_session = requests.Session()
        
        # Connection statistics
        self.connection_stats = ConnectionStats()
        
        # Task state
        self.current_tasks = {}  # task_id -> task_info
        self.task_results = {}   # task_id -> result
        self.task_states = {}    # task_id -> state
        
        # Checkpoints
        self.checkpoints = {}    # task_id -> checkpoint_data
        
        # Message queue for outgoing messages
        self.message_queue = queue.Queue()
        
        # Threads
        self.ws_thread = None
        self.reconnect_thread = None
        self.heartbeat_thread = None
        self.message_sender_thread = None
        
        # Thread control
        self.stop_event = threading.Event()
        
        # Configuration
        self.config = {
            "heartbeat_interval": 5.0,       # Heartbeat interval in seconds
            "initial_reconnect_delay": 1.0,  # Initial reconnection delay in seconds
            "max_reconnect_delay": 60.0,     # Maximum reconnection delay in seconds
            "reconnect_jitter": 0.2,         # Jitter factor for reconnection timing
            "heartbeat_timeout": 15.0,       # Heartbeat timeout in seconds
            "max_reconnect_attempts": 0,     # Max reconnect attempts (0 = unlimited)
            "checkpoint_interval": 300,      # Checkpoint interval in seconds
            "message_retry_count": 3,        # Number of times to retry sending a message
            "connection_health_threshold": 0.7,  # Connection stability threshold for health
            "ws_ping_interval": 30,          # WebSocket ping interval in seconds
            "ws_ping_timeout": 10,           # WebSocket ping timeout in seconds
            "state_sync_batch_size": 100,    # Max number of records to sync at once
        }
        
        # State synchronization
        self.last_sync_time = datetime.now()
        self.sync_lock = threading.Lock()
        self.task_lock = threading.Lock()
        
        # Heartbeat tracking
        self.last_heartbeat_sent = None
        self.last_heartbeat_received = None
        self.heartbeat_sequence = 0
        
        # Recovery tracking
        self.reconnect_attempts = 0
        self.current_reconnect_delay = self.config["initial_reconnect_delay"]
        
        logger.info(f"Worker reconnection manager initialized for worker {worker_id}")
    
    def configure(self, config_updates: Dict[str, Any]):
        """
        Update the configuration of the reconnection manager.
        
        Args:
            config_updates: Dictionary with configuration updates
        """
        self.config.update(config_updates)
        logger.info(f"Reconnection manager configuration updated: {config_updates}")
    
    def start(self):
        """Start the worker reconnection manager."""
        # Reset stop event
        self.stop_event.clear()
        
        # Start message sender thread
        self.message_sender_thread = threading.Thread(
            target=self._message_sender_loop,
            daemon=True
        )
        self.message_sender_thread.start()
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()
        
        # Start initial connection
        self._start_connection()
        
        logger.info(f"Worker reconnection manager started for worker {self.worker_id}")
    
    def stop(self):
        """Stop the worker reconnection manager."""
        # Set stop event to terminate all threads
        self.stop_event.set()
        
        # Close WebSocket connection
        self._close_connection(ConnectionState.SHUTDOWN)
        
        # Wait for threads to terminate
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=5.0)
        
        if self.reconnect_thread and self.reconnect_thread.is_alive():
            self.reconnect_thread.join(timeout=5.0)
            
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5.0)
            
        if self.message_sender_thread and self.message_sender_thread.is_alive():
            self.message_sender_thread.join(timeout=5.0)
        
        logger.info(f"Worker reconnection manager stopped for worker {self.worker_id}")
    
    def is_connected(self) -> bool:
        """
        Check if the worker is currently connected to the coordinator.
        
        Returns:
            True if connected, False otherwise
        """
        return self.connection_state == ConnectionState.CONNECTED
    
    def get_connection_state(self) -> ConnectionState:
        """
        Get the current connection state.
        
        Returns:
            Current ConnectionState enum value
        """
        return self.connection_state
    
    def get_connection_stats(self) -> ConnectionStats:
        """
        Get connection statistics.
        
        Returns:
            ConnectionStats object with connection statistics
        """
        return self.connection_stats
    
    def reset_connection_stats(self):
        """Reset connection statistics."""
        self.connection_stats = ConnectionStats()
    
    def submit_task_result(self, task_id: str, result: Dict[str, Any], error: Optional[Dict] = None):
        """
        Submit a task result to the coordinator.
        
        Args:
            task_id: ID of the task
            result: Task result data
            error: Optional error information
        """
        with self.task_lock:
            # Store result for retry if needed
            self.task_results[task_id] = {
                "result": result,
                "error": error,
                "timestamp": datetime.now().isoformat()
            }
            
            # Remove from current tasks
            if task_id in self.current_tasks:
                del self.current_tasks[task_id]
        
        # Create result message
        message = {
            "type": "task_result",
            "task_id": task_id,
            "worker_id": self.worker_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        if error:
            message["error"] = error
        
        # Queue message for sending
        self._queue_message(message)
    
    def create_checkpoint(self, task_id: str, checkpoint_data: Dict[str, Any]) -> str:
        """
        Create a checkpoint for a task.
        
        Args:
            task_id: ID of the task
            checkpoint_data: Checkpoint data
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Store checkpoint
        self.checkpoints[checkpoint_id] = {
            "task_id": task_id,
            "timestamp": timestamp.isoformat(),
            "data": checkpoint_data
        }
        
        # Update task state with checkpoint information
        with self.task_lock:
            if task_id not in self.task_states:
                self.task_states[task_id] = {}
            
            self.task_states[task_id]["last_checkpoint"] = {
                "checkpoint_id": checkpoint_id,
                "timestamp": timestamp.isoformat()
            }
        
        # Send checkpoint to coordinator
        message = {
            "type": "checkpoint",
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "worker_id": self.worker_id,
            "timestamp": timestamp.isoformat()
        }
        
        # We don't include the full checkpoint data in the message
        # to avoid large messages - the coordinator can request it if needed
        
        # Queue message for sending
        self._queue_message(message)
        
        logger.debug(f"Created checkpoint {checkpoint_id} for task {task_id}")
        return checkpoint_id
    
    def get_latest_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest checkpoint for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Latest checkpoint data or None if no checkpoint exists
        """
        # Find all checkpoints for this task
        task_checkpoints = {
            cp_id: cp
            for cp_id, cp in self.checkpoints.items()
            if cp["task_id"] == task_id
        }
        
        if not task_checkpoints:
            return None
        
        # Find the latest checkpoint
        latest_checkpoint_id = max(
            task_checkpoints.keys(),
            key=lambda cp_id: task_checkpoints[cp_id]["timestamp"]
        )
        
        return self.checkpoints[latest_checkpoint_id]["data"]
    
    def update_task_state(self, task_id: str, state_updates: Dict[str, Any]):
        """
        Update the state of a task.
        
        Args:
            task_id: ID of the task
            state_updates: Dictionary with state updates
        """
        with self.task_lock:
            if task_id not in self.task_states:
                self.task_states[task_id] = {}
            
            # Update state
            self.task_states[task_id].update(state_updates)
        
        # Send state update to coordinator
        message = {
            "type": "task_state",
            "task_id": task_id,
            "worker_id": self.worker_id,
            "state": state_updates,
            "timestamp": datetime.now().isoformat()
        }
        
        # Queue message for sending
        self._queue_message(message)
    
    def force_reconnect(self):
        """Force a reconnection to the coordinator."""
        logger.info("Forcing reconnection to coordinator")
        self._close_connection(ConnectionState.DISCONNECTED)
        self._start_connection()
    
    def _start_connection(self):
        """Start a connection to the coordinator."""
        if self.connection_state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
            return  # Already connected or connecting
        
        # Update state
        self.connection_state = ConnectionState.CONNECTING
        
        # Start WebSocket connection in a new thread
        self.ws_thread = threading.Thread(
            target=self._ws_connection_loop,
            daemon=True
        )
        self.ws_thread.start()
    
    def _ws_connection_loop(self):
        """WebSocket connection thread function."""
        try:
            # Update connection stats
            self.connection_stats.connection_attempts += 1
            
            # Calculate WebSocket URL
            ws_url = self._get_ws_url()
            
            # Create WebSocket connection
            logger.info(f"Connecting to coordinator at {ws_url}")
            
            # Create WebSocket connection with ping/pong protocol enabled
            self.ws_conn = websocket.WebSocketApp(
                ws_url,
                header=self._get_auth_headers(),
                on_open=self._on_ws_open,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close,
                on_ping=self._on_ws_ping,
                on_pong=self._on_ws_pong
            )
            
            # Run WebSocket connection (this blocks until connection closed)
            self.ws_conn.run_forever(
                ping_interval=self.config["ws_ping_interval"],
                ping_timeout=self.config["ws_ping_timeout"]
            )
            
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            logger.debug(traceback.format_exc())
            self._handle_connection_failure()
    
    def _on_ws_open(self, ws):
        """WebSocket on_open callback."""
        logger.info(f"WebSocket connection established for worker {self.worker_id}")
        
        # Update connection state
        self.connection_state = ConnectionState.CONNECTED
        
        # Update connection stats
        self.connection_stats.successful_connections += 1
        self.connection_stats.last_connected = datetime.now()
        
        # Reset reconnection delay
        self.current_reconnect_delay = self.config["initial_reconnect_delay"]
        self.reconnect_attempts = 0
        
        # Send registration message
        self._send_registration()
        
        # Synchronize state
        self._synchronize_state()
    
    def _on_ws_message(self, ws, message):
        """WebSocket on_message callback."""
        try:
            # Parse message
            data = json.loads(message)
            
            # Process message based on type
            msg_type = data.get("type", "")
            
            if msg_type == "heartbeat":
                self._handle_heartbeat(data)
            elif msg_type == "task_assignment":
                self._handle_task_assignment(data)
            elif msg_type == "checkpoint_request":
                self._handle_checkpoint_request(data)
            elif msg_type == "state_request":
                self._handle_state_request(data)
            elif msg_type == "cancel_task":
                self._handle_cancel_task(data)
            elif msg_type == "pause_task":
                self._handle_pause_task(data)
            elif msg_type == "resume_task":
                self._handle_resume_task(data)
            elif msg_type == "error":
                self._handle_error_message(data)
            elif msg_type == "welcome":
                self._handle_welcome(data)
            elif msg_type == "registration_ack":
                self._handle_registration_ack(data)
            # Add handlers for other message types below:
            elif msg_type == "task_result_ack":
                self._handle_task_result_ack(data)
            elif msg_type == "task_state_ack":
                self._handle_task_state_ack(data)
            elif msg_type == "checkpoint_ack":
                self._handle_checkpoint_ack(data)
            else:
                logger.warning(f"Received unknown message type: {msg_type}")
            
        except json.JSONDecodeError:
            logger.error(f"Received invalid JSON: {message[:100]}...")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.debug(traceback.format_exc())
    
    def _on_ws_error(self, ws, error):
        """WebSocket on_error callback."""
        logger.error(f"WebSocket error: {error}")
        # Connection failure will be handled by on_close
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """WebSocket on_close callback."""
        logger.warning(f"WebSocket connection closed: {close_status_code}, {close_msg}")
        
        # Update connection stats
        self.connection_stats.last_disconnected = datetime.now()
        
        # Handle reconnection if not shutting down
        if not self.stop_event.is_set() and self.connection_state != ConnectionState.SHUTDOWN:
            self._handle_connection_failure()
    
    def _on_ws_ping(self, ws, message):
        """WebSocket on_ping callback."""
        logger.debug(f"Received WebSocket ping")
    
    def _on_ws_pong(self, ws, message):
        """WebSocket on_pong callback."""
        logger.debug(f"Received WebSocket pong")
    
    def _send_registration(self):
        """Send worker registration message."""
        message = {
            "type": "registration",
            "worker_id": self.worker_id,
            "capabilities": self.capabilities,
            "timestamp": datetime.now().isoformat()
        }
        
        # Queue message for sending
        self._queue_message(message)
    
    def _synchronize_state(self):
        """Synchronize state with coordinator after reconnection."""
        with self.sync_lock:
            logger.info("Synchronizing state with coordinator")
            
            # Send pending task results
            with self.task_lock:
                for task_id, result_data in self.task_results.items():
                    message = {
                        "type": "task_result",
                        "task_id": task_id,
                        "worker_id": self.worker_id,
                        "result": result_data["result"],
                        "timestamp": result_data["timestamp"]
                    }
                    
                    if "error" in result_data and result_data["error"]:
                        message["error"] = result_data["error"]
                    
                    self._queue_message(message)
                
                # Send current task states
                for task_id, state in self.task_states.items():
                    if task_id in self.current_tasks:
                        message = {
                            "type": "task_state",
                            "task_id": task_id,
                            "worker_id": self.worker_id,
                            "state": state,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        self._queue_message(message)
            
            # Note: Checkpoints are only sent on request to avoid large messages
            
            # Update sync time
            self.last_sync_time = datetime.now()
            
            logger.info("State synchronization completed")
    
    def _handle_connection_failure(self):
        """Handle a connection failure with reconnection logic."""
        # Update connection state
        old_state = self.connection_state
        self.connection_state = ConnectionState.DISCONNECTED
        
        # Increment reconnect attempts
        self.reconnect_attempts += 1
        
        # Check if we've reached the maximum number of reconnect attempts
        if (self.config["max_reconnect_attempts"] > 0 and 
            self.reconnect_attempts >= self.config["max_reconnect_attempts"]):
            logger.error(
                f"Maximum reconnect attempts ({self.config['max_reconnect_attempts']}) reached. "
                f"Giving up."
            )
            self.connection_state = ConnectionState.FAILED
            return
        
        # Calculate reconnect delay with exponential backoff and jitter
        delay = self._calculate_reconnect_delay()
        
        logger.info(f"Will attempt reconnection in {delay:.1f} seconds (attempt {self.reconnect_attempts})")
        
        # Start reconnect thread
        self.reconnect_thread = threading.Thread(
            target=self._delayed_reconnect,
            args=(delay,),
            daemon=True
        )
        self.reconnect_thread.start()
    
    def _delayed_reconnect(self, delay: float):
        """
        Perform a delayed reconnection attempt.
        
        Args:
            delay: Delay in seconds before reconnecting
        """
        # Wait for the specified delay, or until stop event is set
        if self.stop_event.wait(delay):
            return  # Stop event was set, do not reconnect
        
        # Attempt reconnection
        self._start_connection()
    
    def _calculate_reconnect_delay(self) -> float:
        """
        Calculate delay for reconnection using exponential backoff with jitter.
        
        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * 2^(attempts-1)
        base_delay = self.config["initial_reconnect_delay"]
        max_delay = self.config["max_reconnect_delay"]
        jitter_factor = self.config["reconnect_jitter"]
        
        # Calculate delay with exponential backoff
        delay = base_delay * (2 ** (self.reconnect_attempts - 1))
        
        # Apply jitter
        jitter = random.uniform(-jitter_factor * delay, jitter_factor * delay)
        delay += jitter
        
        # Cap at max delay
        delay = min(delay, max_delay)
        
        # Save for next time
        self.current_reconnect_delay = delay
        
        return delay
    
    def _heartbeat_loop(self):
        """Heartbeat thread function."""
        while not self.stop_event.is_set():
            try:
                # Only send heartbeats if connected
                if self.connection_state == ConnectionState.CONNECTED:
                    self._send_heartbeat()
                    
                    # Check if we've received a heartbeat recently
                    if self.last_heartbeat_received is not None:
                        time_since_heartbeat = (datetime.now() - self.last_heartbeat_received).total_seconds()
                        
                        if time_since_heartbeat > self.config["heartbeat_timeout"]:
                            logger.warning(
                                f"No heartbeat received from coordinator for {time_since_heartbeat:.1f}s, "
                                f"forcing reconnection"
                            )
                            self.force_reconnect()
                
                # Track connection statistics
                self._update_connection_stats()
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                logger.debug(traceback.format_exc())
            
            # Wait for next heartbeat interval
            self.stop_event.wait(self.config["heartbeat_interval"])
    
    def _send_heartbeat(self):
        """Send a heartbeat message to the coordinator."""
        # Generate sequence number for latency calculation
        self.heartbeat_sequence += 1
        
        # Create heartbeat message
        message = {
            "type": "heartbeat",
            "worker_id": self.worker_id,
            "sequence": self.heartbeat_sequence,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add current tasks info
        with self.task_lock:
            message["active_tasks"] = list(self.current_tasks.keys())
        
        # Queue message for sending
        self._queue_message(message)
        
        # Record sent time
        self.last_heartbeat_sent = datetime.now()
    
    def _handle_heartbeat(self, data: Dict[str, Any]):
        """
        Handle a heartbeat message from coordinator.
        
        Args:
            data: Heartbeat message data
        """
        # Record received time
        self.last_heartbeat_received = datetime.now()
        
        # Calculate round-trip time if we sent the heartbeat
        if "response_to" in data:
            sequence = data.get("response_to")
            if sequence == self.heartbeat_sequence and self.last_heartbeat_sent:
                latency = (datetime.now() - self.last_heartbeat_sent).total_seconds() * 1000  # ms
                self.connection_stats.add_latency_sample(latency)
                logger.debug(f"Heartbeat latency: {latency:.1f}ms")
        
        # Process pending actions in the heartbeat
        if "actions" in data:
            for action in data["actions"]:
                action_type = action.get("type")
                
                if action_type == "sync_state":
                    self._synchronize_state()
                elif action_type == "resend_task_results":
                    task_ids = action.get("task_ids", [])
                    self._resend_task_results(task_ids)
                elif action_type == "cancel_tasks":
                    task_ids = action.get("task_ids", [])
                    for task_id in task_ids:
                        self._handle_cancel_task({"task_id": task_id})
                elif action_type == "reset_connection_stats":
                    self.reset_connection_stats()
    
    def _update_connection_stats(self):
        """Update connection statistics."""
        now = datetime.now()
        
        # Update connected/disconnected time
        if self.connection_state == ConnectionState.CONNECTED:
            if self.connection_stats.last_disconnected is not None:
                self.connection_stats.disconnected_time += (
                    self.connection_stats.last_connected - self.connection_stats.last_disconnected
                ).total_seconds()
                self.connection_stats.last_disconnected = None
        else:
            if self.connection_stats.last_connected is not None:
                self.connection_stats.connected_time += (
                    self.connection_stats.last_disconnected - self.connection_stats.last_connected
                ).total_seconds()
                self.connection_stats.last_connected = None
    
    def _handle_task_assignment(self, data: Dict[str, Any]):
        """
        Handle a task assignment message from coordinator.
        
        Args:
            data: Task assignment message data
        """
        task_id = data.get("task_id")
        task_config = data.get("config", {})
        
        if not task_id:
            logger.warning("Received task assignment without task_id")
            return
        
        logger.info(f"Received task assignment for task {task_id}")
        
        # Store task
        with self.task_lock:
            self.current_tasks[task_id] = {
                "config": task_config,
                "assigned_at": datetime.now().isoformat(),
                "status": "assigned"
            }
        
        # Execute task if we have an executor
        if self.task_executor:
            try:
                # Execute task in a separate thread
                threading.Thread(
                    target=self._execute_task,
                    args=(task_id, task_config),
                    daemon=True
                ).start()
            except Exception as e:
                logger.error(f"Error starting task {task_id}: {e}")
                error_info = {
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
                self.submit_task_result(task_id, {}, error_info)
        else:
            logger.warning(f"No task executor available for task {task_id}")
            self.submit_task_result(
                task_id, 
                {},
                {"message": "No task executor available"}
            )
    
    def _execute_task(self, task_id: str, task_config: Dict[str, Any]):
        """
        Execute a task using the task executor.
        
        Args:
            task_id: ID of the task
            task_config: Task configuration
        """
        try:
            logger.info(f"Executing task {task_id}")
            
            # Update task status
            with self.task_lock:
                if task_id in self.current_tasks:
                    self.current_tasks[task_id]["status"] = "running"
                    self.current_tasks[task_id]["started_at"] = datetime.now().isoformat()
            
            # Update task state
            self.update_task_state(task_id, {
                "status": "running",
                "started_at": datetime.now().isoformat()
            })
            
            # Execute task
            result = self.task_executor(task_id, task_config)
            
            # Submit result
            self.submit_task_result(task_id, result)
            
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            error_info = {
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            self.submit_task_result(task_id, {}, error_info)
    
    def _handle_checkpoint_request(self, data: Dict[str, Any]):
        """
        Handle a checkpoint request message from coordinator.
        
        Args:
            data: Checkpoint request message data
        """
        task_id = data.get("task_id")
        checkpoint_id = data.get("checkpoint_id")
        
        if not task_id:
            logger.warning("Received checkpoint request without task_id")
            return
        
        logger.info(f"Received checkpoint request for task {task_id}")
        
        # Find checkpoint
        if checkpoint_id:
            # Specific checkpoint requested
            if checkpoint_id in self.checkpoints:
                checkpoint = self.checkpoints[checkpoint_id]
                self._send_checkpoint_data(checkpoint_id, task_id, checkpoint["data"])
            else:
                logger.warning(f"Requested checkpoint {checkpoint_id} not found")
                self._send_checkpoint_error(task_id, f"Checkpoint {checkpoint_id} not found")
        else:
            # Latest checkpoint requested
            checkpoint_data = self.get_latest_checkpoint(task_id)
            if checkpoint_data:
                # Find checkpoint ID
                for cp_id, cp in self.checkpoints.items():
                    if cp["task_id"] == task_id and cp["data"] == checkpoint_data:
                        self._send_checkpoint_data(cp_id, task_id, checkpoint_data)
                        break
            else:
                logger.warning(f"No checkpoint found for task {task_id}")
                self._send_checkpoint_error(task_id, f"No checkpoint found for task {task_id}")
    
    def _send_checkpoint_data(self, checkpoint_id: str, task_id: str, data: Dict[str, Any]):
        """
        Send checkpoint data to coordinator.
        
        Args:
            checkpoint_id: ID of the checkpoint
            task_id: ID of the task
            data: Checkpoint data
        """
        message = {
            "type": "checkpoint_data",
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "worker_id": self.worker_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Queue message for sending
        self._queue_message(message)
    
    def _send_checkpoint_error(self, task_id: str, error_message: str):
        """
        Send checkpoint error to coordinator.
        
        Args:
            task_id: ID of the task
            error_message: Error message
        """
        message = {
            "type": "checkpoint_error",
            "task_id": task_id,
            "worker_id": self.worker_id,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Queue message for sending
        self._queue_message(message)
    
    def _handle_state_request(self, data: Dict[str, Any]):
        """
        Handle a state request message from coordinator.
        
        Args:
            data: State request message data
        """
        task_id = data.get("task_id")
        
        if task_id:
            # Specific task state requested
            with self.task_lock:
                if task_id in self.task_states:
                    state = self.task_states[task_id]
                    self._send_task_state(task_id, state)
                else:
                    logger.warning(f"Requested state for unknown task {task_id}")
                    self._send_state_error(task_id, f"No state found for task {task_id}")
        else:
            # Full state requested
            self._synchronize_state()
    
    def _send_task_state(self, task_id: str, state: Dict[str, Any]):
        """
        Send task state to coordinator.
        
        Args:
            task_id: ID of the task
            state: Task state
        """
        message = {
            "type": "task_state",
            "task_id": task_id,
            "worker_id": self.worker_id,
            "state": state,
            "timestamp": datetime.now().isoformat()
        }
        
        # Queue message for sending
        self._queue_message(message)
    
    def _send_state_error(self, task_id: str, error_message: str):
        """
        Send state error to coordinator.
        
        Args:
            task_id: ID of the task
            error_message: Error message
        """
        message = {
            "type": "state_error",
            "task_id": task_id,
            "worker_id": self.worker_id,
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        # Queue message for sending
        self._queue_message(message)
    
    def _handle_cancel_task(self, data: Dict[str, Any]):
        """
        Handle a cancel task message from coordinator.
        
        Args:
            data: Cancel task message data
        """
        task_id = data.get("task_id")
        
        if not task_id:
            logger.warning("Received cancel task message without task_id")
            return
        
        logger.info(f"Received cancel task message for task {task_id}")
        
        # Remove task from current tasks
        with self.task_lock:
            if task_id in self.current_tasks:
                del self.current_tasks[task_id]
                
                # Send cancellation acknowledgement
                message = {
                    "type": "task_cancelled",
                    "task_id": task_id,
                    "worker_id": self.worker_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Queue message for sending
                self._queue_message(message)
            else:
                logger.warning(f"Requested cancellation of unknown task {task_id}")
    
    def _handle_pause_task(self, data: Dict[str, Any]):
        """
        Handle a pause task message from coordinator.
        
        Args:
            data: Pause task message data
        """
        task_id = data.get("task_id")
        
        if not task_id:
            logger.warning("Received pause task message without task_id")
            return
        
        logger.info(f"Received pause task message for task {task_id}")
        
        # Update task status
        with self.task_lock:
            if task_id in self.current_tasks:
                self.current_tasks[task_id]["status"] = "paused"
                
                # Create checkpoint before pausing
                if task_id in self.task_states:
                    self.create_checkpoint(task_id, self.task_states[task_id])
                
                # Send pause acknowledgement
                message = {
                    "type": "task_paused",
                    "task_id": task_id,
                    "worker_id": self.worker_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Queue message for sending
                self._queue_message(message)
            else:
                logger.warning(f"Requested pause of unknown task {task_id}")
    
    def _handle_resume_task(self, data: Dict[str, Any]):
        """
        Handle a resume task message from coordinator.
        
        Args:
            data: Resume task message data
        """
        task_id = data.get("task_id")
        
        if not task_id:
            logger.warning("Received resume task message without task_id")
            return
        
        logger.info(f"Received resume task message for task {task_id}")
        
        # Update task status
        with self.task_lock:
            if task_id in self.current_tasks:
                self.current_tasks[task_id]["status"] = "running"
                
                # Send resume acknowledgement
                message = {
                    "type": "task_resumed",
                    "task_id": task_id,
                    "worker_id": self.worker_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Queue message for sending
                self._queue_message(message)
            else:
                logger.warning(f"Requested resume of unknown task {task_id}")
    
    def _handle_error_message(self, data: Dict[str, Any]):
        """
        Handle an error message from coordinator.
        
        Args:
            data: Error message data
        """
        error_message = data.get("message", "Unknown error")
        logger.error(f"Received error from coordinator: {error_message}")
        
    def _handle_welcome(self, data: Dict[str, Any]):
        """
        Handle a welcome message from coordinator.
        
        Args:
            data: Welcome message data
        """
        worker_id = data.get("worker_id", "unknown")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        logger.info(f"Received welcome message from coordinator for worker {worker_id}")
        
        # We don't need to do anything specific here, but we could
        # update internal state if needed
        pass
    
    def _handle_registration_ack(self, data: Dict[str, Any]):
        """
        Handle a registration acknowledgement message from coordinator.
        
        Args:
            data: Registration acknowledgement message data
        """
        worker_id = data.get("worker_id", "unknown")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        logger.info(f"Registration confirmed by coordinator for worker {worker_id}")
        
        # Registration is successful, we could update internal state if needed
        pass
    
    def _handle_task_result_ack(self, data: Dict[str, Any]):
        """
        Handle a task result acknowledgement message from coordinator.
        
        Args:
            data: Task result acknowledgement message data
        """
        task_id = data.get("task_id", "unknown")
        worker_id = data.get("worker_id", "unknown")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        logger.debug(f"Task result acknowledged by coordinator for task {task_id}")
        
        # We could clean up any stored task results here if needed
        with self.task_lock:
            if task_id in self.task_results:
                # We know the coordinator received the result, so we can safely remove it
                # from our local storage to save memory
                del self.task_results[task_id]
    
    def _handle_task_state_ack(self, data: Dict[str, Any]):
        """
        Handle a task state acknowledgement message from coordinator.
        
        Args:
            data: Task state acknowledgement message data
        """
        task_id = data.get("task_id", "unknown")
        worker_id = data.get("worker_id", "unknown")
        state_timestamp = data.get("state_timestamp", "unknown")
        logger.debug(f"Task state update acknowledged by coordinator for task {task_id}")
        
        # No specific action needed here, but we could update internal state if needed
        pass
    
    def _handle_checkpoint_ack(self, data: Dict[str, Any]):
        """
        Handle a checkpoint acknowledgement message from coordinator.
        
        Args:
            data: Checkpoint acknowledgement message data
        """
        checkpoint_id = data.get("checkpoint_id", "unknown")
        task_id = data.get("task_id", "unknown")
        worker_id = data.get("worker_id", "unknown")
        logger.debug(f"Checkpoint {checkpoint_id} acknowledged by coordinator for task {task_id}")
        
        # We could clean up older checkpoints here if needed
        # For now, we'll keep them all for resilience
        pass
    
    def _resend_task_results(self, task_ids: List[str]):
        """
        Resend task results to coordinator.
        
        Args:
            task_ids: List of task IDs to resend results for
        """
        with self.task_lock:
            for task_id in task_ids:
                if task_id in self.task_results:
                    result_data = self.task_results[task_id]
                    
                    message = {
                        "type": "task_result",
                        "task_id": task_id,
                        "worker_id": self.worker_id,
                        "result": result_data["result"],
                        "timestamp": result_data["timestamp"]
                    }
                    
                    if "error" in result_data and result_data["error"]:
                        message["error"] = result_data["error"]
                    
                    self._queue_message(message)
                else:
                    logger.warning(f"No result found for task {task_id}")
    
    def _message_sender_loop(self):
        """Message sender thread function."""
        while not self.stop_event.is_set():
            try:
                # Get a message from the queue with timeout
                try:
                    message = self.message_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Send the message
                if self.connection_state == ConnectionState.CONNECTED and self.ws_conn:
                    try:
                        self.ws_conn.send(json.dumps(message))
                        logger.debug(f"Sent message: {message.get('type', 'unknown')}")
                    except Exception as e:
                        logger.error(f"Error sending message: {e}")
                        # Requeue the message
                        self.message_queue.put(message)
                else:
                    # Not connected, requeue the message
                    self.message_queue.put(message)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in message sender loop: {e}")
                logger.debug(traceback.format_exc())
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)
    
    def _queue_message(self, message: Dict[str, Any]):
        """
        Queue a message for sending.
        
        Args:
            message: Message to send
        """
        # Add message to queue
        self.message_queue.put(message)
    
    def _close_connection(self, new_state: ConnectionState):
        """
        Close the WebSocket connection.
        
        Args:
            new_state: New connection state after closing
        """
        # Close WebSocket connection
        if self.ws_conn:
            try:
                self.ws_conn.close()
            except:
                pass
            self.ws_conn = None
        
        # Update connection state
        self.connection_state = new_state
        
        # Update connection stats
        if self.connection_stats.last_connected is not None:
            self.connection_stats.last_disconnected = datetime.now()
    
    def _get_ws_url(self) -> str:
        """
        Get WebSocket URL for coordinator connection.
        
        Returns:
            WebSocket URL string
        """
        # Convert HTTP URL to WebSocket URL
        url = self.coordinator_url.strip()
        if url.startswith("http://"):
            url = url.replace("http://", "ws://")
        elif url.startswith("https://"):
            url = url.replace("https://", "wss://")
        elif not url.startswith(("ws://", "wss://")):
            url = f"ws://{url}"
        
        # Add worker endpoint
        if not url.endswith("/"):
            url += "/"
            
        # Check if the URL already contains the worker endpoint pattern
        # This prevents duplicated path segments
        worker_path = f"api/v1/worker/{self.worker_id}/ws"
        if f"api/v1/worker/" not in url:
            url += worker_path
        
        return url
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        
        Returns:
            Dictionary of HTTP headers
        """
        headers = {}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers


class WorkerReconnectionPlugin:
    """
    Plugin for integrating worker reconnection into the worker.
    
    This class provides a plugin interface for adding worker reconnection
    capabilities to a worker implementation.
    """
    
    def __init__(self, worker):
        """
        Initialize the worker reconnection plugin.
        
        Args:
            worker: Worker instance
        """
        self.worker = worker
        self.reconnection_manager = None
        
        # Configure reconnection manager
        self._configure_reconnection_manager()
    
    def _configure_reconnection_manager(self):
        """Configure the reconnection manager based on worker configuration."""
        # Get worker configuration
        worker_id = getattr(self.worker, "worker_id", str(uuid.uuid4()))
        coordinator_url = getattr(self.worker, "coordinator_url", "")
        api_key = getattr(self.worker, "api_key", None)
        capabilities = getattr(self.worker, "capabilities", {})
        
        # Create reconnection manager
        self.reconnection_manager = WorkerReconnectionManager(
            worker_id=worker_id,
            coordinator_url=coordinator_url,
            api_key=api_key,
            capabilities=capabilities,
            task_executor=self._task_executor_wrapper
        )
        
        # Configure based on worker settings
        config = {}
        
        # Check for worker-specific configuration
        if hasattr(self.worker, "reconnection_config"):
            config.update(self.worker.reconnection_config)
        
        # Apply configuration
        if config:
            self.reconnection_manager.configure(config)
    
    def _task_executor_wrapper(self, task_id: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrap the worker's task executor with reconnection capabilities.
        
        Args:
            task_id: ID of the task
            task_config: Task configuration
            
        Returns:
            Task result
        """
        # Call worker's task executor
        if hasattr(self.worker, "execute_task"):
            return self.worker.execute_task(task_id, task_config)
        
        # Default implementation
        return {"error": "Worker does not implement execute_task method"}
    
    def start(self):
        """Start the reconnection manager."""
        if self.reconnection_manager:
            self.reconnection_manager.start()
    
    def stop(self):
        """Stop the reconnection manager."""
        if self.reconnection_manager:
            self.reconnection_manager.stop()
    
    def is_connected(self) -> bool:
        """
        Check if the worker is connected to the coordinator.
        
        Returns:
            True if connected, False otherwise
        """
        if self.reconnection_manager:
            return self.reconnection_manager.is_connected()
        return False
    
    def get_connection_stats(self) -> Optional[ConnectionStats]:
        """
        Get connection statistics.
        
        Returns:
            ConnectionStats object or None if not available
        """
        if self.reconnection_manager:
            return self.reconnection_manager.get_connection_stats()
        return None
    
    def submit_task_result(self, task_id: str, result: Dict[str, Any], error: Optional[Dict] = None):
        """
        Submit a task result to the coordinator.
        
        Args:
            task_id: ID of the task
            result: Task result data
            error: Optional error information
        """
        if self.reconnection_manager:
            self.reconnection_manager.submit_task_result(task_id, result, error)
    
    def create_checkpoint(self, task_id: str, checkpoint_data: Dict[str, Any]) -> str:
        """
        Create a checkpoint for a task.
        
        Args:
            task_id: ID of the task
            checkpoint_data: Checkpoint data
            
        Returns:
            Checkpoint ID
        """
        if self.reconnection_manager:
            return self.reconnection_manager.create_checkpoint(task_id, checkpoint_data)
        return ""
    
    def get_latest_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest checkpoint for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Latest checkpoint data or None if no checkpoint exists
        """
        if self.reconnection_manager:
            return self.reconnection_manager.get_latest_checkpoint(task_id)
        return None
    
    def update_task_state(self, task_id: str, state_updates: Dict[str, Any]):
        """
        Update the state of a task.
        
        Args:
            task_id: ID of the task
            state_updates: Dictionary with state updates
        """
        if self.reconnection_manager:
            self.reconnection_manager.update_task_state(task_id, state_updates)
    
    def force_reconnect(self):
        """Force a reconnection to the coordinator."""
        if self.reconnection_manager:
            self.reconnection_manager.force_reconnect()


def create_worker_reconnection_plugin(worker) -> WorkerReconnectionPlugin:
    """
    Create a worker reconnection plugin for a worker.
    
    Args:
        worker: Worker instance
        
    Returns:
        Configured WorkerReconnectionPlugin instance
    """
    plugin = WorkerReconnectionPlugin(worker)
    plugin.start()
    
    logger.info(f"Created worker reconnection plugin for worker {getattr(worker, 'worker_id', 'unknown')}")
    return plugin