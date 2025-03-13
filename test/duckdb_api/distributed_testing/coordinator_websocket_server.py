#!/usr/bin/env python3
"""
WebSocket Coordinator Server for Distributed Testing Framework.

This module implements a WebSocket server for the coordinator that manages
connections from distributed testing workers, handles task assignments,
state synchronization, and fault tolerance.
"""

import os
import sys
import json
import time
import uuid
import asyncio
import logging
import signal
import threading
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import traceback
import websockets
from websockets.server import WebSocketServerProtocol

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_websocket_server")


class CoordinatorState:
    """Holds the state of the coordinator."""
    
    def __init__(self):
        """Initialize coordinator state."""
        self.workers = {}              # worker_id -> worker info
        self.tasks = {}                # task_id -> task info
        self.task_results = {}         # task_id -> task result
        self.checkpoints = {}          # checkpoint_id -> checkpoint info
        self.worker_connections = {}   # worker_id -> WebSocket connection
        self.pending_messages = {}     # worker_id -> list of pending messages
        
        # Lock for thread-safe access to state
        self.lock = threading.RLock()
        
        # Task assignment queue
        self.task_queue = asyncio.Queue()
        
        # Worker registration event
        self.worker_registered_event = asyncio.Event()


class CoordinatorWebSocketServer:
    """WebSocket server for the distributed testing coordinator."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize the coordinator WebSocket server.
        
        Args:
            host: Hostname to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.state = CoordinatorState()
        self.server = None
        self.task_manager_task = None
        self.stop_event = asyncio.Event()
    
    async def start(self):
        """Start the WebSocket server."""
        try:
            # Start the server
            self.server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port
            )
            
            # Start the task manager
            self.task_manager_task = asyncio.create_task(self.task_manager())
            
            logger.info(f"Coordinator WebSocket server started on {self.host}:{self.port}")
            
            # Keep running until stopped
            await self.stop_event.wait()
            
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    async def stop(self):
        """Stop the WebSocket server."""
        # Set stop event
        self.stop_event.set()
        
        # Cancel task manager
        if self.task_manager_task:
            self.task_manager_task.cancel()
            try:
                await self.task_manager_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        with self.state.lock:
            for worker_id, connection in list(self.state.worker_connections.items()):
                try:
                    await connection.close()
                except:
                    pass
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("Coordinator WebSocket server stopped")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle a WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            path: Request path
        """
        # Extract worker ID from path
        worker_id = self._extract_worker_id(path)
        if not worker_id:
            logger.warning(f"Connection attempt with invalid path: {path}")
            await websocket.close(code=1008, reason="Invalid path")
            return
        
        # Register worker connection
        with self.state.lock:
            self.state.worker_connections[worker_id] = websocket
            if worker_id not in self.state.workers:
                self.state.workers[worker_id] = {
                    "connected_at": datetime.now().isoformat(),
                    "last_heartbeat": datetime.now().isoformat(),
                    "capabilities": {},
                    "active_tasks": []
                }
            else:
                self.state.workers[worker_id]["connected_at"] = datetime.now().isoformat()
                self.state.workers[worker_id]["last_heartbeat"] = datetime.now().isoformat()
            
            # Check for pending messages
            pending_messages = self.state.pending_messages.get(worker_id, [])
            self.state.pending_messages[worker_id] = []
        
        # Send welcome message
        welcome_message = {
            "type": "welcome",
            "worker_id": worker_id,
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send(json.dumps(welcome_message))
        
        # Send any pending messages
        for message in pending_messages:
            try:
                await websocket.send(json.dumps(message))
                logger.debug(f"Sent pending message to worker {worker_id}: {message.get('type')}")
            except Exception as e:
                logger.error(f"Error sending pending message to worker {worker_id}: {e}")
                # Re-queue message
                with self.state.lock:
                    if worker_id not in self.state.pending_messages:
                        self.state.pending_messages[worker_id] = []
                    self.state.pending_messages[worker_id].append(message)
        
        # Set worker registration event
        self.state.worker_registered_event.set()
        
        # Process messages
        try:
            async for message_data in websocket:
                try:
                    message = json.loads(message_data)
                    await self._process_message(worker_id, message)
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON from worker {worker_id}")
                except Exception as e:
                    logger.error(f"Error processing message from worker {worker_id}: {e}")
                    logger.debug(traceback.format_exc())
        
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"Connection closed normally for worker {worker_id}")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Connection closed with error for worker {worker_id}: {e}")
        except Exception as e:
            logger.error(f"Error handling connection for worker {worker_id}: {e}")
            logger.debug(traceback.format_exc())
        
        finally:
            # Unregister worker connection
            with self.state.lock:
                if worker_id in self.state.worker_connections:
                    del self.state.worker_connections[worker_id]
                    logger.info(f"Worker {worker_id} disconnected")
    
    async def _process_message(self, worker_id: str, message: Dict[str, Any]):
        """
        Process a message from a worker.
        
        Args:
            worker_id: ID of the worker that sent the message
            message: Message data
        """
        message_type = message.get("type", "unknown")
        logger.debug(f"Received {message_type} message from worker {worker_id}")
        
        # Update last heartbeat time
        with self.state.lock:
            if worker_id in self.state.workers:
                self.state.workers[worker_id]["last_heartbeat"] = datetime.now().isoformat()
        
        # Process message based on type
        if message_type == "registration":
            await self._handle_registration(worker_id, message)
        elif message_type == "heartbeat":
            await self._handle_heartbeat(worker_id, message)
        elif message_type == "task_result":
            await self._handle_task_result(worker_id, message)
        elif message_type == "checkpoint":
            await self._handle_checkpoint(worker_id, message)
        elif message_type == "checkpoint_data":
            await self._handle_checkpoint_data(worker_id, message)
        elif message_type == "task_state":
            await self._handle_task_state(worker_id, message)
        elif message_type == "task_cancelled":
            await self._handle_task_cancelled(worker_id, message)
        elif message_type == "task_paused":
            await self._handle_task_paused(worker_id, message)
        elif message_type == "task_resumed":
            await self._handle_task_resumed(worker_id, message)
        else:
            logger.warning(f"Received unknown message type '{message_type}' from worker {worker_id}")
    
    async def _handle_registration(self, worker_id: str, message: Dict[str, Any]):
        """
        Handle a registration message from a worker.
        
        Args:
            worker_id: ID of the worker
            message: Registration message data
        """
        # Update worker capabilities
        capabilities = message.get("capabilities", {})
        with self.state.lock:
            if worker_id in self.state.workers:
                self.state.workers[worker_id]["capabilities"] = capabilities
        
        logger.info(f"Worker {worker_id} registered with capabilities: {capabilities}")
        
        # Send acknowledgement
        ack_message = {
            "type": "registration_ack",
            "worker_id": worker_id,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_worker(worker_id, ack_message)
    
    async def _handle_heartbeat(self, worker_id: str, message: Dict[str, Any]):
        """
        Handle a heartbeat message from a worker.
        
        Args:
            worker_id: ID of the worker
            message: Heartbeat message data
        """
        # Update active tasks
        active_tasks = message.get("active_tasks", [])
        with self.state.lock:
            if worker_id in self.state.workers:
                self.state.workers[worker_id]["active_tasks"] = active_tasks
        
        # Send heartbeat response
        response = {
            "type": "heartbeat",
            "response_to": message.get("sequence"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if there are any actions to include
        actions = []
        
        # Add actions as needed
        # (e.g., sync_state, resend_task_results, cancel_tasks)
        
        if actions:
            response["actions"] = actions
        
        await self._send_message_to_worker(worker_id, response)
    
    async def _handle_task_result(self, worker_id: str, message: Dict[str, Any]):
        """
        Handle a task result message from a worker.
        
        Args:
            worker_id: ID of the worker
            message: Task result message data
        """
        task_id = message.get("task_id")
        result = message.get("result", {})
        error = message.get("error")
        timestamp = message.get("timestamp")
        
        if not task_id:
            logger.warning(f"Received task result without task_id from worker {worker_id}")
            return
        
        # Store task result
        with self.state.lock:
            self.state.task_results[task_id] = {
                "worker_id": worker_id,
                "result": result,
                "error": error,
                "timestamp": timestamp
            }
            
            # Update task status
            if task_id in self.state.tasks:
                self.state.tasks[task_id]["status"] = "completed" if not error else "failed"
                self.state.tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Received result for task {task_id} from worker {worker_id}")
        
        # Send acknowledgement
        ack_message = {
            "type": "task_result_ack",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_worker(worker_id, ack_message)
    
    async def _handle_checkpoint(self, worker_id: str, message: Dict[str, Any]):
        """
        Handle a checkpoint message from a worker.
        
        Args:
            worker_id: ID of the worker
            message: Checkpoint message data
        """
        checkpoint_id = message.get("checkpoint_id")
        task_id = message.get("task_id")
        timestamp = message.get("timestamp")
        
        if not checkpoint_id or not task_id:
            logger.warning(f"Received checkpoint without required IDs from worker {worker_id}")
            return
        
        # Store checkpoint info
        with self.state.lock:
            self.state.checkpoints[checkpoint_id] = {
                "task_id": task_id,
                "worker_id": worker_id,
                "timestamp": timestamp,
                "has_data": False
            }
        
        logger.debug(f"Received checkpoint {checkpoint_id} for task {task_id} from worker {worker_id}")
        
        # Send acknowledgement
        ack_message = {
            "type": "checkpoint_ack",
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_worker(worker_id, ack_message)
    
    async def _handle_checkpoint_data(self, worker_id: str, message: Dict[str, Any]):
        """
        Handle a checkpoint data message from a worker.
        
        Args:
            worker_id: ID of the worker
            message: Checkpoint data message data
        """
        checkpoint_id = message.get("checkpoint_id")
        task_id = message.get("task_id")
        data = message.get("data", {})
        
        if not checkpoint_id or not task_id:
            logger.warning(f"Received checkpoint data without required IDs from worker {worker_id}")
            return
        
        # Store checkpoint data
        with self.state.lock:
            if checkpoint_id in self.state.checkpoints:
                self.state.checkpoints[checkpoint_id]["data"] = data
                self.state.checkpoints[checkpoint_id]["has_data"] = True
            else:
                # Create new checkpoint entry
                self.state.checkpoints[checkpoint_id] = {
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "timestamp": datetime.now().isoformat(),
                    "data": data,
                    "has_data": True
                }
        
        logger.debug(f"Received checkpoint data for checkpoint {checkpoint_id} from worker {worker_id}")
        
        # Send acknowledgement
        ack_message = {
            "type": "checkpoint_data_ack",
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_worker(worker_id, ack_message)
    
    async def _handle_task_state(self, worker_id: str, message: Dict[str, Any]):
        """
        Handle a task state message from a worker.
        
        Args:
            worker_id: ID of the worker
            message: Task state message data
        """
        task_id = message.get("task_id")
        state = message.get("state", {})
        
        if not task_id:
            logger.warning(f"Received task state without task_id from worker {worker_id}")
            return
        
        # Store task state
        with self.state.lock:
            if task_id in self.state.tasks:
                if "state" not in self.state.tasks[task_id]:
                    self.state.tasks[task_id]["state"] = {}
                self.state.tasks[task_id]["state"].update(state)
        
        logger.debug(f"Received state update for task {task_id} from worker {worker_id}")
        
        # Send acknowledgement
        ack_message = {
            "type": "task_state_ack",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_worker(worker_id, ack_message)
    
    async def _handle_task_cancelled(self, worker_id: str, message: Dict[str, Any]):
        """
        Handle a task cancelled message from a worker.
        
        Args:
            worker_id: ID of the worker
            message: Task cancelled message data
        """
        task_id = message.get("task_id")
        
        if not task_id:
            logger.warning(f"Received task cancelled without task_id from worker {worker_id}")
            return
        
        # Update task status
        with self.state.lock:
            if task_id in self.state.tasks:
                self.state.tasks[task_id]["status"] = "cancelled"
                self.state.tasks[task_id]["cancelled_at"] = datetime.now().isoformat()
        
        logger.info(f"Task {task_id} cancelled by worker {worker_id}")
        
        # Send acknowledgement
        ack_message = {
            "type": "task_cancelled_ack",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_worker(worker_id, ack_message)
    
    async def _handle_task_paused(self, worker_id: str, message: Dict[str, Any]):
        """
        Handle a task paused message from a worker.
        
        Args:
            worker_id: ID of the worker
            message: Task paused message data
        """
        task_id = message.get("task_id")
        
        if not task_id:
            logger.warning(f"Received task paused without task_id from worker {worker_id}")
            return
        
        # Update task status
        with self.state.lock:
            if task_id in self.state.tasks:
                self.state.tasks[task_id]["status"] = "paused"
                self.state.tasks[task_id]["paused_at"] = datetime.now().isoformat()
        
        logger.info(f"Task {task_id} paused by worker {worker_id}")
        
        # Send acknowledgement
        ack_message = {
            "type": "task_paused_ack",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_worker(worker_id, ack_message)
    
    async def _handle_task_resumed(self, worker_id: str, message: Dict[str, Any]):
        """
        Handle a task resumed message from a worker.
        
        Args:
            worker_id: ID of the worker
            message: Task resumed message data
        """
        task_id = message.get("task_id")
        
        if not task_id:
            logger.warning(f"Received task resumed without task_id from worker {worker_id}")
            return
        
        # Update task status
        with self.state.lock:
            if task_id in self.state.tasks:
                self.state.tasks[task_id]["status"] = "running"
                self.state.tasks[task_id]["resumed_at"] = datetime.now().isoformat()
        
        logger.info(f"Task {task_id} resumed by worker {worker_id}")
        
        # Send acknowledgement
        ack_message = {
            "type": "task_resumed_ack",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_worker(worker_id, ack_message)
    
    async def _send_message_to_worker(self, worker_id: str, message: Dict[str, Any]):
        """
        Send a message to a worker.
        
        Args:
            worker_id: ID of the worker
            message: Message to send
        """
        # Check if worker is connected
        connection = None
        with self.state.lock:
            connection = self.state.worker_connections.get(worker_id)
            
            # If not connected, queue message for later delivery
            if not connection:
                if worker_id not in self.state.pending_messages:
                    self.state.pending_messages[worker_id] = []
                self.state.pending_messages[worker_id].append(message)
                logger.debug(f"Queued message for disconnected worker {worker_id}: {message.get('type')}")
                return
        
        # Send message
        try:
            await connection.send(json.dumps(message))
            logger.debug(f"Sent message to worker {worker_id}: {message.get('type')}")
        except Exception as e:
            logger.error(f"Error sending message to worker {worker_id}: {e}")
            
            # Queue message for retry
            with self.state.lock:
                if worker_id not in self.state.pending_messages:
                    self.state.pending_messages[worker_id] = []
                self.state.pending_messages[worker_id].append(message)
    
    async def task_manager(self):
        """Task manager loop for handling task assignments."""
        logger.info("Task manager started")
        
        try:
            while not self.stop_event.is_set():
                # Wait for a worker to register
                await self.state.worker_registered_event.wait()
                
                # Clear the event for next time
                self.state.worker_registered_event.clear()
                
                # Assign tasks to available workers
                await self._assign_tasks()
                
                # Sleep for a short time
                await asyncio.sleep(1.0)
        
        except asyncio.CancelledError:
            logger.info("Task manager cancelled")
        except Exception as e:
            logger.error(f"Error in task manager: {e}")
            logger.debug(traceback.format_exc())
    
    async def _assign_tasks(self):
        """Assign tasks to available workers."""
        # Get available workers
        available_workers = []
        with self.state.lock:
            for worker_id, worker_info in self.state.workers.items():
                if worker_id in self.state.worker_connections:
                    # Check if worker has capacity
                    active_tasks = worker_info.get("active_tasks", [])
                    if len(active_tasks) < 5:  # Arbitrary limit for now
                        available_workers.append(worker_id)
        
        # Check if there are any available workers
        if not available_workers:
            return
        
        # Check if there are any tasks to assign
        try:
            # Try to get a task with timeout
            try:
                task_id, task_config = await asyncio.wait_for(self.state.task_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                return
            
            # Select a worker (simple round-robin for now)
            worker_id = available_workers[0]
            
            # Assign task to worker
            logger.info(f"Assigning task {task_id} to worker {worker_id}")
            
            # Update task state
            with self.state.lock:
                self.state.tasks[task_id] = {
                    "worker_id": worker_id,
                    "config": task_config,
                    "status": "assigned",
                    "assigned_at": datetime.now().isoformat()
                }
            
            # Send task assignment message
            assignment_message = {
                "type": "task_assignment",
                "task_id": task_id,
                "config": task_config,
                "timestamp": datetime.now().isoformat()
            }
            await self._send_message_to_worker(worker_id, assignment_message)
            
            # Mark task as done in queue
            self.state.task_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error assigning task: {e}")
            logger.debug(traceback.format_exc())
    
    async def submit_task(self, task_config: Dict[str, Any]) -> str:
        """
        Submit a task for execution.
        
        Args:
            task_config: Task configuration
            
        Returns:
            Task ID
        """
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Add task to queue
        await self.state.task_queue.put((task_id, task_config))
        
        logger.info(f"Submitted task {task_id}")
        
        return task_id
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task result or None if not available
        """
        with self.state.lock:
            return self.state.task_results.get(task_id)
    
    async def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the state of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task state or None if not available
        """
        with self.state.lock:
            task = self.state.tasks.get(task_id)
            if task:
                return task.get("state", {})
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            True if task was cancelled, False otherwise
        """
        # Check if task exists
        worker_id = None
        with self.state.lock:
            if task_id in self.state.tasks:
                worker_id = self.state.tasks[task_id]["worker_id"]
                if self.state.tasks[task_id]["status"] in ["completed", "failed", "cancelled"]:
                    logger.warning(f"Cannot cancel task {task_id} with status {self.state.tasks[task_id]['status']}")
                    return False
            else:
                logger.warning(f"Task {task_id} not found")
                return False
        
        # Send cancel message to worker
        cancel_message = {
            "type": "cancel_task",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_worker(worker_id, cancel_message)
        
        logger.info(f"Cancelled task {task_id}")
        
        return True
    
    async def pause_task(self, task_id: str) -> bool:
        """
        Pause a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            True if task was paused, False otherwise
        """
        # Check if task exists
        worker_id = None
        with self.state.lock:
            if task_id in self.state.tasks:
                worker_id = self.state.tasks[task_id]["worker_id"]
                if self.state.tasks[task_id]["status"] != "running":
                    logger.warning(f"Cannot pause task {task_id} with status {self.state.tasks[task_id]['status']}")
                    return False
            else:
                logger.warning(f"Task {task_id} not found")
                return False
        
        # Send pause message to worker
        pause_message = {
            "type": "pause_task",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_worker(worker_id, pause_message)
        
        logger.info(f"Paused task {task_id}")
        
        return True
    
    async def resume_task(self, task_id: str) -> bool:
        """
        Resume a paused task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            True if task was resumed, False otherwise
        """
        # Check if task exists
        worker_id = None
        with self.state.lock:
            if task_id in self.state.tasks:
                worker_id = self.state.tasks[task_id]["worker_id"]
                if self.state.tasks[task_id]["status"] != "paused":
                    logger.warning(f"Cannot resume task {task_id} with status {self.state.tasks[task_id]['status']}")
                    return False
            else:
                logger.warning(f"Task {task_id} not found")
                return False
        
        # Send resume message to worker
        resume_message = {
            "type": "resume_task",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        await self._send_message_to_worker(worker_id, resume_message)
        
        logger.info(f"Resumed task {task_id}")
        
        return True
    
    async def request_checkpoint(self, task_id: str, checkpoint_id: Optional[str] = None) -> bool:
        """
        Request a checkpoint for a task.
        
        Args:
            task_id: ID of the task
            checkpoint_id: Optional ID of a specific checkpoint
            
        Returns:
            True if request was sent, False otherwise
        """
        # Check if task exists
        worker_id = None
        with self.state.lock:
            if task_id in self.state.tasks:
                worker_id = self.state.tasks[task_id]["worker_id"]
            else:
                logger.warning(f"Task {task_id} not found")
                return False
        
        # Send checkpoint request to worker
        request_message = {
            "type": "checkpoint_request",
            "task_id": task_id,
            "timestamp": datetime.now().isoformat()
        }
        
        if checkpoint_id:
            request_message["checkpoint_id"] = checkpoint_id
        
        await self._send_message_to_worker(worker_id, request_message)
        
        logger.info(f"Requested checkpoint for task {task_id}")
        
        return True
    
    async def check_worker_health(self, worker_id: str) -> bool:
        """
        Check if a worker is healthy.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            True if worker is healthy, False otherwise
        """
        # Check if worker exists and is connected
        with self.state.lock:
            if worker_id not in self.state.workers:
                return False
            
            # Check if connected
            if worker_id not in self.state.worker_connections:
                return False
            
            # Check last heartbeat
            last_heartbeat_str = self.state.workers[worker_id]["last_heartbeat"]
            last_heartbeat = datetime.fromisoformat(last_heartbeat_str)
            
            # Check if heartbeat is recent (within 30 seconds)
            if datetime.now() - last_heartbeat > timedelta(seconds=30):
                return False
        
        return True
    
    def _extract_worker_id(self, path: str) -> Optional[str]:
        """
        Extract worker ID from WebSocket path.
        
        Args:
            path: WebSocket path
            
        Returns:
            Worker ID or None if not found
        """
        # Expected path format: /api/v1/worker/{worker_id}/ws
        parts = path.strip('/').split('/')
        if len(parts) >= 4 and parts[0] == "api" and parts[1] == "v1" and parts[2] == "worker" and parts[-1] == "ws":
            return parts[3]
        return None


async def run_server(host: str, port: int):
    """
    Run the coordinator WebSocket server.
    
    Args:
        host: Hostname to bind to
        port: Port to listen on
    """
    # Create server
    server = CoordinatorWebSocketServer(host, port)
    
    # Set up signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))
    
    # Start server
    await server.start()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Coordinator WebSocket Server")
    
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
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run server
    asyncio.run(run_server(args.host, args.port))