#!/usr/bin/env python3
"""
Test suite for the Worker Reconnection System.

This module contains unit and integration tests for the worker reconnection functionality,
verifying proper handling of network interruptions, state synchronization, and task recovery.
"""

import os
import sys
import time
import json
import uuid
import threading
import unittest
import logging
import socket
import queue
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, List, Tuple

# Add parent directories to path
current_dir = Path(__file__).parent
parent_dir = str(current_dir.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import worker reconnection module
from duckdb_api.distributed_testing.worker_reconnection import (
    ConnectionState, ConnectionStats, WorkerReconnectionManager,
    WorkerReconnectionPlugin, create_worker_reconnection_plugin
)

# Import hardware fault tolerance for integration testing
from duckdb_api.distributed_testing.hardware_aware_fault_tolerance import (
    RecoveryAction, RecoveryStrategy, FailureType
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_worker_reconnection")


class MockWebSocketApp:
    """Mock WebSocketApp for testing WebSocket connections."""
    
    def __init__(self, url, header, on_open, on_message, on_error, on_close, on_ping, on_pong):
        self.url = url
        self.header = header
        self.on_open = on_open
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.on_ping = on_ping
        self.on_pong = on_pong
        self.connected = False
        self.messages = []
        self.closed = False
        self.close_status_code = None
        self.close_msg = None
    
    def run_forever(self, ping_interval=None, ping_timeout=None):
        """Simulate run_forever method."""
        self.connected = True
        self.on_open(self)
        # Keep running until closed
        while self.connected and not self.closed:
            time.sleep(0.1)
    
    def send(self, message):
        """Simulate send method."""
        if not self.connected:
            raise Exception("Connection is closed")
        self.messages.append(message)
    
    def close(self):
        """Simulate close method."""
        self.closed = True
        self.connected = False
        if self.on_close:
            self.on_close(self, 1000, "Normal closure")


class MockCoordinator:
    """Mock coordinator for testing worker-coordinator interactions."""
    
    def __init__(self):
        self.connected_workers = {}
        self.tasks = {}
        self.results = {}
        self.checkpoints = {}
        self.messages_received = []
        self.messages_to_send = queue.Queue()
        self.is_running = False
        self.worker_handlers = {}
    
    def start(self):
        """Start the mock coordinator."""
        self.is_running = True
        
    def stop(self):
        """Stop the mock coordinator."""
        self.is_running = False
        
    def connect_worker(self, worker_id, worker_ws):
        """Connect a worker to the coordinator."""
        self.connected_workers[worker_id] = {
            "ws": worker_ws,
            "connected_at": datetime.now(),
            "last_heartbeat": datetime.now()
        }
        # Send a welcome message
        self.send_message(worker_id, {
            "type": "welcome",
            "worker_id": worker_id,
            "timestamp": datetime.now().isoformat()
        })
        
    def disconnect_worker(self, worker_id):
        """Disconnect a worker from the coordinator."""
        if worker_id in self.connected_workers:
            del self.connected_workers[worker_id]
            
    def assign_task(self, worker_id, task_id, task_config):
        """Assign a task to a worker."""
        self.tasks[task_id] = {
            "worker_id": worker_id,
            "config": task_config,
            "status": "assigned",
            "assigned_at": datetime.now().isoformat()
        }
        # Send task assignment message
        self.send_message(worker_id, {
            "type": "task_assignment",
            "task_id": task_id,
            "config": task_config,
            "timestamp": datetime.now().isoformat()
        })
        
    def receive_message(self, worker_id, message_data):
        """Receive a message from a worker."""
        message = json.loads(message_data)
        self.messages_received.append((worker_id, message))
        
        # Process message based on type
        msg_type = message.get("type", "")
        
        if msg_type == "registration":
            self.connected_workers[worker_id]["capabilities"] = message.get("capabilities", {})
            
        elif msg_type == "heartbeat":
            self.connected_workers[worker_id]["last_heartbeat"] = datetime.now()
            # Respond with a heartbeat response
            self.send_message(worker_id, {
                "type": "heartbeat",
                "response_to": message.get("sequence"),
                "timestamp": datetime.now().isoformat()
            })
            
        elif msg_type == "task_result":
            task_id = message.get("task_id")
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = "completed"
                self.results[task_id] = {
                    "result": message.get("result", {}),
                    "error": message.get("error"),
                    "timestamp": message.get("timestamp")
                }
                
        elif msg_type == "checkpoint":
            checkpoint_id = message.get("checkpoint_id")
            task_id = message.get("task_id")
            self.checkpoints[checkpoint_id] = {
                "task_id": task_id,
                "worker_id": worker_id,
                "timestamp": message.get("timestamp")
            }
            
        elif msg_type == "task_state":
            task_id = message.get("task_id")
            state = message.get("state", {})
            if task_id in self.tasks:
                if "state" not in self.tasks[task_id]:
                    self.tasks[task_id]["state"] = {}
                self.tasks[task_id]["state"].update(state)
                
    def send_message(self, worker_id, message):
        """Send a message to a worker."""
        if worker_id in self.connected_workers and "ws" in self.connected_workers[worker_id]:
            ws = self.connected_workers[worker_id]["ws"]
            if hasattr(ws, "on_message"):
                ws.on_message(ws, json.dumps(message))
            
    def simulate_network_interruption(self, worker_id, duration=2.0):
        """Simulate a network interruption for a worker."""
        if worker_id in self.connected_workers and "ws" in self.connected_workers[worker_id]:
            ws = self.connected_workers[worker_id]["ws"]
            # Close the connection
            if hasattr(ws, "on_close"):
                ws.on_close(ws, 1006, "Connection interrupted")
            # Wait for the specified duration
            time.sleep(duration)
            # Reconnect the worker
            if hasattr(ws, "on_open"):
                ws.on_open(ws)


class TestConnectionStats(unittest.TestCase):
    """Test ConnectionStats class."""
    
    def test_average_latency(self):
        """Test calculation of average latency."""
        stats = ConnectionStats()
        stats.add_latency_sample(10.0)
        stats.add_latency_sample(20.0)
        stats.add_latency_sample(30.0)
        self.assertEqual(stats.average_latency, 20.0)
    
    def test_connection_stability(self):
        """Test calculation of connection stability."""
        stats = ConnectionStats()
        stats.connected_time = 80.0
        stats.disconnected_time = 20.0
        self.assertEqual(stats.connection_stability, 0.8)
    
    def test_connection_success_rate(self):
        """Test calculation of connection success rate."""
        stats = ConnectionStats()
        stats.connection_attempts = 10
        stats.successful_connections = 8
        self.assertEqual(stats.connection_success_rate, 0.8)
    
    def test_latency_samples_limit(self):
        """Test limiting of latency samples."""
        stats = ConnectionStats()
        # Add 150 samples
        for i in range(150):
            stats.add_latency_sample(float(i))
        # Should only keep the last 100
        self.assertEqual(len(stats.latency_samples), 100)
        self.assertEqual(stats.latency_samples[0], 50.0)
        self.assertEqual(stats.latency_samples[-1], 149.0)


class TestWorkerReconnectionManager(unittest.TestCase):
    """Test WorkerReconnectionManager class."""
    
    def setUp(self):
        """Set up test case."""
        self.worker_id = f"test-worker-{uuid.uuid4()}"
        self.coordinator_url = "ws://localhost:8000"
        self.api_key = "test-api-key"
        self.capabilities = {"cpu": 4, "memory": 8}
        self.task_executor = Mock(return_value={"result": "success"})
        
        # Create manager with mocked websocket
        with patch("websocket.WebSocketApp", MockWebSocketApp):
            self.manager = WorkerReconnectionManager(
                worker_id=self.worker_id,
                coordinator_url=self.coordinator_url,
                api_key=self.api_key,
                capabilities=self.capabilities,
                task_executor=self.task_executor
            )
        
        # Create mock coordinator
        self.coordinator = MockCoordinator()
        
    def tearDown(self):
        """Tear down test case."""
        if hasattr(self, "manager"):
            self.manager.stop()
        if hasattr(self, "coordinator"):
            self.coordinator.stop()
    
    def test_initialization(self):
        """Test initialization of WorkerReconnectionManager."""
        self.assertEqual(self.manager.worker_id, self.worker_id)
        self.assertEqual(self.manager.coordinator_url, self.coordinator_url)
        self.assertEqual(self.manager.api_key, self.api_key)
        self.assertEqual(self.manager.capabilities, self.capabilities)
        self.assertEqual(self.manager.task_executor, self.task_executor)
        self.assertEqual(self.manager.connection_state, ConnectionState.DISCONNECTED)
    
    def test_configure(self):
        """Test configuration of WorkerReconnectionManager."""
        config_updates = {
            "heartbeat_interval": 10.0,
            "max_reconnect_delay": 120.0
        }
        self.manager.configure(config_updates)
        self.assertEqual(self.manager.config["heartbeat_interval"], 10.0)
        self.assertEqual(self.manager.config["max_reconnect_delay"], 120.0)
    
    @patch("websocket.WebSocketApp", MockWebSocketApp)
    def test_connection(self):
        """Test connection to coordinator."""
        # Start manager
        self.manager.start()
        # Wait for connection to establish
        time.sleep(0.5)
        # Check connection state
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTED)
        # Stop manager
        self.manager.stop()
        # Check connection state
        self.assertEqual(self.manager.connection_state, ConnectionState.SHUTDOWN)
    
    @patch("websocket.WebSocketApp", MockWebSocketApp)
    def test_is_connected(self):
        """Test is_connected method."""
        # Initially disconnected
        self.assertFalse(self.manager.is_connected())
        # Start manager
        self.manager.start()
        # Wait for connection to establish
        time.sleep(0.5)
        # Should be connected
        self.assertTrue(self.manager.is_connected())
        # Stop manager
        self.manager.stop()
        # Should be disconnected
        self.assertFalse(self.manager.is_connected())
    
    @patch("websocket.WebSocketApp", MockWebSocketApp)
    def test_calculate_reconnect_delay(self):
        """Test calculation of reconnection delay."""
        # Set initial configuration
        self.manager.config["initial_reconnect_delay"] = 1.0
        self.manager.config["max_reconnect_delay"] = 60.0
        self.manager.config["reconnect_jitter"] = 0.0  # Disable jitter for predictable tests
        
        # Test with increasing reconnect attempts
        self.manager.reconnect_attempts = 1
        delay1 = self.manager._calculate_reconnect_delay()
        self.assertEqual(delay1, 1.0)  # 1.0 * (2^0)
        
        self.manager.reconnect_attempts = 2
        delay2 = self.manager._calculate_reconnect_delay()
        self.assertEqual(delay2, 2.0)  # 1.0 * (2^1)
        
        self.manager.reconnect_attempts = 3
        delay3 = self.manager._calculate_reconnect_delay()
        self.assertEqual(delay3, 4.0)  # 1.0 * (2^2)
        
        self.manager.reconnect_attempts = 7
        delay7 = self.manager._calculate_reconnect_delay()
        self.assertEqual(delay7, 60.0)  # Capped at max_reconnect_delay
    
    @patch("websocket.WebSocketApp", MockWebSocketApp)
    def test_task_submission(self):
        """Test task submission and result handling."""
        # Start manager
        self.manager.start()
        # Wait for connection to establish
        time.sleep(0.5)
        
        # Submit a task result
        task_id = "test-task-1"
        result = {"output": "test result"}
        self.manager.submit_task_result(task_id, result)
        
        # Check that result was stored
        self.assertIn(task_id, self.manager.task_results)
        self.assertEqual(self.manager.task_results[task_id]["result"], result)
        
        # Check that a message was queued
        self.assertFalse(self.manager.message_queue.empty())
    
    @patch("websocket.WebSocketApp", MockWebSocketApp)
    def test_checkpoint_creation(self):
        """Test creation of task checkpoints."""
        # Start manager
        self.manager.start()
        # Wait for connection to establish
        time.sleep(0.5)
        
        # Create a checkpoint
        task_id = "test-task-1"
        checkpoint_data = {"progress": 50, "processed_items": 500}
        checkpoint_id = self.manager.create_checkpoint(task_id, checkpoint_data)
        
        # Check that checkpoint was stored
        self.assertIn(checkpoint_id, self.manager.checkpoints)
        self.assertEqual(self.manager.checkpoints[checkpoint_id]["task_id"], task_id)
        self.assertEqual(self.manager.checkpoints[checkpoint_id]["data"], checkpoint_data)
        
        # Check that task state was updated
        self.assertIn(task_id, self.manager.task_states)
        self.assertIn("last_checkpoint", self.manager.task_states[task_id])
        self.assertEqual(self.manager.task_states[task_id]["last_checkpoint"]["checkpoint_id"], checkpoint_id)
        
        # Check that a message was queued
        self.assertFalse(self.manager.message_queue.empty())
    
    @patch("websocket.WebSocketApp", MockWebSocketApp)
    def test_get_latest_checkpoint(self):
        """Test retrieval of latest checkpoint."""
        # Start manager
        self.manager.start()
        # Wait for connection to establish
        time.sleep(0.5)
        
        # Create multiple checkpoints
        task_id = "test-task-1"
        checkpoint_data1 = {"progress": 25, "processed_items": 250}
        checkpoint_data2 = {"progress": 50, "processed_items": 500}
        checkpoint_data3 = {"progress": 75, "processed_items": 750}
        
        # Create checkpoints with delays to ensure different timestamps
        checkpoint_id1 = self.manager.create_checkpoint(task_id, checkpoint_data1)
        time.sleep(0.1)
        checkpoint_id2 = self.manager.create_checkpoint(task_id, checkpoint_data2)
        time.sleep(0.1)
        checkpoint_id3 = self.manager.create_checkpoint(task_id, checkpoint_data3)
        
        # Get latest checkpoint
        latest_checkpoint = self.manager.get_latest_checkpoint(task_id)
        
        # Check that latest checkpoint was returned
        self.assertEqual(latest_checkpoint, checkpoint_data3)
    
    @patch("websocket.WebSocketApp", MockWebSocketApp)
    def test_update_task_state(self):
        """Test updating task state."""
        # Start manager
        self.manager.start()
        # Wait for connection to establish
        time.sleep(0.5)
        
        # Update task state
        task_id = "test-task-1"
        state_updates = {"status": "running", "progress": 25}
        self.manager.update_task_state(task_id, state_updates)
        
        # Check that task state was updated
        self.assertIn(task_id, self.manager.task_states)
        self.assertEqual(self.manager.task_states[task_id]["status"], "running")
        self.assertEqual(self.manager.task_states[task_id]["progress"], 25)
        
        # Check that a message was queued
        self.assertFalse(self.manager.message_queue.empty())
    
    @patch("websocket.WebSocketApp", MockWebSocketApp)
    def test_handle_connection_failure(self):
        """Test handling of connection failures."""
        # Configure for testing
        self.manager.config["initial_reconnect_delay"] = 0.1
        self.manager.config["max_reconnect_delay"] = 0.5
        self.manager.config["reconnect_jitter"] = 0.0
        
        # Start manager
        self.manager.start()
        # Wait for connection to establish
        time.sleep(0.5)
        
        # Simulate connection failure
        self.manager._close_connection(ConnectionState.DISCONNECTED)
        self.manager._handle_connection_failure()
        
        # Check that reconnect attempt was incremented
        self.assertEqual(self.manager.reconnect_attempts, 1)
        
        # Check that connection state is DISCONNECTED
        self.assertEqual(self.manager.connection_state, ConnectionState.DISCONNECTED)
    
    @patch("websocket.WebSocketApp", MockWebSocketApp)
    def test_max_reconnect_attempts(self):
        """Test maximum reconnection attempts."""
        # Configure for testing
        self.manager.config["initial_reconnect_delay"] = 0.1
        self.manager.config["max_reconnect_delay"] = 0.5
        self.manager.config["reconnect_jitter"] = 0.0
        self.manager.config["max_reconnect_attempts"] = 2
        
        # Start manager
        self.manager.start()
        # Wait for connection to establish
        time.sleep(0.5)
        
        # Simulate connection failure exceeding max attempts
        self.manager._close_connection(ConnectionState.DISCONNECTED)
        self.manager.reconnect_attempts = 2
        self.manager._handle_connection_failure()
        
        # Check that connection state is FAILED
        self.assertEqual(self.manager.connection_state, ConnectionState.FAILED)


class TestWorkerReconnectionIntegration(unittest.TestCase):
    """Integration tests for worker reconnection system."""
    
    def setUp(self):
        """Set up test case."""
        self.mock_coordinator = MockCoordinator()
        self.mock_coordinator.start()
        
        # Create worker reconnection manager
        self.worker_id = f"test-worker-{uuid.uuid4()}"
        self.manager = None  # Will be set in individual tests
    
    def tearDown(self):
        """Tear down test case."""
        if self.manager:
            self.manager.stop()
        self.mock_coordinator.stop()
    
    @patch("websocket.WebSocketApp")
    def test_connection_recovery(self, mock_ws_app_class):
        """Test recovery from connection interruptions."""
        # Configure mock WebSocketApp
        mock_ws = MagicMock()
        mock_ws_app_class.return_value = mock_ws
        
        # Set up manager
        self.manager = WorkerReconnectionManager(
            worker_id=self.worker_id,
            coordinator_url="ws://localhost:8000",
            capabilities={"cpu": 4, "memory": 8}
        )
        
        # Customize for faster testing
        self.manager.config["initial_reconnect_delay"] = 0.1
        self.manager.config["max_reconnect_delay"] = 0.5
        self.manager.config["reconnect_jitter"] = 0.0
        self.manager.config["heartbeat_interval"] = 0.2
        
        # Start manager
        self.manager.start()
        
        # Simulate connection establishment
        on_open_handler = None
        for call in mock_ws_app_class.call_args_list:
            args, kwargs = call
            on_open_handler = kwargs.get("on_open")
        
        self.assertIsNotNone(on_open_handler)
        on_open_handler(mock_ws)
        
        # Check initial state
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTED)
        
        # Simulate connection interruption
        on_close_handler = None
        for call in mock_ws_app_class.call_args_list:
            args, kwargs = call
            on_close_handler = kwargs.get("on_close")
        
        self.assertIsNotNone(on_close_handler)
        on_close_handler(mock_ws, 1006, "Connection interrupted")
        
        # Check state after interruption
        self.assertEqual(self.manager.connection_state, ConnectionState.DISCONNECTED)
        
        # Wait for reconnection attempt
        time.sleep(0.5)
        
        # Verify reconnection was attempted
        self.assertTrue(mock_ws_app_class.call_count >= 2)
    
    @patch("websocket.WebSocketApp")
    def test_state_synchronization(self, mock_ws_app_class):
        """Test state synchronization after reconnection."""
        # Configure mock WebSocketApp
        mock_ws = MagicMock()
        mock_ws_app_class.return_value = mock_ws
        
        # Set up manager
        self.manager = WorkerReconnectionManager(
            worker_id=self.worker_id,
            coordinator_url="ws://localhost:8000",
            capabilities={"cpu": 4, "memory": 8}
        )
        
        # Start manager
        self.manager.start()
        
        # Simulate connection establishment
        on_open_handler = None
        for call in mock_ws_app_class.call_args_list:
            args, kwargs = call
            on_open_handler = kwargs.get("on_open")
        
        self.assertIsNotNone(on_open_handler)
        on_open_handler(mock_ws)
        
        # Add some task results and state
        task_id = "test-task-sync"
        self.manager.task_results[task_id] = {
            "result": {"output": "test result"},
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
        
        self.manager.task_states[task_id] = {
            "status": "running",
            "progress": 50
        }
        
        # Reset mock to clear previous calls
        mock_ws.reset_mock()
        
        # Simulate reconnection - trigger synchronization manually
        self.manager._synchronize_state()
        
        # Verify that task results and states were sent
        self.assertTrue(mock_ws.send.call_count > 0)
        
        # Check for task_result message
        task_result_sent = False
        for call in mock_ws.send.call_args_list:
            args, kwargs = call
            message = json.loads(args[0])
            if message.get("type") == "task_result" and message.get("task_id") == task_id:
                task_result_sent = True
                break
        
        self.assertTrue(task_result_sent, "Task result was not synchronized")
        
        # Check for task_state message
        task_state_sent = False
        for call in mock_ws.send.call_args_list:
            args, kwargs = call
            message = json.loads(args[0])
            if message.get("type") == "task_state" and message.get("task_id") == task_id:
                task_state_sent = True
                break
        
        self.assertTrue(task_state_sent, "Task state was not synchronized")
    
    @patch("websocket.WebSocketApp")
    def test_task_resumption_from_checkpoint(self, mock_ws_app_class):
        """Test task resumption from checkpoints."""
        # Configure mock WebSocketApp
        mock_ws = MagicMock()
        mock_ws_app_class.return_value = mock_ws
        
        # Set up manager with task executor
        task_executor_mock = Mock(return_value={"result": "success"})
        self.manager = WorkerReconnectionManager(
            worker_id=self.worker_id,
            coordinator_url="ws://localhost:8000",
            capabilities={"cpu": 4, "memory": 8},
            task_executor=task_executor_mock
        )
        
        # Start manager
        self.manager.start()
        
        # Simulate connection establishment
        on_open_handler = None
        for call in mock_ws_app_class.call_args_list:
            args, kwargs = call
            on_open_handler = kwargs.get("on_open")
        
        self.assertIsNotNone(on_open_handler)
        on_open_handler(mock_ws)
        
        # Create a task with checkpoint
        task_id = "test-task-resume"
        checkpoint_data = {"progress": 50, "processed_items": 500}
        
        # Create checkpoint
        checkpoint_id = self.manager.create_checkpoint(task_id, checkpoint_data)
        
        # Simulate message handler for checkpoint request
        on_message_handler = None
        for call in mock_ws_app_class.call_args_list:
            args, kwargs = call
            on_message_handler = kwargs.get("on_message")
        
        self.assertIsNotNone(on_message_handler)
        
        # Reset mock to clear previous calls
        mock_ws.reset_mock()
        
        # Simulate checkpoint request from coordinator
        checkpoint_request = {
            "type": "checkpoint_request",
            "task_id": task_id,
            "checkpoint_id": checkpoint_id
        }
        on_message_handler(mock_ws, json.dumps(checkpoint_request))
        
        # Verify checkpoint data was sent
        self.assertTrue(mock_ws.send.call_count > 0)
        
        # Check for checkpoint_data message
        checkpoint_data_sent = False
        for call in mock_ws.send.call_args_list:
            args, kwargs = call
            message = json.loads(args[0])
            if message.get("type") == "checkpoint_data" and message.get("checkpoint_id") == checkpoint_id:
                checkpoint_data_sent = True
                checkpoint_content = message.get("data")
                self.assertEqual(checkpoint_content, checkpoint_data)
                break
        
        self.assertTrue(checkpoint_data_sent, "Checkpoint data was not sent")
    
    @patch("websocket.WebSocketApp")
    def test_message_delivery_reliability(self, mock_ws_app_class):
        """Test reliable message delivery."""
        # Configure mock WebSocketApp
        mock_ws = MagicMock()
        mock_ws_app_class.return_value = mock_ws
        
        # Set up manager
        self.manager = WorkerReconnectionManager(
            worker_id=self.worker_id,
            coordinator_url="ws://localhost:8000",
            capabilities={"cpu": 4, "memory": 8}
        )
        
        # Customize for faster testing
        self.manager.config["initial_reconnect_delay"] = 0.1
        
        # Start manager
        self.manager.start()
        
        # Simulate connection establishment
        on_open_handler = None
        for call in mock_ws_app_class.call_args_list:
            args, kwargs = call
            on_open_handler = kwargs.get("on_open")
        
        self.assertIsNotNone(on_open_handler)
        on_open_handler(mock_ws)
        
        # Queue a message when connected
        message1 = {
            "type": "test_message",
            "content": "Message 1"
        }
        self.manager._queue_message(message1)
        
        # Wait for message sender to process
        time.sleep(0.5)
        
        # Verify message was sent
        self.assertTrue(mock_ws.send.call_count > 0)
        
        # Reset mock to clear previous calls
        mock_ws.reset_mock()
        
        # Simulate connection interruption
        on_close_handler = None
        for call in mock_ws_app_class.call_args_list:
            args, kwargs = call
            on_close_handler = kwargs.get("on_close")
        
        self.assertIsNotNone(on_close_handler)
        on_close_handler(mock_ws, 1006, "Connection interrupted")
        
        # Queue message while disconnected
        message2 = {
            "type": "test_message",
            "content": "Message 2"
        }
        self.manager._queue_message(message2)
        
        # Verify no messages sent while disconnected
        self.assertEqual(mock_ws.send.call_count, 0)
        
        # Simulate reconnection
        mock_ws.reset_mock()
        on_open_handler(mock_ws)
        
        # Wait for message sender to process
        time.sleep(0.5)
        
        # Verify queued messages were sent after reconnection
        self.assertTrue(mock_ws.send.call_count > 0)
    
    @patch("websocket.WebSocketApp")
    def test_interaction_with_hardware_fault_tolerance(self, mock_ws_app_class):
        """Test interaction with hardware-aware fault tolerance."""
        # Configure mock WebSocketApp
        mock_ws = MagicMock()
        mock_ws_app_class.return_value = mock_ws
        
        # Create hardware recovery strategy mock
        mock_hardware_recovery = Mock()
        mock_hardware_recovery.get_recovery_actions.return_value = [
            RecoveryAction.RESTART_PROCESS
        ]
        
        # Set up manager
        self.manager = WorkerReconnectionManager(
            worker_id=self.worker_id,
            coordinator_url="ws://localhost:8000",
            capabilities={"cpu": 4, "memory": 8}
        )
        
        # Start manager
        self.manager.start()
        
        # Simulate connection establishment
        on_open_handler = None
        for call in mock_ws_app_class.call_args_list:
            args, kwargs = call
            on_open_handler = kwargs.get("on_open")
        
        self.assertIsNotNone(on_open_handler)
        on_open_handler(mock_ws)
        
        # Add task with hardware failure
        task_id = "test-task-hardware-failure"
        
        # Create task state indicating hardware failure
        hardware_error = {
            "failure_type": str(FailureType.GPU_OUT_OF_MEMORY),
            "error_message": "CUDA out of memory",
            "recovery_attempted": False
        }
        
        self.manager.update_task_state(task_id, {
            "status": "failed",
            "hardware_error": hardware_error
        })
        
        # Verify error state was reported
        self.assertIn(task_id, self.manager.task_states)
        self.assertEqual(self.manager.task_states[task_id]["status"], "failed")
        self.assertEqual(
            self.manager.task_states[task_id]["hardware_error"]["failure_type"],
            str(FailureType.GPU_OUT_OF_MEMORY)
        )
        
        # Verify message was sent
        self.assertTrue(mock_ws.send.call_count > 0)
        
        # Check for task_state message with hardware error
        hardware_error_sent = False
        for call in mock_ws.send.call_args_list:
            args, kwargs = call
            try:
                message = json.loads(args[0])
                if (message.get("type") == "task_state" and 
                    message.get("task_id") == task_id and
                    "hardware_error" in message.get("state", {})):
                    hardware_error_sent = True
                    break
            except:
                continue
        
        self.assertTrue(hardware_error_sent, "Hardware error was not reported")


class TestWorkerReconnectionPlugin(unittest.TestCase):
    """Test WorkerReconnectionPlugin class."""
    
    def test_plugin_initialization(self):
        """Test initialization of WorkerReconnectionPlugin."""
        # Create mock worker
        mock_worker = Mock()
        mock_worker.worker_id = "test-worker"
        mock_worker.coordinator_url = "ws://localhost:8000"
        mock_worker.api_key = "test-api-key"
        mock_worker.capabilities = {"cpu": 4, "memory": 8}
        
        # Create plugin with mocked reconnection manager
        with patch("duckdb_api.distributed_testing.worker_reconnection.WorkerReconnectionManager"):
            plugin = WorkerReconnectionPlugin(mock_worker)
            self.assertEqual(plugin.worker, mock_worker)
            self.assertIsNotNone(plugin.reconnection_manager)
    
    def test_plugin_api_passthrough(self):
        """Test API passthrough of WorkerReconnectionPlugin."""
        # Create mock worker
        mock_worker = Mock()
        mock_worker.worker_id = "test-worker"
        mock_worker.coordinator_url = "ws://localhost:8000"
        
        # Create mock reconnection manager
        mock_manager = Mock()
        mock_manager.is_connected.return_value = True
        mock_manager.get_connection_stats.return_value = ConnectionStats()
        mock_manager.submit_task_result.return_value = None
        mock_manager.create_checkpoint.return_value = "test-checkpoint-id"
        mock_manager.get_latest_checkpoint.return_value = {"progress": 50}
        mock_manager.update_task_state.return_value = None
        mock_manager.force_reconnect.return_value = None
        
        # Create plugin with mocked reconnection manager
        with patch("duckdb_api.distributed_testing.worker_reconnection.WorkerReconnectionManager", return_value=mock_manager):
            plugin = WorkerReconnectionPlugin(mock_worker)
            
            # Test API passthrough
            self.assertTrue(plugin.is_connected())
            self.assertIsNotNone(plugin.get_connection_stats())
            
            plugin.submit_task_result("task-id", {"result": "success"})
            mock_manager.submit_task_result.assert_called_once()
            
            checkpoint_id = plugin.create_checkpoint("task-id", {"progress": 50})
            self.assertEqual(checkpoint_id, "test-checkpoint-id")
            mock_manager.create_checkpoint.assert_called_once()
            
            checkpoint = plugin.get_latest_checkpoint("task-id")
            self.assertEqual(checkpoint, {"progress": 50})
            mock_manager.get_latest_checkpoint.assert_called_once()
            
            plugin.update_task_state("task-id", {"status": "running"})
            mock_manager.update_task_state.assert_called_once()
            
            plugin.force_reconnect()
            mock_manager.force_reconnect.assert_called_once()
    
    def test_create_worker_reconnection_plugin(self):
        """Test create_worker_reconnection_plugin function."""
        # Create mock worker
        mock_worker = Mock()
        mock_worker.worker_id = "test-worker"
        mock_worker.coordinator_url = "ws://localhost:8000"
        
        # Create mock plugin
        mock_plugin = Mock()
        
        # Test with mocked plugin
        with patch("duckdb_api.distributed_testing.worker_reconnection.WorkerReconnectionPlugin", return_value=mock_plugin):
            plugin = create_worker_reconnection_plugin(mock_worker)
            self.assertEqual(plugin, mock_plugin)
            mock_plugin.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()