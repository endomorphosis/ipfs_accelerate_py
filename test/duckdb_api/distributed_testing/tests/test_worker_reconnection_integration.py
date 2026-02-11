#!/usr/bin/env python3
"""
Integration tests for Worker Reconnection System with real WebSocket coordinator.

This module contains integration tests that verify the Worker Reconnection System
works correctly with a real WebSocket coordinator server.
"""

import os
import sys
import time
import json
import uuid
import anyio
import websockets
import threading
import unittest
import logging
import multiprocessing
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from unittest.mock import patch

# Add parent directories to path
current_dir = Path(__file__).parent
parent_dir = str(current_dir.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import worker reconnection module
from data.duckdb.distributed_testing.worker_reconnection import (
    ConnectionState, ConnectionStats, WorkerReconnectionManager,
    WorkerReconnectionPlugin, create_worker_reconnection_plugin
)

# Import coordinator WebSocket server for testing
from data.duckdb.distributed_testing.coordinator_websocket_server import (
    CoordinatorWebSocketServer
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_worker_reconnection_integration")


class CoordinatorServerProcess(multiprocessing.Process):
    """Process for running the coordinator WebSocket server."""

    def __init__(self, host='localhost', port=8765):
        """Initialize the coordinator server process."""
        super().__init__()
        self.host = host
        self.port = port
        self.should_stop = multiprocessing.Event()
        self.ready = multiprocessing.Event()

    def run(self):
        """Run the coordinator server."""
        try:
            # Start AnyIO event loop
            anyio.run(self._run_server)
        except Exception as e:
            logger.error(f"Error in coordinator server process: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    async def _run_server(self):
        """Run the coordinator WebSocket server."""
        try:
            # Create server
            self.server = CoordinatorWebSocketServer(self.host, self.port)

            async with anyio.create_task_group() as tg:
                # Start server
                tg.start_soon(self.server.start)

                # Set ready event after a short delay to ensure server is listening
                await anyio.sleep(2)
                self.ready.set()

                # Wait for stop event
                while not self.should_stop.is_set():
                    await anyio.sleep(0.1)

                # Stop server and cancel background task
                await self.server.stop()
                tg.cancel_scope.cancel()
                
        except Exception as e:
            logger.error(f"Error running coordinator server: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def stop(self):
        """Stop the coordinator server."""
        self.should_stop.set()


class SimpleTaskExecutor:
    """Simple task executor for testing."""
    
    def __init__(self):
        """Initialize the task executor."""
        self.executed_tasks = {}
        self.task_error = None
        self.task_sleep = 0
        self.checkpoint_interval = 0
        self.checkpoint_data = {}
        self.reconnection_manager = None
    
    def execute_task(self, task_id, task_config):
        """
        Execute a task.
        
        Args:
            task_id: ID of the task
            task_config: Task configuration
            
        Returns:
            Task result
        """
        # Store task
        self.executed_tasks[task_id] = {
            "config": task_config,
            "started_at": datetime.now().isoformat()
        }
        
        # Update task state
        if self.reconnection_manager:
            self.reconnection_manager.update_task_state(task_id, {
                "status": "running",
                "progress": 0
            })
        
        # Check if we should simulate an error
        if self.task_error:
            raise Exception(self.task_error)
        
        # Simulate task execution with progress updates and checkpoints
        total_iterations = task_config.get("iterations", 10)
        for i in range(total_iterations):
            # Sleep to simulate work
            time.sleep(self.task_sleep)
            
            # Update progress
            progress = (i + 1) / total_iterations * 100
            if self.reconnection_manager:
                self.reconnection_manager.update_task_state(task_id, {
                    "progress": progress
                })
            
            # Create checkpoint if needed
            if self.checkpoint_interval > 0 and (i + 1) % self.checkpoint_interval == 0:
                checkpoint_data = {
                    "iteration": i + 1,
                    "progress": progress,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add custom checkpoint data if provided
                if task_id in self.checkpoint_data:
                    checkpoint_data.update(self.checkpoint_data[task_id])
                
                # Create checkpoint
                if self.reconnection_manager:
                    self.reconnection_manager.create_checkpoint(task_id, checkpoint_data)
        
        # Return result
        result = {
            "status": "completed",
            "iterations": total_iterations,
            "completed_at": datetime.now().isoformat()
        }
        
        # Add task-specific result data if provided
        if task_id in self.checkpoint_data:
            result.update(self.checkpoint_data[task_id])
        
        return result


class NetworkDisruptorProxy:
    """Proxy that can simulate network disruptions."""
    
    def __init__(self, target_host, target_port):
        """
        Initialize the network disruptor proxy.
        
        Args:
            target_host: Target host to proxy to
            target_port: Target port to proxy to
        """
        self.target_host = target_host
        self.target_port = target_port
        self.server = None
        self.clients = set()
        self.server_task = None
        self.disruption_state = False
        self.disruption_lock = threading.Lock()
    
    async def start(self, host='localhost', port=8766):
        """
        Start the proxy server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.server = await websockets.serve(
            self.handle_client,
            host,
            port
        )
        logger.info(f"Network disruptor proxy started on {host}:{port}")
    
    async def stop(self):
        """Stop the proxy server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all client connections
        for client in list(self.clients):
            try:
                await client.close()
            except:
                pass
        
        self.clients.clear()
        logger.info("Network disruptor proxy stopped")
    
    async def handle_client(self, websocket, path):
        """
        Handle a client connection.
        
        Args:
            websocket: WebSocket connection
            path: Request path
        """
        # Add client to tracking set
        self.clients.add(websocket)
        
        try:
            # Connect to target
            try:
                target_url = f"ws://{self.target_host}:{self.target_port}{path}"
                target_websocket = await websockets.connect(target_url)
            except Exception as e:
                logger.error(f"Error connecting to target: {e}")
                return
            
            # Set up bidirectional relay
            done_event = anyio.Event()

            async def _relay_wrapper(source, target, direction):
                try:
                    await self.relay(source, target, direction)
                finally:
                    done_event.set()

            async with anyio.create_task_group() as tg:
                tg.start_soon(_relay_wrapper, websocket, target_websocket, "client_to_target")
                tg.start_soon(_relay_wrapper, target_websocket, websocket, "target_to_client")
                await done_event.wait()
                tg.cancel_scope.cancel()
            
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        finally:
            # Remove client from tracking set
            self.clients.remove(websocket)
    
    async def relay(self, source, target, direction):
        """
        Relay messages between source and target WebSockets.
        
        Args:
            source: Source WebSocket
            target: Target WebSocket
            direction: Direction of relay (for logging)
        """
        try:
            async for message in source:
                # Check if disruption is active
                with self.disruption_lock:
                    if self.disruption_state:
                        # Drop message during disruption
                        logger.debug(f"Dropping message in direction {direction}: {message[:100]}")
                        continue
                
                # Forward message
                try:
                    await target.send(message)
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    logger.error(f"Error forwarding message in direction {direction}: {e}")
                    break
        
        except websockets.exceptions.ConnectionClosed:
            pass
        
        except Exception as e:
            logger.error(f"Error in relay ({direction}): {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        finally:
            # Close both connections
            try:
                await source.close()
            except:
                pass
            
            try:
                await target.close()
            except:
                pass
    
    def start_disruption(self, duration=5.0):
        """
        Start a network disruption for the specified duration.
        
        Args:
            duration: Duration of disruption in seconds
        """
        def disrupt():
            with self.disruption_lock:
                logger.info(f"Starting network disruption for {duration} seconds")
                self.disruption_state = True
            
            # Sleep for duration
            time.sleep(duration)
            
            with self.disruption_lock:
                logger.info("Ending network disruption")
                self.disruption_state = False
        
        # Start disruption in a separate thread
        threading.Thread(target=disrupt, daemon=True).start()


class TestWorkerReconnectionWithRealCoordinator(unittest.TestCase):
    """Integration tests for Worker Reconnection System with real WebSocket coordinator."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test case class."""
        # Start coordinator server process
        cls.coordinator = CoordinatorServerProcess()
        cls.coordinator.start()
        
        # Wait for server to be ready
        cls.coordinator.ready.wait()
        
        # Create disruptor proxy and run it via an AnyIO blocking portal
        cls.disruptor = NetworkDisruptorProxy('localhost', 8765)
        cls.disruptor_portal = anyio.from_thread.start_blocking_portal()
        cls.disruptor_portal.call(cls.disruptor.start, host='localhost', port=8766)
    
    @classmethod
    def tearDownClass(cls):
        """Tear down the test case class."""
        # Stop network disruptor
        cls.disruptor_portal.call(cls.disruptor.stop)
        cls.disruptor_portal.stop()
        
        # Stop coordinator server
        cls.coordinator.stop()
        cls.coordinator.join()

    def setUp(self):
        """Set up the test case."""
        # Create worker ID
        self.worker_id = f"test-worker-{uuid.uuid4()}"
        
        # Create task executor
        self.executor = SimpleTaskExecutor()
        
        # Create worker reconnection manager
        self.manager = WorkerReconnectionManager(
            worker_id=self.worker_id,
            coordinator_url="ws://localhost:8766/api/v1/worker/{worker_id}/ws",
            capabilities={"cpu": 4, "memory": 8},
            task_executor=self.executor.execute_task
        )
        
        # Set executor's reconnection manager
        self.executor.reconnection_manager = self.manager
        
        # Configure for faster testing
        self.manager.config["heartbeat_interval"] = 1.0
        self.manager.config["initial_reconnect_delay"] = 0.5
        self.manager.config["max_reconnect_delay"] = 2.0
        self.manager.config["reconnect_jitter"] = 0.0
    
    def tearDown(self):
        """Tear down the test case."""
        if hasattr(self, "manager"):
            self.manager.stop()
    
    def test_basic_connection(self):
        """Test basic connection to coordinator."""
        # Start manager
        self.manager.start()
        
        # Wait for connection to establish
        time.sleep(2)
        
        # Check connection state
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTED)
        
        # Stop manager
        self.manager.stop()
        
        # Check connection state
        self.assertNotEqual(self.manager.connection_state, ConnectionState.CONNECTED)
    
    def test_reconnection_after_disruption(self):
        """Test reconnection after network disruption."""
        # Start manager
        self.manager.start()
        
        # Wait for connection to establish
        time.sleep(2)
        
        # Check initial connection state
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTED)
        
        # Start network disruption
        self.disruptor.start_disruption(duration=5.0)
        
        # Wait for disruption to be detected
        time.sleep(3)
        
        # Connection should be lost
        self.assertNotEqual(self.manager.connection_state, ConnectionState.CONNECTED)
        
        # Wait for reconnection (disruption + reconnect delay + buffer)
        time.sleep(7)
        
        # Connection should be re-established
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTED)
        
        # Stop manager
        self.manager.stop()
    
    def test_heartbeat_mechanism(self):
        """Test heartbeat mechanism."""
        # Configure for faster heartbeats
        self.manager.config["heartbeat_interval"] = 0.5
        
        # Start manager
        self.manager.start()
        
        # Wait for connection to establish and heartbeats to occur
        time.sleep(3)
        
        # Check that heartbeats were sent and received
        self.assertIsNotNone(self.manager.last_heartbeat_sent)
        self.assertIsNotNone(self.manager.last_heartbeat_received)
        
        # Check that we received some latency samples
        self.assertGreater(len(self.manager.connection_stats.latency_samples), 0)
        
        # Stop manager
        self.manager.stop()
    
    def test_task_execution_and_result_reporting(self):
        """Test task execution and result reporting."""
        # Start manager
        self.manager.start()
        
        # Wait for connection to establish
        time.sleep(2)
        
        # Submit a task using the coordinator API
        task_config = {"type": "test_task", "iterations": 5}
        async def _submit_task() -> str:
            return await self.coordinator.submit_task(task_config)

        task_id = anyio.run(_submit_task)
        
        # Wait for task execution (5 iterations * 0.1s sleep)
        time.sleep(2)
        
        # Check that task was executed
        self.assertIn(task_id, self.executor.executed_tasks)
        
        # Wait for task completion and result reporting
        time.sleep(3)
        
        # Check task result on coordinator
        async def _get_task_result() -> Dict[str, Any]:
            return await self.coordinator.get_task_result(task_id)

        task_result = anyio.run(_get_task_result)
        
        self.assertIsNotNone(task_result)
        self.assertEqual(task_result["result"]["status"], "completed")
        self.assertEqual(task_result["result"]["iterations"], 5)
    
    def test_task_state_updates_and_checkpoint_creation(self):
        """Test task state updates and checkpoint creation."""
        # Configure task executor
        self.executor.task_sleep = 0.2
        self.executor.checkpoint_interval = 2  # Checkpoint every 2 iterations
        
        # Start manager
        self.manager.start()
        
        # Wait for connection to establish
        time.sleep(2)
        
        # Submit a task using the coordinator API
        task_config = {"type": "test_task", "iterations": 6}
        async def _submit_task() -> str:
            return await self.coordinator.submit_task(task_config)

        task_id = anyio.run(_submit_task)
        
        # Wait for task to start and create some checkpoints
        time.sleep(3)
        
        # Check task state on coordinator
        async def _get_task_state() -> Dict[str, Any]:
            return await self.coordinator.get_task_state(task_id)

        task_state = anyio.run(_get_task_state)
        
        self.assertIsNotNone(task_state)
        self.assertIn("progress", task_state)
        
        # Wait for task completion
        time.sleep(5)
        
        # Check task result on coordinator
        async def _get_task_result() -> Dict[str, Any]:
            return await self.coordinator.get_task_result(task_id)

        task_result = anyio.run(_get_task_result)
        
        self.assertIsNotNone(task_result)
        self.assertEqual(task_result["result"]["status"], "completed")
    
    def test_state_synchronization_after_reconnection(self):
        """Test state synchronization after reconnection."""
        # Configure task executor
        self.executor.task_sleep = 0.2
        
        # Start manager
        self.manager.start()
        
        # Wait for connection to establish
        time.sleep(2)
        
        # Submit a task using the coordinator API
        task_config = {"type": "test_task", "iterations": 10}
        async def _submit_task() -> str:
            return await self.coordinator.submit_task(task_config)

        task_id = anyio.run(_submit_task)
        
        # Wait for task to start
        time.sleep(1)
        
        # Start network disruption while task is running
        self.disruptor.start_disruption(duration=3.0)
        
        # Wait for disruption and reconnection
        time.sleep(5)
        
        # Check that connection is re-established
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTED)
        
        # Wait for task completion
        time.sleep(5)
        
        # Check task result on coordinator
        async def _get_task_result() -> Dict[str, Any]:
            return await self.coordinator.get_task_result(task_id)

        task_result = anyio.run(_get_task_result)
        
        self.assertIsNotNone(task_result)
        self.assertEqual(task_result["result"]["status"], "completed")
    
    def test_task_resumption_from_checkpoint_during_network_outage(self):
        """Test task resumption from checkpoint during network outage."""
        # Configure task executor with checkpoints
        self.executor.task_sleep = 0.2
        self.executor.checkpoint_interval = 2  # Checkpoint every 2 iterations
        
        # Keep track of checkpoint resumptions
        resume_count = [0]
        original_execute_task = self.executor.execute_task
        
        def execute_task_with_resume_tracking(task_id, task_config):
            # Check if we have a checkpoint
            checkpoint_data = self.manager.get_latest_checkpoint(task_id)
            if checkpoint_data:
                # Increment resume count
                resume_count[0] += 1
                # Start from checkpoint
                task_config["start_iteration"] = checkpoint_data.get("iteration", 0)
            
            # Execute task normally
            return original_execute_task(task_id, task_config)
        
        # Replace executor method with tracking version
        self.executor.execute_task = execute_task_with_resume_tracking
        
        # Start manager
        self.manager.start()
        
        # Wait for connection to establish
        time.sleep(2)
        
        # Submit a long-running task
        task_config = {"type": "test_task", "iterations": 20}
        async def _submit_task() -> str:
            return await self.coordinator.submit_task(task_config)

        task_id = anyio.run(_submit_task)
        
        # Wait for task to start and create some checkpoints
        time.sleep(3)
        
        # Simulate a task cancellation and reconnection
        self.manager._close_connection(ConnectionState.DISCONNECTED)
        
        # Wait for reconnection
        time.sleep(2)
        
        # Check that connection is re-established
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTED)
        
        # Wait for task completion
        time.sleep(5)
        
        # Check that task was resumed from checkpoint
        self.assertGreater(resume_count[0], 0)
        
        # Check task result
        async def _get_task_result() -> Dict[str, Any]:
            return await self.coordinator.get_task_result(task_id)

        task_result = anyio.run(_get_task_result)
        
        self.assertIsNotNone(task_result)
        self.assertEqual(task_result["result"]["status"], "completed")
    
    def test_message_delivery_reliability_during_reconnection(self):
        """Test message delivery reliability during reconnection."""
        # Start manager
        self.manager.start()
        
        # Wait for connection to establish
        time.sleep(2)
        
        # Queue a custom task state message
        custom_task_id = str(uuid.uuid4())
        custom_state = {"custom_field": "test_value", "timestamp": datetime.now().isoformat()}
        self.manager.update_task_state(custom_task_id, custom_state)
        
        # Start network disruption
        self.disruptor.start_disruption(duration=3.0)
        
        # Queue another message during disruption
        disruption_task_id = str(uuid.uuid4())
        disruption_state = {"disruption_field": "during_outage", "timestamp": datetime.now().isoformat()}
        self.manager.update_task_state(disruption_task_id, disruption_state)
        
        # Wait for reconnection (disruption + reconnect delay + buffer)
        time.sleep(5)
        
        # Check that connection is re-established
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTED)
        
        # Queue a message after reconnection
        after_task_id = str(uuid.uuid4())
        after_state = {"after_field": "after_reconnect", "timestamp": datetime.now().isoformat()}
        self.manager.update_task_state(after_task_id, after_state)
        
        # Wait for messages to be delivered
        time.sleep(3)
        
        # All messages should have been delivered
        # (This is hard to verify directly with the current API, 
        # but we can check that the message queue is empty)
        self.assertTrue(self.manager.message_queue.empty())
    
    def test_worker_plugin_integration_with_actual_worker(self):
        """Test WorkerReconnectionPlugin integration with an actual worker."""
        # Create a mock worker
        mock_worker = type('Worker', (), {
            'worker_id': self.worker_id,
            'coordinator_url': "ws://localhost:8766/api/v1/worker/{worker_id}/ws",
            'capabilities': {"cpu": 4, "memory": 8},
            'execute_task': self.executor.execute_task,
            'reconnection_config': {
                'heartbeat_interval': 1.0,
                'initial_reconnect_delay': 0.5,
                'max_reconnect_delay': 2.0
            }
        })()
        
        # Create plugin
        plugin = create_worker_reconnection_plugin(mock_worker)
        
        # Wait for connection to establish
        time.sleep(2)
        
        # Check that plugin is connected
        self.assertTrue(plugin.is_connected())
        
        # Start network disruption
        self.disruptor.start_disruption(duration=3.0)
        
        # Wait for disruption to be detected
        time.sleep(2)
        
        # Connection should be lost
        self.assertFalse(plugin.is_connected())
        
        # Wait for reconnection
        time.sleep(4)
        
        # Connection should be re-established
        self.assertTrue(plugin.is_connected())
        
        # Stop plugin
        plugin.stop()


if __name__ == "__main__":
    unittest.main()