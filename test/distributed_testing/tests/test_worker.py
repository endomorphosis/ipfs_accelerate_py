#!/usr/bin/env python3
"""
Tests for the Distributed Testing Framework Worker Node

This module tests the worker node component of the distributed testing framework,
ensuring it can properly connect to the coordinator, handle tasks, and manage its lifecycle.
"""

import anyio
import json
import logging
import os
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets
from security import SecurityManager

from worker import DistributedTestingWorker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDistributedTestingWorker(unittest.TestCase):
    """Test cases for the DistributedTestingWorker class."""

    def setUp(self):
        """Set up test environment."""
        self.coordinator_url = "http://localhost:8080"
        self.api_key = "test_api_key"
        self.worker_id = "test_worker_id"
        
        # Create worker instance with mocked security manager
        with patch('worker.SecurityManager') as mock_security_manager:
            self.mock_security = mock_security_manager.return_value
            self.mock_security.sign_message.side_effect = lambda msg: {**msg, "signature": "test_signature"}
            self.mock_security.verify_message.return_value = True
            
            self.worker = DistributedTestingWorker(
                coordinator_url=self.coordinator_url,
                worker_id=self.worker_id,
                api_key=self.api_key
            )
            self.worker.security_manager = self.mock_security

    def test_init(self):
        """Test worker initialization."""
        self.assertEqual(self.worker.coordinator_url, self.coordinator_url)
        self.assertEqual(self.worker.worker_id, self.worker_id)
        self.assertEqual(self.worker.api_key, self.api_key)
        self.assertIsNone(self.worker.token)
        self.assertIsNone(self.worker.current_task)
        self.assertIsNone(self.worker.db)
        self.assertTrue(self.worker.is_healthy)


class TestDistributedTestingWorkerAsync:
    """Async test cases for the DistributedTestingWorker class."""

    @pytest.fixture
    async def worker_setup(self):
        """Set up worker and mocked connection for testing."""
        coordinator_url = "http://localhost:8080"
        api_key = "test_api_key"
        worker_id = "test_worker_id"
        
        # Create mock websocket
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({
            "type": "register_response",
            "worker_id": worker_id,
            "status": "success"
        }))
        
        # Create worker with mocked components
        with patch('worker.SecurityManager') as mock_security_manager, \
             patch('worker.websockets.connect', new=AsyncMock(return_value=mock_ws)), \
             patch('worker.duckdb.connect') as mock_duckdb_connect:
                
            mock_security = mock_security_manager.return_value
            mock_security.sign_message.side_effect = lambda msg: {**msg, "signature": "test_signature"}
            mock_security.verify_message.return_value = True
            
            worker = DistributedTestingWorker(
                coordinator_url=coordinator_url,
                worker_id=worker_id,
                api_key=api_key
            )
            worker.security_manager = mock_security
            worker.ws = mock_ws
            
            yield worker, mock_ws
    
    @pytest.mark.asyncio
    async def test_connect_to_coordinator(self, worker_setup):
        """Test connection to coordinator."""
        worker, mock_ws = worker_setup
        
        # Setup mock for authentication
        mock_ws.recv.side_effect = [
            json.dumps({"type": "auth_response", "status": "success", "token": "new_token", "worker_id": worker.worker_id}),
            json.dumps({"type": "register_response", "worker_id": worker.worker_id, "status": "success"})
        ]
        
        # Test connection
        result = await worker.connect_to_coordinator()
        
        # Verify connection was successful
        assert result is True
        assert worker.ws_connected is True
        
        # Verify authentication was attempted
        mock_ws.send.assert_any_call(json.dumps({
            "type": "auth",
            "auth_type": "api_key",
            "api_key": worker.api_key,
            "worker_id": worker.worker_id
        }))
        
        # Verify registration was attempted
        # The actual contents of the sent message are harder to verify exactly due to the 
        # hardware detection, so we'll just check that it was sent
        assert mock_ws.send.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_authenticate_success(self, worker_setup):
        """Test successful authentication with coordinator."""
        worker, mock_ws = worker_setup
        
        # Setup mock for successful authentication
        mock_ws.recv.return_value = json.dumps({
            "type": "auth_response",
            "status": "success",
            "token": "new_token",
            "worker_id": worker.worker_id
        })
        
        # Test authentication
        worker.ws_connected = True  # Assume we're connected
        result = await worker._authenticate()
        
        # Verify authentication was successful
        assert result is True
        assert worker.token == "new_token"
        
        # Verify authentication message was sent
        mock_ws.send.assert_called_with(json.dumps({
            "type": "auth",
            "auth_type": "api_key",
            "api_key": worker.api_key,
            "worker_id": worker.worker_id
        }))
    
    @pytest.mark.asyncio
    async def test_authenticate_failure(self, worker_setup):
        """Test failed authentication with coordinator."""
        worker, mock_ws = worker_setup
        
        # Setup mock for failed authentication
        mock_ws.recv.return_value = json.dumps({
            "type": "auth_response",
            "status": "failure",
            "message": "Invalid API key"
        })
        
        # Test authentication
        worker.ws_connected = True  # Assume we're connected
        result = await worker._authenticate()
        
        # Verify authentication failed
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_heartbeat(self, worker_setup):
        """Test sending heartbeat to coordinator."""
        worker, mock_ws = worker_setup
        
        # Setup mock for heartbeat response
        mock_ws.recv.return_value = json.dumps({
            "type": "heartbeat_response",
            "status": "success"
        })
        
        # Test sending heartbeat
        worker.ws_connected = True  # Assume we're connected
        
        # Mock health check to avoid actual hardware checks
        with patch.object(worker, 'check_health', AsyncMock(return_value=True)):
            result = await worker.send_heartbeat()
        
        # Verify heartbeat was successful
        assert result is True
        
        # Verify heartbeat message was sent
        mock_ws.send.assert_called_once()
        
        # Get the sent message and verify it has the right format
        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "heartbeat"
        assert sent_message["worker_id"] == worker.worker_id
        assert "timestamp" in sent_message
        assert "hardware_metrics" in sent_message
    
    @pytest.mark.asyncio
    async def test_task_execution_success(self, worker_setup):
        """Test successful task execution."""
        worker, mock_ws = worker_setup
        
        # Create a benchmark task for testing
        task = {
            "task_id": "test_task_1",
            "type": "benchmark",
            "config": {
                "model": "test_model",
                "batch_sizes": [1, 2, 4],
                "precision": "fp32",
                "iterations": 3
            },
            "status": "received",
            "received": datetime.now().isoformat()
        }
        
        # Mock the _execute_benchmark_task method to avoid actual computation
        with patch.object(worker, '_execute_benchmark_task', AsyncMock(return_value={
            "model": "test_model",
            "precision": "fp32",
            "iterations": 3,
            "batch_sizes": {
                "1": {"latency_ms": 10, "throughput_items_per_second": 100, "memory_mb": 1024},
                "2": {"latency_ms": 15, "throughput_items_per_second": 133, "memory_mb": 1152},
                "4": {"latency_ms": 25, "throughput_items_per_second": 160, "memory_mb": 1280}
            }
        })), patch.object(worker, '_send_task_result', AsyncMock()):
            
            # Execute the task
            result = await worker._execute_task(task)
            
            # Verify task was completed successfully
            assert result["status"] == "completed"
            assert "ended" in result
            assert "result" in result
            
            # Verify the expected task flow
            worker._execute_benchmark_task.assert_called_once_with(task)
            worker._send_task_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_execution_failure(self, worker_setup):
        """Test failed task execution with error handling."""
        worker, mock_ws = worker_setup
        
        # Create a test task for testing
        task = {
            "task_id": "test_task_2",
            "type": "test",
            "config": {
                "test_file": "test_worker.py",
                "test_args": []
            },
            "status": "received",
            "received": datetime.now().isoformat()
        }
        
        # Mock the _execute_test_task method to raise an exception
        with patch.object(worker, '_execute_test_task', AsyncMock(side_effect=Exception("Test failed"))), \
             patch.object(worker, '_send_task_error', AsyncMock()):
            
            # Execute the task
            result = await worker._execute_task(task)
            
            # Verify task failed
            assert result["status"] == "failed"
            assert "error" in result
            assert result["error"] == "Test failed"
            
            # Verify the expected task flow
            worker._execute_test_task.assert_called_once_with(task)
            worker._send_task_error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_execution_cancelled(self, worker_setup):
        """Test cancellation of a task during execution."""
        worker, mock_ws = worker_setup
        
        # Create a custom task for testing
        task = {
            "task_id": "test_task_3",
            "type": "custom",
            "config": {
                "name": "test_custom_task"
            },
            "status": "received",
            "received": datetime.now().isoformat()
        }
        
        # Mock the _execute_custom_task method to sleep and be cancellable
        async def mock_execute_custom_task(task):
            # Simulate slow execution to allow cancellation
            try:
                await anyio.sleep(10)
                return {"name": "test_custom_task", "success": True}
            except anyio.get_cancelled_exc_class():
                raise
        
        with patch.object(worker, '_execute_custom_task', side_effect=mock_execute_custom_task):
            async with anyio.create_task_group() as tg:
                tg.start_soon(worker._execute_task, task)
                await anyio.sleep(0.1)
                tg.cancel_scope.cancel()
    
    @pytest.mark.asyncio
    async def test_check_health(self, worker_setup):
        """Test worker health check functionality."""
        worker, _ = worker_setup
        
        # Test a healthy state
        worker.ws_connected = True
        with patch('worker.psutil.cpu_percent', return_value=50), \
             patch('worker.psutil.virtual_memory', return_value=MagicMock(percent=60)):
            
            result = await worker.check_health()
            
            # Verify health status
            assert result is True
            assert worker.is_healthy is True
            assert worker.health_metrics["cpu_percent"] == 50
            assert worker.health_metrics["memory_percent"] == 60
        
        # Test an unhealthy state - high CPU
        with patch('worker.psutil.cpu_percent', return_value=96), \
             patch('worker.psutil.virtual_memory', return_value=MagicMock(percent=60)):
            
            result = await worker.check_health()
            
            # Verify health status
            assert result is False
            assert worker.is_healthy is False
            assert worker.health_metrics["cpu_percent"] == 96
        
        # Test an unhealthy state - high memory
        with patch('worker.psutil.cpu_percent', return_value=50), \
             patch('worker.psutil.virtual_memory', return_value=MagicMock(percent=96)):
            
            result = await worker.check_health()
            
            # Verify health status
            assert result is False
            assert worker.is_healthy is False
            assert worker.health_metrics["memory_percent"] == 96
        
        # Test an unhealthy state - disconnected
        worker.ws_connected = False
        with patch('worker.psutil.cpu_percent', return_value=50), \
             patch('worker.psutil.virtual_memory', return_value=MagicMock(percent=60)):
            
            result = await worker.check_health()
            
            # Verify health status
            assert result is False
            assert worker.is_healthy is False
    
    @pytest.mark.asyncio
    async def test_collect_hardware_metrics(self, worker_setup):
        """Test collection of hardware metrics."""
        worker, _ = worker_setup
        
        # Mock hardware metrics
        with patch('worker.psutil.cpu_percent', return_value=50), \
             patch('worker.psutil.virtual_memory', return_value=MagicMock(
                percent=60, 
                used=8 * 1024**3,  # 8 GB
                available=12 * 1024**3  # 12 GB
             )), \
             patch('worker.HAS_TORCH', False):  # Disable GPU metrics for this test
            
            # Collect metrics
            metrics = worker._collect_hardware_metrics()
            
            # Verify metrics
            assert metrics["cpu_percent"] == 50
            assert metrics["memory_percent"] == 60
            assert metrics["memory_used_gb"] == 8
            assert metrics["memory_available_gb"] == 12
            
            # GPU metrics should not be present
            assert "gpu" not in metrics


if __name__ == '__main__':
    unittest.main()