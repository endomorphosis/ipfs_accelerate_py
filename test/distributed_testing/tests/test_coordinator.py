#!/usr/bin/env python3
"""
Tests for the Distributed Testing Framework Coordinator

This module tests the coordinator component of the distributed testing framework,
ensuring it can properly manage worker nodes, distribute tasks, and handle results.
"""

import anyio
import json
import logging
import os
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

websockets = pytest.importorskip("websockets")
pytest.importorskip("aiohttp")
from aiohttp import web

from coordinator import DistributedTestingCoordinator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDistributedTestingCoordinator(unittest.TestCase):
    """Test cases for the DistributedTestingCoordinator class."""

    def setUp(self):
        """Set up test environment."""
        # Mock database
        with patch('coordinator.duckdb.connect') as mock_duckdb_connect, \
             patch('coordinator.SecurityManager') as mock_security_manager, \
             patch('coordinator.HealthMonitor') as mock_health_monitor, \
             patch('coordinator.TaskScheduler') as mock_task_scheduler, \
             patch('coordinator.AdaptiveLoadBalancer') as mock_load_balancer, \
             patch('coordinator.PluginManager') as mock_plugin_manager:
             
            # Create coordinator instance
            self.db_path = "test_db.duckdb"
            self.coordinator = DistributedTestingCoordinator(
                db_path=self.db_path,
                host="localhost",
                port=8080
            )
            
            # Store mocks for verification
            self.mock_db = mock_duckdb_connect.return_value
            self.mock_security = self.coordinator.security_manager
            self.mock_health_monitor = self.coordinator.health_monitor
            self.mock_task_scheduler = self.coordinator.task_scheduler
            self.mock_load_balancer = self.coordinator.load_balancer
            self.mock_plugin_manager = self.coordinator.plugin_manager

    def test_init(self):
        """Test coordinator initialization."""
        # Test basic initialization
        self.assertEqual(self.coordinator.db_path, self.db_path)
        self.assertEqual(self.coordinator.host, "localhost")
        self.assertEqual(self.coordinator.port, 8080)
        
        # Verify component initialization
        self.assertIsNotNone(self.coordinator.security_manager)
        self.assertIsNotNone(self.coordinator.health_monitor)
        self.assertIsNotNone(self.coordinator.task_scheduler)
        self.assertIsNotNone(self.coordinator.load_balancer)
        self.assertIsNotNone(self.coordinator.plugin_manager)
        
        # Verify initial state
        self.assertEqual(len(self.coordinator.workers), 0)
        self.assertEqual(len(self.coordinator.tasks), 0)
        self.assertEqual(len(self.coordinator.pending_tasks), 0)
        self.assertEqual(len(self.coordinator.running_tasks), 0)
        self.assertEqual(len(self.coordinator.completed_tasks), 0)
        self.assertEqual(len(self.coordinator.failed_tasks), 0)
    
    def test_init_with_disabled_components(self):
        """Test coordinator initialization with disabled components."""
        with patch('coordinator.duckdb.connect') as mock_duckdb_connect, \
             patch('coordinator.SecurityManager') as mock_security_manager:
             
            # Create coordinator with disabled components
            coordinator = DistributedTestingCoordinator(
                db_path=self.db_path,
                host="localhost",
                port=8080,
                enable_advanced_scheduler=False,
                enable_health_monitor=False,
                enable_load_balancer=False,
                enable_auto_recovery=False,
                enable_redundancy=False,
                enable_plugins=False,
                enable_enhanced_error_handling=False
            )
            
            # Verify component initialization
            self.assertIsNotNone(coordinator.security_manager)
            self.assertIsNone(coordinator.health_monitor)
            self.assertIsNone(coordinator.task_scheduler)
            self.assertIsNone(coordinator.load_balancer)
            self.assertIsNone(coordinator.plugin_manager)


@pytest.fixture
async def coordinator_setup():
    """Set up coordinator for async tests with mocked dependencies."""
    # Mock all dependencies
    with patch('coordinator.duckdb.connect'), \
         patch('coordinator.SecurityManager') as mock_security_manager, \
         patch('coordinator.HealthMonitor'), \
         patch('coordinator.TaskScheduler'), \
         patch('coordinator.AdaptiveLoadBalancer'), \
         patch('coordinator.PluginManager'):
         
        # Create security manager mock
        mock_security = mock_security_manager.return_value
        mock_security.verify_token = AsyncMock(return_value=True)
        mock_security.verify_api_key = AsyncMock(return_value=True)
        mock_security.generate_token = AsyncMock(return_value="test_token")
        mock_security.sign_message = lambda msg: {**msg, "signature": "test_signature"}
        mock_security.verify_message = lambda msg: True
        
        # Create coordinator instance
        coordinator = DistributedTestingCoordinator(
            db_path="test_db.duckdb",
            host="localhost",
            port=8080
        )
        
        # Setup initial state for testing
        coordinator.security_manager = mock_security
        
        yield coordinator


class TestCoordinatorAsync:
    """Async test cases for the DistributedTestingCoordinator class."""
    
    @pytest.mark.asyncio
    async def test_handle_worker_registration(self, coordinator_setup):
        """Test handling worker registration."""
        coordinator = coordinator_setup
        
        # Mock websocket
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.receive_json = AsyncMock(return_value={
            "type": "register",
            "worker_id": "test_worker_id",
            "hostname": "test_host",
            "capabilities": {
                "hardware": ["cpu", "cuda"],
                "cpu": {"cores": 8, "threads": 16},
                "gpu": {"name": "Test GPU", "memory_gb": 8}
            }
        })
        
        # Test registration handler with patch to avoid web dependencies
        with patch.object(coordinator, '_send_response', AsyncMock()):
            # Call the registration handler
            await coordinator._handle_worker_registration(mock_ws, {
                "type": "register",
                "worker_id": "test_worker_id",
                "hostname": "test_host",
                "capabilities": {
                    "hardware": ["cpu", "cuda"],
                    "cpu": {"cores": 8, "threads": 16},
                    "gpu": {"name": "Test GPU", "memory_gb": 8}
                }
            })
            
            # Verify worker was registered
            assert "test_worker_id" in coordinator.workers
            assert coordinator.workers["test_worker_id"]["hostname"] == "test_host"
            assert "cpu" in coordinator.workers["test_worker_id"]["capabilities"]["hardware"]
            assert "cuda" in coordinator.workers["test_worker_id"]["capabilities"]["hardware"]
            
            # Verify response was sent
            coordinator._send_response.assert_called_once()
            
            # Verify registration response format
            call_args = coordinator._send_response.call_args[0]
            assert call_args[0] == mock_ws
            assert call_args[1]["type"] == "register_response"
            assert call_args[1]["status"] == "success"
            assert call_args[1]["worker_id"] == "test_worker_id"
    
    @pytest.mark.asyncio
    async def test_handle_worker_heartbeat(self, coordinator_setup):
        """Test handling worker heartbeat."""
        coordinator = coordinator_setup
        
        # Register a worker
        worker_id = "test_worker_id"
        coordinator.workers[worker_id] = {
            "worker_id": worker_id,
            "hostname": "test_host",
            "capabilities": {
                "hardware": ["cpu", "cuda"],
                "cpu": {"cores": 8, "threads": 16},
                "gpu": {"name": "Test GPU", "memory_gb": 8}
            },
            "status": "idle",
            "last_heartbeat": datetime.now().isoformat(),
            "connected": True
        }
        
        # Create heartbeat message
        heartbeat = {
            "type": "heartbeat",
            "worker_id": worker_id,
            "timestamp": datetime.now().isoformat(),
            "hardware_metrics": {
                "cpu_percent": 50,
                "memory_percent": 60,
                "gpu": [{"index": 0, "memory_utilization_percent": 30}]
            },
            "health_status": {
                "is_healthy": True,
                "health_metrics": {
                    "resource_healthy": True,
                    "connection_healthy": True
                }
            }
        }
        
        # Mock websocket
        mock_ws = AsyncMock()
        
        # Test heartbeat handler with patch to avoid web dependencies
        with patch.object(coordinator, '_send_response', AsyncMock()):
            # Call the heartbeat handler
            await coordinator._handle_worker_heartbeat(mock_ws, heartbeat)
            
            # Verify heartbeat was processed
            assert coordinator.workers[worker_id]["status"] == "idle"
            assert "last_heartbeat" in coordinator.workers[worker_id]
            assert coordinator.workers[worker_id]["hardware_metrics"]["cpu_percent"] == 50
            assert coordinator.workers[worker_id]["hardware_metrics"]["memory_percent"] == 60
            assert coordinator.workers[worker_id]["health_status"]["is_healthy"] is True
            
            # Verify response was sent
            coordinator._send_response.assert_called_once()
            
            # Verify heartbeat response format
            call_args = coordinator._send_response.call_args[0]
            assert call_args[0] == mock_ws
            assert call_args[1]["type"] == "heartbeat_response"
            assert call_args[1]["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_handle_task_result(self, coordinator_setup):
        """Test handling task result."""
        coordinator = coordinator_setup
        
        # Register a worker
        worker_id = "test_worker_id"
        coordinator.workers[worker_id] = {
            "worker_id": worker_id,
            "hostname": "test_host",
            "capabilities": {
                "hardware": ["cpu", "cuda"],
                "cpu": {"cores": 8, "threads": 16},
                "gpu": {"name": "Test GPU", "memory_gb": 8}
            },
            "status": "busy",
            "last_heartbeat": datetime.now().isoformat(),
            "connected": True
        }
        
        # Create a task
        task_id = "test_task_id"
        task = {
            "task_id": task_id,
            "type": "benchmark",
            "status": "running",
            "worker_id": worker_id,
            "config": {
                "model": "test_model",
                "batch_sizes": [1, 2, 4]
            },
            "created": datetime.now().isoformat(),
            "started": datetime.now().isoformat()
        }
        coordinator.tasks[task_id] = task
        coordinator.running_tasks[task_id] = worker_id
        
        # Create task result message
        result = {
            "type": "task_result",
            "worker_id": worker_id,
            "task_id": task_id,
            "status": "completed",
            "execution_time_seconds": 10.5,
            "hardware_metrics": {
                "cpu_percent": 50,
                "memory_percent": 60
            },
            "result": {
                "model": "test_model",
                "precision": "fp32",
                "iterations": 3,
                "batch_sizes": {
                    "1": {"latency_ms": 10, "throughput_items_per_second": 100, "memory_mb": 1024},
                    "2": {"latency_ms": 15, "throughput_items_per_second": 133, "memory_mb": 1152},
                    "4": {"latency_ms": 25, "throughput_items_per_second": 160, "memory_mb": 1280}
                }
            }
        }
        
        # Mock websocket
        mock_ws = AsyncMock()
        
        # Test task result handler with patches to avoid web dependencies
        with patch.object(coordinator, '_send_response', AsyncMock()), \
             patch.object(coordinator, '_save_task_result', AsyncMock()), \
             patch.object(coordinator, '_notify_task_completion', AsyncMock()):
            
            # Call the task result handler
            await coordinator._handle_task_result(mock_ws, result)
            
            # Verify task was updated
            assert coordinator.tasks[task_id]["status"] == "completed"
            assert "completed" in coordinator.tasks[task_id]
            assert "execution_time_seconds" in coordinator.tasks[task_id]
            assert "result" in coordinator.tasks[task_id]
            
            # Verify task tracking was updated
            assert task_id in coordinator.completed_tasks
            assert task_id not in coordinator.running_tasks
            
            # Verify worker status was updated
            assert coordinator.workers[worker_id]["status"] == "idle"
            
            # Verify results were saved
            coordinator._save_task_result.assert_called_once()
            
            # Verify notifications were sent
            coordinator._notify_task_completion.assert_called_once()
            
            # Verify response was sent
            coordinator._send_response.assert_called_once()
            
            # Verify task result response format
            call_args = coordinator._send_response.call_args[0]
            assert call_args[0] == mock_ws
            assert call_args[1]["type"] == "task_result_response"
            assert call_args[1]["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_handle_task_error(self, coordinator_setup):
        """Test handling task error."""
        coordinator = coordinator_setup
        
        # Register a worker
        worker_id = "test_worker_id"
        coordinator.workers[worker_id] = {
            "worker_id": worker_id,
            "hostname": "test_host",
            "capabilities": {"hardware": ["cpu"]},
            "status": "busy",
            "last_heartbeat": datetime.now().isoformat(),
            "connected": True
        }
        
        # Create a task
        task_id = "test_task_id"
        task = {
            "task_id": task_id,
            "type": "test",
            "status": "running",
            "worker_id": worker_id,
            "config": {
                "test_file": "test_worker.py"
            },
            "created": datetime.now().isoformat(),
            "started": datetime.now().isoformat()
        }
        coordinator.tasks[task_id] = task
        coordinator.running_tasks[task_id] = worker_id
        
        # Create task error message
        error_result = {
            "type": "task_result",
            "worker_id": worker_id,
            "task_id": task_id,
            "status": "failed",
            "execution_time_seconds": 5.2,
            "hardware_metrics": {
                "cpu_percent": 50,
                "memory_percent": 60
            },
            "error": "Test execution failed with error: AssertionError"
        }
        
        # Mock websocket
        mock_ws = AsyncMock()
        
        # Test task error handler with patches to avoid web dependencies
        with patch.object(coordinator, '_send_response', AsyncMock()), \
             patch.object(coordinator, '_save_task_result', AsyncMock()), \
             patch.object(coordinator, '_notify_task_failure', AsyncMock()):
            
            # Call the task result handler
            await coordinator._handle_task_result(mock_ws, error_result)
            
            # Verify task was updated
            assert coordinator.tasks[task_id]["status"] == "failed"
            assert "completed" in coordinator.tasks[task_id]
            assert "execution_time_seconds" in coordinator.tasks[task_id]
            assert "error" in coordinator.tasks[task_id]
            
            # Verify task tracking was updated
            assert task_id in coordinator.failed_tasks
            assert task_id not in coordinator.running_tasks
            
            # Verify worker status was updated
            assert coordinator.workers[worker_id]["status"] == "idle"
            
            # Verify results were saved
            coordinator._save_task_result.assert_called_once()
            
            # Verify notifications were sent
            coordinator._notify_task_failure.assert_called_once()
            
            # Verify response was sent
            coordinator._send_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_authenticate_worker_api_key(self, coordinator_setup):
        """Test worker authentication with API key."""
        coordinator = coordinator_setup
        
        # Mock websocket
        mock_ws = AsyncMock()
        
        # Mock API key authentication
        coordinator.security_manager.verify_api_key.return_value = True
        
        # Create auth message
        auth_message = {
            "type": "auth",
            "auth_type": "api_key",
            "api_key": "test_api_key",
            "worker_id": "test_worker_id"
        }
        
        # Test authentication handler
        with patch.object(coordinator, '_send_response', AsyncMock()):
            # Call the authentication handler
            result = await coordinator._authenticate_worker(mock_ws, auth_message)
            
            # Verify authentication was successful
            assert result is True
            
            # Verify API key was verified
            coordinator.security_manager.verify_api_key.assert_called_once_with("test_api_key")
            
            # Verify token was generated
            coordinator.security_manager.generate_token.assert_called_once()
            
            # Verify response was sent
            coordinator._send_response.assert_called_once()
            
            # Verify auth response format
            call_args = coordinator._send_response.call_args[0]
            assert call_args[0] == mock_ws
            assert call_args[1]["type"] == "auth_response"
            assert call_args[1]["status"] == "success"
            assert "token" in call_args[1]
    
    @pytest.mark.asyncio
    async def test_authenticate_worker_token(self, coordinator_setup):
        """Test worker authentication with token."""
        coordinator = coordinator_setup
        
        # Mock websocket
        mock_ws = AsyncMock()
        
        # Mock token authentication
        coordinator.security_manager.verify_token.return_value = True
        
        # Create auth message
        auth_message = {
            "type": "auth",
            "auth_type": "token",
            "token": "test_token"
        }
        
        # Test authentication handler
        with patch.object(coordinator, '_send_response', AsyncMock()):
            # Call the authentication handler
            result = await coordinator._authenticate_worker(mock_ws, auth_message)
            
            # Verify authentication was successful
            assert result is True
            
            # Verify token was verified
            coordinator.security_manager.verify_token.assert_called_once_with("test_token")
            
            # Verify response was sent
            coordinator._send_response.assert_called_once()
            
            # Verify auth response format
            call_args = coordinator._send_response.call_args[0]
            assert call_args[0] == mock_ws
            assert call_args[1]["type"] == "auth_response"
            assert call_args[1]["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_authenticate_worker_failed(self, coordinator_setup):
        """Test failed worker authentication."""
        coordinator = coordinator_setup
        
        # Mock websocket
        mock_ws = AsyncMock()
        
        # Mock failed API key authentication
        coordinator.security_manager.verify_api_key.return_value = False
        
        # Create auth message
        auth_message = {
            "type": "auth",
            "auth_type": "api_key",
            "api_key": "invalid_api_key",
            "worker_id": "test_worker_id"
        }
        
        # Test authentication handler
        with patch.object(coordinator, '_send_response', AsyncMock()):
            # Call the authentication handler
            result = await coordinator._authenticate_worker(mock_ws, auth_message)
            
            # Verify authentication failed
            assert result is False
            
            # Verify API key was verified
            coordinator.security_manager.verify_api_key.assert_called_once_with("invalid_api_key")
            
            # Verify response was sent
            coordinator._send_response.assert_called_once()
            
            # Verify auth response format
            call_args = coordinator._send_response.call_args[0]
            assert call_args[0] == mock_ws
            assert call_args[1]["type"] == "auth_response"
            assert call_args[1]["status"] == "failure"
            assert "message" in call_args[1]
    
    @pytest.mark.asyncio
    async def test_find_worker_for_task(self, coordinator_setup):
        """Test finding a worker for a task."""
        coordinator = coordinator_setup
        
        # Add multiple workers with different capabilities
        coordinator.workers = {
            "worker1": {
                "worker_id": "worker1",
                "status": "idle",
                "capabilities": {
                    "hardware": ["cpu"],
                    "cpu": {"cores": 4}
                },
                "connected": True,
                "health_status": {"is_healthy": True}
            },
            "worker2": {
                "worker_id": "worker2",
                "status": "idle",
                "capabilities": {
                    "hardware": ["cpu", "cuda"],
                    "cpu": {"cores": 8},
                    "gpu": {"name": "RTX 3080", "memory_gb": 10}
                },
                "connected": True,
                "health_status": {"is_healthy": True}
            },
            "worker3": {
                "worker_id": "worker3",
                "status": "busy",
                "capabilities": {
                    "hardware": ["cpu", "cuda"],
                    "cpu": {"cores": 12},
                    "gpu": {"name": "RTX 3090", "memory_gb": 24}
                },
                "connected": True,
                "health_status": {"is_healthy": True}
            }
        }
        
        # Create a CPU task
        cpu_task = {
            "task_id": "cpu_task",
            "type": "benchmark",
            "config": {
                "model": "model1",
                "hardware": "cpu"
            }
        }
        
        # Create a GPU task
        gpu_task = {
            "task_id": "gpu_task",
            "type": "benchmark",
            "config": {
                "model": "model2",
                "hardware": "cuda"
            }
        }
        
        # Test with coordinator that has a load balancer
        with patch.object(coordinator.load_balancer, 'select_worker_for_task', 
                         side_effect=lambda task, workers: "worker2" if "cuda" in task["config"]["hardware"] else "worker1"):
            
            # Find worker for CPU task
            worker_id = coordinator._find_worker_for_task(cpu_task)
            assert worker_id == "worker1"
            
            # Find worker for GPU task
            worker_id = coordinator._find_worker_for_task(gpu_task)
            assert worker_id == "worker2"
    
    @pytest.mark.asyncio
    async def test_assign_task_to_worker(self, coordinator_setup):
        """Test assigning a task to a worker."""
        coordinator = coordinator_setup
        
        # Add a worker
        worker_id = "test_worker_id"
        coordinator.workers[worker_id] = {
            "worker_id": worker_id,
            "hostname": "test_host",
            "capabilities": {"hardware": ["cpu", "cuda"]},
            "status": "idle",
            "last_heartbeat": datetime.now().isoformat(),
            "connected": True,
            "ws": AsyncMock()  # Mock websocket
        }
        
        # Create a task
        task_id = "test_task_id"
        task = {
            "task_id": task_id,
            "type": "benchmark",
            "status": "pending",
            "config": {
                "model": "test_model",
                "batch_sizes": [1, 2, 4]
            },
            "created": datetime.now().isoformat()
        }
        coordinator.tasks[task_id] = task
        coordinator.pending_tasks.add(task_id)
        
        # Test assign task with a patch for _send_task
        with patch.object(coordinator, '_send_task', AsyncMock(return_value=True)):
            # Assign task to worker
            success = await coordinator._assign_task_to_worker(task, worker_id)
            
            # Verify assignment was successful
            assert success is True
            
            # Verify task state was updated
            assert coordinator.tasks[task_id]["status"] == "assigned"
            assert coordinator.tasks[task_id]["worker_id"] == worker_id
            assert "assigned" in coordinator.tasks[task_id]
            
            # Verify task tracking was updated
            assert task_id not in coordinator.pending_tasks
            assert task_id in coordinator.running_tasks
            assert coordinator.running_tasks[task_id] == worker_id
            
            # Verify worker status was updated
            assert coordinator.workers[worker_id]["status"] == "busy"
            
            # Verify task was sent to worker
            coordinator._send_task.assert_called_once()
            
            # Verify send_task parameters
            call_args = coordinator._send_task.call_args[0]
            assert call_args[0] == coordinator.workers[worker_id]["ws"]
            assert call_args[1]["task_id"] == task_id
            assert call_args[1]["task_type"] == task["type"]


if __name__ == '__main__':
    unittest.main()