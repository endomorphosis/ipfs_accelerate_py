#!/usr/bin/env python3
"""
Unit tests for the integration between circuit breaker pattern and coordinator.

This module tests the integration between the circuit breaker pattern and
the coordinator service.
"""

import os
import sys
import unittest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import circuit breaker and coordinator integration
from duckdb_api.distributed_testing.circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitOpenError, CircuitBreakerRegistry
)
from duckdb_api.distributed_testing.coordinator_circuit_breaker_integration import (
    CoordinatorCircuitBreakerIntegration
)
from duckdb_api.distributed_testing.coordinator_integration import (
    integrate_circuit_breaker_with_coordinator
)


class TestCoordinatorCircuitBreakerIntegration(unittest.TestCase):
    """Test cases for the coordinator circuit breaker integration."""
    
    def setUp(self):
        """Set up a test environment."""
        # Create mock coordinator
        self.coordinator = MagicMock()
        self.coordinator.worker_manager = MagicMock()
        self.coordinator.task_manager = MagicMock()
        
        # Create coordinator circuit breaker integration
        self.integration = CoordinatorCircuitBreakerIntegration(self.coordinator)
    
    def test_initialization(self):
        """Test initialization of the integration."""
        self.assertEqual(self.integration.coordinator, self.coordinator)
        self.assertIsNotNone(self.integration.circuit_registry)
    
    def test_get_worker_circuit(self):
        """Test getting a worker circuit."""
        # Get a worker circuit
        circuit = self.integration.get_worker_circuit("worker1")
        
        # Check that the circuit was created correctly
        self.assertEqual(circuit.name, "worker_worker1")
        self.assertEqual(circuit.get_state(), CircuitState.CLOSED)
        
        # Getting the same circuit again should return the same instance
        circuit2 = self.integration.get_worker_circuit("worker1")
        self.assertEqual(circuit, circuit2)
    
    def test_get_task_circuit(self):
        """Test getting a task circuit."""
        # Get a task circuit
        circuit = self.integration.get_task_circuit("task1", "benchmark")
        
        # Check that the circuit was created correctly
        self.assertEqual(circuit.name, "task_benchmark")
        self.assertEqual(circuit.get_state(), CircuitState.CLOSED)
        
        # Getting the same circuit again should return the same instance
        circuit2 = self.integration.get_task_circuit("task2", "benchmark")
        self.assertEqual(circuit, circuit2)
    
    def test_get_endpoint_circuit(self):
        """Test getting an endpoint circuit."""
        # Get an endpoint circuit
        circuit = self.integration.get_endpoint_circuit("api/tasks")
        
        # Check that the circuit was created correctly
        self.assertEqual(circuit.name, "endpoint_api/tasks")
        self.assertEqual(circuit.get_state(), CircuitState.CLOSED)
        
        # Getting the same circuit again should return the same instance
        circuit2 = self.integration.get_endpoint_circuit("api/tasks")
        self.assertEqual(circuit, circuit2)
    
    def test_wrap_worker_execution(self):
        """Test wrapping worker execution with circuit breaker."""
        # Mock a worker circuit
        mock_circuit = MagicMock(spec=CircuitBreaker)
        mock_circuit.execute = MagicMock(return_value="result")
        
        # Replace get_worker_circuit to return our mock
        self.integration.get_worker_circuit = MagicMock(return_value=mock_circuit)
        
        # Wrap worker execution
        action = lambda: "result"
        fallback = lambda: "fallback"
        result = self.integration.wrap_worker_execution("worker1", action, fallback)
        
        # Check that the circuit's execute method was called correctly
        mock_circuit.execute.assert_called_once_with(action, fallback)
        self.assertEqual(result, "result")
    
    def test_wrap_task_execution(self):
        """Test wrapping task execution with circuit breaker."""
        # Mock a task circuit
        mock_circuit = MagicMock(spec=CircuitBreaker)
        mock_circuit.execute = MagicMock(return_value="result")
        
        # Replace get_task_circuit to return our mock
        self.integration.get_task_circuit = MagicMock(return_value=mock_circuit)
        
        # Wrap task execution
        action = lambda: "result"
        fallback = lambda: "fallback"
        result = self.integration.wrap_task_execution("task1", "benchmark", action, fallback)
        
        # Check that the circuit's execute method was called correctly
        mock_circuit.execute.assert_called_once_with(action, fallback)
        self.assertEqual(result, "result")
    
    def test_on_worker_failure(self):
        """Test handling worker failure."""
        # Mock a worker circuit
        mock_circuit = MagicMock(spec=CircuitBreaker)
        mock_circuit.execute = MagicMock(side_effect=Exception("Worker failure"))
        
        # Replace get_worker_circuit to return our mock
        self.integration.get_worker_circuit = MagicMock(return_value=mock_circuit)
        
        # Handle worker failure
        self.integration.on_worker_failure("worker1", "crash")
        
        # Check that the circuit's execute method was called
        mock_circuit.execute.assert_called_once()
    
    def test_on_task_failure(self):
        """Test handling task failure."""
        # Mock a task circuit
        mock_circuit = MagicMock(spec=CircuitBreaker)
        mock_circuit.execute = MagicMock(side_effect=Exception("Task failure"))
        
        # Replace get_task_circuit to return our mock
        self.integration.get_task_circuit = MagicMock(return_value=mock_circuit)
        
        # Handle task failure
        self.integration.on_task_failure("task1", "benchmark", "error")
        
        # Check that the circuit's execute method was called
        mock_circuit.execute.assert_called_once()
    
    def test_get_circuit_breaker_metrics(self):
        """Test getting circuit breaker metrics."""
        # Add some circuits to the registry
        worker_circuit = CircuitBreaker(name="worker_worker1")
        task_circuit = CircuitBreaker(name="task_benchmark")
        self.integration.circuit_registry.register(worker_circuit)
        self.integration.circuit_registry.register(task_circuit)
        
        # Get metrics
        metrics = self.integration.get_circuit_breaker_metrics()
        
        # Check metrics
        self.assertIn("worker_circuits", metrics)
        self.assertIn("task_circuits", metrics)
        self.assertIn("endpoint_circuits", metrics)
        self.assertIn("global_health", metrics)
        self.assertIn("last_update", metrics)
        
        # Check specific metrics
        self.assertIn("worker1", metrics["worker_circuits"])
        self.assertIn("benchmark", metrics["task_circuits"])


class TestCoordinatorIntegration(unittest.TestCase):
    """Test cases for integrating circuit breaker with coordinator."""
    
    def setUp(self):
        """Set up a test environment."""
        # Create mock coordinator
        self.coordinator = MagicMock()
        self.coordinator.worker_manager = MagicMock()
        self.coordinator.worker_manager.assign_task_to_worker = AsyncMock()
        self.coordinator.worker_manager.handle_worker_failure = AsyncMock()
        
        self.coordinator.task_manager = MagicMock()
        self.coordinator.task_manager.start_task = AsyncMock()
        self.coordinator.task_manager.handle_task_failure = AsyncMock()
        self.coordinator.task_manager.get_task_info = AsyncMock(return_value={"type": "benchmark"})
        self.coordinator.task_manager.update_task_status = AsyncMock()
        self.coordinator.task_manager.update_task_priority = AsyncMock()
    
    def test_integrate_circuit_breaker_with_coordinator(self):
        """Test integrating circuit breaker with coordinator."""
        # Integrate circuit breaker with coordinator
        result = integrate_circuit_breaker_with_coordinator(self.coordinator)
        
        # Check result
        self.assertTrue(result)
        
        # Check that circuit_breaker_integration was added to coordinator
        self.assertTrue(hasattr(self.coordinator, "circuit_breaker_integration"))
        self.assertIsInstance(self.coordinator.circuit_breaker_integration, CoordinatorCircuitBreakerIntegration)


if __name__ == "__main__":
    unittest.main()