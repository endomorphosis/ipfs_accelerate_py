#!/usr/bin/env python3
"""
Unit tests for the integration between circuit breaker pattern and fault tolerance.

This module tests the integration between the circuit breaker pattern and
the hardware-aware fault tolerance system.
"""

import os
import sys
import unittest
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import circuit breaker and fault tolerance
from data.duckdb.distributed_testing.circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitOpenError, CircuitBreakerRegistry
)
from data.duckdb.distributed_testing.hardware_aware_fault_tolerance import (
    HardwareAwareFaultToleranceManager, FailureContext, RecoveryAction, RecoveryStrategy, FailureType
)
from data.duckdb.distributed_testing.fault_tolerance_integration import (
    CircuitBreakerIntegration, create_fault_tolerance_integration, apply_recovery_with_circuit_breaker
)


class TestFaultToleranceIntegration(unittest.TestCase):
    """Test cases for the fault tolerance integration."""
    
    def setUp(self):
        """Set up a test environment."""
        # Create mocks
        self.db_manager = MagicMock()
        self.coordinator = MagicMock()
        self.task_scheduler = MagicMock()
        
        # Create hardware-aware fault tolerance manager
        self.fault_tolerance_manager = MagicMock(spec=HardwareAwareFaultToleranceManager)
        self.fault_tolerance_manager._determine_fallback_hardware_class = MagicMock(return_value="CPU")
        
        # Create fault tolerance integration
        self.integration = CircuitBreakerIntegration(self.fault_tolerance_manager)
    
    def test_initialization(self):
        """Test initialization of the integration."""
        self.assertIsNotNone(self.integration)
        self.assertEqual(self.integration.fault_tolerance_manager, self.fault_tolerance_manager)
        self.assertIsNotNone(self.integration.circuit_registry)
        self.assertIn("failure_threshold", self.integration.worker_circuit_config)
        self.assertIn("failure_threshold", self.integration.hardware_circuit_config)
        self.assertIn("failure_threshold", self.integration.task_type_circuit_config)
    
    def test_get_worker_circuit(self):
        """Test getting a worker circuit breaker."""
        circuit = self.integration.get_worker_circuit("worker1")
        self.assertEqual(circuit.name, "worker_worker1")
        self.assertEqual(circuit.state, CircuitState.CLOSED)
        
        # Should get the same circuit breaker for the same worker
        circuit2 = self.integration.get_worker_circuit("worker1")
        self.assertIs(circuit, circuit2)
    
    def test_get_hardware_circuit(self):
        """Test getting a hardware circuit breaker."""
        circuit = self.integration.get_hardware_circuit("GPU")
        self.assertEqual(circuit.name, "hardware_GPU")
        self.assertEqual(circuit.state, CircuitState.CLOSED)
        
        # Should get the same circuit breaker for the same hardware class
        circuit2 = self.integration.get_hardware_circuit("GPU")
        self.assertIs(circuit, circuit2)
    
    def test_get_task_type_circuit(self):
        """Test getting a task type circuit breaker."""
        circuit = self.integration.get_task_type_circuit("benchmark")
        self.assertEqual(circuit.name, "task_type_benchmark")
        self.assertEqual(circuit.state, CircuitState.CLOSED)
        
        # Should get the same circuit breaker for the same task type
        circuit2 = self.integration.get_task_type_circuit("benchmark")
        self.assertIs(circuit, circuit2)
    
    def test_handle_failure_no_open_circuits(self):
        """Test handling a failure with no open circuits."""
        # Mock the fault tolerance manager
        mock_recovery_action = RecoveryAction(
            strategy=RecoveryStrategy.DELAYED_RETRY,
            message="Test recovery action"
        )
        self.fault_tolerance_manager._determine_recovery_strategy = MagicMock(return_value=mock_recovery_action)
        self.fault_tolerance_manager._get_task = MagicMock(return_value={"type": "benchmark"})
        
        # Create a failure context
        failure_context = FailureContext(
            task_id="task1",
            worker_id="worker1",
            error_type=FailureType.SOFTWARE_ERROR,
            error_message="Test error"
        )
        
        # Handle the failure
        recovery_action = self.integration.handle_failure(failure_context)
        
        # Should use the default recovery strategy
        self.assertEqual(recovery_action.strategy, RecoveryStrategy.DELAYED_RETRY)
        self.assertEqual(recovery_action.message, "Test recovery action")
        
        # Should have called determine_recovery_strategy
        self.fault_tolerance_manager._determine_recovery_strategy.assert_called_once_with(failure_context)
    
    def test_handle_failure_worker_circuit_open(self):
        """Test handling a failure with worker circuit open."""
        # Mock the fault tolerance manager
        self.fault_tolerance_manager._get_task = MagicMock(return_value={"type": "benchmark"})
        
        # Get the worker circuit and open it
        worker_circuit = self.integration.get_worker_circuit("worker1")
        worker_circuit.state = CircuitState.OPEN
        
        # Create a failure context
        failure_context = FailureContext(
            task_id="task1",
            worker_id="worker1",
            error_type=FailureType.SOFTWARE_ERROR,
            error_message="Test error"
        )
        
        # Handle the failure
        recovery_action = self.integration.handle_failure(failure_context)
        
        # Should override the recovery strategy for worker circuit
        self.assertEqual(recovery_action.strategy, RecoveryStrategy.DIFFERENT_WORKER)
        self.assertIn("worker", recovery_action.message)
        
        # Should not have called determine_recovery_strategy
        self.fault_tolerance_manager._determine_recovery_strategy.assert_not_called()
    
    def test_handle_failure_hardware_circuit_open(self):
        """Test handling a failure with hardware circuit open."""
        # Mock the fault tolerance manager
        self.fault_tolerance_manager._get_task = MagicMock(return_value={"type": "benchmark"})
        
        # Create a hardware profile mock
        hardware_profile = MagicMock()
        hardware_profile.hardware_class.name = "GPU"
        
        # Get the hardware circuit and open it
        hardware_circuit = self.integration.get_hardware_circuit("GPU")
        hardware_circuit.state = CircuitState.OPEN
        
        # Create a failure context
        failure_context = FailureContext(
            task_id="task1",
            worker_id="worker1",
            hardware_profile=hardware_profile,
            error_type=FailureType.HARDWARE_ERROR,
            error_message="Test error"
        )
        
        # Handle the failure
        recovery_action = self.integration.handle_failure(failure_context)
        
        # Should override the recovery strategy for hardware circuit
        self.assertEqual(recovery_action.strategy, RecoveryStrategy.DIFFERENT_HARDWARE_CLASS)
        self.assertIn("hardware", recovery_action.message)
        self.assertIn("hardware", recovery_action.hardware_requirements)
        
        # Should have called _determine_fallback_hardware_class
        self.fault_tolerance_manager._determine_fallback_hardware_class.assert_called_once_with("GPU")
    
    def test_handle_failure_task_type_circuit_open(self):
        """Test handling a failure with task type circuit open."""
        # Mock the fault tolerance manager
        self.fault_tolerance_manager._get_task = MagicMock(return_value={"type": "benchmark"})
        
        # Get the task type circuit and open it
        task_type_circuit = self.integration.get_task_type_circuit("benchmark")
        task_type_circuit.state = CircuitState.OPEN
        
        # Create a failure context
        failure_context = FailureContext(
            task_id="task1",
            worker_id="worker1",
            error_type=FailureType.SOFTWARE_ERROR,
            error_message="Test error"
        )
        
        # Handle the failure
        recovery_action = self.integration.handle_failure(failure_context)
        
        # Should override the recovery strategy for task type circuit
        self.assertEqual(recovery_action.strategy, RecoveryStrategy.DELAYED_RETRY)
        self.assertIn("task type", recovery_action.message)
        self.assertEqual(recovery_action.delay, 60.0)
    
    def test_track_failure(self):
        """Test tracking a failure in circuit breakers."""
        # Mock the fault tolerance manager
        self.fault_tolerance_manager._get_task = MagicMock(return_value={"type": "benchmark"})
        
        # Create a hardware profile mock
        hardware_profile = MagicMock()
        hardware_profile.hardware_class.name = "GPU"
        
        # Create a failure context
        failure_context = FailureContext(
            task_id="task1",
            worker_id="worker1",
            hardware_profile=hardware_profile,
            error_type=FailureType.HARDWARE_ERROR,
            error_message="Test error"
        )
        
        # Track the failure
        self.integration._track_failure(failure_context)
        
        # Get the circuit breakers
        worker_circuit = self.integration.get_worker_circuit("worker1")
        hardware_circuit = self.integration.get_hardware_circuit("GPU")
        task_type_circuit = self.integration.get_task_type_circuit("benchmark")
        
        # Hardware circuit should have been updated for HARDWARE_ERROR
        self.assertEqual(hardware_circuit.total_failures, 1)
        self.assertEqual(hardware_circuit.failure_count, 1)
        
        # Other circuits should not have been updated
        self.assertEqual(worker_circuit.total_failures, 0)
        self.assertEqual(worker_circuit.failure_count, 0)
        self.assertEqual(task_type_circuit.total_failures, 0)
        self.assertEqual(task_type_circuit.failure_count, 0)
    
    def test_track_success(self):
        """Test tracking a successful execution."""
        # Track success
        self.integration.track_success(
            task_id="task1",
            worker_id="worker1",
            hardware_class="GPU",
            task_type="benchmark"
        )
        
        # Get the circuit breakers
        worker_circuit = self.integration.get_worker_circuit("worker1")
        hardware_circuit = self.integration.get_hardware_circuit("GPU")
        task_type_circuit = self.integration.get_task_type_circuit("benchmark")
        
        # All circuits should have been updated
        self.assertEqual(worker_circuit.total_successes, 1)
        self.assertEqual(hardware_circuit.total_successes, 1)
        self.assertEqual(task_type_circuit.total_successes, 1)
    
    def test_reset_worker_circuit(self):
        """Test resetting a worker circuit breaker."""
        # Get the worker circuit and open it
        worker_circuit = self.integration.get_worker_circuit("worker1")
        worker_circuit.state = CircuitState.OPEN
        worker_circuit.failure_count = 5
        
        # Reset the circuit
        self.integration.reset_worker_circuit("worker1")
        
        # Circuit should be closed and reset
        self.assertEqual(worker_circuit.state, CircuitState.CLOSED)
        self.assertEqual(worker_circuit.failure_count, 0)
    
    def test_reset_hardware_circuit(self):
        """Test resetting a hardware circuit breaker."""
        # Get the hardware circuit and open it
        hardware_circuit = self.integration.get_hardware_circuit("GPU")
        hardware_circuit.state = CircuitState.OPEN
        hardware_circuit.failure_count = 5
        
        # Reset the circuit
        self.integration.reset_hardware_circuit("GPU")
        
        # Circuit should be closed and reset
        self.assertEqual(hardware_circuit.state, CircuitState.CLOSED)
        self.assertEqual(hardware_circuit.failure_count, 0)
    
    def test_reset_task_type_circuit(self):
        """Test resetting a task type circuit breaker."""
        # Get the task type circuit and open it
        task_type_circuit = self.integration.get_task_type_circuit("benchmark")
        task_type_circuit.state = CircuitState.OPEN
        task_type_circuit.failure_count = 5
        
        # Reset the circuit
        self.integration.reset_task_type_circuit("benchmark")
        
        # Circuit should be closed and reset
        self.assertEqual(task_type_circuit.state, CircuitState.CLOSED)
        self.assertEqual(task_type_circuit.failure_count, 0)
    
    def test_reset_all_circuits(self):
        """Test resetting all circuit breakers."""
        # Get circuit breakers and open them
        worker_circuit = self.integration.get_worker_circuit("worker1")
        hardware_circuit = self.integration.get_hardware_circuit("GPU")
        task_type_circuit = self.integration.get_task_type_circuit("benchmark")
        
        worker_circuit.state = CircuitState.OPEN
        hardware_circuit.state = CircuitState.OPEN
        task_type_circuit.state = CircuitState.OPEN
        
        # Reset all circuits
        self.integration.reset_all_circuits()
        
        # All circuits should be closed and reset
        self.assertEqual(worker_circuit.state, CircuitState.CLOSED)
        self.assertEqual(hardware_circuit.state, CircuitState.CLOSED)
        self.assertEqual(task_type_circuit.state, CircuitState.CLOSED)
    
    def test_get_health_metrics(self):
        """Test getting health metrics for all circuit breakers."""
        # Get circuit breakers and add some activity
        worker_circuit = self.integration.get_worker_circuit("worker1")
        hardware_circuit = self.integration.get_hardware_circuit("GPU")
        task_type_circuit = self.integration.get_task_type_circuit("benchmark")
        
        worker_circuit._on_success()
        hardware_circuit._on_failure()
        task_type_circuit._on_success()
        
        # Get health metrics
        metrics = self.integration.get_health_metrics()
        
        # Should have correct structure
        self.assertIn("aggregate", metrics)
        self.assertIn("workers", metrics)
        self.assertIn("hardware_classes", metrics)
        self.assertIn("task_types", metrics)
        self.assertIn("timestamp", metrics)
        
        # Should have correct data
        self.assertIn("worker1", metrics["workers"])
        self.assertIn("GPU", metrics["hardware_classes"])
        self.assertIn("benchmark", metrics["task_types"])
        
        self.assertEqual(metrics["workers"]["worker1"]["total_successes"], 1)
        self.assertEqual(metrics["hardware_classes"]["GPU"]["total_failures"], 1)
        self.assertEqual(metrics["task_types"]["benchmark"]["total_successes"], 1)
    
    def test_get_worker_health(self):
        """Test getting health metrics for a specific worker."""
        # Get the worker circuit and add some activity
        worker_circuit = self.integration.get_worker_circuit("worker1")
        worker_circuit._on_success()
        
        # Get worker health metrics
        metrics = self.integration.get_worker_health("worker1")
        
        # Should have correct data
        self.assertEqual(metrics["name"], "worker_worker1")
        self.assertEqual(metrics["total_successes"], 1)
        self.assertEqual(metrics["state"], CircuitState.CLOSED.name)
    
    def test_get_hardware_health(self):
        """Test getting health metrics for a specific hardware class."""
        # Get the hardware circuit and add some activity
        hardware_circuit = self.integration.get_hardware_circuit("GPU")
        hardware_circuit._on_failure()
        
        # Get hardware health metrics
        metrics = self.integration.get_hardware_health("GPU")
        
        # Should have correct data
        self.assertEqual(metrics["name"], "hardware_GPU")
        self.assertEqual(metrics["total_failures"], 1)
        self.assertEqual(metrics["state"], CircuitState.CLOSED.name)
    
    def test_get_task_type_health(self):
        """Test getting health metrics for a specific task type."""
        # Get the task type circuit and add some activity
        task_type_circuit = self.integration.get_task_type_circuit("benchmark")
        task_type_circuit._on_success()
        
        # Get task type health metrics
        metrics = self.integration.get_task_type_health("benchmark")
        
        # Should have correct data
        self.assertEqual(metrics["name"], "task_type_benchmark")
        self.assertEqual(metrics["total_successes"], 1)
        self.assertEqual(metrics["state"], CircuitState.CLOSED.name)
    
    def test_configure_worker_circuits(self):
        """Test configuring worker circuit breakers."""
        # Configure worker circuits
        self.integration.configure_worker_circuits({
            "failure_threshold": 10,
            "recovery_timeout": 120.0
        })
        
        # Configuration should be updated
        self.assertEqual(self.integration.worker_circuit_config["failure_threshold"], 10)
        self.assertEqual(self.integration.worker_circuit_config["recovery_timeout"], 120.0)
    
    def test_configure_hardware_circuits(self):
        """Test configuring hardware circuit breakers."""
        # Configure hardware circuits
        self.integration.configure_hardware_circuits({
            "failure_threshold": 5,
            "recovery_timeout": 300.0
        })
        
        # Configuration should be updated
        self.assertEqual(self.integration.hardware_circuit_config["failure_threshold"], 5)
        self.assertEqual(self.integration.hardware_circuit_config["recovery_timeout"], 300.0)
    
    def test_configure_task_type_circuits(self):
        """Test configuring task type circuit breakers."""
        # Configure task type circuits
        self.integration.configure_task_type_circuits({
            "failure_threshold": 15,
            "recovery_timeout": 600.0
        })
        
        # Configuration should be updated
        self.assertEqual(self.integration.task_type_circuit_config["failure_threshold"], 15)
        self.assertEqual(self.integration.task_type_circuit_config["recovery_timeout"], 600.0)
    
    def test_is_worker_circuit_open(self):
        """Test checking if a worker circuit breaker is open."""
        # Get the worker circuit and leave it closed
        worker_circuit = self.integration.get_worker_circuit("worker1")
        
        # Should be closed
        self.assertFalse(self.integration.is_worker_circuit_open("worker1"))
        
        # Open the circuit
        worker_circuit.state = CircuitState.OPEN
        
        # Should be open
        self.assertTrue(self.integration.is_worker_circuit_open("worker1"))
    
    def test_is_hardware_circuit_open(self):
        """Test checking if a hardware circuit breaker is open."""
        # Get the hardware circuit and leave it closed
        hardware_circuit = self.integration.get_hardware_circuit("GPU")
        
        # Should be closed
        self.assertFalse(self.integration.is_hardware_circuit_open("GPU"))
        
        # Open the circuit
        hardware_circuit.state = CircuitState.OPEN
        
        # Should be open
        self.assertTrue(self.integration.is_hardware_circuit_open("GPU"))
    
    def test_is_task_type_circuit_open(self):
        """Test checking if a task type circuit breaker is open."""
        # Get the task type circuit and leave it closed
        task_type_circuit = self.integration.get_task_type_circuit("benchmark")
        
        # Should be closed
        self.assertFalse(self.integration.is_task_type_circuit_open("benchmark"))
        
        # Open the circuit
        task_type_circuit.state = CircuitState.OPEN
        
        # Should be open
        self.assertTrue(self.integration.is_task_type_circuit_open("benchmark"))
    
    def test_create_fault_tolerance_integration(self):
        """Test creating a fault tolerance integration."""
        # Create a fault tolerance integration
        integration = create_fault_tolerance_integration(self.fault_tolerance_manager)
        
        # Should be a CircuitBreakerIntegration instance
        self.assertIsInstance(integration, CircuitBreakerIntegration)
        self.assertEqual(integration.fault_tolerance_manager, self.fault_tolerance_manager)
    
    def test_apply_recovery_with_circuit_breaker(self):
        """Test applying a recovery action with circuit breaker protection."""
        # Mock the handle_failure method to return a recovery action
        mock_recovery_action = RecoveryAction(
            strategy=RecoveryStrategy.DELAYED_RETRY,
            message="Test recovery action"
        )
        self.integration.handle_failure = MagicMock(return_value=mock_recovery_action)
        
        # Create a failure context
        failure_context = FailureContext(
            task_id="task1",
            worker_id="worker1",
            error_type=FailureType.SOFTWARE_ERROR,
            error_message="Test error"
        )
        
        # Mock the coordinator to return success
        self.coordinator.retry_task = MagicMock(return_value=True)
        
        # Apply recovery with circuit breaker
        result = apply_recovery_with_circuit_breaker(
            task_id="task1",
            failure_context=failure_context,
            integration=self.integration,
            coordinator=self.coordinator
        )
        
        # Should have called handle_failure and retry_task
        self.integration.handle_failure.assert_called_once_with(failure_context)
        self.coordinator.retry_task.assert_called_once()
        self.assertTrue(result)
    
    def test_apply_recovery_with_circuit_breaker_no_coordinator(self):
        """Test applying a recovery action with no coordinator."""
        # Create a failure context
        failure_context = FailureContext(
            task_id="task1",
            worker_id="worker1",
            error_type=FailureType.SOFTWARE_ERROR,
            error_message="Test error"
        )
        
        # Apply recovery with circuit breaker but no coordinator
        result = apply_recovery_with_circuit_breaker(
            task_id="task1",
            failure_context=failure_context,
            integration=self.integration,
            coordinator=None
        )
        
        # Should fail
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()