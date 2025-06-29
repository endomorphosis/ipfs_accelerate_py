#!/usr/bin/env python3
"""
Unit tests for the Circuit Breaker pattern implementation.

This module provides comprehensive tests for the circuit breaker implementation,
verifying its behavior across all three states: CLOSED, OPEN, and HALF_OPEN.
"""

import os
import sys
import time
import unittest
import threading
import random
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import circuit breaker
from duckdb_api.distributed_testing.circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitOpenError, CircuitBreakerRegistry,
    circuit_breaker, create_worker_circuit_breaker, create_endpoint_circuit_breaker,
    worker_circuit_breaker, endpoint_circuit_breaker
)


class TestCircuitBreaker(unittest.TestCase):
    """Test cases for CircuitBreaker class."""
    
    def setUp(self):
        """Set up a fresh circuit breaker for each test."""
        self.circuit = CircuitBreaker(
            name="test_circuit",
            failure_threshold=3,
            recovery_timeout=0.1,  # Short timeout for testing
            half_open_max_calls=2,
            reset_timeout_factor=2.0,
            success_threshold=2
        )
    
    def test_initial_state(self):
        """Test the initial state of the circuit breaker."""
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit.failure_count, 0)
        self.assertEqual(self.circuit.success_count, 0)
        self.assertEqual(self.circuit.total_failures, 0)
        self.assertEqual(self.circuit.total_successes, 0)
        self.assertEqual(self.circuit.open_circuits_count, 0)
    
    def test_successful_execution(self):
        """Test successful execution."""
        result = self.circuit.execute(lambda: "success")
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit.total_successes, 1)
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
    
    def test_failed_execution(self):
        """Test failed execution."""
        with self.assertRaises(ValueError):
            self.circuit.execute(lambda: (_ for _ in ()).throw(ValueError("test failure")))
        
        self.assertEqual(self.circuit.total_failures, 1)
        self.assertEqual(self.circuit.failure_count, 1)
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
    
    def test_fallback_execution(self):
        """Test fallback execution on failure."""
        result = self.circuit.execute(
            lambda: (_ for _ in ()).throw(ValueError("test failure")),
            fallback=lambda: "fallback"
        )
        
        self.assertEqual(result, "fallback")
        self.assertEqual(self.circuit.total_failures, 1)
        self.assertEqual(self.circuit.failure_count, 1)
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
    
    def test_circuit_opens_after_threshold(self):
        """Test that circuit opens after failure threshold is reached."""
        # Fail 3 times to reach threshold
        for _ in range(3):
            try:
                self.circuit.execute(lambda: (_ for _ in ()).throw(ValueError("test failure")))
            except ValueError:
                pass
        
        self.assertEqual(self.circuit.state, CircuitState.OPEN)
        self.assertEqual(self.circuit.total_failures, 3)
        self.assertEqual(self.circuit.failure_count, 3)
        self.assertEqual(self.circuit.open_circuits_count, 1)
    
    def test_open_circuit_blocks_execution(self):
        """Test that an open circuit blocks execution."""
        # Open the circuit
        self.circuit.state = CircuitState.OPEN
        self.circuit.last_failure_time = datetime.now()
        
        # Try to execute with open circuit
        with self.assertRaises(CircuitOpenError):
            self.circuit.execute(lambda: "should not execute")
    
    def test_open_circuit_uses_fallback(self):
        """Test that an open circuit uses fallback if provided."""
        # Open the circuit
        self.circuit.state = CircuitState.OPEN
        self.circuit.last_failure_time = datetime.now()
        
        # Try to execute with open circuit and fallback
        result = self.circuit.execute(
            lambda: "should not execute",
            fallback=lambda: "fallback"
        )
        
        self.assertEqual(result, "fallback")
    
    def test_circuit_half_opens_after_timeout(self):
        """Test that circuit transitions to half-open after timeout."""
        # Open the circuit
        self.circuit.state = CircuitState.OPEN
        self.circuit.last_failure_time = datetime.now() - timedelta(seconds=0.2)  # Past the recovery timeout
        
        # Try to execute with open circuit that should transition to half-open
        try:
            self.circuit.execute(lambda: (_ for _ in ()).throw(ValueError("test failure")))
        except ValueError:
            pass
        
        self.assertEqual(self.circuit.state, CircuitState.OPEN)  # Failed again, back to open
        self.assertEqual(self.circuit.half_open_failures, 1)
    
    def test_circuit_closes_after_success_in_half_open(self):
        """Test that circuit closes after successful calls in half-open state."""
        # Set to half-open state
        self.circuit._transition_to_half_open()
        
        # Execute successfully twice (success threshold is 2)
        self.circuit.execute(lambda: "success 1")
        self.circuit.execute(lambda: "success 2")
        
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit.success_count, 2)
    
    def test_half_open_limited_calls(self):
        """Test that half-open state limits concurrent calls."""
        # Set to half-open state
        self.circuit._transition_to_half_open()
        
        # Consume all allowed calls
        self.circuit.half_open_calls = self.circuit.half_open_max_calls
        
        # Try to execute with too many half-open calls
        with self.assertRaises(CircuitOpenError):
            self.circuit.execute(lambda: "should not execute")
    
    def test_reset(self):
        """Test manual reset of circuit."""
        # Open the circuit
        self.circuit.state = CircuitState.OPEN
        self.circuit.failure_count = 5
        self.circuit.last_failure_time = datetime.now()
        
        # Reset the circuit
        self.circuit.reset()
        
        self.assertEqual(self.circuit.state, CircuitState.CLOSED)
        self.assertEqual(self.circuit.failure_count, 0)
    
    def test_metrics(self):
        """Test metrics collection."""
        # Generate some activity
        for _ in range(2):
            self.circuit.execute(lambda: "success")
        
        try:
            self.circuit.execute(lambda: (_ for _ in ()).throw(ValueError("test failure")))
        except ValueError:
            pass
        
        # Get metrics
        metrics = self.circuit.get_metrics()
        
        self.assertEqual(metrics["total_successes"], 2)
        self.assertEqual(metrics["total_failures"], 1)
        self.assertEqual(metrics["state"], CircuitState.CLOSED.name)
        self.assertIn("health_percentage", metrics)
    
    def test_exponential_backoff(self):
        """Test that recovery timeout increases with exponential backoff."""
        initial_timeout = self.circuit.current_reset_timeout
        
        # Open the circuit
        self.circuit._transition_to_open()
        first_timeout = self.circuit.current_reset_timeout
        
        # Open it again (simulating a failed recovery)
        self.circuit._transition_to_open()
        second_timeout = self.circuit.current_reset_timeout
        
        self.assertGreater(first_timeout, initial_timeout)
        self.assertGreater(second_timeout, first_timeout)
        self.assertLessEqual(second_timeout, self.circuit.max_reset_timeout)


class TestCircuitBreakerRegistry(unittest.TestCase):
    """Test cases for CircuitBreakerRegistry class."""
    
    def setUp(self):
        """Set up a fresh registry for each test."""
        self.registry = CircuitBreakerRegistry()
    
    def test_get_or_create(self):
        """Test getting or creating a circuit breaker."""
        circuit1 = self.registry.get_or_create("test_circuit")
        circuit2 = self.registry.get_or_create("test_circuit")
        
        self.assertIs(circuit1, circuit2)  # Should be the same instance
        self.assertEqual(len(self.registry.circuits), 1)
    
    def test_get(self):
        """Test getting an existing circuit breaker."""
        self.registry.get_or_create("test_circuit")
        circuit = self.registry.get("test_circuit")
        
        self.assertIsNotNone(circuit)
        self.assertEqual(circuit.name, "test_circuit")
        
        # Get non-existent circuit
        circuit = self.registry.get("non_existent")
        self.assertIsNone(circuit)
    
    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        circuit1 = self.registry.get_or_create("test_circuit1")
        circuit2 = self.registry.get_or_create("test_circuit2")
        
        # Open both circuits
        circuit1._transition_to_open()
        circuit2._transition_to_open()
        
        # Reset all
        self.registry.reset_all()
        
        self.assertEqual(circuit1.state, CircuitState.CLOSED)
        self.assertEqual(circuit2.state, CircuitState.CLOSED)
    
    def test_get_all_metrics(self):
        """Test getting metrics for all circuit breakers."""
        self.registry.get_or_create("test_circuit1")
        self.registry.get_or_create("test_circuit2")
        
        metrics = self.registry.get_all_metrics()
        
        self.assertEqual(len(metrics), 2)
        self.assertIn("test_circuit1", metrics)
        self.assertIn("test_circuit2", metrics)
    
    def test_get_aggregate_metrics(self):
        """Test getting aggregate metrics."""
        circuit1 = self.registry.get_or_create("test_circuit1")
        circuit2 = self.registry.get_or_create("test_circuit2")
        
        # Open one circuit
        circuit1._transition_to_open()
        
        metrics = self.registry.get_aggregate_metrics()
        
        self.assertEqual(metrics["total_circuits"], 2)
        self.assertEqual(metrics["open_circuits"], 1)
        self.assertEqual(metrics["closed_circuits"], 1)
        self.assertEqual(metrics["half_open_circuits"], 0)
        self.assertIn("overall_health", metrics)
    
    def test_execute(self):
        """Test executing with a circuit breaker from the registry."""
        self.registry.get_or_create("test_circuit")
        
        result = self.registry.execute("test_circuit", lambda: "success")
        self.assertEqual(result, "success")
        
        # Test with non-existent circuit
        with self.assertRaises(KeyError):
            self.registry.execute("non_existent", lambda: "success")


class TestCircuitBreakerDecorator(unittest.TestCase):
    """Test cases for circuit breaker decorator."""
    
    def setUp(self):
        """Set up a fresh test environment."""
        # Clear global registry
        self.original_circuits = global_registry.circuits.copy()
        global_registry.circuits.clear()
    
    def tearDown(self):
        """Restore global registry."""
        global_registry.circuits = self.original_circuits
    
    def test_decorator(self):
        """Test circuit breaker decorator."""
        # Define a decorated function
        @circuit_breaker("decorated_test", failure_threshold=2, recovery_timeout=0.1)
        def decorated_function():
            return "success"
        
        # Call the decorated function
        result = decorated_function()
        self.assertEqual(result, "success")
        
        # Check that circuit was created
        self.assertIn("decorated_test", global_registry.circuits)
        
        # Check metrics
        circuit = global_registry.get("decorated_test")
        self.assertEqual(circuit.total_successes, 1)
    
    def test_worker_circuit_breaker(self):
        """Test worker-specific circuit breaker creation."""
        circuit = create_worker_circuit_breaker("worker1")
        self.assertEqual(circuit.name, "worker_worker1")
        self.assertIn("worker_worker1", global_registry.circuits)
    
    def test_endpoint_circuit_breaker(self):
        """Test endpoint-specific circuit breaker creation."""
        circuit = create_endpoint_circuit_breaker("endpoint1")
        self.assertEqual(circuit.name, "endpoint_endpoint1")
        self.assertIn("endpoint_endpoint1", global_registry.circuits)
    
    def test_worker_decorator(self):
        """Test worker-specific circuit breaker decorator."""
        # Define a decorated function
        @worker_circuit_breaker("worker1")
        def worker_function():
            return "worker success"
        
        # Call the decorated function
        result = worker_function()
        self.assertEqual(result, "worker success")
        
        # Check that circuit was created
        self.assertIn("worker_worker1", global_registry.circuits)
    
    def test_endpoint_decorator(self):
        """Test endpoint-specific circuit breaker decorator."""
        # Define a decorated function
        @endpoint_circuit_breaker("endpoint1")
        def endpoint_function():
            return "endpoint success"
        
        # Call the decorated function
        result = endpoint_function()
        self.assertEqual(result, "endpoint success")
        
        # Check that circuit was created
        self.assertIn("endpoint_endpoint1", global_registry.circuits)


class TestCircuitBreakerUnderLoad(unittest.TestCase):
    """Test circuit breaker behavior under concurrent load."""
    
    def setUp(self):
        """Set up a fresh circuit breaker for each test."""
        self.circuit = CircuitBreaker(
            name="load_test_circuit",
            failure_threshold=5,
            recovery_timeout=0.1,
            half_open_max_calls=3,
            reset_timeout_factor=2.0,
            success_threshold=3
        )
    
    def test_concurrent_execution(self):
        """Test concurrent execution with the circuit breaker."""
        # Number of threads to use
        num_threads = 20
        
        # Track results
        results = []
        errors = []
        
        # Function that will be executed by each thread
        def test_function(thread_id):
            try:
                # Even threads succeed, odd threads fail
                if thread_id % 2 == 0:
                    result = self.circuit.execute(
                        lambda: f"success from thread {thread_id}"
                    )
                    results.append(result)
                else:
                    try:
                        self.circuit.execute(
                            lambda: (_ for _ in ()).throw(ValueError(f"failure from thread {thread_id}"))
                        )
                    except ValueError as e:
                        errors.append(str(e))
                    except CircuitOpenError:
                        # Circuit opened, which is expected
                        pass
            except Exception as e:
                errors.append(f"Unexpected error: {str(e)}")
        
        # Create and start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=test_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertGreaterEqual(len(results), 1)  # At least some threads should succeed
        self.assertGreaterEqual(len(errors), 1)   # At least some threads should fail
        
        # Circuit should be in OPEN state
        self.assertEqual(self.circuit.state, CircuitState.OPEN)
        
        # Check metrics
        metrics = self.circuit.get_metrics()
        self.assertEqual(metrics["state"], CircuitState.OPEN.name)
        self.assertGreater(metrics["total_successes"], 0)
        self.assertGreater(metrics["total_failures"], 0)


if __name__ == '__main__':
    unittest.main()