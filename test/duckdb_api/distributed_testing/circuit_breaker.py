#!/usr/bin/env python3
"""
Circuit Breaker Pattern Implementation for Distributed Testing Framework

This module implements the Circuit Breaker pattern for the distributed testing framework,
providing a mechanism to prevent cascading failures by temporarily disabling problematic
components when they start to fail consistently.

The circuit breaker has three states:
1. CLOSED: Normal operation, requests are allowed through
2. OPEN: Failing state, requests are blocked
3. HALF_OPEN: Testing state, limited requests are allowed to check if the issue is resolved

Key features:
1. Configurable thresholds for circuit opening
2. Automatic testing and recovery
3. Exponential backoff for recovery attempts
4. Detailed metrics and health reporting
5. Integration with the fault tolerance system
"""

import time
import logging
import threading
import random
from enum import Enum, auto
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("circuit_breaker")


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = auto()  # Normal operation, requests are allowed through
    OPEN = auto()    # Failing state, requests are blocked
    HALF_OPEN = auto()  # Testing state, limited requests are allowed to check if the issue is resolved


class CircuitBreaker:
    """
    Implementation of the Circuit Breaker pattern.
    
    This class provides a way to detect when a component is failing and
    prevent additional requests until the component has recovered.
    """
    
    def __init__(self, name: str, failure_threshold: int = 3, 
                recovery_timeout: float = 60.0, half_open_max_calls: int = 1,
                reset_timeout_factor: float = 2.0, max_reset_timeout: float = 300.0,
                success_threshold: int = 2):
        """
        Initialize a new circuit breaker.
        
        Args:
            name: Name of the circuit (for identification/logging)
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Base time in seconds to wait before trying recovery
            half_open_max_calls: Maximum number of test calls in HALF_OPEN state
            reset_timeout_factor: Factor to increase timeout after each failed recovery
            max_reset_timeout: Maximum timeout between recovery attempts
            success_threshold: Number of successes required to close the circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.reset_timeout_factor = reset_timeout_factor
        self.max_reset_timeout = max_reset_timeout
        self.success_threshold = success_threshold
        
        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change_time = datetime.now()
        self.current_reset_timeout = recovery_timeout
        
        # Metrics
        self.total_failures = 0
        self.total_successes = 0
        self.open_circuits_count = 0
        self.trip_times = []  # Times when circuit was opened
        self.recovery_times = []  # Times when circuit was closed after being open
        
        # Locking
        self.state_lock = threading.RLock()
        
        # Half-open state tracking
        self.half_open_calls = 0
        self.half_open_failures = 0
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def execute(self, action: Callable, fallback: Optional[Callable] = None):
        """
        Execute an action with circuit breaker protection.
        
        Args:
            action: Function to execute
            fallback: Optional fallback function to execute if circuit is open
            
        Returns:
            Result of the action or fallback
            
        Raises:
            Exception: If circuit is open and no fallback is provided
        """
        with self.state_lock:
            if self.state == CircuitState.OPEN:
                # Check if it's time to try recovery
                current_time = datetime.now()
                time_since_last_failure = (current_time - self.last_failure_time).total_seconds() \
                    if self.last_failure_time else float('inf')
                
                if time_since_last_failure >= self.current_reset_timeout:
                    # Try recovery, transition to half-open
                    self._transition_to_half_open()
                else:
                    # Circuit is still open
                    if fallback:
                        logger.debug(f"Circuit '{self.name}' is OPEN, using fallback")
                        return fallback()
                    else:
                        logger.warning(f"Circuit '{self.name}' is OPEN, no fallback provided")
                        raise CircuitOpenError(f"Circuit '{self.name}' is open")
            
            # If we're in HALF_OPEN state, only allow limited calls
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    # Too many calls in half-open state
                    if fallback:
                        logger.debug(f"Circuit '{self.name}' is HALF_OPEN with too many calls, using fallback")
                        return fallback()
                    else:
                        logger.warning(f"Circuit '{self.name}' is HALF_OPEN with too many calls, no fallback provided")
                        raise CircuitOpenError(f"Circuit '{self.name}' is half-open with too many calls")
                
                # Increment the call counter
                self.half_open_calls += 1
        
        # Execute the action
        try:
            result = action()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            
            # If we have a fallback, use it
            if fallback:
                logger.debug(f"Action failed in circuit '{self.name}', using fallback")
                return fallback()
            else:
                # No fallback, propagate the exception
                logger.warning(f"Action failed in circuit '{self.name}', no fallback provided")
                raise
    
    def _on_success(self):
        """Handle successful execution."""
        with self.state_lock:
            self.total_successes += 1
            
            if self.state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self.failure_count = 0
            elif self.state == CircuitState.HALF_OPEN:
                # Increment success count in half-open state
                self.success_count += 1
                
                # Check if we've had enough successes to close the circuit
                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()
    
    def _on_failure(self):
        """Handle failed execution."""
        with self.state_lock:
            self.total_failures += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.CLOSED:
                # Increment failure count in closed state
                self.failure_count += 1
                
                # Check if we've had enough failures to open the circuit
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                # In half-open state, any failure opens the circuit again
                self.half_open_failures += 1
                self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        logger.warning(f"Circuit '{self.name}' transitioned to OPEN state")
        self.state = CircuitState.OPEN
        self.last_state_change_time = datetime.now()
        self.current_reset_timeout = min(
            self.current_reset_timeout * self.reset_timeout_factor,
            self.max_reset_timeout
        )
        self.trip_times.append(datetime.now())
        self.open_circuits_count += 1
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        logger.info(f"Circuit '{self.name}' transitioned to HALF_OPEN state")
        self.state = CircuitState.HALF_OPEN
        self.last_state_change_time = datetime.now()
        self.half_open_calls = 0
        self.half_open_failures = 0
        self.success_count = 0
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info(f"Circuit '{self.name}' transitioned to CLOSED state")
        self.state = CircuitState.CLOSED
        self.last_state_change_time = datetime.now()
        self.failure_count = 0
        self.current_reset_timeout = self.recovery_timeout  # Reset timeout
        self.recovery_times.append(datetime.now())
    
    def reset(self):
        """Force reset the circuit to CLOSED state."""
        with self.state_lock:
            logger.info(f"Circuit '{self.name}' forcibly reset to CLOSED state")
            self._transition_to_closed()
    
    def get_state(self) -> CircuitState:
        """
        Get the current state of the circuit.
        
        Returns:
            Current circuit state
        """
        with self.state_lock:
            return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the circuit.
        
        Returns:
            Dictionary with circuit metrics
        """
        with self.state_lock:
            time_in_current_state = (datetime.now() - self.last_state_change_time).total_seconds()
            
            metrics = {
                "name": self.name,
                "state": self.state.name,
                "time_in_current_state": time_in_current_state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "open_circuits_count": self.open_circuits_count,
                "trip_times": [t.isoformat() for t in self.trip_times],
                "recovery_times": [t.isoformat() for t in self.recovery_times],
                "current_reset_timeout": self.current_reset_timeout,
                "health_percentage": self._calculate_health_percentage()
            }
            
            return metrics
    
    def _calculate_health_percentage(self) -> float:
        """
        Calculate a health percentage for the circuit.
        
        Returns:
            Health percentage (0-100)
        """
        # Default health is 100%
        health = 100.0
        
        # If state is OPEN, health is at most 20%
        if self.state == CircuitState.OPEN:
            return max(0, 20.0 - (self.open_circuits_count * 2))
        
        # If state is HALF_OPEN, health is at most 60%
        if self.state == CircuitState.HALF_OPEN:
            base_health = 40.0 + (self.success_count * 10.0)
            return min(base_health, 60.0)
        
        # If state is CLOSED, health is based on recent failures
        total_calls = self.total_successes + self.total_failures
        
        # If no calls yet, health is 100%
        if total_calls == 0:
            return 100.0
        
        # Adjust health based on recent failure rate, but never below 70% for CLOSED state
        if self.failure_count > 0:
            health = 100.0 - (self.failure_count * 10.0)
            return max(70.0, health)
        
        return health


class CircuitOpenError(Exception):
    """Exception raised when a circuit is open."""
    pass


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    This class provides a way to create, access, and monitor multiple circuit breakers
    as well as aggregate metrics across all circuits.
    """
    
    def __init__(self):
        """Initialize the circuit breaker registry."""
        self.circuits: Dict[str, CircuitBreaker] = {}
        self.registry_lock = threading.RLock()
        logger.info("Circuit breaker registry initialized")
    
    def get_or_create(self, name: str, **kwargs) -> CircuitBreaker:
        """
        Get an existing circuit breaker or create a new one.
        
        Args:
            name: Name of the circuit breaker
            **kwargs: Parameters to pass to CircuitBreaker constructor if creating
            
        Returns:
            CircuitBreaker instance
        """
        with self.registry_lock:
            if name not in self.circuits:
                self.circuits[name] = CircuitBreaker(name=name, **kwargs)
            return self.circuits[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get an existing circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            
        Returns:
            CircuitBreaker instance or None if not found
        """
        with self.registry_lock:
            return self.circuits.get(name)
    
    def reset_all(self):
        """Reset all circuit breakers to CLOSED state."""
        with self.registry_lock:
            for circuit in self.circuits.values():
                circuit.reset()
            logger.info("All circuit breakers have been reset")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all circuit breakers.
        
        Returns:
            Dictionary mapping circuit names to metrics
        """
        with self.registry_lock:
            return {name: circuit.get_metrics() for name, circuit in self.circuits.items()}
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate metrics across all circuit breakers.
        
        Returns:
            Dictionary with aggregate metrics
        """
        with self.registry_lock:
            total_circuits = len(self.circuits)
            open_circuits = sum(1 for c in self.circuits.values() if c.get_state() == CircuitState.OPEN)
            half_open_circuits = sum(1 for c in self.circuits.values() if c.get_state() == CircuitState.HALF_OPEN)
            closed_circuits = sum(1 for c in self.circuits.values() if c.get_state() == CircuitState.CLOSED)
            
            total_failures = sum(c.total_failures for c in self.circuits.values())
            total_successes = sum(c.total_successes for c in self.circuits.values())
            total_open_circuits = sum(c.open_circuits_count for c in self.circuits.values())
            
            # Calculate average health
            health_values = [c._calculate_health_percentage() for c in self.circuits.values()]
            avg_health = sum(health_values) / total_circuits if total_circuits > 0 else 100.0
            
            return {
                "total_circuits": total_circuits,
                "open_circuits": open_circuits,
                "half_open_circuits": half_open_circuits,
                "closed_circuits": closed_circuits,
                "total_failures": total_failures,
                "total_successes": total_successes,
                "total_open_circuits": total_open_circuits,
                "average_health": avg_health,
                "overall_health": self._calculate_overall_health(open_circuits, half_open_circuits, total_circuits)
            }
    
    def _calculate_overall_health(self, open_circuits: int, half_open_circuits: int, total_circuits: int) -> float:
        """
        Calculate overall health percentage of the system based on circuit states.
        
        Args:
            open_circuits: Number of open circuits
            half_open_circuits: Number of half-open circuits
            total_circuits: Total number of circuits
            
        Returns:
            Overall health percentage (0-100)
        """
        if total_circuits == 0:
            return 100.0
        
        # Weight: OPEN circuits reduce health significantly, HALF_OPEN less so
        open_weight = 1.0
        half_open_weight = 0.4
        
        health_reduction = (
            (open_circuits * open_weight) + 
            (half_open_circuits * half_open_weight)
        ) / total_circuits * 100.0
        
        return max(0.0, 100.0 - health_reduction)
    
    def execute(self, circuit_name: str, action: Callable, fallback: Optional[Callable] = None):
        """
        Execute an action with circuit breaker protection.
        
        Args:
            circuit_name: Name of the circuit breaker to use
            action: Function to execute
            fallback: Optional fallback function to execute if circuit is open
            
        Returns:
            Result of the action or fallback
            
        Raises:
            Exception: If circuit is open and no fallback is provided
            KeyError: If circuit is not found
        """
        with self.registry_lock:
            if circuit_name not in self.circuits:
                raise KeyError(f"Circuit breaker '{circuit_name}' not found")
            
            circuit = self.circuits[circuit_name]
        
        return circuit.execute(action, fallback)


# Global registry instance for convenience
global_registry = CircuitBreakerRegistry()


def circuit_breaker(circuit_name: str, failure_threshold: int = 3, 
                   recovery_timeout: float = 60.0, **kwargs):
    """
    Decorator for adding circuit breaker protection to functions.
    
    Args:
        circuit_name: Name of the circuit breaker to use
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds to wait before trying recovery
        **kwargs: Additional parameters to pass to CircuitBreaker constructor
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            circuit = global_registry.get_or_create(
                circuit_name, 
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                **kwargs
            )
            
            return circuit.execute(
                lambda: func(*args, **kwargs),
                fallback=lambda: None
            )
        return wrapper
    return decorator


def create_worker_circuit_breaker(worker_id: str, failure_threshold: int = 3, 
                                recovery_timeout: float = 60.0) -> CircuitBreaker:
    """
    Create a circuit breaker for a worker.
    
    Args:
        worker_id: ID of the worker
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds to wait before trying recovery
        
    Returns:
        CircuitBreaker instance
    """
    circuit_name = f"worker_{worker_id}"
    return global_registry.get_or_create(
        circuit_name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )


def create_endpoint_circuit_breaker(endpoint_name: str, failure_threshold: int = 3,
                                   recovery_timeout: float = 60.0) -> CircuitBreaker:
    """
    Create a circuit breaker for an API endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds to wait before trying recovery
        
    Returns:
        CircuitBreaker instance
    """
    circuit_name = f"endpoint_{endpoint_name}"
    return global_registry.get_or_create(
        circuit_name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )


def worker_circuit_breaker(worker_id: str):
    """
    Decorator for adding worker-specific circuit breaker protection.
    
    Args:
        worker_id: ID of the worker
        
    Returns:
        Decorated function
    """
    return circuit_breaker(f"worker_{worker_id}")


def endpoint_circuit_breaker(endpoint_name: str):
    """
    Decorator for adding endpoint-specific circuit breaker protection.
    
    Args:
        endpoint_name: Name of the endpoint
        
    Returns:
        Decorated function
    """
    return circuit_breaker(f"endpoint_{endpoint_name}")