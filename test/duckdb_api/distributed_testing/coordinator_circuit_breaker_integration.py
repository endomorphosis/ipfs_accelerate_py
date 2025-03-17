#!/usr/bin/env python3
"""
Circuit Breaker Integration with Coordinator Service

This module integrates the Circuit Breaker pattern with the Coordinator service
to provide fault tolerance and prevent cascading failures in the distributed testing framework.

Key features:
1. Integration with CoordinatorServer
2. Worker-specific circuit breakers 
3. Task-specific circuit breakers
4. Automatic recovery strategies
5. Health monitoring and visualization
"""

import logging
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from datetime import datetime, timedelta
from enum import Enum, auto

# Import circuit breaker
from duckdb_api.distributed_testing.circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitOpenError, CircuitBreakerRegistry,
    create_worker_circuit_breaker, create_endpoint_circuit_breaker
)

# Import hardware-aware fault tolerance integration
from duckdb_api.distributed_testing.fault_tolerance_integration import (
    CircuitBreakerIntegration
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("coordinator_circuit_breaker_integration")


class CoordinatorCircuitBreakerIntegration:
    """
    Integration between Circuit Breaker pattern and Coordinator service.
    
    This class provides methods to integrate circuit breakers with the coordinator
    to protect against cascading failures when workers or tasks fail.
    """
    
    def __init__(self, coordinator):
        """
        Initialize the coordinator integration.
        
        Args:
            coordinator: The coordinator server instance
        """
        self.coordinator = coordinator
        self.circuit_registry = CircuitBreakerRegistry()
        
        # Get fault tolerance manager if available
        self.fault_tolerance_manager = getattr(coordinator, 'fault_tolerance_manager', None)
        
        # Create fault tolerance integration if manager is available
        self.fault_tolerance_integration = None
        if self.fault_tolerance_manager:
            self.fault_tolerance_integration = CircuitBreakerIntegration(self.fault_tolerance_manager)
            logger.info("Created integration with hardware-aware fault tolerance system")
        
        # Metrics collection
        self.metrics_lock = threading.RLock()
        self.metrics = {
            "worker_circuits": {},
            "task_circuits": {},
            "endpoint_circuits": {},
            "global_health": 100.0,
            "last_update": datetime.now().isoformat()
        }
        
        # Worker state tracking
        self.worker_states = {}
        
        # Start metrics collection thread
        self.metrics_thread = threading.Thread(target=self._collect_metrics, daemon=True)
        self.metrics_thread.start()
        
        logger.info("Coordinator circuit breaker integration initialized")
    
    def wrap_worker_execution(self, worker_id: str, action: Callable, fallback: Optional[Callable] = None):
        """
        Wrap a worker-related action with circuit breaker protection.
        
        Args:
            worker_id: ID of the worker
            action: Function to execute
            fallback: Optional fallback function if circuit is open
            
        Returns:
            Result of the action or fallback
        """
        # Get or create worker circuit breaker
        circuit = self.get_worker_circuit(worker_id)
        
        # Execute with circuit breaker protection
        return circuit.execute(action, fallback)
    
    def wrap_task_execution(self, task_id: str, task_type: str, action: Callable, fallback: Optional[Callable] = None):
        """
        Wrap a task-related action with circuit breaker protection.
        
        Args:
            task_id: ID of the task
            task_type: Type of the task
            action: Function to execute
            fallback: Optional fallback function if circuit is open
            
        Returns:
            Result of the action or fallback
        """
        # Get or create task circuit breaker
        circuit = self.get_task_circuit(task_id, task_type)
        
        # Execute with circuit breaker protection
        return circuit.execute(action, fallback)
    
    def get_worker_circuit(self, worker_id: str) -> CircuitBreaker:
        """
        Get a circuit breaker for a worker.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            CircuitBreaker instance for the worker
        """
        # Use fault tolerance integration if available
        if self.fault_tolerance_integration:
            return self.fault_tolerance_integration.get_worker_circuit(worker_id)
        
        # Otherwise, use our own registry
        circuit_name = f"worker_{worker_id}"
        
        if not self.circuit_registry.exists(circuit_name):
            circuit = create_worker_circuit_breaker(circuit_name)
            self.circuit_registry.register(circuit)
        
        return self.circuit_registry.get(circuit_name)
    
    def get_task_circuit(self, task_id: str, task_type: str) -> CircuitBreaker:
        """
        Get a circuit breaker for a task.
        
        Args:
            task_id: ID of the task
            task_type: Type of the task
            
        Returns:
            CircuitBreaker instance for the task
        """
        # Use fault tolerance integration if available
        if self.fault_tolerance_integration:
            return self.fault_tolerance_integration.get_task_type_circuit(task_type)
        
        # Otherwise, use our own registry
        circuit_name = f"task_{task_type}"
        
        if not self.circuit_registry.exists(circuit_name):
            circuit = CircuitBreaker(
                name=circuit_name,
                failure_threshold=10,
                recovery_timeout=30.0,
                reset_timeout_factor=1.5,
                max_reset_timeout=300.0
            )
            self.circuit_registry.register(circuit)
        
        return self.circuit_registry.get(circuit_name)
    
    def get_endpoint_circuit(self, endpoint: str) -> CircuitBreaker:
        """
        Get a circuit breaker for an API endpoint.
        
        Args:
            endpoint: Endpoint path
            
        Returns:
            CircuitBreaker instance for the endpoint
        """
        # Use fault tolerance integration if available
        if self.fault_tolerance_integration:
            return self.fault_tolerance_integration.get_endpoint_circuit(endpoint)
        
        # Otherwise, use our own registry
        circuit_name = f"endpoint_{endpoint}"
        
        if not self.circuit_registry.exists(circuit_name):
            circuit = create_endpoint_circuit_breaker(circuit_name)
            self.circuit_registry.register(circuit)
        
        return self.circuit_registry.get(circuit_name)
    
    def on_worker_failure(self, worker_id: str, failure_type: str) -> None:
        """
        Handle a worker failure.
        
        Args:
            worker_id: ID of the failed worker
            failure_type: Type of failure
        """
        # Use fault tolerance integration if available
        if self.fault_tolerance_integration:
            self.fault_tolerance_integration.on_worker_failure(worker_id, failure_type)
        else:
            # Get the worker circuit breaker
            circuit = self.get_worker_circuit(worker_id)
            
            # Force a failure to increment the failure count
            try:
                circuit.execute(lambda: exec('raise Exception("Worker failure")'))
            except Exception:
                pass
    
    def on_task_failure(self, task_id: str, task_type: str, failure_type: str) -> None:
        """
        Handle a task failure.
        
        Args:
            task_id: ID of the failed task
            task_type: Type of the task
            failure_type: Type of failure
        """
        # Use fault tolerance integration if available
        if self.fault_tolerance_integration:
            self.fault_tolerance_integration.on_task_failure(task_id, task_type, failure_type)
        else:
            # Get the task circuit breaker
            circuit = self.get_task_circuit(task_id, task_type)
            
            # Force a failure to increment the failure count
            try:
                circuit.execute(lambda: exec('raise Exception("Task failure")'))
            except Exception:
                pass
    
    def get_circuit_breaker_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for all circuit breakers.
        
        Returns:
            Dictionary containing circuit breaker metrics
        """
        with self.metrics_lock:
            return self.metrics.copy()
    
    def _collect_metrics(self) -> None:
        """Background thread to periodically collect metrics."""
        while True:
            try:
                # Get all circuit breaker metrics
                with self.metrics_lock:
                    # Update worker circuits
                    worker_circuits = {}
                    for name, circuit in self.circuit_registry.circuits.items():
                        if name.startswith("worker_"):
                            metrics = circuit.get_metrics()
                            worker_id = name.replace("worker_", "")
                            worker_circuits[worker_id] = metrics
                    
                    # Update task circuits
                    task_circuits = {}
                    for name, circuit in self.circuit_registry.circuits.items():
                        if name.startswith("task_"):
                            metrics = circuit.get_metrics()
                            task_type = name.replace("task_", "")
                            task_circuits[task_type] = metrics
                    
                    # Update endpoint circuits
                    endpoint_circuits = {}
                    for name, circuit in self.circuit_registry.circuits.items():
                        if name.startswith("endpoint_"):
                            metrics = circuit.get_metrics()
                            endpoint = name.replace("endpoint_", "")
                            endpoint_circuits[endpoint] = metrics
                    
                    # Calculate global health
                    all_healths = []
                    for metrics in worker_circuits.values():
                        all_healths.append(metrics.get("health_percentage", 100.0))
                    for metrics in task_circuits.values():
                        all_healths.append(metrics.get("health_percentage", 100.0))
                    
                    global_health = 100.0
                    if all_healths:
                        global_health = sum(all_healths) / len(all_healths)
                    
                    # Update metrics
                    self.metrics = {
                        "worker_circuits": worker_circuits,
                        "task_circuits": task_circuits,
                        "endpoint_circuits": endpoint_circuits,
                        "global_health": global_health,
                        "last_update": datetime.now().isoformat()
                    }
                
                # Also collect metrics from fault tolerance integration if available
                if self.fault_tolerance_integration:
                    tolerance_metrics = self.fault_tolerance_integration.get_all_metrics()
                    with self.metrics_lock:
                        self.metrics["fault_tolerance"] = tolerance_metrics
            
            except Exception as e:
                logger.error(f"Error collecting circuit breaker metrics: {e}")
            
            # Sleep for 5 seconds
            time.sleep(5)