#!/usr/bin/env python3
"""
Integration between Circuit Breaker Pattern and Fault Tolerance System

This module integrates the circuit breaker pattern with the hardware-aware
fault tolerance system to provide comprehensive protection against failures.
It allows the fault tolerance system to use circuit breakers for specific
failure scenarios and provides unified monitoring and visualization.

Key features:
1. Integration with hardware-aware fault tolerance
2. Worker-specific circuit breakers
3. Hardware class-specific circuit breakers
4. Task type-specific circuit breakers
5. Shared metrics and health monitoring
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

# Import hardware-aware fault tolerance
from duckdb_api.distributed_testing.hardware_aware_fault_tolerance import (
    HardwareAwareFaultToleranceManager, FailureContext, RecoveryAction, RecoveryStrategy, FailureType
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("fault_tolerance_integration")


class CircuitBreakerIntegration:
    """
    Integration between circuit breaker pattern and hardware-aware fault tolerance.
    
    This class provides a unified interface for using circuit breakers with
    the fault tolerance system.
    """
    
    def __init__(self, fault_tolerance_manager: HardwareAwareFaultToleranceManager):
        """
        Initialize the integration.
        
        Args:
            fault_tolerance_manager: Hardware-aware fault tolerance manager
        """
        self.fault_tolerance_manager = fault_tolerance_manager
        self.circuit_registry = CircuitBreakerRegistry()
        
        # Circuit configuration for different scenarios
        self.worker_circuit_config = {
            "failure_threshold": 5,
            "recovery_timeout": 60.0,
            "reset_timeout_factor": 2.0,
            "max_reset_timeout": 600.0,
            "success_threshold": 3
        }
        
        self.hardware_circuit_config = {
            "failure_threshold": 3,
            "recovery_timeout": 120.0,
            "reset_timeout_factor": 1.5,
            "max_reset_timeout": 1800.0,
            "success_threshold": 5
        }
        
        self.task_type_circuit_config = {
            "failure_threshold": 10,
            "recovery_timeout": 300.0,
            "reset_timeout_factor": 1.2,
            "max_reset_timeout": 3600.0,
            "success_threshold": 2
        }
        
        # Mapping of failure types to circuit breaker types
        self.failure_circuit_mapping = {
            FailureType.WORKER_CRASH: "worker",
            FailureType.WORKER_DISCONNECTION: "worker",
            FailureType.HARDWARE_ERROR: "hardware",
            FailureType.BROWSER_FAILURE: "worker",
            FailureType.RESOURCE_EXHAUSTION: "hardware"
        }
        
        logger.info("Circuit breaker integration initialized")
    
    def get_worker_circuit(self, worker_id: str) -> CircuitBreaker:
        """
        Get a circuit breaker for a worker.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            CircuitBreaker instance
        """
        circuit_name = f"worker_{worker_id}"
        return self.circuit_registry.get_or_create(
            circuit_name, 
            **self.worker_circuit_config
        )
    
    def get_hardware_circuit(self, hardware_class: str) -> CircuitBreaker:
        """
        Get a circuit breaker for a hardware class.
        
        Args:
            hardware_class: Hardware class name
            
        Returns:
            CircuitBreaker instance
        """
        circuit_name = f"hardware_{hardware_class}"
        return self.circuit_registry.get_or_create(
            circuit_name, 
            **self.hardware_circuit_config
        )
    
    def get_task_type_circuit(self, task_type: str) -> CircuitBreaker:
        """
        Get a circuit breaker for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            CircuitBreaker instance
        """
        circuit_name = f"task_type_{task_type}"
        return self.circuit_registry.get_or_create(
            circuit_name, 
            **self.task_type_circuit_config
        )
    
    def handle_failure(self, failure_context: FailureContext) -> RecoveryAction:
        """
        Handle a failure with circuit breaker integration.
        
        Args:
            failure_context: Failure context
            
        Returns:
            RecoveryAction to take
        """
        # Determine which circuit breakers to use
        worker_id = failure_context.worker_id
        hardware_class = "UNKNOWN"
        task_type = "unknown"
        
        # Extract hardware class if available
        if failure_context.hardware_profile and failure_context.hardware_profile.hardware_class:
            hardware_class = failure_context.hardware_profile.hardware_class.name
        
        # Extract task type if available
        task = self.fault_tolerance_manager._get_task(failure_context.task_id)
        if task and "type" in task:
            task_type = task["type"]
        
        # Update circuit breakers
        worker_circuit = self.get_worker_circuit(worker_id)
        hardware_circuit = self.get_hardware_circuit(hardware_class)
        task_type_circuit = self.get_task_type_circuit(task_type)
        
        # Check if any circuits are open
        open_circuits = []
        if worker_circuit.get_state() == CircuitState.OPEN:
            open_circuits.append(f"worker_{worker_id}")
        
        if hardware_circuit.get_state() == CircuitState.OPEN:
            open_circuits.append(f"hardware_{hardware_class}")
        
        if task_type_circuit.get_state() == CircuitState.OPEN:
            open_circuits.append(f"task_type_{task_type}")
        
        # If any circuits are open, modify the recovery action
        if open_circuits:
            logger.warning(f"Circuits open for task {failure_context.task_id}: {open_circuits}")
            
            # Determine alternative strategy based on open circuits
            if "worker" in open_circuits[0]:
                # Worker circuit is open, use a different worker
                return RecoveryAction(
                    strategy=RecoveryStrategy.DIFFERENT_WORKER,
                    message=f"Circuit breaker open for worker {worker_id}, using different worker"
                )
            elif "hardware" in open_circuits[0]:
                # Hardware circuit is open, use a different hardware class
                return RecoveryAction(
                    strategy=RecoveryStrategy.DIFFERENT_HARDWARE_CLASS,
                    message=f"Circuit breaker open for hardware {hardware_class}, using different hardware class",
                    hardware_requirements={"hardware": [self.fault_tolerance_manager._determine_fallback_hardware_class(hardware_class).lower()]}
                )
            else:
                # Task type circuit is open, use a more conservative approach
                return RecoveryAction(
                    strategy=RecoveryStrategy.DELAYED_RETRY,
                    delay=60.0,  # Longer delay for task type circuit breaker
                    message=f"Circuit breaker open for task type {task_type}, using delayed retry with longer delay"
                )
        
        # No circuits are open, track the failure
        self._track_failure(failure_context)
        
        # Use default fault tolerance strategy
        return self.fault_tolerance_manager._determine_recovery_strategy(failure_context)
    
    def _track_failure(self, failure_context: FailureContext):
        """
        Track a failure in relevant circuit breakers.
        
        Args:
            failure_context: Failure context
        """
        worker_id = failure_context.worker_id
        hardware_class = "UNKNOWN"
        task_type = "unknown"
        
        # Extract hardware class if available
        if failure_context.hardware_profile and failure_context.hardware_profile.hardware_class:
            hardware_class = failure_context.hardware_profile.hardware_class.name
        
        # Extract task type if available
        task = self.fault_tolerance_manager._get_task(failure_context.task_id)
        if task and "type" in task:
            task_type = task["type"]
        
        # Get the circuit breakers
        worker_circuit = self.get_worker_circuit(worker_id)
        hardware_circuit = self.get_hardware_circuit(hardware_class)
        task_type_circuit = self.get_task_type_circuit(task_type)
        
        # Determine which circuit breakers to update based on failure type
        circuit_type = self.failure_circuit_mapping.get(failure_context.error_type, None)
        
        if circuit_type == "worker":
            # Update worker circuit
            worker_circuit._on_failure()
            logger.debug(f"Updated worker circuit for {worker_id} due to {failure_context.error_type.name}")
        elif circuit_type == "hardware":
            # Update hardware circuit
            hardware_circuit._on_failure()
            logger.debug(f"Updated hardware circuit for {hardware_class} due to {failure_context.error_type.name}")
        else:
            # Update task type circuit for other errors
            task_type_circuit._on_failure()
            logger.debug(f"Updated task type circuit for {task_type} due to {failure_context.error_type.name}")
    
    def track_success(self, task_id: str, worker_id: str, hardware_class: str = None, task_type: str = None):
        """
        Track a successful execution.
        
        Args:
            task_id: ID of the task
            worker_id: ID of the worker
            hardware_class: Optional hardware class name
            task_type: Optional task type
        """
        # Default values if not provided
        hardware_class = hardware_class or "UNKNOWN"
        task_type = task_type or "unknown"
        
        # Get the circuit breakers
        worker_circuit = self.get_worker_circuit(worker_id)
        hardware_circuit = self.get_hardware_circuit(hardware_class)
        task_type_circuit = self.get_task_type_circuit(task_type)
        
        # Update all circuits
        worker_circuit._on_success()
        hardware_circuit._on_success()
        task_type_circuit._on_success()
        
        logger.debug(f"Tracked success for task {task_id} on worker {worker_id}")
    
    def reset_worker_circuit(self, worker_id: str):
        """
        Reset a worker's circuit breaker.
        
        Args:
            worker_id: ID of the worker
        """
        circuit = self.get_worker_circuit(worker_id)
        circuit.reset()
        logger.info(f"Reset circuit breaker for worker {worker_id}")
    
    def reset_hardware_circuit(self, hardware_class: str):
        """
        Reset a hardware class's circuit breaker.
        
        Args:
            hardware_class: Hardware class name
        """
        circuit = self.get_hardware_circuit(hardware_class)
        circuit.reset()
        logger.info(f"Reset circuit breaker for hardware class {hardware_class}")
    
    def reset_task_type_circuit(self, task_type: str):
        """
        Reset a task type's circuit breaker.
        
        Args:
            task_type: Task type
        """
        circuit = self.get_task_type_circuit(task_type)
        circuit.reset()
        logger.info(f"Reset circuit breaker for task type {task_type}")
    
    def reset_all_circuits(self):
        """Reset all circuit breakers."""
        self.circuit_registry.reset_all()
        logger.info("Reset all circuit breakers")
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """
        Get health metrics for all circuit breakers.
        
        Returns:
            Dictionary with health metrics
        """
        # Get metrics from registry
        individual_metrics = self.circuit_registry.get_all_metrics()
        aggregate_metrics = self.circuit_registry.get_aggregate_metrics()
        
        # Group metrics by type
        worker_metrics = {}
        hardware_metrics = {}
        task_type_metrics = {}
        
        for circuit_name, metrics in individual_metrics.items():
            if circuit_name.startswith("worker_"):
                worker_id = circuit_name[7:]  # Remove "worker_" prefix
                worker_metrics[worker_id] = metrics
            elif circuit_name.startswith("hardware_"):
                hardware_class = circuit_name[9:]  # Remove "hardware_" prefix
                hardware_metrics[hardware_class] = metrics
            elif circuit_name.startswith("task_type_"):
                task_type = circuit_name[10:]  # Remove "task_type_" prefix
                task_type_metrics[task_type] = metrics
        
        # Build complete metrics object
        health_metrics = {
            "aggregate": aggregate_metrics,
            "workers": worker_metrics,
            "hardware_classes": hardware_metrics,
            "task_types": task_type_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        return health_metrics
    
    def get_worker_health(self, worker_id: str) -> Dict[str, Any]:
        """
        Get health metrics for a specific worker.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            Dictionary with worker health metrics
        """
        circuit = self.get_worker_circuit(worker_id)
        return circuit.get_metrics()
    
    def get_hardware_health(self, hardware_class: str) -> Dict[str, Any]:
        """
        Get health metrics for a specific hardware class.
        
        Args:
            hardware_class: Hardware class name
            
        Returns:
            Dictionary with hardware health metrics
        """
        circuit = self.get_hardware_circuit(hardware_class)
        return circuit.get_metrics()
    
    def get_task_type_health(self, task_type: str) -> Dict[str, Any]:
        """
        Get health metrics for a specific task type.
        
        Args:
            task_type: Task type
            
        Returns:
            Dictionary with task type health metrics
        """
        circuit = self.get_task_type_circuit(task_type)
        return circuit.get_metrics()
    
    def configure_worker_circuits(self, config: Dict[str, Any]):
        """
        Configure worker circuit breakers.
        
        Args:
            config: Configuration parameters
        """
        self.worker_circuit_config.update(config)
        logger.info(f"Updated worker circuit configuration: {config}")
    
    def configure_hardware_circuits(self, config: Dict[str, Any]):
        """
        Configure hardware circuit breakers.
        
        Args:
            config: Configuration parameters
        """
        self.hardware_circuit_config.update(config)
        logger.info(f"Updated hardware circuit configuration: {config}")
    
    def configure_task_type_circuits(self, config: Dict[str, Any]):
        """
        Configure task type circuit breakers.
        
        Args:
            config: Configuration parameters
        """
        self.task_type_circuit_config.update(config)
        logger.info(f"Updated task type circuit configuration: {config}")
    
    def is_worker_circuit_open(self, worker_id: str) -> bool:
        """
        Check if a worker's circuit breaker is open.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            True if the circuit is open, False otherwise
        """
        circuit = self.get_worker_circuit(worker_id)
        return circuit.get_state() == CircuitState.OPEN
    
    def is_hardware_circuit_open(self, hardware_class: str) -> bool:
        """
        Check if a hardware class's circuit breaker is open.
        
        Args:
            hardware_class: Hardware class name
            
        Returns:
            True if the circuit is open, False otherwise
        """
        circuit = self.get_hardware_circuit(hardware_class)
        return circuit.get_state() == CircuitState.OPEN
    
    def is_task_type_circuit_open(self, task_type: str) -> bool:
        """
        Check if a task type's circuit breaker is open.
        
        Args:
            task_type: Task type
            
        Returns:
            True if the circuit is open, False otherwise
        """
        circuit = self.get_task_type_circuit(task_type)
        return circuit.get_state() == CircuitState.OPEN


def create_fault_tolerance_integration(fault_tolerance_manager: HardwareAwareFaultToleranceManager) -> CircuitBreakerIntegration:
    """
    Create a circuit breaker integration with fault tolerance.
    
    Args:
        fault_tolerance_manager: Hardware-aware fault tolerance manager
        
    Returns:
        CircuitBreakerIntegration instance
    """
    return CircuitBreakerIntegration(fault_tolerance_manager)


def apply_recovery_with_circuit_breaker(task_id: str, failure_context: FailureContext,
                                      integration: CircuitBreakerIntegration,
                                      coordinator=None) -> bool:
    """
    Apply a recovery action with circuit breaker protection.
    
    Args:
        task_id: ID of the failed task
        failure_context: Failure context
        integration: Circuit breaker integration
        coordinator: Coordinator instance
        
    Returns:
        True if the recovery action was applied successfully
    """
    # Get recovery action using circuit breaker integration
    recovery_action = integration.handle_failure(failure_context)
    
    # Now apply the recovery action
    if coordinator:
        try:
            # Apply the action based on recovery strategy
            if recovery_action.strategy == RecoveryStrategy.IMMEDIATE_RETRY:
                # Retry immediately on the same worker
                coordinator.retry_task(task_id, worker_id=recovery_action.worker_id)
                
            elif recovery_action.strategy == RecoveryStrategy.DELAYED_RETRY:
                # Schedule delayed retry
                if recovery_action.delay > 0:
                    # Use scheduler to delay the task
                    threading.Timer(
                        recovery_action.delay, 
                        lambda: coordinator.retry_task(task_id, worker_id=recovery_action.worker_id)
                    ).start()
                else:
                    coordinator.retry_task(task_id, worker_id=recovery_action.worker_id)
                
            elif recovery_action.strategy == RecoveryStrategy.DIFFERENT_WORKER:
                # Retry on a different worker
                coordinator.retry_task(task_id, exclude_workers=[failure_context.worker_id])
                
            elif recovery_action.strategy == RecoveryStrategy.DIFFERENT_HARDWARE_CLASS:
                # Retry on different hardware class
                if recovery_action.hardware_requirements:
                    coordinator.update_task_requirements(task_id, recovery_action.hardware_requirements)
                coordinator.retry_task(task_id)
                
            elif recovery_action.strategy == RecoveryStrategy.REDUCED_PRECISION:
                # Retry with reduced precision
                if recovery_action.modified_task:
                    coordinator.update_task_config(task_id, recovery_action.modified_task.get("config", {}))
                coordinator.retry_task(task_id)
                
            elif recovery_action.strategy == RecoveryStrategy.REDUCED_BATCH_SIZE:
                # Retry with reduced batch size
                if recovery_action.modified_task:
                    coordinator.update_task_config(task_id, recovery_action.modified_task.get("config", {}))
                coordinator.retry_task(task_id)
                
            elif recovery_action.strategy == RecoveryStrategy.FALLBACK_CPU:
                # Fallback to CPU execution
                if recovery_action.hardware_requirements:
                    coordinator.update_task_requirements(task_id, recovery_action.hardware_requirements)
                coordinator.retry_task(task_id)
                
            elif recovery_action.strategy == RecoveryStrategy.BROWSER_RESTART:
                # Restart browser and retry
                if recovery_action.worker_id:
                    coordinator.restart_worker_browser(recovery_action.worker_id)
                coordinator.retry_task(task_id)
                
            elif recovery_action.strategy == RecoveryStrategy.RESET_WORKER_STATE:
                # Reset worker state and retry
                if recovery_action.worker_id:
                    coordinator.reset_worker_state(recovery_action.worker_id)
                coordinator.retry_task(task_id)
                
            elif recovery_action.strategy == RecoveryStrategy.ESCALATION:
                # Escalate to human operator
                coordinator.escalate_task(task_id, recovery_action.message)
                
            else:
                logger.warning(f"Unsupported recovery strategy: {recovery_action.strategy}")
                return False
            
            logger.info(f"Applied recovery action for task {task_id}: {recovery_action.strategy.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying recovery action: {e}")
            return False
    else:
        logger.error("Cannot apply recovery action: No coordinator provided")
        return False