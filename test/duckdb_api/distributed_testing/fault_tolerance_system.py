#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fault Tolerance System for the Distributed Testing Framework.

This module provides comprehensive fault tolerance capabilities for the Distributed Testing
Framework, including automatic retries, fallback mechanisms, circuit breaking, and 
recovery strategies for various types of failures.
"""

import os
import sys
import json
import uuid
import time
import anyio
import logging
import threading
import random
import math
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Enumeration for error severity levels."""
    LOW = "low"                # Minor issue, can be safely retried
    MEDIUM = "medium"          # Moderate issue, retry with caution
    HIGH = "high"              # Serious issue, limited retries
    CRITICAL = "critical"      # Critical issue, no retry


class ErrorCategory(Enum):
    """Enumeration for categorizing errors."""
    NETWORK = "network"                # Network-related errors
    RESOURCE = "resource"              # Resource allocation errors
    WORKER = "worker"                  # Worker node errors
    TASK = "task"                      # Task execution errors
    DATA = "data"                      # Data-related errors
    HARDWARE = "hardware"              # Hardware-related errors
    AUTHENTICATION = "authentication"  # Authentication errors
    AUTHORIZATION = "authorization"    # Authorization errors
    TIMEOUT = "timeout"                # Timeout errors
    UNKNOWN = "unknown"                # Unknown errors


class RecoveryAction(Enum):
    """Enumeration for recovery actions."""
    RETRY = "retry"                    # Retry the operation
    FALLBACK = "fallback"              # Use fallback mechanism
    REASSIGN = "reassign"              # Reassign to a different worker
    THROTTLE = "throttle"              # Apply throttling
    CIRCUIT_BREAK = "circuit_break"    # Apply circuit breaking
    ABORT = "abort"                    # Abort the operation
    RESET = "reset"                    # Reset worker or connection
    NOTIFY = "notify"                  # Send notification


class FaultToleranceSystem:
    """
    Comprehensive fault tolerance system for the Distributed Testing Framework.
    
    This class provides capabilities for:
    1. Automatic retries with exponential backoff
    2. Circuit breaking to prevent cascading failures
    3. Error categorization and severity assessment
    4. Fallback mechanisms for degraded operation
    5. Recovery strategies tailored to different error types
    6. Health monitoring and alerting
    7. Failure statistics and trend analysis
    """

    def __init__(self, 
                 coordinator=None, 
                 task_manager=None, 
                 worker_manager=None,
                 max_retries=3,
                 circuit_break_threshold=5,
                 circuit_break_timeout=300,
                 error_window_size=100,
                 error_rate_threshold=0.5):
        """
        Initialize the fault tolerance system.
        
        Args:
            coordinator: The coordinator server instance
            task_manager: The task manager instance
            worker_manager: The worker manager instance
            max_retries (int): Maximum number of retries for operations
            circuit_break_threshold (int): Number of errors before circuit breaking
            circuit_break_timeout (int): Timeout in seconds for circuit breaker reset
            error_window_size (int): Size of sliding window for error rate calculation
            error_rate_threshold (float): Threshold for error rate alerting
        """
        self.coordinator = coordinator
        self.task_manager = task_manager
        self.worker_manager = worker_manager
        
        # Configuration
        self.max_retries = max_retries
        self.circuit_break_threshold = circuit_break_threshold
        self.circuit_break_timeout = circuit_break_timeout
        self.error_window_size = error_window_size
        self.error_rate_threshold = error_rate_threshold
        
        # Internal state
        self.worker_errors = defaultdict(list)  # Maps worker_id to list of errors
        self.task_errors = defaultdict(list)  # Maps task_id to list of errors
        self.circuit_breakers = {}  # Maps service_key to circuit breaker state
        self.error_history = deque(maxlen=error_window_size)  # Recent errors
        self.retry_counts = defaultdict(int)  # Maps operation_id to retry count
        self.fallbacks = {}  # Maps service_key to fallback
        self.recovery_strategies = {}  # Maps error category to recovery strategy
        
        # Locks for thread safety
        self.worker_errors_lock = threading.RLock()
        self.task_errors_lock = threading.RLock()
        self.circuit_breakers_lock = threading.RLock()
        self.retry_counts_lock = threading.RLock()
        
        # Initialize recovery strategies
        self._init_recovery_strategies()
        
        # Start background threads
        self.stop_event = threading.Event()
        self.error_rate_monitor_thread = threading.Thread(target=self._monitor_error_rate)
        self.error_rate_monitor_thread.daemon = True
        self.error_rate_monitor_thread.start()
        
        self.circuit_breaker_monitor_thread = threading.Thread(target=self._monitor_circuit_breakers)
        self.circuit_breaker_monitor_thread.daemon = True
        self.circuit_breaker_monitor_thread.start()
        
        logger.info("Fault Tolerance System initialized")

    def handle_error(self, 
                     error: Exception,
                     context: Dict[str, Any],
                     worker_id: Optional[str] = None,
                     task_id: Optional[str] = None,
                     operation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle an error and determine appropriate recovery action.
        
        Args:
            error: The exception that occurred
            context: Dictionary with context information about the operation
            worker_id: Worker ID if applicable
            task_id: Task ID if applicable
            operation_id: Unique ID for the operation (generated if not provided)
            
        Returns:
            Dict: Recovery action and details
        """
        # Generate operation ID if not provided
        if not operation_id:
            operation_id = str(uuid.uuid4())
        
        # Categorize and assess error
        error_info = self._categorize_error(error, context)
        error_category = error_info["category"]
        error_severity = error_info["severity"]
        
        # Record error
        self._record_error(error_info, worker_id, task_id, operation_id)
        
        # Check circuit breaker
        service_key = context.get("service_key")
        if service_key and self._is_circuit_open(service_key):
            logger.warning(f"Circuit breaker open for {service_key}, using fallback")
            return {
                "action": RecoveryAction.FALLBACK,
                "fallback": self._get_fallback(service_key),
                "reason": "Circuit breaker open"
            }
        
        # Get current retry count
        retry_count = self._get_retry_count(operation_id)
        
        # Determine recovery action based on error category and severity
        recovery_action = self._determine_recovery_action(
            error_category, error_severity, retry_count, context
        )
        
        # If retry action, increment retry counter
        if recovery_action["action"] == RecoveryAction.RETRY:
            self._increment_retry_count(operation_id)
            
            # Apply exponential backoff
            backoff_seconds = 2 ** retry_count
            recovery_action["backoff_seconds"] = min(backoff_seconds, 60)  # Max 60 seconds
        
        # If circuit break action, open circuit breaker
        elif recovery_action["action"] == RecoveryAction.CIRCUIT_BREAK and service_key:
            self._open_circuit(service_key)
            recovery_action["fallback"] = self._get_fallback(service_key)
        
        # Add metadata to recovery action
        recovery_action["error_category"] = error_category
        recovery_action["error_severity"] = error_severity
        recovery_action["retry_count"] = retry_count
        recovery_action["operation_id"] = operation_id
        
        return recovery_action

    def retry_operation(self, operation_func, 
                        args=None, 
                        kwargs=None, 
                        context=None, 
                        fallback_func=None,
                        max_retries=None) -> Tuple[bool, Any]:
        """
        Retry an operation with exponential backoff and fallback.
        
        Args:
            operation_func: Function to execute
            args: Args to pass to function
            kwargs: Keyword args to pass to function
            context: Context information for error handling
            fallback_func: Fallback function to call if all retries fail
            max_retries: Override default max retries
            
        Returns:
            Tuple[bool, Any]: Success flag and result or error
        """
        args = args or ()
        kwargs = kwargs or {}
        context = context or {}
        operation_id = str(uuid.uuid4())
        max_retries_count = max_retries or self.max_retries
        
        # Initialize retry counter
        retry_count = 0
        
        while retry_count <= max_retries_count:
            try:
                # Execute operation
                result = operation_func(*args, **kwargs)
                return True, result
                
            except Exception as e:
                # Handle error
                recovery_action = self.handle_error(
                    error=e,
                    context=context,
                    operation_id=operation_id
                )
                
                # If not retry action or reached max retries, break
                if recovery_action["action"] != RecoveryAction.RETRY or retry_count >= max_retries_count:
                    break
                
                # Apply backoff
                backoff_seconds = recovery_action.get("backoff_seconds", 2 ** retry_count)
                logger.info(f"Retrying operation after {backoff_seconds}s (attempt {retry_count + 1}/{max_retries_count})")
                time.sleep(backoff_seconds)
                
                # Increment retry counter
                retry_count += 1
        
        # All retries failed, use fallback if available
        if fallback_func:
            try:
                fallback_result = fallback_func(*args, **kwargs)
                return True, fallback_result
            except Exception as fallback_error:
                logger.exception(f"Fallback function failed: {fallback_error}")
                return False, fallback_error
        
        return False, f"Operation failed after {retry_count} retries"

    def register_fallback(self, service_key: str, fallback: Any) -> None:
        """
        Register a fallback for a service.
        
        Args:
            service_key: Identifier for the service
            fallback: Fallback to use when service fails
        """
        self.fallbacks[service_key] = fallback
        logger.info(f"Registered fallback for {service_key}")

    def register_recovery_strategy(self, 
                                  error_category: Union[str, ErrorCategory], 
                                  strategy: Callable) -> None:
        """
        Register a custom recovery strategy for an error category.
        
        Args:
            error_category: Error category to handle
            strategy: Function that returns a recovery action
        """
        if isinstance(error_category, str):
            error_category = ErrorCategory(error_category)
            
        self.recovery_strategies[error_category] = strategy
        logger.info(f"Registered recovery strategy for {error_category.value}")

    def reset_circuit_breaker(self, service_key: str) -> bool:
        """
        Reset (close) a circuit breaker.
        
        Args:
            service_key: Identifier for the service
            
        Returns:
            bool: True if reset was successful
        """
        with self.circuit_breakers_lock:
            if service_key in self.circuit_breakers:
                self.circuit_breakers[service_key] = {
                    "state": "closed",
                    "error_count": 0,
                    "last_error_time": None,
                    "reset_time": datetime.now()
                }
                logger.info(f"Reset circuit breaker for {service_key}")
                return True
                
        return False

    def reset_retry_count(self, operation_id: str) -> bool:
        """
        Reset retry count for an operation.
        
        Args:
            operation_id: Operation ID
            
        Returns:
            bool: True if reset was successful
        """
        with self.retry_counts_lock:
            if operation_id in self.retry_counts:
                self.retry_counts[operation_id] = 0
                logger.info(f"Reset retry count for operation {operation_id}")
                return True
                
        return False

    def get_error_statistics(self, 
                            worker_id: Optional[str] = None,
                            task_id: Optional[str] = None,
                            time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get error statistics for a worker, task, or globally.
        
        Args:
            worker_id: Filter by worker ID
            task_id: Filter by task ID
            time_window: Time window in seconds
            
        Returns:
            Dict: Error statistics
        """
        stats = {
            "total_errors": 0,
            "categories": defaultdict(int),
            "severities": defaultdict(int),
            "error_rate": 0.0,
            "time_distribution": defaultdict(int)
        }
        
        # Apply time window filter
        current_time = datetime.now()
        min_time = current_time - timedelta(seconds=time_window) if time_window else None
        
        # Process worker errors
        if worker_id:
            with self.worker_errors_lock:
                errors = self.worker_errors.get(worker_id, [])
                for error in errors:
                    if min_time and error["timestamp"] < min_time:
                        continue
                    
                    stats["total_errors"] += 1
                    stats["categories"][error["category"].value] += 1
                    stats["severities"][error["severity"].value] += 1
                    
                    # Group by hour
                    hour = error["timestamp"].strftime("%Y-%m-%d %H:00:00")
                    stats["time_distribution"][hour] += 1
        
        # Process task errors
        elif task_id:
            with self.task_errors_lock:
                errors = self.task_errors.get(task_id, [])
                for error in errors:
                    if min_time and error["timestamp"] < min_time:
                        continue
                    
                    stats["total_errors"] += 1
                    stats["categories"][error["category"].value] += 1
                    stats["severities"][error["severity"].value] += 1
                    
                    # Group by hour
                    hour = error["timestamp"].strftime("%Y-%m-%d %H:00:00")
                    stats["time_distribution"][hour] += 1
        
        # Global statistics
        else:
            # Use error history for global statistics
            for error in self.error_history:
                if min_time and error["timestamp"] < min_time:
                    continue
                
                stats["total_errors"] += 1
                stats["categories"][error["category"].value] += 1
                stats["severities"][error["severity"].value] += 1
                
                # Group by hour
                hour = error["timestamp"].strftime("%Y-%m-%d %H:00:00")
                stats["time_distribution"][hour] += 1
            
            # Calculate error rate
            if self.error_history:
                stats["error_rate"] = len(self.error_history) / self.error_window_size
        
        return stats

    def get_circuit_breaker_status(self, service_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of circuit breakers.
        
        Args:
            service_key: Filter by service key
            
        Returns:
            Dict: Circuit breaker status
        """
        with self.circuit_breakers_lock:
            if service_key:
                if service_key in self.circuit_breakers:
                    return {service_key: self.circuit_breakers[service_key]}
                return {}
            
            return {k: v for k, v in self.circuit_breakers.items()}

    def stop(self) -> None:
        """Stop the fault tolerance system and clean up resources."""
        logger.info("Stopping Fault Tolerance System")
        self.stop_event.set()
        
        if self.error_rate_monitor_thread.is_alive():
            self.error_rate_monitor_thread.join(timeout=5)
            
        if self.circuit_breaker_monitor_thread.is_alive():
            self.circuit_breaker_monitor_thread.join(timeout=5)

    def _init_recovery_strategies(self) -> None:
        """Initialize default recovery strategies for error categories."""
        # Network errors: Retry with exponential backoff
        self.recovery_strategies[ErrorCategory.NETWORK] = lambda severity, retry_count, context: {
            "action": RecoveryAction.RETRY if retry_count < self.max_retries else RecoveryAction.CIRCUIT_BREAK,
            "reason": "Network error"
        }
        
        # Resource errors: Throttle and retry
        self.recovery_strategies[ErrorCategory.RESOURCE] = lambda severity, retry_count, context: {
            "action": RecoveryAction.THROTTLE if retry_count < 1 else 
                     (RecoveryAction.RETRY if retry_count < self.max_retries else RecoveryAction.ABORT),
            "reason": "Resource allocation error",
            "throttle_seconds": 5 * (retry_count + 1)
        }
        
        # Worker errors: Reassign to different worker
        self.recovery_strategies[ErrorCategory.WORKER] = lambda severity, retry_count, context: {
            "action": RecoveryAction.REASSIGN if context.get("worker_id") else 
                     (RecoveryAction.RETRY if retry_count < self.max_retries else RecoveryAction.ABORT),
            "reason": "Worker error"
        }
        
        # Task errors: Retry for transient errors, abort for others
        self.recovery_strategies[ErrorCategory.TASK] = lambda severity, retry_count, context: {
            "action": RecoveryAction.RETRY if severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM] and retry_count < self.max_retries else RecoveryAction.ABORT,
            "reason": "Task execution error"
        }
        
        # Data errors: Abort operation
        self.recovery_strategies[ErrorCategory.DATA] = lambda severity, retry_count, context: {
            "action": RecoveryAction.ABORT,
            "reason": "Data error"
        }
        
        # Hardware errors: Try fallback hardware or abort
        self.recovery_strategies[ErrorCategory.HARDWARE] = lambda severity, retry_count, context: {
            "action": RecoveryAction.FALLBACK if context.get("alternative_hardware") else RecoveryAction.ABORT,
            "reason": "Hardware error",
            "fallback": context.get("alternative_hardware")
        }
        
        # Authentication errors: Retry with new credentials or abort
        self.recovery_strategies[ErrorCategory.AUTHENTICATION] = lambda severity, retry_count, context: {
            "action": RecoveryAction.RETRY if context.get("refresh_credentials") and retry_count < 1 else RecoveryAction.ABORT,
            "reason": "Authentication error",
            "refresh_credentials": True
        }
        
        # Authorization errors: Abort operation
        self.recovery_strategies[ErrorCategory.AUTHORIZATION] = lambda severity, retry_count, context: {
            "action": RecoveryAction.ABORT,
            "reason": "Authorization error"
        }
        
        # Timeout errors: Retry with increased timeout
        self.recovery_strategies[ErrorCategory.TIMEOUT] = lambda severity, retry_count, context: {
            "action": RecoveryAction.RETRY if retry_count < self.max_retries else RecoveryAction.ABORT,
            "reason": "Timeout error",
            "increase_timeout": True,
            "timeout_multiplier": min(2.0, 1.0 + retry_count * 0.5)  # Increase timeout by 50% each retry
        }
        
        # Unknown errors: Retry limited times
        self.recovery_strategies[ErrorCategory.UNKNOWN] = lambda severity, retry_count, context: {
            "action": RecoveryAction.RETRY if retry_count < 1 else RecoveryAction.ABORT,
            "reason": "Unknown error"
        }

    def _categorize_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Categorize an error and determine its severity.
        
        Args:
            error: The exception
            context: Context information
            
        Returns:
            Dict: Error categorization
        """
        error_str = str(error)
        error_type = type(error).__name__
        
        # Default category and severity
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        
        # Network errors
        if any(s in error_str.lower() for s in ["connection", "network", "socket", "timeout", "unreachable"]):
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.MEDIUM
        
        # Resource errors
        elif any(s in error_str.lower() for s in ["resource", "memory", "capacity", "overflow", "full"]):
            category = ErrorCategory.RESOURCE
            severity = ErrorSeverity.HIGH
        
        # Worker errors
        elif "worker" in error_str.lower() or context.get("worker_id"):
            category = ErrorCategory.WORKER
            severity = ErrorSeverity.MEDIUM
        
        # Task errors
        elif "task" in error_str.lower() or context.get("task_id"):
            category = ErrorCategory.TASK
            severity = ErrorSeverity.MEDIUM
        
        # Data errors
        elif any(s in error_str.lower() for s in ["data", "invalid", "corrupt", "format", "parsing"]):
            category = ErrorCategory.DATA
            severity = ErrorSeverity.HIGH
        
        # Hardware errors
        elif any(s in error_str.lower() for s in ["hardware", "device", "gpu", "cpu", "memory"]):
            category = ErrorCategory.HARDWARE
            severity = ErrorSeverity.HIGH
        
        # Authentication errors
        elif any(s in error_str.lower() for s in ["authentication", "login", "credentials", "password"]):
            category = ErrorCategory.AUTHENTICATION
            severity = ErrorSeverity.MEDIUM
        
        # Authorization errors
        elif any(s in error_str.lower() for s in ["authorization", "permission", "access", "forbidden"]):
            category = ErrorCategory.AUTHORIZATION
            severity = ErrorSeverity.HIGH
        
        # Timeout errors
        elif "timeout" in error_str.lower():
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.MEDIUM
        
        # Adjust severity based on context
        if "critical" in error_str.lower() or context.get("critical"):
            severity = ErrorSeverity.CRITICAL
        elif "high" in error_str.lower() or context.get("high_severity"):
            severity = ErrorSeverity.HIGH
        elif "low" in error_str.lower() or context.get("low_severity"):
            severity = ErrorSeverity.LOW
        
        return {
            "error": error,
            "error_type": error_type,
            "error_message": error_str,
            "category": category,
            "severity": severity,
            "timestamp": datetime.now(),
            "context": context
        }

    def _record_error(self, 
                      error_info: Dict[str, Any], 
                      worker_id: Optional[str] = None,
                      task_id: Optional[str] = None,
                      operation_id: Optional[str] = None) -> None:
        """
        Record an error in the system.
        
        Args:
            error_info: Error information dictionary
            worker_id: Worker ID if applicable
            task_id: Task ID if applicable
            operation_id: Operation ID if applicable
        """
        # Add to error history
        self.error_history.append(error_info)
        
        # Record in worker errors
        if worker_id:
            with self.worker_errors_lock:
                self.worker_errors[worker_id].append(error_info)
        
        # Record in task errors
        if task_id:
            with self.task_errors_lock:
                self.task_errors[task_id].append(error_info)
        
        # Update circuit breaker error count if service_key is in context
        service_key = error_info["context"].get("service_key")
        if service_key:
            self._increment_circuit_error(service_key)
        
        logger.info(f"Recorded {error_info['category'].value} error with {error_info['severity'].value} severity " + 
                   f"for {'worker ' + worker_id if worker_id else ''} {'task ' + task_id if task_id else ''} " +
                   f"{'operation ' + operation_id if operation_id else ''}")

    def _get_retry_count(self, operation_id: str) -> int:
        """
        Get the current retry count for an operation.
        
        Args:
            operation_id: Operation ID
            
        Returns:
            int: Current retry count
        """
        with self.retry_counts_lock:
            return self.retry_counts.get(operation_id, 0)

    def _increment_retry_count(self, operation_id: str) -> int:
        """
        Increment the retry count for an operation.
        
        Args:
            operation_id: Operation ID
            
        Returns:
            int: New retry count
        """
        with self.retry_counts_lock:
            self.retry_counts[operation_id] += 1
            return self.retry_counts[operation_id]

    def _determine_recovery_action(self, 
                                   error_category: ErrorCategory, 
                                   error_severity: ErrorSeverity,
                                   retry_count: int,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the appropriate recovery action based on error category and severity.
        
        Args:
            error_category: Category of the error
            error_severity: Severity of the error
            retry_count: Current retry count
            context: Context information
            
        Returns:
            Dict: Recovery action and details
        """
        # If error severity is CRITICAL, abort regardless of category
        if error_severity == ErrorSeverity.CRITICAL:
            return {
                "action": RecoveryAction.ABORT,
                "reason": "Critical error"
            }
        
        # Get the appropriate recovery strategy for this category
        strategy = self.recovery_strategies.get(error_category)
        if strategy:
            return strategy(error_severity, retry_count, context)
        
        # Default strategy: retry for low/medium severity, abort for high
        if error_severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM] and retry_count < self.max_retries:
            return {
                "action": RecoveryAction.RETRY,
                "reason": "Transient error"
            }
        else:
            return {
                "action": RecoveryAction.ABORT,
                "reason": "Persistent error"
            }

    def _is_circuit_open(self, service_key: str) -> bool:
        """
        Check if a circuit breaker is open.
        
        Args:
            service_key: Service identifier
            
        Returns:
            bool: True if circuit is open
        """
        with self.circuit_breakers_lock:
            if service_key not in self.circuit_breakers:
                return False
                
            circuit = self.circuit_breakers[service_key]
            return circuit["state"] == "open"

    def _open_circuit(self, service_key: str) -> None:
        """
        Open a circuit breaker.
        
        Args:
            service_key: Service identifier
        """
        with self.circuit_breakers_lock:
            # Initialize circuit breaker if not exist
            if service_key not in self.circuit_breakers:
                self.circuit_breakers[service_key] = {
                    "state": "closed",
                    "error_count": 0,
                    "last_error_time": None,
                    "reset_time": None
                }
            
            # Open the circuit
            self.circuit_breakers[service_key]["state"] = "open"
            self.circuit_breakers[service_key]["last_error_time"] = datetime.now()
            
            logger.warning(f"Circuit breaker opened for {service_key}")

    def _increment_circuit_error(self, service_key: str) -> None:
        """
        Increment error count for a circuit breaker.
        
        Args:
            service_key: Service identifier
        """
        with self.circuit_breakers_lock:
            # Initialize circuit breaker if not exist
            if service_key not in self.circuit_breakers:
                self.circuit_breakers[service_key] = {
                    "state": "closed",
                    "error_count": 0,
                    "last_error_time": None,
                    "reset_time": None
                }
            
            circuit = self.circuit_breakers[service_key]
            circuit["error_count"] += 1
            circuit["last_error_time"] = datetime.now()
            
            # Open circuit if threshold reached
            if circuit["state"] == "closed" and circuit["error_count"] >= self.circuit_break_threshold:
                circuit["state"] = "open"
                logger.warning(f"Circuit breaker opened for {service_key} after {circuit['error_count']} errors")

    def _get_fallback(self, service_key: str) -> Optional[Any]:
        """
        Get fallback for a service.
        
        Args:
            service_key: Service identifier
            
        Returns:
            Optional[Any]: Fallback or None
        """
        return self.fallbacks.get(service_key)

    def _monitor_error_rate(self) -> None:
        """Background thread to monitor error rate."""
        last_check_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                # Check every 10 seconds
                if current_time - last_check_time >= 10:
                    # Calculate error rate
                    error_rate = len(self.error_history) / self.error_window_size if self.error_history else 0
                    
                    # Alert if error rate exceeds threshold
                    if error_rate > self.error_rate_threshold:
                        logger.warning(f"Error rate alert: {error_rate:.2f} (threshold: {self.error_rate_threshold})")
                        
                        # Notify coordinator if available
                        if self.coordinator and hasattr(self.coordinator, 'notify_alert'):
                            self.coordinator.notify_alert({
                                "type": "error_rate",
                                "error_rate": error_rate,
                                "threshold": self.error_rate_threshold,
                                "timestamp": datetime.now().isoformat()
                            })
                    
                    last_check_time = current_time
                
                # Sleep briefly
                time.sleep(1)
                
            except Exception as e:
                logger.exception(f"Error in error rate monitor: {e}")
                time.sleep(10)  # Sleep longer on error

    def _monitor_circuit_breakers(self) -> None:
        """Background thread to monitor and reset circuit breakers."""
        while not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                
                with self.circuit_breakers_lock:
                    for service_key, circuit in list(self.circuit_breakers.items()):
                        # Skip if circuit is closed
                        if circuit["state"] != "open":
                            continue
                        
                        # Check if timeout elapsed
                        if circuit["last_error_time"]:
                            elapsed_seconds = (current_time - circuit["last_error_time"]).total_seconds()
                            
                            if elapsed_seconds >= self.circuit_break_timeout:
                                # Reset to half-open state
                                circuit["state"] = "half-open"
                                circuit["reset_time"] = current_time
                                logger.info(f"Circuit breaker for {service_key} reset to half-open after {elapsed_seconds:.1f}s")
                
                # Sleep for a short time
                time.sleep(5)
                
            except Exception as e:
                logger.exception(f"Error in circuit breaker monitor: {e}")
                time.sleep(10)  # Sleep longer on error