#!/usr/bin/env python3
"""
Distributed Testing Framework - Error Handler

This module implements enhanced error handling for the distributed testing framework,
providing graceful failure handling, error categorization, and intelligent recovery strategies.

Key features:
- Error categorization
- Recovery strategies for different error types
- Automatic retry with exponential backoff
- Error aggregation and pattern detection
- Comprehensive error reporting
"""

import os
import re
import abc
import enum
import json
import time
import random
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple, Set, Type, Union
from dataclasses import dataclass, field
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("distributed_error_handler")

class ErrorCategory(enum.Enum):
    """Standardized error categories for distributed testing."""
    
    # Resource-related errors
    RESOURCE_EXHAUSTED = "resource_exhausted"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    
    # Network-related errors
    NETWORK_TIMEOUT = "network_timeout"
    NETWORK_CONNECTION_ERROR = "network_connection_error"
    NETWORK_SERVER_ERROR = "network_server_error"
    
    # Test execution errors
    TEST_ASSERTION_ERROR = "test_assertion_error"
    TEST_IMPORT_ERROR = "test_import_error"
    TEST_DEPENDENCY_ERROR = "test_dependency_error"
    TEST_SYNTAX_ERROR = "test_syntax_error"
    
    # Hardware-related errors
    HARDWARE_NOT_AVAILABLE = "hardware_not_available"
    HARDWARE_MISMATCH = "hardware_mismatch"
    HARDWARE_COMPATIBILITY_ERROR = "hardware_compatibility_error"
    
    # System errors
    SYSTEM_CRASH = "system_crash"
    SYSTEM_RESOURCE_LIMIT = "system_resource_limit"
    
    # Worker errors
    WORKER_DISCONNECTED = "worker_disconnected"
    WORKER_TIMEOUT = "worker_timeout"
    WORKER_CRASHED = "worker_crashed"
    
    # Unknown/other
    UNKNOWN = "unknown"


class RetryPolicy:
    """Configures retry behavior for tasks based on error categories."""
    
    def __init__(
        self,
        max_retries=3,
        retry_delay_seconds=60,
        retry_backoff_factor=2,
        max_retry_delay_seconds=3600,
        retry_jitter=0.2,
        categories_to_retry=None,
        categories_to_skip=None
    ):
        """Initialize retry policy with configuration.
        
        Args:
            max_retries (int): Maximum number of retry attempts
            retry_delay_seconds (int): Initial delay between retries in seconds
            retry_backoff_factor (float): Factor to multiply delay by after each attempt
            max_retry_delay_seconds (int): Maximum delay between retries in seconds
            retry_jitter (float): Random jitter factor (0-1) to add to delay
            categories_to_retry (List[ErrorCategory], optional): Error categories to retry
            categories_to_skip (List[ErrorCategory], optional): Error categories to never retry
        """
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.retry_backoff_factor = retry_backoff_factor
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.retry_jitter = retry_jitter
        self.categories_to_retry = categories_to_retry or []
        self.categories_to_skip = categories_to_skip or []
    
    def should_retry(self, error_category: ErrorCategory, attempt_count: int) -> bool:
        """Determine if a task should be retried.
        
        Args:
            error_category (ErrorCategory): The category of the error
            attempt_count (int): The number of attempts made so far
            
        Returns:
            bool: True if the task should be retried, False otherwise
        """
        # Skip if max retries exceeded
        if attempt_count >= self.max_retries:
            return False
        
        # Skip if error category is in categories_to_skip
        if error_category in self.categories_to_skip:
            return False
        
        # Retry if categories_to_retry is empty (retry all) or error category is in categories_to_retry
        return not self.categories_to_retry or error_category in self.categories_to_retry
    
    def get_retry_delay(self, attempt_count: int) -> float:
        """Calculate retry delay with exponential backoff and jitter.
        
        Args:
            attempt_count (int): The number of attempts made so far
            
        Returns:
            float: The delay in seconds before the next retry
        """
        # Calculate base delay with exponential backoff
        delay = self.retry_delay_seconds * (self.retry_backoff_factor ** (attempt_count - 1))
        
        # Apply maximum delay limit
        delay = min(delay, self.max_retry_delay_seconds)
        
        # Add random jitter
        jitter_factor = 1.0 + random.uniform(-self.retry_jitter, self.retry_jitter)
        delay = delay * jitter_factor
        
        return delay


class RecoveryStrategy(abc.ABC):
    """Base class for error recovery strategies."""
    
    @abc.abstractmethod
    def recover(self, task_id: str, error: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to recover from an error.
        
        Args:
            task_id (str): The ID of the task that failed
            error (Dict[str, Any]): Error information
            context (Dict[str, Any]): Additional context for recovery
            
        Returns:
            Dict[str, Any]: Recovery result including success status and actions taken
        """
        pass
    
    @abc.abstractmethod
    def is_applicable(self, error_category: ErrorCategory, hardware_type: Optional[str] = None) -> bool:
        """Check if this strategy applies to the error.
        
        Args:
            error_category (ErrorCategory): The category of the error
            hardware_type (str, optional): The type of hardware involved
            
        Returns:
            bool: True if this strategy is applicable, False otherwise
        """
        pass


class ResourceRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for resource-related errors."""
    
    def recover(self, task_id: str, error: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from resource errors by freeing or reallocating resources.
        
        Args:
            task_id (str): The ID of the task that failed
            error (Dict[str, Any]): Error information
            context (Dict[str, Any]): Additional context for recovery
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        recovery_result = {
            "success": False,
            "strategy": "resource",
            "actions_taken": [],
            "retry_recommended": False
        }
        
        # Get worker ID from context
        worker_id = context.get("worker_id")
        if not worker_id:
            recovery_result["message"] = "No worker ID in context, cannot recover"
            return recovery_result
        
        # Determine resource recovery action based on error category
        error_category = error.get("category", ErrorCategory.UNKNOWN)
        
        if error_category == ErrorCategory.RESOURCE_EXHAUSTED:
            # Free resources on the worker
            recovery_result["actions_taken"].append("request_resource_cleanup")
            recovery_result["retry_recommended"] = True
            
        elif error_category == ErrorCategory.RESOURCE_UNAVAILABLE:
            # Mark resource as unavailable and try different resource
            recovery_result["actions_taken"].append("mark_resource_unavailable")
            recovery_result["actions_taken"].append("reallocate_task")
            recovery_result["retry_recommended"] = True
            
        recovery_result["success"] = len(recovery_result["actions_taken"]) > 0
        return recovery_result
    
    def is_applicable(self, error_category: ErrorCategory, hardware_type: Optional[str] = None) -> bool:
        """Check if this strategy applies to resource errors.
        
        Args:
            error_category (ErrorCategory): The category of the error
            hardware_type (str, optional): The type of hardware involved
            
        Returns:
            bool: True if this strategy is applicable, False otherwise
        """
        return error_category in [
            ErrorCategory.RESOURCE_EXHAUSTED,
            ErrorCategory.RESOURCE_UNAVAILABLE,
            ErrorCategory.RESOURCE_NOT_FOUND
        ]


class NetworkRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for network-related errors."""
    
    def recover(self, task_id: str, error: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from network errors by reconnecting or failing over.
        
        Args:
            task_id (str): The ID of the task that failed
            error (Dict[str, Any]): Error information
            context (Dict[str, Any]): Additional context for recovery
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        recovery_result = {
            "success": False,
            "strategy": "network",
            "actions_taken": [],
            "retry_recommended": False
        }
        
        # Get worker ID from context
        worker_id = context.get("worker_id")
        if not worker_id:
            recovery_result["message"] = "No worker ID in context, cannot recover"
            return recovery_result
        
        # Determine network recovery action based on error category
        error_category = error.get("category", ErrorCategory.UNKNOWN)
        
        if error_category == ErrorCategory.NETWORK_TIMEOUT:
            # Increase timeout for retry
            recovery_result["actions_taken"].append("increase_timeout")
            recovery_result["retry_recommended"] = True
            
        elif error_category == ErrorCategory.NETWORK_CONNECTION_ERROR:
            # Request reconnection and retry
            recovery_result["actions_taken"].append("reconnect")
            recovery_result["retry_recommended"] = True
            
        elif error_category == ErrorCategory.NETWORK_SERVER_ERROR:
            # Failover to another server if available
            recovery_result["actions_taken"].append("failover")
            recovery_result["retry_recommended"] = True
            
        recovery_result["success"] = len(recovery_result["actions_taken"]) > 0
        return recovery_result
    
    def is_applicable(self, error_category: ErrorCategory, hardware_type: Optional[str] = None) -> bool:
        """Check if this strategy applies to network errors.
        
        Args:
            error_category (ErrorCategory): The category of the error
            hardware_type (str, optional): The type of hardware involved
            
        Returns:
            bool: True if this strategy is applicable, False otherwise
        """
        return error_category in [
            ErrorCategory.NETWORK_TIMEOUT,
            ErrorCategory.NETWORK_CONNECTION_ERROR,
            ErrorCategory.NETWORK_SERVER_ERROR
        ]


class HardwareRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for hardware-related errors."""
    
    def recover(self, task_id: str, error: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from hardware errors by reallocating to different hardware.
        
        Args:
            task_id (str): The ID of the task that failed
            error (Dict[str, Any]): Error information
            context (Dict[str, Any]): Additional context for recovery
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        recovery_result = {
            "success": False,
            "strategy": "hardware",
            "actions_taken": [],
            "retry_recommended": False
        }
        
        # Get worker ID and hardware type from context
        worker_id = context.get("worker_id")
        hardware_type = context.get("hardware_type")
        
        if not worker_id or not hardware_type:
            recovery_result["message"] = "Missing worker ID or hardware type in context"
            return recovery_result
        
        # Determine hardware recovery action based on error category
        error_category = error.get("category", ErrorCategory.UNKNOWN)
        
        if error_category == ErrorCategory.HARDWARE_NOT_AVAILABLE:
            # Mark hardware as unavailable and reallocate to different hardware
            recovery_result["actions_taken"].append(f"mark_hardware_unavailable:{hardware_type}:{worker_id}")
            recovery_result["actions_taken"].append("reallocate_to_alternative_hardware")
            recovery_result["retry_recommended"] = True
            
        elif error_category == ErrorCategory.HARDWARE_MISMATCH:
            # Reallocate to compatible hardware
            recovery_result["actions_taken"].append("reallocate_to_compatible_hardware")
            recovery_result["retry_recommended"] = True
            
        elif error_category == ErrorCategory.HARDWARE_COMPATIBILITY_ERROR:
            # Update hardware requirements and reallocate
            recovery_result["actions_taken"].append("update_hardware_requirements")
            recovery_result["actions_taken"].append("reallocate_to_compatible_hardware")
            recovery_result["retry_recommended"] = True
            
        recovery_result["success"] = len(recovery_result["actions_taken"]) > 0
        return recovery_result
    
    def is_applicable(self, error_category: ErrorCategory, hardware_type: Optional[str] = None) -> bool:
        """Check if this strategy applies to hardware errors.
        
        Args:
            error_category (ErrorCategory): The category of the error
            hardware_type (str, optional): The type of hardware involved
            
        Returns:
            bool: True if this strategy is applicable, False otherwise
        """
        return error_category in [
            ErrorCategory.HARDWARE_NOT_AVAILABLE,
            ErrorCategory.HARDWARE_MISMATCH,
            ErrorCategory.HARDWARE_COMPATIBILITY_ERROR
        ]


class WorkerRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for worker-related errors."""
    
    def recover(self, task_id: str, error: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from worker errors by reassigning tasks.
        
        Args:
            task_id (str): The ID of the task that failed
            error (Dict[str, Any]): Error information
            context (Dict[str, Any]): Additional context for recovery
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        recovery_result = {
            "success": False,
            "strategy": "worker",
            "actions_taken": [],
            "retry_recommended": False
        }
        
        # Get worker ID from context
        worker_id = context.get("worker_id")
        if not worker_id:
            recovery_result["message"] = "No worker ID in context, cannot recover"
            return recovery_result
        
        # Determine worker recovery action based on error category
        error_category = error.get("category", ErrorCategory.UNKNOWN)
        
        if error_category == ErrorCategory.WORKER_DISCONNECTED:
            # Mark worker as unavailable and reassign task
            recovery_result["actions_taken"].append(f"mark_worker_unavailable:{worker_id}")
            recovery_result["actions_taken"].append("reassign_task")
            recovery_result["retry_recommended"] = True
            
        elif error_category == ErrorCategory.WORKER_TIMEOUT:
            # Mark worker as slow and reassign task with increased timeout
            recovery_result["actions_taken"].append(f"mark_worker_slow:{worker_id}")
            recovery_result["actions_taken"].append("reassign_task_with_increased_timeout")
            recovery_result["retry_recommended"] = True
            
        elif error_category == ErrorCategory.WORKER_CRASHED:
            # Mark worker as crashed and reassign task to different worker
            recovery_result["actions_taken"].append(f"mark_worker_crashed:{worker_id}")
            recovery_result["actions_taken"].append("reassign_task_to_different_worker")
            recovery_result["retry_recommended"] = True
            
        recovery_result["success"] = len(recovery_result["actions_taken"]) > 0
        return recovery_result
    
    def is_applicable(self, error_category: ErrorCategory, hardware_type: Optional[str] = None) -> bool:
        """Check if this strategy applies to worker errors.
        
        Args:
            error_category (ErrorCategory): The category of the error
            hardware_type (str, optional): The type of hardware involved
            
        Returns:
            bool: True if this strategy is applicable, False otherwise
        """
        return error_category in [
            ErrorCategory.WORKER_DISCONNECTED,
            ErrorCategory.WORKER_TIMEOUT,
            ErrorCategory.WORKER_CRASHED
        ]


class TestExecutionRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for test execution errors."""
    
    def recover(self, task_id: str, error: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from test execution errors with appropriate strategies.
        
        Args:
            task_id (str): The ID of the task that failed
            error (Dict[str, Any]): Error information
            context (Dict[str, Any]): Additional context for recovery
            
        Returns:
            Dict[str, Any]: Recovery result
        """
        recovery_result = {
            "success": False,
            "strategy": "test_execution",
            "actions_taken": [],
            "retry_recommended": False
        }
        
        # Determine test execution recovery action based on error category
        error_category = error.get("category", ErrorCategory.UNKNOWN)
        
        if error_category == ErrorCategory.TEST_IMPORT_ERROR:
            # Check for missing dependencies and install if possible
            recovery_result["actions_taken"].append("check_dependencies")
            recovery_result["retry_recommended"] = True
            
        elif error_category == ErrorCategory.TEST_DEPENDENCY_ERROR:
            # Resolve dependencies and retry
            recovery_result["actions_taken"].append("resolve_dependencies")
            recovery_result["retry_recommended"] = True
            
        elif error_category == ErrorCategory.TEST_ASSERTION_ERROR:
            # Don't retry, test is genuinely failing
            recovery_result["actions_taken"].append("record_test_failure")
            recovery_result["retry_recommended"] = False
            
        elif error_category == ErrorCategory.TEST_SYNTAX_ERROR:
            # Don't retry, test has syntax issues
            recovery_result["actions_taken"].append("record_test_error")
            recovery_result["retry_recommended"] = False
            
        recovery_result["success"] = len(recovery_result["actions_taken"]) > 0
        return recovery_result
    
    def is_applicable(self, error_category: ErrorCategory, hardware_type: Optional[str] = None) -> bool:
        """Check if this strategy applies to test execution errors.
        
        Args:
            error_category (ErrorCategory): The category of the error
            hardware_type (str, optional): The type of hardware involved
            
        Returns:
            bool: True if this strategy is applicable, False otherwise
        """
        return error_category in [
            ErrorCategory.TEST_ASSERTION_ERROR,
            ErrorCategory.TEST_IMPORT_ERROR,
            ErrorCategory.TEST_DEPENDENCY_ERROR,
            ErrorCategory.TEST_SYNTAX_ERROR
        ]


class ErrorAggregator:
    """Aggregates and analyzes related errors."""
    
    def __init__(self, similarity_threshold=0.8, max_errors_per_group=100):
        """Initialize the error aggregator.
        
        Args:
            similarity_threshold (float): Threshold for considering errors similar (0-1)
            max_errors_per_group (int): Maximum number of errors to store per group
        """
        self.error_groups = {}  # group_id -> list of errors
        self.similarity_threshold = similarity_threshold
        self.max_errors_per_group = max_errors_per_group
        self.next_group_id = 1
    
    def add_error(self, error: Dict[str, Any]) -> int:
        """Add an error to the appropriate group.
        
        Args:
            error (Dict[str, Any]): Error information
            
        Returns:
            int: The group ID the error was added to
        """
        # Calculate similarity with existing groups
        best_group_id = None
        best_similarity = 0
        
        for group_id, group_errors in self.error_groups.items():
            # Compare with the first error in the group (the representative)
            if group_errors:
                similarity = self._calculate_similarity(error, group_errors[0])
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_group_id = group_id
        
        # If no similar group found, create a new group
        if best_group_id is None:
            best_group_id = self.next_group_id
            self.error_groups[best_group_id] = []
            self.next_group_id += 1
        
        # Add error to group (up to max_errors_per_group)
        group = self.error_groups[best_group_id]
        if len(group) < self.max_errors_per_group:
            group.append(error)
        
        return best_group_id
    
    def get_error_groups(self) -> Dict[int, List[Dict[str, Any]]]:
        """Get all error groups.
        
        Returns:
            Dict[int, List[Dict[str, Any]]]: Group ID -> list of errors
        """
        return self.error_groups
    
    def get_frequent_errors(self, min_count=3) -> List[Tuple[int, int, Dict[str, Any]]]:
        """Get frequently occurring errors.
        
        Args:
            min_count (int): Minimum number of occurrences to be considered frequent
            
        Returns:
            List[Tuple[int, int, Dict[str, Any]]]: List of (group_id, count, representative_error)
        """
        result = []
        
        for group_id, errors in self.error_groups.items():
            if len(errors) >= min_count:
                result.append((group_id, len(errors), errors[0]))
        
        # Sort by frequency (descending)
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
    
    def _calculate_similarity(self, error1: Dict[str, Any], error2: Dict[str, Any]) -> float:
        """Calculate similarity between two errors.
        
        Args:
            error1 (Dict[str, Any]): First error
            error2 (Dict[str, Any]): Second error
            
        Returns:
            float: Similarity score (0-1)
        """
        # Start with a base score
        score = 0.0
        total_weight = 0.0
        
        # Check type similarity (high weight)
        if error1.get("type") == error2.get("type"):
            score += 0.4
        total_weight += 0.4
        
        # Check category similarity (high weight)
        if error1.get("category") == error2.get("category"):
            score += 0.3
        total_weight += 0.3
        
        # Check message similarity using simple string comparison (medium weight)
        msg1 = error1.get("message", "")
        msg2 = error2.get("message", "")
        
        if msg1 and msg2:
            # Simple string similarity using Jaccard index of words
            words1 = set(re.findall(r'\w+', msg1.lower()))
            words2 = set(re.findall(r'\w+', msg2.lower()))
            
            if words1 and words2:
                jaccard = len(words1.intersection(words2)) / len(words1.union(words2))
                score += 0.2 * jaccard
            
        total_weight += 0.2
        
        # Check hardware context similarity (low weight)
        hw1 = error1.get("hardware_context", {})
        hw2 = error2.get("hardware_context", {})
        
        if hw1 and hw2:
            hw_similarity = 0.0
            
            # Check hardware type
            if hw1.get("hardware_type") == hw2.get("hardware_type"):
                hw_similarity += 1.0
                
            # Normalize hardware similarity
            score += 0.1 * hw_similarity
            
        total_weight += 0.1
        
        # Normalize score
        if total_weight > 0:
            final_score = score / total_weight
        else:
            final_score = 0.0
            
        return final_score


class DistributedErrorHandler:
    """Handles errors in distributed test execution."""
    
    def __init__(self, config=None):
        """Initialize error handler with configuration.
        
        Args:
            config (Dict[str, Any], optional): Configuration for the error handler
        """
        self.config = config or {}
        self.error_categories = self._initialize_error_categories()
        self.retry_policies = self._initialize_retry_policies()
        self.active_errors = {}  # task_id -> error info
        self.error_patterns = self._initialize_error_patterns()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.error_aggregator = ErrorAggregator(
            similarity_threshold=self.config.get("similarity_threshold", 0.8)
        )
        self.logger = logging.getLogger("distributed_error_handler")
    
    def _initialize_error_categories(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error category mapping.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping error types to categories
        """
        # Default error category mapping based on error type
        default_categories = {
            "ResourceError": ErrorCategory.RESOURCE_UNAVAILABLE,
            "MemoryError": ErrorCategory.RESOURCE_EXHAUSTED,
            "TimeoutError": ErrorCategory.NETWORK_TIMEOUT,
            "ConnectionError": ErrorCategory.NETWORK_CONNECTION_ERROR,
            "ConnectionRefusedError": ErrorCategory.NETWORK_CONNECTION_ERROR,
            "ConnectionResetError": ErrorCategory.NETWORK_CONNECTION_ERROR,
            "HTTPError": ErrorCategory.NETWORK_SERVER_ERROR,
            "ImportError": ErrorCategory.TEST_IMPORT_ERROR,
            "ModuleNotFoundError": ErrorCategory.TEST_IMPORT_ERROR,
            "AssertionError": ErrorCategory.TEST_ASSERTION_ERROR,
            "SyntaxError": ErrorCategory.TEST_SYNTAX_ERROR,
            "HardwareError": ErrorCategory.HARDWARE_NOT_AVAILABLE,
            "WorkerDisconnectedError": ErrorCategory.WORKER_DISCONNECTED,
            "WorkerTimeoutError": ErrorCategory.WORKER_TIMEOUT,
            "WorkerCrashError": ErrorCategory.WORKER_CRASHED,
            "SystemError": ErrorCategory.SYSTEM_CRASH,
            "RuntimeError": ErrorCategory.UNKNOWN
        }
        
        # Merge with custom categories from config
        custom_categories = self.config.get("error_categories", {})
        return {**default_categories, **custom_categories}
    
    def _initialize_retry_policies(self) -> Dict[ErrorCategory, RetryPolicy]:
        """Initialize retry policies for different error categories.
        
        Returns:
            Dict[ErrorCategory, RetryPolicy]: Dictionary mapping error categories to retry policies
        """
        # Default retry policies
        default_policies = {
            # Resource errors - retry with backoff
            ErrorCategory.RESOURCE_EXHAUSTED: RetryPolicy(
                max_retries=5,
                retry_delay_seconds=30,
                retry_backoff_factor=2
            ),
            ErrorCategory.RESOURCE_UNAVAILABLE: RetryPolicy(
                max_retries=3,
                retry_delay_seconds=60,
                retry_backoff_factor=2
            ),
            
            # Network errors - retry with quick backoff
            ErrorCategory.NETWORK_TIMEOUT: RetryPolicy(
                max_retries=5,
                retry_delay_seconds=10,
                retry_backoff_factor=1.5
            ),
            ErrorCategory.NETWORK_CONNECTION_ERROR: RetryPolicy(
                max_retries=5,
                retry_delay_seconds=15,
                retry_backoff_factor=1.5
            ),
            ErrorCategory.NETWORK_SERVER_ERROR: RetryPolicy(
                max_retries=3,
                retry_delay_seconds=30,
                retry_backoff_factor=2
            ),
            
            # Worker errors - retry on different worker
            ErrorCategory.WORKER_DISCONNECTED: RetryPolicy(
                max_retries=3,
                retry_delay_seconds=5,
                retry_backoff_factor=1.5
            ),
            ErrorCategory.WORKER_TIMEOUT: RetryPolicy(
                max_retries=3,
                retry_delay_seconds=10,
                retry_backoff_factor=2
            ),
            ErrorCategory.WORKER_CRASHED: RetryPolicy(
                max_retries=3,
                retry_delay_seconds=15,
                retry_backoff_factor=2
            ),
            
            # Hardware errors - retry with different hardware
            ErrorCategory.HARDWARE_NOT_AVAILABLE: RetryPolicy(
                max_retries=3,
                retry_delay_seconds=10,
                retry_backoff_factor=1
            ),
            ErrorCategory.HARDWARE_MISMATCH: RetryPolicy(
                max_retries=2,
                retry_delay_seconds=5,
                retry_backoff_factor=1
            ),
            
            # Test errors - limited retries
            ErrorCategory.TEST_IMPORT_ERROR: RetryPolicy(
                max_retries=2,
                retry_delay_seconds=5,
                retry_backoff_factor=1
            ),
            ErrorCategory.TEST_DEPENDENCY_ERROR: RetryPolicy(
                max_retries=2,
                retry_delay_seconds=5,
                retry_backoff_factor=1
            ),
            
            # No retry for these categories
            ErrorCategory.TEST_ASSERTION_ERROR: RetryPolicy(
                max_retries=0
            ),
            ErrorCategory.TEST_SYNTAX_ERROR: RetryPolicy(
                max_retries=0
            ),
            
            # Default policy for unknown errors
            ErrorCategory.UNKNOWN: RetryPolicy(
                max_retries=1,
                retry_delay_seconds=30,
                retry_backoff_factor=2
            )
        }
        
        # Merge with custom policies from config
        custom_policies = {}
        for category_name, policy_config in self.config.get("retry_policies", {}).items():
            try:
                category = ErrorCategory[category_name]
                custom_policies[category] = RetryPolicy(**policy_config)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Invalid retry policy configuration: {e}")
        
        return {**default_policies, **custom_policies}
    
    def _initialize_error_patterns(self) -> Dict[str, ErrorCategory]:
        """Initialize patterns for matching error messages to categories.
        
        Returns:
            Dict[str, ErrorCategory]: Dictionary mapping regex patterns to error categories
        """
        # Default error message patterns
        default_patterns = {
            # Resource errors
            r"(?i)out\s+of\s+memory": ErrorCategory.RESOURCE_EXHAUSTED,
            r"(?i)memory\s+allocation\s+failed": ErrorCategory.RESOURCE_EXHAUSTED,
            r"(?i)resource\s+temporarily\s+unavailable": ErrorCategory.RESOURCE_UNAVAILABLE,
            r"(?i)disk\s+quota\s+exceeded": ErrorCategory.RESOURCE_EXHAUSTED,
            
            # Network errors
            r"(?i)connection\s+(?:timed?\s*out|timeout)": ErrorCategory.NETWORK_TIMEOUT,
            r"(?i)could\s+not\s+connect": ErrorCategory.NETWORK_CONNECTION_ERROR,
            r"(?i)connection\s+(?:refused|reset)": ErrorCategory.NETWORK_CONNECTION_ERROR,
            r"(?i)server\s+error": ErrorCategory.NETWORK_SERVER_ERROR,
            r"(?i)5\d\d\s+(?:error|response)": ErrorCategory.NETWORK_SERVER_ERROR,
            
            # Hardware errors
            r"(?i)(?:GPU|CUDA|ROCm|TPU)\s+not\s+available": ErrorCategory.HARDWARE_NOT_AVAILABLE,
            r"(?i)incompatible\s+hardware": ErrorCategory.HARDWARE_COMPATIBILITY_ERROR,
            r"(?i)hardware\s+(?:mismatch|incompatible)": ErrorCategory.HARDWARE_MISMATCH,
            
            # Worker errors
            r"(?i)worker\s+disconnected": ErrorCategory.WORKER_DISCONNECTED,
            r"(?i)worker\s+timed?\s*out": ErrorCategory.WORKER_TIMEOUT,
            r"(?i)worker\s+crashed": ErrorCategory.WORKER_CRASHED,
            
            # Test execution errors
            r"(?i)module\s+(?:not\s+found|missing)": ErrorCategory.TEST_IMPORT_ERROR,
            r"(?i)no\s+module\s+named": ErrorCategory.TEST_IMPORT_ERROR,
            r"(?i)import\s+error": ErrorCategory.TEST_IMPORT_ERROR,
            r"(?i)assertion\s+(?:failed|error)": ErrorCategory.TEST_ASSERTION_ERROR,
            r"(?i)syntax\s+error": ErrorCategory.TEST_SYNTAX_ERROR,
            r"(?i)(?:circular|missing|unresolved)\s+dependency": ErrorCategory.TEST_DEPENDENCY_ERROR
        }
        
        # Merge with custom patterns from config
        custom_patterns = {}
        for pattern, category_name in self.config.get("error_patterns", {}).items():
            try:
                category = ErrorCategory[category_name]
                custom_patterns[pattern] = category
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Invalid error pattern configuration: {e}")
        
        return {**default_patterns, **custom_patterns}
    
    def _initialize_recovery_strategies(self) -> List[RecoveryStrategy]:
        """Initialize recovery strategies.
        
        Returns:
            List[RecoveryStrategy]: List of recovery strategy instances
        """
        return [
            ResourceRecoveryStrategy(),
            NetworkRecoveryStrategy(),
            HardwareRecoveryStrategy(),
            WorkerRecoveryStrategy(),
            TestExecutionRecoveryStrategy()
        ]
    
    def categorize_error(self, error: Dict[str, Any]) -> ErrorCategory:
        """Categorize an error based on type and message.
        
        Args:
            error (Dict[str, Any]): Error information
            
        Returns:
            ErrorCategory: The category of the error
        """
        error_type = error.get("type", "")
        error_message = error.get("message", "")
        
        # Check if error type is directly mapped
        if error_type in self.error_categories:
            return self.error_categories[error_type]
        
        # Check if error message matches any pattern
        for pattern, category in self.error_patterns.items():
            if re.search(pattern, error_message):
                return category
        
        # Default to unknown category
        return ErrorCategory.UNKNOWN
    
    def should_retry(self, task_id: str, error: Dict[str, Any], attempt_count: int) -> Tuple[bool, Optional[float]]:
        """Determine if a task should be retried based on error category.
        
        Args:
            task_id (str): The ID of the task that failed
            error (Dict[str, Any]): Error information
            attempt_count (int): The number of attempts made so far
            
        Returns:
            Tuple[bool, Optional[float]]: (should_retry, retry_delay_seconds)
        """
        # Get error category
        category = error.get("category", ErrorCategory.UNKNOWN)
        
        # Get retry policy for this category
        retry_policy = self.retry_policies.get(category, self.retry_policies[ErrorCategory.UNKNOWN])
        
        # Check if retry is recommended
        should_retry = retry_policy.should_retry(category, attempt_count)
        
        # Calculate retry delay if retry is recommended
        retry_delay = retry_policy.get_retry_delay(attempt_count) if should_retry else None
        
        return should_retry, retry_delay
    
    def get_recovery_strategy(self, error_category: ErrorCategory, hardware_type: Optional[str] = None) -> Optional[RecoveryStrategy]:
        """Get appropriate recovery strategy for error + hardware combo.
        
        Args:
            error_category (ErrorCategory): The category of the error
            hardware_type (str, optional): The type of hardware involved
            
        Returns:
            Optional[RecoveryStrategy]: The recovery strategy to use, or None if none applicable
        """
        for strategy in self.recovery_strategies:
            if strategy.is_applicable(error_category, hardware_type):
                return strategy
        
        return None
    
    def aggregate_related_errors(self, new_error: Dict[str, Any]) -> int:
        """Group related errors together for analysis.
        
        Args:
            new_error (Dict[str, Any]): New error information
            
        Returns:
            int: The group ID the error was added to
        """
        return self.error_aggregator.add_error(new_error)
    
    def log_and_report_error(self, task_id: str, error: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> None:
        """Log and report error with additional context.
        
        Args:
            task_id (str): The ID of the task that failed
            error (Dict[str, Any]): Error information
            context (Dict[str, Any], optional): Additional context for the error
        """
        # Ensure error has a category
        if "category" not in error:
            error["category"] = self.categorize_error(error)
        
        # Add timestamp and task ID
        error["timestamp"] = datetime.now().isoformat()
        error["task_id"] = task_id
        
        # Add context if provided
        if context:
            error["context"] = context
        
        # Log the error
        self.logger.error(
            f"Task {task_id} failed with {error['type']}: {error['message']} "
            f"(Category: {error['category'].value})"
        )
        
        # Store in active errors
        self.active_errors[task_id] = error
        
        # Aggregate with related errors
        group_id = self.aggregate_related_errors(error)
        error["group_id"] = group_id
        
        # Check for frequent error patterns
        frequent_errors = self.error_aggregator.get_frequent_errors()
        for group_id, count, representative in frequent_errors:
            if count >= 5 and representative.get("category") == error.get("category"):
                self.logger.warning(
                    f"Detected frequent error pattern: {representative['type']} occurred {count} times. "
                    f"Category: {representative['category'].value}"
                )
    
    def handle_error(self, task_id: str, error: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main entry point for error handling.
        
        Args:
            task_id (str): The ID of the task that failed
            error (Dict[str, Any]): Error information
            context (Dict[str, Any], optional): Additional context for the error
            
        Returns:
            Dict[str, Any]: Error handling result
        """
        # Ensure context exists
        context = context or {}
        
        # Initialize result
        result = {
            "task_id": task_id,
            "error_handled": False,
            "retry": False,
            "retry_delay": None,
            "recovery_action": None,
            "error_category": None
        }
        
        # Categorize error if not already categorized
        if "category" not in error:
            error["category"] = self.categorize_error(error)
        
        result["error_category"] = error["category"]
        
        # Log and report the error
        self.log_and_report_error(task_id, error, context)
        
        # Get attempt count from context
        attempt_count = context.get("attempt_count", 1)
        
        # Check if retry is recommended
        should_retry, retry_delay = self.should_retry(task_id, error, attempt_count)
        result["retry"] = should_retry
        result["retry_delay"] = retry_delay
        
        # Apply recovery strategy if applicable
        hardware_type = context.get("hardware_type")
        recovery_strategy = self.get_recovery_strategy(error["category"], hardware_type)
        
        if recovery_strategy:
            recovery_result = recovery_strategy.recover(task_id, error, context)
            result["recovery_action"] = recovery_result
            result["error_handled"] = recovery_result.get("success", False)
            
            # If the recovery strategy recommends retry, override previous retry decision
            if recovery_result.get("retry_recommended"):
                result["retry"] = True
        
        return result


# Example usage
if __name__ == "__main__":
    # Create error handler
    handler = DistributedErrorHandler()
    
    # Example error
    error = {
        "type": "ConnectionError",
        "message": "Connection refused by remote host",
        "traceback": "...stack trace...",
        "hardware_context": {
            "hardware_type": "cuda",
            "device_id": 0
        }
    }
    
    # Handle error
    context = {
        "worker_id": "worker1",
        "hardware_type": "cuda",
        "attempt_count": 1
    }
    
    result = handler.handle_error("task123", error, context)
    
    # Print result
    print(json.dumps(result, indent=2, default=str))