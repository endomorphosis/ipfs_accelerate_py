#!/usr/bin/env python3
"""
Distributed Error Handler for Distributed Testing Framework

This module provides a comprehensive error handling system for the distributed testing
framework. It categorizes errors, provides graceful failure handling, supports automatic
retry for transient failures, and aggregates related failures to simplify troubleshooting.

Key features:
- Error categorization by type and severity
- Graceful failure handling with customizable recovery strategies
- Automatic retry with configurable retry policies
- Error aggregation for related test failures
- Detailed error reporting with context-aware information
- Integration with execution orchestrator and dependency manager
- Customizable error hooks for specialized handling
"""

import anyio
import asyncio
import logging
import time
import traceback
import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import concurrent.futures

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("distributed_error_handler")


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    INFO = "info"           # Informational only, no action needed
    LOW = "low"             # Minor issue, simple retry may resolve
    MEDIUM = "medium"       # Significant issue requiring attention
    HIGH = "high"           # Critical issue affecting test execution
    CRITICAL = "critical"   # Severe issue affecting system stability


class ErrorType(Enum):
    """Types of errors encountered in distributed testing."""
    
    # Infrastructure errors
    NETWORK = "network"             # Network connectivity issues
    RESOURCE = "resource"           # Resource allocation/availability issues
    HARDWARE = "hardware"           # Hardware-related failures
    SYSTEM = "system"               # Operating system issues
    DATABASE = "database"           # Database access/query issues
    
    # Software errors
    DEPENDENCY = "dependency"       # Dependency resolution failures
    TIMEOUT = "timeout"             # Operation timeouts
    VALIDATION = "validation"       # Data validation failures
    ASSERTION = "assertion"         # Test assertion failures
    CONFIGURATION = "configuration" # Configuration issues
    
    # Security errors
    AUTHENTICATION = "authentication"  # Authentication failures
    AUTHORIZATION = "authorization"    # Authorization issues
    
    # Coordination errors
    WORKER = "worker"               # Worker/agent failures
    COORDINATION = "coordination"   # Worker coordination issues
    SCHEDULING = "scheduling"       # Test scheduling issues
    STATE = "state"                 # State management issues
    
    # Test errors
    TEST_SETUP = "test_setup"       # Test setup failures
    TEST_EXECUTION = "test_execution" # Test execution failures
    TEST_TEARDOWN = "test_teardown" # Test teardown failures
    TEST_ENVIRONMENT = "test_environment" # Test environment issues
    
    # Unknown errors
    UNKNOWN = "unknown"             # Unclassified errors


@dataclass
class ErrorContext:
    """Context information for an error."""
    component: str                      # Component where error occurred
    operation: str                      # Operation being performed
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None       # User ID associated with operation
    request_id: Optional[str] = None    # Request ID associated with operation
    environment: Dict[str, Any] = field(default_factory=dict)  # Environment info
    stack_trace: Optional[str] = None   # Stack trace if available
    related_entities: Dict[str, Any] = field(default_factory=dict)  # Related entities
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class ErrorReport:
    """Comprehensive error report."""
    error_id: str                        # Unique ID for this error
    error_type: ErrorType                # Type of error 
    error_severity: ErrorSeverity        # Severity of error
    message: str                         # Error message
    context: ErrorContext                # Context information
    exception: Optional[Exception] = None  # Original exception if available
    retry_count: int = 0                 # Number of retry attempts
    retry_successful: Optional[bool] = None  # Whether retry was successful
    aggregated_count: int = 1            # Count of aggregated similar errors
    related_errors: List[str] = field(default_factory=list)  # Related error IDs
    resolution_status: str = "open"      # Status: open, retrying, resolved, failed
    resolution_time: Optional[datetime] = None  # When error was resolved
    resolution_strategy: Optional[str] = None  # Strategy used to resolve
    recovery_details: Dict[str, Any] = field(default_factory=dict)  # Recovery details


class RetryPolicy:
    """Policy for automatic retry of failed operations."""
    
    def __init__(
            self,
            max_retries: int = 3,
            initial_delay_ms: int = 500,
            max_delay_ms: int = 30000,
            backoff_factor: float = 2.0,
            jitter: bool = True,
            retry_on_error_types: Optional[List[ErrorType]] = None,
            skip_on_error_types: Optional[List[ErrorType]] = None,
            skip_if_severity_above: Optional[ErrorSeverity] = None,
            custom_retry_condition: Optional[Callable[[ErrorReport], bool]] = None
        ):
        """
        Initialize a retry policy.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay_ms: Initial delay between retries in milliseconds
            max_delay_ms: Maximum delay between retries in milliseconds
            backoff_factor: Exponential backoff factor
            jitter: Whether to add jitter to delays to prevent thundering herd
            retry_on_error_types: List of error types to retry (None means all types)
            skip_on_error_types: List of error types to skip retrying
            skip_if_severity_above: Skip retry if severity is above this level
            custom_retry_condition: Custom function to determine if retry should be attempted
        """
        self.max_retries = max_retries
        self.initial_delay_ms = initial_delay_ms
        self.max_delay_ms = max_delay_ms
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retry_on_error_types = retry_on_error_types
        self.skip_on_error_types = skip_on_error_types
        self.skip_if_severity_above = skip_if_severity_above
        self.custom_retry_condition = custom_retry_condition
    
    def should_retry(self, error_report: ErrorReport) -> bool:
        """
        Determine if the operation should be retried.
        
        Args:
            error_report: Error report for the failed operation
            
        Returns:
            True if the operation should be retried, False otherwise
        """
        # Check retry count
        if error_report.retry_count >= self.max_retries:
            return False
        
        # Check error type filters
        if self.retry_on_error_types is not None:
            # Handle both Enum and string value comparisons
            if isinstance(error_report.error_type, str):
                if error_report.error_type not in [et.value if hasattr(et, 'value') else et for et in self.retry_on_error_types]:
                    return False
            else:
                if error_report.error_type not in self.retry_on_error_types:
                    return False
        
        if self.skip_on_error_types is not None:
            # Handle both Enum and string value comparisons
            if isinstance(error_report.error_type, str):
                if error_report.error_type in [et.value if hasattr(et, 'value') else et for et in self.skip_on_error_types]:
                    return False
            else:
                if error_report.error_type in self.skip_on_error_types:
                    return False
        
        # Check severity filter
        if self.skip_if_severity_above is not None:
            if isinstance(error_report.error_severity, str):
                # Convert string to enum value for comparison if needed
                severity_value = error_report.error_severity
                skip_value = self.skip_if_severity_above.value if hasattr(self.skip_if_severity_above, 'value') else self.skip_if_severity_above
                if severity_value > skip_value:
                    return False
            else:
                # Compare enum values
                if getattr(error_report.error_severity, 'value', error_report.error_severity) > getattr(self.skip_if_severity_above, 'value', self.skip_if_severity_above):
                    return False
        
        # Check custom condition
        if self.custom_retry_condition is not None:
            return self.custom_retry_condition(error_report)
        
        # Default to retry
        return True
    
    def get_delay_ms(self, retry_count: int) -> int:
        """
        Get the delay before the next retry attempt.
        
        Args:
            retry_count: Current retry count (0-based)
            
        Returns:
            Delay in milliseconds
        """
        import random
        
        # Calculate base delay with exponential backoff
        delay = self.initial_delay_ms * (self.backoff_factor ** retry_count)
        
        # Cap at max delay
        delay = min(delay, self.max_delay_ms)
        
        # Add jitter if enabled
        if self.jitter:
            # Add up to 25% jitter
            jitter_factor = random.uniform(0.75, 1.25)
            delay *= jitter_factor
        
        return int(delay)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert retry policy to a dictionary."""
        return {
            "max_retries": self.max_retries,
            "initial_delay_ms": self.initial_delay_ms,
            "max_delay_ms": self.max_delay_ms,
            "backoff_factor": self.backoff_factor,
            "jitter": self.jitter,
            "retry_on_error_types": [et.value for et in self.retry_on_error_types] if self.retry_on_error_types else None,
            "skip_on_error_types": [et.value for et in self.skip_on_error_types] if self.skip_on_error_types else None,
            "skip_if_severity_above": self.skip_if_severity_above.value if self.skip_if_severity_above else None
        }


class ErrorAggregator:
    """Aggregates related errors to simplify troubleshooting."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize the error aggregator.
        
        Args:
            similarity_threshold: Threshold for considering errors similar (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.error_groups: Dict[str, List[ErrorReport]] = {}
        self.error_fingerprints: Dict[str, str] = {}  # Maps error ID to group ID
    
    def get_error_fingerprint(self, error_report: ErrorReport) -> str:
        """
        Generate a fingerprint for an error to use for similarity matching.
        
        Args:
            error_report: The error report
            
        Returns:
            Error fingerprint string
        """
        # Use key attributes to create a fingerprint
        fingerprint_parts = [
            error_report.error_type.value,
            error_report.context.component,
            error_report.context.operation
        ]
        
        # Add exception type if available
        if error_report.exception:
            fingerprint_parts.append(type(error_report.exception).__name__)
        
        # Extract the most relevant part of the message (first line or first 100 chars)
        message = error_report.message.split('\n')[0][:100]
        fingerprint_parts.append(message)
        
        return "::".join(fingerprint_parts)
    
    def calculate_similarity(self, fingerprint1: str, fingerprint2: str) -> float:
        """
        Calculate similarity between two error fingerprints.
        
        Args:
            fingerprint1: First fingerprint
            fingerprint2: Second fingerprint
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Simple implementation using Jaccard similarity of parts
        parts1 = set(fingerprint1.split('::'))
        parts2 = set(fingerprint2.split('::'))
        
        # Calculate Jaccard similarity
        intersection = len(parts1.intersection(parts2))
        union = len(parts1.union(parts2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def add_error(self, error_report: ErrorReport) -> Tuple[str, bool]:
        """
        Add an error report to the aggregator.
        
        Args:
            error_report: The error report to add
            
        Returns:
            Tuple of (group_id, is_new_group)
        """
        # Generate fingerprint
        fingerprint = self.get_error_fingerprint(error_report)
        
        # Check for similar existing groups
        best_match = None
        best_similarity = 0.0
        
        for group_id, group in self.error_groups.items():
            # Use the first error in the group as the reference
            reference = group[0]
            reference_fingerprint = self.get_error_fingerprint(reference)
            
            # Calculate similarity
            similarity = self.calculate_similarity(fingerprint, reference_fingerprint)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_match = group_id
                best_similarity = similarity
        
        if best_match:
            # Add to existing group
            group = self.error_groups[best_match]
            group.append(error_report)
            
            # Update aggregation count on the first error in the group
            group[0].aggregated_count = len(group)
            
            # Add related error ID to the first error
            group[0].related_errors.append(error_report.error_id)
            
            # Record the group for this error
            self.error_fingerprints[error_report.error_id] = best_match
            
            return best_match, False
        else:
            # Create new group with this error's ID as the group ID
            group_id = error_report.error_id
            self.error_groups[group_id] = [error_report]
            self.error_fingerprints[error_report.error_id] = group_id
            
            return group_id, True
    
    def get_group(self, error_id: str) -> Optional[List[ErrorReport]]:
        """
        Get the group containing a specific error.
        
        Args:
            error_id: The error ID to look up
            
        Returns:
            List of error reports in the group, or None if not found
        """
        if error_id not in self.error_fingerprints:
            return None
        
        group_id = self.error_fingerprints[error_id]
        return self.error_groups.get(group_id)
    
    def get_primary_error(self, error_id: str) -> Optional[ErrorReport]:
        """
        Get the primary error for a group.
        
        Args:
            error_id: Any error ID in the group
            
        Returns:
            The primary error report for the group, or None if not found
        """
        group = self.get_group(error_id)
        if not group:
            return None
        
        return group[0]  # First error is the primary
    
    def get_all_groups(self) -> Dict[str, List[ErrorReport]]:
        """Get all error groups."""
        return self.error_groups
    
    def get_group_summaries(self) -> List[Dict[str, Any]]:
        """
        Get summaries of all error groups.
        
        Returns:
            List of dictionaries with group summaries
        """
        summaries = []
        
        for group_id, group in self.error_groups.items():
            primary = group[0]
            
            summary = {
                "group_id": group_id,
                "error_type": primary.error_type.value,
                "error_severity": primary.error_severity.value,
                "component": primary.context.component,
                "operation": primary.context.operation,
                "message": primary.message,
                "count": len(group),
                "first_seen": min(e.context.timestamp for e in group),
                "last_seen": max(e.context.timestamp for e in group),
                "resolution_status": primary.resolution_status
            }
            
            summaries.append(summary)
        
        return summaries


class DistributedErrorHandler:
    """
    Error handling system for distributed testing framework.
    
    This class is responsible for:
    - Categorizing and logging errors
    - Implementing retry policies
    - Aggregating related errors
    - Coordinating error recovery strategies
    - Providing error reporting and metrics
    """
    
    def __init__(self):
        """Initialize the distributed error handler."""
        # Error storage
        self.errors: Dict[str, ErrorReport] = {}
        
        # Configuration
        self.default_retry_policy = RetryPolicy()
        self.component_retry_policies: Dict[str, RetryPolicy] = {}
        
        # Error aggregation
        self.error_aggregator = ErrorAggregator()
        
        # Error hooks
        self.error_hooks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Error metrics
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.retry_success_counts: Dict[str, int] = defaultdict(int)
        self.retry_failure_counts: Dict[str, int] = defaultdict(int)
        
        # Database connection for persistent storage
        self.db_connection = None
        
        logger.info("Distributed Error Handler initialized")
    
    def set_db_connection(self, db_connection: Any) -> None:
        """Set the database connection for error persistence."""
        self.db_connection = db_connection
        
        # Create error tracking tables if needed
        self._create_schema()
    
    def _create_schema(self) -> None:
        """Create database schema for error tracking."""
        if not self.db_connection:
            return
        
        try:
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS error_reports (
                error_id VARCHAR PRIMARY KEY,
                error_type VARCHAR,
                error_severity VARCHAR,
                message TEXT,
                component VARCHAR,
                operation VARCHAR,
                timestamp TIMESTAMP,
                retry_count INTEGER,
                retry_successful BOOLEAN,
                aggregated_count INTEGER,
                resolution_status VARCHAR,
                resolution_time TIMESTAMP,
                resolution_strategy VARCHAR,
                context_data JSON,
                recovery_details JSON
            )
            """)
            
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS error_relations (
                error_id VARCHAR,
                related_error_id VARCHAR,
                relation_type VARCHAR,
                UNIQUE(error_id, related_error_id)
            )
            """)
            
            logger.debug("Error tracking database schema created")
        except Exception as e:
            logger.error(f"Failed to create error tracking schema: {str(e)}")
    
    def _persist_error(self, error_report: ErrorReport) -> None:
        """Persist error to database."""
        if not self.db_connection:
            return
        
        try:
            # Convert context to JSON
            context_data = {
                "component": error_report.context.component,
                "operation": error_report.context.operation,
                "timestamp": error_report.context.timestamp.isoformat(),
                "user_id": error_report.context.user_id,
                "request_id": error_report.context.request_id,
                "environment": error_report.context.environment,
                "stack_trace": error_report.context.stack_trace,
                "related_entities": error_report.context.related_entities,
                "metadata": error_report.context.metadata
            }
            
            # Insert error report
            self.db_connection.execute("""
            INSERT INTO error_reports (
                error_id, error_type, error_severity, message, component, operation, 
                timestamp, retry_count, retry_successful, aggregated_count, 
                resolution_status, resolution_time, resolution_strategy, 
                context_data, recovery_details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(error_id) DO UPDATE SET
                retry_count = excluded.retry_count,
                retry_successful = excluded.retry_successful,
                aggregated_count = excluded.aggregated_count,
                resolution_status = excluded.resolution_status,
                resolution_time = excluded.resolution_time,
                resolution_strategy = excluded.resolution_strategy,
                recovery_details = excluded.recovery_details
            """, (
                error_report.error_id,
                error_report.error_type.value,
                error_report.error_severity.value,
                error_report.message,
                error_report.context.component,
                error_report.context.operation,
                error_report.context.timestamp,
                error_report.retry_count,
                error_report.retry_successful,
                error_report.aggregated_count,
                error_report.resolution_status,
                error_report.resolution_time,
                error_report.resolution_strategy,
                json.dumps(context_data),
                json.dumps(error_report.recovery_details)
            ))
            
            # Insert related errors
            for related_id in error_report.related_errors:
                self.db_connection.execute("""
                INSERT INTO error_relations (error_id, related_error_id, relation_type)
                VALUES (?, ?, 'aggregated')
                ON CONFLICT(error_id, related_error_id) DO NOTHING
                """, (error_report.error_id, related_id))
            
            logger.debug(f"Persisted error {error_report.error_id} to database")
        except Exception as e:
            logger.error(f"Failed to persist error to database: {str(e)}")
    
    def register_error_hook(self, error_type: Union[str, ErrorType, List[Union[str, ErrorType]]], 
                         hook: Callable[[ErrorReport], None]) -> None:
        """
        Register a hook to be called when an error of the specified type occurs.
        
        Args:
            error_type: Error type(s) to trigger the hook
            hook: Function to call when an error occurs
        """
        if isinstance(error_type, list):
            # Register for multiple error types
            for et in error_type:
                et_value = et.value if isinstance(et, ErrorType) else et
                self.error_hooks[et_value].append(hook)
        else:
            # Register for a single error type
            et_value = error_type.value if isinstance(error_type, ErrorType) else error_type
            self.error_hooks[et_value].append(hook)
        
        logger.debug(f"Registered error hook for {error_type}")
    
    def set_retry_policy(self, component: str, retry_policy: RetryPolicy) -> None:
        """
        Set retry policy for a specific component.
        
        Args:
            component: Component name
            retry_policy: Retry policy for the component
        """
        self.component_retry_policies[component] = retry_policy
        logger.debug(f"Set retry policy for component {component}")
    
    def get_retry_policy(self, component: str) -> RetryPolicy:
        """
        Get retry policy for a component.
        
        Args:
            component: Component name
            
        Returns:
            Retry policy for the component, or default policy if not set
        """
        return self.component_retry_policies.get(component, self.default_retry_policy)
    
    def categorize_error(self, exception: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorType, ErrorSeverity]:
        """
        Categorize an error based on the exception and context.
        
        Args:
            exception: The exception to categorize
            context: Additional context information
            
        Returns:
            Tuple of (error_type, error_severity)
        """
        context = context or {}
        error_type = ErrorType.UNKNOWN
        error_severity = ErrorSeverity.MEDIUM
        
        # Extract useful information
        exception_name = type(exception).__name__
        exception_module = type(exception).__module__
        error_message = str(exception)
        component = context.get("component", "unknown")
        operation = context.get("operation", "unknown")
        
        # Categorize based on exception type
        if exception_name in ["ConnectionError", "ConnectionRefusedError", "ConnectionResetError"]:
            error_type = ErrorType.NETWORK
            error_severity = ErrorSeverity.MEDIUM
        
        elif exception_name in ["TimeoutError", "asyncio.TimeoutError"]:
            error_type = ErrorType.TIMEOUT
            error_severity = ErrorSeverity.MEDIUM
        
        elif exception_name in ["MemoryError", "ResourceWarning"]:
            error_type = ErrorType.RESOURCE
            error_severity = ErrorSeverity.HIGH
        
        elif exception_name in ["AssertionError"]:
            error_type = ErrorType.ASSERTION
            error_severity = ErrorSeverity.MEDIUM
        
        elif exception_name in ["ValueError", "TypeError", "KeyError", "IndexError"]:
            error_type = ErrorType.VALIDATION
            error_severity = ErrorSeverity.MEDIUM
        
        elif exception_name in ["PermissionError", "AccessDenied"]:
            error_type = ErrorType.AUTHORIZATION
            error_severity = ErrorSeverity.HIGH
        
        elif exception_name in ["AuthenticationError", "LoginFailure"]:
            error_type = ErrorType.AUTHENTICATION
            error_severity = ErrorSeverity.HIGH
        
        elif "database" in component.lower() or "db" in component.lower():
            error_type = ErrorType.DATABASE
            error_severity = ErrorSeverity.MEDIUM
        
        elif "test" in component.lower():
            if "setup" in operation.lower():
                error_type = ErrorType.TEST_SETUP
            elif "teardown" in operation.lower():
                error_type = ErrorType.TEST_TEARDOWN
            else:
                error_type = ErrorType.TEST_EXECUTION
            error_severity = ErrorSeverity.MEDIUM
        
        elif "coordinator" in component.lower() or "orchestrator" in component.lower():
            if "schedule" in operation.lower():
                error_type = ErrorType.SCHEDULING
            else:
                error_type = ErrorType.COORDINATION
            error_severity = ErrorSeverity.HIGH
        
        # Check message contents for additional clues
        if any(kw in error_message.lower() for kw in ["configuration", "config", "settings"]):
            error_type = ErrorType.CONFIGURATION
            error_severity = ErrorSeverity.MEDIUM
        
        elif any(kw in error_message.lower() for kw in ["dependency", "require", "import", "module"]):
            error_type = ErrorType.DEPENDENCY
            error_severity = ErrorSeverity.MEDIUM
        
        # Adjust severity based on context
        if context.get("critical", False):
            error_severity = ErrorSeverity.CRITICAL
        elif context.get("high_priority", False):
            error_severity = ErrorSeverity.HIGH
        elif context.get("low_priority", False):
            error_severity = ErrorSeverity.LOW
        
        return error_type, error_severity
    
    def create_error_report(self, exception: Exception, context_data: Dict[str, Any] = None) -> ErrorReport:
        """
        Create a detailed error report from an exception.
        
        Args:
            exception: The exception to report
            context_data: Additional context information
            
        Returns:
            Detailed error report
        """
        context_data = context_data or {}
        
        # Generate stack trace
        stack_trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        
        # Categorize the error
        error_type, error_severity = self.categorize_error(exception, context_data)
        
        # Create error context
        context = ErrorContext(
            component=context_data.get("component", "unknown"),
            operation=context_data.get("operation", "unknown"),
            timestamp=datetime.now(),
            user_id=context_data.get("user_id"),
            request_id=context_data.get("request_id"),
            environment=context_data.get("environment", {}),
            stack_trace=stack_trace,
            related_entities=context_data.get("related_entities", {}),
            metadata=context_data.get("metadata", {})
        )
        
        # Create error report
        error_id = f"err_{uuid.uuid4().hex}"
        error_report = ErrorReport(
            error_id=error_id,
            error_type=error_type,
            error_severity=error_severity,
            message=str(exception),
            context=context,
            exception=exception
        )
        
        # Store the error
        self.errors[error_id] = error_report
        
        # Aggregate the error
        group_id, is_new_group = self.error_aggregator.add_error(error_report)
        
        # Update metrics
        error_type_str = error_type.value
        self.error_counts[error_type_str] += 1
        
        # Persist to database if available
        self._persist_error(error_report)
        
        # Call error hooks
        for hook in self.error_hooks.get(error_type_str, []):
            try:
                hook(error_report)
            except Exception as e:
                logger.error(f"Error in error hook: {str(e)}")
        
        # Log the error
        if error_severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.error(f"SEVERE ERROR [{error_id}]: {error_type.value} in {context.component}/{context.operation}: {str(exception)}")
        else:
            logger.warning(f"Error [{error_id}]: {error_type.value} in {context.component}/{context.operation}: {str(exception)}")
        
        return error_report
    
    async def retry_operation(self, operation: Callable, args: Tuple = None, kwargs: Dict[str, Any] = None, 
                          context: Dict[str, Any] = None) -> Tuple[Any, Optional[ErrorReport]]:
        """
        Execute an operation with automatic retry based on retry policy.
        
        Args:
            operation: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            context: Additional context information
            
        Returns:
            Tuple of (result, error_report)
            If successful, error_report will be None
            If failed after retries, result will be None and error_report will contain details
        """
        args = args or ()
        kwargs = kwargs or {}
        context = context or {}
        
        # Get component and operation names
        component = context.get("component", "unknown")
        operation_name = context.get("operation", operation.__name__)
        
        # Get retry policy for this component
        retry_policy = self.get_retry_policy(component)
        
        # Track retry attempts
        retry_count = 0
        last_error = None
        error_report = None
        
        # Execute with retries
        while True:
            try:
                # Execute the operation
                result = operation(*args, **kwargs)
                
                # Handle coroutines
                if asyncio.iscoroutine(result):
                    result = await result
                
                # If we got here after retries, update the error report
                if error_report:
                    error_report.retry_successful = True
                    error_report.resolution_status = "resolved"
                    error_report.resolution_time = datetime.now()
                    error_report.resolution_strategy = "retry"
                    error_report.recovery_details["successful_retry"] = {
                        "attempt": retry_count,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Update metrics
                    self.retry_success_counts[error_report.error_type.value] += 1
                    
                    # Persist updated error report
                    self._persist_error(error_report)
                
                # Return successful result
                return result, None
            
            except Exception as e:
                # Create error report for the first failure
                if not error_report:
                    # Update context with component and operation
                    context_data = {
                        "component": component,
                        "operation": operation_name,
                        **context
                    }
                    
                    # Create error report
                    error_report = self.create_error_report(e, context_data)
                    last_error = e
                else:
                    # Update error report for retry failure
                    error_report.retry_count += 1
                    
                    # Store the latest exception
                    last_error = e
                    
                    # Update error message if different
                    if str(e) != error_report.message:
                        error_report.message = f"{error_report.message} | Retry {retry_count}: {str(e)}"
                
                # Check if we should retry
                retry_count += 1
                if not retry_policy.should_retry(error_report) or retry_count > retry_policy.max_retries:
                    # No more retries, mark as failed
                    error_report.retry_successful = False
                    error_report.resolution_status = "failed"
                    error_report.recovery_details["failed_retries"] = {
                        "attempts": retry_count,
                        "last_error": str(last_error)
                    }
                    
                    # Update metrics
                    self.retry_failure_counts[error_report.error_type.value] += 1
                    
                    # Persist final error state
                    self._persist_error(error_report)
                    
                    # Log failure
                    logger.error(f"Operation {operation_name} failed after {retry_count} retries: {str(last_error)}")
                    
                    # Return failure
                    return None, error_report
                
                # Delay before retry
                delay_ms = retry_policy.get_delay_ms(retry_count - 1)
                delay_sec = delay_ms / 1000.0
                
                # Log retry attempt
                logger.warning(f"Retrying operation {operation_name} after error: {str(e)} (attempt {retry_count}/{retry_policy.max_retries}, delay {delay_ms}ms)")
                
                # Update error report
                error_report.retry_count = retry_count
                error_report.resolution_status = "retrying"
                if "retry_attempts" not in error_report.recovery_details:
                    error_report.recovery_details["retry_attempts"] = []
                
                error_report.recovery_details["retry_attempts"].append({
                    "attempt": retry_count,
                    "timestamp": datetime.now().isoformat(),
                    "delay_ms": delay_ms,
                    "error": str(last_error)
                })
                
                # Persist retry status
                self._persist_error(error_report)
                
                # Wait before retry
                await anyio.sleep(delay_sec)
    
    def retry_operation_sync(self, operation: Callable, args: Tuple = None, kwargs: Dict[str, Any] = None, 
                         context: Dict[str, Any] = None) -> Tuple[Any, Optional[ErrorReport]]:
        """
        Synchronous version of retry_operation.
        
        Args:
            operation: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            context: Additional context information
            
        Returns:
            Tuple of (result, error_report)
        """
        args = args or ()
        kwargs = kwargs or {}
        context = context or {}
        
        # Get component and operation names
        component = context.get("component", "unknown")
        operation_name = context.get("operation", operation.__name__)
        
        # Get retry policy for this component
        retry_policy = self.get_retry_policy(component)
        
        # Track retry attempts
        retry_count = 0
        last_error = None
        error_report = None
        
        # Execute with retries
        while True:
            try:
                # Execute the operation
                result = operation(*args, **kwargs)
                
                # If we got here after retries, update the error report
                if error_report:
                    error_report.retry_successful = True
                    error_report.resolution_status = "resolved"
                    error_report.resolution_time = datetime.now()
                    error_report.resolution_strategy = "retry"
                    error_report.recovery_details["successful_retry"] = {
                        "attempt": retry_count,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Update metrics
                    self.retry_success_counts[error_report.error_type.value] += 1
                    
                    # Persist updated error report
                    self._persist_error(error_report)
                
                # Return successful result
                return result, None
            
            except Exception as e:
                # Create error report for the first failure
                if not error_report:
                    # Update context with component and operation
                    context_data = {
                        "component": component,
                        "operation": operation_name,
                        **context
                    }
                    
                    # Create error report
                    error_report = self.create_error_report(e, context_data)
                    last_error = e
                else:
                    # Update error report for retry failure
                    error_report.retry_count += 1
                    
                    # Store the latest exception
                    last_error = e
                    
                    # Update error message if different
                    if str(e) != error_report.message:
                        error_report.message = f"{error_report.message} | Retry {retry_count}: {str(e)}"
                
                # Check if we should retry
                retry_count += 1
                if not retry_policy.should_retry(error_report) or retry_count > retry_policy.max_retries:
                    # No more retries, mark as failed
                    error_report.retry_successful = False
                    error_report.resolution_status = "failed"
                    error_report.recovery_details["failed_retries"] = {
                        "attempts": retry_count,
                        "last_error": str(last_error)
                    }
                    
                    # Update metrics
                    self.retry_failure_counts[error_report.error_type.value] += 1
                    
                    # Persist final error state
                    self._persist_error(error_report)
                    
                    # Log failure
                    logger.error(f"Operation {operation_name} failed after {retry_count} retries: {str(last_error)}")
                    
                    # Return failure
                    return None, error_report
                
                # Delay before retry
                delay_ms = retry_policy.get_delay_ms(retry_count - 1)
                delay_sec = delay_ms / 1000.0
                
                # Log retry attempt
                logger.warning(f"Retrying operation {operation_name} after error: {str(e)} (attempt {retry_count}/{retry_policy.max_retries}, delay {delay_ms}ms)")
                
                # Update error report
                error_report.retry_count = retry_count
                error_report.resolution_status = "retrying"
                if "retry_attempts" not in error_report.recovery_details:
                    error_report.recovery_details["retry_attempts"] = []
                
                error_report.recovery_details["retry_attempts"].append({
                    "attempt": retry_count,
                    "timestamp": datetime.now().isoformat(),
                    "delay_ms": delay_ms,
                    "error": str(last_error)
                })
                
                # Persist retry status
                self._persist_error(error_report)
                
                # Wait before retry
                time.sleep(delay_sec)
    
    def handle_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorReport:
        """
        Handle an error without retry.
        
        Args:
            exception: The exception to handle
            context: Additional context information
            
        Returns:
            Error report for the handled error
        """
        # Create error report
        error_report = self.create_error_report(exception, context)
        
        # Log appropriate message based on severity
        if error_report.error_severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error in {error_report.context.component}/{error_report.context.operation}: {error_report.message}")
        elif error_report.error_severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error in {error_report.context.component}/{error_report.context.operation}: {error_report.message}")
        else:
            logger.warning(f"Error in {error_report.context.component}/{error_report.context.operation}: {error_report.message}")
        
        return error_report
    
    def resolve_error(self, error_id: str, resolution: str, details: Dict[str, Any] = None) -> bool:
        """
        Mark an error as resolved.
        
        Args:
            error_id: ID of the error to resolve
            resolution: Resolution description
            details: Additional resolution details
            
        Returns:
            True if error was found and resolved, False otherwise
        """
        if error_id not in self.errors:
            logger.warning(f"Attempted to resolve unknown error {error_id}")
            return False
        
        # Get the error report
        error_report = self.errors[error_id]
        
        # Update error report
        error_report.resolution_status = "resolved"
        error_report.resolution_time = datetime.now()
        error_report.resolution_strategy = resolution
        error_report.recovery_details["manual_resolution"] = {
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        # Persist updated error report
        self._persist_error(error_report)
        
        logger.info(f"Error {error_id} marked as resolved: {resolution}")
        
        return True
    
    def get_error_report(self, error_id: str) -> Optional[ErrorReport]:
        """
        Get an error report by ID.
        
        Args:
            error_id: ID of the error
            
        Returns:
            Error report if found, None otherwise
        """
        return self.errors.get(error_id)
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """
        Get error metrics.
        
        Returns:
            Dictionary with error metrics
        """
        # Calculate success rates by error type
        retry_success_rates = {}
        for error_type in set(self.retry_success_counts.keys()).union(self.retry_failure_counts.keys()):
            successes = self.retry_success_counts.get(error_type, 0)
            failures = self.retry_failure_counts.get(error_type, 0)
            total = successes + failures
            rate = successes / total if total > 0 else 0.0
            retry_success_rates[error_type] = rate
        
        # Calculate aggregation metrics
        error_groups = self.error_aggregator.get_all_groups()
        aggregation_metrics = {
            "total_groups": len(error_groups),
            "total_errors": sum(len(group) for group in error_groups.values()),
            "largest_group_size": max((len(group) for group in error_groups.values()), default=0),
            "avg_group_size": sum(len(group) for group in error_groups.values()) / len(error_groups) if error_groups else 0
        }
        
        return {
            "error_counts": dict(self.error_counts),
            "retry_success_counts": dict(self.retry_success_counts),
            "retry_failure_counts": dict(self.retry_failure_counts),
            "retry_success_rates": retry_success_rates,
            "aggregation": aggregation_metrics,
            "total_errors": len(self.errors),
            "unresolved_errors": sum(1 for e in self.errors.values() if e.resolution_status != "resolved")
        }
    
    def get_unresolved_errors(self, component: Optional[str] = None, severity: Optional[ErrorSeverity] = None) -> List[ErrorReport]:
        """
        Get unresolved errors, optionally filtered by component or severity.
        
        Args:
            component: Optional component to filter by
            severity: Optional minimum severity to filter by
            
        Returns:
            List of unresolved error reports
        """
        unresolved = []
        
        for error_report in self.errors.values():
            if error_report.resolution_status != "resolved":
                # Apply component filter if specified
                if component and error_report.context.component != component:
                    continue
                
                # Apply severity filter if specified
                if severity and error_report.error_severity.value < severity.value:
                    continue
                
                unresolved.append(error_report)
        
        return unresolved
    
    def get_error_history(self, component: Optional[str] = None, 
                       period_hours: Optional[int] = None) -> Dict[str, List[ErrorReport]]:
        """
        Get error history for analysis, optionally filtered by component and time period.
        
        Args:
            component: Optional component to filter by
            period_hours: Optional time period to limit history (in hours)
            
        Returns:
            Dictionary mapping error types to lists of error reports
        """
        history = defaultdict(list)
        
        # Calculate cutoff time if period specified
        cutoff = None
        if period_hours:
            cutoff = datetime.now() - timedelta(hours=period_hours)
        
        for error_report in self.errors.values():
            # Apply component filter if specified
            if component and error_report.context.component != component:
                continue
            
            # Apply time filter if specified
            if cutoff and error_report.context.timestamp < cutoff:
                continue
            
            # Group by error type
            history[error_report.error_type.value].append(error_report)
        
        return dict(history)
    
    def integrate_with_execution_orchestrator(self, execution_orchestrator):
        """
        Integrate with the execution orchestrator to handle test execution errors.
        
        Args:
            execution_orchestrator: The execution orchestrator instance
        """
        # Define error handling hook for test execution
        def handle_test_execution_error(test_id, error, context=None):
            context = context or {}
            context.update({
                "component": "test_execution",
                "operation": f"execute_test:{test_id}",
                "related_entities": {"test_id": test_id}
            })
            
            # Create error report
            error_report = self.create_error_report(error, context)
            
            # Log error
            logger.error(f"Test execution error for test {test_id}: {str(error)}")
            
            return error_report
        
        # Set up hooks in the execution orchestrator
        def test_end_hook(test_id, status, result, error):
            if error:
                # Only handle errors
                handle_test_execution_error(test_id, Exception(error), {
                    "test_status": status.value if hasattr(status, "value") else status
                })
        
        # Register the hook
        execution_orchestrator.set_test_end_hook(test_end_hook)
        
        logger.info("Integrated error handler with execution orchestrator")
    
    def integrate_with_dependency_manager(self, dependency_manager):
        """
        Integrate with the dependency manager to handle dependency resolution errors.
        
        Args:
            dependency_manager: The dependency manager instance
        """
        # Store original validate_dependencies method
        original_validate = dependency_manager.validate_dependencies
        
        # Replace with error-wrapped version
        def error_wrapped_validate():
            try:
                return original_validate()
            except Exception as e:
                # Handle the error
                context = {
                    "component": "dependency_manager",
                    "operation": "validate_dependencies"
                }
                error_report = self.create_error_report(e, context)
                logger.error(f"Dependency validation error: {str(e)}")
                
                # Re-raise with error info
                raise type(e)(f"{str(e)} (Error ID: {error_report.error_id})")
        
        # Replace the method
        dependency_manager.validate_dependencies = error_wrapped_validate
        
        logger.info("Integrated error handler with dependency manager")
    
    def get_error_report_summary(self, error_id: str) -> Dict[str, Any]:
        """
        Get a summary of an error report suitable for display.
        
        Args:
            error_id: ID of the error
            
        Returns:
            Dictionary with error summary
        """
        error_report = self.get_error_report(error_id)
        if not error_report:
            return {"error": f"Error {error_id} not found"}
        
        # Get the error group
        group = self.error_aggregator.get_group(error_id) or []
        
        summary = {
            "error_id": error_report.error_id,
            "error_type": error_report.error_type.value,
            "error_severity": error_report.error_severity.value,
            "message": error_report.message,
            "component": error_report.context.component,
            "operation": error_report.context.operation,
            "timestamp": error_report.context.timestamp.isoformat(),
            "resolution_status": error_report.resolution_status,
            "retries": error_report.retry_count,
            "retry_successful": error_report.retry_successful,
            "group_size": len(group),
            "stack_trace": error_report.context.stack_trace,
            "related_entities": error_report.context.related_entities,
            "metadata": error_report.context.metadata
        }
        
        return summary
    
    def clear_resolved_errors(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear resolved errors from memory (not from database).
        
        Args:
            older_than_hours: Only clear errors older than this many hours
            
        Returns:
            Number of errors cleared
        """
        to_clear = []
        
        # Calculate cutoff time if specified
        cutoff = None
        if older_than_hours:
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
        
        for error_id, error_report in self.errors.items():
            if error_report.resolution_status == "resolved":
                if cutoff and error_report.resolution_time and error_report.resolution_time < cutoff:
                    to_clear.append(error_id)
                elif not cutoff:
                    to_clear.append(error_id)
        
        # Remove errors
        for error_id in to_clear:
            del self.errors[error_id]
        
        logger.info(f"Cleared {len(to_clear)} resolved errors from memory")
        
        return len(to_clear)


# Helper functions for common error handling patterns

def safe_execute(func, *args, error_handler=None, **kwargs):
    """
    Execute a function safely, handling any exceptions.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        error_handler: Optional error handler instance
        **kwargs: Keyword arguments
        
    Returns:
        Function result or None if an exception occurred
    """
    # Extract context parameters that should not be passed to the function
    component = kwargs.pop("component", "unknown") if "component" in kwargs else "unknown"
    operation = kwargs.pop("operation", func.__name__) if "operation" in kwargs else func.__name__
    context_arg = kwargs.pop("context", {}) if "context" in kwargs else {}
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_handler:
            context = {
                "component": component,
                "operation": operation,
                **context_arg
            }
            error_handler.handle_error(e, context)
        else:
            logger.exception(f"Error in {func.__name__}: {str(e)}")
        return None


async def safe_execute_async(func, *args, error_handler=None, **kwargs):
    """
    Execute an async function safely, handling any exceptions.
    
    Args:
        func: Async function to execute
        *args: Positional arguments
        error_handler: Optional error handler instance
        **kwargs: Keyword arguments
        
    Returns:
        Function result or None if an exception occurred
    """
    # Extract context parameters that should not be passed to the function
    component = kwargs.pop("component", "unknown") if "component" in kwargs else "unknown"
    operation = kwargs.pop("operation", func.__name__) if "operation" in kwargs else func.__name__
    context_arg = kwargs.pop("context", {}) if "context" in kwargs else {}
    
    try:
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    except Exception as e:
        if error_handler:
            context = {
                "component": component,
                "operation": operation,
                **context_arg
            }
            error_handler.handle_error(e, context)
        else:
            logger.exception(f"Error in async {func.__name__}: {str(e)}")
        return None


# Example usage
if __name__ == "__main__":
    # Create error handler
    error_handler = DistributedErrorHandler()
    
    # Define a function that might fail
    def risky_operation(value):
        if value < 0:
            raise ValueError("Value cannot be negative")
        if value > 100:
            raise OverflowError("Value too large")
        return value * 2
    
    # Try with different values
    for test_value in [-5, 50, 200]:
        try:
            result = risky_operation(test_value)
            print(f"Result for {test_value}: {result}")
        except Exception as e:
            # Handle the error
            error_report = error_handler.handle_error(
                e, 
                {"component": "example", "operation": "risky_operation", "input_value": test_value}
            )
            print(f"Error handled: {error_report.error_id} - {error_report.error_type.value} - {error_report.message}")
    
    # Try with retry
    async def test_retry():
        print("\nTesting retry functionality:")
        
        # Set a custom retry policy for the example component
        error_handler.set_retry_policy("example", RetryPolicy(
            max_retries=3,
            initial_delay_ms=100,
            backoff_factor=1.5
        ))
        
        # Function that fails the first two times
        attempt = 0
        def flaky_operation():
            nonlocal attempt
            attempt += 1
            if attempt <= 2:
                raise ConnectionError(f"Connection failed (attempt {attempt})")
            return "Success on attempt " + str(attempt)
        
        # Retry the flaky operation
        result, error = await error_handler.retry_operation(
            flaky_operation,
            context={"component": "example", "operation": "flaky_operation"}
        )
        
        if result:
            print(f"Operation succeeded after retries: {result}")
        else:
            print(f"Operation failed after retries: {error.message}")
    
    # Run the async test
    anyio.run(test_retry())
    
    # Print error metrics
    print("\nError Metrics:")
    metrics = error_handler.get_error_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")