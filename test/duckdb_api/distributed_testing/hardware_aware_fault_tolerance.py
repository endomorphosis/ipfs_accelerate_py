#!/usr/bin/env python3
"""
Hardware-Aware Fault Tolerance System for Distributed Testing Framework

This module implements a comprehensive fault tolerance system that is aware of different
hardware types and their specific failure modes. It provides specialized recovery strategies
for different hardware platforms (CPUs, GPUs, TPUs, browsers with WebGPU/WebNN),
state persistence mechanisms, failure pattern detection, and task checkpointing.

Key features:
1. Hardware-specific recovery strategies for different types of hardware
2. Intelligent retry policies with exponential backoff and jitter
3. Failure pattern detection and prevention
4. Task state persistence and recovery
5. Checkpoint and resume for long-running tasks
6. Integration with heterogeneous hardware scheduler

This component was implemented ahead of schedule (March 13, 2025 vs. planned June 12-19, 2025).
Additional enhancements implemented ahead of schedule (March 13, 2025):
1. Machine Learning-Based Pattern Detection - originally planned for June 2025
2. Comprehensive Visualization System - precursor to the monitoring dashboard planned for June 19-26, 2025
"""

import os
import sys
import json
import time
import logging
import threading
import uuid
import random
import traceback
import importlib
import matplotlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from enum import Enum, auto
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_fault_tolerance")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import hardware taxonomy and heterogeneous scheduler
from duckdb_api.distributed_testing.hardware_taxonomy import (
    HardwareClass, HardwareArchitecture, HardwareVendor, 
    SoftwareBackend, PrecisionType, HardwareCapabilityProfile
)
from duckdb_api.distributed_testing.heterogeneous_scheduler import (
    HeterogeneousScheduler, WorkloadProfile, TestTask, WorkerState
)


class FailureType(Enum):
    """Types of failures that can occur in distributed testing."""
    HARDWARE_ERROR = auto()             # Hardware failure (GPU crashed, TPU unresponsive)
    SOFTWARE_ERROR = auto()             # Software error (runtime error, exception)
    RESOURCE_EXHAUSTION = auto()        # Out of memory, disk space, etc.
    TIMEOUT = auto()                    # Task took too long to execute
    COMMUNICATION_ERROR = auto()        # Network or communication failure
    BROWSER_FAILURE = auto()            # Browser crash, WebGPU context lost, etc.
    TRANSIENT_ERROR = auto()            # Temporary error expected to resolve on retry
    PERSISTENT_ERROR = auto()           # Error that persists across retries
    WORKER_CRASH = auto()               # Worker process crashed
    WORKER_DISCONNECTION = auto()       # Worker disconnected unexpectedly
    COORDINATOR_ERROR = auto()          # Coordinator-side error
    UNKNOWN = auto()                    # Unknown error type


class RecoveryStrategy(Enum):
    """Strategies for recovering from failures."""
    IMMEDIATE_RETRY = auto()            # Retry immediately on the same worker
    DELAYED_RETRY = auto()              # Retry after a delay on the same worker
    DIFFERENT_WORKER = auto()           # Retry on a different worker with similar capabilities
    DIFFERENT_HARDWARE_CLASS = auto()   # Retry on a worker with different hardware class
    REDUCED_PRECISION = auto()          # Retry with reduced precision (e.g., fp16 instead of fp32)
    REDUCED_BATCH_SIZE = auto()         # Retry with smaller batch size
    SIMPLIFIED_MODEL = auto()           # Retry with a simpler model variant
    FALLBACK_CPU = auto()               # Fall back to CPU execution
    BROWSER_RESTART = auto()            # Restart the browser instance
    SKIP_CHECKPOINTING = auto()         # Continue without checkpointing
    RESET_WORKER_STATE = auto()         # Reset worker state and retry
    ESCALATION = auto()                 # Escalate to human operator


@dataclass
class FailureContext:
    """Context information about a failure."""
    task_id: str
    worker_id: str
    hardware_profile: Optional[HardwareCapabilityProfile] = None
    error_message: str = ""
    error_type: FailureType = FailureType.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.now)
    attempt: int = 1
    stacktrace: str = ""
    task_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RecoveryAction:
    """Action to take to recover from a failure."""
    strategy: RecoveryStrategy
    worker_id: Optional[str] = None  # Target worker for retry (if applicable)
    delay: float = 0.0  # Delay before retry in seconds
    modified_task: Optional[Dict[str, Any]] = None  # Modified task config if applicable
    checkpoint_data: Optional[Dict[str, Any]] = None  # Checkpoint data for resume
    priority_adjustment: int = 0  # Adjust task priority for retry
    message: str = ""  # Informational message about recovery action
    hardware_requirements: Optional[Dict[str, Any]] = None  # Modified hardware requirements


class HardwareAwareFaultToleranceManager:
    """
    Hardware-aware fault tolerance manager for distributed testing framework.
    
    This class provides fault tolerance capabilities with specialized recovery
    strategies for different hardware types, failure pattern detection, and
    intelligent retry policies.
    """
    
    def __init__(self, db_manager=None, scheduler=None, coordinator=None, enable_ml=False):
        """
        Initialize the hardware-aware fault tolerance manager.
        
        Args:
            db_manager: Database manager for state persistence
            scheduler: Reference to the heterogeneous scheduler
            coordinator: Reference to the coordinator
            enable_ml: Enable machine learning-based pattern detection
        """
        self.db_manager = db_manager
        self.scheduler = scheduler
        self.enable_ml = enable_ml
        self.ml_detector = None
        
        # Initialize ML pattern detector if enabled
        if self.enable_ml:
            try:
                # Import dynamically to avoid circular imports
                ml_module = importlib.import_module("duckdb_api.distributed_testing.ml_pattern_detection")
                self.ml_detector = ml_module.MLPatternDetector(db_manager=db_manager)
                logger.info("ML-based pattern detection enabled")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Failed to initialize ML pattern detection: {e}")
                self.enable_ml = False
        self.coordinator = coordinator
        
        # Configuration
        self.config = {
            "max_retries": 3,  # Maximum number of retry attempts
            "base_delay": 2.0,  # Base delay for exponential backoff (seconds)
            "max_delay": 60.0,  # Maximum delay for exponential backoff (seconds)
            "jitter_factor": 0.2,  # Jitter factor for randomizing delay
            "checkpoint_interval": 300,  # Interval between checkpoints (seconds)
            "state_persistence_enabled": True,  # Enable state persistence
            "failure_history_size": 100,  # Number of failures to track for pattern detection
            "failure_pattern_threshold": 3,  # Number of similar failures to consider a pattern
            "failure_pattern_timeframe": 3600,  # Timeframe for pattern detection (seconds)
            "recovery_strategies": {
                # Default recovery strategies for each hardware class
                "CPU": [
                    RecoveryStrategy.DELAYED_RETRY,
                    RecoveryStrategy.DIFFERENT_WORKER,
                    RecoveryStrategy.REDUCED_BATCH_SIZE
                ],
                "GPU": [
                    RecoveryStrategy.DELAYED_RETRY,
                    RecoveryStrategy.DIFFERENT_WORKER,
                    RecoveryStrategy.REDUCED_PRECISION,
                    RecoveryStrategy.REDUCED_BATCH_SIZE,
                    RecoveryStrategy.FALLBACK_CPU
                ],
                "TPU": [
                    RecoveryStrategy.DELAYED_RETRY,
                    RecoveryStrategy.DIFFERENT_WORKER,
                    RecoveryStrategy.REDUCED_BATCH_SIZE,
                    RecoveryStrategy.FALLBACK_CPU
                ],
                "NPU": [
                    RecoveryStrategy.DELAYED_RETRY,
                    RecoveryStrategy.DIFFERENT_WORKER,
                    RecoveryStrategy.REDUCED_PRECISION,
                    RecoveryStrategy.FALLBACK_CPU
                ],
                "WEBGPU": [
                    RecoveryStrategy.BROWSER_RESTART,
                    RecoveryStrategy.DELAYED_RETRY,
                    RecoveryStrategy.DIFFERENT_WORKER,
                    RecoveryStrategy.REDUCED_PRECISION,
                    RecoveryStrategy.FALLBACK_CPU
                ],
                "WEBNN": [
                    RecoveryStrategy.BROWSER_RESTART,
                    RecoveryStrategy.DELAYED_RETRY,
                    RecoveryStrategy.DIFFERENT_WORKER,
                    RecoveryStrategy.FALLBACK_CPU
                ]
            },
            "browser_specific_strategies": {
                "chrome": [
                    RecoveryStrategy.BROWSER_RESTART,
                    RecoveryStrategy.DELAYED_RETRY,
                    RecoveryStrategy.DIFFERENT_WORKER
                ],
                "firefox": [
                    RecoveryStrategy.BROWSER_RESTART,
                    RecoveryStrategy.DELAYED_RETRY,
                    RecoveryStrategy.DIFFERENT_WORKER
                ],
                "safari": [
                    RecoveryStrategy.BROWSER_RESTART,
                    RecoveryStrategy.DELAYED_RETRY,
                    RecoveryStrategy.DIFFERENT_WORKER
                ],
                "edge": [
                    RecoveryStrategy.BROWSER_RESTART,
                    RecoveryStrategy.DELAYED_RETRY,
                    RecoveryStrategy.DIFFERENT_WORKER
                ]
            },
            "error_specific_strategies": {
                # Specific strategies for certain error types
                "out_of_memory": [
                    RecoveryStrategy.REDUCED_BATCH_SIZE,
                    RecoveryStrategy.REDUCED_PRECISION,
                    RecoveryStrategy.DIFFERENT_WORKER,
                    RecoveryStrategy.FALLBACK_CPU
                ],
                "cuda_error": [
                    RecoveryStrategy.DELAYED_RETRY,
                    RecoveryStrategy.DIFFERENT_WORKER,
                    RecoveryStrategy.FALLBACK_CPU
                ],
                "browser_crash": [
                    RecoveryStrategy.BROWSER_RESTART,
                    RecoveryStrategy.DIFFERENT_WORKER,
                    RecoveryStrategy.FALLBACK_CPU
                ],
                "webgpu_context_lost": [
                    RecoveryStrategy.BROWSER_RESTART,
                    RecoveryStrategy.DIFFERENT_WORKER,
                    RecoveryStrategy.FALLBACK_CPU
                ]
            }
        }
        
        # Failure tracking
        self.failure_history: List[FailureContext] = []
        self.failure_patterns: Dict[str, Dict[str, Any]] = {}
        self.failure_pattern_lock = threading.Lock()
        
        # Task state management
        self.task_states: Dict[str, Dict[str, Any]] = {}
        self.task_state_lock = threading.Lock()
        
        # Checkpoint management
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.checkpoint_lock = threading.Lock()
        
        # Recovery history
        self.recovery_history: Dict[str, List[RecoveryAction]] = {}
        self.recovery_history_lock = threading.Lock()
        
        # Handler threads
        self.checkpoint_thread = None
        self.checkpoint_stop_event = threading.Event()
        
        logger.info("Hardware-aware fault tolerance manager initialized")
    
    def configure(self, config_updates: Dict[str, Any]):
        """
        Update the configuration of the fault tolerance manager.
        
        Args:
            config_updates: Dictionary with configuration updates
        """
        self.config.update(config_updates)
        logger.info(f"Fault tolerance configuration updated: {config_updates}")
    
    def start(self):
        """Start the fault tolerance manager."""
        # Load persisted state if available
        if self.config["state_persistence_enabled"] and self.db_manager:
            self._load_persisted_state()
        
        # Start checkpoint thread
        self.checkpoint_stop_event.clear()
        self.checkpoint_thread = threading.Thread(
            target=self._checkpoint_loop,
            daemon=True
        )
        self.checkpoint_thread.start()
        
        logger.info("Hardware-aware fault tolerance manager started")
    
    def stop(self):
        """Stop the fault tolerance manager."""
        # Stop checkpoint thread
        if self.checkpoint_thread and self.checkpoint_thread.is_alive():
            self.checkpoint_stop_event.set()
            self.checkpoint_thread.join(timeout=5.0)
            if self.checkpoint_thread.is_alive():
                logger.warning("Checkpoint thread did not stop gracefully")
        
        # Persist state
        if self.config["state_persistence_enabled"] and self.db_manager:
            self._persist_state()
        
        logger.info("Hardware-aware fault tolerance manager stopped")
    
    def handle_failure(self, task_id: str, worker_id: str, 
                      error_info: Dict[str, Any]) -> RecoveryAction:
        """
        Handle a task failure and determine recovery action.
        
        Args:
            task_id: ID of the failed task
            worker_id: ID of the worker where the failure occurred
            error_info: Information about the error
            
        Returns:
            RecoveryAction with the determined recovery strategy
        """
        # Get task and worker information
        task = self._get_task(task_id)
        worker = self._get_worker(worker_id)
        
        if not task:
            logger.warning(f"Cannot handle failure for unknown task {task_id}")
            return RecoveryAction(
                strategy=RecoveryStrategy.ESCALATION,
                message=f"Cannot handle failure for unknown task {task_id}"
            )
        
        # Create failure context
        failure_context = self._create_failure_context(task_id, worker_id, error_info)
        
        # Track failure in history
        self._add_to_failure_history(failure_context)
        
        # Check if max retries exceeded
        retry_count = task.get("retry_count", 0)
        if retry_count >= self.config["max_retries"]:
            logger.warning(f"Task {task_id} exceeded maximum retry count ({self.config['max_retries']})")
            return RecoveryAction(
                strategy=RecoveryStrategy.ESCALATION,
                message=f"Exceeded maximum retry count ({self.config['max_retries']})"
            )
        
        # Check for failure patterns using traditional method
        pattern_action = self._check_failure_patterns(failure_context)
        if pattern_action:
            return pattern_action
            
        # Check for ML-based patterns if enabled
        ml_action = self._check_ml_patterns(failure_context)
        if ml_action:
            return ml_action
        
        # Determine recovery strategy based on hardware type, error type, etc.
        recovery_action = self._determine_recovery_strategy(failure_context)
        
        # Update task state with recovery information
        self._update_task_state(task_id, {
            "last_failure": {
                "timestamp": failure_context.timestamp.isoformat(),
                "error_type": failure_context.error_type.name,
                "error_message": failure_context.error_message,
                "recovery_action": {
                    "strategy": recovery_action.strategy.name,
                    "worker_id": recovery_action.worker_id,
                    "delay": recovery_action.delay
                }
            },
            "retry_count": retry_count + 1
        })
        
        # Record recovery action in history
        self._add_to_recovery_history(task_id, recovery_action)
        
        logger.info(f"Task {task_id} failed on worker {worker_id}, recovery strategy: {recovery_action.strategy.name}")
        return recovery_action
    
    def get_task_state(self, task_id: str) -> Dict[str, Any]:
        """
        Get the current state of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task state dictionary
        """
        with self.task_state_lock:
            return self.task_states.get(task_id, {}).copy()
    
    def update_task_state(self, task_id: str, state_updates: Dict[str, Any]):
        """
        Update the state of a task.
        
        Args:
            task_id: ID of the task
            state_updates: Dictionary with state updates
        """
        with self.task_state_lock:
            if task_id not in self.task_states:
                self.task_states[task_id] = {}
            self.task_states[task_id].update(state_updates)
    
    def create_checkpoint(self, task_id: str, checkpoint_data: Dict[str, Any]) -> str:
        """
        Create a checkpoint for a task.
        
        Args:
            task_id: ID of the task
            checkpoint_data: Checkpoint data
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        with self.checkpoint_lock:
            self.checkpoints[checkpoint_id] = {
                "task_id": task_id,
                "timestamp": timestamp.isoformat(),
                "data": checkpoint_data
            }
        
        # Update task state with checkpoint information
        self._update_task_state(task_id, {
            "last_checkpoint": {
                "checkpoint_id": checkpoint_id,
                "timestamp": timestamp.isoformat()
            }
        })
        
        # Persist checkpoint if enabled
        if self.config["state_persistence_enabled"] and self.db_manager:
            self._persist_checkpoint(checkpoint_id)
        
        logger.debug(f"Created checkpoint {checkpoint_id} for task {task_id}")
        return checkpoint_id
    
    def get_latest_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest checkpoint for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Latest checkpoint data or None if no checkpoint exists
        """
        with self.checkpoint_lock:
            # Find all checkpoints for this task
            task_checkpoints = {
                cp_id: cp
                for cp_id, cp in self.checkpoints.items()
                if cp["task_id"] == task_id
            }
            
            if not task_checkpoints:
                return None
            
            # Find the latest checkpoint
            latest_checkpoint_id = max(
                task_checkpoints.keys(),
                key=lambda cp_id: task_checkpoints[cp_id]["timestamp"]
            )
            
            return self.checkpoints[latest_checkpoint_id]["data"]
    
    def get_recovery_history(self, task_id: str) -> List[RecoveryAction]:
        """
        Get the recovery history for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of recovery actions
        """
        with self.recovery_history_lock:
            return self.recovery_history.get(task_id, []).copy()
    
    def get_failure_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detected failure patterns.
        
        Returns:
            Dictionary of failure patterns
        """
        with self.failure_pattern_lock:
            return self.failure_patterns.copy()
    
    def create_visualization(self, output_dir="./visualizations") -> Optional[str]:
        """
        Create visualizations for this fault tolerance manager.
        
        Args:
            output_dir: Directory where visualization files will be saved
            
        Returns:
            Path to the generated report file, or None if visualization failed
        """
        try:
            # Use the helper function to create visualizations
            return visualize_fault_tolerance(self, output_dir)
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
    
    def _create_failure_context(self, task_id: str, worker_id: str, 
                               error_info: Dict[str, Any]) -> FailureContext:
        """
        Create a failure context from error information.
        
        Args:
            task_id: ID of the failed task
            worker_id: ID of the worker where the failure occurred
            error_info: Information about the error
            
        Returns:
            FailureContext object
        """
        # Get task state
        task_state = self.get_task_state(task_id)
        
        # Get hardware profile
        hardware_profile = None
        worker = self._get_worker(worker_id)
        if worker and "hardware_profile" in worker:
            hardware_profile = worker["hardware_profile"]
        
        # Determine error type
        error_type = self._categorize_error(error_info)
        
        # Get retry count
        retry_count = task_state.get("retry_count", 0) + 1
        
        # Create failure context
        return FailureContext(
            task_id=task_id,
            worker_id=worker_id,
            hardware_profile=hardware_profile,
            error_message=error_info.get("message", ""),
            error_type=error_type,
            timestamp=datetime.now(),
            attempt=retry_count,
            stacktrace=error_info.get("stacktrace", ""),
            task_state=task_state
        )
    
    def _categorize_error(self, error_info: Dict[str, Any]) -> FailureType:
        """
        Categorize an error based on error information.
        
        Args:
            error_info: Information about the error
            
        Returns:
            FailureType enum value
        """
        error_message = error_info.get("message", "").lower()
        error_type = error_info.get("type", "").lower()
        
        # Check for known error patterns
        if "out of memory" in error_message or "oom" in error_message:
            return FailureType.RESOURCE_EXHAUSTION
        
        if "cuda error" in error_message or "cuda failure" in error_message:
            return FailureType.HARDWARE_ERROR
        
        if "browser crash" in error_message or "browser disconnected" in error_message:
            return FailureType.BROWSER_FAILURE
        
        if "webgpu context lost" in error_message:
            return FailureType.BROWSER_FAILURE
        
        if "timeout" in error_message or "timed out" in error_message:
            return FailureType.TIMEOUT
            
        if "connection" in error_message:
            return FailureType.COMMUNICATION_ERROR
        
        if "worker crash" in error_message or "worker disconnected" in error_message:
            return FailureType.WORKER_CRASH
        
        # Default to software error
        return FailureType.SOFTWARE_ERROR
    
    def _add_to_failure_history(self, failure_context: FailureContext):
        """
        Add a failure to the history and check for patterns.
        
        Args:
            failure_context: Failure context to add
        """
        with self.failure_pattern_lock:
            # Add to history
            self.failure_history.append(failure_context)
            
            # Limit history size
            if len(self.failure_history) > self.config["failure_history_size"]:
                self.failure_history = self.failure_history[-self.config["failure_history_size"]:]
            
            # Check for patterns
            self._detect_failure_patterns()
    
    def _detect_failure_patterns(self):
        """
        Detect patterns in failure history.
        
        This analyzes recent failures to identify recurring patterns
        that might indicate systemic issues.
        """
        # Group failures by various attributes
        by_worker_hardware = {}
        by_error_type = {}
        by_worker_id = {}
        
        # Only consider recent failures
        cutoff_time = datetime.now() - timedelta(seconds=self.config["failure_pattern_timeframe"])
        recent_failures = [f for f in self.failure_history if f.timestamp > cutoff_time]
        
        for failure in recent_failures:
            # Group by hardware type
            if failure.hardware_profile:
                hw_class = failure.hardware_profile.hardware_class
                hw_key = hw_class.name if hw_class else "UNKNOWN"
                
                if hw_key not in by_worker_hardware:
                    by_worker_hardware[hw_key] = []
                by_worker_hardware[hw_key].append(failure)
            
            # Group by error type
            error_key = failure.error_type.name
            if error_key not in by_error_type:
                by_error_type[error_key] = []
            by_error_type[error_key].append(failure)
            
            # Group by worker
            if failure.worker_id not in by_worker_id:
                by_worker_id[failure.worker_id] = []
            by_worker_id[failure.worker_id].append(failure)
        
        # Check for hardware-specific patterns
        self._check_pattern_group(by_worker_hardware, "hardware_class")
        
        # Check for error-specific patterns
        self._check_pattern_group(by_error_type, "error_type")
        
        # Check for worker-specific patterns
        self._check_pattern_group(by_worker_id, "worker_id")
    
    def _check_pattern_group(self, grouped_failures: Dict[str, List[FailureContext]], 
                            group_type: str):
        """
        Check for patterns in a group of failures.
        
        Args:
            grouped_failures: Dictionary mapping group keys to lists of failures
            group_type: Type of grouping (e.g., "hardware_class", "error_type")
        """
        threshold = self.config["failure_pattern_threshold"]
        
        for key, failures in grouped_failures.items():
            if len(failures) >= threshold:
                # We have a potential pattern
                pattern_id = f"{group_type}_{key}_{int(time.time())}"
                
                # Check if similar pattern already exists
                if self._similar_pattern_exists(group_type, key):
                    continue
                
                # Create new pattern
                self.failure_patterns[pattern_id] = {
                    "type": group_type,
                    "key": key,
                    "count": len(failures),
                    "first_seen": min(f.timestamp for f in failures).isoformat(),
                    "last_seen": max(f.timestamp for f in failures).isoformat(),
                    "task_ids": list(set(f.task_id for f in failures)),
                    "worker_ids": list(set(f.worker_id for f in failures)),
                    "error_types": list(set(f.error_type.name for f in failures)),
                    "recommended_action": self._recommend_action_for_pattern(group_type, key, failures)
                }
                
                logger.warning(
                    f"Detected failure pattern: {group_type}={key}, "
                    f"count={len(failures)}, "
                    f"recommended_action={self.failure_patterns[pattern_id]['recommended_action']}"
                )
    
    def _similar_pattern_exists(self, pattern_type: str, pattern_key: str) -> bool:
        """
        Check if a similar pattern already exists.
        
        Args:
            pattern_type: Type of pattern (e.g., "hardware_class", "error_type")
            pattern_key: Key value for the pattern
            
        Returns:
            True if a similar pattern exists, False otherwise
        """
        for pattern_id, pattern in self.failure_patterns.items():
            if pattern["type"] == pattern_type and pattern["key"] == pattern_key:
                # Update existing pattern instead of creating a new one
                return True
        return False
    
    def _recommend_action_for_pattern(self, pattern_type: str, pattern_key: str, 
                                     failures: List[FailureContext]) -> str:
        """
        Recommend an action for a detected failure pattern.
        
        Args:
            pattern_type: Type of pattern (e.g., "hardware_class", "error_type")
            pattern_key: Key value for the pattern
            failures: List of failures in this pattern
            
        Returns:
            Recommended action as a string
        """
        if pattern_type == "hardware_class":
            # Hardware-specific recommendations
            if pattern_key == "GPU":
                return "Consider using different GPU workers or falling back to CPU"
            elif pattern_key == "TPU":
                return "Consider using different TPU workers or falling back to CPU"
            elif pattern_key == "WEBGPU" or pattern_key == "WEBNN":
                return "Consider restarting browser instances or using different browser types"
            else:
                return "Consider using different hardware class"
                
        elif pattern_type == "error_type":
            # Error-specific recommendations
            if pattern_key == "RESOURCE_EXHAUSTION":
                return "Reduce batch size or model complexity"
            elif pattern_key == "BROWSER_FAILURE":
                return "Restart browser instances or use different browser types"
            elif pattern_key == "HARDWARE_ERROR":
                return "Check hardware status or use different hardware"
            elif pattern_key == "TIMEOUT":
                return "Increase timeout limits or optimize task execution"
            else:
                return "Investigate common error pattern"
                
        elif pattern_type == "worker_id":
            # Worker-specific recommendations
            return f"Take worker {pattern_key} offline for investigation"
            
        # Default recommendation
        return "Investigate pattern and take appropriate action"
    
    def _check_failure_patterns(self, failure_context: FailureContext) -> Optional[RecoveryAction]:
        """
        Check if a failure matches known patterns and determine recovery action.
        
        Args:
            failure_context: Current failure context
            
        Returns:
            RecoveryAction if a pattern match is found, None otherwise
        """
        with self.failure_pattern_lock:
            # Skip if no patterns
            if not self.failure_patterns:
                return None
            
            # Check for hardware class patterns
            if failure_context.hardware_profile:
                hw_class = failure_context.hardware_profile.hardware_class
                hw_key = hw_class.name if hw_class else "UNKNOWN"
                
                for pattern_id, pattern in self.failure_patterns.items():
                    if pattern["type"] == "hardware_class" and pattern["key"] == hw_key:
                        # Found a matching hardware pattern
                        return self._get_pattern_recovery_action(pattern, failure_context)
            
            # Check for error type patterns
            error_key = failure_context.error_type.name
            for pattern_id, pattern in self.failure_patterns.items():
                if pattern["type"] == "error_type" and pattern["key"] == error_key:
                    # Found a matching error pattern
                    return self._get_pattern_recovery_action(pattern, failure_context)
            
            # Check for worker patterns
            for pattern_id, pattern in self.failure_patterns.items():
                if pattern["type"] == "worker_id" and pattern["key"] == failure_context.worker_id:
                    # Found a matching worker pattern
                    return self._get_pattern_recovery_action(pattern, failure_context)
            
            # No pattern match
            return None
    
    def _check_ml_patterns(self, failure_context: FailureContext) -> Optional[RecoveryAction]:
        """
        Check for ML-detected patterns that can inform recovery strategy.
        
        Args:
            failure_context: Context information about the failure
            
        Returns:
            RecoveryAction if an ML pattern was detected, None otherwise
        """
        if not self.enable_ml or not self.ml_detector:
            return None
            
        try:
            # Add failure to ML detector
            self.ml_detector.add_failure(failure_context)
            
            # Get ML-based recovery strategy recommendation
            recommended_strategy = self.ml_detector.recommend_strategy(failure_context)
            if recommended_strategy:
                logger.info(f"ML pattern detection recommended strategy: {recommended_strategy.name}")
                return RecoveryAction(
                    strategy=recommended_strategy,
                    message=f"ML pattern detection recommended {recommended_strategy.name} based on historical data",
                    worker_id=failure_context.worker_id
                )
                
            # Check for newly detected patterns
            ml_patterns = self.ml_detector.detect_patterns()
            if ml_patterns:
                # Find the highest confidence pattern
                best_pattern = max(ml_patterns, key=lambda p: p.get("confidence", 0))
                if best_pattern.get("confidence", 0) >= 0.7:  # Only use high-confidence patterns
                    logger.info(f"ML pattern detected: {best_pattern['description']}")
                    
                    # Try to map the recommended strategy to a RecoveryStrategy enum
                    strategy_name = best_pattern.get("recommended_strategy", "DELAYED_RETRY")
                    try:
                        strategy = RecoveryStrategy[strategy_name]
                    except (KeyError, ValueError):
                        strategy = RecoveryStrategy.DELAYED_RETRY
                    
                    return RecoveryAction(
                        strategy=strategy,
                        message=f"ML pattern detection: {best_pattern['description']}",
                        worker_id=failure_context.worker_id
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Error in ML pattern detection: {e}")
            return None
    
    def _get_pattern_recovery_action(self, pattern: Dict[str, Any], 
                                    failure_context: FailureContext) -> RecoveryAction:
        """
        Get recovery action based on a pattern match.
        
        Args:
            pattern: Matched failure pattern
            failure_context: Current failure context
            
        Returns:
            RecoveryAction based on the pattern
        """
        # Determine recovery strategy based on pattern type
        if pattern["type"] == "hardware_class":
            if pattern["key"] == "GPU":
                return RecoveryAction(
                    strategy=RecoveryStrategy.DIFFERENT_HARDWARE_CLASS,
                    message=f"Switching from GPU to CPU due to pattern: {pattern['recommended_action']}",
                    hardware_requirements={"hardware": ["cpu"]}
                )
            elif pattern["key"] == "TPU":
                return RecoveryAction(
                    strategy=RecoveryStrategy.DIFFERENT_HARDWARE_CLASS,
                    message=f"Switching from TPU to CPU due to pattern: {pattern['recommended_action']}",
                    hardware_requirements={"hardware": ["cpu"]}
                )
            elif pattern["key"] in ["WEBGPU", "WEBNN"]:
                return RecoveryAction(
                    strategy=RecoveryStrategy.BROWSER_RESTART,
                    message=f"Restarting browser due to pattern: {pattern['recommended_action']}"
                )
            else:
                return RecoveryAction(
                    strategy=RecoveryStrategy.DIFFERENT_WORKER,
                    message=f"Switching to different worker due to pattern: {pattern['recommended_action']}"
                )
                
        elif pattern["type"] == "error_type":
            if pattern["key"] == "RESOURCE_EXHAUSTION":
                return RecoveryAction(
                    strategy=RecoveryStrategy.REDUCED_BATCH_SIZE,
                    message=f"Reducing batch size due to pattern: {pattern['recommended_action']}",
                    modified_task=self._create_reduced_batch_task(failure_context.task_id)
                )
            elif pattern["key"] == "BROWSER_FAILURE":
                return RecoveryAction(
                    strategy=RecoveryStrategy.BROWSER_RESTART,
                    message=f"Restarting browser due to pattern: {pattern['recommended_action']}"
                )
            elif pattern["key"] == "HARDWARE_ERROR":
                return RecoveryAction(
                    strategy=RecoveryStrategy.DIFFERENT_WORKER,
                    message=f"Switching to different worker due to pattern: {pattern['recommended_action']}"
                )
            else:
                return RecoveryAction(
                    strategy=RecoveryStrategy.DELAYED_RETRY,
                    message=f"Delayed retry due to pattern: {pattern['recommended_action']}",
                    delay=self._calculate_retry_delay(failure_context.attempt)
                )
                
        elif pattern["type"] == "worker_id":
            return RecoveryAction(
                strategy=RecoveryStrategy.DIFFERENT_WORKER,
                message=f"Switching from problematic worker due to pattern: {pattern['recommended_action']}"
            )
            
        # Default action
        return RecoveryAction(
            strategy=RecoveryStrategy.DIFFERENT_WORKER,
            message=f"Generic recovery due to pattern: {pattern['recommended_action']}"
        )
    
    def _determine_recovery_strategy(self, failure_context: FailureContext) -> RecoveryAction:
        """
        Determine the best recovery strategy for a failure.
        
        Args:
            failure_context: Failure context
            
        Returns:
            RecoveryAction with the determined strategy
        """
        # Get hardware class
        hardware_class = "CPU"  # Default
        if failure_context.hardware_profile and failure_context.hardware_profile.hardware_class:
            hardware_class = failure_context.hardware_profile.hardware_class.name
        
        # Get error message
        error_message = failure_context.error_message.lower()
        
        # Check for specific error conditions
        if "out of memory" in error_message or "oom" in error_message:
            return self._handle_out_of_memory(failure_context)
            
        if "cuda error" in error_message or "cuda failure" in error_message:
            return self._handle_cuda_error(failure_context)
            
        if "browser crash" in error_message or "browser disconnected" in error_message:
            return self._handle_browser_error(failure_context)
            
        if "webgpu context lost" in error_message:
            return self._handle_webgpu_context_lost(failure_context)
        
        # Get strategies for this hardware class
        if hardware_class in self.config["recovery_strategies"]:
            strategies = self.config["recovery_strategies"][hardware_class]
        else:
            # Default to CPU strategies
            strategies = self.config["recovery_strategies"]["CPU"]
        
        # Select strategy based on attempt number
        strategy_index = min(failure_context.attempt - 1, len(strategies) - 1)
        strategy = strategies[strategy_index]
        
        # Create recovery action based on strategy
        if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
            return RecoveryAction(
                strategy=strategy,
                worker_id=failure_context.worker_id,
                message="Retrying immediately on same worker"
            )
            
        elif strategy == RecoveryStrategy.DELAYED_RETRY:
            delay = self._calculate_retry_delay(failure_context.attempt)
            return RecoveryAction(
                strategy=strategy,
                worker_id=failure_context.worker_id,
                delay=delay,
                message=f"Retrying with delay of {delay:.1f}s on same worker"
            )
            
        elif strategy == RecoveryStrategy.DIFFERENT_WORKER:
            return RecoveryAction(
                strategy=strategy,
                message="Retrying on different worker with similar capabilities"
            )
            
        elif strategy == RecoveryStrategy.DIFFERENT_HARDWARE_CLASS:
            # Determine fallback hardware class
            fallback_class = self._determine_fallback_hardware_class(hardware_class)
            requirements = {"hardware": [fallback_class.lower()]}
            
            return RecoveryAction(
                strategy=strategy,
                hardware_requirements=requirements,
                message=f"Retrying on {fallback_class} hardware"
            )
            
        elif strategy == RecoveryStrategy.REDUCED_PRECISION:
            modified_task = self._create_reduced_precision_task(failure_context.task_id)
            return RecoveryAction(
                strategy=strategy,
                modified_task=modified_task,
                message="Retrying with reduced precision"
            )
            
        elif strategy == RecoveryStrategy.REDUCED_BATCH_SIZE:
            modified_task = self._create_reduced_batch_task(failure_context.task_id)
            return RecoveryAction(
                strategy=strategy,
                modified_task=modified_task,
                message="Retrying with reduced batch size"
            )
            
        elif strategy == RecoveryStrategy.FALLBACK_CPU:
            return RecoveryAction(
                strategy=strategy,
                hardware_requirements={"hardware": ["cpu"]},
                message="Falling back to CPU execution"
            )
            
        elif strategy == RecoveryStrategy.BROWSER_RESTART:
            return RecoveryAction(
                strategy=strategy,
                message="Restarting browser instance"
            )
            
        # Default to escalation if no strategy matched
        return RecoveryAction(
            strategy=RecoveryStrategy.ESCALATION,
            message="No suitable recovery strategy found"
        )
    
    def _handle_out_of_memory(self, failure_context: FailureContext) -> RecoveryAction:
        """
        Handle out of memory errors.
        
        Args:
            failure_context: Failure context
            
        Returns:
            RecoveryAction for OOM error
        """
        # Try reduced batch size first
        if failure_context.attempt == 1:
            modified_task = self._create_reduced_batch_task(failure_context.task_id)
            return RecoveryAction(
                strategy=RecoveryStrategy.REDUCED_BATCH_SIZE,
                modified_task=modified_task,
                message="Retrying with reduced batch size due to OOM"
            )
        
        # Then try reduced precision
        elif failure_context.attempt == 2:
            modified_task = self._create_reduced_precision_task(failure_context.task_id)
            return RecoveryAction(
                strategy=RecoveryStrategy.REDUCED_PRECISION,
                modified_task=modified_task,
                message="Retrying with reduced precision due to OOM"
            )
        
        # Finally fall back to CPU
        else:
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_CPU,
                hardware_requirements={"hardware": ["cpu"]},
                message="Falling back to CPU execution due to OOM"
            )
    
    def _handle_cuda_error(self, failure_context: FailureContext) -> RecoveryAction:
        """
        Handle CUDA errors.
        
        Args:
            failure_context: Failure context
            
        Returns:
            RecoveryAction for CUDA error
        """
        # Try a different worker first
        if failure_context.attempt == 1:
            return RecoveryAction(
                strategy=RecoveryStrategy.DIFFERENT_WORKER,
                message="Switching to different worker due to CUDA error"
            )
        
        # Then try delayed retry
        elif failure_context.attempt == 2:
            delay = self._calculate_retry_delay(failure_context.attempt)
            return RecoveryAction(
                strategy=RecoveryStrategy.DELAYED_RETRY,
                delay=delay,
                message=f"Delayed retry ({delay:.1f}s) due to CUDA error"
            )
        
        # Finally fall back to CPU
        else:
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_CPU,
                hardware_requirements={"hardware": ["cpu"]},
                message="Falling back to CPU execution due to CUDA error"
            )
    
    def _handle_browser_error(self, failure_context: FailureContext) -> RecoveryAction:
        """
        Handle browser errors.
        
        Args:
            failure_context: Failure context
            
        Returns:
            RecoveryAction for browser error
        """
        # Try browser restart first
        if failure_context.attempt == 1:
            return RecoveryAction(
                strategy=RecoveryStrategy.BROWSER_RESTART,
                message="Restarting browser due to browser crash"
            )
        
        # Then try a different worker
        elif failure_context.attempt == 2:
            return RecoveryAction(
                strategy=RecoveryStrategy.DIFFERENT_WORKER,
                message="Switching to different worker due to browser crash"
            )
        
        # Finally fall back to CPU
        else:
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_CPU,
                hardware_requirements={"hardware": ["cpu"]},
                message="Falling back to CPU execution due to browser crash"
            )
    
    def _handle_webgpu_context_lost(self, failure_context: FailureContext) -> RecoveryAction:
        """
        Handle WebGPU context lost errors.
        
        Args:
            failure_context: Failure context
            
        Returns:
            RecoveryAction for WebGPU context lost
        """
        # Try browser restart first
        if failure_context.attempt == 1:
            return RecoveryAction(
                strategy=RecoveryStrategy.BROWSER_RESTART,
                message="Restarting browser due to WebGPU context lost"
            )
        
        # Then try a different worker
        elif failure_context.attempt == 2:
            return RecoveryAction(
                strategy=RecoveryStrategy.DIFFERENT_WORKER,
                message="Switching to different worker due to WebGPU context lost"
            )
        
        # Finally fall back to CPU
        else:
            return RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_CPU,
                hardware_requirements={"hardware": ["cpu"]},
                message="Falling back to CPU execution due to WebGPU context lost"
            )
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry using exponential backoff with jitter.
        
        Args:
            attempt: Retry attempt number
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * 2^(attempt-1)
        base_delay = self.config["base_delay"]
        max_delay = self.config["max_delay"]
        jitter_factor = self.config["jitter_factor"]
        
        # Calculate delay with exponential backoff
        delay = base_delay * (2 ** (attempt - 1))
        
        # Apply jitter
        jitter = random.uniform(-jitter_factor * delay, jitter_factor * delay)
        delay += jitter
        
        # Cap at max delay
        return min(delay, max_delay)
    
    def _determine_fallback_hardware_class(self, current_class: str) -> str:
        """
        Determine fallback hardware class for a given hardware class.
        
        Args:
            current_class: Current hardware class
            
        Returns:
            Fallback hardware class
        """
        # Define fallback hierarchy
        fallback_map = {
            "GPU": "CPU",
            "TPU": "CPU",
            "NPU": "CPU",
            "WEBGPU": "CPU",
            "WEBNN": "CPU",
            "DSP": "CPU",
            "FPGA": "CPU"
        }
        
        return fallback_map.get(current_class, "CPU")
    
    def _create_reduced_precision_task(self, task_id: str) -> Dict[str, Any]:
        """
        Create a modified task config with reduced precision.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Modified task configuration
        """
        task = self._get_task(task_id)
        if not task:
            return {}
        
        # Clone the task config
        modified_task = task.copy()
        
        # Check if there's a config field
        if "config" not in modified_task:
            return modified_task
        
        config = modified_task["config"]
        
        # Convert precision if specified
        if "precision" in config:
            current_precision = config["precision"].lower()
            
            # Define precision fallbacks
            if current_precision == "fp32":
                config["precision"] = "fp16"
            elif current_precision == "fp16":
                config["precision"] = "int8"
            elif current_precision == "bf16":
                config["precision"] = "int8"
        else:
            # Add precision if not specified
            config["precision"] = "fp16"
        
        return modified_task
    
    def _create_reduced_batch_task(self, task_id: str) -> Dict[str, Any]:
        """
        Create a modified task config with reduced batch size.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Modified task configuration
        """
        task = self._get_task(task_id)
        if not task:
            return {}
        
        # Clone the task config
        modified_task = task.copy()
        
        # Check if there's a config field
        if "config" not in modified_task:
            return modified_task
        
        config = modified_task["config"]
        
        # Reduce batch size if specified
        if "batch_size" in config:
            current_batch = config["batch_size"]
            # Reduce by half, but minimum of 1
            config["batch_size"] = max(1, current_batch // 2)
        elif "batch_sizes" in config:
            # If multiple batch sizes are specified
            current_batches = config["batch_sizes"]
            if current_batches:
                # Reduce each batch size by half, but minimum of 1
                config["batch_sizes"] = [max(1, batch // 2) for batch in current_batches]
        else:
            # Add a batch size if not specified
            config["batch_size"] = 1
        
        return modified_task
    
    def _add_to_recovery_history(self, task_id: str, recovery_action: RecoveryAction):
        """
        Add a recovery action to the history.
        
        Args:
            task_id: ID of the task
            recovery_action: Recovery action to add
        """
        with self.recovery_history_lock:
            if task_id not in self.recovery_history:
                self.recovery_history[task_id] = []
            
            # Add to history
            self.recovery_history[task_id].append(recovery_action)
            
            # Update ML detector if enabled
            if self.enable_ml and self.ml_detector:
                try:
                    # We don't know the outcome yet, so assume neutral
                    # The outcome will be updated when we get feedback
                    self.ml_detector.update_recovery_result(
                        task_id=task_id,
                        strategy=recovery_action.strategy,
                        success=True  # Assume success for now
                    )
                except Exception as e:
                    logger.warning(f"Error updating ML detector: {e}")
    
    def _update_task_state(self, task_id: str, state_updates: Dict[str, Any]):
        """
        Update task state internally.
        
        Args:
            task_id: ID of the task
            state_updates: Dictionary with state updates
        """
        with self.task_state_lock:
            if task_id not in self.task_states:
                self.task_states[task_id] = {}
            
            # Update state
            self.task_states[task_id].update(state_updates)
    
    def _checkpoint_loop(self):
        """Background thread for periodic checkpointing."""
        while not self.checkpoint_stop_event.is_set():
            try:
                # Create checkpoints for running tasks
                self._create_periodic_checkpoints()
                
                # Persist state
                if self.config["state_persistence_enabled"] and self.db_manager:
                    self._persist_state()
                
            except Exception as e:
                logger.error(f"Error in checkpoint loop: {e}")
                logger.debug(traceback.format_exc())
            
            # Wait for next checkpoint interval
            self.checkpoint_stop_event.wait(self.config["checkpoint_interval"])
    
    def _create_periodic_checkpoints(self):
        """Create checkpoints for all running tasks."""
        # If no coordinator, can't get running tasks
        if not self.coordinator:
            return
        
        # Get running tasks
        running_tasks = self._get_running_tasks()
        
        for task_id in running_tasks:
            # Skip if recent checkpoint exists
            last_checkpoint = self.get_task_state(task_id).get("last_checkpoint", {})
            if last_checkpoint:
                # Check if checkpoint is recent enough
                checkpoint_time = datetime.fromisoformat(last_checkpoint.get("timestamp", "2000-01-01T00:00:00"))
                age = (datetime.now() - checkpoint_time).total_seconds()
                
                if age < self.config["checkpoint_interval"]:
                    # Recent enough, skip
                    continue
            
            try:
                # Try to create a checkpoint for this task
                task_data = self._get_task(task_id)
                if task_data:
                    # Create a checkpoint with task state
                    self.create_checkpoint(task_id, {
                        "task_data": task_data,
                        "state": self.get_task_state(task_id)
                    })
            except Exception as e:
                logger.warning(f"Error creating checkpoint for task {task_id}: {e}")
    
    def _persist_state(self):
        """Persist state to database."""
        if not self.db_manager:
            return
        
        try:
            # Create state snapshot
            state = {
                "timestamp": datetime.now().isoformat(),
                "task_states": self.task_states,
                "failure_patterns": self.failure_patterns,
                "recovery_history": {
                    task_id: [self._recovery_action_to_dict(action) for action in actions]
                    for task_id, actions in self.recovery_history.items()
                }
            }
            
            # Add ML detector state if enabled
            if self.enable_ml and self.ml_detector:
                try:
                    state["ml_detector_state"] = self.ml_detector.save_state()
                except Exception as e:
                    logger.warning(f"Error saving ML detector state: {e}")
            
            # Save to database
            self.db_manager.save_fault_tolerance_state(state)
            logger.debug("Persisted fault tolerance state")
            
        except Exception as e:
            logger.error(f"Error persisting state: {e}")
    
    def _persist_checkpoint(self, checkpoint_id: str):
        """
        Persist a checkpoint to database.
        
        Args:
            checkpoint_id: ID of the checkpoint to persist
        """
        if not self.db_manager:
            return
        
        try:
            with self.checkpoint_lock:
                if checkpoint_id not in self.checkpoints:
                    return
                
                checkpoint = self.checkpoints[checkpoint_id]
            
            # Save to database
            self.db_manager.save_checkpoint(checkpoint_id, checkpoint)
            logger.debug(f"Persisted checkpoint {checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Error persisting checkpoint {checkpoint_id}: {e}")
    
    def _load_persisted_state(self):
        """Load persisted state from database."""
        if not self.db_manager:
            return
        
        try:
            # Load state from database
            state = self.db_manager.get_fault_tolerance_state()
            
            if state:
                # Restore task states
                if "task_states" in state:
                    with self.task_state_lock:
                        self.task_states.update(state["task_states"])
                
                # Restore failure patterns
                if "failure_patterns" in state:
                    with self.failure_pattern_lock:
                        self.failure_patterns.update(state["failure_patterns"])
                
                # Restore recovery history
                if "recovery_history" in state:
                    with self.recovery_history_lock:
                        for task_id, actions in state["recovery_history"].items():
                            self.recovery_history[task_id] = [
                                self._dict_to_recovery_action(action)
                                for action in actions
                            ]
                
                # Load checkpoints
                self._load_checkpoints()
                
                # Load ML detector state if enabled
                if self.enable_ml and self.ml_detector and "ml_detector_state" in state:
                    try:
                        self.ml_detector.load_state(state["ml_detector_state"])
                        logger.info("Loaded ML detector state")
                    except Exception as e:
                        logger.warning(f"Error loading ML detector state: {e}")
                
                logger.info("Loaded persisted fault tolerance state")
            else:
                logger.info("No persisted fault tolerance state found")
                
        except Exception as e:
            logger.error(f"Error loading persisted state: {e}")
    
    def _load_checkpoints(self):
        """Load checkpoints from database."""
        if not self.db_manager:
            return
        
        try:
            # Load checkpoints from database
            checkpoints = self.db_manager.get_all_checkpoints()
            
            if checkpoints:
                with self.checkpoint_lock:
                    self.checkpoints.update(checkpoints)
                
                logger.info(f"Loaded {len(checkpoints)} checkpoints")
            else:
                logger.info("No checkpoints found")
                
        except Exception as e:
            logger.error(f"Error loading checkpoints: {e}")
    
    def _recovery_action_to_dict(self, action: RecoveryAction) -> Dict[str, Any]:
        """
        Convert a RecoveryAction to a dictionary for persistence.
        
        Args:
            action: RecoveryAction to convert
            
        Returns:
            Dictionary representation of the action
        """
        return {
            "strategy": action.strategy.name,
            "worker_id": action.worker_id,
            "delay": action.delay,
            "modified_task": action.modified_task,
            "checkpoint_data": action.checkpoint_data,
            "priority_adjustment": action.priority_adjustment,
            "message": action.message,
            "hardware_requirements": action.hardware_requirements
        }
    
    def _dict_to_recovery_action(self, data: Dict[str, Any]) -> RecoveryAction:
        """
        Convert a dictionary to a RecoveryAction.
        
        Args:
            data: Dictionary representation of an action
            
        Returns:
            RecoveryAction object
        """
        try:
            strategy = RecoveryStrategy[data["strategy"]]
        except (KeyError, ValueError):
            strategy = RecoveryStrategy.DELAYED_RETRY  # Default
        
        return RecoveryAction(
            strategy=strategy,
            worker_id=data.get("worker_id"),
            delay=data.get("delay", 0.0),
            modified_task=data.get("modified_task"),
            checkpoint_data=data.get("checkpoint_data"),
            priority_adjustment=data.get("priority_adjustment", 0),
            message=data.get("message", ""),
            hardware_requirements=data.get("hardware_requirements")
        )
    
    def _get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task data by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task data dictionary or None if not found
        """
        if self.coordinator and hasattr(self.coordinator, "task_manager"):
            return self.coordinator.task_manager.get_task(task_id)
        return None
    
    def _get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get worker data by ID.
        
        Args:
            worker_id: ID of the worker
            
        Returns:
            Worker data dictionary or None if not found
        """
        if self.coordinator and hasattr(self.coordinator, "worker_manager"):
            return self.coordinator.worker_manager.get_worker(worker_id)
        return None
    
    def _get_running_tasks(self) -> List[str]:
        """
        Get list of currently running task IDs.
        
        Returns:
            List of task IDs
        """
        if self.coordinator and hasattr(self.coordinator, "task_manager"):
            return self.coordinator.task_manager.get_running_tasks()
        return []


def create_recovery_manager(coordinator, db_manager=None, scheduler=None, enable_ml=False) -> HardwareAwareFaultToleranceManager:
    """
    Create a hardware-aware fault tolerance manager.
    
    Args:
        coordinator: Coordinator instance
        db_manager: Optional database manager
        scheduler: Optional heterogeneous scheduler
        enable_ml: Enable machine learning-based pattern detection
        
    Returns:
        Configured HardwareAwareFaultToleranceManager instance
    """
    # Create manager
    manager = HardwareAwareFaultToleranceManager(
        db_manager=db_manager,
        scheduler=scheduler,
        enable_ml=enable_ml,
        coordinator=coordinator
    )
    
    # Start the manager
    manager.start()
    
    logger.info("Created hardware-aware fault tolerance manager")
    return manager


# Helper functions for working with recovery actions

def visualize_fault_tolerance(manager, output_dir="./visualizations"):
    """
    Create visualizations for the fault tolerance system.
    
    Args:
        manager: HardwareAwareFaultToleranceManager instance
        output_dir: Directory to save visualizations
        
    Returns:
        Path to the generated report
    """
    try:
        # Import visualization module
        visualization = importlib.import_module("duckdb_api.distributed_testing.fault_tolerance_visualization")
        
        # Create visualizer and generate report
        report_path = visualization.visualize_fault_tolerance_system(
            fault_tolerance_manager=manager,
            output_dir=output_dir
        )
        
        return report_path
        
    except ImportError:
        logger.warning("Fault tolerance visualization module not available")
        return None
    except Exception as e:
        logger.error(f"Error visualizing fault tolerance system: {e}")
        return None

def apply_recovery_action(task_id: str, action: RecoveryAction, 
                         coordinator=None, scheduler=None) -> bool:
    """
    Apply a recovery action to a failed task.
    
    Args:
        task_id: ID of the failed task
        action: Recovery action to apply
        coordinator: Coordinator instance (optional)
        scheduler: Scheduler instance (optional)
        
    Returns:
        True if the action was applied successfully, False otherwise
    """
    if not coordinator:
        logger.error("Cannot apply recovery action: No coordinator provided")
        return False
    
    try:
        # Apply the action based on recovery strategy
        if action.strategy == RecoveryStrategy.IMMEDIATE_RETRY:
            # Retry immediately on the same worker
            coordinator.retry_task(task_id, worker_id=action.worker_id)
            
        elif action.strategy == RecoveryStrategy.DELAYED_RETRY:
            # Schedule delayed retry
            if action.delay > 0:
                # Use scheduler to delay the task
                threading.Timer(
                    action.delay, 
                    lambda: coordinator.retry_task(task_id, worker_id=action.worker_id)
                ).start()
            else:
                coordinator.retry_task(task_id, worker_id=action.worker_id)
            
        elif action.strategy == RecoveryStrategy.DIFFERENT_WORKER:
            # Retry on a different worker
            coordinator.retry_task(task_id, exclude_workers=[action.worker_id])
            
        elif action.strategy == RecoveryStrategy.DIFFERENT_HARDWARE_CLASS:
            # Retry on different hardware class
            if action.hardware_requirements:
                coordinator.update_task_requirements(task_id, action.hardware_requirements)
            coordinator.retry_task(task_id)
            
        elif action.strategy == RecoveryStrategy.REDUCED_PRECISION:
            # Retry with reduced precision
            if action.modified_task:
                coordinator.update_task_config(task_id, action.modified_task.get("config", {}))
            coordinator.retry_task(task_id)
            
        elif action.strategy == RecoveryStrategy.REDUCED_BATCH_SIZE:
            # Retry with reduced batch size
            if action.modified_task:
                coordinator.update_task_config(task_id, action.modified_task.get("config", {}))
            coordinator.retry_task(task_id)
            
        elif action.strategy == RecoveryStrategy.FALLBACK_CPU:
            # Fallback to CPU execution
            if action.hardware_requirements:
                coordinator.update_task_requirements(task_id, action.hardware_requirements)
            coordinator.retry_task(task_id)
            
        elif action.strategy == RecoveryStrategy.BROWSER_RESTART:
            # Restart browser and retry
            if action.worker_id:
                coordinator.restart_worker_browser(action.worker_id)
            coordinator.retry_task(task_id)
            
        elif action.strategy == RecoveryStrategy.RESET_WORKER_STATE:
            # Reset worker state and retry
            if action.worker_id:
                coordinator.reset_worker_state(action.worker_id)
            coordinator.retry_task(task_id)
            
        elif action.strategy == RecoveryStrategy.ESCALATION:
            # Escalate to human operator
            coordinator.escalate_task(task_id, action.message)
            
        else:
            logger.warning(f"Unsupported recovery strategy: {action.strategy}")
            return False
        
        logger.info(f"Applied recovery action for task {task_id}: {action.strategy.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error applying recovery action: {e}")
        return False