#!/usr/bin/env python3
"""
Enhanced Error Recovery for WebGPU/WebNN Resource Pool Bridge

This module extends the WebGPU/WebNN Resource Pool Bridge with enhanced error
recovery capabilities from the distributed testing framework's error recovery
with performance tracking system.

Integration points:
1. WebGPU/WebNN Resource Pool Bridge
2. Error Recovery with Performance Tracking
3. Distributed State Management
4. Performance History Tracking
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field

# Import error recovery with performance tracking
from error_recovery_with_performance import (
    ErrorRecoveryWithPerformance,
    RecoveryPerformanceMetric,
    RecoveryPerformanceRecord,
    RecoveryStrategyScore,
    ProgressiveRecoveryLevel
)

# Import hardware-test matcher for hardware-aware recovery
from hardware_test_matcher import TestHardwareMatcher

# Import distributed error handler for error categorization
from distributed_error_handler import DistributedErrorHandler, ErrorType, ErrorSeverity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResourcePoolErrorCategory(Enum):
    """Error categories specific to WebGPU/WebNN resource pool."""
    BROWSER_CONNECTION = auto()
    MODEL_INITIALIZATION = auto()
    MODEL_INFERENCE = auto()
    MODEL_MIGRATION = auto()
    MODEL_SHARD = auto()
    STATE_SYNC = auto()
    PERFORMANCE_TRACKING = auto()

@dataclass
class ModelExecutionContext:
    """Context for model execution operations."""
    model_id: str
    model_name: str
    model_type: str
    browser_id: str
    operation_type: str
    inputs: Any = None
    start_time: float = field(default_factory=time.time)
    result: Any = None
    error: Optional[Exception] = None
    recovery_attempts: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class EnhancedResourcePoolRecovery:
    """
    Enhanced recovery manager for WebGPU/WebNN Resource Pool.
    
    This class extends the resource pool's recovery capabilities with the
    error recovery with performance tracking system.
    """
    
    def __init__(
        self,
        state_manager: Any = None,
        performance_tracker: Any = None,
        sharding_manager: Any = None,
        connection_pool: Dict[str, Any] = None,
        recovery_database_path: str = "./resource_pool_recovery.db",
        enable_performance_tracking: bool = True,
        enable_hardware_aware_recovery: bool = True,
        enable_progressive_recovery: bool = True,
        adaptive_timeouts: bool = True
    ):
        """
        Initialize the enhanced resource pool recovery manager.
        
        Args:
            state_manager: Reference to browser state manager
            performance_tracker: Reference to performance history tracker
            sharding_manager: Reference to sharded model manager
            connection_pool: Reference to browser connection pool
            recovery_database_path: Path to recovery database
            enable_performance_tracking: Whether to enable performance tracking
            enable_hardware_aware_recovery: Whether to enable hardware-aware recovery
            enable_progressive_recovery: Whether to enable progressive recovery
            adaptive_timeouts: Whether to enable adaptive timeouts
        """
        self.state_manager = state_manager
        self.performance_tracker = performance_tracker
        self.sharding_manager = sharding_manager
        self.connection_pool = connection_pool
        
        # Create error recovery with performance tracking instance
        self.error_recovery = ErrorRecoveryWithPerformance(
            database_path=recovery_database_path,
            enable_performance_tracking=enable_performance_tracking,
            enable_progressive_recovery=enable_progressive_recovery,
            adaptive_timeouts=adaptive_timeouts
        )
        
        # Create hardware matcher for hardware-aware recovery
        self.hardware_matcher = TestHardwareMatcher() if enable_hardware_aware_recovery else None
        
        # Create distributed error handler for error categorization
        self.error_handler = DistributedErrorHandler()
        
        # Track active operations
        self.active_operations = {}
        
        # Track model recovery settings
        self.model_recovery_settings = {}
        
        # Track recovery history
        self.recovery_history = []
        
        # Error to recovery strategy mapping
        self.error_strategy_mapping = self._create_error_strategy_mapping()
        
        logger.info("EnhancedResourcePoolRecovery initialized")
    
    async def initialize(self):
        """Initialize the enhanced recovery manager."""
        logger.info("Initializing EnhancedResourcePoolRecovery...")
        
        # Initialize error recovery with performance
        await self.error_recovery.initialize()
        
        # Initialize hardware matcher if enabled
        if self.hardware_matcher:
            await self.hardware_matcher.initialize()
        
        # Initialize error handler
        await self.error_handler.initialize()
        
        # Register resource pool specific error categories
        self._register_error_categories()
        
        # Load previous recovery history if available
        if hasattr(self.error_recovery, 'load_recovery_history'):
            self.recovery_history = await self.error_recovery.load_recovery_history()
        
        logger.info("EnhancedResourcePoolRecovery initialization complete")
    
    def _register_error_categories(self):
        """Register resource pool specific error categories with error handler."""
        error_categories = {
            ResourcePoolErrorCategory.BROWSER_CONNECTION: {
                "name": "Browser Connection",
                "description": "Errors related to browser connection management",
                "default_severity": ErrorSeverity.HIGH
            },
            ResourcePoolErrorCategory.MODEL_INITIALIZATION: {
                "name": "Model Initialization",
                "description": "Errors during model initialization",
                "default_severity": ErrorSeverity.MEDIUM
            },
            ResourcePoolErrorCategory.MODEL_INFERENCE: {
                "name": "Model Inference",
                "description": "Errors during model inference execution",
                "default_severity": ErrorSeverity.HIGH
            },
            ResourcePoolErrorCategory.MODEL_MIGRATION: {
                "name": "Model Migration",
                "description": "Errors during model migration between browsers",
                "default_severity": ErrorSeverity.MEDIUM
            },
            ResourcePoolErrorCategory.MODEL_SHARD: {
                "name": "Model Shard",
                "description": "Errors related to sharded model execution",
                "default_severity": ErrorSeverity.HIGH
            },
            ResourcePoolErrorCategory.STATE_SYNC: {
                "name": "State Synchronization",
                "description": "Errors during state synchronization",
                "default_severity": ErrorSeverity.MEDIUM
            },
            ResourcePoolErrorCategory.PERFORMANCE_TRACKING: {
                "name": "Performance Tracking",
                "description": "Errors in performance tracking",
                "default_severity": ErrorSeverity.LOW
            }
        }
        
        for category, info in error_categories.items():
            self.error_handler.register_error_category(
                category,
                info["name"],
                info["description"],
                info["default_severity"]
            )
    
    def _create_error_strategy_mapping(self):
        """Create mapping of error types to recovery strategies."""
        return {
            ResourcePoolErrorCategory.BROWSER_CONNECTION: {
                "strategies": ["browser_reconnect", "browser_restart", "browser_recreate"],
                "default_level": ProgressiveRecoveryLevel.MEDIUM
            },
            ResourcePoolErrorCategory.MODEL_INITIALIZATION: {
                "strategies": ["model_reinitialize", "model_migrate", "model_recreate"],
                "default_level": ProgressiveRecoveryLevel.MEDIUM
            },
            ResourcePoolErrorCategory.MODEL_INFERENCE: {
                "strategies": ["operation_retry", "model_reset", "model_migrate"],
                "default_level": ProgressiveRecoveryLevel.LOW
            },
            ResourcePoolErrorCategory.MODEL_MIGRATION: {
                "strategies": ["migration_retry", "alternative_browser", "model_recreate"],
                "default_level": ProgressiveRecoveryLevel.MEDIUM
            },
            ResourcePoolErrorCategory.MODEL_SHARD: {
                "strategies": ["shard_retry", "shard_reassign", "full_retry"],
                "default_level": ProgressiveRecoveryLevel.HIGH
            },
            ResourcePoolErrorCategory.STATE_SYNC: {
                "strategies": ["sync_retry", "partial_rebuild", "full_rebuild"],
                "default_level": ProgressiveRecoveryLevel.MEDIUM
            },
            ResourcePoolErrorCategory.PERFORMANCE_TRACKING: {
                "strategies": ["soft_reset", "clear_cache", "rebuild_index"],
                "default_level": ProgressiveRecoveryLevel.LOW
            }
        }
    
    async def set_model_recovery_settings(
        self,
        model_id: str,
        recovery_timeout: int,
        state_persistence: bool,
        failover_strategy: str,
        priority_level: str = "medium"
    ):
        """
        Set recovery settings for a specific model.
        
        Args:
            model_id: Model ID
            recovery_timeout: Maximum recovery time in seconds
            state_persistence: Whether to persist state between sessions
            failover_strategy: Strategy for failover (immediate, progressive, etc.)
            priority_level: Priority level for recovery (low, medium, high)
        """
        # Store settings
        self.model_recovery_settings[model_id] = {
            "recovery_timeout": recovery_timeout,
            "state_persistence": state_persistence,
            "failover_strategy": failover_strategy,
            "priority_level": priority_level
        }
        
        # Convert priority level to ProgressiveRecoveryLevel enum
        priority_enum = ProgressiveRecoveryLevel.MEDIUM
        if priority_level.lower() == "low":
            priority_enum = ProgressiveRecoveryLevel.LOW
        elif priority_level.lower() == "high":
            priority_enum = ProgressiveRecoveryLevel.HIGH
        
        # Register with error recovery system
        await self.error_recovery.register_component(
            component_id=model_id,
            component_type="model",
            recovery_timeout=recovery_timeout,
            priority_level=priority_enum,
            state_persistence=state_persistence,
            metadata={
                "model_id": model_id,
                "failover_strategy": failover_strategy
            }
        )
        
        logger.info(f"Recovery settings updated for model {model_id}")
    
    async def start_operation(
        self,
        model_id: str,
        model_name: str,
        model_type: str,
        browser_id: str,
        operation_type: str,
        inputs: Any = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Start tracking an operation for recovery purposes.
        
        Args:
            model_id: ID of the model performing the operation
            model_name: Name of the model
            model_type: Type of model (text_embedding, vision, audio, etc.)
            browser_id: ID of the browser running the model
            operation_type: Type of operation
            inputs: Inputs to the operation (for retry)
            metadata: Additional metadata for the operation
            
        Returns:
            Operation ID
        """
        # Generate operation ID
        operation_id = f"op-{uuid.uuid4().hex}"
        
        # Create context
        context = ModelExecutionContext(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            browser_id=browser_id,
            operation_type=operation_type,
            inputs=inputs,
            start_time=time.time(),
            performance_metrics={}
        )
        
        # Store context
        self.active_operations[operation_id] = context
        
        # Start operation tracking in error recovery system
        await self.error_recovery.start_operation(
            operation_id=operation_id,
            component_id=model_id,
            operation_type=operation_type,
            metadata=metadata or {
                "model_name": model_name,
                "model_type": model_type,
                "browser_id": browser_id,
                "inputs_hash": self._calculate_inputs_hash(inputs)
            }
        )
        
        # Record operation in state manager if available
        if self.state_manager:
            await self.state_manager.record_operation(
                operation_id=operation_id,
                model_id=model_id,
                operation_type=operation_type,
                start_time=datetime.now().isoformat(),
                status="started",
                metadata=metadata
            )
        
        logger.debug(f"Started tracking operation {operation_id} for model {model_id}")
        return operation_id
    
    async def complete_operation(
        self,
        operation_id: str,
        result: Any = None,
        performance_metrics: Dict[str, Any] = None
    ):
        """
        Mark an operation as completed.
        
        Args:
            operation_id: Operation ID
            result: Result of the operation
            performance_metrics: Performance metrics for the operation
        """
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found, cannot complete")
            return False
        
        # Get context
        context = self.active_operations[operation_id]
        
        # Store result
        context.result = result
        
        # Calculate duration
        duration = time.time() - context.start_time
        
        # Store performance metrics
        if performance_metrics:
            context.performance_metrics.update(performance_metrics)
        
        # Add duration if not provided
        if "duration" not in context.performance_metrics:
            context.performance_metrics["duration"] = duration
        
        # Complete operation in error recovery system
        await self.error_recovery.complete_operation(
            operation_id=operation_id,
            success=True,
            result=result,
            execution_time=duration,
            performance_metrics=context.performance_metrics
        )
        
        # Record in state manager if available
        if self.state_manager:
            await self.state_manager.complete_operation(
                operation_id=operation_id,
                status="completed",
                end_time=datetime.now().isoformat(),
                result=result
            )
        
        # Record performance data if available
        if self.performance_tracker:
            await self.performance_tracker.record_operation_performance(
                browser_id=context.browser_id,
                model_id=context.model_id,
                model_type=context.model_type,
                operation_type=context.operation_type,
                latency=duration * 1000,  # convert to ms
                success=True,
                metadata=context.performance_metrics
            )
        
        # Remove from active operations
        del self.active_operations[operation_id]
        
        logger.debug(f"Operation {operation_id} completed")
        return True
    
    async def fail_operation(
        self,
        operation_id: str,
        error: Exception,
        error_category: Union[ResourcePoolErrorCategory, ErrorType],
        error_severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ):
        """
        Mark an operation as failed.
        
        Args:
            operation_id: Operation ID
            error: Exception that caused the failure
            error_category: Category of the error
            error_severity: Severity of the error
        """
        if operation_id not in self.active_operations:
            logger.warning(f"Operation {operation_id} not found, cannot mark as failed")
            return False
        
        # Get context
        context = self.active_operations[operation_id]
        
        # Store error
        context.error = error
        
        # Calculate duration
        duration = time.time() - context.start_time
        
        # Register error with error handler
        # Create error context
        context_data = {
            "component": context.model_id,
            "operation": operation_id,
            "metadata": {
                "model_name": context.model_name,
                "model_type": context.model_type,
                "browser_id": context.browser_id,
                "operation_type": context.operation_type,
                "duration": duration
            }
        }
        
        # Handle the error
        error_report = self.error_handler.handle_error(error, context_data)
        error_id = error_report.error_id
        
        # Fail operation in error recovery system
        await self.error_recovery.fail_operation(
            operation_id=operation_id,
            error_message=str(error),
            error_id=error_id,
            execution_time=duration
        )
        
        # Record in state manager if available
        if self.state_manager:
            await self.state_manager.complete_operation(
                operation_id=operation_id,
                status="failed",
                end_time=datetime.now().isoformat(),
                result={"error": str(error)}
            )
        
        # Record performance data if available
        if self.performance_tracker:
            await self.performance_tracker.record_operation_performance(
                browser_id=context.browser_id,
                model_id=context.model_id,
                model_type=context.model_type,
                operation_type=context.operation_type,
                latency=duration * 1000,  # convert to ms
                success=False,
                metadata={
                    "error": str(error),
                    "error_category": str(error_category),
                    "error_severity": str(error_severity)
                }
            )
        
        logger.warning(f"Operation {operation_id} failed with error: {str(error)}")
        return True
    
    async def recover_model_operation(
        self,
        model_id: str,
        operation_type: str,
        error: str,
        inputs: Any = None,
        browser_id: str = None,
        model_info: Dict[str, Any] = None
    ) -> Tuple[bool, Any]:
        """
        Recover from a failed model operation.
        
        Args:
            model_id: ID of the model that failed
            operation_type: Type of operation that failed
            error: Error message
            inputs: Inputs to the operation (for retry)
            browser_id: Browser ID (optional)
            model_info: Additional model information
            
        Returns:
            Tuple of (success, recovered_model)
        """
        # Get model state from state manager if available
        model_state = None
        if self.state_manager:
            model_state = self.state_manager.get_model_state(model_id)
        
        # Get browser ID from model state if not provided
        if not browser_id and model_state:
            browser_id = model_state.get("browser_id")
        
        # Get browser state
        browser_state = None
        if self.state_manager and browser_id:
            browser_state = self.state_manager.get_browser_state(browser_id)
        
        # Get model name and type
        model_name = model_info.get("name") if model_info else None
        model_type = model_info.get("type") if model_info else None
        
        if model_state:
            model_name = model_name or model_state.get("name")
            model_type = model_type or model_state.get("type")
        
        # Create recovery context
        recovery_context = {
            "model_id": model_id,
            "model_name": model_name,
            "model_type": model_type,
            "browser_id": browser_id,
            "operation_type": operation_type,
            "error": error,
            "inputs": inputs
        }
        
        # Determine recovery strategy using error recovery system
        recovery_result = await self.error_recovery.recover_operation(
            component_id=model_id,
            operation_type=operation_type,
            error_message=error,
            context=recovery_context
        )
        
        if not recovery_result.success:
            logger.error(f"Recovery failed for model {model_id}: {recovery_result.error_message}")
            return False, None
        
        # Extract new browser ID from recovery result
        if 'new_browser_id' in recovery_result.context:
            new_browser_id = recovery_result.context['new_browser_id']
        elif 'browser_id' in recovery_result.context:
            new_browser_id = recovery_result.context['browser_id']
        else:
            new_browser_id = browser_id
        
        # Create a recovered model proxy (real implementation would create actual model)
        if recovery_result.recovery_action == "model_migrate" or recovery_result.recovery_action == "browser_recreate":
            # Update model browser in state manager
            if self.state_manager and new_browser_id != browser_id:
                await self.state_manager.update_model_browser(model_id, new_browser_id)
        
        # In real implementation, would recreate the model proxy
        # For now, import and create a simple proxy
        from resource_pool_bridge import ModelProxy
        
        recovered_model = ModelProxy(
            model_id=model_id,
            model_name=model_name or "unknown",
            model_type=model_type or "unknown",
            browser_id=new_browser_id
        )
        
        # Record recovery attempt in history
        self.recovery_history.append({
            "model_id": model_id,
            "operation_type": operation_type,
            "error": error,
            "recovery_action": recovery_result.recovery_action,
            "success": True,
            "time": datetime.now().isoformat()
        })
        
        logger.info(f"Model {model_id} recovered with action {recovery_result.recovery_action}")
        return True, recovered_model
    
    async def recover_browser(
        self,
        browser_id: str,
        error: str,
        browser_type: str = None
    ) -> Tuple[bool, str]:
        """
        Recover from a browser failure.
        
        Args:
            browser_id: Browser ID
            error: Error message
            browser_type: Browser type (chrome, firefox, edge)
            
        Returns:
            Tuple of (success, new_browser_id)
        """
        # Get browser state from state manager if available
        browser_state = None
        if self.state_manager:
            browser_state = self.state_manager.get_browser_state(browser_id)
        
        # Get browser type from state if not provided
        if not browser_type and browser_state:
            browser_type = browser_state.get("type")
        
        # Create recovery context
        recovery_context = {
            "browser_id": browser_id,
            "browser_type": browser_type,
            "error": error
        }
        
        # Determine recovery strategy using error recovery system
        recovery_result = await self.error_recovery.recover_operation(
            component_id=browser_id,
            operation_type="browser_connection",
            error_message=error,
            context=recovery_context
        )
        
        if not recovery_result.success:
            logger.error(f"Recovery failed for browser {browser_id}: {recovery_result.error_message}")
            return False, None
        
        # Generate new browser ID
        new_browser_id = f"{browser_type or 'browser'}-{uuid.uuid4().hex[:8]}"
        
        # Create new browser in connection pool
        if self.connection_pool is not None:
            # Add to connection pool
            self.connection_pool[new_browser_id] = {
                'id': new_browser_id,
                'type': browser_type or 'chrome',
                'status': 'initializing',
                'capabilities': {},
                'created_at': datetime.now().isoformat(),
                'active_models': set(),
                'performance_metrics': {}
            }
            
            # Simulate browser initialization
            await asyncio.sleep(0.1)
            
            # Update status
            self.connection_pool[new_browser_id]['status'] = 'ready'
        
        # Register browser with state manager if available
        if self.state_manager:
            await self.state_manager.register_browser(
                browser_id=new_browser_id,
                browser_type=browser_type or 'chrome',
                capabilities={}
            )
        
        # Record recovery attempt in history
        self.recovery_history.append({
            "browser_id": browser_id,
            "new_browser_id": new_browser_id,
            "recovery_action": recovery_result.recovery_action,
            "success": True,
            "time": datetime.now().isoformat()
        })
        
        logger.info(f"Browser {browser_id} recovered with new browser {new_browser_id}")
        return True, new_browser_id
    
    async def recover_sharded_model(
        self,
        model_id: str,
        failed_shard_ids: List[str],
        error: str,
        strategy: str = "retry_failed_shards"
    ) -> bool:
        """
        Recover from a failure in a sharded model.
        
        Args:
            model_id: Sharded model ID
            failed_shard_ids: IDs of failed shards
            error: Error message
            strategy: Recovery strategy (retry_failed_shards, reassign_shards, full_retry)
            
        Returns:
            Success flag
        """
        # Get sharded model state
        sharded_model = None
        if self.sharding_manager:
            sharded_model = self.sharding_manager.get_sharded_model(model_id)
        
        if not sharded_model:
            logger.error(f"Sharded model {model_id} not found")
            return False
        
        # Create recovery context
        recovery_context = {
            "model_id": model_id,
            "failed_shard_ids": failed_shard_ids,
            "error": error,
            "strategy": strategy
        }
        
        # Determine recovery strategy using error recovery system
        recovery_result = await self.error_recovery.recover_operation(
            component_id=model_id,
            operation_type="sharded_model",
            error_message=error,
            context=recovery_context
        )
        
        if not recovery_result.success:
            logger.error(f"Recovery failed for sharded model {model_id}: {recovery_result.error_message}")
            return False
        
        # Recover shards based on strategy
        if strategy == "retry_failed_shards" or recovery_result.recovery_action == "shard_retry":
            # Retry only failed shards
            for shard_id in failed_shard_ids:
                success = await self._retry_shard(model_id, shard_id)
                if not success:
                    logger.error(f"Failed to retry shard {shard_id} of model {model_id}")
                    return False
        
        elif strategy == "reassign_shards" or recovery_result.recovery_action == "shard_reassign":
            # Reassign failed shards to different browsers
            for shard_id in failed_shard_ids:
                success = await self._reassign_shard(model_id, shard_id)
                if not success:
                    logger.error(f"Failed to reassign shard {shard_id} of model {model_id}")
                    return False
        
        elif strategy == "full_retry" or recovery_result.recovery_action == "full_retry":
            # Retry the entire sharded model
            success = await self._full_retry_sharded_model(model_id)
            if not success:
                logger.error(f"Failed to perform full retry of sharded model {model_id}")
                return False
        
        # Record recovery attempt in history
        self.recovery_history.append({
            "model_id": model_id,
            "failed_shard_ids": failed_shard_ids,
            "recovery_action": recovery_result.recovery_action,
            "success": True,
            "time": datetime.now().isoformat()
        })
        
        logger.info(f"Sharded model {model_id} recovered with action {recovery_result.recovery_action}")
        return True
    
    async def _retry_shard(self, model_id: str, shard_id: str) -> bool:
        """
        Retry a failed shard on the same browser.
        
        Args:
            model_id: Sharded model ID
            shard_id: ID of the shard to retry
            
        Returns:
            Success flag
        """
        # In real implementation, would reset and retry the shard
        # For now, just simulate success
        await asyncio.sleep(0.1)
        
        logger.info(f"Retried shard {shard_id} of model {model_id}")
        return True
    
    async def _reassign_shard(self, model_id: str, shard_id: str) -> bool:
        """
        Reassign a failed shard to a different browser.
        
        Args:
            model_id: Sharded model ID
            shard_id: ID of the shard to reassign
            
        Returns:
            Success flag
        """
        # Get sharded model info
        sharded_model = None
        if self.sharding_manager:
            sharded_model = self.sharding_manager.get_sharded_model(model_id)
        
        if not sharded_model:
            logger.error(f"Sharded model {model_id} not found")
            return False
        
        # Get shard info
        shard_info = sharded_model.get("shards", {}).get(shard_id)
        if not shard_info:
            logger.error(f"Shard {shard_id} not found in model {model_id}")
            return False
        
        # Get current browser ID
        current_browser_id = shard_info.get("browser_id")
        if not current_browser_id:
            logger.error(f"Browser ID not found for shard {shard_id}")
            return False
        
        # Find a new browser
        new_browser_id = None
        if self.connection_pool:
            # Find browser with fewest models
            candidates = []
            for browser_id, browser in self.connection_pool.items():
                if browser_id != current_browser_id and browser.get("status") == "ready":
                    load = len(browser.get("active_models", set()))
                    candidates.append((browser_id, load))
            
            if candidates:
                # Sort by load (ascending)
                candidates.sort(key=lambda x: x[1])
                new_browser_id = candidates[0][0]
        
        if not new_browser_id:
            logger.error(f"No suitable browser found for shard {shard_id}")
            return False
        
        # Update shard browser
        if self.sharding_manager:
            await self.sharding_manager.update_shard_browser(model_id, shard_id, new_browser_id)
        
        logger.info(f"Reassigned shard {shard_id} from browser {current_browser_id} to {new_browser_id}")
        return True
    
    async def _full_retry_sharded_model(self, model_id: str) -> bool:
        """
        Perform a full retry of a sharded model.
        
        Args:
            model_id: Sharded model ID
            
        Returns:
            Success flag
        """
        # In real implementation, would reset and retry the entire model
        # For now, just simulate success
        await asyncio.sleep(0.2)
        
        logger.info(f"Performed full retry of sharded model {model_id}")
        return True
    
    def _calculate_inputs_hash(self, inputs: Any) -> str:
        """Calculate a hash for operation inputs."""
        if inputs is None:
            return "none"
        
        try:
            # Serialize inputs to JSON
            if isinstance(inputs, (dict, list)):
                inputs_str = json.dumps(inputs, sort_keys=True)
            else:
                inputs_str = str(inputs)
            
            # Calculate hash
            return hashlib.md5(inputs_str.encode()).hexdigest()
        except:
            return str(hash(str(inputs)))
    
    async def analyze_recovery_performance(
        self,
        model_type: str = None,
        time_range: str = "7d",
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance of recovery strategies.
        
        Args:
            model_type: Filter by model type
            time_range: Time range for analysis
            metrics: Specific metrics to include
            
        Returns:
            Performance analysis results
        """
        # Use error recovery performance analysis
        return await self.error_recovery.analyze_recovery_performance(
            component_type="model" if model_type else None,
            component_filter={"model_type": model_type} if model_type else None,
            time_range=time_range,
            metrics=metrics
        )
    
    async def optimize_recovery_strategies(self) -> Dict[str, Any]:
        """
        Optimize recovery strategies based on performance history.
        
        Returns:
            Optimization results
        """
        # Use error recovery strategy optimization
        optimization_result = await self.error_recovery.optimize_recovery_strategies()
        
        # Apply optimizations to model recovery settings
        for component_id, strategies in optimization_result.get("component_strategies", {}).items():
            if component_id in self.model_recovery_settings:
                self.model_recovery_settings[component_id]["optimized_strategies"] = strategies
        
        return optimization_result

# Integration with WebGPU/WebNN Resource Pool Bridge
class FaultTolerantModelProxyEnhanced:
    """
    Enhanced fault-tolerant proxy for a model in the browser.
    
    This class extends the basic ModelProxy with enhanced fault tolerance capabilities.
    """
    
    def __init__(
        self,
        model_id: str,
        model_name: str,
        model_type: str,
        browser_id: str,
        recovery_manager: Any
    ):
        """
        Initialize the enhanced fault-tolerant model proxy.
        
        Args:
            model_id: Unique identifier for the model
            model_name: Name of the model
            model_type: Type of model
            browser_id: ID of the browser running the model
            recovery_manager: Reference to the enhanced recovery manager
        """
        self.model_id = model_id
        self.model_name = model_name
        self.model_type = model_type
        self.browser_id = browser_id
        self.recovery_manager = recovery_manager
        
        logger.debug(f"Created enhanced fault-tolerant model proxy for {model_id}")
    
    async def __call__(self, inputs: Any) -> Any:
        """
        Run inference with the model with enhanced automatic recovery.
        
        Args:
            inputs: Input data for inference
            
        Returns:
            Inference results
        """
        try:
            # Start tracking operation
            operation_id = await self.recovery_manager.start_operation(
                model_id=self.model_id,
                model_name=self.model_name,
                model_type=self.model_type,
                browser_id=self.browser_id,
                operation_type="inference",
                inputs=inputs
            )
            
            # Simulate model inference (real implementation would call actual model)
            logger.debug(f"Running inference with model {self.model_id} on browser {self.browser_id}")
            
            # Simulate processing time
            await asyncio.sleep(0.5)
            
            # Generate result based on input
            if isinstance(inputs, dict):
                result = {"result": f"Processed {inputs}", "model_id": self.model_id}
            elif isinstance(inputs, list):
                result = [f"Processed item {i}" for i in range(len(inputs))]
            else:
                result = f"Processed {inputs}"
            
            # Complete operation
            await self.recovery_manager.complete_operation(
                operation_id=operation_id,
                result=result,
                performance_metrics={
                    "latency_ms": 500,
                    "tokens_processed": len(str(inputs)) if inputs else 0
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error running inference with model {self.model_id}: {str(e)}")
            
            # Mark operation as failed
            if 'operation_id' in locals():
                await self.recovery_manager.fail_operation(
                    operation_id=operation_id,
                    error=e,
                    error_category=ResourcePoolErrorCategory.MODEL_INFERENCE
                )
            
            # Attempt recovery
            if self.recovery_manager:
                recovered, new_model = await self.recovery_manager.recover_model_operation(
                    model_id=self.model_id,
                    operation_type="inference",
                    error=str(e),
                    inputs=inputs,
                    browser_id=self.browser_id,
                    model_info={
                        "name": self.model_name,
                        "type": self.model_type
                    }
                )
                
                if recovered and new_model:
                    # Update our browser ID if model was migrated
                    self.browser_id = new_model.browser_id
                    
                    # Try again with the recovered model
                    logger.info(f"Retrying inference after recovery with browser {self.browser_id}")
                    
                    # Run inference on recovered model
                    return await new_model(inputs)
                else:
                    # Recovery failed
                    raise Exception(f"Model operation failed and recovery was unsuccessful: {str(e)}")
            else:
                # No recovery manager
                raise
    
    async def get_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "id": self.model_id,
            "name": self.model_name,
            "type": self.model_type,
            "browser_id": self.browser_id,
            "fault_tolerance": "enhanced"
        }