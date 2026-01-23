#!/usr/bin/env python3
"""
Error Recovery with Performance Tracking for Distributed Testing Framework

This module provides enhanced error recovery capabilities with performance history tracking
for the distributed testing framework. It builds on the existing error handling system and
adds performance-based strategy selection, adaptive timeouts, and progressive recovery.

Key features:
- Performance history tracking for recovery strategies
- Data-driven recovery strategy selection based on historical performance
- Adaptive recovery timeouts based on performance patterns
- Progressive recovery strategies with escalation for persistent errors
- Integration with hardware-test matcher for hardware-aware recovery
- Recovery performance analytics and reporting
"""

import anyio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Tuple, Callable, Union, DefaultDict
from dataclasses import dataclass, field
from collections import defaultdict, deque
import traceback
import copy
import math
import statistics

# Import related modules
from distributed_error_handler import (
    DistributedErrorHandler, ErrorType, ErrorSeverity, ErrorReport, 
    ErrorContext, RetryPolicy, safe_execute, safe_execute_async
)

from hardware_test_matcher import (
    TestHardwareMatcher, TestProfile, TestType, TestRequirementType,
    TestRequirement, HardwareTestMatch
)

from error_recovery_strategies import (
    ErrorCategory, RecoveryStrategy, RetryStrategy, WorkerRecoveryStrategy,
    DatabaseRecoveryStrategy, CoordinatorRecoveryStrategy, SystemRecoveryStrategy,
    EnhancedErrorRecoveryManager
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("error_recovery_with_performance")


class RecoveryPerformanceMetric(Enum):
    """Performance metrics for recovery strategies."""
    SUCCESS_RATE = "success_rate"         # Success rate of recovery strategy
    RECOVERY_TIME = "recovery_time"       # Time taken for recovery
    RESOURCE_USAGE = "resource_usage"     # Resources used during recovery
    IMPACT_SCORE = "impact_score"         # Impact on system during recovery
    STABILITY = "stability"               # Post-recovery stability
    TASK_RECOVERY = "task_recovery"       # Success rate of task recovery


@dataclass
class RecoveryPerformanceRecord:
    """Performance record for a recovery strategy."""
    strategy_id: str
    strategy_name: str
    error_type: str
    execution_time_seconds: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    affected_tasks: int = 0
    task_recovery_success: int = 0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    impact_score: float = 0.0
    post_recovery_stability: float = 1.0
    metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class RecoveryStrategyScore:
    """Score for a recovery strategy."""
    strategy_id: str
    strategy_name: str
    error_type: str
    success_rate: float
    average_recovery_time: float
    resource_efficiency: float
    impact_score: float
    stability_score: float
    task_recovery_rate: float
    overall_score: float
    sample_count: int
    last_used: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    

class ProgressiveRecoveryLevel(Enum):
    """Levels for progressive recovery escalation."""
    LEVEL_1 = 1  # Basic retry
    LEVEL_2 = 2  # Enhanced retry with advanced parameters
    LEVEL_3 = 3  # Component restart
    LEVEL_4 = 4  # Component recovery with state reset
    LEVEL_5 = 5  # System-wide intervention


class ErrorRecoveryWithPerformance:
    """
    Enhanced error recovery system with performance tracking.
    
    This class provides an enhanced error recovery system that tracks performance
    of different recovery strategies and uses historical data to select the most
    effective strategy for each error type.
    """
    
    def __init__(
            self,
            error_handler: DistributedErrorHandler,
            hardware_matcher: Optional[TestHardwareMatcher] = None,
            recovery_manager: Optional[EnhancedErrorRecoveryManager] = None,
            coordinator=None,
            db_connection=None
        ):
        """
        Initialize the enhanced error recovery system.
        
        Args:
            error_handler: The distributed error handler
            hardware_matcher: Optional hardware test matcher for hardware-aware recovery
            recovery_manager: Optional enhanced recovery manager
            coordinator: Optional reference to the coordinator
            db_connection: Optional database connection
        """
        self.error_handler = error_handler
        self.hardware_matcher = hardware_matcher
        self.recovery_manager = recovery_manager
        self.coordinator = coordinator
        self.db_connection = db_connection
        
        # Performance history
        self.performance_history: Dict[str, List[RecoveryPerformanceRecord]] = defaultdict(list)
        
        # Strategy scores
        self.strategy_scores: Dict[str, Dict[str, RecoveryStrategyScore]] = defaultdict(dict)
        
        # Recovery timeouts
        self.default_timeout = 30.0  # Default timeout in seconds
        self.timeout_history: Dict[str, List[float]] = defaultdict(list)
        self.adaptive_timeouts: Dict[str, float] = {}
        
        # Progressive recovery tracking
        self.error_recovery_levels: Dict[str, int] = {}  # error_id -> current level
        self.error_recovery_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Local cache to prevent DB calls during recovery
        self.recovery_strategy_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize database tables
        if self.db_connection:
            self._create_performance_tables()
            
        # Load historical performance data
        self._load_performance_history()
        
        # Register with error handler for notifications
        if self.error_handler:
            # Register for all error types
            for error_type in ErrorType:
                self.error_handler.register_error_hook(
                    error_type,
                    self._error_notification_hook
                )
        
        logger.info("Error Recovery with Performance initialized")

    def _create_performance_tables(self):
        """Create database tables for performance tracking."""
        try:
            # Recovery performance records
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS recovery_performance (
                id INTEGER PRIMARY KEY,
                strategy_id VARCHAR,
                strategy_name VARCHAR,
                error_type VARCHAR,
                execution_time_seconds FLOAT,
                success BOOLEAN,
                timestamp TIMESTAMP,
                affected_tasks INTEGER,
                task_recovery_success INTEGER,
                resource_usage JSON,
                impact_score FLOAT,
                post_recovery_stability FLOAT,
                metrics JSON,
                context JSON
            )
            """)
            
            # Strategy scores
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS recovery_strategy_scores (
                strategy_id VARCHAR,
                strategy_name VARCHAR,
                error_type VARCHAR,
                success_rate FLOAT,
                average_recovery_time FLOAT,
                resource_efficiency FLOAT,
                impact_score FLOAT,
                stability_score FLOAT,
                task_recovery_rate FLOAT,
                overall_score FLOAT,
                sample_count INTEGER,
                last_used TIMESTAMP,
                metrics JSON,
                PRIMARY KEY (strategy_id, error_type)
            )
            """)
            
            # Recovery timeouts history
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS recovery_timeouts (
                error_type VARCHAR,
                strategy_id VARCHAR,
                timeout_seconds FLOAT,
                timestamp TIMESTAMP,
                success BOOLEAN,
                PRIMARY KEY (error_type, strategy_id, timestamp)
            )
            """)
            
            # Progressive recovery history
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS progressive_recovery_history (
                error_id VARCHAR,
                recovery_level INTEGER,
                strategy_id VARCHAR,
                strategy_name VARCHAR,
                timestamp TIMESTAMP,
                success BOOLEAN,
                details JSON,
                PRIMARY KEY (error_id, recovery_level)
            )
            """)
            
            logger.info("Recovery performance tables created")
        except Exception as e:
            logger.error(f"Error creating performance tables: {str(e)}")
    
    def _load_performance_history(self):
        """Load historical performance data from database."""
        if not self.db_connection:
            logger.warning("No database connection, cannot load performance history")
            return
        
        try:
            # Load recovery performance records
            result = self.db_connection.execute("""
            SELECT 
                strategy_id, strategy_name, error_type, execution_time_seconds,
                success, timestamp, affected_tasks, task_recovery_success,
                resource_usage, impact_score, post_recovery_stability, 
                metrics, context
            FROM recovery_performance
            WHERE timestamp > datetime('now', '-30 day')
            """).fetchall()
            
            for row in result:
                (
                    strategy_id, strategy_name, error_type, execution_time_seconds,
                    success, timestamp, affected_tasks, task_recovery_success,
                    resource_usage, impact_score, post_recovery_stability,
                    metrics, context
                ) = row
                
                record = RecoveryPerformanceRecord(
                    strategy_id=strategy_id,
                    strategy_name=strategy_name,
                    error_type=error_type,
                    execution_time_seconds=execution_time_seconds,
                    success=success,
                    timestamp=timestamp,
                    affected_tasks=affected_tasks,
                    task_recovery_success=task_recovery_success,
                    resource_usage=json.loads(resource_usage) if resource_usage else {},
                    impact_score=impact_score,
                    post_recovery_stability=post_recovery_stability,
                    metrics=json.loads(metrics) if metrics else {},
                    context=json.loads(context) if context else {}
                )
                
                # Group by strategy_id
                self.performance_history[strategy_id].append(record)
            
            # Load strategy scores
            result = self.db_connection.execute("""
            SELECT 
                strategy_id, strategy_name, error_type, success_rate,
                average_recovery_time, resource_efficiency, impact_score,
                stability_score, task_recovery_rate, overall_score,
                sample_count, last_used, metrics
            FROM recovery_strategy_scores
            """).fetchall()
            
            for row in result:
                (
                    strategy_id, strategy_name, error_type, success_rate,
                    average_recovery_time, resource_efficiency, impact_score,
                    stability_score, task_recovery_rate, overall_score,
                    sample_count, last_used, metrics
                ) = row
                
                score = RecoveryStrategyScore(
                    strategy_id=strategy_id,
                    strategy_name=strategy_name,
                    error_type=error_type,
                    success_rate=success_rate,
                    average_recovery_time=average_recovery_time,
                    resource_efficiency=resource_efficiency,
                    impact_score=impact_score,
                    stability_score=stability_score,
                    task_recovery_rate=task_recovery_rate,
                    overall_score=overall_score,
                    sample_count=sample_count,
                    last_used=last_used,
                    metrics=json.loads(metrics) if metrics else {}
                )
                
                # Store by error_type and strategy_id
                self.strategy_scores[error_type][strategy_id] = score
            
            # Load timeout history
            result = self.db_connection.execute("""
            SELECT 
                error_type, strategy_id, timeout_seconds
            FROM recovery_timeouts
            WHERE timestamp > datetime('now', '-30 day')
            """).fetchall()
            
            for row in result:
                error_type, strategy_id, timeout_seconds = row
                key = f"{error_type}:{strategy_id}"
                self.timeout_history[key].append(timeout_seconds)
            
            # Calculate adaptive timeouts
            self._update_adaptive_timeouts()
            
            logger.info(f"Loaded performance history: {sum(len(records) for records in self.performance_history.values())} records, {sum(len(scores) for scores in self.strategy_scores.values())} scores")
        except Exception as e:
            logger.error(f"Error loading performance history: {str(e)}")
    
    def _update_adaptive_timeouts(self):
        """Update adaptive timeouts based on historical data."""
        for key, timeouts in self.timeout_history.items():
            if not timeouts:
                continue
                
            # Use 95th percentile as timeout to cover most cases
            if len(timeouts) > 10:
                # With enough data, use percentile
                timeouts.sort()
                idx = min(int(len(timeouts) * 0.95), len(timeouts) - 1)
                timeout = timeouts[idx]
            else:
                # With limited data, use mean + 2*stddev
                if len(timeouts) > 1:
                    mean = statistics.mean(timeouts)
                    stddev = statistics.stdev(timeouts)
                    timeout = mean + 2 * stddev
                else:
                    timeout = timeouts[0] * 1.5  # Add 50% margin
            
            # Apply reasonable bounds
            timeout = max(1.0, min(timeout, 300.0))  # Between 1 and 300 seconds
            
            self.adaptive_timeouts[key] = timeout
    
    def _error_notification_hook(self, error_report: ErrorReport):
        """
        Hook called by error handler when an error occurs.
        
        Args:
            error_report: The error report
        """
        # This is just a notification - actual recovery is initiated by calling recover()
        logger.info(f"Error notification received: {error_report.error_id} - {error_report.error_type.value}")
    
    async def recover(self, error_report: ErrorReport) -> Tuple[bool, Dict[str, Any]]:
        """
        Recover from an error using performance-based strategy selection.
        
        Args:
            error_report: The error report
            
        Returns:
            Tuple of (success, recovery_info)
        """
        # Start tracking recovery
        recovery_start = time.time()
        
        # Get error type
        error_type = error_report.error_type.value
        
        # Get current recovery level for this error (or start at level 1)
        recovery_level = self.error_recovery_levels.get(error_report.error_id, 1)
        
        # Select best strategy based on historical performance and current level
        strategy, strategy_id = await self._select_best_strategy(error_type, recovery_level)
        
        if not strategy:
            logger.error(f"No recovery strategy available for error type {error_type}")
            return False, {"error": "No recovery strategy available"}
        
        # Get adaptive timeout for this strategy
        timeout = self._get_adaptive_timeout(error_type, strategy_id)
        
        # Prepare recovery info
        recovery_info = {
            "error_id": error_report.error_id,
            "error_type": error_type,
            "strategy_id": strategy_id,
            "strategy_name": strategy.name,
            "recovery_level": recovery_level,
            "started": datetime.now().isoformat(),
            "timeout": timeout,
            "success": False,
            "affected_tasks": 0,
            "task_recovery_success": 0
        }
        
        # Track resource usage before recovery
        resources_before = await self._get_resource_usage()
        
        # Gather affected tasks
        affected_tasks = await self._get_affected_tasks(error_report)
        recovery_info["affected_tasks"] = len(affected_tasks)
        
        try:
            # Execute strategy with timeout
            success = await # TODO: Replace with anyio.fail_after - asyncio.wait_for(
                strategy.execute(self._convert_error_report(error_report)),
                timeout=timeout
            )
            
            # Check task recovery success
            recovered_tasks = await self._check_task_recovery(affected_tasks)
            recovery_info["task_recovery_success"] = recovered_tasks
            
            # Update recovery info
            recovery_info["success"] = success
            recovery_info["completed"] = datetime.now().isoformat()
            recovery_info["execution_time"] = time.time() - recovery_start
            
            # Track resource usage after recovery
            resources_after = await self._get_resource_usage()
            resource_diff = self._calculate_resource_diff(resources_before, resources_after)
            recovery_info["resource_usage"] = resource_diff
            
            # Calculate impact score
            impact_score = self._calculate_impact_score(resource_diff, recovery_info["execution_time"], affected_tasks)
            recovery_info["impact_score"] = impact_score
            
            # Check post-recovery stability
            stability_score = await self._check_stability()
            recovery_info["stability_score"] = stability_score
            
            # Record performance
            self._record_recovery_performance(
                strategy_id=strategy_id,
                strategy_name=strategy.name,
                error_type=error_type,
                execution_time=recovery_info["execution_time"],
                success=success,
                affected_tasks=recovery_info["affected_tasks"],
                task_recovery_success=recovery_info["task_recovery_success"],
                resource_usage=resource_diff,
                impact_score=impact_score,
                stability_score=stability_score,
                context={
                    "error_id": error_report.error_id,
                    "recovery_level": recovery_level,
                    "component": error_report.context.component,
                    "operation": error_report.context.operation
                }
            )
            
            # Update scores
            self._update_strategy_scores(strategy_id, error_type)
            
            # Handle progressive recovery
            if success:
                # Reset recovery level on success
                self.error_recovery_levels.pop(error_report.error_id, None)
            else:
                # Escalate to next level on failure
                next_level = min(recovery_level + 1, ProgressiveRecoveryLevel.LEVEL_5.value)
                self.error_recovery_levels[error_report.error_id] = next_level
                
                # Record progression
                self._record_progressive_recovery(
                    error_id=error_report.error_id,
                    old_level=recovery_level,
                    new_level=next_level,
                    strategy_id=strategy_id,
                    strategy_name=strategy.name,
                    success=success
                )
                
                # If not at max level, retry with escalated strategy
                if next_level <= ProgressiveRecoveryLevel.LEVEL_5.value and next_level > recovery_level:
                    logger.info(f"Progressive recovery: escalating from level {recovery_level} to {next_level} for error {error_report.error_id}")
                    return await self.recover(error_report)
            
            return success, recovery_info
        except asyncio.TimeoutError:
            # Recovery timed out
            logger.error(f"Recovery timed out after {timeout} seconds for error {error_report.error_id}")
            
            execution_time = time.time() - recovery_start
            recovery_info["success"] = False
            recovery_info["completed"] = datetime.now().isoformat()
            recovery_info["execution_time"] = execution_time
            recovery_info["timeout"] = True
            
            # Record timeout failure
            self._record_timeout(error_type, strategy_id, timeout, False)
            
            # Track resource usage after timeout
            resources_after = await self._get_resource_usage()
            resource_diff = self._calculate_resource_diff(resources_before, resources_after)
            recovery_info["resource_usage"] = resource_diff
            
            # Record performance (failure due to timeout)
            self._record_recovery_performance(
                strategy_id=strategy_id,
                strategy_name=strategy.name,
                error_type=error_type,
                execution_time=execution_time,
                success=False,
                affected_tasks=recovery_info["affected_tasks"],
                task_recovery_success=0,  # Assume no tasks recovered on timeout
                resource_usage=resource_diff,
                impact_score=1.0,  # High impact due to timeout
                stability_score=0.0,  # Low stability due to timeout
                context={
                    "error_id": error_report.error_id,
                    "recovery_level": recovery_level,
                    "component": error_report.context.component,
                    "operation": error_report.context.operation,
                    "timeout": True
                }
            )
            
            # Update scores
            self._update_strategy_scores(strategy_id, error_type)
            
            # Escalate to next level
            next_level = min(recovery_level + 1, ProgressiveRecoveryLevel.LEVEL_5.value)
            self.error_recovery_levels[error_report.error_id] = next_level
            
            # Record progression
            self._record_progressive_recovery(
                error_id=error_report.error_id,
                old_level=recovery_level,
                new_level=next_level,
                strategy_id=strategy_id,
                strategy_name=strategy.name,
                success=False,
                details={"timeout": True}
            )
            
            # If not at max level, retry with escalated strategy
            if next_level <= ProgressiveRecoveryLevel.LEVEL_5.value and next_level > recovery_level:
                logger.info(f"Progressive recovery: escalating from level {recovery_level} to {next_level} for error {error_report.error_id} after timeout")
                return await self.recover(error_report)
            
            return False, recovery_info
        except Exception as e:
            # Recovery failed with exception
            logger.error(f"Recovery failed with exception: {str(e)}")
            logger.error(traceback.format_exc())
            
            execution_time = time.time() - recovery_start
            recovery_info["success"] = False
            recovery_info["completed"] = datetime.now().isoformat()
            recovery_info["execution_time"] = execution_time
            recovery_info["error"] = str(e)
            
            # Track resource usage after error
            resources_after = await self._get_resource_usage()
            resource_diff = self._calculate_resource_diff(resources_before, resources_after)
            recovery_info["resource_usage"] = resource_diff
            
            # Record performance (failure due to exception)
            self._record_recovery_performance(
                strategy_id=strategy_id,
                strategy_name=strategy.name,
                error_type=error_type,
                execution_time=execution_time,
                success=False,
                affected_tasks=recovery_info["affected_tasks"],
                task_recovery_success=0,  # Assume no tasks recovered on exception
                resource_usage=resource_diff,
                impact_score=1.0,  # High impact due to exception
                stability_score=0.0,  # Low stability due to exception
                context={
                    "error_id": error_report.error_id,
                    "recovery_level": recovery_level,
                    "component": error_report.context.component,
                    "operation": error_report.context.operation,
                    "exception": str(e)
                }
            )
            
            # Update scores
            self._update_strategy_scores(strategy_id, error_type)
            
            # Escalate to next level
            next_level = min(recovery_level + 1, ProgressiveRecoveryLevel.LEVEL_5.value)
            self.error_recovery_levels[error_report.error_id] = next_level
            
            # Record progression
            self._record_progressive_recovery(
                error_id=error_report.error_id,
                old_level=recovery_level,
                new_level=next_level,
                strategy_id=strategy_id,
                strategy_name=strategy.name,
                success=False,
                details={"exception": str(e)}
            )
            
            # If not at max level, retry with escalated strategy
            if next_level <= ProgressiveRecoveryLevel.LEVEL_5.value and next_level > recovery_level:
                logger.info(f"Progressive recovery: escalating from level {recovery_level} to {next_level} for error {error_report.error_id} after exception")
                return await self.recover(error_report)
            
            return False, recovery_info
    
    async def _select_best_strategy(self, error_type: str, recovery_level: int) -> Tuple[Any, str]:
        """
        Select the best recovery strategy based on historical performance.
        
        Args:
            error_type: The error type
            recovery_level: The current recovery level
            
        Returns:
            Tuple of (strategy, strategy_id)
        """
        if not self.recovery_manager:
            logger.warning("No recovery manager available for strategy selection")
            return None, ""
        
        # Convert error_type to ErrorCategory if possible
        category = error_type
        for ec in ErrorCategory:
            if ec.name == error_type or ec.value == error_type:
                category = ec.value
                break
        
        strategies = {}
        
        # Get available strategies from manager
        for strategy_id, strategy in self.recovery_manager.strategies.items():
            # Filter by recovery level
            if recovery_level == ProgressiveRecoveryLevel.LEVEL_1.value:
                if strategy.level.value == "low":
                    strategies[strategy_id] = strategy
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_2.value:
                if strategy.level.value in ["low", "medium"]:
                    strategies[strategy_id] = strategy
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_3.value:
                if strategy.level.value in ["medium"]:
                    strategies[strategy_id] = strategy
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_4.value:
                if strategy.level.value in ["medium", "high"]:
                    strategies[strategy_id] = strategy
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_5.value:
                if strategy.level.value in ["high", "critical"]:
                    strategies[strategy_id] = strategy
        
        # If no strategies match recovery level, use any available
        if not strategies and self.recovery_manager.strategies:
            strategies = self.recovery_manager.strategies
        
        # If still no strategies, use default retry strategy
        if not strategies:
            logger.warning(f"No strategies found for error type {error_type}, using default retry strategy")
            strategy = RetryStrategy(self.coordinator)
            return strategy, "retry"
        
        # If there's only one strategy, use it
        if len(strategies) == 1:
            strategy_id, strategy = next(iter(strategies.items()))
            return strategy, strategy_id
        
        # Score each strategy
        scored_strategies = []
        
        for strategy_id, strategy in strategies.items():
            # Get score from history
            score = self.strategy_scores.get(error_type, {}).get(strategy_id)
            
            # If no history, use default score based on recovery level match
            if not score:
                base_score = 0.5  # Default score
                
                # Adjust score based on match to recovery level
                if recovery_level == ProgressiveRecoveryLevel.LEVEL_1.value and strategy.level.value == "low":
                    base_score = 0.8
                elif recovery_level == ProgressiveRecoveryLevel.LEVEL_2.value and strategy.level.value == "low":
                    base_score = 0.7
                elif recovery_level == ProgressiveRecoveryLevel.LEVEL_2.value and strategy.level.value == "medium":
                    base_score = 0.8
                elif recovery_level == ProgressiveRecoveryLevel.LEVEL_3.value and strategy.level.value == "medium":
                    base_score = 0.8
                elif recovery_level == ProgressiveRecoveryLevel.LEVEL_4.value and strategy.level.value == "high":
                    base_score = 0.8
                elif recovery_level == ProgressiveRecoveryLevel.LEVEL_5.value and strategy.level.value == "critical":
                    base_score = 0.9
                
                # Create default score
                score = RecoveryStrategyScore(
                    strategy_id=strategy_id,
                    strategy_name=strategy.name,
                    error_type=error_type,
                    success_rate=base_score,
                    average_recovery_time=10.0,
                    resource_efficiency=0.7,
                    impact_score=0.3,
                    stability_score=0.7,
                    task_recovery_rate=0.7,
                    overall_score=base_score,
                    sample_count=0
                )
            
            # Add to scored strategies
            scored_strategies.append((strategy, strategy_id, score.overall_score))
        
        # Sort by score (highest first)
        scored_strategies.sort(key=lambda x: x[2], reverse=True)
        
        # Get best strategy
        if scored_strategies:
            strategy, strategy_id, score = scored_strategies[0]
            logger.info(f"Selected strategy {strategy_id} for error type {error_type} with score {score:.2f}")
            return strategy, strategy_id
        
        # Fallback to default retry strategy
        logger.warning(f"No scored strategies found for error type {error_type}, using default retry strategy")
        strategy = RetryStrategy(self.coordinator)
        return strategy, "retry"
    
    def _get_adaptive_timeout(self, error_type: str, strategy_id: str) -> float:
        """
        Get adaptive timeout for a strategy based on historical performance.
        
        Args:
            error_type: The error type
            strategy_id: The strategy ID
            
        Returns:
            Timeout in seconds
        """
        key = f"{error_type}:{strategy_id}"
        
        # If we have an adaptive timeout, use it
        if key in self.adaptive_timeouts:
            return self.adaptive_timeouts[key]
        
        # Otherwise use default timeout
        return self.default_timeout
    
    def _record_timeout(self, error_type: str, strategy_id: str, timeout: float, success: bool):
        """
        Record timeout for adaptive timeout calculation.
        
        Args:
            error_type: The error type
            strategy_id: The strategy ID
            timeout: The timeout in seconds
            success: Whether the operation succeeded within timeout
        """
        key = f"{error_type}:{strategy_id}"
        self.timeout_history[key].append(timeout)
        
        # Keep history size manageable
        if len(self.timeout_history[key]) > 100:
            self.timeout_history[key] = self.timeout_history[key][-100:]
        
        # Update adaptive timeouts
        self._update_adaptive_timeouts()
        
        # Persist to database
        if self.db_connection:
            try:
                self.db_connection.execute("""
                INSERT INTO recovery_timeouts (
                    error_type, strategy_id, timeout_seconds, timestamp, success
                ) VALUES (?, ?, ?, ?, ?)
                """, (
                    error_type,
                    strategy_id,
                    timeout,
                    datetime.now(),
                    success
                ))
            except Exception as e:
                logger.error(f"Error recording timeout: {str(e)}")
    
    def _record_recovery_performance(
            self,
            strategy_id: str,
            strategy_name: str,
            error_type: str,
            execution_time: float,
            success: bool,
            affected_tasks: int,
            task_recovery_success: int,
            resource_usage: Dict[str, float],
            impact_score: float,
            stability_score: float,
            context: Dict[str, Any] = None
        ):
        """
        Record performance metrics for a recovery strategy.
        
        Args:
            strategy_id: The strategy ID
            strategy_name: The strategy name
            error_type: The error type
            execution_time: Execution time in seconds
            success: Whether recovery was successful
            affected_tasks: Number of affected tasks
            task_recovery_success: Number of successfully recovered tasks
            resource_usage: Resource usage metrics
            impact_score: Impact score (0-1, lower is better)
            stability_score: Stability score (0-1, higher is better)
            context: Additional context
        """
        context = context or {}
        
        # Calculate metrics
        metrics = {
            "success_rate": 1.0 if success else 0.0,
            "recovery_time": execution_time,
            "resource_usage": sum(resource_usage.values()) if resource_usage else 0.0,
            "impact_score": impact_score,
            "stability": stability_score,
            "task_recovery": task_recovery_success / affected_tasks if affected_tasks > 0 else 1.0
        }
        
        # Create record
        record = RecoveryPerformanceRecord(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            error_type=error_type,
            execution_time_seconds=execution_time,
            success=success,
            affected_tasks=affected_tasks,
            task_recovery_success=task_recovery_success,
            resource_usage=resource_usage,
            impact_score=impact_score,
            post_recovery_stability=stability_score,
            metrics=metrics,
            context=context
        )
        
        # Add to history
        self.performance_history[strategy_id].append(record)
        
        # Keep history size manageable
        if len(self.performance_history[strategy_id]) > 100:
            self.performance_history[strategy_id] = self.performance_history[strategy_id][-100:]
        
        # Persist to database
        if self.db_connection:
            try:
                self.db_connection.execute("""
                INSERT INTO recovery_performance (
                    strategy_id, strategy_name, error_type, execution_time_seconds,
                    success, timestamp, affected_tasks, task_recovery_success,
                    resource_usage, impact_score, post_recovery_stability,
                    metrics, context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_id,
                    strategy_name,
                    error_type,
                    execution_time,
                    success,
                    datetime.now(),
                    affected_tasks,
                    task_recovery_success,
                    json.dumps(resource_usage),
                    impact_score,
                    stability_score,
                    json.dumps(metrics),
                    json.dumps(context)
                ))
            except Exception as e:
                logger.error(f"Error recording performance: {str(e)}")
    
    def _update_strategy_scores(self, strategy_id: str, error_type: str):
        """
        Update scores for a strategy based on historical performance.
        
        Args:
            strategy_id: The strategy ID
            error_type: The error type
        """
        # Get relevant performance records
        records = [
            record for record in self.performance_history.get(strategy_id, [])
            if record.error_type == error_type
        ]
        
        if not records:
            return
        
        # Calculate metrics
        success_rate = sum(1.0 if record.success else 0.0 for record in records) / len(records)
        
        # Only use successful recoveries for time calculation
        successful_records = [record for record in records if record.success]
        average_recovery_time = (
            sum(record.execution_time_seconds for record in successful_records) / len(successful_records)
            if successful_records else float('inf')
        )
        
        # Resource efficiency (lower resource usage is better)
        resource_values = []
        for record in records:
            usage_sum = sum(record.resource_usage.values()) if record.resource_usage else 0.0
            resource_values.append(usage_sum)
        
        resource_efficiency = 1.0
        if resource_values:
            max_resource = max(resource_values) if resource_values else 1.0
            if max_resource > 0:
                avg_resource = sum(resource_values) / len(resource_values)
                # Higher score for lower resource usage
                resource_efficiency = 1.0 - (avg_resource / (max_resource * 2))
        
        # Impact score (average)
        impact_score = sum(record.impact_score for record in records) / len(records)
        
        # Stability score (average)
        stability_score = sum(record.post_recovery_stability for record in records) / len(records)
        
        # Task recovery rate
        task_recovery_rates = []
        for record in records:
            if record.affected_tasks > 0:
                rate = record.task_recovery_success / record.affected_tasks
                task_recovery_rates.append(rate)
        
        task_recovery_rate = (
            sum(task_recovery_rates) / len(task_recovery_rates)
            if task_recovery_rates else 1.0
        )
        
        # Calculate overall score
        # Weights for different factors (should sum to 1.0)
        weights = {
            "success_rate": 0.4,
            "recovery_time": 0.15,
            "resource_efficiency": 0.1,
            "impact_score": 0.1,
            "stability_score": 0.1,
            "task_recovery_rate": 0.15
        }
        
        # Normalize recovery time to 0-1 scale (lower is better)
        # Anything over 60 seconds is considered slow, less than 1 second is excellent
        if average_recovery_time == float('inf'):
            recovery_time_score = 0.0
        else:
            recovery_time_score = max(0.0, min(1.0, 1.0 - (average_recovery_time / 60.0)))
        
        # Calculate overall score
        overall_score = (
            weights["success_rate"] * success_rate +
            weights["recovery_time"] * recovery_time_score +
            weights["resource_efficiency"] * resource_efficiency +
            weights["impact_score"] * (1.0 - impact_score) +  # Lower impact is better
            weights["stability_score"] * stability_score +
            weights["task_recovery_rate"] * task_recovery_rate
        )
        
        # Create or update score
        metrics = {
            "success_rate": success_rate,
            "recovery_time": average_recovery_time,
            "recovery_time_score": recovery_time_score,
            "resource_efficiency": resource_efficiency,
            "impact_score": impact_score,
            "stability_score": stability_score,
            "task_recovery_rate": task_recovery_rate
        }
        
        # Get strategy name from records
        strategy_name = records[0].strategy_name if records else "unknown"
        
        score = RecoveryStrategyScore(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            error_type=error_type,
            success_rate=success_rate,
            average_recovery_time=average_recovery_time,
            resource_efficiency=resource_efficiency,
            impact_score=impact_score,
            stability_score=stability_score,
            task_recovery_rate=task_recovery_rate,
            overall_score=overall_score,
            sample_count=len(records),
            last_used=datetime.now(),
            metrics=metrics
        )
        
        # Update score
        self.strategy_scores[error_type][strategy_id] = score
        
        # Persist to database
        if self.db_connection:
            try:
                self.db_connection.execute("""
                INSERT OR REPLACE INTO recovery_strategy_scores (
                    strategy_id, strategy_name, error_type, success_rate,
                    average_recovery_time, resource_efficiency, impact_score,
                    stability_score, task_recovery_rate, overall_score,
                    sample_count, last_used, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_id,
                    strategy_name,
                    error_type,
                    success_rate,
                    average_recovery_time,
                    resource_efficiency,
                    impact_score,
                    stability_score,
                    task_recovery_rate,
                    overall_score,
                    len(records),
                    datetime.now(),
                    json.dumps(metrics)
                ))
            except Exception as e:
                logger.error(f"Error updating strategy scores: {str(e)}")
    
    def _record_progressive_recovery(
            self,
            error_id: str,
            old_level: int,
            new_level: int,
            strategy_id: str,
            strategy_name: str,
            success: bool,
            details: Dict[str, Any] = None
        ):
        """
        Record progressive recovery history.
        
        Args:
            error_id: The error ID
            old_level: The old recovery level
            new_level: The new recovery level
            strategy_id: The strategy ID
            strategy_name: The strategy name
            success: Whether recovery was successful
            details: Additional details
        """
        details = details or {}
        
        # Record in history
        history_entry = {
            "error_id": error_id,
            "old_level": old_level,
            "new_level": new_level,
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "details": details
        }
        
        self.error_recovery_history[error_id].append(history_entry)
        
        # Persist to database
        if self.db_connection:
            try:
                self.db_connection.execute("""
                INSERT INTO progressive_recovery_history (
                    error_id, recovery_level, strategy_id, strategy_name,
                    timestamp, success, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    error_id,
                    old_level,  # Record the level that was used
                    strategy_id,
                    strategy_name,
                    datetime.now(),
                    success,
                    json.dumps({
                        "old_level": old_level,
                        "new_level": new_level,
                        **details
                    })
                ))
            except Exception as e:
                logger.error(f"Error recording progressive recovery: {str(e)}")
    
    async def _get_resource_usage(self) -> Dict[str, float]:
        """
        Get current resource usage.
        
        Returns:
            Dictionary with resource usage metrics
        """
        import psutil
        import os
        
        # Basic resource metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        memory_mb = memory_info.used / (1024 * 1024)
        
        # Process-specific metrics
        process = psutil.Process(os.getpid())
        process_cpu = process.cpu_percent(interval=0.1)
        process_memory_info = process.memory_info()
        process_memory_mb = process_memory_info.rss / (1024 * 1024)
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_percent = disk_usage.percent
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_mb": memory_mb,
            "process_cpu": process_cpu,
            "process_memory_mb": process_memory_mb,
            "disk_percent": disk_percent
        }
    
    def _calculate_resource_diff(self, before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate difference in resource usage.
        
        Args:
            before: Resource usage before recovery
            after: Resource usage after recovery
            
        Returns:
            Dictionary with resource usage differences
        """
        diff = {}
        
        for key in before:
            if key in after:
                diff[key] = after[key] - before[key]
        
        return diff
    
    def _calculate_impact_score(self, resource_diff: Dict[str, float], execution_time: float, affected_tasks: int) -> float:
        """
        Calculate impact score of a recovery operation.
        
        Args:
            resource_diff: Resource usage difference
            execution_time: Execution time in seconds
            affected_tasks: Number of affected tasks
            
        Returns:
            Impact score (0-1, lower is better)
        """
        # Memory impact (percentage of total memory)
        memory_impact = abs(resource_diff.get("memory_percent", 0.0)) / 100.0
        
        # CPU impact
        cpu_impact = abs(resource_diff.get("cpu_percent", 0.0)) / 100.0
        
        # Process memory impact (MB)
        process_memory_impact = abs(resource_diff.get("process_memory_mb", 0.0)) / 1000.0  # Normalize to 0-1 range
        process_memory_impact = min(process_memory_impact, 1.0)  # Cap at 1.0
        
        # Time impact (normalized to 0-1 range)
        time_impact = min(execution_time / 60.0, 1.0)  # Cap at 1.0 (anything over 60s is high impact)
        
        # Task impact (percentage of affected tasks compared to all tasks)
        task_impact = 0.0
        if hasattr(self.coordinator, 'tasks') and self.coordinator.tasks:
            total_tasks = len(self.coordinator.tasks)
            if total_tasks > 0:
                task_impact = affected_tasks / total_tasks
        
        # Calculate overall impact score
        # Weights should sum to 1.0
        weights = {
            "memory_impact": 0.2,
            "cpu_impact": 0.2,
            "process_memory_impact": 0.1,
            "time_impact": 0.3,
            "task_impact": 0.2
        }
        
        impact_score = (
            weights["memory_impact"] * memory_impact +
            weights["cpu_impact"] * cpu_impact +
            weights["process_memory_impact"] * process_memory_impact +
            weights["time_impact"] * time_impact +
            weights["task_impact"] * task_impact
        )
        
        return impact_score
    
    async def _check_stability(self) -> float:
        """
        Check system stability after recovery.
        
        Returns:
            Stability score (0-1, higher is better)
        """
        # Simple implementation - check system is responsive after recovery
        try:
            # Check if coordinator is responsive
            if self.coordinator and hasattr(self.coordinator, 'db'):
                try:
                    # Try a simple query
                    self.coordinator.db.execute("SELECT 1").fetchone()
                    db_responsive = True
                except Exception:
                    db_responsive = False
            else:
                db_responsive = True  # Assume true if no DB to check
            
            # Check if worker connections are active
            worker_connections_active = True
            if self.coordinator and hasattr(self.coordinator, 'worker_connections'):
                total_workers = len(self.coordinator.worker_connections)
                active_workers = sum(1 for ws in self.coordinator.worker_connections.values() if not ws.closed)
                worker_connections_active = active_workers == total_workers if total_workers > 0 else True
            
            # Check tasks are being processed
            tasks_processing = True
            if self.coordinator and hasattr(self.coordinator, 'running_tasks') and hasattr(self.coordinator, 'pending_tasks'):
                # Healthy if either tasks are running or we've processed all pending tasks
                tasks_processing = len(self.coordinator.running_tasks) > 0 or len(self.coordinator.pending_tasks) == 0
            
            # Calculate stability score
            factors = [db_responsive, worker_connections_active, tasks_processing]
            stability_score = sum(1.0 for factor in factors if factor) / len(factors)
            
            return stability_score
        except Exception as e:
            logger.error(f"Error checking stability: {str(e)}")
            return 0.5  # Medium stability if we can't check
    
    async def _get_affected_tasks(self, error_report: ErrorReport) -> Set[str]:
        """
        Get tasks affected by an error.
        
        Args:
            error_report: The error report
            
        Returns:
            Set of affected task IDs
        """
        affected_tasks = set()
        
        # Extract context for better matching
        component = error_report.context.component
        operation = error_report.context.operation
        related_entities = error_report.context.related_entities
        
        # Check if error directly mentions task IDs
        if related_entities and "task_id" in related_entities:
            task_id = related_entities["task_id"]
            if isinstance(task_id, list):
                affected_tasks.update(task_id)
            else:
                affected_tasks.add(task_id)
                
        if related_entities and "task_ids" in related_entities:
            task_ids = related_entities["task_ids"]
            if isinstance(task_ids, list):
                affected_tasks.update(task_ids)
            elif isinstance(task_ids, str):
                affected_tasks.add(task_ids)
        
        # If error is related to a worker, get all tasks assigned to that worker
        if component == "worker" and "worker_id" in related_entities:
            worker_id = related_entities["worker_id"]
            
            if hasattr(self.coordinator, 'running_tasks'):
                for task_id, assigned_worker_id in self.coordinator.running_tasks.items():
                    if assigned_worker_id == worker_id:
                        affected_tasks.add(task_id)
        
        # For database errors, consider all running tasks affected
        if component == "database" and hasattr(self.coordinator, 'running_tasks'):
            affected_tasks.update(self.coordinator.running_tasks.keys())
        
        # For coordination errors, consider all pending and running tasks affected
        if component == "coordinator" and operation in ["schedule", "assign"]:
            if hasattr(self.coordinator, 'running_tasks'):
                affected_tasks.update(self.coordinator.running_tasks.keys())
            
            if hasattr(self.coordinator, 'pending_tasks'):
                affected_tasks.update(self.coordinator.pending_tasks)
        
        return affected_tasks
    
    async def _check_task_recovery(self, affected_tasks: Set[str]) -> int:
        """
        Check how many affected tasks were successfully recovered.
        
        Args:
            affected_tasks: Set of affected task IDs
            
        Returns:
            Number of successfully recovered tasks
        """
        recovered_count = 0
        
        if not hasattr(self.coordinator, 'tasks'):
            return 0
        
        for task_id in affected_tasks:
            if task_id in self.coordinator.tasks:
                task = self.coordinator.tasks[task_id]
                
                # Check if task is now in a good state
                if task.get("status") in ["pending", "running", "completed"]:
                    recovered_count += 1
                
                # Check if task was moved to failed tasks but has been retried
                if task.get("status") == "failed" and task.get("retried_id"):
                    retried_id = task.get("retried_id")
                    if retried_id in self.coordinator.tasks:
                        retried_task = self.coordinator.tasks[retried_id]
                        if retried_task.get("status") in ["pending", "running", "completed"]:
                            recovered_count += 1
        
        return recovered_count
    
    def _convert_error_report(self, error_report: ErrorReport) -> Dict[str, Any]:
        """
        Convert an ErrorReport to a dictionary for the recovery manager.
        
        Args:
            error_report: The error report
            
        Returns:
            Dictionary representation of the error report
        """
        error_info = {
            "type": error_report.error_type.value,
            "message": error_report.message,
            "category": self._map_error_type_to_category(error_report.error_type),
            "timestamp": error_report.context.timestamp.isoformat(),
            "context": {
                "component": error_report.context.component,
                "operation": error_report.context.operation,
                "user_id": error_report.context.user_id,
                "request_id": error_report.context.request_id,
                "environment": error_report.context.environment,
                "related_entities": error_report.context.related_entities,
                "metadata": error_report.context.metadata
            }
        }
        
        return error_info
    
    def _map_error_type_to_category(self, error_type: ErrorType) -> str:
        """
        Map ErrorType to ErrorCategory.
        
        Args:
            error_type: The error type
            
        Returns:
            Error category string
        """
        mapping = {
            ErrorType.NETWORK: ErrorCategory.CONNECTION.value,
            ErrorType.TIMEOUT: ErrorCategory.TIMEOUT.value,
            ErrorType.RESOURCE: ErrorCategory.SYSTEM_RESOURCE.value,
            ErrorType.SYSTEM: ErrorCategory.SYSTEM_RESOURCE.value,
            ErrorType.DATABASE: ErrorCategory.DB_CONNECTION.value,
            ErrorType.DEPENDENCY: ErrorCategory.TASK_ERROR.value,
            ErrorType.VALIDATION: ErrorCategory.TASK_ERROR.value,
            ErrorType.ASSERTION: ErrorCategory.TASK_ERROR.value,
            ErrorType.CONFIGURATION: ErrorCategory.TASK_ERROR.value,
            ErrorType.AUTHENTICATION: ErrorCategory.AUTH_ERROR.value,
            ErrorType.AUTHORIZATION: ErrorCategory.UNAUTHORIZED.value,
            ErrorType.COORDINATION: ErrorCategory.COORDINATOR_ERROR.value,
            ErrorType.SCHEDULING: ErrorCategory.COORDINATOR_ERROR.value,
            ErrorType.STATE: ErrorCategory.STATE_ERROR.value,
            ErrorType.TEST_SETUP: ErrorCategory.TASK_ERROR.value,
            ErrorType.TEST_EXECUTION: ErrorCategory.TASK_ERROR.value,
            ErrorType.TEST_TEARDOWN: ErrorCategory.TASK_ERROR.value,
            ErrorType.TEST_ENVIRONMENT: ErrorCategory.TASK_ERROR.value,
            ErrorType.UNKNOWN: ErrorCategory.UNKNOWN.value
        }
        
        return mapping.get(error_type, ErrorCategory.UNKNOWN.value)
    
    def get_performance_metrics(self, strategy_id: Optional[str] = None, error_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for recovery strategies.
        
        Args:
            strategy_id: Optional strategy ID to filter by
            error_type: Optional error type to filter by
            
        Returns:
            Dictionary with performance metrics
        """
        # Collect stats for each strategy
        strategy_stats = {}
        
        # Filter scores
        filtered_scores = {}
        for et, strategies in self.strategy_scores.items():
            if error_type and et != error_type:
                continue
                
            filtered_scores[et] = {}
            for sid, score in strategies.items():
                if strategy_id and sid != strategy_id:
                    continue
                    
                filtered_scores[et][sid] = score
        
        # Calculate metrics
        for error_type, strategies in filtered_scores.items():
            for strategy_id, score in strategies.items():
                if strategy_id not in strategy_stats:
                    strategy_stats[strategy_id] = {
                        "name": score.strategy_name,
                        "total_samples": 0,
                        "error_types": [],
                        "success_rate": 0.0,
                        "avg_recovery_time": 0.0,
                        "resource_efficiency": 0.0,
                        "impact_score": 0.0,
                        "stability_score": 0.0,
                        "task_recovery_rate": 0.0,
                        "overall_score": 0.0,
                        "by_error_type": {}
                    }
                
                # Update stats
                stats = strategy_stats[strategy_id]
                stats["total_samples"] += score.sample_count
                if error_type not in stats["error_types"]:
                    stats["error_types"].append(error_type)
                
                stats["by_error_type"][error_type] = {
                    "samples": score.sample_count,
                    "success_rate": score.success_rate,
                    "avg_recovery_time": score.average_recovery_time,
                    "resource_efficiency": score.resource_efficiency,
                    "impact_score": score.impact_score,
                    "stability_score": score.stability_score,
                    "task_recovery_rate": score.task_recovery_rate,
                    "overall_score": score.overall_score,
                    "last_used": score.last_used.isoformat() if score.last_used else None,
                    "metrics": score.metrics
                }
        
        # Calculate overall stats for each strategy
        for sid, stats in strategy_stats.items():
            if stats["total_samples"] > 0:
                # Average metrics across error types, weighted by sample count
                total_weighted_success_rate = 0.0
                total_weighted_recovery_time = 0.0
                total_weighted_resource_efficiency = 0.0
                total_weighted_impact_score = 0.0
                total_weighted_stability_score = 0.0
                total_weighted_task_recovery_rate = 0.0
                total_weighted_overall_score = 0.0
                
                for et, et_stats in stats["by_error_type"].items():
                    weight = et_stats["samples"] / stats["total_samples"]
                    total_weighted_success_rate += et_stats["success_rate"] * weight
                    total_weighted_recovery_time += et_stats["avg_recovery_time"] * weight
                    total_weighted_resource_efficiency += et_stats["resource_efficiency"] * weight
                    total_weighted_impact_score += et_stats["impact_score"] * weight
                    total_weighted_stability_score += et_stats["stability_score"] * weight
                    total_weighted_task_recovery_rate += et_stats["task_recovery_rate"] * weight
                    total_weighted_overall_score += et_stats["overall_score"] * weight
                
                stats["success_rate"] = total_weighted_success_rate
                stats["avg_recovery_time"] = total_weighted_recovery_time
                stats["resource_efficiency"] = total_weighted_resource_efficiency
                stats["impact_score"] = total_weighted_impact_score
                stats["stability_score"] = total_weighted_stability_score
                stats["task_recovery_rate"] = total_weighted_task_recovery_rate
                stats["overall_score"] = total_weighted_overall_score
        
        # Top strategies by error type
        top_strategies = {}
        for et, strategies in filtered_scores.items():
            if not strategies:
                continue
                
            best_sid = None
            best_score = -1.0
            
            for sid, score in strategies.items():
                if score.overall_score > best_score:
                    best_score = score.overall_score
                    best_sid = sid
            
            if best_sid:
                top_strategies[et] = {
                    "strategy_id": best_sid,
                    "strategy_name": strategies[best_sid].strategy_name,
                    "score": best_score
                }
        
        return {
            "strategy_stats": strategy_stats,
            "top_strategies": top_strategies,
            "summary": {
                "total_strategies": len(strategy_stats),
                "total_error_types": len(filtered_scores),
                "average_success_rate": sum(s["success_rate"] for s in strategy_stats.values()) / len(strategy_stats) if strategy_stats else 0.0,
                "average_recovery_time": sum(s["avg_recovery_time"] for s in strategy_stats.values()) / len(strategy_stats) if strategy_stats else 0.0
            }
        }

    def get_strategy_recommendations(self, error_type: str) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations for an error type.
        
        Args:
            error_type: The error type
            
        Returns:
            List of recommended strategies sorted by score
        """
        # Get scores for this error type
        scores = list(self.strategy_scores.get(error_type, {}).values())
        
        # Sort by overall score
        scores.sort(key=lambda s: s.overall_score, reverse=True)
        
        # Convert to list of dictionaries
        recommendations = []
        for score in scores:
            recommendations.append({
                "strategy_id": score.strategy_id,
                "strategy_name": score.strategy_name,
                "score": score.overall_score,
                "success_rate": score.success_rate,
                "avg_recovery_time": score.average_recovery_time,
                "sample_count": score.sample_count,
                "last_used": score.last_used.isoformat() if score.last_used else None,
                "metrics": score.metrics
            })
        
        return recommendations
    
    def get_progressive_recovery_history(self, error_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get progressive recovery history.
        
        Args:
            error_id: Optional error ID to filter by
            
        Returns:
            Dictionary with progressive recovery history
        """
        if error_id:
            # Get history for specific error
            history = self.error_recovery_history.get(error_id, [])
            return {
                "error_id": error_id,
                "history": history,
                "current_level": self.error_recovery_levels.get(error_id, 1)
            }
        else:
            # Get summary of all errors
            result = {
                "errors": [],
                "summary": {
                    "level_1_count": 0,
                    "level_2_count": 0,
                    "level_3_count": 0,
                    "level_4_count": 0,
                    "level_5_count": 0,
                    "successful_recoveries": 0,
                    "failed_recoveries": 0
                }
            }
            
            # Process each error
            for error_id, history in self.error_recovery_history.items():
                level = self.error_recovery_levels.get(error_id, 1)
                
                # Count by level
                if level == 1:
                    result["summary"]["level_1_count"] += 1
                elif level == 2:
                    result["summary"]["level_2_count"] += 1
                elif level == 3:
                    result["summary"]["level_3_count"] += 1
                elif level == 4:
                    result["summary"]["level_4_count"] += 1
                elif level == 5:
                    result["summary"]["level_5_count"] += 1
                
                # Check if most recent attempt was successful
                last_success = history[-1]["success"] if history else False
                if last_success:
                    result["summary"]["successful_recoveries"] += 1
                else:
                    result["summary"]["failed_recoveries"] += 1
                
                # Add error details
                result["errors"].append({
                    "error_id": error_id,
                    "current_level": level,
                    "attempts": len(history),
                    "last_attempt_success": last_success,
                    "last_attempt_time": history[-1]["timestamp"] if history else None
                })
            
            return result
    
    def get_adaptive_timeouts(self, error_type: Optional[str] = None, strategy_id: Optional[str] = None) -> Dict[str, float]:
        """
        Get adaptive timeouts for strategies.
        
        Args:
            error_type: Optional error type to filter by
            strategy_id: Optional strategy ID to filter by
            
        Returns:
            Dictionary mapping keys to timeout values
        """
        result = {}
        
        for key, timeout in self.adaptive_timeouts.items():
            # Parse key to get error_type and strategy_id
            parts = key.split(":")
            if len(parts) == 2:
                et, sid = parts
                
                # Apply filters
                if error_type and et != error_type:
                    continue
                
                if strategy_id and sid != strategy_id:
                    continue
                
                result[key] = timeout
        
        return result
    
    async def analyze_recovery_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze recovery performance across all strategies and error types.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with performance analysis
        """
        if not self.db_connection:
            logger.warning("No database connection, cannot analyze performance")
            return {"error": "No database connection"}
        
        try:
            # Load recent performance records
            result = self.db_connection.execute("""
            SELECT 
                strategy_id, strategy_name, error_type, execution_time_seconds,
                success, timestamp, affected_tasks, task_recovery_success,
                impact_score, post_recovery_stability
            FROM recovery_performance
            WHERE timestamp > datetime('now', '-' || ? || ' day')
            """, (days,)).fetchall()
            
            # Process records
            records = []
            for row in result:
                (
                    strategy_id, strategy_name, error_type, execution_time_seconds,
                    success, timestamp, affected_tasks, task_recovery_success,
                    impact_score, post_recovery_stability
                ) = row
                
                records.append({
                    "strategy_id": strategy_id,
                    "strategy_name": strategy_name,
                    "error_type": error_type,
                    "execution_time": execution_time_seconds,
                    "success": success,
                    "timestamp": timestamp,
                    "affected_tasks": affected_tasks,
                    "task_recovery_success": task_recovery_success,
                    "impact_score": impact_score,
                    "stability_score": post_recovery_stability
                })
            
            # Group by error type
            by_error_type = defaultdict(list)
            for record in records:
                by_error_type[record["error_type"]].append(record)
            
            # Group by strategy
            by_strategy = defaultdict(list)
            for record in records:
                by_strategy[record["strategy_id"]].append(record)
            
            # Calculate metrics
            
            # Overall metrics
            overall_metrics = {
                "total_recoveries": len(records),
                "successful_recoveries": sum(1 for r in records if r["success"]),
                "avg_execution_time": sum(r["execution_time"] for r in records) / len(records) if records else 0.0,
                "avg_impact_score": sum(r["impact_score"] for r in records) / len(records) if records else 0.0,
                "avg_stability_score": sum(r["stability_score"] for r in records) / len(records) if records else 0.0,
                "success_rate": sum(1 for r in records if r["success"]) / len(records) if records else 0.0
            }
            
            # Metrics by error type
            error_type_metrics = {}
            for et, et_records in by_error_type.items():
                error_type_metrics[et] = {
                    "total_recoveries": len(et_records),
                    "successful_recoveries": sum(1 for r in et_records if r["success"]),
                    "avg_execution_time": sum(r["execution_time"] for r in et_records) / len(et_records) if et_records else 0.0,
                    "avg_impact_score": sum(r["impact_score"] for r in et_records) / len(et_records) if et_records else 0.0,
                    "avg_stability_score": sum(r["stability_score"] for r in et_records) / len(et_records) if et_records else 0.0,
                    "success_rate": sum(1 for r in et_records if r["success"]) / len(et_records) if et_records else 0.0
                }
            
            # Metrics by strategy
            strategy_metrics = {}
            for sid, s_records in by_strategy.items():
                strategy_name = s_records[0]["strategy_name"] if s_records else "unknown"
                strategy_metrics[sid] = {
                    "name": strategy_name,
                    "total_recoveries": len(s_records),
                    "successful_recoveries": sum(1 for r in s_records if r["success"]),
                    "avg_execution_time": sum(r["execution_time"] for r in s_records) / len(s_records) if s_records else 0.0,
                    "avg_impact_score": sum(r["impact_score"] for r in s_records) / len(s_records) if s_records else 0.0,
                    "avg_stability_score": sum(r["stability_score"] for r in s_records) / len(s_records) if s_records else 0.0,
                    "success_rate": sum(1 for r in s_records if r["success"]) / len(s_records) if s_records else 0.0
                }
            
            # Time-based analysis
            from collections import OrderedDict
            import datetime as dt
            
            # Group by day
            days_map = OrderedDict()
            start_date = dt.datetime.now() - dt.timedelta(days=days)
            for i in range(days):
                day = start_date + dt.timedelta(days=i)
                day_str = day.strftime("%Y-%m-%d")
                days_map[day_str] = {
                    "date": day_str,
                    "total_recoveries": 0,
                    "successful_recoveries": 0,
                    "avg_execution_time": 0.0,
                    "success_rate": 0.0
                }
            
            # Fill in data
            for record in records:
                timestamp = record["timestamp"]
                if isinstance(timestamp, str):
                    # Parse timestamp string if necessary
                    date_obj = dt.datetime.fromisoformat(timestamp.split(" ")[0])
                elif isinstance(timestamp, dt.datetime):
                    date_obj = timestamp
                else:
                    continue
                
                day_str = date_obj.strftime("%Y-%m-%d")
                if day_str in days_map:
                    days_map[day_str]["total_recoveries"] += 1
                    if record["success"]:
                        days_map[day_str]["successful_recoveries"] += 1
                        
                    # Update average execution time
                    current_avg = days_map[day_str]["avg_execution_time"]
                    current_count = days_map[day_str]["total_recoveries"]
                    if current_count > 1:
                        days_map[day_str]["avg_execution_time"] = (current_avg * (current_count - 1) + record["execution_time"]) / current_count
                    else:
                        days_map[day_str]["avg_execution_time"] = record["execution_time"]
                    
                    # Update success rate
                    days_map[day_str]["success_rate"] = days_map[day_str]["successful_recoveries"] / days_map[day_str]["total_recoveries"]
            
            # Convert to list
            time_series = list(days_map.values())
            
            # Prune days with no data
            time_series = [day for day in time_series if day["total_recoveries"] > 0]
            
            return {
                "overall": overall_metrics,
                "by_error_type": error_type_metrics,
                "by_strategy": strategy_metrics,
                "time_series": time_series,
                "record_count": len(records),
                "period_days": days
            }
        except Exception as e:
            logger.error(f"Error analyzing recovery performance: {str(e)}")
            return {"error": str(e)}


# Demo and testing code
if __name__ == "__main__":
    import duckdb
    import asyncio
    
    async def run_demo():
        print("Error Recovery with Performance Tracking Demo")
        
        # Create error handler
        error_handler = DistributedErrorHandler()
        
        # Create recovery manager
        recovery_manager = EnhancedErrorRecoveryManager(None)
        
        # Create demo coordinator
        class DemoCoordinator:
            def __init__(self):
                self.worker_connections = {}
                self.tasks = {}
                self.running_tasks = {}
                self.pending_tasks = set()
                self.db = duckdb.connect(":memory:")
        
        coordinator = DemoCoordinator()
        
        # Create recovery system
        recovery_system = ErrorRecoveryWithPerformance(
            error_handler=error_handler,
            recovery_manager=recovery_manager,
            coordinator=coordinator,
            db_connection=coordinator.db
        )
        
        # Test recovery
        print("Creating test error...")
        error_report = error_handler.create_error_report(
            Exception("Test error"),
            {
                "component": "test",
                "operation": "demo",
                "critical": False
            }
        )
        
        print(f"Recovering from error {error_report.error_id}...")
        success, info = await recovery_system.recover(error_report)
        
        print(f"Recovery {'successful' if success else 'failed'}")
        print(f"Recovery info: {json.dumps(info, indent=2)}")
        
        # Get metrics
        print("\nPerformance metrics:")
        metrics = recovery_system.get_performance_metrics()
        print(json.dumps(metrics, indent=2))

    anyio.run(run_demo())