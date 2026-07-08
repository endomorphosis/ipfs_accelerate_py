#!/usr/bin/env python3
"""
Performance-Based Error Recovery for Distributed Testing Framework

This module enhances the error recovery system with performance tracking, hardware awareness,
and adaptive recovery strategies. It integrates with the existing error handling system and
provides optimized recovery strategies based on historical performance data.

Key features:
- Performance history tracking for recovery strategies
- Adaptive timeouts based on performance patterns
- Progressive recovery with escalation for persistent errors
- Hardware-aware recovery strategy selection
- Recovery analytics and performance visualization
- Integration with the distributed testing database

Usage:
    Import this module in coordinator.py to enhance error recovery capabilities.
"""

import anyio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union
import traceback
import copy
import statistics
from collections import defaultdict, deque
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("performance_recovery")

# Import existing error handler and strategies
# Prefer package-relative imports so this module works when imported as
# `distributed_testing.*` (pytest/anyio), while keeping a fallback for
# script-style execution.
try:
    from .distributed_error_handler import DistributedErrorHandler, ErrorReport
    from .error_recovery_strategies import (
        ErrorCategory,
        RecoveryStrategy,
        EnhancedErrorRecoveryManager,
        RetryStrategy,
        WorkerRecoveryStrategy,
        DatabaseRecoveryStrategy,
        CoordinatorRecoveryStrategy,
        SystemRecoveryStrategy,
    )
except Exception:  # pragma: no cover
    from distributed_error_handler import DistributedErrorHandler, ErrorReport
    from error_recovery_strategies import (
        ErrorCategory,
        RecoveryStrategy,
        EnhancedErrorRecoveryManager,
        RetryStrategy,
        WorkerRecoveryStrategy,
        DatabaseRecoveryStrategy,
        CoordinatorRecoveryStrategy,
        SystemRecoveryStrategy,
    )

class RecoveryPerformanceMetric(Enum):
    """Performance metrics for recovery strategies."""
    SUCCESS_RATE = "success_rate"         # Success rate of recovery strategy
    RECOVERY_TIME = "recovery_time"       # Time taken for recovery
    RESOURCE_USAGE = "resource_usage"     # Resources used during recovery
    IMPACT_SCORE = "impact_score"         # Impact on system during recovery
    STABILITY = "stability"               # Post-recovery stability
    TASK_RECOVERY = "task_recovery"       # Success rate of task recovery


class ProgressiveRecoveryLevel(Enum):
    """Levels for progressive recovery escalation."""
    LEVEL_1 = 1  # Basic retry with minimal impact
    LEVEL_2 = 2  # Enhanced retry with extended parameters
    LEVEL_3 = 3  # Component restart/reset
    LEVEL_4 = 4  # System-component recovery
    LEVEL_5 = 5  # Full system recovery


class RecoveryPerformanceRecord:
    """Performance record for a recovery strategy."""
    
    def __init__(
        self,
        strategy_id: str,
        strategy_name: str,
        error_type: str,
        execution_time: float,
        success: bool,
        hardware_id: Optional[str] = None,
        affected_tasks: int = 0,
        recovered_tasks: int = 0
    ):
        """Initialize a new performance record."""
        self.strategy_id = strategy_id
        self.strategy_name = strategy_name
        self.error_type = error_type
        self.execution_time = execution_time
        self.success = success
        self.timestamp = datetime.now()
        self.hardware_id = hardware_id
        self.affected_tasks = affected_tasks
        self.recovered_tasks = recovered_tasks
        self.resource_usage = {}
        self.impact_score = 0.0
        self.stability_score = 0.0
        self.context = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "error_type": self.error_type,
            "execution_time": self.execution_time,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "hardware_id": self.hardware_id,
            "affected_tasks": self.affected_tasks,
            "recovered_tasks": self.recovered_tasks,
            "resource_usage": self.resource_usage,
            "impact_score": self.impact_score,
            "stability_score": self.stability_score,
            "context": self.context
        }


class PerformanceBasedErrorRecovery:
    """
    Performance-based error recovery system for the distributed testing framework.
    
    This class enhances the error recovery with:
    - Performance history tracking
    - Adaptive strategy selection
    - Progressive recovery
    - Hardware-aware recovery
    """
    
    def __init__(
        self,
        error_handler: DistributedErrorHandler,
        recovery_manager: EnhancedErrorRecoveryManager,
        coordinator=None,
        db_connection=None
    ):
        """
        Initialize the performance-based error recovery system.
        
        Args:
            error_handler: The distributed error handler
            recovery_manager: The enhanced recovery manager
            coordinator: The coordinator instance
            db_connection: Database connection for storing performance data
        """
        self.error_handler = error_handler
        self.recovery_manager = recovery_manager
        self.coordinator = coordinator
        self.db_connection = db_connection
        
        # Performance history
        self.performance_history: Dict[str, List[RecoveryPerformanceRecord]] = defaultdict(list)
        
        # Strategy scores - {error_type: {strategy_id: score}}
        self.strategy_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Adaptive timeouts
        self.adaptive_timeouts: Dict[str, float] = {}
        self.default_timeout = 30.0  # Default timeout in seconds
        
        # Progressive recovery tracking - {error_id: level}
        self.error_recovery_levels: Dict[str, int] = {}
        
        # Recovery history
        self.recovery_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Initialize database tables
        if self.db_connection:
            self._create_performance_tables()
            self._load_performance_history()
        
        # Register as error hook
        if self.error_handler and hasattr(self.error_handler, 'register_error_hook'):
            self.error_handler.register_error_hook("*", self._error_notification_hook)
        
        logger.info("Performance-based error recovery system initialized")
    
    def _create_performance_tables(self):
        """Create database tables for performance tracking."""
        try:
            # Check schema version
            self._check_and_upgrade_schema()
            
            # Performance history table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS recovery_performance (
                id INTEGER PRIMARY KEY,
                strategy_id VARCHAR,
                strategy_name VARCHAR,
                error_type VARCHAR,
                execution_time FLOAT,
                success BOOLEAN,
                timestamp TIMESTAMP,
                hardware_id VARCHAR,
                affected_tasks INTEGER,
                recovered_tasks INTEGER,
                resource_usage JSON,
                impact_score FLOAT,
                stability_score FLOAT,
                context JSON
            )
            """)
            
            # Strategy scores table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS strategy_scores (
                error_type VARCHAR,
                strategy_id VARCHAR,
                score FLOAT,
                last_updated TIMESTAMP,
                samples INTEGER,
                metrics JSON,
                PRIMARY KEY (error_type, strategy_id)
            )
            """)
            
            # Adaptive timeouts table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS adaptive_timeouts (
                error_type VARCHAR,
                strategy_id VARCHAR,
                timeout FLOAT,
                last_updated TIMESTAMP,
                PRIMARY KEY (error_type, strategy_id)
            )
            """)
            
            # Progressive recovery table
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS progressive_recovery (
                error_id VARCHAR PRIMARY KEY,
                current_level INTEGER,
                last_updated TIMESTAMP,
                history JSON
            )
            """)
            
            # Schema version tracking
            self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS schema_versions (
                component VARCHAR PRIMARY KEY, 
                version INTEGER,
                last_updated TIMESTAMP
            )
            """)
            
            # Set schema version if not exists
            self.db_connection.execute("""
            INSERT OR IGNORE INTO schema_versions (component, version, last_updated)
            VALUES ('performance_recovery', 1, CURRENT_TIMESTAMP)
            """)
            
            logger.info("Performance tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
    
    def _check_and_upgrade_schema(self):
        """Check schema version and perform upgrades if necessary."""
        try:
            # Check if schema_versions table exists
            result = self.db_connection.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='schema_versions'
            """).fetchone()
            
            if not result:
                # First time setup, no upgrades needed
                return
            
            # Get current schema version
            result = self.db_connection.execute("""
            SELECT version FROM schema_versions 
            WHERE component='performance_recovery'
            """).fetchone()
            
            current_version = result[0] if result else 0
            
            # Apply upgrades based on version
            if current_version < 1:
                # Upgrade to version 1 - Fix the recovery_performance table
                logger.info("Upgrading performance recovery schema to version 1")
                
                # Check if recovery_performance table exists
                result = self.db_connection.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='recovery_performance'
                """).fetchone()
                
                if result:
                    # Table exists, check if it has autoincrement
                    try:
                        # Check what columns exist in the current table - DuckDB compatible version
                        table_info = self.db_connection.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name='recovery_performance' AND table_schema='main'
                        """).fetchall()
                        
                        column_names = [col[0].lower() for col in table_info]
                        
                        # Attempt to create a temporary table with correct schema
                        self.db_connection.execute("""
                        CREATE TEMP TABLE recovery_performance_new (
                            id INTEGER PRIMARY KEY,
                            strategy_id VARCHAR,
                            strategy_name VARCHAR,
                            error_type VARCHAR,
                            execution_time FLOAT,
                            success BOOLEAN,
                            timestamp TIMESTAMP,
                            hardware_id VARCHAR,
                            affected_tasks INTEGER,
                            recovered_tasks INTEGER,
                            resource_usage JSON,
                            impact_score FLOAT,
                            stability_score FLOAT,
                            context JSON
                        )
                        """)
                        
                        # Build dynamic SQL for migration based on existing columns
                        select_clause = []
                        for col in ["id", "strategy_id", "strategy_name", "error_type", "execution_time", "success", "timestamp"]:
                            if col.lower() in column_names:
                                select_clause.append(col)
                                
                        # Add default values for missing columns
                        if "hardware_id" not in column_names:
                            select_clause.append("NULL as hardware_id")
                        else:
                            select_clause.append("hardware_id")
                            
                        if "affected_tasks" not in column_names:
                            select_clause.append("0 as affected_tasks")
                        else:
                            select_clause.append("affected_tasks")
                            
                        if "recovered_tasks" not in column_names:
                            select_clause.append("0 as recovered_tasks")
                        else:
                            select_clause.append("recovered_tasks")
                            
                        if "resource_usage" not in column_names:
                            select_clause.append("'{}' as resource_usage")
                        else:
                            select_clause.append("resource_usage")
                            
                        if "impact_score" not in column_names:
                            select_clause.append("0.0 as impact_score")
                        else:
                            select_clause.append("impact_score")
                            
                        if "stability_score" not in column_names:
                            select_clause.append("1.0 as stability_score")
                        else:
                            select_clause.append("stability_score")
                            
                        if "context" not in column_names:
                            select_clause.append("'{}' as context")
                        else:
                            select_clause.append("context")
                        
                        # Copy data using dynamic SQL
                        copy_sql = f"""
                        INSERT INTO recovery_performance_new 
                            (id, strategy_id, strategy_name, error_type, execution_time,
                            success, timestamp, hardware_id, affected_tasks,
                            recovered_tasks, resource_usage, impact_score,
                            stability_score, context)
                        SELECT 
                            {', '.join(select_clause)}
                        FROM recovery_performance
                        """
                        
                        self.db_connection.execute(copy_sql)
                        
                        # Drop old table
                        self.db_connection.execute("DROP TABLE recovery_performance")
                        
                        # Rename new table
                        self.db_connection.execute("ALTER TABLE recovery_performance_new RENAME TO recovery_performance")
                        
                        logger.info("Successfully migrated recovery_performance table to use AUTOINCREMENT")
                    except Exception as e:
                        logger.error(f"Error upgrading recovery_performance table: {str(e)}")
                
                # Update schema version
                self.db_connection.execute("""
                INSERT OR REPLACE INTO schema_versions (component, version, last_updated)
                VALUES ('performance_recovery', 1, CURRENT_TIMESTAMP)
                """)
            
        except Exception as e:
            logger.error(f"Error checking or upgrading schema: {str(e)}")
    
    def _load_performance_history(self):
        """Load performance history from database."""
        if not self.db_connection:
            return
        
        try:
            # Load performance records
            # Check if the table has all columns first
            table_info = self.db_connection.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='recovery_performance'
            """).fetchall()
            
            column_names = [col[0] for col in table_info]
            
            # Build query based on available columns
            basic_columns = ["strategy_id", "strategy_name", "error_type", "execution_time", "success"]
            optional_columns = ["timestamp", "hardware_id", "affected_tasks", "recovered_tasks", 
                              "resource_usage", "impact_score", "stability_score", "context"]
            
            select_columns = []
            for col in basic_columns + optional_columns:
                if col.upper() in [c.upper() for c in column_names]:
                    select_columns.append(col)
                elif col in ["resource_usage", "impact_score", "stability_score", "context"]:
                    # Add default values for optional JSON columns
                    select_columns.append(f"NULL as {col}")
                else:
                    # Add default values for other optional columns
                    select_columns.append(f"NULL as {col}")
            
            # Build the query
            query = f"""
            SELECT {', '.join(select_columns)}
            FROM recovery_performance
            """
            
            # Add timestamp filter if timestamp column exists
            if "TIMESTAMP" in [c.upper() for c in column_names]:
                query += " WHERE timestamp > (CURRENT_TIMESTAMP - INTERVAL '30 days')"
            
            result = self.db_connection.execute(query).fetchall()
            
            for row in result:
                (
                    strategy_id, strategy_name, error_type, execution_time,
                    success, timestamp, hardware_id, affected_tasks,
                    recovered_tasks, resource_usage, impact_score,
                    stability_score, context
                ) = row
                
                record = RecoveryPerformanceRecord(
                    strategy_id=strategy_id,
                    strategy_name=strategy_name,
                    error_type=error_type,
                    execution_time=execution_time,
                    success=success,
                    hardware_id=hardware_id,
                    affected_tasks=affected_tasks,
                    recovered_tasks=recovered_tasks
                )
                
                # Set additional fields
                record.timestamp = timestamp
                record.resource_usage = json.loads(resource_usage) if resource_usage else {}
                record.impact_score = impact_score
                record.stability_score = stability_score
                record.context = json.loads(context) if context else {}
                
                # Add to memory
                self.performance_history[strategy_id].append(record)
            
            # Load strategy scores
            result = self.db_connection.execute("""
            SELECT error_type, strategy_id, score, samples, metrics
            FROM strategy_scores
            """).fetchall()
            
            for row in result:
                error_type, strategy_id, score, samples, metrics = row
                self.strategy_scores[error_type][strategy_id] = score
            
            # Load adaptive timeouts
            result = self.db_connection.execute("""
            SELECT error_type, strategy_id, timeout
            FROM adaptive_timeouts
            """).fetchall()
            
            for row in result:
                error_type, strategy_id, timeout = row
                key = f"{error_type}:{strategy_id}"
                self.adaptive_timeouts[key] = timeout
            
            # Load progressive recovery levels
            result = self.db_connection.execute("""
            SELECT error_id, current_level, history
            FROM progressive_recovery
            """).fetchall()
            
            for row in result:
                error_id, current_level, history = row
                self.error_recovery_levels[error_id] = current_level
                self.recovery_history[error_id] = json.loads(history) if history else []
            
            logger.info(f"Loaded {sum(len(records) for records in self.performance_history.values())} performance records")
        except Exception as e:
            logger.error(f"Error loading performance history: {str(e)}")
    
    def _error_notification_hook(self, error_report: Any):
        """
        Hook called when an error is registered with the error handler.
        
        Args:
            error_report: The error report object
        """
        # Just log the notification - recovery will be initiated separately
        logger.debug(f"Error notification received: {getattr(error_report, 'error_id', 'unknown')}")
    
    async def recover(self, error_report: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Recover from an error using performance-based strategy selection.
        
        Args:
            error_report: The error report
            
        Returns:
            Tuple of (success, recovery_info)
        """
        error_id = getattr(error_report, 'error_id', str(uuid.uuid4()))
        error_type = self._get_error_type(error_report)
        
        # Start tracking performance
        start_time = time.time()
        
        # Get current recovery level (or start at level 1)
        recovery_level = self.error_recovery_levels.get(error_id, 1)
        
        error_dict = self._convert_error_report(error_report)

        # Select best strategy based on performance history
        strategy, strategy_id = await self._select_best_strategy(error_type, recovery_level, error_dict)
        
        if not strategy:
            logger.error(f"No recovery strategy available for error type: {error_type}")
            return False, {"error": "No recovery strategy available"}
        
        # Get adaptive timeout
        timeout = self._get_adaptive_timeout(error_type, strategy_id)
        
        # Prepare recovery info
        recovery_info = {
            "error_id": error_id,
            "error_type": error_type,
            "strategy_id": strategy_id,
            "strategy_name": strategy.name,
            "recovery_level": recovery_level,
            "timeout": timeout,
            "start_time": datetime.now().isoformat()
        }
        
        # Get affected tasks
        affected_tasks = await self._get_affected_tasks(error_report)
        recovery_info["affected_tasks"] = len(affected_tasks)
        
        # Get resource usage before recovery
        resources_before = self._get_resource_usage()
        
        try:
            # Execute strategy with timeout
            with anyio.fail_after(timeout):
                success = await strategy.execute(error_dict)
            
            # Record execution time
            execution_time = time.time() - start_time
            
            # Get resource usage after recovery
            resources_after = self._get_resource_usage()
            resource_diff = self._calculate_resource_diff(resources_before, resources_after)
            
            # Check stability
            stability_score = await self._check_stability()
            
            # Check task recovery
            recovered_tasks = await self._check_task_recovery(affected_tasks)
            
            # Calculate impact score
            impact_score = self._calculate_impact_score(
                resource_diff, 
                execution_time,
                recovery_info["affected_tasks"],
                recovered_tasks
            )
            
            # Update recovery info
            recovery_info.update({
                "success": success,
                "execution_time": execution_time,
                "end_time": datetime.now().isoformat(),
                "resource_usage": resource_diff,
                "stability_score": stability_score,
                "recovered_tasks": recovered_tasks,
                "impact_score": impact_score
            })
            
            # Record performance
            self._record_performance(
                strategy_id=strategy_id,
                strategy_name=strategy.name,
                error_type=error_type,
                execution_time=execution_time,
                success=success,
                affected_tasks=len(affected_tasks),
                recovered_tasks=recovered_tasks,
                resource_usage=resource_diff,
                impact_score=impact_score,
                stability_score=stability_score,
                context={
                    "error_id": error_id,
                    "component": getattr(error_report, 'component', 'unknown'),
                    "operation": getattr(error_report, 'operation', 'unknown'),
                    "recovery_level": recovery_level
                }
            )
            
            # Update strategy scores
            self._update_strategy_scores(strategy_id, error_type)
            
            # Handle progressive recovery
            if success:
                # Reset recovery level on success
                if error_id in self.error_recovery_levels:
                    del self.error_recovery_levels[error_id]
            else:
                # Escalate to next level
                next_level = min(recovery_level + 1, ProgressiveRecoveryLevel.LEVEL_5.value)
                self.error_recovery_levels[error_id] = next_level
                
                # Record progression only when the level actually changes
                if next_level != recovery_level:
                    self._record_progression(
                        error_id=error_id,
                        old_level=recovery_level,
                        new_level=next_level,
                        strategy_id=strategy_id,
                        success=success
                    )
                
                # If not at max level, retry with escalated strategy
                if next_level <= ProgressiveRecoveryLevel.LEVEL_5.value and next_level > recovery_level:
                    logger.info(f"Progressive recovery: escalating from level {recovery_level} to {next_level}")
                    return await self.recover(error_report)
            
            return success, recovery_info
        
        except TimeoutError:
            # Recovery timed out
            execution_time = time.time() - start_time
            
            # Get resource usage after timeout
            resources_after = self._get_resource_usage()
            resource_diff = self._calculate_resource_diff(resources_before, resources_after)
            
            # Update recovery info
            recovery_info.update({
                "success": False,
                "execution_time": execution_time,
                "end_time": datetime.now().isoformat(),
                "timeout": True,
                "resource_usage": resource_diff,
                "recovered_tasks": 0
            })
            
            # Record timeout in performance history
            self._record_performance(
                strategy_id=strategy_id,
                strategy_name=strategy.name,
                error_type=error_type,
                execution_time=execution_time,
                success=False,
                affected_tasks=len(affected_tasks),
                recovered_tasks=0,
                resource_usage=resource_diff,
                impact_score=1.0,  # Highest impact score for timeouts
                stability_score=0.0,  # Lowest stability for timeouts
                context={
                    "error_id": error_id,
                    "recovery_level": recovery_level,
                    "timeout": True
                }
            )
            
            # Update adaptive timeout
            self._update_adaptive_timeout(error_type, strategy_id, timeout, False)
            
            # Update strategy scores
            self._update_strategy_scores(strategy_id, error_type)
            
            # Escalate to next level
            next_level = min(recovery_level + 1, ProgressiveRecoveryLevel.LEVEL_5.value)
            self.error_recovery_levels[error_id] = next_level
            
            # Record progression only when the level actually changes
            if next_level != recovery_level:
                self._record_progression(
                    error_id=error_id,
                    old_level=recovery_level,
                    new_level=next_level,
                    strategy_id=strategy_id,
                    success=False,
                    details={"timeout": True}
                )
            
            # If not at max level, retry with escalated strategy
            if next_level <= ProgressiveRecoveryLevel.LEVEL_5.value and next_level > recovery_level:
                logger.info(f"Progressive recovery after timeout: escalating from level {recovery_level} to {next_level}")
                return await self.recover(error_report)
            
            return False, recovery_info
        
        except Exception as e:
            # Recovery failed with exception
            logger.error(f"Recovery failed with exception: {str(e)}")
            logger.error(traceback.format_exc())
            
            execution_time = time.time() - start_time
            
            # Get resource usage after exception
            resources_after = self._get_resource_usage()
            resource_diff = self._calculate_resource_diff(resources_before, resources_after)
            
            # Update recovery info
            recovery_info.update({
                "success": False,
                "execution_time": execution_time,
                "end_time": datetime.now().isoformat(),
                "error": str(e),
                "resource_usage": resource_diff,
                "recovered_tasks": 0
            })
            
            # Record failure in performance history
            self._record_performance(
                strategy_id=strategy_id,
                strategy_name=strategy.name,
                error_type=error_type,
                execution_time=execution_time,
                success=False,
                affected_tasks=len(affected_tasks),
                recovered_tasks=0,
                resource_usage=resource_diff,
                impact_score=1.0,  # Highest impact score for exceptions
                stability_score=0.0,  # Lowest stability for exceptions
                context={
                    "error_id": error_id,
                    "recovery_level": recovery_level,
                    "exception": str(e)
                }
            )
            
            # Update strategy scores
            self._update_strategy_scores(strategy_id, error_type)
            
            # Escalate to next level
            next_level = min(recovery_level + 1, ProgressiveRecoveryLevel.LEVEL_5.value)
            self.error_recovery_levels[error_id] = next_level
            
            # Record progression only when the level actually changes
            if next_level != recovery_level:
                self._record_progression(
                    error_id=error_id,
                    old_level=recovery_level,
                    new_level=next_level,
                    strategy_id=strategy_id,
                    success=False,
                    details={"exception": str(e)}
                )
            
            # If not at max level, retry with escalated strategy
            if next_level <= ProgressiveRecoveryLevel.LEVEL_5.value and next_level > recovery_level:
                logger.info(f"Progressive recovery after exception: escalating from level {recovery_level} to {next_level}")
                return await self.recover(error_report)
            
            return False, recovery_info
    
    async def _select_best_strategy(
        self,
        error_type: str,
        recovery_level: int,
        error_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, str]:
        """
        Select the best recovery strategy based on historical performance and level.
        
        Args:
            error_type: The error type
            recovery_level: Current recovery level
            
        Returns:
            Tuple of (strategy, strategy_id)
        """
        error_info = error_info or {}

        if not self.recovery_manager or not hasattr(self.recovery_manager, 'strategies'):
            logger.warning("No recovery manager available")
            return None, ""
        
        # Filter strategies by recovery level and applicability
        candidates = {}
        for strategy_id, strategy in self.recovery_manager.strategies.items():
            level_value = getattr(strategy.level, 'value', strategy.level)
            if hasattr(strategy, "is_applicable") and not strategy.is_applicable(error_info):
                continue
            
            # Match strategies based on recovery level
            if recovery_level == ProgressiveRecoveryLevel.LEVEL_1.value:
                if level_value == "low":
                    candidates[strategy_id] = strategy
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_2.value:
                if level_value in ["low", "medium"]:
                    candidates[strategy_id] = strategy
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_3.value:
                if level_value == "medium":
                    candidates[strategy_id] = strategy
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_4.value:
                if level_value in ["medium", "high"]:
                    candidates[strategy_id] = strategy
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_5.value:
                if level_value in ["high", "critical"]:
                    candidates[strategy_id] = strategy
        
        # If no candidates match level criteria, use all strategies
        if not candidates:
            candidates = {
                strategy_id: strategy
                for strategy_id, strategy in self.recovery_manager.strategies.items()
                if not hasattr(strategy, "is_applicable") or strategy.is_applicable(error_info)
            }
        
        # If still no candidates, use default retry strategy
        if not candidates:
            logger.warning(f"No strategies found for error type {error_type}, using default retry")
            default_strategy = RetryStrategy(self.coordinator)
            return default_strategy, "retry"
        
        # If only one candidate, use it
        if len(candidates) == 1:
            strategy_id, strategy = next(iter(candidates.items()))
            return strategy, strategy_id
        
        # Score candidates based on historical performance
        scored_candidates = []
        for strategy_id, strategy in candidates.items():
            # Get historical score
            score = self.strategy_scores.get(error_type, {}).get(strategy_id, 0.5)
            
            # Add strategy level match bonus
            level_value = getattr(strategy.level, 'value', strategy.level)
            level_match_bonus = 0.0
            
            if recovery_level == ProgressiveRecoveryLevel.LEVEL_1.value and level_value == "low":
                level_match_bonus = 0.2
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_2.value and level_value == "medium":
                level_match_bonus = 0.2
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_3.value and level_value == "medium":
                level_match_bonus = 0.2
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_4.value and level_value == "high":
                level_match_bonus = 0.2
            elif recovery_level == ProgressiveRecoveryLevel.LEVEL_5.value and level_value == "critical":
                level_match_bonus = 0.3
            
            # Apply bonus for appropriate level match
            adjusted_score = min(score + level_match_bonus, 1.0)
            
            scored_candidates.append((strategy, strategy_id, adjusted_score))
        
        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Use highest scoring strategy
        strategy, strategy_id, score = scored_candidates[0]
        logger.info(f"Selected strategy {strategy_id} with score {score:.2f} for error type {error_type}")
        
        return strategy, strategy_id
    
    def _get_error_type(self, error_report: Any) -> str:
        """Get error type from error report."""
        # Handle different error report formats
        if hasattr(error_report, 'error_type'):
            # Check if it's an enum or string
            error_type = error_report.error_type
            if hasattr(error_type, 'value'):
                return error_type.value
            return str(error_type)
        
        # Try other possible attributes
        for attr in ['type', 'category', 'code']:
            if hasattr(error_report, attr):
                value = getattr(error_report, attr)
                if hasattr(value, 'value'):
                    return value.value
                return str(value)
        
        # Default to unknown
        return "unknown"
    
    def _convert_error_report(self, error_report: Any) -> Dict[str, Any]:
        """Convert error report to dictionary format for recovery strategies."""
        # Create error info dictionary
        error_info = {
            "type": self._get_error_type(error_report),
            "message": getattr(error_report, 'message', str(error_report)),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add context if available
        if hasattr(error_report, 'context'):
            context = error_report.context
            error_info["context"] = {
                "component": getattr(context, 'component', 'unknown'),
                "operation": getattr(context, 'operation', 'unknown')
            }
            
            # Add additional context fields if available
            for field in ['user_id', 'request_id', 'environment', 'related_entities', 'metadata']:
                if hasattr(context, field):
                    error_info["context"][field] = getattr(context, field)
            
        # Map to ErrorCategory
        error_type = self._get_error_type(error_report)
        error_info["category"] = self._map_error_type_to_category(error_type)
        
        return error_info
    
    def _map_error_type_to_category(self, error_type: str) -> str:
        """Map error type to ErrorCategory."""
        # Direct mapping for known categories
        for category in ErrorCategory:
            if category.name.lower() == error_type.lower() or category.value.lower() == error_type.lower():
                return category.value
        
        # Keyword-based mapping
        keyword_mapping = {
            "network": ErrorCategory.CONNECTION.value,
            "connection": ErrorCategory.CONNECTION.value,
            "timeout": ErrorCategory.TIMEOUT.value,
            "worker": ErrorCategory.WORKER_OFFLINE.value,
            "resource": ErrorCategory.SYSTEM_RESOURCE.value,
            "database": ErrorCategory.DB_CONNECTION.value,
            "db": ErrorCategory.DB_CONNECTION.value,
            "task": ErrorCategory.TASK_ERROR.value,
            "auth": ErrorCategory.AUTH_ERROR.value,
            "permission": ErrorCategory.UNAUTHORIZED.value,
            "coordinator": ErrorCategory.COORDINATOR_ERROR.value,
            "state": ErrorCategory.STATE_ERROR.value,
            "system": ErrorCategory.SYSTEM_RESOURCE.value
        }
        
        for keyword, category in keyword_mapping.items():
            if keyword in error_type.lower():
                return category
        
        # Default to unknown
        return ErrorCategory.UNKNOWN.value
    
    def _get_adaptive_timeout(self, error_type: str, strategy_id: str) -> float:
        """Get adaptive timeout for error type and strategy."""
        key = f"{error_type}:{strategy_id}"
        
        # Return adaptive timeout if available
        if key in self.adaptive_timeouts:
            return self.adaptive_timeouts[key]
        
        # Otherwise return default timeout
        return self.default_timeout
    
    def _update_adaptive_timeout(self, error_type: str, strategy_id: str, current_timeout: float, success: bool):
        """Update adaptive timeout based on execution result."""
        key = f"{error_type}:{strategy_id}"
        
        # Initialize timeout history for this key if needed
        if key not in self.adaptive_timeouts:
            self.adaptive_timeouts[key] = current_timeout
        
        # Adjust timeout based on success/failure
        current_value = self.adaptive_timeouts[key]
        
        if success:
            # If successful, gradually reduce timeout (but not below 50% of current)
            new_timeout = max(current_value * 0.9, current_timeout * 0.5, 5.0)  # Min 5 seconds
        else:
            # If timeout occurred, increase by 50%
            new_timeout = min(current_value * 1.5, 300.0)  # Max 5 minutes
        
        # Update timeout
        self.adaptive_timeouts[key] = new_timeout
        
        # Persist to database
        if self.db_connection:
            try:
                self.db_connection.execute("""
                INSERT OR REPLACE INTO adaptive_timeouts
                (error_type, strategy_id, timeout, last_updated)
                VALUES (?, ?, ?, ?)
                """, (error_type, strategy_id, new_timeout, datetime.now()))
            except Exception as e:
                logger.error(f"Error updating adaptive timeout: {str(e)}")
        
        logger.debug(f"Updated adaptive timeout for {key}: {current_value:.1f}s -> {new_timeout:.1f}s")
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage statistics."""
        try:
            import psutil
            
            # System-wide CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Process-specific resources
            process = psutil.Process()
            process_cpu = process.cpu_percent(interval=0.1)
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent
            net_recv = net_io.bytes_recv
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "process_cpu": process_cpu,
                "process_memory_mb": process_memory,
                "disk_percent": disk_percent,
                "net_sent_bytes": net_sent,
                "net_recv_bytes": net_recv
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "process_cpu": 0.0,
                "process_memory_mb": 0.0,
                "disk_percent": 0.0,
                "net_sent_bytes": 0,
                "net_recv_bytes": 0
            }
        except Exception as e:
            logger.warning(f"Error getting resource usage: {str(e)}")
            return {}
    
    def _calculate_resource_diff(self, before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        """Calculate difference in resource usage."""
        diff = {}
        
        # Calculate differences for each metric
        for key in before:
            if key in after:
                diff[key] = after[key] - before[key]
        
        return diff
    
    def _calculate_impact_score(
        self, 
        resource_diff: Dict[str, float],
        execution_time: float,
        affected_tasks: int,
        recovered_tasks: int
    ) -> float:
        """
        Calculate impact score of recovery operation.
        
        The impact score ranges from 0.0 (no impact) to 1.0 (severe impact).
        It considers resource usage, execution time, and task impact.
        """
        # Memory impact (0-0.25)
        memory_impact = min(abs(resource_diff.get("memory_percent", 0)) / 100.0, 0.25)
        
        # CPU impact (0-0.25)
        cpu_impact = min(abs(resource_diff.get("cpu_percent", 0)) / 100.0, 0.25) 
        
        # Time impact (0-0.25)
        # Normalize execution time: 0-5s is minimal, 60s+ is severe
        time_impact = min(execution_time / 240.0, 0.25)
        
        # Task impact (0-0.25)
        # Consider both affected tasks and recovery success rate
        task_count_impact = 0.0
        if hasattr(self.coordinator, 'tasks') and self.coordinator.tasks:
            total_tasks = len(self.coordinator.tasks)
            if total_tasks > 0:
                task_count_impact = min(affected_tasks / total_tasks, 0.125)
        
        # Recovery success impact
        recovery_impact = 0.0
        if affected_tasks > 0:
            recovery_rate = recovered_tasks / affected_tasks
            recovery_impact = 0.125 * (1 - recovery_rate)
        
        # Calculate total impact
        impact_score = memory_impact + cpu_impact + time_impact + task_count_impact + recovery_impact
        
        # Ensure score is in 0-1 range
        return min(max(impact_score, 0.0), 1.0)
    
    async def _check_stability(self) -> float:
        """
        Check system stability after recovery.
        
        Returns a stability score from 0.0 (unstable) to 1.0 (fully stable).
        """
        stability_factors = []
        
        # Check coordinator responsiveness
        if hasattr(self.coordinator, 'db'):
            try:
                self.coordinator.db.execute("SELECT 1").fetchone()
                stability_factors.append(1.0)  # Database responsive
            except Exception:
                stability_factors.append(0.0)  # Database not responsive
        
        # Check worker connections
        if hasattr(self.coordinator, 'worker_connections'):
            total_workers = len(self.coordinator.worker_connections)
            active_workers = 0
            
            for ws in self.coordinator.worker_connections.values():
                if not getattr(ws, 'closed', True):
                    active_workers += 1
            
            if total_workers > 0:
                worker_stability = active_workers / total_workers
                stability_factors.append(worker_stability)
        
        # Check task scheduling
        if (hasattr(self.coordinator, 'running_tasks') and 
            hasattr(self.coordinator, 'pending_tasks')):
            # Good state: either some tasks running or no pending tasks
            if self.coordinator.running_tasks or not self.coordinator.pending_tasks:
                stability_factors.append(1.0)
            else:
                # Tasks pending but none running
                stability_factors.append(0.5)
        
        # Check system resources
        resources = self._get_resource_usage()
        cpu_percent = resources.get("cpu_percent", 0)
        memory_percent = resources.get("memory_percent", 0)
        
        if cpu_percent < 90:  # CPU not critically high
            stability_factors.append(1.0 - (cpu_percent / 100.0) * 0.5)  # Scale 0.5-1.0
        else:
            stability_factors.append(0.0)  # Critical CPU usage
            
        if memory_percent < 90:  # Memory not critically high
            stability_factors.append(1.0 - (memory_percent / 100.0) * 0.5)  # Scale 0.5-1.0
        else:
            stability_factors.append(0.0)  # Critical memory usage
        
        # Calculate overall stability
        if stability_factors:
            stability_score = sum(stability_factors) / len(stability_factors)
            logger.debug(f"Stability score: {stability_score:.2f} (factors: {stability_factors})")
            return stability_score
        else:
            # Default to medium stability if no factors available
            return 0.5
    
    async def _get_affected_tasks(self, error_report: Any) -> Set[str]:
        """
        Get tasks affected by an error.
        
        Args:
            error_report: Error report object
            
        Returns:
            Set of affected task IDs
        """
        affected_tasks = set()
        
        # Try to extract from error context
        if hasattr(error_report, 'context'):
            context = error_report.context
            
            # Check for related_entities.task_id or similar
            if hasattr(context, 'related_entities'):
                related = context.related_entities

                def _related_get(key: str):
                    if isinstance(related, dict):
                        return related.get(key)
                    return getattr(related, key, None)
                
                # Check for task_id field
                task_id = _related_get('task_id')
                if task_id:
                    if isinstance(task_id, list):
                        affected_tasks.update(task_id)
                    else:
                        affected_tasks.add(str(task_id))
                
                # Check for task_ids field
                task_ids = _related_get('task_ids')
                if task_ids:
                    if isinstance(task_ids, list):
                        affected_tasks.update(task_ids)
                    else:
                        affected_tasks.add(str(task_ids))
            
            # Handle different context formats
            if hasattr(context, 'task_id'):
                task_id = context.task_id
                if isinstance(task_id, list):
                    affected_tasks.update(task_id)
                elif task_id:
                    affected_tasks.add(str(task_id))
        
        # If no specific tasks found, infer from error type and component
        if not affected_tasks:
            error_type = self._get_error_type(error_report)
            component = getattr(error_report, 'component', '')
            if not component and hasattr(error_report, 'context'):
                component = getattr(error_report.context, 'component', '')
            
            # For worker errors, get all tasks assigned to that worker
            if 'worker' in component.lower() and hasattr(self.coordinator, 'running_tasks'):
                # Try to extract worker_id
                worker_id = None
                if hasattr(error_report, 'context') and hasattr(error_report.context, 'related_entities'):
                    rel = error_report.context.related_entities
                    if isinstance(rel, dict):
                        worker_id = rel.get('worker_id')
                    else:
                        worker_id = getattr(rel, 'worker_id', None)
                
                if worker_id:
                    # Get all tasks assigned to this worker
                    for task_id, assigned_worker in self.coordinator.running_tasks.items():
                        if assigned_worker == worker_id:
                            affected_tasks.add(task_id)
                else:
                    # Without specific worker_id, assume all running tasks may be affected
                    affected_tasks.update(self.coordinator.running_tasks.keys())
            
            # For database errors, consider all running tasks affected
            elif 'database' in component.lower() and hasattr(self.coordinator, 'running_tasks'):
                affected_tasks.update(self.coordinator.running_tasks.keys())
            
            # For coordinator errors, consider all pending and running tasks affected
            elif 'coordinator' in component.lower():
                if hasattr(self.coordinator, 'running_tasks'):
                    affected_tasks.update(self.coordinator.running_tasks.keys())
                
                if hasattr(self.coordinator, 'pending_tasks'):
                    affected_tasks.update(self.coordinator.pending_tasks)
        
        return affected_tasks
    
    async def _check_task_recovery(self, affected_tasks: Set[str]) -> int:
        """
        Check how many affected tasks were successfully recovered.
        
        Args:
            affected_tasks: Set of task IDs that were affected by the error
            
        Returns:
            Number of successfully recovered tasks
        """
        if not hasattr(self.coordinator, 'tasks'):
            return 0
        
        recovered_count = 0
        
        for task_id in affected_tasks:
            if task_id in self.coordinator.tasks:
                task = self.coordinator.tasks[task_id]
                
                # Task is successfully recovered if its status is good
                if task.get("status") in ["pending", "running", "completed"]:
                    recovered_count += 1
                
                # Check if task has been retried successfully
                elif task.get("status") == "failed" and task.get("retried_task_id"):
                    retried_id = task.get("retried_task_id")
                    if retried_id in self.coordinator.tasks:
                        retried_task = self.coordinator.tasks[retried_id]
                        if retried_task.get("status") in ["pending", "running", "completed"]:
                            recovered_count += 1
        
        return recovered_count
    
    def _record_performance(
        self,
        strategy_id: str,
        strategy_name: str,
        error_type: str,
        execution_time: float,
        success: bool,
        affected_tasks: int = 0,
        recovered_tasks: int = 0,
        resource_usage: Dict[str, float] = None,
        impact_score: float = 0.0,
        stability_score: float = 1.0,
        context: Dict[str, Any] = None
    ):
        """
        Record performance data for a recovery strategy.
        
        Args:
            strategy_id: ID of the strategy
            strategy_name: Name of the strategy
            error_type: Type of error
            execution_time: Execution time in seconds
            success: Whether recovery was successful
            affected_tasks: Number of affected tasks
            recovered_tasks: Number of successfully recovered tasks
            resource_usage: Resource usage metrics
            impact_score: Impact score (0-1, lower is better)
            stability_score: Stability score (0-1, higher is better)
            context: Additional context information
        """
        resource_usage = resource_usage or {}
        context = context or {}
        
        # Create record
        record = RecoveryPerformanceRecord(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            error_type=error_type,
            execution_time=execution_time,
            success=success,
            affected_tasks=affected_tasks,
            recovered_tasks=recovered_tasks
        )
        
        # Set additional fields
        record.resource_usage = resource_usage
        record.impact_score = impact_score
        record.stability_score = stability_score
        record.context = context
        
        # Add to in-memory history
        self.performance_history[strategy_id].append(record)
        
        # Keep history size manageable
        if len(self.performance_history[strategy_id]) > 100:
            self.performance_history[strategy_id] = self.performance_history[strategy_id][-100:]
        
        # Persist to database
        if self.db_connection:
            try:
                # Get next ID for the record (DuckDB doesn't have AUTOINCREMENT)
                next_id = 1
                try:
                    # Get max ID if there are existing records
                    max_id = self.db_connection.execute("""
                    SELECT MAX(id) FROM recovery_performance
                    """).fetchone()[0]
                    
                    if max_id is not None:
                        next_id = max_id + 1
                except Exception:
                    # If query fails (e.g., empty table), use default (1)
                    pass
                
                self.db_connection.execute("""
                INSERT INTO recovery_performance (
                    id, strategy_id, strategy_name, error_type, execution_time,
                    success, timestamp, hardware_id, affected_tasks,
                    recovered_tasks, resource_usage, impact_score,
                    stability_score, context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    next_id,
                    strategy_id,
                    strategy_name,
                    error_type,
                    execution_time,
                    success,
                    datetime.now(),
                    None,  # hardware_id
                    affected_tasks,
                    recovered_tasks,
                    json.dumps(resource_usage),
                    impact_score,
                    stability_score,
                    json.dumps(context)
                ))
            except Exception as e:
                logger.error(f"Error recording performance data: {str(e)}")
    
    def _update_strategy_scores(self, strategy_id: str, error_type: str):
        """
        Update scores for a strategy based on performance history.
        
        Args:
            strategy_id: ID of the strategy
            error_type: Type of error
        """
        # Get relevant performance records
        records = [
            r for r in self.performance_history.get(strategy_id, [])
            if r.error_type == error_type
        ]
        
        if not records:
            logger.debug(f"No performance records for {strategy_id} with error type {error_type}")
            return
        
        # Calculate success rate
        success_rate = sum(1 for r in records if r.success) / len(records)
        
        # Calculate execution time (for successful recoveries)
        successful_records = [r for r in records if r.success]
        if successful_records:
            avg_execution_time = sum(r.execution_time for r in successful_records) / len(successful_records)
        else:
            avg_execution_time = float('inf')
        
        # Calculate impact score (lower is better)
        avg_impact = sum(r.impact_score for r in records) / len(records)
        
        # Calculate stability score (higher is better)
        avg_stability = sum(r.stability_score for r in records) / len(records)
        
        # Calculate task recovery rate
        task_recovery_rates = []
        for r in records:
            if r.affected_tasks > 0:
                rate = r.recovered_tasks / r.affected_tasks
                task_recovery_rates.append(rate)
        
        task_recovery_rate = (
            sum(task_recovery_rates) / len(task_recovery_rates)
            if task_recovery_rates else 0.0
        )
        
        # Calculate overall score
        # Weights for different factors (sum to 1.0)
        weights = {
            "success_rate": 0.4,
            "execution_time": 0.15,
            "impact_score": 0.15,
            "stability_score": 0.15,
            "task_recovery_rate": 0.15
        }
        
        # Normalize execution time to 0-1 scale (lower is better)
        if avg_execution_time == float('inf'):
            time_score = 0.0
        else:
            # Scale: 0-1s excellent (1.0), 30s+ poor (0.0)
            time_score = max(0.0, min(1.0, 1.0 - (avg_execution_time / 30.0)))
        
        # Calculate overall score
        overall_score = (
            weights["success_rate"] * success_rate +
            weights["execution_time"] * time_score +
            weights["impact_score"] * (1.0 - avg_impact) +  # Invert so lower impact is better
            weights["stability_score"] * avg_stability +
            weights["task_recovery_rate"] * task_recovery_rate
        )
        
        # Update in-memory score
        self.strategy_scores[error_type][strategy_id] = overall_score
        
        # Create metrics for database
        metrics = {
            "success_rate": success_rate,
            "execution_time": avg_execution_time,
            "time_score": time_score,
            "impact_score": avg_impact,
            "stability_score": avg_stability,
            "task_recovery_rate": task_recovery_rate
        }
        
        # Persist to database
        if self.db_connection:
            try:
                self.db_connection.execute("""
                INSERT OR REPLACE INTO strategy_scores
                (error_type, strategy_id, score, last_updated, samples, metrics)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    error_type,
                    strategy_id,
                    overall_score,
                    datetime.now(),
                    len(records),
                    json.dumps(metrics)
                ))
            except Exception as e:
                logger.error(f"Error updating strategy score: {str(e)}")
        
        logger.debug(f"Updated strategy score for {strategy_id} with error type {error_type}: {overall_score:.2f}")
    
    def _record_progression(
        self,
        error_id: str,
        old_level: int,
        new_level: int,
        strategy_id: str,
        success: bool,
        details: Dict[str, Any] = None
    ):
        """
        Record progressive recovery history.
        
        Args:
            error_id: ID of the error
            old_level: Old recovery level
            new_level: New recovery level
            strategy_id: ID of the strategy used
            success: Whether recovery was successful
            details: Additional details
        """
        details = details or {}
        
        # Create history entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "old_level": old_level,
            "new_level": new_level,
            "strategy_id": strategy_id,
            "success": success,
            "details": details
        }
        
        # Add to in-memory history
        self.recovery_history[error_id].append(entry)
        
        # Keep history size manageable
        if len(self.recovery_history[error_id]) > 10:
            self.recovery_history[error_id] = self.recovery_history[error_id][-10:]
        
        # Persist to database
        if self.db_connection:
            try:
                # Convert history to JSON
                history_json = json.dumps(self.recovery_history[error_id])
                
                self.db_connection.execute("""
                INSERT OR REPLACE INTO progressive_recovery
                (error_id, current_level, last_updated, history)
                VALUES (?, ?, ?, ?)
                """, (
                    error_id,
                    new_level,
                    datetime.now(),
                    history_json
                ))
            except Exception as e:
                logger.error(f"Error recording progressive recovery: {str(e)}")
    
    def get_performance_metrics(
        self,
        error_type: Optional[str] = None,
        strategy_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get performance metrics for recovery strategies.
        
        Args:
            error_type: Optional filter by error type
            strategy_id: Optional filter by strategy
            days: Number of days to include in analysis
            
        Returns:
            Dictionary with performance metrics
        """
        # Query database for comprehensive metrics
        if self.db_connection:
            try:
                # Build query conditions using proper DuckDB date function
                days_filter = f"timestamp > (CURRENT_TIMESTAMP - INTERVAL '{days} days')"
                conditions = [days_filter]
                params = []
                
                if error_type:
                    conditions.append("error_type = ?")
                    params.append(error_type)
                
                if strategy_id:
                    conditions.append("strategy_id = ?")
                    params.append(strategy_id)
                
                # Query performance records
                where_clause = " AND ".join(conditions)
                query = f"""
                SELECT 
                    strategy_id, 
                    MIN(strategy_name) as strategy_name, -- Use MIN to get a single value
                    error_type, 
                    AVG(execution_time) as avg_execution_time,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    COUNT(*) as total,
                    AVG(impact_score) as avg_impact,
                    AVG(stability_score) as avg_stability,
                    SUM(recovered_tasks) as recovered_tasks,
                    SUM(affected_tasks) as affected_tasks
                FROM recovery_performance
                WHERE {where_clause}
                GROUP BY strategy_id, error_type
                """
                
                result = self.db_connection.execute(query, params).fetchall()
                
                # Process results
                strategies = {}
                for row in result:
                    (
                        strategy_id, strategy_name, error_type, 
                        avg_execution_time, successes, total,
                        avg_impact, avg_stability,
                        recovered_tasks, affected_tasks
                    ) = row
                    
                    success_rate = successes / total if total > 0 else 0
                    recovery_rate = recovered_tasks / affected_tasks if affected_tasks > 0 else 0
                    
                    # Get current score
                    score = self.strategy_scores.get(error_type, {}).get(strategy_id, 0.0)
                    
                    if strategy_id not in strategies:
                        strategies[strategy_id] = {
                            "strategy_id": strategy_id,
                            "strategy_name": strategy_name,
                            "error_types": {},
                            "total_executions": 0,
                            "success_rate": 0.0,
                            "avg_execution_time": 0.0,
                            "overall_score": 0.0
                        }
                    
                    # Add error type stats
                    strategies[strategy_id]["error_types"][error_type] = {
                        "executions": total,
                        "success_rate": success_rate,
                        "avg_execution_time": avg_execution_time,
                        "avg_impact": avg_impact,
                        "avg_stability": avg_stability,
                        "task_recovery_rate": recovery_rate,
                        "score": score
                    }
                    
                    # Update totals
                    strategies[strategy_id]["total_executions"] += total
                
                # Calculate overall metrics for each strategy
                for sid, strategy in strategies.items():
                    total_execs = strategy["total_executions"]
                    weighted_success = 0.0
                    weighted_time = 0.0
                    weighted_score = 0.0
                    
                    for et_stats in strategy["error_types"].values():
                        weight = et_stats["executions"] / total_execs
                        weighted_success += et_stats["success_rate"] * weight
                        weighted_time += et_stats["avg_execution_time"] * weight
                        weighted_score += et_stats["score"] * weight
                    
                    strategy["success_rate"] = weighted_success
                    strategy["avg_execution_time"] = weighted_time
                    strategy["overall_score"] = weighted_score
                
                # Overall statistics
                total_executions = sum(s["total_executions"] for s in strategies.values())
                total_success_rate = sum(s["success_rate"] * s["total_executions"] for s in strategies.values()) / total_executions if total_executions > 0 else 0
                
                # Top strategies by error type
                top_strategies = {}
                for error_type in set(et for s in strategies.values() for et in s["error_types"].keys()):
                    best_sid = None
                    best_score = -1.0
                    
                    for sid, strategy in strategies.items():
                        if error_type in strategy["error_types"]:
                            score = strategy["error_types"][error_type]["score"]
                            if score > best_score:
                                best_score = score
                                best_sid = sid
                    
                    if best_sid:
                        top_strategies[error_type] = {
                            "strategy_id": best_sid,
                            "strategy_name": strategies[best_sid]["strategy_name"],
                            "score": best_score
                        }
                
                return {
                    "strategies": strategies,
                    "top_strategies": top_strategies,
                    "overall": {
                        "total_executions": total_executions,
                        "overall_success_rate": total_success_rate,
                        "strategy_count": len(strategies),
                        "error_type_count": len(set(et for s in strategies.values() for et in s["error_types"].keys()))
                    }
                }
            except Exception as e:
                logger.error(f"Error getting performance metrics from database: {str(e)}")
        
        # Fallback to in-memory metrics
        strategies = {}
        for strategy_id, records in self.performance_history.items():
            # Filter by time
            cutoff = datetime.now() - timedelta(days=days)
            records = [r for r in records if r.timestamp > cutoff]
            
            # Filter by error type
            if error_type:
                records = [r for r in records if r.error_type == error_type]
            
            # Skip if no records
            if not records:
                continue
            
            # Group by error type
            by_error_type = defaultdict(list)
            for record in records:
                by_error_type[record.error_type].append(record)
            
            # Skip if no relevant error types
            if not by_error_type:
                continue
            
            # Get strategy name
            strategy_name = records[0].strategy_name
            
            # Initialize strategy entry
            strategies[strategy_id] = {
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "error_types": {},
                "total_executions": len(records),
                "success_rate": sum(1 for r in records if r.success) / len(records),
                "avg_execution_time": sum(r.execution_time for r in records) / len(records),
                "overall_score": self.strategy_scores.get(error_type, {}).get(strategy_id, 0.0) if error_type else 0.0
            }
            
            # Add per-error-type metrics
            for et, et_records in by_error_type.items():
                success_rate = sum(1 for r in et_records if r.success) / len(et_records)
                avg_execution_time = sum(r.execution_time for r in et_records) / len(et_records)
                
                # Calculate task recovery rate
                recovery_rates = []
                for r in et_records:
                    if r.affected_tasks > 0:
                        rate = r.recovered_tasks / r.affected_tasks
                        recovery_rates.append(rate)
                
                recovery_rate = sum(recovery_rates) / len(recovery_rates) if recovery_rates else 0.0
                
                # Add error type stats
                strategies[strategy_id]["error_types"][et] = {
                    "executions": len(et_records),
                    "success_rate": success_rate,
                    "avg_execution_time": avg_execution_time,
                    "avg_impact": sum(r.impact_score for r in et_records) / len(et_records),
                    "avg_stability": sum(r.stability_score for r in et_records) / len(et_records),
                    "task_recovery_rate": recovery_rate,
                    "score": self.strategy_scores.get(et, {}).get(strategy_id, 0.0)
                }
            
            # Calculate overall score if not filtered by error type
            if not error_type:
                # Weight by number of executions for each error type
                total_execs = strategies[strategy_id]["total_executions"]
                weighted_score = 0.0
                
                for et, et_stats in strategies[strategy_id]["error_types"].items():
                    weight = et_stats["executions"] / total_execs
                    score = self.strategy_scores.get(et, {}).get(strategy_id, 0.0)
                    weighted_score += score * weight
                
                strategies[strategy_id]["overall_score"] = weighted_score
        
        # Overall statistics
        total_executions = sum(s["total_executions"] for s in strategies.values())
        total_success_rate = sum(s["success_rate"] * s["total_executions"] for s in strategies.values()) / total_executions if total_executions > 0 else 0
        
        # Top strategies by error type
        top_strategies = {}
        for error_type in set(et for s in strategies.values() for et in s["error_types"].keys()):
            best_sid = None
            best_score = -1.0
            
            for sid, strategy in strategies.items():
                if error_type in strategy["error_types"]:
                    score = strategy["error_types"][error_type]["score"]
                    if score > best_score:
                        best_score = score
                        best_sid = sid
            
            if best_sid:
                top_strategies[error_type] = {
                    "strategy_id": best_sid,
                    "strategy_name": strategies[best_sid]["strategy_name"],
                    "score": best_score
                }
        
        return {
            "strategies": strategies,
            "top_strategies": top_strategies,
            "overall": {
                "total_executions": total_executions,
                "overall_success_rate": total_success_rate,
                "strategy_count": len(strategies),
                "error_type_count": len(set(et for s in strategies.values() for et in s["error_types"].keys()))
            }
        }
    
    def get_timeouts(self) -> Dict[str, float]:
        """Get current adaptive timeouts."""
        return self.adaptive_timeouts
    
    def get_recovery_levels(self) -> Dict[str, int]:
        """Get current recovery levels for errors."""
        return self.error_recovery_levels
    
    def get_recovery_history(self, error_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get recovery history for all errors or a specific error.
        
        Args:
            error_id: Optional filter by error ID
            
        Returns:
            Dictionary with recovery history
        """
        if error_id:
            return {
                "error_id": error_id,
                "history": self.recovery_history.get(error_id, []),
                "current_level": self.error_recovery_levels.get(error_id, 1)
            }
        
        # Return summary of all errors
        errors = []
        for eid, history in self.recovery_history.items():
            level = self.error_recovery_levels.get(eid, 1)
            last_attempt = history[-1] if history else None
            
            errors.append({
                "error_id": eid,
                "current_level": level,
                "attempts": len(history),
                "last_attempt": last_attempt
            })
        
        return {
            "errors": errors,
            "summary": {
                "error_count": len(errors),
                "level_1_count": sum(1 for e in errors if e["current_level"] == 1),
                "level_2_count": sum(1 for e in errors if e["current_level"] == 2),
                "level_3_count": sum(1 for e in errors if e["current_level"] == 3),
                "level_4_count": sum(1 for e in errors if e["current_level"] == 4),
                "level_5_count": sum(1 for e in errors if e["current_level"] == 5)
            }
        }
    
    def reset_recovery_level(self, error_id: str):
        """
        Reset recovery level for an error.
        
        Args:
            error_id: The error ID to reset
        """
        if error_id in self.error_recovery_levels:
            del self.error_recovery_levels[error_id]
            
            # Update database
            if self.db_connection:
                try:
                    self.db_connection.execute("""
                    DELETE FROM progressive_recovery
                    WHERE error_id = ?
                    """, (error_id,))
                except Exception as e:
                    logger.error(f"Error resetting recovery level in database: {str(e)}")
            
            logger.info(f"Reset recovery level for error {error_id}")
            return True
        return False
    
    def reset_all_recovery_levels(self):
        """Reset all recovery levels."""
        self.error_recovery_levels.clear()
        
        # Update database
        if self.db_connection:
            try:
                self.db_connection.execute("DELETE FROM progressive_recovery")
            except Exception as e:
                logger.error(f"Error resetting all recovery levels in database: {str(e)}")
        
        logger.info("Reset all recovery levels")
        return True
    
    def get_strategy_recommendations(self, error_type: str) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations for an error type.
        
        Args:
            error_type: The error type
            
        Returns:
            List of recommended strategies sorted by score
        """
        # Get strategies and scores for this error type
        strategies = []
        
        # Get scores from database
        if self.db_connection:
            try:
                result = self.db_connection.execute("""
                SELECT strategy_id, score, metrics, samples
                FROM strategy_scores
                WHERE error_type = ?
                ORDER BY score DESC
                """, (error_type,)).fetchall()
                
                for row in result:
                    strategy_id, score, metrics_json, samples = row
                    
                    # Get strategy name
                    strategy_name = "unknown"
                    if self.recovery_manager and hasattr(self.recovery_manager, 'strategies'):
                        strategy = self.recovery_manager.strategies.get(strategy_id)
                        if strategy:
                            strategy_name = strategy.name
                    
                    metrics = json.loads(metrics_json) if metrics_json else {}
                    
                    strategies.append({
                        "strategy_id": strategy_id,
                        "strategy_name": strategy_name,
                        "score": score,
                        "samples": samples,
                        "metrics": metrics
                    })
                
                return strategies
            except Exception as e:
                logger.error(f"Error getting strategy recommendations from database: {str(e)}")
        
        # Fallback to in-memory scores
        for strategy_id, score in self.strategy_scores.get(error_type, {}).items():
            # Get strategy name
            strategy_name = "unknown"
            if self.recovery_manager and hasattr(self.recovery_manager, 'strategies'):
                strategy = self.recovery_manager.strategies.get(strategy_id)
                if strategy:
                    strategy_name = strategy.name
            
            strategies.append({
                "strategy_id": strategy_id,
                "strategy_name": strategy_name,
                "score": score,
                "samples": sum(1 for r in self.performance_history.get(strategy_id, []) if r.error_type == error_type)
            })
        
        # Sort by score (highest first)
        strategies.sort(key=lambda s: s["score"], reverse=True)
        
        return strategies


if __name__ == "__main__":
    # Demo and testing code
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the performance-based error recovery system")
    parser.add_argument("--db-path", help="Path to DuckDB database file", default=":memory:")
    parser.add_argument("--test-errors", action="store_true", help="Test error recovery")
    parser.add_argument("--get-metrics", action="store_true", help="Get performance metrics")
    args = parser.parse_args()
    
    async def run_tests():
        print("Testing Performance-Based Error Recovery System")
        
        # Mock coordinator for testing
        class MockCoordinator:
            def __init__(self):
                self.tasks = {}
                self.running_tasks = {}
                self.pending_tasks = set()
                self.worker_connections = {}
                
                import duckdb
                self.db = duckdb.connect(args.db_path)
                
                # Add some test tasks
                for i in range(10):
                    task_id = f"task-{i}"
                    self.tasks[task_id] = {
                        "task_id": task_id,
                        "status": "pending" if i < 3 else "running" if i < 8 else "completed",
                        "type": "test",
                        "created": datetime.now().isoformat()
                    }
                    
                    if self.tasks[task_id]["status"] == "running":
                        self.running_tasks[task_id] = f"worker-{i % 3}"
                    elif self.tasks[task_id]["status"] == "pending":
                        self.pending_tasks.add(task_id)
        
        # Create mock error handler
        class MockErrorHandler:
            def __init__(self):
                self.error_hooks = {}
            
            def register_error_hook(self, error_type, hook):
                self.error_hooks[error_type] = hook
            
            def create_error_report(self, error, context=None):
                # Simple error report
                return type('ErrorReport', (), {
                    'error_id': str(uuid.uuid4()),
                    'error_type': context.get('error_type', 'unknown'),
                    'message': str(error),
                    'context': type('Context', (), context or {}),
                    'component': context.get('component', 'unknown'),
                    'operation': context.get('operation', 'unknown')
                })
        
        # Create mock recovery strategies
        class MockRecoveryStrategy:
            def __init__(self, name, level, success_rate=1.0, delay=0.0):
                self.name = name
                self.level = level
                self.success_rate = success_rate
                self.delay = delay
            
            async def execute(self, error_info):
                # Simulate execution
                await anyio.sleep(self.delay)
                
                # Return success based on probability
                return random.random() < self.success_rate
        
        # Create mock recovery manager
        class MockRecoveryManager:
            def __init__(self):
                import random
                self.strategies = {
                    "retry": MockRecoveryStrategy("retry", "low", 0.8, 0.5),
                    "worker_restart": MockRecoveryStrategy("worker_restart", "medium", 0.7, 1.0),
                    "db_reconnect": MockRecoveryStrategy("db_reconnect", "medium", 0.9, 0.8),
                    "coordinator_restart": MockRecoveryStrategy("coordinator_restart", "high", 0.6, 2.0),
                    "system_recovery": MockRecoveryStrategy("system_recovery", "critical", 0.9, 3.0)
                }
        
        # Set up test components
        import random
        coordinator = MockCoordinator()
        error_handler = MockErrorHandler()
        recovery_manager = MockRecoveryManager()
        
        # Create recovery system
        recovery_system = PerformanceBasedErrorRecovery(
            error_handler=error_handler,
            recovery_manager=recovery_manager,
            coordinator=coordinator,
            db_connection=coordinator.db
        )
        
        if args.test_errors:
            # Test different error types
            error_types = [
                "connection", "timeout", "worker_offline", "task_error",
                "db_connection", "coordinator_error"
            ]
            
            for error_type in error_types:
                print(f"\nTesting recovery for error type: {error_type}")
                
                # Create mock error report
                error_report = error_handler.create_error_report(
                    Exception(f"Test {error_type} error"),
                    {
                        "error_type": error_type,
                        "component": error_type.split("_")[0],
                        "operation": "test",
                        "related_entities": {
                            "task_id": "task-1",
                            "worker_id": "worker-1"
                        }
                    }
                )
                
                # Recover from error
                success, info = await recovery_system.recover(error_report)
                
                print(f"Recovery {'successful' if success else 'failed'}")
                print(f"Strategy: {info['strategy_name']}")
                print(f"Execution time: {info['execution_time']:.2f}s")
                print(f"Affected tasks: {info.get('affected_tasks', 0)}")
                print(f"Recovered tasks: {info.get('recovered_tasks', 0)}")
                
                # For some errors, test progressive recovery
                if error_type in ["worker_offline", "db_connection"]:
                    # Force failure for first attempt
                    if success:
                        success, info = await recovery_system.recover(error_report)
                        print(f"Second recovery attempt: {'successful' if success else 'failed'}")
                        print(f"Strategy: {info['strategy_name']}")
                        print(f"Recovery level: {info['recovery_level']}")
        
        if args.get_metrics:
            # Get performance metrics
            print("\nPerformance metrics:")
            metrics = recovery_system.get_performance_metrics()
            
            print(f"Total executions: {metrics['overall']['total_executions']}")
            print(f"Overall success rate: {metrics['overall']['overall_success_rate']:.2f}")
            print(f"Strategy count: {metrics['overall']['strategy_count']}")
            
            print("\nTop strategies by error type:")
            for error_type, strategy in metrics['top_strategies'].items():
                print(f"  {error_type}: {strategy['strategy_name']} (score: {strategy['score']:.2f})")
            
            # Get adaptive timeouts
            print("\nAdaptive timeouts:")
            timeouts = recovery_system.get_timeouts()
            for key, timeout in timeouts.items():
                print(f"  {key}: {timeout:.2f}s")
            
            # Get recovery levels
            print("\nRecovery levels:")
            levels = recovery_system.get_recovery_levels()
            for error_id, level in levels.items():
                print(f"  {error_id}: Level {level}")
            
            # Get recovery history
            print("\nRecovery history:")
            history = recovery_system.get_recovery_history()
            print(f"  Total errors: {len(history['errors'])}")
            print(f"  Level 1: {history['summary']['level_1_count']}")
            print(f"  Level 2: {history['summary']['level_2_count']}")
            print(f"  Level 3: {history['summary']['level_3_count']}")
            print(f"  Level 4: {history['summary']['level_4_count']}")
            print(f"  Level 5: {history['summary']['level_5_count']}")
    
    # Run tests
    anyio.run(run_tests())