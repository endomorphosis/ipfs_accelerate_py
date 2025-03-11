#!/usr/bin/env python3
"""
Distributed Testing Framework - Enhanced Error Recovery Strategies

This module implements enhanced error recovery strategies for the distributed testing framework.
It provides a comprehensive approach to handling different types of failures, with specialized
recovery procedures for different failure scenarios.

Key features:
- Categorized error handling for different failure types
- Progressive recovery with escalation for persistent errors
- Coordinated recovery for distributed system failures
- Dependency-aware recovery with topological sorting
- Adaptive timeout and retry strategies
- Post-recovery verification and validation
- Recovery history tracking and analysis

Usage:
    Import this module in coordinator.py to enhance the error recovery capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("error_recovery")

class ErrorCategory(Enum):
    """Categories of errors for specialized recovery strategies."""
    
    # Connection-related errors
    CONNECTION = "connection"         # Network connection issues
    TIMEOUT = "timeout"               # Request timeouts
    
    # Worker-related errors
    WORKER_OFFLINE = "worker_offline" # Worker node offline
    WORKER_CRASH = "worker_crash"     # Worker process crashed
    WORKER_RESOURCE = "worker_resource" # Worker resource exhaustion
    
    # Task-related errors
    TASK_TIMEOUT = "task_timeout"     # Task execution timeout
    TASK_ERROR = "task_error"         # Task execution error
    TASK_RESOURCE = "task_resource"   # Task resource limits exceeded
    
    # Database-related errors
    DB_CONNECTION = "db_connection"   # Database connection issues
    DB_QUERY = "db_query"             # Database query errors
    DB_INTEGRITY = "db_integrity"     # Database integrity issues
    
    # Coordinator-related errors
    COORDINATOR_ERROR = "coordinator_error" # Coordinator internal error
    STATE_ERROR = "state_error"       # Distributed state errors
    COORDINATOR_CRASH = "coordinator_crash" # Coordinator process crashed
    
    # Security-related errors
    AUTH_ERROR = "auth_error"         # Authentication errors
    UNAUTHORIZED = "unauthorized"     # Unauthorized access attempts
    
    # System-wide errors
    SYSTEM_RESOURCE = "system_resource" # System resource exhaustion
    DISK_FULL = "disk_full"           # Disk space exhaustion
    SYSTEM_OVERLOAD = "system_overload" # System overload
    
    # Unknown errors
    UNKNOWN = "unknown"               # Unclassified errors

class RecoveryLevel(Enum):
    """Levels of recovery severity/effort."""
    
    LOW = "low"           # Simple retries, reconnects
    MEDIUM = "medium"     # Service restarts, task reassignment
    HIGH = "high"         # Full component recovery, restart
    CRITICAL = "critical" # System-wide recovery procedures
    MANUAL = "manual"     # Requires manual intervention

class RecoveryStrategy:
    """Base class for recovery strategies."""
    
    def __init__(self, coordinator, name: str, level: RecoveryLevel):
        """
        Initialize the recovery strategy.
        
        Args:
            coordinator: Reference to the coordinator instance
            name: Name of the strategy
            level: Recovery level
        """
        self.coordinator = coordinator
        self.name = name
        self.level = level
        self.success_rate = 1.0  # Initial success rate estimate (optimistic)
        self.attempts = 0
        self.successes = 0
        
        logger.debug(f"Initialized recovery strategy: {name} (level: {level.value})")
    
    async def execute(self, error_info: Dict[str, Any]) -> bool:
        """
        Execute the recovery strategy.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            # Increment attempts counter
            self.attempts += 1
            
            # Record start time
            start_time = time.time()
            
            # Execute strategy implementation
            success = await self._execute_impl(error_info)
            
            # Record execution time
            execution_time = time.time() - start_time
            
            # Update success rate
            if success:
                self.successes += 1
            self.success_rate = self.successes / self.attempts if self.attempts > 0 else 0.0
            
            # Log outcome
            if success:
                logger.info(f"Recovery strategy {self.name} succeeded in {execution_time:.2f}s (success rate: {self.success_rate:.2f})")
            else:
                logger.warning(f"Recovery strategy {self.name} failed after {execution_time:.2f}s (success rate: {self.success_rate:.2f})")
            
            return success
        except Exception as e:
            logger.error(f"Error executing recovery strategy {self.name}: {str(e)}")
            # Update success rate
            self.success_rate = self.successes / self.attempts if self.attempts > 0 else 0.0
            return False
    
    async def _execute_impl(self, error_info: Dict[str, Any]) -> bool:
        """
        Implementation of recovery strategy.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # To be implemented by specific strategies
        raise NotImplementedError("Recovery strategy implementation not provided")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the strategy.
        
        Returns:
            Dictionary with strategy information
        """
        return {
            "name": self.name,
            "level": self.level.value,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": self.success_rate
        }

class RetryStrategy(RecoveryStrategy):
    """Simple retry strategy with exponential backoff."""
    
    def __init__(self, coordinator, max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
        """
        Initialize the retry strategy.
        
        Args:
            coordinator: Reference to the coordinator instance
            max_retries: Maximum number of retries
            initial_delay: Initial delay between retries in seconds
            backoff_factor: Backoff factor for exponential backoff
        """
        super().__init__(coordinator, "retry", RecoveryLevel.LOW)
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
    
    async def _execute_impl(self, error_info: Dict[str, Any]) -> bool:
        """
        Implement retry strategy.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if retry was successful, False otherwise
        """
        # Get the operation to retry
        operation = error_info.get("operation")
        if not operation or not callable(operation):
            logger.error("Retry strategy requires a callable operation")
            return False
        
        # Get retry arguments
        args = error_info.get("args", [])
        kwargs = error_info.get("kwargs", {})
        
        # Try the operation with retries
        for retry in range(self.max_retries):
            try:
                # Execute operation
                result = operation(*args, **kwargs)
                
                # Handle async operations
                if asyncio.iscoroutine(result):
                    result = await result
                
                # Success!
                return True
            except Exception as e:
                # Calculate delay with exponential backoff
                delay = self.initial_delay * (self.backoff_factor ** retry)
                
                logger.info(f"Retry {retry+1}/{self.max_retries} failed: {str(e)}. Retrying in {delay:.2f}s...")
                
                # Wait before next retry
                await asyncio.sleep(delay)
        
        # All retries failed
        logger.warning(f"All {self.max_retries} retries failed")
        return False

class WorkerRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for worker failures."""
    
    def __init__(self, coordinator):
        """
        Initialize the worker recovery strategy.
        
        Args:
            coordinator: Reference to the coordinator instance
        """
        super().__init__(coordinator, "worker_recovery", RecoveryLevel.MEDIUM)
    
    async def _execute_impl(self, error_info: Dict[str, Any]) -> bool:
        """
        Implement worker recovery strategy.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Get worker ID
        worker_id = error_info.get("worker_id")
        if not worker_id:
            logger.error("Worker recovery strategy requires a worker_id")
            return False
        
        # Get worker information
        worker = None
        if hasattr(self.coordinator, 'workers'):
            worker = self.coordinator.workers.get(worker_id)
        
        if not worker:
            logger.warning(f"Worker {worker_id} not found in coordinator state")
            return False
        
        # Close any existing connection
        if hasattr(self.coordinator, 'worker_connections'):
            if worker_id in self.coordinator.worker_connections:
                try:
                    ws = self.coordinator.worker_connections[worker_id]
                    await ws.close()
                except Exception:
                    pass
                
                # Remove from connections
                del self.coordinator.worker_connections[worker_id]
        
        # Mark worker as recovery_pending in database
        try:
            if hasattr(self.coordinator, 'db'):
                self.coordinator.db.execute(
                """
                UPDATE worker_nodes
                SET status = 'recovery_pending'
                WHERE worker_id = ?
                """,
                (worker_id,)
                )
        except Exception as e:
            logger.error(f"Error updating worker status in database: {str(e)}")
        
        # Update worker status in memory
        worker["status"] = "recovery_pending"
        
        # Find tasks assigned to this worker
        recovered_tasks = 0
        if hasattr(self.coordinator, 'running_tasks'):
            for task_id, assigned_worker_id in list(self.coordinator.running_tasks.items()):
                if assigned_worker_id == worker_id:
                    # Requeue task
                    await self._requeue_task(task_id, worker_id, "worker_failure")
                    recovered_tasks += 1
        
        logger.info(f"Recovered {recovered_tasks} tasks from failed worker {worker_id}")
        
        # Worker may reconnect later, but we've handled the tasks
        return True
    
    async def _requeue_task(self, task_id: str, worker_id: str, reason: str):
        """
        Requeue a task after worker failure.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID that was running the task
            reason: Reason for requeuing
        """
        if not hasattr(self.coordinator, 'tasks'):
            return
            
        logger.info(f"Requeuing task {task_id} due to {reason}")
        
        try:
            # Check if task exists
            if task_id not in self.coordinator.tasks:
                logger.warning(f"Task {task_id} not found, cannot requeue")
                return
            
            task = self.coordinator.tasks[task_id]
            
            # Check retry policy
            retry_policy = task.get("config", {}).get("retry_policy", {})
            max_retries = retry_policy.get("max_retries", 3)
            
            # Increment attempts
            if "attempts" not in task:
                task["attempts"] = 1
            task["attempts"] += 1
            
            # Check if max retries exceeded
            if task["attempts"] > max_retries:
                logger.warning(f"Max retries ({max_retries}) exceeded for task {task_id}")
                
                # Mark task as failed
                task["status"] = "failed"
                task["error"] = f"Max retries exceeded due to {reason}"
                
                # Add to failed tasks
                if hasattr(self.coordinator, 'failed_tasks'):
                    self.coordinator.failed_tasks.add(task_id)
                
                # Remove from running tasks
                if hasattr(self.coordinator, 'running_tasks'):
                    if task_id in self.coordinator.running_tasks:
                        del self.coordinator.running_tasks[task_id]
                
                # Record failure in database
                if hasattr(self.coordinator, 'db'):
                    self.coordinator.db.execute(
                    """
                    UPDATE distributed_tasks
                    SET status = 'failed', end_time = ?, worker_id = NULL,
                        attempts = ?
                    WHERE task_id = ?
                    """,
                    (datetime.now(), task["attempts"], task_id)
                    )
                    
                    # Record task execution history
                    self.coordinator.db.execute(
                    """
                    INSERT INTO task_execution_history (
                        task_id, worker_id, attempt, status, start_time,
                        end_time, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        worker_id,
                        task["attempts"],
                        "failed",
                        task.get("started", datetime.now().isoformat()),
                        datetime.now(),
                        f"Max retries exceeded due to {reason}"
                    )
                    )
                
                return
            
            # Requeue task
            logger.info(f"Requeuing task {task_id} (attempt {task['attempts']}/{max_retries})")
            
            # Update task status
            task["status"] = "pending"
            if "started" in task:
                del task["started"]
            if "worker_id" in task:
                del task["worker_id"]
            
            # Add to pending tasks
            if hasattr(self.coordinator, 'pending_tasks'):
                self.coordinator.pending_tasks.add(task_id)
            
            # Remove from running tasks
            if hasattr(self.coordinator, 'running_tasks'):
                if task_id in self.coordinator.running_tasks:
                    del self.coordinator.running_tasks[task_id]
            
            # Update task in database
            if hasattr(self.coordinator, 'db'):
                self.coordinator.db.execute(
                """
                UPDATE distributed_tasks
                SET status = 'pending', worker_id = NULL, start_time = NULL,
                    attempts = ?
                WHERE task_id = ?
                """,
                (task["attempts"], task_id)
                )
                
                # Record task recovery in database
                self.coordinator.db.execute(
                """
                INSERT INTO task_recovery_history (
                    task_id, worker_id, recovery_time, attempt_number,
                    success, recovery_type, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    worker_id,
                    datetime.now(),
                    task["attempts"],
                    True,
                    "requeue",
                    json.dumps({
                        "reason": reason,
                        "requeue_time": datetime.now().isoformat(),
                        "max_retries": max_retries
                    })
                )
                )
            
            # Try to assign task immediately if we have available workers
            if hasattr(self.coordinator, '_assign_pending_tasks'):
                await self.coordinator._assign_pending_tasks()
            
        except Exception as e:
            logger.error(f"Error requeuing task {task_id}: {str(e)}")

class DatabaseRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for database failures."""
    
    def __init__(self, coordinator):
        """
        Initialize the database recovery strategy.
        
        Args:
            coordinator: Reference to the coordinator instance
        """
        super().__init__(coordinator, "database_recovery", RecoveryLevel.HIGH)
    
    async def _execute_impl(self, error_info: Dict[str, Any]) -> bool:
        """
        Implement database recovery strategy.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Get error category
        category = error_info.get("category", ErrorCategory.UNKNOWN.value)
        
        # Get db_path
        if not hasattr(self.coordinator, 'db_path'):
            logger.error("Database recovery strategy requires coordinator.db_path")
            return False
        
        db_path = self.coordinator.db_path
        
        # Different strategies for different error types
        if category == ErrorCategory.DB_CONNECTION.value:
            # Connection issue - try to reconnect
            return await self._handle_connection_error(db_path)
        elif category == ErrorCategory.DB_QUERY.value:
            # Query error - try to fix query
            return await self._handle_query_error(error_info)
        elif category == ErrorCategory.DB_INTEGRITY.value:
            # Integrity issue - try to restore from backup
            return await self._handle_integrity_error(db_path)
        else:
            # Unknown database error - try generic recovery
            return await self._handle_generic_error(db_path)
    
    async def _handle_connection_error(self, db_path: str) -> bool:
        """
        Handle database connection error.
        
        Args:
            db_path: Path to database
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Attempting to reconnect to database at {db_path}")
        
        try:
            # Close existing connection if any
            if hasattr(self.coordinator, 'db') and self.coordinator.db:
                try:
                    self.coordinator.db.close()
                except Exception:
                    pass
            
            # Wait a bit before reconnecting
            await asyncio.sleep(1.0)
            
            # Try to reconnect
            import duckdb
            self.coordinator.db = duckdb.connect(db_path)
            
            # Verify connection with a simple query
            result = self.coordinator.db.execute("SELECT 1").fetchone()
            if result and result[0] == 1:
                logger.info(f"Successfully reconnected to database at {db_path}")
                return True
            else:
                logger.warning(f"Reconnection to database at {db_path} returned unexpected result")
                return False
        except Exception as e:
            logger.error(f"Failed to reconnect to database at {db_path}: {str(e)}")
            return False
    
    async def _handle_query_error(self, error_info: Dict[str, Any]) -> bool:
        """
        Handle database query error.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Get the query
        query = error_info.get("query")
        if not query:
            logger.error("Query error recovery requires the query")
            return False
        
        # Get the parameters
        params = error_info.get("params", [])
        
        # Get the error message
        error_message = error_info.get("error", "")
        
        logger.info(f"Attempting to recover from query error: {error_message}")
        
        try:
            # Some basic query fixes
            fixed_query = query
            
            # Try to fix some common errors
            if "no such table" in error_message.lower():
                # Table doesn't exist, check schema
                if hasattr(self.coordinator, '_create_schema'):
                    logger.info("Attempting to recreate schema")
                    await asyncio.to_thread(self.coordinator._create_schema)
                    
                    # Retry the query
                    self.coordinator.db.execute(query, params)
                    return True
            elif "syntax error" in error_message.lower():
                # Syntax error, not much we can do automatically
                logger.warning(f"Query syntax error, cannot auto-fix: {query}")
                return False
            elif "constraint failed" in error_message.lower():
                # Constraint violation, more complex to handle
                logger.warning(f"Constraint violation, cannot auto-fix: {query}")
                return False
            else:
                # Unknown query error, try running the query again
                logger.info(f"Retrying query: {query}")
                self.coordinator.db.execute(query, params)
                return True
                
            return False
        except Exception as e:
            logger.error(f"Failed to fix query error: {str(e)}")
            return False
    
    async def _handle_integrity_error(self, db_path: str) -> bool:
        """
        Handle database integrity error.
        
        Args:
            db_path: Path to database
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Attempting to recover from database integrity error for {db_path}")
        
        try:
            # Close existing connection
            if hasattr(self.coordinator, 'db') and self.coordinator.db:
                try:
                    self.coordinator.db.close()
                except Exception:
                    pass
            
            # Try to find a backup
            import os
            import glob
            
            # Check for backups in the same directory
            db_dir = os.path.dirname(db_path)
            db_name = os.path.basename(db_path)
            backup_pattern = os.path.join(db_dir, f"{db_name}.backup*")
            
            backup_files = glob.glob(backup_pattern)
            if not backup_files:
                logger.warning(f"No backup files found matching {backup_pattern}")
                return False
            
            # Sort backups by modification time (newest first)
            backup_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            
            # Try to restore from the newest backup
            newest_backup = backup_files[0]
            logger.info(f"Attempting to restore from backup: {newest_backup}")
            
            # Rename current database to .corrupted
            corrupted_path = f"{db_path}.corrupted.{int(time.time())}"
            os.rename(db_path, corrupted_path)
            
            # Copy backup to original path
            import shutil
            shutil.copy2(newest_backup, db_path)
            
            # Try to reconnect
            import duckdb
            self.coordinator.db = duckdb.connect(db_path)
            
            # Verify connection with a simple query
            result = self.coordinator.db.execute("SELECT 1").fetchone()
            if result and result[0] == 1:
                logger.info(f"Successfully restored database from backup: {newest_backup}")
                return True
            else:
                logger.warning(f"Database restored from {newest_backup} but verification failed")
                return False
        except Exception as e:
            logger.error(f"Failed to restore database from backup: {str(e)}")
            return False
    
    async def _handle_generic_error(self, db_path: str) -> bool:
        """
        Handle generic database error.
        
        Args:
            db_path: Path to database
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Attempting generic recovery for database at {db_path}")
        
        try:
            # Close existing connection if any
            if hasattr(self.coordinator, 'db') and self.coordinator.db:
                try:
                    self.coordinator.db.close()
                except Exception:
                    pass
            
            # Wait a bit before reconnecting
            await asyncio.sleep(2.0)
            
            # Try to reconnect
            import duckdb
            self.coordinator.db = duckdb.connect(db_path)
            
            # Verify connection with a simple query
            try:
                result = self.coordinator.db.execute("SELECT 1").fetchone()
                if result and result[0] == 1:
                    logger.info(f"Successfully reconnected to database at {db_path}")
                    
                    # Try to recreate schema
                    if hasattr(self.coordinator, '_create_schema'):
                        logger.info("Recreating database schema")
                        await asyncio.to_thread(self.coordinator._create_schema)
                    
                    return True
                else:
                    logger.warning(f"Reconnection to database at {db_path} returned unexpected result")
                    return False
            except Exception as e:
                logger.error(f"Failed to verify database connection: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Failed to perform generic database recovery: {str(e)}")
            return False

class CoordinatorRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for coordinator failures."""
    
    def __init__(self, coordinator):
        """
        Initialize the coordinator recovery strategy.
        
        Args:
            coordinator: Reference to the coordinator instance
        """
        super().__init__(coordinator, "coordinator_recovery", RecoveryLevel.CRITICAL)
    
    async def _execute_impl(self, error_info: Dict[str, Any]) -> bool:
        """
        Implement coordinator recovery strategy.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Get error category
        category = error_info.get("category", ErrorCategory.UNKNOWN.value)
        
        # Different strategies for different error types
        if category == ErrorCategory.COORDINATOR_ERROR.value:
            # Internal error - try to recover from error
            return await self._handle_coordinator_error(error_info)
        elif category == ErrorCategory.STATE_ERROR.value:
            # State error - try to recover distributed state
            return await self._handle_state_error(error_info)
        elif category == ErrorCategory.COORDINATOR_CRASH.value:
            # Crash - try to restart coordinator (but this is running inside the coordinator...)
            return await self._handle_coordinator_crash(error_info)
        else:
            # Unknown coordinator error - try generic recovery
            return await self._handle_generic_error(error_info)
    
    async def _handle_coordinator_error(self, error_info: Dict[str, Any]) -> bool:
        """
        Handle coordinator internal error.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info("Attempting to recover from coordinator internal error")
        
        try:
            # Reset internal state
            
            # Re-initialize worker connections if needed
            if hasattr(self.coordinator, 'worker_connections'):
                for worker_id, ws in list(self.coordinator.worker_connections.items()):
                    try:
                        await ws.close()
                    except Exception:
                        pass
                
                self.coordinator.worker_connections = {}
            
            # Update worker statuses
            if hasattr(self.coordinator, 'workers') and hasattr(self.coordinator, 'db'):
                try:
                    # Update workers in database
                    for worker_id, worker in list(self.coordinator.workers.items()):
                        if worker.get("status") == "active":
                            # Set to recovery_pending to allow reconnection
                            self.coordinator.db.execute(
                            """
                            UPDATE worker_nodes
                            SET status = 'recovery_pending'
                            WHERE worker_id = ?
                            """,
                            (worker_id,)
                            )
                            
                            worker["status"] = "recovery_pending"
                except Exception as e:
                    logger.error(f"Error updating worker statuses: {str(e)}")
            
            # Reset task assignments
            if hasattr(self.coordinator, 'running_tasks') and hasattr(self.coordinator, 'pending_tasks'):
                for task_id, worker_id in list(self.coordinator.running_tasks.items()):
                    if task_id in self.coordinator.tasks:
                        # Reset task to pending
                        task = self.coordinator.tasks[task_id]
                        task["status"] = "pending"
                        if "started" in task:
                            del task["started"]
                        if "worker_id" in task:
                            del task["worker_id"]
                        
                        # Add to pending tasks
                        self.coordinator.pending_tasks.add(task_id)
                
                # Clear running tasks
                self.coordinator.running_tasks = {}
            
            # Also update tasks in database
            if hasattr(self.coordinator, 'db'):
                try:
                    # Update tasks that are running but might be stalled
                    self.coordinator.db.execute(
                    """
                    UPDATE distributed_tasks
                    SET status = 'pending', worker_id = NULL, start_time = NULL
                    WHERE status = 'running' AND start_time < ?
                    """,
                    (datetime.now() - timedelta(minutes=30),)  # Reset tasks running for >30 minutes
                    )
                except Exception as e:
                    logger.error(f"Error updating task statuses in database: {str(e)}")
            
            logger.info("Successfully reset coordinator internal state")
            return True
        except Exception as e:
            logger.error(f"Failed to handle coordinator error: {str(e)}")
            return False
    
    async def _handle_state_error(self, error_info: Dict[str, Any]) -> bool:
        """
        Handle distributed state error.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info("Attempting to recover from distributed state error")
        
        try:
            # Check for distributed state manager
            if hasattr(self.coordinator, 'state_manager'):
                logger.info("Recreating distributed state from database")
                
                # Recreate worker state from database
                if hasattr(self.coordinator, 'db'):
                    try:
                        # Load workers from database
                        result = self.coordinator.db.execute(
                        """
                        SELECT 
                            worker_id, hostname, registration_time, last_heartbeat,
                            status, capabilities, hardware_metrics, tags
                        FROM worker_nodes
                        """).fetchall()
                        
                        workers = {}
                        for row in result:
                            worker_id, hostname, reg_time, last_heartbeat, status, capabilities, hardware_metrics, tags = row
                            
                            workers[worker_id] = {
                                "worker_id": worker_id,
                                "hostname": hostname,
                                "registration_time": reg_time,
                                "last_heartbeat": last_heartbeat,
                                "status": status,
                                "capabilities": json.loads(capabilities) if capabilities else {},
                                "hardware_metrics": json.loads(hardware_metrics) if hardware_metrics else {},
                                "tags": json.loads(tags) if tags else {}
                            }
                        
                        # Update worker state in state manager
                        self.coordinator.state_manager.update_batch("workers", workers)
                        logger.info(f"Recovered {len(workers)} workers from database")
                    except Exception as e:
                        logger.error(f"Error recovering workers from database: {str(e)}")
                    
                    try:
                        # Load tasks from database
                        result = self.coordinator.db.execute(
                        """
                        SELECT 
                            task_id, type, priority, status, create_time, 
                            start_time, end_time, worker_id, attempts, config, requirements
                        FROM distributed_tasks
                        WHERE status IN ('pending', 'running')
                        """).fetchall()
                        
                        tasks = {}
                        for row in result:
                            task_id, task_type, priority, status, create_time, start_time, end_time, worker_id, attempts, config, requirements = row
                            
                            tasks[task_id] = {
                                "task_id": task_id,
                                "type": task_type,
                                "priority": priority,
                                "status": status,
                                "created": create_time,
                                "started": start_time,
                                "ended": end_time,
                                "worker_id": worker_id,
                                "attempts": attempts,
                                "config": json.loads(config) if config else {},
                                "requirements": json.loads(requirements) if requirements else {}
                            }
                        
                        # Update task state in state manager
                        self.coordinator.state_manager.update_batch("tasks", tasks)
                        logger.info(f"Recovered {len(tasks)} tasks from database")
                    except Exception as e:
                        logger.error(f"Error recovering tasks from database: {str(e)}")
                
                # Force state synchronization
                self.coordinator.state_manager.changes_pending = True
                
                logger.info("Distributed state recovery completed")
                return True
            else:
                logger.warning("No distributed state manager available for recovery")
                return False
        except Exception as e:
            logger.error(f"Failed to handle state error: {str(e)}")
            return False
    
    async def _handle_coordinator_crash(self, error_info: Dict[str, Any]) -> bool:
        """
        Handle coordinator crash.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info("Attempting to recover from coordinator crash")
        
        # Since this is running inside the coordinator, we can only do soft recovery
        try:
            # Reset internal state (similar to coordinator_error)
            await self._handle_coordinator_error(error_info)
            
            # Try to reconnect database
            if hasattr(self.coordinator, 'db_path'):
                db_recovery = DatabaseRecoveryStrategy(self.coordinator)
                db_recovery_success = await db_recovery.execute({
                    "category": ErrorCategory.DB_CONNECTION.value
                })
                
                if not db_recovery_success:
                    logger.warning("Database recovery failed during coordinator crash recovery")
            
            # If we have redundancy manager, try to restore from backup node
            if hasattr(self.coordinator, 'redundancy_manager'):
                logger.info("Attempting to restore state from backup coordinator node")
                
                try:
                    # Force state synchronization from other nodes
                    await self.coordinator.redundancy_manager._sync_state_from_leader()
                    logger.info("State synchronized from backup coordinator node")
                except Exception as e:
                    logger.error(f"Error synchronizing state from backup node: {str(e)}")
            
            logger.info("Coordinator crash recovery completed")
            return True
        except Exception as e:
            logger.error(f"Failed to handle coordinator crash: {str(e)}")
            return False
    
    async def _handle_generic_error(self, error_info: Dict[str, Any]) -> bool:
        """
        Handle generic coordinator error.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info("Attempting generic coordinator recovery")
        
        try:
            # Combine coordinator_error and state_error recovery
            await self._handle_coordinator_error(error_info)
            await self._handle_state_error(error_info)
            
            logger.info("Generic coordinator recovery completed")
            return True
        except Exception as e:
            logger.error(f"Failed to handle generic coordinator error: {str(e)}")
            return False

class SystemRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for system-wide failures."""
    
    def __init__(self, coordinator):
        """
        Initialize the system recovery strategy.
        
        Args:
            coordinator: Reference to the coordinator instance
        """
        super().__init__(coordinator, "system_recovery", RecoveryLevel.CRITICAL)
    
    async def _execute_impl(self, error_info: Dict[str, Any]) -> bool:
        """
        Implement system recovery strategy.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Get error category
        category = error_info.get("category", ErrorCategory.UNKNOWN.value)
        
        # Different strategies for different error types
        if category == ErrorCategory.SYSTEM_RESOURCE.value:
            # Resource exhaustion - try to free resources
            return await self._handle_resource_exhaustion(error_info)
        elif category == ErrorCategory.DISK_FULL.value:
            # Disk full - try to free disk space
            return await self._handle_disk_full(error_info)
        elif category == ErrorCategory.SYSTEM_OVERLOAD.value:
            # System overload - try load shedding
            return await self._handle_system_overload(error_info)
        else:
            # Unknown system error - try generic recovery
            return await self._handle_generic_error(error_info)
    
    async def _handle_resource_exhaustion(self, error_info: Dict[str, Any]) -> bool:
        """
        Handle system resource exhaustion.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info("Attempting to recover from system resource exhaustion")
        
        try:
            # Record emergency recovery in database
            if hasattr(self.coordinator, 'db'):
                try:
                    self.coordinator.db.execute(
                    """
                    INSERT INTO system_recovery_history (
                        recovery_time, system_status, affected_workers,
                        affected_tasks, details
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(),
                        "resource_exhaustion",
                        len(self.coordinator.workers) if hasattr(self.coordinator, 'workers') else 0,
                        len(self.coordinator.running_tasks) if hasattr(self.coordinator, 'running_tasks') else 0,
                        json.dumps({
                            "resource_exhaustion": True,
                            "error_info": error_info,
                            "initiated_time": datetime.now().isoformat()
                        })
                    )
                    )
                except Exception as e:
                    logger.error(f"Error recording system recovery in database: {str(e)}")
            
            # Reduce system load
            
            # 1. Pause task assignment
            if hasattr(self.coordinator, 'task_scheduler'):
                self.coordinator.task_scheduler.pause_scheduling = True
                logger.info("Paused task assignment to reduce system load")
            
            # 2. Reduce worker load if possible
            if hasattr(self.coordinator, 'workers') and hasattr(self.coordinator, 'db'):
                active_workers = sum(1 for w in self.coordinator.workers.values() if w.get("status") == "active")
                
                # If too many active workers, reduce load
                if active_workers > 5:  # Arbitrary threshold
                    logger.info(f"Reducing active workers from {active_workers} to 5")
                    
                    # Sort workers by active tasks
                    worker_tasks = {}
                    for task_id, worker_id in self.coordinator.running_tasks.items():
                        worker_tasks[worker_id] = worker_tasks.get(worker_id, 0) + 1
                    
                    # Sort workers by task count (highest first)
                    sorted_workers = sorted(
                        [w for w in self.coordinator.workers.values() if w.get("status") == "active"],
                        key=lambda w: worker_tasks.get(w.get("worker_id", ""), 0),
                        reverse=True
                    )
                    
                    # Keep the 5 busiest workers, pause the rest
                    for worker in sorted_workers[5:]:
                        worker_id = worker.get("worker_id")
                        if worker_id:
                            # Set status to paused in database
                            self.coordinator.db.execute(
                            """
                            UPDATE worker_nodes
                            SET status = 'paused'
                            WHERE worker_id = ?
                            """,
                            (worker_id,)
                            )
                            
                            # Update in-memory status
                            worker["status"] = "paused"
                            
                            logger.info(f"Paused worker {worker_id} to reduce system load")
            
            # 3. Pause non-critical worker heartbeats
            if hasattr(self.coordinator, 'health_monitor'):
                self.coordinator.health_monitor.check_interval = 30  # Increase health check interval
                logger.info("Increased health check interval to reduce system load")
            
            # 4. Reduce database operations
            if hasattr(self.coordinator, 'db_write_interval'):
                self.coordinator.db_write_interval = 60  # Increase database write interval
                logger.info("Increased database write interval to reduce system load")
            
            # Wait for resource situation to improve
            await asyncio.sleep(10)
            
            # Resume normal operation
            if hasattr(self.coordinator, 'task_scheduler'):
                self.coordinator.task_scheduler.pause_scheduling = False
                logger.info("Resumed task assignment")
            
            logger.info("System resource recovery completed")
            return True
        except Exception as e:
            logger.error(f"Failed to handle system resource exhaustion: {str(e)}")
            return False
    
    async def _handle_disk_full(self, error_info: Dict[str, Any]) -> bool:
        """
        Handle disk full error.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info("Attempting to recover from disk full error")
        
        try:
            # Try to free disk space by cleaning up temporary files
            import os
            import tempfile
            import glob
            
            # 1. Clean up temporary directory
            temp_dir = tempfile.gettempdir()
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            
            # Delete old temporary files (older than 1 day)
            deleted_bytes = 0
            deleted_count = 0
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_age = time.time() - os.path.getmtime(file_path)
                        if file_age > 86400:  # 1 day in seconds
                            file_size = os.path.getsize(file_path)
                            os.unlink(file_path)
                            deleted_bytes += file_size
                            deleted_count += 1
                    except Exception:
                        continue
            
            logger.info(f"Deleted {deleted_count} temporary files, freed {deleted_bytes / 1024 / 1024:.2f} MB")
            
            # 2. Clean up log files
            log_pattern = "*.log*"
            log_files = glob.glob(log_pattern)
            
            for log_file in log_files:
                try:
                    file_age = time.time() - os.path.getmtime(log_file)
                    if file_age > 604800:  # 7 days in seconds
                        file_size = os.path.getsize(log_file)
                        os.unlink(log_file)
                        deleted_bytes += file_size
                        deleted_count += 1
                except Exception:
                    continue
            
            logger.info(f"Deleted {deleted_count} old log files, freed {deleted_bytes / 1024 / 1024:.2f} MB total")
            
            # 3. Clean up old database backups if any
            if hasattr(self.coordinator, 'db_path'):
                db_dir = os.path.dirname(self.coordinator.db_path)
                db_name = os.path.basename(self.coordinator.db_path)
                backup_pattern = os.path.join(db_dir, f"{db_name}.backup*")
                
                backup_files = glob.glob(backup_pattern)
                
                # Sort backups by modification time (oldest first)
                backup_files.sort(key=lambda f: os.path.getmtime(f))
                
                # Keep latest 2 backups, delete the rest
                for backup_file in backup_files[:-2]:
                    try:
                        file_size = os.path.getsize(backup_file)
                        os.unlink(backup_file)
                        deleted_bytes += file_size
                        deleted_count += 1
                    except Exception:
                        continue
                
                logger.info(f"Deleted {deleted_count} old database backups, freed {deleted_bytes / 1024 / 1024:.2f} MB total")
            
            return True
        except Exception as e:
            logger.error(f"Failed to handle disk full error: {str(e)}")
            return False
    
    async def _handle_system_overload(self, error_info: Dict[str, Any]) -> bool:
        """
        Handle system overload.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info("Attempting to recover from system overload")
        
        try:
            # Record emergency recovery in database
            if hasattr(self.coordinator, 'db'):
                try:
                    self.coordinator.db.execute(
                    """
                    INSERT INTO system_recovery_history (
                        recovery_time, system_status, affected_workers,
                        affected_tasks, details
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(),
                        "system_overload",
                        len(self.coordinator.workers) if hasattr(self.coordinator, 'workers') else 0,
                        len(self.coordinator.running_tasks) if hasattr(self.coordinator, 'running_tasks') else 0,
                        json.dumps({
                            "system_overload": True,
                            "error_info": error_info,
                            "initiated_time": datetime.now().isoformat()
                        })
                    )
                    )
                except Exception as e:
                    logger.error(f"Error recording system recovery in database: {str(e)}")
            
            # 1. Pause task scheduling
            if hasattr(self.coordinator, 'task_scheduler'):
                self.coordinator.task_scheduler.pause_scheduling = True
                logger.info("Paused task scheduling")
            
            # 2. Pause non-essential services
            if hasattr(self.coordinator, 'health_monitor'):
                self.coordinator.health_monitor.check_interval = 60  # Increase health check interval
                logger.info("Increased health check interval")
            
            # 3. Reduce concurrency
            if hasattr(self.coordinator, 'max_concurrent_tasks'):
                self.coordinator.max_concurrent_tasks = max(1, self.coordinator.max_concurrent_tasks // 2)
                logger.info(f"Reduced max concurrent tasks to {self.coordinator.max_concurrent_tasks}")
            
            # 4. Limit active workers
            if hasattr(self.coordinator, 'workers') and hasattr(self.coordinator, 'db'):
                active_workers = [w for w in self.coordinator.workers.values() if w.get("status") == "active"]
                
                # Keep only half of active workers
                limit = max(1, len(active_workers) // 2)
                
                for worker in active_workers[limit:]:
                    worker_id = worker.get("worker_id")
                    if worker_id:
                        # Set status to paused in database
                        self.coordinator.db.execute(
                        """
                        UPDATE worker_nodes
                        SET status = 'paused'
                        WHERE worker_id = ?
                        """,
                        (worker_id,)
                        )
                        
                        # Update in-memory status
                        worker["status"] = "paused"
                        
                        logger.info(f"Paused worker {worker_id} due to system overload")
            
            # Wait for a bit
            await asyncio.sleep(30)
            
            # Gradually resume services
            
            # 1. Resume task scheduling
            if hasattr(self.coordinator, 'task_scheduler'):
                self.coordinator.task_scheduler.pause_scheduling = False
                logger.info("Resumed task scheduling")
            
            # 2. Gradually restore settings
            if hasattr(self.coordinator, 'health_monitor'):
                self.coordinator.health_monitor.check_interval = 30  # Partially restore health check interval
                logger.info("Partially restored health check interval")
            
            logger.info("System overload recovery completed")
            return True
        except Exception as e:
            logger.error(f"Failed to handle system overload: {str(e)}")
            return False
    
    async def _handle_generic_error(self, error_info: Dict[str, Any]) -> bool:
        """
        Handle generic system error.
        
        Args:
            error_info: Information about the error
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info("Attempting generic system recovery")
        
        try:
            # Record recovery in database
            if hasattr(self.coordinator, 'db'):
                try:
                    self.coordinator.db.execute(
                    """
                    INSERT INTO system_recovery_history (
                        recovery_time, system_status, affected_workers,
                        affected_tasks, details
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(),
                        "generic_error",
                        len(self.coordinator.workers) if hasattr(self.coordinator, 'workers') else 0,
                        len(self.coordinator.running_tasks) if hasattr(self.coordinator, 'running_tasks') else 0,
                        json.dumps({
                            "generic_system_error": True,
                            "error_info": error_info,
                            "initiated_time": datetime.now().isoformat()
                        })
                    )
                    )
                except Exception as e:
                    logger.error(f"Error recording system recovery in database: {str(e)}")
            
            # Combine strategies from resource and overload handlers
            await self._handle_resource_exhaustion(error_info)
            await self._handle_system_overload(error_info)
            
            logger.info("Generic system recovery completed")
            return True
        except Exception as e:
            logger.error(f"Failed to handle generic system error: {str(e)}")
            return False

class EnhancedErrorRecoveryManager:
    """Manager for enhanced error recovery strategies."""
    
    def __init__(self, coordinator):
        """
        Initialize the error recovery manager.
        
        Args:
            coordinator: Reference to the coordinator instance
        """
        self.coordinator = coordinator
        
        # Initialize recovery strategies
        self.strategies = {
            # Basic strategies
            "retry": RetryStrategy(coordinator),
            
            # Category-specific strategies
            "worker": WorkerRecoveryStrategy(coordinator),
            "database": DatabaseRecoveryStrategy(coordinator),
            "coordinator": CoordinatorRecoveryStrategy(coordinator),
            "system": SystemRecoveryStrategy(coordinator),
        }
        
        # Error categorization mapping
        self.error_type_to_strategy = {
            # Connection errors
            ErrorCategory.CONNECTION.value: "retry",
            ErrorCategory.TIMEOUT.value: "retry",
            
            # Worker errors
            ErrorCategory.WORKER_OFFLINE.value: "worker",
            ErrorCategory.WORKER_CRASH.value: "worker",
            ErrorCategory.WORKER_RESOURCE.value: "worker",
            
            # Task errors
            ErrorCategory.TASK_TIMEOUT.value: "retry",
            ErrorCategory.TASK_ERROR.value: "retry",
            ErrorCategory.TASK_RESOURCE.value: "retry",
            
            # Database errors
            ErrorCategory.DB_CONNECTION.value: "database",
            ErrorCategory.DB_QUERY.value: "database",
            ErrorCategory.DB_INTEGRITY.value: "database",
            
            # Coordinator errors
            ErrorCategory.COORDINATOR_ERROR.value: "coordinator",
            ErrorCategory.STATE_ERROR.value: "coordinator",
            ErrorCategory.COORDINATOR_CRASH.value: "coordinator",
            
            # Security errors
            ErrorCategory.AUTH_ERROR.value: "retry",
            ErrorCategory.UNAUTHORIZED.value: "retry",
            
            # System errors
            ErrorCategory.SYSTEM_RESOURCE.value: "system",
            ErrorCategory.DISK_FULL.value: "system",
            ErrorCategory.SYSTEM_OVERLOAD.value: "system",
            
            # Unknown errors
            ErrorCategory.UNKNOWN.value: "retry",
        }
        
        # Recovery history
        self.recovery_history = []
        self.max_history_size = 100
        
        logger.info("EnhancedErrorRecoveryManager initialized with 5 strategies")
    
    def get_strategy_for_error(self, error_info: Dict[str, Any]) -> Optional[RecoveryStrategy]:
        """
        Get the appropriate recovery strategy for an error.
        
        Args:
            error_info: Information about the error
            
        Returns:
            Recovery strategy or None if no strategy is available
        """
        # Get error category
        category = error_info.get("category", ErrorCategory.UNKNOWN.value)
        
        # Get strategy name from mapping
        strategy_name = self.error_type_to_strategy.get(category, "retry")
        
        # Get strategy
        return self.strategies.get(strategy_name)
    
    def categorize_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Categorize an error.
        
        Args:
            error: Exception object
            context: Optional context information
            
        Returns:
            Dictionary with error information
        """
        error_type = type(error).__name__
        error_message = str(error)
        error_traceback = traceback.format_exc()
        
        # Default category
        category = ErrorCategory.UNKNOWN.value
        
        # Context defaults
        context = context or {}
        
        # Categorize based on error type and message
        if error_type in ["ConnectionError", "ConnectionRefusedError", "ConnectionResetError"]:
            category = ErrorCategory.CONNECTION.value
        elif error_type in ["TimeoutError", "asyncio.TimeoutError"]:
            category = ErrorCategory.TIMEOUT.value
        elif error_type in ["DatabaseError", "OperationalError"]:
            if "connection" in error_message.lower():
                category = ErrorCategory.DB_CONNECTION.value
            elif "query" in error_message.lower() or "sql" in error_message.lower():
                category = ErrorCategory.DB_QUERY.value
            elif "integrity" in error_message.lower() or "constraint" in error_message.lower():
                category = ErrorCategory.DB_INTEGRITY.value
            else:
                category = ErrorCategory.DB_CONNECTION.value
        elif "worker" in context.get("component", "").lower():
            if "offline" in error_message.lower() or "disconnected" in error_message.lower():
                category = ErrorCategory.WORKER_OFFLINE.value
            elif "crash" in error_message.lower():
                category = ErrorCategory.WORKER_CRASH.value
            elif "resource" in error_message.lower() or "memory" in error_message.lower():
                category = ErrorCategory.WORKER_RESOURCE.value
            else:
                category = ErrorCategory.WORKER_OFFLINE.value
        elif "task" in context.get("component", "").lower():
            if "timeout" in error_message.lower():
                category = ErrorCategory.TASK_TIMEOUT.value
            elif "resource" in error_message.lower() or "memory" in error_message.lower():
                category = ErrorCategory.TASK_RESOURCE.value
            else:
                category = ErrorCategory.TASK_ERROR.value
        elif "coordinator" in context.get("component", "").lower():
            if "state" in error_message.lower():
                category = ErrorCategory.STATE_ERROR.value
            elif "crash" in error_message.lower():
                category = ErrorCategory.COORDINATOR_CRASH.value
            else:
                category = ErrorCategory.COORDINATOR_ERROR.value
        elif "auth" in error_message.lower() or "authentication" in error_message.lower():
            category = ErrorCategory.AUTH_ERROR.value
        elif "unauthorized" in error_message.lower() or "permission" in error_message.lower():
            category = ErrorCategory.UNAUTHORIZED.value
        elif "disk" in error_message.lower() and "full" in error_message.lower():
            category = ErrorCategory.DISK_FULL.value
        elif "system" in error_message.lower() and "overload" in error_message.lower():
            category = ErrorCategory.SYSTEM_OVERLOAD.value
        elif "resource" in error_message.lower() or "memory" in error_message.lower():
            category = ErrorCategory.SYSTEM_RESOURCE.value
        
        # Create error info
        error_info = {
            "type": error_type,
            "message": error_message,
            "traceback": error_traceback,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "context": context
        }
        
        return error_info
    
    async def recover(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Recover from an error.
        
        Args:
            error: Exception object
            context: Optional context information
            
        Returns:
            Tuple of (success, recovery_info)
        """
        # Categorize error
        error_info = self.categorize_error(error, context)
        
        # Get strategy
        strategy = self.get_strategy_for_error(error_info)
        
        # Recovery info
        recovery_info = {
            "error_info": error_info,
            "strategy": strategy.name if strategy else None,
            "started": datetime.now().isoformat(),
            "success": False,
            "duration": 0
        }
        
        if not strategy:
            logger.error(f"No recovery strategy available for error category: {error_info['category']}")
            return False, recovery_info
        
        # Record start time
        start_time = time.time()
        
        try:
            # Execute strategy
            success = await strategy.execute(error_info)
            
            # Record end time
            end_time = time.time()
            duration = end_time - start_time
            
            # Update recovery info
            recovery_info["success"] = success
            recovery_info["duration"] = duration
            recovery_info["completed"] = datetime.now().isoformat()
            
            # Add to history
            self.recovery_history.append(recovery_info)
            
            # Trim history if needed
            if len(self.recovery_history) > self.max_history_size:
                self.recovery_history = self.recovery_history[-self.max_history_size:]
            
            return success, recovery_info
        except Exception as e:
            logger.error(f"Error during recovery: {str(e)}")
            
            # Record end time
            end_time = time.time()
            duration = end_time - start_time
            
            # Update recovery info
            recovery_info["success"] = False
            recovery_info["duration"] = duration
            recovery_info["completed"] = datetime.now().isoformat()
            recovery_info["recovery_error"] = str(e)
            
            # Add to history
            self.recovery_history.append(recovery_info)
            
            # Trim history if needed
            if len(self.recovery_history) > self.max_history_size:
                self.recovery_history = self.recovery_history[-self.max_history_size:]
            
            return False, recovery_info
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """
        Get recovery history.
        
        Returns:
            List of recovery history entries
        """
        return self.recovery_history.copy()
    
    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all strategies.
        
        Returns:
            Dictionary mapping strategy names to statistics
        """
        return {name: strategy.get_strategy_info() for name, strategy in self.strategies.items()}
    
    def add_custom_strategy(self, name: str, strategy: RecoveryStrategy) -> bool:
        """
        Add a custom recovery strategy.
        
        Args:
            name: Name of the strategy
            strategy: Recovery strategy
            
        Returns:
            True if successful, False otherwise
        """
        if name in self.strategies:
            logger.warning(f"Recovery strategy {name} already exists, replacing")
        
        self.strategies[name] = strategy
        logger.info(f"Added custom recovery strategy: {name}")
        
        return True
    
    def register_error_category_strategy(self, category: ErrorCategory, strategy_name: str) -> bool:
        """
        Register a strategy for an error category.
        
        Args:
            category: Error category
            strategy_name: Name of strategy to use
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_name not in self.strategies:
            logger.error(f"Recovery strategy {strategy_name} not found")
            return False
        
        self.error_type_to_strategy[category.value] = strategy_name
        logger.info(f"Registered strategy {strategy_name} for error category {category.value}")
        
        return True
    
    def record_error_recovery(self, error_info: Dict[str, Any], strategy_name: str, success: bool, duration: float):
        """
        Record error recovery in the database.
        
        Args:
            error_info: Error information
            strategy_name: Name of strategy used
            success: Whether recovery was successful
            duration: Duration of recovery in seconds
        """
        if not hasattr(self.coordinator, 'db'):
            return
            
        try:
            # Record in database
            self.coordinator.db.execute(
            """
            INSERT INTO error_recovery_history (
                error_category, error_type, strategy_name, 
                success, duration, recovery_time, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                error_info.get("category"),
                error_info.get("type"),
                strategy_name,
                success,
                duration,
                datetime.now(),
                json.dumps({
                    "error_message": error_info.get("message"),
                    "error_context": error_info.get("context"),
                    "recovery_info": {
                        "success": success,
                        "duration": duration,
                        "strategy": strategy_name
                    }
                })
            )
            )
        except Exception as e:
            logger.error(f"Error recording recovery in database: {str(e)}")
    
    def ensure_error_recovery_table(self):
        """Ensure error recovery history table exists in the database."""
        if not hasattr(self.coordinator, 'db'):
            return
            
        try:
            # Create table if not exists
            self.coordinator.db.execute(
            """
            CREATE TABLE IF NOT EXISTS error_recovery_history (
                id INTEGER PRIMARY KEY,
                error_category VARCHAR,
                error_type VARCHAR,
                strategy_name VARCHAR,
                success BOOLEAN,
                duration FLOAT,
                recovery_time TIMESTAMP,
                details JSON
            )
            """)
            
            logger.info("Error recovery history table created or verified")
        except Exception as e:
            logger.error(f"Error ensuring error recovery table: {str(e)}")
    
    async def initialize(self):
        """Initialize the error recovery manager."""
        # Ensure database table
        self.ensure_error_recovery_table()
        
        logger.info("EnhancedErrorRecoveryManager initialized")