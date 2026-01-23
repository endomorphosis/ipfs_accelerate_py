#!/usr/bin/env python3
"""
Distributed Testing Framework - Health Monitoring Module

This module provides health monitoring functionality for the distributed testing framework.
It tracks worker health, monitors task execution, and implements auto-recovery mechanisms.

Usage:
    Import this module in coordinator.py to enable health monitoring features.
"""

import anyio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthMonitor:
    """Health monitor for distributed testing framework."""
    
    def __init__(
        self,
        coordinator,
        heartbeat_timeout: int = 30,
        task_timeout_multiplier: float = 3.0,
        auto_recovery: bool = True,
        check_interval: int = 10,
        max_recovery_attempts: int = 5
    ):
        """
        Initialize the health monitor.
        
        Args:
            coordinator: Reference to the coordinator instance
            heartbeat_timeout: Timeout for worker heartbeats in seconds
            task_timeout_multiplier: Multiplier for task timeout based on estimated execution time
            auto_recovery: Whether to enable automatic recovery
            check_interval: Interval for health checks in seconds
            max_recovery_attempts: Maximum number of recovery attempts per worker
        """
        self.coordinator = coordinator
        self.heartbeat_timeout = heartbeat_timeout
        self.task_timeout_multiplier = task_timeout_multiplier
        self.auto_recovery = auto_recovery
        self.check_interval = check_interval
        self.max_recovery_attempts = max_recovery_attempts
        
        # Task timeout tracking
        self.task_timeout_estimates: Dict[str, float] = {}
        
        # Worker health metrics
        self.worker_health_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Task health metrics
        self.task_health_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Recovery tracking
        self.recovery_attempts: Dict[str, int] = {}  # worker_id -> attempts
        self.task_recovery_attempts: Dict[str, int] = {}  # task_id -> attempts
        
        # Recovery pending checks
        self.recovery_pending: Dict[str, datetime] = {}  # worker_id -> check_after_time
        
        # System health status
        self.system_health_status = "healthy"  # healthy, degraded, critical
        
        logger.info("Health monitor initialized")
    
    async def start_monitoring(self):
        """Start the health monitoring loop."""
        logger.info("Starting health monitoring loop")
        
        # Create database tables if needed
        await self._create_database_tables()
        
        # Start background tasks
        # TODO: Replace with task group - asyncio.create_task(self._system_health_monitor())
        
        while True:
            try:
                # Check worker health
                await self.check_worker_health()
                
                # Check task health
                await self.check_task_health()
                
                # Update health metrics in database
                await self.update_health_metrics_in_db()
                
                # Check recovery status of workers in recovery_pending
                await self._check_recovery_pending()
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {str(e)}")
            
            # Sleep until next check
            await anyio.sleep(self.check_interval)
    
    async def _create_database_tables(self):
        """Create required database tables for health monitoring."""
        try:
            # Worker health metrics table
            self.coordinator.db.execute("""
            CREATE TABLE IF NOT EXISTS worker_health_metrics (
                id INTEGER PRIMARY KEY,
                worker_id VARCHAR,
                check_time TIMESTAMP,
                heartbeat_age_seconds FLOAT,
                status VARCHAR,
                metrics JSON
            )
            """)
            
            # Task health metrics table
            self.coordinator.db.execute("""
            CREATE TABLE IF NOT EXISTS task_health_metrics (
                id INTEGER PRIMARY KEY,
                task_id VARCHAR,
                worker_id VARCHAR,
                check_time TIMESTAMP,
                running_time_seconds FLOAT,
                timeout_seconds FLOAT,
                status VARCHAR,
                metrics JSON
            )
            """)
            
            # Worker recovery attempts table
            self.coordinator.db.execute("""
            CREATE TABLE IF NOT EXISTS worker_recovery_attempts (
                id INTEGER PRIMARY KEY,
                worker_id VARCHAR,
                attempt_number INTEGER,
                attempt_time TIMESTAMP,
                status VARCHAR,
                completion_time TIMESTAMP,
                success BOOLEAN,
                details JSON
            )
            """)
            
            # Task recovery attempts table
            self.coordinator.db.execute("""
            CREATE TABLE IF NOT EXISTS task_recovery_attempts (
                id INTEGER PRIMARY KEY,
                task_id VARCHAR,
                attempt_number INTEGER,
                attempt_time TIMESTAMP,
                status VARCHAR,
                worker_id VARCHAR,
                details JSON
            )
            """)
            
            # System health status table
            self.coordinator.db.execute("""
            CREATE TABLE IF NOT EXISTS system_health_status (
                id INTEGER PRIMARY KEY,
                check_time TIMESTAMP,
                status VARCHAR,
                active_workers INTEGER,
                total_workers INTEGER,
                active_percentage FLOAT,
                tasks_pending INTEGER,
                tasks_running INTEGER,
                tasks_completed INTEGER,
                tasks_failed INTEGER,
                details JSON
            )
            """)
            
            # Worker permanent failures table
            self.coordinator.db.execute("""
            CREATE TABLE IF NOT EXISTS worker_permanent_failures (
                id INTEGER PRIMARY KEY,
                worker_id VARCHAR,
                failure_time TIMESTAMP,
                recovery_attempts INTEGER,
                last_known_status VARCHAR,
                last_heartbeat VARCHAR,
                failure_reason VARCHAR
            )
            """)
            
            logger.info("Health monitoring database tables created")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
    
    async def check_worker_health(self):
        """Check health of all workers."""
        now = datetime.now()
        
        for worker_id, worker in list(self.coordinator.workers.items()):
            # Skip workers already marked as permanent failures
            if worker.get("status") == "permanent_failure":
                continue
                
            # Skip offline workers that are in recovery_pending
            if worker.get("status") == "recovery_pending" and worker_id in self.recovery_pending:
                continue
            
            # Check heartbeat timestamp
            last_heartbeat_str = worker.get("last_heartbeat", "1970-01-01T00:00:00")
            try:
                last_heartbeat = datetime.fromisoformat(last_heartbeat_str)
                heartbeat_age = (now - last_heartbeat).total_seconds()
            except (ValueError, TypeError):
                logger.warning(f"Invalid heartbeat timestamp for worker {worker_id}: {last_heartbeat_str}")
                heartbeat_age = float('inf')
            
            # Update health metrics
            self.worker_health_metrics[worker_id] = {
                "worker_id": worker_id,
                "last_heartbeat": last_heartbeat_str,
                "heartbeat_age_seconds": heartbeat_age,
                "status": worker.get("status", "unknown"),
                "check_time": now.isoformat(),
            }
            
            # Check if heartbeat is too old for active or idle workers
            if worker.get("status") in ["active", "idle"] and heartbeat_age > self.heartbeat_timeout:
                logger.warning(f"Worker {worker_id} heartbeat is {heartbeat_age:.1f} seconds old (timeout: {self.heartbeat_timeout}s)")
                
                # Mark worker as offline
                worker["status"] = "offline"
                
                # Update database
                self.coordinator.db.execute(
                """
                UPDATE worker_nodes
                SET status = 'offline', last_heartbeat = ?
                WHERE worker_id = ?
                """,
                (now, worker_id)
                )
                
                # Handle failed tasks
                await self.handle_worker_failure(worker_id)
                
                # Attempt recovery if enabled
                if self.auto_recovery:
                    await self.attempt_worker_recovery(worker_id)
    
    async def check_task_health(self):
        """Check health of all running tasks."""
        now = datetime.now()
        
        for task_id, worker_id in list(self.coordinator.running_tasks.items()):
            # Skip if task no longer exists
            if task_id not in self.coordinator.tasks:
                continue
            
            task = self.coordinator.tasks[task_id]
            
            # Calculate task running time
            if "started" in task:
                try:
                    started = datetime.fromisoformat(task["started"])
                    running_time = (now - started).total_seconds()
                except (ValueError, TypeError):
                    logger.warning(f"Invalid start time for task {task_id}: {task.get('started')}")
                    running_time = 0
            else:
                running_time = 0
            
            # Get timeout for this task
            timeout = self.get_task_timeout(task_id, task)
            
            # Update health metrics
            self.task_health_metrics[task_id] = {
                "task_id": task_id,
                "worker_id": worker_id,
                "running_time_seconds": running_time,
                "timeout_seconds": timeout,
                "status": task.get("status", "unknown"),
                "check_time": now.isoformat(),
            }
            
            # Check if task has timed out
            if running_time > timeout:
                logger.warning(f"Task {task_id} has been running for {running_time:.1f} seconds on worker {worker_id} (timeout: {timeout}s)")
                
                # Handle task timeout
                await self.handle_task_timeout(task_id, worker_id)
    
    def get_task_timeout(self, task_id: str, task: Dict[str, Any]) -> float:
        """
        Get timeout for a task.
        
        Args:
            task_id: Task ID
            task: Task information
            
        Returns:
            Timeout in seconds
        """
        # Check if task has explicit timeout in config
        task_config = task.get("config", {})
        if isinstance(task_config, dict) and "timeout_seconds" in task_config:
            return float(task_config["timeout_seconds"])
        
        # Check if we have an estimated timeout for this task type
        task_type = task.get("type", "unknown")
        estimated_timeout = self.task_timeout_estimates.get(f"{task_type}", None)
        
        if estimated_timeout is not None:
            # Apply multiplier for safety margin
            return estimated_timeout * self.task_timeout_multiplier
        
        # Default timeouts based on task type
        default_timeouts = {
            "benchmark": 600,  # 10 minutes
            "test": 300,       # 5 minutes
            "custom": 300,     # 5 minutes
        }
        
        return default_timeouts.get(task_type, 600)  # Default to 10 minutes
    
    async def handle_worker_failure(self, worker_id: str):
        """
        Handle worker failure.
        
        Args:
            worker_id: Worker ID
        """
        logger.info(f"Handling failure of worker {worker_id}")
        
        # Find all tasks assigned to this worker
        for task_id, assigned_worker_id in list(self.coordinator.running_tasks.items()):
            if assigned_worker_id == worker_id:
                logger.info(f"Marking task {task_id} as failed due to worker failure")
                
                # Update task status
                await self.coordinator._update_task_status(task_id, "failed", {
                    "error": f"Worker {worker_id} failed or disconnected"
                })
                
                # Update failure metrics in database
                self.coordinator.db.execute(
                """
                INSERT INTO task_execution_history (
                    task_id, worker_id, attempt, status, start_time, end_time,
                    execution_time_seconds, error_message, hardware_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id, worker_id,
                    self.coordinator.tasks[task_id].get("attempts", 1),
                    "failed",
                    datetime.fromisoformat(self.coordinator.tasks[task_id].get("started", datetime.now().isoformat())),
                    datetime.now(),
                    0,
                    f"Worker {worker_id} failed or disconnected",
                    "{}"
                )
                )
                
                # If auto-recovery is enabled, requeue the task
                if self.auto_recovery:
                    await self.requeue_task(task_id)
    
    async def handle_task_timeout(self, task_id: str, worker_id: str):
        """
        Handle task timeout.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID
        """
        logger.info(f"Handling timeout of task {task_id} on worker {worker_id}")
        
        # Update task status
        await self.coordinator._update_task_status(task_id, "failed", {
            "error": "Task execution timed out"
        })
        
        # Update timeout metrics in database
        self.coordinator.db.execute(
        """
        INSERT INTO task_execution_history (
            task_id, worker_id, attempt, status, start_time, end_time,
            execution_time_seconds, error_message, hardware_metrics
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id, worker_id,
            self.coordinator.tasks[task_id].get("attempts", 1),
            "failed",
            datetime.fromisoformat(self.coordinator.tasks[task_id].get("started", datetime.now().isoformat())),
            datetime.now(),
            0,
            "Task execution timed out",
            "{}"
        )
        )
        
        # If auto-recovery is enabled, requeue the task
        if self.auto_recovery:
            await self.requeue_task(task_id)
    
    async def attempt_worker_recovery(self, worker_id: str):
        """
        Attempt to recover a failed worker.
        
        Args:
            worker_id: Worker ID
        """
        # Increment recovery attempts
        self.recovery_attempts[worker_id] = self.recovery_attempts.get(worker_id, 0) + 1
        attempts = self.recovery_attempts[worker_id]
        
        # Log recovery attempt
        logger.info(f"Attempting recovery of worker {worker_id} (attempt {attempts})")
        
        # If we've tried too many times, mark as permanently offline
        if attempts > self.max_recovery_attempts:
            logger.warning(f"Worker {worker_id} has failed {attempts} times, marking as permanently offline")
            
            # Update worker status in database
            self.coordinator.db.execute(
            """
            UPDATE worker_nodes
            SET status = 'permanent_failure', 
                last_heartbeat = ?
            WHERE worker_id = ?
            """,
            (datetime.now(), worker_id)
            )
            
            # Update in-memory status
            if worker_id in self.coordinator.workers:
                self.coordinator.workers[worker_id]["status"] = "permanent_failure"
                
            # Log the permanent failure
            logger.error(f"Worker {worker_id} has been marked as permanently failed after {attempts} recovery attempts")
            
            # Record the permanent failure in the health metrics database
            try:
                # Record the permanent failure
                self.coordinator.db.execute(
                """
                INSERT INTO worker_permanent_failures (
                    worker_id, failure_time, recovery_attempts, 
                    last_known_status, last_heartbeat, failure_reason
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    worker_id,
                    datetime.now(),
                    attempts,
                    "offline",
                    self.coordinator.workers[worker_id].get("last_heartbeat", "unknown"),
                    "Maximum recovery attempts exceeded"
                )
                )
            except Exception as e:
                logger.error(f"Error recording permanent failure for worker {worker_id}: {str(e)}")
            
            return
        
        # Update worker status in database for recovery
        self.coordinator.db.execute(
        """
        UPDATE worker_nodes
        SET status = 'recovery_pending',
            last_heartbeat = ?
        WHERE worker_id = ?
        """,
        (datetime.now(), worker_id)
        )
        
        # Update worker status in memory
        if worker_id in self.coordinator.workers:
            self.coordinator.workers[worker_id]["status"] = "recovery_pending"
            
        # Implement active recovery steps
        
        # 1. Store recovery attempt in database
        try:
            # Record this recovery attempt
            self.coordinator.db.execute(
            """
            INSERT INTO worker_recovery_attempts (
                worker_id, attempt_number, attempt_time, status, details
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                worker_id,
                attempts,
                datetime.now(),
                "initiated",
                json.dumps({
                    "recovery_strategy": "auto_reconnect",
                    "expected_timeout": 60,  # Give the worker 60 seconds to reconnect
                    "initiated_by": "health_monitor"
                })
            )
            )
        except Exception as e:
            logger.error(f"Error recording recovery attempt for worker {worker_id}: {str(e)}")
            
        # 2. Set recovery pending check time (check after 60 seconds)
        self.recovery_pending[worker_id] = datetime.now() + timedelta(seconds=60)
    
    async def _check_recovery_pending(self):
        """Check the status of workers in recovery_pending mode."""
        now = datetime.now()
        
        for worker_id, check_after in list(self.recovery_pending.items()):
            if now >= check_after:
                # Time to check if worker has recovered
                if worker_id in self.coordinator.workers:
                    worker = self.coordinator.workers[worker_id]
                    
                    # Check if worker is back online
                    if worker.get("status") in ["active", "idle"]:
                        # Worker has recovered!
                        logger.info(f"Worker {worker_id} has recovered successfully")
                        
                        # Update recovery attempt status in database
                        try:
                            self.coordinator.db.execute(
                            """
                            UPDATE worker_recovery_attempts
                            SET status = 'completed',
                                completion_time = ?,
                                success = TRUE,
                                details = json_set(details, '$.completion_time', ?)
                            WHERE worker_id = ? AND status = 'initiated'
                            ORDER BY attempt_time DESC
                            LIMIT 1
                            """,
                            (datetime.now(), datetime.now().isoformat(), worker_id)
                            )
                        except Exception as e:
                            logger.error(f"Error updating recovery status for worker {worker_id}: {str(e)}")
                        
                        # Remove from recovery pending
                        del self.recovery_pending[worker_id]
                    else:
                        # Worker has not recovered yet
                        logger.warning(f"Worker {worker_id} has not recovered, status: {worker.get('status')}")
                        
                        # If still in recovery_pending, retry recovery
                        if worker.get("status") == "recovery_pending":
                            # Retry recovery with another attempt
                            await self.attempt_worker_recovery(worker_id)
                        else:
                            # Remove from recovery pending (will be picked up by regular health check)
                            del self.recovery_pending[worker_id]
                else:
                    # Worker is no longer in the workers list
                    logger.warning(f"Worker {worker_id} is no longer in the workers list")
                    
                    # Update recovery attempt status in database
                    try:
                        self.coordinator.db.execute(
                        """
                        UPDATE worker_recovery_attempts
                        SET status = 'completed',
                            completion_time = ?,
                            success = FALSE,
                            details = json_set(details, '$.completion_time', ?)
                        WHERE worker_id = ? AND status = 'initiated'
                        ORDER BY attempt_time DESC
                        LIMIT 1
                        """,
                        (datetime.now(), datetime.now().isoformat(), worker_id)
                        )
                    except Exception as e:
                        logger.error(f"Error updating recovery status for worker {worker_id}: {str(e)}")
                    
                    # Remove from recovery pending
                    del self.recovery_pending[worker_id]
    
    async def requeue_task(self, task_id: str):
        """
        Requeue a failed task.
        
        Args:
            task_id: Task ID
        """
        # Skip if task no longer exists
        if task_id not in self.coordinator.tasks:
            return
        
        # Get task retry policy
        task = self.coordinator.tasks[task_id]
        task_config = task.get("config", {})
        retry_policy = task_config.get("retry_policy", {})
        
        # Get max retries from policy (default to 3)
        max_retries = retry_policy.get("max_retries", 3)
        
        # Increment task recovery attempts
        self.task_recovery_attempts[task_id] = self.task_recovery_attempts.get(task_id, 0) + 1
        current_attempt = self.task_recovery_attempts[task_id]
        
        # Check if max retries exceeded
        if current_attempt > max_retries:
            logger.warning(f"Max retries ({max_retries}) exceeded for task {task_id}")
            
            # Record the failed recovery attempt
            try:
                self.coordinator.db.execute(
                """
                INSERT INTO task_recovery_attempts (
                    task_id, attempt_number, attempt_time, status, details
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    current_attempt,
                    datetime.now(),
                    "failed",
                    json.dumps({
                        "reason": "max_retries_exceeded",
                        "max_retries": max_retries,
                        "failure_time": datetime.now().isoformat()
                    })
                )
                )
            except Exception as e:
                logger.error(f"Error recording task recovery failure for task {task_id}: {str(e)}")
            
            return
        
        # Log requeue attempt
        logger.info(f"Requeuing task {task_id} (attempt {current_attempt}/{max_retries})")
        
        # Update task in database
        self.coordinator.db.execute(
        """
        UPDATE distributed_tasks
        SET status = 'pending', worker_id = NULL, start_time = NULL
        WHERE task_id = ?
        """,
        (task_id,)
        )
        
        # Update task in memory
        task["status"] = "pending"
        if "started" in task:
            del task["started"]
        if "worker_id" in task:
            del task["worker_id"]
        
        # Add to pending tasks
        self.coordinator.pending_tasks.add(task_id)
        
        # Remove from running tasks
        if task_id in self.coordinator.running_tasks:
            del self.coordinator.running_tasks[task_id]
        
        # Record the recovery attempt
        try:
            self.coordinator.db.execute(
            """
            INSERT INTO task_recovery_attempts (
                task_id, attempt_number, attempt_time, status, details
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                task_id,
                current_attempt,
                datetime.now(),
                "initiated",
                json.dumps({
                    "requeue_time": datetime.now().isoformat(),
                    "previous_attempts": current_attempt - 1,
                    "max_retries": max_retries
                })
            )
            )
        except Exception as e:
            logger.error(f"Error recording task recovery attempt for task {task_id}: {str(e)}")
        
        # Try to assign task to a new worker
        await self.coordinator._assign_pending_tasks()
    
    async def update_health_metrics_in_db(self):
        """Update health metrics in the database."""
        now = datetime.now()
        
        try:
            # Update worker health metrics in database
            for worker_id, metrics in self.worker_health_metrics.items():
                # Insert metrics
                self.coordinator.db.execute(
                """
                INSERT INTO worker_health_metrics (
                    worker_id, check_time, heartbeat_age_seconds, status, metrics
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    worker_id,
                    now,
                    metrics.get("heartbeat_age_seconds", 0),
                    metrics.get("status", "unknown"),
                    json.dumps(metrics)
                )
                )
            
            # Update task health metrics in database
            for task_id, metrics in self.task_health_metrics.items():
                # Insert metrics
                self.coordinator.db.execute(
                """
                INSERT INTO task_health_metrics (
                    task_id, worker_id, check_time, running_time_seconds,
                    timeout_seconds, status, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    metrics.get("worker_id", "unknown"),
                    now,
                    metrics.get("running_time_seconds", 0),
                    metrics.get("timeout_seconds", 0),
                    metrics.get("status", "unknown"),
                    json.dumps(metrics)
                )
                )
            
        except Exception as e:
            logger.error(f"Error updating health metrics in database: {str(e)}")
    
    async def _system_health_monitor(self):
        """Monitor overall system health and update status."""
        while True:
            try:
                # Get worker health summary
                worker_summary = self.get_worker_health_summary()
                
                # Get task health summary
                task_summary = self.get_task_health_summary()
                
                # Determine system health status
                if worker_summary["active_percentage"] < 50 or worker_summary["total_workers"] == 0:
                    # Less than 50% of workers are active or no workers registered
                    status = "critical"
                elif worker_summary["active_percentage"] < 80:
                    # Less than 80% of workers are active
                    status = "degraded"
                else:
                    # 80% or more of workers are active
                    status = "healthy"
                
                # Update system health status
                self.system_health_status = status
                
                # Update system health status in database
                self.coordinator.db.execute(
                """
                INSERT INTO system_health_status (
                    check_time, status, active_workers, total_workers, active_percentage,
                    tasks_pending, tasks_running, tasks_completed, tasks_failed, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(),
                    status,
                    worker_summary["active_workers"],
                    worker_summary["total_workers"],
                    worker_summary["active_percentage"],
                    task_summary["pending_tasks"],
                    task_summary["running_tasks"],
                    task_summary["completed_tasks"],
                    task_summary["failed_tasks"],
                    json.dumps({
                        "worker_summary": worker_summary,
                        "task_summary": task_summary
                    })
                )
                )
                
                # Log system health status
                logger.info(f"System health status: {status}")
                
                # If status is critical, send alert
                if status == "critical":
                    logger.critical(f"SYSTEM HEALTH CRITICAL: Only {worker_summary['active_percentage']:.1f}% of workers are active!")
                
            except Exception as e:
                logger.error(f"Error in system health monitor: {str(e)}")
            
            # Sleep for system health check interval
            await anyio.sleep(60)  # Check every minute
    
    def update_task_timeout_estimate(self, task_type: str, execution_time: float):
        """
        Update task timeout estimate based on actual execution time.
        
        Args:
            task_type: Task type
            execution_time: Actual execution time in seconds
        """
        # Only update if execution time is reasonable
        if execution_time <= 0 or execution_time > 3600 * 24:  # 24 hours max
            return
        
        key = f"{task_type}"
        
        if key not in self.task_timeout_estimates:
            # First estimate
            self.task_timeout_estimates[key] = execution_time
        else:
            # Exponential moving average (alpha = 0.3)
            alpha = 0.3
            current = self.task_timeout_estimates[key]
            self.task_timeout_estimates[key] = (1 - alpha) * current + alpha * execution_time
        
        logger.debug(f"Updated timeout estimate for {task_type}: {self.task_timeout_estimates[key]:.2f}s")
    
    def get_worker_health_summary(self) -> Dict[str, Any]:
        """
        Get summary of worker health.
        
        Returns:
            Summary of worker health
        """
        total_workers = len(self.coordinator.workers)
        active_workers = sum(1 for w in self.coordinator.workers.values() if w.get("status") == "active")
        idle_workers = sum(1 for w in self.coordinator.workers.values() if w.get("status") == "idle")
        offline_workers = sum(1 for w in self.coordinator.workers.values() if w.get("status") == "offline")
        recovery_pending = sum(1 for w in self.coordinator.workers.values() if w.get("status") == "recovery_pending")
        permanent_failure = sum(1 for w in self.coordinator.workers.values() if w.get("status") == "permanent_failure")
        
        summary = {
            "total_workers": total_workers,
            "active_workers": active_workers,
            "idle_workers": idle_workers,
            "offline_workers": offline_workers,
            "recovery_pending": recovery_pending,
            "permanent_failure": permanent_failure,
            "active_percentage": (active_workers / total_workers * 100) if total_workers > 0 else 0,
            "recovery_attempts": sum(self.recovery_attempts.values()),
            "health_check_interval": self.check_interval,
            "heartbeat_timeout": self.heartbeat_timeout,
            "auto_recovery_enabled": self.auto_recovery,
            "system_health_status": self.system_health_status
        }
        
        return summary
    
    def get_task_health_summary(self) -> Dict[str, Any]:
        """
        Get summary of task health.
        
        Returns:
            Summary of task health
        """
        total_tasks = len(self.coordinator.tasks)
        pending_tasks = len(self.coordinator.pending_tasks)
        running_tasks = len(self.coordinator.running_tasks)
        completed_tasks = len(self.coordinator.completed_tasks)
        failed_tasks = len(self.coordinator.failed_tasks)
        
        summary = {
            "total_tasks": total_tasks,
            "pending_tasks": pending_tasks,
            "running_tasks": running_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "completion_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "failure_rate": (failed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "task_recovery_attempts": sum(self.task_recovery_attempts.values()),
            "timeout_estimates": dict(self.task_timeout_estimates),
        }
        
        return summary
    
    def get_worker_details(self, worker_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a worker.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            Worker details
        """
        # Get basic worker info
        if worker_id not in self.coordinator.workers:
            return {"error": "Worker not found"}
        
        worker = self.coordinator.workers[worker_id]
        
        # Get recovery history from database
        recovery_history = []
        try:
            result = self.coordinator.db.execute(
            """
            SELECT 
                attempt_number, attempt_time, status, completion_time, 
                success, details
            FROM worker_recovery_attempts
            WHERE worker_id = ?
            ORDER BY attempt_time DESC
            LIMIT 10
            """,
            (worker_id,)
            ).fetchall()
            
            for row in result:
                recovery_attempt = {
                    "attempt_number": row[0],
                    "attempt_time": row[1],
                    "status": row[2],
                    "completion_time": row[3],
                    "success": row[4],
                    "details": json.loads(row[5]) if row[5] else {}
                }
                recovery_history.append(recovery_attempt)
        except Exception as e:
            logger.error(f"Error getting recovery history for worker {worker_id}: {str(e)}")
        
        # Get health metrics history
        health_history = []
        try:
            result = self.coordinator.db.execute(
            """
            SELECT 
                check_time, heartbeat_age_seconds, status, metrics
            FROM worker_health_metrics
            WHERE worker_id = ?
            ORDER BY check_time DESC
            LIMIT 10
            """,
            (worker_id,)
            ).fetchall()
            
            for row in result:
                health_metric = {
                    "check_time": row[0],
                    "heartbeat_age_seconds": row[1],
                    "status": row[2],
                    "metrics": json.loads(row[3]) if row[3] else {}
                }
                health_history.append(health_metric)
        except Exception as e:
            logger.error(f"Error getting health history for worker {worker_id}: {str(e)}")
        
        # Get tasks assigned to this worker
        assigned_tasks = []
        for task_id, assigned_worker_id in self.coordinator.running_tasks.items():
            if assigned_worker_id == worker_id:
                task = self.coordinator.tasks.get(task_id, {})
                assigned_tasks.append({
                    "task_id": task_id,
                    "type": task.get("type", "unknown"),
                    "status": task.get("status", "unknown"),
                    "started": task.get("started", "unknown")
                })
        
        # Combine all information
        details = {
            "worker_id": worker_id,
            "hostname": worker.get("hostname", "unknown"),
            "status": worker.get("status", "unknown"),
            "last_heartbeat": worker.get("last_heartbeat", "unknown"),
            "capabilities": worker.get("capabilities", {}),
            "hardware_metrics": worker.get("hardware_metrics", {}),
            "recovery_attempts": self.recovery_attempts.get(worker_id, 0),
            "recovery_history": recovery_history,
            "health_history": health_history,
            "assigned_tasks": assigned_tasks,
            "in_recovery_pending": worker_id in self.recovery_pending
        }
        
        return details
    
    def get_task_details(self, task_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task details
        """
        # Get basic task info
        if task_id not in self.coordinator.tasks:
            return {"error": "Task not found"}
        
        task = self.coordinator.tasks[task_id]
        
        # Get recovery history from database
        recovery_history = []
        try:
            result = self.coordinator.db.execute(
            """
            SELECT 
                attempt_number, attempt_time, status, worker_id, details
            FROM task_recovery_attempts
            WHERE task_id = ?
            ORDER BY attempt_time DESC
            LIMIT 10
            """,
            (task_id,)
            ).fetchall()
            
            for row in result:
                recovery_attempt = {
                    "attempt_number": row[0],
                    "attempt_time": row[1],
                    "status": row[2],
                    "worker_id": row[3],
                    "details": json.loads(row[4]) if row[4] else {}
                }
                recovery_history.append(recovery_attempt)
        except Exception as e:
            logger.error(f"Error getting recovery history for task {task_id}: {str(e)}")
        
        # Get health metrics history
        health_history = []
        try:
            result = self.coordinator.db.execute(
            """
            SELECT 
                check_time, worker_id, running_time_seconds, timeout_seconds, status, metrics
            FROM task_health_metrics
            WHERE task_id = ?
            ORDER BY check_time DESC
            LIMIT 10
            """,
            (task_id,)
            ).fetchall()
            
            for row in result:
                health_metric = {
                    "check_time": row[0],
                    "worker_id": row[1],
                    "running_time_seconds": row[2],
                    "timeout_seconds": row[3],
                    "status": row[4],
                    "metrics": json.loads(row[5]) if row[5] else {}
                }
                health_history.append(health_metric)
        except Exception as e:
            logger.error(f"Error getting health history for task {task_id}: {str(e)}")
        
        # Get execution history
        execution_history = []
        try:
            result = self.coordinator.db.execute(
            """
            SELECT 
                worker_id, attempt, status, start_time, end_time, 
                execution_time_seconds, error_message
            FROM task_execution_history
            WHERE task_id = ?
            ORDER BY start_time DESC
            LIMIT 10
            """,
            (task_id,)
            ).fetchall()
            
            for row in result:
                execution = {
                    "worker_id": row[0],
                    "attempt": row[1],
                    "status": row[2],
                    "start_time": row[3],
                    "end_time": row[4],
                    "execution_time_seconds": row[5],
                    "error_message": row[6]
                }
                execution_history.append(execution)
        except Exception as e:
            logger.error(f"Error getting execution history for task {task_id}: {str(e)}")
        
        # Combine all information
        details = {
            "task_id": task_id,
            "type": task.get("type", "unknown"),
            "status": task.get("status", "unknown"),
            "priority": task.get("priority", 0),
            "created": task.get("created", "unknown"),
            "started": task.get("started", "unknown"),
            "ended": task.get("ended", "unknown"),
            "worker_id": task.get("worker_id", "unknown"),
            "config": task.get("config", {}),
            "requirements": task.get("requirements", {}),
            "recovery_attempts": self.task_recovery_attempts.get(task_id, 0),
            "recovery_history": recovery_history,
            "health_history": health_history,
            "execution_history": execution_history
        }
        
        return details