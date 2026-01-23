#!/usr/bin/env python3
"""
Distributed Testing Framework - Auto Recovery Module

This module provides auto-recovery capabilities for the distributed testing framework.
It works with the health monitoring system to detect and recover from failures.

Usage:
    Import this module in coordinator.py to enable auto-recovery functionality.
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

class AutoRecoveryManager:
    """Auto recovery manager for distributed testing framework."""
    
    def __init__(
        self,
        coordinator,
        health_monitor,
        max_recovery_attempts: int = 5,
        recovery_interval: int = 60,
        enable_proactive_recovery: bool = True
    ):
        """
        Initialize the auto recovery manager.
        
        Args:
            coordinator: Reference to the coordinator instance
            health_monitor: Reference to the health monitor instance
            max_recovery_attempts: Maximum number of recovery attempts per worker
            recovery_interval: Interval between recovery attempts in seconds
            enable_proactive_recovery: Whether to enable proactive recovery
        """
        self.coordinator = coordinator
        self.health_monitor = health_monitor
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_interval = recovery_interval
        self.enable_proactive_recovery = enable_proactive_recovery
        
        # Recovery tracking
        self.recovery_attempts: Dict[str, int] = {}  # worker_id -> attempts
        self.recovery_history: Dict[str, List[Dict[str, Any]]] = {}  # worker_id -> recovery history
        self.pending_recovery: Set[str] = set()  # Set of worker_ids with pending recovery
        
        # Task recovery tracking
        self.task_recovery_attempts: Dict[str, int] = {}  # task_id -> attempts
        self.task_recovery_history: Dict[str, List[Dict[str, Any]]] = {}  # task_id -> recovery history
        
        # System status
        self.recovery_in_progress = False
        
        logger.info("Auto recovery manager initialized")
    
    async def start_recovery_monitoring(self):
        """Start the recovery monitoring loop."""
        logger.info("Starting recovery monitoring loop")
        
        # Create database tables if needed
        await self._create_database_tables()
        
        # Initialize from database
        await self._load_recovery_data_from_db()
        
        while True:
            try:
                # Check system health status
                system_health = self.health_monitor.system_health_status
                
                if system_health == "critical":
                    logger.critical("System health is CRITICAL - triggering emergency recovery")
                    await self._handle_system_critical_state()
                
                # Check for unhealthy workers
                await self._check_unhealthy_workers()
                
                # Check for pending recovery tasks
                await self._process_pending_recovery()
                
                # Check for stalled tasks
                if self.enable_proactive_recovery:
                    await self._check_stalled_tasks()
                
            except Exception as e:
                logger.error(f"Error in recovery monitoring loop: {str(e)}")
            
            # Sleep until next check
            await anyio.sleep(self.recovery_interval)
    
    async def _create_database_tables(self):
        """Create required database tables for auto recovery."""
        try:
            # Worker recovery history table
            self.coordinator.db.execute("""
            CREATE TABLE IF NOT EXISTS worker_recovery_history (
                id INTEGER PRIMARY KEY,
                worker_id VARCHAR,
                recovery_time TIMESTAMP,
                attempt_number INTEGER,
                success BOOLEAN,
                recovery_type VARCHAR,
                error_message VARCHAR,
                details JSON
            )
            """)
            
            # Task recovery history table
            self.coordinator.db.execute("""
            CREATE TABLE IF NOT EXISTS task_recovery_history (
                id INTEGER PRIMARY KEY,
                task_id VARCHAR,
                worker_id VARCHAR,
                recovery_time TIMESTAMP,
                attempt_number INTEGER,
                success BOOLEAN,
                recovery_type VARCHAR,
                error_message VARCHAR,
                details JSON
            )
            """)
            
            # System recovery history table
            self.coordinator.db.execute("""
            CREATE TABLE IF NOT EXISTS system_recovery_history (
                id INTEGER PRIMARY KEY,
                recovery_time TIMESTAMP,
                system_status VARCHAR,
                success BOOLEAN,
                affected_workers INTEGER,
                affected_tasks INTEGER,
                details JSON
            )
            """)
            
            logger.info("Auto recovery database tables created")
            
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
    
    async def _load_recovery_data_from_db(self):
        """Load recovery data from database."""
        try:
            # Load worker recovery attempts
            results = self.coordinator.db.execute("""
            SELECT worker_id, COUNT(*) as attempts
            FROM worker_recovery_history
            WHERE recovery_time > datetime('now', '-1 day')
            GROUP BY worker_id
            """).fetchall()
            
            for row in results:
                self.recovery_attempts[row[0]] = row[1]
            
            # Load task recovery attempts
            results = self.coordinator.db.execute("""
            SELECT task_id, COUNT(*) as attempts
            FROM task_recovery_history
            WHERE recovery_time > datetime('now', '-1 day')
            GROUP BY task_id
            """).fetchall()
            
            for row in results:
                self.task_recovery_attempts[row[0]] = row[1]
            
            logger.info(f"Loaded recovery data from database: {len(self.recovery_attempts)} workers, {len(self.task_recovery_attempts)} tasks")
            
        except Exception as e:
            logger.error(f"Error loading recovery data from database: {str(e)}")
    
    async def _check_unhealthy_workers(self):
        """Check for unhealthy workers and initiate recovery if needed."""
        # Get unhealthy workers from health monitor
        for worker_id, metrics in self.health_monitor.worker_health_metrics.items():
            heartbeat_age = metrics.get("heartbeat_age_seconds", 0)
            status = metrics.get("status", "unknown")
            
            if status in ["offline", "error"] or heartbeat_age > self.health_monitor.heartbeat_timeout:
                # Check if worker is already in recovery
                if worker_id in self.pending_recovery:
                    continue
                
                # Check if we've tried too many times
                attempts = self.recovery_attempts.get(worker_id, 0)
                
                if attempts >= self.max_recovery_attempts:
                    logger.warning(f"Worker {worker_id} has failed {attempts} times, marking as permanent failure")
                    await self._mark_worker_permanent_failure(worker_id)
                else:
                    # Schedule recovery
                    await self._schedule_worker_recovery(worker_id)
    
    async def _schedule_worker_recovery(self, worker_id: str):
        """Schedule recovery for a worker."""
        # Add to pending recovery
        self.pending_recovery.add(worker_id)
        
        # Increment recovery attempts
        self.recovery_attempts[worker_id] = self.recovery_attempts.get(worker_id, 0) + 1
        attempts = self.recovery_attempts[worker_id]
        
        # Log recovery attempt
        logger.info(f"Scheduling recovery for worker {worker_id} (attempt {attempts}/{self.max_recovery_attempts})")
        
        # Record recovery attempt in database
        try:
            self.coordinator.db.execute(
            """
            INSERT INTO worker_recovery_history (
                worker_id, recovery_time, attempt_number, 
                success, recovery_type, details
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                worker_id,
                datetime.now(),
                attempts,
                False,  # Success will be updated after recovery
                "scheduled",
                json.dumps({
                    "scheduled_time": datetime.now().isoformat(),
                    "recovery_type": "automatic",
                    "max_attempts": self.max_recovery_attempts
                })
            )
            )
        except Exception as e:
            logger.error(f"Error recording recovery attempt for worker {worker_id}: {str(e)}")
    
    async def _process_pending_recovery(self):
        """Process pending recovery tasks."""
        for worker_id in list(self.pending_recovery):
            try:
                # Attempt recovery
                logger.info(f"Processing recovery for worker {worker_id}")
                
                # If worker is back online, mark recovery as successful
                if worker_id in self.coordinator.workers:
                    worker = self.coordinator.workers[worker_id]
                    
                    if worker.get("status") in ["active", "idle"]:
                        logger.info(f"Worker {worker_id} is back online, recovery successful")
                        
                        # Update recovery history
                        await self._record_recovery_success(worker_id)
                        
                        # Remove from pending recovery
                        self.pending_recovery.remove(worker_id)
                        continue
                
                # Worker is still offline, attempt recovery
                success = await self._attempt_worker_recovery(worker_id)
                
                if success:
                    logger.info(f"Recovery successful for worker {worker_id}")
                    
                    # Update recovery history
                    await self._record_recovery_success(worker_id)
                    
                    # Remove from pending recovery
                    self.pending_recovery.remove(worker_id)
                else:
                    logger.warning(f"Recovery failed for worker {worker_id}")
                    
                    # Keep in pending recovery for next attempt
                    # We'll remove it after max attempts
                    if self.recovery_attempts.get(worker_id, 0) >= self.max_recovery_attempts:
                        logger.error(f"Worker {worker_id} has reached maximum recovery attempts")
                        await self._mark_worker_permanent_failure(worker_id)
                        self.pending_recovery.remove(worker_id)
                
            except Exception as e:
                logger.error(f"Error processing recovery for worker {worker_id}: {str(e)}")
    
    async def _attempt_worker_recovery(self, worker_id: str) -> bool:
        """
        Attempt to recover a worker.
        
        Args:
            worker_id: Worker ID
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Set recovery in progress
        self.recovery_in_progress = True
        
        try:
            # Check if worker is connected
            if worker_id in self.coordinator.worker_connections:
                logger.info(f"Worker {worker_id} is connected but may be unresponsive")
                
                # Close existing connection to force reconnect
                ws = self.coordinator.worker_connections[worker_id]
                
                try:
                    await ws.close()
                except Exception:
                    pass
                
                # Remove from connections
                del self.coordinator.worker_connections[worker_id]
            
            # Mark worker as recovery_pending in database
            self.coordinator.db.execute(
            """
            UPDATE worker_nodes
            SET status = 'recovery_pending'
            WHERE worker_id = ?
            """,
            (worker_id,)
            )
            
            # Update worker status in memory
            if worker_id in self.coordinator.workers:
                self.coordinator.workers[worker_id]["status"] = "recovery_pending"
            
            # Find tasks assigned to this worker
            recovered_tasks = 0
            for task_id, assigned_worker_id in list(self.coordinator.running_tasks.items()):
                if assigned_worker_id == worker_id:
                    # Requeue task
                    await self._requeue_task(task_id, worker_id, "worker_failure")
                    recovered_tasks += 1
            
            logger.info(f"Recovered {recovered_tasks} tasks from failed worker {worker_id}")
            
            # Wait for worker to reconnect (if it will)
            # In a real implementation, we might have more active recovery steps here
            
            # For now, we'll consider recovery "successful" if we've handled the tasks
            # The worker may reconnect later
            return True
            
        except Exception as e:
            logger.error(f"Error attempting recovery for worker {worker_id}: {str(e)}")
            return False
        finally:
            # Reset recovery in progress
            self.recovery_in_progress = False
    
    async def _record_recovery_success(self, worker_id: str):
        """Record successful recovery in database."""
        try:
            # Get the most recent recovery attempt
            result = self.coordinator.db.execute(
            """
            SELECT id FROM worker_recovery_history
            WHERE worker_id = ?
            ORDER BY recovery_time DESC
            LIMIT 1
            """,
            (worker_id,)
            ).fetchone()
            
            if result:
                recovery_id = result[0]
                
                # Update recovery record
                self.coordinator.db.execute(
                """
                UPDATE worker_recovery_history
                SET success = TRUE,
                    details = json_set(details, '$.completion_time', ?)
                WHERE id = ?
                """,
                (datetime.now().isoformat(), recovery_id)
                )
                
                logger.info(f"Recorded successful recovery for worker {worker_id}")
        except Exception as e:
            logger.error(f"Error recording recovery success for worker {worker_id}: {str(e)}")
    
    async def _mark_worker_permanent_failure(self, worker_id: str):
        """Mark a worker as permanently failed."""
        logger.error(f"Marking worker {worker_id} as permanently failed after {self.recovery_attempts.get(worker_id, 0)} recovery attempts")
        
        try:
            # Update worker status in database
            self.coordinator.db.execute(
            """
            UPDATE worker_nodes
            SET status = 'permanent_failure'
            WHERE worker_id = ?
            """,
            (worker_id,)
            )
            
            # Update in-memory status
            if worker_id in self.coordinator.workers:
                self.coordinator.workers[worker_id]["status"] = "permanent_failure"
                
            # Log the permanent failure
            self.coordinator.db.execute(
            """
            INSERT INTO worker_permanent_failures (
                worker_id, failure_time, recovery_attempts,
                last_known_status, failure_reason
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                worker_id,
                datetime.now(),
                self.recovery_attempts.get(worker_id, 0),
                "offline",
                "Maximum recovery attempts exceeded"
            )
            )
            
            # Handle tasks assigned to this worker
            for task_id, assigned_worker_id in list(self.coordinator.running_tasks.items()):
                if assigned_worker_id == worker_id:
                    # Requeue task with recovery note
                    await self._requeue_task(task_id, worker_id, "permanent_worker_failure")
            
        except Exception as e:
            logger.error(f"Error marking worker as permanent failure: {str(e)}")
    
    async def _requeue_task(self, task_id: str, worker_id: str, reason: str):
        """
        Requeue a task after worker failure.
        
        Args:
            task_id: Task ID
            worker_id: Worker ID that was running the task
            reason: Reason for requeuing
        """
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
                self.coordinator.failed_tasks.add(task_id)
                
                # Remove from running tasks
                if task_id in self.coordinator.running_tasks:
                    del self.coordinator.running_tasks[task_id]
                
                # Record failure in database
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
            self.coordinator.pending_tasks.add(task_id)
            
            # Remove from running tasks
            if task_id in self.coordinator.running_tasks:
                del self.coordinator.running_tasks[task_id]
            
            # Update task in database
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
            await self.coordinator._assign_pending_tasks()
            
        except Exception as e:
            logger.error(f"Error requeuing task {task_id}: {str(e)}")
    
    async def _check_stalled_tasks(self):
        """Check for stalled tasks that may need recovery."""
        now = datetime.now()
        
        for task_id, worker_id in list(self.coordinator.running_tasks.items()):
            # Skip if task no longer exists
            if task_id not in self.coordinator.tasks:
                continue
            
            task = self.coordinator.tasks[task_id]
            
            # Skip if no start time
            if "started" not in task:
                continue
            
            # Calculate running time
            try:
                started = datetime.fromisoformat(task["started"])
                running_time = (now - started).total_seconds()
            except Exception:
                continue
            
            # Get timeout from task_monitor
            timeout = self.health_monitor.get_task_timeout(task_id, task)
            
            # Check if task has been running too long
            if running_time > timeout:
                logger.warning(f"Task {task_id} has been running for {running_time:.1f} seconds (timeout: {timeout}s), may be stalled")
                
                # Check if worker is healthy
                worker_healthy = True
                if worker_id in self.health_monitor.worker_health_metrics:
                    worker_status = self.health_monitor.worker_health_metrics[worker_id].get("status", "unknown")
                    worker_healthy = worker_status in ["active", "idle"]
                
                if not worker_healthy:
                    logger.warning(f"Worker {worker_id} appears unhealthy, recovering task {task_id}")
                    await self._requeue_task(task_id, worker_id, "stalled_task_unhealthy_worker")
                else:
                    # Worker looks healthy, but task may be stuck
                    # We could implement more sophisticated detection here
                    pass
    
    async def _handle_system_critical_state(self):
        """Handle critical system state with emergency recovery."""
        logger.critical("EMERGENCY RECOVERY: System is in critical state")
        
        # Record emergency recovery in database
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
                "critical",
                len(self.coordinator.workers),
                len(self.coordinator.running_tasks),
                json.dumps({
                    "emergency_recovery": True,
                    "active_workers_percentage": self.health_monitor.get_worker_health_summary().get("active_percentage", 0),
                    "initiated_time": datetime.now().isoformat()
                })
            )
            )
        except Exception as e:
            logger.error(f"Error recording system recovery: {str(e)}")
        
        # Implement emergency recovery steps
        # This could include restarting services, recovering database, etc.
        # For now, we'll focus on task recovery
        
        # 1. Requeue all running tasks
        requeued_tasks = 0
        for task_id, worker_id in list(self.coordinator.running_tasks.items()):
            await self._requeue_task(task_id, worker_id, "system_critical_emergency")
            requeued_tasks += 1
        
        logger.info(f"Emergency recovery: requeued {requeued_tasks} tasks")
        
        # 2. Reset worker connections to force reconnect
        for worker_id, ws in list(self.coordinator.worker_connections.items()):
            try:
                await ws.close()
            except Exception:
                pass
        
        self.coordinator.worker_connections.clear()
        
        # 3. Mark all workers as recovery_pending
        for worker_id, worker in self.coordinator.workers.items():
            if worker.get("status") not in ["permanent_failure"]:
                worker["status"] = "recovery_pending"
        
        # Update database
        try:
            self.coordinator.db.execute(
            """
            UPDATE worker_nodes
            SET status = 'recovery_pending'
            WHERE status != 'permanent_failure'
            """)
        except Exception as e:
            logger.error(f"Error updating worker status during emergency recovery: {str(e)}")
        
        # 4. Try to reassign pending tasks
        await self.coordinator._assign_pending_tasks()
        
        # Record completion
        try:
            self.coordinator.db.execute(
            """
            UPDATE system_recovery_history
            SET success = ?,
                details = json_set(details, '$.completion_time', ?),
                details = json_set(details, '$.requeued_tasks', ?)
            WHERE recovery_time = (
                SELECT MAX(recovery_time) FROM system_recovery_history
            )
            """,
            (
                True,
                datetime.now().isoformat(),
                requeued_tasks
            )
            )
        except Exception as e:
            logger.error(f"Error updating system recovery record: {str(e)}")
        
        logger.info("Emergency recovery procedures completed")